"""PyTorch Dataset for sequential driving data.

Loads HDF5 episodes and returns fixed-length clips for RWKV training
with Truncated Backpropagation Through Time (TBPTT).

Each item is a contiguous sequence of frames from a single episode,
suitable for recurrent model training where temporal continuity matters.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

# ImageNet normalization constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class DrivingSequenceDataset(Dataset):
    """Dataset that returns fixed-length sequential clips from driving episodes.

    Each __getitem__ call returns a contiguous clip of `seq_len` frames from
    a single episode. Clips are sampled by sliding a window over each episode.

    Args:
        data_dir: Path to directory containing episode HDF5 files.
        seq_len: Number of frames per clip (default: 32).
        stride: Stride between clip start positions within an episode (default: 16).
        split: One of "train" or "val" (default: "train").
        val_fraction: Fraction of episodes reserved for validation (default: 0.2).
        seed: Random seed for reproducible train/val split (default: 42).
        augment: Whether to apply data augmentation (default: True for train).
    """

    def __init__(
        self,
        data_dir: str | Path,
        seq_len: int = 32,
        stride: int = 16,
        split: str = "train",
        val_fraction: float = 0.2,
        seed: int = 42,
        augment: bool | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.stride = stride
        self.split = split

        if augment is None:
            self.augment = split == "train"
        else:
            self.augment = augment

        # Discover episode files
        all_episodes = sorted(self.data_dir.glob("episode_*.h5"))
        if not all_episodes:
            raise FileNotFoundError(f"No episode_*.h5 files found in {self.data_dir}")

        # Deterministic episode-level train/val split
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(all_episodes))
        n_val = max(1, int(len(all_episodes) * val_fraction))

        if split == "val":
            selected_indices = sorted(indices[:n_val])
        elif split == "train":
            selected_indices = sorted(indices[n_val:])
        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        self.episode_files = [all_episodes[i] for i in selected_indices]

        # Build index of (episode_idx, start_frame) pairs
        self.clips: list[tuple[int, int]] = []
        self.episode_nav_counts: list[dict[int, int]] = []

        for ep_idx, ep_path in enumerate(self.episode_files):
            with h5py.File(ep_path, "r") as f:
                n_frames = f["images"].shape[0]
                nav_cmds = f["nav_command"][:]

            # Count nav commands for balanced sampling
            nav_counts = {}
            for cmd in range(4):
                nav_counts[cmd] = int(np.sum(nav_cmds == cmd))
            self.episode_nav_counts.append(nav_counts)

            # Generate clip start positions
            if n_frames >= seq_len:
                for start in range(0, n_frames - seq_len + 1, stride):
                    self.clips.append((ep_idx, start))

        if not self.clips:
            raise ValueError(
                f"No valid clips found (seq_len={seq_len} may exceed all episode lengths)"
            )

        # Compute per-clip sampling weights (inverse of LaneFollow frequency)
        self._clip_weights = self._compute_sampling_weights()

    def _compute_sampling_weights(self) -> torch.Tensor:
        """Compute sampling weights to balance turn distribution.

        Episodes with more turns (Left/Right) get higher weight to
        counteract the natural LaneFollow dominance.
        """
        weights = []
        for ep_idx, _start in self.clips:
            nav = self.episode_nav_counts[ep_idx]
            total = sum(nav.values())
            if total == 0:
                weights.append(1.0)
                continue
            # Weight = inverse of LaneFollow fraction (more turns → higher weight)
            lane_follow_frac = nav.get(0, 0) / total
            # Avoid division by zero; cap minimum fraction
            weight = 1.0 / max(lane_follow_frac, 0.1)
            weights.append(weight)

        w = torch.tensor(weights, dtype=torch.float64)
        w = w / w.sum()  # Normalize
        return w

    def get_sampler(self) -> WeightedRandomSampler:
        """Get a WeightedRandomSampler for balanced training.

        Usage:
            dataset = DrivingSequenceDataset(...)
            sampler = dataset.get_sampler()
            loader = DataLoader(dataset, sampler=sampler, ...)
        """
        return WeightedRandomSampler(
            weights=self._clip_weights,
            num_samples=len(self),
            replacement=True,
        )

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load a single clip.

        Returns:
            Dictionary with:
                images: (seq_len, 3, 224, 224) float32, ImageNet-normalized
                controls: (seq_len, 2) float32 [steering, throttle]
                nav_command: (seq_len,) int64
                velocity: (seq_len, 3) float32
        """
        ep_idx, start = self.clips[idx]
        end = start + self.seq_len

        ep_path = self.episode_files[ep_idx]

        with h5py.File(ep_path, "r") as f:
            images = f["images"][start:end]  # (T, H, W, 3) uint8
            steering = f["steering"][start:end]  # (T,) float32
            throttle = f["throttle"][start:end]  # (T,) float32
            nav_cmd = f["nav_command"][start:end]  # (T,) int32
            velocity = f["velocity"][start:end]  # (T, 3) float32

        # Convert images: uint8 HWC → float32 CHW, then normalize
        images = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).float() / 255.0
        images = (images - IMAGENET_MEAN) / IMAGENET_STD

        steering = torch.from_numpy(steering.copy())
        throttle = torch.from_numpy(throttle.copy())
        controls = torch.stack([steering, throttle], dim=-1)  # (T, 2)

        nav_cmd = torch.from_numpy(nav_cmd.copy()).long()
        velocity = torch.from_numpy(velocity.copy())

        # Data augmentation (train only)
        if self.augment:
            images, controls = self._augment(images, controls)

        return {
            "images": images,
            "controls": controls,
            "nav_command": nav_cmd,
            "velocity": velocity,
        }

    def _augment(
        self,
        images: torch.Tensor,
        controls: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation to a clip.

        Augmentations:
            - Color jitter (brightness, contrast, saturation)
            - Horizontal flip (with steering sign flip)

        All augmentations are applied consistently across the entire clip
        to maintain temporal coherence.

        Args:
            images: (T, 3, H, W) normalized float32
            controls: (T, 2) [steering, throttle]

        Returns:
            Augmented (images, controls) tuple.
        """
        # Color jitter: same perturbation for entire clip
        if torch.rand(1).item() < 0.5:
            # Brightness
            brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.4  # [0.8, 1.2]
            images = images * brightness

        if torch.rand(1).item() < 0.3:
            # Contrast
            mean = images.mean(dim=(2, 3), keepdim=True)
            contrast = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
            images = (images - mean) * contrast + mean

        # Horizontal flip with steering sign flip
        if torch.rand(1).item() < 0.5:
            images = images.flip(dims=[-1])  # Flip W dimension
            controls[:, 0] = -controls[:, 0]  # Flip steering sign

        return images, controls

    def get_split_info(self) -> dict[str, Any]:
        """Return metadata about this dataset split."""
        return {
            "split": self.split,
            "n_episodes": len(self.episode_files),
            "n_clips": len(self.clips),
            "seq_len": self.seq_len,
            "stride": self.stride,
            "augment": self.augment,
            "episode_files": [p.name for p in self.episode_files],
        }
