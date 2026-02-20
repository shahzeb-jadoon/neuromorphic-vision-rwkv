"""PyTorch Dataset for sequential driving data.

Loads HDF5 episodes and returns fixed-length clips for RWKV training
with Truncated Backpropagation Through Time (TBPTT).

Each item is a contiguous sequence of frames from a single episode,
suitable for recurrent model training where temporal continuity matters.

Schema v3 changes:
    - Images are 3-camera panoramic (H, W_total, 3) where W_total = 3 * W_cam.
    - Navigation commands use RoadOption values (LEFT=1 .. LANEFOLLOW=4).
    - Position, rotation, traffic light state available as extras.
    - Horizontal flip augmentation swaps left/right camera panels.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from src.data import hdf5_schema as S

# ImageNet normalization constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Velocity z-score normalisation constants.
# TODO: Recompute from real collection data and update.
VELOCITY_STATS_VERSION = "placeholder_v0"
VELOCITY_MEAN = torch.tensor([5.0, 0.0, 0.0])
VELOCITY_STD = torch.tensor([4.0, 0.5, 0.1])


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
        n_cameras: Number of cameras in the panoramic image (default: 3).
            Used by the horizontal flip augmentation to swap camera panels.
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
        n_cameras: int = 3,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.stride = stride
        self.split = split
        self.n_cameras = n_cameras

        if VELOCITY_STATS_VERSION == "placeholder_v0":
            warnings.warn(
                "Using placeholder velocity normalization stats "
                f"(VELOCITY_STATS_VERSION={VELOCITY_STATS_VERSION!r}). "
                "Recompute from real collection data before training — "
                "see scripts/analyze_dataset.py for per-axis statistics.",
                UserWarning,
                stacklevel=2,
            )

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

        # --- Schema version gate ---
        # Fail fast if any episode is not schema v3+.  Older schemas
        # lack panoramic images and RoadOption nav commands, leading to
        # cryptic KeyErrors deep in training loops.
        _MIN_SCHEMA = 3
        for ep_path in self.episode_files:
            with h5py.File(ep_path, "r") as f:
                sv = 0
                if S.METADATA_GROUP in f:
                    sv = int(f[S.METADATA_GROUP].attrs.get(S.META_SCHEMA_VERSION, 0))
                if sv < _MIN_SCHEMA:
                    raise ValueError(
                        f"Episode {ep_path.name} has schema_version={sv} "
                        f"(minimum required: {_MIN_SCHEMA}).  Re-collect "
                        f"with the current ExpertCollector to produce v3 data."
                    )

        # Per-worker HDF5 file handle cache (fork-safe via worker_init_fn)
        self._h5_cache: dict[int, h5py.File] = {}

        # Build index of (episode_idx, start_frame) pairs
        self.clips: list[tuple[int, int]] = []
        # Per-episode nav command arrays (kept for per-clip weight computation)
        _episode_nav_arrays: list[np.ndarray] = []

        for ep_idx, ep_path in enumerate(self.episode_files):
            with h5py.File(ep_path, "r") as f:
                n_frames = f[S.IMAGES].shape[0]
                nav_cmds = f[S.NAV_COMMAND][:]

                # Truncate at collision frame to exclude post-collision
                # garbage data (vehicle stuck/spinning)
                if S.METADATA_GROUP in f and S.META_COLLISION_FRAME in f[S.METADATA_GROUP].attrs:
                    collision_frame = int(f[S.METADATA_GROUP].attrs[S.META_COLLISION_FRAME])
                    if collision_frame > 0:
                        n_frames = min(n_frames, collision_frame)
                        nav_cmds = nav_cmds[:n_frames]

            _episode_nav_arrays.append(nav_cmds)

            # Generate clip start positions
            if n_frames >= seq_len:
                for start in range(0, n_frames - seq_len + 1, stride):
                    self.clips.append((ep_idx, start))

        if not self.clips:
            raise ValueError(
                f"No valid clips found (seq_len={seq_len} may exceed all episode lengths)"
            )

        # Compute per-clip sampling weights (inverse of LaneFollow frequency)
        self._clip_weights = self._compute_sampling_weights(_episode_nav_arrays)

    def _compute_sampling_weights(self, episode_nav_arrays: list[np.ndarray]) -> torch.Tensor:
        """Compute per-clip sampling weights to balance turn distribution.

        Each clip's weight is derived from its own nav command distribution
        (not the parent episode's), so clips with more turns get genuinely
        higher sampling probability.
        """
        weights = []
        for ep_idx, start in self.clips:
            clip_nav = episode_nav_arrays[ep_idx][start : start + self.seq_len]
            total = len(clip_nav)
            if total == 0:
                weights.append(1.0)
                continue
            # Weight = inverse of LaneFollow fraction (more turns → higher weight)
            # Schema v3: LANEFOLLOW=4; schema v2: LaneFollow=0
            lane_follow = int(np.sum((clip_nav == 4) | (clip_nav == 0)))
            lane_follow_frac = lane_follow / total
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
            weights=self._clip_weights.tolist(),
            num_samples=len(self),
            replacement=True,
        )

    def _get_h5(self, ep_idx: int) -> h5py.File:
        """Return a cached h5py.File handle for the given episode."""
        if ep_idx not in self._h5_cache:
            self._h5_cache[ep_idx] = h5py.File(self.episode_files[ep_idx], "r")
        return self._h5_cache[ep_idx]

    def close(self) -> None:
        """Close all cached HDF5 file handles."""
        cache = getattr(self, "_h5_cache", {})
        for fh in cache.values():
            try:
                fh.close()
            except Exception:
                pass
        cache.clear()

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "DrivingSequenceDataset":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load a single clip.

        Returns:
            Dictionary with:
                images: (seq_len, 3, H, W_total) float32, ImageNet-normalized.
                    W_total = n_cameras * per_camera_width (panoramic).
                controls: (seq_len, 2) float32 [steering, acceleration]
                    acceleration = throttle - brake  ∈ [-1, 1]
                nav_command: (seq_len,) int64  (RoadOption values)
                velocity: (seq_len, 3) float32
        """
        ep_idx, start = self.clips[idx]
        end = start + self.seq_len

        f = self._get_h5(ep_idx)
        images = f[S.IMAGES][start:end]  # (T, H, W_total, 3) uint8
        steering = f[S.STEERING][start:end]
        throttle = f[S.THROTTLE][start:end]
        brake = f[S.BRAKE][start:end]
        nav_cmd = f[S.NAV_COMMAND][start:end]
        velocity = f[S.VELOCITY][start:end]

        # Convert images: uint8 HWC → float32 CHW [0, 1]
        images = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).float() / 255.0

        steering = torch.from_numpy(steering.copy())
        throttle = torch.from_numpy(throttle.copy())
        brake = torch.from_numpy(brake.copy())
        acceleration = throttle - brake
        controls = torch.stack([steering, acceleration], dim=-1)

        nav_cmd = torch.from_numpy(nav_cmd.copy()).long()
        velocity = torch.from_numpy(velocity.copy())

        # Z-score normalise velocity
        velocity = (velocity - VELOCITY_MEAN) / VELOCITY_STD.clamp(min=1e-6)

        # Data augmentation BEFORE normalization (operates on [0, 1] pixel values)
        if self.augment:
            images, controls = self._augment(images, controls)

        # ImageNet normalization (after augmentation so jitter operates on real intensities)
        images = (images - IMAGENET_MEAN) / IMAGENET_STD

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

        Note: Must be called BEFORE ImageNet normalization so that
        brightness/contrast adjustments operate on real pixel values
        in [0, 1], not z-scores.

        Args:
            images: (T, 3, H, W) float32 in [0, 1]
            controls: (T, 2) [steering, acceleration]

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

        # Horizontal flip with steering sign flip.
        # For panoramic images (n_cameras > 1), we must also swap the
        # camera panels: [Left|Centre|Right] → [Right'|Centre'|Left']
        # so the spatial layout stays physically consistent.
        if torch.rand(1).item() < 0.5:
            if self.n_cameras > 1:
                # Split into panels, reverse order, flip each panel
                T, C, H, W_total = images.shape
                panel_w = W_total // self.n_cameras
                panels = [
                    images[:, :, :, i * panel_w : (i + 1) * panel_w] for i in range(self.n_cameras)
                ]
                # Reverse panel order AND horizontally flip each panel
                panels = [p.flip(dims=[-1]) for p in reversed(panels)]
                images = torch.cat(panels, dim=-1)
            else:
                images = images.flip(dims=[-1])
            controls[:, 0] = -controls[:, 0]  # Flip steering sign

        # Clamp to valid [0, 1] range after brightness/contrast adjustments
        images = images.clamp(0.0, 1.0)

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


def driving_worker_init_fn(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` for fork-safe HDF5 caching.

    After a DataLoader worker is forked, inherited h5py.File handles are
    invalid.  This callback clears the cache so handles are re-opened
    lazily in the new process.

    Also seeds the per-worker PyTorch RNG so that data augmentation
    (random flips, colour jitter) is reproducible across runs with the
    same number of workers.

    Usage::

        loader = DataLoader(
            dataset,
            num_workers=4,
            worker_init_fn=driving_worker_init_fn,
        )
    """
    import torch.utils.data as _data

    info = _data.get_worker_info()
    if info is not None:
        ds = info.dataset
        if hasattr(ds, "_h5_cache"):
            ds._h5_cache = {}

        # Deterministic per-worker seed for augmentation reproducibility.
        # base_seed is set by the DataLoader's generator or global seed.
        base_seed = torch.initial_seed() % (2**32)
        torch.manual_seed(base_seed + info.id)
        np.random.seed((base_seed + info.id) % (2**32))
