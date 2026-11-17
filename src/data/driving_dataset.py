"""PyTorch Dataset for sequential driving data.

Loads HDF5 episodes and returns fixed-length clips for RWKV training
with Truncated Backpropagation Through Time (TBPTT).

Each item is a contiguous sequence of frames from a single episode,
suitable for recurrent model training where temporal continuity matters.

Episodes are read through src.data.schema_adapter, which maps every supported schema
to one sample format. Images are 3-camera panoramas (H, 3 * W_cam, 3), so the horizontal
flip augmentation also swaps the left and right camera panels.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler

from src.data import labels, schema_adapter

# ImageNet normalization constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Velocity z-score normalisation constants.
# Computed from cleaned v1 corpus (1,099 episodes, 3,624,673 frames)
# on 2026-04-21 via Welford's algorithm.
# World-frame (vx, vy, vz) in m/s.
VELOCITY_STATS_VERSION = "v1_cleaned_2026_04_21"
VELOCITY_MEAN = torch.tensor([-0.069918, -0.101425, 0.001077])
VELOCITY_STD = torch.tensor([4.864434, 3.784082, 0.128879])

# Speed normalisation: raw target_speed is stored in km/h.
# Dividing by MAX_SPEED_KMH maps it to [0, 1] so speed_loss MSE stays on
# the same scale as control_loss (~0.01–0.1) rather than 100–10,000×.
# To recover km/h at inference: speed_pred_kmh = model_output * MAX_SPEED_KMH
MAX_SPEED_KMH: float = 120.0

# Future-waypoint targets: frames ahead of each frame, in the vehicle's frame.
WAYPOINT_OFFSETS = (5, 10, 15, 20)


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
        resize_height: Optional output height for on-the-fly resize.
        resize_width: Optional output width for on-the-fly resize.
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
        resize_height: int | None = None,
        resize_width: int | None = None,
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

        if (resize_height is None) != (resize_width is None):
            raise ValueError("resize_height and resize_width must both be set, or both be None")
        if resize_height is not None and resize_height <= 0:
            raise ValueError(f"resize_height must be positive, got {resize_height}")
        if resize_width is not None and resize_width <= 0:
            raise ValueError(f"resize_width must be positive, got {resize_width}")
        self.resize_height = resize_height
        self.resize_width = resize_width

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

        # Per-worker HDF5 file handle cache (fork-safe via worker_init_fn)
        self._h5_cache: dict[int, h5py.File] = {}

        # Build index of (episode_idx, start_frame) pairs
        self.clips: list[tuple[int, int]] = []
        # Per-episode nav command arrays (kept for per-clip weight computation)
        _episode_nav_arrays: list[np.ndarray] = []

        for ep_idx, ep_path in enumerate(self.episode_files):
            with h5py.File(ep_path, "r") as f:
                schema_adapter.check_supported(f, ep_path.name)
                # Frames after a collision (vehicle stuck or spinning) are dropped.
                n_frames = schema_adapter.usable_frames(f)
                nav_cmds = schema_adapter.nav_commands(f)

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

    # Stationary-frame loss down-weight.
    # Frames with speed < this threshold (m/s) get reduced loss weight
    # to prevent the 42% stationary majority from dominating gradients.
    STATIONARY_SPEED_THRESH: float = 0.14  # ~0.5 km/h
    STATIONARY_LOSS_WEIGHT: float = 0.1

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load a single clip.

        Returns:
            Dictionary with:
                images: (seq_len, 3, H_out, W_out) float32, ImageNet-normalized.
                    By default H_out/W_out match stored panoramic frames
                    (H=320, W=3*camera_width). If resize_* is set, images are
                    resized on the fly for ablations and memory tuning.
                controls: (seq_len, 2) float32 [steering, acceleration]
                    acceleration = throttle - brake  ∈ [-1, 1]
                nav_command: (seq_len,) int64  (RoadOption values, 0-indexed)
                velocity: (seq_len, 3) float32  (z-score normalised)
                loss_weight: (seq_len,) float32 — per-frame weight
                    (1.0 for moving frames, 0.1 for stationary)
                target_speed: (seq_len,) float32  (normalised ∈ [0, 1];
                    multiply by MAX_SPEED_KMH to recover km/h)
                driving_mode: (seq_len,) int64  (DrivingMode enum, for aux loss)
                future_waypoints: (seq_len, 4, 2) float32  (positions at t+5, t+10,
                    t+15, t+20 frames in the vehicle's frame at t: x forward,
                    y right, metres; for aux loss)
        """
        ep_idx, start = self.clips[idx]
        end = start + self.seq_len

        w = schema_adapter.read_window(self._get_h5(ep_idx), start, end, max(WAYPOINT_OFFSETS))

        # Convert images: uint8 HWC → float32 CHW [0, 1]
        images = torch.from_numpy(w.images.copy()).permute(0, 3, 1, 2).float() / 255.0

        steering = torch.from_numpy(w.steering.copy())
        throttle = torch.from_numpy(w.throttle.copy())
        brake = torch.from_numpy(w.brake.copy())
        acceleration = throttle - brake
        controls = torch.stack([steering, acceleration], dim=-1)

        # Remap RoadOption values 1-6 → 0-5 for nn.Embedding(6, dim).
        # Schema v3 values ∈ {1..4} (LEFT..LANEFOLLOW), extended to {1..6}
        # with CHANGELANELEFT=5, CHANGELANERIGHT=6 via extrapolation or
        # future expert collections.
        nav_cmd = torch.from_numpy(w.nav_command.copy()).long() - 1
        velocity = torch.from_numpy(w.velocity.copy())

        # Per-frame loss weight: down-weight stationary frames.
        # Compute speed from raw velocity BEFORE z-score normalisation.
        speed = velocity[:, :2].norm(dim=-1)  # horizontal speed (m/s)
        loss_weight = torch.where(
            speed < self.STATIONARY_SPEED_THRESH,
            torch.tensor(self.STATIONARY_LOSS_WEIGHT),
            torch.tensor(1.0),
        )

        # Z-score normalise velocity
        velocity = (velocity - VELOCITY_MEAN) / VELOCITY_STD.clamp(min=1e-6)

        # Data augmentation BEFORE normalization (operates on [0, 1] pixel values)
        if self.augment:
            images, controls = self._augment(images, controls)

        # Optional on-the-fly resize for resolution ablation experiments.
        if self.resize_height is not None and self.resize_width is not None:
            images = F.interpolate(
                images,
                size=(self.resize_height, self.resize_width),
                mode="bilinear",
                align_corners=False,
            )

        # ImageNet normalization (after augmentation so jitter operates on real intensities)
        images = (images - IMAGENET_MEAN) / IMAGENET_STD

        # Auxiliary targets
        target_speed_t = torch.from_numpy(w.target_speed.copy()) / MAX_SPEED_KMH
        # DrivingMode values 1-8 (newer modes folded by the adapter) → 0-7 for CrossEntropy.
        driving_mode_t = torch.from_numpy(w.driving_mode.copy()).long() - 1

        future_waypoints = torch.from_numpy(
            labels.ego_waypoints(w.position, w.yaw, WAYPOINT_OFFSETS)
        )  # (T, K, 2): x forward, y right, metres

        return {
            "images": images,
            "controls": controls,
            "nav_command": nav_cmd,
            "velocity": velocity,
            "loss_weight": loss_weight,
            "target_speed": target_speed_t,
            "driving_mode": driving_mode_t,
            "future_waypoints": future_waypoints,
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
