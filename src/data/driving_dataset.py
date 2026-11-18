"""PyTorch Dataset for sequential driving data.

Loads HDF5 episodes and returns fixed-length clips for RWKV training
with Truncated Backpropagation Through Time (TBPTT).

Each item is a contiguous sequence of frames from a single episode,
suitable for recurrent model training where temporal continuity matters.

Episodes are read through src.data.schema_adapter, which maps every supported schema
to one sample format. Images are 3-camera panoramas (H, 3 * W_cam, 3).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from src.data import labels, schema_adapter
from src.eval.metrics import turn_frames

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

# Every episode is read at this frame rate, the rate of the cached and v7 images.
FPS = 10

# Future-waypoint targets: frames ahead of each frame (0.5-2 s), in the vehicle's frame.
WAYPOINT_OFFSETS = (5, 10, 15, 20)

# Nav command (0-indexed RoadOption) seen in a mirror: LEFT and RIGHT swap, as do the
# two lane changes; STRAIGHT and LANEFOLLOW stay.
MIRRORED_COMMAND = torch.tensor([1, 0, 2, 3, 5, 4])

# A training clip's sampling weight is 1 + TURN_WEIGHT x its share of turning frames
# (|steering| > 0.1); see docs/LABELS.md.
TURN_WEIGHT = 5.0

ClipKey = tuple[int, int]  # (episode index, start frame)

# Open episode files per process. Every open HDF5 file keeps its own metadata and chunk
# caches, so a handle per episode visited grows a worker's memory without bound.
MAX_OPEN_FILES = 16


def mirror(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The sample as it would be recorded in a left-right mirror image of the world.

    Flipping the panorama also moves each side camera's view to the opposite panel.
    Steering and waypoint y change sign and left and right commands swap; every other
    field is unchanged (velocity is world-frame and only its magnitude is used).
    """
    return {
        **sample,
        "images": sample["images"].flip(-1),
        "controls": sample["controls"] * torch.tensor([-1.0, 1.0]),
        "nav_command": MIRRORED_COMMAND[sample["nav_command"]],
        "future_waypoints": sample["future_waypoints"] * torch.tensor([1.0, -1.0]),
    }


class RandomClipSampler(Sampler[ClipKey]):
    """Clips that may start at any usable frame, drawn with replacement.

    `weights[e][s]` is the relative chance of episode e's clip starting at frame s. Each
    epoch draws fresh starts, so every usable frame can be seen, not only those of the
    dataset's fixed clips.
    """

    def __init__(self, weights: list[np.ndarray], num_samples: int) -> None:
        self.weights = torch.from_numpy(np.concatenate(weights))
        self.first = np.cumsum([0] + [len(w) for w in weights])  # first flat index of each episode
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def keys(self, flat: np.ndarray) -> list[ClipKey]:
        """The clip of each index into the episodes' starts, concatenated."""
        episodes = np.searchsorted(self.first, flat, side="right") - 1
        return list(zip(episodes.tolist(), (flat - self.first[episodes]).tolist()))

    def __iter__(self) -> Iterator[ClipKey]:
        yield from self.keys(
            torch.multinomial(self.weights, self.num_samples, replacement=True).numpy()
        )


class EpisodeOrderedSampler(Sampler[ClipKey]):
    """Consecutive clips of each episode, for training that carries state across clips.

    Row b of every batch follows lane b, a run of whole episodes clip after clip, so a
    model can carry row b's state into the next batch unless that clip starts a new
    episode (its start is not the previous clip's end). Each epoch shuffles the episodes
    and shifts each episode's clip boundaries by a random offset, so every usable frame
    can be drawn. Use with a DataLoader of the same `batch_size` and no shuffling.
    """

    def __init__(self, episode_frames: list[int], seq_len: int, batch_size: int) -> None:
        self.episode_frames = episode_frames
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.steps = sum(n // seq_len for n in episode_frames) // batch_size

    def __len__(self) -> int:
        return self.steps * self.batch_size

    def __iter__(self) -> Iterator[ClipKey]:
        stream: list[ClipKey] = []
        for e in torch.randperm(len(self.episode_frames)).tolist():
            n = self.episode_frames[e]
            offset = int(torch.randint(n % self.seq_len + 1, ()))
            stream += [(e, s) for s in range(offset, n - self.seq_len + 1, self.seq_len)]
        for t in range(self.steps):
            for lane in range(self.batch_size):
                yield stream[lane * self.steps + t]


class DrivingSequenceDataset(Dataset):
    """Dataset that returns fixed-length sequential clips from driving episodes.

    Each __getitem__ call returns a contiguous clip of `seq_len` frames from a single
    episode: by index, one of the fixed clips every `stride` frames (`clips`); by
    (episode, start) key, the clip a sampler chose.

    Args:
        data_dir: Path to directory containing episode HDF5 files.
        seq_len: Number of frames per clip (default: 32).
        stride: Frames between the fixed clips; also sets the clips per epoch drawn by
            `get_sampler` (default: 16).
        fps: Frame rate every episode is read at (default: FPS).
        split: One of "train" or "val" (default: "train").
        val_fraction: Fraction of episodes reserved for validation (default: 0.2).
        seed: Random seed for reproducible train/val split (default: 42).
        augment: Whether to apply colour jitter and, to half of the clips, `mirror`
            (default: True for train).
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
        fps: int = FPS,
        augment: bool | None = None,
        resize_height: int | None = None,
        resize_width: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.stride = stride
        self.fps = fps
        self.split = split

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
        self.size = (
            None if resize_height is None or resize_width is None else (resize_height, resize_width)
        )

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

        # Open HDF5 files of this process, least recently used first (see driving_worker_init_fn)
        self._h5_cache: dict[int, h5py.File] = {}

        # Per episode: usable frames, which of them turn, and their navigation commands
        # (derived from the path, see schema_adapter.commands). Fixed clips every `stride`.
        self.episode_frames: list[int] = []
        self.turning: list[np.ndarray] = []
        self.commands: list[np.ndarray] = []
        self.clips: list[ClipKey] = []
        for ep_idx, ep_path in enumerate(self.episode_files):
            with h5py.File(ep_path, "r") as f:
                schema_adapter.check_supported(f, ep_path.name)
                steering, _, command = schema_adapter.read_labels(f, fps)
            self.episode_frames.append(len(steering))
            self.turning.append(turn_frames(steering))
            self.commands.append(command.astype(np.int8))
            self.clips += [(ep_idx, s) for s in range(0, len(steering) - seq_len + 1, stride)]

        if not self.clips:
            raise ValueError(
                f"No valid clips found (seq_len={seq_len} may exceed all episode lengths)"
            )

    def get_sampler(self, weighted: bool = True) -> RandomClipSampler:
        """Training sampler: `len(self)` clips per epoch, starting at any usable frame.

        Weighted, a clip's chance is 1 + TURN_WEIGHT x its share of turning frames;
        otherwise every start is equally likely.
        """
        weights = []
        for turning in self.turning:
            count = np.concatenate([[0], np.cumsum(turning)])
            share = (count[self.seq_len :] - count[: -self.seq_len]) / self.seq_len
            weights.append(1.0 + TURN_WEIGHT * share if weighted else np.ones_like(share))
        return RandomClipSampler(weights, len(self))

    def _get_h5(self, ep_idx: int) -> h5py.File:
        """Episode `ep_idx`, opened once; past MAX_OPEN_FILES the least recently used closes."""
        f = self._h5_cache.pop(ep_idx, None)
        if f is None:
            f = h5py.File(self.episode_files[ep_idx], "r")
            if len(self._h5_cache) >= MAX_OPEN_FILES:
                self._h5_cache.pop(next(iter(self._h5_cache))).close()
        self._h5_cache[ep_idx] = f
        return f

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
    # Stationary frames (horizontal speed below this, m/s) get a reduced loss weight so the
    # easiest third of the data does not dominate; see docs/LABELS.md.
    STATIONARY_SPEED_THRESH: float = 0.14  # ~0.5 km/h
    STATIONARY_LOSS_WEIGHT: float = 0.1

    def __getitem__(self, key: int | ClipKey) -> dict[str, torch.Tensor]:
        """Load the clip `clips[key]`, or the clip (episode, start) a sampler chose.

        Returns:
            Dictionary with:
                images: (seq_len, 3, H_out, W_out) float32, ImageNet-normalized.
                    By default H_out/W_out match the stored frames; resize_* sets
                    them, resizing in uint8 before conversion to float.
                controls: (seq_len, 2) float32 [steering, acceleration]
                    acceleration = throttle - brake  ∈ [-1, 1]
                nav_command: (seq_len,) int64  (RoadOption derived from the path, 0-indexed)
                velocity: (seq_len, 3) float32  (z-score normalised)
                loss_weight: (seq_len,) float32 — per-frame weight
                    (1.0 for moving frames, 0.1 for stationary)
                target_speed: (seq_len,) float32  (normalised ∈ [0, 1];
                    multiply by MAX_SPEED_KMH to recover km/h)
                driving_mode: (seq_len,) int64  (DrivingMode enum, for aux loss)
                future_waypoints: (seq_len, 4, 2) float32  (positions at t+5, t+10,
                    t+15, t+20 frames in the vehicle's frame at t: x forward,
                    y right, metres; for aux loss)
                waypoint_mask: (seq_len, 4) bool  (False where a waypoint lies past
                    the end of the recording and is clamped; left out of the loss)
        """
        ep_idx, start = key if isinstance(key, tuple) else self.clips[key]
        if not 0 <= start <= self.episode_frames[ep_idx] - self.seq_len:
            raise IndexError(f"clip at frame {start} overruns episode {ep_idx}'s usable frames")
        end = start + self.seq_len

        w = schema_adapter.read_window(
            self._get_h5(ep_idx), start, end, max(WAYPOINT_OFFSETS), fps=self.fps, size=self.size
        )

        # uint8 HWC → float32 CHW in [0, 1]; image arithmetic below is in place to bound memory
        images = torch.from_numpy(w.images).permute(0, 3, 1, 2).float().div_(255.0)

        steering = torch.from_numpy(w.steering.copy())
        throttle = torch.from_numpy(w.throttle.copy())
        brake = torch.from_numpy(w.brake.copy())
        acceleration = throttle - brake
        controls = torch.stack([steering, acceleration], dim=-1)

        # The command derived from the path (RoadOption 1-6), 0-indexed for nn.Embedding(6, dim).
        nav_cmd = torch.from_numpy(self.commands[ep_idx][start:end]).long() - 1
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

        if self.augment:
            images = self._jitter(images)

        # ImageNet normalization (after augmentation so jitter operates on real intensities)
        images = images.sub_(IMAGENET_MEAN).div_(IMAGENET_STD)

        # Auxiliary targets
        target_speed_t = torch.from_numpy(w.target_speed.copy()) / MAX_SPEED_KMH
        # DrivingMode values 1-8 (newer modes folded by the adapter) → 0-7 for CrossEntropy.
        driving_mode_t = torch.from_numpy(w.driving_mode.copy()).long() - 1

        future_waypoints = torch.from_numpy(
            labels.ego_waypoints(w.position, w.yaw, WAYPOINT_OFFSETS)
        )  # (T, K, 2): x forward, y right, metres
        # A waypoint past the end of the recording is clamped to its last position, which
        # is not where the vehicle went: it is kept out of the waypoint loss.
        ahead = np.arange(len(w.yaw))[:, None] + np.asarray(WAYPOINT_OFFSETS)
        waypoint_mask = torch.from_numpy(ahead < len(w.position))

        sample = {
            "images": images,
            "controls": controls,
            "nav_command": nav_cmd,
            "velocity": velocity,
            "loss_weight": loss_weight,
            "target_speed": target_speed_t,
            "driving_mode": driving_mode_t,
            "future_waypoints": future_waypoints,
            "waypoint_mask": waypoint_mask,
        }
        if self.augment and torch.rand(1).item() < 0.5:
            sample = mirror(sample)
        return sample

    @staticmethod
    def _jitter(images: torch.Tensor) -> torch.Tensor:
        """Random brightness and contrast, the same for every frame of the clip.

        Applied before ImageNet normalisation, on (T, 3, H, W) values in [0, 1].
        """
        if torch.rand(1).item() < 0.5:
            images.mul_(1.0 + (torch.rand(1).item() - 0.5) * 0.4)  # [0.8, 1.2]
        if torch.rand(1).item() < 0.3:
            mean = images.mean(dim=(2, 3), keepdim=True)
            images.sub_(mean).mul_(1.0 + (torch.rand(1).item() - 0.5) * 0.4).add_(mean)
        return images.clamp_(0.0, 1.0)

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
