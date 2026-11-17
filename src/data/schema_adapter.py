"""Read driving episodes of every supported schema into one sample format.

Schemas v3 to v6 store the training fields under the same names and units; v4 to v6
only add telemetry. Every difference that matters for training is handled here, so the
dataset sees one format whatever wrote the file:

- v6 experts also write DrivingMode 9 (YieldStop). The sample format keeps the eight v3
  classes, so YieldStop is folded into StopSignStop: both mean the car stopped for a sign.
- A training cache (scripts/build_testbed_cache.py) keeps the full-rate signals and one
  JPEG image every META_IMAGE_STRIDE source frames. It reads as an episode at that lower
  rate: frame k is source frame k * stride.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import h5py
import numpy as np
from PIL import Image

from src.data import hdf5_schema as S

SUPPORTED_SCHEMAS = range(3, 7)
MODE_FOLD = {9: 8}


@dataclass(frozen=True)
class Window:
    """Frames [start, end) of one episode. `position` continues `lookahead` frames past
    `end`, clamped at the episode end, for future-waypoint targets."""

    images: np.ndarray  # (T, H, W, 3) uint8
    steering: np.ndarray  # (T,) float32
    throttle: np.ndarray  # (T,) float32
    brake: np.ndarray  # (T,) float32
    nav_command: np.ndarray  # (T,) RoadOption 1..6
    velocity: np.ndarray  # (T, 3) float32, m/s, world frame
    target_speed: np.ndarray  # (T,) float32, km/h
    driving_mode: np.ndarray  # (T,) DrivingMode 1..8
    position: np.ndarray  # (T + lookahead, 3) float32, metres, world frame


def schema_version(f: h5py.File) -> int:
    if S.METADATA_GROUP not in f:
        return 0
    return int(f[S.METADATA_GROUP].attrs.get(S.META_SCHEMA_VERSION, 0))


def check_supported(f: h5py.File, name: str) -> None:
    version = schema_version(f)
    if version not in SUPPORTED_SCHEMAS:
        first, last = SUPPORTED_SCHEMAS.start, SUPPORTED_SCHEMAS.stop - 1
        raise ValueError(f"{name}: schema v{version} is not supported (v{first} to v{last})")


def is_cache(f: h5py.File) -> bool:
    return S.IMAGES_JPEG in f


def _stride(f: h5py.File) -> int:
    return int(f[S.METADATA_GROUP].attrs[S.META_IMAGE_STRIDE]) if is_cache(f) else 1


def _frames(f: h5py.File) -> int:
    """Frames with a signal row: every image frame of a cache, the signal length otherwise."""
    return int(f[S.IMAGES_JPEG].shape[0]) if is_cache(f) else int(f[S.POSITION].shape[0])


def _rows(f: h5py.File, name: str, start: int, end: int) -> np.ndarray:
    """Signal `name` at frames [start, end)."""
    stride = _stride(f)
    if stride == 1:
        return np.asarray(f[name][start:end])
    return np.asarray(f[name][start * stride : (end - 1) * stride + 1 : stride])


def usable_frames(f: h5py.File) -> int:
    """Frame count, cut at the collision frame when the episode ended in one."""
    n = int((f[S.IMAGES_JPEG] if is_cache(f) else f[S.IMAGES]).shape[0])
    meta = f[S.METADATA_GROUP].attrs if S.METADATA_GROUP in f else {}
    collision = int(meta.get(S.META_COLLISION_FRAME, 0))
    stride = _stride(f)
    return min(n, -(-collision // stride)) if collision > 0 else n


def nav_commands(f: h5py.File) -> np.ndarray:
    return _rows(f, S.NAV_COMMAND, 0, usable_frames(f))


def read_window(f: h5py.File, start: int, end: int, lookahead: int) -> Window:
    if is_cache(f):
        images = np.stack(
            [
                np.asarray(Image.open(io.BytesIO(j.tobytes())).convert("RGB"))
                for j in f[S.IMAGES_JPEG][start:end]
            ]
        )
    else:
        images = f[S.IMAGES][start:end]
    mode = _rows(f, S.DRIVING_MODE, start, end)
    for written, sample in MODE_FOLD.items():
        mode[mode == written] = sample
    return Window(
        images=images,
        steering=_rows(f, S.STEERING, start, end),
        throttle=_rows(f, S.THROTTLE, start, end),
        brake=_rows(f, S.BRAKE, start, end),
        nav_command=_rows(f, S.NAV_COMMAND, start, end),
        velocity=_rows(f, S.VELOCITY, start, end),
        target_speed=_rows(f, S.TARGET_SPEED, start, end),
        driving_mode=mode,
        position=_rows(f, S.POSITION, start, min(end + lookahead, _frames(f))),
    )


def read_labels(f: h5py.File) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Steering, acceleration (throttle - brake) and nav command over the usable frames."""
    n = usable_frames(f)
    steering = _rows(f, S.STEERING, 0, n)
    acceleration = _rows(f, S.THROTTLE, 0, n) - _rows(f, S.BRAKE, 0, n)
    return steering, acceleration, _rows(f, S.NAV_COMMAND, 0, n)
