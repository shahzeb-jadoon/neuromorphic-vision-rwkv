"""Read driving episodes of every supported schema into one sample format.

Schemas v3 to v6 store the training fields under the same names and units; v4 to v6
only add telemetry. Every difference that matters for training is handled here, so the
dataset sees one format whatever wrote the file:

- v6 experts also write DrivingMode 9 (YieldStop). The sample format keeps the eight v3
  classes, so YieldStop is folded into StopSignStop: both mean the car stopped for a sign.
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np

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


def usable_frames(f: h5py.File) -> int:
    """Frame count, cut at the collision frame when the episode ended in one."""
    n = int(f[S.IMAGES].shape[0])
    meta = f[S.METADATA_GROUP].attrs if S.METADATA_GROUP in f else {}
    collision = int(meta.get(S.META_COLLISION_FRAME, 0))
    return min(n, collision) if collision > 0 else n


def nav_commands(f: h5py.File) -> np.ndarray:
    return np.asarray(f[S.NAV_COMMAND][: usable_frames(f)])


def read_window(f: h5py.File, start: int, end: int, lookahead: int) -> Window:
    mode = f[S.DRIVING_MODE][start:end]
    for written, sample in MODE_FOLD.items():
        mode[mode == written] = sample
    future_end = min(end + lookahead, f[S.POSITION].shape[0])
    return Window(
        images=f[S.IMAGES][start:end],
        steering=f[S.STEERING][start:end],
        throttle=f[S.THROTTLE][start:end],
        brake=f[S.BRAKE][start:end],
        nav_command=f[S.NAV_COMMAND][start:end],
        velocity=f[S.VELOCITY][start:end],
        target_speed=f[S.TARGET_SPEED][start:end],
        driving_mode=mode,
        position=f[S.POSITION][start:future_end],
    )


def read_labels(f: h5py.File) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Steering, acceleration (throttle - brake) and nav command over the usable frames."""
    n = usable_frames(f)
    steering = np.asarray(f[S.STEERING][:n])
    acceleration = np.asarray(f[S.THROTTLE][:n]) - np.asarray(f[S.BRAKE][:n])
    return steering, acceleration, np.asarray(f[S.NAV_COMMAND][:n])
