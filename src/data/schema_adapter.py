"""Read driving episodes of every supported schema into one sample format.

Schemas v3 to v6 store the training fields under the same names and units; v4 to v6
only add telemetry. Every difference that matters for training is handled here, so the
dataset sees one format whatever wrote the file:

- v6 experts also write DrivingMode 9 (YieldStop). The sample format keeps the eight v3
  classes, so YieldStop is folded into StopSignStop: both mean the car stopped for a sign.
- A training cache (scripts/build_testbed_cache.py) keeps the full-rate signals and one
  JPEG image every META_IMAGE_STRIDE source frames. It reads as an episode at that lower
  rate: frame k is source frame k * stride.

Only an episode's usable frames (src.data.labels.usable_end) are read for training, and
a nav command outside RoadOption 1-6 there is an error, never clamped.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import h5py
import numpy as np
from PIL import Image

from src.data import hdf5_schema as S
from src.data import labels

SUPPORTED_SCHEMAS = range(3, 7)
MODE_FOLD = {9: 8}


@dataclass(frozen=True)
class Window:
    """Frames [start, end) of one episode. `position` continues `lookahead` frames past
    `end`, clamped at the episode end, for future-waypoint targets. `images` is empty
    when the window was read without them."""

    images: np.ndarray  # (T, H, W, 3) uint8
    steering: np.ndarray  # (T,) float32
    throttle: np.ndarray  # (T,) float32
    brake: np.ndarray  # (T,) float32
    nav_command: np.ndarray  # (T,) RoadOption 1..6
    velocity: np.ndarray  # (T, 3) float32, m/s, world frame
    target_speed: np.ndarray  # (T,) float32, km/h
    driving_mode: np.ndarray  # (T,) DrivingMode 1..8
    position: np.ndarray  # (T + lookahead, 3) float32, metres, world frame
    yaw: np.ndarray  # (T,) float32, degrees, world frame


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
    """Frames that make valid targets, at this file's rate. A golden-subset excerpt is
    used whole: its status and end describe the source episode, not the excerpt."""
    meta = f[S.METADATA_GROUP].attrs
    speed = np.linalg.norm(f[S.VELOCITY][()], axis=1)
    if S.META_GOLDEN_SOURCE_FILE in meta:
        end = len(speed)
    else:
        end = labels.usable_end(
            speed,
            int(meta[S.META_FPS]),
            str(meta.get(S.META_STATUS, "")),
            int(meta.get(S.META_COLLISION_FRAME, 0)),
        )
    return min(_frames(f), -(-end // _stride(f)))


def nav_commands(f: h5py.File) -> np.ndarray:
    """RoadOption (1-6) of every usable frame; raises on any other value."""
    nav = _rows(f, S.NAV_COMMAND, 0, usable_frames(f))
    bad = np.flatnonzero(~np.isin(nav, list(S.VALID_NAV_COMMANDS)))
    if bad.size:
        raise ValueError(
            f"{f.filename}: nav command {nav[bad[0]]} at frame {bad[0]} is not a RoadOption 1-6"
        )
    return nav


def read_window(f: h5py.File, start: int, end: int, lookahead: int, images: bool = True) -> Window:
    if not images:
        pixels = np.empty((0, 0, 0, 3), dtype=np.uint8)
    elif is_cache(f):
        pixels = np.stack(
            [
                np.asarray(Image.open(io.BytesIO(j.tobytes())).convert("RGB"))
                for j in f[S.IMAGES_JPEG][start:end]
            ]
        )
    else:
        pixels = f[S.IMAGES][start:end]
    mode = _rows(f, S.DRIVING_MODE, start, end)
    for written, sample in MODE_FOLD.items():
        mode[mode == written] = sample
    return Window(
        images=pixels,
        steering=_rows(f, S.STEERING, start, end),
        throttle=_rows(f, S.THROTTLE, start, end),
        brake=_rows(f, S.BRAKE, start, end),
        nav_command=_rows(f, S.NAV_COMMAND, start, end),
        velocity=_rows(f, S.VELOCITY, start, end),
        target_speed=_rows(f, S.TARGET_SPEED, start, end),
        driving_mode=mode,
        position=_rows(f, S.POSITION, start, min(end + lookahead, _frames(f))),
        yaw=_rows(f, S.ROTATION, start, end)[:, 1],
    )


def read_labels(f: h5py.File) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Steering, acceleration (throttle - brake) and nav command over the usable frames."""
    nav = nav_commands(f)
    n = len(nav)
    steering = _rows(f, S.STEERING, 0, n)
    acceleration = _rows(f, S.THROTTLE, 0, n) - _rows(f, S.BRAKE, 0, n)
    return steering, acceleration, nav
