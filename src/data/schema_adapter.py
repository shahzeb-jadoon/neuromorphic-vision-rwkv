"""Read driving episodes of every supported schema into one sample format.

Schemas v3 to v6 store the training fields under the same names and units; v4 to v6
only add telemetry. Every difference that matters for training is handled here, so the
dataset sees one format whatever wrote the file:

- v6 experts also write DrivingMode 9 (YieldStop). The sample format keeps the eight v3
  classes, so YieldStop is folded into StopSignStop: both mean the car stopped for a sign.
- A training cache (scripts/build_testbed_cache.py) keeps the full-rate signals and one
  JPEG image every META_IMAGE_STRIDE signal rows.

The readers take `fps`, the rate to read an episode at (by default, the rate of its
stored images): frame k is signal row k * (recorded fps / `fps`). Labels therefore come
from the full-rate signals at the instants of the frames read, whether the images are
raw or cached. Images are resized in uint8, one frame at a time; cached JPEGs first
decode at the smallest DCT scale that covers the requested size. Only an episode's usable
frames (src.data.labels.usable_end) are read for training. The navigation command is
derived from the path driven (src.data.labels.path_commands), since the recorded one can
end before the turn it announces; a recorded command outside RoadOption 1-6 is still an
error, never clamped, as it marks a corrupt file.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import h5py
import numpy as np
import torch
import torch.nn.functional as F
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
    nav_command: np.ndarray  # (T,) RoadOption 1..6 as recorded; training uses commands()
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


def _stored_every(f: h5py.File) -> int:
    """Signal rows per stored image."""
    return int(f[S.METADATA_GROUP].attrs.get(S.META_IMAGE_STRIDE, 1))


def _step(f: h5py.File, fps: int | None) -> int:
    """Signal rows per frame read at `fps`."""
    stored = _stored_every(f)
    if fps is None:
        return stored
    recorded = int(f[S.METADATA_GROUP].attrs[S.META_FPS])
    rows, rest = divmod(recorded, fps)
    if rest or rows % stored:
        raise ValueError(
            f"{f.filename}: cannot be read at {fps} fps "
            f"(signals at {recorded} fps, an image every {stored} rows)"
        )
    return rows


def _rows(f: h5py.File, name: str, start: int, end: int, step: int) -> np.ndarray:
    """Signal `name` at frames [start, end), `step` rows apart."""
    return np.asarray(f[name][start * step : end * step : step])


def _usable_end(f: h5py.File) -> int:
    """End (exclusive) of the signal rows that make valid targets. A golden-subset excerpt
    is used whole: its status and end describe the source episode, not the excerpt."""
    meta = f[S.METADATA_GROUP].attrs
    speed = np.linalg.norm(f[S.VELOCITY][()], axis=1)
    if S.META_GOLDEN_SOURCE_FILE in meta:
        return len(speed)
    return labels.usable_end(
        speed,
        int(meta[S.META_FPS]),
        str(meta.get(S.META_STATUS, "")),
        int(meta.get(S.META_COLLISION_FRAME, 0)),
    )


def usable_frames(f: h5py.File, fps: int | None = None) -> int:
    """Frames that make valid targets, read at `fps`."""
    return -(-_usable_end(f) // _step(f, fps))


def nav_commands(f: h5py.File, fps: int | None = None) -> np.ndarray:
    """RoadOption (1-6) of every usable frame; raises on any other recorded value."""
    nav = np.asarray(f[S.NAV_COMMAND][: _usable_end(f)])
    bad = np.flatnonzero(~np.isin(nav, list(S.VALID_NAV_COMMANDS)))
    if bad.size:
        raise ValueError(
            f"{f.filename}: nav command {nav[bad[0]]} at frame {bad[0]} is not a RoadOption 1-6"
        )
    return nav[:: _step(f, fps)]


def commands(f: h5py.File, fps: int | None = None) -> np.ndarray:
    """Navigation command (RoadOption) of every usable frame, from the path driven."""
    every_row = labels.path_commands(f[S.POSITION][()], f[S.ROTATION][:, 1])
    return every_row[: _usable_end(f) : _step(f, fps)]


def _decode(jpeg: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
    """A cached JPEG as (H, W, 3) uint8, decoded at the smallest DCT scale (1, 1/2, 1/4 or
    1/8) that is still at least `size` (height, width)."""
    image = Image.open(io.BytesIO(jpeg.tobytes()))
    if size is not None:
        image.draft("RGB", (size[1], size[0]))
    return np.asarray(image.convert("RGB"))


def _resize(pixels: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
    """(H, W, 3) uint8 resized to `size` (height, width) bilinearly, as the cache builder
    resizes, without leaving uint8; never enlarged."""
    if size is None or pixels.shape[:2] == size:
        return pixels
    if pixels.shape[0] < size[0] or pixels.shape[1] < size[1]:
        raise ValueError(f"{pixels.shape[0]}x{pixels.shape[1]} images cannot be enlarged to {size}")
    x = torch.from_numpy(pixels).permute(2, 0, 1)[None]
    y = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return np.asarray(y[0].permute(1, 2, 0))


def read_window(
    f: h5py.File,
    start: int,
    end: int,
    lookahead: int,
    *,
    fps: int | None = None,
    size: tuple[int, int] | None = None,
    images: bool = True,
) -> Window:
    """Frames [start, end) read at `fps`, images resized to `size` (height, width)."""
    step = _step(f, fps)
    if not images:
        pixels = np.empty((0, 0, 0, 3), dtype=np.uint8)
    else:
        every = step // _stored_every(f)
        first, last = start * every, (end - 1) * every + 1
        if is_cache(f):
            jpegs = f[S.IMAGES_JPEG][first:last:every]
            pixels = np.stack([_resize(_decode(j, size), size) for j in jpegs])
        else:  # frame by frame, which bounds memory; a strided read of chunked images is slow
            raw = f[S.IMAGES]
            pixels = np.stack([_resize(raw[i], size) for i in range(first, last, every)])
    mode = _rows(f, S.DRIVING_MODE, start, end, step)
    for written, sample in MODE_FOLD.items():
        mode[mode == written] = sample
    frames = -(-int(f[S.POSITION].shape[0]) // step)
    return Window(
        images=pixels,
        steering=_rows(f, S.STEERING, start, end, step),
        throttle=_rows(f, S.THROTTLE, start, end, step),
        brake=_rows(f, S.BRAKE, start, end, step),
        nav_command=_rows(f, S.NAV_COMMAND, start, end, step),
        velocity=_rows(f, S.VELOCITY, start, end, step),
        target_speed=_rows(f, S.TARGET_SPEED, start, end, step),
        driving_mode=mode,
        position=_rows(f, S.POSITION, start, min(end + lookahead, frames), step),
        yaw=_rows(f, S.ROTATION, start, end, step)[:, 1],
    )


def read_labels(f: h5py.File, fps: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Steering, acceleration (throttle - brake) and navigation command (`commands`) over
    the usable frames; raises on a recorded command outside RoadOption 1-6."""
    nav_commands(f)
    nav = commands(f, fps)
    step, n = _step(f, fps), len(nav)
    steering = _rows(f, S.STEERING, 0, n, step)
    acceleration = _rows(f, S.THROTTLE, 0, n, step) - _rows(f, S.BRAKE, 0, n, step)
    return steering, acceleration, nav
