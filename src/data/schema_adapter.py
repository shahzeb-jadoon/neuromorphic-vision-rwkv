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
frames (src.data.labels.usable_end) are read for training.

The navigation command, target point and situations (src.data.labels.route) are derived
from the path driven and the junction outlines of its town, read from JUNCTIONS_FILE
beside the episodes (scripts/export_junctions.py), since the recorded command can end
before the turn it announces. The same file holds the stop lines of the town's traffic
lights, which find the red-light stops the expert made in the wrong place
(src.data.labels.misplaced_stops). A recorded command outside RoadOption 1-6 is still an
error, never clamped, as it marks a corrupt file.
"""

from __future__ import annotations

import functools
import io
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data import hdf5_schema as S
from src.data import labels

SUPPORTED_SCHEMAS = range(3, 7)
MODE_FOLD = {9: 8}
JUNCTIONS_FILE = "junctions.json"  # junction outlines and stop lines per town, beside the episodes


@dataclass(frozen=True)
class Window:
    """Frames [start, end) of one episode. `position` continues `lookahead` frames past
    `end`, clamped at the episode end, for future-waypoint targets. `images` is empty
    when the window was read without them."""

    images: np.ndarray  # (T, H, W, 3) uint8
    steering: np.ndarray  # (T,) float32
    throttle: np.ndarray  # (T,) float32
    brake: np.ndarray  # (T,) float32
    nav_command: np.ndarray  # (T,) RoadOption 1..6 as recorded; training uses read_labels()
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


Junctions = tuple[list[tuple[int, np.ndarray]], np.ndarray]  # outlines by id, (M, 3) stop lines


@functools.lru_cache(maxsize=8)
def _junction_file(directory: str) -> dict[str, Junctions]:
    path = Path(directory) / JUNCTIONS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; scripts/export_junctions.py writes the junction outlines "
            "that the navigation labels need"
        )
    towns = json.loads(path.read_text(encoding="utf-8"))
    if any(not isinstance(town, dict) for town in towns.values()):
        raise ValueError(f"{path} has no stop lines; export it again with export_junctions.py")
    return {
        name: (
            [(j["id"], np.asarray(j["outline"], np.float64)) for j in town["junctions"]],
            np.asarray(town["stop_lines"], np.float64).reshape(-1, 3),
        )
        for name, town in towns.items()
    }


def town_junctions(town: str, directory: str | Path) -> Junctions:
    """The junction outlines and traffic-light stop lines of `town`, from JUNCTIONS_FILE in
    `directory`."""
    towns = _junction_file(str(directory))
    if town not in towns:
        path = Path(directory) / JUNCTIONS_FILE
        raise ValueError(f"{path} has no junction outlines for {town!r}")
    return towns[town]


def _junctions(f: h5py.File) -> Junctions:
    """The junction outlines and traffic-light stop lines of the episode's town."""
    town = f[S.METADATA_GROUP].attrs.get(S.META_TOWN, "")
    town = town.decode() if isinstance(town, bytes) else str(town)
    return town_junctions(town, Path(f.filename).parent)


def _fold(mode: np.ndarray) -> np.ndarray:
    for written, sample in MODE_FOLD.items():
        mode[mode == written] = sample
    return mode


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
    mode = _fold(_rows(f, S.DRIVING_MODE, start, end, step))
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


@dataclass(frozen=True)
class Labels:
    """Per-frame labels of an episode's usable frames, computed from its full-rate signals."""

    steering: np.ndarray  # (N,) float32
    acceleration: np.ndarray  # (N,) float32, throttle - brake
    speed: np.ndarray  # (N,) float32, horizontal, m/s
    command: np.ndarray  # (N,) RoadOption (src.data.labels.route)
    target_point: np.ndarray  # (N, 2) float32, metres, x forward and y right
    sparse_target_point: np.ndarray  # (N, 2) float32, the next sparse route point, likewise
    situation: np.ndarray  # (N, len(labels.SITUATIONS)) bool
    excluded: np.ndarray  # (N,) bool: kept out of the loss (src.data.labels.misplaced_stops)


def read_labels(f: h5py.File, fps: int | None = None) -> Labels:
    """Labels of the usable frames read at `fps`; raises on a recorded command outside
    RoadOption 1-6."""
    nav_commands(f)
    outlines, stop_lines = _junctions(f)
    position, yaw = f[S.POSITION][()], f[S.ROTATION][:, 1]
    path = labels.route(position, yaw, labels.junction_ids(position[:, :2], outlines))
    step = _step(f, fps)
    rows = slice(0, _usable_end(f), step)
    # Usable rows end before the padding that some v3 files keep in traffic_light_distance.
    speed = np.linalg.norm(f[S.VELOCITY][rows, :2], axis=1)
    light_state, light_distance = f[S.TRAFFIC_LIGHT_STATE][rows], f[S.TRAFFIC_LIGHT_DISTANCE][rows]
    vehicle_gap = labels.vehicle_gaps(f[S.NEARBY_VEHICLES][rows])
    situation = labels.situations(
        path[rows],
        speed,
        light_state,
        light_distance,
        f[S.STOP_SIGN_DISTANCE][rows],
        vehicle_gap,
        labels.pedestrian_distances(f[S.NEARBY_PEDESTRIANS][rows]),
    )
    rate = int(f[S.METADATA_GROUP].attrs[S.META_FPS]) // step
    return Labels(
        steering=f[S.STEERING][rows],
        acceleration=(f[S.THROTTLE][rows] - f[S.BRAKE][rows]),
        speed=speed,
        command=path.command[rows],
        target_point=path.target[rows],
        sparse_target_point=path.sparse_target[rows],
        situation=situation,
        excluded=labels.misplaced_stops(
            position[rows],
            yaw[rows],
            speed,
            light_state,
            light_distance,
            vehicle_gap,
            stop_lines,
            rate,
        ),
    )
