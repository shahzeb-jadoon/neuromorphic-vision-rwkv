"""Expert data collector for CARLA autonomous driving episodes.

Drives a custom **PrivilegedExpert** agent through pre-planned routes
while recording per-frame observations (3-camera panoramic, controls,
navigation commands, ego state, nearby-actor metadata) to HDF5 files.

Schema version: 3 (see ``src/data/hdf5_schema.py``).

Key differences from the v2 collector:
    - Uses ``PrivilegedExpert`` (custom PID + privileged perception)
      instead of CARLA's TrafficManager autopilot.
    - Spawns 3 RGB cameras (left / centre / right) for panoramic input.
    - Stores ego position, rotation, traffic light state, and
      nearby-actor metadata for future-proofing (GPS, LiDAR, etc.).

Handles three failure modes:
    - CARLA crash recovery (reconnect with retries)
    - Episode termination (completed / collision / timeout / memory_limit)
    - Campaign resumption via collection_manifest.json
"""

from __future__ import annotations

from src.data import hdf5_schema as S

import ctypes
import ctypes.util
import gc
import hashlib
import json
import logging
import math
import os
import queue
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# RoadOption is a pure-Python IntEnum with no carla dependency,
# so import it unconditionally.
from agents.navigation.local_planner import RoadOption

# ---------------------------------------------------------------------------
# posix_fadvise — evict clean page cache after HDF5 flushes
# ---------------------------------------------------------------------------
# Without this, every HDF5 write stays in Linux page cache indefinitely,
# inflating VmmemWSL to 19+ GB on Windows.  With periodic
# fadvise(DONTNEED) the cache stays below ~2 MB.
# See docs/MEMORY_DIAGNOSTICS_FINDINGS.md §15 for full analysis.
POSIX_FADV_DONTNEED = 4
_CACHE_EVICT_INTERVAL = 100  # frames between flush+evict cycles

_libc = None
try:
    _libc_name = ctypes.util.find_library("c")
    if _libc_name:
        _libc = ctypes.CDLL(_libc_name, use_errno=True)
except OSError:
    pass


def _posix_fadvise(fd: int, offset: int, length: int, advice: int) -> int:
    """Call posix_fadvise(3) via ctypes.  Returns 0 on success."""
    if _libc is None:
        return -1
    # int posix_fadvise(int fd, off_t offset, off_t length, int advice)
    return int(_libc.posix_fadvise(fd, ctypes.c_int64(offset), ctypes.c_int64(length), advice))


try:
    import carla

    from agents.expert.privileged_expert import (  # noqa: F401
        ExpertConfig,
        FrameInfo,
        PrivilegedExpert,
    )
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    CARLA_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CARLA_AVAILABLE = False

    # Provide a lightweight stub so EpisodeConfig (which references
    # ExpertConfig in its default_factory) can still be defined.
    @dataclass
    class ExpertConfig:  # type: ignore[no-redef]
        """Stub — real config lives in agents.expert.privileged_expert."""

        target_speed: float = 30.0


logger = logging.getLogger(__name__)


@dataclass
class EpisodeConfig:
    """Configuration for a single data collection episode."""

    town: str = "Town01"
    weather: str = "ClearNoon"
    n_vehicles: int = 50
    n_walkers: int = 30
    max_frames: int = 2000
    timeout_speed: float = 0.1  # m/s — below this for timeout_duration → discard
    timeout_duration: float = 75.0  # seconds stuck before early termination
    fps: int = 20

    # 3-camera panoramic setup
    camera_width: int = 640  # per-camera width
    camera_height: int = 320  # per-camera height
    camera_fov: int = 60
    camera_yaws: tuple[float, ...] = (-60.0, 0.0, 60.0)  # left, centre, right

    # Camera mount (vehicle frame, metres)
    camera_x: float = 1.5
    camera_z: float = 2.4

    # Visualization / recording
    spectator_follow: bool = False  # Attach CARLA spectator camera to ego vehicle
    spectator_offset: tuple[float, float, float] = (-8.0, 0.0, 5.0)  # (x, y, z) behind/above
    record_simulation: bool = False  # Use CARLA built-in recorder
    debug_draw: bool = False  # Draw route + trajectory in CARLA viewport

    # Expert agent
    expert_config: ExpertConfig = field(default_factory=ExpertConfig)

    # F3: RSS memory guard — CARLA's C++ client leaks anonymous memory
    # on most maps.  When RSS exceeds this limit (MB), the episode is
    # stopped gracefully and the worker exits so the orchestrator can
    # restart it.  0 = disabled.
    rss_limit_mb: int = 8192  # 8 GB default


@dataclass
class EpisodeResult:
    """Result of a single episode collection."""

    status: str  # "completed", "collision", "crashed", "timeout", "memory_limit"
    filename: str | None = None
    episode_hash: str | None = None
    n_frames: int = 0
    town: str = ""
    weather: str = ""
    n_vehicles: int = 0
    n_walkers: int = 0
    config_hash: str = ""
    nav_distribution: dict[str, int] = field(default_factory=dict)
    collision_frame: int | None = None
    collision_other_actor: str | None = None
    error_message: str | None = None
    duration_seconds: float = 0.0
    recording_path: str | None = None


# ---------------------------------------------------------------------------
# Streaming HDF5 writer — writes one frame at a time to avoid peak memory
# ---------------------------------------------------------------------------


class _StreamingEpisodeWriter:
    """Incrementally writes episode data to an HDF5 file.

    Instead of accumulating every frame in Python lists and calling
    ``np.stack`` at the end (which doubles peak memory to ~14 GB for a
    4000-frame panoramic episode), this writer pre-allocates resizable
    HDF5 datasets and appends each frame directly.

    Peak image memory: **1 frame** (~1.76 MB) instead of **N frames**.
    """

    def __init__(
        self,
        filepath: Path,
        config: EpisodeConfig,
        max_frames: int,
    ) -> None:
        self.filepath = filepath
        self.config = config
        self.n_written = 0

        pano_w = config.camera_width * len(config.camera_yaws)
        h, w, c = config.camera_height, pano_w, 3

        self._h5 = h5py.File(filepath, "w")

        # Pre-allocate with maxshape=(None, ...) for extensibility.
        # Initial shape is (max_frames, ...) — we resize down at finalize.
        self._ds_images = self._h5.create_dataset(
            S.IMAGES,
            shape=(max_frames, h, w, c),
            dtype=np.uint8,
            maxshape=(None, h, w, c),
            chunks=(1, h, w, c),
            compression="gzip",
            compression_opts=1,
        )
        self._ds_steering = self._h5.create_dataset(
            S.STEERING, shape=(max_frames,), dtype=np.float32, maxshape=(None,)
        )
        self._ds_throttle = self._h5.create_dataset(
            S.THROTTLE, shape=(max_frames,), dtype=np.float32, maxshape=(None,)
        )
        self._ds_brake = self._h5.create_dataset(
            S.BRAKE, shape=(max_frames,), dtype=np.float32, maxshape=(None,)
        )
        self._ds_nav = self._h5.create_dataset(
            S.NAV_COMMAND, shape=(max_frames,), dtype=np.int32, maxshape=(None,)
        )
        self._ds_pos = self._h5.create_dataset(
            S.POSITION, shape=(max_frames, 3), dtype=np.float32, maxshape=(None, 3)
        )
        self._ds_rot = self._h5.create_dataset(
            S.ROTATION, shape=(max_frames, 3), dtype=np.float32, maxshape=(None, 3)
        )
        self._ds_vel = self._h5.create_dataset(
            S.VELOCITY, shape=(max_frames, 3), dtype=np.float32, maxshape=(None, 3)
        )
        self._ds_tl = self._h5.create_dataset(
            S.TRAFFIC_LIGHT_STATE, shape=(max_frames,), dtype=np.int32, maxshape=(None,)
        )
        self._ds_tl_dist = self._h5.create_dataset(
            S.TRAFFIC_LIGHT_DISTANCE, shape=(max_frames,), dtype=np.float32, maxshape=(None,)
        )
        self._ds_ss = self._h5.create_dataset(
            S.STOP_SIGN_DISTANCE, shape=(max_frames,), dtype=np.float32, maxshape=(None,)
        )
        self._ds_ts = self._h5.create_dataset(
            S.TARGET_SPEED, shape=(max_frames,), dtype=np.float32, maxshape=(None,)
        )
        self._ds_dm = self._h5.create_dataset(
            S.DRIVING_MODE, shape=(max_frames,), dtype=np.int32, maxshape=(None,)
        )
        self._ds_ncorr = self._h5.create_dataset(
            S.N_CORRIDOR_SEGMENTS, shape=(max_frames,), dtype=np.int32, maxshape=(None,)
        )
        self._ds_wpbs = self._h5.create_dataset(
            S.WP_BUFFER_SIZE, shape=(max_frames,), dtype=np.int32, maxshape=(None,)
        )
        self._ds_wpbh = self._h5.create_dataset(
            S.WP_BUFFER_HEAD_DIST, shape=(max_frames,), dtype=np.float32, maxshape=(None,)
        )
        self._ds_sat = self._h5.create_dataset(
            S.SAT_DETECTIONS, shape=(max_frames,), dtype=np.int32, maxshape=(None,)
        )
        self._ds_lat = self._h5.create_dataset(
            S.LATERAL_DETECTIONS, shape=(max_frames,), dtype=np.int32, maxshape=(None,)
        )

        # Variable-length string datasets for actor JSON
        dt_str = h5py.string_dtype()
        self._ds_nveh = self._h5.create_dataset(
            S.NEARBY_VEHICLES, shape=(max_frames,), dtype=dt_str, maxshape=(None,)
        )
        self._ds_nped = self._h5.create_dataset(
            S.NEARBY_PEDESTRIANS, shape=(max_frames,), dtype=dt_str, maxshape=(None,)
        )

    # -- Per-frame append --------------------------------------------------

    def _flush_and_evict_cache(self) -> None:
        """Flush HDF5 buffers to disk and evict pages from Linux page cache.

        Without this, every HDF5 write stays in the kernel's page cache,
        inflating VmmemWSL to 19+ GB on Windows/WSL2.  Calling
        ``posix_fadvise(POSIX_FADV_DONTNEED)`` after each flush tells the
        kernel to drop the clean pages, keeping cache below ~2 MB.

        See docs/MEMORY_DIAGNOSTICS_FINDINGS.md §15 for the full analysis.
        """
        try:
            self._h5.flush()
            fd = self._h5.id.get_vfd_handle()
            os.fsync(fd)
            _posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
        except Exception:
            # Non-fatal — data is still written, just cached longer.
            pass

    def append_frame(
        self,
        panoramic: np.ndarray,
        control: Any,
        info: Any,
    ) -> None:
        """Write one frame of data directly to the HDF5 datasets."""
        i = self.n_written

        self._ds_images[i] = panoramic
        self._ds_steering[i] = float(control.steer)
        self._ds_throttle[i] = float(control.throttle)
        self._ds_brake[i] = float(control.brake)
        self._ds_nav[i] = info.nav_command
        self._ds_pos[i] = np.array(info.position, dtype=np.float32)
        self._ds_rot[i] = np.array(info.rotation, dtype=np.float32)
        self._ds_vel[i] = np.array(info.velocity, dtype=np.float32)
        self._ds_tl[i] = info.traffic_light_state
        self._ds_tl_dist[i] = info.traffic_light_distance
        self._ds_ss[i] = info.stop_sign_distance
        self._ds_ts[i] = info.target_speed
        self._ds_dm[i] = info.driving_mode
        self._ds_ncorr[i] = info.n_corridor_segments
        self._ds_wpbs[i] = info.wp_buffer_size
        self._ds_wpbh[i] = info.wp_buffer_head_dist
        self._ds_sat[i] = info.sat_detections
        self._ds_lat[i] = info.lateral_detections
        self._ds_nveh[i] = json.dumps(info.nearby_vehicles)
        self._ds_nped[i] = json.dumps(info.nearby_pedestrians)

        self.n_written += 1

        # Periodically flush + evict page cache to prevent VmmemWSL bloat.
        if self.n_written % _CACHE_EVICT_INTERVAL == 0:
            self._flush_and_evict_cache()

    # -- Finalize ----------------------------------------------------------

    def finalize(
        self,
        status: str,
        collision_frame: int | None,
    ) -> EpisodeResult:
        """Resize datasets to actual frame count, write metadata, close."""
        n = self.n_written
        config = self.config

        if n == 0:
            self._h5.close()
            self.filepath.unlink(missing_ok=True)
            return EpisodeResult(
                status="crashed",
                error_message="No frames collected",
            )

        # Trim datasets to actual length
        for ds in [
            self._ds_images,
            self._ds_steering,
            self._ds_throttle,
            self._ds_brake,
            self._ds_nav,
            self._ds_pos,
            self._ds_rot,
            self._ds_vel,
            self._ds_tl,
            self._ds_ss,
            self._ds_ts,
            self._ds_dm,
            self._ds_ncorr,
            self._ds_wpbs,
            self._ds_wpbh,
            self._ds_sat,
            self._ds_lat,
            self._ds_nveh,
            self._ds_nped,
        ]:
            ds.resize(n, axis=0)

        # Derived: stationary flag
        vel = self._ds_vel[:]
        speed_ms = np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2 + vel[:, 2] ** 2)
        self._h5.create_dataset(S.STATIONARY, data=(speed_ms * 3.6 < 0.5))

        # Metadata
        pano_w = config.camera_width * len(config.camera_yaws)
        meta = self._h5.create_group(S.METADATA_GROUP)
        meta.attrs[S.META_SCHEMA_VERSION] = S.SCHEMA_VERSION
        meta.attrs[S.META_TOWN] = config.town
        meta.attrs[S.META_WEATHER] = config.weather
        meta.attrs[S.META_FPS] = config.fps
        meta.attrs[S.META_N_FRAMES] = n
        meta.attrs[S.META_STATUS] = status
        meta.attrs[S.META_CAMERA_WIDTH] = config.camera_width
        meta.attrs[S.META_CAMERA_HEIGHT] = config.camera_height
        meta.attrs[S.META_N_CAMERAS] = len(config.camera_yaws)
        meta.attrs[S.META_CAMERA_FOV] = config.camera_fov
        meta.attrs[S.META_CAMERA_YAWS] = list(config.camera_yaws)
        meta.attrs[S.META_EXPERT_TYPE] = "privileged_expert"
        meta.attrs[S.META_EXPERT_TARGET_SPEED] = config.expert_config.target_speed
        if collision_frame is not None:
            meta.attrs[S.META_COLLISION_FRAME] = collision_frame

        # Final cache eviction before close — release all remaining pages.
        self._flush_and_evict_cache()
        self._h5.close()

        # Nav distribution
        nav = self._read_nav_distribution()

        filename = self.filepath.name
        episode_hash = filename.replace("episode_", "").replace(".h5", "")

        logger.info(
            "Saved %s — %d frames (%dx%d panoramic), status=%s, town=%s",
            filename,
            n,
            pano_w,
            config.camera_height,
            status,
            config.town,
        )

        return EpisodeResult(
            status=status,
            filename=filename,
            episode_hash=episode_hash,
            n_frames=n,
            town=config.town,
            weather=config.weather,
            n_vehicles=config.n_vehicles,
            n_walkers=config.n_walkers,
            nav_distribution=nav,
            collision_frame=collision_frame,
        )

    def _read_nav_distribution(self) -> dict[str, int]:
        """Read nav command distribution from the (already-written) dataset."""
        nav_names = {
            int(RoadOption.LANEFOLLOW): "LaneFollow",
            int(RoadOption.LEFT): "Left",
            int(RoadOption.RIGHT): "Right",
            int(RoadOption.STRAIGHT): "Straight",
        }
        # Re-open briefly to read the trimmed nav dataset
        with h5py.File(self.filepath, "r") as f:
            nav = f[S.NAV_COMMAND][:]
        dist = {}
        for cmd_id, cmd_name in nav_names.items():
            dist[cmd_name] = int(np.sum(nav == cmd_id))
        return dist

    def abort(self) -> None:
        """Close the file without finalizing (e.g. on crash with no data)."""
        try:
            self._h5.close()
        except Exception:
            pass
        self.filepath.unlink(missing_ok=True)


class ExpertDataCollector:
    """Collects expert driving data from CARLA using PrivilegedExpert.

    Drives a fully custom expert agent with PID control and privileged
    hazard perception.  Records 3-camera panoramic images, expert
    controls, navigation commands, ego state, and nearby-actor metadata.

    Usage::

        collector = ExpertDataCollector(host="auto", port=2000,
                                        output_dir="data/collected")
        collector.connect()
        result = collector.collect_episode(config, route_waypoints)

    Args:
        host: CARLA server host ("auto" to detect from WSL2 gateway).
        port: CARLA server port.
        output_dir: Directory to save HDF5 files and manifest.
        crash_retries: Number of reconnection attempts on CARLA crash.
        crash_wait: Seconds to wait between reconnection attempts.
    """

    def __init__(
        self,
        host: str = "auto",
        port: int = 2000,
        output_dir: str | Path = "data/collected",
        crash_retries: int = 3,
        crash_wait: float = 30.0,
    ) -> None:
        if not CARLA_AVAILABLE:
            raise RuntimeError("carla package not installed. Run: pip install carla==0.9.16")

        self.port = port
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.crash_retries = crash_retries
        self.crash_wait = crash_wait

        # Resolve host
        if host == "auto":
            self.host = self._detect_wsl2_host() or "localhost"
        else:
            self.host = host

        self.client: carla.Client | None = None
        self.world: carla.World | None = None

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_wsl2_host() -> str | None:
        """Detect Windows host IP from WSL2 default gateway."""
        import subprocess

        try:
            result = subprocess.run(
                ["ip", "route", "show", "default"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip().split()[2]
        except (subprocess.CalledProcessError, IndexError, FileNotFoundError):
            return None

    def connect(self) -> None:
        """Connect to CARLA server."""
        logger.info("Connecting to CARLA at %s:%d", self.host, self.port)
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(60.0)
        self.world = self.client.get_world()
        logger.info(
            "Connected — server=%s, map=%s",
            self.client.get_server_version(),
            self.world.get_map().name,
        )

    def teardown_world(self) -> None:
        """Destroy all actors and flush C++ client state before map change.

        CARLA's C++ client library (libcarla_client) may retain internal
        sensor data buffers after actors are destroyed at the Python level.
        This method aggressively cleans up to prevent memory leaks during
        ``client.load_world()`` map transitions.

        Call this BEFORE ``client.load_world()`` when switching towns.
        """
        if self.world is None:
            return

        # 1. Destroy all actors we can find in the CARLA world
        try:
            actors = self.world.get_actors()
            # Stop and destroy sensors first (cameras, collision, lidar, etc.)
            sensors = actors.filter("sensor.*")
            for sensor in sensors:
                try:
                    sensor.stop()
                except Exception:
                    pass
                try:
                    sensor.destroy()
                except Exception:
                    pass

            # Destroy vehicles (except hero/ego if any — they should
            # already be gone from episode cleanup)
            vehicles = actors.filter("vehicle.*")
            for v in vehicles:
                try:
                    v.set_autopilot(False)
                    v.destroy()
                except Exception:
                    pass

            # Destroy walkers and controllers
            walkers = actors.filter("walker.*")
            for w in walkers:
                try:
                    w.destroy()
                except Exception:
                    pass
            controllers = actors.filter("controller.*")
            for c in controllers:
                try:
                    c.stop()
                    c.destroy()
                except Exception:
                    pass

            logger.info(
                "Teardown: destroyed %d sensors, %d vehicles, %d walkers",
                len(sensors),
                len(vehicles),
                len(walkers),
            )
        except Exception as e:
            logger.warning("Teardown actor cleanup failed: %s", e)

        # 2. Drop our Python reference to the old world
        self.world = None

        # 3. Force garbage collection to release C-level pointers
        gc.collect()

    def _reconnect(self) -> bool:
        """Attempt to reconnect after CARLA crash."""
        for attempt in range(1, self.crash_retries + 1):
            logger.warning(
                "Reconnection attempt %d/%d (waiting %ds)...",
                attempt,
                self.crash_retries,
                int(self.crash_wait),
            )
            time.sleep(self.crash_wait)
            try:
                self.connect()
                return True
            except Exception as e:
                logger.error("Reconnection failed: %s", e)
        return False

    @staticmethod
    def _get_rss_mb() -> int:
        """Read current process RSS (Resident Set Size) in MB.

        Uses ``/proc/self/status`` on Linux for a zero-overhead read
        (no subprocess, no psutil dependency).  Returns 0 on non-Linux
        platforms so the RSS guard is effectively disabled.
        """
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        # Format: "VmRSS:   12345 kB"
                        return int(line.split()[1]) // 1024
        except (FileNotFoundError, OSError):
            return 0
        return 0

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _update_spectator(
        world: "carla.World",
        vehicle: "carla.Vehicle",
        offset: tuple[float, float, float] = (-8.0, 0.0, 5.0),
    ) -> None:
        """Move the CARLA spectator camera to follow the ego vehicle.

        The spectator is positioned behind and above the vehicle so
        you can watch it drive in the CARLA viewport in real-time.

        Args:
            world: CARLA world instance.
            vehicle: Ego vehicle to follow.
            offset: (x, y, z) offset in vehicle-local frame.
                    Default ``(-8, 0, 5)`` gives a chase-cam view.
        """
        vt = vehicle.get_transform()
        # Compute world-space offset using vehicle's forward/right/up
        fwd = vt.get_forward_vector()
        right = vt.get_right_vector()
        up = carla.Location(x=0.0, y=0.0, z=1.0)

        cam_loc = vt.location + carla.Location(
            x=fwd.x * offset[0] + right.x * offset[1] + up.x * offset[2],
            y=fwd.y * offset[0] + right.y * offset[1] + up.y * offset[2],
            z=fwd.z * offset[0] + right.z * offset[1] + up.z * offset[2],
        )
        cam_rot = carla.Rotation(
            pitch=-15.0,
            yaw=vt.rotation.yaw,
            roll=0.0,
        )
        world.get_spectator().set_transform(carla.Transform(cam_loc, cam_rot))

    @staticmethod
    def _draw_route_debug(
        world: "carla.World",
        expert: "PrivilegedExpert",
        vehicle: "carla.Vehicle",
        frame_idx: int,
        info: Any = None,
    ) -> None:
        """Draw debug markers in the CARLA viewport.

        Renders several layers so you can see:
          1. **Planned route** — green dots along the remaining route
          2. **Actual position** — orange breadcrumb at current location
          3. **Upcoming target** — cyan sphere at the expert's steering target
          4. **Detection corridor** — magenta outline of the curve-aware
             lane corridor used for vehicle detection
          5. **Nearby vehicles** — red box for lead (in-lane) vehicles,
             yellow box for nearby but out-of-lane vehicles
          6. **HUD** — driving mode, speed, target speed, lead distance

        All markers use ``life_time=0.3`` so they refresh each tick without
        accumulating clutter.
        """
        debug = world.debug
        life = 0.3  # seconds — slightly longer than 1 tick so no flicker

        # 1. Planned route — draw every 5th remaining waypoint in green
        route = expert._route
        idx = expert._route_index
        GREEN = carla.Color(0, 255, 0)
        for i in range(idx, min(len(route), idx + 200), 5):
            wp, _opt = route[i]
            loc = wp.transform.location
            debug.draw_point(
                carla.Location(loc.x, loc.y, loc.z + 0.5),
                size=0.08,
                color=GREEN,
                life_time=life,
            )

        # Draw route as connected line segments (every 2nd point)
        for i in range(idx, min(len(route) - 1, idx + 200), 2):
            wp_a = route[i][0].transform.location
            wp_b = route[i + 1][0].transform.location
            debug.draw_line(
                carla.Location(wp_a.x, wp_a.y, wp_a.z + 0.5),
                carla.Location(wp_b.x, wp_b.y, wp_b.z + 0.5),
                thickness=0.06,
                color=GREEN,
                life_time=life,
            )

        # 2. Actual position — orange breadcrumb (longer life for trail)
        ego_loc = vehicle.get_transform().location
        ORANGE = carla.Color(255, 165, 0)
        debug.draw_point(
            carla.Location(ego_loc.x, ego_loc.y, ego_loc.z + 0.5),
            size=0.10,
            color=ORANGE,
            life_time=8.0,  # persist for 8s to form a visible trail
        )

        # 3. Current steering target — cyan sphere
        if expert._wp_buffer:
            target_wp = expert._wp_buffer[0]
            t_loc = target_wp[0].transform.location
            CYAN = carla.Color(0, 255, 255)
            debug.draw_point(
                carla.Location(t_loc.x, t_loc.y, t_loc.z + 1.0),
                size=0.15,
                color=CYAN,
                life_time=life,
            )
            # Line from ego to target
            debug.draw_line(
                carla.Location(ego_loc.x, ego_loc.y, ego_loc.z + 0.5),
                carla.Location(t_loc.x, t_loc.y, t_loc.z + 1.0),
                thickness=0.04,
                color=CYAN,
                life_time=life,
            )

        # 4. Detection corridor — magenta outline
        corridor: Any | None = None
        corridor_3d = getattr(expert, "_corridor_3d_segments", None)
        if corridor_3d:
            MAGENTA = carla.Color(255, 0, 255)
            for quad in corridor_3d:
                if len(quad) != 4:
                    continue
                for i in range(4):
                    p1 = quad[i]
                    p2 = quad[(i + 1) % 4]
                    debug.draw_line(
                        carla.Location(float(p1[0]), float(p1[1]), float(p1[2])),
                        carla.Location(float(p2[0]), float(p2[1]), float(p2[2])),
                        thickness=0.04,
                        color=MAGENTA,
                        life_time=life,
                    )
        else:
            corridor = getattr(expert, "_corridor", None)
            if corridor is None or len(corridor) < 3:
                corridor = None
        if corridor is not None:
            MAGENTA = carla.Color(255, 0, 255)
            z = ego_loc.z + 0.3
            n = len(corridor)
            for i in range(n):
                p1 = corridor[i]
                p2 = corridor[(i + 1) % n]
                debug.draw_line(
                    carla.Location(float(p1[0]), float(p1[1]), z),
                    carla.Location(float(p2[0]), float(p2[1]), z),
                    thickness=0.04,
                    color=MAGENTA,
                    life_time=life,
                )

        # 5. Nearby vehicle bounding boxes
        #    Red    = in corridor AND within emergency braking distance
        #    Orange = in corridor but further away
        #    Yellow = detected but not in lane / not ahead
        RED = carla.Color(255, 0, 0)
        V_ORANGE = carla.Color(101, 67, 33)
        YELLOW = carla.Color(255, 255, 0)

        if info is not None:
            emrg_dist = expert.config.vehicle_emergency_distance

            for i, v_info in enumerate(info.nearby_vehicles):
                obb_flag = v_info.get("obb_in_lane", False)
                is_ahead_same = v_info.get("is_ahead", False) and v_info.get("same_direction", True)

                color = YELLOW
                if obb_flag and is_ahead_same:
                    if v_info["distance"] < emrg_dist:
                        color = RED
                    else:
                        color = V_ORANGE
                elif obb_flag:
                    color = V_ORANGE

                # Draw marker at the detected vehicle's true world position
                # when available; fallback to ego-relative reconstruction.
                if all(k in v_info for k in ("world_x", "world_y", "world_z")):
                    vx = float(v_info["world_x"])
                    vy = float(v_info["world_y"])
                    vz = float(v_info["world_z"]) + 1.2
                else:
                    fwd = vehicle.get_transform().get_forward_vector()
                    right = carla.Vector3D(-fwd.y, fwd.x, 0)
                    vx = ego_loc.x + fwd.x * v_info["longitudinal"] + right.x * v_info["lateral"]
                    vy = ego_loc.y + fwd.y * v_info["longitudinal"] + right.y * v_info["lateral"]
                    vz = ego_loc.z + 1.5
                debug.draw_point(
                    carla.Location(vx, vy, vz),
                    size=0.15,
                    color=color,
                    life_time=life,
                )
                # Distance label
                debug.draw_string(
                    carla.Location(vx, vy, vz + 0.5),
                    f"{v_info['distance']:.0f}m",
                    draw_shadow=True,
                    color=color,
                    life_time=life,
                )

            # 5b. Nearby pedestrian markers
            #     Red    = in lane corridor (ahead + lateral < ped half-width)
            #     Yellow = nearby but out of lane
            P_RED = carla.Color(255, 50, 50)
            P_YELLOW = carla.Color(255, 255, 100)
            ped_half_w = expert.config.pedestrian_lane_half_width
            fwd = vehicle.get_transform().get_forward_vector()
            right = carla.Vector3D(-fwd.y, fwd.x, 0)
            for p_info in getattr(info, "nearby_pedestrians", []):
                in_lane = p_info["is_ahead"] and abs(p_info["lateral"]) < ped_half_w
                p_color = P_RED if in_lane else P_YELLOW
                px = ego_loc.x + fwd.x * p_info["longitudinal"] + right.x * p_info["lateral"]
                py = ego_loc.y + fwd.y * p_info["longitudinal"] + right.y * p_info["lateral"]
                pz = ego_loc.z + 1.0
                debug.draw_point(
                    carla.Location(px, py, pz),
                    size=0.12,
                    color=p_color,
                    life_time=life,
                )
                debug.draw_string(
                    carla.Location(px, py, pz + 0.5),
                    f"{p_info['distance']:.0f}m",
                    draw_shadow=True,
                    color=p_color,
                    life_time=life,
                )

        # 6. HUD — frame, mode, speed, target speed
        from agents.expert.privileged_expert import DrivingMode

        mode_name = DrivingMode(info.driving_mode).name if info is not None else "?"
        spd = info.current_speed if info is not None else 0.0
        tgt = info.target_speed if info is not None else 0.0

        # Lead vehicle distance from nearby_vehicles (union check)
        lead_str = "--"
        half_w = getattr(expert, "_cached_half_lane", expert.config.lane_half_width)
        if info is not None:
            for v_info in info.nearby_vehicles:
                is_ahead_same = v_info["is_ahead"] and v_info["same_direction"]
                in_lane = v_info["obb_in_lane"] or abs(v_info["lateral"]) < half_w
                if is_ahead_same and in_lane:
                    lead_str = f"{v_info['distance']:.0f}m"
                    break

        hud_text = f"F{frame_idx} | {mode_name}\n" f"{spd:.0f}/{tgt:.0f} km/h | lead:{lead_str}"
        debug.draw_string(
            carla.Location(ego_loc.x, ego_loc.y, ego_loc.z + 3.0),
            hud_text,
            draw_shadow=True,
            color=carla.Color(255, 255, 255),
            life_time=life,
        )

    @staticmethod
    def _spawn_cameras(
        world: "carla.World",
        vehicle: "carla.Vehicle",
        config: EpisodeConfig,
    ) -> list[tuple["carla.Actor", queue.Queue]]:
        """Spawn 3 RGB cameras and return (actor, image_queue) pairs.

        Cameras are ordered LEFT, CENTRE, RIGHT so that horizontal
        concatenation produces a seamless panoramic image.
        """
        bp_lib = world.get_blueprint_library()
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(config.camera_width))
        camera_bp.set_attribute("image_size_y", str(config.camera_height))
        camera_bp.set_attribute("fov", str(config.camera_fov))

        cameras: list[tuple[carla.Actor, queue.Queue]] = []

        for yaw in config.camera_yaws:
            cam_transform = carla.Transform(
                carla.Location(x=config.camera_x, z=config.camera_z),
                carla.Rotation(yaw=yaw),
            )
            cam = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)
            img_q: queue.Queue = queue.Queue()
            cam.listen(lambda img, q=img_q: q.put(img))
            cameras.append((cam, img_q))

        return cameras

    @staticmethod
    def _capture_panoramic(
        cameras: list[tuple["carla.Actor", queue.Queue]],
        config: EpisodeConfig,
        timeout: float = 2.0,
    ) -> np.ndarray:
        """Capture one frame from all cameras and stitch horizontally.

        Returns:
            (H, W_total, 3) uint8 RGB panoramic image where
            W_total = n_cameras x per_camera_width.
        """
        panels: list[np.ndarray] = []

        for _cam, img_q in cameras:
            try:
                carla_image = img_q.get(timeout=timeout)
                raw = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
                raw = raw.reshape(config.camera_height, config.camera_width, 4)
                rgb = raw[:, :, :3][:, :, ::-1].copy()  # BGRA → RGB
                panels.append(rgb)
            except queue.Empty:
                panels.append(
                    np.zeros(
                        (config.camera_height, config.camera_width, 3),
                        dtype=np.uint8,
                    )
                )

        return np.concatenate(panels, axis=1)  # (H, W_total, 3)

    # ------------------------------------------------------------------
    # Episode collection
    # ------------------------------------------------------------------

    def collect_episode(
        self,
        config: EpisodeConfig,
        route_waypoints: list | None = None,
    ) -> EpisodeResult:
        """Collect a single episode of expert driving data.

        Uses the custom PrivilegedExpert agent for driving.  Navigation
        commands come directly from the route planner — no heuristic
        inference needed.

        Args:
            config: Episode configuration (cameras, expert, etc.).
            route_waypoints: Pre-computed route as
                ``[(carla.Transform, RoadOption_str), ...]``.
                If ``None``, a random route is generated.

        Returns:
            EpisodeResult with status and metadata.
        """
        if self.client is None:
            self.connect()

        start_time = time.time()
        actors: list = []
        npc_vehicles: list = []
        walkers: list = []
        walker_controllers: list = []
        original_settings = None
        cameras: list[tuple[carla.Actor, queue.Queue]] = []

        # Streaming HDF5 writer (replaces in-memory buffer lists)
        # Generate hash-based filename upfront
        timestamp_bytes = int(time.time() * 1_000_000).to_bytes(8, "big")
        episode_hash = hashlib.sha256(timestamp_bytes).hexdigest()[:8]
        episode_filename = f"episode_{episode_hash}.h5"
        episode_filepath = self.output_dir / episode_filename
        writer: _StreamingEpisodeWriter | None = None
        collision_frame: int | None = None

        try:
            assert self.world is not None
            assert self.client is not None

            # --- Setup world ---
            if self.world.get_map().name.split("/")[-1] != config.town:
                logger.info("Loading map: %s", config.town)
                self.client.set_timeout(120.0)
                self.client.load_world(config.town)
                self.world = self.client.get_world()
                self.client.set_timeout(60.0)
                time.sleep(5.0)

            carla_map = self.world.get_map()
            original_settings = self.world.get_settings()

            # Enable synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / config.fps
            self.world.apply_settings(settings)

            # Set weather
            weather_preset = getattr(carla.WeatherParameters, config.weather, None)
            if weather_preset is not None:
                self.world.set_weather(weather_preset)

            # --- Spawn ego vehicle ---
            bp_lib = self.world.get_blueprint_library()
            vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
            spawn_points = carla_map.get_spawn_points()

            # When a pre-planned route is provided, teleport the vehicle
            # directly to the route's first waypoint transform.  This
            # guarantees the car starts on the route facing the correct
            # direction (same approach TransFuser / Leaderboard uses).
            ego_spawn_loc: Any = None

            if route_waypoints:
                # Lift spawn 0.5m above road surface so the vehicle's
                # physics bounding-box doesn't clip the ground (route
                # waypoint transforms sit at exact road level).
                raw = route_waypoints[0][0]  # carla.Transform
                spawn_transform = carla.Transform(
                    carla.Location(
                        x=raw.location.x,
                        y=raw.location.y,
                        z=raw.location.z + 0.5,
                    ),
                    raw.rotation,
                )
                vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
                if vehicle is None:
                    # Fallback: exact location may be obstructed; try
                    # closest spawn point instead.
                    route_start_loc = raw.location
                    distances = [sp.location.distance(route_start_loc) for sp in spawn_points]
                    spawn_idx = int(np.argmin(distances))
                    vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_points[spawn_idx])
                    ego_spawn_loc = spawn_points[spawn_idx].location
                    logger.warning(
                        "Teleport spawn failed; fell back to spawn point %d "
                        "(%.1fm from route start)",
                        spawn_idx,
                        distances[spawn_idx],
                    )
                else:
                    ego_spawn_loc = spawn_transform.location
                    logger.info(
                        "Teleported ego to route start at (%.1f, %.1f, %.1f)",
                        spawn_transform.location.x,
                        spawn_transform.location.y,
                        spawn_transform.location.z,
                    )
            else:
                spawn_idx = np.random.randint(len(spawn_points))
                vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_points[spawn_idx])
                ego_spawn_loc = spawn_points[spawn_idx].location

            if vehicle is None:
                return EpisodeResult(
                    status="crashed",
                    error_message="Failed to spawn vehicle",
                    n_vehicles=config.n_vehicles,
                    n_walkers=config.n_walkers,
                    duration_seconds=time.time() - start_time,
                )

            actors.append(vehicle)

            # --- Spawn NPC traffic (driven by TrafficManager) ---
            traffic_manager = self.client.get_trafficmanager(8000)
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_global_distance_to_leading_vehicle(2.5)

            if config.n_vehicles > 0:
                npc_bps = bp_lib.filter("vehicle.*")
                npc_bps = [bp for bp in npc_bps if int(bp.get_attribute("number_of_wheels")) == 4]
                available_spawns = [
                    sp
                    for sp in spawn_points
                    if ego_spawn_loc is None or sp.location.distance(ego_spawn_loc) > 50.0
                ]
                np.random.shuffle(available_spawns)
                n_to_spawn = min(config.n_vehicles, len(available_spawns))

                for i in range(n_to_spawn):
                    npc_bp = np.random.choice(npc_bps)
                    npc = self.world.try_spawn_actor(npc_bp, available_spawns[i])
                    if npc is not None:
                        npc.set_autopilot(True, traffic_manager.get_port())
                        npc_vehicles.append(npc)

                logger.info(
                    "Spawned %d NPC vehicles (requested %d)",
                    len(npc_vehicles),
                    config.n_vehicles,
                )

            # --- Spawn NPC walkers with AI controllers ---
            walkers = []
            walker_controllers = []
            if config.n_walkers > 0:
                walker_bps = bp_lib.filter("walker.pedestrian.*")
                # Gather random spawn points on sidewalks
                walker_spawn_points: list[Any] = []
                for _ in range(config.n_walkers * 2):
                    loc = self.world.get_random_location_from_navigation()
                    if loc is not None:
                        wp = carla.Transform(location=loc)
                        walker_spawn_points.append(wp)
                    if len(walker_spawn_points) >= config.n_walkers:
                        break

                n_walkers_to_spawn = min(config.n_walkers, len(walker_spawn_points))
                for i in range(n_walkers_to_spawn):
                    w_bp = np.random.choice(walker_bps)
                    # Enable crossing for ~30 % of walkers
                    if w_bp.has_attribute("is_invincible"):
                        w_bp.set_attribute("is_invincible", "false")
                    walker = self.world.try_spawn_actor(w_bp, walker_spawn_points[i])
                    if walker is not None:
                        walkers.append(walker)

                # Tick so walkers materialise before attaching controllers
                self.world.tick()

                # Attach WalkerAIController to each walker
                controller_bp = bp_lib.find("controller.ai.walker")
                for walker in walkers:
                    ctrl = self.world.spawn_actor(
                        controller_bp, carla.Transform(), attach_to=walker
                    )
                    walker_controllers.append(ctrl)

                # Tick again so controllers are ready
                self.world.tick()

                # Start walking: random destinations + varied speeds
                for ctrl in walker_controllers:
                    dest = self.world.get_random_location_from_navigation()
                    if dest is not None:
                        ctrl.start()
                        ctrl.go_to_location(dest)
                        # Vary speed: 1.0–2.5 m/s (walking to jogging)
                        ctrl.set_max_speed(1.0 + np.random.random() * 1.5)

                logger.info(
                    "Spawned %d NPC walkers (requested %d)",
                    len(walkers),
                    config.n_walkers,
                )

            # --- Attach 3 cameras ---
            cameras = self._spawn_cameras(self.world, vehicle, config)
            for cam, _ in cameras:
                actors.append(cam)

            # --- Attach collision sensor ---
            collision_bp = bp_lib.find("sensor.other.collision")
            collision_sensor = self.world.spawn_actor(
                collision_bp, carla.Transform(), attach_to=vehicle
            )
            actors.append(collision_sensor)
            collision_detected = [False]
            collision_other_actor: list[str | None] = [None]

            def on_collision(event: Any) -> None:
                collision_detected[0] = True
                try:
                    collision_other_actor[0] = event.other_actor.type_id
                except Exception:
                    collision_other_actor[0] = "unknown"

            collision_sensor.listen(on_collision)

            # --- Setup PrivilegedExpert ---
            route_planner = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)
            expert = PrivilegedExpert(vehicle, self.world, route_planner, config.expert_config)

            # Build route
            if route_waypoints:
                # route_waypoints = [(transform, road_option_str), ...]
                route = []
                for transform, road_opt_name in route_waypoints:
                    wp = carla_map.get_waypoint(transform.location)
                    opt = self._parse_road_option(str(road_opt_name))
                    route.append((wp, opt))
                expert.set_route(route)
                logger.info(
                    "Set custom route with %d waypoints for expert",
                    len(route),
                )
            else:
                # Random destination
                dest_idx = np.random.randint(len(spawn_points))
                while dest_idx == spawn_idx:
                    dest_idx = np.random.randint(len(spawn_points))
                expert.set_destination(
                    spawn_points[spawn_idx].location,
                    spawn_points[dest_idx].location,
                )
                logger.info("Set random route for expert")

            # Warm-up ticks
            for _ in range(10):
                self.world.tick()

            # Drain warm-up images from all cameras
            for _cam, img_q in cameras:
                while not img_q.empty():
                    try:
                        img_q.get_nowait()
                    except queue.Empty:
                        break

            # --- Start CARLA recorder if requested ---
            recording_path: str | None = None
            if config.record_simulation:
                rec_name = f"episode_{int(time.time())}.log"
                recording_path = rec_name
                self.client.start_recorder(rec_name, True)
                logger.info("CARLA recorder started: %s", rec_name)

            # --- Tick loop ---
            stuck_start: float | None = None
            distance_traveled = 0.0
            prev_location = vehicle.get_transform().location
            dt = 1.0 / config.fps
            memory_limit_hit = False

            # Pre-tick RSS baseline: catch leaks that accumulated during
            # setup (warm-up ticks, actor spawning, etc.)
            if config.rss_limit_mb > 0:
                rss = self._get_rss_mb()
                logger.info(
                    "  [RSS guard] baseline before tick loop: %d MB " "(limit %d MB)",
                    rss,
                    config.rss_limit_mb,
                )
                if rss > config.rss_limit_mb:
                    logger.warning(
                        "RSS memory guard: %d MB exceeds limit "
                        "%d MB before tick loop — aborting episode",
                        rss,
                        config.rss_limit_mb,
                    )
                    memory_limit_hit = True

            # Create streaming HDF5 writer (writes one frame at a time)
            writer = _StreamingEpisodeWriter(
                filepath=episode_filepath,
                config=config,
                max_frames=config.max_frames,
            )

            # Track last RSS log time (log every 30s regardless of frame
            # count, since the C++ leak is wall-clock-proportional).
            _last_rss_log = time.time()

            for frame_idx in range(config.max_frames):
                if memory_limit_hit:
                    break

                # Check if expert finished route
                if expert.done():
                    logger.info("Expert completed route at frame %d", frame_idx)
                    break

                self.world.tick()

                # --- F3: RSS memory guard ---
                # Check EVERY frame — reading /proc/self/status costs
                # ~5 µs, negligible vs. the ~50 ms CARLA tick.  The C++
                # leak is wall-clock-proportional (~40-100 MB/s), NOT
                # frame-proportional, so infrequent frame-based checks
                # miss it when tick rate drops under memory pressure.
                if config.rss_limit_mb > 0:
                    rss = self._get_rss_mb()
                    now = time.time()
                    if now - _last_rss_log >= 30.0:
                        logger.info(
                            "  [RSS guard] frame %d, t=%.0fs: " "%d MB (limit %d MB)",
                            frame_idx + 1,
                            now - start_time,
                            rss,
                            config.rss_limit_mb,
                        )
                        _last_rss_log = now
                    if rss > config.rss_limit_mb:
                        logger.warning(
                            "RSS memory guard: %d MB exceeds limit "
                            "%d MB at frame %d (t=%.0fs) "
                            "— stopping episode",
                            rss,
                            config.rss_limit_mb,
                            frame_idx + 1,
                            now - start_time,
                        )
                        memory_limit_hit = True
                        break

                # --- Expert step ---
                control, info = expert.run_step(dt)
                vehicle.apply_control(control)

                # --- Spectator follow (real-time visualization) ---
                if config.spectator_follow:
                    self._update_spectator(self.world, vehicle, config.spectator_offset)

                # --- In-CARLA debug drawing (route + trajectory) ---
                if config.debug_draw:
                    self._draw_route_debug(
                        self.world,
                        expert,
                        vehicle,
                        frame_idx,
                        info,
                    )

                # --- Capture panoramic image ---
                panoramic = self._capture_panoramic(cameras, config)

                # --- Stream frame directly to HDF5 (no in-memory buffer) ---
                writer.append_frame(panoramic, control, info)

                # Free the panoramic array immediately
                del panoramic

                # --- Distance tracking ---
                cur_loc = vehicle.get_transform().location
                dx = cur_loc.x - prev_location.x
                dy = cur_loc.y - prev_location.y
                distance_traveled += math.sqrt(dx**2 + dy**2)
                prev_location = cur_loc

                # --- Collision check ---
                if collision_detected[0] and collision_frame is None:
                    collision_frame = frame_idx
                    logger.warning(
                        "Collision detected at frame %d with %s",
                        frame_idx,
                        collision_other_actor[0] or "unknown",
                    )
                    break  # Stop collecting — post-collision data is unusable

                # --- Timeout (stuck) check ---
                speed_ms = math.sqrt(
                    info.velocity[0] ** 2 + info.velocity[1] ** 2 + info.velocity[2] ** 2
                )
                if speed_ms < config.timeout_speed:
                    if stuck_start is None:
                        stuck_start = time.time()
                    elif time.time() - stuck_start > config.timeout_duration:
                        logger.warning(
                            "Timeout: vehicle stuck for >%ds",
                            config.timeout_duration,
                        )
                        break
                else:
                    stuck_start = None

                # --- Progress logging ---
                if (frame_idx + 1) % 500 == 0:
                    logger.info(
                        "  Frame %d/%d — speed=%.1f km/h, dist=%.0fm, "
                        "steer=%.3f, mode=%s, route=%.0f%%",
                        frame_idx + 1,
                        config.max_frames,
                        info.current_speed,
                        distance_traveled,
                        control.steer,
                        info.driving_mode,
                        info.route_completion * 100,
                    )

            # --- Determine status ---
            timed_out = stuck_start is not None and (
                time.time() - stuck_start > config.timeout_duration
            )
            if memory_limit_hit:
                status = "memory_limit"
            elif timed_out:
                status = "timeout"
            elif collision_frame is not None:
                status = "collision"
            else:
                status = "completed"

            # --- Finalize streaming HDF5 ---
            result = writer.finalize(status=status, collision_frame=collision_frame)
            result.collision_other_actor = collision_other_actor[0]
            result.duration_seconds = time.time() - start_time
            result.recording_path = recording_path
            return result

        except Exception as e:
            logger.error("Episode crashed: %s\n%s", e, traceback.format_exc())

            # Finalize partial episode if writer has data
            if writer is not None and writer.n_written > 10:
                try:
                    result = writer.finalize(
                        status="crashed",
                        collision_frame=collision_frame,
                    )
                    result.error_message = str(e)
                    result.duration_seconds = time.time() - start_time
                    return result
                except Exception:
                    pass
            elif writer is not None:
                writer.abort()

            return EpisodeResult(
                status="crashed",
                error_message=str(e),
                n_frames=writer.n_written if writer is not None else 0,
                town=config.town,
                weather=config.weather,
                n_vehicles=config.n_vehicles,
                n_walkers=config.n_walkers,
                duration_seconds=time.time() - start_time,
            )

        finally:
            # Stop CARLA recorder if it was started
            if config.record_simulation and self.client is not None:
                try:
                    self.client.stop_recorder()
                    logger.info("CARLA recorder stopped")
                except Exception:
                    pass

            # Stop sensors first
            for actor in actors:
                try:
                    if actor.is_alive and hasattr(actor, "stop"):
                        actor.stop()
                except Exception:
                    pass

            # Destroy NPC vehicles
            for npc in npc_vehicles:
                try:
                    if npc.is_alive:
                        npc.set_autopilot(False)
                        npc.destroy()
                except Exception:
                    pass

            # Destroy walker controllers then walkers
            for ctrl in walker_controllers:
                try:
                    if ctrl.is_alive:
                        ctrl.stop()
                        ctrl.destroy()
                except Exception:
                    pass
            for w in walkers:
                try:
                    if w.is_alive:
                        w.destroy()
                except Exception:
                    pass

            # Destroy ego actors (sensors, collision, vehicle)
            for actor in reversed(actors):
                try:
                    if actor.is_alive:
                        actor.destroy()
                except Exception:
                    pass

            # Restore world settings
            if original_settings is not None:
                try:
                    assert self.world is not None
                    self.world.apply_settings(original_settings)
                except Exception:
                    pass

            # Force garbage collection to free any lingering references
            gc.collect()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_road_option(name: str) -> RoadOption:
        """Parse a road option string to RoadOption enum."""
        mapping = {
            "LaneFollow": RoadOption.LANEFOLLOW,
            "Left": RoadOption.LEFT,
            "Right": RoadOption.RIGHT,
            "Straight": RoadOption.STRAIGHT,
            "RoadOption.LANEFOLLOW": RoadOption.LANEFOLLOW,
            "RoadOption.LEFT": RoadOption.LEFT,
            "RoadOption.RIGHT": RoadOption.RIGHT,
            "RoadOption.STRAIGHT": RoadOption.STRAIGHT,
            "LANEFOLLOW": RoadOption.LANEFOLLOW,
            "LEFT": RoadOption.LEFT,
            "RIGHT": RoadOption.RIGHT,
            "STRAIGHT": RoadOption.STRAIGHT,
            "CHANGELANELEFT": RoadOption.CHANGELANELEFT,
            "CHANGELANERIGHT": RoadOption.CHANGELANERIGHT,
            "ChangeLaneLeft": RoadOption.CHANGELANELEFT,
            "ChangeLaneRight": RoadOption.CHANGELANERIGHT,
        }
        return mapping.get(name, RoadOption.LANEFOLLOW)


class CollectionManifest:
    """Manages the collection_manifest.json for campaign resumption.

    Tracks every episode: hash, town, weather, frame count, turn
    distribution, status, and timestamp.
    """

    def __init__(self, manifest_path: str | Path) -> None:
        self.path = Path(manifest_path)
        self.data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        """Load existing manifest or create empty."""
        if self.path.exists():
            with open(self.path) as f:
                data: dict[str, Any] = json.load(f)
                return data
        return {
            "created": datetime.now(timezone.utc).isoformat(),
            "total_episodes": 0,
            "total_frames": 0,
            "episodes": [],
        }

    def add_episode(self, result: EpisodeResult) -> None:
        """Add an episode result to the manifest."""
        entry = {
            "filename": result.filename,
            "hash": result.episode_hash,
            "town": result.town,
            "weather": result.weather,
            "n_vehicles": result.n_vehicles,
            "n_walkers": result.n_walkers,
            "config_hash": result.config_hash,
            "n_frames": result.n_frames,
            "status": result.status,
            "nav_distribution": result.nav_distribution,
            "collision_frame": result.collision_frame,
            "collision_other_actor": result.collision_other_actor,
            "duration_seconds": result.duration_seconds,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if result.error_message:
            entry["error"] = result.error_message

        self.data["episodes"].append(entry)
        self.data["total_episodes"] = len(self.data["episodes"])
        self.data["total_frames"] = sum(e["n_frames"] for e in self.data["episodes"])
        self.data["last_updated"] = datetime.now(timezone.utc).isoformat()

    def save(self) -> None:
        """Write manifest to disk."""
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def get_completed_configs(self) -> set[tuple[str, str, int, int]]:
        """Return set of (town, weather, n_vehicles, n_walkers) completed."""
        return {
            (
                e["town"],
                e["weather"],
                e.get("n_vehicles", 0),
                e.get("n_walkers", 0),
            )
            for e in self.data["episodes"]
            if e["status"] == "completed"
        }

    def get_completed_counts(self) -> dict[tuple[str, str, int, int], int]:
        """Return count of completed episodes per (town, weather, n_vehicles, n_walkers)."""
        counts: dict[tuple[str, str, int, int], int] = {}
        for e in self.data["episodes"]:
            if e["status"] == "completed":
                key = (
                    e["town"],
                    e["weather"],
                    e.get("n_vehicles", 0),
                    e.get("n_walkers", 0),
                )
                counts[key] = counts.get(key, 0) + 1
        return counts

    def get_config_hashes(
        self,
        town: str,
        weather: str,
        n_vehicles: int,
        n_walkers: int,
    ) -> set[str]:
        """Return distinct config hashes for a scenario key.

        Used by C-4 audit: if the returned set is non-empty and the
        current config hash is not in it, data-affecting parameters
        changed between collection runs.
        """
        hashes: set[str] = set()
        for e in self.data["episodes"]:
            if (
                e["town"] == town
                and e["weather"] == weather
                and e.get("n_vehicles", 0) == n_vehicles
                and e.get("n_walkers", 0) == n_walkers
                and e.get("config_hash")
            ):
                hashes.add(e["config_hash"])
        return hashes

    def summary(self) -> str:
        """Return human-readable summary."""
        total_hrs = self.data["total_frames"] / (20 * 3600)
        status_counts: dict[str, int] = {}
        for e in self.data["episodes"]:
            s = e["status"]
            status_counts[s] = status_counts.get(s, 0) + 1

        lines = [
            f"Episodes: {self.data['total_episodes']}",
            f"Frames:   {self.data['total_frames']}" f" ({total_hrs:.1f} hours at 20 FPS)",
            f"Statuses: {status_counts}",
        ]
        return "\n".join(lines)
