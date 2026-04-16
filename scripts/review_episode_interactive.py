#!/usr/bin/env python3
"""Interactive frame-by-frame episode reviewer in live CARLA.

This script replays an HDF5 episode timeline as debug overlays in CARLA,
allowing manual frame stepping and free spectator movement (mouse + WASD/QE)
in the UE viewport.

By default it uses a focused SAT overlay mode (corridor + bbox actor
overlays) to keep frame stepping responsive.

Collector parity mode remains available via ``--draw-mode collector``
and uses ``ExpertDataCollector._draw_route_debug``.

Primary controls (global keyboard mode):
- Space: step forward one frame.
- Hold Space: continuous forward playback.
- Shift+Space: step backward one frame.
- Hold Shift+Space: continuous backward playback.
- Esc: quit.

If global keyboard hooks are unavailable, terminal fallback controls are used:
- Space: step forward one frame.
- b: step backward one frame.
- p: toggle continuous forward playback.
- o: toggle continuous backward playback.
- q: quit.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import logging
import math
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import h5py
import numpy as np

try:
    import carla
except ImportError as exc:  # pragma: no cover
    raise SystemExit("CARLA Python API is required. Activate your CARLA env first.") from exc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

HEADING_LABEL_STYLE = "compass16"

MODE_NAMES = {
    1: "FOLLOW_LANE",
    2: "APPROACHING_TURN",
    3: "TURNING",
    4: "TRAFFIC_LIGHT_STOP",
    5: "VEHICLE_FOLLOWING",
    6: "PEDESTRIAN_STOP",
    7: "EMERGENCY_STOP",
    8: "STOP_SIGN_STOP",
    9: "YIELD_STOP",
}

NAV_NAMES = {
    1: "LEFT",
    2: "RIGHT",
    3: "STRAIGHT",
    4: "LANEFOLLOW",
    5: "CHANGELANELEFT",
    6: "CHANGELANERIGHT",
}

TL_NAMES = {
    0: "NONE",
    1: "GREEN",
    2: "YELLOW",
    3: "RED",
}

EGO_COLOR = carla.Color(0, 255, 0)
ACTOR_COLOR = carla.Color(255, 255, 0)
SAT_COLOR = carla.Color(255, 0, 0)
CORRIDOR_COLOR = carla.Color(255, 0, 255)
NEXT_WP_COLOR = carla.Color(0, 255, 0)
TARGET_COLOR = carla.Color(0, 170, 255)
INDIGO_COLOR = carla.Color(102, 0, 255)
WHITE_COLOR = carla.Color(255, 255, 255)


@dataclass
class EpisodeData:
    path: Path
    town: str
    fps: int
    status: str
    collision_frame: int | None
    n_frames: int
    position: np.ndarray
    rotation: np.ndarray
    velocity: np.ndarray
    nav_command: np.ndarray
    driving_mode: np.ndarray
    target_speed: np.ndarray
    route_completion: np.ndarray
    traffic_light_state: np.ndarray
    traffic_light_actor_id: np.ndarray
    traffic_light_distance: np.ndarray
    stop_sign_distance: np.ndarray
    yield_sign_distance: np.ndarray
    wp_buffer_head_index: np.ndarray
    steer: np.ndarray
    nearby_vehicles_raw: np.ndarray
    nearby_pedestrians_raw: np.ndarray


@dataclass
class RouteData:
    path: Path | None
    points_xyz: np.ndarray
    points_junction: np.ndarray
    points_yaw: np.ndarray
    points_road_id: np.ndarray
    points_lane_id: np.ndarray
    options: list[str]


@dataclass
class InputState:
    shift_down: bool = False
    space_down: bool = False
    quit_requested: bool = False
    step_request: int = 0
    hold_started_at: float = 0.0
    play_direction: int = 0  # -1 back, +1 forward, 0 paused


@dataclass
class SatRenderContext:
    last_frame_idx: int = -1
    vehicle_prev: list[tuple[np.ndarray, float]] = field(default_factory=list)
    pedestrian_prev: list[tuple[np.ndarray, float]] = field(default_factory=list)


@dataclass
class RoadLaneCache:
    by_road_lane: dict[int, dict[int, list[Any]]]


def _detect_wsl2_host() -> str:
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
    except Exception:
        return "localhost"


def _wait_for_world_ready(
    client: carla.Client,
    expected_town: str | None,
    timeout_sec: float,
) -> carla.World:
    """Poll CARLA until world RPCs are responsive and (optionally) town matches."""
    deadline = time.time() + max(5.0, float(timeout_sec))
    last_exc: Exception | None = None

    while time.time() < deadline:
        try:
            world = client.get_world()
            town = world.get_map().name.split("/")[-1]
            if expected_town and town != expected_town:
                time.sleep(1.0)
                continue
            _ = world.get_settings()
            return world
        except RuntimeError as exc:
            last_exc = exc
            time.sleep(1.0)

    if last_exc is not None:
        raise RuntimeError(f"CARLA world not ready within {float(timeout_sec):.0f}s") from last_exc
    raise RuntimeError(f"CARLA world not ready within {float(timeout_sec):.0f}s")


def _enable_synchronous_mode(
    client: carla.Client,
    fps: int,
    attempts: int = 3,
    retry_delay_sec: float = 2.0,
) -> tuple[carla.World, Any]:
    """Enable synchronous mode with retries for transient RPC timeouts."""
    total_attempts = max(1, int(attempts))
    last_exc: Exception | None = None

    for attempt in range(1, total_attempts + 1):
        world = client.get_world()
        original_settings = world.get_settings()
        new_settings = world.get_settings()
        new_settings.synchronous_mode = True
        new_settings.fixed_delta_seconds = 1.0 / max(int(fps), 1)
        try:
            world.apply_settings(new_settings)
            return world, original_settings
        except RuntimeError as exc:
            last_exc = exc
            if attempt >= total_attempts:
                break
            logger.warning(
                "apply_settings attempt %d/%d failed (%s). Retrying in %.1fs.",
                attempt,
                total_attempts,
                exc,
                float(retry_delay_sec),
            )
            time.sleep(max(0.1, float(retry_delay_sec)))

    raise RuntimeError("Failed to enable CARLA synchronous mode") from last_exc


def _configure_file_logging(log_path: Path) -> None:
    """Add optional file logging while preserving existing console output."""
    resolved = log_path.expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    resolved = resolved.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if Path(handler.baseFilename).resolve() == resolved:
                    return
            except Exception:
                continue

    file_handler = logging.FileHandler(resolved, encoding="utf-8")
    file_handler.setLevel(root_logger.level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    )
    root_logger.addHandler(file_handler)
    logger.info("File logging enabled: %s", resolved)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive HDF5 episode reviewer in CARLA")
    parser.add_argument("--episode", required=True, help="Path to episode_*.h5 file")
    parser.add_argument(
        "--route",
        default=None,
        help="Optional route sidecar path (default: episode_<id>.route.json next to H5)",
    )
    parser.add_argument("--host", default="auto", help="CARLA host (default: auto WSL2 gateway)")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--start-frame", type=int, default=0, help="Initial frame index")
    parser.add_argument("--end-frame", type=int, default=-1, help="Last frame index (-1 = end)")
    parser.add_argument(
        "--hold-delay-sec", type=float, default=0.20, help="Delay before hold-repeat"
    )
    parser.add_argument(
        "--repeat-rate-hz",
        type=float,
        default=20.0,
        help="Frame repeat rate while held (20Hz matches collector tick)",
    )
    parser.add_argument(
        "--control-mode",
        choices=["global", "terminal"],
        default="global",
        help="Input mode. global=Space/Shift+Space with UE focused (requires pynput).",
    )
    parser.add_argument(
        "--auto-load-town",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load episode town map if current CARLA map differs",
    )
    parser.add_argument(
        "--set-initial-view",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set an initial overhead spectator view at start frame",
    )
    parser.add_argument(
        "--draw-life-time",
        type=float,
        default=0.30,
        help="CARLA debug primitive lifetime in seconds (collector default: 0.3)",
    )
    parser.add_argument(
        "--draw-mode",
        choices=["sat", "collector", "simple"],
        default="sat",
        help=(
            "sat=focused SAT corridor + bbox overlays, "
            "collector=ExpertDataCollector debug draw parity, "
            "simple=lightweight fallback"
        ),
    )
    parser.add_argument(
        "--sat-radius-m",
        type=float,
        default=20.0,
        help="Draw actor bboxes only for nearby actors within this distance",
    )
    parser.add_argument(
        "--sat-corridor-length-m",
        type=float,
        default=20.0,
        help="Forward SAT corridor length",
    )
    parser.add_argument(
        "--sat-half-width-m",
        type=float,
        default=2.5,
        help="Half width for SAT corridor",
    )
    parser.add_argument("--ego-length-m", type=float, default=4.6, help="Ego bbox length")
    parser.add_argument("--ego-width-m", type=float, default=2.0, help="Ego bbox width")
    parser.add_argument(
        "--vehicle-length-m",
        type=float,
        default=4.5,
        help="Fallback nearby-vehicle bbox length",
    )
    parser.add_argument(
        "--vehicle-width-m",
        type=float,
        default=2.0,
        help="Fallback nearby-vehicle bbox width",
    )
    parser.add_argument(
        "--pedestrian-length-m",
        type=float,
        default=0.6,
        help="Nearby pedestrian bbox length",
    )
    parser.add_argument(
        "--pedestrian-width-m",
        type=float,
        default=0.6,
        help="Nearby pedestrian bbox width",
    )
    parser.add_argument("--history-frames", type=int, default=120, help="Past path length to draw")
    parser.add_argument("--future-frames", type=int, default=120, help="Future path length to draw")
    parser.add_argument(
        "--route-window",
        type=int,
        default=24,
        help="Route sidecar points to draw before/after current wp index",
    )
    parser.add_argument(
        "--direction-label-style",
        choices=["compass16", "cardinal4"],
        default="compass16",
        help="Direction label mode for HUD text",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path for startup/runtime diagnostics",
    )
    return parser.parse_args()


def _load_episode(path: Path) -> EpisodeData:
    with h5py.File(path, "r") as h:
        meta = h["metadata"]
        town = str(meta.attrs.get("town", ""))
        fps = int(meta.attrs.get("fps", 20))
        status = str(meta.attrs.get("status", ""))
        collision_frame_attr = meta.attrs.get("collision_frame", None)
        collision_frame = None if collision_frame_attr is None else int(collision_frame_attr)

        n_frames = int(meta.attrs.get("n_frames", h["position"].shape[0]))
        position = h["position"][:n_frames, :3].astype(np.float32)
        rotation = h["rotation"][:n_frames, :3].astype(np.float32)
        velocity = h["velocity"][:n_frames, :3].astype(np.float32)
        nav_command = h["nav_command"][:n_frames].astype(np.int32)
        driving_mode = h["driving_mode"][:n_frames].astype(np.int32)
        if "target_speed" in h:
            target_speed = h["target_speed"][:n_frames].astype(np.float32)
        else:
            target_speed = np.zeros((n_frames,), dtype=np.float32)
        route_completion = h["route_completion"][:n_frames].astype(np.float32)
        traffic_light_state = h["traffic_light_state"][:n_frames].astype(np.int32)
        if "traffic_light_actor_id" in h:
            traffic_light_actor_id = h["traffic_light_actor_id"][:n_frames].astype(np.int32)
        else:
            traffic_light_actor_id = np.full((n_frames,), -1, dtype=np.int32)
        traffic_light_distance = h["traffic_light_distance"][:n_frames].astype(np.float32)
        stop_sign_distance = h["stop_sign_distance"][:n_frames].astype(np.float32)
        yield_sign_distance = h["yield_sign_distance"][:n_frames].astype(np.float32)
        steer = h["steering"][:n_frames].astype(np.float32)

        if "nearby_vehicles" in h:
            nearby_vehicles_raw = h["nearby_vehicles"][:n_frames]
        else:
            nearby_vehicles_raw = np.array(["[]"] * n_frames, dtype=object)

        if "nearby_pedestrians" in h:
            nearby_pedestrians_raw = h["nearby_pedestrians"][:n_frames]
        else:
            nearby_pedestrians_raw = np.array(["[]"] * n_frames, dtype=object)

        if "wp_buffer_head_index" in h:
            wp_buffer_head_index = h["wp_buffer_head_index"][:n_frames].astype(np.int32)
        else:
            wp_buffer_head_index = np.full((n_frames,), -1, dtype=np.int32)

    return EpisodeData(
        path=path,
        town=town,
        fps=fps,
        status=status,
        collision_frame=collision_frame,
        n_frames=n_frames,
        position=position,
        rotation=rotation,
        velocity=velocity,
        nav_command=nav_command,
        driving_mode=driving_mode,
        target_speed=target_speed,
        route_completion=route_completion,
        traffic_light_state=traffic_light_state,
        traffic_light_actor_id=traffic_light_actor_id,
        traffic_light_distance=traffic_light_distance,
        stop_sign_distance=stop_sign_distance,
        yield_sign_distance=yield_sign_distance,
        wp_buffer_head_index=wp_buffer_head_index,
        steer=steer,
        nearby_vehicles_raw=nearby_vehicles_raw,
        nearby_pedestrians_raw=nearby_pedestrians_raw,
    )


def _load_route(path: Path | None) -> RouteData:
    if path is None or not path.exists():
        return RouteData(
            path=None,
            points_xyz=np.zeros((0, 3), dtype=np.float32),
            points_junction=np.zeros((0,), dtype=bool),
            points_yaw=np.zeros((0,), dtype=np.float32),
            points_road_id=np.zeros((0,), dtype=np.int32),
            points_lane_id=np.zeros((0,), dtype=np.int32),
            options=[],
        )

    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Failed to read route sidecar %s: %s", path, exc)
        return RouteData(
            path=None,
            points_xyz=np.zeros((0, 3), dtype=np.float32),
            points_junction=np.zeros((0,), dtype=bool),
            points_yaw=np.zeros((0,), dtype=np.float32),
            points_road_id=np.zeros((0,), dtype=np.int32),
            points_lane_id=np.zeros((0,), dtype=np.int32),
            options=[],
        )

    wps = data.get("waypoints", [])
    if not isinstance(wps, list) or not wps:
        return RouteData(
            path=path,
            points_xyz=np.zeros((0, 3), dtype=np.float32),
            points_junction=np.zeros((0,), dtype=bool),
            points_yaw=np.zeros((0,), dtype=np.float32),
            points_road_id=np.zeros((0,), dtype=np.int32),
            points_lane_id=np.zeros((0,), dtype=np.int32),
            options=[],
        )

    xyz = np.array(
        [[float(w.get("x", 0.0)), float(w.get("y", 0.0)), float(w.get("z", 0.0))] for w in wps],
        dtype=np.float32,
    )
    is_junction = np.array([bool(w.get("is_junction", False)) for w in wps], dtype=bool)
    yaw = np.array([float(w.get("yaw", 0.0)) for w in wps], dtype=np.float32)
    road_id = np.array([int(w.get("road_id", -1)) for w in wps], dtype=np.int32)
    lane_id = np.array([int(w.get("lane_id", 0)) for w in wps], dtype=np.int32)
    options = [str(w.get("road_option", "LANEFOLLOW")) for w in wps]
    return RouteData(
        path=path,
        points_xyz=xyz,
        points_junction=is_junction,
        points_yaw=yaw,
        points_road_id=road_id,
        points_lane_id=lane_id,
        options=options,
    )


def _decode_json_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        text = value.decode("utf-8", errors="ignore")
    else:
        text = str(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


class _ReplayWaypoint:
    def __init__(self, x: float, y: float, z: float, yaw: float, is_junction: bool) -> None:
        self.transform = carla.Transform(
            carla.Location(x=float(x), y=float(y), z=float(z)),
            carla.Rotation(pitch=0.0, yaw=float(yaw), roll=0.0),
        )
        self.is_junction = bool(is_junction)

    def get_junction(self) -> None:
        return None


class _ReplayVehicle:
    def __init__(self) -> None:
        self._transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0), carla.Rotation())

    def set_pose(self, position: np.ndarray, rotation: np.ndarray) -> None:
        self._transform = carla.Transform(
            carla.Location(
                x=float(position[0]),
                y=float(position[1]),
                z=float(position[2]),
            ),
            carla.Rotation(
                pitch=float(rotation[0]),
                yaw=float(rotation[1]),
                roll=float(rotation[2]),
            ),
        )

    def get_transform(self) -> carla.Transform:
        return self._transform


class _ReplayExpertDebugProxy:
    def __init__(self, route: RouteData, carla_map: carla.Map) -> None:
        from agents.expert.privileged_expert import ExpertConfig
        from agents.navigation.local_planner import RoadOption

        self.config = ExpertConfig()
        self._carla_map = carla_map
        self._route: list[tuple[Any, Any]] = []
        self._route_index = 0
        self._wp_buffer: deque[tuple[Any, Any]] = deque()
        self._corridor_3d_segments = None
        self._corridor = None
        self._cached_half_lane = float(self.config.lane_half_width)
        self._stop_sign_active = None
        self._yield_active = None

        fallback_option = RoadOption.LANEFOLLOW
        for i in range(route.points_xyz.shape[0]):
            xyz = route.points_xyz[i]
            yaw = float(route.points_yaw[i]) if i < route.points_yaw.shape[0] else 0.0
            opt_name = route.options[i] if i < len(route.options) else "LANEFOLLOW"
            option = getattr(RoadOption, opt_name, fallback_option)
            wp = _ReplayWaypoint(
                x=float(xyz[0]),
                y=float(xyz[1]),
                z=float(xyz[2]),
                yaw=yaw,
                is_junction=(
                    bool(route.points_junction[i]) if i < route.points_junction.shape[0] else False
                ),
            )
            self._route.append((wp, option))

    def sync_route_state(self, wp_head_index: int) -> None:
        if not self._route:
            self._route_index = 0
            self._wp_buffer = deque()
            return

        if int(wp_head_index) >= 0:
            self._route_index = max(0, min(int(wp_head_index), len(self._route) - 1))
        else:
            self._route_index = max(0, min(self._route_index, len(self._route) - 1))

        end = min(len(self._route), self._route_index + 24)
        self._wp_buffer = deque(self._route[self._route_index : end])

    def _route_waypoint_sources(self, lookahead: int = 180) -> Iterator[tuple[Any, Any]]:
        if not self._route:
            return
        start = max(0, min(self._route_index, len(self._route) - 1))
        end = min(len(self._route), start + max(1, int(lookahead)))
        for i in range(start, end):
            rwp, ropt = self._route[i]
            map_wp = self._carla_map.get_waypoint(
                rwp.transform.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if map_wp is not None:
                yield map_wp, ropt
            else:
                yield rwp, ropt

    @staticmethod
    def _lead_vehicle_info(
        nearby_vehicles: list[dict[str, Any]],
        _ego_speed_kmh: float,
    ) -> tuple[float | None, float | None, float | None]:
        best: tuple[float, float, float] | None = None
        for v in nearby_vehicles:
            is_ahead = bool(v.get("is_ahead", False))
            same_direction = bool(v.get("same_direction", True))
            if not is_ahead or not same_direction:
                continue
            try:
                distance = float(v.get("distance", 1e9))
            except (TypeError, ValueError):
                continue
            if distance < 0.0:
                continue
            speed = float(v.get("speed", v.get("speed_kmh", 0.0)))
            lateral = float(v.get("lateral", 0.0))
            if best is None or distance < best[0]:
                best = (distance, speed, lateral)
        if best is None:
            return None, None, None
        return best


@dataclass
class CollectorDebugContext:
    draw_fn: Any
    expert: _ReplayExpertDebugProxy
    vehicle: _ReplayVehicle


def _build_collector_debug_context(world: carla.World, route: RouteData) -> CollectorDebugContext:
    from src.data.expert_collector import ExpertDataCollector

    return CollectorDebugContext(
        draw_fn=ExpertDataCollector._draw_route_debug,
        expert=_ReplayExpertDebugProxy(route=route, carla_map=world.get_map()),
        vehicle=_ReplayVehicle(),
    )


def _build_debug_info(ep: EpisodeData, frame_idx: int) -> SimpleNamespace:
    speed_kmh = float(np.linalg.norm(ep.velocity[frame_idx]) * 3.6)
    nearby_vehicles = _decode_json_list(ep.nearby_vehicles_raw[frame_idx])
    nearby_pedestrians = _decode_json_list(ep.nearby_pedestrians_raw[frame_idx])
    mode_value = int(ep.driving_mode[frame_idx])
    if mode_value not in MODE_NAMES:
        mode_value = 1
    return SimpleNamespace(
        current_speed=speed_kmh,
        target_speed=float(ep.target_speed[frame_idx]),
        driving_mode=mode_value,
        nearby_vehicles=nearby_vehicles,
        nearby_pedestrians=nearby_pedestrians,
        traffic_light_actor_id=int(ep.traffic_light_actor_id[frame_idx]),
        stop_sign_distance=float(ep.stop_sign_distance[frame_idx]),
        yield_sign_distance=float(ep.yield_sign_distance[frame_idx]),
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_route_head_index(ep: EpisodeData, route: RouteData, frame_idx: int) -> int:
    n = route.points_xyz.shape[0]
    if n <= 0:
        return -1

    raw = int(ep.wp_buffer_head_index[frame_idx])
    if raw < 0 or raw >= n:
        raw = 0

    ego_xy = ep.position[frame_idx, :2].astype(np.float64)
    yaw = math.radians(float(ep.rotation[frame_idx, 1]))
    fwd = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)

    lo = max(0, raw - 20)
    hi = min(n, raw + 80)
    window = route.points_xyz[lo:hi, :2].astype(np.float64)
    if window.shape[0] == 0:
        return raw

    rel = window - ego_xy
    longitudinal = rel @ fwd
    dist = np.linalg.norm(rel, axis=1)
    ahead = np.where(longitudinal >= -1.0)[0]
    if ahead.size > 0:
        best_local = int(ahead[np.argmin(dist[ahead])])
        return lo + best_local

    all_rel = route.points_xyz[:, :2].astype(np.float64) - ego_xy
    return int(np.argmin(np.linalg.norm(all_rel, axis=1)))


def _build_road_lane_cache(carla_map: Any, spacing_m: float = 2.0) -> RoadLaneCache:
    by_road_lane: dict[int, dict[int, list[Any]]] = {}
    for wp in carla_map.generate_waypoints(float(spacing_m)):
        if wp.lane_type != carla.LaneType.Driving:
            continue
        road_id = int(getattr(wp, "road_id", -1))
        lane_id = int(getattr(wp, "lane_id", 0))
        if road_id < 0 or lane_id == 0:
            continue
        by_road_lane.setdefault(road_id, {}).setdefault(lane_id, []).append(wp)
    return RoadLaneCache(by_road_lane=by_road_lane)


def _find_cached_lane_waypoint(
    road_cache: RoadLaneCache | None,
    road_id: int | None,
    lane_id: int | None,
    ref_loc: Any,
) -> Any:
    if road_cache is None or road_id is None or lane_id is None:
        return None
    lane_group = road_cache.by_road_lane.get(int(road_id), {}).get(int(lane_id), [])
    if not lane_group:
        return None

    best = None
    best_d2 = float("inf")
    rx = float(ref_loc.x)
    ry = float(ref_loc.y)
    for wp in lane_group:
        wloc = wp.transform.location
        dx = float(wloc.x) - rx
        dy = float(wloc.y) - ry
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best = wp
    return best


def _road_direction_from_lane_ids(
    road_cache: RoadLaneCache | None,
    road_id: int | None,
    lane_id: int | None,
) -> str:
    if road_cache is None or road_id is None:
        return "--"

    lane_map = road_cache.by_road_lane.get(int(road_id), {})
    if not lane_map:
        return "--"

    lane_ids = sorted(int(lid) for lid in lane_map.keys() if int(lid) != 0)
    if not lane_ids:
        return "--"

    def _group_dir(ids: list[int]) -> str:
        yaws: list[float] = []
        for lid in ids:
            samples = lane_map.get(lid, [])
            if not samples:
                continue
            step = max(1, len(samples) // 25)
            for wp in samples[::step]:
                yaws.append(float(wp.transform.rotation.yaw))
        if not yaws:
            return "--"
        return _yaw_to_cardinal(_circular_mean_deg(yaws))

    positive = [lid for lid in lane_ids if lid > 0]
    negative = [lid for lid in lane_ids if lid < 0]

    if positive and negative:
        dir_pos = _group_dir(positive)
        dir_neg = _group_dir(negative)
        if lane_id is not None:
            dominant = dir_pos if int(lane_id) > 0 else dir_neg
            opposite = dir_neg if int(lane_id) > 0 else dir_pos
            return f"{dominant} / {opposite} (bi-dir)"
        return f"{dir_pos} / {dir_neg} (bi-dir)"

    single_group = positive if positive else negative
    return f"{_group_dir(single_group)} (one-way)"


def _lane_position_from_lane_id(
    road_cache: RoadLaneCache | None,
    road_id: int | None,
    lane_id: int | None,
) -> str:
    if road_cache is None or road_id is None or lane_id is None or int(lane_id) == 0:
        return "--"

    lane_map = road_cache.by_road_lane.get(int(road_id), {})
    if not lane_map:
        return "--"

    lid = int(lane_id)
    sign = 1 if lid > 0 else -1
    same_side = sorted((x for x in lane_map.keys() if int(x) * sign > 0), reverse=True)
    if not same_side:
        return "--"
    if len(same_side) == 1:
        return "single"

    if lid not in same_side:
        return "--"

    idx_from_right = same_side.index(lid) + 1
    idx_from_left = len(same_side) - idx_from_right + 1

    if idx_from_right == 1:
        return "first from right"
    if idx_from_left == 1:
        return "first from left"
    return f"{_ordinal(idx_from_right)} from right ({_ordinal(idx_from_left)} from left)"


def _bbox_corners_2d(
    center_xy: np.ndarray,
    yaw_deg: float,
    length_m: float,
    width_m: float,
) -> np.ndarray:
    half_l = max(0.1, float(length_m) * 0.5)
    half_w = max(0.1, float(width_m) * 0.5)
    yaw = math.radians(float(yaw_deg))
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    fwd = np.array([cos_y, sin_y], dtype=np.float64)
    right = np.array([-sin_y, cos_y], dtype=np.float64)

    return np.array(
        [
            center_xy + fwd * half_l + right * half_w,
            center_xy + fwd * half_l - right * half_w,
            center_xy - fwd * half_l - right * half_w,
            center_xy - fwd * half_l + right * half_w,
        ],
        dtype=np.float64,
    )


def _sat_overlap(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
    def _project(vertices: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
        dots = vertices @ axis
        return float(dots.min()), float(dots.max())

    for poly in (poly_a, poly_b):
        n = len(poly)
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            axis = np.array([-edge[1], edge[0]], dtype=np.float64)
            norm = float(np.linalg.norm(axis))
            if norm < 1e-8:
                continue
            axis /= norm

            min_a, max_a = _project(poly_a, axis)
            min_b, max_b = _project(poly_b, axis)
            if max_a < min_b or max_b < min_a:
                return False
    return True


def _local_to_world_center(
    ego_xy: np.ndarray,
    yaw_deg: float,
    longitudinal: float,
    lateral: float,
    default_z: float,
) -> tuple[np.ndarray, float]:
    yaw = math.radians(float(yaw_deg))
    fwd = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    right = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    xy = ego_xy + fwd * float(longitudinal) + right * float(lateral)
    return xy, float(default_z)


def _dot2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[0] + a[1] * b[1])


def _local_to_world_no_mutate(tf: Any, loc: Any) -> Any:
    yaw = math.radians(float(tf.rotation.yaw))
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    x = cos_y * float(loc.x) - sin_y * float(loc.y) + float(tf.location.x)
    y = sin_y * float(loc.x) + cos_y * float(loc.y) + float(tf.location.y)
    z = float(loc.z) + float(tf.location.z)
    return carla.Location(x=x, y=y, z=z)


def _circular_mean_deg(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sum(math.sin(math.radians(v)) for v in values)
    c = sum(math.cos(math.radians(v)) for v in values)
    return math.degrees(math.atan2(s, c))


def _collect_same_direction_lane_band(base_wp: Any) -> list[Any]:
    band = [base_wp]
    seen_ids = {int(base_wp.id)}
    base_sign = 1 if int(getattr(base_wp, "lane_id", 1)) >= 0 else -1

    def _walk(side: str) -> None:
        current = base_wp
        while True:
            next_wp = current.get_left_lane() if side == "left" else current.get_right_lane()
            if next_wp is None:
                return
            if next_wp.lane_type != carla.LaneType.Driving:
                return
            next_sign = 1 if int(getattr(next_wp, "lane_id", 1)) >= 0 else -1
            if next_sign != base_sign:
                return
            next_id = int(next_wp.id)
            if next_id in seen_ids:
                return
            seen_ids.add(next_id)
            band.append(next_wp)
            current = next_wp

    _walk("left")
    _walk("right")
    return band


def _collect_road_driving_lane_band(base_wp: Any) -> list[Any]:
    band = [base_wp]
    seen_lane_ids = {int(getattr(base_wp, "lane_id", 0))}
    base_road_id = int(getattr(base_wp, "road_id", -1))

    def _walk(side: str) -> None:
        current = base_wp
        while True:
            next_wp = current.get_left_lane() if side == "left" else current.get_right_lane()
            if next_wp is None:
                return
            if next_wp.lane_type != carla.LaneType.Driving:
                return
            if int(getattr(next_wp, "road_id", -2)) != base_road_id:
                return
            lane_id = int(getattr(next_wp, "lane_id", 0))
            if lane_id in seen_lane_ids:
                return
            seen_lane_ids.add(lane_id)
            band.append(next_wp)
            current = next_wp

    _walk("left")
    _walk("right")
    return band


def _compute_lane_zone(
    waypoints: list[Any],
    z_offset: float = 0.06,
) -> tuple[Any, Any, float, float] | None:
    if not waypoints:
        return None

    locs = [wp.transform.location for wp in waypoints]
    yaws = [float(wp.transform.rotation.yaw) for wp in waypoints]
    lane_widths = [float(getattr(wp, "lane_width", 3.5)) for wp in waypoints]

    center = carla.Location(
        x=float(np.mean([loc.x for loc in locs])),
        y=float(np.mean([loc.y for loc in locs])),
        z=float(np.mean([loc.z for loc in locs])) + z_offset,
    )
    yaw = _circular_mean_deg(yaws)
    lane_w = max(2.6, float(np.mean(lane_widths)) if lane_widths else 3.5)

    lane_span = 0.0
    for i in range(len(locs)):
        for j in range(i + 1, len(locs)):
            lane_span = max(lane_span, float(locs[i].distance(locs[j])))

    total_width = lane_w if len(locs) == 1 else lane_span + lane_w
    depth = max(1.2, min(2.2, lane_w * 0.6))
    rot = carla.Rotation(pitch=0.0, yaw=float(yaw), roll=0.0)
    return center, rot, total_width, depth


def _collect_tl_waypoints(tl: Any) -> list[Any]:
    try:
        swps = list(tl.get_stop_waypoints())
    except (AttributeError, RuntimeError):
        swps = []
    if swps:
        return swps
    try:
        return list(tl.get_affected_lane_waypoints())
    except (AttributeError, RuntimeError):
        return []


def _collect_sign_waypoints(carla_map: Any, sign: Any, trigger_loc: Any) -> list[Any]:
    try:
        affected = list(sign.get_affected_lane_waypoints())
    except (AttributeError, RuntimeError):
        affected = []

    driving = [wp for wp in affected if wp.lane_type == carla.LaneType.Driving]
    if driving:
        return driving

    base_wp = carla_map.get_waypoint(
        trigger_loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if base_wp is None:
        return []
    return _collect_same_direction_lane_band(base_wp)


def _road_id_at(carla_map: Any, loc: Any) -> int | None:
    wp = carla_map.get_waypoint(
        loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if wp is None:
        return None
    return int(wp.road_id)


def _ego_road_width_cap(carla_map: Any, ego_loc: Any) -> tuple[float, int | None]:
    ego_wp = carla_map.get_waypoint(
        ego_loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if ego_wp is None:
        return 12.0, None

    band = _collect_road_driving_lane_band(ego_wp)
    if not band:
        return max(2.6, float(getattr(ego_wp, "lane_width", 3.5))), int(ego_wp.road_id)

    total_width = sum(max(0.1, float(getattr(wp, "lane_width", 3.5))) for wp in band)
    return max(2.6, float(total_width)), int(ego_wp.road_id)


def _trigger_edge_toward_junction(
    carla_map: Any,
    sign_tf: Any,
    trigger_center: Any,
    trigger_extent_x: float,
) -> Any:
    yaw = math.radians(float(sign_tf.rotation.yaw))
    fwd_x = math.cos(yaw)
    fwd_y = math.sin(yaw)

    cand_plus = carla.Location(
        x=float(trigger_center.x) + fwd_x * trigger_extent_x,
        y=float(trigger_center.y) + fwd_y * trigger_extent_x,
        z=float(trigger_center.z),
    )
    cand_minus = carla.Location(
        x=float(trigger_center.x) - fwd_x * trigger_extent_x,
        y=float(trigger_center.y) - fwd_y * trigger_extent_x,
        z=float(trigger_center.z),
    )

    def _junction_score(loc: Any) -> int:
        wp = carla_map.get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if wp is None:
            return 2
        return 0 if wp.is_junction else 1

    plus_score = _junction_score(cand_plus)
    minus_score = _junction_score(cand_minus)
    if plus_score < minus_score:
        return cand_plus
    if minus_score < plus_score:
        return cand_minus
    return cand_plus


def _draw_active_control_zone(
    world: carla.World,
    ego_loc: Any,
    ego_fwd_2d: np.ndarray,
    traffic_light_actor_id: int,
    stop_sign_distance: float,
    yield_sign_distance: float,
    life_time: float,
) -> None:
    debug = world.debug
    carla_map = world.get_map()
    stop_search_radius = 70.0

    fwd_norm = float(np.linalg.norm(ego_fwd_2d))
    if fwd_norm < 1e-6:
        ego_fwd_2d = np.array([1.0, 0.0], dtype=np.float64)
    else:
        ego_fwd_2d = ego_fwd_2d / fwd_norm

    road_width_cap, _ = _ego_road_width_cap(carla_map, ego_loc)
    controls: list[dict[str, Any]] = []

    def _ahead_longitudinal(target_loc: Any) -> float:
        rel = np.array(
            [float(target_loc.x - ego_loc.x), float(target_loc.y - ego_loc.y)],
            dtype=np.float64,
        )
        return _dot2(rel, ego_fwd_2d)

    actors = world.get_actors()

    for tl in actors.filter("traffic.traffic_light*"):
        tl_loc = tl.get_transform().location
        if tl_loc.distance(ego_loc) > stop_search_radius:
            continue

        stop_wps = _collect_tl_waypoints(tl)
        ahead_wps: list[tuple[Any, Any, float]] = []
        for wp in stop_wps:
            loc = wp.transform.location
            if loc.distance(ego_loc) > stop_search_radius:
                continue
            longitudinal = _ahead_longitudinal(loc)
            if longitudinal <= 0.0:
                continue
            ahead_wps.append((wp, loc, longitudinal))
        if not ahead_wps:
            continue

        zone = _compute_lane_zone([wp for wp, _loc, _lon in ahead_wps], z_offset=0.85)
        if zone is None:
            continue
        zone_center, zone_rot, zone_width, zone_depth = zone
        zone_width = min(zone_width, road_width_cap)

        _best_wp, best_loc, best_long = min(
            ahead_wps,
            key=lambda item: float(item[1].distance(ego_loc)),
        )
        controls.append(
            {
                "kind": "TL",
                "actor_id": int(tl.id),
                "shape": "box",
                "box_center": zone_center,
                "box_rot": zone_rot,
                "box_depth": float(zone_depth),
                "box_width": float(zone_width),
                "box_height_half": 2.0,
                "anchor_loc": best_loc,
                "label_loc": carla.Location(
                    x=float(tl_loc.x),
                    y=float(tl_loc.y),
                    z=float(tl_loc.z) + 2.8,
                ),
                "distance": float(best_loc.distance(ego_loc)),
                "longitudinal": float(best_long),
                "road_id": _road_id_at(carla_map, best_loc),
            }
        )

    for sign in actors.filter("traffic.stop*"):
        trigger = getattr(sign, "trigger_volume", None)
        if trigger is None:
            continue

        sign_tf = sign.get_transform()
        trigger_loc = _local_to_world_no_mutate(sign_tf, trigger.location)
        if trigger_loc.distance(ego_loc) > stop_search_radius:
            continue
        longitudinal = _ahead_longitudinal(trigger_loc)
        if longitudinal <= 0.0:
            continue

        sign_wps = _collect_sign_waypoints(carla_map, sign, trigger_loc)
        zone = _compute_lane_zone(sign_wps, z_offset=0.06)
        if zone is not None:
            zone_center, zone_rot, zone_width, zone_depth = zone
        else:
            zone_center = carla.Location(
                x=float(trigger_loc.x),
                y=float(trigger_loc.y),
                z=float(trigger_loc.z) + 0.06,
            )
            zone_rot = carla.Rotation(
                pitch=0.0,
                yaw=float(sign_tf.rotation.yaw),
                roll=0.0,
            )
            zone_width = max(2.6, float(trigger.extent.y) * 2.0)
            zone_depth = max(1.2, float(trigger.extent.x) * 2.0)

        zone_width = min(zone_width, road_width_cap)
        sign_loc = sign_tf.location
        controls.append(
            {
                "kind": "STOP",
                "actor_id": int(sign.id),
                "shape": "box",
                "box_center": zone_center,
                "box_rot": zone_rot,
                "box_depth": float(zone_depth),
                "box_width": float(zone_width),
                "box_height_half": 1.0,
                "anchor_loc": zone_center,
                "label_loc": carla.Location(
                    x=float(sign_loc.x),
                    y=float(sign_loc.y),
                    z=float(sign_loc.z) + 2.4,
                ),
                "distance": float(zone_center.distance(ego_loc)),
                "longitudinal": float(longitudinal),
                "road_id": _road_id_at(carla_map, zone_center),
            }
        )

    for sign in actors.filter("traffic.yield*"):
        trigger = getattr(sign, "trigger_volume", None)
        if trigger is None:
            continue

        sign_tf = sign.get_transform()
        trigger_loc = _local_to_world_no_mutate(sign_tf, trigger.location)
        if trigger_loc.distance(ego_loc) > stop_search_radius:
            continue
        longitudinal = _ahead_longitudinal(trigger_loc)
        if longitudinal <= 0.0:
            continue

        sign_wps = _collect_sign_waypoints(carla_map, sign, trigger_loc)
        zone = _compute_lane_zone(sign_wps, z_offset=0.06)
        if zone is not None:
            _zone_center, zone_rot, zone_width, _zone_depth = zone
        else:
            zone_rot = carla.Rotation(
                pitch=0.0,
                yaw=float(sign_tf.rotation.yaw),
                roll=0.0,
            )
            zone_width = max(2.6, float(trigger.extent.y) * 2.0)

        zone_width = min(zone_width, road_width_cap)
        line_center = _trigger_edge_toward_junction(
            carla_map,
            sign_tf,
            trigger_loc,
            max(0.1, float(trigger.extent.x)),
        )
        yaw = math.radians(float(zone_rot.yaw))
        right_x = -math.sin(yaw)
        right_y = math.cos(yaw)
        half_width = float(zone_width) * 0.5
        line_z = float(line_center.z) + 0.03
        p1 = carla.Location(
            x=float(line_center.x) - right_x * half_width,
            y=float(line_center.y) - right_y * half_width,
            z=line_z,
        )
        p2 = carla.Location(
            x=float(line_center.x) + right_x * half_width,
            y=float(line_center.y) + right_y * half_width,
            z=line_z,
        )

        sign_loc = sign_tf.location
        line_longitudinal = _ahead_longitudinal(line_center)
        controls.append(
            {
                "kind": "YIELD",
                "actor_id": int(sign.id),
                "shape": "line",
                "line_p1": p1,
                "line_p2": p2,
                "anchor_loc": line_center,
                "label_loc": carla.Location(
                    x=float(sign_loc.x),
                    y=float(sign_loc.y),
                    z=float(sign_loc.z) + 2.4,
                ),
                "distance": float(line_center.distance(ego_loc)),
                "longitudinal": float(line_longitudinal),
                "road_id": _road_id_at(carla_map, line_center),
            }
        )

    if not controls:
        return

    active_control: dict[str, Any] | None = None

    if int(traffic_light_actor_id) > 0:
        for c in controls:
            if c["kind"] == "TL" and int(c.get("actor_id", -1)) == int(traffic_light_actor_id):
                active_control = c
                break

    if active_control is None and float(stop_sign_distance) > 0.0:
        stop_controls = [
            c for c in controls if c["kind"] == "STOP" and float(c.get("longitudinal", -1.0)) > 0.0
        ]
        if stop_controls:
            active_control = min(
                stop_controls,
                key=lambda c: abs(float(c["distance"]) - float(stop_sign_distance)),
            )

    if active_control is None and float(yield_sign_distance) > 0.0:
        yield_controls = [
            c for c in controls if c["kind"] == "YIELD" and float(c.get("longitudinal", -1.0)) > 0.0
        ]
        if yield_controls:
            active_control = min(
                yield_controls,
                key=lambda c: abs(float(c["distance"]) - float(yield_sign_distance)),
            )

    if active_control is None:
        return

    if active_control["shape"] == "box":
        debug.draw_box(
            carla.BoundingBox(
                active_control["box_center"],
                carla.Vector3D(
                    x=float(active_control["box_depth"]) * 0.5,
                    y=float(active_control["box_width"]) * 0.5,
                    z=float(active_control["box_height_half"]),
                ),
            ),
            active_control["box_rot"],
            thickness=0.13,
            color=INDIGO_COLOR,
            life_time=life_time,
        )
    else:
        debug.draw_line(
            active_control["line_p1"],
            active_control["line_p2"],
            thickness=0.12,
            color=INDIGO_COLOR,
            life_time=life_time,
        )

    marker = active_control["label_loc"]
    kind = active_control["kind"]
    dist = float(active_control["distance"])
    debug.draw_point(
        marker,
        size=0.15,
        color=WHITE_COLOR,
        life_time=life_time,
    )
    debug.draw_string(
        carla.Location(marker.x, marker.y, marker.z + 0.55),
        f"{kind} {dist:.1f}m",
        draw_shadow=True,
        color=WHITE_COLOR,
        life_time=life_time,
    )
    debug.draw_line(
        marker,
        carla.Location(
            x=float(active_control["anchor_loc"].x),
            y=float(active_control["anchor_loc"].y),
            z=float(active_control["anchor_loc"].z) + 0.10,
        ),
        thickness=0.04,
        color=WHITE_COLOR,
        life_time=life_time,
    )


def _angle_diff_deg(a: float, b: float) -> float:
    d = (float(a) - float(b) + 180.0) % 360.0 - 180.0
    return abs(d)


def _yaw_to_cardinal(yaw_deg: float) -> str:
    y = (float(yaw_deg) + 360.0) % 360.0
    if HEADING_LABEL_STYLE == "cardinal4":
        if y < 45.0 or y >= 315.0:
            return "Eastbound"
        if y < 135.0:
            return "Southbound"
        if y < 225.0:
            return "Westbound"
        return "Northbound"

    # CARLA yaw uses 0 deg = EAST, 90 deg = SOUTH.
    labels = (
        "EAST",
        "EAST-SOUTHEAST",
        "SOUTHEAST",
        "SOUTH-SOUTHEAST",
        "SOUTH",
        "SOUTH-SOUTHWEST",
        "SOUTHWEST",
        "WEST-SOUTHWEST",
        "WEST",
        "WEST-NORTHWEST",
        "NORTHWEST",
        "NORTH-NORTHWEST",
        "NORTH",
        "NORTH-NORTHEAST",
        "NORTHEAST",
        "EAST-NORTHEAST",
    )
    idx = int(((y + 11.25) % 360.0) // 22.5)
    return labels[idx]


def _ordinal(n: int) -> str:
    if n <= 0:
        return str(n)
    if n == 1:
        return "first"
    if n == 2:
        return "second"
    if n == 3:
        return "third"
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _lane_position_descriptor(wp: Any, reference_yaw_deg: float) -> str:
    if wp is None:
        return "--"

    same_dir = _collect_same_direction_lane_band(wp)
    if len(same_dir) <= 1:
        return "single"

    ref_yaw = math.radians(float(reference_yaw_deg))
    right_vec = np.array([-math.sin(ref_yaw), math.cos(ref_yaw)], dtype=np.float64)

    centre = np.array(
        [float(wp.transform.location.x), float(wp.transform.location.y)], dtype=np.float64
    )
    entries: list[tuple[float, Any]] = []
    for cand in same_dir:
        c = np.array(
            [float(cand.transform.location.x), float(cand.transform.location.y)],
            dtype=np.float64,
        )
        lateral = _dot2(c - centre, right_vec)
        entries.append((lateral, cand))

    entries.sort(key=lambda item: item[0], reverse=True)  # right-most first
    idx_from_right = 1
    for i, (_lat, cand) in enumerate(entries):
        if int(cand.id) == int(wp.id):
            idx_from_right = i + 1
            break
    idx_from_left = len(entries) - idx_from_right + 1

    if idx_from_right == 1:
        return "first from right"
    if idx_from_left == 1:
        return "first from left"
    return f"{_ordinal(idx_from_right)} from right ({_ordinal(idx_from_left)} from left)"


def _avg_yaw_from_waypoints(waypoints: list[Any]) -> float:
    if not waypoints:
        return 0.0
    return _circular_mean_deg([float(wp.transform.rotation.yaw) for wp in waypoints])


def _road_traffic_direction(carla_map: Any, wp: Any) -> str:
    if wp is None:
        return "--"
    road_band = _collect_road_driving_lane_band(wp)
    if not road_band:
        return _yaw_to_cardinal(float(wp.transform.rotation.yaw))

    lane_id = int(getattr(wp, "lane_id", 0))
    sign = 1 if lane_id >= 0 else -1
    same = [
        cand for cand in road_band if (1 if int(getattr(cand, "lane_id", 0)) >= 0 else -1) == sign
    ]
    opp = [
        cand for cand in road_band if (1 if int(getattr(cand, "lane_id", 0)) >= 0 else -1) != sign
    ]

    same_dir = _yaw_to_cardinal(_avg_yaw_from_waypoints(same if same else [wp]))
    if not opp:
        return f"{same_dir} (one-way)"

    opp_dir = _yaw_to_cardinal(_avg_yaw_from_waypoints(opp))
    if same_dir == opp_dir:
        return f"{same_dir}"
    return f"{same_dir} / {opp_dir} (bi-dir)"


def _draw_spectator_corner_hud(
    world: carla.World,
    lines: list[tuple[str, carla.Color]],
    life_time: float,
) -> None:
    if not lines:
        return

    spect_tf = world.get_spectator().get_transform()
    rot = spect_tf.rotation
    yaw = math.radians(float(rot.yaw))
    pitch = math.radians(float(rot.pitch))

    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    forward = np.array([cp * cy, cp * sy, sp], dtype=np.float64)
    right = np.array([-sy, cy, 0.0], dtype=np.float64)
    up = np.array([-sp * cy, -sp * sy, cp], dtype=np.float64)

    base = np.array(
        [float(spect_tf.location.x), float(spect_tf.location.y), float(spect_tf.location.z)],
        dtype=np.float64,
    )

    anchor = base + forward * 10.0 - right * 7.5 + up * 3.9
    line_step = 0.34

    debug = world.debug
    for i, (text, color) in enumerate(lines):
        p = anchor - up * (i * line_step)
        debug.draw_string(
            carla.Location(x=float(p[0]), y=float(p[1]), z=float(p[2])),
            text,
            draw_shadow=True,
            color=color,
            life_time=life_time,
            persistent_lines=False,
        )


def _draw_lane_direction_hud(
    world: carla.World,
    ep: EpisodeData,
    route: RouteData,
    frame_idx: int,
    ridx: int,
    ego_loc: Any,
    road_cache: RoadLaneCache | None,
    life_time: float,
) -> None:
    carla_map = world.get_map()

    ego_yaw = float(ep.rotation[frame_idx, 1])
    ego_heading = _yaw_to_cardinal(ego_yaw)

    live_wp = carla_map.get_waypoint(
        ego_loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    live_road_id_val: int | None = None
    live_lane_id_val: int | None = None
    if live_wp is not None:
        live_road_id_val = int(getattr(live_wp, "road_id", -1))
        live_lane_id_val = int(getattr(live_wp, "lane_id", 0))

    ep_road_id_val: int | None = None
    ep_lane_id_val: int | None = None
    ep_ref_loc = None
    if 0 <= ridx < route.points_xyz.shape[0]:
        ep_road_id_val = (
            int(route.points_road_id[ridx]) if ridx < route.points_road_id.shape[0] else None
        )
        ep_lane_id_val = (
            int(route.points_lane_id[ridx]) if ridx < route.points_lane_id.shape[0] else None
        )
        rp = route.points_xyz[ridx]
        ep_ref_loc = carla.Location(x=float(rp[0]), y=float(rp[1]), z=float(rp[2]))

    ep_wp = None
    if ep_ref_loc is not None:
        ep_wp = _find_cached_lane_waypoint(
            road_cache=road_cache,
            road_id=ep_road_id_val,
            lane_id=ep_lane_id_val,
            ref_loc=ep_ref_loc,
        )
        if ep_wp is None:
            ep_wp = carla_map.get_waypoint(
                ep_ref_loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )

    live_junction = "--"
    if live_wp is not None and live_wp.is_junction:
        junction = live_wp.get_junction()
        if junction is not None:
            live_junction = str(int(junction.id))

    ep_junction = "--"
    if ep_wp is not None and ep_wp.is_junction:
        junction = ep_wp.get_junction()
        if junction is not None:
            ep_junction = str(int(junction.id))

    live_lane_id = "--" if live_lane_id_val is None else str(int(live_lane_id_val))
    ep_lane_id = "--" if ep_lane_id_val is None else str(int(ep_lane_id_val))
    live_road_id = "--" if live_road_id_val is None else str(int(live_road_id_val))
    ep_road_id = "--" if ep_road_id_val is None else str(int(ep_road_id_val))

    live_lane_pos = _lane_position_descriptor(live_wp, ego_yaw)
    ep_lane_ref_yaw = (
        float(route.points_yaw[ridx]) if (0 <= ridx < route.points_yaw.shape[0]) else ego_yaw
    )
    ep_lane_pos = _lane_position_descriptor(ep_wp, ep_lane_ref_yaw)
    if ep_lane_pos == "--" and ep_road_id_val is not None and ep_lane_id_val is not None:
        ep_lane_pos = _lane_position_from_lane_id(
            road_cache=road_cache,
            road_id=ep_road_id_val,
            lane_id=ep_lane_id_val,
        )

    live_lane_dir = (
        "--" if live_wp is None else _yaw_to_cardinal(float(live_wp.transform.rotation.yaw))
    )
    if 0 <= ridx < route.points_yaw.shape[0]:
        ep_lane_dir = _yaw_to_cardinal(float(route.points_yaw[ridx]))
    elif ep_wp is not None:
        ep_lane_dir = _yaw_to_cardinal(float(ep_wp.transform.rotation.yaw))
    else:
        ep_lane_dir = "--"

    live_road_dir = _road_direction_from_lane_ids(
        road_cache=road_cache,
        road_id=live_road_id_val,
        lane_id=live_lane_id_val,
    )
    ep_road_dir = _road_direction_from_lane_ids(
        road_cache=road_cache,
        road_id=ep_road_id_val,
        lane_id=ep_lane_id_val,
    )

    lane_match_text = "MISMATCH"
    lane_match_color = SAT_COLOR
    if (
        live_lane_id_val is not None
        and ep_lane_id_val is not None
        and live_road_id_val is not None
        and ep_road_id_val is not None
        and int(live_lane_id_val) == int(ep_lane_id_val)
        and int(live_road_id_val) == int(ep_road_id_val)
    ):
        lane_match_text = "MATCH"
        lane_match_color = EGO_COLOR

    live_lane_pair = "--"
    if live_road_id_val is not None and live_lane_id_val is not None:
        live_lane_pair = f"r{int(live_road_id_val)}/l{int(live_lane_id_val)}"

    ep_lane_pair = "--"
    if ep_road_id_val is not None and ep_lane_id_val is not None:
        ep_lane_pair = f"r{int(ep_road_id_val)}/l{int(ep_lane_id_val)}"

    match_text = "--"
    match_color = WHITE_COLOR
    if live_wp is not None:
        lane_yaw = float(live_wp.transform.rotation.yaw)
        direction_match = _angle_diff_deg(ego_yaw, lane_yaw) <= 90.0
        match_text = "MATCH" if direction_match else "MISMATCH"
        match_color = EGO_COLOR if direction_match else SAT_COLOR

    hud_lines: list[tuple[str, carla.Color]] = [
        (f"Junction ID  live={live_junction}  episode={ep_junction}", WHITE_COLOR),
        (f"Ego yaw={ego_yaw:.1f}°  heading={ego_heading}", WHITE_COLOR),
        (f"Road ID      live={live_road_id}  episode={ep_road_id}", WHITE_COLOR),
        (f"Lane ID      live={live_lane_id}  episode={ep_lane_id}", WHITE_COLOR),
        (f"Lane pos     live={live_lane_pos}  episode={ep_lane_pos}", WHITE_COLOR),
        (f"Lane dir     live={live_lane_dir}  episode={ep_lane_dir}", WHITE_COLOR),
        (f"Road dir     live={live_road_dir}  episode={ep_road_dir}", WHITE_COLOR),
        (
            f"Live vs Episode lane: {lane_match_text}  ({live_lane_pair} vs {ep_lane_pair})",
            lane_match_color,
        ),
        (f"Ego vs traffic direction: {match_text}", match_color),
    ]
    _draw_spectator_corner_hud(world, hud_lines, life_time)


def _draw_junction_boxes(
    world: carla.World,
    route: RouteData,
    ridx: int,
    ego_loc: Any,
    life_time: float,
) -> None:
    carla_map = world.get_map()
    selected_junction_boxes: list[Any] = []
    selected_junction_ids: set[int] = set()
    junction_radius = 85.0

    ego_wp = carla_map.get_waypoint(
        ego_loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if ego_wp is not None and ego_wp.is_junction:
        junction = ego_wp.get_junction()
        if junction is not None:
            jid = int(junction.id)
            selected_junction_ids.add(jid)
            selected_junction_boxes.append(junction.bounding_box)

    if ridx >= 0 and route.points_xyz.shape[0] > 0:
        end = min(route.points_xyz.shape[0], ridx + 180)
        for i in range(ridx, end):
            if i >= route.points_junction.shape[0] or not bool(route.points_junction[i]):
                continue
            rp = route.points_xyz[i]
            wp = carla_map.get_waypoint(
                carla.Location(x=float(rp[0]), y=float(rp[1]), z=float(rp[2])),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if wp is None or not wp.is_junction:
                continue
            junction = wp.get_junction()
            if junction is None:
                continue
            jid = int(junction.id)
            if jid in selected_junction_ids:
                continue
            selected_junction_ids.add(jid)
            selected_junction_boxes.append(junction.bounding_box)
            if len(selected_junction_boxes) >= 2:
                break

    for bb in selected_junction_boxes:
        if bb.location.distance(ego_loc) > junction_radius:
            continue
        world.debug.draw_box(
            bb,
            getattr(bb, "rotation", carla.Rotation()),
            thickness=0.08,
            color=NEXT_WP_COLOR,
            life_time=life_time,
        )


def _draw_next_waypoints_and_target(
    world: carla.World,
    route: RouteData,
    ridx: int,
    ego_loc: Any,
    life_time: float,
) -> None:
    if ridx < 0 or route.points_xyz.shape[0] <= 0:
        return

    debug = world.debug
    end = min(route.points_xyz.shape[0], ridx + 5)
    if end <= ridx:
        return

    for i in range(ridx, end):
        p = route.points_xyz[i]
        debug.draw_point(
            carla.Location(x=float(p[0]), y=float(p[1]), z=float(p[2]) + 0.55),
            size=0.10,
            color=NEXT_WP_COLOR,
            life_time=life_time,
            persistent_lines=False,
        )

    for i in range(ridx, max(ridx, end - 1)):
        a = route.points_xyz[i]
        b = route.points_xyz[i + 1]
        debug.draw_line(
            carla.Location(x=float(a[0]), y=float(a[1]), z=float(a[2]) + 0.55),
            carla.Location(x=float(b[0]), y=float(b[1]), z=float(b[2]) + 0.55),
            thickness=0.05,
            color=NEXT_WP_COLOR,
            life_time=life_time,
            persistent_lines=False,
        )

    target = route.points_xyz[ridx]
    target_loc = carla.Location(
        x=float(target[0]),
        y=float(target[1]),
        z=float(target[2]) + 1.0,
    )
    debug.draw_point(
        target_loc,
        size=0.15,
        color=TARGET_COLOR,
        life_time=life_time,
        persistent_lines=False,
    )
    debug.draw_line(
        carla.Location(ego_loc.x, ego_loc.y, ego_loc.z + 0.6),
        target_loc,
        thickness=0.05,
        color=TARGET_COLOR,
        life_time=life_time,
        persistent_lines=False,
    )


def _build_sat_corridor_segments(
    ep: EpisodeData,
    route: RouteData,
    frame_idx: int,
    ridx: int,
    corridor_length_m: float,
    half_width_m: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    pos = ep.position[frame_idx]
    ego_xy = np.array([float(pos[0]), float(pos[1])], dtype=np.float64)
    ego_z = float(pos[2])
    yaw_deg = float(ep.rotation[frame_idx, 1])
    corridor_length_m = max(1.0, float(corridor_length_m))
    half_width_m = max(0.5, float(half_width_m))

    samples: list[tuple[np.ndarray, float]] = []

    if route.points_xyz.shape[0] > 0 and ridx >= 0:
        prev_center = ego_xy
        prev_z = ego_z
        traveled = 0.0
        start = min(ridx, route.points_xyz.shape[0] - 1)
        for i in range(start, route.points_xyz.shape[0]):
            rp = route.points_xyz[i]
            center = np.array([float(rp[0]), float(rp[1])], dtype=np.float64)
            center_z = float(rp[2])
            seg_vec = center - prev_center
            seg_len = float(np.linalg.norm(seg_vec))
            if seg_len < 1e-6:
                continue

            if traveled + seg_len > corridor_length_m:
                remain = max(0.0, corridor_length_m - traveled)
                ratio = remain / seg_len if seg_len > 1e-6 else 0.0
                center = prev_center + seg_vec * ratio
                center_z = prev_z + (center_z - prev_z) * ratio
                samples.append((center, center_z))
                break

            samples.append((center, center_z))
            traveled += seg_len
            prev_center = center
            prev_z = center_z
            if traveled >= corridor_length_m:
                break

    if not samples:
        yaw = math.radians(yaw_deg)
        fwd = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
        samples = [(ego_xy + fwd * corridor_length_m, ego_z)]

    segments_2d: list[np.ndarray] = []
    segments_3d: list[np.ndarray] = []

    prev_center = ego_xy
    prev_z = ego_z
    for center, center_z in samples:
        seg_vec = center - prev_center
        seg_len = float(np.linalg.norm(seg_vec))
        if seg_len < 1e-6:
            continue

        seg_dir = seg_vec / seg_len
        seg_right = np.array([-seg_dir[1], seg_dir[0]], dtype=np.float64)

        prev_l = prev_center - seg_right * half_width_m
        prev_r = prev_center + seg_right * half_width_m
        cur_l = center - seg_right * half_width_m
        cur_r = center + seg_right * half_width_m

        quad_2d = np.array([prev_l, prev_r, cur_r, cur_l], dtype=np.float64)
        quad_3d = np.array(
            [
                [prev_l[0], prev_l[1], prev_z + 0.2],
                [prev_r[0], prev_r[1], prev_z + 0.2],
                [cur_r[0], cur_r[1], center_z + 0.2],
                [cur_l[0], cur_l[1], center_z + 0.2],
            ],
            dtype=np.float64,
        )

        segments_2d.append(quad_2d)
        segments_3d.append(quad_3d)
        prev_center = center
        prev_z = center_z

    return segments_2d, segments_3d


def _draw_quad_outline(
    debug: carla.DebugHelper,
    quad_3d: np.ndarray,
    color: carla.Color,
    life_time: float,
    thickness: float = 0.07,
) -> None:
    for i in range(4):
        p1 = quad_3d[i]
        p2 = quad_3d[(i + 1) % 4]
        debug.draw_line(
            carla.Location(x=float(p1[0]), y=float(p1[1]), z=float(p1[2])),
            carla.Location(x=float(p2[0]), y=float(p2[1]), z=float(p2[2])),
            thickness=thickness,
            color=color,
            life_time=life_time,
            persistent_lines=False,
        )


def _draw_bbox_outline(
    debug: carla.DebugHelper,
    corners_2d: np.ndarray,
    z: float,
    color: carla.Color,
    life_time: float,
    thickness: float = 0.09,
) -> None:
    for i in range(4):
        p1 = corners_2d[i]
        p2 = corners_2d[(i + 1) % 4]
        debug.draw_line(
            carla.Location(x=float(p1[0]), y=float(p1[1]), z=float(z)),
            carla.Location(x=float(p2[0]), y=float(p2[1]), z=float(z)),
            thickness=thickness,
            color=color,
            life_time=life_time,
            persistent_lines=False,
        )


def _draw_frame_sat(
    world: carla.World,
    frame_idx: int,
    ep: EpisodeData,
    route: RouteData,
    sat_ctx: SatRenderContext,
    road_cache: RoadLaneCache | None,
    draw_life_time: float,
    sat_radius_m: float,
    sat_corridor_length_m: float,
    sat_half_width_m: float,
    ego_length_m: float,
    ego_width_m: float,
    vehicle_length_m: float,
    vehicle_width_m: float,
    pedestrian_length_m: float,
    pedestrian_width_m: float,
) -> None:
    debug = world.debug

    pos = ep.position[frame_idx]
    ego_xy = np.array([float(pos[0]), float(pos[1])], dtype=np.float64)
    ego_z = float(pos[2])
    yaw_deg = float(ep.rotation[frame_idx, 1])
    ego_loc = carla.Location(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))

    ridx = _resolve_route_head_index(ep, route, frame_idx)

    corridor_segments_2d, corridor_segments_3d = _build_sat_corridor_segments(
        ep=ep,
        route=route,
        frame_idx=frame_idx,
        ridx=ridx,
        corridor_length_m=sat_corridor_length_m,
        half_width_m=sat_half_width_m,
    )
    for quad in corridor_segments_3d:
        _draw_quad_outline(debug, quad, CORRIDOR_COLOR, draw_life_time, thickness=0.06)

    _draw_next_waypoints_and_target(world, route, ridx, ego_loc, draw_life_time)
    _draw_junction_boxes(world, route, ridx, ego_loc, draw_life_time)

    yaw_rad = math.radians(yaw_deg)
    ego_fwd_2d = np.array([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=np.float64)
    _draw_active_control_zone(
        world=world,
        ego_loc=ego_loc,
        ego_fwd_2d=ego_fwd_2d,
        traffic_light_actor_id=int(ep.traffic_light_actor_id[frame_idx]),
        stop_sign_distance=float(ep.stop_sign_distance[frame_idx]),
        yield_sign_distance=float(ep.yield_sign_distance[frame_idx]),
        life_time=draw_life_time,
    )
    _draw_lane_direction_hud(
        world=world,
        ep=ep,
        route=route,
        frame_idx=frame_idx,
        ridx=ridx,
        ego_loc=ego_loc,
        road_cache=road_cache,
        life_time=draw_life_time,
    )

    # Ego bbox (green)
    ego_bbox = _bbox_corners_2d(
        center_xy=ego_xy,
        yaw_deg=yaw_deg,
        length_m=ego_length_m,
        width_m=ego_width_m,
    )
    _draw_bbox_outline(
        debug,
        corners_2d=ego_bbox,
        z=ego_z + 0.12,
        color=EGO_COLOR,
        life_time=draw_life_time,
        thickness=0.11,
    )

    nearby_vehicles = _decode_json_list(ep.nearby_vehicles_raw[frame_idx])
    nearby_pedestrians = _decode_json_list(ep.nearby_pedestrians_raw[frame_idx])
    radius_m = max(0.1, float(sat_radius_m))
    sequential_forward = frame_idx == sat_ctx.last_frame_idx + 1

    def _sat_hit(corners_2d: np.ndarray) -> bool:
        return any(_sat_overlap(seg, corners_2d) for seg in corridor_segments_2d)

    def _estimate_orientations(
        centers: list[np.ndarray],
        previous: list[tuple[np.ndarray, float]],
        max_match_dist: float,
        fallback_yaw: float,
    ) -> list[float]:
        if not centers:
            return []
        used_prev: set[int] = set()
        yaws: list[float] = []
        for center in centers:
            yaw = float(fallback_yaw)
            if sequential_forward and previous:
                best_idx = -1
                best_dist = float(max_match_dist)
                for idx, (prev_center, _prev_yaw) in enumerate(previous):
                    if idx in used_prev:
                        continue
                    dist = float(np.linalg.norm(center - prev_center))
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx
                if best_idx >= 0:
                    used_prev.add(best_idx)
                    prev_center, prev_yaw = previous[best_idx]
                    delta = center - prev_center
                    if float(np.linalg.norm(delta)) > 0.03:
                        yaw = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
                    else:
                        yaw = float(prev_yaw)
            yaws.append(yaw)
        return yaws

    vehicle_items: list[dict[str, Any]] = []
    for v in nearby_vehicles:
        dist = _safe_float(v.get("distance", v.get("center_distance", 1e9)), 1e9)
        if dist > radius_m:
            continue

        if "world_x" in v and "world_y" in v:
            center_xy = np.array(
                [_safe_float(v.get("world_x")), _safe_float(v.get("world_y"))], dtype=np.float64
            )
            center_z = _safe_float(v.get("world_z"), ego_z)
        else:
            center_xy, center_z = _local_to_world_center(
                ego_xy,
                yaw_deg,
                _safe_float(v.get("longitudinal", 0.0)),
                _safe_float(v.get("lateral", 0.0)),
                ego_z,
            )

        vehicle_items.append(
            {
                "center_xy": center_xy,
                "center_z": center_z,
                "obb_in_lane": bool(v.get("obb_in_lane", False)),
            }
        )

    vehicle_centers = [item["center_xy"] for item in vehicle_items]
    vehicle_yaws = _estimate_orientations(
        centers=vehicle_centers,
        previous=sat_ctx.vehicle_prev,
        max_match_dist=4.5,
        fallback_yaw=0.0,
    )
    sat_hits = 0

    for item, veh_yaw in zip(vehicle_items, vehicle_yaws):
        center_xy = item["center_xy"]
        center_z = float(item["center_z"])

        veh_bbox = _bbox_corners_2d(
            center_xy=center_xy,
            yaw_deg=veh_yaw,
            length_m=vehicle_length_m,
            width_m=vehicle_width_m,
        )
        sat = bool(item["obb_in_lane"]) or _sat_hit(veh_bbox)
        if sat:
            sat_hits += 1
        color = SAT_COLOR if sat else ACTOR_COLOR
        _draw_bbox_outline(
            debug,
            corners_2d=veh_bbox,
            z=center_z + 0.15,
            color=color,
            life_time=draw_life_time,
            thickness=0.09,
        )

    sat_ctx.vehicle_prev = [
        (item["center_xy"].copy(), yaw) for item, yaw in zip(vehicle_items, vehicle_yaws)
    ]

    pedestrian_items: list[dict[str, Any]] = []
    for p in nearby_pedestrians:
        dist = _safe_float(p.get("distance", 1e9), 1e9)
        if dist > radius_m:
            continue

        center_xy, center_z = _local_to_world_center(
            ego_xy,
            yaw_deg,
            _safe_float(p.get("longitudinal", 0.0)),
            _safe_float(p.get("lateral", 0.0)),
            ego_z,
        )
        pedestrian_items.append(
            {
                "center_xy": center_xy,
                "center_z": center_z,
            }
        )

    ped_centers = [item["center_xy"] for item in pedestrian_items]
    ped_yaws = _estimate_orientations(
        centers=ped_centers,
        previous=sat_ctx.pedestrian_prev,
        max_match_dist=2.0,
        fallback_yaw=0.0,
    )

    for item, ped_yaw in zip(pedestrian_items, ped_yaws):
        center_xy = item["center_xy"]
        center_z = float(item["center_z"])
        ped_bbox = _bbox_corners_2d(
            center_xy=center_xy,
            yaw_deg=ped_yaw,
            length_m=pedestrian_length_m,
            width_m=pedestrian_width_m,
        )
        sat = _sat_hit(ped_bbox)
        color = SAT_COLOR if sat else ACTOR_COLOR
        _draw_bbox_outline(
            debug,
            corners_2d=ped_bbox,
            z=center_z + 0.08,
            color=color,
            life_time=draw_life_time,
            thickness=0.08,
        )

    sat_ctx.pedestrian_prev = [
        (item["center_xy"].copy(), yaw) for item, yaw in zip(pedestrian_items, ped_yaws)
    ]

    speed_kmh = float(np.linalg.norm(ep.velocity[frame_idx]) * 3.6)
    debug.draw_string(
        carla.Location(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]) + 2.5),
        (
            f"F{frame_idx}/{ep.n_frames - 1} "
            f"speed={speed_kmh:.1f}kmh "
            f"near<= {radius_m:.0f}m "
            f"sat_hits={sat_hits} "
            f"wp={ridx}"
        ),
        draw_shadow=True,
        color=WHITE_COLOR,
        life_time=draw_life_time,
        persistent_lines=False,
    )

    sat_ctx.last_frame_idx = frame_idx


def _draw_polyline(
    debug: carla.DebugHelper,
    pts: np.ndarray,
    color: carla.Color,
    life_time: float,
    z_offset: float = 0.2,
) -> None:
    if pts.shape[0] < 2:
        return
    for i in range(pts.shape[0] - 1):
        a = pts[i]
        b = pts[i + 1]
        debug.draw_line(
            carla.Location(x=float(a[0]), y=float(a[1]), z=float(a[2]) + z_offset),
            carla.Location(x=float(b[0]), y=float(b[1]), z=float(b[2]) + z_offset),
            thickness=0.08,
            color=color,
            life_time=life_time,
            persistent_lines=False,
        )


def _draw_frame_simple(
    world: carla.World,
    frame_idx: int,
    ep: EpisodeData,
    route: RouteData,
    draw_life_time: float,
    history_frames: int,
    future_frames: int,
    route_window: int,
) -> None:
    debug = world.debug

    pos = ep.position[frame_idx]
    yaw_deg = float(ep.rotation[frame_idx, 1])
    yaw_rad = math.radians(yaw_deg)
    forward = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0], dtype=np.float32)
    speed_kmh = float(np.linalg.norm(ep.velocity[frame_idx]) * 3.6)

    # Past and future ego path context
    hs = max(0, frame_idx - history_frames)
    fs = min(ep.n_frames - 1, frame_idx + future_frames)
    _draw_polyline(
        debug,
        ep.position[hs : frame_idx + 1],
        color=carla.Color(80, 220, 80),
        life_time=draw_life_time,
        z_offset=0.20,
    )
    _draw_polyline(
        debug,
        ep.position[frame_idx : fs + 1],
        color=carla.Color(80, 180, 255),
        life_time=draw_life_time,
        z_offset=0.20,
    )

    # Local route context around current wp head index
    ridx = int(ep.wp_buffer_head_index[frame_idx])
    if route.points_xyz.shape[0] > 0 and ridx >= 0:
        lo = max(0, ridx - route_window)
        hi = min(route.points_xyz.shape[0], ridx + route_window + 1)
        seg = route.points_xyz[lo:hi]
        _draw_polyline(
            debug,
            seg,
            color=carla.Color(255, 200, 80),
            life_time=draw_life_time,
            z_offset=0.35,
        )
        if 0 <= ridx < route.points_xyz.shape[0]:
            p = route.points_xyz[ridx]
            debug.draw_point(
                carla.Location(x=float(p[0]), y=float(p[1]), z=float(p[2]) + 0.50),
                size=0.14,
                color=carla.Color(255, 180, 30),
                life_time=draw_life_time,
                persistent_lines=False,
            )
            if 0 <= ridx < route.points_junction.shape[0] and route.points_junction[ridx]:
                debug.draw_string(
                    carla.Location(x=float(p[0]), y=float(p[1]), z=float(p[2]) + 1.2),
                    "JUNCTION",
                    draw_shadow=False,
                    color=carla.Color(255, 120, 255),
                    life_time=draw_life_time,
                    persistent_lines=False,
                )

    # Ego marker + heading arrow
    ego_loc = carla.Location(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]) + 0.55)
    arrow_tip = carla.Location(
        x=float(pos[0] + forward[0] * 2.2),
        y=float(pos[1] + forward[1] * 2.2),
        z=float(pos[2]) + 0.65,
    )
    debug.draw_point(
        ego_loc,
        size=0.18,
        color=carla.Color(255, 40, 40),
        life_time=draw_life_time,
        persistent_lines=False,
    )
    debug.draw_arrow(
        ego_loc,
        arrow_tip,
        thickness=0.08,
        arrow_size=0.16,
        color=carla.Color(255, 40, 40),
        life_time=draw_life_time,
        persistent_lines=False,
    )

    nav = NAV_NAMES.get(int(ep.nav_command[frame_idx]), str(int(ep.nav_command[frame_idx])))
    mode = MODE_NAMES.get(int(ep.driving_mode[frame_idx]), str(int(ep.driving_mode[frame_idx])))
    tl_name = TL_NAMES.get(
        int(ep.traffic_light_state[frame_idx]), str(int(ep.traffic_light_state[frame_idx]))
    )

    tl_d = float(ep.traffic_light_distance[frame_idx])
    ss_d = float(ep.stop_sign_distance[frame_idx])
    ys_d = float(ep.yield_sign_distance[frame_idx])

    text_lines = [
        f"frame {frame_idx}/{ep.n_frames - 1}",
        f"speed={speed_kmh:.1f} km/h steer={float(ep.steer[frame_idx]):+.3f}",
        f"nav={nav} mode={mode}",
        f"route={float(ep.route_completion[frame_idx]) * 100.0:.1f}% wp_idx={ridx}",
        f"TL={tl_name} ({tl_d:.1f}m)  SS={ss_d:.1f}m  YS={ys_d:.1f}m",
    ]

    for i, line in enumerate(text_lines):
        debug.draw_string(
            carla.Location(
                x=float(pos[0]),
                y=float(pos[1]),
                z=float(pos[2]) + 1.6 + i * 0.25,
            ),
            line,
            draw_shadow=False,
            color=carla.Color(255, 255, 255),
            life_time=draw_life_time,
            persistent_lines=False,
        )


def _draw_frame_collector(
    world: carla.World,
    frame_idx: int,
    ep: EpisodeData,
    collector_ctx: CollectorDebugContext,
) -> None:
    collector_ctx.vehicle.set_pose(ep.position[frame_idx], ep.rotation[frame_idx])
    collector_ctx.expert.sync_route_state(int(ep.wp_buffer_head_index[frame_idx]))
    info = _build_debug_info(ep, frame_idx)
    collector_ctx.draw_fn(
        world,
        collector_ctx.expert,
        collector_ctx.vehicle,
        frame_idx,
        info,
    )


def _set_initial_view(world: carla.World, ep: EpisodeData, frame_idx: int) -> None:
    pos = ep.position[frame_idx]
    spect = world.get_spectator()
    spect.set_transform(
        carla.Transform(
            carla.Location(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]) + 45.0),
            carla.Rotation(pitch=-70.0, yaw=float(ep.rotation[frame_idx, 1]), roll=0.0),
        )
    )


def _setup_global_keyboard(state: InputState) -> tuple[Any | None, bool]:
    """Return (listener, enabled_bool)."""
    try:
        from pynput import keyboard as pynput_keyboard  # type: ignore
    except Exception as exc:
        logger.warning(
            "Global keyboard mode unavailable (%s). Falling back to terminal controls.",
            exc,
        )
        return None, False

    def _is_shift(k: Any) -> bool:
        return k in (
            pynput_keyboard.Key.shift,
            pynput_keyboard.Key.shift_l,
            pynput_keyboard.Key.shift_r,
        )

    def on_press(key: Any) -> None:
        if _is_shift(key):
            state.shift_down = True
            return
        if key == pynput_keyboard.Key.esc:
            state.quit_requested = True
            return
        if key == pynput_keyboard.Key.space:
            now = time.time()
            if not state.space_down:
                state.hold_started_at = now
            state.space_down = True
            state.step_request += -1 if state.shift_down else 1

    def on_release(key: Any) -> None:
        if _is_shift(key):
            state.shift_down = False
            return
        if key == pynput_keyboard.Key.space:
            state.space_down = False

    listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener, True


def _read_terminal_keys(state: InputState) -> None:
    """Non-blocking terminal control fallback."""
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not ready:
        return

    data = sys.stdin.read(1)
    if not data:
        return

    if data == "q" or data == "\x1b":
        state.quit_requested = True
    elif data == " ":
        state.step_request += 1
    elif data == "b":
        state.step_request -= 1
    elif data == "p":
        state.play_direction = 0 if state.play_direction == 1 else 1
    elif data == "o":
        state.play_direction = 0 if state.play_direction == -1 else -1


def _compute_frame_delta(
    state: InputState,
    now: float,
    hold_delay_sec: float,
    repeat_rate_hz: float,
    last_repeat_time: float,
) -> tuple[int, float]:
    """Compute frame delta for this loop and updated repeat timestamp."""
    if state.step_request != 0:
        delta = 1 if state.step_request > 0 else -1
        state.step_request -= delta
        return delta, last_repeat_time

    if state.play_direction != 0:
        return state.play_direction, last_repeat_time

    if state.space_down:
        if now - state.hold_started_at >= hold_delay_sec:
            interval = 1.0 / max(repeat_rate_hz, 1.0)
            if now - last_repeat_time >= interval:
                delta = -1 if state.shift_down else 1
                return delta, now

    return 0, last_repeat_time


def main() -> None:
    args = _parse_args()

    if args.log_file:
        _configure_file_logging(Path(str(args.log_file)))

    global HEADING_LABEL_STYLE
    HEADING_LABEL_STYLE = str(args.direction_label_style)
    logger.info("Direction label style: %s", HEADING_LABEL_STYLE)

    episode_path = Path(args.episode)
    if not episode_path.exists():
        raise SystemExit(f"Episode file not found: {episode_path}")

    route_path: Path | None
    if args.route:
        route_path = Path(args.route)
    else:
        route_path = episode_path.with_suffix(".route.json")

    ep = _load_episode(episode_path)
    route = _load_route(route_path)

    host = _detect_wsl2_host() if args.host == "auto" else args.host
    logger.info("Connecting to CARLA at %s:%d", host, args.port)
    client = carla.Client(host, args.port)
    rpc_timeout_sec = 60.0
    map_load_timeout_sec = 240.0
    client.set_timeout(rpc_timeout_sec)
    world = client.get_world()

    if args.auto_load_town and ep.town:
        cur = world.get_map().name.split("/")[-1]
        if cur != ep.town:
            logger.info("Loading map %s (current=%s)", ep.town, cur)
            client.set_timeout(map_load_timeout_sec)
            try:
                client.load_world(ep.town)
                world = _wait_for_world_ready(
                    client,
                    expected_town=ep.town,
                    timeout_sec=map_load_timeout_sec,
                )
            finally:
                client.set_timeout(rpc_timeout_sec)

    original_settings = None
    sync_mode_enabled = False

    try:
        world, original_settings = _enable_synchronous_mode(
            client,
            fps=int(ep.fps),
            attempts=3,
            retry_delay_sec=2.0,
        )
        sync_mode_enabled = True

        start_frame = max(0, min(int(args.start_frame), ep.n_frames - 1))
        if int(args.end_frame) < 0:
            end_frame = ep.n_frames - 1
        else:
            end_frame = max(start_frame, min(int(args.end_frame), ep.n_frames - 1))

        frame_idx = start_frame
        logger.info(
            "Episode loaded: %s | town=%s | status=%s | frames=%d | window=%d..%d",
            ep.path.name,
            ep.town,
            ep.status,
            ep.n_frames,
            start_frame,
            end_frame,
        )
        if ep.collision_frame is not None:
            logger.info("Collision frame in episode metadata: %d", ep.collision_frame)
        if route.path is not None:
            logger.info(
                "Route sidecar loaded: %s (%d points)",
                route.path.name,
                route.points_xyz.shape[0],
            )
        else:
            logger.info("No route sidecar loaded")

        collector_ctx: CollectorDebugContext | None = None
        sat_ctx: SatRenderContext | None = None
        road_cache: RoadLaneCache | None = None
        draw_mode = str(args.draw_mode)
        if draw_mode == "collector":
            try:
                collector_ctx = _build_collector_debug_context(world=world, route=route)
                logger.info("Draw mode: collector parity (ExpertDataCollector._draw_route_debug)")
                if float(args.draw_life_time) != 0.3:
                    logger.info(
                        "--draw-life-time is ignored in collector mode "
                        "(collector uses life_time=0.3)"
                    )
            except Exception as exc:
                logger.warning(
                    "Collector draw mode unavailable (%s). Falling back to simple overlays.",
                    exc,
                )
                draw_mode = "simple"
        if draw_mode == "sat":
            sat_ctx = SatRenderContext()
            try:
                road_cache = _build_road_lane_cache(world.get_map(), spacing_m=2.0)
                logger.info(
                    "Road lane cache built for SAT HUD (%d roads)", len(road_cache.by_road_lane)
                )
            except Exception as exc:
                road_cache = None
                logger.warning(
                    "Road lane cache unavailable (%s); SAT HUD may be less accurate", exc
                )
            logger.info(
                "Draw mode: SAT overlays (no route trails), radius=%.1fm "
                "corridor=%.1fm half-width=%.2fm",
                float(args.sat_radius_m),
                float(args.sat_corridor_length_m),
                float(args.sat_half_width_m),
            )
        elif draw_mode == "simple":
            logger.info("Draw mode: simple overlays")

        if args.set_initial_view:
            _set_initial_view(world, ep, frame_idx)

        state = InputState()
        listener = None
        global_enabled = False
        if args.control_mode == "global":
            listener, global_enabled = _setup_global_keyboard(state)

        use_terminal = (args.control_mode == "terminal") or (not global_enabled)
        terminal_backup = False
        old_tty = None
        if use_terminal:
            logger.info(
                "Terminal controls: [Space]=next, [b]=prev, [p]=toggle forward, "
                "[o]=toggle reverse, [q]=quit"
            )
            old_tty = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
            attrs = termios.tcgetattr(sys.stdin.fileno())
            attrs[3] = attrs[3] & ~termios.ECHO
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, attrs)
        else:
            logger.info(
                "Global controls: [Space]=next/hold forward, "
                "[Shift+Space]=prev/hold reverse, [Esc]=quit"
            )
            if sys.stdin.isatty():
                old_tty = termios.tcgetattr(sys.stdin.fileno())
                tty.setcbreak(sys.stdin.fileno())
                attrs = termios.tcgetattr(sys.stdin.fileno())
                attrs[3] = attrs[3] & ~termios.ECHO
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, attrs)
                terminal_backup = True
                logger.info(
                    "Global mode terminal backup enabled: "
                    "[Space]/[b]/[p]/[o]/[q]/[Esc] also work from this terminal."
                )

        last_repeat_time = 0.0
        frame_period_sec = 1.0 / max(float(ep.fps), 1.0)
        while not state.quit_requested:
            loop_start = time.time()

            if use_terminal or terminal_backup:
                _read_terminal_keys(state)

            now = loop_start
            delta, last_repeat_time = _compute_frame_delta(
                state,
                now,
                hold_delay_sec=float(args.hold_delay_sec),
                repeat_rate_hz=float(args.repeat_rate_hz),
                last_repeat_time=last_repeat_time,
            )

            if delta != 0:
                frame_idx = max(start_frame, min(end_frame, frame_idx + delta))

            if draw_mode == "collector" and collector_ctx is not None:
                _draw_frame_collector(world, frame_idx, ep, collector_ctx)
            elif draw_mode == "sat":
                _draw_frame_sat(
                    world,
                    frame_idx,
                    ep,
                    route,
                    sat_ctx=sat_ctx if sat_ctx is not None else SatRenderContext(),
                    road_cache=road_cache,
                    draw_life_time=float(args.draw_life_time),
                    sat_radius_m=float(args.sat_radius_m),
                    sat_corridor_length_m=float(args.sat_corridor_length_m),
                    sat_half_width_m=float(args.sat_half_width_m),
                    ego_length_m=float(args.ego_length_m),
                    ego_width_m=float(args.ego_width_m),
                    vehicle_length_m=float(args.vehicle_length_m),
                    vehicle_width_m=float(args.vehicle_width_m),
                    pedestrian_length_m=float(args.pedestrian_length_m),
                    pedestrian_width_m=float(args.pedestrian_width_m),
                )
            else:
                _draw_frame_simple(
                    world,
                    frame_idx,
                    ep,
                    route,
                    draw_life_time=float(args.draw_life_time),
                    history_frames=int(args.history_frames),
                    future_frames=int(args.future_frames),
                    route_window=int(args.route_window),
                )

            world.tick()

            elapsed = time.time() - loop_start
            if elapsed < frame_period_sec:
                time.sleep(frame_period_sec - elapsed)

            if frame_idx in (start_frame, end_frame) and delta != 0 and state.play_direction != 0:
                state.play_direction = 0

            if ep.collision_frame is not None and frame_idx == ep.collision_frame:
                # Keep visualizing at collision frame unless user continues.
                pass

        if listener is not None:
            listener.stop()

        if old_tty is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_tty)

    finally:
        if sync_mode_enabled and original_settings is not None:
            try:
                client.get_world().apply_settings(original_settings)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to restore original CARLA world settings: %s", exc)


if __name__ == "__main__":
    main()
