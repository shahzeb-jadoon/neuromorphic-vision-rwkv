#!/usr/bin/env python3
"""Preview proposed debug color semantics without changing production debug_draw.

This test-only script runs short live previews across one or more CARLA towns,
spawns traffic around an ego vehicle, and renders policy overlays using the
approved 7-color palette:
  GREEN   #00FF00  junctions
  SKY     #00AAFF  steering target marker
  MAGENTA #FF00FF  SAT corridor outline
  RED     #FF0000  emergency actors
  YELLOW  #FFFF00  SAT/lateral predicted-crossing actors
  WHITE   #FFFFFF  nearby non-critical actors + HUD
  INDIGO  #6600FF  stop targets (TL/STOP/YIELD)

Default run validates Town03, Town07, Town10HD in sequence.
"""

from __future__ import annotations

import argparse
import math
import random
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import carla
except ImportError as exc:  # pragma: no cover
    raise SystemExit("CARLA Python API is required. Activate your CARLA env first.") from exc


DEFAULT_TICK_SECONDS = 0.05

GREEN = carla.Color(0, 255, 0)
SKY = carla.Color(0, 170, 255)
MAGENTA = carla.Color(255, 0, 255)
RED = carla.Color(255, 0, 0)
YELLOW = carla.Color(255, 255, 0)
WHITE = carla.Color(255, 255, 255)
INDIGO = carla.Color(102, 0, 255)


@dataclass
class ScenarioActors:
    """Actors spawned by this script for one town run."""

    ego: Any
    vehicles: list[Any] = field(default_factory=list)
    walkers: list[Any] = field(default_factory=list)
    walker_controllers: list[Any] = field(default_factory=list)

    def all_spawned(self) -> list[Any]:
        return [*self.walker_controllers, *self.walkers, *self.vehicles, self.ego]


@dataclass
class OverlayCounts:
    """Per-tick overlay diagnostics."""

    vehicles_white: int = 0
    vehicles_yellow: int = 0
    vehicles_red: int = 0
    peds_white: int = 0
    peds_yellow: int = 0
    peds_red: int = 0
    stop_targets: int = 0
    junction_boxes: int = 0


@dataclass
class ControlCandidate:
    """A traffic control candidate currently ahead of ego."""

    kind: str
    actor: Any
    anchor_loc: Any
    label_loc: Any
    distance: float
    longitudinal: float
    road_id: int | None = None


def get_windows_host_ip() -> str | None:
    """Best-effort host auto-discovery for WSL2 setups."""
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


def resolve_host(host: str) -> str:
    if host != "auto":
        return host
    auto_host = get_windows_host_ip()
    return auto_host if auto_host else "localhost"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview proposed debug color policy in CARLA towns.",
    )
    parser.add_argument("--host", default="auto", help="CARLA host (auto|ip|hostname)")
    parser.add_argument("--port", type=int, default=2000, help="CARLA RPC port")
    parser.add_argument(
        "--towns",
        nargs="+",
        default=["Town03", "Town07", "Town10HD"],
        help="Town names to preview in sequence",
    )
    parser.add_argument(
        "--seconds-per-town",
        type=float,
        default=45.0,
        help="Preview duration per town",
    )
    parser.add_argument("--tm-port", type=int, default=8000, help="TrafficManager port")
    parser.add_argument(
        "--traffic-vehicles",
        type=int,
        default=32,
        help="Number of NPC vehicles to spawn",
    )
    parser.add_argument(
        "--walkers",
        type=int,
        default=20,
        help="Number of pedestrians to spawn near ego",
    )
    parser.add_argument(
        "--walker-radius",
        type=float,
        default=45.0,
        help="Spawn radius for pedestrians around ego spawn",
    )
    parser.add_argument(
        "--vicinity-radius",
        type=float,
        default=40.0,
        help="Nearby actor radius for white/yellow/red classification",
    )
    parser.add_argument(
        "--vehicle-emergency-distance",
        type=float,
        default=6.0,
        help="Vehicle emergency threshold (red)",
    )
    parser.add_argument(
        "--ped-emergency-distance",
        type=float,
        default=4.0,
        help="Pedestrian emergency threshold (red)",
    )
    parser.add_argument(
        "--ped-lateral-threshold",
        type=float,
        default=3.0,
        help="Lateral threshold for pedestrian yellow classification",
    )
    parser.add_argument(
        "--ped-prediction-sec",
        type=float,
        default=1.0,
        help="Prediction horizon for pedestrian lateral check",
    )
    parser.add_argument(
        "--corridor-length",
        type=float,
        default=28.0,
        help="Forward corridor length (meters)",
    )
    parser.add_argument(
        "--corridor-half-width",
        type=float,
        default=2.6,
        help="Minimum half-width for SAT corridor (meters)",
    )
    parser.add_argument(
        "--stop-search-radius",
        type=float,
        default=70.0,
        help="Search radius for TL/STOP/YIELD target markers",
    )
    parser.add_argument(
        "--junction-radius",
        type=float,
        default=85.0,
        help="Draw nearby junction boxes inside this radius",
    )
    parser.add_argument(
        "--junction-sample-spacing",
        type=float,
        default=6.0,
        help="Waypoint spacing used when sampling unique junction bounds",
    )
    parser.add_argument(
        "--fixed-delta",
        type=float,
        default=DEFAULT_TICK_SECONDS,
        help="Synchronous simulation step",
    )
    parser.add_argument(
        "--life-time",
        type=float,
        default=0.20,
        help="Debug primitive life_time (seconds)",
    )
    parser.add_argument(
        "--draw-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw compact text labels for actors",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    return parser.parse_args()


def _dot2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[0] + a[1] * b[1])


def sat_overlap(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
    """Separating Axis Theorem overlap test for convex polygons."""

    def _project(vertices: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
        dots = vertices @ axis
        return float(dots.min()), float(dots.max())

    for poly in (poly_a, poly_b):
        n = len(poly)
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            axis = np.array([-edge[1], edge[0]], dtype=np.float64)
            norm = np.linalg.norm(axis)
            if norm < 1e-8:
                continue
            axis /= norm
            min_a, max_a = _project(poly_a, axis)
            min_b, max_b = _project(poly_b, axis)
            if max_a < min_b or max_b < min_a:
                return False
    return True


def get_actor_obb_2d(actor: Any) -> np.ndarray:
    """Return actor OBB corners in world XY without in-place CARLA transforms."""
    bb = actor.bounding_box
    cx = float(bb.location.x)
    cy = float(bb.location.y)
    ex = float(bb.extent.x)
    ey = float(bb.extent.y)

    tf = actor.get_transform()
    yaw = math.radians(tf.rotation.yaw)
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)

    local_offsets = [
        (cx + ex, cy + ey),
        (cx + ex, cy - ey),
        (cx - ex, cy - ey),
        (cx - ex, cy + ey),
    ]

    out = np.empty((4, 2), dtype=np.float64)
    for i, (lx, ly) in enumerate(local_offsets):
        out[i, 0] = cos_y * lx - sin_y * ly + tf.location.x
        out[i, 1] = sin_y * lx + cos_y * ly + tf.location.y
    return out


def local_to_world_no_mutate(tf: Any, loc: Any) -> Any:
    """Transform local point by yaw translation only (sufficient for sign triggers)."""
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
    """Collect contiguous driving lanes on the same road as the base waypoint."""
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


def _ego_road_width_cap(carla_map: Any, ego_loc: Any) -> tuple[float, int | None]:
    """Return (max road width cap, ego road_id) for current ego location."""
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


def _road_id_at(carla_map: Any, loc: Any) -> int | None:
    wp = carla_map.get_waypoint(
        loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if wp is None:
        return None
    return int(wp.road_id)


def _trigger_edge_toward_junction(
    carla_map: Any,
    sign_tf: Any,
    trigger_center: Any,
    trigger_extent_x: float,
) -> Any:
    """Pick the trigger-volume edge that is oriented toward the junction."""
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


def _compute_lane_zone(
    waypoints: list[Any], z_offset: float = 0.06
) -> tuple[Any, Any, float, float] | None:
    """Return (center, rotation, total_width, depth) for lane-aligned road markings."""
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


def draw_polygon(
    debug: Any,
    poly: np.ndarray,
    z: float,
    color: Any,
    life_time: float,
    thickness: float = 0.06,
) -> None:
    n = len(poly)
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        debug.draw_line(
            carla.Location(float(p1[0]), float(p1[1]), z),
            carla.Location(float(p2[0]), float(p2[1]), z),
            thickness=thickness,
            color=color,
            life_time=life_time,
        )


def build_forward_corridor(ego_tf: Any, length: float, half_width: float) -> np.ndarray:
    loc = ego_tf.location
    fwd = ego_tf.get_forward_vector()
    fwd_2d = np.array([float(fwd.x), float(fwd.y)], dtype=np.float64)
    fwd_norm = np.linalg.norm(fwd_2d)
    if fwd_norm < 1e-6:
        fwd_2d = np.array([1.0, 0.0], dtype=np.float64)
    else:
        fwd_2d /= fwd_norm

    right_2d = np.array([-fwd_2d[1], fwd_2d[0]], dtype=np.float64)
    ego_xy = np.array([float(loc.x), float(loc.y)], dtype=np.float64)
    ahead_xy = ego_xy + fwd_2d * float(length)

    return np.array(
        [
            ego_xy - right_2d * half_width,
            ego_xy + right_2d * half_width,
            ahead_xy + right_2d * half_width,
            ahead_xy - right_2d * half_width,
        ],
        dtype=np.float64,
    )


def is_ahead(ego_loc: Any, ego_fwd_2d: np.ndarray, target_loc: Any) -> bool:
    delta = np.array(
        [float(target_loc.x) - float(ego_loc.x), float(target_loc.y) - float(ego_loc.y)],
        dtype=np.float64,
    )
    return _dot2(delta, ego_fwd_2d) > 0.0


def collect_junction_boxes(carla_map: Any, sample_spacing: float) -> list[Any]:
    """Collect unique junction bounding boxes from map waypoints."""
    seen_ids: set[int] = set()
    boxes: list[Any] = []

    for wp in carla_map.generate_waypoints(sample_spacing):
        if not wp.is_junction:
            continue
        junction = wp.get_junction()
        if junction is None:
            continue
        j_id = int(junction.id)
        if j_id in seen_ids:
            continue
        seen_ids.add(j_id)
        boxes.append(junction.bounding_box)
    return boxes


def pick_vehicle_blueprint(bp_lib: Any, rng: random.Random) -> Any:
    vehicles = bp_lib.filter("vehicle.*")
    candidates = []
    for bp in vehicles:
        if bp.has_attribute("number_of_wheels"):
            try:
                if int(bp.get_attribute("number_of_wheels").as_int()) != 4:
                    continue
            except ValueError:
                continue
        candidates.append(bp)
    if not candidates:
        try:
            return bp_lib.find("vehicle.tesla.model3")
        except (RuntimeError, IndexError, ValueError) as exc:
            raise RuntimeError("No suitable vehicle blueprint found") from exc
    return rng.choice(candidates)


def spawn_ego(world: Any, rng: random.Random) -> tuple[Any, Any]:
    bp_lib = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    if not spawns:
        raise RuntimeError("No spawn points available in current map")

    rng.shuffle(spawns)
    bp = None
    try:
        bp = bp_lib.find("vehicle.tesla.model3")
    except (RuntimeError, IndexError, ValueError):
        bp = None
    if bp is None:
        bp = pick_vehicle_blueprint(bp_lib, rng)
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "hero")

    for sp in spawns:
        ego = world.try_spawn_actor(bp, sp)
        if ego is not None:
            return ego, sp
    raise RuntimeError("Failed to spawn ego vehicle")


def spawn_npc_vehicles(
    world: Any,
    tm_port: int,
    ego_spawn: Any,
    count: int,
    rng: random.Random,
) -> list[Any]:
    bp_lib = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    rng.shuffle(spawns)

    out: list[Any] = []
    for sp in spawns:
        if len(out) >= count:
            break
        if sp.location.distance(ego_spawn.location) < 8.0:
            continue

        bp = pick_vehicle_blueprint(bp_lib, rng)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "autopilot")
        if bp.has_attribute("color"):
            colors = bp.get_attribute("color").recommended_values
            if colors:
                bp.set_attribute("color", rng.choice(colors))

        actor = world.try_spawn_actor(bp, sp)
        if actor is None:
            continue
        actor.set_autopilot(True, tm_port)
        out.append(actor)

    return out


def spawn_walkers_near_ego(
    world: Any,
    ego_spawn: Any,
    walker_count: int,
    radius: float,
    rng: random.Random,
) -> tuple[list[Any], list[Any]]:
    bp_lib = world.get_blueprint_library()
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    if not walker_bps:
        return [], []

    try:
        controller_bp = bp_lib.find("controller.ai.walker")
    except (RuntimeError, IndexError, ValueError):
        return [], []

    walkers: list[Any] = []
    controllers: list[Any] = []
    attempts = 0
    max_attempts = max(100, walker_count * 20)

    while len(walkers) < walker_count and attempts < max_attempts:
        attempts += 1
        nav_loc = world.get_random_location_from_navigation()
        if nav_loc is None:
            continue
        if nav_loc.distance(ego_spawn.location) > radius:
            continue

        walker_bp = rng.choice(walker_bps)
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")

        walker = world.try_spawn_actor(walker_bp, carla.Transform(nav_loc))
        if walker is None:
            continue

        controller = world.try_spawn_actor(controller_bp, carla.Transform(), walker)
        if controller is None:
            walker.destroy()
            continue

        walkers.append(walker)
        controllers.append(controller)

    world.tick()
    for controller in controllers:
        try:
            controller.start()
            dst = world.get_random_location_from_navigation()
            if dst is not None:
                controller.go_to_location(dst)
            controller.set_max_speed(rng.uniform(0.9, 1.5))
        except RuntimeError:
            continue

    return walkers, controllers


def destroy_actors(actors: list[Any]) -> None:
    for actor in actors:
        try:
            if actor is not None and actor.is_alive:
                actor.destroy()
        except RuntimeError:
            continue


def follow_ego_with_spectator(world: Any, ego: Any) -> None:
    tf = ego.get_transform()
    fwd = tf.get_forward_vector()

    cam_loc = carla.Location(
        x=tf.location.x - fwd.x * 10.0,
        y=tf.location.y - fwd.y * 10.0,
        z=tf.location.z + 5.0,
    )
    cam_rot = carla.Rotation(pitch=-18.0, yaw=tf.rotation.yaw, roll=0.0)
    world.get_spectator().set_transform(carla.Transform(cam_loc, cam_rot))


def draw_stop_targets(
    world: Any,
    carla_map: Any,
    ego_loc: Any,
    ego_fwd_2d: np.ndarray,
    args: argparse.Namespace,
) -> tuple[int, ControlCandidate | None]:
    debug = world.debug
    actors = world.get_actors()
    life_time = args.life_time
    count = 0
    candidates: list[ControlCandidate] = []
    road_width_cap, ego_road_id = _ego_road_width_cap(carla_map, ego_loc)

    def _ahead_longitudinal(target_loc: Any) -> float:
        rel = np.array(
            [float(target_loc.x - ego_loc.x), float(target_loc.y - ego_loc.y)],
            dtype=np.float64,
        )
        return _dot2(rel, ego_fwd_2d)

    for tl in actors.filter("traffic.traffic_light*"):
        tl_loc = tl.get_transform().location
        if tl_loc.distance(ego_loc) > args.stop_search_radius:
            continue

        stop_wps = _collect_tl_waypoints(tl)

        ahead_wps: list[tuple[Any, Any, float]] = []
        for wp in stop_wps:
            loc = wp.transform.location
            if loc.distance(ego_loc) > args.stop_search_radius:
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
        debug.draw_box(
            carla.BoundingBox(
                zone_center,
                carla.Vector3D(
                    x=float(zone_depth * 0.5),
                    y=float(zone_width * 0.5),
                    z=1.0,
                ),
            ),
            zone_rot,
            thickness=0.13,
            color=INDIGO,
            life_time=life_time,
        )

        _best_wp, best_loc, best_long = min(
            ahead_wps,
            key=lambda item: float(item[1].distance(ego_loc)),
        )
        candidates.append(
            ControlCandidate(
                kind="TL",
                actor=tl,
                anchor_loc=best_loc,
                label_loc=carla.Location(
                    x=float(tl_loc.x),
                    y=float(tl_loc.y),
                    z=float(tl_loc.z) + 2.8,
                ),
                distance=float(best_loc.distance(ego_loc)),
                longitudinal=float(best_long),
                road_id=_road_id_at(carla_map, best_loc),
            )
        )
        count += 1

    for sign in actors.filter("traffic.stop*"):
        trigger = getattr(sign, "trigger_volume", None)
        if trigger is None:
            continue

        sign_tf = sign.get_transform()
        trigger_loc = local_to_world_no_mutate(sign_tf, trigger.location)
        if trigger_loc.distance(ego_loc) > args.stop_search_radius:
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

        debug.draw_box(
            carla.BoundingBox(
                zone_center,
                carla.Vector3D(
                    x=float(zone_depth * 0.5),
                    y=float(zone_width * 0.5),
                    z=0.06,
                ),
            ),
            zone_rot,
            thickness=0.10,
            color=INDIGO,
            life_time=life_time,
        )

        sign_loc = sign_tf.location
        candidates.append(
            ControlCandidate(
                kind="STOP",
                actor=sign,
                anchor_loc=zone_center,
                label_loc=carla.Location(
                    x=float(sign_loc.x),
                    y=float(sign_loc.y),
                    z=float(sign_loc.z) + 2.4,
                ),
                distance=float(zone_center.distance(ego_loc)),
                longitudinal=float(longitudinal),
                road_id=_road_id_at(carla_map, zone_center),
            )
        )
        count += 1

    for sign in actors.filter("traffic.yield*"):
        trigger = getattr(sign, "trigger_volume", None)
        if trigger is None:
            continue

        sign_tf = sign.get_transform()
        trigger_loc = local_to_world_no_mutate(sign_tf, trigger.location)
        if trigger_loc.distance(ego_loc) > args.stop_search_radius:
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
        debug.draw_line(
            p1,
            p2,
            thickness=0.12,
            color=INDIGO,
            life_time=life_time,
        )

        sign_loc = sign_tf.location
        line_longitudinal = _ahead_longitudinal(line_center)
        candidates.append(
            ControlCandidate(
                kind="YIELD",
                actor=sign,
                anchor_loc=line_center,
                label_loc=carla.Location(
                    x=float(sign_loc.x),
                    y=float(sign_loc.y),
                    z=float(sign_loc.z) + 2.4,
                ),
                distance=float(line_center.distance(ego_loc)),
                longitudinal=float(line_longitudinal),
                road_id=_road_id_at(carla_map, line_center),
            )
        )
        count += 1

    active_target = None
    if candidates:
        same_road = [c for c in candidates if ego_road_id is not None and c.road_id == ego_road_id]
        pool = same_road if same_road else candidates
        active_target = min(pool, key=lambda c: (c.longitudinal, c.distance))
        debug.draw_point(
            active_target.label_loc,
            size=0.15,
            color=WHITE,
            life_time=life_time,
        )
        debug.draw_string(
            carla.Location(
                x=float(active_target.label_loc.x),
                y=float(active_target.label_loc.y),
                z=float(active_target.label_loc.z) + 0.55,
            ),
            f"{active_target.kind} {active_target.distance:.1f}m",
            draw_shadow=True,
            color=WHITE,
            life_time=life_time,
        )
        debug.draw_line(
            active_target.label_loc,
            carla.Location(
                x=float(active_target.anchor_loc.x),
                y=float(active_target.anchor_loc.y),
                z=float(active_target.anchor_loc.z) + 0.10,
            ),
            thickness=0.04,
            color=WHITE,
            life_time=life_time,
        )

    return count, active_target


def draw_overlay(
    world: Any,
    carla_map: Any,
    ego: Any,
    junction_boxes: list[Any],
    args: argparse.Namespace,
    town: str,
    elapsed: float,
) -> OverlayCounts:
    debug = world.debug
    life_time = args.life_time
    counts = OverlayCounts()

    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    fwd = ego_tf.get_forward_vector()
    fwd_2d = np.array([float(fwd.x), float(fwd.y)], dtype=np.float64)
    fwd_norm = np.linalg.norm(fwd_2d)
    if fwd_norm < 1e-6:
        fwd_2d = np.array([1.0, 0.0], dtype=np.float64)
    else:
        fwd_2d /= fwd_norm
    right_2d = np.array([-fwd_2d[1], fwd_2d[0]], dtype=np.float64)

    ego_wp = carla_map.get_waypoint(
        ego_loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    lane_half = args.corridor_half_width
    if ego_wp is not None:
        lane_half = max(lane_half, 0.5 * float(ego_wp.lane_width))

    corridor = build_forward_corridor(
        ego_tf,
        length=args.corridor_length,
        half_width=lane_half,
    )
    draw_polygon(
        debug,
        corridor,
        z=float(ego_loc.z) + 0.25,
        color=MAGENTA,
        life_time=life_time,
        thickness=0.06,
    )

    if ego_wp is not None:
        nxt = ego_wp.next(12.0)
        if nxt:
            tloc = nxt[0].transform.location
            debug.draw_point(
                carla.Location(tloc.x, tloc.y, tloc.z + 0.9),
                size=0.14,
                color=SKY,
                life_time=life_time,
            )
            debug.draw_line(
                carla.Location(ego_loc.x, ego_loc.y, ego_loc.z + 0.7),
                carla.Location(tloc.x, tloc.y, tloc.z + 0.9),
                thickness=0.04,
                color=SKY,
                life_time=life_time,
            )

    actors = world.get_actors()

    for veh in actors.filter("vehicle.*"):
        if veh.id == ego.id:
            continue
        loc = veh.get_location()
        distance = float(loc.distance(ego_loc))
        if distance > args.vicinity_radius:
            continue

        veh_obb = get_actor_obb_2d(veh)
        sat_hit = sat_overlap(corridor, veh_obb)
        emergency = distance <= args.vehicle_emergency_distance

        if emergency:
            color = RED
            counts.vehicles_red += 1
        elif sat_hit:
            color = YELLOW
            counts.vehicles_yellow += 1
        else:
            color = WHITE
            counts.vehicles_white += 1

        debug.draw_point(
            carla.Location(loc.x, loc.y, loc.z + 1.2),
            size=0.13,
            color=color,
            life_time=life_time,
        )

        if args.draw_labels:
            tag = "V:R" if emergency else ("V:Y" if sat_hit else "V:W")
            debug.draw_string(
                carla.Location(loc.x, loc.y, loc.z + 1.7),
                f"{tag} {distance:.0f}m",
                draw_shadow=True,
                color=color,
                life_time=life_time,
            )

    for ped in actors.filter("walker.pedestrian.*"):
        loc = ped.get_location()
        distance = float(loc.distance(ego_loc))
        if distance > args.vicinity_radius:
            continue

        rel = np.array([float(loc.x - ego_loc.x), float(loc.y - ego_loc.y)], dtype=np.float64)
        longitudinal = _dot2(rel, fwd_2d)
        lateral = _dot2(rel, right_2d)

        vel = ped.get_velocity()
        pred_loc = np.array(
            [
                float(loc.x + vel.x * args.ped_prediction_sec),
                float(loc.y + vel.y * args.ped_prediction_sec),
            ],
            dtype=np.float64,
        )
        pred_rel = pred_loc - np.array([float(ego_loc.x), float(ego_loc.y)], dtype=np.float64)
        pred_longitudinal = _dot2(pred_rel, fwd_2d)
        pred_lateral = _dot2(pred_rel, right_2d)

        in_lane_now = longitudinal > 0.0 and abs(lateral) <= args.ped_lateral_threshold
        in_lane_pred = pred_longitudinal > 0.0 and abs(pred_lateral) <= args.ped_lateral_threshold

        lateral_or_pred = in_lane_now or in_lane_pred

        ped_obb = get_actor_obb_2d(ped)
        sat_hit = sat_overlap(corridor, ped_obb)
        yellow_hit = sat_hit or lateral_or_pred
        emergency_now = in_lane_now and longitudinal <= args.ped_emergency_distance
        emergency_pred = in_lane_pred and pred_longitudinal <= args.ped_emergency_distance
        emergency_dist = yellow_hit and distance <= args.ped_emergency_distance
        emergency = emergency_now or emergency_pred or emergency_dist

        if emergency:
            color = RED
            counts.peds_red += 1
        elif yellow_hit:
            color = YELLOW
            counts.peds_yellow += 1
        else:
            color = WHITE
            counts.peds_white += 1

        debug.draw_point(
            carla.Location(loc.x, loc.y, loc.z + 1.0),
            size=0.11,
            color=color,
            life_time=life_time,
        )

        if lateral_or_pred:
            pred_color = RED if emergency else YELLOW
            debug.draw_point(
                carla.Location(float(pred_loc[0]), float(pred_loc[1]), loc.z + 1.0),
                size=0.08,
                color=pred_color,
                life_time=life_time,
            )

        if args.draw_labels:
            tag = "P:R" if emergency else ("P:Y" if yellow_hit else "P:W")
            debug.draw_string(
                carla.Location(loc.x, loc.y, loc.z + 1.45),
                f"{tag} {distance:.0f}m",
                draw_shadow=True,
                color=color,
                life_time=life_time,
            )

    counts.stop_targets, active_control = draw_stop_targets(
        world,
        carla_map,
        ego_loc,
        fwd_2d,
        args,
    )

    follow_text = "none"
    if active_control is not None:
        follow_text = f"{active_control.kind} {active_control.distance:.1f}m"

    for bb in junction_boxes:
        if bb.location.distance(ego_loc) > args.junction_radius:
            continue
        debug.draw_box(
            bb,
            getattr(bb, "rotation", carla.Rotation()),
            thickness=0.08,
            color=GREEN,
            life_time=life_time,
        )
        counts.junction_boxes += 1

    hud = (
        f"{town} | t={elapsed:.1f}s\n"
        f"veh W/Y/R: {counts.vehicles_white}/{counts.vehicles_yellow}/{counts.vehicles_red} | "
        f"ped W/Y/R: {counts.peds_white}/{counts.peds_yellow}/{counts.peds_red}\n"
        f"stop targets: {counts.stop_targets} | follow: {follow_text} | "
        f"junction boxes: {counts.junction_boxes}"
    )
    debug.draw_string(
        carla.Location(ego_loc.x, ego_loc.y, ego_loc.z + 3.2),
        hud,
        draw_shadow=True,
        color=WHITE,
        life_time=life_time,
    )

    return counts


def configure_sync_mode(world: Any, fixed_delta: float) -> Any:
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta
    world.apply_settings(settings)
    return original


def restore_world_settings(world: Any, original_settings: Any) -> None:
    world.apply_settings(original_settings)


def setup_scenario(
    world: Any,
    tm: Any,
    args: argparse.Namespace,
    rng: random.Random,
) -> ScenarioActors:
    ego, ego_spawn = spawn_ego(world, rng)
    ego.set_autopilot(True, tm.get_port())

    vehicles = spawn_npc_vehicles(
        world,
        tm_port=tm.get_port(),
        ego_spawn=ego_spawn,
        count=args.traffic_vehicles,
        rng=rng,
    )

    walkers, controllers = spawn_walkers_near_ego(
        world,
        ego_spawn=ego_spawn,
        walker_count=args.walkers,
        radius=args.walker_radius,
        rng=rng,
    )

    world.tick()
    return ScenarioActors(
        ego=ego,
        vehicles=vehicles,
        walkers=walkers,
        walker_controllers=controllers,
    )


def run_town_preview(
    client: Any,
    tm: Any,
    town: str,
    args: argparse.Namespace,
    rng: random.Random,
) -> None:
    print(f"\n=== Loading {town} ===")
    world = client.get_world()
    current_map = world.get_map().name
    if not current_map.endswith(town):
        world = client.load_world(town)
        time.sleep(2.0)
    else:
        print(f"Map {town} already loaded")

    world = client.get_world()
    carla_map = world.get_map()

    original_settings = configure_sync_mode(world, args.fixed_delta)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.global_percentage_speed_difference(0.0)

    scenario: ScenarioActors | None = None

    try:
        junction_boxes = collect_junction_boxes(carla_map, args.junction_sample_spacing)
        print(f"Collected {len(junction_boxes)} unique junction boxes")

        scenario = setup_scenario(world, tm, args, rng)
        print(
            "Spawned actors: "
            f"ego=1, vehicles={len(scenario.vehicles)}, "
            f"walkers={len(scenario.walkers)}"
        )

        total_ticks = max(1, int(args.seconds_per_town / args.fixed_delta))
        t_start = time.time()

        for tick_idx in range(total_ticks):
            world.tick()

            if not scenario.ego.is_alive:
                print("Ego actor no longer alive; stopping this town preview")
                break

            follow_ego_with_spectator(world, scenario.ego)
            elapsed = time.time() - t_start
            counts = draw_overlay(
                world=world,
                carla_map=carla_map,
                ego=scenario.ego,
                junction_boxes=junction_boxes,
                args=args,
                town=town,
                elapsed=elapsed,
            )

            if tick_idx % int(max(1, round(1.0 / args.fixed_delta))) == 0:
                veh_counts = (
                    f"{counts.vehicles_white}/"
                    f"{counts.vehicles_yellow}/"
                    f"{counts.vehicles_red}"
                )
                ped_counts = f"{counts.peds_white}/" f"{counts.peds_yellow}/" f"{counts.peds_red}"
                print(
                    f"  t={elapsed:5.1f}s "
                    f"veh(W/Y/R)={veh_counts} "
                    f"ped(W/Y/R)={ped_counts} "
                    f"stop={counts.stop_targets} jct={counts.junction_boxes}"
                )

        print(f"Completed {town} preview")

    finally:
        if scenario is not None:
            for controller in scenario.walker_controllers:
                try:
                    controller.stop()
                except RuntimeError:
                    continue
            destroy_actors(scenario.all_spawned())
            world.tick()

        tm.set_synchronous_mode(False)
        restore_world_settings(world, original_settings)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    host = resolve_host(args.host)
    print(f"Connecting to CARLA at {host}:{args.port}")

    client = carla.Client(host, args.port)
    client.set_timeout(45.0)
    tm = client.get_trafficmanager(args.tm_port)

    print("Preview palette (approved 7-color list):")
    print("  GREEN  #00FF00 | SKY #00AAFF | MAGENTA #FF00FF")
    print("  RED    #FF0000 | YELLOW #FFFF00 | WHITE #FFFFFF | INDIGO #6600FF")

    try:
        for town in args.towns:
            run_town_preview(client, tm, town, args, rng)
    except KeyboardInterrupt:
        print("\nInterrupted by user; stopping preview")


if __name__ == "__main__":
    main()
