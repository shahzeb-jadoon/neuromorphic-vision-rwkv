#!/usr/bin/env python3
"""Manual CARLA driving with SAT overlays and lane-direction HUD.

This script spawns an ego vehicle and lets you drive it manually while
rendering the same visualization layers as scripts/review_episode_interactive.py:

- SAT corridor (magenta)
- Next 5 waypoints (green) + target point/line (cyan)
- Junction boxes
- Indigo control zones for TL/STOP/YIELD with distance labels
- Ego/actor 2D footprint boxes with SAT hit coloring
- Top-left lane/direction HUD

Global controls (recommended):
- W: toggle throttle on/off
- A / D: steering nudge left / right
- S: brake (while held in global keyboard mode)
- Space: hand brake
- R: toggle reverse gear
- P: pause/resume driving (free-camera while paused)
- Esc: quit

Terminal fallback controls:
- w: toggle throttle on/off
- s: brake
- x: neutral throttle+brake
- a/d: steering nudge left / right
- c: center steering
- space: toggle hand brake
- r: toggle reverse
- p: pause/resume driving (free-camera while paused)
- q: quit
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

try:
    import carla
except ImportError as exc:  # pragma: no cover
    raise SystemExit("CARLA Python API is required. Activate your CARLA env first.") from exc

from scripts import review_episode_interactive as viz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class DriveState:
    throttle: bool = False
    throttle_toggle_latch: bool = False
    brake: bool = False
    steer_pulse_dir: int = 0
    steer_pulse_until: float = 0.0
    hand_brake: bool = False
    reverse: bool = False
    reverse_latch: bool = False
    pause_latch: bool = False
    paused: bool = False
    quit_requested: bool = False
    last_global_event_time: float = 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual driving with SAT/HUD overlays")
    parser.add_argument("--host", default="auto", help="CARLA host (default: auto WSL2 gateway)")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--town", default=None, help="Optional town to load (e.g., Town03)")
    parser.add_argument(
        "--vehicle-filter",
        default="vehicle.tesla.model3",
        help="Blueprint filter for ego vehicle",
    )
    parser.add_argument(
        "--spawn-index",
        type=int,
        default=0,
        help="Spawn point index (wraps if out of range)",
    )
    parser.add_argument(
        "--control-mode",
        choices=["global", "terminal"],
        default="global",
        help="Input mode. global uses pynput keyboard hooks.",
    )
    parser.add_argument(
        "--tick-hz",
        type=float,
        default=20.0,
        help="Simulation tick rate",
    )
    parser.add_argument(
        "--draw-life-time",
        type=float,
        default=0.30,
        help="CARLA debug primitive lifetime in seconds",
    )
    parser.add_argument(
        "--sat-radius-m",
        type=float,
        default=20.0,
        help="Draw actor bboxes within this radius",
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
    parser.add_argument(
        "--route-step-m",
        type=float,
        default=2.0,
        help="Waypoint sampling step for live route preview",
    )
    parser.add_argument(
        "--route-lookahead-m",
        type=float,
        default=60.0,
        help="Live route preview length",
    )
    parser.add_argument(
        "--spectator-follow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Follow ego with spectator camera",
    )
    parser.add_argument(
        "--spectator-drive-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When global keyboard hooks produce no events, infer WASD intent from "
            "spectator movement in the CARLA window"
        ),
    )
    return parser.parse_args()


def _spawn_ego(world: carla.World, vehicle_filter: str, spawn_index: int) -> carla.Vehicle:
    bp_lib = world.get_blueprint_library()
    bps = bp_lib.filter(vehicle_filter)
    if not bps:
        bps = bp_lib.filter("vehicle.*")
    if not bps:
        raise RuntimeError("No vehicle blueprints found")

    bp = random.choice(list(bps))
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "hero")

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points in current map")

    idx = int(spawn_index) % len(spawn_points)
    ordered_indices = [idx] + [i for i in range(len(spawn_points)) if i != idx]

    for i in ordered_indices:
        actor = world.try_spawn_actor(bp, spawn_points[i])
        if actor is not None:
            logger.info("Spawned ego at spawn index %d with blueprint %s", i, bp.id)
            return actor

    raise RuntimeError("Failed to spawn ego vehicle at any spawn point")


def _update_spectator(
    world: carla.World, vehicle: carla.Vehicle, offset: tuple[float, float, float]
) -> None:
    vt = vehicle.get_transform()
    fwd = vt.get_forward_vector()
    right = vt.get_right_vector()
    up = carla.Location(x=0.0, y=0.0, z=1.0)

    cam_loc = vt.location + carla.Location(
        x=fwd.x * offset[0] + right.x * offset[1] + up.x * offset[2],
        y=fwd.y * offset[0] + right.y * offset[1] + up.y * offset[2],
        z=fwd.z * offset[0] + right.z * offset[1] + up.z * offset[2],
    )
    cam_rot = carla.Rotation(pitch=-15.0, yaw=vt.rotation.yaw, roll=0.0)
    world.get_spectator().set_transform(carla.Transform(cam_loc, cam_rot))


def _safe_transform_bb_center(actor: carla.Actor) -> carla.Location:
    bb = actor.bounding_box
    tf = actor.get_transform()
    local = carla.Location(bb.location.x, bb.location.y, bb.location.z)
    return tf.transform(local)


def _choose_next_waypoint(current_wp: Any, step_m: float) -> Any | None:
    candidates = list(current_wp.next(max(0.5, float(step_m))))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    cur_fwd = current_wp.transform.get_forward_vector()
    cur_v = np.array([float(cur_fwd.x), float(cur_fwd.y)], dtype=np.float64)
    norm = float(np.linalg.norm(cur_v))
    if norm < 1e-6:
        cur_v = np.array([1.0, 0.0], dtype=np.float64)
    else:
        cur_v /= norm

    best = candidates[0]
    best_score = -1e9
    for cand in candidates:
        cand_fwd = cand.transform.get_forward_vector()
        cand_v = np.array([float(cand_fwd.x), float(cand_fwd.y)], dtype=np.float64)
        cn = float(np.linalg.norm(cand_v))
        if cn < 1e-6:
            continue
        cand_v /= cn
        score = float(cand_v @ cur_v)
        if score > best_score:
            best_score = score
            best = cand
    return best


def _build_live_route_data(
    world: carla.World,
    vehicle: carla.Vehicle,
    route_step_m: float,
    route_lookahead_m: float,
) -> viz.RouteData:
    carla_map = world.get_map()
    ego_tf = vehicle.get_transform()
    ego_wp = carla_map.get_waypoint(
        ego_tf.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if ego_wp is None:
        return viz.RouteData(
            path=None,
            points_xyz=np.zeros((0, 3), dtype=np.float32),
            points_junction=np.zeros((0,), dtype=bool),
            points_yaw=np.zeros((0,), dtype=np.float32),
            points_road_id=np.zeros((0,), dtype=np.int32),
            points_lane_id=np.zeros((0,), dtype=np.int32),
            options=[],
        )

    waypoints = [ego_wp]
    seen_ids = {int(ego_wp.id)}
    traveled = 0.0
    current = ego_wp

    while traveled < max(2.0, float(route_lookahead_m)) and len(waypoints) < 180:
        nxt = _choose_next_waypoint(current, route_step_m)
        if nxt is None:
            break
        nxt_id = int(nxt.id)
        if nxt_id in seen_ids:
            break
        traveled += float(current.transform.location.distance(nxt.transform.location))
        waypoints.append(nxt)
        seen_ids.add(nxt_id)
        current = nxt

    xyz = np.array(
        [
            [
                float(wp.transform.location.x),
                float(wp.transform.location.y),
                float(wp.transform.location.z),
            ]
            for wp in waypoints
        ],
        dtype=np.float32,
    )
    is_junction = np.array([bool(wp.is_junction) for wp in waypoints], dtype=bool)
    yaw = np.array([float(wp.transform.rotation.yaw) for wp in waypoints], dtype=np.float32)
    road_id = np.array([int(getattr(wp, "road_id", -1)) for wp in waypoints], dtype=np.int32)
    lane_id = np.array([int(getattr(wp, "lane_id", 0)) for wp in waypoints], dtype=np.int32)

    return viz.RouteData(
        path=None,
        points_xyz=xyz,
        points_junction=is_junction,
        points_yaw=yaw,
        points_road_id=road_id,
        points_lane_id=lane_id,
        options=["LANEFOLLOW"] * len(waypoints),
    )


def _compute_control_targets(
    world: carla.World,
    ego_loc: carla.Location,
    ego_fwd_2d: np.ndarray,
) -> tuple[int, float, float]:
    stop_search_radius = 70.0
    actors = world.get_actors()

    fwd_norm = float(np.linalg.norm(ego_fwd_2d))
    if fwd_norm < 1e-6:
        ego_fwd_2d = np.array([1.0, 0.0], dtype=np.float64)
    else:
        ego_fwd_2d = ego_fwd_2d / fwd_norm

    def _ahead_longitudinal(loc: carla.Location) -> float:
        rel = np.array([float(loc.x - ego_loc.x), float(loc.y - ego_loc.y)], dtype=np.float64)
        return viz._dot2(rel, ego_fwd_2d)

    tl_actor_id = -1
    tl_best_dist = 1e9
    for tl in actors.filter("traffic.traffic_light*"):
        tl_loc = tl.get_transform().location
        if tl_loc.distance(ego_loc) > stop_search_radius:
            continue
        for wp in viz._collect_tl_waypoints(tl):
            loc = wp.transform.location
            if loc.distance(ego_loc) > stop_search_radius:
                continue
            if _ahead_longitudinal(loc) <= 0.0:
                continue
            dist = float(loc.distance(ego_loc))
            if dist < tl_best_dist:
                tl_best_dist = dist
                tl_actor_id = int(tl.id)

    stop_dist = -1.0
    best_stop = 1e9
    for sign in actors.filter("traffic.stop*"):
        trigger = getattr(sign, "trigger_volume", None)
        if trigger is None:
            continue
        sign_tf = sign.get_transform()
        trigger_loc = viz._local_to_world_no_mutate(sign_tf, trigger.location)
        if trigger_loc.distance(ego_loc) > stop_search_radius:
            continue
        if _ahead_longitudinal(trigger_loc) <= 0.0:
            continue
        dist = float(trigger_loc.distance(ego_loc))
        if dist < best_stop:
            best_stop = dist
            stop_dist = dist

    yield_dist = -1.0
    best_yield = 1e9
    for sign in actors.filter("traffic.yield*"):
        trigger = getattr(sign, "trigger_volume", None)
        if trigger is None:
            continue
        sign_tf = sign.get_transform()
        trigger_loc = viz._local_to_world_no_mutate(sign_tf, trigger.location)
        if trigger_loc.distance(ego_loc) > stop_search_radius:
            continue
        if _ahead_longitudinal(trigger_loc) <= 0.0:
            continue
        dist = float(trigger_loc.distance(ego_loc))
        if dist < best_yield:
            best_yield = dist
            yield_dist = dist

    return tl_actor_id, stop_dist, yield_dist


def _draw_live_frame(
    world: carla.World,
    vehicle: carla.Vehicle,
    road_cache: viz.RoadLaneCache | None,
    args: argparse.Namespace,
) -> None:
    tf = vehicle.get_transform()
    ego_loc = tf.location
    ego_yaw = float(tf.rotation.yaw)

    route = _build_live_route_data(
        world=world,
        vehicle=vehicle,
        route_step_m=float(args.route_step_m),
        route_lookahead_m=max(
            float(args.route_lookahead_m), float(args.sat_corridor_length_m) + 5.0
        ),
    )
    if route.points_xyz.shape[0] == 0:
        return

    ep_stub: viz.EpisodeData = cast(
        viz.EpisodeData,
        SimpleNamespace(
            position=np.array(
                [[float(ego_loc.x), float(ego_loc.y), float(ego_loc.z)]],
                dtype=np.float32,
            ),
            rotation=np.array(
                [[float(tf.rotation.pitch), float(tf.rotation.yaw), float(tf.rotation.roll)]],
                dtype=np.float32,
            ),
            wp_buffer_head_index=np.array([0], dtype=np.int32),
            wp_buffer_size=np.array([0], dtype=np.int32),
            wp_buffer_head_dist=np.array([0.0], dtype=np.float32),
            arc_active=np.array([0], dtype=np.int8),
            steer_from_arc=np.array([0], dtype=np.int8),
        ),
    )

    corridor_2d, corridor_3d = viz._build_sat_corridor_segments(
        ep=ep_stub,
        route=route,
        frame_idx=0,
        ridx=0,
        corridor_length_m=float(args.sat_corridor_length_m),
        half_width_m=float(args.sat_half_width_m),
    )

    for quad in corridor_3d:
        viz._draw_quad_outline(
            world.debug, quad, viz.CORRIDOR_COLOR, float(args.draw_life_time), thickness=0.06
        )

    viz._draw_next_waypoints_and_target(world, route, 0, ego_loc, float(args.draw_life_time))
    viz._draw_junction_boxes(world, route, 0, ego_loc, float(args.draw_life_time))

    yaw_rad = math.radians(ego_yaw)
    ego_fwd_2d = np.array([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=np.float64)
    tl_actor_id, stop_dist, yield_dist = _compute_control_targets(world, ego_loc, ego_fwd_2d)
    viz._draw_active_control_zone(
        world=world,
        ego_loc=ego_loc,
        ego_fwd_2d=ego_fwd_2d,
        traffic_light_actor_id=int(tl_actor_id),
        stop_sign_distance=float(stop_dist),
        yield_sign_distance=float(yield_dist),
        life_time=float(args.draw_life_time),
    )

    ego_center = _safe_transform_bb_center(vehicle)
    ego_bb = vehicle.bounding_box
    ego_len = max(float(args.vehicle_length_m), float(ego_bb.extent.x) * 2.0)
    ego_wid = max(float(args.vehicle_width_m), float(ego_bb.extent.y) * 2.0)
    ego_corners = viz._bbox_corners_2d(
        center_xy=np.array([float(ego_center.x), float(ego_center.y)], dtype=np.float64),
        yaw_deg=ego_yaw,
        length_m=ego_len,
        width_m=ego_wid,
    )
    viz._draw_bbox_outline(
        world.debug,
        ego_corners,
        z=float(ego_center.z) + 0.12,
        color=viz.EGO_COLOR,
        life_time=float(args.draw_life_time),
        thickness=0.11,
    )

    sat_radius = max(0.1, float(args.sat_radius_m))

    for actor in world.get_actors().filter("*vehicle*"):
        if int(actor.id) == int(vehicle.id):
            continue
        center = _safe_transform_bb_center(actor)
        if float(center.distance(ego_center)) > sat_radius:
            continue

        bb = actor.bounding_box
        length_m = max(float(args.vehicle_length_m), float(bb.extent.x) * 2.0)
        width_m = max(float(args.vehicle_width_m), float(bb.extent.y) * 2.0)
        yaw = float(actor.get_transform().rotation.yaw)
        corners = viz._bbox_corners_2d(
            center_xy=np.array([float(center.x), float(center.y)], dtype=np.float64),
            yaw_deg=yaw,
            length_m=length_m,
            width_m=width_m,
        )
        sat_hit = any(viz._sat_overlap(seg, corners) for seg in corridor_2d)
        color = viz.SAT_COLOR if sat_hit else viz.ACTOR_COLOR
        viz._draw_bbox_outline(
            world.debug,
            corners,
            z=float(center.z) + 0.15,
            color=color,
            life_time=float(args.draw_life_time),
            thickness=0.09,
        )

    for actor in world.get_actors().filter("*walker*"):
        center = _safe_transform_bb_center(actor)
        if float(center.distance(ego_center)) > sat_radius:
            continue

        bb = actor.bounding_box
        length_m = max(float(args.pedestrian_length_m), float(bb.extent.x) * 2.0)
        width_m = max(float(args.pedestrian_width_m), float(bb.extent.y) * 2.0)
        yaw = float(actor.get_transform().rotation.yaw)
        corners = viz._bbox_corners_2d(
            center_xy=np.array([float(center.x), float(center.y)], dtype=np.float64),
            yaw_deg=yaw,
            length_m=length_m,
            width_m=width_m,
        )
        sat_hit = any(viz._sat_overlap(seg, corners) for seg in corridor_2d)
        color = viz.SAT_COLOR if sat_hit else viz.ACTOR_COLOR
        viz._draw_bbox_outline(
            world.debug,
            corners,
            z=float(center.z) + 0.10,
            color=color,
            life_time=float(args.draw_life_time),
            thickness=0.08,
        )

    viz._draw_lane_direction_hud(
        world=world,
        ep=ep_stub,
        route=route,
        frame_idx=0,
        ridx=0,
        ego_loc=ego_loc,
        road_cache=road_cache,
        life_time=float(args.draw_life_time),
    )


def _setup_global_keyboard(state: DriveState) -> tuple[Any | None, bool]:
    try:
        from pynput import keyboard as pynput_keyboard  # type: ignore
    except Exception as exc:
        logger.warning("Global keyboard mode unavailable (%s). Falling back to terminal mode.", exc)
        return None, False

    def _key_char(key: Any) -> str | None:
        try:
            ch = key.char
        except AttributeError:
            ch = None
        return ch.lower() if isinstance(ch, str) else None

    def _push_steer_pulse(direction: int, duration_s: float = 0.14) -> None:
        if direction == 0:
            state.steer_pulse_dir = 0
            state.steer_pulse_until = 0.0
            return
        state.steer_pulse_dir = -1 if direction < 0 else 1
        state.steer_pulse_until = time.time() + max(0.02, float(duration_s))

    def on_press(key: Any) -> None:
        state.last_global_event_time = time.time()
        ch = _key_char(key)
        if key == pynput_keyboard.Key.esc:
            state.quit_requested = True
            return
        if ch == "w" and not state.throttle_toggle_latch:
            state.throttle = not state.throttle
            state.throttle_toggle_latch = True
            if state.throttle:
                state.brake = False
        elif ch == "s":
            state.brake = True
            state.throttle = False
        elif ch == "a":
            _push_steer_pulse(-1)
        elif ch == "d":
            _push_steer_pulse(+1)
        elif ch == "r" and not state.reverse_latch:
            state.reverse = not state.reverse
            state.reverse_latch = True
        elif ch == "p" and not state.pause_latch:
            state.paused = not state.paused
            state.pause_latch = True
            state.throttle = False
            state.brake = False
            _push_steer_pulse(0)
            logger.info(
                "Driving %s; %s",
                "paused" if state.paused else "resumed",
                "free-camera enabled" if state.paused else "follow controls enabled",
            )
        elif key == pynput_keyboard.Key.space:
            state.hand_brake = True

    def on_release(key: Any) -> None:
        state.last_global_event_time = time.time()
        ch = _key_char(key)
        if ch == "w":
            state.throttle_toggle_latch = False
        elif ch == "s":
            state.brake = False
        elif ch == "r":
            state.reverse_latch = False
        elif ch == "p":
            state.pause_latch = False
        elif key == pynput_keyboard.Key.space:
            state.hand_brake = False

    listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener, True


def _read_terminal_keys(state: DriveState) -> None:
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not ready:
        return

    ch = sys.stdin.read(1)
    if not ch:
        return

    ch = ch.lower()

    def _push_steer_pulse(direction: int, duration_s: float = 0.14) -> None:
        if direction == 0:
            state.steer_pulse_dir = 0
            state.steer_pulse_until = 0.0
            return
        state.steer_pulse_dir = -1 if direction < 0 else 1
        state.steer_pulse_until = time.time() + max(0.02, float(duration_s))

    if ch == "q":
        state.quit_requested = True
    elif ch == "w":
        state.throttle = not state.throttle
        if state.throttle:
            state.brake = False
    elif ch == "s":
        state.brake = True
        state.throttle = False
    elif ch == "x":
        state.throttle = False
        state.brake = False
    elif ch == "a":
        _push_steer_pulse(-1)
    elif ch == "d":
        _push_steer_pulse(+1)
    elif ch == "c":
        _push_steer_pulse(0)
    elif ch == "r":
        state.reverse = not state.reverse
    elif ch == "p":
        state.paused = not state.paused
        state.throttle = False
        state.brake = False
        _push_steer_pulse(0)
        logger.info(
            "Driving %s; %s",
            "paused" if state.paused else "resumed",
            "free-camera enabled" if state.paused else "follow controls enabled",
        )
    elif ch == " ":
        state.hand_brake = not state.hand_brake


def _apply_drive_control(vehicle: carla.Vehicle, state: DriveState) -> None:
    steer_left = False
    steer_right = False
    now = time.time()
    if state.steer_pulse_dir != 0 and now <= state.steer_pulse_until:
        steer_left = state.steer_pulse_dir < 0
        steer_right = state.steer_pulse_dir > 0
    else:
        state.steer_pulse_dir = 0

    _apply_drive_control_from_intent(
        vehicle=vehicle,
        throttle=bool(state.throttle),
        brake=bool(state.brake),
        steer_left=bool(steer_left),
        steer_right=bool(steer_right),
        hand_brake=bool(state.hand_brake),
        reverse=bool(state.reverse),
    )


def _spectator_delta_intent(
    vehicle: carla.Vehicle,
    prev_spectator_tf: carla.Transform | None,
    cur_spectator_tf: carla.Transform,
) -> tuple[bool, bool, bool, bool]:
    """Infer throttle/brake/steer from spectator displacement between ticks."""
    if prev_spectator_tf is None:
        return False, False, False, False

    delta = np.array(
        [
            float(cur_spectator_tf.location.x - prev_spectator_tf.location.x),
            float(cur_spectator_tf.location.y - prev_spectator_tf.location.y),
        ],
        dtype=np.float64,
    )

    if float(np.linalg.norm(delta)) < 0.01:
        return False, False, False, False

    vt = vehicle.get_transform()
    fwd = vt.get_forward_vector()
    right = vt.get_right_vector()
    fwd_2d = np.array([float(fwd.x), float(fwd.y)], dtype=np.float64)
    right_2d = np.array([float(right.x), float(right.y)], dtype=np.float64)

    fn = float(np.linalg.norm(fwd_2d))
    rn = float(np.linalg.norm(right_2d))
    if fn < 1e-6 or rn < 1e-6:
        return False, False, False, False
    fwd_2d /= fn
    right_2d /= rn

    longitudinal = float(delta @ fwd_2d)
    lateral = float(delta @ right_2d)

    move_thr = 0.04
    steer_thr = 0.03
    throttle = longitudinal > move_thr
    brake = longitudinal < -move_thr
    steer_left = lateral < -steer_thr
    steer_right = lateral > steer_thr
    return throttle, brake, steer_left, steer_right


def _apply_drive_control_from_intent(
    vehicle: carla.Vehicle,
    throttle: bool,
    brake: bool,
    steer_left: bool,
    steer_right: bool,
    hand_brake: bool,
    reverse: bool,
) -> None:
    control = carla.VehicleControl()
    control.reverse = bool(reverse)
    control.hand_brake = bool(hand_brake)

    if throttle and not brake:
        control.throttle = 0.55
        control.brake = 0.0
    elif brake and not throttle:
        control.throttle = 0.0
        control.brake = 0.70
    else:
        control.throttle = 0.0
        control.brake = 0.0

    if steer_left and not steer_right:
        control.steer = -0.55
    elif steer_right and not steer_left:
        control.steer = 0.55
    else:
        control.steer = 0.0

    vehicle.apply_control(control)


def main() -> None:
    args = _parse_args()

    host = viz._detect_wsl2_host() if args.host == "auto" else args.host
    logger.info("Connecting to CARLA at %s:%d", host, args.port)
    client = carla.Client(host, int(args.port))
    client.set_timeout(60.0)
    world = client.get_world()

    if args.town:
        cur = world.get_map().name.split("/")[-1]
        if cur != args.town:
            logger.info("Loading map %s (current=%s)", args.town, cur)
            client.load_world(args.town)
            world = client.get_world()

    original_settings = world.get_settings()
    ego: carla.Vehicle | None = None
    listener = None
    old_tty = None

    try:
        new_settings = world.get_settings()
        new_settings.synchronous_mode = True
        new_settings.fixed_delta_seconds = 1.0 / max(float(args.tick_hz), 1.0)
        world.apply_settings(new_settings)

        ego = _spawn_ego(world, str(args.vehicle_filter), int(args.spawn_index))
        road_cache = viz._build_road_lane_cache(world.get_map(), spacing_m=2.0)
        logger.info("Road lane cache built (%d roads)", len(road_cache.by_road_lane))

        state = DriveState()
        if args.control_mode == "global":
            listener, global_enabled = _setup_global_keyboard(state)
        else:
            global_enabled = False

        use_terminal = (args.control_mode == "terminal") or (not global_enabled)
        terminal_backup = False
        if use_terminal:
            logger.info(
                "Terminal controls: w=toggle throttle, s=brake, x=neutral, "
                "a/d=steering nudge, c=center steer, space=toggle handbrake, "
                "r=toggle reverse, p=pause/resume, q=quit"
            )
            old_tty = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
            attrs = termios.tcgetattr(sys.stdin.fileno())
            attrs[3] = attrs[3] & ~termios.ECHO
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, attrs)
        else:
            logger.info(
                "Global controls: W toggle throttle, A/D steering nudge, S brake, "
                "Space handbrake, R reverse toggle, P pause/resume, Esc quit"
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
                    "w(toggle)/s/x/a/d(nudge)/c/space/r/p/q are also accepted "
                    "from this terminal."
                )

        frame_period_sec = 1.0 / max(float(args.tick_hz), 1.0)
        prev_spectator_tf = world.get_spectator().get_transform()
        fallback_notice_logged = False
        while not state.quit_requested:
            loop_start = time.time()
            cur_spectator_tf = world.get_spectator().get_transform()

            if use_terminal or terminal_backup:
                _read_terminal_keys(state)

            used_fallback = False
            if state.paused:
                _apply_drive_control_from_intent(
                    vehicle=ego,
                    throttle=False,
                    brake=True,
                    steer_left=False,
                    steer_right=False,
                    hand_brake=True,
                    reverse=False,
                )
            else:
                if (
                    args.control_mode == "global"
                    and global_enabled
                    and bool(args.spectator_drive_fallback)
                    and not any(
                        [
                            state.throttle,
                            state.brake,
                            (state.steer_pulse_dir != 0 and time.time() <= state.steer_pulse_until),
                            state.hand_brake,
                        ]
                    )
                ):
                    throttle, brake, steer_left, steer_right = _spectator_delta_intent(
                        vehicle=ego,
                        prev_spectator_tf=prev_spectator_tf,
                        cur_spectator_tf=cur_spectator_tf,
                    )
                    if throttle or brake or steer_left or steer_right:
                        _apply_drive_control_from_intent(
                            vehicle=ego,
                            throttle=throttle,
                            brake=brake,
                            steer_left=steer_left,
                            steer_right=steer_right,
                            hand_brake=False,
                            reverse=False,
                        )
                        used_fallback = True
                        if not fallback_notice_logged:
                            logger.info(
                                "Using CARLA-window WASD fallback via spectator "
                                "motion (no global key events needed)."
                            )
                            fallback_notice_logged = True

                if not used_fallback:
                    _apply_drive_control(ego, state)

            prev_spectator_tf = cur_spectator_tf

            if bool(args.spectator_follow) and not state.paused:
                _update_spectator(world, ego, offset=(-8.0, 0.0, 5.0))

            _draw_live_frame(world, ego, road_cache, args)

            world.tick()

            elapsed = time.time() - loop_start
            if elapsed < frame_period_sec:
                time.sleep(frame_period_sec - elapsed)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if listener is not None:
            listener.stop()

        if old_tty is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_tty)

        if ego is not None:
            try:
                ego.destroy()
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to destroy ego actor: %s", exc)

        try:
            world.apply_settings(original_settings)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to restore original CARLA world settings: %s", exc)


if __name__ == "__main__":
    main()
