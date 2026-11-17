#!/usr/bin/env python3
"""High-confidence stop-decision triage replay.

Runs one targeted CARLA replay with the project expert and exports:
1) Per-frame CSV dump with stop-decision evidence.
2) Screenshot timeline for visual correlation.

Per-frame dump includes:
- driving_mode
- lead vehicle id
- lead distance
- lead same_direction
- lead obb_in_lane
- dynamic emergency distance (dyn_emergency)
- selected control id
"""

from __future__ import annotations

import argparse
import csv
import os
import queue
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

try:
    import carla
except ImportError as exc:  # pragma: no cover
    raise SystemExit("CARLA Python API not found. Activate the CARLA environment first.") from exc

from agents.expert.privileged_expert import DrivingMode, ExpertConfig, PrivilegedExpert
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption

MODE_NAMES = {
    int(DrivingMode.FOLLOW_LANE): "FOLLOW_LANE",
    int(DrivingMode.APPROACHING_TURN): "APPROACHING_TURN",
    int(DrivingMode.TURNING): "TURNING",
    int(DrivingMode.TRAFFIC_LIGHT_STOP): "TRAFFIC_LIGHT_STOP",
    int(DrivingMode.VEHICLE_FOLLOWING): "VEHICLE_FOLLOWING",
    int(DrivingMode.PEDESTRIAN_STOP): "PEDESTRIAN_STOP",
    int(DrivingMode.EMERGENCY_STOP): "EMERGENCY_STOP",
    int(DrivingMode.STOP_SIGN_STOP): "STOP_SIGN_STOP",
    int(DrivingMode.YIELD_STOP): "YIELD_STOP",
}


@dataclass
class TriageRow:
    frame: int
    driving_mode_id: int
    driving_mode_name: str
    current_speed_kmh: float
    target_speed_kmh: float
    selected_control_kind: str
    selected_control_id: int
    tl_actor_id: int
    tl_distance_m: float
    stop_sign_active_id: int
    stop_sign_distance_m: float
    yield_sign_active_id: int
    yield_sign_distance_m: float
    lead_vehicle_id: int
    lead_distance_m: float
    lead_same_direction: int
    lead_obb_in_lane: int
    dyn_emergency_m: float
    sat_detections: int
    lateral_detections: int
    screenshot_file: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one targeted replay and dump per-frame stop-decision evidence "
            "plus screenshot timeline."
        )
    )
    parser.add_argument("--host", default="172.20.0.1", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--town", default="Town07", help="Town name")
    parser.add_argument("--weather", default="ClearNoon", help="CARLA weather preset")
    parser.add_argument("--fps", type=int, default=20, help="Synchronous FPS")
    parser.add_argument("--max-frames", type=int, default=1200, help="Maximum replay frames")
    parser.add_argument("--n-vehicles", type=int, default=50, help="NPC vehicle count")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument(
        "--min-route-waypoints",
        type=int,
        default=450,
        help="Minimum route length (waypoints) before replay starts",
    )
    parser.add_argument(
        "--screenshot-every",
        type=int,
        default=8,
        help="Capture every Nth frame (plus stop-related frames)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/triage_replay",
        help="Output directory for CSV and screenshots",
    )
    return parser.parse_args()


def _mode_name(mode_id: int) -> str:
    return MODE_NAMES.get(mode_id, f"UNKNOWN_{mode_id}")


def _pick_vehicle_blueprint(bp_lib: Any, rng: random.Random) -> Any:
    candidates = []
    for bp in bp_lib.filter("vehicle.*"):
        if bp.has_attribute("number_of_wheels"):
            try:
                if int(bp.get_attribute("number_of_wheels").as_int()) != 4:
                    continue
            except ValueError:
                continue
        candidates.append(bp)
    if not candidates:
        raise RuntimeError("No suitable vehicle blueprint found")
    return rng.choice(candidates)


def _spawn_ego(world: Any, rng: random.Random) -> tuple[Any, Any]:
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points in current town")

    rng.shuffle(spawn_points)
    try:
        ego_bp = bp_lib.find("vehicle.tesla.model3")
    except RuntimeError:
        ego_bp = _pick_vehicle_blueprint(bp_lib, rng)

    if ego_bp.has_attribute("role_name"):
        ego_bp.set_attribute("role_name", "hero")

    for sp in spawn_points:
        ego = world.try_spawn_actor(ego_bp, sp)
        if ego is not None:
            return ego, sp
    raise RuntimeError("Failed to spawn ego vehicle")


def _spawn_npc_vehicles(
    world: Any, tm_port: int, ego_spawn: Any, count: int, rng: random.Random
) -> list[Any]:
    if count <= 0:
        return []

    bp_lib = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    rng.shuffle(spawns)

    out: list[Any] = []
    for sp in spawns:
        if len(out) >= count:
            break
        if sp.location.distance(ego_spawn.location) < 12.0:
            continue

        bp = _pick_vehicle_blueprint(bp_lib, rng)
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


def _build_route(
    carla_map: Any,
    route_planner: GlobalRoutePlanner,
    start_spawn: Any,
    rng: random.Random,
    min_route_waypoints: int,
) -> list[tuple[Any, RoadOption]]:
    spawns = carla_map.get_spawn_points()
    if not spawns:
        raise RuntimeError("No spawn points available for route build")

    for _ in range(120):
        dest = rng.choice(spawns)
        if dest.location.distance(start_spawn.location) < 70.0:
            continue
        route = cast(
            list[tuple[Any, RoadOption]],
            route_planner.trace_route(start_spawn.location, dest.location),
        )
        if len(route) < min_route_waypoints:
            continue
        has_intersection = any(opt != RoadOption.LANEFOLLOW for _wp, opt in route)
        if has_intersection:
            return route

    # Last-resort fallback: longest route found in random attempts.
    best_route: list[tuple[Any, RoadOption]] = []
    for _ in range(40):
        dest = rng.choice(spawns)
        route = cast(
            list[tuple[Any, RoadOption]],
            route_planner.trace_route(start_spawn.location, dest.location),
        )
        if len(route) > len(best_route):
            best_route = route
    if best_route:
        return best_route

    raise RuntimeError("Failed to generate a usable route")


def _attach_camera(world: Any, ego: Any) -> tuple[Any, queue.Queue]:
    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "1280")
    cam_bp.set_attribute("image_size_y", "720")
    cam_bp.set_attribute("fov", "100")

    cam_tf = carla.Transform(
        carla.Location(x=1.6, z=1.7),
        carla.Rotation(pitch=-8.0, yaw=0.0, roll=0.0),
    )
    cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
    q: queue.Queue = queue.Queue()
    cam.listen(lambda img, qq=q: qq.put(img))
    return cam, q


def _image_to_rgb(image: Any) -> np.ndarray:
    raw = np.frombuffer(image.raw_data, dtype=np.uint8)
    raw = raw.reshape(image.height, image.width, 4)
    return raw[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB


def _vehicle_centroid(actor: Any) -> Any:
    tf = actor.get_transform()
    bb = actor.bounding_box
    local = carla.Location(bb.location.x, bb.location.y, bb.location.z)
    return tf.transform(local)


def _match_vehicle_id(world: Any, ego_id: int, wx: float, wy: float, wz: float) -> int:
    target = carla.Location(x=float(wx), y=float(wy), z=float(wz))
    best_id = -1
    best_dist = float("inf")

    for actor in world.get_actors().filter("*vehicle*"):
        if int(actor.id) == ego_id:
            continue
        c = _vehicle_centroid(actor)
        d = float(c.distance(target))
        if d < best_dist:
            best_dist = d
            best_id = int(actor.id)

    if best_dist > 3.0:
        return -1
    return best_id


def _pick_lead(info: Any, cached_half_lane: float) -> dict[str, Any] | None:
    for v in info.nearby_vehicles:
        if not bool(v.get("is_ahead", False)):
            continue
        if not bool(v.get("same_direction", True)) and not bool(v.get("obb_in_lane", False)):
            continue

        obb_flag = bool(v.get("obb_in_lane", False))
        lateral_flag = abs(float(v.get("lateral", 999.0))) < float(cached_half_lane)
        if obb_flag or lateral_flag:
            return cast(dict[str, Any], v)
    return None


def _selected_control(info: Any, expert: Any) -> tuple[str, int]:
    tl_id = int(getattr(info, "traffic_light_actor_id", -1))
    if tl_id > 0:
        return "TL", tl_id

    ss_id = getattr(expert, "_stop_sign_active", None)
    if ss_id is not None:
        return "STOP", int(ss_id)

    ys_id = getattr(expert, "_yield_active", None)
    if ys_id is not None:
        return "YIELD", int(ys_id)

    return "NONE", -1


def _destroy_actors(actors: list[Any]) -> None:
    for actor in actors:
        try:
            if actor is not None and actor.is_alive:
                actor.destroy()
        except RuntimeError:
            continue


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir) / f"triage_{run_stamp}_{args.town}"
    shots_dir = out_root / "screenshots"
    out_root.mkdir(parents=True, exist_ok=True)
    shots_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_root / "frame_dump.csv"

    client = carla.Client(args.host, args.port)
    client.set_timeout(40.0)
    world = client.get_world()
    traffic_manager = None

    original_settings = world.get_settings()
    actors: list[Any] = []

    try:
        print(f"Loading town: {args.town}")
        client.set_timeout(120.0)
        world = client.load_world(args.town)
        client.set_timeout(40.0)
        time.sleep(3.0)

        weather_preset = getattr(carla.WeatherParameters, args.weather, None)
        if weather_preset is not None:
            world.set_weather(weather_preset)

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / float(args.fps)
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)

        carla_map = world.get_map()
        ego, ego_spawn = _spawn_ego(world, rng)
        actors.append(ego)

        npcs = _spawn_npc_vehicles(
            world, traffic_manager.get_port(), ego_spawn, args.n_vehicles, rng
        )
        actors.extend(npcs)

        route_planner = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)
        route = _build_route(carla_map, route_planner, ego_spawn, rng, args.min_route_waypoints)

        expert_cfg = ExpertConfig()
        expert = PrivilegedExpert(ego, world, route_planner, expert_cfg)
        expert.set_route(route)

        cam, cam_q = _attach_camera(world, ego)
        actors.append(cam)

        # Warmup ticks
        for _ in range(20):
            world.tick()
        while True:
            try:
                cam_q.get_nowait()
            except queue.Empty:
                break

        print("Triage replay started")
        print(f"  output: {out_root}")
        print(f"  route waypoints: {len(route)}")
        print(f"  npc vehicles: {len(npcs)}")

        fieldnames = [
            "frame",
            "driving_mode_id",
            "driving_mode_name",
            "current_speed_kmh",
            "target_speed_kmh",
            "selected_control_kind",
            "selected_control_id",
            "tl_actor_id",
            "tl_distance_m",
            "stop_sign_active_id",
            "stop_sign_distance_m",
            "yield_sign_active_id",
            "yield_sign_distance_m",
            "lead_vehicle_id",
            "lead_distance_m",
            "lead_same_direction",
            "lead_obb_in_lane",
            "dyn_emergency_m",
            "sat_detections",
            "lateral_detections",
            "screenshot_file",
        ]

        rows_written = 0
        with csv_path.open("w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()

            dt = 1.0 / float(args.fps)
            for frame in range(args.max_frames):
                world.tick()

                control, info = expert.run_step(dt)
                ego.apply_control(control)

                if expert.done():
                    print(f"Route completed at frame {frame}")
                    break

                latest_img = None
                while True:
                    try:
                        latest_img = cam_q.get_nowait()
                    except queue.Empty:
                        break
                if latest_img is None:
                    try:
                        latest_img = cam_q.get(timeout=1.0)
                    except queue.Empty:
                        latest_img = None

                mode_id = int(info.driving_mode)
                mode_name = _mode_name(mode_id)

                selected_kind, selected_id = _selected_control(info, expert)

                lead = _pick_lead(
                    info, float(getattr(expert, "_cached_half_lane", expert.config.lane_half_width))
                )
                if lead is not None:
                    lead_id = _match_vehicle_id(
                        world,
                        int(ego.id),
                        float(lead.get("world_x", 0.0)),
                        float(lead.get("world_y", 0.0)),
                        float(lead.get("world_z", 0.0)),
                    )
                    lead_distance = float(lead.get("distance", -1.0))
                    lead_same = int(bool(lead.get("same_direction", True)))
                    lead_obb = int(bool(lead.get("obb_in_lane", False)))
                else:
                    lead_id = -1
                    lead_distance = -1.0
                    lead_same = -1
                    lead_obb = -1

                speed_ms = float(info.current_speed) / 3.6
                dyn_emergency = max(
                    float(expert.config.vehicle_emergency_distance),
                    speed_ms * float(expert.config.vehicle_emergency_time),
                )

                screenshot_file = ""
                capture_mode = mode_id in {
                    int(DrivingMode.TRAFFIC_LIGHT_STOP),
                    int(DrivingMode.STOP_SIGN_STOP),
                    int(DrivingMode.YIELD_STOP),
                    int(DrivingMode.EMERGENCY_STOP),
                    int(DrivingMode.VEHICLE_FOLLOWING),
                }
                should_capture = (frame % max(1, args.screenshot_every) == 0) or capture_mode
                if latest_img is not None and should_capture:
                    rgb = _image_to_rgb(latest_img)
                    shot_name = (
                        f"frame_{frame:05d}_mode_{mode_id}_{selected_kind}_{selected_id}.png"
                    )
                    shot_path = shots_dir / shot_name
                    Image.fromarray(rgb).save(shot_path)
                    screenshot_file = str(shot_path.relative_to(out_root))

                row = TriageRow(
                    frame=frame,
                    driving_mode_id=mode_id,
                    driving_mode_name=mode_name,
                    current_speed_kmh=float(info.current_speed),
                    target_speed_kmh=float(info.target_speed),
                    selected_control_kind=selected_kind,
                    selected_control_id=int(selected_id),
                    tl_actor_id=int(getattr(info, "traffic_light_actor_id", -1)),
                    tl_distance_m=float(getattr(info, "traffic_light_distance", -1.0)),
                    stop_sign_active_id=int(getattr(expert, "_stop_sign_active", -1) or -1),
                    stop_sign_distance_m=float(getattr(info, "stop_sign_distance", -1.0)),
                    yield_sign_active_id=int(getattr(expert, "_yield_active", -1) or -1),
                    yield_sign_distance_m=float(getattr(info, "yield_sign_distance", -1.0)),
                    lead_vehicle_id=lead_id,
                    lead_distance_m=lead_distance,
                    lead_same_direction=lead_same,
                    lead_obb_in_lane=lead_obb,
                    dyn_emergency_m=float(dyn_emergency),
                    sat_detections=int(getattr(info, "sat_detections", -1)),
                    lateral_detections=int(getattr(info, "lateral_detections", -1)),
                    screenshot_file=screenshot_file,
                )
                writer.writerow(row.__dict__)
                rows_written += 1

                if frame % 100 == 0:
                    print(
                        f"frame={frame:04d} mode={mode_name:<18} "
                        f"ctrl={selected_kind}:{selected_id:<6} "
                        f"lead={lead_id}:{lead_distance:.2f}m dynE={dyn_emergency:.2f}"
                    )

        print("Triage replay finished")
        print(f"  frame dump: {csv_path}")
        print(f"  screenshots: {shots_dir}")
        print(f"  rows written: {rows_written}")

    finally:
        for actor in actors:
            try:
                if actor is not None and hasattr(actor, "stop"):
                    actor.stop()
            except RuntimeError:
                continue

        # Do not explicitly destroy actors here: some CARLA builds
        # can throw uncaught C++ runtime errors during teardown even
        # after successful replay completion. We rely on process exit
        # and the next load_world() call to reset simulator state.
        if traffic_manager is not None:
            try:
                traffic_manager.set_synchronous_mode(False)
            except RuntimeError:
                pass
        try:
            world.apply_settings(original_settings)
        except RuntimeError:
            pass


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        exit_code = 1
        traceback.print_exc()
    finally:
        # CARLA Python proxies occasionally throw at interpreter teardown
        # after all useful artifacts are already written. Exit explicitly
        # to preserve the real script result code.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
