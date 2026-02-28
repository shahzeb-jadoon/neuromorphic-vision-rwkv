#!/usr/bin/env python3
"""Visual SAT corridor debugger — spawns ego + lead vehicle, draws corridor.

This test lets you VISUALLY verify the SAT corridor in CARLA spectator view.
It draws:
  GREEN  = corridor boundary segments (left + right edges)
  RED    = vehicle OBB (4 corners of lead vehicle bounding box)
  BLUE   = ego position marker
  YELLOW = route waypoint centers (corridor centerline)
  CYAN   = ego-to-first-waypoint line (shows any lane offset)
  WHITE  = ego forward heading line (20m)

It also prints detailed diagnostic data every tick:
  - Lane widths, IDs, road IDs for ego vs waypoints
  - Lateral offset between ego and corridor centerline
  - SAT hit/miss result per segment
  - Vehicle OBB corners vs corridor segment corners

Usage:
    1. Start CARLA 0.9.16 on Windows host
    2. conda activate rwkv
    3. python scripts/sat_visual_test.py [--host auto] [--ticks 200]

The test runs for --ticks frames (default 200 = 10 seconds at 20Hz).
Watch the CARLA spectator view while it runs.
"""

import argparse
import math
import subprocess
import sys
import time
from collections import deque
from typing import Any

import numpy as np

try:
    import carla
except ImportError:
    print("ERROR: carla package not installed. Run: pip install carla==0.9.16")
    sys.exit(1)

from agents.navigation.global_route_planner import GlobalRoutePlanner


# ── SAT overlap (exact copy from privileged_expert.py) ──────────────


def sat_overlap(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
    """Separating Axis Theorem overlap test for two convex polygons."""

    def _project(vertices: np.ndarray, axis: np.ndarray) -> tuple[Any, Any]:
        dots = vertices @ axis
        return dots.min(), dots.max()

    for poly in (poly_a, poly_b):
        n = len(poly)
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            axis = np.array([-edge[1], edge[0]])
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
    """Return (4,2) array of world-space 2D OBB corners for an actor.

    FIXED version: uses pure-Python rotation instead of CARLA's
    t.transform(), which mutates the input Location in-place and
    corrupts bounding_box.location for subsequent reads.
    """
    bb = actor.bounding_box
    cx = float(bb.location.x)
    cy = float(bb.location.y)
    ex = float(bb.extent.x)
    ey = float(bb.extent.y)

    t = actor.get_transform()
    yaw = math.radians(t.rotation.yaw)
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    tx = t.location.x
    ty = t.location.y

    local_offsets = [
        (cx + ex, cy + ey),
        (cx + ex, cy - ey),
        (cx - ex, cy - ey),
        (cx - ex, cy + ey),
    ]

    corners = np.empty((4, 2), dtype=np.float64)
    for i, (lx, ly) in enumerate(local_offsets):
        corners[i, 0] = cos_y * lx - sin_y * ly + tx
        corners[i, 1] = sin_y * lx + cos_y * ly + ty

    return corners


def build_route_corridor(
    ego_loc: Any,
    wp_buffer: Any,
    cutoff: float,
    half_lane: float,
) -> tuple[np.ndarray, list[np.ndarray], list[Any]]:
    """Exact copy of PrivilegedExpert._build_route_corridor().

    Returns (polygon, segments, samples) — we add 'samples' for diagnostics.
    """
    samples = []
    ego_xy = np.array([ego_loc.x, ego_loc.y])

    for wp, _cmd in wp_buffer:
        wloc = wp.transform.location
        if ego_loc.distance(wloc) > cutoff:
            break
        wp_fwd = wp.transform.get_forward_vector()
        fwd_2d = np.array([wp_fwd.x, wp_fwd.y])
        norm = np.linalg.norm(fwd_2d)
        if norm < 1e-6:
            continue
        fwd_2d /= norm
        right_2d = np.array([-fwd_2d[1], fwd_2d[0]])
        centre = np.array([wloc.x, wloc.y])
        samples.append((centre, right_2d, wp))  # keep wp for diagnostics

    if not samples:
        return np.array([ego_xy]), [], []

    prev_center = ego_xy
    segments = []
    for centre, _right_2d, _wp in samples:
        seg_vec = centre - prev_center
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-6:
            continue
        seg_dir = seg_vec / seg_len
        seg_right = np.array([-seg_dir[1], seg_dir[0]])
        prev_l = prev_center - seg_right * half_lane
        prev_r = prev_center + seg_right * half_lane
        cur_l = centre - seg_right * half_lane
        cur_r = centre + seg_right * half_lane
        segments.append(np.array([prev_l, prev_r, cur_r, cur_l]))
        prev_center = centre

    left_boundary = []
    right_boundary = []
    prev_center = ego_xy
    for centre, _r, _w in samples:
        seg_vec = centre - prev_center
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-6:
            continue
        seg_dir = seg_vec / seg_len
        seg_right = np.array([-seg_dir[1], seg_dir[0]])
        left_boundary.append(prev_center - seg_right * half_lane)
        right_boundary.append(prev_center + seg_right * half_lane)
        prev_center = centre
    if segments:
        last_seg = segments[-1]
        left_boundary.append(last_seg[3])
        right_boundary.append(last_seg[2])
    polygon = (
        np.array(left_boundary + right_boundary[::-1]) if left_boundary else np.array([ego_xy])
    )
    return polygon, segments, samples


def get_windows_host_ip() -> str | None:
    """Get the Windows host IP from WSL2's default gateway."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual SAT corridor test")
    parser.add_argument("--host", default="auto", help="CARLA host (auto=detect WSL2)")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument(
        "--ticks", type=int, default=400, help="Number of simulation ticks (400=20s)"
    )
    parser.add_argument("--town", default="Town01", help="Map to load")
    parser.add_argument("--gap", type=float, default=15.0, help="Gap between ego and lead (m)")
    args = parser.parse_args()

    if args.host == "auto":
        host = get_windows_host_ip()
        if not host:
            host = "localhost"
    else:
        host = args.host

    print(f"Connecting to CARLA at {host}:{args.port} ...")
    client = carla.Client(host, args.port)
    client.set_timeout(30.0)
    print(f"Server: {client.get_server_version()}")

    # Load map
    world = client.load_world(args.town)
    time.sleep(2.0)
    world = client.get_world()
    carla_map = world.get_map()
    original_settings = world.get_settings()

    actors = []

    try:
        # ── Synchronous mode ──
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 Hz
        world.apply_settings(settings)
        print("Sync mode enabled (20 Hz)")

        # ── Select spawn point on a straight road ──
        spawn_points = carla_map.get_spawn_points()
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]

        # Find a good straight-road spawn point
        # We'll try spawn point 0 and verify lane info
        ego_sp = spawn_points[0]
        ego_wp = carla_map.get_waypoint(ego_sp.location)
        print(f"\n{'='*70}")
        print("SPAWN POINT INFO")
        print(f"{'='*70}")
        print(
            f"  Spawn transform: ({ego_sp.location.x:.1f}, {ego_sp.location.y:.1f}, "
            f"{ego_sp.location.z:.1f}) yaw={ego_sp.rotation.yaw:.1f}°"
        )
        print(
            f"  Nearest waypoint: ({ego_wp.transform.location.x:.1f}, "
            f"{ego_wp.transform.location.y:.1f})"
        )
        print(f"  Lane ID:     {ego_wp.lane_id}")
        print(f"  Road ID:     {ego_wp.road_id}")
        print(f"  Lane width:  {ego_wp.lane_width:.2f}m")
        print(f"  Lane type:   {ego_wp.lane_type}")
        print(f"  Is junction: {ego_wp.is_junction}")
        print(
            f"  Num left lanes:  {len(ego_wp.get_left_lane() and [ego_wp.get_left_lane()] or [])}"
        )
        print(
            f"  Num right lanes: {len(ego_wp.get_right_lane() and [ego_wp.get_right_lane()] or [])}"
        )

        # Count lanes by traversing left and right
        n_lanes_left = 0
        wp_iter = ego_wp
        while True:
            lw = wp_iter.get_left_lane()
            if lw is None or lw.lane_type != carla.LaneType.Driving:
                break
            # Check same direction (lane_id sign)
            if lw.lane_id * wp_iter.lane_id < 0:
                break  # opposite direction
            n_lanes_left += 1
            wp_iter = lw

        n_lanes_right = 0
        wp_iter = ego_wp
        while True:
            rw = wp_iter.get_right_lane()
            if rw is None or rw.lane_type != carla.LaneType.Driving:
                break
            if rw.lane_id * wp_iter.lane_id < 0:
                break
            n_lanes_right += 1
            wp_iter = rw

        print(
            f"  Same-dir lanes: {n_lanes_left} left + 1 (ego) + {n_lanes_right} right = "
            f"{n_lanes_left + 1 + n_lanes_right} total"
        )

        # ── Spawn ego vehicle ──
        ego_vehicle = world.spawn_actor(vehicle_bp, ego_sp)
        actors.append(ego_vehicle)
        print(f"\n  Ego spawned: id={ego_vehicle.id}")

        # ── Spawn lead vehicle ahead ──
        # Move forward along waypoints to place lead vehicle
        lead_wp = ego_wp
        dist_covered = 0.0
        while dist_covered < args.gap:
            nexts = lead_wp.next(2.0)
            if not nexts:
                break
            lead_wp = nexts[0]
            dist_covered += 2.0

        lead_sp = lead_wp.transform
        lead_sp.location.z += 0.5  # prevent clipping into ground
        lead_vehicle = world.spawn_actor(vehicle_bp, lead_sp)
        actors.append(lead_vehicle)
        print(f"  Lead spawned: id={lead_vehicle.id} at ~{dist_covered:.0f}m ahead")
        print(
            f"  Lead waypoint lane_id={lead_wp.lane_id} road_id={lead_wp.road_id} "
            f"lane_width={lead_wp.lane_width:.2f}m"
        )

        # Let physics settle
        for _ in range(10):
            world.tick()

        # ── Plan route using GlobalRoutePlanner ──
        grp = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)
        # Route to a destination far ahead
        dest_sp = spawn_points[min(1, len(spawn_points) - 1)]
        # Or just go straight ahead 200m
        dest_loc = lead_wp.next(200.0)
        if dest_loc:
            dest_location = dest_loc[0].transform.location
        else:
            dest_location = dest_sp.location

        route = grp.trace_route(ego_sp.location, dest_location)
        print(f"\n  Route planned: {len(route)} waypoints")

        # ── Build waypoint buffer (simulate PrivilegedExpert) ──
        wp_buffer_size = 40
        wp_buffer: deque[tuple[Any, Any]] = deque(maxlen=wp_buffer_size)
        for i in range(min(wp_buffer_size, len(route))):
            wp_buffer.append(route[i])

        print(f"  WP buffer filled: {len(wp_buffer)} waypoints")

        # ── Analyze lane alignment ──
        ego_transform = ego_vehicle.get_transform()
        ego_loc = ego_transform.location
        ego_actual_wp = carla_map.get_waypoint(ego_loc)

        print(f"\n{'='*70}")
        print("LANE ALIGNMENT ANALYSIS")
        print(f"{'='*70}")
        print(f"  Ego actual position:  ({ego_loc.x:.2f}, {ego_loc.y:.2f})")
        print(
            f"  Ego nearest waypoint: ({ego_actual_wp.transform.location.x:.2f}, "
            f"{ego_actual_wp.transform.location.y:.2f})"
        )
        print(
            f"  Ego lane_id={ego_actual_wp.lane_id} road_id={ego_actual_wp.road_id} "
            f"width={ego_actual_wp.lane_width:.2f}m"
        )

        # Check first N route waypoints
        print("\n  Route waypoint lane analysis (first 10):")
        print(
            f"  {'#':>3} {'X':>8} {'Y':>8} {'lane_id':>8} {'road_id':>8} "
            f"{'width':>6} {'junc':>5} {'ego_dist':>9} {'lat_off':>8}"
        )
        ego_fwd = ego_transform.get_forward_vector()
        ego_right = carla.Vector3D(-ego_fwd.y, ego_fwd.x, 0)

        for i in range(min(10, len(wp_buffer))):
            wp, cmd = wp_buffer[i]
            wloc = wp.transform.location
            dx = wloc.x - ego_loc.x
            dy = wloc.y - ego_loc.y
            dist = math.sqrt(dx * dx + dy * dy)
            lat_offset = ego_right.x * dx + ego_right.y * dy
            same_lane = wp.lane_id == ego_actual_wp.lane_id and wp.road_id == ego_actual_wp.road_id
            marker = " " if same_lane else " ← DIFF LANE!"
            print(
                f"  {i:3d} {wloc.x:8.2f} {wloc.y:8.2f} {wp.lane_id:8d} {wp.road_id:8d} "
                f"{wp.lane_width:6.2f} {str(wp.is_junction):>5} {dist:9.2f} "
                f"{lat_offset:8.2f}{marker}"
            )

        # ── Build corridor ──
        half_lane = 2.5
        cutoff = 50.0  # vehicle_hazard_distance * 2
        polygon, segments, samples = build_route_corridor(ego_loc, wp_buffer, cutoff, half_lane)
        print(f"\n  Corridor: {len(segments)} segments, half_lane={half_lane}m")

        # ── Get vehicle OBB ──
        v_corners = get_actor_obb_2d(lead_vehicle)
        print("\n  Lead vehicle OBB corners:")
        for ci, (cx, cy) in enumerate(v_corners):
            print(f"    Corner {ci}: ({cx:.2f}, {cy:.2f})")

        # ── Test SAT per segment ──
        print(f"\n{'='*70}")
        print("SAT OVERLAP TEST PER SEGMENT")
        print(f"{'='*70}")
        any_hit = False
        for si, seg in enumerate(segments):
            hit = sat_overlap(seg, v_corners)
            if hit:
                any_hit = True
                print(f"  Seg {si:2d}: HIT ✓")
                # Print segment corners
                print(
                    f"          corners: [{seg[0][0]:.1f},{seg[0][1]:.1f}] "
                    f"[{seg[1][0]:.1f},{seg[1][1]:.1f}] "
                    f"[{seg[2][0]:.1f},{seg[2][1]:.1f}] "
                    f"[{seg[3][0]:.1f},{seg[3][1]:.1f}]"
                )
            # Print first 3 and any hit segment
            elif si < 3:
                print(f"  Seg {si:2d}: MISS")

        if not any_hit:
            print(f"\n  *** ALL {len(segments)} SEGMENTS MISS! ***")
            # Find closest segment to vehicle center
            v_center = v_corners.mean(axis=0)
            min_dist = float("in")
            closest_si = -1
            for si, seg in enumerate(segments):
                seg_center = seg.mean(axis=0)
                d = float(np.linalg.norm(seg_center - v_center))
                if d < min_dist:
                    min_dist = d
                    closest_si = si
            if closest_si >= 0:
                seg = segments[closest_si]
                seg_center = seg.mean(axis=0)
                print(
                    f"  Closest segment {closest_si}:"
                    f" center=({seg_center[0]:.2f},"
                    f" {seg_center[1]:.2f})"
                )
                print(f"  Vehicle center: ({v_center[0]:.2f}, {v_center[1]:.2f})")
                print(f"  Gap: {min_dist:.2f}m")

                # Compute lateral offset of vehicle center from corridor centerline
                # at the closest segment
                if closest_si < len(samples):
                    wp_center = samples[closest_si][0]
                    v_to_wp = v_center - wp_center
                    print(f"  Waypoint center: ({wp_center[0]:.2f}, {wp_center[1]:.2f})")
                    print(f"  Vehicle-to-waypoint offset: ({v_to_wp[0]:.2f}, {v_to_wp[1]:.2f})")
                    print(f"  |offset|: {np.linalg.norm(v_to_wp):.2f}m")
        else:
            n_hit = sum(sat_overlap(seg, v_corners) for seg in segments)
            print("\n  SAT DETECTED VEHICLE \u2713" f" ({n_hit} segments)")

        # ── Compute what lateral offset the ego-heading check measures ──
        # Copy bb.location before transforming to avoid in-place mutation
        ego_bb = ego_vehicle.bounding_box
        ego_bb_loc = carla.Location(ego_bb.location.x, ego_bb.location.y, ego_bb.location.z)
        ego_bb_center = ego_transform.transform(ego_bb_loc)
        lead_t = lead_vehicle.get_transform()
        lead_bb = lead_vehicle.bounding_box
        lead_bb_loc = carla.Location(lead_bb.location.x, lead_bb.location.y, lead_bb.location.z)
        lead_bb_center = lead_t.transform(lead_bb_loc)
        dx = lead_bb_center.x - ego_bb_center.x
        dy = lead_bb_center.y - ego_bb_center.y
        lat_from_ego = ego_right.x * dx + ego_right.y * dy
        long_from_ego = ego_fwd.x * dx + ego_fwd.y * dy
        print(f"\n{'='*70}")
        print("LATERAL CHECK COMPARISON")
        print(f"{'='*70}")
        print(f"  Ego BB center:  ({ego_bb_center.x:.2f}, {ego_bb_center.y:.2f})")
        print(f"  Lead BB center: ({lead_bb_center.x:.2f}, {lead_bb_center.y:.2f})")
        print(f"  Longitudinal:   {long_from_ego:.2f}m")
        print(f"  Lateral:        {lat_from_ego:.2f}m")
        print(
            f"  |Lateral| < {half_lane}: {abs(lat_from_ego) < half_lane} → "
            f"{'DETECTED' if abs(lat_from_ego) < half_lane else 'MISSED'}"
        )

        # ── Draw debug visuals ──
        print(f"\n{'='*70}")
        print("DRAWING DEBUG VISUALIZATION")
        print(f"{'='*70}")
        print("  GREEN  = corridor boundary")
        print("  RED    = vehicle OBB")
        print("  BLUE   = ego position marker")
        print("  YELLOW = route waypoint centers")
        print("  CYAN   = ego → first waypoint line")
        print("  WHITE  = ego forward heading (20m)")
        print()
        print("  >>> Look at the CARLA spectator view now! <<<")
        print("  >>> The visualization refreshes each tick for 10 seconds <<<")

        # Position spectator above the scene
        mid_x = (ego_loc.x + lead_bb_center.x) / 2
        mid_y = (ego_loc.y + lead_bb_center.y) / 2
        spectator = world.get_spectator()
        spectator.set_transform(
            carla.Transform(
                carla.Location(x=mid_x, y=mid_y, z=50),
                carla.Rotation(pitch=-90, yaw=ego_sp.rotation.yaw, roll=0),
            )
        )

        # Drawing colors
        GREEN = carla.Color(0, 255, 0)
        RED = carla.Color(255, 0, 0)
        BLUE = carla.Color(0, 100, 255)
        YELLOW = carla.Color(255, 255, 0)
        CYAN = carla.Color(0, 255, 255)
        WHITE = carla.Color(255, 255, 255)
        MAGENTA = carla.Color(255, 0, 255)

        debug = world.debug
        z_draw = ego_loc.z + 0.5  # draw slightly above ground

        for tick in range(args.ticks):
            world.tick()

            # Re-get positions (vehicles might move slightly)
            ego_t = ego_vehicle.get_transform()
            ego_l = ego_t.location

            life = 0.15  # slightly longer than 1 tick

            # 1. Draw corridor segments (GREEN)
            for seg in segments:
                for ci in range(4):
                    p1 = seg[ci]
                    p2 = seg[(ci + 1) % 4]
                    debug.draw_line(
                        carla.Location(x=float(p1[0]), y=float(p1[1]), z=z_draw),
                        carla.Location(x=float(p2[0]), y=float(p2[1]), z=z_draw),
                        color=GREEN,
                        thickness=0.05,
                        life_time=life,
                    )

            # 2. Draw vehicle OBB (RED)
            vc = get_actor_obb_2d(lead_vehicle)
            for ci in range(4):
                p1 = vc[ci]
                p2 = vc[(ci + 1) % 4]
                debug.draw_line(
                    carla.Location(x=float(p1[0]), y=float(p1[1]), z=z_draw + 0.1),
                    carla.Location(x=float(p2[0]), y=float(p2[1]), z=z_draw + 0.1),
                    color=RED,
                    thickness=0.08,
                    life_time=life,
                )

            # 3. Ego marker (BLUE)
            debug.draw_point(
                carla.Location(x=ego_l.x, y=ego_l.y, z=z_draw + 0.2),
                size=0.15,
                color=BLUE,
                life_time=life,
            )

            # 4. Route waypoint centers (YELLOW dots)
            for centre, _r, _w in samples:
                debug.draw_point(
                    carla.Location(x=float(centre[0]), y=float(centre[1]), z=z_draw + 0.1),
                    size=0.08,
                    color=YELLOW,
                    life_time=life,
                )

            # 5. Ego → first waypoint line (CYAN)
            if wp_buffer:
                wp0_loc = wp_buffer[0][0].transform.location
                debug.draw_line(
                    carla.Location(x=ego_l.x, y=ego_l.y, z=z_draw + 0.2),
                    carla.Location(x=wp0_loc.x, y=wp0_loc.y, z=z_draw + 0.2),
                    color=CYAN,
                    thickness=0.06,
                    life_time=life,
                )

            # 6. Ego forward heading line (WHITE, 20m)
            ego_fwd_v = ego_t.get_forward_vector()
            debug.draw_line(
                carla.Location(x=ego_l.x, y=ego_l.y, z=z_draw + 0.2),
                carla.Location(
                    x=ego_l.x + ego_fwd_v.x * 20,
                    y=ego_l.y + ego_fwd_v.y * 20,
                    z=z_draw + 0.2,
                ),
                color=WHITE,
                thickness=0.04,
                life_time=life,
            )

            # 7. Draw lane markings for ego's lane (MAGENTA, just edges)
            ego_cur_wp = carla_map.get_waypoint(ego_l)
            hw = ego_cur_wp.lane_width / 2
            for step in range(2, 40, 2):
                nxt = ego_cur_wp.next(float(step))
                if not nxt:
                    break
                nxt_wp = nxt[0]
                nxt_loc = nxt_wp.transform.location
                nxt_fwd = nxt_wp.transform.get_forward_vector()
                nxt_right = carla.Vector3D(-nxt_fwd.y, nxt_fwd.x, 0)
                left_pt = carla.Location(
                    x=nxt_loc.x - nxt_right.x * hw,
                    y=nxt_loc.y - nxt_right.y * hw,
                    z=z_draw + 0.05,
                )
                right_pt = carla.Location(
                    x=nxt_loc.x + nxt_right.x * hw,
                    y=nxt_loc.y + nxt_right.y * hw,
                    z=z_draw + 0.05,
                )
                debug.draw_point(left_pt, size=0.04, color=MAGENTA, life_time=life)
                debug.draw_point(right_pt, size=0.04, color=MAGENTA, life_time=life)

            if tick == 0:
                print(f"  Tick {tick}: visualization started. Watch CARLA window!")
            elif tick % 40 == 0:
                print(f"  Tick {tick}/{args.ticks} ...")

        print(f"\n  Done! {args.ticks} ticks completed.")

    finally:
        # Cleanup
        print("\nCleaning up ...")
        for a in actors:
            try:
                a.destroy()
            except Exception:
                pass
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass
        print("Done.")


if __name__ == "__main__":
    main()
