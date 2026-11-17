#!/usr/bin/env python3
"""Visualize episode traffic lights and stop order in live CARLA.

Draws a magenta box around selected traffic lights and labels each as
``TL<episode_id>``. Also draws stop markers (``Stop 1``, ``Stop 2``, ...)
at the ego's first near-stop frame for each TL in the episode.

The drawing refreshes continuously, so you can freely move the spectator
camera in the CARLA UE window using mouse + WASD/QE controls.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np

try:
    import carla
except ImportError as exc:  # pragma: no cover
    raise SystemExit("CARLA Python API is required. Activate your CARLA env first.") from exc


MAGENTA = carla.Color(255, 0, 255)
WHITE = carla.Color(255, 255, 255)
CYAN = carla.Color(0, 255, 255)
LIME = carla.Color(120, 255, 120)
GREEN = carla.Color(0, 255, 0)
ORANGE = carla.Color(255, 165, 0)
SKY = carla.Color(0, 170, 255)
YELLOW = carla.Color(255, 255, 0)


TOWN_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "Town10HD": {
        "episode": "data/test_collection/episode_cf183fc1.h5",
        "tl_ids": [2778, 2789, 2779, 2782],
    },
    "Town06": {
        "episode": "data/test_collection/episode_51be5089.h5",
        "tl_ids": [3114, 3127],
    },
}


@dataclass
class EpisodeTLStop:
    requested_tl_id: int
    first_stop_frame: int | None
    raw_first_stop_frame: int | None
    stop_order: int | None
    stop_position: tuple[float, float, float] | None
    stop_line_position: tuple[float, float, float] | None


@dataclass
class BoundTL:
    stop: EpisodeTLStop
    actor: Any
    bind_mode: str
    bind_distance: float
    zone_center: Any | None
    zone_rotation: Any | None
    zone_extent: Any | None
    zone_source: str


@dataclass
class LiveTLHighlight:
    live_id: int
    actor: Any
    zone_center: Any | None
    zone_rotation: Any | None
    zone_extent: Any | None
    zone_source: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw TL stop-zone overlays for a recorded episode in live CARLA."
    )
    parser.add_argument("--host", default="172.20.0.1", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--town", default="Town10HD", help="Expected town name fragment")
    parser.add_argument(
        "--episode",
        default=None,
        help="Episode HDF5 path (optional if town profile default exists)",
    )
    parser.add_argument(
        "--tl-ids",
        type=int,
        nargs="+",
        default=None,
        help="Episode traffic light IDs to visualize (optional if town profile default exists)",
    )
    parser.add_argument(
        "--stop-speed-kmh",
        type=float,
        default=0.8,
        help="Near-stop speed threshold (km/h) for stop-order extraction",
    )
    parser.add_argument(
        "--life-time",
        type=float,
        default=0.65,
        help="Debug primitive lifetime in seconds",
    )
    parser.add_argument(
        "--refresh-sec",
        type=float,
        default=0.20,
        help="Overlay refresh period in seconds",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=0.0,
        help="Run duration in seconds (0 means run until Ctrl+C)",
    )
    parser.add_argument(
        "--set-initial-view",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set spectator to a top-down initial overview",
    )
    parser.add_argument(
        "--auto-load-town",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically load --town if the current CARLA world differs",
    )
    parser.add_argument(
        "--hud-side",
        choices=["left", "right"],
        default="left",
        help="Side for the live location bar in the spectator view",
    )
    parser.add_argument(
        "--compass-side",
        choices=["left", "right", "opposite"],
        default="opposite",
        help="Deprecated and ignored (compass overlay is disabled)",
    )
    parser.add_argument(
        "--highlight-live-ids",
        type=int,
        nargs="*",
        default=[],
        help="Extra live traffic-light actor IDs to draw in green",
    )
    parser.add_argument(
        "--allow-initial-hold-stop",
        action="store_true",
        help="Use earliest near-stop even if no prior motion happened for that TL",
    )
    parser.add_argument(
        "--min-prior-speed-kmh",
        type=float,
        default=2.0,
        help="Minimum earlier speed (same TL) required to qualify as a true stop-after-motion",
    )
    parser.add_argument(
        "--width-overlays",
        type=float,
        nargs="*",
        default=[],
        help=(
            "Additional width-scale overlays for stop zones; example: --width-overlays 0.75 0.65"
        ),
    )
    parser.add_argument(
        "--show-no-stop-tls",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Include TLs with no near-stop frame in this episode (these display as F?)"),
    )
    return parser.parse_args()


def _normalize_town_label(name: str) -> str:
    up = name.upper()
    if "TOWN10HD" in up:
        return "Town10HD"
    if "TOWN06" in up:
        return "Town06"
    return name


def _resolve_profile_defaults(town: str) -> tuple[str | None, list[int] | None]:
    profile = TOWN_PROFILE_DEFAULTS.get(_normalize_town_label(town))
    if profile is None:
        return None, None
    return str(profile["episode"]), list(profile["tl_ids"])


def _resolve_episode_and_tl_ids(
    args: argparse.Namespace,
) -> tuple[Path, list[int], bool, bool]:
    episode_arg = args.episode
    tl_ids = list(args.tl_ids) if args.tl_ids is not None else None

    auto_episode = False
    auto_tl_ids = False

    prof_episode, prof_tl_ids = _resolve_profile_defaults(args.town)
    if episode_arg is None and prof_episode is not None:
        episode_arg = prof_episode
        auto_episode = True
    if (tl_ids is None or len(tl_ids) == 0) and prof_tl_ids is not None:
        tl_ids = list(prof_tl_ids)
        auto_tl_ids = True

    if episode_arg is None:
        raise SystemExit(
            "No --episode provided and no default profile for this town. Pass --episode explicitly."
        )
    if tl_ids is None or len(tl_ids) == 0:
        raise SystemExit(
            "No --tl-ids provided and no default profile for this town. Pass --tl-ids explicitly."
        )

    episode_path = _resolve_episode_path(episode_arg)
    return episode_path, [int(v) for v in tl_ids], auto_episode, auto_tl_ids


def _resolve_episode_path(episode_arg: str) -> Path:
    """Resolve episode path with friendly fallbacks.

    Accepts either a full/relative path or a bare filename.
    """
    raw = Path(episode_arg)
    if raw.exists():
        return raw

    candidates = [
        Path("data/test_collection") / episode_arg,
        Path("data/collected") / episode_arg,
    ]
    for cand in candidates:
        if cand.exists():
            return cand

    names = [str(raw)] + [str(c) for c in candidates]
    joined = "\n  - ".join(names)
    raise SystemExit(
        "Episode file not found. Tried:\n"
        f"  - {joined}\n"
        "Tip: pass --episode data/test_collection/episode_cf183fc1.h5"
    )


def _valid_xyz(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.zeros(arr.shape[0], dtype=bool)
    return cast(
        np.ndarray,
        ~np.all(np.isclose(arr[:, :3], np.array([-1.0, -1.0, -1.0])), axis=1),
    )


def _median_stop_line(
    tl_ids: np.ndarray,
    tl_states: np.ndarray,
    stop_lines: np.ndarray,
    target_tl_id: int,
) -> tuple[float, float, float] | None:
    primary = (tl_ids == target_tl_id) & (tl_states >= 2)
    valid = primary & _valid_xyz(stop_lines)
    if not np.any(valid):
        fallback = (tl_ids == target_tl_id) & _valid_xyz(stop_lines)
        valid = fallback
    if not np.any(valid):
        return None
    pts = stop_lines[valid, :3]
    med = np.median(pts, axis=0)
    return float(med[0]), float(med[1]), float(med[2])


def _extract_episode_stops(
    episode_path: Path,
    tl_ids: list[int],
    stop_speed_kmh: float,
    allow_initial_hold_stop: bool,
    min_prior_speed_kmh: float,
) -> list[EpisodeTLStop]:
    with h5py.File(episode_path, "r") as h:
        ep_tl_id = h["traffic_light_actor_id"][:]
        ep_tl_state = h["traffic_light_state"][:]
        ep_vel = h["velocity"][:, :3]
        ep_pos = h["position"][:, :3]
        ep_stop = h["traffic_light_stop_line_position"][:, :3]

    speed_kmh = np.linalg.norm(ep_vel, axis=1) * 3.6
    stops: list[EpisodeTLStop] = []
    for tl_id in tl_ids:
        mask = (ep_tl_id == tl_id) & (ep_tl_state >= 2) & (speed_kmh <= stop_speed_kmh)
        stop_frames = np.where(mask)[0]
        raw_first_stop_frame = int(stop_frames[0]) if stop_frames.size > 0 else None

        first_stop_frame = raw_first_stop_frame
        if stop_frames.size > 0 and not allow_initial_hold_stop:
            first_stop_frame = None
            for sf in stop_frames:
                # For a true stop event, we require that this TL had
                # prior motion first; this avoids classifying spawn hold
                # as the stop point when the run starts under red.
                prior_motion = np.any(
                    (ep_tl_id[:sf] == tl_id) & (speed_kmh[:sf] > min_prior_speed_kmh)
                )
                if prior_motion:
                    first_stop_frame = int(sf)
                    break
            if first_stop_frame is None:
                first_stop_frame = raw_first_stop_frame

        stop_position = None
        if first_stop_frame is not None:
            p = ep_pos[first_stop_frame]
            stop_position = (float(p[0]), float(p[1]), float(p[2]))

        stop_line_position = _median_stop_line(ep_tl_id, ep_tl_state, ep_stop, tl_id)

        stops.append(
            EpisodeTLStop(
                requested_tl_id=tl_id,
                first_stop_frame=first_stop_frame,
                raw_first_stop_frame=raw_first_stop_frame,
                stop_order=None,
                stop_position=stop_position,
                stop_line_position=stop_line_position,
            )
        )

    ordered_with_frame = [s for s in stops if s.first_stop_frame is not None]
    ordered = sorted(
        ordered_with_frame,
        key=lambda s: cast(int, s.first_stop_frame),
    )
    for idx, stop in enumerate(ordered, start=1):
        stop.stop_order = idx

    return stops


def _resolve_live_tl_actor(
    world: Any,
    requested_tl_id: int,
    stop_line_position: tuple[float, float, float] | None,
) -> tuple[Any | None, str, float]:
    direct = world.get_actor(requested_tl_id)
    if direct is not None and "traffic_light" in direct.type_id:
        return direct, "direct", 0.0

    if stop_line_position is None:
        return None, "unresolved", math.inf

    lights = list(world.get_actors().filter("*traffic_light*"))
    if not lights:
        return None, "unresolved", math.inf

    sx, sy, sz = stop_line_position
    target = carla.Location(x=sx, y=sy, z=sz)

    def _dist(loc: carla.Location) -> float:
        return float(target.distance(loc))

    best_tl = None
    best_dist = math.inf
    best_mode = "unresolved"

    # Primary match: closest stop-waypoint location (most semantically
    # aligned with where vehicles stop for the light).
    for tl in lights:
        try:
            swps = list(tl.get_stop_waypoints())
        except (AttributeError, RuntimeError):
            swps = []
        if not swps:
            continue
        d = min(_dist(wp.transform.location) for wp in swps)
        if d < best_dist:
            best_tl = tl
            best_dist = d
            best_mode = "nearest-stopwp"

    # Secondary: affected lane waypoints.
    if best_tl is None:
        for tl in lights:
            try:
                awps = list(tl.get_affected_lane_waypoints())
            except (AttributeError, RuntimeError):
                awps = []
            if not awps:
                continue
            d = min(_dist(wp.transform.location) for wp in awps)
            if d < best_dist:
                best_tl = tl
                best_dist = d
                best_mode = "nearest-affectedwp"

    # Final fallback: bounding box center (often near stop trigger), then pole.
    if best_tl is None:
        for tl in lights:
            bb_center = tl.get_transform().transform(tl.bounding_box.location)
            d = _dist(bb_center)
            if d < best_dist:
                best_tl = tl
                best_dist = d
                best_mode = "nearest-bbox"

    if best_tl is None:
        nearest = min(lights, key=lambda tl: tl.get_transform().location.distance(target))
        return nearest, "nearest-pole", float(nearest.get_transform().location.distance(target))

    return best_tl, best_mode, float(best_dist)


def _circular_mean_deg(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sum(math.sin(math.radians(v)) for v in values)
    c = sum(math.cos(math.radians(v)) for v in values)
    return math.degrees(math.atan2(s, c))


def _collect_tl_waypoints(tl: Any) -> tuple[list[Any], str]:
    try:
        swps = list(tl.get_stop_waypoints())
    except (AttributeError, RuntimeError):
        swps = []
    if swps:
        return swps, "stopwp"

    try:
        awps = list(tl.get_affected_lane_waypoints())
    except (AttributeError, RuntimeError):
        awps = []
    if awps:
        return awps, "affectedwp"

    return [], "bbox"


def _compute_tl_stop_zone(tl: Any) -> tuple[Any | None, Any | None, Any | None, str]:
    """Return lane-local stop zone geometry for drawing.

    The zone is aligned to stop-waypoint lanes and spans full lane coverage
    across the controlled approach.
    """
    wps, source = _collect_tl_waypoints(tl)
    if not wps:
        return None, None, None, source

    locs = [wp.transform.location for wp in wps]
    yaws = [float(wp.transform.rotation.yaw) for wp in wps]
    lane_widths = [float(getattr(wp, "lane_width", 3.5)) for wp in wps]

    center = carla.Location(
        x=float(np.mean([loc.x for loc in locs])),
        y=float(np.mean([loc.y for loc in locs])),
        z=float(np.mean([loc.z for loc in locs])) + 0.85,
    )

    yaw = _circular_mean_deg(yaws)
    lane_w = max(2.6, float(np.mean(lane_widths)) if lane_widths else 3.5)

    lane_span = 0.0
    for i in range(len(locs)):
        for j in range(i + 1, len(locs)):
            lane_span = max(lane_span, locs[i].distance(locs[j]))

    if len(locs) == 1:
        total_width = lane_w
    else:
        # Full lane coverage: center-to-center span plus half-lane padding
        # on both sides.
        total_width = lane_span + lane_w

    depth = max(1.2, min(2.2, lane_w * 0.6))
    extent = carla.Vector3D(
        x=float(depth * 0.5),
        y=float(total_width * 0.5),
        z=1.0,
    )
    rot = carla.Rotation(pitch=0.0, yaw=float(yaw), roll=0.0)
    return center, rot, extent, source


def _bound_tl_list(world: Any, stops: list[EpisodeTLStop]) -> list[BoundTL]:
    out: list[BoundTL] = []
    for stop in stops:
        actor, mode, dist = _resolve_live_tl_actor(
            world, stop.requested_tl_id, stop.stop_line_position
        )
        if actor is None:
            print(f"[warn] TL{stop.requested_tl_id}: could not bind to live traffic light")
            continue
        zone_center, zone_rot, zone_extent, zone_source = _compute_tl_stop_zone(actor)
        out.append(
            BoundTL(
                stop=stop,
                actor=actor,
                bind_mode=mode,
                bind_distance=dist,
                zone_center=zone_center,
                zone_rotation=zone_rot,
                zone_extent=zone_extent,
                zone_source=zone_source,
            )
        )
    return out


def _collect_live_highlights(world: Any, live_ids: list[int]) -> list[LiveTLHighlight]:
    out: list[LiveTLHighlight] = []
    for lid in live_ids:
        actor = world.get_actor(lid)
        if actor is None:
            print(f"[warn] live_id:{lid} not found in current world")
            continue
        if "traffic_light" not in actor.type_id:
            print(f"[warn] live_id:{lid} exists but is not a traffic light ({actor.type_id})")
            continue
        zone_center, zone_rot, zone_extent, zone_source = _compute_tl_stop_zone(actor)
        out.append(
            LiveTLHighlight(
                live_id=lid,
                actor=actor,
                zone_center=zone_center,
                zone_rotation=zone_rot,
                zone_extent=zone_extent,
                zone_source=zone_source,
            )
        )
    return out


def _set_initial_spectator_view(world: Any, bound_tls: list[BoundTL]) -> None:
    points: list[tuple[float, float, float]] = []
    for item in bound_tls:
        loc = item.actor.get_transform().location
        points.append((float(loc.x), float(loc.y), float(loc.z)))
        if item.stop.stop_position is not None:
            points.append(item.stop.stop_position)
    if not points:
        return

    arr = np.array(points, dtype=np.float64)
    cx = float(np.mean(arr[:, 0]))
    cy = float(np.mean(arr[:, 1]))
    min_x, max_x = float(np.min(arr[:, 0])), float(np.max(arr[:, 0]))
    min_y, max_y = float(np.min(arr[:, 1])), float(np.max(arr[:, 1]))
    span = max(max_x - min_x, max_y - min_y)
    z = max(35.0, min(120.0, span * 1.4 + 20.0))

    spectator = world.get_spectator()
    spectator.set_transform(
        carla.Transform(
            carla.Location(x=cx, y=cy, z=z),
            carla.Rotation(pitch=-88.0, yaw=0.0, roll=0.0),
        )
    )


def _yaw_to_heading_label(yaw_deg: float) -> str:
    yawr = math.radians(yaw_deg)
    fx = math.cos(yawr)
    fy = math.sin(yawr)
    if abs(fx) >= abs(fy):
        return "eastbound" if fx >= 0.0 else "westbound"
    # CARLA Town axes: +Y points south in this setup.
    return "southbound" if fy >= 0.0 else "northbound"


def _offset_location(
    origin: carla.Location,
    forward: carla.Vector3D,
    right: carla.Vector3D,
    up: carla.Vector3D,
    forward_m: float = 0.0,
    right_m: float = 0.0,
    up_m: float = 0.0,
) -> carla.Location:
    return carla.Location(
        x=origin.x + forward.x * forward_m + right.x * right_m + up.x * up_m,
        y=origin.y + forward.y * forward_m + right.y * right_m + up.y * up_m,
        z=origin.z + forward.z * forward_m + right.z * right_m + up.z * up_m,
    )


def _traffic_light_state_name(tl: Any) -> str:
    state = getattr(tl, "state", None)
    mapping = {
        carla.TrafficLightState.Red: "RED",
        carla.TrafficLightState.Yellow: "YELLOW",
        carla.TrafficLightState.Green: "GREEN",
        carla.TrafficLightState.Off: "OFF",
        carla.TrafficLightState.Unknown: "UNKNOWN",
    }
    if state in mapping:
        return mapping[state]
    if state is None:
        return "?"
    return str(state).split(".")[-1].upper()


def _draw_navigation_hud(
    world: Any,
    hud_side: str,
    life_time: float,
) -> None:
    debug = world.debug
    spectator = world.get_spectator()
    tf = spectator.get_transform()
    loc = tf.location
    yaw = (float(tf.rotation.yaw) + 360.0) % 360.0
    heading = _yaw_to_heading_label(yaw)

    forward = tf.get_forward_vector()
    right = tf.get_right_vector()
    up = tf.get_up_vector()

    hud_forward = 14.0
    hud_up = 3.4
    hud_side_offset = 8.0
    line_step = 0.55

    # --- Location bar ---
    hud_sign = -1.0 if hud_side == "left" else 1.0
    hud_anchor = _offset_location(
        loc,
        forward,
        right,
        up,
        forward_m=hud_forward,
        right_m=hud_sign * hud_side_offset,
        up_m=hud_up,
    )

    hud_lines = [
        "Location Bar",
        f"x={loc.x:.2f}  y={loc.y:.2f}",
        f"yaw={yaw:.1f} deg",
        f"heading={heading}",
    ]
    for i, text in enumerate(hud_lines):
        line_loc = _offset_location(
            hud_anchor,
            forward,
            right,
            up,
            up_m=-(i * line_step),
        )
        debug.draw_string(
            line_loc,
            text,
            draw_shadow=True,
            color=WHITE,
            life_time=life_time,
            persistent_lines=False,
        )


def _overlay_color_for_index(idx: int) -> Any:
    palette = [ORANGE, SKY, WHITE]
    return palette[idx] if idx < len(palette) else WHITE


def _overlay_label_for_index(idx: int, scale: float) -> str:
    color = "ORANGE" if idx == 0 else "SKY" if idx == 1 else "WHITE"
    return f"{int(round(scale * 100))}%={color}"


def _draw_overlay(
    world: Any,
    bound_tls: list[BoundTL],
    life_time: float,
    width_overlays: list[float],
) -> None:
    debug = world.debug

    # Draw traffic lights and labels.
    for item in bound_tls:
        tl = item.actor
        tl_tf = tl.get_transform()
        tl_state = _traffic_light_state_name(tl)
        if (
            item.zone_center is not None
            and item.zone_rotation is not None
            and item.zone_extent is not None
        ):
            box_center = item.zone_center
            box_rot = item.zone_rotation
            box_extent = item.zone_extent
        else:
            bb = tl.bounding_box
            box_center = tl_tf.transform(bb.location)
            box_rot = tl_tf.rotation
            box_extent = carla.Vector3D(
                x=max(float(bb.extent.x) * 2.0, 0.55),
                y=max(float(bb.extent.y) * 2.0, 0.55),
                z=max(float(bb.extent.z) * 2.4, 2.2),
            )
        draw_bb = carla.BoundingBox(box_center, box_extent)
        debug.draw_box(
            draw_bb,
            box_rot,
            thickness=0.13,
            color=MAGENTA,
            life_time=life_time,
        )

        # Junction-side exit edge (front edge in waypoint yaw direction).
        yawr = math.radians(float(box_rot.yaw))
        fwd = carla.Vector3D(math.cos(yawr), math.sin(yawr), 0.0)
        right = carla.Vector3D(-math.sin(yawr), math.cos(yawr), 0.0)
        front_center = carla.Location(
            x=box_center.x + fwd.x * box_extent.x,
            y=box_center.y + fwd.y * box_extent.x,
            z=box_center.z,
        )
        exit_a = carla.Location(
            x=front_center.x - right.x * box_extent.y,
            y=front_center.y - right.y * box_extent.y,
            z=front_center.z + 0.02,
        )
        exit_b = carla.Location(
            x=front_center.x + right.x * box_extent.y,
            y=front_center.y + right.y * box_extent.y,
            z=front_center.z + 0.02,
        )
        debug.draw_line(
            exit_a,
            exit_b,
            thickness=0.14,
            color=YELLOW,
            life_time=life_time,
        )
        exit_center = front_center
        debug.draw_point(
            carla.Location(x=exit_center.x, y=exit_center.y, z=exit_center.z + 0.03),
            size=0.14,
            color=YELLOW,
            life_time=life_time,
        )
        debug.draw_string(
            carla.Location(x=exit_center.x, y=exit_center.y, z=exit_center.z + 0.55),
            "CENTER TARGET",
            draw_shadow=True,
            color=YELLOW,
            life_time=life_time,
            persistent_lines=False,
        )

        # Optional narrower width overlays to compare lane-fit visually.
        for idx, scale in enumerate(width_overlays):
            if scale <= 0.0:
                continue
            if scale >= 0.999:
                continue
            overlay_extent = carla.Vector3D(
                x=box_extent.x,
                y=max(0.08, box_extent.y * scale),
                z=box_extent.z,
            )
            debug.draw_box(
                carla.BoundingBox(box_center, overlay_extent),
                box_rot,
                thickness=0.10,
                color=_overlay_color_for_index(idx),
                life_time=life_time,
            )

        frame_text = (
            f"F{item.stop.first_stop_frame}" if item.stop.first_stop_frame is not None else "F?"
        )
        top_text = f"TL{item.stop.requested_tl_id} [{tl_state}]"
        if item.stop.stop_order is not None:
            top_text += f"  Stop {item.stop.stop_order}"
        top_text += f"  {frame_text}"
        if (
            item.stop.raw_first_stop_frame is not None
            and item.stop.first_stop_frame is not None
            and item.stop.raw_first_stop_frame != item.stop.first_stop_frame
        ):
            top_text += f"  raw:F{item.stop.raw_first_stop_frame}"
        if item.bind_mode != "direct":
            top_text += f"  live_id:{tl.id}"

        debug.draw_string(
            carla.Location(
                x=box_center.x,
                y=box_center.y,
                z=box_center.z + box_extent.z + 1.3,
            ),
            top_text,
            draw_shadow=True,
            color=MAGENTA,
            life_time=life_time,
            persistent_lines=False,
        )

    # Draw stop markers (where ego first near-stopped for each TL).
    for item in bound_tls:
        if item.stop.stop_position is None:
            continue
        sx, sy, sz = item.stop.stop_position
        stop_loc = carla.Location(x=sx, y=sy, z=sz + 0.35)
        debug.draw_point(stop_loc, size=0.22, color=MAGENTA, life_time=life_time)

        stop_text = f"Stop {item.stop.stop_order}" if item.stop.stop_order is not None else "Stop ?"
        frame_text = (
            f"F{item.stop.first_stop_frame}" if item.stop.first_stop_frame is not None else "F?"
        )
        label = f"{stop_text}  {frame_text} -> TL{item.stop.requested_tl_id}"
        if (
            item.stop.raw_first_stop_frame is not None
            and item.stop.first_stop_frame is not None
            and item.stop.raw_first_stop_frame != item.stop.first_stop_frame
        ):
            label += f"  (raw:F{item.stop.raw_first_stop_frame})"

        debug.draw_string(
            carla.Location(x=sx, y=sy, z=sz + 1.0),
            label,
            draw_shadow=True,
            color=WHITE,
            life_time=life_time,
            persistent_lines=False,
        )

        tl_loc = item.actor.get_transform().location
        line_target = carla.Location(x=tl_loc.x, y=tl_loc.y, z=tl_loc.z + 2.0)
        debug.draw_line(
            line_target,
            stop_loc,
            thickness=0.06,
            color=WHITE,
            life_time=life_time,
        )


def _draw_live_highlights(
    world: Any,
    highlights: list[LiveTLHighlight],
    life_time: float,
) -> None:
    if not highlights:
        return
    debug = world.debug
    for item in highlights:
        tl = item.actor
        if (
            item.zone_center is not None
            and item.zone_rotation is not None
            and item.zone_extent is not None
        ):
            center = item.zone_center
            rot = item.zone_rotation
            extent = item.zone_extent
        else:
            tf = tl.get_transform()
            bb = tl.bounding_box
            center = tf.transform(bb.location)
            rot = tf.rotation
            extent = carla.Vector3D(
                x=max(float(bb.extent.x) * 2.3, 0.7),
                y=max(float(bb.extent.y) * 2.3, 0.7),
                z=max(float(bb.extent.z) * 2.7, 2.5),
            )
        draw_bb = carla.BoundingBox(center, extent)
        debug.draw_box(
            draw_bb,
            rot,
            thickness=0.14,
            color=GREEN,
            life_time=life_time,
        )
        state = _traffic_light_state_name(tl)
        debug.draw_string(
            carla.Location(x=center.x, y=center.y, z=center.z + extent.z + 1.5),
            f"LIVE{item.live_id} [{state}]",
            draw_shadow=True,
            color=GREEN,
            life_time=life_time,
            persistent_lines=False,
        )


def main() -> None:
    args = _parse_args()
    episode_path, tl_ids, auto_episode, auto_tl_ids = _resolve_episode_and_tl_ids(args)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    map_name = world.get_map().name
    if args.auto_load_town and args.town not in map_name:
        print(f"[info] Current world is '{map_name}', loading '{args.town}'...")
        client.set_timeout(120.0)
        client.load_world(args.town)
        world = client.get_world()
        client.set_timeout(10.0)
        map_name = world.get_map().name
        print(f"[info] Loaded world: {map_name}")
    elif args.town not in map_name:
        print(f"[warn] Current world is '{map_name}', expected '{args.town}'")

    stops = _extract_episode_stops(
        episode_path,
        tl_ids,
        args.stop_speed_kmh,
        args.allow_initial_hold_stop,
        args.min_prior_speed_kmh,
    )

    stops_for_overlay = stops
    if not args.show_no_stop_tls:
        hidden = [s for s in stops if s.first_stop_frame is None]
        if hidden:
            hidden_txt = ", ".join(f"TL{s.requested_tl_id}" for s in hidden)
            print(
                "[info] Hiding TLs with no near-stop frame in this episode: "
                f"{hidden_txt} (use --show-no-stop-tls to include)"
            )
        stops_for_overlay = [s for s in stops if s.first_stop_frame is not None]
        if not stops_for_overlay:
            print("[warn] No near-stop frames found; drawing all requested TLs.")
            stops_for_overlay = stops

    bound_tls = _bound_tl_list(world, stops_for_overlay)
    if not bound_tls:
        raise SystemExit("No traffic lights could be bound. Nothing to draw.")
    live_highlights = _collect_live_highlights(world, args.highlight_live_ids)

    print("=" * 76)
    print(f"Traffic-Light Visual Overlay (magenta) for episode {episode_path.name}")
    print("=" * 76)
    if auto_episode:
        print(f"[auto] episode from town profile: {episode_path}")
    if auto_tl_ids:
        print(f"[auto] tl_ids from town profile: {tl_ids}")
    width_overlays = [w for w in args.width_overlays if 0.0 < w < 1.0]

    for item in sorted(
        bound_tls,
        key=lambda b: (b.stop.stop_order is None, b.stop.stop_order or 999),
    ):
        frame_txt = (
            str(item.stop.first_stop_frame) if item.stop.first_stop_frame is not None else "none"
        )
        order_txt = str(item.stop.stop_order) if item.stop.stop_order is not None else "none"
        print(
            f"TL{item.stop.requested_tl_id}: stop_order={order_txt} first_stop_frame={frame_txt} "
            f"bind={item.bind_mode} live_id={item.actor.id} bind_dist={item.bind_distance:.2f}m"
        )
        if item.zone_extent is not None:
            base_width = float(item.zone_extent.y * 2.0)
            print(f"  zone width (100%): {base_width:.2f} m")
            for i, scale in enumerate(width_overlays):
                print(
                    f"  zone width ({int(round(scale * 100))}%): {base_width * scale:.2f} m "
                    f"color={('ORANGE' if i == 0 else 'SKY' if i == 1 else 'WHITE')}"
                )
        if (
            item.zone_center is not None
            and item.zone_rotation is not None
            and item.zone_extent is not None
            and item.stop.stop_position is not None
        ):
            sx, sy, sz = item.stop.stop_position
            yawr = math.radians(float(item.zone_rotation.yaw))
            exit_cx = float(item.zone_center.x) + math.cos(yawr) * float(item.zone_extent.x)
            exit_cy = float(item.zone_center.y) + math.sin(yawr) * float(item.zone_extent.x)
            dx = sx - exit_cx
            dy = sy - exit_cy
            fwdx, fwdy = math.cos(yawr), math.sin(yawr)
            rightx, righty = -math.sin(yawr), math.cos(yawr)
            lon_err = dx * fwdx + dy * fwdy
            lat_err = dx * rightx + dy * righty
            print(
                "  stop-center error vs yellow exit-edge center: "
                f"longitudinal={lon_err:.2f}m lateral={lat_err:.2f}m"
            )
        if (
            item.stop.raw_first_stop_frame is not None
            and item.stop.first_stop_frame is not None
            and item.stop.raw_first_stop_frame != item.stop.first_stop_frame
        ):
            print(
                f"  note: raw first near-stop was F{item.stop.raw_first_stop_frame}, "
                f"displaying first stop-after-motion F{item.stop.first_stop_frame}"
            )

    if live_highlights:
        print("\nExtra green live-ID highlights:")
        for h in live_highlights:
            print(f"  live_id:{h.live_id} state={_traffic_light_state_name(h.actor)}")

    print("\nControls: click CARLA viewport, hold right mouse button, use WASD + Q/E to fly.")
    print(
        "Overlay: magenta box = lane-local stop zone for that TL, "
        "white text/point = where ego first near-stopped for it."
    )
    print("Exit edge: junction-side edge of magenta box is highlighted in YELLOW.")
    print("Center target: yellow dot/text marks the desired center-stop point at that exit edge.")
    if width_overlays:
        print(
            "Width overlays: "
            + ", ".join(
                [_overlay_label_for_index(i, scale) for i, scale in enumerate(width_overlays)]
            )
        )
    else:
        print("Width overlays: disabled. Re-enable with --width-overlays 0.75 0.65")
    print("Green boxes: explicitly requested live traffic-light IDs.")
    print("HUD: live location bar + heading.")
    print("Press Ctrl+C in this terminal when done.\n")

    if args.set_initial_view:
        _set_initial_spectator_view(world, bound_tls)

    start = time.time()
    try:
        while True:
            _draw_overlay(world, bound_tls, args.life_time, width_overlays)
            _draw_live_highlights(world, live_highlights, args.life_time)
            _draw_navigation_hud(world, args.hud_side, args.life_time)
            if args.duration_sec > 0 and (time.time() - start) >= args.duration_sec:
                break
            time.sleep(max(0.03, args.refresh_sec))
    except KeyboardInterrupt:
        pass

    print("Visualizer stopped.")


if __name__ == "__main__":
    main()
