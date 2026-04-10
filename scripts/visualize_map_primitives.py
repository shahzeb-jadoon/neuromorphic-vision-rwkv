#!/usr/bin/env python3
"""Draw map-wide traffic controls and junction overlays in live CARLA.

This visualizer draws:
- All traffic lights (actor body + stop/affected lane points)
- All stop signs (actor body + trigger volume)
- All yield signs (actor body + trigger volume)
- All junction bounding volumes

The overlay refreshes continuously so you can fly spectator freely using
mouse + WASD + Q/E in the CARLA viewport.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Any

try:
    import carla
except ImportError as exc:  # pragma: no cover
    raise SystemExit("CARLA Python API is required. Activate your CARLA env first.") from exc


TL_COLOR = carla.Color(238, 51, 119)  # #EE3377
TL_STOP_COLOR = carla.Color(102, 204, 238)  # #66CCEE
STOP_COLOR = carla.Color(187, 85, 102)  # #BB5566
STOP_TV_COLOR = carla.Color(238, 119, 51)  # #EE7733
YIELD_COLOR = carla.Color(204, 187, 68)  # #CCBB44
YIELD_TV_COLOR = carla.Color(153, 153, 51)  # #999933
JUNCTION_COLOR = carla.Color(68, 119, 170)  # #4477AA
HUD_TEXT = carla.Color(255, 255, 255)


@dataclass
class JunctionOverlay:
    junction_id: int
    center: Any
    extent: Any
    rotation: Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw all TL/stop/yield/junction overlays in live CARLA.",
    )
    parser.add_argument("--host", default="172.20.0.1", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument(
        "--town", default="Town10HD", help="Town to load (Town10 auto-maps to Town10HD)"
    )
    parser.add_argument(
        "--auto-load-town",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load --town if the current world does not match",
    )
    parser.add_argument(
        "--draw-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw ID labels for objects",
    )
    parser.add_argument(
        "--draw-stop-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw TL stop/affected lane points",
    )
    parser.add_argument(
        "--set-initial-view",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set spectator to a high-level initial overview",
    )
    parser.add_argument(
        "--junction-sampling",
        type=float,
        default=5.0,
        help="Waypoint sampling (m) used to discover unique junctions",
    )
    parser.add_argument(
        "--refresh-sec",
        type=float,
        default=0.20,
        help="Overlay refresh period (s)",
    )
    parser.add_argument(
        "--life-time",
        type=float,
        default=0.65,
        help="CARLA debug primitive lifetime (s)",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=0.0,
        help="Run for N seconds (0 means until Ctrl+C)",
    )
    parser.add_argument(
        "--label-distance",
        type=float,
        default=180.0,
        help="Only draw labels within this distance from spectator (m)",
    )
    return parser.parse_args()


def _normalize_town_name(name: str) -> str:
    raw = name.strip()
    lower = raw.lower()
    if lower == "town10":
        return "Town10HD"
    if lower == "town10hd":
        return "Town10HD"
    if lower.startswith("town"):
        suffix = raw[4:]
        if suffix:
            return f"Town{suffix}"
    return raw


def _load_target_world(client: Any, requested_town: str, auto_load: bool) -> Any:
    world = client.get_world()
    cur_name = world.get_map().name
    want = _normalize_town_name(requested_town)

    if auto_load and want not in cur_name:
        print(f"[info] Current world is '{cur_name}', loading '{want}'...")
        client.set_timeout(120.0)
        client.load_world(want)
        world = client.get_world()
        client.set_timeout(10.0)
        cur_name = world.get_map().name
        print(f"[info] Loaded world: {cur_name}")
    elif want not in cur_name:
        print(f"[warn] Current world '{cur_name}' does not match requested '{want}'")

    return world


def _actor_body_box(actor: Any, z_pad: float = 0.0, min_extent: float = 0.45) -> tuple[Any, Any]:
    tf = actor.get_transform()
    bb = actor.bounding_box
    center = tf.transform(bb.location)
    center = carla.Location(x=center.x, y=center.y, z=center.z + z_pad)
    extent = carla.Vector3D(
        x=max(float(bb.extent.x), min_extent),
        y=max(float(bb.extent.y), min_extent),
        z=max(float(bb.extent.z), min_extent),
    )
    return carla.BoundingBox(center, extent), tf.rotation


def _trigger_volume_box(
    actor: Any, z_pad: float = 0.1, min_extent: float = 0.2
) -> tuple[Any, Any] | None:
    try:
        tv = actor.trigger_volume
        tf = actor.get_transform()
        center = tf.transform(tv.location)
        center = carla.Location(x=center.x, y=center.y, z=center.z + z_pad)
        extent = carla.Vector3D(
            x=max(float(tv.extent.x), min_extent),
            y=max(float(tv.extent.y), min_extent),
            z=max(float(tv.extent.z), min_extent),
        )
        return carla.BoundingBox(center, extent), tf.rotation
    except (AttributeError, RuntimeError):
        return None


def _collect_tl_points(tl: Any) -> list[Any]:
    points: list[Any] = []
    try:
        for wp in tl.get_stop_waypoints():
            points.append(wp.transform.location)
    except (AttributeError, RuntimeError):
        pass

    if points:
        return points

    try:
        for wp in tl.get_affected_lane_waypoints():
            points.append(wp.transform.location)
    except (AttributeError, RuntimeError):
        pass

    return points


def _collect_junctions(carla_map: Any, sampling: float) -> list[JunctionOverlay]:
    by_id: dict[int, JunctionOverlay] = {}
    step = max(1.0, float(sampling))

    for wp in carla_map.generate_waypoints(step):
        if not wp.is_junction:
            continue
        try:
            junction = wp.get_junction()
        except RuntimeError:
            continue
        if junction is None:
            continue

        jid = int(junction.id)
        if jid in by_id:
            continue

        bb = junction.bounding_box
        center = carla.Location(x=bb.location.x, y=bb.location.y, z=bb.location.z + 0.2)
        extent = carla.Vector3D(
            x=max(float(bb.extent.x), 1.0),
            y=max(float(bb.extent.y), 1.0),
            z=max(float(bb.extent.z), 0.6),
        )
        rotation = getattr(bb, "rotation", carla.Rotation())
        by_id[jid] = JunctionOverlay(
            junction_id=jid,
            center=center,
            extent=extent,
            rotation=rotation,
        )

    return [by_id[jid] for jid in sorted(by_id)]


def _set_initial_spectator(world: Any, tls: list[Any], stops: list[Any], yields: list[Any]) -> None:
    points: list[tuple[float, float, float]] = []
    for actor in tls + stops + yields:
        loc = actor.get_transform().location
        points.append((float(loc.x), float(loc.y), float(loc.z)))

    if not points:
        return

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    z = max(45.0, min(220.0, span * 0.35 + 70.0))

    spectator = world.get_spectator()
    spectator.set_transform(
        carla.Transform(
            carla.Location(x=cx, y=cy, z=z),
            carla.Rotation(pitch=-88.0, yaw=0.0, roll=0.0),
        )
    )


def _dist(a: Any, b: Any) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _draw_hud(world: Any, counts: dict[str, int], life_time: float) -> None:
    debug = world.debug
    spec = world.get_spectator()
    tf = spec.get_transform()

    fwd = tf.get_forward_vector()
    right = tf.get_right_vector()
    up = tf.get_up_vector()
    loc = tf.location

    def off(forward_m: float, right_m: float, up_m: float) -> Any:
        return carla.Location(
            x=loc.x + fwd.x * forward_m + right.x * right_m + up.x * up_m,
            y=loc.y + fwd.y * forward_m + right.y * right_m + up.y * up_m,
            z=loc.z + fwd.z * forward_m + right.z * right_m + up.z * up_m,
        )

    lines = [
        "Map Primitives Overlay",
        (
            f"TL: {counts['tls']}  Stop: {counts['stops']}  "
            f"Yield: {counts['yields']}  Junctions: {counts['junctions']}"
        ),
        (
            "Colors: TL=Magenta, TL stop points=Cyan, "
            "Stop=Red/Orange, Yield=Yellow/Olive, Junction=Blue"
        ),
        "Fly camera: hold right mouse + WASD/QE",
    ]

    for i, text in enumerate(lines):
        debug.draw_string(
            off(12.0, -8.5, 3.2 - i * 0.55),
            text,
            draw_shadow=True,
            color=HUD_TEXT,
            life_time=life_time,
            persistent_lines=False,
        )


def _draw_overlay(
    world: Any,
    tls: list[Any],
    stops: list[Any],
    yields: list[Any],
    junctions: list[JunctionOverlay],
    draw_labels: bool,
    draw_stop_points: bool,
    label_dist: float,
    life_time: float,
) -> None:
    debug = world.debug
    spec_loc = world.get_spectator().get_transform().location

    # Traffic lights
    for tl in tls:
        box, rot = _actor_body_box(tl, z_pad=0.2, min_extent=0.5)
        debug.draw_box(box, rot, thickness=0.13, color=TL_COLOR, life_time=life_time)

        if draw_stop_points:
            for wp_loc in _collect_tl_points(tl):
                p = carla.Location(x=wp_loc.x, y=wp_loc.y, z=wp_loc.z + 0.25)
                debug.draw_point(p, size=0.09, color=TL_STOP_COLOR, life_time=life_time)

        if draw_labels and _dist(spec_loc, box.location) <= label_dist:
            state = str(getattr(tl, "state", "?")).split(".")[-1].upper()
            debug.draw_string(
                carla.Location(
                    x=box.location.x, y=box.location.y, z=box.location.z + box.extent.z + 1.2
                ),
                f"TL {tl.id} [{state}]",
                draw_shadow=True,
                color=TL_COLOR,
                life_time=life_time,
                persistent_lines=False,
            )

    # Stop signs
    for ss in stops:
        box, rot = _actor_body_box(ss, z_pad=0.1, min_extent=0.4)
        debug.draw_box(box, rot, thickness=0.12, color=STOP_COLOR, life_time=life_time)

        tv = _trigger_volume_box(ss, z_pad=0.12)
        if tv is not None:
            tv_box, tv_rot = tv
            debug.draw_box(tv_box, tv_rot, thickness=0.10, color=STOP_TV_COLOR, life_time=life_time)

        if draw_labels and _dist(spec_loc, box.location) <= label_dist:
            debug.draw_string(
                carla.Location(
                    x=box.location.x, y=box.location.y, z=box.location.z + box.extent.z + 1.0
                ),
                f"STOP {ss.id}",
                draw_shadow=True,
                color=STOP_COLOR,
                life_time=life_time,
                persistent_lines=False,
            )

    # Yield signs
    for ys in yields:
        box, rot = _actor_body_box(ys, z_pad=0.1, min_extent=0.4)
        debug.draw_box(box, rot, thickness=0.12, color=YIELD_COLOR, life_time=life_time)

        tv = _trigger_volume_box(ys, z_pad=0.12)
        if tv is not None:
            tv_box, tv_rot = tv
            debug.draw_box(
                tv_box, tv_rot, thickness=0.10, color=YIELD_TV_COLOR, life_time=life_time
            )

        if draw_labels and _dist(spec_loc, box.location) <= label_dist:
            debug.draw_string(
                carla.Location(
                    x=box.location.x, y=box.location.y, z=box.location.z + box.extent.z + 1.0
                ),
                f"YIELD {ys.id}",
                draw_shadow=True,
                color=YIELD_COLOR,
                life_time=life_time,
                persistent_lines=False,
            )

    # Junctions
    for j in junctions:
        bb = carla.BoundingBox(j.center, j.extent)
        debug.draw_box(bb, j.rotation, thickness=0.09, color=JUNCTION_COLOR, life_time=life_time)

        if draw_labels and _dist(spec_loc, j.center) <= label_dist:
            debug.draw_string(
                carla.Location(x=j.center.x, y=j.center.y, z=j.center.z + j.extent.z + 1.0),
                f"JUNCTION {j.junction_id}",
                draw_shadow=True,
                color=JUNCTION_COLOR,
                life_time=life_time,
                persistent_lines=False,
            )


def main() -> None:
    args = _parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    world = _load_target_world(client, args.town, args.auto_load_town)
    carla_map = world.get_map()

    actors = world.get_actors()
    tls = list(actors.filter("traffic.traffic_light"))
    stops = list(actors.filter("traffic.stop"))
    yields = list(actors.filter("traffic.yield"))
    junctions = _collect_junctions(carla_map, args.junction_sampling)

    counts = {
        "tls": len(tls),
        "stops": len(stops),
        "yields": len(yields),
        "junctions": len(junctions),
    }

    print("=" * 76)
    print("Map Primitives Overlay")
    print("=" * 76)
    print(f"World: {carla_map.name}")
    print(
        "Counts: "
        f"traffic_lights={counts['tls']} stop_signs={counts['stops']} "
        f"yield_signs={counts['yields']} junctions={counts['junctions']}"
    )
    print("Controls: click CARLA viewport, hold right mouse, use WASD + Q/E to fly.")
    print("Press Ctrl+C in this terminal when done.\n")

    if args.set_initial_view:
        _set_initial_spectator(world, tls, stops, yields)

    start = time.time()
    try:
        while True:
            _draw_overlay(
                world,
                tls,
                stops,
                yields,
                junctions,
                draw_labels=args.draw_labels,
                draw_stop_points=args.draw_stop_points,
                label_dist=float(args.label_distance),
                life_time=float(args.life_time),
            )
            _draw_hud(world, counts, float(args.life_time))

            if args.duration_sec > 0 and (time.time() - start) >= args.duration_sec:
                break
            time.sleep(max(0.03, float(args.refresh_sec)))
    except KeyboardInterrupt:
        pass

    print("Visualizer stopped.")


if __name__ == "__main__":
    main()
