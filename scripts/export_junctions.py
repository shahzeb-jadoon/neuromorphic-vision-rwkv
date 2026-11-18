"""Export the junction areas of CARLA towns, for labels that need to know where junctions are.

Loads each town in a running CARLA server (CARLA_HOST, CARLA_PORT), samples its driving
lanes inside every junction every 0.5 m, widens them by half a lane on each side and writes
the convex outline of each junction (world x, y in metres) to OUT.json. Loading a town
replaces the server's world; the town loaded before is loaded again at the end.

usage: python scripts/export_junctions.py OUT.json [--towns Town01 Town03 ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import carla
import numpy as np
from scipy.spatial import ConvexHull  # type: ignore[import-untyped]

from src.utils.settings import setting

V3_TOWNS = ["Town01", "Town03", "Town04", "Town05", "Town06", "Town10HD"]


def junction_outlines(world_map: carla.Map) -> dict[int, list[list[float]]]:
    edges: dict[int, list[tuple[float, float]]] = {}
    for wp in world_map.generate_waypoints(0.5):
        if not wp.is_junction:
            continue
        centre, right, half = (
            wp.transform.location,
            wp.transform.get_right_vector(),
            wp.lane_width / 2,
        )
        for side in (-half, half):
            point = (centre.x + side * right.x, centre.y + side * right.y)
            edges.setdefault(wp.get_junction().id, []).append(point)
    outlines = {}
    for junction, points in edges.items():
        xy = np.array(points)
        outlines[junction] = xy[ConvexHull(xy).vertices].round(2).tolist()
    return outlines


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("out", type=Path)
    p.add_argument("--towns", nargs="+", default=V3_TOWNS)
    args = p.parse_args()

    client = carla.Client(setting("CARLA_HOST"), int(setting("CARLA_PORT")))
    client.set_timeout(180.0)
    before = client.get_world().get_map().name.split("/")[-1]
    towns = {}
    try:
        for town in args.towns:
            outlines = junction_outlines(client.load_world(town).get_map())
            towns[town] = [{"id": j, "outline": o} for j, o in sorted(outlines.items())]
            print(f"{town}: {len(outlines)} junctions", flush=True)
    finally:
        client.load_world(before)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(towns), encoding="utf-8")


if __name__ == "__main__":
    main()
