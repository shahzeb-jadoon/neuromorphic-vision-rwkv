"""Pick a frozen route suite from the maps of a running CARLA server.

    python scripts/make_route_suite.py --out configs/routes/suite_v1.json

For each town, routes between spawn points are planned at the leaderboard's 1 m spacing and
trimmed to start and end outside junctions. A route is stored as keypoints about every 50 m
outside junctions, so it does not depend on how a later planner joins its ends. It is kept if
its length is in range, it turns at two junctions or more and it plans again to the same
length through its keypoints. Of those, a town's routes are picked one at a time, each
covering the most road the town's earlier picks did not. Each route records its turns, lane
changes and traffic lights, so results can be split by them. Traffic (20, 50 or 100 vehicles;
10, 30 or 50 walkers) and the recording's 10 weather presets rotate over the suite. Once
committed, a suite is never edited: a change is a new version.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import carla
import numpy as np

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption
from agents.navigation.route import plan_route
from src.data.labels import ROUTE_RESOLUTION
from src.eval.closed_loop import ROUTE_MISMATCH, red_light_lines
from src.eval.driving_score import cumulative_length
from src.utils.settings import setting

TOWNS = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"]  # Town02 is held out
WEATHERS = [  # configs/collection.yaml, the presets v3 was recorded in
    "ClearNoon", "CloudyNoon", "WetNoon", "SoftRainNoon", "HardRainNoon",
    "ClearSunset", "CloudySunset", "WetSunset", "MidRainyNoon", "SoftRainSunset",
]  # fmt: skip
TRAFFIC = [(20, 10), (50, 30), (100, 50)]  # vehicles, walkers
SMALL_TOWNS = {"Town02": 50}  # most vehicles a town with few spawn points gets
KEYPOINT_SPACING = 50.0  # m
COVERED = 5.0  # m: road within this of an earlier pick's route counts as covered
TURNS = {RoadOption.LEFT, RoadOption.RIGHT}
LANE_CHANGES = {RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT}


def describe(route: list[tuple[Any, Any]], stop_lines: np.ndarray) -> dict[str, Any]:
    xy = np.array([[wp.transform.location.x, wp.transform.location.y] for wp, _ in route])
    options = [option for _, option in route]
    turns = sum(o in TURNS and o != p for p, o in zip([None, *options], options))
    near = np.linalg.norm(xy[:, None] - stop_lines[None], axis=2).min(axis=0, initial=np.inf)
    return {
        "xy": xy,
        "length_m": float(cumulative_length(xy)[-1]),
        "turns": turns,
        "lane_changes": sum(
            o in LANE_CHANGES and o != p for p, o in zip([None, *options], options)
        ),
        "lights": int((near < 1.5).sum()),
    }


def clear(town: Any, wp: Any) -> bool:
    """Whether a route point is outside junctions and on the lane the map gives back for its
    location: where lanes overlap, a planner or a spawn may join a point to the wrong one."""
    if wp.is_junction:
        return False
    back = town.get_waypoint(wp.transform.location)
    return bool((back.road_id, back.lane_id) == (wp.road_id, wp.lane_id))


def trim(town: Any, route: list[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
    """The route from its first clear point to its last."""
    first = next((i for i, (wp, _) in enumerate(route) if clear(town, wp)), len(route))
    last = next((i for i in range(len(route) - 1, -1, -1) if clear(town, route[i][0])), -1)
    return route[first : last + 1]


def keypoints(town: Any, route: list[tuple[Any, Any]]) -> list[list[float]]:
    """The first and last route points and a clear one about every 50 m between."""
    along = cumulative_length(np.array([[w.transform.location.x, w.transform.location.y]
                                        for w, _ in route]))  # fmt: skip
    picks, mark = [0], KEYPOINT_SPACING
    for i, (wp, _) in enumerate(route[:-1]):
        if along[i] >= mark and clear(town, wp):
            picks.append(i)
            mark = along[i] + KEYPOINT_SPACING
    picks.append(len(route) - 1)
    points = [route[i][0].transform.location for i in picks]
    return [[round(q.x, 2), round(q.y, 2), round(q.z, 2)] for q in points]


def replans(planner: Any, points: list[list[float]], length_m: float) -> bool:
    """Whether the route planned through the keypoints has the route's length."""
    route = plan_route(planner, [carla.Location(*q) for q in points])
    xy = np.array([[w.transform.location.x, w.transform.location.y] for w, _ in route])
    return len(xy) > 1 and abs(cumulative_length(xy)[-1] - length_m) <= ROUTE_MISMATCH


def pick_town(
    world: Any, n: int, length: tuple[float, float], candidates: int, rng: random.Random
) -> list[dict[str, Any]]:
    town = world.get_map()
    planner = GlobalRoutePlanner(town, ROUTE_RESOLUTION)
    lines, _ = red_light_lines(world)
    stop_lines = np.array([(line.a + line.b) / 2 for line in lines]).reshape(-1, 2)
    spawns = town.get_spawn_points()
    pool = []
    for _ in range(candidates):
        a, b = rng.sample(spawns, 2)
        route = trim(town, planner.trace_route(a.location, b.location))
        if len(route) < 2:
            continue
        info = describe(route, stop_lines)
        if length[0] <= info["length_m"] <= length[1] and info["turns"] >= 2:
            info["keypoints"] = keypoints(town, route)
            if replans(planner, info["keypoints"], info["length_m"]):
                pool.append((route, info))
    picked: list[tuple[Any, dict[str, Any]]] = []
    covered = np.zeros((0, 2))
    while pool and len(picked) < n:

        def new_road(item: tuple[Any, dict[str, Any]]) -> float:
            xy = item[1]["xy"]
            if not len(covered):
                return float(item[1]["length_m"])
            gaps = np.min(np.linalg.norm(xy[:, None] - covered[None], axis=2), axis=1)
            return float((gaps > COVERED).sum() * ROUTE_RESOLUTION)

        best = max(pool, key=new_road)
        pool.remove(best)
        picked.append(best)
        covered = np.vstack([covered, best[1]["xy"]])
    return [
        {
            "keypoints": info["keypoints"],
            "length_m": round(info["length_m"], 1),
            "turns": info["turns"],
            "lane_changes": info["lane_changes"],
            "lights": info["lights"],
        }
        for route, info in picked
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--towns", nargs="*", default=TOWNS)
    parser.add_argument("--per-town", type=int, default=4)
    parser.add_argument("--min-length", type=float, default=500.0)
    parser.add_argument("--max-length", type=float, default=1200.0)
    parser.add_argument("--candidates", type=int, default=1000, help="routes planned per town")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    client = carla.Client(setting("CARLA_HOST"), int(setting("CARLA_PORT", "2000")))
    client.set_timeout(120.0)
    rng = random.Random(args.seed)
    routes: list[dict[str, Any]] = []
    for town in args.towns:
        world = client.load_world(town)
        picks = pick_town(
            world, args.per_town, (args.min_length, args.max_length), args.candidates, rng
        )
        for i, pick in enumerate(picks):
            k = len(routes)
            vehicles, walkers = TRAFFIC[k % len(TRAFFIC)]
            vehicles = min(vehicles, SMALL_TOWNS.get(town, vehicles))
            routes.append({
                "id": f"{town.lower()}_{i + 1}",
                "town": town,
                "seed": k,
                "vehicles": vehicles,
                "walkers": walkers,
                "weather": WEATHERS[k % len(WEATHERS)],
                **pick,
            })  # fmt: skip
            r = routes[-1]
            print(
                f"{r['id']:>12} {r['length_m']:7.1f} m  {r['turns']} turns  "
                f"{r['lane_changes']} lane changes  {r['lights']} lights  {vehicles} vehicles  "
                f"{r['weather']}"
            )
    suite = {
        "version": args.out.stem,
        "generator": vars(args) | {"out": str(args.out)},
        "routes": routes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(suite, indent=1) + "\n")


if __name__ == "__main__":
    main()
