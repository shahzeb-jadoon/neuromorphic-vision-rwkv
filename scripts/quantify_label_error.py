"""Quantify navigation command labeling error.

Compares the heuristic _infer_nav_command() labels against ground-truth
RoadOption labels from CARLA's GlobalRoutePlanner by generating balanced
routes and applying both labeling methods to the same waypoints.

No episodes need to be collected — this runs against the live CARLA server.

Usage:
    conda activate rwkv
    python scripts/quantify_label_error.py
    python scripts/quantify_label_error.py --towns Town01 Town03 --routes-per-town 10
"""

from __future__ import annotations

import argparse
import logging
import math
from collections import defaultdict
from typing import Any

import numpy as np

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption
from src.data.expert_collector import ExpertDataCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Nav command IDs matching _infer_nav_command output
NAV_NAMES = {0: "LaneFollow", 1: "Left", 2: "Right", 3: "Straight"}

# RoadOption → heuristic nav ID mapping
ROAD_OPTION_TO_NAV = {
    RoadOption.LANEFOLLOW: 0,
    RoadOption.LEFT: 1,
    RoadOption.RIGHT: 2,
    RoadOption.STRAIGHT: 3,
    RoadOption.CHANGELANELEFT: 0,
    RoadOption.CHANGELANERIGHT: 0,
    RoadOption.VOID: 0,
}


def heuristic_nav_command_from_waypoints(
    current_wp: Any,
    next_wp: Any,
) -> int:
    """Simulate _infer_nav_command using waypoint data.

    Replicates the heuristic logic from ExpertDataCollector._infer_nav_command
    but works with route waypoints instead of a live vehicle.

    Args:
        current_wp: Current CARLA waypoint.
        next_wp: Next waypoint along the route.

    Returns:
        Navigation command ID (0-3).
    """
    # Vehicle heading at current waypoint (the forward vector)
    fwd = current_wp.transform.rotation.get_forward_vector()
    fwd_2d = np.array([fwd.x, fwd.y])

    # Direction from current to next waypoint
    cur_loc = current_wp.transform.location
    nxt_loc = next_wp.transform.location
    dir_2d = np.array([nxt_loc.x - cur_loc.x, nxt_loc.y - cur_loc.y])

    fwd_norm = np.linalg.norm(fwd_2d)
    dir_norm = np.linalg.norm(dir_2d)
    if fwd_norm < 1e-6 or dir_norm < 1e-6:
        return 0

    fwd_2d = fwd_2d / fwd_norm
    dir_2d = dir_2d / dir_norm

    cross = fwd_2d[0] * dir_2d[1] - fwd_2d[1] * dir_2d[0]
    dot = np.dot(fwd_2d, dir_2d)
    angle_deg = math.degrees(math.atan2(cross, dot))

    threshold = 10.0
    at_junction = current_wp.is_junction

    if abs(angle_deg) < threshold:
        if at_junction:
            return 3  # Straight
        return 0  # LaneFollow
    elif angle_deg < -threshold:
        return 1  # Left
    else:
        return 2  # Right


def compare_on_routes(
    world: Any,
    town: str,
    n_routes: int = 10,
    rng: np.random.Generator | None = None,
) -> dict:
    """Generate routes and compare heuristic vs ground-truth labels.

    Args:
        world: CARLA world instance.
        town: Town name.
        n_routes: Number of random routes to generate.
        rng: Random number generator.

    Returns:
        Dict with per-class confusion counts and overall statistics.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    grp = GlobalRoutePlanner(carla_map, 2.0)

    confusion: dict[tuple[str, str], int] = defaultdict(int)
    total = 0
    agree = 0

    for route_idx in range(n_routes):
        # Pick random start and end
        indices = rng.choice(len(spawn_points), size=2, replace=False)
        start_loc = spawn_points[int(indices[0])].location
        end_loc = spawn_points[int(indices[1])].location

        try:
            route = grp.trace_route(start_loc, end_loc)
        except Exception as e:
            logger.warning("Route %d failed: %s", route_idx, e)
            continue

        if len(route) < 5:
            continue

        # Compare labels at each waypoint along the route
        for i in range(len(route) - 1):
            wp, road_option = route[i]
            next_wp, _ = route[i + 1]

            # Ground-truth label
            gt_nav = ROAD_OPTION_TO_NAV.get(road_option, 0)
            gt_name = NAV_NAMES[gt_nav]

            # Heuristic label
            heuristic_nav = heuristic_nav_command_from_waypoints(wp, next_wp)
            heuristic_name = NAV_NAMES[heuristic_nav]

            confusion[(heuristic_name, gt_name)] += 1
            total += 1
            if heuristic_nav == gt_nav:
                agree += 1

    accuracy = agree / total if total > 0 else 0.0
    logger.info(
        "%s: %d routes, %d waypoints compared, %.1f%% agreement",
        town,
        n_routes,
        total,
        accuracy * 100,
    )

    return {
        "town": town,
        "n_routes": n_routes,
        "total": total,
        "agreed": agree,
        "accuracy": accuracy,
        "confusion": dict(confusion),
    }


def print_results(all_results: list[dict]) -> None:
    """Print summary with confusion matrix."""
    print("\n" + "=" * 70)
    print("NAVIGATION COMMAND LABELING ERROR ANALYSIS")
    print("Heuristic (_infer_nav_command) vs Ground Truth (GlobalRoutePlanner)")
    print("=" * 70)

    total_all = 0
    agreed_all = 0
    all_confusion: dict[tuple[str, str], int] = defaultdict(int)

    for r in all_results:
        print(f"\n{r['town']}:")
        print(f"  Routes: {r['n_routes']}, waypoints compared: {r['total']}")
        print(f"  Agreement: {r['agreed']}/{r['total']} ({r['accuracy']:.1%})")
        total_all += r["total"]
        agreed_all += r["agreed"]
        for k, v in r["confusion"].items():
            all_confusion[k] += v

    if total_all == 0:
        print("\nNo waypoints compared.")
        return

    overall_acc = agreed_all / total_all
    labels = ["LaneFollow", "Left", "Right", "Straight"]

    # Build confusion matrix
    matrix = np.zeros((4, 4), dtype=int)
    for (h_name, gt_name), count in all_confusion.items():
        if h_name in labels and gt_name in labels:
            matrix[labels.index(h_name), labels.index(gt_name)] = count

    print(f"\n{'='*70}")
    print(f"OVERALL: {agreed_all}/{total_all} ({overall_acc:.1%} agreement)")
    print("\nConfusion matrix (rows=heuristic, cols=ground_truth):")
    header = f"{'Heuristic↓ GT→':>16}" + "".join(f"{lbl:>12}" for lbl in labels)
    print(header)
    print("-" * (16 + 12 * len(labels)))
    for i, label in enumerate(labels):
        row = f"{label:>16}" + "".join(f"{matrix[i, j]:>12}" for j in range(4))
        # Per-row accuracy
        row_total = matrix[i, :].sum()
        row_acc = matrix[i, i] / row_total if row_total > 0 else 0.0
        row += f"  ({row_acc:.0%})"
        print(row)

    print("\nPer-class precision (ground-truth column accuracy):")
    for j, label in enumerate(labels):
        col_total = matrix[:, j].sum()
        correct = matrix[j, j]
        prec = correct / col_total if col_total > 0 else 0.0
        print(f"  {label:>12}: {correct}/{col_total} ({prec:.1%})")

    print(f"\nOverall labeling error rate: {1 - overall_acc:.1%}")
    print(
        "\nKey findings:"
        "\n  - Report this error rate as a known limitation of proxy labelling"
        "\n  - Heuristic is most accurate for LaneFollow (dominant class)"
        "\n  - Turns (Left/Right) may have higher mislabeling rates"
        "\n  - Straight-through-junction vs LaneFollow confusion is expected"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantify nav command labeling error (heuristic vs GlobalRoutePlanner)"
    )
    parser.add_argument(
        "--towns",
        nargs="+",
        default=["Town01", "Town03", "Town04", "Town05"],
    )
    parser.add_argument("--routes-per-town", type=int, default=20)
    parser.add_argument("--host", type=str, default="auto")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Connect
    collector = ExpertDataCollector(
        host=args.host, port=args.port, output_dir="/tmp/label_error_scratch"
    )
    collector.connect()

    rng = np.random.default_rng(args.seed)
    all_results = []

    for town in args.towns:
        logger.info("Loading %s...", town)
        try:
            assert collector.world is not None
            assert collector.client is not None
            current_map_name = collector.world.get_map().name
            if town not in current_map_name:
                collector.client.load_world(town)
                import time

                time.sleep(5)
                collector.world = collector.client.get_world()
            else:
                logger.info("Already on %s, skipping load", town)

            result = compare_on_routes(collector.world, town, args.routes_per_town, rng)
            all_results.append(result)
        except Exception as e:
            logger.error("Failed on %s: %s", town, e)

    print_results(all_results)


if __name__ == "__main__":
    main()
