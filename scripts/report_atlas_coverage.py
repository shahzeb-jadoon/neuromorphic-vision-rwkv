#!/usr/bin/env python3
"""Recompute and report atlas traversal coverage from emitted manifests.

This script reconstructs the selected edge routes using each manifest's saved
settings, then reports traversed-vs-total coverage for roads, lanes, junction
IDs, and inner-junction roads per town/solver.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import networkx as nx

import scripts.build_route_atlas_cpp as atlas


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 3)


def _town_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("Town")])


def _collect_totals(edges: list[atlas.EdgeGeometry]) -> dict[str, set[Any]]:
    total_roads = {int(edge.road_id) for edge in edges}
    total_lanes = {(int(edge.road_id), int(edge.section_id), int(edge.lane_id)) for edge in edges}
    total_junctions: set[int] = set()
    for edge in edges:
        if edge.start_junction_id is not None:
            total_junctions.add(int(edge.start_junction_id))
        if edge.end_junction_id is not None:
            total_junctions.add(int(edge.end_junction_id))
        total_junctions.update(int(v) for v in edge.sampled_junction_ids)
        total_junctions.update(int(v) for v in edge.sampled_start_junction_ids)
        total_junctions.update(int(v) for v in edge.sampled_end_junction_ids)

    total_inner_j_roads = {int(edge.road_id) for edge in edges if atlas._edge_is_junction(edge)}
    return {
        "roads": total_roads,
        "lanes": total_lanes,
        "junctions": total_junctions,
        "inner_j_roads": total_inner_j_roads,
    }


def _collect_traversed(
    edge_ids: set[int],
    edges: list[atlas.EdgeGeometry],
) -> dict[str, set[Any]]:
    traversed_roads = {int(edges[edge_id].road_id) for edge_id in edge_ids}
    traversed_lanes = {
        (int(edges[edge_id].road_id), int(edges[edge_id].section_id), int(edges[edge_id].lane_id))
        for edge_id in edge_ids
    }

    traversed_junctions: set[int] = set()
    for edge_id in edge_ids:
        edge = edges[edge_id]
        if edge.start_junction_id is not None:
            traversed_junctions.add(int(edge.start_junction_id))
        if edge.end_junction_id is not None:
            traversed_junctions.add(int(edge.end_junction_id))
        traversed_junctions.update(int(v) for v in edge.sampled_junction_ids)
        traversed_junctions.update(int(v) for v in edge.sampled_start_junction_ids)
        traversed_junctions.update(int(v) for v in edge.sampled_end_junction_ids)

    traversed_inner_j_roads = {
        int(edges[edge_id].road_id)
        for edge_id in edge_ids
        if atlas._edge_is_junction(edges[edge_id])
    }
    return {
        "roads": traversed_roads,
        "lanes": traversed_lanes,
        "junctions": traversed_junctions,
        "inner_j_roads": traversed_inner_j_roads,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report atlas traversal coverage from any atlas output root"
    )
    parser.add_argument(
        "--atlas-root",
        type=Path,
        required=True,
        help="Atlas output root (contains TownXX/cpp-*/atlas_manifest.json)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/atlas_cpp/topology_cache"),
        help="Topology cache directory used to rebuild graph/edges",
    )
    parser.add_argument(
        "--solver",
        choices=["cpp-approx", "cpp-exact", "both"],
        default="both",
        help="Solver subset to report",
    )
    args = parser.parse_args()

    root = args.atlas_root
    cache_dir = args.cache_dir
    if not root.exists():
        raise SystemExit(f"ERROR: atlas root not found: {root}")
    if not cache_dir.exists():
        raise SystemExit(f"ERROR: cache dir not found: {cache_dir}")

    solvers = ["cpp-approx", "cpp-exact"] if args.solver == "both" else [args.solver]

    rows: list[dict[str, Any]] = []
    agg_total: dict[str, set[Any]] = {
        "roads": set(),
        "lanes": set(),
        "junctions": set(),
        "inner_j_roads": set(),
    }
    agg_traversed: dict[str, set[Any]] = {
        "roads": set(),
        "lanes": set(),
        "junctions": set(),
        "inner_j_roads": set(),
    }

    for town_dir in _town_dirs(root):
        town = town_dir.name
        cache_path = cache_dir / f"{town}.json"
        if not cache_path.exists():
            continue

        _, edges = atlas._load_topology_cache(cache_path)
        _, dg_all = atlas._build_graphs(edges)
        components = [set(comp) for comp in nx.weakly_connected_components(dg_all)]
        combos, transition_to_combo = atlas._build_junction_combo_catalog(edges)
        totals = _collect_totals(edges)

        for solver in solvers:
            manifest_path = town_dir / solver / "atlas_manifest.json"
            if not manifest_path.exists():
                continue

            manifest = json.loads(manifest_path.read_text())
            settings = manifest.get("settings", {})

            component_sequences: list[list[int]] = []
            for component_nodes in components:
                component_edge_ids = [
                    edge.edge_id
                    for edge in edges
                    if edge.u in component_nodes and edge.v in component_nodes
                ]
                if not component_edge_ids:
                    continue
                plan = atlas._plan_component(component_edge_ids, edges, solver)
                if plan.edge_sequence:
                    component_sequences.append(plan.edge_sequence)

            selected_edge_routes, _, _ = atlas._run_multi_pass_route_selection(
                component_sequences=component_sequences,
                edges=edges,
                all_combo_ids=set(combos.keys()),
                transition_to_combo=transition_to_combo,
                max_routes_per_town=int(settings.get("max_routes_per_town", 150)),
                combo_repeat_target=int(settings.get("combo_repeat_target", 1)),
                min_waypoints=int(settings.get("chunk_min_waypoints", 120)),
                max_waypoints=int(settings.get("chunk_max_waypoints", 260)),
                novelty_objective=bool(settings.get("novelty_objective", True)),
                anti_repeat_combos_per_route=bool(
                    settings.get("anti_repeat_combos_per_route", True)
                ),
                per_route_combo_repeat_limit=int(settings.get("per_route_combo_repeat_limit", 1)),
                route_end_min_waypoints_before_junction=int(
                    settings.get("route_end_min_waypoints_before_junction", 10)
                ),
                route_end_min_waypoints_after_junction=int(
                    settings.get("route_end_min_waypoints_after_junction", 2)
                ),
                multi_pass_max_passes=int(settings.get("multi_pass_max_passes", 8)),
                multi_pass_min_passes=int(settings.get("multi_pass_min_passes", 1)),
                target_combo_coverage_ratio=float(
                    settings.get("multi_pass_target_coverage_ratio", 1.0)
                ),
                target_repeat_coverage_ratio=float(
                    settings.get("multi_pass_target_repeat_coverage_ratio", 1.0)
                ),
                seed=int(settings.get("seed", 42)),
            )

            kept_edge_routes: list[list[int]] = []
            for edge_route in selected_edge_routes:
                route_points = atlas._annotate_road_options(
                    edge_sequence=edge_route,
                    edges=edges,
                    straight_threshold_deg=float(settings.get("straight_threshold_deg", 15.0)),
                    turn_threshold_deg=float(settings.get("turn_threshold_deg", 35.0)),
                )
                waypoint_count = int(len(route_points))
                min_wp = max(2, int(settings.get("chunk_min_waypoints", 120)))
                max_wp = int(settings.get("chunk_max_waypoints", 260))
                if waypoint_count < min_wp:
                    continue
                if max_wp > 0 and waypoint_count > max_wp:
                    continue
                kept_edge_routes.append(edge_route)

            traversed_edge_ids = {edge_id for route in kept_edge_routes for edge_id in route}
            traversed = _collect_traversed(traversed_edge_ids, edges)

            rows.append(
                {
                    "town": town,
                    "solver": solver,
                    "manifest_routes": int(manifest.get("route_count", 0)),
                    "recomputed_routes": int(len(kept_edge_routes)),
                    "roads": (
                        len(traversed["roads"]),
                        len(totals["roads"]),
                        _ratio(len(traversed["roads"]), len(totals["roads"])),
                    ),
                    "lanes": (
                        len(traversed["lanes"]),
                        len(totals["lanes"]),
                        _ratio(len(traversed["lanes"]), len(totals["lanes"])),
                    ),
                    "junctions": (
                        len(traversed["junctions"]),
                        len(totals["junctions"]),
                        _ratio(len(traversed["junctions"]), len(totals["junctions"])),
                    ),
                    "inner_j_roads": (
                        len(traversed["inner_j_roads"]),
                        len(totals["inner_j_roads"]),
                        _ratio(len(traversed["inner_j_roads"]), len(totals["inner_j_roads"])),
                    ),
                }
            )

            agg_total["roads"].update((town, value) for value in totals["roads"])
            agg_total["lanes"].update((town, *value) for value in totals["lanes"])
            agg_total["junctions"].update((town, value) for value in totals["junctions"])
            agg_total["inner_j_roads"].update((town, value) for value in totals["inner_j_roads"])

            agg_traversed["roads"].update((town, value) for value in traversed["roads"])
            agg_traversed["lanes"].update((town, *value) for value in traversed["lanes"])
            agg_traversed["junctions"].update((town, value) for value in traversed["junctions"])
            agg_traversed["inner_j_roads"].update(
                (town, value) for value in traversed["inner_j_roads"]
            )

    print(
        "town solver manifest_routes recomputed_routes "
        "roads traversed/total(cov) lanes traversed/total(cov) "
        "junctions traversed/total(cov) inner_j_roads traversed/total(cov)"
    )
    for row in rows:
        print(
            row["town"],
            row["solver"],
            row["manifest_routes"],
            row["recomputed_routes"],
            f"{row['roads'][0]}/{row['roads'][1]}({row['roads'][2]})",
            f"{row['lanes'][0]}/{row['lanes'][1]}({row['lanes'][2]})",
            f"{row['junctions'][0]}/{row['junctions'][1]}({row['junctions'][2]})",
            f"{row['inner_j_roads'][0]}/{row['inner_j_roads'][1]}({row['inner_j_roads'][2]})",
        )

    print("--- aggregate across all towns under atlas-root (town-scoped IDs) ---")
    for key in ("roads", "lanes", "junctions", "inner_j_roads"):
        n = len(agg_traversed[key])
        d = len(agg_total[key])
        print(key, n, "/", d, "cov", _ratio(n, d))


if __name__ == "__main__":
    main()
