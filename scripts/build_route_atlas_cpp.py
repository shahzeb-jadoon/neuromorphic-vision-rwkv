#!/usr/bin/env python3
"""Build standalone CPP-style route atlases for CARLA towns.

This script is intentionally decoupled from collection runtime code so route-bank
planning can be iterated independently before integration.

Key features:
- Two solver modes: cpp-approx and cpp-exact
- JSON route-bank + manifest output
- Input modes: carla-live, topology-cache, auto (prefer cache, fallback live)
- Town-sequential processing for memory safety
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, cast

import networkx as nx
import yaml

try:
    import carla

    CARLA_AVAILABLE = True
except ImportError:
    carla = None
    CARLA_AVAILABLE = False


logger = logging.getLogger("build_route_atlas_cpp")

RUN_STATE_SCHEMA_VERSION = 1
JUNCTION_COMBO_LEDGER_SCHEMA_VERSION = 1


@dataclass
class EdgeGeometry:
    """Directed lane-segment edge geometry for atlas planning."""

    edge_id: int
    segment_id: str
    u: int
    v: int
    road_id: int
    section_id: int
    lane_id: int
    end_road_id: int
    end_section_id: int
    end_lane_id: int
    start_is_junction: bool
    end_is_junction: bool
    start_junction_id: int | None
    end_junction_id: int | None
    weight_m: float
    points: list[tuple[float, float, float, float]]  # x,y,z,yaw


@dataclass
class ComponentPlan:
    """Planned traversal details for one weakly-connected component."""

    edge_sequence: list[int]
    original_edge_count: int
    duplicated_edge_count: int
    assignment_pairs: int
    eulerian: bool
    solver_fallback_used: bool


@dataclass
class JunctionCombo:
    """Junction transition combo represented by in-edge -> out-edge at a shared node."""

    combo_id: str
    junction_id: int
    via_node_id: int
    incoming_edge_id: int
    outgoing_edge_id: int
    incoming_segment_id: str
    outgoing_segment_id: str
    incoming_lane: tuple[int, int, int]
    outgoing_lane: tuple[int, int, int]


@dataclass
class RouteChunkCandidate:
    """Route candidate represented as edge-id chunk plus precomputed combo counts."""

    edge_ids: list[int]
    combo_hit_counts: dict[str, int]
    component_index: int
    component_chunk_index: int


def _normalize_town_name(name: str) -> str:
    base = str(name).split("/")[-1]
    if base.endswith("_Opt"):
        base = base[:-4]
    return base


def _angle_wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _distance_xyz(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    dz = float(a[2]) - float(b[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _polyline_length(points: list[tuple[float, float, float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    length = 0.0
    for i in range(1, len(points)):
        length += _distance_xyz(points[i - 1], points[i])
    return float(length)


def _load_config_towns(config_path: Path) -> list[str]:
    if not config_path.exists():
        return []
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    towns = list(cfg.get("towns", [])) + list(cfg.get("eval_towns", []))
    out: list[str] = []
    seen: set[str] = set()
    for town in towns:
        t = _normalize_town_name(str(town))
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _solver_modes(solver_arg: str) -> list[str]:
    return [solver_arg] if solver_arg != "both" else ["cpp-approx", "cpp-exact"]


def _combo_key(town: str, solver: str) -> str:
    return f"{town}::{solver}"


def _state_file_path(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output)
    if args.run_state_file is None:
        return output_dir / "atlas_run_state.json"
    return Path(args.run_state_file)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _load_run_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {
            "schema_version": RUN_STATE_SCHEMA_VERSION,
            "created_at_unix": int(time.time()),
            "updated_at_unix": int(time.time()),
            "combos": {},
        }

    payload_obj = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload_obj, dict):
        raise RuntimeError(f"Invalid run-state payload in {state_path}; expected JSON object")
    payload = cast(dict[str, Any], payload_obj)
    if int(payload.get("schema_version", 0)) != RUN_STATE_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported run-state schema in {state_path}; expected {RUN_STATE_SCHEMA_VERSION}"
        )
    if "combos" not in payload or not isinstance(payload["combos"], dict):
        payload["combos"] = {}
    payload.setdefault("updated_at_unix", int(time.time()))
    return payload


def _save_run_state(state_path: Path, state: dict[str, Any]) -> None:
    state["updated_at_unix"] = int(time.time())
    _atomic_write_json(state_path, state)


def _ensure_combo_entry(state: dict[str, Any], town: str, solver: str) -> dict[str, Any]:
    combos = state.setdefault("combos", {})
    key = _combo_key(town, solver)
    entry = combos.get(key)
    if not isinstance(entry, dict):
        entry = {
            "town": town,
            "solver": solver,
            "status": "pending",
            "attempts": 0,
        }
        combos[key] = entry
    else:
        entry["town"] = town
        entry["solver"] = solver
        entry.setdefault("status", "pending")
        entry.setdefault("attempts", 0)
    return entry


def _mark_combo_status(
    state: dict[str, Any],
    state_path: Path,
    town: str,
    solver: str,
    *,
    status: str,
    error: str | None = None,
) -> None:
    entry = _ensure_combo_entry(state, town, solver)
    now = int(time.time())
    entry["last_update_unix"] = now

    if status == "in_progress":
        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        entry["last_start_unix"] = now
        entry["error"] = None
    elif status == "completed":
        entry["completed_at_unix"] = now
        entry["error"] = None
    elif status == "failed":
        entry["error"] = (error or "").strip()[:1000]

    entry["status"] = status
    _save_run_state(state_path, state)


def _sync_completion_from_manifest(
    args: argparse.Namespace,
    state: dict[str, Any],
    state_path: Path,
    town: str,
    solver: str,
) -> bool:
    manifest_path = args.output / town / solver / "atlas_manifest.json"
    entry = _ensure_combo_entry(state, town, solver)

    if manifest_path.exists():
        if entry.get("status") != "completed":
            entry["status"] = "completed"
            entry["completed_at_unix"] = int(time.time())
            entry["error"] = None
            _save_run_state(state_path, state)
        return True

    if entry.get("status") == "completed":
        entry["status"] = "pending"
        entry["error"] = "manifest_missing"
        _save_run_state(state_path, state)
    return False


def _discover_towns_from_cache(cache_dir: Path) -> list[str]:
    towns: list[str] = []
    for p in sorted(cache_dir.glob("*.json")):
        towns.append(p.stem)
    return towns


def _discover_towns_from_carla(client: Any, include_opt: bool = False) -> list[str]:
    maps = client.get_available_maps()
    names: set[str] = set()
    for raw in maps:
        base = str(raw).split("/")[-1]
        if "Town" not in base:
            continue
        if not include_opt and base.endswith("_Opt"):
            continue
        names.add(_normalize_town_name(base))
    return sorted(names)


def _sample_segment_waypoints(
    wp_start: Any,
    wp_end: Any,
    sampling_resolution: float,
) -> list[Any]:
    """Sample points along a topology segment, mirroring planner logic."""
    lane_key = (int(wp_start.road_id), int(wp_start.section_id), int(wp_start.lane_id))
    sampled = [wp_start]
    visited = {int(wp_start.id)}
    current = wp_start
    step = max(0.5, float(sampling_resolution))

    for _ in range(1024):
        if current.transform.location.distance(wp_end.transform.location) <= max(0.8, step * 1.2):
            break
        nexts = current.next(step)
        if not nexts:
            break

        same_lane = [
            nxt
            for nxt in nexts
            if (int(nxt.road_id), int(nxt.section_id), int(nxt.lane_id)) == lane_key
        ]
        candidates = same_lane or nexts
        nxt = min(
            candidates, key=lambda cand: cand.transform.location.distance(wp_end.transform.location)
        )
        nxt_id = int(nxt.id)
        if nxt_id in visited:
            break
        visited.add(nxt_id)
        sampled.append(nxt)
        current = nxt

    if sampled[-1].transform.location.distance(wp_end.transform.location) > 0.1:
        sampled.append(wp_end)
    return sampled


def _build_edges_from_carla(world: Any, sampling_resolution: float) -> list[EdgeGeometry]:
    wmap = world.get_map()
    topology = list(wmap.get_topology())

    edges: list[EdgeGeometry] = []
    seen_segment_ids: set[str] = set()

    for wp_start, wp_end in topology:
        if wp_start.lane_type != carla.LaneType.Driving:
            continue
        if wp_end.lane_type != carla.LaneType.Driving:
            continue

        segment_id = f"{int(wp_start.id)}:{int(wp_end.id)}"
        if segment_id in seen_segment_ids:
            continue
        seen_segment_ids.add(segment_id)

        sampled = _sample_segment_waypoints(wp_start, wp_end, sampling_resolution)
        points = [
            (
                float(wp.transform.location.x),
                float(wp.transform.location.y),
                float(wp.transform.location.z),
                float(wp.transform.rotation.yaw),
            )
            for wp in sampled
        ]

        edges.append(
            EdgeGeometry(
                edge_id=len(edges),
                segment_id=segment_id,
                u=int(wp_start.id),
                v=int(wp_end.id),
                road_id=int(wp_start.road_id),
                section_id=int(wp_start.section_id),
                lane_id=int(wp_start.lane_id),
                end_road_id=int(wp_end.road_id),
                end_section_id=int(wp_end.section_id),
                end_lane_id=int(wp_end.lane_id),
                start_is_junction=bool(wp_start.is_junction),
                end_is_junction=bool(wp_end.is_junction),
                start_junction_id=(
                    int(wp_start.get_junction().id) if bool(wp_start.is_junction) else None
                ),
                end_junction_id=(
                    int(wp_end.get_junction().id) if bool(wp_end.is_junction) else None
                ),
                weight_m=max(0.01, _polyline_length(points)),
                points=points,
            )
        )

    return edges


def _edge_to_dict(edge: EdgeGeometry) -> dict[str, Any]:
    return {
        "edge_id": edge.edge_id,
        "segment_id": edge.segment_id,
        "u": edge.u,
        "v": edge.v,
        "road_id": edge.road_id,
        "section_id": edge.section_id,
        "lane_id": edge.lane_id,
        "end_road_id": edge.end_road_id,
        "end_section_id": edge.end_section_id,
        "end_lane_id": edge.end_lane_id,
        "start_is_junction": edge.start_is_junction,
        "end_is_junction": edge.end_is_junction,
        "start_junction_id": edge.start_junction_id,
        "end_junction_id": edge.end_junction_id,
        "weight_m": edge.weight_m,
        "points": [[p[0], p[1], p[2], p[3]] for p in edge.points],
    }


def _edge_from_dict(raw: dict[str, Any]) -> EdgeGeometry:
    start_junction_id = raw.get("start_junction_id")
    end_junction_id = raw.get("end_junction_id")
    if start_junction_id is not None:
        start_junction_id = int(start_junction_id)
    if end_junction_id is not None:
        end_junction_id = int(end_junction_id)

    return EdgeGeometry(
        edge_id=int(raw["edge_id"]),
        segment_id=str(raw["segment_id"]),
        u=int(raw["u"]),
        v=int(raw["v"]),
        road_id=int(raw.get("road_id", 0)),
        section_id=int(raw.get("section_id", 0)),
        lane_id=int(raw.get("lane_id", 0)),
        end_road_id=int(raw.get("end_road_id", raw.get("road_id", 0))),
        end_section_id=int(raw.get("end_section_id", raw.get("section_id", 0))),
        end_lane_id=int(raw.get("end_lane_id", raw.get("lane_id", 0))),
        start_is_junction=bool(raw.get("start_is_junction", start_junction_id is not None)),
        end_is_junction=bool(raw.get("end_is_junction", end_junction_id is not None)),
        start_junction_id=start_junction_id,
        end_junction_id=end_junction_id,
        weight_m=float(raw.get("weight_m", 0.01)),
        points=[
            (float(p[0]), float(p[1]), float(p[2]), float(p[3])) for p in raw.get("points", [])
        ],
    )


def _save_topology_cache(
    cache_path: Path, town: str, sampling_resolution: float, edges: list[EdgeGeometry]
) -> None:
    payload = {
        "town": town,
        "sampling_resolution_m": float(sampling_resolution),
        "edge_count": len(edges),
        "edges": [_edge_to_dict(edge) for edge in edges],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload), encoding="utf-8")


def _load_topology_cache(cache_path: Path) -> tuple[float, list[EdgeGeometry]]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    sampling_resolution = float(payload.get("sampling_resolution_m", 2.0))
    edges = [_edge_from_dict(raw) for raw in payload.get("edges", [])]
    return sampling_resolution, edges


def _build_graphs(
    edges: list[EdgeGeometry], edge_ids: set[int] | None = None
) -> tuple[nx.MultiDiGraph, nx.DiGraph]:
    mdg = nx.MultiDiGraph()
    dg = nx.DiGraph()

    for edge in edges:
        if edge_ids is not None and edge.edge_id not in edge_ids:
            continue

        mdg.add_edge(edge.u, edge.v, key=edge.edge_id, edge_id=edge.edge_id, weight_m=edge.weight_m)

        if dg.has_edge(edge.u, edge.v):
            if float(edge.weight_m) < float(dg[edge.u][edge.v]["weight_m"]):
                dg[edge.u][edge.v]["weight_m"] = float(edge.weight_m)
                dg[edge.u][edge.v]["edge_id"] = int(edge.edge_id)
        else:
            dg.add_edge(edge.u, edge.v, weight_m=float(edge.weight_m), edge_id=int(edge.edge_id))

    return mdg, dg


def _compute_imbalance(mdg: nx.MultiDiGraph) -> dict[int, int]:
    imbalance: dict[int, int] = {}
    for node in mdg.nodes:
        b = int(mdg.out_degree(node) - mdg.in_degree(node))
        if b != 0:
            imbalance[int(node)] = b
    return imbalance


def _approx_assignments(
    dg: nx.DiGraph, imbalance: dict[int, int]
) -> tuple[dict[tuple[int, int], int], bool]:
    """Greedy balancing assignment (approximate)."""
    supplies = {node: -b for node, b in imbalance.items() if b < 0}
    demands = {node: b for node, b in imbalance.items() if b > 0}
    assignments: dict[tuple[int, int], int] = defaultdict(int)

    unresolved = False

    while True:
        active_supplies = [(qty, node) for node, qty in supplies.items() if qty > 0]
        if not active_supplies:
            break

        _, source = max(active_supplies)
        lengths = nx.single_source_dijkstra_path_length(dg, source, weight="weight_m")

        while supplies[source] > 0:
            candidates = [
                (float(lengths[t]), t) for t, qty in demands.items() if qty > 0 and t in lengths
            ]
            if not candidates:
                unresolved = True
                supplies[source] = 0
                break

            _, sink = min(candidates, key=lambda item: item[0])
            amount = min(int(supplies[source]), int(demands[sink]))
            assignments[(source, sink)] += amount
            supplies[source] -= amount
            demands[sink] -= amount

    if any(qty > 0 for qty in demands.values()):
        unresolved = True

    return dict(assignments), unresolved


def _exact_assignments(
    dg: nx.DiGraph, imbalance: dict[int, int]
) -> tuple[dict[tuple[int, int], int], bool]:
    """Min-cost flow balancing assignment (exact for the modeled bipartite costs)."""
    supplies = {node: -b for node, b in imbalance.items() if b < 0}
    demands = {node: b for node, b in imbalance.items() if b > 0}

    flow_graph = nx.DiGraph()
    for node, qty in supplies.items():
        flow_graph.add_node(("S", int(node)), demand=-int(qty))
    for node, qty in demands.items():
        flow_graph.add_node(("T", int(node)), demand=int(qty))

    total_flow = int(sum(supplies.values()))

    for source in supplies:
        lengths = nx.single_source_dijkstra_path_length(dg, source, weight="weight_m")
        for sink in demands:
            if sink not in lengths:
                continue
            cost_int = max(1, int(round(float(lengths[sink]) * 100.0)))
            flow_graph.add_edge(
                ("S", int(source)), ("T", int(sink)), weight=cost_int, capacity=total_flow
            )

    try:
        _, flow = nx.network_simplex(flow_graph)
    except Exception:
        return {}, True

    assignments: dict[tuple[int, int], int] = {}
    for s_node, targets in flow.items():
        if s_node[0] != "S":
            continue
        source = int(s_node[1])
        for t_node, value in targets.items():
            if t_node[0] != "T":
                continue
            if int(value) <= 0:
                continue
            sink = int(t_node[1])
            assignments[(source, sink)] = int(value)

    unresolved = False
    sent: defaultdict[int, int] = defaultdict(int)
    received: defaultdict[int, int] = defaultdict(int)
    for (source, sink), value in assignments.items():
        sent[source] += value
        received[sink] += value

    for source, qty in supplies.items():
        if sent[source] != qty:
            unresolved = True
            break
    for sink, qty in demands.items():
        if received[sink] != qty:
            unresolved = True
            break

    return assignments, unresolved


def _pairwise(values: list[int]) -> Iterable[tuple[int, int]]:
    for idx in range(1, len(values)):
        yield int(values[idx - 1]), int(values[idx])


def _shortest_path_edge_ids(dg: nx.DiGraph, source: int, sink: int) -> list[int]:
    node_path = nx.shortest_path(dg, source, sink, weight="weight_m")
    edge_ids: list[int] = []
    for u, v in _pairwise([int(n) for n in node_path]):
        edge_ids.append(int(dg[u][v]["edge_id"]))
    return edge_ids


def _fallback_cover_walk(mdg_aug: nx.MultiDiGraph, dg: nx.DiGraph) -> list[int]:
    """Fallback edge-cover walk when a strict Eulerian circuit is unavailable."""
    unvisited: set[tuple[int, int, Any]] = set()
    out_map: dict[int, set[tuple[int, int, Any]]] = defaultdict(set)
    edge_id_lookup: dict[tuple[int, int, Any], int] = {}

    for u, v, k, data in mdg_aug.edges(keys=True, data=True):
        key = (int(u), int(v), k)
        unvisited.add(key)
        out_map[int(u)].add(key)
        edge_id_lookup[key] = int(data["edge_id"])

    if not unvisited:
        return []

    first = next(iter(unvisited))
    current = int(first[0])
    sequence: list[int] = []

    while unvisited:
        outgoing = out_map.get(current, set())
        if outgoing:
            edge_key = next(iter(outgoing))
            out_map[current].remove(edge_key)
            if edge_key in unvisited:
                unvisited.remove(edge_key)
                sequence.append(edge_id_lookup[edge_key])
            current = int(edge_key[1])
            continue

        targets = [node for node, edges in out_map.items() if edges]
        if not targets:
            break

        lengths = nx.single_source_dijkstra_path_length(dg, current, weight="weight_m")
        reachable = [(float(lengths[t]), t) for t in targets if t in lengths]
        if not reachable:
            current = targets[0]
            continue

        _, target = min(reachable, key=lambda item: item[0])
        path_nodes = [int(n) for n in nx.shortest_path(dg, current, target, weight="weight_m")]
        for u, v in _pairwise(path_nodes):
            sequence.append(int(dg[u][v]["edge_id"]))
        current = int(target)

    return sequence


def _plan_component(
    component_edge_ids: list[int],
    all_edges: list[EdgeGeometry],
    solver: str,
) -> ComponentPlan:
    component_edge_set = set(component_edge_ids)
    mdg, dg = _build_graphs(all_edges, component_edge_set)
    imbalance = _compute_imbalance(mdg)

    assignments: dict[tuple[int, int], int] = {}
    unresolved = False
    fallback_used = False

    if imbalance:
        if solver == "cpp-exact":
            assignments, unresolved = _exact_assignments(dg, imbalance)
            if unresolved:
                fallback_used = True
                assignments, unresolved = _approx_assignments(dg, imbalance)
        else:
            assignments, unresolved = _approx_assignments(dg, imbalance)

    duplicate_edge_ids: list[int] = []
    for (source, sink), amount in assignments.items():
        if amount <= 0:
            continue
        try:
            path_edge_ids = _shortest_path_edge_ids(dg, source, sink)
        except Exception:
            unresolved = True
            continue

        for _ in range(int(amount)):
            duplicate_edge_ids.extend(path_edge_ids)

    mdg_aug = nx.MultiDiGraph()
    for edge_id in component_edge_ids:
        edge = all_edges[edge_id]
        mdg_aug.add_edge(edge.u, edge.v, edge_id=int(edge.edge_id), weight_m=float(edge.weight_m))
    for edge_id in duplicate_edge_ids:
        edge = all_edges[edge_id]
        mdg_aug.add_edge(edge.u, edge.v, edge_id=int(edge.edge_id), weight_m=float(edge.weight_m))

    edge_sequence: list[int]
    if nx.is_eulerian(mdg_aug):
        circuit = list(nx.eulerian_circuit(mdg_aug, keys=True))
        edge_sequence = [int(mdg_aug[u][v][k]["edge_id"]) for u, v, k in circuit]
        eulerian = True
    else:
        edge_sequence = _fallback_cover_walk(mdg_aug, dg)
        eulerian = False

    if unresolved:
        fallback_used = True

    return ComponentPlan(
        edge_sequence=edge_sequence,
        original_edge_count=len(component_edge_ids),
        duplicated_edge_count=len(duplicate_edge_ids),
        assignment_pairs=len(assignments),
        eulerian=eulerian,
        solver_fallback_used=fallback_used,
    )


def _sequence_to_points(
    edge_sequence: list[int], edges: list[EdgeGeometry]
) -> list[tuple[float, float, float, float]]:
    points: list[tuple[float, float, float, float]] = []

    for edge_id in edge_sequence:
        edge_points = edges[edge_id].points
        if not edge_points:
            continue
        if not points:
            points.extend(edge_points)
            continue

        if _distance_xyz(points[-1], edge_points[0]) <= 0.05:
            points.extend(edge_points[1:])
        else:
            points.extend(edge_points)

    deduped: list[tuple[float, float, float, float]] = []
    for p in points:
        if deduped and _distance_xyz(deduped[-1], p) < 1e-6:
            continue
        deduped.append(p)
    return deduped


def _annotate_road_options(
    points: list[tuple[float, float, float, float]],
    straight_threshold_deg: float,
    turn_threshold_deg: float,
) -> list[dict[str, Any]]:
    if not points:
        return []

    annotated: list[dict[str, Any]] = []
    prev_yaw = float(points[0][3])
    annotated.append(
        {
            "x": float(points[0][0]),
            "y": float(points[0][1]),
            "z": float(points[0][2]),
            "yaw": float(points[0][3]),
            "road_option": "LANEFOLLOW",
        }
    )

    for idx in range(1, len(points)):
        x, y, z, yaw = points[idx]
        delta = _angle_wrap_deg(float(yaw) - prev_yaw)
        if abs(delta) <= straight_threshold_deg:
            cmd = "LANEFOLLOW"
        elif abs(delta) >= turn_threshold_deg:
            cmd = "RIGHT" if delta > 0.0 else "LEFT"
        else:
            cmd = "STRAIGHT"

        annotated.append(
            {
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "yaw": float(yaw),
                "road_option": cmd,
            }
        )
        prev_yaw = float(yaw)

    return annotated


def _route_length(route_points: list[dict[str, Any]]) -> float:
    if len(route_points) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(route_points)):
        a = route_points[i - 1]
        b = route_points[i]
        dx = float(a["x"]) - float(b["x"])
        dy = float(a["y"]) - float(b["y"])
        dz = float(a["z"]) - float(b["z"])
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return float(total)


def _split_route_points(
    route_points: list[dict[str, Any]],
    min_waypoints: int,
    max_waypoints: int,
    max_routes: int,
) -> list[list[dict[str, Any]]]:
    if not route_points:
        return []

    chunks: list[list[dict[str, Any]]] = []
    start = 0
    n = len(route_points)

    while start < n:
        end = min(start + max_waypoints, n)
        chunk = route_points[start:end]

        if len(chunk) < 2:
            break

        if len(chunk) < min_waypoints and chunks:
            chunks[-1].extend(chunk[1:])
            break

        chunks.append(chunk)

        if end >= n:
            break
        start = end - 1

    if max_routes > 0 and len(chunks) > max_routes:
        if max_routes == 1:
            chunks = [chunks[0]]
        else:
            idxs = sorted(
                {
                    int(round(i * (len(chunks) - 1) / float(max_routes - 1)))
                    for i in range(max_routes)
                }
            )
            chunks = [chunks[i] for i in idxs]

    return chunks


def _edge_waypoint_increment(previous_edge: EdgeGeometry | None, edge: EdgeGeometry) -> int:
    if not edge.points:
        return 1
    increment = int(len(edge.points))
    if previous_edge is not None and previous_edge.points:
        if _distance_xyz(previous_edge.points[-1], edge.points[0]) <= 0.05:
            increment = max(1, increment - 1)
    return max(1, increment)


def _split_edge_sequence_for_routes(
    edge_sequence: list[int],
    edges: list[EdgeGeometry],
    transition_to_combo: dict[tuple[int, int], str],
    min_waypoints: int,
    max_waypoints: int,
    max_routes: int,
    anti_repeat_combos_per_route: bool,
    per_route_combo_repeat_limit: int,
) -> list[list[int]]:
    if not edge_sequence:
        return []

    chunks: list[list[int]] = []
    idx = 0
    total_edges = len(edge_sequence)

    while idx < total_edges:
        if max_routes > 0 and len(chunks) >= max_routes:
            break

        chunk: list[int] = []
        combo_counts: defaultdict[str, int] = defaultdict(int)
        waypoint_count = 0
        previous_edge_id: int | None = None

        while idx < total_edges:
            edge_id = int(edge_sequence[idx])
            edge = edges[edge_id]

            combo_id: str | None = None
            if previous_edge_id is not None:
                combo_id = transition_to_combo.get((int(previous_edge_id), edge_id))
                if (
                    anti_repeat_combos_per_route
                    and per_route_combo_repeat_limit > 0
                    and combo_id is not None
                    and int(combo_counts[combo_id]) >= int(per_route_combo_repeat_limit)
                ):
                    break

            previous_edge = edges[previous_edge_id] if previous_edge_id is not None else None
            increment = _edge_waypoint_increment(previous_edge, edge)

            if chunk and max_waypoints > 0 and (waypoint_count + increment) > max_waypoints:
                break

            chunk.append(edge_id)
            waypoint_count += increment
            if combo_id is not None:
                combo_counts[combo_id] += 1

            previous_edge_id = edge_id
            idx += 1

            if max_waypoints > 0 and waypoint_count >= max_waypoints:
                break

        if not chunk:
            edge_id = int(edge_sequence[idx])
            chunk = [edge_id]
            waypoint_count = max(1, len(edges[edge_id].points))
            idx += 1

        if waypoint_count < min_waypoints and idx < total_edges:
            while idx < total_edges and waypoint_count < min_waypoints:
                edge_id = int(edge_sequence[idx])
                edge = edges[edge_id]

                combo_id = None
                if previous_edge_id is not None:
                    combo_id = transition_to_combo.get((int(previous_edge_id), edge_id))
                    if (
                        anti_repeat_combos_per_route
                        and per_route_combo_repeat_limit > 0
                        and combo_id is not None
                        and int(combo_counts[combo_id]) >= int(per_route_combo_repeat_limit)
                    ):
                        break

                previous_edge = edges[previous_edge_id] if previous_edge_id is not None else None
                increment = _edge_waypoint_increment(previous_edge, edge)

                if max_waypoints > 0 and (waypoint_count + increment) > max_waypoints:
                    break

                chunk.append(edge_id)
                waypoint_count += increment
                if combo_id is not None:
                    combo_counts[combo_id] += 1

                previous_edge_id = edge_id
                idx += 1

        if waypoint_count < min_waypoints and chunks:
            chunks[-1].extend(chunk)
        else:
            chunks.append(chunk)

    return chunks


def _combo_novelty_gain(
    candidate_combo_hits: dict[str, int],
    observed_combo_hits: dict[str, int],
    repeat_target: int,
) -> tuple[int, int]:
    novelty_gain = 0
    new_combo_count = 0

    for combo_id, count in candidate_combo_hits.items():
        count_i = int(count)
        current = int(observed_combo_hits.get(combo_id, 0))
        deficit = max(0, int(repeat_target) - current)
        novelty_gain += min(deficit, count_i)
        if current == 0 and count_i > 0:
            new_combo_count += 1

    return int(novelty_gain), int(new_combo_count)


def _build_route_chunk_candidates(
    component_sequences: list[list[int]],
    edges: list[EdgeGeometry],
    transition_to_combo: dict[tuple[int, int], str],
    min_waypoints: int,
    max_waypoints: int,
    max_routes_per_town: int,
    anti_repeat_combos_per_route: bool,
    per_route_combo_repeat_limit: int,
) -> list[RouteChunkCandidate]:
    candidates: list[RouteChunkCandidate] = []
    max_chunks_per_component = max_routes_per_town if max_routes_per_town > 0 else 0

    for component_index, edge_sequence in enumerate(component_sequences):
        chunks = _split_edge_sequence_for_routes(
            edge_sequence=edge_sequence,
            edges=edges,
            transition_to_combo=transition_to_combo,
            min_waypoints=int(min_waypoints),
            max_waypoints=int(max_waypoints),
            max_routes=int(max_chunks_per_component),
            anti_repeat_combos_per_route=bool(anti_repeat_combos_per_route),
            per_route_combo_repeat_limit=int(per_route_combo_repeat_limit),
        )
        for chunk_index, chunk in enumerate(chunks):
            combo_hit_counts = _count_observed_combo_hits(chunk, transition_to_combo)
            candidates.append(
                RouteChunkCandidate(
                    edge_ids=chunk,
                    combo_hit_counts=combo_hit_counts,
                    component_index=int(component_index),
                    component_chunk_index=int(chunk_index),
                )
            )

    return candidates


def _select_route_chunks_with_novelty(
    candidates: list[RouteChunkCandidate],
    max_routes: int,
    combo_repeat_target: int,
    novelty_objective: bool,
    initial_combo_hits: dict[str, int] | None = None,
) -> tuple[list[list[int]], dict[str, int], dict[str, Any]]:
    if not candidates:
        return (
            [],
            {},
            {
                "candidate_route_count": 0,
                "selected_route_count": 0,
                "novelty_enabled": bool(novelty_objective),
                "novelty_positive_route_count": 0,
            },
        )

    selected_chunks: list[list[int]] = []
    observed_combo_hits: defaultdict[str, int] = defaultdict(int)
    incremental_combo_hits: defaultdict[str, int] = defaultdict(int)
    if initial_combo_hits is not None:
        for combo_id, count in initial_combo_hits.items():
            observed_combo_hits[str(combo_id)] = int(count)
    used: set[int] = set()
    novelty_positive_route_count = 0
    selection_trace: list[dict[str, int]] = []

    while True:
        if max_routes > 0 and len(selected_chunks) >= max_routes:
            break

        remaining_indices = [idx for idx in range(len(candidates)) if idx not in used]
        if not remaining_indices:
            break

        selected_index: int
        chosen_gain = 0
        chosen_new_count = 0

        if novelty_objective:
            scored: list[tuple[int, int, int, int, int, int, int]] = []
            for idx in remaining_indices:
                candidate = candidates[idx]
                gain, new_count = _combo_novelty_gain(
                    candidate.combo_hit_counts,
                    observed_combo_hits,
                    combo_repeat_target,
                )
                scored.append(
                    (
                        int(gain),
                        int(new_count),
                        int(len(candidate.combo_hit_counts)),
                        int(-len(candidate.edge_ids)),
                        int(-candidate.component_index),
                        int(-candidate.component_chunk_index),
                        int(idx),
                    )
                )
            scored.sort(reverse=True)
            top = scored[0]
            selected_index = int(top[-1])
            chosen_gain = int(top[0])
            chosen_new_count = int(top[1])

            if chosen_gain <= 0:
                selected_index = int(min(remaining_indices))
                chosen_gain = 0
                chosen_new_count = 0
        else:
            selected_index = int(min(remaining_indices))

        used.add(selected_index)
        chosen = candidates[selected_index]
        selected_chunks.append(list(chosen.edge_ids))
        for combo_id, count in chosen.combo_hit_counts.items():
            observed_combo_hits[combo_id] += int(count)
            incremental_combo_hits[combo_id] += int(count)

        if chosen_gain > 0:
            novelty_positive_route_count += 1

        selection_trace.append(
            {
                "candidate_index": int(selected_index),
                "novelty_gain": int(chosen_gain),
                "new_combo_count": int(chosen_new_count),
                "component_index": int(chosen.component_index),
                "component_chunk_index": int(chosen.component_chunk_index),
            }
        )

    selection_stats: dict[str, Any] = {
        "candidate_route_count": int(len(candidates)),
        "selected_route_count": int(len(selected_chunks)),
        "novelty_enabled": bool(novelty_objective),
        "novelty_positive_route_count": int(novelty_positive_route_count),
        "selected_combo_observations": int(sum(incremental_combo_hits.values())),
        "selection_trace": selection_trace,
    }
    return selected_chunks, dict(incremental_combo_hits), selection_stats


def _variant_component_sequence(
    edge_sequence: list[int],
    pass_index: int,
    component_index: int,
    seed: int,
) -> list[int]:
    if not edge_sequence:
        return []
    if pass_index <= 0:
        return list(edge_sequence)

    variant = list(edge_sequence)
    mix = int(seed) + int(pass_index) * 1009 + int(component_index) * 131
    if mix % 2 == 1 and len(variant) > 1:
        variant = list(reversed(variant))

    rotation = int(mix % len(variant))
    if rotation > 0:
        variant = variant[rotation:] + variant[:rotation]
    return variant


def _combo_progress_metrics(
    all_combo_ids: set[str], combo_hits: dict[str, int], repeat_target: int
) -> dict[str, Any]:
    possible_combo_count = int(len(all_combo_ids))
    observed_combo_count = int(
        sum(1 for combo_id in all_combo_ids if int(combo_hits.get(combo_id, 0)) > 0)
    )
    repeat_target_combo_count = int(
        sum(
            1
            for combo_id in all_combo_ids
            if int(combo_hits.get(combo_id, 0)) >= int(repeat_target)
        )
    )

    return {
        "possible_combo_count": possible_combo_count,
        "observed_combo_count": observed_combo_count,
        "coverage_ratio": _safe_ratio(observed_combo_count, possible_combo_count),
        "repeat_target_combo_count": repeat_target_combo_count,
        "repeat_target_coverage_ratio": _safe_ratio(
            repeat_target_combo_count, possible_combo_count
        ),
    }


def _run_multi_pass_route_selection(
    component_sequences: list[list[int]],
    edges: list[EdgeGeometry],
    all_combo_ids: set[str],
    transition_to_combo: dict[tuple[int, int], str],
    *,
    max_routes_per_town: int,
    combo_repeat_target: int,
    min_waypoints: int,
    max_waypoints: int,
    novelty_objective: bool,
    anti_repeat_combos_per_route: bool,
    per_route_combo_repeat_limit: int,
    multi_pass_max_passes: int,
    multi_pass_min_passes: int,
    target_combo_coverage_ratio: float,
    target_repeat_coverage_ratio: float,
    seed: int,
) -> tuple[list[list[int]], dict[str, int], dict[str, Any]]:
    if not component_sequences:
        return (
            [],
            {},
            {
                "passes_executed": 0,
                "stop_reason": "no_components",
                "target_combo_coverage_ratio": float(target_combo_coverage_ratio),
                "target_repeat_coverage_ratio": float(target_repeat_coverage_ratio),
                "pass_summaries": [],
            },
        )

    selected_routes: list[list[int]] = []
    seen_route_signatures: set[tuple[int, ...]] = set()
    combo_hits: defaultdict[str, int] = defaultdict(int)
    pass_summaries: list[dict[str, Any]] = []

    max_passes = max(1, int(multi_pass_max_passes))
    min_passes = max(1, int(multi_pass_min_passes))
    stop_reason = "max_passes_reached"

    for pass_index in range(max_passes):
        if max_routes_per_town > 0:
            remaining_routes = int(max_routes_per_town - len(selected_routes))
            if remaining_routes <= 0:
                stop_reason = "max_routes_reached"
                break
        else:
            remaining_routes = 0

        pass_sequences: list[list[int]] = []
        for component_index, edge_sequence in enumerate(component_sequences):
            variant = _variant_component_sequence(
                edge_sequence=edge_sequence,
                pass_index=int(pass_index),
                component_index=int(component_index),
                seed=int(seed),
            )
            if variant:
                pass_sequences.append(variant)

        pass_candidates = _build_route_chunk_candidates(
            component_sequences=pass_sequences,
            edges=edges,
            transition_to_combo=transition_to_combo,
            min_waypoints=int(min_waypoints),
            max_waypoints=int(max_waypoints),
            max_routes_per_town=int(remaining_routes),
            anti_repeat_combos_per_route=bool(anti_repeat_combos_per_route),
            per_route_combo_repeat_limit=int(per_route_combo_repeat_limit),
        )

        unique_candidates: list[RouteChunkCandidate] = []
        for candidate in pass_candidates:
            signature = tuple(candidate.edge_ids)
            if signature in seen_route_signatures:
                continue
            unique_candidates.append(candidate)

        selected_in_pass, incremental_hits, pass_selection_stats = (
            _select_route_chunks_with_novelty(
                candidates=unique_candidates,
                max_routes=int(remaining_routes),
                combo_repeat_target=int(combo_repeat_target),
                novelty_objective=bool(novelty_objective),
                initial_combo_hits=dict(combo_hits),
            )
        )

        for combo_id, count in incremental_hits.items():
            combo_hits[combo_id] += int(count)

        for edge_route in selected_in_pass:
            seen_route_signatures.add(tuple(edge_route))
            selected_routes.append(edge_route)

        progress = _combo_progress_metrics(
            all_combo_ids=all_combo_ids,
            combo_hits=dict(combo_hits),
            repeat_target=int(combo_repeat_target),
        )

        pass_summaries.append(
            {
                "pass_index": int(pass_index),
                "selected_route_count": int(len(selected_in_pass)),
                "cumulative_route_count": int(len(selected_routes)),
                "progress": progress,
                "selection": pass_selection_stats,
            }
        )

        coverage_met = float(progress["coverage_ratio"]) >= float(target_combo_coverage_ratio)
        repeat_met = float(progress["repeat_target_coverage_ratio"]) >= float(
            target_repeat_coverage_ratio
        )
        if (pass_index + 1) >= min_passes and coverage_met and repeat_met:
            stop_reason = "combo_targets_met"
            break

        if not selected_in_pass:
            stop_reason = "no_additional_routes_available"
            break

    if (
        stop_reason == "max_passes_reached"
        and max_routes_per_town > 0
        and len(selected_routes) >= max_routes_per_town
    ):
        stop_reason = "max_routes_reached"

    final_progress = _combo_progress_metrics(
        all_combo_ids=all_combo_ids,
        combo_hits=dict(combo_hits),
        repeat_target=int(combo_repeat_target),
    )
    planning_stats: dict[str, Any] = {
        "passes_executed": int(len(pass_summaries)),
        "stop_reason": stop_reason,
        "target_combo_coverage_ratio": float(target_combo_coverage_ratio),
        "target_repeat_coverage_ratio": float(target_repeat_coverage_ratio),
        "final_progress": final_progress,
        "pass_summaries": pass_summaries,
    }
    return selected_routes, dict(combo_hits), planning_stats


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _build_junction_combo(incoming: EdgeGeometry, outgoing: EdgeGeometry) -> JunctionCombo | None:
    if incoming.v != outgoing.u:
        return None

    junction_id: int | None = None
    if incoming.end_is_junction and incoming.end_junction_id is not None:
        junction_id = int(incoming.end_junction_id)
    elif outgoing.start_is_junction and outgoing.start_junction_id is not None:
        junction_id = int(outgoing.start_junction_id)

    if junction_id is None:
        return None

    via_node_id = int(outgoing.u)
    incoming_lane = (
        int(incoming.end_road_id),
        int(incoming.end_section_id),
        int(incoming.end_lane_id),
    )
    outgoing_lane = (int(outgoing.road_id), int(outgoing.section_id), int(outgoing.lane_id))
    combo_id = (
        f"j{junction_id}|n{via_node_id}|" f"in:{incoming.segment_id}|out:{outgoing.segment_id}"
    )

    return JunctionCombo(
        combo_id=combo_id,
        junction_id=junction_id,
        via_node_id=via_node_id,
        incoming_edge_id=int(incoming.edge_id),
        outgoing_edge_id=int(outgoing.edge_id),
        incoming_segment_id=str(incoming.segment_id),
        outgoing_segment_id=str(outgoing.segment_id),
        incoming_lane=incoming_lane,
        outgoing_lane=outgoing_lane,
    )


def _build_junction_combo_catalog(
    edges: list[EdgeGeometry],
) -> tuple[dict[str, JunctionCombo], dict[tuple[int, int], str]]:
    incoming_by_node: dict[int, list[EdgeGeometry]] = defaultdict(list)
    outgoing_by_node: dict[int, list[EdgeGeometry]] = defaultdict(list)

    for edge in edges:
        incoming_by_node[int(edge.v)].append(edge)
        outgoing_by_node[int(edge.u)].append(edge)

    combos: dict[str, JunctionCombo] = {}
    transition_to_combo: dict[tuple[int, int], str] = {}

    shared_nodes = sorted(set(incoming_by_node.keys()).intersection(set(outgoing_by_node.keys())))
    for node in shared_nodes:
        incomings = incoming_by_node[node]
        outgoings = outgoing_by_node[node]
        for incoming in incomings:
            for outgoing in outgoings:
                combo = _build_junction_combo(incoming, outgoing)
                if combo is None:
                    continue
                combos.setdefault(combo.combo_id, combo)
                transition_to_combo[(int(incoming.edge_id), int(outgoing.edge_id))] = combo.combo_id

    return combos, transition_to_combo


def _count_observed_combo_hits(
    edge_sequence: list[int],
    transition_to_combo: dict[tuple[int, int], str],
) -> dict[str, int]:
    hits: dict[str, int] = defaultdict(int)
    for incoming_edge_id, outgoing_edge_id in _pairwise(edge_sequence):
        combo_id = transition_to_combo.get((int(incoming_edge_id), int(outgoing_edge_id)))
        if combo_id is None:
            continue
        hits[combo_id] += 1
    return dict(hits)


def _combo_summary(
    combos: dict[str, JunctionCombo],
    combo_hits: dict[str, int],
    repeat_target: int,
) -> dict[str, Any]:
    possible_combo_count = int(len(combos))
    observed_combo_count = int(
        sum(1 for combo_id in combos if int(combo_hits.get(combo_id, 0)) > 0)
    )
    satisfied_repeat_target = int(
        sum(1 for combo_id in combos if int(combo_hits.get(combo_id, 0)) >= int(repeat_target))
    )
    total_observations = int(sum(int(v) for v in combo_hits.values()))

    return {
        "schema_version": JUNCTION_COMBO_LEDGER_SCHEMA_VERSION,
        "repeat_target": int(repeat_target),
        "possible_combo_count": possible_combo_count,
        "observed_combo_count": observed_combo_count,
        "coverage_ratio": _safe_ratio(observed_combo_count, possible_combo_count),
        "repeat_target_combo_count": satisfied_repeat_target,
        "repeat_target_coverage_ratio": _safe_ratio(satisfied_repeat_target, possible_combo_count),
        "total_observations": total_observations,
    }


def _build_junction_combo_ledger_payload(
    town: str,
    solver: str,
    combos: dict[str, JunctionCombo],
    combo_hits: dict[str, int],
    repeat_target: int,
) -> dict[str, Any]:
    summary = _combo_summary(combos, combo_hits, repeat_target)

    combo_rows: list[dict[str, Any]] = []
    for combo_id in sorted(combos.keys()):
        combo = combos[combo_id]
        hit_count = int(combo_hits.get(combo_id, 0))
        combo_rows.append(
            {
                "combo_id": combo.combo_id,
                "junction_id": int(combo.junction_id),
                "via_node_id": int(combo.via_node_id),
                "incoming_edge_id": int(combo.incoming_edge_id),
                "outgoing_edge_id": int(combo.outgoing_edge_id),
                "incoming_segment_id": combo.incoming_segment_id,
                "outgoing_segment_id": combo.outgoing_segment_id,
                "incoming_lane": {
                    "road_id": int(combo.incoming_lane[0]),
                    "section_id": int(combo.incoming_lane[1]),
                    "lane_id": int(combo.incoming_lane[2]),
                },
                "outgoing_lane": {
                    "road_id": int(combo.outgoing_lane[0]),
                    "section_id": int(combo.outgoing_lane[1]),
                    "lane_id": int(combo.outgoing_lane[2]),
                },
                "hit_count": hit_count,
                "satisfied_repeat_target": bool(hit_count >= int(repeat_target)),
            }
        )

    return {
        "schema_version": JUNCTION_COMBO_LEDGER_SCHEMA_VERSION,
        "created_at_unix": int(time.time()),
        "town": town,
        "solver": solver,
        "summary": summary,
        "combos": combo_rows,
    }


def _write_solver_outputs(
    output_dir: Path,
    town: str,
    solver: str,
    routes: list[list[dict[str, Any]]],
    edge_stats: dict[str, Any],
    combo_stats: dict[str, Any],
    combo_ledger: dict[str, Any],
    settings: dict[str, Any],
    resume_existing_routes: bool,
) -> None:
    solver_dir = output_dir / town / solver
    routes_dir = solver_dir / "routes"
    routes_dir.mkdir(parents=True, exist_ok=True)

    route_records: list[dict[str, Any]] = []
    total_waypoints = 0
    total_length_m = 0.0

    for idx, route_points in enumerate(routes, start=1):
        route_id = f"{town}_{solver}_{idx:04d}"
        route_len = _route_length(route_points)
        route_path = routes_dir / f"route_{idx:04d}.json"
        waypoint_count = len(route_points)
        estimated_length_m = round(route_len, 3)

        reused_existing = False
        if resume_existing_routes and route_path.exists():
            try:
                existing = json.loads(route_path.read_text(encoding="utf-8"))
                if (
                    str(existing.get("town")) == town
                    and str(existing.get("solver")) == solver
                    and int(existing.get("waypoint_count", -1)) > 0
                ):
                    waypoint_count = int(existing.get("waypoint_count", waypoint_count))
                    estimated_length_m = float(
                        existing.get("estimated_length_m", estimated_length_m)
                    )
                    reused_existing = True
            except Exception:
                reused_existing = False

        if not reused_existing:
            route_payload = {
                "route_id": route_id,
                "town": town,
                "solver": solver,
                "waypoint_count": waypoint_count,
                "estimated_length_m": estimated_length_m,
                "waypoints": route_points,
            }
            route_path.write_text(json.dumps(route_payload), encoding="utf-8")

        total_waypoints += waypoint_count
        total_length_m += float(estimated_length_m)

        route_records.append(
            {
                "route_id": route_id,
                "file": str(route_path.relative_to(solver_dir)),
                "waypoint_count": int(waypoint_count),
                "estimated_length_m": round(float(estimated_length_m), 3),
            }
        )

    manifest = {
        "town": town,
        "solver": solver,
        "created_at_unix": int(time.time()),
        "route_count": len(routes),
        "total_waypoints": int(total_waypoints),
        "total_length_m": round(total_length_m, 3),
        "edge_stats": edge_stats,
        "combo_stats": combo_stats,
        "settings": settings,
        "routes": route_records,
    }

    combo_ledger_path = solver_dir / "junction_combo_ledger.json"
    combo_ledger_path.write_text(json.dumps(combo_ledger, indent=2), encoding="utf-8")

    manifest_path = solver_dir / "atlas_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _resolve_towns(args: argparse.Namespace, client: Any | None) -> list[str]:
    selected: list[str] = []

    if args.towns:
        selected.extend([_normalize_town_name(t) for t in args.towns])

    if args.all_towns:
        if client is not None:
            selected.extend(_discover_towns_from_carla(client, include_opt=args.include_opt_maps))
        else:
            selected.extend(_discover_towns_from_cache(args.cache_dir))

    if not selected:
        selected.extend(_load_config_towns(args.config))

    deduped: list[str] = []
    seen: set[str] = set()
    for town in selected:
        if not town:
            continue
        if town in seen:
            continue
        seen.add(town)
        deduped.append(town)

    return deduped


def _load_or_build_edges(
    args: argparse.Namespace,
    town: str,
    client: Any | None,
) -> tuple[float, list[EdgeGeometry], str]:
    cache_path = args.cache_dir / f"{town}.json"

    if args.input_mode in {"topology-cache", "auto"} and cache_path.exists():
        sampling_resolution, edges = _load_topology_cache(cache_path)
        return sampling_resolution, edges, "topology-cache"

    if args.input_mode == "topology-cache" and not cache_path.exists():
        raise FileNotFoundError(f"Topology cache missing: {cache_path}")

    if client is None:
        raise RuntimeError("CARLA client unavailable for live topology extraction")

    logger.info("Loading world %s for live topology extraction", town)
    world = client.load_world(town)
    sampling_resolution = float(args.sampling_resolution_m)
    edges = _build_edges_from_carla(world, sampling_resolution)

    if args.write_topology_cache:
        _save_topology_cache(cache_path, town, sampling_resolution, edges)

    return sampling_resolution, edges, "carla-live"


def _process_town(town: str, args: argparse.Namespace, client: Any | None) -> None:
    t0 = time.perf_counter()

    sampling_resolution, edges, resolved_input_mode = _load_or_build_edges(args, town, client)
    if not edges:
        logger.warning("Town %s has no driving edges; skipping", town)
        return

    all_mdg, all_dg = _build_graphs(edges)
    components = [set(comp) for comp in nx.weakly_connected_components(all_dg)]
    junction_combos, transition_to_combo = _build_junction_combo_catalog(edges)

    logger.info(
        "Town %s: edges=%d nodes=%d weak_components=%d input_mode=%s",
        town,
        len(edges),
        all_dg.number_of_nodes(),
        len(components),
        resolved_input_mode,
    )

    solver_modes = _solver_modes(args.solver)
    state_path = _state_file_path(args)
    state = _load_run_state(state_path) if args.resume else None

    for solver in solver_modes:
        if (
            args.resume
            and state is not None
            and _sync_completion_from_manifest(args, state, state_path, town, solver)
        ):
            logger.info("Skipping town %s solver=%s (already completed)", town, solver)
            continue

        if args.resume and state is not None:
            _mark_combo_status(state, state_path, town, solver, status="in_progress")

        logger.info("Planning town %s with solver=%s", town, solver)

        try:
            planned_components: list[ComponentPlan] = []
            component_sequences: list[list[int]] = []

            for component_nodes in components:
                component_edge_ids = [
                    edge.edge_id
                    for edge in edges
                    if edge.u in component_nodes and edge.v in component_nodes
                ]
                if not component_edge_ids:
                    continue

                plan = _plan_component(component_edge_ids, edges, solver)
                planned_components.append(plan)
                if plan.edge_sequence:
                    component_sequences.append(plan.edge_sequence)

            selected_edge_routes, selected_combo_hits, route_selection_stats = (
                _run_multi_pass_route_selection(
                    component_sequences=component_sequences,
                    edges=edges,
                    all_combo_ids=set(junction_combos.keys()),
                    transition_to_combo=transition_to_combo,
                    max_routes_per_town=int(args.max_routes_per_town),
                    combo_repeat_target=int(args.combo_repeat_target),
                    min_waypoints=int(args.chunk_min_waypoints),
                    max_waypoints=int(args.chunk_max_waypoints),
                    novelty_objective=bool(args.novelty_objective),
                    anti_repeat_combos_per_route=bool(args.anti_repeat_combos_per_route),
                    per_route_combo_repeat_limit=int(args.per_route_combo_repeat_limit),
                    multi_pass_max_passes=int(args.multi_pass_max_passes),
                    multi_pass_min_passes=int(args.multi_pass_min_passes),
                    target_combo_coverage_ratio=float(args.multi_pass_target_coverage_ratio),
                    target_repeat_coverage_ratio=float(
                        args.multi_pass_target_repeat_coverage_ratio
                    ),
                    seed=int(args.seed),
                )
            )

            routes: list[list[dict[str, Any]]] = []
            kept_edge_routes: list[list[int]] = []
            for edge_route in selected_edge_routes:
                stitched_points = _sequence_to_points(edge_route, edges)
                if len(stitched_points) < 2:
                    continue
                routes.append(
                    _annotate_road_options(
                        stitched_points,
                        straight_threshold_deg=float(args.straight_threshold_deg),
                        turn_threshold_deg=float(args.turn_threshold_deg),
                    )
                )
                kept_edge_routes.append(edge_route)

            full_edge_sequence = [
                edge_id for edge_route in kept_edge_routes for edge_id in edge_route
            ]

            combo_hits = _count_observed_combo_hits(full_edge_sequence, transition_to_combo)
            combo_ledger = _build_junction_combo_ledger_payload(
                town=town,
                solver=solver,
                combos=junction_combos,
                combo_hits=combo_hits,
                repeat_target=int(args.combo_repeat_target),
            )
            combo_stats = dict(combo_ledger["summary"])
            combo_stats["route_selection"] = route_selection_stats
            combo_stats["selected_combo_observations"] = int(sum(selected_combo_hits.values()))

            duplicated_total = sum(plan.duplicated_edge_count for plan in planned_components)
            eulerian_components = sum(1 for plan in planned_components if plan.eulerian)
            fallback_components = sum(1 for plan in planned_components if plan.solver_fallback_used)

            edge_stats = {
                "driving_edge_count": len(edges),
                "traversal_edge_steps": len(full_edge_sequence),
                "unique_traversed_edges": len(set(full_edge_sequence)),
                "weak_component_count": len(planned_components),
                "eulerian_component_count": int(eulerian_components),
                "solver_fallback_component_count": int(fallback_components),
                "duplicated_edge_count": int(duplicated_total),
                "approx_edge_coverage_ratio": round(
                    (len(set(full_edge_sequence)) / float(len(edges))) if edges else 0.0,
                    6,
                ),
            }

            settings = {
                "input_mode": resolved_input_mode,
                "sampling_resolution_m": float(sampling_resolution),
                "target_coverage": float(args.target_coverage),
                "chunk_min_waypoints": int(args.chunk_min_waypoints),
                "chunk_max_waypoints": int(args.chunk_max_waypoints),
                "max_routes_per_town": int(args.max_routes_per_town),
                "combo_repeat_target": int(args.combo_repeat_target),
                "novelty_objective": bool(args.novelty_objective),
                "anti_repeat_combos_per_route": bool(args.anti_repeat_combos_per_route),
                "per_route_combo_repeat_limit": int(args.per_route_combo_repeat_limit),
                "multi_pass_max_passes": int(args.multi_pass_max_passes),
                "multi_pass_min_passes": int(args.multi_pass_min_passes),
                "multi_pass_target_coverage_ratio": float(args.multi_pass_target_coverage_ratio),
                "multi_pass_target_repeat_coverage_ratio": float(
                    args.multi_pass_target_repeat_coverage_ratio
                ),
                "solver": solver,
                "seed": int(args.seed),
                "carla_host": args.carla_host,
                "carla_port": int(args.carla_port),
            }

            _write_solver_outputs(
                args.output,
                town,
                solver,
                routes,
                edge_stats,
                combo_stats,
                combo_ledger,
                settings,
                resume_existing_routes=bool(args.resume),
            )

            if args.resume and state is not None:
                _mark_combo_status(state, state_path, town, solver, status="completed")

            logger.info(
                (
                    "Town %s solver=%s done: routes=%d waypoints=%d "
                    "traversed_edges=%d unique_edges=%d combo_coverage=%.3f"
                ),
                town,
                solver,
                len(routes),
                sum(len(route) for route in routes),
                len(full_edge_sequence),
                len(set(full_edge_sequence)),
                float(combo_stats["coverage_ratio"]),
            )
        except Exception as exc:
            if args.resume and state is not None:
                _mark_combo_status(state, state_path, town, solver, status="failed", error=str(exc))
            logger.exception("Town %s solver=%s failed: %s", town, solver, exc)

    logger.info("Town %s complete in %.1fs", town, time.perf_counter() - t0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build standalone CPP route atlas artifacts")

    parser.add_argument(
        "--town",
        dest="town_groups",
        nargs="+",
        action="append",
        default=[],
        help="Town selection (supports one or many values per flag; repeatable)",
    )
    parser.add_argument("--all-towns", action="store_true", help="Process all discoverable towns")
    parser.add_argument(
        "--include-opt-maps",
        action="store_true",
        help="When discovering CARLA maps, include *_Opt map variants",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/collection.yaml"),
        help="Collection config used for default town list fallback",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/atlas_cpp"),
        help="Output root for atlas artifacts",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/atlas_cpp/topology_cache"),
        help="Topology cache directory",
    )

    parser.add_argument(
        "--input-mode",
        choices=["auto", "carla-live", "topology-cache"],
        default="auto",
        help="Input source: auto(cache->live), live only, or cache only",
    )
    parser.add_argument(
        "--write-topology-cache",
        action="store_true",
        help="Write topology cache after live extraction",
    )

    parser.add_argument("--carla-host", type=str, default="172.20.0.1", help="CARLA host")
    parser.add_argument("--carla-port", type=int, default=2000, help="CARLA port")

    parser.add_argument(
        "--solver",
        choices=["cpp-approx", "cpp-exact", "both"],
        default="both",
        help="Solver mode",
    )

    parser.add_argument(
        "--target-coverage", type=float, default=1.0, help="Target coverage metadata"
    )
    parser.add_argument(
        "--sampling-resolution-m", type=float, default=2.0, help="Segment sampling resolution"
    )
    parser.add_argument(
        "--chunk-min-waypoints", type=int, default=120, help="Min waypoints per route chunk"
    )
    parser.add_argument(
        "--chunk-max-waypoints", type=int, default=260, help="Max waypoints per route chunk"
    )
    parser.add_argument(
        "--max-routes-per-town",
        type=int,
        default=150,
        help="Upper bound on route files emitted per town/solver",
    )
    parser.add_argument(
        "--combo-repeat-target",
        type=int,
        default=1,
        help="Minimum hit target per junction-combo for ledger satisfaction metrics",
    )
    parser.add_argument(
        "--novelty-objective",
        dest="novelty_objective",
        action="store_true",
        default=True,
        help="Enable novelty-aware route chunk selection against combo repeat deficits",
    )
    parser.add_argument(
        "--no-novelty-objective",
        dest="novelty_objective",
        action="store_false",
        help="Disable novelty-aware route chunk selection",
    )
    parser.add_argument(
        "--anti-repeat-combos-per-route",
        dest="anti_repeat_combos_per_route",
        action="store_true",
        default=True,
        help="Prevent repeating the same junction-combo more than the per-route limit",
    )
    parser.add_argument(
        "--no-anti-repeat-combos-per-route",
        dest="anti_repeat_combos_per_route",
        action="store_false",
        help="Allow repeated junction-combos within a single route",
    )
    parser.add_argument(
        "--per-route-combo-repeat-limit",
        type=int,
        default=1,
        help="Maximum times a junction-combo may appear in one emitted route",
    )
    parser.add_argument(
        "--multi-pass-max-passes",
        type=int,
        default=8,
        help="Maximum diversified planning passes for route selection",
    )
    parser.add_argument(
        "--multi-pass-min-passes",
        type=int,
        default=1,
        help="Minimum number of planning passes before early stop checks apply",
    )
    parser.add_argument(
        "--multi-pass-target-coverage-ratio",
        type=float,
        default=1.0,
        help="Coverage ratio threshold used by multi-pass stop criteria",
    )
    parser.add_argument(
        "--multi-pass-target-repeat-coverage-ratio",
        type=float,
        default=1.0,
        help="Repeat-target coverage ratio threshold used by multi-pass stop criteria",
    )

    parser.add_argument(
        "--straight-threshold-deg",
        type=float,
        default=15.0,
        help="Yaw delta threshold for straight/lanefollow labeling",
    )
    parser.add_argument(
        "--turn-threshold-deg",
        type=float,
        default=35.0,
        help="Yaw delta threshold for left/right labeling",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume mode: skip completed town/solver combos and continue unfinished work",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume mode and rebuild all requested town/solver combos",
    )
    parser.add_argument(
        "--run-state-file",
        type=Path,
        default=None,
        help="Path to resume state JSON (default: <output>/atlas_run_state.json)",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()
    args.towns = [_normalize_town_name(town) for group in args.town_groups for town in group]
    args.combo_repeat_target = max(1, int(args.combo_repeat_target))
    args.per_route_combo_repeat_limit = max(1, int(args.per_route_combo_repeat_limit))
    args.multi_pass_max_passes = max(1, int(args.multi_pass_max_passes))
    args.multi_pass_min_passes = max(1, int(args.multi_pass_min_passes))
    args.multi_pass_target_coverage_ratio = min(
        1.0, max(0.0, float(args.multi_pass_target_coverage_ratio))
    )
    args.multi_pass_target_repeat_coverage_ratio = min(
        1.0, max(0.0, float(args.multi_pass_target_repeat_coverage_ratio))
    )
    return args


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    random.seed(int(args.seed))
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    client = None
    need_live = args.input_mode in {"auto", "carla-live"}
    if need_live:
        if not CARLA_AVAILABLE:
            if args.input_mode == "carla-live":
                raise RuntimeError("CARLA Python API not available for --input-mode=carla-live")
            logger.warning("CARLA unavailable; auto mode will use cache only")
        else:
            client = carla.Client(args.carla_host, int(args.carla_port))
            client.set_timeout(240.0)

    towns = _resolve_towns(args, client)
    if not towns:
        raise RuntimeError(
            "No towns resolved. Provide --town, --all-towns, or valid config/cache inputs"
        )

    if args.resume:
        state_path = _state_file_path(args)
        state = _load_run_state(state_path)
        state["last_requested_towns"] = list(towns)
        state["last_requested_solver_modes"] = _solver_modes(args.solver)
        _save_run_state(state_path, state)

    logger.info("Resolved towns: %s", ", ".join(towns))

    for town in towns:
        try:
            _process_town(town, args, client)
        except Exception as exc:
            logger.exception("Town %s failed: %s", town, exc)


if __name__ == "__main__":
    main()
