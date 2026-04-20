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
TOPOLOGY_CACHE_SCHEMA_VERSION = 4


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
    sampled_is_junction: bool
    sampled_junction_ids: list[int]
    sampled_start_junction_ids: list[int]
    sampled_end_junction_ids: list[int]
    road_option_hint: str
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
    towns = list(cfg.get("towns", []))
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


def _single_or_none(values: list[int]) -> int | None:
    if len(values) != 1:
        return None
    return int(values[0])


def _collect_sampled_junction_ids(
    sampled_waypoints: list[Any],
    boundary_window: int = 3,
) -> tuple[list[int], list[int], list[int]]:
    if not sampled_waypoints:
        return [], [], []

    n = len(sampled_waypoints)
    window = max(1, min(int(boundary_window), n))

    all_ids: set[int] = set()
    start_ids: set[int] = set()
    end_ids: set[int] = set()

    for idx, wp in enumerate(sampled_waypoints):
        if not bool(wp.is_junction):
            continue
        junction = wp.get_junction()
        if junction is None:
            continue

        junction_id = int(junction.id)
        all_ids.add(junction_id)

        if idx < window:
            start_ids.add(junction_id)
        if idx >= (n - window):
            end_ids.add(junction_id)

    return sorted(all_ids), sorted(start_ids), sorted(end_ids)


def _point_from_waypoint(wp: Any) -> tuple[float, float, float, float]:
    return (
        float(wp.transform.location.x),
        float(wp.transform.location.y),
        float(wp.transform.location.z),
        float(wp.transform.rotation.yaw),
    )


def _resolve_entry_edge_for_waypoint(
    waypoint: Any,
    edges: list[EdgeGeometry],
    entry_edge_ids_by_node: dict[int, list[int]],
    lane_edge_ids: dict[tuple[int, int, int], list[int]],
) -> int | None:
    direct = entry_edge_ids_by_node.get(int(waypoint.id), [])
    wp_point = _point_from_waypoint(waypoint)

    if len(direct) == 1:
        return int(direct[0])

    if direct:
        return int(
            min(
                direct,
                key=lambda edge_id: (
                    _distance_xyz(
                        wp_point,
                        edges[int(edge_id)].points[0] if edges[int(edge_id)].points else wp_point,
                    ),
                    int(edge_id),
                ),
            )
        )

    lane_key = (int(waypoint.road_id), int(waypoint.section_id), int(waypoint.lane_id))
    candidates = lane_edge_ids.get(lane_key, [])
    if not candidates:
        return None

    return int(
        min(
            candidates,
            key=lambda edge_id: (
                _distance_xyz(
                    wp_point,
                    edges[int(edge_id)].points[0] if edges[int(edge_id)].points else wp_point,
                ),
                int(edge_id),
            ),
        )
    )


def _lane_change_allowed(waypoint: Any, direction: str) -> bool:
    lane_change_flag = waypoint.lane_change
    if direction == "left":
        flag_allowed = lane_change_flag in (carla.LaneChange.Left, carla.LaneChange.Both)
        marking = waypoint.left_lane_marking
        marking_allowed = bool(
            marking is not None and (marking.lane_change & carla.LaneChange.Left)
        )
        return bool(flag_allowed or marking_allowed)

    if direction == "right":
        flag_allowed = lane_change_flag in (carla.LaneChange.Right, carla.LaneChange.Both)
        marking = waypoint.right_lane_marking
        marking_allowed = bool(
            marking is not None and (marking.lane_change & carla.LaneChange.Right)
        )
        return bool(flag_allowed or marking_allowed)

    return False


def _augment_edges_with_lane_changes(
    edges: list[EdgeGeometry],
    sampled_waypoints_by_edge_id: dict[int, list[Any]],
) -> list[EdgeGeometry]:
    if not edges:
        return edges

    entry_edge_ids_by_node: defaultdict[int, list[int]] = defaultdict(list)
    lane_edge_ids: defaultdict[tuple[int, int, int], list[int]] = defaultdict(list)
    for edge in edges:
        entry_edge_ids_by_node[int(edge.u)].append(int(edge.edge_id))
        lane_edge_ids[(int(edge.road_id), int(edge.section_id), int(edge.lane_id))].append(
            int(edge.edge_id)
        )

    added_edges: list[EdgeGeometry] = []
    seen_signatures: set[tuple[int, int, str]] = set()

    for source_edge_id, sampled_waypoints in sampled_waypoints_by_edge_id.items():
        source_edge = edges[int(source_edge_id)]
        if _edge_is_junction(source_edge):
            continue

        left_found = False
        right_found = False

        for source_wp in sampled_waypoints:
            if bool(source_wp.is_junction):
                break

            directions = (
                ("left", "CHANGELANELEFT", source_wp.get_left_lane, left_found),
                ("right", "CHANGELANERIGHT", source_wp.get_right_lane, right_found),
            )

            for direction, road_option, lane_fn, already_found in directions:
                if already_found:
                    continue
                if not _lane_change_allowed(source_wp, direction):
                    continue

                target_wp = lane_fn()
                if target_wp is None:
                    continue
                if target_wp.lane_type != carla.LaneType.Driving:
                    continue
                if bool(target_wp.is_junction):
                    continue
                if int(target_wp.road_id) != int(source_wp.road_id):
                    continue
                if int(target_wp.section_id) != int(source_wp.section_id):
                    continue

                target_edge_id = _resolve_entry_edge_for_waypoint(
                    waypoint=target_wp,
                    edges=edges,
                    entry_edge_ids_by_node=dict(entry_edge_ids_by_node),
                    lane_edge_ids=dict(lane_edge_ids),
                )
                if target_edge_id is None:
                    continue

                target_edge = edges[int(target_edge_id)]
                if _edge_is_junction(target_edge):
                    continue

                source_node_id = int(source_edge.u)
                target_node_id = int(target_edge.u)
                if source_node_id == target_node_id:
                    continue

                signature = (source_node_id, target_node_id, str(road_option))
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)

                source_point = (
                    source_edge.points[0] if source_edge.points else _point_from_waypoint(source_wp)
                )
                target_point = (
                    target_edge.points[0] if target_edge.points else _point_from_waypoint(target_wp)
                )
                lateral_dist = _distance_xyz(source_point, target_point)
                if lateral_dist <= 0.05:
                    continue

                new_edge_id = int(len(edges) + len(added_edges))
                added_edges.append(
                    EdgeGeometry(
                        edge_id=new_edge_id,
                        segment_id=(
                            f"lc:{source_edge.segment_id}->{target_edge.segment_id}:{direction}"
                        ),
                        u=source_node_id,
                        v=target_node_id,
                        road_id=int(source_edge.road_id),
                        section_id=int(source_edge.section_id),
                        lane_id=int(source_edge.lane_id),
                        end_road_id=int(target_edge.road_id),
                        end_section_id=int(target_edge.section_id),
                        end_lane_id=int(target_edge.lane_id),
                        start_is_junction=False,
                        end_is_junction=False,
                        start_junction_id=None,
                        end_junction_id=None,
                        sampled_is_junction=False,
                        sampled_junction_ids=[],
                        sampled_start_junction_ids=[],
                        sampled_end_junction_ids=[],
                        road_option_hint=str(road_option),
                        weight_m=max(0.5, float(lateral_dist)),
                        points=[source_point, target_point],
                    )
                )

                if direction == "left":
                    left_found = True
                elif direction == "right":
                    right_found = True

            if left_found and right_found:
                break

    if not added_edges:
        return edges
    return edges + added_edges


def _build_edges_from_carla(world: Any, sampling_resolution: float) -> list[EdgeGeometry]:
    wmap = world.get_map()
    topology = list(wmap.get_topology())

    edges: list[EdgeGeometry] = []
    sampled_waypoints_by_edge_id: dict[int, list[Any]] = {}
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
        sampled_junction_ids, sampled_start_junction_ids, sampled_end_junction_ids = (
            _collect_sampled_junction_ids(sampled)
        )
        points = [
            (
                float(wp.transform.location.x),
                float(wp.transform.location.y),
                float(wp.transform.location.z),
                float(wp.transform.rotation.yaw),
            )
            for wp in sampled
        ]

        edge_id = int(len(edges))
        edges.append(
            EdgeGeometry(
                edge_id=edge_id,
                segment_id=segment_id,
                u=int(wp_start.id),
                v=int(wp_end.id),
                road_id=int(wp_start.road_id),
                section_id=int(wp_start.section_id),
                lane_id=int(wp_start.lane_id),
                end_road_id=int(wp_end.road_id),
                end_section_id=int(wp_end.section_id),
                end_lane_id=int(wp_end.lane_id),
                start_is_junction=bool(wp_start.is_junction) or bool(sampled_start_junction_ids),
                end_is_junction=bool(wp_end.is_junction) or bool(sampled_end_junction_ids),
                start_junction_id=(
                    int(wp_start.get_junction().id)
                    if bool(wp_start.is_junction)
                    else _single_or_none(sampled_start_junction_ids)
                ),
                end_junction_id=(
                    int(wp_end.get_junction().id)
                    if bool(wp_end.is_junction)
                    else _single_or_none(sampled_end_junction_ids)
                ),
                sampled_is_junction=bool(sampled_junction_ids),
                sampled_junction_ids=sampled_junction_ids,
                sampled_start_junction_ids=sampled_start_junction_ids,
                sampled_end_junction_ids=sampled_end_junction_ids,
                road_option_hint="LANEFOLLOW",
                weight_m=max(0.01, _polyline_length(points)),
                points=points,
            )
        )
        sampled_waypoints_by_edge_id[edge_id] = sampled

    return _augment_edges_with_lane_changes(edges, sampled_waypoints_by_edge_id)


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
        "sampled_is_junction": edge.sampled_is_junction,
        "sampled_junction_ids": [int(v) for v in edge.sampled_junction_ids],
        "sampled_start_junction_ids": [int(v) for v in edge.sampled_start_junction_ids],
        "sampled_end_junction_ids": [int(v) for v in edge.sampled_end_junction_ids],
        "road_option_hint": edge.road_option_hint,
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

    sampled_junction_ids = [int(v) for v in raw.get("sampled_junction_ids", [])]
    sampled_start_junction_ids = [int(v) for v in raw.get("sampled_start_junction_ids", [])]
    sampled_end_junction_ids = [int(v) for v in raw.get("sampled_end_junction_ids", [])]
    road_option_hint = str(raw.get("road_option_hint", "LANEFOLLOW"))

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
        start_is_junction=bool(
            raw.get(
                "start_is_junction",
                start_junction_id is not None or bool(sampled_start_junction_ids),
            )
        ),
        end_is_junction=bool(
            raw.get(
                "end_is_junction",
                end_junction_id is not None or bool(sampled_end_junction_ids),
            )
        ),
        start_junction_id=start_junction_id,
        end_junction_id=end_junction_id,
        sampled_is_junction=bool(raw.get("sampled_is_junction", bool(sampled_junction_ids))),
        sampled_junction_ids=sampled_junction_ids,
        sampled_start_junction_ids=sampled_start_junction_ids,
        sampled_end_junction_ids=sampled_end_junction_ids,
        road_option_hint=road_option_hint,
        weight_m=float(raw.get("weight_m", 0.01)),
        points=[
            (float(p[0]), float(p[1]), float(p[2]), float(p[3])) for p in raw.get("points", [])
        ],
    )


def _save_topology_cache(
    cache_path: Path, town: str, sampling_resolution: float, edges: list[EdgeGeometry]
) -> None:
    payload = {
        "schema_version": TOPOLOGY_CACHE_SCHEMA_VERSION,
        "created_at_unix": int(time.time()),
        "town": town,
        "sampling_resolution_m": float(sampling_resolution),
        "edge_count": len(edges),
        "edges": [_edge_to_dict(edge) for edge in edges],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload), encoding="utf-8")


def _load_topology_cache(cache_path: Path) -> tuple[float, list[EdgeGeometry]]:
    payload_obj = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(payload_obj, dict):
        raise RuntimeError(f"Invalid topology cache payload in {cache_path}; expected JSON object")
    payload = cast(dict[str, Any], payload_obj)

    schema_version = int(payload.get("schema_version", 0))
    if schema_version != TOPOLOGY_CACHE_SCHEMA_VERSION:
        raise RuntimeError(
            (
                f"Unsupported topology cache schema in {cache_path}; "
                f"expected {TOPOLOGY_CACHE_SCHEMA_VERSION}, got {schema_version}"
            )
        )

    raw_edges = payload.get("edges", [])
    if not isinstance(raw_edges, list):
        raise RuntimeError(f"Invalid topology cache edges in {cache_path}; expected list")

    required_fields = (
        "start_is_junction",
        "end_is_junction",
        "start_junction_id",
        "end_junction_id",
        "sampled_is_junction",
        "sampled_junction_ids",
        "sampled_start_junction_ids",
        "sampled_end_junction_ids",
        "road_option_hint",
    )
    for idx, raw in enumerate(raw_edges):
        if not isinstance(raw, dict):
            raise RuntimeError(
                f"Invalid edge entry at index {idx} in {cache_path}; expected object"
            )
        missing = [field for field in required_fields if field not in raw]
        if missing:
            raise RuntimeError(
                f"Stale topology cache {cache_path}; missing edge fields: {', '.join(missing)}"
            )
        if not isinstance(raw.get("sampled_junction_ids"), list):
            raise RuntimeError(
                f"Invalid topology cache {cache_path}; sampled_junction_ids must be a list"
            )
        if not isinstance(raw.get("sampled_start_junction_ids"), list):
            raise RuntimeError(
                f"Invalid topology cache {cache_path}; sampled_start_junction_ids must be a list"
            )
        if not isinstance(raw.get("sampled_end_junction_ids"), list):
            raise RuntimeError(
                f"Invalid topology cache {cache_path}; sampled_end_junction_ids must be a list"
            )

    sampling_resolution = float(payload.get("sampling_resolution_m", 2.0))
    edges = [_edge_from_dict(cast(dict[str, Any], raw)) for raw in raw_edges]
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


def _edge_is_junction(edge: EdgeGeometry) -> bool:
    return bool(
        edge.start_is_junction
        or edge.end_is_junction
        or edge.sampled_is_junction
        or edge.sampled_junction_ids
    )


def _edge_start_junction_ids(edge: EdgeGeometry) -> set[int]:
    ids = {int(v) for v in edge.sampled_start_junction_ids}
    if edge.start_junction_id is not None:
        ids.add(int(edge.start_junction_id))
    return ids


def _edge_end_junction_ids(edge: EdgeGeometry) -> set[int]:
    ids = {int(v) for v in edge.sampled_end_junction_ids}
    if edge.end_junction_id is not None:
        ids.add(int(edge.end_junction_id))
    return ids


def _edge_entry_yaw(edge: EdgeGeometry) -> float:
    if edge.points:
        return float(edge.points[0][3])
    return 0.0


def _edge_exit_yaw(edge: EdgeGeometry) -> float:
    if edge.points:
        return float(edge.points[-1][3])
    return 0.0


def _classify_junction_run_command(
    edge_sequence: list[int],
    edges: list[EdgeGeometry],
    run_start: int,
    run_end: int,
    straight_threshold_deg: float,
    turn_threshold_deg: float,
) -> str:
    prev_idx = run_start - 1
    while prev_idx >= 0 and _edge_is_junction(edges[int(edge_sequence[prev_idx])]):
        prev_idx -= 1

    next_idx = run_end + 1
    while next_idx < len(edge_sequence) and _edge_is_junction(edges[int(edge_sequence[next_idx])]):
        next_idx += 1

    run_first = edges[int(edge_sequence[run_start])]
    run_last = edges[int(edge_sequence[run_end])]

    if prev_idx >= 0:
        entry_yaw = _edge_exit_yaw(edges[int(edge_sequence[prev_idx])])
    else:
        entry_yaw = _edge_entry_yaw(run_first)

    if next_idx < len(edge_sequence):
        exit_yaw = _edge_entry_yaw(edges[int(edge_sequence[next_idx])])
    else:
        exit_yaw = _edge_exit_yaw(run_last)

    # Parity guard with lane-graph planner semantics: if a junction run exits
    # onto the same lane lineage, treat it as straight regardless of local yaw
    # artifacts around junction polygons.
    if prev_idx >= 0 and next_idx < len(edge_sequence):
        prev_edge = edges[int(edge_sequence[prev_idx])]
        next_edge = edges[int(edge_sequence[next_idx])]
        same_start_lane = (
            int(prev_edge.road_id) == int(next_edge.road_id)
            and int(prev_edge.section_id) == int(next_edge.section_id)
            and int(prev_edge.lane_id) == int(next_edge.lane_id)
        )
        same_end_to_start_lane = (
            int(prev_edge.end_road_id) == int(next_edge.road_id)
            and int(prev_edge.end_section_id) == int(next_edge.section_id)
            and int(prev_edge.end_lane_id) == int(next_edge.lane_id)
        )
        if same_start_lane or same_end_to_start_lane:
            return "STRAIGHT"

    total_delta = _angle_wrap_deg(float(exit_yaw) - float(entry_yaw))
    if abs(total_delta) <= float(straight_threshold_deg):
        return "STRAIGHT"
    if abs(total_delta) < float(turn_threshold_deg):
        return "STRAIGHT"
    return "RIGHT" if total_delta > 0.0 else "LEFT"


def _edge_commands_for_route(
    edge_sequence: list[int],
    edges: list[EdgeGeometry],
    straight_threshold_deg: float,
    turn_threshold_deg: float,
) -> list[str]:
    if not edge_sequence:
        return []

    commands: list[str] = [
        str(edges[int(edge_id)].road_option_hint or "LANEFOLLOW") for edge_id in edge_sequence
    ]
    idx = 0
    while idx < len(edge_sequence):
        edge = edges[int(edge_sequence[idx])]
        if commands[idx] != "LANEFOLLOW":
            idx += 1
            continue

        if not _edge_is_junction(edge):
            commands[idx] = "LANEFOLLOW"
            idx += 1
            continue

        run_end = idx
        while run_end + 1 < len(edge_sequence):
            nxt_edge = edges[int(edge_sequence[run_end + 1])]
            if commands[run_end + 1] != "LANEFOLLOW" or not _edge_is_junction(nxt_edge):
                break
            run_end += 1

        run_cmd = _classify_junction_run_command(
            edge_sequence=edge_sequence,
            edges=edges,
            run_start=int(idx),
            run_end=int(run_end),
            straight_threshold_deg=float(straight_threshold_deg),
            turn_threshold_deg=float(turn_threshold_deg),
        )
        for j in range(idx, run_end + 1):
            commands[j] = run_cmd
        idx = run_end + 1

    return commands


def _annotate_road_options(
    edge_sequence: list[int],
    edges: list[EdgeGeometry],
    straight_threshold_deg: float,
    turn_threshold_deg: float,
) -> list[dict[str, Any]]:
    if not edge_sequence:
        return []

    edge_commands = _edge_commands_for_route(
        edge_sequence=edge_sequence,
        edges=edges,
        straight_threshold_deg=float(straight_threshold_deg),
        turn_threshold_deg=float(turn_threshold_deg),
    )

    annotated: list[dict[str, Any]] = []
    for idx, edge_id in enumerate(edge_sequence):
        edge = edges[int(edge_id)]
        points = edge.points
        if not points:
            continue

        start_offset = 0
        if annotated:
            last = annotated[-1]
            prev_tuple = (
                float(last["x"]),
                float(last["y"]),
                float(last["z"]),
                float(last["yaw"]),
            )
            if _distance_xyz(prev_tuple, points[0]) <= 0.05:
                start_offset = 1

        cmd = edge_commands[idx]
        for p in points[start_offset:]:
            annotated.append(
                {
                    "x": float(p[0]),
                    "y": float(p[1]),
                    "z": float(p[2]),
                    "yaw": float(p[3]),
                    "road_option": cmd,
                }
            )

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


def _edge_sequence_waypoint_count(edge_sequence: list[int], edges: list[EdgeGeometry]) -> int:
    waypoint_count = 0
    previous_edge: EdgeGeometry | None = None
    for edge_id in edge_sequence:
        edge = edges[int(edge_id)]
        waypoint_count += int(_edge_waypoint_increment(previous_edge, edge))
        previous_edge = edge
    return int(max(0, waypoint_count))


def _build_route_end_safety_context(
    edge_sequence: list[int],
    edges: list[EdgeGeometry],
) -> tuple[list[int], list[bool], list[int | None], list[int | None]]:
    n = len(edge_sequence)
    if n <= 0:
        return [0], [], [], []

    junction_flags = [_edge_is_junction(edges[int(edge_id)]) for edge_id in edge_sequence]

    increments: list[int] = []
    previous_edge: EdgeGeometry | None = None
    for edge_id in edge_sequence:
        edge = edges[int(edge_id)]
        increments.append(int(_edge_waypoint_increment(previous_edge, edge)))
        previous_edge = edge

    waypoint_prefix = [0]
    for increment in increments:
        waypoint_prefix.append(int(waypoint_prefix[-1]) + int(increment))

    prev_junction_run_end_at_or_before: list[int | None] = [None] * n
    last_run_end: int | None = None
    for idx in range(n):
        if junction_flags[idx] and (idx == (n - 1) or not junction_flags[idx + 1]):
            last_run_end = int(idx)
        prev_junction_run_end_at_or_before[idx] = last_run_end

    next_junction_run_start_after: list[int | None] = [None] * n
    next_run_start: int | None = None
    for idx in range(n - 1, -1, -1):
        next_junction_run_start_after[idx] = next_run_start
        if junction_flags[idx] and (idx == 0 or not junction_flags[idx - 1]):
            next_run_start = int(idx)

    return (
        waypoint_prefix,
        junction_flags,
        prev_junction_run_end_at_or_before,
        next_junction_run_start_after,
    )


def _waypoint_range_sum(waypoint_prefix: list[int], start_idx: int, end_idx: int) -> int:
    if start_idx > end_idx:
        return 0
    return int(waypoint_prefix[end_idx + 1] - waypoint_prefix[start_idx])


def _route_end_is_junction_safe(
    end_index: int,
    waypoint_prefix: list[int],
    junction_flags: list[bool],
    prev_junction_run_end_at_or_before: list[int | None],
    next_junction_run_start_after: list[int | None],
    min_waypoints_before_next_junction: int,
    min_waypoints_after_prev_junction: int,
) -> bool:
    if end_index < 0 or end_index >= len(junction_flags):
        return False

    # Never terminate within a junction-tagged edge.
    if bool(junction_flags[end_index]):
        return False

    waypoints_before_next_junction: int | None = None
    next_run_start = next_junction_run_start_after[end_index]
    if next_run_start is not None:
        waypoints_before_next_junction = _waypoint_range_sum(
            waypoint_prefix=waypoint_prefix,
            start_idx=int(end_index + 1),
            end_idx=int(next_run_start - 1),
        )

    waypoints_after_prev_junction: int | None = None
    prev_run_end = prev_junction_run_end_at_or_before[end_index]
    if prev_run_end is not None and end_index > int(prev_run_end):
        waypoints_after_prev_junction = _waypoint_range_sum(
            waypoint_prefix=waypoint_prefix,
            start_idx=int(prev_run_end + 1),
            end_idx=int(end_index),
        )

    # Endpoint is acceptable when it is either sufficiently before the next
    # junction entrance or sufficiently after the previous junction exit.
    if (
        waypoints_before_next_junction is not None
        and waypoints_before_next_junction < int(min_waypoints_before_next_junction)
        and (
            waypoints_after_prev_junction is None
            or waypoints_after_prev_junction < int(min_waypoints_after_prev_junction)
        )
    ):
        return False

    if (
        waypoints_after_prev_junction is not None
        and waypoints_after_prev_junction < int(min_waypoints_after_prev_junction)
        and (
            waypoints_before_next_junction is None
            or waypoints_before_next_junction < int(min_waypoints_before_next_junction)
        )
    ):
        return False

    return True


def _adjust_chunk_end_cuts_for_junction_clearance(
    initial_cuts: list[int],
    consumed_end_index: int,
    waypoint_prefix: list[int],
    junction_flags: list[bool],
    prev_junction_run_end_at_or_before: list[int | None],
    next_junction_run_start_after: list[int | None],
    min_waypoints_before_next_junction: int,
    min_waypoints_after_prev_junction: int,
) -> list[int]:
    if not initial_cuts:
        return []

    adjusted_cuts = [int(cut) for cut in initial_cuts]

    def _is_safe(cut: int) -> bool:
        return _route_end_is_junction_safe(
            end_index=int(cut),
            waypoint_prefix=waypoint_prefix,
            junction_flags=junction_flags,
            prev_junction_run_end_at_or_before=prev_junction_run_end_at_or_before,
            next_junction_run_start_after=next_junction_run_start_after,
            min_waypoints_before_next_junction=int(min_waypoints_before_next_junction),
            min_waypoints_after_prev_junction=int(min_waypoints_after_prev_junction),
        )

    idx = 0
    while idx < len(adjusted_cuts):
        left_bound = int(adjusted_cuts[idx - 1] + 1) if idx > 0 else 0
        adjusted_cuts[idx] = int(
            max(left_bound, min(int(adjusted_cuts[idx]), int(consumed_end_index)))
        )

        # Prefer moving the split forward; if needed, absorb single-edge next
        # chunks until a safe boundary is found.
        while not _is_safe(int(adjusted_cuts[idx])):
            if idx < len(adjusted_cuts) - 1:
                if int(adjusted_cuts[idx]) < int(adjusted_cuts[idx + 1] - 1):
                    adjusted_cuts[idx] = int(adjusted_cuts[idx] + 1)
                else:
                    adjusted_cuts[idx] = int(adjusted_cuts[idx + 1])
                    del adjusted_cuts[idx + 1]
            else:
                break

        # If forward movement cannot resolve the boundary, retreat backward.
        left_bound = int(adjusted_cuts[idx - 1] + 1) if idx > 0 else 0
        while not _is_safe(int(adjusted_cuts[idx])) and int(adjusted_cuts[idx]) > left_bound:
            adjusted_cuts[idx] = int(adjusted_cuts[idx] - 1)

        idx += 1

    return [int(cut) for cut in adjusted_cuts]


def _split_edge_sequence_for_routes(
    edge_sequence: list[int],
    edges: list[EdgeGeometry],
    transition_to_combo: dict[tuple[int, int], str],
    min_waypoints: int,
    max_waypoints: int,
    max_routes: int,
    anti_repeat_combos_per_route: bool,
    per_route_combo_repeat_limit: int,
    route_end_min_waypoints_before_junction: int,
    route_end_min_waypoints_after_junction: int,
) -> list[list[int]]:
    if not edge_sequence:
        return []

    chunks: list[list[int]] = []
    chunk_waypoint_counts: list[int] = []
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
            can_merge = bool(max_waypoints <= 0) or (
                int(chunk_waypoint_counts[-1]) + int(waypoint_count) <= int(max_waypoints)
            )
            if can_merge:
                chunks[-1].extend(chunk)
                chunk_waypoint_counts[-1] = int(chunk_waypoint_counts[-1]) + int(waypoint_count)
            else:
                chunks.append(chunk)
                chunk_waypoint_counts.append(int(waypoint_count))
        else:
            chunks.append(chunk)
            chunk_waypoint_counts.append(int(waypoint_count))

    if not chunks:
        return []

    initial_cuts: list[int] = []
    cursor = 0
    for chunk in chunks:
        if not chunk:
            continue
        cursor += len(chunk)
        initial_cuts.append(int(cursor - 1))

    if not initial_cuts:
        return []

    (
        waypoint_prefix,
        junction_flags,
        prev_junction_run_end_at_or_before,
        next_junction_run_start_after,
    ) = _build_route_end_safety_context(edge_sequence=edge_sequence, edges=edges)

    adjusted_cuts = _adjust_chunk_end_cuts_for_junction_clearance(
        initial_cuts=initial_cuts,
        consumed_end_index=int(initial_cuts[-1]),
        waypoint_prefix=waypoint_prefix,
        junction_flags=junction_flags,
        prev_junction_run_end_at_or_before=prev_junction_run_end_at_or_before,
        next_junction_run_start_after=next_junction_run_start_after,
        min_waypoints_before_next_junction=int(route_end_min_waypoints_before_junction),
        min_waypoints_after_prev_junction=int(route_end_min_waypoints_after_junction),
    )

    safe_cuts: list[int] = []
    for cut in adjusted_cuts:
        if _route_end_is_junction_safe(
            end_index=int(cut),
            waypoint_prefix=waypoint_prefix,
            junction_flags=junction_flags,
            prev_junction_run_end_at_or_before=prev_junction_run_end_at_or_before,
            next_junction_run_start_after=next_junction_run_start_after,
            min_waypoints_before_next_junction=int(route_end_min_waypoints_before_junction),
            min_waypoints_after_prev_junction=int(route_end_min_waypoints_after_junction),
        ):
            safe_cuts.append(int(cut))

    if not safe_cuts:
        return []

    adjusted_chunks: list[list[int]] = []
    start_idx = 0
    for cut in safe_cuts:
        end_idx = int(cut)
        if end_idx < start_idx:
            end_idx = int(start_idx)
        adjusted_chunks.append(
            [int(edge_id) for edge_id in edge_sequence[start_idx : int(end_idx + 1)]]
        )
        start_idx = int(end_idx + 1)

    contract_chunks: list[list[int]] = []
    for chunk in adjusted_chunks:
        waypoint_count = int(_edge_sequence_waypoint_count(chunk, edges))
        if waypoint_count < int(min_waypoints):
            continue
        if max_waypoints > 0 and waypoint_count > int(max_waypoints):
            continue
        contract_chunks.append(chunk)

    return contract_chunks


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
    route_end_min_waypoints_before_junction: int,
    route_end_min_waypoints_after_junction: int,
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
            route_end_min_waypoints_before_junction=int(route_end_min_waypoints_before_junction),
            route_end_min_waypoints_after_junction=int(route_end_min_waypoints_after_junction),
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
        "coverage_ratio": _coverage_ratio_or_complete(observed_combo_count, possible_combo_count),
        "repeat_target_combo_count": repeat_target_combo_count,
        "repeat_target_coverage_ratio": _coverage_ratio_or_complete(
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
    route_end_min_waypoints_before_junction: int,
    route_end_min_waypoints_after_junction: int,
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
            route_end_min_waypoints_before_junction=int(route_end_min_waypoints_before_junction),
            route_end_min_waypoints_after_junction=int(route_end_min_waypoints_after_junction),
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


def _coverage_ratio_or_complete(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return _safe_ratio(numerator, denominator)


def _build_junction_combo(incoming: EdgeGeometry, outgoing: EdgeGeometry) -> JunctionCombo | None:
    if incoming.v != outgoing.u:
        return None

    if not (_edge_is_junction(incoming) or _edge_is_junction(outgoing)):
        return None

    incoming_end_ids = _edge_end_junction_ids(incoming)
    outgoing_start_ids = _edge_start_junction_ids(outgoing)
    shared_boundary_ids = sorted(incoming_end_ids.intersection(outgoing_start_ids))

    junction_id: int | None = None
    if shared_boundary_ids:
        junction_id = int(shared_boundary_ids[0])
    elif incoming.end_is_junction and incoming.end_junction_id is not None:
        junction_id = int(incoming.end_junction_id)
    elif outgoing.start_is_junction and outgoing.start_junction_id is not None:
        junction_id = int(outgoing.start_junction_id)
    else:
        boundary_union = sorted(incoming_end_ids.union(outgoing_start_ids))
        if len(boundary_union) == 1:
            junction_id = int(boundary_union[0])

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
        "coverage_ratio": _coverage_ratio_or_complete(observed_combo_count, possible_combo_count),
        "repeat_target_combo_count": satisfied_repeat_target,
        "repeat_target_coverage_ratio": _coverage_ratio_or_complete(
            satisfied_repeat_target, possible_combo_count
        ),
        "total_observations": total_observations,
    }


def _junction_evidence_summary(edges: list[EdgeGeometry]) -> dict[str, int]:
    endpoint_count = 0
    sampled_count = 0
    endpoint_only_count = 0
    sampled_only_count = 0
    both_count = 0

    for edge in edges:
        endpoint_flag = bool(edge.start_is_junction or edge.end_is_junction)
        sampled_flag = bool(edge.sampled_is_junction or edge.sampled_junction_ids)
        if endpoint_flag:
            endpoint_count += 1
        if sampled_flag:
            sampled_count += 1
        if endpoint_flag and sampled_flag:
            both_count += 1
        elif endpoint_flag:
            endpoint_only_count += 1
        elif sampled_flag:
            sampled_only_count += 1

    return {
        "endpoint_junction_edge_count": int(endpoint_count),
        "sampled_junction_edge_count": int(sampled_count),
        "endpoint_only_junction_edge_count": int(endpoint_only_count),
        "sampled_only_junction_edge_count": int(sampled_only_count),
        "endpoint_and_sampled_junction_edge_count": int(both_count),
    }


def _lane_change_edge_summary(edges: list[EdgeGeometry]) -> dict[str, int]:
    left = sum(1 for edge in edges if str(edge.road_option_hint) == "CHANGELANELEFT")
    right = sum(1 for edge in edges if str(edge.road_option_hint) == "CHANGELANERIGHT")
    lane_change_total = int(left + right)
    return {
        "lane_change_edge_count": lane_change_total,
        "lane_change_left_edge_count": int(left),
        "lane_change_right_edge_count": int(right),
        "lanefollow_edge_count": int(
            sum(1 for edge in edges if str(edge.road_option_hint) == "LANEFOLLOW")
        ),
    }


def _validate_solver_outputs(
    *,
    edge_routes: list[list[int]],
    routes: list[list[dict[str, Any]]],
    edges: list[EdgeGeometry],
    combo_stats: dict[str, Any],
    resolved_input_mode: str,
    min_waypoints: int,
    max_waypoints: int,
    route_end_min_waypoints_before_junction: int,
    route_end_min_waypoints_after_junction: int,
    strict: bool,
) -> dict[str, Any]:
    allowed_commands = {
        "LANEFOLLOW",
        "STRAIGHT",
        "LEFT",
        "RIGHT",
        "CHANGELANELEFT",
        "CHANGELANERIGHT",
    }

    command_counts: defaultdict[str, int] = defaultdict(int)
    invalid_command_counts: defaultdict[str, int] = defaultdict(int)

    oversized_routes = 0
    undersized_routes = 0
    unsafe_route_end_count = 0
    junction_routes = 0
    junction_routes_without_turn_command = 0
    lane_change_routes = 0
    lane_change_routes_without_lane_change_command = 0

    for edge_route, route_points in zip(edge_routes, routes):
        waypoint_count = int(len(route_points))
        if waypoint_count < int(min_waypoints):
            undersized_routes += 1
        if max_waypoints > 0 and waypoint_count > int(max_waypoints):
            oversized_routes += 1

        labels = [str(point.get("road_option", "")) for point in route_points]
        for label in labels:
            command_counts[label] += 1
            if label not in allowed_commands:
                invalid_command_counts[label] += 1

        has_junction_edge = any(_edge_is_junction(edges[int(edge_id)]) for edge_id in edge_route)
        if has_junction_edge:
            junction_routes += 1
            if not any(label in {"STRAIGHT", "LEFT", "RIGHT"} for label in labels):
                junction_routes_without_turn_command += 1

        has_lane_change_edge = any(
            str(edges[int(edge_id)].road_option_hint) in {"CHANGELANELEFT", "CHANGELANERIGHT"}
            for edge_id in edge_route
        )
        if has_lane_change_edge:
            lane_change_routes += 1
            if not any(label in {"CHANGELANELEFT", "CHANGELANERIGHT"} for label in labels):
                lane_change_routes_without_lane_change_command += 1

        if edge_route:
            end_edge = edges[int(edge_route[-1])]
            if _edge_is_junction(end_edge):
                unsafe_route_end_count += 1

    total_commands = int(sum(command_counts.values()))
    lanefollow_commands = int(command_counts.get("LANEFOLLOW", 0))
    non_lanefollow_commands = int(total_commands - lanefollow_commands)

    has_any_junction_edge = any(_edge_is_junction(edge) for edge in edges)
    combo_possible_count = int(combo_stats.get("possible_combo_count", 0))
    combo_space_empty_live = bool(
        resolved_input_mode == "carla-live" and has_any_junction_edge and combo_possible_count <= 0
    )

    no_oversized_routes = bool(oversized_routes == 0)
    no_undersized_routes = bool(undersized_routes == 0)
    no_unsafe_route_endcuts = bool(unsafe_route_end_count == 0)
    non_empty_combo_space_live = bool(not combo_space_empty_live)

    command_distribution_sane = bool(
        not invalid_command_counts
        and (junction_routes == 0 or junction_routes_without_turn_command == 0)
        and (lane_change_routes == 0 or lane_change_routes_without_lane_change_command == 0)
        and (junction_routes == 0 or non_lanefollow_commands > 0)
    )

    gate_errors: list[str] = []
    if not no_oversized_routes:
        gate_errors.append(f"oversized_routes={oversized_routes}")
    if not no_undersized_routes:
        gate_errors.append(f"undersized_routes={undersized_routes}")
    if not no_unsafe_route_endcuts:
        gate_errors.append(f"unsafe_route_endcuts={unsafe_route_end_count}")
    if not non_empty_combo_space_live:
        gate_errors.append("live_combo_space_empty")
    if not command_distribution_sane:
        gate_errors.append("command_distribution_sanity_failed")

    validation_summary: dict[str, Any] = {
        "strict": bool(strict),
        "route_count": int(len(routes)),
        "total_commands": int(total_commands),
        "command_counts": {k: int(v) for k, v in sorted(command_counts.items())},
        "invalid_command_counts": {k: int(v) for k, v in sorted(invalid_command_counts.items())},
        "junction_routes": int(junction_routes),
        "junction_routes_without_turn_command": int(junction_routes_without_turn_command),
        "lane_change_routes": int(lane_change_routes),
        "lane_change_routes_without_lane_change_command": int(
            lane_change_routes_without_lane_change_command
        ),
        "oversized_routes": int(oversized_routes),
        "undersized_routes": int(undersized_routes),
        "unsafe_route_endcuts": int(unsafe_route_end_count),
        "combo_possible_count": int(combo_possible_count),
        "has_any_junction_edge": bool(has_any_junction_edge),
        "non_lanefollow_commands": int(non_lanefollow_commands),
        "gates": {
            "no_oversized_routes": no_oversized_routes,
            "no_undersized_routes": no_undersized_routes,
            "no_unsafe_route_endcuts": no_unsafe_route_endcuts,
            "non_empty_combo_space_live": non_empty_combo_space_live,
            "command_distribution_sane": command_distribution_sane,
        },
        "errors": gate_errors,
    }

    if strict and gate_errors:
        raise RuntimeError("Validation gates failed: " + ", ".join(gate_errors))

    return validation_summary


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
    validation_gates: dict[str, Any],
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
        "validation_gates": validation_gates,
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
        try:
            sampling_resolution, edges = _load_topology_cache(cache_path)
            return sampling_resolution, edges, "topology-cache"
        except Exception as exc:
            if args.input_mode == "topology-cache":
                raise RuntimeError(
                    (
                        f"Topology cache {cache_path} is stale/invalid ({exc}). "
                        "Rebuild cache using --input-mode carla-live --write-topology-cache"
                    )
                ) from exc
            logger.warning(
                (
                    "Ignoring stale/invalid topology cache %s (%s); "
                    "falling back to live extraction"
                ),
                cache_path,
                exc,
            )

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
                    route_end_min_waypoints_before_junction=int(
                        args.route_end_min_waypoints_before_junction
                    ),
                    route_end_min_waypoints_after_junction=int(
                        args.route_end_min_waypoints_after_junction
                    ),
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
            dropped_undersized_routes = 0
            dropped_oversized_routes = 0
            for edge_route in selected_edge_routes:
                route_points = _annotate_road_options(
                    edge_sequence=edge_route,
                    edges=edges,
                    straight_threshold_deg=float(args.straight_threshold_deg),
                    turn_threshold_deg=float(args.turn_threshold_deg),
                )
                waypoint_count = int(len(route_points))
                min_waypoints = max(2, int(args.chunk_min_waypoints))
                max_waypoints = int(args.chunk_max_waypoints)
                if waypoint_count < min_waypoints:
                    dropped_undersized_routes += 1
                    continue
                if max_waypoints > 0 and waypoint_count > max_waypoints:
                    dropped_oversized_routes += 1
                    continue
                routes.append(route_points)
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

            validation_gates = _validate_solver_outputs(
                edge_routes=kept_edge_routes,
                routes=routes,
                edges=edges,
                combo_stats=combo_stats,
                resolved_input_mode=resolved_input_mode,
                min_waypoints=max(2, int(args.chunk_min_waypoints)),
                max_waypoints=int(args.chunk_max_waypoints),
                route_end_min_waypoints_before_junction=int(
                    args.route_end_min_waypoints_before_junction
                ),
                route_end_min_waypoints_after_junction=int(
                    args.route_end_min_waypoints_after_junction
                ),
                strict=bool(args.strict_validation_gates),
            )

            duplicated_total = sum(plan.duplicated_edge_count for plan in planned_components)
            eulerian_components = sum(1 for plan in planned_components if plan.eulerian)
            fallback_components = sum(1 for plan in planned_components if plan.solver_fallback_used)

            edge_stats = {
                "atlas_edge_count": len(edges),
                "selected_edge_route_count_pre_contract": int(len(selected_edge_routes)),
                "selected_edge_route_count_post_contract": int(len(kept_edge_routes)),
                "dropped_undersized_routes": int(dropped_undersized_routes),
                "dropped_oversized_routes": int(dropped_oversized_routes),
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
                "junction_evidence": _junction_evidence_summary(edges),
                "lane_change_edges": _lane_change_edge_summary(edges),
            }

            settings = {
                "input_mode": resolved_input_mode,
                "sampling_resolution_m": float(sampling_resolution),
                "topology_cache_schema_version": int(TOPOLOGY_CACHE_SCHEMA_VERSION),
                "target_coverage": float(args.target_coverage),
                "chunk_min_waypoints": int(args.chunk_min_waypoints),
                "chunk_max_waypoints": int(args.chunk_max_waypoints),
                "route_end_min_waypoints_before_junction": int(
                    args.route_end_min_waypoints_before_junction
                ),
                "route_end_min_waypoints_after_junction": int(
                    args.route_end_min_waypoints_after_junction
                ),
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
                "strict_validation_gates": bool(args.strict_validation_gates),
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
                validation_gates,
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
            if validation_gates.get("errors"):
                logger.warning(
                    "Town %s solver=%s validation gate warnings: %s",
                    town,
                    solver,
                    validation_gates.get("errors"),
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
        "--route-end-min-waypoints-before-junction",
        type=int,
        default=10,
        help=(
            "Minimum non-junction waypoints required between a route end and the next "
            "junction entrance"
        ),
    )
    parser.add_argument(
        "--route-end-min-waypoints-after-junction",
        type=int,
        default=2,
        help=(
            "Minimum non-junction waypoints required after the previous junction before "
            "allowing a route end"
        ),
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
    parser.add_argument(
        "--strict-validation-gates",
        dest="strict_validation_gates",
        action="store_true",
        default=True,
        help="Fail a town/solver run when route-atlas validation gates are violated",
    )
    parser.add_argument(
        "--no-strict-validation-gates",
        dest="strict_validation_gates",
        action="store_false",
        help="Log validation gate violations without failing the run",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()
    args.towns = [_normalize_town_name(town) for group in args.town_groups for town in group]
    args.combo_repeat_target = max(1, int(args.combo_repeat_target))
    args.per_route_combo_repeat_limit = max(1, int(args.per_route_combo_repeat_limit))
    args.route_end_min_waypoints_before_junction = max(
        0, int(args.route_end_min_waypoints_before_junction)
    )
    args.route_end_min_waypoints_after_junction = max(
        0, int(args.route_end_min_waypoints_after_junction)
    )
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

    large_campaign_towns = {"Town12", "Town15"}
    large_selected = sorted(town for town in towns if town in large_campaign_towns)
    if large_selected and args.input_mode in {"carla-live", "auto"}:
        logger.warning(
            (
                "Large-map campaign includes %s using %s mode. For repeated all-town "
                "campaigns, run one live pass with --write-topology-cache, then prefer "
                "--input-mode topology-cache for improved memory stability."
            ),
            ", ".join(large_selected),
            args.input_mode,
        )
    if large_selected and args.input_mode == "topology-cache":
        logger.info(
            "Using topology-cache mode for large maps (%s): memory-stable campaign mode active",
            ", ".join(large_selected),
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
