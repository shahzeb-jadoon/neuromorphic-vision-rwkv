"""Lane-segment graph route planner for CARLA.

This planner intentionally models *lane segments* (not per-metre waypoints)
to reduce graph cardinality while preserving lane-level legality:

* Each topology segment start waypoint is a node.
* Forward edges follow lane continuity across map topology.
* Lateral edges model legal lane changes on non-junction lanes.

Important junction insight used by this implementation:

Roads inside a junction act as connector roads that link incoming outside
roads to outgoing outside roads. The planner explicitly tracks these
connector links in diagnostics for forensic analysis.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple

import carla

from agents.navigation.local_planner import RoadOption

logger = logging.getLogger(__name__)


GRAPH_CACHE_VERSION = 2
GRAPH_CACHE_LOGIC_TAG = "lane-graph-v2"


class EdgeKind(Enum):
    """Internal graph edge categories."""

    FORWARD = "forward"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"


@dataclass(frozen=True)
class Edge:
    """Directed graph edge between lane segments."""

    to_id: str
    cost: float
    distance_m: float
    kind: EdgeKind


@dataclass(frozen=True)
class SegmentNode:
    """Lane-segment graph node.

    ``segment_id`` is anchored to CARLA's start waypoint ID for the segment.
    """

    segment_id: str
    road_id: int
    section_id: int
    lane_id: int
    is_junction: bool
    entry_xyz: Tuple[float, float, float]
    exit_xyz: Tuple[float, float, float]
    entry_yaw: float
    exit_yaw: float
    centerline: Tuple[Tuple[float, float, float, float], ...]
    length_m: float


class LaneGraphPlanner:
    """Lane-segment route planner that returns ``(Waypoint, RoadOption)`` traces.

    Parameters
    ----------
    wmap:
        CARLA map object.
    sampling_resolution:
        Spacing (metres) used for centerline sampling and successor probing.
    lane_change_base_cost:
        Base additive cost for lane-change edges. Higher values reduce
        unnecessary lane changes.
    late_lane_change_penalty:
        Additional additive cost for lane changes close to junctions.
    lane_change_freeze_distance_m:
        Distance-to-junction threshold under which lane changes are heavily
        penalized.
    turn_threshold_deg:
        Absolute heading-change threshold to classify LEFT/RIGHT vs STRAIGHT.
    max_search_expansions:
        Hard cap for A* node expansions to prevent pathological searches.
    cache_dir:
        Optional directory for serialized graph caches.
    """

    def __init__(
        self,
        wmap: carla.Map,
        sampling_resolution: float = 1.0,
        lane_change_base_cost: float = 25.0,
        late_lane_change_penalty: float = 300.0,
        lane_change_freeze_distance_m: float = 30.0,
        turn_threshold_deg: float = 35.0,
        max_search_expansions: int = 250_000,
        cache_dir: str | Path | None = None,
    ) -> None:
        self._wmap = wmap
        self._sampling_resolution = max(0.2, float(sampling_resolution))
        self._lane_change_base_cost = float(lane_change_base_cost)
        self._late_lane_change_penalty = max(0.0, float(late_lane_change_penalty))
        self._lane_change_freeze_distance_m = max(0.0, float(lane_change_freeze_distance_m))
        self._turn_threshold_deg = max(5.0, float(turn_threshold_deg))
        self._max_search_expansions = max(10_000, int(max_search_expansions))

        self._segments: Dict[str, SegmentNode] = {}
        self._edges: Dict[str, List[Edge]] = {}
        self._edge_kind_lookup: Dict[Tuple[str, str], EdgeKind] = {}
        self._lane_members: Dict[Tuple[int, int, int], List[str]] = {}
        self._entry_wp_by_segment: Dict[str, carla.Waypoint] = {}
        self._exit_wp_by_segment: Dict[str, carla.Waypoint] = {}
        self._entry_wp_id_to_segment: Dict[int, str] = {}
        self._dist_to_junction_m: Dict[str, float] = {}
        self._connector_links_by_road: Dict[int, dict[str, Any]] = {}
        self._segment_wp_cache: Dict[str, List[carla.Waypoint]] = {}

        self._cache_dir = self._resolve_cache_dir(cache_dir)
        self._cache_hit = False
        self._graph_diagnostics: dict[str, Any] = {}
        self._last_plan_diagnostics: dict[str, Any] | None = None
        self._grp_fallback: Any | None = None

        self._build_or_load_graph()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trace_route(
        self,
        origin: carla.Location,
        destination: carla.Location,
    ) -> List[Tuple[carla.Waypoint, RoadOption]]:
        """Plan a route from ``origin`` to ``destination``.

        Returns
        -------
        list[tuple[carla.Waypoint, RoadOption]]
            Dense lane-level route suitable for ``PrivilegedExpert.set_route``.
        """
        start_wp = self._wmap.get_waypoint(
            origin,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        end_wp = self._wmap.get_waypoint(
            destination,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if start_wp is None or end_wp is None:
            logger.warning(
                "LaneGraphPlanner projection failed for origin/destination; attempting GRP fallback"
            )
            grp_route = self._trace_route_with_grp_fallback(origin, destination)
            if grp_route:
                return grp_route
            raise RuntimeError("LaneGraphPlanner: failed to project origin/destination to driving lanes")

        start_id = self._nearest_segment_id(start_wp)
        goal_id = self._nearest_segment_id(end_wp)

        path_start = time.perf_counter()
        segment_path, search_stats = self._a_star(start_id, goal_id)
        search_ms = (time.perf_counter() - path_start) * 1000.0

        if not segment_path:
            logger.warning(
                "LaneGraphPlanner found no segment path from %s/%s to %s/%s; attempting GRP fallback",
                start_wp.road_id,
                start_wp.lane_id,
                end_wp.road_id,
                end_wp.lane_id,
            )
            grp_route = self._trace_route_with_grp_fallback(origin, destination)
            if grp_route:
                return grp_route
            raise RuntimeError(
                "LaneGraphPlanner: no segment route found "
                f"from {start_wp.road_id}/{start_wp.lane_id} to {end_wp.road_id}/{end_wp.lane_id}"
            )

        commands = self._infer_commands(segment_path)
        route: List[Tuple[carla.Waypoint, RoadOption]] = []
        for idx, seg_id in enumerate(segment_path):
            cmd = commands[idx]
            for wp in self._segment_waypoints(seg_id):
                if route:
                    prev_wp = route[-1][0]
                    if self._same_lane(prev_wp, wp):
                        prev_loc = prev_wp.transform.location
                        wp_loc = wp.transform.location
                        if prev_loc.distance(wp_loc) < max(0.4, self._sampling_resolution * 0.4):
                            continue
                route.append((wp, cmd))

        # Ensure final point aligns with destination lane key.
        if not route:
            grp_route = self._trace_route_with_grp_fallback(origin, destination)
            if grp_route:
                return grp_route
            raise RuntimeError("LaneGraphPlanner: segment decode produced an empty route")

        last_wp = route[-1][0]
        if (
            last_wp.road_id != end_wp.road_id
            or last_wp.section_id != end_wp.section_id
            or last_wp.lane_id != end_wp.lane_id
        ):
            route.append((end_wp, RoadOption.LANEFOLLOW))

        self._last_plan_diagnostics = {
            "backend": "lane_segment_graph",
            "cache_hit": bool(self._cache_hit),
            "search": {
                "start_segment_id": start_id,
                "goal_segment_id": goal_id,
                "segment_hops": len(segment_path),
                "expansions": int(search_stats.get("expansions", 0)),
                "path_cost": float(search_stats.get("path_cost", 0.0)),
                "search_ms": round(search_ms, 3),
                "max_search_expansions": int(self._max_search_expansions),
            },
            "route": {
                "n_waypoints": len(route),
                "junction_segments_in_path": int(
                    sum(1 for seg_id in segment_path if self._segments[seg_id].is_junction)
                ),
                "segment_ids": segment_path,
                "segment_links": self._segment_links_for_path(segment_path),
            },
            "graph": self.get_graph_diagnostics(),
        }

        return route

    def get_graph_diagnostics(self) -> dict[str, Any]:
        """Return static graph diagnostics for logging/sidecars."""
        return dict(self._graph_diagnostics)

    def get_last_plan_diagnostics(self) -> dict[str, Any] | None:
        """Return the most recent per-route planning diagnostics."""
        if self._last_plan_diagnostics is None:
            return None
        return dict(self._last_plan_diagnostics)

    def _trace_route_with_grp_fallback(
        self,
        origin: carla.Location,
        destination: carla.Location,
    ) -> List[Tuple[carla.Waypoint, RoadOption]]:
        """Fallback to CARLA's GlobalRoutePlanner when lane-graph search fails."""
        try:
            from agents.navigation.global_route_planner import GlobalRoutePlanner

            if self._grp_fallback is None:
                self._grp_fallback = GlobalRoutePlanner(self._wmap, sampling_resolution=2.0)

            route = self._grp_fallback.trace_route(origin, destination)
            if route:
                logger.warning("LaneGraphPlanner GRP fallback succeeded with %d route waypoints", len(route))
            else:
                logger.warning("LaneGraphPlanner GRP fallback returned an empty route")
            return route
        except Exception as exc:
            logger.error("LaneGraphPlanner GRP fallback failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Graph construction / cache
    # ------------------------------------------------------------------

    def _resolve_cache_dir(self, cache_dir: str | Path | None) -> Path:
        if cache_dir is not None:
            return Path(cache_dir)
        env_dir = os.getenv("RWKV_LANE_SEGMENT_CACHE_DIR")
        if env_dir:
            return Path(env_dir)
        return Path("data/analysis/lane_segment_graph_cache")

    def _cache_path(self) -> Path:
        map_name = str(getattr(self._wmap, "name", "unknown_map")).replace("/", "_")
        key_blob = (
            f"{GRAPH_CACHE_VERSION}|{GRAPH_CACHE_LOGIC_TAG}|{map_name}|{self._sampling_resolution:.3f}|"
            f"{self._lane_change_base_cost:.3f}|{self._late_lane_change_penalty:.3f}|"
            f"{self._lane_change_freeze_distance_m:.3f}"
        )
        digest = hashlib.sha1(key_blob.encode("utf-8")).hexdigest()[:12]
        return self._cache_dir / f"{map_name}.lane_segment_graph.{digest}.json"

    def _build_or_load_graph(self) -> None:
        """Load graph from cache if possible, otherwise build and cache it."""
        t0 = time.perf_counter()
        cache_path = self._cache_path()
        loaded = False
        if cache_path.exists():
            loaded = self._load_graph_cache(cache_path)

        if loaded:
            self._cache_hit = True
            self._graph_diagnostics["cache_path"] = str(cache_path)
            self._graph_diagnostics["cache_hit"] = True
            self._graph_diagnostics["load_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            logger.info(
                "LaneGraphPlanner loaded cached lane-segment graph: segments=%d edges=%d connector_roads=%d",
                len(self._segments),
                sum(len(v) for v in self._edges.values()),
                len(self._connector_links_by_road),
            )
            return

        self._cache_hit = False
        self._build_graph_from_map()
        self._graph_diagnostics["cache_path"] = str(cache_path)
        self._graph_diagnostics["cache_hit"] = False
        self._graph_diagnostics["build_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_graph_cache(cache_path)
        except Exception as exc:
            logger.warning("LaneGraphPlanner failed to write cache %s: %s", cache_path, exc)

    def _build_graph_from_map(self) -> None:
        """Build a lane-segment graph from CARLA topology and lane-change rules."""
        topology = list(self._wmap.get_topology())
        if not topology:
            raise RuntimeError("LaneGraphPlanner: map topology is empty")

        self._segments.clear()
        self._edges.clear()
        self._edge_kind_lookup.clear()
        self._lane_members.clear()
        self._entry_wp_by_segment.clear()
        self._exit_wp_by_segment.clear()
        self._entry_wp_id_to_segment.clear()
        self._segment_wp_cache.clear()

        build_t0 = time.perf_counter()

        for wp_start, wp_end in topology:
            if wp_start.lane_type != carla.LaneType.Driving:
                continue
            if wp_end.lane_type != carla.LaneType.Driving:
                continue

            seg_id = self._segment_id_for_waypoint(wp_start)
            if seg_id in self._segments:
                continue

            sampled_wps = self._sample_segment_waypoints(wp_start, wp_end)
            centerline = tuple(
                (
                    float(wp.transform.location.x),
                    float(wp.transform.location.y),
                    float(wp.transform.location.z),
                    float(wp.transform.rotation.yaw),
                )
                for wp in sampled_wps
            )
            length_m = 0.0
            for idx in range(1, len(sampled_wps)):
                length_m += sampled_wps[idx - 1].transform.location.distance(
                    sampled_wps[idx].transform.location
                )

            node = SegmentNode(
                segment_id=seg_id,
                road_id=int(wp_start.road_id),
                section_id=int(wp_start.section_id),
                lane_id=int(wp_start.lane_id),
                is_junction=bool(
                    wp_start.is_junction
                    or wp_end.is_junction
                    or any(bool(getattr(wp, "is_junction", False)) for wp in sampled_wps)
                ),
                entry_xyz=(
                    float(wp_start.transform.location.x),
                    float(wp_start.transform.location.y),
                    float(wp_start.transform.location.z),
                ),
                exit_xyz=(
                    float(wp_end.transform.location.x),
                    float(wp_end.transform.location.y),
                    float(wp_end.transform.location.z),
                ),
                entry_yaw=float(wp_start.transform.rotation.yaw),
                exit_yaw=float(wp_end.transform.rotation.yaw),
                centerline=centerline,
                length_m=float(length_m),
            )
            self._segments[seg_id] = node
            self._edges[seg_id] = []
            self._entry_wp_by_segment[seg_id] = wp_start
            self._exit_wp_by_segment[seg_id] = wp_end
            self._entry_wp_id_to_segment[int(wp_start.id)] = seg_id
            self._segment_wp_cache[seg_id] = list(sampled_wps)

            lane_key = (node.road_id, node.section_id, node.lane_id)
            self._lane_members.setdefault(lane_key, []).append(seg_id)

        # Forward continuity edges.
        for seg_id, node in self._segments.items():
            end_wp = self._exit_wp_by_segment[seg_id]
            direct_id = self._entry_wp_id_to_segment.get(int(end_wp.id))
            if direct_id is not None and direct_id != seg_id:
                dist = self._distance_xyz(node.exit_xyz, self._segments[direct_id].entry_xyz)
                self._upsert_edge(seg_id, direct_id, dist, dist, EdgeKind.FORWARD)

            for nxt in end_wp.next(max(0.5, self._sampling_resolution)):
                if nxt.lane_type != carla.LaneType.Driving:
                    continue
                to_id = self._resolve_segment_for_waypoint(nxt)
                if to_id is None or to_id == seg_id:
                    continue
                dist = self._distance_xyz(node.exit_xyz, self._segments[to_id].entry_xyz)
                if dist <= 1e-3:
                    continue
                self._upsert_edge(seg_id, to_id, dist, dist, EdgeKind.FORWARD)

        self._dist_to_junction_m = self._compute_distance_to_junction_forward()

        # Lateral lane-change edges.
        for seg_id, node in self._segments.items():
            wp = self._entry_wp_by_segment[seg_id]
            if node.is_junction:
                continue
            self._add_lane_change_edges(seg_id, wp)

        self._connector_links_by_road = self._build_junction_connector_links()

        self._graph_diagnostics = {
            "backend": "lane_segment_graph",
            "map_name": str(getattr(self._wmap, "name", "")),
            "sampling_resolution_m": float(self._sampling_resolution),
            "segment_count": int(len(self._segments)),
            "edge_count": int(sum(len(v) for v in self._edges.values())),
            "forward_edge_count": int(
                sum(1 for edges in self._edges.values() for edge in edges if edge.kind == EdgeKind.FORWARD)
            ),
            "lane_change_edge_count": int(
                sum(1 for edges in self._edges.values() for edge in edges if edge.kind != EdgeKind.FORWARD)
            ),
            "junction_segment_count": int(sum(1 for node in self._segments.values() if node.is_junction)),
            "connector_road_count": int(len(self._connector_links_by_road)),
            "connector_links": self._connector_links_by_road,
            "build_topology_edges": int(len(topology)),
            "build_ms": round((time.perf_counter() - build_t0) * 1000.0, 3),
        }

        logger.info(
            "LaneGraphPlanner built lane-segment graph: segments=%d edges=%d junction_segments=%d connector_roads=%d",
            len(self._segments),
            sum(len(v) for v in self._edges.values()),
            sum(1 for node in self._segments.values() if node.is_junction),
            len(self._connector_links_by_road),
        )

    def _save_graph_cache(self, cache_path: Path) -> None:
        payload = {
            "cache_version": GRAPH_CACHE_VERSION,
            "cache_logic_tag": GRAPH_CACHE_LOGIC_TAG,
            "map_name": str(getattr(self._wmap, "name", "")),
            "sampling_resolution_m": float(self._sampling_resolution),
            "lane_change_base_cost": float(self._lane_change_base_cost),
            "late_lane_change_penalty": float(self._late_lane_change_penalty),
            "lane_change_freeze_distance_m": float(self._lane_change_freeze_distance_m),
            "segments": [
                {
                    "segment_id": node.segment_id,
                    "road_id": node.road_id,
                    "section_id": node.section_id,
                    "lane_id": node.lane_id,
                    "is_junction": node.is_junction,
                    "entry_xyz": list(node.entry_xyz),
                    "exit_xyz": list(node.exit_xyz),
                    "entry_yaw": node.entry_yaw,
                    "exit_yaw": node.exit_yaw,
                    "centerline": [list(p) for p in node.centerline],
                    "length_m": node.length_m,
                }
                for node in self._segments.values()
            ],
            "edges": {
                seg_id: [
                    {
                        "to_id": edge.to_id,
                        "cost": edge.cost,
                        "distance_m": edge.distance_m,
                        "kind": edge.kind.value,
                    }
                    for edge in edges
                ]
                for seg_id, edges in self._edges.items()
            },
            "dist_to_junction_m": self._dist_to_junction_m,
            "connector_links_by_road": self._connector_links_by_road,
        }
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_graph_cache(self, cache_path: Path) -> bool:
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return False

        if int(payload.get("cache_version", -1)) != GRAPH_CACHE_VERSION:
            return False
        if str(payload.get("cache_logic_tag", "")) != GRAPH_CACHE_LOGIC_TAG:
            return False
        if str(payload.get("map_name", "")) != str(getattr(self._wmap, "name", "")):
            return False

        try:
            self._segments = {}
            self._edges = {}
            self._edge_kind_lookup = {}
            self._lane_members = {}
            self._entry_wp_by_segment = {}
            self._exit_wp_by_segment = {}
            self._entry_wp_id_to_segment = {}
            self._segment_wp_cache = {}

            for raw in payload.get("segments", []):
                node = SegmentNode(
                    segment_id=str(raw["segment_id"]),
                    road_id=int(raw["road_id"]),
                    section_id=int(raw["section_id"]),
                    lane_id=int(raw["lane_id"]),
                    is_junction=bool(raw["is_junction"]),
                    entry_xyz=tuple(float(v) for v in raw["entry_xyz"]),
                    exit_xyz=tuple(float(v) for v in raw["exit_xyz"]),
                    entry_yaw=float(raw["entry_yaw"]),
                    exit_yaw=float(raw["exit_yaw"]),
                    centerline=tuple(
                        (float(p[0]), float(p[1]), float(p[2]), float(p[3]))
                        for p in raw.get("centerline", [])
                    ),
                    length_m=float(raw["length_m"]),
                )
                self._segments[node.segment_id] = node
                self._edges[node.segment_id] = []
                lane_key = (node.road_id, node.section_id, node.lane_id)
                self._lane_members.setdefault(lane_key, []).append(node.segment_id)

            for seg_id, edges in payload.get("edges", {}).items():
                if seg_id not in self._segments:
                    continue
                parsed: list[Edge] = []
                for raw_edge in edges:
                    to_id = str(raw_edge.get("to_id", ""))
                    if to_id not in self._segments:
                        continue
                    kind = EdgeKind(str(raw_edge.get("kind", EdgeKind.FORWARD.value)))
                    edge = Edge(
                        to_id=to_id,
                        cost=float(raw_edge.get("cost", 0.0)),
                        distance_m=float(raw_edge.get("distance_m", 0.0)),
                        kind=kind,
                    )
                    parsed.append(edge)
                    self._edge_kind_lookup[(seg_id, to_id)] = kind
                self._edges[seg_id] = parsed

            self._dist_to_junction_m = {
                str(k): float(v) for k, v in payload.get("dist_to_junction_m", {}).items()
            }
            self._connector_links_by_road = {
                int(k): dict(v) for k, v in payload.get("connector_links_by_road", {}).items()
            }

            self._graph_diagnostics = {
                "backend": "lane_segment_graph",
                "map_name": str(getattr(self._wmap, "name", "")),
                "sampling_resolution_m": float(self._sampling_resolution),
                "segment_count": int(len(self._segments)),
                "edge_count": int(sum(len(v) for v in self._edges.values())),
                "forward_edge_count": int(
                    sum(1 for edges in self._edges.values() for edge in edges if edge.kind == EdgeKind.FORWARD)
                ),
                "lane_change_edge_count": int(
                    sum(1 for edges in self._edges.values() for edge in edges if edge.kind != EdgeKind.FORWARD)
                ),
                "junction_segment_count": int(
                    sum(1 for node in self._segments.values() if node.is_junction)
                ),
                "connector_road_count": int(len(self._connector_links_by_road)),
                "connector_links": self._connector_links_by_road,
            }
            return True
        except Exception:
            return False

    def _segment_id_for_waypoint(self, wp: carla.Waypoint) -> str:
        return str(int(wp.id))

    def _sample_segment_waypoints(
        self,
        wp_start: carla.Waypoint,
        wp_end: carla.Waypoint,
    ) -> List[carla.Waypoint]:
        lane_key = (int(wp_start.road_id), int(wp_start.section_id), int(wp_start.lane_id))
        sampled = [wp_start]
        visited = {int(wp_start.id)}
        current = wp_start
        step = max(0.5, self._sampling_resolution)
        max_steps = 1024

        for _ in range(max_steps):
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
                candidates,
                key=lambda cand: cand.transform.location.distance(wp_end.transform.location),
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

    def _upsert_edge(
        self,
        from_id: str,
        to_id: str,
        cost: float,
        distance_m: float,
        kind: EdgeKind,
    ) -> None:
        key = (from_id, to_id)
        prev_kind = self._edge_kind_lookup.get(key)

        # Prefer lane-change semantics over forward if both exist between
        # the same node pair; otherwise keep the cheaper edge.
        if prev_kind is not None:
            if prev_kind == kind:
                edges = self._edges[from_id]
                for idx, edge in enumerate(edges):
                    if edge.to_id == to_id and edge.kind == kind:
                        if cost < edge.cost:
                            edges[idx] = Edge(to_id=to_id, cost=cost, distance_m=distance_m, kind=kind)
                        return
            elif prev_kind in (EdgeKind.LANE_CHANGE_LEFT, EdgeKind.LANE_CHANGE_RIGHT):
                return
            elif kind in (EdgeKind.LANE_CHANGE_LEFT, EdgeKind.LANE_CHANGE_RIGHT):
                self._edges[from_id] = [
                    edge
                    for edge in self._edges[from_id]
                    if not (edge.to_id == to_id and edge.kind == EdgeKind.FORWARD)
                ]
            else:
                return

        self._edges[from_id].append(Edge(to_id=to_id, cost=cost, distance_m=distance_m, kind=kind))
        self._edge_kind_lookup[key] = kind

    def _resolve_segment_for_waypoint(self, wp: carla.Waypoint) -> str | None:
        direct = self._entry_wp_id_to_segment.get(int(wp.id))
        if direct is not None:
            return direct
        lane_key = (int(wp.road_id), int(wp.section_id), int(wp.lane_id))
        candidates = self._lane_members.get(lane_key)
        if not candidates:
            return None
        wp_loc = wp.transform.location
        return min(
            candidates,
            key=lambda seg_id: wp_loc.distance(self._entry_location(seg_id)),
        )

    def _add_lane_change_edges(self, node_id: str, wp: carla.Waypoint) -> None:
        lane_change_flags = wp.lane_change

        if lane_change_flags in (carla.LaneChange.Left, carla.LaneChange.Both):
            self._maybe_add_lateral_edge(node_id, wp, wp.get_left_lane(), EdgeKind.LANE_CHANGE_LEFT)

        if lane_change_flags in (carla.LaneChange.Right, carla.LaneChange.Both):
            self._maybe_add_lateral_edge(node_id, wp, wp.get_right_lane(), EdgeKind.LANE_CHANGE_RIGHT)

    def _maybe_add_lateral_edge(
        self,
        from_id: str,
        from_wp: carla.Waypoint,
        to_wp: carla.Waypoint | None,
        kind: EdgeKind,
    ) -> None:
        if to_wp is None:
            return
        if to_wp.lane_type != carla.LaneType.Driving:
            return
        if to_wp.is_junction:
            return
        if int(to_wp.road_id) != int(from_wp.road_id):
            return
        if int(to_wp.section_id) != int(from_wp.section_id):
            return

        yaw_delta = abs(self._normalize_angle_deg(float(to_wp.transform.rotation.yaw - from_wp.transform.rotation.yaw)))
        if yaw_delta > 30.0:
            return

        to_id = self._resolve_segment_for_waypoint(to_wp)
        if to_id is None:
            return
        to_node = self._segments[to_id]
        if to_node.is_junction:
            return
        lateral_dist = from_wp.transform.location.distance(to_wp.transform.location)
        if lateral_dist <= 1e-3:
            return

        cost = self._lane_change_base_cost + lateral_dist
        dist_to_junction = self._dist_to_junction_m.get(from_id, math.inf)
        if dist_to_junction <= self._lane_change_freeze_distance_m:
            slack = max(0.0, self._lane_change_freeze_distance_m - dist_to_junction)
            scale = 1.0 + (slack / max(1.0, self._lane_change_freeze_distance_m))
            cost += self._late_lane_change_penalty * scale

        self._upsert_edge(from_id, to_id, cost, lateral_dist, kind)

    def _compute_distance_to_junction_forward(self) -> Dict[str, float]:
        """Compute shortest forward distance from each segment to any junction segment."""
        reverse_longitudinal: Dict[str, List[Tuple[str, float]]] = {}
        for from_id, edges in self._edges.items():
            for edge in edges:
                if edge.kind != EdgeKind.FORWARD:
                    continue
                reverse_longitudinal.setdefault(edge.to_id, []).append((from_id, edge.distance_m))

        dist: Dict[str, float] = {node_id: math.inf for node_id in self._segments}
        heap: List[Tuple[float, str]] = []

        for node_id, node in self._segments.items():
            if node.is_junction:
                dist[node_id] = 0.0
                heapq.heappush(heap, (0.0, node_id))

        while heap:
            best_dist, node_id = heapq.heappop(heap)
            if best_dist > dist[node_id]:
                continue
            for prev_id, step_dist in reverse_longitudinal.get(node_id, []):
                cand = best_dist + step_dist
                if cand < dist[prev_id]:
                    dist[prev_id] = cand
                    heapq.heappush(heap, (cand, prev_id))

        return dist

    def _build_junction_connector_links(self) -> Dict[int, dict[str, Any]]:
        """Summarize connector-road links in junctions.

        Junction connector roads bridge outside roads. This function makes that
        linkage explicit for diagnostics and sidecar traceability.
        """
        reverse_forward: Dict[str, List[str]] = {}
        for from_id, edges in self._edges.items():
            for edge in edges:
                if edge.kind != EdgeKind.FORWARD:
                    continue
                reverse_forward.setdefault(edge.to_id, []).append(from_id)

        by_road: Dict[int, dict[str, Any]] = {}
        for seg_id, node in self._segments.items():
            if not node.is_junction:
                continue

            incoming_roads = {
                self._segments[pred_id].road_id
                for pred_id in reverse_forward.get(seg_id, [])
                if not self._segments[pred_id].is_junction
            }
            outgoing_roads = {
                self._segments[edge.to_id].road_id
                for edge in self._edges.get(seg_id, [])
                if edge.kind == EdgeKind.FORWARD and not self._segments[edge.to_id].is_junction
            }

            if not incoming_roads and not outgoing_roads:
                continue

            road_entry = by_road.setdefault(
                int(node.road_id),
                {
                    "connector_road_id": int(node.road_id),
                    "connector_segment_ids": [],
                    "entry_road_ids": set(),
                    "exit_road_ids": set(),
                    "links": set(),
                },
            )
            road_entry["connector_segment_ids"].append(seg_id)
            road_entry["entry_road_ids"].update(incoming_roads)
            road_entry["exit_road_ids"].update(outgoing_roads)
            for rid_in in incoming_roads:
                for rid_out in outgoing_roads:
                    road_entry["links"].add((int(rid_in), int(rid_out)))

        normalized: Dict[int, dict[str, Any]] = {}
        for road_id, entry in by_road.items():
            links = sorted(entry["links"])
            normalized[road_id] = {
                "connector_road_id": int(road_id),
                "connector_segment_ids": sorted(set(entry["connector_segment_ids"])),
                "entry_road_ids": sorted(int(rid) for rid in entry["entry_road_ids"]),
                "exit_road_ids": sorted(int(rid) for rid in entry["exit_road_ids"]),
                "links": [
                    {
                        "from_road_id": int(rid_in),
                        "to_road_id": int(rid_out),
                    }
                    for rid_in, rid_out in links
                ],
            }
        return normalized

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _a_star(self, start_id: str, goal_id: str) -> Tuple[List[str], dict[str, float]]:
        """Run A* over the lane-segment graph."""
        if start_id == goal_id:
            return [start_id], {"expansions": 0.0, "path_cost": 0.0}

        goal_xyz = self._segment_midpoint_xyz(goal_id)
        g_score: Dict[str, float] = {start_id: 0.0}
        came_from: Dict[str, str] = {}
        open_heap: List[Tuple[float, float, str]] = [
            (self._heuristic(start_id, goal_xyz), 0.0, start_id)
        ]

        expansions = 0
        while open_heap:
            _, cost_so_far, node_id = heapq.heappop(open_heap)
            if cost_so_far > g_score.get(node_id, math.inf):
                continue

            if node_id == goal_id:
                return self._reconstruct_path(came_from, goal_id), {
                    "expansions": float(expansions),
                    "path_cost": float(cost_so_far),
                }

            expansions += 1
            if expansions > self._max_search_expansions:
                logger.warning(
                    "LaneGraphPlanner A* expansion cap reached (%d)", self._max_search_expansions
                )
                break

            for edge in self._edges.get(node_id, []):
                cand = cost_so_far + edge.cost
                if cand >= g_score.get(edge.to_id, math.inf):
                    continue
                g_score[edge.to_id] = cand
                came_from[edge.to_id] = node_id
                f = cand + self._heuristic(edge.to_id, goal_xyz)
                heapq.heappush(open_heap, (f, cand, edge.to_id))

        return [], {"expansions": float(expansions), "path_cost": float(math.inf)}

    def _heuristic(self, node_id: str, goal_xyz: Tuple[float, float, float]) -> float:
        return self._distance_xyz(self._segment_midpoint_xyz(node_id), goal_xyz)

    @staticmethod
    def _reconstruct_path(came_from: Dict[str, str], goal_id: str) -> List[str]:
        path = [goal_id]
        cur = goal_id
        while cur in came_from:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Route decoding
    # ------------------------------------------------------------------

    def _nearest_segment_id(self, wp: carla.Waypoint) -> str:
        lane_key = (int(wp.road_id), int(wp.section_id), int(wp.lane_id))
        candidates = self._lane_members.get(lane_key)
        if not candidates:
            candidates = list(self._segments.keys())
        if not candidates:
            raise RuntimeError("LaneGraphPlanner: graph has no segments")
        wp_loc = wp.transform.location
        return min(candidates, key=lambda seg_id: wp_loc.distance(self._entry_location(seg_id)))

    def _infer_commands(self, segment_path: List[str]) -> List[RoadOption]:
        commands: List[RoadOption] = [RoadOption.LANEFOLLOW] * len(segment_path)
        active_junction_cmd: RoadOption | None = None

        for idx in range(1, len(segment_path)):
            prev_id = segment_path[idx - 1]
            cur_id = segment_path[idx]
            kind = self._edge_kind_lookup.get((prev_id, cur_id), EdgeKind.FORWARD)

            if kind == EdgeKind.LANE_CHANGE_LEFT:
                commands[idx] = RoadOption.CHANGELANELEFT
                active_junction_cmd = None
                continue

            if kind == EdgeKind.LANE_CHANGE_RIGHT:
                commands[idx] = RoadOption.CHANGELANERIGHT
                active_junction_cmd = None
                continue

            prev_seg = self._segments[prev_id]
            cur_seg = self._segments[cur_id]

            if active_junction_cmd is not None:
                commands[idx] = active_junction_cmd
                if not cur_seg.is_junction:
                    active_junction_cmd = None
                continue

            if not prev_seg.is_junction and cur_seg.is_junction:
                active_junction_cmd = self._classify_junction_turn(segment_path, idx)
                commands[idx] = active_junction_cmd
                continue

            if prev_seg.is_junction and cur_seg.is_junction:
                active_junction_cmd = self._classify_junction_turn(segment_path, idx)
                commands[idx] = active_junction_cmd
                continue

            commands[idx] = RoadOption.LANEFOLLOW

        if len(commands) > 1 and commands[0] == RoadOption.LANEFOLLOW:
            commands[0] = commands[1]

        return commands

    def _classify_junction_turn(self, segment_path: List[str], idx: int) -> RoadOption:
        if idx <= 0:
            return RoadOption.STRAIGHT

        prev_seg = self._segments[segment_path[idx - 1]]
        cur_seg = self._segments[segment_path[idx]]
        entry_vec = self._direction_vec_xyz(
            self._segment_midpoint_xyz(segment_path[idx - 1]),
            self._segment_midpoint_xyz(segment_path[idx]),
        )
        if entry_vec is None:
            return RoadOption.STRAIGHT

        exit_vec = None
        # Prefer first edge that exits a junction sequence.
        for j in range(idx + 1, len(segment_path)):
            last_seg = self._segments[segment_path[j - 1]]
            next_seg = self._segments[segment_path[j]]
            if last_seg.is_junction and not next_seg.is_junction:
                exit_vec = self._direction_vec_xyz(
                    self._segment_midpoint_xyz(segment_path[j - 1]),
                    self._segment_midpoint_xyz(segment_path[j]),
                )
                break

        if exit_vec is None and idx + 1 < len(segment_path):
            exit_vec = self._direction_vec_xyz(
                self._segment_midpoint_xyz(segment_path[idx]),
                self._segment_midpoint_xyz(segment_path[idx + 1]),
            )

        if exit_vec is None:
            return RoadOption.STRAIGHT

        signed = self._signed_angle_deg(entry_vec, exit_vec)
        if abs(signed) < self._turn_threshold_deg:
            return RoadOption.STRAIGHT
        return RoadOption.LEFT if signed < 0.0 else RoadOption.RIGHT

    @staticmethod
    def _direction_vec_xyz(
        p_a: Tuple[float, float, float],
        p_b: Tuple[float, float, float],
    ) -> Tuple[float, float] | None:
        dx = float(p_b[0] - p_a[0])
        dy = float(p_b[1] - p_a[1])
        mag = math.hypot(dx, dy)
        if mag <= 1e-6:
            return None
        return (dx / mag, dy / mag)

    def _segment_waypoints(self, seg_id: str) -> List[carla.Waypoint]:
        cached = self._segment_wp_cache.get(seg_id)
        if cached:
            return cached

        node = self._segments[seg_id]
        sampled: List[carla.Waypoint] = []
        for x, y, z, _yaw in node.centerline:
            wp = self._wmap.get_waypoint(
                carla.Location(float(x), float(y), float(z)),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if wp is None:
                continue
            if sampled:
                prev = sampled[-1]
                if self._same_lane(prev, wp):
                    if prev.transform.location.distance(wp.transform.location) < max(
                        0.4,
                        self._sampling_resolution * 0.4,
                    ):
                        continue
            sampled.append(wp)

        self._segment_wp_cache[seg_id] = sampled
        return sampled

    def _segment_links_for_path(self, segment_path: List[str]) -> List[dict[str, Any]]:
        links: List[dict[str, Any]] = []
        for idx in range(1, len(segment_path)):
            prev_id = segment_path[idx - 1]
            cur_id = segment_path[idx]
            prev_seg = self._segments[prev_id]
            cur_seg = self._segments[cur_id]
            via_connector = bool(prev_seg.is_junction or cur_seg.is_junction)
            links.append(
                {
                    "from_segment_id": prev_id,
                    "to_segment_id": cur_id,
                    "from_road_id": int(prev_seg.road_id),
                    "to_road_id": int(cur_seg.road_id),
                    "from_lane_id": int(prev_seg.lane_id),
                    "to_lane_id": int(cur_seg.lane_id),
                    "edge_kind": self._edge_kind_lookup.get((prev_id, cur_id), EdgeKind.FORWARD).value,
                    "via_junction_connector": via_connector,
                }
            )
        return links

    def _entry_location(self, seg_id: str) -> carla.Location:
        node = self._segments[seg_id]
        return carla.Location(node.entry_xyz[0], node.entry_xyz[1], node.entry_xyz[2])

    def _segment_midpoint_xyz(self, seg_id: str) -> Tuple[float, float, float]:
        node = self._segments[seg_id]
        if node.centerline:
            mid = node.centerline[len(node.centerline) // 2]
            return (float(mid[0]), float(mid[1]), float(mid[2]))
        return node.entry_xyz

    @staticmethod
    def _distance_xyz(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        dx = float(a[0] - b[0])
        dy = float(a[1] - b[1])
        dz = float(a[2] - b[2])
        return float(math.sqrt(dx * dx + dy * dy + dz * dz))

    @staticmethod
    def _same_lane(wp_a: carla.Waypoint, wp_b: carla.Waypoint) -> bool:
        return (
            int(wp_a.road_id) == int(wp_b.road_id)
            and int(wp_a.section_id) == int(wp_b.section_id)
            and int(wp_a.lane_id) == int(wp_b.lane_id)
        )

    @staticmethod
    def _signed_angle_deg(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = max(-1.0, min(1.0, v1[0] * v2[0] + v1[1] * v2[1]))
        return math.degrees(math.atan2(cross, dot))

    @staticmethod
    def _normalize_angle_deg(angle: float) -> float:
        while angle > 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle
