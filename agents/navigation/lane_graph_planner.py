"""Lane-graph route planner for CARLA.

This planner builds a directed graph over driving-lane waypoints and runs
A* to produce a lane-legal route. Unlike GlobalRoutePlanner's road-level
abstraction, this preserves lane-level transitions (including connectors)
so junction approach lanes can be enforced naturally.
"""

from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple

import carla

from agents.navigation.local_planner import RoadOption

logger = logging.getLogger(__name__)


class EdgeKind(Enum):
    """Internal graph edge categories."""

    FORWARD = "forward"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"


@dataclass(frozen=True)
class Edge:
    """Directed graph edge between sampled lane waypoints."""

    to_id: int
    cost: float
    distance_m: float
    kind: EdgeKind


class LaneGraphPlanner:
    """Lane-level route planner that returns ``(Waypoint, RoadOption)`` traces.

    Parameters
    ----------
    wmap:
        CARLA map object.
    sampling_resolution:
        Spacing (metres) used for graph waypoint sampling and forward links.
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
    ) -> None:
        self._wmap = wmap
        self._sampling_resolution = max(0.2, float(sampling_resolution))
        self._lane_change_base_cost = float(lane_change_base_cost)
        self._late_lane_change_penalty = max(0.0, float(late_lane_change_penalty))
        self._lane_change_freeze_distance_m = max(0.0, float(lane_change_freeze_distance_m))
        self._turn_threshold_deg = max(5.0, float(turn_threshold_deg))
        self._max_search_expansions = max(10_000, int(max_search_expansions))

        self._nodes: Dict[int, carla.Waypoint] = {}
        self._edges: Dict[int, List[Edge]] = {}
        self._edge_kind_lookup: Dict[Tuple[int, int], EdgeKind] = {}
        self._lane_members: Dict[Tuple[int, int, int], List[int]] = {}
        self._dist_to_junction_m: Dict[int, float] = {}
        self._grp_fallback: Any | None = None

        self._build_graph()

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

        start_id = self._nearest_node_id(start_wp)
        goal_id = self._nearest_node_id(end_wp)
        node_path = self._a_star(start_id, goal_id)
        if not node_path:
            logger.warning(
                "LaneGraphPlanner found no lane-level path from %s/%s to %s/%s; attempting GRP fallback",
                start_wp.road_id,
                start_wp.lane_id,
                end_wp.road_id,
                end_wp.lane_id,
            )
            grp_route = self._trace_route_with_grp_fallback(origin, destination)
            if grp_route:
                return grp_route
            raise RuntimeError(
                "LaneGraphPlanner: no lane-level route found "
                f"from {start_wp.road_id}/{start_wp.lane_id} to {end_wp.road_id}/{end_wp.lane_id}"
            )

        commands = self._infer_commands(node_path)
        route: List[Tuple[carla.Waypoint, RoadOption]] = []
        for idx, node_id in enumerate(node_path):
            route.append((self._nodes[node_id], commands[idx]))

        # Ensure final point aligns with destination lane key.
        last_wp = route[-1][0]
        if (
            last_wp.road_id != end_wp.road_id
            or last_wp.section_id != end_wp.section_id
            or last_wp.lane_id != end_wp.lane_id
        ):
            route.append((end_wp, RoadOption.LANEFOLLOW))

        return route

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
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> None:
        """Build sampled lane graph and junction-distance annotations."""
        sampled = self._wmap.generate_waypoints(self._sampling_resolution)

        for wp in sampled:
            if wp.lane_type != carla.LaneType.Driving:
                continue
            self._register_node(wp)

        # Longitudinal edges first so we can compute distance-to-junction on
        # the forward graph before adding lateral lane-change edges.
        pending_ids = list(self._nodes.keys())
        cursor = 0
        while cursor < len(pending_ids):
            node_id = pending_ids[cursor]
            cursor += 1
            wp = self._nodes[node_id]
            for nxt in wp.next(self._sampling_resolution):
                if nxt.lane_type != carla.LaneType.Driving:
                    continue
                is_new = int(nxt.id) not in self._nodes
                to_id = self._register_node(nxt)
                if is_new:
                    pending_ids.append(to_id)
                if to_id == node_id:
                    continue
                dist = wp.transform.location.distance(nxt.transform.location)
                if dist <= 1e-3:
                    continue
                self._upsert_edge(node_id, to_id, dist, dist, EdgeKind.FORWARD)

        self._dist_to_junction_m = self._compute_distance_to_junction_forward()

        for node_id, wp in list(self._nodes.items()):
            if wp.is_junction:
                continue
            self._add_lane_change_edges(node_id, wp)

        logger.info(
            "LaneGraphPlanner built graph: nodes=%d edges=%d res=%.2fm",
            len(self._nodes),
            sum(len(v) for v in self._edges.values()),
            self._sampling_resolution,
        )

    def _register_node(self, wp: carla.Waypoint) -> int:
        node_id = int(wp.id)
        if node_id in self._nodes:
            return node_id
        self._nodes[node_id] = wp
        self._edges[node_id] = []
        lane_key = (int(wp.road_id), int(wp.section_id), int(wp.lane_id))
        self._lane_members.setdefault(lane_key, []).append(node_id)
        return node_id

    def _upsert_edge(
        self,
        from_id: int,
        to_id: int,
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

    def _add_lane_change_edges(self, node_id: int, wp: carla.Waypoint) -> None:
        lane_change_flags = wp.lane_change

        if lane_change_flags in (carla.LaneChange.Left, carla.LaneChange.Both):
            self._maybe_add_lateral_edge(node_id, wp, wp.get_left_lane(), EdgeKind.LANE_CHANGE_LEFT)

        if lane_change_flags in (carla.LaneChange.Right, carla.LaneChange.Both):
            self._maybe_add_lateral_edge(node_id, wp, wp.get_right_lane(), EdgeKind.LANE_CHANGE_RIGHT)

    def _maybe_add_lateral_edge(
        self,
        from_id: int,
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

        to_id = int(to_wp.id)
        if to_id not in self._nodes:
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

    def _compute_distance_to_junction_forward(self) -> Dict[int, float]:
        """Compute shortest forward distance from each node to any junction."""
        reverse_longitudinal: Dict[int, List[Tuple[int, float]]] = {}
        for from_id, edges in self._edges.items():
            for edge in edges:
                if edge.kind != EdgeKind.FORWARD:
                    continue
                reverse_longitudinal.setdefault(edge.to_id, []).append((from_id, edge.distance_m))

        dist: Dict[int, float] = {node_id: math.inf for node_id in self._nodes}
        heap: List[Tuple[float, int]] = []

        for node_id, wp in self._nodes.items():
            if wp.is_junction:
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

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _a_star(self, start_id: int, goal_id: int) -> List[int]:
        """Run A* over the lane graph."""
        if start_id == goal_id:
            return [start_id]

        goal_loc = self._nodes[goal_id].transform.location
        g_score: Dict[int, float] = {start_id: 0.0}
        came_from: Dict[int, int] = {}
        open_heap: List[Tuple[float, float, int]] = [
            (self._heuristic(start_id, goal_loc), 0.0, start_id)
        ]

        expansions = 0
        while open_heap:
            _, cost_so_far, node_id = heapq.heappop(open_heap)
            if cost_so_far > g_score.get(node_id, math.inf):
                continue

            if node_id == goal_id:
                return self._reconstruct_path(came_from, goal_id)

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
                f = cand + self._heuristic(edge.to_id, goal_loc)
                heapq.heappush(open_heap, (f, cand, edge.to_id))

        return []

    def _heuristic(self, node_id: int, goal_loc: carla.Location) -> float:
        wp_loc = self._nodes[node_id].transform.location
        return wp_loc.distance(goal_loc)

    @staticmethod
    def _reconstruct_path(came_from: Dict[int, int], goal_id: int) -> List[int]:
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

    def _nearest_node_id(self, wp: carla.Waypoint) -> int:
        direct_id = int(wp.id)
        if direct_id in self._nodes:
            return direct_id

        lane_key = (int(wp.road_id), int(wp.section_id), int(wp.lane_id))
        candidates = self._lane_members.get(lane_key)
        if not candidates:
            # Fallback: global nearest sampled node.
            candidates = list(self._nodes.keys())

        wp_loc = wp.transform.location
        best_id = candidates[0]
        best_dist = self._nodes[best_id].transform.location.distance(wp_loc)
        for node_id in candidates[1:]:
            dist = self._nodes[node_id].transform.location.distance(wp_loc)
            if dist < best_dist:
                best_dist = dist
                best_id = node_id
        return best_id

    def _infer_commands(self, node_path: List[int]) -> List[RoadOption]:
        commands: List[RoadOption] = [RoadOption.LANEFOLLOW] * len(node_path)
        active_junction_cmd: RoadOption | None = None

        for idx in range(1, len(node_path)):
            prev_id = node_path[idx - 1]
            cur_id = node_path[idx]
            kind = self._edge_kind_lookup.get((prev_id, cur_id), EdgeKind.FORWARD)

            if kind == EdgeKind.LANE_CHANGE_LEFT:
                commands[idx] = RoadOption.CHANGELANELEFT
                active_junction_cmd = None
                continue

            if kind == EdgeKind.LANE_CHANGE_RIGHT:
                commands[idx] = RoadOption.CHANGELANERIGHT
                active_junction_cmd = None
                continue

            prev_wp = self._nodes[prev_id]
            cur_wp = self._nodes[cur_id]

            if active_junction_cmd is not None:
                commands[idx] = active_junction_cmd
                if not cur_wp.is_junction:
                    active_junction_cmd = None
                continue

            if not prev_wp.is_junction and cur_wp.is_junction:
                active_junction_cmd = self._classify_junction_turn(node_path, idx)
                commands[idx] = active_junction_cmd
                continue

            if prev_wp.is_junction and cur_wp.is_junction:
                active_junction_cmd = self._classify_junction_turn(node_path, idx)
                commands[idx] = active_junction_cmd
                continue

            commands[idx] = RoadOption.LANEFOLLOW

        if len(commands) > 1 and commands[0] == RoadOption.LANEFOLLOW:
            commands[0] = commands[1]

        return commands

    def _classify_junction_turn(self, node_path: List[int], idx: int) -> RoadOption:
        if idx <= 0:
            return RoadOption.STRAIGHT

        prev_wp = self._nodes[node_path[idx - 1]]
        cur_wp = self._nodes[node_path[idx]]
        entry_vec = self._direction_vec(prev_wp, cur_wp)
        if entry_vec is None:
            return RoadOption.STRAIGHT

        exit_vec = None
        # Prefer first edge that exits a junction sequence.
        for j in range(idx + 1, len(node_path)):
            last_wp = self._nodes[node_path[j - 1]]
            next_wp = self._nodes[node_path[j]]
            if last_wp.is_junction and not next_wp.is_junction:
                exit_vec = self._direction_vec(last_wp, next_wp)
                break

        if exit_vec is None and idx + 1 < len(node_path):
            nxt_wp = self._nodes[node_path[idx + 1]]
            exit_vec = self._direction_vec(cur_wp, nxt_wp)

        if exit_vec is None:
            return RoadOption.STRAIGHT

        signed = self._signed_angle_deg(entry_vec, exit_vec)
        if abs(signed) < self._turn_threshold_deg:
            return RoadOption.STRAIGHT
        return RoadOption.LEFT if signed < 0.0 else RoadOption.RIGHT

    @staticmethod
    def _direction_vec(wp_a: carla.Waypoint, wp_b: carla.Waypoint) -> Tuple[float, float] | None:
        loc_a = wp_a.transform.location
        loc_b = wp_b.transform.location
        dx = float(loc_b.x - loc_a.x)
        dy = float(loc_b.y - loc_a.y)
        mag = math.hypot(dx, dy)
        if mag <= 1e-6:
            return None
        return (dx / mag, dy / mag)

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
