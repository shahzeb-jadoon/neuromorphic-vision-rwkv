"""Balanced route generator for CARLA data collection.

Uses CARLA's topology API to find intersections and chain waypoints
that force the autopilot through left/right turns, counteracting
the natural straight-driving bias.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import carla  # noqa: F401

    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RouteConfig:
    """Configuration for route generation."""

    strategy: str = "balanced"  # "random" or "balanced"
    target_left: float = 0.15
    target_right: float = 0.15
    target_straight: float = 0.10
    min_waypoints: int = 100
    max_waypoints: int = 500
    sampling_resolution: float = 2.0  # meters between waypoints
    route_planner_backend: str = "lane_graph"
    lane_graph_sampling_resolution_m: float = 1.0
    lane_graph_lane_change_cost: float = 25.0
    lane_graph_late_lane_change_penalty: float = 300.0
    lane_graph_lane_change_freeze_distance_m: float = 30.0
    lane_graph_turn_threshold_deg: float = 35.0
    lane_graph_max_search_expansions: int = 250_000
    validate_junction_approach_exit: bool = True
    validation_max_regen_attempts: int = 3
    validation_max_issues: int = 16
    campaign_turn_balance_enabled: bool = False
    campaign_turn_balance_max_adjust: float = 0.05
    campaign_turn_balance_smoothing_alpha: float = 0.35
    campaign_turn_balance_min_turn_samples: int = 20


class BalancedRouteGenerator:
    """Generates driving routes with balanced turn distributions.

    Uses CARLA's waypoint API and road topology to identify intersections
    and create routes that include a target proportion of turns.

    Args:
        world: CARLA world instance.
        config: Route generation configuration.
    """

    def __init__(
        self,
        world: Any,  # carla.World
        config: RouteConfig | None = None,
    ) -> None:
        if not CARLA_AVAILABLE:
            raise RuntimeError("carla package not installed")

        self.world = world
        self.config = config or RouteConfig()
        self.map = world.get_map()
        self.topology = self.map.get_topology()
        self.spawn_points = self.map.get_spawn_points()
        self.route_planner = self._build_route_planner()
        self._campaign_turn_counts: dict[str, int] = {
            "LEFT": 0,
            "RIGHT": 0,
            "STRAIGHT": 0,
        }
        self._campaign_turn_deficit_ema: dict[str, float] = {
            "LEFT": 0.0,
            "RIGHT": 0.0,
            "STRAIGHT": 0.0,
        }

        # Build intersection graph
        self._junctions = self._find_junctions()
        self._approach_exit_links = self._build_approach_exit_links()
        logger.info("Found %d junctions in %s", len(self._junctions), self.map.name)
        logger.info(
            "Route validator approach->exit links: approaches=%d links=%d",
            len(self._approach_exit_links),
            sum(len(v) for v in self._approach_exit_links.values()),
        )

    def _build_route_planner(self) -> Any:
        """Build the route planner backend used for segment tracing."""
        backend = str(self.config.route_planner_backend).strip().lower()

        if backend == "lane_graph":
            try:
                from agents.navigation.lane_graph_planner import LaneGraphPlanner

                planner = LaneGraphPlanner(
                    self.map,
                    sampling_resolution=self.config.lane_graph_sampling_resolution_m,
                    lane_change_base_cost=self.config.lane_graph_lane_change_cost,
                    late_lane_change_penalty=self.config.lane_graph_late_lane_change_penalty,
                    lane_change_freeze_distance_m=(
                        self.config.lane_graph_lane_change_freeze_distance_m
                    ),
                    turn_threshold_deg=self.config.lane_graph_turn_threshold_deg,
                    max_search_expansions=self.config.lane_graph_max_search_expansions,
                )
                logger.info("RouteGenerator planner backend: lane_graph")
                return planner
            except Exception as e:
                logger.warning(
                    "Lane-graph planner init failed (%s); falling back to GRP",
                    e,
                )

        from agents.navigation.global_route_planner import GlobalRoutePlanner

        logger.info("RouteGenerator planner backend: grp")
        return GlobalRoutePlanner(self.map, self.config.sampling_resolution)

    def _find_junctions(self) -> list[dict[str, Any]]:
        """Find all junctions and their possible exit directions.

        Returns:
            List of junction dicts with 'location', 'junction_id',
            and 'exits' (list of direction labels).
        """
        junction_map: dict[int, dict] = {}

        for wp_start, wp_end in self.topology:
            if wp_start.is_junction:
                jid = wp_start.get_junction().id
                if jid not in junction_map:
                    junction_map[jid] = {
                        "junction_id": jid,
                        "location": wp_start.transform.location,
                        "entry_waypoints": [],
                        "exit_waypoints": [],
                    }
                junction_map[jid]["entry_waypoints"].append(wp_start)
                junction_map[jid]["exit_waypoints"].append(wp_end)

        return list(junction_map.values())

    def _build_approach_exit_links(self) -> dict[tuple[int, int], set[tuple[int, int]]]:
        """Build legal approach->exit lane mappings using topology connectors."""
        connector_incoming: dict[int, set[tuple[int, int]]] = defaultdict(set)
        connector_outgoing: dict[int, set[tuple[int, int]]] = defaultdict(set)

        for wp_start, wp_end in self.topology:
            if wp_start.lane_type != carla.LaneType.Driving:
                continue
            if wp_end.lane_type != carla.LaneType.Driving:
                continue

            start_is_junction = bool(getattr(wp_start, "is_junction", False))
            end_is_junction = bool(getattr(wp_end, "is_junction", False))

            if (not start_is_junction) and end_is_junction:
                approach_key = (int(wp_start.road_id), int(wp_start.lane_id))
                connector_incoming[int(wp_end.road_id)].add(approach_key)

            if start_is_junction and (not end_is_junction):
                connector_road_id = int(wp_start.road_id)
                exit_key = (int(wp_end.road_id), int(wp_end.lane_id))
                connector_outgoing[connector_road_id].add(exit_key)

        links: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
        for connector_road_id, incoming in connector_incoming.items():
            outgoing = connector_outgoing.get(connector_road_id, set())
            if not incoming or not outgoing:
                continue
            for approach_key in incoming:
                links[approach_key].update(outgoing)

        return {k: set(v) for k, v in links.items()}

    def generate_route(
        self,
        rng: np.random.Generator | None = None,
    ) -> list[tuple[Any, str]]:
        """Generate a single balanced route.

        Returns:
            List of (carla.Transform, RoadOption_name) tuples suitable
            for BehaviorAgent.set_global_plan().
        """
        if rng is None:
            rng = np.random.default_rng()

        max_attempts = max(1, int(self.config.validation_max_regen_attempts))
        for attempt_idx in range(max_attempts):
            if self.config.strategy == "random":
                route = self._generate_random_route(rng)
            elif self.config.strategy == "balanced":
                route = self._generate_balanced_route(rng)
            else:
                raise ValueError(f"Unknown strategy: {self.config.strategy}")

            if not route:
                continue

            if not bool(self.config.validate_junction_approach_exit):
                self._record_campaign_turn_counts(route)
                return route

            report = self._validate_approach_conditioned_route(route)
            if int(report["invalid_transition_count"]) == 0:
                self._record_campaign_turn_counts(route)
                return route

            logger.warning(
                "Route validator rejected route (%d invalid transition(s)) on attempt %d/%d",
                int(report["invalid_transition_count"]),
                attempt_idx + 1,
                max_attempts,
            )
            if report["issues"]:
                logger.warning("Route validator sample issue: %s", report["issues"][0])

        return []

    def _validate_approach_conditioned_route(
        self,
        route: list[tuple[Any, str]],
    ) -> dict[str, Any]:
        """Validate that approach->exit transitions match legal connector mappings."""
        report: dict[str, Any] = {
            "checked_transition_count": 0,
            "invalid_transition_count": 0,
            "issues": [],
        }
        if not route:
            return report
        if not self._approach_exit_links:
            return report

        projected: list[tuple[int, Any]] = []
        for idx, (transform, _road_option) in enumerate(route):
            wp = self.map.get_waypoint(
                transform.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if wp is None:
                continue
            projected.append((idx, wp))

        if len(projected) < 2:
            return report

        max_issues = max(1, int(self.config.validation_max_issues))
        pending_approach: tuple[int, int] | None = None

        for (idx_a, wp_a), (idx_b, wp_b) in zip(projected, projected[1:]):
            a_is_junction = bool(getattr(wp_a, "is_junction", False))
            b_is_junction = bool(getattr(wp_b, "is_junction", False))

            if (not a_is_junction) and b_is_junction:
                pending_approach = (int(wp_a.road_id), int(wp_a.lane_id))
                continue

            if pending_approach is not None and a_is_junction and (not b_is_junction):
                report["checked_transition_count"] += 1
                exit_key = (int(wp_b.road_id), int(wp_b.lane_id))
                allowed = self._approach_exit_links.get(pending_approach)
                if allowed is not None and len(allowed) > 0 and exit_key not in allowed:
                    report["invalid_transition_count"] += 1
                    if len(report["issues"]) < max_issues:
                        report["issues"].append(
                            {
                                "from_index": int(idx_a),
                                "to_index": int(idx_b),
                                "approach_road_id": int(pending_approach[0]),
                                "approach_lane_id": int(pending_approach[1]),
                                "exit_road_id": int(exit_key[0]),
                                "exit_lane_id": int(exit_key[1]),
                                "allowed_exits": [
                                    {
                                        "road_id": int(road_id),
                                        "lane_id": int(lane_id),
                                    }
                                    for road_id, lane_id in sorted(allowed)
                                ],
                            }
                        )
                pending_approach = None
                continue

            if not b_is_junction:
                pending_approach = None

        return report

    def _generate_random_route(
        self,
        rng: np.random.Generator,
    ) -> list[tuple[Any, str]]:
        """Generate a simple random start-to-end route."""
        indices = rng.choice(len(self.spawn_points), size=2, replace=False)
        start = self.spawn_points[int(indices[0])]
        end = self.spawn_points[int(indices[1])]

        start_wp = self.map.get_waypoint(start.location)
        end_wp = self.map.get_waypoint(end.location)

        # Use CARLA's built-in route planner
        route = self._trace_route(start_wp, end_wp)
        return route

    def _generate_balanced_route(
        self,
        rng: np.random.Generator,
    ) -> list[tuple[Any, str]]:
        """Generate a route that passes through junctions for turn diversity.

        Strategy:
        1. Pick a random start point
        2. Find nearby junctions
        3. Chain multiple junction-to-junction segments
        4. Track turn distribution and bias toward underrepresented turns
        """
        if not self._junctions:
            logger.warning("No junctions found, falling back to random route")
            return self._generate_random_route(rng)

        # Start from random spawn point
        start_idx = rng.integers(len(self.spawn_points))
        current_wp = self.map.get_waypoint(self.spawn_points[int(start_idx)].location)

        route_segments: list[tuple[Any, str]] = []
        turn_counts: dict[str, int] = {
            "LEFT": 0,
            "RIGHT": 0,
            "STRAIGHT": 0,
        }
        target_fracs = self._target_fracs_for_route()
        total_waypoints = 0

        max_segments = 20  # Safety limit

        for _ in range(max_segments):
            remaining_budget = self.config.max_waypoints - total_waypoints
            if remaining_budget <= 0:
                break

            # Find the nearest junction we haven't visited recently
            target_junction = self._find_target_junction(
                current_wp.transform.location,
                turn_counts,
                rng,
            )

            if target_junction is None:
                break

            # Get a waypoint near the junction entry
            junction_entry = target_junction["entry_waypoints"]
            if not junction_entry:
                continue

            entry_wp = junction_entry[int(rng.integers(len(junction_entry)))]

            # Trace route from current position to junction
            segment = self._trace_route(current_wp, entry_wp)
            if not segment:
                continue

            if len(segment) > remaining_budget:
                segment = segment[:remaining_budget]

            # Count turns in this segment
            self._update_turn_counts(turn_counts, segment)

            added = self._append_route_segment(route_segments, segment)
            total_waypoints += added

            if route_segments:
                segment_tail_wp = self._waypoint_from_transform(route_segments[-1][0])
                if segment_tail_wp is not None:
                    current_wp = segment_tail_wp

            remaining_budget = self.config.max_waypoints - total_waypoints
            if remaining_budget <= 0:
                break

            # Move current position past the junction using a traced
            # connector path, not a random coordinate jump.
            exit_wps = target_junction["exit_waypoints"]
            if exit_wps:
                junction_segment = self._select_junction_exit_segment(
                    current_wp,
                    exit_wps,
                    turn_counts,
                    rng,
                    target_fracs=target_fracs,
                )
                if junction_segment:
                    if len(junction_segment) > remaining_budget:
                        junction_segment = junction_segment[:remaining_budget]

                    self._update_turn_counts(turn_counts, junction_segment)
                    added = self._append_route_segment(route_segments, junction_segment)
                    total_waypoints += added

                    if route_segments:
                        junction_tail_wp = self._waypoint_from_transform(route_segments[-1][0])
                        if junction_tail_wp is not None:
                            current_wp = junction_tail_wp

                    if total_waypoints >= self.config.max_waypoints:
                        break

        # If we didn't get enough waypoints, extend with random driving
        if total_waypoints < self.config.min_waypoints:
            remaining_budget = self.config.max_waypoints - total_waypoints
            if remaining_budget > 0:
                extra_needed = min(self.config.min_waypoints - total_waypoints, remaining_budget)
                extra = self._extend_route(current_wp, extra_needed)
                total_waypoints += self._append_route_segment(route_segments, extra)

        return route_segments

    @staticmethod
    def _canonical_option_name(option_name: str) -> str:
        normalized = str(option_name).strip().replace("_", "").replace(" ", "").lower()
        mapping = {
            "left": "LEFT",
            "right": "RIGHT",
            "straight": "STRAIGHT",
            "lanefollow": "LANEFOLLOW",
            "changelaneleft": "CHANGELANELEFT",
            "changelaneright": "CHANGELANERIGHT",
            "void": "VOID",
        }
        return mapping.get(normalized, str(option_name).strip().upper())

    def _update_turn_counts(
        self,
        turn_counts: dict[str, int],
        segment: list[tuple[Any, str]],
    ) -> None:
        for _, road_option in segment:
            canonical = self._canonical_option_name(road_option)
            if canonical in ("LEFT", "RIGHT", "STRAIGHT"):
                turn_counts[canonical] = int(turn_counts.get(canonical, 0)) + 1

    def _ensure_campaign_state(self) -> None:
        if not hasattr(self, "_campaign_turn_counts"):
            self._campaign_turn_counts = {
                "LEFT": 0,
                "RIGHT": 0,
                "STRAIGHT": 0,
            }
        if not hasattr(self, "_campaign_turn_deficit_ema"):
            self._campaign_turn_deficit_ema = {
                "LEFT": 0.0,
                "RIGHT": 0.0,
                "STRAIGHT": 0.0,
            }

    def _base_target_fracs(self) -> dict[str, float]:
        return {
            "LEFT": max(0.0, float(self.config.target_left)),
            "RIGHT": max(0.0, float(self.config.target_right)),
            "STRAIGHT": max(0.0, float(self.config.target_straight)),
        }

    def _target_fracs_for_route(self) -> dict[str, float]:
        base = self._base_target_fracs()
        if not bool(self.config.campaign_turn_balance_enabled):
            return base

        self._ensure_campaign_state()

        labels = list(base.keys())
        observed_total = sum(int(self._campaign_turn_counts.get(lbl, 0)) for lbl in labels)
        min_samples = max(0, int(self.config.campaign_turn_balance_min_turn_samples))
        if observed_total < min_samples:
            return base

        alpha = float(np.clip(float(self.config.campaign_turn_balance_smoothing_alpha), 0.0, 1.0))
        max_adjust = max(0.0, float(self.config.campaign_turn_balance_max_adjust))

        adjusted: dict[str, float] = {}
        for lbl in labels:
            base_frac = base[lbl]
            observed_frac = float(self._campaign_turn_counts.get(lbl, 0)) / float(observed_total)
            deficit = base_frac - observed_frac
            prev_ema = float(self._campaign_turn_deficit_ema.get(lbl, 0.0))
            smoothed_deficit = alpha * deficit + (1.0 - alpha) * prev_ema
            self._campaign_turn_deficit_ema[lbl] = smoothed_deficit

            correction = float(np.clip(smoothed_deficit, -max_adjust, max_adjust))
            adjusted[lbl] = max(0.0, base_frac + correction)

        if sum(adjusted.values()) <= 1e-9:
            return base

        return adjusted

    def _record_campaign_turn_counts(
        self,
        route: list[tuple[Any, str]],
    ) -> None:
        if not route:
            return

        self._ensure_campaign_state()

        for _, road_option in route:
            canonical = self._canonical_option_name(road_option)
            if canonical in self._campaign_turn_counts:
                self._campaign_turn_counts[canonical] += 1

    def _choose_desired_turn(
        self,
        turn_counts: dict[str, int],
        rng: np.random.Generator,
        target_fracs: dict[str, float] | None = None,
    ) -> str:
        if target_fracs is None:
            target_fracs = self._target_fracs_for_route()

        labels = list(target_fracs.keys())
        total_target = sum(target_fracs.values())
        if total_target <= 1e-9:
            return "STRAIGHT"

        observed_total = sum(int(turn_counts.get(lbl, 0)) for lbl in labels)
        if observed_total <= 0:
            weights = np.array([target_fracs[lbl] for lbl in labels], dtype=np.float64)
        else:
            deficits = []
            for lbl in labels:
                observed_frac = float(turn_counts.get(lbl, 0)) / float(observed_total)
                deficit = max(target_fracs[lbl] - observed_frac, 0.0)
                deficits.append(deficit)
            deficit_sum = float(sum(deficits))
            if deficit_sum > 1e-9:
                weights = np.array(deficits, dtype=np.float64)
            else:
                weights = np.array([target_fracs[lbl] for lbl in labels], dtype=np.float64)

        weights = weights / weights.sum()
        chosen = int(rng.choice(len(labels), p=weights))
        return labels[chosen]

    def _select_junction_exit_segment(
        self,
        current_wp: Any,
        exit_waypoints: list[Any],
        turn_counts: dict[str, int],
        rng: np.random.Generator,
        target_fracs: dict[str, float] | None = None,
    ) -> list[tuple[Any, str]]:
        if not exit_waypoints:
            return []

        desired_turn = self._choose_desired_turn(turn_counts, rng, target_fracs=target_fracs)
        candidate_segments: list[tuple[int, int, int, list[tuple[Any, str]]]] = []

        for candidate_idx in rng.permutation(len(exit_waypoints)):
            exit_wp = exit_waypoints[int(candidate_idx)]
            segment = self._trace_route(current_wp, exit_wp)
            if not segment:
                continue

            desired_hits = 0
            any_turn_hits = 0
            for _, road_option in segment:
                canonical = self._canonical_option_name(road_option)
                if canonical in ("LEFT", "RIGHT", "STRAIGHT"):
                    any_turn_hits += 1
                    if canonical == desired_turn:
                        desired_hits += 1

            candidate_segments.append(
                (
                    desired_hits,
                    any_turn_hits,
                    -len(segment),
                    segment,
                )
            )

        if not candidate_segments:
            return []

        candidate_segments.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return candidate_segments[0][3]

    def _find_target_junction(
        self,
        current_location: Any,
        turn_counts: dict[str, int],
        rng: np.random.Generator,
    ) -> dict | None:
        """Find the best junction to route through next.

        Biases toward junctions that would help balance the turn distribution.
        """
        if not self._junctions:
            return None

        # Calculate distances to all junctions
        dist_list: list[float] = []
        for junc in self._junctions:
            loc = junc["location"]
            dist = ((current_location.x - loc.x) ** 2 + (current_location.y - loc.y) ** 2) ** 0.5
            dist_list.append(dist)

        distances = np.array(dist_list)

        # Filter to reachable junctions (not too close, not too far)
        mask = (distances > 20.0) & (distances < 500.0)
        valid_indices = np.where(mask)[0]

        if len(valid_indices) == 0:
            return None

        # Weight by inverse distance (prefer closer junctions)
        valid_dists = distances[valid_indices]
        weights = 1.0 / (valid_dists + 1.0)
        weights = weights / weights.sum()

        chosen_idx = valid_indices[rng.choice(len(valid_indices), p=weights)]
        return self._junctions[int(chosen_idx)]

    def _trace_route(
        self,
        start_wp: Any,
        end_wp: Any,
    ) -> list[tuple[Any, str]]:
        """Trace a route between two waypoints using configured planner.

        Returns list of (transform, road_option_name) tuples.
        """
        try:
            route = self.route_planner.trace_route(
                start_wp.transform.location,
                end_wp.transform.location,
            )

            result = []
            for wp, road_option in route:
                option_name = getattr(road_option, "name", str(road_option))
                result.append((wp.transform, option_name))

            return result

        except Exception as e:
            logger.error("Route tracing failed: %s", e)
            return []

    def _append_route_segment(
        self,
        route_segments: list[tuple[Any, str]],
        segment: list[tuple[Any, str]],
    ) -> int:
        """Append a traced segment while skipping duplicated seam points."""
        if not segment:
            return 0

        start_idx = 0
        if route_segments:
            last_loc = route_segments[-1][0].location
            tol = max(0.05, float(self.config.sampling_resolution) * 0.05)
            while start_idx < len(segment):
                cur_loc = segment[start_idx][0].location
                if last_loc.distance(cur_loc) <= tol:
                    start_idx += 1
                    continue
                break

        if start_idx >= len(segment):
            return 0

        route_segments.extend(segment[start_idx:])
        return len(segment) - start_idx

    def _waypoint_from_transform(self, transform: Any) -> Any | None:
        """Project a transform back to the nearest driving waypoint."""
        try:
            return self.map.get_waypoint(
                transform.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except Exception:
            return None

    def _extend_route(
        self,
        start_wp: Any,
        n_waypoints: int,
    ) -> list[tuple[Any, str]]:
        """Extend a route by following the road forward."""
        result = []
        current = start_wp

        for _ in range(n_waypoints):
            next_wps = current.next(self.config.sampling_resolution)
            if not next_wps:
                break
            current = next_wps[0]
            result.append((current.transform, "LaneFollow"))

        return result

    def get_distribution_report(
        self,
        route: list[tuple[Any, str]],
    ) -> dict[str, float]:
        """Compute turn distribution for a generated route."""
        counts: dict[str, int] = defaultdict(int)
        for _, road_option in route:
            counts[road_option] += 1

        total = len(route)
        if total == 0:
            return {}

        return {option: count / total for option, count in sorted(counts.items())}
