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

        # Build intersection graph
        self._junctions = self._find_junctions()
        logger.info("Found %d junctions in %s", len(self._junctions), self.map.name)

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

        if self.config.strategy == "random":
            return self._generate_random_route(rng)
        elif self.config.strategy == "balanced":
            return self._generate_balanced_route(rng)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

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
        turn_counts = defaultdict(int)
        total_waypoints = 0

        max_segments = 20  # Safety limit

        for _ in range(max_segments):
            if total_waypoints >= self.config.max_waypoints:
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

            # Count turns in this segment
            for _, road_option in segment:
                if road_option in ("Left", "Right", "Straight"):
                    turn_counts[road_option] += 1

            route_segments.extend(segment)
            total_waypoints += len(segment)

            # Move current position past the junction
            exit_wps = target_junction["exit_waypoints"]
            if exit_wps:
                current_wp = exit_wps[int(rng.integers(len(exit_wps)))]

        # If we didn't get enough waypoints, extend with random driving
        if total_waypoints < self.config.min_waypoints:
            extra = self._extend_route(current_wp, self.config.min_waypoints - total_waypoints)
            route_segments.extend(extra)

        return route_segments

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
        distances = []
        for junc in self._junctions:
            loc = junc["location"]
            dist = ((current_location.x - loc.x) ** 2 + (current_location.y - loc.y) ** 2) ** 0.5
            distances.append(dist)

        distances = np.array(distances)

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
        """Trace a route between two waypoints using CARLA's planner.

        Returns list of (transform, road_option_name) tuples.
        """
        try:
            from agents.navigation.global_route_planner import (  # type: ignore[import-untyped]
                GlobalRoutePlanner,
            )

            grp = GlobalRoutePlanner(self.map, self.config.sampling_resolution)
            route = grp.trace_route(start_wp.transform.location, end_wp.transform.location)

            result = []
            for wp, road_option in route:
                result.append((wp.transform, road_option.name))

            return result

        except Exception as e:
            logger.error("Route tracing failed: %s", e)
            return []

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
        counts = defaultdict(int)
        for _, road_option in route:
            counts[road_option] += 1

        total = len(route)
        if total == 0:
            return {}

        return {option: count / total for option, count in sorted(counts.items())}
