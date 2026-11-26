"""A route planned through keypoints in order, as the CARLA leaderboard plans its routes."""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from src.data.labels import ROUTE_RESOLUTION


def plan_route(planner: Any, keypoints: Sequence[Any]) -> List[Tuple[Any, Any]]:
    """The planner's (waypoint, road option) pairs through the keypoints (carla.Location), with
    the point two legs share kept once."""
    route: List[Tuple[Any, Any]] = []
    for a, b in zip(keypoints, keypoints[1:]):
        part = planner.trace_route(a, b)
        if route and part:
            gap = part[0][0].transform.location.distance(route[-1][0].transform.location)
            if gap < 0.5 * ROUTE_RESOLUTION:
                part = part[1:]
        route += part
    return route
