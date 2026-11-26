"""Leaderboard-style scoring of a closed-loop route, on plain per-tick values.

Rules and constants follow the CARLA Leaderboard 2.0: leaderboard/utils/statistics_manager.py,
and scenario_runner's scenarioatomics/atomic_criteria.py and timer.py (branch leaderboard-2.0).
The criteria a route carries there are route completion, outside route lanes, collisions, red
lights, stop signs, deviation from the route and being blocked, plus the route timeout. Its
minimum-speed criterion, which compares the ego with the background traffic, is not scored.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

PENALTY = {
    "collision_pedestrian": 0.50,
    "collision_vehicle": 0.60,
    "collision_static": 0.65,
    "red_light": 0.70,
    "stop_sign": 0.80,
}
COLLISION_RADIUS = 5.0  # m: a collision this close to the last counted one is the same one
COLLISION_ID_TIME = 5.0  # s: a collision with the same actor this soon is the same one
COLLISION_MIN_SPEED = 0.1  # m/s: collisions while the ego is slower are not counted
RED_LIGHT_DISTANCE = 15.0  # m: a light farther than this from the ego is not checked
STOP_SPEED = 0.1  # m/s: slower than this counts as stopping at a stop sign
BLOCKED_SPEED = 0.1  # m/s
BLOCKED_TIME = 180.0  # s slower than BLOCKED_SPEED ends the route
DEVIATION_DISTANCE = 30.0  # m from the route ends it
DEVIATION_SAFE = 15.0  # m: route passed farther than this from the car counts as off route
DEVIATION_SHARE = 30.0  # % of the route passed off route ends it
TIMEOUT_MIN = 300.0  # s before any progress
TIMEOUT_LIMIT_SHARE = 0.10  # each metre of progress adds 1 / (0.10 * speed limit) seconds
COMPLETION_DISTANCE = 10.0  # m from the last route point, with COMPLETION_SHARE passed
COMPLETION_SHARE = 99.0  # %
LANE_MARGIN = 0.5  # m beyond half the lane width counts as outside the lanes
WRONG_HEADING = 120.0  # deg between the car and the lane on leaving a junction
LANE_FLIP = 150.0  # deg between consecutive lanes that marks a change of direction


def cumulative_length(points: np.ndarray) -> np.ndarray:
    """Metres along the points from the first to each."""
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])


def passed(position: np.ndarray, point: np.ndarray, forward: np.ndarray) -> bool:
    """Whether the car at position is past a route point whose direction is forward."""
    return float(np.dot(position - point, forward)) > 0


def tail_crosses(
    position: np.ndarray, yaw_deg: float, half_length: float, a: np.ndarray, b: np.ndarray
) -> bool:
    """Whether the segment behind the car, from 0.8 to 1 + 1 half-lengths back, crosses a-b."""
    back = -np.array([math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))])
    p, q = position + back * 0.8 * half_length, position + back * (half_length + 1.0)
    return _segments_cross(p, q, a, b)


def _segments_cross(p: np.ndarray, q: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    def side(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
        return float((v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0]))

    return side(p, q, a) * side(p, q, b) <= 0 and side(a, b, p) * side(a, b, q) <= 0


class RouteProgress:
    """Share of the route passed: the farthest of the next two route points the car is past."""

    WINDOW = 2

    def __init__(self, points: np.ndarray, forwards: np.ndarray) -> None:
        self.points, self.forwards = np.asarray(points, float), np.asarray(forwards, float)
        self.cumulative = cumulative_length(self.points)
        self.index = 0
        self.completion = 0.0

    @property
    def length(self) -> float:
        return float(self.cumulative[-1])

    @property
    def metres(self) -> float:
        return float(self.cumulative[self.index])

    def update(self, position: np.ndarray) -> float:
        """Advance with the car's position; returns the metres of route newly passed."""
        before = self.index
        for i in range(self.index, min(self.index + self.WINDOW + 1, len(self.points))):
            if passed(position, self.points[i], self.forwards[i]):
                self.index = i
        self.completion = 100.0 * self.metres / max(self.length, 1e-6)
        if self.completion > COMPLETION_SHARE and (
            np.linalg.norm(position - self.points[-1]) < COMPLETION_DISTANCE
        ):
            self.completion = 100.0
        return float(self.cumulative[self.index] - self.cumulative[before])

    @property
    def done(self) -> bool:
        return self.completion >= 100.0


class RouteDeviation:
    """Ends the route when the car is 30 m from it, or has passed 30% of it more than 15 m away."""

    WINDOW = 5

    def __init__(self, points: np.ndarray) -> None:
        self.points = np.asarray(points, float)
        self.cumulative = cumulative_length(self.points)
        self.index = 0
        self.off_metres = 0.0

    def update(self, position: np.ndarray) -> bool:
        window = self.points[self.index : self.index + self.WINDOW + 1]
        distances = np.linalg.norm(window - position, axis=1)
        nearest = len(distances) - 1 - int(np.argmin(distances[::-1]))  # ties go to the later
        if distances[nearest] >= DEVIATION_DISTANCE:
            return True
        if nearest > 0:
            start, self.index = self.index, self.index + nearest
            if distances[nearest] >= DEVIATION_SAFE:
                self.off_metres += float(self.cumulative[self.index] - self.cumulative[start])
        return 100.0 * self.off_metres / max(float(self.cumulative[-1]), 1e-6) > DEVIATION_SHARE


class RouteTimeout:
    """Starts at 300 s; each metre of progress adds 1 / (10% of the speed limit) seconds."""

    def __init__(self) -> None:
        self.seconds = TIMEOUT_MIN

    def update(self, elapsed: float, progress_metres: float, speed_limit_kmh: float) -> bool:
        if progress_metres > 0:
            limit = max(speed_limit_kmh, 1.0) / 3.6
            self.seconds += progress_metres / (TIMEOUT_LIMIT_SHARE * limit)
        return elapsed > self.seconds


class Blocked:
    """Ends the route after 180 s below 0.1 m/s, counted from the last time it was faster."""

    def __init__(self) -> None:
        self.since: float | None = None

    def update(self, elapsed: float, speed: float) -> bool:
        if self.since is None or speed >= BLOCKED_SPEED:
            self.since = elapsed
        return elapsed - self.since > BLOCKED_TIME


def collision_kind(type_id: str) -> str | None:
    if ("static" in type_id or "traffic" in type_id) and "sidewalk" not in type_id:
        return "collision_static"
    if "vehicle" in type_id:
        return "collision_vehicle"
    if "walker" in type_id:
        return "collision_pedestrian"
    return None


class CollisionFilter:
    """Counts a collision once: not again with the same actor within 5 s, nor within 5 m of the
    last counted one, nor while the ego is slower than 0.1 m/s."""

    def __init__(self) -> None:
        self.last_id: int | None = None
        self.last_time = -math.inf
        self.last_place: np.ndarray | None = None

    def near_last(self, position: np.ndarray) -> bool:
        if self.last_place is None:
            return False
        return float(np.linalg.norm(position - self.last_place)) <= COLLISION_RADIUS

    def update(self, elapsed: float, position: np.ndarray) -> None:
        if not self.near_last(position):
            self.last_place = None
        if elapsed - self.last_time > COLLISION_ID_TIME:
            self.last_id = None

    def count(
        self, elapsed: float, position: np.ndarray, speed: float, other_id: int, type_id: str
    ) -> str | None:
        """The kind of collision to count, or None if it is the same one or not counted."""
        if other_id == self.last_id or speed < COLLISION_MIN_SPEED or self.near_last(position):
            return None
        kind = collision_kind(type_id)
        if kind is None:
            return None
        self.last_id = other_id if other_id != 0 else None
        self.last_time, self.last_place = elapsed, np.array(position, float)
        return kind


class StopSigns:
    """A stop sign is run when the car leaves its zone without having gone below 0.1 m/s."""

    def __init__(self) -> None:
        self.target: int | None = None
        self.stopped = False
        self.last_failed: int | None = None

    def update(self, affecting: Collection[int], ahead: int | None, speed: float) -> int | None:
        """affecting: stop signs whose zone holds the car's path just ahead; ahead: the one to
        watch, if the car drives with its lane. Returns a stop sign run."""
        if self.target is None:
            self.target = ahead
            return None
        self.stopped = self.stopped or speed < STOP_SPEED
        if self.target in affecting:
            return None
        run = None
        if not self.stopped and self.last_failed != self.target:
            run = self.last_failed = self.target
        self.target, self.stopped = None, False
        return run


@dataclass
class StopLine:
    """Where a traffic light stops one lane: across the lane, 0.6 lane widths each side."""

    light_id: int
    a: np.ndarray
    b: np.ndarray
    road_id: int
    lane_id: int
    forward: np.ndarray  # the lane's direction


class RedLights:
    """A red light is run when the segment behind the car crosses the light's stop line, in that
    lane and going its way, with the light within 15 m of the car. Counted once per light in a
    row."""

    def __init__(self, lines: Sequence[StopLine]) -> None:
        self.lines = list(lines)
        self.last_id: int | None = None

    def update(
        self,
        position: np.ndarray,
        yaw_deg: float,
        half_length: float,
        tail_lane: tuple[int, int],
        red: Mapping[int, np.ndarray],
    ) -> int | None:
        """red: centre of each red light's trigger zone, by light id. Returns a light run."""
        forward = np.array([math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))])
        for line in self.lines:
            centre = red.get(line.light_id)
            if centre is None or line.light_id == self.last_id:
                continue
            if np.linalg.norm(np.asarray(centre)[:2] - position) > RED_LIGHT_DISTANCE:
                continue
            if tail_lane != (line.road_id, line.lane_id) or np.dot(forward, line.forward) <= 0:
                continue
            if tail_crosses(position, yaw_deg, half_length, line.a, line.b):
                self.last_id = line.light_id
                return line.light_id
        return None


@dataclass
class LanePlace:
    """The driving lane under the car, as CARLA's map projects it."""

    road_id: int
    lane_id: int
    junction: bool
    yaw: float  # deg
    distance: float  # m from the car to the lane centre
    width: float  # m


class OutsideLanes:
    """Share of the route the car passed off the driving lanes or against the lane's direction."""

    def __init__(self) -> None:
        self.total = 0.0
        self.outside = 0.0
        self.wrong_lane = False
        self.last: LanePlace | None = None

    def update(self, place: LanePlace, car_yaw: float, progress_metres: float) -> bool:
        """Returns whether the car is outside or in a wrong lane now."""
        off_road = place.distance > place.width / 2 + LANE_MARGIN
        last = self.last
        if place.junction:
            self.wrong_lane = False
        elif last is not None and (last.road_id, last.lane_id) != (place.road_id, place.lane_id):
            if last.junction:
                diff = (car_yaw - place.yaw) % 360
                self.wrong_lane = WRONG_HEADING < diff < 360 - WRONG_HEADING
            elif LANE_FLIP < (place.yaw - last.yaw) % 360 < 360 - LANE_FLIP:
                self.wrong_lane = not self.wrong_lane
        self.last = place
        bad = off_road or self.wrong_lane
        self.total += progress_metres
        self.outside += progress_metres if bad else 0.0
        return bad

    @property
    def percentage(self) -> float:
        return 100.0 * self.outside / self.total if self.total > 0 else 0.0


@dataclass
class Infraction:
    kind: str
    tick: int
    x: float
    y: float
    detail: str = ""


@dataclass
class RouteScore:
    route_completion: float  # %
    infraction_penalty: float
    driving_score: float  # %
    outside_lanes: float  # % of the route passed
    status: str  # Perfect, Completed, or Failed - <reason>
    infractions: list[Infraction] = field(default_factory=list)

    def count(self, kind: str) -> int:
        return sum(i.kind == kind for i in self.infractions)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RouteScore:
        infractions = [Infraction(**i) for i in data["infractions"]]
        return cls(**{**data, "infractions": infractions})


def score_route(
    completion: float,
    infractions: Sequence[Infraction],
    outside_lanes: float,
    failure: str | None = None,
) -> RouteScore:
    """Driving score = route completion x the product of a penalty per infraction, times
    (1 - outside share). A route that ends early (deviation, blocked, timeout) keeps its
    completion and penalty; only its status says why it ended."""
    penalty = 1.0
    for infraction in infractions:
        penalty *= PENALTY[infraction.kind]
    penalty *= 1.0 - outside_lanes / 100.0
    completion = round(completion, 6)
    if completion >= 100.0:
        status = "Completed" if infractions or outside_lanes > 0 else "Perfect"
    else:
        status = f"Failed - {failure or 'route not finished'}"
    return RouteScore(
        route_completion=completion,
        infraction_penalty=round(penalty, 6),
        driving_score=round(max(completion * penalty, 0.0), 6),
        outside_lanes=round(outside_lanes, 6),
        status=status,
        infractions=list(infractions),
    )


def summarize(scores: Sequence[RouteScore], lengths_m: Sequence[float]) -> dict[str, float]:
    """Means over routes, and each infraction per km driven (route length x completion)."""
    km = max(sum(L / 1000 * s.route_completion / 100 for s, L in zip(scores, lengths_m)), 0.001)
    out = {
        "driving_score": float(np.mean([s.driving_score for s in scores])),
        "route_completion": float(np.mean([s.route_completion for s in scores])),
        "infraction_penalty": float(np.mean([s.infraction_penalty for s in scores])),
        "outside_lanes": float(np.mean([s.outside_lanes for s in scores])),
        "km_driven": km,
    }
    for kind in PENALTY:
        out[f"{kind}_per_km"] = sum(s.count(kind) for s in scores) / km
    for reason in ("deviated", "blocked", "timed out"):
        out[f"{reason.replace(' ', '_')}_share"] = float(
            np.mean([reason in s.status for s in scores])
        )
    return out
