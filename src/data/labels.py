"""Training targets computed from an episode's raw signals."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, fields

import numpy as np
from matplotlib.path import Path as Outline

from src.data import hdf5_schema as S

PRE_COLLISION_S = 3.0  # the expert was already failing this close to a collision
STUCK_SPEED = 0.1  # m/s; the collector times an episode out after standing below it


def usable_end(speed: np.ndarray, fps: int, status: str, collision_frame: int) -> int:
    """End (exclusive) of the recorded frames that make valid targets.

    speed: (N,) speed of every recorded frame, m/s. Frames from PRE_COLLISION_S before a
    collision on are dropped. An episode that timed out loses its final stop: the
    vehicle stood still until the collector gave up, so those frames show it stuck.
    """
    end = len(speed)
    if collision_frame > 0:
        end = min(end, collision_frame - round(PRE_COLLISION_S * fps))
    if status == "timeout":
        moving = np.flatnonzero(speed[: max(end, 0)] >= STUCK_SPEED)
        end = int(moving[-1]) + 1 if moving.size else 0
    return max(end, 0)


def ego_waypoints(position: np.ndarray, yaw: np.ndarray, offsets: Sequence[int]) -> np.ndarray:
    """Positions at frames t + offsets, in the vehicle's frame at t.

    position: (T + lookahead, 3) world positions in metres; frames past its end are
    clamped to the last one. yaw: (T,) CARLA yaw in degrees, clockwise seen from above.
    Returns (T, K, 2) float32 offsets in metres: x forward, y to the right.
    """
    t = len(yaw)
    future = np.minimum(np.arange(t)[:, None] + np.asarray(offsets), len(position) - 1)
    d = position[future, :2].astype(np.float64) - position[:t, None, :2]
    return _vehicle_frame(d[..., 0], d[..., 1], yaw[:, None])


def _vehicle_frame(dx: np.ndarray, dy: np.ndarray, yaw: np.ndarray | float) -> np.ndarray:
    """World offsets (..., ) as (..., 2) float32: x forward and y right of a vehicle at yaw."""
    psi = np.radians(np.asarray(yaw, np.float64))
    cos, sin = np.cos(psi), np.sin(psi)
    return np.stack([cos * dx + sin * dy, -sin * dx + cos * dy], axis=-1).astype(np.float32)


# Route labels: what the path does at junctions, where it leads, and the situation the
# vehicle is in. Computed from a path (a driven trajectory or a planned route), the junction
# outlines of its town (scripts/export_junctions.py) and, for situations, the traffic rules
# and road users the expert recorded around it.

# RoadOption values, with CARLA's meaning: LEFT, RIGHT or STRAIGHT through a junction.
LEFT, RIGHT, STRAIGHT, LANEFOLLOW = 1, 2, 3, 4
APPROACH = 30.0  # m: a junction's command, and "approaching", start this far before it
STRAIGHT_LIMIT = 35.0  # deg: less heading change across a junction is going straight, as in
# CARLA's route planner (GlobalRoutePlanner._turn_decision)
TURN_MARGIN = (5.0, 10.0)  # m measured before entering and after leaving, to cover the turn
MIN_CROSSING = 3.0  # m: a shorter visit only clips a junction's outline
JUNCTION_GAP = 10.0  # m: a junction's command starts as soon as the previous junction is left
# if its approach would otherwise begin less than this after it
TARGET_DISTANCE = 30.0  # m along the path to the target point
TURNING_CURVATURE = 3.0  # deg/m: sharper than v3's road curves (a radius under 19 m)
PATH_STEP = 0.5  # m: resolution of the path's heading
STANDING_SPEED = 0.14  # m/s (0.5 km/h)
RULE_AHEAD = 30.0  # m: a red light or stop sign this close ahead is approached or, standing,
# stopped for; this includes waiting in a queue for it
VEHICLE_REACH = 25.0  # m, bumper to bumper: a vehicle in the path this close is followed or,
# standing, stopped for (the expert's shortest following distance)
VEHICLE_LANE = 1.75  # m: a vehicle centred this close to the line straight ahead is in the path
PEDESTRIAN_REACH = 15.0  # m: the expert's pedestrian hazard distance
PEDESTRIAN_LANE = 2.5  # m: a pedestrian this close to the line straight ahead, now or 1 s on, is
# in the path (the expert's lane half-width)

SITUATIONS = (
    "approaching junction",
    "in junction",
    "turning",
    "approaching red light",
    "stopped at red light",
    "approaching stop sign",
    "stopped at stop sign",
    "following vehicle",
    "stopped for vehicle",
    "yielding to pedestrian",
    "stopped for pedestrian",
)
_RED = [code for code, name in S.TRAFFIC_LIGHT_STATE_NAMES.items() if name in ("Yellow", "Red")]


def junction_ids(xy: np.ndarray, outlines: Sequence[tuple[int, np.ndarray]]) -> np.ndarray:
    """(N,) id of the junction outline each point lies in, -1 outside all of them."""
    ids = np.full(len(xy), -1)
    for junction, outline in outlines:
        ids[Outline(outline).contains_points(xy)] = junction
    return ids


@dataclass(frozen=True)
class Route:
    command: np.ndarray  # (N,) RoadOption: LEFT, RIGHT or STRAIGHT from APPROACH before a
    # junction until it is left, LANEFOLLOW elsewhere
    approaching: np.ndarray  # (N,) bool: within APPROACH before a junction
    in_junction: np.ndarray  # (N,) bool
    turning: np.ndarray  # (N,) bool: the path curves by TURNING_CURVATURE or more here
    target: np.ndarray  # (N, 2) float32: TARGET_DISTANCE ahead along the path, x forward
    # and y right of the vehicle, metres

    def __getitem__(self, rows: slice) -> Route:
        return Route(*(getattr(self, field.name)[rows] for field in fields(self)))


def route(position: np.ndarray, yaw: np.ndarray, junction: np.ndarray) -> Route:
    """Route labels of every row of a path.

    position: (N, 2+) world positions in metres; yaw: (N,) CARLA yaw in degrees, clockwise
    seen from above, so a right turn increases it; junction: (N,) from junction_ids.
    Everything is measured along the distance travelled, so a vehicle waiting before a
    junction keeps its command. Past the end of the path, the target point continues
    straight along the final heading.
    """
    xy = position[:, :2].astype(np.float64)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    heading = np.degrees(np.unwrap(np.radians(yaw.astype(np.float64))))
    moving = np.r_[True, np.diff(s) > 1e-3]

    def heading_at(distance: np.ndarray | float) -> np.ndarray:
        return np.asarray(np.interp(distance, s[moving], heading[moving]))

    cut = np.flatnonzero(np.diff(junction)) + 1
    crossings = [
        (a, b)
        for a, b in zip(np.r_[0, cut], np.r_[cut, len(s)])
        if junction[a] >= 0 and s[b - 1] - s[a] >= MIN_CROSSING
    ]
    command = np.full(len(s), LANEFOLLOW)
    in_junction = np.zeros(len(s), bool)
    approaching = np.zeros(len(s), bool)
    left = [s[b - 1] for a, b in crossings]  # where each junction is left
    # Latest junction first, so an earlier one keeps its rows from the next one's approach.
    for i in reversed(range(len(crossings))):
        a, b = crossings[i]
        entry, leave = s[a], s[b - 1]
        turn = heading_at(leave + TURN_MARGIN[1]) - heading_at(entry - TURN_MARGIN[0])
        way = STRAIGHT if abs(turn) < STRAIGHT_LIMIT else RIGHT if turn > 0 else LEFT
        start = entry - APPROACH
        if i > 0 and start - left[i - 1] < JUNCTION_GAP:
            start = left[i - 1]
        command[(s >= start) & (s <= leave)] = way
        approaching |= (s >= start) & (s < entry)
        in_junction[a:b] = True
    approaching &= ~in_junction

    grid = np.arange(0.0, max(s[-1], 2 * PATH_STEP), PATH_STEP)
    h = heading_at(grid)
    curvature = (np.interp(grid + 1, grid, h) - np.interp(grid - 1, grid, h)) / 2
    turning = np.abs(np.interp(s, grid, curvature)) >= TURNING_CURVATURE

    ahead = s + TARGET_DISTANCE
    beyond = np.maximum(ahead - s[-1], 0.0)
    end = np.radians(heading[-1])
    tx = np.interp(ahead, s[moving], xy[moving, 0]) + beyond * np.cos(end)
    ty = np.interp(ahead, s[moving], xy[moving, 1]) + beyond * np.sin(end)
    target = _vehicle_frame(tx - xy[:, 0], ty - xy[:, 1], yaw)
    return Route(command, approaching, in_junction, turning, target)


def vehicle_gaps(records: Sequence[str | bytes]) -> np.ndarray:
    """(N,) bumper-to-bumper gap in metres to the nearest vehicle in the path ahead, inf if none.

    records: the expert's nearby-vehicle JSON of each row. A vehicle is in the path when its
    box overlaps the expert's corridor along the route, or its centre lies within
    VEHICLE_LANE of the line straight ahead, whichever way it is heading.
    """
    gaps = np.full(len(records), np.inf)
    for i, record in enumerate(records):
        if len(record) > 2:
            ahead = [
                v["distance"]
                for v in json.loads(record)
                if v["longitudinal"] > 0 and (v["obb_in_lane"] or abs(v["lateral"]) < VEHICLE_LANE)
            ]
            gaps[i] = min(ahead, default=np.inf)
    return gaps


def pedestrian_distances(records: Sequence[str | bytes]) -> np.ndarray:
    """(N,) distance in metres to the nearest pedestrian in the path ahead, inf if none.

    records: the expert's nearby-pedestrian JSON of each row. A pedestrian is in the path when
    within PEDESTRIAN_LANE of the line straight ahead, now or where its velocity takes it in
    1 s; its distance is then the nearer of the two.
    """
    found = np.full(len(records), np.inf)
    for i, record in enumerate(records):
        if len(record) > 2:
            found[i] = min(
                (
                    min(p["distance"], p["predicted_distance"])
                    for p in json.loads(record)
                    if (p["is_ahead"] and abs(p["lateral"]) < PEDESTRIAN_LANE)
                    or (p["predicted_ahead"] and abs(p["predicted_lateral"]) < PEDESTRIAN_LANE)
                ),
                default=np.inf,
            )
    return found


def situations(
    route: Route,
    speed: np.ndarray,
    light_state: np.ndarray,
    light_distance: np.ndarray,
    stop_distance: np.ndarray,
    vehicle_gap: np.ndarray,
    pedestrian_distance: np.ndarray,
) -> np.ndarray:
    """(N, len(SITUATIONS)) bool: which situations hold at each row; several can at once.

    speed in m/s; light_state as TRAFFIC_LIGHT_STATE_NAMES codes, with light_distance and
    stop_distance in metres, negative when there is none; vehicle_gap and
    pedestrian_distance from vehicle_gaps and pedestrian_distances. Each cause comes as a
    pair, moving and standing: a "stopped" situation always stands still.
    """
    standing = speed < STANDING_SPEED
    red = _red_ahead(light_state, light_distance)
    sign = (stop_distance >= 0) & (stop_distance <= RULE_AHEAD)
    vehicle = vehicle_gap <= VEHICLE_REACH
    pedestrian = pedestrian_distance <= PEDESTRIAN_REACH
    return np.stack(
        [
            route.approaching,
            route.in_junction,
            route.turning,
            red & ~standing,
            red & standing,
            sign & ~standing,
            sign & standing,
            vehicle & ~standing,
            vehicle & standing,
            pedestrian & ~standing,
            pedestrian & standing,
        ],
        axis=1,
    )


def _red_ahead(light_state: np.ndarray, light_distance: np.ndarray) -> np.ndarray:
    return np.isin(light_state, _RED) & (light_distance >= 0) & (light_distance <= RULE_AHEAD)


# The expert sometimes stops for a red light in the wrong place: it notices the light late and
# stops inside the junction, or misjudges the stop line and stops well before it. Those stops,
# and the approach that led to them, are kept out of training.
STOP_LINE_REACH = 20.0  # m along the heading: the stop line a standing vehicle is measured from
STOP_LINE_SIDE = 10.0  # m: lines of the other lanes of the same approach count
STOP_LINE_HEADING = 30.0  # deg: a line is in the vehicle's direction within this
PAST_LINE = 4.0  # m: a vehicle centre further past the line stands inside the junction; a
# correct stop has it within 3 m of the line
SHORT_OF_LINE = 6.0  # m: further before the line, with nothing in the path, is too early
MIN_STOP = 2.0  # s: a shorter stop is a pause while creeping up to the line
STOP_APPROACH = 5.0  # s before a misplaced stop are excluded with it: the expert usually
# reacts to a red light 5 s before standing


def stop_line_ahead(xy: np.ndarray, yaw: float, stop_lines: np.ndarray) -> float | None:
    """Metres along the heading from a vehicle at xy (world, metres) and CARLA yaw (degrees)
    to the nearest stop line in its direction, negative once passed; None if there is none
    within STOP_LINE_REACH. stop_lines: (M, 3) x, y and CARLA yaw of traffic-light stop lines.
    """
    local = _vehicle_frame(stop_lines[:, 0] - xy[0], stop_lines[:, 1] - xy[1], yaw)
    turn = (stop_lines[:, 2] - yaw + 180) % 360 - 180
    near = (
        (np.abs(turn) < STOP_LINE_HEADING)
        & (np.abs(local[:, 1]) < STOP_LINE_SIDE)
        & (np.abs(local[:, 0]) < STOP_LINE_REACH)
    )
    if not near.any():
        return None
    along = local[near, 0]
    return float(along[np.argmin(np.abs(along))])


def misplaced_stops(
    position: np.ndarray,
    yaw: np.ndarray,
    speed: np.ndarray,
    light_state: np.ndarray,
    light_distance: np.ndarray,
    vehicle_gap: np.ndarray,
    stop_lines: np.ndarray,
    fps: int,
) -> np.ndarray:
    """(N,) bool: rows of red-light stops made in the wrong place, and the STOP_APPROACH
    before each.

    Rows are `fps` apart; position (N, 2+) and yaw (N,) as in route(), the other signals as
    in situations(). A stop is a run of standing rows, at least MIN_STOP long, that begins at
    a red or yellow light within RULE_AHEAD with no vehicle in the path; it is misplaced when
    the vehicle stands more than PAST_LINE past its stop line or more than SHORT_OF_LINE
    before it.
    """
    excluded = np.zeros(len(speed), bool)
    edges = np.flatnonzero(np.diff(np.r_[0, (speed < STANDING_SPEED).astype(np.int8), 0]))
    red = _red_ahead(light_state, light_distance)
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start < MIN_STOP * fps or not red[start] or vehicle_gap[start] <= VEHICLE_REACH:
            continue
        line = stop_line_ahead(position[start, :2], float(yaw[start]), stop_lines)
        if line is not None and not -PAST_LINE <= line <= SHORT_OF_LINE:
            excluded[max(start - round(STOP_APPROACH * fps), 0) : end] = True
    return excluded
