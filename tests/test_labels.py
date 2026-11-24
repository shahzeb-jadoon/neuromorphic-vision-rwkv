"""Training targets computed from raw signals."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.data.labels import (
    APPROACH,
    LANEFOLLOW,
    LEFT,
    MIN_CROSSING,
    RIGHT,
    SITUATIONS,
    STRAIGHT,
    TARGET_DISTANCE,
    Route,
    ego_waypoints,
    junction_ids,
    misplaced_stops,
    pedestrian_distances,
    route,
    route_progress,
    situations,
    target_on_route,
    usable_end,
    vehicle_gaps,
)

OFFSETS = (5, 10, 15, 20)


@pytest.mark.parametrize("yaw", [0.0, 37.0, 90.0, -135.0, 180.0])
def test_straight_drive_is_straight_ahead(yaw: float) -> None:
    heading = np.radians(yaw)
    steps = np.arange(60)[:, None] * 0.5 * np.array([np.cos(heading), np.sin(heading), 0.0])
    position = (np.array([120.0, -40.0, 0.3]) + steps).astype(np.float32)
    wp = ego_waypoints(position, np.full(40, yaw, np.float32), OFFSETS)
    assert wp.shape == (40, 4, 2)
    assert np.allclose(wp[..., 0], 0.5 * np.array(OFFSETS), atol=1e-4)
    assert np.allclose(wp[..., 1], 0.0, atol=1e-4)


def test_invariant_to_rotating_the_world() -> None:
    rng = np.random.default_rng(0)
    position = np.cumsum(rng.normal(size=(50, 3)), axis=0).astype(np.float32)
    yaw = rng.uniform(-180, 180, size=30).astype(np.float32)
    theta = np.radians(73.0)
    turn = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    turned = (position @ turn.T + [30.0, 5.0, 0.0]).astype(np.float32)
    assert np.allclose(
        ego_waypoints(position, yaw, OFFSETS),
        ego_waypoints(turned, yaw + np.float32(73.0), OFFSETS),
        atol=1e-3,
    )


def test_right_is_positive_and_left_negative() -> None:
    # Facing +y (yaw 90): +x is on the vehicle's left in CARLA's left-handed frame.
    position = np.array([[0, 0, 0], [-1, 5, 0], [1, 5, 0]], dtype=np.float32)
    wp = ego_waypoints(position, np.array([90.0], np.float32), (1, 2))
    assert np.allclose(wp[0], [[5, 1], [5, -1]], atol=1e-6)


def test_offsets_past_the_end_use_the_last_position() -> None:
    position = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
    wp = ego_waypoints(position, np.zeros(2, np.float32), (1, 5))
    assert np.allclose(wp[..., 0], [[1, 3], [1, 2]])


def test_clean_episodes_are_used_whole() -> None:
    speed = np.ones(100)
    assert usable_end(speed, 20, "completed", 0) == 100
    speed[60:] = 0.0  # stopped at a light when the frame budget ran out
    assert usable_end(speed, 20, "completed", 0) == 100


def test_three_seconds_before_a_collision_are_dropped() -> None:
    assert usable_end(np.ones(200), 20, "collision", 150) == 90
    assert usable_end(np.ones(200), 10, "collision", 150) == 120
    assert usable_end(np.ones(200), 20, "collision", 40) == 0


def test_a_timeout_loses_its_final_stop() -> None:
    speed = np.r_[np.ones(50), np.zeros(10), np.ones(30), np.full(70, 0.05)]
    assert usable_end(speed, 20, "timeout", 0) == 90
    assert usable_end(np.zeros(40), 20, "timeout", 0) == 0


def drive(turns: list[tuple[float, float]], step: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Rows every `step` metres along straights and arcs: (length m, heading change deg)."""
    heading = [0.0]
    for length, change in turns:
        n = int(round(length / step))
        heading += list(heading[-1] + change * np.arange(1, n + 1) / n)
    psi = np.radians(heading)
    xy = np.cumsum(np.c_[np.cos(psi), np.sin(psi)] * step, axis=0)
    return np.c_[xy, np.zeros(len(xy))], np.degrees(psi)


def square(x0: float, x1: float, y0: float, y1: float) -> list[tuple[int, np.ndarray]]:
    return [(7, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float))]


def route_of(position: np.ndarray, yaw: np.ndarray, outlines: list) -> Route:
    return route(position, yaw, junction_ids(position[:, :2], outlines))


@pytest.mark.parametrize("sign, command", [(1, RIGHT), (-1, LEFT)])
def test_a_junction_turn_is_commanded_from_its_approach_until_left(sign: int, command: int) -> None:
    arc = np.pi / 2 * 10  # 90 degrees at a 10 m radius, from (60, 0) to (70, 10 * sign)
    position, yaw = drive([(60, 0), (arc, sign * 90), (60, 0)])
    r = route_of(position, yaw, square(57, 73, -13, 13))
    s = np.arange(len(yaw)) * 0.25
    leave = 60 + arc + 3  # the path leaves the square at |y| = 13
    assert set(r.command[(s > 57 - APPROACH + 1) & (s < leave - 1)]) == {command}
    assert set(r.command[(s < 57 - APPROACH - 1) | (s > leave + 1)]) == {LANEFOLLOW}
    assert r.approaching[(s > 28) & (s < 56)].all() and not r.approaching[s > 58].any()
    assert r.in_junction[(s > 58) & (s < leave - 1)].all()


def test_going_straight_through_a_junction_is_straight() -> None:
    position, yaw = drive([(150, 0)])
    r = route_of(position, yaw, square(60, 75, -8, 8))
    s = np.arange(len(yaw)) * 0.25
    assert set(r.command[(s > 31) & (s < 74)]) == {STRAIGHT}
    assert set(r.command[(s < 29) | (s > 76)]) == {LANEFOLLOW}


def test_junctions_close_together_leave_no_lane_following_between() -> None:
    position, yaw = drive([(200, 0)])
    near = square(60, 75, -8, 8) + [(8, square(110, 125, -8, 8)[0][1])]  # approach at 80 m
    r = route_of(position, yaw, near)
    s = np.arange(len(yaw)) * 0.25
    assert set(r.command[(s > 31) & (s < 124)]) == {STRAIGHT}


def test_clipping_an_outline_is_not_a_junction() -> None:
    position, yaw = drive([(150, 0)])
    r = route_of(position, yaw, square(60, 60 + MIN_CROSSING - 1, -8, 8))
    assert set(r.command) == {LANEFOLLOW} and not r.in_junction.any()


def test_waiting_before_a_junction_keeps_its_command() -> None:
    position, yaw = drive([(150, 0)])
    wait = int(45 / 0.25)  # stand still for 400 rows 15 m before the junction
    position = np.insert(position, wait, np.repeat(position[wait : wait + 1], 400, axis=0), axis=0)
    yaw = np.insert(yaw, wait, np.full(400, yaw[wait]))
    r = route_of(position, yaw, square(60, 75, -8, 8))
    assert (
        set(r.command[wait : wait + 400]) == {STRAIGHT} and r.approaching[wait : wait + 400].all()
    )


def test_a_bend_between_junctions_is_lane_following_but_turning() -> None:
    position, yaw = drive([(40, 0), (np.pi / 2 * 10, 90), (40, 0)])
    r = route_of(position, yaw, [])
    s = np.arange(len(yaw)) * 0.25
    assert set(r.command) == {LANEFOLLOW}
    assert r.turning[(s > 43) & (s < 53)].all() and not r.turning[(s < 37) | (s > 59)].any()


def test_target_point_lies_ahead_along_the_path() -> None:
    straight, yaw = drive([(100, 0)])
    assert np.allclose(route_of(straight, yaw, []).target, [TARGET_DISTANCE, 0.0], atol=1e-3)
    position, yaw = drive([(60, 0), (np.pi / 2 * 10, 90), (60, 0)])
    target = route_of(position, yaw, []).target[int(40 / 0.25)]  # 30 m on: 10 m into the arc
    assert np.allclose(target, [20 + 10 * np.sin(1.0), 10 - 10 * np.cos(1.0)], atol=0.1)


def test_a_planned_route_gives_the_target_point_of_the_same_path() -> None:
    position, yaw = drive([(60, 0), (np.pi / 2 * 10, 90), (60, 0)])
    expected = route_of(position, yaw, []).target
    for t in range(0, len(yaw), 37):
        got = target_on_route(position[t], yaw[t], position[t + 1 :])
        assert np.allclose(got, expected[t], atol=0.05)


def test_progress_moves_on_past_a_point_the_vehicle_never_came_near() -> None:
    planned = np.c_[np.arange(200.0), np.zeros(200)]
    planned[50, 1] = 8.0  # the vehicle passes this point 8 m away
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(planned, axis=0), axis=1))]
    index = 0
    for x in np.arange(0.0, 120.0, 1.4):
        index = route_progress(planned, distance, index, np.array([x, 0.0]))
        assert abs(planned[index, 0] - x) <= 1.0 and planned[index, 1] == 0.0
    assert route_progress(planned, distance, index, np.zeros(2)) == index  # never moves back
    assert route_progress(planned, distance, 0, np.array([150.0, 0.0])) < 50  # searched 50 m on


def test_without_a_route_the_target_point_lies_straight_ahead() -> None:
    got = target_on_route(np.array([5.0, -3.0]), 90.0, np.empty((0, 2)))
    assert np.allclose(got, [TARGET_DISTANCE, 0.0], atol=1e-3)


def test_situations_pair_each_cause_moving_and_standing() -> None:
    names = {name: i for i, name in enumerate(SITUATIONS)}
    position, yaw = drive([(10, 0)])
    r = route_of(position[:10], yaw[:10], [])
    red, green, inf = 3, 1, np.inf  # TRAFFIC_LIGHT_STATE_NAMES codes
    got = situations(
        r,
        speed=np.array([5, 0, 0, 5, 5, 0, 5, 0, 5, 0]),
        light_state=np.array([red, red, green, red, 0, 0, 0, 0, 0, 0]),
        light_distance=np.array([25, 28, 10, 40, -1, -1, -1, -1, -1, -1]),
        stop_distance=np.array([-1, -1, -1, -1, 20, 5, -1, -1, -1, -1]),
        vehicle_gap=np.array([inf, inf, inf, inf, inf, inf, 20, 4, 30, inf]),
        pedestrian_distance=np.array([inf] * 8 + [10, 12]),
    )
    expected = [
        {"approaching red light"},
        {"stopped at red light"},
        set(),
        set(),
        {"approaching stop sign"},
        {"stopped at stop sign"},
        {"following vehicle"},
        {"stopped for vehicle"},
        {"yielding to pedestrian"},
        {"stopped for pedestrian"},
    ]
    for row, want in zip(got, expected):
        assert {name for name, i in names.items() if row[i]} == want


def test_road_users_count_only_in_the_path_ahead() -> None:
    def vehicle(lon: float, lat: float, gap: float, corridor: bool = False) -> dict:
        return {"distance": gap, "longitudinal": lon, "lateral": lat, "obb_in_lane": corridor}

    def walker(lat: float, later: float, ahead: bool = True) -> dict:
        return {
            "distance": 10.0,
            "lateral": lat,
            "is_ahead": ahead,
            "predicted_distance": 9.5,
            "predicted_lateral": later,
            "predicted_ahead": ahead,
        }

    vehicles = [
        [],
        [vehicle(12, 0.5, 7.5), vehicle(8, 3.5, 3.5)],  # ahead, and in the next lane
        [vehicle(9, -6, 4.0, corridor=True)],  # across the corridor of a turn
        [vehicle(-6, 0, 1.5)],  # behind
    ]
    assert vehicle_gaps([json.dumps(v) for v in vehicles]).tolist() == [np.inf, 7.5, 4.0, np.inf]
    walkers = [[walker(1.0, 1.0)], [walker(4.0, 2.0)], [walker(4.0, 4.5)], [walker(0, 0, False)]]
    got = pedestrian_distances([json.dumps(w) for w in walkers])
    assert got.tolist() == [9.5, 9.5, np.inf, np.inf]  # in the lane, stepping in, pavement, behind


@pytest.mark.parametrize(
    "stand, seconds, light, gap, misplaced",
    [
        (99.5, 3, 3, np.inf, False),  # at the line
        (106.0, 3, 3, np.inf, True),  # 6 m past it, inside the junction
        (92.0, 3, 3, np.inf, True),  # 8 m short with nothing in the path
        (92.0, 1, 3, np.inf, False),  # a pause while creeping up to the line
        (92.0, 3, 3, 4.0, False),  # 8 m short, queueing behind a vehicle
        (106.0, 3, 1, np.inf, False),  # not a red-light stop
    ],
)
def test_a_red_light_stop_in_the_wrong_place_is_excluded_with_its_approach(
    stand: float, seconds: int, light: int, gap: float, misplaced: bool
) -> None:
    fps = 10  # 6 s at 10 m/s, standing at x = stand, then 3 s on
    still = seconds * fps
    x = np.r_[stand - 60 + np.arange(60.0), np.full(still, stand), stand + np.arange(1, 31.0)]
    speed = np.r_[np.full(60, 10.0), np.zeros(still), np.full(30, 10.0)]
    lines = np.array([[100.0, 1.75, 0.0], [104.0, 6.0, 90.0]])  # this lane's, the cross street's
    got = misplaced_stops(
        np.c_[x, np.zeros_like(x)],
        np.zeros_like(x),
        speed,
        np.full(len(x), light),
        np.full(len(x), 8.0),
        np.full(len(x), gap),
        lines,
        fps,
    )
    rows = np.arange(len(x))
    assert np.array_equal(got, misplaced & (rows >= 10) & (rows < 60 + still))
