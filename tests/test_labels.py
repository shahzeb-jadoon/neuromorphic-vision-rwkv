"""Training targets computed from raw signals."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.labels import (
    LANEFOLLOW,
    LEFT,
    RIGHT,
    TURN_APPROACH,
    ego_waypoints,
    path_commands,
    usable_end,
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


@pytest.mark.parametrize("sign, command", [(1, RIGHT), (-1, LEFT)])
def test_a_junction_turn_is_announced_until_it_ends(sign: int, command: int) -> None:
    arc = np.pi / 2 * 10  # 90 degrees at a 10 m radius
    position, yaw = drive([(60, 0), (arc, sign * 90), (60, 0)])
    s = np.arange(len(yaw)) * 0.25
    got = path_commands(position, yaw)
    turn_end = 60 + arc
    assert set(got[(s > 60 - TURN_APPROACH + 2) & (s < turn_end - 1)]) == {command}
    assert set(got[(s < 60 - TURN_APPROACH - 3) | (s > turn_end + 3)]) == {LANEFOLLOW}


def test_a_road_curve_is_lane_following() -> None:
    position, yaw = drive([(40, 0), (np.pi / 3 * 60, 60), (40, 0)])  # 60 degrees, 60 m radius
    assert set(path_commands(position, yaw)) == {LANEFOLLOW}


def test_waiting_before_a_turn_keeps_its_command() -> None:
    position, yaw = drive([(60, 0), (np.pi / 2 * 10, -90), (40, 0)])
    wait = int(
        (60 - TURN_APPROACH / 2) / 0.25
    )  # stand still for 400 rows halfway along the approach
    position = np.insert(position, wait, np.repeat(position[wait : wait + 1], 400, axis=0), axis=0)
    yaw = np.insert(yaw, wait, np.full(400, yaw[wait]))
    assert set(path_commands(position, yaw)[wait : wait + 400]) == {LEFT}


def test_close_turns_the_same_way_share_one_command() -> None:
    arc = np.pi / 2 * 10
    position, yaw = drive([(60, 0), (arc, -90), (35, 0), (arc, -90), (60, 0)])
    s = np.arange(len(yaw)) * 0.25
    got = path_commands(position, yaw)
    assert set(got[(s > 60 - TURN_APPROACH + 2) & (s < 60 + 2 * arc + 35 - 1)]) == {LEFT}
