"""Training targets computed from raw signals."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.labels import ego_waypoints, usable_end

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
