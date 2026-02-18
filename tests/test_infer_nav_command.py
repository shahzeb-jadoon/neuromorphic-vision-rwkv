"""Tests for ExpertDataCollector._infer_nav_command().

The method is a @staticmethod that computes a navigation command (0–3)
from the signed angle between the vehicle's forward vector and the
direction to a lookahead waypoint. We test it with lightweight mock
objects — no CARLA installation required.

Run with: pytest tests/test_infer_nav_command.py -v
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from unittest.mock import MagicMock


from src.data.expert_collector import ExpertDataCollector

# ── Mock CARLA geometry objects ────────────────────────────────────────


@dataclass
class MockVector3D:
    x: float
    y: float
    z: float = 0.0


@dataclass
class MockLocation:
    x: float
    y: float
    z: float = 0.0


class MockTransform:
    """Minimal mock for carla.Transform."""

    def __init__(self, location: MockLocation, forward: MockVector3D) -> None:
        self.location = location
        self._forward = forward

    def get_forward_vector(self) -> MockVector3D:
        return self._forward


class MockWaypoint:
    """Minimal mock for carla.Waypoint."""

    def __init__(
        self,
        location: MockLocation,
        is_junction: bool = False,
        next_wps: list | None = None,
    ) -> None:
        self.transform = MockTransform(location, MockVector3D(1.0, 0.0))
        self.is_junction = is_junction
        self._next_wps = next_wps if next_wps is not None else []

    def next(self, _distance: float) -> list:
        return self._next_wps


def _make_scenario(
    veh_x: float,
    veh_y: float,
    fwd_x: float,
    fwd_y: float,
    target_x: float,
    target_y: float,
    is_junction: bool = False,
) -> tuple[MockTransform, MockWaypoint, MagicMock]:
    """Build mock vehicle transform + waypoint for a given scenario."""
    vehicle_transform = MockTransform(
        location=MockLocation(veh_x, veh_y),
        forward=MockVector3D(fwd_x, fwd_y),
    )
    target_wp = MockWaypoint(MockLocation(target_x, target_y))
    current_wp = MockWaypoint(
        MockLocation(veh_x, veh_y),
        is_junction=is_junction,
        next_wps=[target_wp],
    )
    carla_map = MagicMock()
    return vehicle_transform, current_wp, carla_map


# ── Test Cases ─────────────────────────────────────────────────────────


class TestInferNavCommand:
    """Test suite for _infer_nav_command static method."""

    def test_straight_ahead_returns_lane_follow(self) -> None:
        """Vehicle going straight, target straight ahead → LaneFollow (0)."""
        veh_tf, wp, cmap = _make_scenario(
            veh_x=0, veh_y=0, fwd_x=1, fwd_y=0, target_x=10, target_y=0
        )
        result = ExpertDataCollector._infer_nav_command(veh_tf, wp, cmap)
        assert result == 0, f"Expected LaneFollow (0), got {result}"

    def test_hard_left_turn(self) -> None:
        """Target is 90° to the left → Left (1).

        Vehicle faces +X, target is at -Y direction (CARLA left-handed).
        In CARLA: negative cross product → Left.
        """
        veh_tf, wp, cmap = _make_scenario(
            veh_x=0, veh_y=0, fwd_x=1, fwd_y=0, target_x=0, target_y=-10
        )
        result = ExpertDataCollector._infer_nav_command(veh_tf, wp, cmap)
        assert result == 1, f"Expected Left (1), got {result}"

    def test_hard_right_turn(self) -> None:
        """Target is 90° to the right → Right (2).

        Vehicle faces +X, target is at +Y direction.
        """
        veh_tf, wp, cmap = _make_scenario(
            veh_x=0, veh_y=0, fwd_x=1, fwd_y=0, target_x=0, target_y=10
        )
        result = ExpertDataCollector._infer_nav_command(veh_tf, wp, cmap)
        assert result == 2, f"Expected Right (2), got {result}"

    def test_junction_straight_through(self) -> None:
        """At a junction with small angle → Straight (3)."""
        veh_tf, wp, cmap = _make_scenario(
            veh_x=0,
            veh_y=0,
            fwd_x=1,
            fwd_y=0,
            target_x=10,
            target_y=0.5,
            is_junction=True,
        )
        result = ExpertDataCollector._infer_nav_command(veh_tf, wp, cmap)
        assert result == 3, f"Expected Straight (3), got {result}"

    def test_no_next_waypoints_returns_lane_follow(self) -> None:
        """When there are no next waypoints → default LaneFollow (0)."""
        vehicle_transform = MockTransform(
            location=MockLocation(0, 0),
            forward=MockVector3D(1, 0),
        )
        empty_wp = MockWaypoint(MockLocation(0, 0), next_wps=[])
        carla_map = MagicMock()
        result = ExpertDataCollector._infer_nav_command(vehicle_transform, empty_wp, carla_map)
        assert result == 0

    def test_zero_forward_vector_returns_lane_follow(self) -> None:
        """When forward vector is zero → default LaneFollow (0)."""
        vehicle_transform = MockTransform(
            location=MockLocation(0, 0),
            forward=MockVector3D(0, 0),  # Zero vector
        )
        target_wp = MockWaypoint(MockLocation(10, 0))
        current_wp = MockWaypoint(MockLocation(0, 0), next_wps=[target_wp])
        carla_map = MagicMock()
        result = ExpertDataCollector._infer_nav_command(vehicle_transform, current_wp, carla_map)
        assert result == 0

    def test_boundary_angle_just_below_threshold(self) -> None:
        """Angle just below 15° threshold → LaneFollow (0)."""
        # target at 14° (below 15° threshold)
        angle = 14.0
        rad = math.radians(angle)
        veh_tf, wp, cmap = _make_scenario(
            veh_x=0,
            veh_y=0,
            fwd_x=1,
            fwd_y=0,
            target_x=10 * math.cos(rad),
            target_y=10 * math.sin(rad),
        )
        result = ExpertDataCollector._infer_nav_command(veh_tf, wp, cmap)
        assert result == 0, f"14° should be LaneFollow (0), got {result}"

    def test_boundary_angle_just_above_threshold(self) -> None:
        """Angle just above 15° threshold → Right (2)."""
        angle = 16.0
        rad = math.radians(angle)
        veh_tf, wp, cmap = _make_scenario(
            veh_x=0,
            veh_y=0,
            fwd_x=1,
            fwd_y=0,
            target_x=10 * math.cos(rad),
            target_y=10 * math.sin(rad),
        )
        result = ExpertDataCollector._infer_nav_command(veh_tf, wp, cmap)
        assert result == 2, f"16° should be Right (2), got {result}"

    def test_moderate_left_angle(self) -> None:
        """Angle of -45° (moderate left) → Left (1)."""
        angle = -45.0
        rad = math.radians(angle)
        veh_tf, wp, cmap = _make_scenario(
            veh_x=0,
            veh_y=0,
            fwd_x=1,
            fwd_y=0,
            target_x=10 * math.cos(rad),
            target_y=10 * math.sin(rad),
        )
        result = ExpertDataCollector._infer_nav_command(veh_tf, wp, cmap)
        assert result == 1, f"-45° should be Left (1), got {result}"
