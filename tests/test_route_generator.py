"""Tests for BalancedRouteGenerator.

Since the class requires a live CARLA world instance, we test:
1. RouteConfig defaults
2. get_distribution_report() — pure logic
3. _find_target_junction() — geometry with mocks
4. _extend_route() — waypoint chaining with mocks

Run with: pytest tests/test_route_generator.py -v
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data.route_generator import BalancedRouteGenerator, RouteConfig

# ── Test RouteConfig defaults ──────────────────────────────────────────


class TestRouteConfig:
    """Test suite for RouteConfig dataclass defaults."""

    def test_default_strategy(self) -> None:
        config = RouteConfig()
        assert config.strategy == "balanced"

    def test_default_targets_sum_under_one(self) -> None:
        config = RouteConfig()
        total = config.target_left + config.target_right + config.target_straight
        assert total <= 1.0, f"Target proportions sum to {total}, expected <= 1.0"

    def test_default_waypoint_range(self) -> None:
        config = RouteConfig()
        assert config.min_waypoints < config.max_waypoints
        assert config.min_waypoints > 0

    def test_custom_config(self) -> None:
        config = RouteConfig(
            strategy="random",
            target_left=0.3,
            target_right=0.3,
            target_straight=0.2,
            min_waypoints=50,
            max_waypoints=200,
        )
        assert config.strategy == "random"
        assert config.target_left == 0.3


# ── Test get_distribution_report ───────────────────────────────────────


class TestDistributionReport:
    """Test get_distribution_report — pure logic, no CARLA needed.

    We instantiate with a patched __init__ to avoid CARLA dependency.
    """

    @pytest.fixture
    def generator(self) -> BalancedRouteGenerator:
        """Create a BalancedRouteGenerator with mocked __init__."""
        from src.data.route_generator import BalancedRouteGenerator

        with patch.object(BalancedRouteGenerator, "__init__", lambda self, *a, **kw: None):
            gen = BalancedRouteGenerator.__new__(BalancedRouteGenerator)
            return gen

    def test_empty_route(self, generator: BalancedRouteGenerator) -> None:
        report = generator.get_distribution_report([])
        assert report == {}

    def test_all_lane_follow(self, generator: BalancedRouteGenerator) -> None:
        route = [(MagicMock(), "LaneFollow")] * 10
        report = generator.get_distribution_report(route)
        assert report["LaneFollow"] == 1.0
        assert len(report) == 1

    def test_mixed_route(self, generator: BalancedRouteGenerator) -> None:
        route = (
            [(MagicMock(), "LaneFollow")] * 6
            + [(MagicMock(), "Left")] * 2
            + [(MagicMock(), "Right")] * 1
            + [(MagicMock(), "Straight")] * 1
        )
        report = generator.get_distribution_report(route)
        assert report["LaneFollow"] == pytest.approx(0.6)
        assert report["Left"] == pytest.approx(0.2)
        assert report["Right"] == pytest.approx(0.1)
        assert report["Straight"] == pytest.approx(0.1)


# ── Test _find_target_junction ─────────────────────────────────────────


@dataclass
class MockLocation:
    x: float
    y: float
    z: float = 0.0


class TestFindTargetJunction:
    """Test _find_target_junction — geometry + distance filtering."""

    @pytest.fixture
    def generator(self) -> BalancedRouteGenerator:
        """Create a BalancedRouteGenerator with mocked __init__ and junctions."""
        from src.data.route_generator import BalancedRouteGenerator

        with patch.object(BalancedRouteGenerator, "__init__", lambda self, *a, **kw: None):
            gen = BalancedRouteGenerator.__new__(BalancedRouteGenerator)
            gen.config = RouteConfig()
            gen._junctions = [
                {
                    "junction_id": 1,
                    "location": MockLocation(100.0, 0.0),
                    "entry_waypoints": [],
                    "exit_waypoints": [],
                },
                {
                    "junction_id": 2,
                    "location": MockLocation(200.0, 0.0),
                    "entry_waypoints": [],
                    "exit_waypoints": [],
                },
                {
                    "junction_id": 3,
                    "location": MockLocation(10.0, 0.0),  # too close (< 20m)
                    "entry_waypoints": [],
                    "exit_waypoints": [],
                },
            ]
            return gen

    def test_returns_valid_junction(self, generator: BalancedRouteGenerator) -> None:
        current = MockLocation(0.0, 0.0)
        rng = np.random.default_rng(42)
        result = generator._find_target_junction(current, {}, rng)
        assert result is not None
        assert result["junction_id"] in (1, 2)  # Junction 3 too close

    def test_filters_too_close(self, generator: BalancedRouteGenerator) -> None:
        """Junction at 10m should be filtered out (threshold is 20m)."""
        current = MockLocation(0.0, 0.0)
        rng = np.random.default_rng(42)
        # Run multiple times to verify junction 3 is never selected
        for _ in range(20):
            result = generator._find_target_junction(current, {}, rng)
            if result is not None:
                assert result["junction_id"] != 3

    def test_no_valid_junctions(self) -> None:
        """All junctions too far → returns None."""
        from src.data.route_generator import BalancedRouteGenerator

        with patch.object(BalancedRouteGenerator, "__init__", lambda self, *a, **kw: None):
            gen = BalancedRouteGenerator.__new__(BalancedRouteGenerator)
            gen.config = RouteConfig()
            gen._junctions = [
                {
                    "junction_id": 99,
                    "location": MockLocation(10000.0, 10000.0),  # too far
                    "entry_waypoints": [],
                    "exit_waypoints": [],
                },
            ]
            result = gen._find_target_junction(MockLocation(0, 0), {}, np.random.default_rng())
            assert result is None

    def test_empty_junctions(self) -> None:
        """No junctions at all → returns None."""
        from src.data.route_generator import BalancedRouteGenerator

        with patch.object(BalancedRouteGenerator, "__init__", lambda self, *a, **kw: None):
            gen = BalancedRouteGenerator.__new__(BalancedRouteGenerator)
            gen.config = RouteConfig()
            gen._junctions = []
            result = gen._find_target_junction(MockLocation(0, 0), {}, np.random.default_rng())
            assert result is None


# ── Test _extend_route ─────────────────────────────────────────────────


class TestExtendRoute:
    """Test _extend_route — waypoint chaining logic."""

    @pytest.fixture
    def generator(self) -> BalancedRouteGenerator:
        from src.data.route_generator import BalancedRouteGenerator

        with patch.object(BalancedRouteGenerator, "__init__", lambda self, *a, **kw: None):
            gen = BalancedRouteGenerator.__new__(BalancedRouteGenerator)
            gen.config = RouteConfig()
            return gen

    def test_extends_correct_length(self, generator: BalancedRouteGenerator) -> None:
        """Extends route by the requested number of waypoints."""
        # Build a chain of mock waypoints
        waypoints = []
        for i in range(10):
            wp = MagicMock()
            wp.transform = MagicMock()
            waypoints.append(wp)

        # Each waypoint.next() returns the next one
        for i in range(len(waypoints) - 1):
            waypoints[i].next = MagicMock(return_value=[waypoints[i + 1]])
        waypoints[-1].next = MagicMock(return_value=[])

        result = generator._extend_route(waypoints[0], 5)
        assert len(result) == 5
        # All should be LaneFollow
        for _, option in result:
            assert option == "LaneFollow"

    def test_stops_at_road_end(self, generator: BalancedRouteGenerator) -> None:
        """Stops early if no next waypoints."""
        wp = MagicMock()
        wp.next = MagicMock(return_value=[])
        result = generator._extend_route(wp, 100)
        assert len(result) == 0
