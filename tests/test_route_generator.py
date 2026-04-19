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

from src.data.route_generator import BalancedRouteGenerator, RouteConfig, carla

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

    def distance(self, other: "MockLocation") -> float:
        dx = float(self.x - other.x)
        dy = float(self.y - other.y)
        dz = float(self.z - other.z)
        return float((dx * dx + dy * dy + dz * dz) ** 0.5)


@dataclass
class MockTransform:
    location: MockLocation


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


class TestBalancedRouteStitching:
    @staticmethod
    def _make_waypoint(x: float, y: float) -> MagicMock:
        wp = MagicMock()
        wp.transform = MockTransform(MockLocation(x, y))
        return wp

    def test_traces_through_selected_junction_exit(self) -> None:
        with patch.object(BalancedRouteGenerator, "__init__", lambda self, *a, **kw: None):
            gen = BalancedRouteGenerator.__new__(BalancedRouteGenerator)

        gen.config = RouteConfig(min_waypoints=0, max_waypoints=500, sampling_resolution=2.0)

        spawn_loc = MockLocation(0.0, 0.0)
        spawn = MagicMock()
        spawn.location = spawn_loc
        gen.spawn_points = [spawn]

        start_wp = self._make_waypoint(0.0, 0.0)
        entry_wp = self._make_waypoint(10.0, 0.0)
        exit_wp = self._make_waypoint(12.0, 0.0)

        gen.map = MagicMock()

        def _map_get_waypoint(location: MockLocation, *args: object, **kwargs: object) -> MagicMock:
            if location is spawn_loc:
                return start_wp
            if abs(float(location.x) - 10.0) < 1e-6:
                return entry_wp
            if abs(float(location.x) - 12.0) < 1e-6:
                return exit_wp
            return start_wp

        target_junction = {
            "junction_id": 1,
            "location": MockLocation(10.0, 0.0),
            "entry_waypoints": [entry_wp],
            "exit_waypoints": [exit_wp],
        }
        gen._junctions = [target_junction]

        seg_to_entry = [
            (MockTransform(MockLocation(1.0, 0.0)), "LaneFollow"),
            (MockTransform(MockLocation(10.0, 0.0)), "LaneFollow"),
        ]
        seg_through_junction = [
            (MockTransform(MockLocation(10.0, 0.0)), "LaneFollow"),
            (MockTransform(MockLocation(12.0, 0.0)), "Left"),
        ]
        with patch.object(gen.map, "get_waypoint", side_effect=_map_get_waypoint):
            with patch.object(
                gen,
                "_find_target_junction",
                side_effect=[target_junction, None],
            ):
                with patch.object(
                    gen,
                    "_trace_route",
                    side_effect=[seg_to_entry, seg_through_junction],
                ) as trace_mock:
                    route = gen._generate_balanced_route(np.random.default_rng(0))

                    coords = [(float(tf.location.x), float(tf.location.y)) for tf, _ in route]
                    assert coords == [(1.0, 0.0), (10.0, 0.0), (12.0, 0.0)]
                    assert trace_mock.call_count == 2
                    assert trace_mock.call_args_list[1].args[0] is entry_wp
                    assert trace_mock.call_args_list[1].args[1] is exit_wp

    def test_failed_junction_trace_keeps_continuous_tail(self) -> None:
        with patch.object(BalancedRouteGenerator, "__init__", lambda self, *a, **kw: None):
            gen = BalancedRouteGenerator.__new__(BalancedRouteGenerator)

        gen.config = RouteConfig(min_waypoints=6, max_waypoints=500, sampling_resolution=2.0)

        spawn_loc = MockLocation(0.0, 0.0)
        spawn = MagicMock()
        spawn.location = spawn_loc
        gen.spawn_points = [spawn]

        start_wp = self._make_waypoint(0.0, 0.0)
        entry_wp = self._make_waypoint(10.0, 0.0)
        exit_wp = self._make_waypoint(12.0, 0.0)

        gen.map = MagicMock()

        def _map_get_waypoint(location: MockLocation, *args: object, **kwargs: object) -> MagicMock:
            if location is spawn_loc:
                return start_wp
            if abs(float(location.x) - 10.0) < 1e-6:
                return entry_wp
            if abs(float(location.x) - 12.0) < 1e-6:
                return exit_wp
            return start_wp

        target_junction = {
            "junction_id": 1,
            "location": MockLocation(10.0, 0.0),
            "entry_waypoints": [entry_wp],
            "exit_waypoints": [exit_wp],
        }
        gen._junctions = [target_junction]

        seg_to_entry = [
            (MockTransform(MockLocation(1.0, 0.0)), "LaneFollow"),
            (MockTransform(MockLocation(10.0, 0.0)), "LaneFollow"),
        ]
        with patch.object(gen.map, "get_waypoint", side_effect=_map_get_waypoint):
            with patch.object(
                gen,
                "_find_target_junction",
                side_effect=[target_junction, None],
            ):
                with patch.object(
                    gen, "_trace_route", side_effect=[seg_to_entry, []]
                ) as trace_mock:
                    with patch.object(gen, "_extend_route", return_value=[]) as extend_mock:
                        gen._generate_balanced_route(np.random.default_rng(0))

                        assert trace_mock.call_count == 2
                        assert extend_mock.call_count == 1
                        assert extend_mock.call_args.args[0] is entry_wp


class TestBalancedRouteBudgetAndTargets:
    @staticmethod
    def _make_transform(x: float, y: float) -> MockTransform:
        return MockTransform(MockLocation(x, y))

    @staticmethod
    def _make_waypoint(x: float, y: float) -> MagicMock:
        wp = MagicMock()
        wp.transform = MockTransform(MockLocation(x, y))
        return wp

    def test_balanced_route_clamps_to_max_waypoints(self) -> None:
        with patch.object(BalancedRouteGenerator, "__init__", lambda self, *a, **kw: None):
            gen = BalancedRouteGenerator.__new__(BalancedRouteGenerator)

        gen.config = RouteConfig(min_waypoints=0, max_waypoints=5, sampling_resolution=2.0)

        spawn_loc = MockLocation(0.0, 0.0)
        spawn = MagicMock()
        spawn.location = spawn_loc
        gen.spawn_points = [spawn]

        start_wp = self._make_waypoint(0.0, 0.0)
        entry_wp = self._make_waypoint(10.0, 0.0)
        exit_wp = self._make_waypoint(12.0, 0.0)

        gen.map = MagicMock()

        def _map_get_waypoint(location: MockLocation, *args: object, **kwargs: object) -> MagicMock:
            if location is spawn_loc:
                return start_wp
            if abs(float(location.x) - 10.0) < 1e-6:
                return entry_wp
            if abs(float(location.x) - 12.0) < 1e-6:
                return exit_wp
            return start_wp

        target_junction = {
            "junction_id": 1,
            "location": MockLocation(10.0, 0.0),
            "entry_waypoints": [entry_wp],
            "exit_waypoints": [exit_wp],
        }
        gen._junctions = [target_junction]

        long_segment = [(self._make_transform(float(i), 0.0), "LaneFollow") for i in range(12)]

        with patch.object(gen.map, "get_waypoint", side_effect=_map_get_waypoint):
            with patch.object(gen, "_find_target_junction", side_effect=[target_junction, None]):
                with patch.object(gen, "_trace_route", return_value=long_segment):
                    with patch.object(gen, "_select_junction_exit_segment") as select_exit_mock:
                        route = gen._generate_balanced_route(np.random.default_rng(0))

        assert len(route) == 5
        assert [float(tf.location.x) for tf, _ in route] == [0.0, 1.0, 2.0, 3.0, 4.0]
        # Once the segment budget is exhausted, no junction exit segment is traced.
        assert select_exit_mock.call_count == 0

    def test_choose_desired_turn_honors_extreme_target_distribution(self) -> None:
        with patch.object(BalancedRouteGenerator, "__init__", lambda self, *a, **kw: None):
            gen = BalancedRouteGenerator.__new__(BalancedRouteGenerator)

        gen.config = RouteConfig(target_left=0.0, target_right=1.0, target_straight=0.0)
        turn_counts = {"LEFT": 50, "RIGHT": 0, "STRAIGHT": 30}

        for _ in range(20):
            desired = gen._choose_desired_turn(turn_counts, np.random.default_rng(_))
            assert desired == "RIGHT"

    def test_select_junction_exit_segment_prefers_desired_turn(self) -> None:
        with patch.object(BalancedRouteGenerator, "__init__", lambda self, *a, **kw: None):
            gen = BalancedRouteGenerator.__new__(BalancedRouteGenerator)

        gen.config = RouteConfig(target_left=0.1, target_right=0.8, target_straight=0.1)
        current_wp = self._make_waypoint(0.0, 0.0)
        exit_left = self._make_waypoint(10.0, 0.0)
        exit_right = self._make_waypoint(12.0, 0.0)
        turn_counts = {"LEFT": 40, "RIGHT": 0, "STRAIGHT": 10}

        left_segment = [(self._make_transform(1.0, 0.0), "Left")]
        right_segment = [(self._make_transform(1.0, 1.0), "Right")]

        def _trace_route(_start: object, end_wp: object) -> list[tuple[MockTransform, str]]:
            if end_wp is exit_left:
                return left_segment
            return right_segment

        with patch.object(gen, "_trace_route", side_effect=_trace_route):
            with patch.object(gen, "_choose_desired_turn", return_value="RIGHT"):
                selected = gen._select_junction_exit_segment(
                    current_wp,
                    [exit_left, exit_right],
                    turn_counts,
                    np.random.default_rng(0),
                )

        assert selected == right_segment


class _TopoWaypoint:
    def __init__(self, road_id: int, lane_id: int, is_junction: bool, x: float) -> None:
        self.road_id = road_id
        self.lane_id = lane_id
        self.is_junction = is_junction
        self.lane_type = carla.LaneType.Driving
        self.transform = MockTransform(MockLocation(x, 0.0))


class TestApproachConditionedValidation:
    def test_builds_approach_exit_links_from_topology(self) -> None:
        with patch.object(BalancedRouteGenerator, "__init__", lambda self, *a, **kw: None):
            gen = BalancedRouteGenerator.__new__(BalancedRouteGenerator)

        # 48/1 can legally exit via connector 791->30/1 and 727->32/-2.
        wp_48 = _TopoWaypoint(48, 1, False, 0.0)
        wp_791 = _TopoWaypoint(791, 1, True, 1.0)
        wp_727 = _TopoWaypoint(727, 1, True, 1.0)
        wp_30 = _TopoWaypoint(30, 1, False, 2.0)
        wp_32 = _TopoWaypoint(32, -2, False, 2.0)

        gen.topology = [
            (wp_48, wp_791),
            (wp_791, wp_30),
            (wp_48, wp_727),
            (wp_727, wp_32),
        ]

        links = gen._build_approach_exit_links()

        assert (48, 1) in links
        assert links[(48, 1)] == {(30, 1), (32, -2)}

    def test_flags_invalid_approach_exit_transition(self) -> None:
        with patch.object(BalancedRouteGenerator, "__init__", lambda self, *a, **kw: None):
            gen = BalancedRouteGenerator.__new__(BalancedRouteGenerator)

        gen.config = RouteConfig(validation_max_issues=4)
        gen._approach_exit_links = {(48, 1): {(30, 1)}}

        wp_a = _TopoWaypoint(48, 1, False, 0.0)
        wp_b = _TopoWaypoint(791, 1, True, 1.0)
        wp_c = _TopoWaypoint(39, -2, False, 2.0)  # invalid exit for approach 48/1

        route = [
            (MockTransform(MockLocation(0.0, 0.0)), "LaneFollow"),
            (MockTransform(MockLocation(1.0, 0.0)), "Straight"),
            (MockTransform(MockLocation(2.0, 0.0)), "LaneFollow"),
        ]

        gen.map = MagicMock()

        def _map_get_waypoint(
            location: MockLocation, *args: object, **kwargs: object
        ) -> _TopoWaypoint:
            if abs(float(location.x) - 0.0) < 1e-6:
                return wp_a
            if abs(float(location.x) - 1.0) < 1e-6:
                return wp_b
            return wp_c

        with patch.object(gen.map, "get_waypoint", side_effect=_map_get_waypoint):
            report = gen._validate_approach_conditioned_route(route)

        assert report["checked_transition_count"] == 1
        assert report["invalid_transition_count"] == 1
        assert len(report["issues"]) == 1

    def test_generate_route_retries_until_validation_passes(self) -> None:
        with patch.object(BalancedRouteGenerator, "__init__", lambda self, *a, **kw: None):
            gen = BalancedRouteGenerator.__new__(BalancedRouteGenerator)

        gen.config = RouteConfig(
            strategy="balanced",
            validate_junction_approach_exit=True,
            validation_max_regen_attempts=2,
        )

        route_bad = [(MockTransform(MockLocation(0.0, 0.0)), "LaneFollow")]
        route_good = [(MockTransform(MockLocation(1.0, 0.0)), "LaneFollow")]

        with patch.object(gen, "_generate_balanced_route", side_effect=[route_bad, route_good]):
            with patch.object(
                gen,
                "_validate_approach_conditioned_route",
                side_effect=[
                    {"invalid_transition_count": 1, "issues": []},
                    {"invalid_transition_count": 0, "issues": []},
                ],
            ) as validate_mock:
                route = gen.generate_route(np.random.default_rng(0))

        assert route == route_good
        assert validate_mock.call_count == 2

    def test_generate_route_returns_empty_after_validation_rejects_all(self) -> None:
        with patch.object(BalancedRouteGenerator, "__init__", lambda self, *a, **kw: None):
            gen = BalancedRouteGenerator.__new__(BalancedRouteGenerator)

        gen.config = RouteConfig(
            strategy="balanced",
            validate_junction_approach_exit=True,
            validation_max_regen_attempts=2,
        )

        route_bad = [(MockTransform(MockLocation(0.0, 0.0)), "LaneFollow")]

        with patch.object(gen, "_generate_balanced_route", side_effect=[route_bad, route_bad]):
            with patch.object(
                gen,
                "_validate_approach_conditioned_route",
                side_effect=[
                    {"invalid_transition_count": 1, "issues": []},
                    {"invalid_transition_count": 1, "issues": []},
                ],
            ) as validate_mock:
                route = gen.generate_route(np.random.default_rng(0))

        assert route == []
        assert validate_mock.call_count == 2
