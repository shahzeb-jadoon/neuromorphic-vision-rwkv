"""Focused logic tests for privileged expert planner behavior."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

from pytest import MonkeyPatch

from agents.expert.privileged_expert import (
    DrivingMode,
    ExpertConfig,
    PlannerPolicyException,
    PlannerPolicyReason,
    PrivilegedExpert,
    RoadOption,
)


class _Loc:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z

    def distance(self, other: "_Loc") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return float((dx * dx + dy * dy + dz * dz) ** 0.5)


class _Transform:
    def __init__(self, loc: _Loc, yaw: float = 0.0) -> None:
        self.location = loc
        self.rotation = SimpleNamespace(yaw=yaw)

    def get_forward_vector(self) -> SimpleNamespace:
        return SimpleNamespace(x=1.0, y=0.0)

    def get_right_vector(self) -> SimpleNamespace:
        return SimpleNamespace(x=0.0, y=1.0)


class _Waypoint:
    _id_counter = 1

    def __init__(self, x: float, y: float, road_id: int, lane_id: int, yaw: float = 0.0) -> None:
        self.id = _Waypoint._id_counter
        _Waypoint._id_counter += 1
        self.transform = _Transform(_Loc(x, y), yaw=yaw)
        self.road_id = road_id
        self.lane_id = lane_id
        self.is_junction = False
        self._left: _Waypoint | None = None
        self._right: _Waypoint | None = None
        self._nexts: list[_Waypoint] = []

    def get_left_lane(self) -> "_Waypoint | None":
        return self._left

    def get_right_lane(self) -> "_Waypoint | None":
        return self._right

    def next(self, _step: float) -> list["_Waypoint"]:
        return list(self._nexts)


def _build_expert() -> PrivilegedExpert:
    expert = PrivilegedExpert.__new__(PrivilegedExpert)
    expert.config = ExpertConfig()
    expert._ego_half_length = 1.0
    expert._tl_proceed_committed = False
    expert._turn_distance = None
    expert._smoothed_curvature = 0.0
    expert._wp_buffer = deque()
    expert._route = []
    expert._route_index = 0
    expert._route_length = 0
    expert._route_tl_candidates = set()
    expert._prev_lane_wp = None
    expert._prev_target = 30.0
    expert._cached_half_lane = expert.config.lane_half_width
    expert._gap_fill_rpcs = 0
    expert._gap_fill_diverged = 0
    expert._gap_fill_budget_exhausted = False
    expert._gap_fill_tried = set()
    expert._arc_buffer = deque()
    expert._planner_policy_reason = PlannerPolicyReason.NONE
    expert._planner_policy_exception = PlannerPolicyException.NONE
    expert._dist_to_next_junction = -1.0
    expert._frame_pre_junction_lane_change_violation = 0
    expert._frame_consecutive_lane_change_violation = 0
    expert._frame_straight_exit_lane_violation = 0
    expert._legality_wrong_way_filtered = 0
    expert._legality_do_not_enter_filtered = 0
    expert._legality_turn_mismatch_filtered = 0
    expert._policy_distance_traveled = 0.0
    expert._policy_last_location = None
    expert._policy_last_lane_change_distance = -1e9
    setattr(expert, "_route_curvature", lambda: 0.0)
    expert.vehicle = SimpleNamespace(
        bounding_box=SimpleNamespace(extent=SimpleNamespace(y=1.0)),
        get_transform=lambda: _Transform(_Loc(0.0, 0.0)),
        is_at_traffic_light=lambda: False,
        get_traffic_light=lambda: None,
    )
    expert.map = SimpleNamespace(get_waypoint=lambda _loc: _Waypoint(0.0, 0.0, 1, 1))
    expert.world = SimpleNamespace(get_actor=lambda _id: None)
    setattr(expert, "_stop_line_distance", lambda *_args, **_kwargs: (None, None))
    return expert


def test_plan_speed_tl_mode_not_masked_by_lower_target(monkeypatch: MonkeyPatch) -> None:
    expert = _build_expert()
    expert._prev_target = 10.0
    fake_tl = SimpleNamespace(Green=1, Yellow=2, Red=3)
    monkeypatch.setattr(
        "agents.expert.privileged_expert.carla", SimpleNamespace(TrafficLightState=fake_tl)
    )

    target, mode = expert._plan_speed(
        speed_kmh=10.0,
        nav_command=RoadOption.LANEFOLLOW,
        tl_state=fake_tl.Red,
        tl_distance=30.0,
        ss_distance=None,
        ss_committed=False,
        lead_distance=None,
        lead_speed=None,
        lead_lateral=None,
        ped_dist=None,
        base_target=10.0,
    )

    assert mode == DrivingMode.TRAFFIC_LIGHT_STOP
    assert target == 10.0


def test_plan_speed_stopped_lead_yields_to_red_tl(monkeypatch: MonkeyPatch) -> None:
    expert = _build_expert()
    expert._prev_target = 20.0
    fake_tl = SimpleNamespace(Green=1, Yellow=2, Red=3)
    monkeypatch.setattr(
        "agents.expert.privileged_expert.carla", SimpleNamespace(TrafficLightState=fake_tl)
    )

    target, mode = expert._plan_speed(
        speed_kmh=20.0,
        nav_command=RoadOption.LANEFOLLOW,
        tl_state=fake_tl.Red,
        tl_distance=18.0,
        ss_distance=None,
        ss_committed=False,
        lead_distance=8.0,
        lead_speed=0.0,
        lead_lateral=0.0,
        ped_dist=None,
        base_target=20.0,
    )

    assert mode == DrivingMode.TRAFFIC_LIGHT_STOP
    assert target > 0.0


def test_plan_speed_stopped_lead_in_emergency_range_overrides_tl(monkeypatch: MonkeyPatch) -> None:
    expert = _build_expert()
    expert._prev_target = 30.0
    fake_tl = SimpleNamespace(Green=1, Yellow=2, Red=3)
    monkeypatch.setattr(
        "agents.expert.privileged_expert.carla", SimpleNamespace(TrafficLightState=fake_tl)
    )

    target, mode = expert._plan_speed(
        speed_kmh=20.0,
        nav_command=RoadOption.LANEFOLLOW,
        tl_state=fake_tl.Red,
        tl_distance=45.0,
        ss_distance=None,
        ss_committed=False,
        lead_distance=4.0,
        lead_speed=0.0,
        lead_lateral=0.0,
        ped_dist=None,
        base_target=30.0,
    )

    assert mode == DrivingMode.EMERGENCY_STOP
    assert target == 0.0


def test_plan_speed_yellow_requires_junction_clearance(monkeypatch: MonkeyPatch) -> None:
    expert = _build_expert()
    expert._prev_target = 45.0
    setattr(expert, "_estimate_junction_clearance_distance", lambda: 80.0)

    fake_tl = SimpleNamespace(Green=1, Yellow=2, Red=3)
    monkeypatch.setattr(
        "agents.expert.privileged_expert.carla", SimpleNamespace(TrafficLightState=fake_tl)
    )

    target, mode = expert._plan_speed(
        speed_kmh=40.0,
        nav_command=RoadOption.LANEFOLLOW,
        tl_state=fake_tl.Yellow,
        tl_distance=20.0,
        ss_distance=None,
        ss_committed=False,
        lead_distance=None,
        lead_speed=None,
        lead_lateral=None,
        ped_dist=None,
        base_target=45.0,
    )

    assert mode == DrivingMode.TRAFFIC_LIGHT_STOP
    assert target >= 0.0


def test_plan_speed_yellow_commit_respects_max_distance(monkeypatch: MonkeyPatch) -> None:
    expert = _build_expert()
    expert._prev_target = 60.0
    setattr(expert, "_estimate_junction_clearance_distance", lambda: 10.0)

    fake_tl = SimpleNamespace(Green=1, Yellow=2, Red=3)
    monkeypatch.setattr(
        "agents.expert.privileged_expert.carla", SimpleNamespace(TrafficLightState=fake_tl)
    )

    target, mode = expert._plan_speed(
        speed_kmh=60.0,
        nav_command=RoadOption.LANEFOLLOW,
        tl_state=fake_tl.Yellow,
        tl_distance=40.0,
        ss_distance=None,
        ss_committed=False,
        lead_distance=None,
        lead_speed=None,
        lead_lateral=None,
        ped_dist=None,
        base_target=60.0,
    )

    assert mode == DrivingMode.TRAFFIC_LIGHT_STOP
    assert expert._tl_proceed_committed is False
    assert target < 60.0


def test_plan_speed_full_stop_clears_yellow_commit(monkeypatch: MonkeyPatch) -> None:
    expert = _build_expert()
    expert._prev_target = 20.0
    expert._tl_proceed_committed = True

    fake_tl = SimpleNamespace(Green=1, Yellow=2, Red=3)
    monkeypatch.setattr(
        "agents.expert.privileged_expert.carla", SimpleNamespace(TrafficLightState=fake_tl)
    )

    target, mode = expert._plan_speed(
        speed_kmh=0.0,
        nav_command=RoadOption.LANEFOLLOW,
        tl_state=fake_tl.Red,
        tl_distance=1.0,
        ss_distance=None,
        ss_committed=False,
        lead_distance=None,
        lead_speed=None,
        lead_lateral=None,
        ped_dist=None,
        base_target=20.0,
    )

    assert expert._tl_proceed_committed is False
    assert mode == DrivingMode.TRAFFIC_LIGHT_STOP
    assert target == 0.0


def test_plan_speed_moving_keeps_yellow_commit(monkeypatch: MonkeyPatch) -> None:
    expert = _build_expert()
    expert._prev_target = 20.0
    expert._tl_proceed_committed = True

    fake_tl = SimpleNamespace(Green=1, Yellow=2, Red=3)
    monkeypatch.setattr(
        "agents.expert.privileged_expert.carla", SimpleNamespace(TrafficLightState=fake_tl)
    )

    target, mode = expert._plan_speed(
        speed_kmh=12.0,
        nav_command=RoadOption.LANEFOLLOW,
        tl_state=fake_tl.Red,
        tl_distance=20.0,
        ss_distance=None,
        ss_committed=False,
        lead_distance=None,
        lead_speed=None,
        lead_lateral=None,
        ped_dist=None,
        base_target=20.0,
    )

    assert expert._tl_proceed_committed is True
    assert mode == DrivingMode.FOLLOW_LANE
    assert target == 20.0


def test_check_traffic_light_uses_stop_line_not_pole_for_stale_clear() -> None:
    expert = _build_expert()
    expert._tl_waiting = 7
    setattr(expert, "_tl_stop_loc", _Loc(20.0, 0.0))
    expert._tl_proceed_committed = False

    tl_actor = SimpleNamespace(
        state=3,
        get_transform=lambda: _Transform(_Loc(-10.0, 0.0)),
    )
    expert.world = SimpleNamespace(get_actor=lambda _id: tl_actor)

    state, dist = expert._check_traffic_light(_Transform(_Loc(0.0, 0.0)))

    assert state == 3
    assert dist is not None
    assert round(dist, 1) == 20.0
    assert expert._tl_waiting == 7


def test_check_traffic_light_does_not_return_pole_distance_when_stop_line_missing() -> None:
    expert = _build_expert()
    expert._tl_waiting = 9
    setattr(expert, "_tl_stop_loc", None)

    tl_actor = SimpleNamespace(
        state=3,
        get_transform=lambda: _Transform(_Loc(10.0, 0.0)),
    )

    class _Actors:
        @staticmethod
        def filter(_pattern: str) -> list[object]:
            return []

    expert.world = SimpleNamespace(
        get_actor=lambda _id: tl_actor,
        get_actors=lambda: _Actors(),
    )

    state, dist = expert._check_traffic_light(_Transform(_Loc(0.0, 0.0)))

    assert state is None
    assert dist is None
    assert expert._tl_waiting == 9


def test_refresh_tl_stop_loc_preserves_previous_when_unavailable() -> None:
    expert = _build_expert()
    expert._tl_waiting = 21
    previous_stop = _Loc(30.0, 0.0)
    setattr(expert, "_tl_stop_loc", previous_stop)
    setattr(expert, "_stop_line_distance", lambda *_args, **_kwargs: (None, None))

    tl_actor = SimpleNamespace(get_transform=lambda: _Transform(_Loc(80.0, 0.0)))
    expert.world = SimpleNamespace(get_actor=lambda _id: tl_actor)

    expert._refresh_tl_stop_loc(_Transform(_Loc(0.0, 0.0)))

    assert expert._tl_stop_loc is previous_stop


def test_stop_line_distance_matches_route_aware_junction_seam() -> None:
    expert = _build_expert()
    ego_wp = _Waypoint(0.0, 0.0, road_id=10, lane_id=1, yaw=0.0)
    route_wp = _Waypoint(8.0, 0.0, road_id=11, lane_id=1, yaw=0.0)
    awp = _Waypoint(8.5, 0.0, road_id=11, lane_id=1, yaw=0.0)

    expert.map = SimpleNamespace(get_waypoint=lambda _loc: ego_wp)
    expert._wp_buffer = deque([(route_wp, RoadOption.LANEFOLLOW)])

    tl_actor = SimpleNamespace(get_affected_lane_waypoints=lambda: [awp])
    dist, stop_loc = PrivilegedExpert._stop_line_distance(
        expert,
        tl_actor,
        _Transform(_Loc(0.0, 0.0), yaw=0.0),
    )

    assert dist is not None
    assert stop_loc is awp.transform.location


def test_get_target_overrides_with_physical_lane_change_label() -> None:
    expert = _build_expert()

    current_wp = _Waypoint(0.0, 0.0, road_id=10, lane_id=2)
    prev_wp = _Waypoint(-1.0, 0.0, road_id=10, lane_id=1)
    setattr(prev_wp, "_left", current_wp)

    far_wp = _Waypoint(15.0, 0.0, road_id=10, lane_id=2)
    expert._wp_buffer = deque([(far_wp, RoadOption.LANEFOLLOW)])
    setattr(expert, "_prev_lane_wp", prev_wp)
    expert.map = SimpleNamespace(get_waypoint=lambda _loc: current_wp)

    _target_wp, nav_label, nav_command = expert._get_target(_Transform(_Loc(0.0, 0.0)))

    assert nav_command == RoadOption.LANEFOLLOW
    assert nav_label == RoadOption.CHANGELANELEFT


def test_turning_keeps_stopped_vehicle_in_corridor_as_blocker(monkeypatch: MonkeyPatch) -> None:
    expert = _build_expert()
    expert._prev_target = 20.0
    expert._turn_distance = 12.0

    junction_wp = _Waypoint(0.0, 0.0, road_id=10, lane_id=1)
    junction_wp.is_junction = True
    expert.map = SimpleNamespace(get_waypoint=lambda _loc: junction_wp)

    fake_tl = SimpleNamespace(Green=1, Yellow=2, Red=3)
    monkeypatch.setattr(
        "agents.expert.privileged_expert.carla", SimpleNamespace(TrafficLightState=fake_tl)
    )
    setattr(expert, "_dynamic_half_lane", lambda _t: 1.75)

    target, mode = expert._plan_speed(
        speed_kmh=20.0,
        nav_command=RoadOption.LEFT,
        tl_state=None,
        tl_distance=None,
        ss_distance=None,
        ss_committed=False,
        lead_distance=20.0,
        lead_speed=0.0,
        lead_lateral=1.5,
        ped_dist=None,
        base_target=20.0,
    )

    assert mode == DrivingMode.EMERGENCY_STOP
    assert target == 0.0


def test_closest_pedestrian_distance_uses_hybrid_floor() -> None:
    expert = _build_expert()
    setattr(expert, "_dynamic_half_lane", lambda _t: 1.75)

    ped = {
        "is_ahead": True,
        "lateral": 2.2,
        "distance": 12.0,
        "predicted_ahead": False,
    }
    dist = expert._closest_pedestrian_distance([ped], _Transform(_Loc(0.0, 0.0)))

    assert dist == 12.0


def test_fill_junction_gaps_prefers_bezier_for_junction_gap() -> None:
    expert = _build_expert()

    wp_prev = _Waypoint(0.0, 0.0, road_id=10, lane_id=1, yaw=0.0)
    wp_next = _Waypoint(8.0, 0.0, road_id=10, lane_id=1, yaw=0.0)
    projected = [
        _Waypoint(2.0, 0.0, road_id=10, lane_id=1, yaw=0.0),
        _Waypoint(4.0, 0.0, road_id=10, lane_id=1, yaw=0.0),
        _Waypoint(6.0, 0.0, road_id=10, lane_id=1, yaw=0.0),
    ]
    wrong_branch = _Waypoint(2.0, 2.0, road_id=11, lane_id=1, yaw=35.0)
    next_calls = {"count": 0}

    def _counting_next(_step: float) -> list[_Waypoint]:
        next_calls["count"] += 1
        return [wrong_branch]

    wp_prev.next = _counting_next  # type: ignore[method-assign]

    def _project_to_route(loc: _Loc) -> _Waypoint:
        return min(projected, key=lambda wp: abs(wp.transform.location.x - loc.x))

    expert.map = SimpleNamespace(get_waypoint=_project_to_route)
    expert._wp_buffer = deque(
        [
            (wp_prev, RoadOption.LANEFOLLOW),
            (wp_next, RoadOption.LANEFOLLOW),
        ]
    )

    expert._fill_junction_gaps(None)

    # RPC fallback must not have been called (Bezier was preferred).
    assert next_calls["count"] == 0
    # Bezier fills land in _arc_buffer, not _wp_buffer.
    arc_wps = [wp for wp, _cmd in expert._arc_buffer]
    assert any(wp in arc_wps for wp in projected), "Bezier fills not found in _arc_buffer"
    assert wrong_branch not in arc_wps


def test_fill_junction_gaps_bezier_goes_to_arc_buffer_not_wp_buffer() -> None:
    """Bezier arc fills must land in _arc_buffer, never in _wp_buffer.

    Dual-buffer design: _wp_buffer stays as a pure route tracker.  Arc fills
    are placed in _arc_buffer so no overflow can displace post-junction route
    waypoints and no _route_index rewind is needed.
    """
    expert = _build_expert()

    wp_prev = _Waypoint(0.0, 0.0, road_id=10, lane_id=1, yaw=0.0)
    wp_next = _Waypoint(8.0, 0.0, road_id=10, lane_id=1, yaw=0.0)
    projected = [
        _Waypoint(2.0, 0.0, road_id=10, lane_id=1, yaw=0.0),
        _Waypoint(4.0, 0.0, road_id=10, lane_id=1, yaw=0.0),
        _Waypoint(6.0, 0.0, road_id=10, lane_id=1, yaw=0.0),
    ]

    def _project(loc: object) -> "_Waypoint":
        x = float(getattr(loc, "x", 0.0))
        return min(projected, key=lambda wp: abs(wp.transform.location.x - x))

    expert.map = SimpleNamespace(get_waypoint=_project)
    expert._wp_buffer = deque([(wp_prev, RoadOption.LANEFOLLOW), (wp_next, RoadOption.LANEFOLLOW)])

    expert._fill_junction_gaps(None)

    # Arc fills landed in _arc_buffer.
    assert len(expert._arc_buffer) > 0, "Bezier fills were not placed in _arc_buffer"
    # _wp_buffer must still contain the junction exit (wp_next) — not disturbed.
    route_wps = [wp for wp, _ in expert._wp_buffer]
    assert wp_next in route_wps, "junction exit (wp_next) was removed from _wp_buffer"
    # No arc waypoints in _wp_buffer (they should only be in _arc_buffer).
    arc_wps = {wp for wp, _ in expert._arc_buffer}
    for wp in route_wps:
        assert (
            wp not in arc_wps
        ), f"arc waypoint leaked into _wp_buffer at x={wp.transform.location.x}"


def test_fill_junction_gaps_does_not_start_second_arc_while_one_is_active() -> None:
    """If _arc_buffer is already populated, gap fill must not overwrite it."""
    expert = _build_expert()

    wp_prev = _Waypoint(0.0, 0.0, road_id=10, lane_id=1, yaw=0.0)
    wp_next = _Waypoint(8.0, 0.0, road_id=10, lane_id=1, yaw=0.0)

    sentinel_arc_wp = _Waypoint(1.0, 0.0, road_id=99, lane_id=1)
    expert._arc_buffer = deque([(sentinel_arc_wp, RoadOption.LANEFOLLOW)])

    def _project(loc: object) -> "_Waypoint":
        return _Waypoint(float(getattr(loc, "x", 0.0)), 0.0, road_id=10, lane_id=1, yaw=0.0)

    expert.map = SimpleNamespace(get_waypoint=_project)
    expert._wp_buffer = deque([(wp_prev, RoadOption.LANEFOLLOW), (wp_next, RoadOption.LANEFOLLOW)])

    expert._fill_junction_gaps(None)

    # Active arc must not have been replaced.
    assert (
        expert._arc_buffer[0][0] is sentinel_arc_wp
    ), "_arc_buffer was overwritten while an arc was already active"


def test_advance_route_uses_segment_progress_even_when_heading_would_disagree() -> None:
    expert = _build_expert()
    expert.config.min_follow_distance = 0.1

    wp0 = _Waypoint(0.0, 0.0, road_id=1, lane_id=1)
    wp1 = _Waypoint(4.0, 0.0, road_id=1, lane_id=1)
    wp2 = _Waypoint(8.0, 0.0, road_id=1, lane_id=1)
    expert._wp_buffer = deque(
        [
            (wp0, RoadOption.LANEFOLLOW),
            (wp1, RoadOption.LANEFOLLOW),
            (wp2, RoadOption.LANEFOLLOW),
        ]
    )
    setattr(expert, "_fill_buffer", lambda _transform: None)

    # x=1.0 is before midpoint (2.0) on segment [0->4], so wp0 should remain.
    expert._advance_route(_Transform(_Loc(1.0, 0.0), yaw=180.0))
    assert expert._wp_buffer[0][0] is wp0

    # x=2.5 is past midpoint, so wp0 should now be consumed.
    expert._advance_route(_Transform(_Loc(2.5, 0.0), yaw=180.0))
    assert expert._wp_buffer[0][0] is wp1


def test_densify_route_transitions_inserts_waypoints_for_large_gap() -> None:
    expert = _build_expert()
    expert.config.route_densify_enabled = True
    expert.config.route_densify_step = 1.0

    wp0 = _Waypoint(0.0, 0.0, road_id=1, lane_id=1)
    wp2 = _Waypoint(2.0, 0.0, road_id=1, lane_id=1)
    wp4 = _Waypoint(4.0, 0.0, road_id=1, lane_id=1)
    wp6 = _Waypoint(6.0, 0.0, road_id=1, lane_id=1)
    wp8 = _Waypoint(8.0, 0.0, road_id=1, lane_id=1)
    wp10 = _Waypoint(10.0, 0.0, road_id=1, lane_id=1)
    wp12 = _Waypoint(12.0, 0.0, road_id=1, lane_id=1)

    wp0._nexts = [wp2]
    wp2._nexts = [wp4]
    wp4._nexts = [wp6]
    wp6._nexts = [wp8]
    wp8._nexts = [wp10]
    wp10._nexts = [wp12]

    route = [(wp0, RoadOption.LANEFOLLOW), (wp12, RoadOption.LEFT)]
    dense = expert._densify_route_transitions(route)

    assert len(dense) > len(route)
    assert dense[0][0] is wp0
    assert dense[-1][0] is wp12
    assert all(cmd == RoadOption.LEFT for _wp, cmd in dense[1:])


def test_densify_route_transitions_uses_projection_fallback() -> None:
    expert = _build_expert()
    expert.config.route_densify_enabled = True
    expert.config.route_densify_step = 1.0
    expert.config.route_densify_projection_fallback = True

    wp0 = _Waypoint(0.0, 0.0, road_id=1, lane_id=1)
    wp8 = _Waypoint(8.0, 0.0, road_id=2, lane_id=1)
    wrong = _Waypoint(1.0, 4.0, road_id=9, lane_id=1)
    wp0._nexts = [wrong]
    wrong._nexts = []

    projected = [
        _Waypoint(2.0, 0.0, road_id=2, lane_id=1),
        _Waypoint(4.0, 0.0, road_id=2, lane_id=1),
        _Waypoint(6.0, 0.0, road_id=2, lane_id=1),
    ]

    def _project(loc: object) -> _Waypoint:
        x = float(getattr(loc, "x", 0.0))
        return min(projected, key=lambda wp: abs(wp.transform.location.x - x))

    expert.map = SimpleNamespace(get_waypoint=_project)

    route = [(wp0, RoadOption.LANEFOLLOW), (wp8, RoadOption.LEFT)]
    dense = expert._densify_route_transitions(route)

    inserted = [wp for wp, _ in dense[1:-1]]
    assert len(inserted) >= 2
    assert any(wp in inserted for wp in projected)


def test_densify_route_transitions_can_be_disabled() -> None:
    expert = _build_expert()
    expert.config.route_densify_enabled = False

    wp0 = _Waypoint(0.0, 0.0, road_id=1, lane_id=1)
    wp10 = _Waypoint(10.0, 0.0, road_id=1, lane_id=1)
    route = [(wp0, RoadOption.LANEFOLLOW), (wp10, RoadOption.LEFT)]

    dense = expert._densify_route_transitions(route)

    assert len(dense) == len(route)
    assert dense[0][0] is wp0
    assert dense[1][0] is wp10


def test_route_legality_filters_wrong_way_transition() -> None:
    expert = _build_expert()
    expert.config.planner_wrong_way_max_yaw_diff_deg = 80.0

    wp0 = _Waypoint(0.0, 0.0, road_id=1, lane_id=1, yaw=0.0)
    wrong_way = _Waypoint(-6.0, 0.0, road_id=1, lane_id=1, yaw=180.0)
    wp_ok = _Waypoint(6.0, 0.0, road_id=1, lane_id=1, yaw=0.0)

    filtered = expert._enforce_route_legality(
        [
            (wp0, RoadOption.LANEFOLLOW),
            (wrong_way, RoadOption.LANEFOLLOW),
            (wp_ok, RoadOption.LANEFOLLOW),
        ]
    )

    assert [wp for wp, _ in filtered] == [wp0, wp_ok]
    assert expert._legality_wrong_way_filtered >= 1


def test_route_legality_rejects_lane_sign_flip_without_exception() -> None:
    expert = _build_expert()
    expert.config.planner_allow_lane_id_sign_flip = False

    wp0 = _Waypoint(0.0, 0.0, road_id=1, lane_id=1, yaw=0.0)
    wp1 = _Waypoint(4.0, 0.0, road_id=1, lane_id=-1, yaw=0.0)

    legal = expert._is_route_transition_legal(
        wp0,
        wp1,
        RoadOption.LANEFOLLOW,
        count_violation=True,
    )

    assert legal is False
    assert expert._legality_do_not_enter_filtered == 1


def test_route_legality_rejects_turn_mismatch() -> None:
    expert = _build_expert()

    wp0 = _Waypoint(0.0, 0.0, road_id=1, lane_id=1, yaw=0.0)
    wp_right = _Waypoint(5.0, -5.0, road_id=2, lane_id=1, yaw=-45.0)

    legal = expert._is_route_transition_legal(
        wp0,
        wp_right,
        RoadOption.LEFT,
        count_violation=True,
    )

    assert legal is False
    assert expert._legality_turn_mismatch_filtered == 1


def test_route_end_extension_skips_illegal_next_branch() -> None:
    expert = _build_expert()
    expert._route_index = 1
    expert._route_length = 1

    last_wp = _Waypoint(0.0, 0.0, road_id=1, lane_id=1, yaw=0.0)
    bad_next = _Waypoint(-3.0, 0.0, road_id=1, lane_id=1, yaw=180.0)
    good_next = _Waypoint(3.0, 0.0, road_id=1, lane_id=1, yaw=0.0)
    last_wp._nexts = [bad_next, good_next]
    bad_next._nexts = []
    good_next._nexts = []

    expert._wp_buffer = deque([(last_wp, RoadOption.LANEFOLLOW)], maxlen=5)

    expert._fill_buffer(None)

    assert len(expert._wp_buffer) == 2
    assert expert._wp_buffer[1][0] is good_next
    assert expert._legality_wrong_way_filtered >= 1


def test_lane_change_freeze_blocks_non_mandatory_change() -> None:
    expert = _build_expert()
    expert._policy_distance_traveled = 120.0
    expert._policy_last_lane_change_distance = 0.0

    lc_wp = _Waypoint(8.0, 0.0, road_id=1, lane_id=1)
    junc_wp = _Waypoint(12.0, 0.0, road_id=2, lane_id=1)
    junc_wp.is_junction = True
    expert._wp_buffer = deque(
        [
            (lc_wp, RoadOption.CHANGELANELEFT),
            (junc_wp, RoadOption.STRAIGHT),
        ]
    )

    ego_wp = _Waypoint(0.0, 0.0, road_id=1, lane_id=1)
    label = expert._apply_lane_change_policy(
        RoadOption.CHANGELANELEFT,
        ego_wp,
        _Loc(0.0, 0.0),
    )

    assert label == RoadOption.LANEFOLLOW
    assert expert._frame_pre_junction_lane_change_violation == 1
    assert expert._planner_policy_reason == PlannerPolicyReason.PRE_JUNCTION_LANE_CHANGE_BLOCKED


def test_lane_change_mandatory_exit_exception_allows_change() -> None:
    expert = _build_expert()
    expert._policy_distance_traveled = 120.0
    expert._policy_last_lane_change_distance = 0.0

    lc_wp = _Waypoint(8.0, 0.0, road_id=1, lane_id=1)
    junc_wp = _Waypoint(12.0, 0.0, road_id=2, lane_id=1)
    junc_wp.is_junction = True
    expert._wp_buffer = deque(
        [
            (lc_wp, RoadOption.CHANGELANELEFT),
            (junc_wp, RoadOption.LEFT),
        ]
    )

    ego_wp = _Waypoint(0.0, 0.0, road_id=1, lane_id=1)
    label = expert._apply_lane_change_policy(
        RoadOption.CHANGELANELEFT,
        ego_wp,
        _Loc(0.0, 0.0),
    )

    assert label == RoadOption.CHANGELANELEFT
    assert expert._planner_policy_exception == PlannerPolicyException.MANDATORY_EXIT
    assert expert._frame_pre_junction_lane_change_violation == 0


def test_consecutive_lane_change_blocked_without_exception() -> None:
    expert = _build_expert()
    expert._policy_distance_traveled = 50.0
    expert._policy_last_lane_change_distance = 45.0
    expert._wp_buffer = deque([(_Waypoint(30.0, 0.0, road_id=1, lane_id=1), RoadOption.LANEFOLLOW)])

    ego_wp = _Waypoint(0.0, 0.0, road_id=1, lane_id=1)
    label = expert._apply_lane_change_policy(
        RoadOption.CHANGELANERIGHT,
        ego_wp,
        _Loc(0.0, 0.0),
    )

    assert label == RoadOption.LANEFOLLOW
    assert expert._frame_consecutive_lane_change_violation == 1
    assert expert._planner_policy_reason == PlannerPolicyReason.CONSECUTIVE_LANE_CHANGE_BLOCKED


def test_straight_exit_lane_warning_records_violation() -> None:
    expert = _build_expert()
    junc_wp = _Waypoint(15.0, 0.0, road_id=2, lane_id=1)
    junc_wp.is_junction = True
    expert._wp_buffer = deque(
        [
            (_Waypoint(8.0, 0.0, road_id=1, lane_id=1), RoadOption.LANEFOLLOW),
            (junc_wp, RoadOption.STRAIGHT),
        ]
    )
    setattr(expert, "_is_exit_only_lane_for_straight", lambda _ego_wp: True)

    ego_wp = _Waypoint(0.0, 0.0, road_id=1, lane_id=1)
    label = expert._apply_lane_change_policy(
        RoadOption.STRAIGHT,
        ego_wp,
        _Loc(0.0, 0.0),
    )

    assert label == RoadOption.STRAIGHT
    assert expert._frame_straight_exit_lane_violation == 1
    assert expert._planner_policy_reason == PlannerPolicyReason.STRAIGHT_EXIT_LANE_WARNING
