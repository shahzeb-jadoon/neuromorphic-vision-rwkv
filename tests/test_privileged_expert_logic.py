"""Focused logic tests for privileged expert planner behavior."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

from pytest import MonkeyPatch

from agents.expert.privileged_expert import (
    DrivingMode,
    ExpertConfig,
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


def test_fill_junction_gaps_chooses_route_aligned_branch() -> None:
    expert = _build_expert()

    wp_prev = _Waypoint(0.0, 0.0, road_id=10, lane_id=1, yaw=0.0)
    wp_next = _Waypoint(8.0, 0.0, road_id=10, lane_id=1, yaw=0.0)

    wrong_branch = _Waypoint(2.0, 2.0, road_id=11, lane_id=1, yaw=35.0)
    right_branch = _Waypoint(2.0, 0.0, road_id=10, lane_id=1, yaw=0.0)
    near_target = _Waypoint(6.5, 0.0, road_id=10, lane_id=1, yaw=0.0)

    wp_prev._nexts = [wrong_branch, right_branch]  # wrong branch first by list order
    right_branch._nexts = [near_target]
    wrong_branch._nexts = [_Waypoint(4.5, 3.0, road_id=11, lane_id=1, yaw=45.0)]

    expert._wp_buffer = deque(
        [
            (wp_prev, RoadOption.LANEFOLLOW),
            (wp_next, RoadOption.LANEFOLLOW),
        ]
    )

    expert._fill_junction_gaps(None)
    filled_wps = [wp for wp, _cmd in expert._wp_buffer]

    assert right_branch in filled_wps
    assert wrong_branch not in filled_wps


def test_fill_junction_gaps_prefers_bezier_before_branch_fallback() -> None:
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
    filled_wps = [wp for wp, _cmd in expert._wp_buffer]

    assert next_calls["count"] == 0
    assert any(wp in filled_wps for wp in projected)
    assert wrong_branch not in filled_wps


def test_fill_buffer_route_end_prefers_lane_order_continuity() -> None:
    expert = _build_expert()

    last_wp = _Waypoint(0.0, 0.0, road_id=50, lane_id=2, yaw=90.0)
    wrong_lane = _Waypoint(0.0, 2.0, road_id=60, lane_id=4, yaw=90.0)
    right_lane = _Waypoint(0.0, 2.0, road_id=60, lane_id=2, yaw=90.0)

    # Wrong lane comes first to emulate unstable nexts[] ordering.
    last_wp._nexts = [wrong_lane, right_lane]
    wrong_lane._nexts = []
    right_lane._nexts = []

    expert._wp_buffer = deque([(last_wp, RoadOption.LANEFOLLOW)])
    expert._route = []
    expert._route_index = 0
    expert._route_length = 0

    expert._fill_buffer(None)
    filled_wps = [wp for wp, _cmd in expert._wp_buffer]

    assert len(filled_wps) >= 2
    assert filled_wps[1] is right_lane


def test_advance_route_uses_segment_progress() -> None:
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

    expert._advance_route(_Transform(_Loc(2.5, 0.0), yaw=180.0))

    assert len(expert._wp_buffer) == 2
    assert expert._wp_buffer[0][0] is wp1


def test_advance_route_caps_waypoint_pops_per_tick() -> None:
    expert = _build_expert()
    expert.config.min_follow_distance = 0.1
    expert.config.max_waypoint_pops_per_tick = 2

    wps = [_Waypoint(float(i * 2), 0.0, road_id=1, lane_id=1) for i in range(5)]
    expert._wp_buffer = deque([(wp, RoadOption.LANEFOLLOW) for wp in wps])
    setattr(expert, "_fill_buffer", lambda _transform: None)

    expert._advance_route(_Transform(_Loc(100.0, 0.0), yaw=0.0))

    assert len(expert._wp_buffer) == 3
    assert expert._wp_buffer[0][0] is wps[2]
