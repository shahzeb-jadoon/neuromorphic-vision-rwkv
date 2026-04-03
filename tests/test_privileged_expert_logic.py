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


class _Waypoint:
    def __init__(self, x: float, y: float, road_id: int, lane_id: int, yaw: float = 0.0) -> None:
        self.transform = _Transform(_Loc(x, y), yaw=yaw)
        self.road_id = road_id
        self.lane_id = lane_id
        self.is_junction = False
        self._left: _Waypoint | None = None
        self._right: _Waypoint | None = None

    def get_left_lane(self) -> "_Waypoint | None":
        return self._left

    def get_right_lane(self) -> "_Waypoint | None":
        return self._right


def _build_expert() -> PrivilegedExpert:
    expert = PrivilegedExpert.__new__(PrivilegedExpert)
    expert.config = ExpertConfig()
    expert._ego_half_length = 1.0
    expert._tl_proceed_committed = False
    expert._turn_distance = None
    expert._smoothed_curvature = 0.0
    expert._wp_buffer = deque()
    expert._prev_lane_wp = None
    expert._prev_target = 30.0
    expert._cached_half_lane = expert.config.lane_half_width
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
        lead_distance=4.0,
        lead_speed=0.0,
        lead_lateral=0.0,
        ped_dist=None,
        base_target=20.0,
    )

    assert mode == DrivingMode.TRAFFIC_LIGHT_STOP
    assert target > 0.0


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
