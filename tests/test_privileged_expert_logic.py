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
        self.lane_width = 3.5
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
    expert._route_wp_identity_to_index = {}
    expert._route_index = 0
    expert._route_length = 0
    expert._route_tl_candidates = set()
    expert._prev_lane_wp = None
    expert._prev_target = 30.0
    expert._cached_half_lane = expert.config.lane_half_width
    expert._gap_fill_rpcs = 0
    expert._gap_fill_diverged = 0
    expert._gap_fill_budget_exhausted = False
    expert._gap_fill_rpc_tried = set()
    expert._gap_fill_bezier_tried = set()
    expert._gap_fill_rpc_attempts = {}
    expert._gap_fill_bezier_attempts = {}
    expert._gap_fill_pair_blacklist = set()
    expert._arc_buffer = deque()
    expert._arc_watchdog_stall_frames = 0
    expert._arc_watchdog_prev_head_dist = None
    expert._arc_watchdog_trigger_count = 0
    expert._steer_from_arc = 0
    expert._planner_policy_reason = PlannerPolicyReason.NONE
    expert._planner_policy_exception = PlannerPolicyException.NONE
    expert._dist_to_next_junction = -1.0
    expert._frame_pre_junction_lane_change_violation = 0
    expert._frame_consecutive_lane_change_violation = 0
    expert._frame_straight_exit_lane_violation = 0
    expert._frame_turn_lane_legality_violation = 0
    expert._frame_roundabout_lane_change_violation = 0
    expert._frame_route_watchdog_triggered = 0
    expert._legality_wrong_way_filtered = 0
    expert._legality_do_not_enter_filtered = 0
    expert._legality_turn_mismatch_filtered = 0
    expert._legality_continuity_kept = 0
    expert._route_watchdog_consecutive_frames = 0
    expert._route_watchdog_progress_stall_frames = 0
    expert._route_watchdog_prev_route_completion = None
    expert._policy_distance_traveled = 0.0
    expert._policy_last_location = None
    expert._policy_last_lane_change_distance = -1e9
    expert._policy_last_roundabout_lane_change_distance = -1e9
    expert._policy_block_streak = 0
    expert._topology_cycle_road_ids = set()
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


def test_expert_config_defaults_use_conservative_arc_settings() -> None:
    cfg = ExpertConfig()
    assert cfg.gap_fill_bezier_retry_limit == 1
    assert cfg.gap_fill_bezier_max_pair_index == 1
    assert cfg.planner_policy_steer_consistency_gate_enabled is True
    assert cfg.arc_watchdog_enabled is True


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


def test_plan_speed_stopped_lead_uses_queue_following_under_red_tl(
    monkeypatch: MonkeyPatch,
) -> None:
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

    assert mode == DrivingMode.VEHICLE_FOLLOWING
    assert 0.0 < target < 20.0


def test_plan_speed_fast_lead_keeps_traffic_light_authority(monkeypatch: MonkeyPatch) -> None:
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
        lead_speed=18.0,
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


def test_lead_vehicle_info_rejects_oncoming_adjacent_overlap() -> None:
    expert = _build_expert()
    expert.config.vehicle_oncoming_conflict_gate_enabled = True
    expert.config.oncoming_lead_lateral_max_m = 1.2

    nearby = [
        {
            "is_ahead": True,
            "same_direction": False,
            "obb_in_lane": True,
            "distance": 0.0,
            "longitudinal": 3.8,
            "lateral": -2.4,
            "speed_kmh": 0.0,
        }
    ]

    lead_distance, lead_speed, lead_lateral = expert._lead_vehicle_info(nearby, ego_speed_kmh=15.0)

    assert lead_distance is None
    assert lead_speed is None
    assert lead_lateral is None


def test_lead_vehicle_info_accepts_oncoming_immediate_conflict() -> None:
    expert = _build_expert()
    expert.config.vehicle_oncoming_conflict_gate_enabled = True
    expert.config.oncoming_lead_lateral_max_m = 1.2
    expert.config.oncoming_lead_longitudinal_max_m = 12.0
    expert.config.oncoming_lead_ttc_max_s = 1.5

    nearby = [
        {
            "is_ahead": True,
            "same_direction": False,
            "obb_in_lane": True,
            "distance": 3.0,
            "longitudinal": 3.0,
            "lateral": 0.3,
            "speed_kmh": 10.0,
        }
    ]

    lead_distance, lead_speed, lead_lateral = expert._lead_vehicle_info(nearby, ego_speed_kmh=30.0)

    assert lead_distance == 3.0
    assert lead_speed == 10.0
    assert lead_lateral == 0.3


def test_lead_vehicle_info_rejects_same_direction_adjacent_sat_overlap() -> None:
    expert = _build_expert()
    expert._cached_half_lane = 1.7
    expert.config.vehicle_adjacent_sat_gate_enabled = True
    expert.config.vehicle_adjacent_sat_lateral_margin_m = 0.35
    expert.config.vehicle_adjacent_sat_vertical_max_m = 1.0

    nearby = [
        {
            "is_ahead": True,
            "same_direction": True,
            "obb_in_lane": True,
            "distance": 4.0,
            "longitudinal": 5.0,
            "lateral": 3.2,
            "vertical_sep": 0.2,
            "speed_kmh": 8.0,
        }
    ]

    lead_distance, lead_speed, lead_lateral = expert._lead_vehicle_info(nearby, ego_speed_kmh=20.0)

    assert lead_distance is None
    assert lead_speed is None
    assert lead_lateral is None


def test_lead_vehicle_info_rejects_same_direction_sat_with_large_vertical_separation() -> None:
    expert = _build_expert()
    expert._cached_half_lane = 1.7
    expert.config.vehicle_adjacent_sat_gate_enabled = True
    expert.config.vehicle_adjacent_sat_lateral_margin_m = 0.35
    expert.config.vehicle_adjacent_sat_vertical_max_m = 1.0

    nearby = [
        {
            "is_ahead": True,
            "same_direction": True,
            "obb_in_lane": True,
            "distance": 3.0,
            "longitudinal": 4.0,
            "lateral": 0.3,
            "vertical_sep": 1.4,
            "speed_kmh": 5.0,
        }
    ]

    lead_distance, lead_speed, lead_lateral = expert._lead_vehicle_info(nearby, ego_speed_kmh=10.0)

    assert lead_distance is None
    assert lead_speed is None
    assert lead_lateral is None


def test_lead_vehicle_info_accepts_same_direction_sat_conflict_in_corridor() -> None:
    expert = _build_expert()
    expert._cached_half_lane = 1.7
    expert.config.vehicle_adjacent_sat_gate_enabled = True
    expert.config.vehicle_adjacent_sat_lateral_margin_m = 0.35
    expert.config.vehicle_adjacent_sat_vertical_max_m = 1.0

    nearby = [
        {
            "is_ahead": True,
            "same_direction": True,
            "obb_in_lane": True,
            "distance": 3.2,
            "longitudinal": 4.5,
            "lateral": 0.4,
            "vertical_sep": 0.1,
            "speed_kmh": 7.5,
        }
    ]

    lead_distance, lead_speed, lead_lateral = expert._lead_vehicle_info(nearby, ego_speed_kmh=12.0)

    assert lead_distance == 3.2
    assert lead_speed == 7.5
    assert lead_lateral == 0.4


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


def test_dynamic_half_lane_uses_97pct_lane_half_when_wider_than_ego() -> None:
    expert = _build_expert()
    wp = _Waypoint(0.0, 0.0, road_id=1, lane_id=1)
    wp.lane_width = 3.5
    expert.map = SimpleNamespace(get_waypoint=lambda _loc: wp)
    expert.vehicle = SimpleNamespace(
        bounding_box=SimpleNamespace(extent=SimpleNamespace(y=1.0)),
    )

    half = expert._dynamic_half_lane(_Transform(_Loc(0.0, 0.0)))

    assert round(half, 4) == round(0.97 * (3.5 / 2.0), 4)


def test_dynamic_half_lane_never_goes_below_ego_half_width() -> None:
    expert = _build_expert()
    wp = _Waypoint(0.0, 0.0, road_id=1, lane_id=1)
    wp.lane_width = 1.5
    expert.map = SimpleNamespace(get_waypoint=lambda _loc: wp)
    expert.vehicle = SimpleNamespace(
        bounding_box=SimpleNamespace(extent=SimpleNamespace(y=1.1)),
    )

    half = expert._dynamic_half_lane(_Transform(_Loc(0.0, 0.0)))

    assert half == 1.1


def test_detect_pedestrians_includes_world_and_vertical_fields(monkeypatch: MonkeyPatch) -> None:
    expert = _build_expert()
    monkeypatch.setattr(
        "agents.expert.privileged_expert.carla",
        SimpleNamespace(Vector3D=lambda x, y, z: SimpleNamespace(x=x, y=y, z=z)),
    )

    walker_loc = _Loc(6.0, 1.0, 1.2)
    walker = SimpleNamespace(
        get_transform=lambda: _Transform(walker_loc),
        get_velocity=lambda: SimpleNamespace(x=1.0, y=0.0, z=0.2),
    )

    class _Actors:
        @staticmethod
        def filter(_pattern: str) -> list[object]:
            return [walker]

    expert.world = SimpleNamespace(get_actors=lambda: _Actors())

    nearby = expert._detect_pedestrians(_Transform(_Loc(0.0, 0.0, 0.0)))

    assert len(nearby) == 1
    ped = nearby[0]
    assert ped["world_x"] == 6.0
    assert ped["world_y"] == 1.0
    assert ped["world_z"] == 1.2
    assert ped["vertical_sep"] == 1.2
    assert ped["predicted_world_x"] == 7.0
    assert ped["predicted_world_y"] == 1.0
    assert ped["predicted_world_z"] == 1.4
    assert ped["predicted_vertical_sep"] == 1.4


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


def test_fill_junction_gaps_allows_rpc_retry_after_bezier_failure() -> None:
    expert = _build_expert()
    expert.config.gap_fill_bezier_retry_limit = 1
    expert.config.gap_fill_rpc_retry_limit = 1

    wp_prev = _Waypoint(0.0, 0.0, road_id=10, lane_id=1, yaw=0.0)
    wp_next = _Waypoint(8.0, 0.0, road_id=10, lane_id=1, yaw=0.0)
    wp2 = _Waypoint(2.0, 0.0, road_id=10, lane_id=1, yaw=0.0)
    wp4 = _Waypoint(4.0, 0.0, road_id=10, lane_id=1, yaw=0.0)
    wp7 = _Waypoint(7.0, 0.0, road_id=10, lane_id=1, yaw=0.0)

    wp_prev._nexts = [wp2]
    wp2._nexts = [wp4]
    wp4._nexts = [wp7]
    wp7._nexts = []

    expert._wp_buffer = deque([(wp_prev, RoadOption.LANEFOLLOW), (wp_next, RoadOption.LANEFOLLOW)])

    # First pass: Bezier fails while RPC is temporarily marked exhausted.
    pair_key = (wp_prev.id, wp_next.id)
    expert._gap_fill_rpc_attempts[pair_key] = 1
    expert.map = SimpleNamespace(get_waypoint=lambda _loc: None)
    expert._fill_junction_gaps(None)
    assert pair_key in expert._gap_fill_bezier_tried
    assert pair_key not in expert._gap_fill_rpc_tried
    assert expert._gap_fill_rpcs == 0

    # Second pass: enable RPC retries; pair should still be RPC-eligible.
    expert._gap_fill_rpc_attempts[pair_key] = 0
    expert._fill_junction_gaps(None)
    assert pair_key in expert._gap_fill_rpc_tried
    assert expert._gap_fill_rpcs > 0


def test_fill_junction_gaps_rpc_uses_adjacent_lane_bridge_for_lateral_gap() -> None:
    expert = _build_expert()
    expert.config.gap_fill_bezier_enabled = False
    expert.config.gap_fill_rpc_retry_limit = 1

    wp_prev = _Waypoint(0.0, 0.0, road_id=10, lane_id=1, yaw=0.0)
    wp_next = _Waypoint(8.0, 0.0, road_id=10, lane_id=2, yaw=0.0)

    # Same-lane forward chain is wrong-road and should not be used.
    wp_wrong = _Waypoint(2.0, 4.0, road_id=11, lane_id=1, yaw=0.0)
    wp_prev._nexts = [wp_wrong]
    wp_wrong._nexts = []

    # Adjacent lane bridge (left) plus forward chain should converge.
    wp_left = _Waypoint(0.0, 1.5, road_id=10, lane_id=2, yaw=0.0)
    wp_left2 = _Waypoint(4.0, 1.5, road_id=10, lane_id=2, yaw=0.0)
    wp_left3 = _Waypoint(7.0, 1.5, road_id=10, lane_id=2, yaw=0.0)
    wp_prev._left = wp_left
    wp_left._nexts = [wp_left2]
    wp_left2._nexts = [wp_left3]
    wp_left3._nexts = []

    expert._wp_buffer = deque(
        [
            (wp_prev, RoadOption.LANEFOLLOW),
            (wp_next, RoadOption.CHANGELANELEFT),
        ]
    )

    expert._fill_junction_gaps(None)

    route_wps = [wp for wp, _ in expert._wp_buffer]
    assert wp_left in route_wps
    assert wp_left2 in route_wps
    assert wp_wrong not in route_wps
    assert expert._gap_fill_diverged == 0


def test_fill_junction_gaps_startup_circuit_breaker_blacklists_diverging_pair() -> None:
    expert = _build_expert()
    expert.config.gap_fill_startup_circuit_breaker_enabled = True
    expert.config.gap_fill_bezier_retry_limit = 1
    expert.config.gap_fill_rpc_retry_limit = 2

    wp_prev = _Waypoint(0.0, 0.0, road_id=10, lane_id=1, yaw=0.0)
    wp_next = _Waypoint(8.0, 0.0, road_id=10, lane_id=1, yaw=0.0)
    wp_wrong = _Waypoint(2.0, 2.0, road_id=11, lane_id=1, yaw=0.0)
    wp_prev._nexts = [wp_wrong]
    wp_wrong._nexts = []

    pair_key = (wp_prev.id, wp_next.id)
    expert._wp_buffer = deque([(wp_prev, RoadOption.LANEFOLLOW), (wp_next, RoadOption.LANEFOLLOW)])
    expert.map = SimpleNamespace(get_waypoint=lambda _loc: None)

    expert._fill_junction_gaps(None)

    assert pair_key in expert._gap_fill_pair_blacklist
    first_rpc_count = expert._gap_fill_rpcs

    # Second startup pass should skip this pair entirely.
    expert._fill_junction_gaps(None)
    assert expert._gap_fill_rpcs == first_rpc_count


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


def test_advance_route_arc_watchdog_clears_stalled_arc_buffer() -> None:
    expert = _build_expert()
    expert.config.arc_watchdog_enabled = True
    expert.config.arc_watchdog_stall_min_frames = 1
    expert.config.arc_watchdog_progress_epsilon_m = 0.01

    arc0 = _Waypoint(6.0, 0.0, road_id=10, lane_id=1)
    arc1 = _Waypoint(10.0, 0.0, road_id=10, lane_id=1)
    route0 = _Waypoint(12.0, 0.0, road_id=10, lane_id=1)
    route1 = _Waypoint(14.0, 0.0, road_id=10, lane_id=1)

    expert._arc_buffer = deque(
        [
            (arc0, RoadOption.LANEFOLLOW),
            (arc1, RoadOption.LANEFOLLOW),
        ]
    )
    expert._wp_buffer = deque(
        [
            (route0, RoadOption.LANEFOLLOW),
            (route1, RoadOption.LANEFOLLOW),
        ]
    )
    setattr(expert, "_fill_buffer", lambda _transform: None)

    ego_transform = _Transform(_Loc(0.0, 0.0), yaw=0.0)
    expert._advance_route(ego_transform)
    assert len(expert._arc_buffer) == 2

    expert._advance_route(ego_transform)
    assert len(expert._arc_buffer) == 0
    assert expert._arc_watchdog_trigger_count == 1


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


def test_route_legality_continuity_guard_keeps_large_skip_gap() -> None:
    expert = _build_expert()
    expert.config.planner_wrong_way_max_yaw_diff_deg = 80.0
    expert.config.planner_legality_max_skip_gap_m = 5.0

    wp0 = _Waypoint(0.0, 0.0, road_id=1, lane_id=1, yaw=0.0)
    wrong_way_far = _Waypoint(-20.0, 0.0, road_id=1, lane_id=1, yaw=180.0)
    wp_ok = _Waypoint(6.0, 0.0, road_id=1, lane_id=1, yaw=0.0)

    filtered = expert._enforce_route_legality(
        [
            (wp0, RoadOption.LANEFOLLOW),
            (wrong_way_far, RoadOption.LANEFOLLOW),
            (wp_ok, RoadOption.LANEFOLLOW),
        ]
    )

    assert filtered[0][0] is wp0
    assert filtered[1][0] is wrong_way_far
    assert expert._legality_wrong_way_filtered >= 1
    assert expert._legality_continuity_kept >= 1


def test_route_tail_hardening_trims_terminal_severe_gap_suffix() -> None:
    expert = _build_expert()
    expert.config.planner_tail_gap_hardening_enabled = True
    expert.config.planner_tail_severe_gap_m = 20.0

    wp0 = _Waypoint(0.0, 0.0, road_id=1, lane_id=1, yaw=0.0)
    wp1 = _Waypoint(1.0, 0.0, road_id=1, lane_id=1, yaw=0.0)
    wp2 = _Waypoint(2.0, 0.0, road_id=1, lane_id=1, yaw=0.0)
    wp3 = _Waypoint(35.0, 0.0, road_id=1, lane_id=1, yaw=0.0)
    wp4 = _Waypoint(66.0, 0.0, road_id=1, lane_id=1, yaw=0.0)

    hardened = expert._harden_route_tail_severe_gaps(
        [
            (wp0, RoadOption.LANEFOLLOW),
            (wp1, RoadOption.LANEFOLLOW),
            (wp2, RoadOption.LANEFOLLOW),
            (wp3, RoadOption.LANEFOLLOW),
            (wp4, RoadOption.LANEFOLLOW),
        ]
    )

    assert [wp for wp, _ in hardened] == [wp0, wp1, wp2]


def test_route_tail_hardening_does_not_trim_non_terminal_gap() -> None:
    expert = _build_expert()
    expert.config.planner_tail_gap_hardening_enabled = True
    expert.config.planner_tail_severe_gap_m = 20.0

    wp0 = _Waypoint(0.0, 0.0, road_id=1, lane_id=1, yaw=0.0)
    wp1 = _Waypoint(30.0, 0.0, road_id=1, lane_id=1, yaw=0.0)
    wp2 = _Waypoint(31.0, 0.0, road_id=1, lane_id=1, yaw=0.0)
    wp3 = _Waypoint(32.0, 0.0, road_id=1, lane_id=1, yaw=0.0)

    route = [
        (wp0, RoadOption.LANEFOLLOW),
        (wp1, RoadOption.LANEFOLLOW),
        (wp2, RoadOption.LANEFOLLOW),
        (wp3, RoadOption.LANEFOLLOW),
    ]
    hardened = expert._harden_route_tail_severe_gaps(route)

    assert [wp for wp, _ in hardened] == [wp0, wp1, wp2, wp3]


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


def test_policy_steer_consistency_gate_retargets_from_arc_to_route() -> None:
    expert = _build_expert()
    expert.config.planner_policy_steer_consistency_gate_enabled = True
    expert._planner_policy_reason = PlannerPolicyReason.PRE_JUNCTION_LANE_CHANGE_BLOCKED
    expert._planner_policy_exception = PlannerPolicyException.NONE

    arc_wp = _Waypoint(5.0, 1.5, road_id=1, lane_id=2)
    route_wp = _Waypoint(6.0, 0.0, road_id=1, lane_id=1)
    expert._arc_buffer = deque([(arc_wp, RoadOption.LANEFOLLOW)])
    expert._wp_buffer = deque([(route_wp, RoadOption.LANEFOLLOW)])
    expert._steer_from_arc = 1

    target, blocked = expert._enforce_policy_steer_consistency(
        target_wp=arc_wp,
        nav_command_route=RoadOption.CHANGELANELEFT,
        nav_label=RoadOption.LANEFOLLOW,
    )

    assert blocked is True
    assert target is route_wp
    assert expert._steer_from_arc == 0
    assert expert._planner_policy_exception == PlannerPolicyException.BLOCKED_LANE


def test_policy_steer_consistency_gate_can_be_disabled() -> None:
    expert = _build_expert()
    expert.config.planner_policy_steer_consistency_gate_enabled = False
    expert._planner_policy_reason = PlannerPolicyReason.CONSECUTIVE_LANE_CHANGE_BLOCKED

    arc_wp = _Waypoint(5.0, 1.5, road_id=1, lane_id=2)
    route_wp = _Waypoint(6.0, 0.0, road_id=1, lane_id=1)
    expert._arc_buffer = deque([(arc_wp, RoadOption.LANEFOLLOW)])
    expert._wp_buffer = deque([(route_wp, RoadOption.LANEFOLLOW)])
    expert._steer_from_arc = 1

    target, blocked = expert._enforce_policy_steer_consistency(
        target_wp=arc_wp,
        nav_command_route=RoadOption.CHANGELANERIGHT,
        nav_label=RoadOption.LANEFOLLOW,
    )

    assert blocked is True
    assert target is arc_wp
    assert expert._steer_from_arc == 1


def test_turn_lane_legality_blocks_unsupported_turn_command() -> None:
    expert = _build_expert()
    expert.config.planner_turn_lane_validation_enabled = True

    ego_wp = _Waypoint(0.0, 0.0, road_id=1, lane_id=1, yaw=0.0)
    approach_wp = _Waypoint(8.0, 0.0, road_id=1, lane_id=1, yaw=0.0)
    left_turn_wp = _Waypoint(12.0, 0.0, road_id=2, lane_id=1, yaw=45.0)
    left_turn_wp.is_junction = True
    expert._wp_buffer = deque(
        [
            (approach_wp, RoadOption.LANEFOLLOW),
            (left_turn_wp, RoadOption.LEFT),
        ]
    )

    label = expert._apply_lane_change_policy(RoadOption.RIGHT, ego_wp, _Loc(0.0, 0.0))

    assert label == RoadOption.LANEFOLLOW
    assert expert._frame_turn_lane_legality_violation == 1
    assert expert._planner_policy_reason == PlannerPolicyReason.TURN_LANE_LEGALITY_BLOCKED


def test_turn_lane_legality_allows_supported_turn_command() -> None:
    expert = _build_expert()
    expert.config.planner_turn_lane_validation_enabled = True

    ego_wp = _Waypoint(0.0, 0.0, road_id=1, lane_id=1, yaw=0.0)
    approach_wp = _Waypoint(8.0, 0.0, road_id=1, lane_id=1, yaw=0.0)
    left_turn_wp = _Waypoint(12.0, 0.0, road_id=2, lane_id=1, yaw=45.0)
    left_turn_wp.is_junction = True
    expert._wp_buffer = deque(
        [
            (approach_wp, RoadOption.LANEFOLLOW),
            (left_turn_wp, RoadOption.LEFT),
        ]
    )

    label = expert._apply_lane_change_policy(RoadOption.LEFT, ego_wp, _Loc(0.0, 0.0))

    assert label == RoadOption.LEFT
    assert expert._frame_turn_lane_legality_violation == 0


def test_roundabout_lane_change_cooldown_blocks_repeated_change() -> None:
    expert = _build_expert()
    expert.config.planner_roundabout_commitment_enabled = True
    expert.config.planner_roundabout_lane_change_cooldown_m = 35.0
    expert.config.planner_roundabout_context_window_m = 40.0
    expert.config.planner_lane_change_freeze_distance_m = 0.0
    expert.config.planner_lane_change_consecutive_window_m = 0.0
    expert._policy_distance_traveled = 120.0
    expert._policy_last_lane_change_distance = 0.0
    expert._policy_last_roundabout_lane_change_distance = 100.0

    ego_wp = _Waypoint(0.0, 0.0, road_id=1, lane_id=1, yaw=0.0)
    ego_wp.is_junction = True
    left_next = _Waypoint(2.0, 2.0, road_id=2, lane_id=1, yaw=45.0)
    right_next = _Waypoint(2.0, -2.0, road_id=3, lane_id=1, yaw=-45.0)
    ego_wp._nexts = [left_next, right_next]

    junc_wp = _Waypoint(8.0, 0.0, road_id=1, lane_id=1)
    junc_wp.is_junction = True
    expert._wp_buffer = deque([(junc_wp, RoadOption.LANEFOLLOW)])

    label = expert._apply_lane_change_policy(RoadOption.CHANGELANELEFT, ego_wp, _Loc(0.0, 0.0))

    assert label == RoadOption.LANEFOLLOW
    assert expert._frame_roundabout_lane_change_violation == 1
    assert expert._planner_policy_reason == PlannerPolicyReason.ROUNDABOUT_LANE_CHANGE_BLOCKED


def test_roundabout_lane_change_allowed_after_cooldown_distance() -> None:
    expert = _build_expert()
    expert.config.planner_roundabout_commitment_enabled = True
    expert.config.planner_roundabout_lane_change_cooldown_m = 35.0
    expert.config.planner_roundabout_context_window_m = 40.0
    expert.config.planner_lane_change_freeze_distance_m = 0.0
    expert.config.planner_lane_change_consecutive_window_m = 0.0
    expert._policy_distance_traveled = 180.0
    expert._policy_last_lane_change_distance = 0.0
    expert._policy_last_roundabout_lane_change_distance = 100.0

    ego_wp = _Waypoint(0.0, 0.0, road_id=1, lane_id=1, yaw=0.0)
    ego_wp.is_junction = True
    left_next = _Waypoint(2.0, 2.0, road_id=2, lane_id=1, yaw=45.0)
    right_next = _Waypoint(2.0, -2.0, road_id=3, lane_id=1, yaw=-45.0)
    ego_wp._nexts = [left_next, right_next]

    junc_wp = _Waypoint(8.0, 0.0, road_id=1, lane_id=1)
    junc_wp.is_junction = True
    expert._wp_buffer = deque([(junc_wp, RoadOption.LANEFOLLOW)])

    label = expert._apply_lane_change_policy(RoadOption.CHANGELANERIGHT, ego_wp, _Loc(0.0, 0.0))

    assert label == RoadOption.CHANGELANERIGHT
    assert expert._frame_roundabout_lane_change_violation == 0


def test_roundabout_context_uses_topology_cycle_and_route_mix() -> None:
    expert = _build_expert()
    expert.config.planner_roundabout_commitment_enabled = True
    expert.config.planner_roundabout_lane_change_cooldown_m = 35.0
    expert.config.planner_roundabout_context_window_m = 40.0
    expert.config.planner_lane_change_freeze_distance_m = 0.0
    expert.config.planner_lane_change_consecutive_window_m = 0.0
    expert._policy_distance_traveled = 120.0
    expert._policy_last_lane_change_distance = 0.0
    expert._policy_last_roundabout_lane_change_distance = 100.0
    expert._topology_cycle_road_ids = {5}

    ego_wp = _Waypoint(0.0, 0.0, road_id=5, lane_id=1, yaw=0.0)
    ego_wp.is_junction = False
    wp_l = _Waypoint(6.0, 0.0, road_id=5, lane_id=1, yaw=0.0)
    wp_r = _Waypoint(12.0, 0.0, road_id=5, lane_id=1, yaw=0.0)
    expert._wp_buffer = deque(
        [
            (wp_l, RoadOption.CHANGELANELEFT),
            (wp_r, RoadOption.CHANGELANERIGHT),
        ]
    )

    label = expert._apply_lane_change_policy(RoadOption.CHANGELANELEFT, ego_wp, _Loc(0.0, 0.0))

    assert label == RoadOption.LANEFOLLOW
    assert expert._frame_roundabout_lane_change_violation == 1


def test_mandatory_exit_lane_change_requires_turn_within_window() -> None:
    expert = _build_expert()
    expert.config.planner_mandatory_exit_window_m = 30.0

    lc_wp = _Waypoint(10.0, 0.0, road_id=1, lane_id=1)
    turn_wp = _Waypoint(20.0, 0.0, road_id=2, lane_id=1)
    expert._wp_buffer = deque(
        [
            (lc_wp, RoadOption.CHANGELANELEFT),
            (turn_wp, RoadOption.RIGHT),
        ]
    )

    assert expert._is_mandatory_exit_lane_change(_Loc(0.0, 0.0)) is True


def test_mandatory_exit_lane_change_ignores_far_turn_outside_window() -> None:
    expert = _build_expert()
    expert.config.planner_mandatory_exit_window_m = 30.0

    lc_wp = _Waypoint(10.0, 0.0, road_id=1, lane_id=1)
    far_turn_wp = _Waypoint(70.0, 0.0, road_id=2, lane_id=1)
    expert._wp_buffer = deque(
        [
            (lc_wp, RoadOption.CHANGELANELEFT),
            (far_turn_wp, RoadOption.RIGHT),
        ]
    )

    assert expert._is_mandatory_exit_lane_change(_Loc(0.0, 0.0)) is False


def test_route_watchdog_triggers_emergency_stop_after_streak() -> None:
    expert = _build_expert()
    expert.config.route_watchdog_enabled = True
    expert.config.route_watchdog_wp_buffer_head_dist_m = 30.0
    expert.config.route_watchdog_min_corridor_segments = 1
    expert.config.route_watchdog_min_frames = 2
    expert._n_corridor_segments = 1

    target, mode = expert._apply_route_continuity_watchdog(
        buf_head_dist=45.0,
        target_speed=20.0,
        mode=DrivingMode.FOLLOW_LANE,
        speed_kmh=20.0,
        route_completion=0.2,
        heading_error_deg=0.0,
        tl_state=None,
        tl_distance=None,
    )
    assert mode == DrivingMode.FOLLOW_LANE
    assert target == 20.0
    assert expert._frame_route_watchdog_triggered == 0

    target, mode = expert._apply_route_continuity_watchdog(
        buf_head_dist=44.0,
        target_speed=18.0,
        mode=DrivingMode.FOLLOW_LANE,
        speed_kmh=18.0,
        route_completion=0.2,
        heading_error_deg=0.0,
        tl_state=None,
        tl_distance=None,
    )
    assert mode == DrivingMode.EMERGENCY_STOP
    assert target == 0.0
    assert expert._frame_route_watchdog_triggered == 1
    assert expert._planner_policy_reason == PlannerPolicyReason.ROUTE_CONTINUITY_WATCHDOG


def test_route_watchdog_resets_streak_when_corridor_recovers() -> None:
    expert = _build_expert()
    expert.config.route_watchdog_enabled = True
    expert.config.route_watchdog_wp_buffer_head_dist_m = 30.0
    expert.config.route_watchdog_min_corridor_segments = 1
    expert.config.route_watchdog_min_frames = 2

    expert._n_corridor_segments = 1
    _ = expert._apply_route_continuity_watchdog(
        buf_head_dist=40.0,
        target_speed=25.0,
        mode=DrivingMode.FOLLOW_LANE,
        speed_kmh=25.0,
        route_completion=0.2,
        heading_error_deg=0.0,
        tl_state=None,
        tl_distance=None,
    )
    assert expert._route_watchdog_consecutive_frames == 1

    expert._n_corridor_segments = 2
    target, mode = expert._apply_route_continuity_watchdog(
        buf_head_dist=8.0,
        target_speed=25.0,
        mode=DrivingMode.FOLLOW_LANE,
        speed_kmh=25.0,
        route_completion=0.21,
        heading_error_deg=0.0,
        tl_state=None,
        tl_distance=None,
    )
    assert expert._route_watchdog_consecutive_frames == 0
    assert expert._frame_route_watchdog_triggered == 0
    assert mode == DrivingMode.FOLLOW_LANE
    assert target == 25.0


def test_route_watchdog_stall_signature_triggers_without_corridor_collapse() -> None:
    expert = _build_expert()
    expert.config.route_watchdog_enabled = True
    expert.config.route_watchdog_min_corridor_segments = 1
    expert.config.route_watchdog_wp_buffer_head_dist_m = 30.0
    expert.config.route_watchdog_stall_min_frames = 2
    expert.config.route_watchdog_progress_epsilon = 1e-4
    expert.config.route_watchdog_min_moving_speed_kmh = 5.0
    expert.config.route_watchdog_heading_mismatch_deg = 60.0
    expert._n_corridor_segments = 20

    target, mode = expert._apply_route_continuity_watchdog(
        buf_head_dist=4.0,
        target_speed=24.0,
        mode=DrivingMode.FOLLOW_LANE,
        speed_kmh=20.0,
        route_completion=0.45,
        heading_error_deg=95.0,
        tl_state=None,
        tl_distance=None,
    )
    assert mode == DrivingMode.FOLLOW_LANE
    assert target == 24.0

    target, mode = expert._apply_route_continuity_watchdog(
        buf_head_dist=4.0,
        target_speed=24.0,
        mode=DrivingMode.FOLLOW_LANE,
        speed_kmh=20.0,
        route_completion=0.45,
        heading_error_deg=100.0,
        tl_state=None,
        tl_distance=None,
    )
    assert mode == DrivingMode.EMERGENCY_STOP
    assert target == 0.0
    assert expert._frame_route_watchdog_triggered == 1


def test_route_watchdog_tl_hold_exempts_stall_signature(monkeypatch: MonkeyPatch) -> None:
    expert = _build_expert()
    expert.config.route_watchdog_enabled = True
    expert.config.route_watchdog_stall_min_frames = 1
    expert.config.route_watchdog_progress_epsilon = 1e-4
    expert.config.route_watchdog_min_moving_speed_kmh = 5.0
    expert.config.route_watchdog_heading_mismatch_deg = 60.0
    expert.config.route_watchdog_tl_hold_exempt_speed_kmh = 8.0
    expert._n_corridor_segments = 20

    fake_tl = SimpleNamespace(Green=1, Yellow=2, Red=3)
    monkeypatch.setattr(
        "agents.expert.privileged_expert.carla", SimpleNamespace(TrafficLightState=fake_tl)
    )

    target, mode = expert._apply_route_continuity_watchdog(
        buf_head_dist=5.0,
        target_speed=12.0,
        mode=DrivingMode.TRAFFIC_LIGHT_STOP,
        speed_kmh=6.5,
        route_completion=0.2,
        heading_error_deg=95.0,
        tl_state=fake_tl.Red,
        tl_distance=10.0,
    )
    assert mode == DrivingMode.TRAFFIC_LIGHT_STOP
    assert target == 12.0
    assert expert._frame_route_watchdog_triggered == 0


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
