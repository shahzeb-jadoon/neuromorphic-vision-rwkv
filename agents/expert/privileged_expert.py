"""Privileged Expert — Full custom expert agent for CARLA.

Uses privileged simulator access (ground-truth positions, traffic light
states, actor locations) to produce optimal driving demonstrations for
imitation learning.  Written from scratch — no dependency on CARLA's
BehaviorAgent or BasicAgent classes.

Design principles:
    1. Smooth, learnable behaviour (damped PID, look-ahead steering)
    2. Full privileged info exported per frame for post-hoc analysis
    3. Conservative safety margins (better to stop than crash)
    4. Explicit driving-mode state for interpretability

Depends only on:
    - ``carla`` Python package (``pip install carla==0.9.16``)
    - ``agents.navigation.global_route_planner.GlobalRoutePlanner``
      (already vendored)
    - ``agents.navigation.local_planner.RoadOption`` enum (already vendored)

Reference implementations studied:
    - TransFuser autopilot.py (Chitta et al., 2022)
    - Learning by Cheating expert (Chen et al., 2020)

Copyright (c) 2025–2026, Shahzeb Jadoon.  All Rights Reserved.
See LICENSE in the project root for terms.  The ``agents/navigation/``
module is vendored from CARLA and retains its original MIT license.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum, auto
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import carla

    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False

try:
    from agents.navigation.local_planner import RoadOption
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except (ImportError, ModuleNotFoundError):
    pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ExpertConfig:
    """Tunable parameters for the privileged expert.

    All speeds are in **km/h** unless stated otherwise.
    All distances are in **metres**.
    """

    # Target cruising speed (used as fallback when road has no speed-limit sign)
    target_speed: float = 30.0
    max_speed: float = 50.0
    # speed_limit_margin: float = 0.90  # drive at 90% of posted limit

    # --- PID: longitudinal (speed) ---
    # NOTE: gains tuned for RAW km/h error (P4-C).  Previous gains
    # (0.8 / 0.05 / 0.01) were tuned for normalised error dividing
    # by target_speed, which capped proportional authority and
    # prevented the car from reaching cruising speed.
    speed_kp: float = 0.07
    speed_ki: float = 0.005
    speed_kd: float = 0.002

    # --- PID: lateral (steering) — straight-line defaults ---
    steer_kp: float = 1.2
    steer_ki: float = 0.05
    steer_kd: float = 0.15

    # --- PID gains for turns (blended smoothly via curvature) ---
    # Slightly higher than straight-line gains for tighter control.
    turn_speed_kp: float = 0.08
    turn_speed_ki: float = 0.008
    turn_speed_kd: float = 0.003
    turn_steer_kp: float = 1.5
    turn_steer_ki: float = 0.05
    turn_steer_kd: float = 0.3

    # --- Waypoint following ---
    min_follow_distance: float = 3.0  # metres — pop threshold
    target_follow_distance: float = 5.0  # metres — desired look-ahead
    turn_follow_distance: float = 2.0  # metres — shorter look-ahead in turns
    max_follow_distance: float = 20.0  # metres — upper bound
    waypoint_buffer_size: int = 60

    # --- Safety thresholds ---
    traffic_light_brake_distance: float = 80.0  # metres — start braking
    traffic_light_stop_distance: float = 0.5  # metres — full stop
    traffic_light_route_match_distance: float = 6.0  # metres — AWP-to-route tolerance for Tier 3
    traffic_light_route_match_yaw_deg: float = 40.0  # degrees — AWP/route heading alignment gate
    traffic_light_stored_stop_max_yaw_diff_deg: float = (
        60.0  # degrees — reject stale stored stop-lines with large heading mismatch
    )
    traffic_light_stored_stop_lateral_margin: float = (
        1.0  # metres — extra lateral tolerance beyond half-lane width
    )
    tl_comfort_decel: float = 1.5  # m/s² — comfortable deceleration for TL
    tl_commit_decel: float = 2.8  # m/s² — deceleration threshold for yellow commitment
    tl_commit_max_stop_distance: float = (
        25.0  # m — max stop-line distance to allow yellow commitment
    )
    tl_clearance_margin: float = 8.0  # m — must clear stop line by this to commit on yellow
    tl_half_junction_clearance_factor: float = (
        0.5  # require at least half-junction clearance on yellow commit
    )
    turn_comfort_decel: float = 1.5  # m/s² — gentler braking for turn approach
    vehicle_hazard_distance: float = 25.0  # metres — start slowing (min)
    vehicle_emergency_distance: float = 5.0  # metres — emergency (min)
    vehicle_reaction_time: float = 2.0  # seconds — scale hazard dist with speed
    vehicle_emergency_time: float = 1.2  # seconds — scale emergency with speed
    vehicle_comfort_decel: float = 3.0  # m/s² — comfortable braking for following
    vehicle_allow_lateral_fallback: bool = False  # if False, only SAT (obb_in_lane) can select lead
    vehicle_oncoming_conflict_gate_enabled: bool = True
    oncoming_lead_lateral_max_m: float = 1.2
    oncoming_lead_longitudinal_max_m: float = 12.0
    oncoming_lead_ttc_max_s: float = 1.5
    oncoming_lead_min_closing_speed_kmh: float = 5.0
    pedestrian_hazard_distance: float = 15.0
    pedestrian_emergency_distance: float = 5.0
    lane_half_width: float = 2.5  # metres — lateral fallback (used only when ego_wp unavailable)
    corridor_vehicle_half_width_margin: float = (
        0.2  # metres — extra lateral margin beyond ego half-width
    )
    corridor_margin_max: float = 0.0  # metres — max SAT corridor bleed into adjacent lane
    corridor_margin_min: float = 0.0  # metres — min SAT corridor bleed into adjacent lane
    corridor_max_vertical_separation: float = 2.5  # metres — reject SAT hits on stacked roads
    pedestrian_lane_half_width: float = 3.0  # metres — wider lateral band for pedestrians

    # --- Stop signs ---
    stop_sign_brake_distance: float = 40.0  # metres — start braking
    stop_sign_stop_distance: float = 4.0  # metres — full stop
    stop_sign_comfort_decel: float = 1.5  # m/s² — comfortable deceleration
    stop_sign_creep_speed: float = 8.0  # km/h — Phase 1 floor; car creeps until trigger volume
    stop_sign_wait_ticks: int = 40  # ~2 s at 20 Hz — minimum pause
    stop_sign_route_proximity: float = 2.5  # metres — 2D horizontal TV-center-to-route threshold
    stop_sign_commit_radius: float = 4.0  # metres — 2D ego-to-TV-center triggers Phase 2 full stop

    # --- Yield signs ---
    yield_scan_distance: float = 35.0  # metres — start checking for yield ahead
    yield_route_proximity: float = 2.5  # metres — max TV-center-to-route 2D distance
    yield_slow_speed: float = 12.0  # km/h — creep speed while checking for conflicts
    yield_comfort_decel: float = 1.5  # m/s² — comfortable deceleration for yield
    yield_conflict_distance: float = 25.0  # metres — scan radius for cross-traffic
    yield_conflict_angle_deg: float = 30.0  # degrees — heading divergence to count as cross-traffic
    yield_wait_ticks: int = 20  # ~1 s at 20 Hz — hold after conflict clears before proceeding
    yield_clear_behind_m: float = 3.0  # metres — sign behind ego by this much → clear

    # --- Turn behaviour ---
    turn_speed_factor: float = 0.45  # target speed multiplier for turns

    # --- PIDs ---
    turn_anticipation_distance: float = 30.0  # start slowing this far ahead

    # --- Speed limits / curvature ---
    curvature_speed_factor: float = 0.7  # reduce by 30% on curves
    curvature_threshold_deg: float = 20.0  # angle change between successive wps

    # --- Cross-track correction (DISABLED — unbounded value causes
    #     steering spikes on waypoint transitions; see commit history) ---
    # cross_track_gain: float = 0.3

    # --- Noise injection (for training data diversity) ---
    steering_noise_sigma: float = 1e-3  # Gaussian noise std on steering

    # --- Rate limiter ---
    max_target_drop_per_frame: float = 2.0  # km/h — prevents target jumps

    # --- Junction gap filling ---
    # Controls how _fill_junction_gaps() fills large waypoint gaps at
    # junctions by querying wp.next() from the CARLA server.
    #
    # The original code had an unbounded while loop that diverged at
    # junctions (wp_prev and wp_next on different roads), issuing
    # thousands of CARLA server RPCs per frame and leaking ~19 GB of
    # server-side memory.  These parameters bound that loop.
    #
    # gap_fill_max_factor: max iterations = gap_m / fill_step * factor.
    #   Default 3.0 means a 10m gap allows at most 15 iterations.
    #   Set to 0 to disable gap filling entirely.
    #   Set to float('inf') for unbounded (NOT RECOMMENDED — OOM risk).
    # gap_fill_diverge_factor: break if distance from origin > gap * factor.
    #   Default 2.0 — if we've traveled twice the original gap distance,
    #   we're following the wrong road.
    # gap_fill_rpc_budget: cumulative wp.next() RPC limit per episode.
    #   Default 50_000.  When exceeded, gap filling is disabled for the
    #   rest of the episode to cap server-side memory.  A warning is logged
    #   when the budget is exhausted.
    gap_fill_max_factor: float = 3.0
    gap_fill_diverge_factor: float = 2.0
    gap_fill_rpc_budget: int = 50_000
    gap_fill_bezier_enabled: bool = True
    gap_fill_bezier_retry_limit: int = 2
    gap_fill_rpc_retry_limit: int = 2
    gap_fill_bezier_max_pair_index: int = 2
    gap_fill_bezier_step: float = 0.5  # metres
    gap_fill_bezier_tangent_factor: float = 0.35  # control-handle length = gap * factor
    gap_fill_bezier_max_project_error: float = 2.5  # metres

    # --- Route densification (long-term robust path smoothing) ---
    # Preprocess sparse route transitions once in set_route() so runtime
    # control does not depend on per-frame gap-repair timing.
    route_densify_enabled: bool = True
    route_densify_gap_threshold: float = 4.0  # metres
    route_densify_step: float = 1.0  # metres
    route_densify_max_factor: float = 3.0
    route_densify_diverge_factor: float = 2.0
    route_densify_projection_fallback: bool = True

    # --- Waypoint planner policy scaffolding ---
    planner_policy_enabled: bool = True
    planner_legality_filter_enabled: bool = True
    planner_wrong_way_max_yaw_diff_deg: float = 110.0
    planner_allow_lane_id_sign_flip: bool = False
    planner_turn_mismatch_threshold_deg: float = 20.0
    planner_turn_lane_validation_enabled: bool = True
    planner_lane_change_freeze_distance_m: float = 30.0
    planner_lane_change_consecutive_window_m: float = 30.0
    planner_mandatory_exit_window_m: float = 30.0
    planner_exit_lane_leave_distance_m: float = 30.0
    planner_legality_max_skip_gap_m: float = 30.0
    route_watchdog_enabled: bool = True
    route_watchdog_wp_buffer_head_dist_m: float = 30.0
    route_watchdog_min_corridor_segments: int = 1
    route_watchdog_min_frames: int = 3
    route_watchdog_stall_min_frames: int = 80
    route_watchdog_progress_epsilon: float = 1e-4
    route_watchdog_min_moving_speed_kmh: float = 5.0
    route_watchdog_heading_mismatch_deg: float = 75.0
    route_watchdog_tl_hold_exempt_speed_kmh: float = 8.0
    planner_roundabout_commitment_enabled: bool = True
    planner_roundabout_context_window_m: float = 35.0
    planner_roundabout_lane_change_cooldown_m: float = 35.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class DrivingMode(IntEnum):
    """Human-readable driving mode for logging."""

    FOLLOW_LANE = auto()
    APPROACHING_TURN = auto()
    TURNING = auto()
    TRAFFIC_LIGHT_STOP = auto()
    VEHICLE_FOLLOWING = auto()
    PEDESTRIAN_STOP = auto()
    EMERGENCY_STOP = auto()
    STOP_SIGN_STOP = auto()
    YIELD_STOP = auto()


class PlannerPolicyReason(IntEnum):
    """Per-frame planner policy reason code for diagnostics and analytics."""

    NONE = 0
    LEGALITY_FILTERED = auto()
    PRE_JUNCTION_LANE_CHANGE_BLOCKED = auto()
    CONSECUTIVE_LANE_CHANGE_BLOCKED = auto()
    STRAIGHT_EXIT_LANE_WARNING = auto()
    TURN_LANE_LEGALITY_BLOCKED = auto()
    ROUNDABOUT_LANE_CHANGE_BLOCKED = auto()
    ROUTE_CONTINUITY_WATCHDOG = auto()


class PlannerPolicyException(IntEnum):
    """Per-frame planner policy exception code for explicit audit trails."""

    NONE = 0
    MANDATORY_EXIT = auto()
    BLOCKED_LANE = auto()


@dataclass
class FrameInfo:
    """Privileged information exported for every frame.

    Stored in HDF5 for post-hoc analysis, debugging, and potential
    future use as auxiliary training signals.
    """

    # Navigation
    nav_command: int  # RoadOption value
    route_completion: float  # 0.0 → 1.0

    # Targets
    target_speed: float  # km/h
    current_speed: float  # km/h
    driving_mode: int  # DrivingMode value

    # Ego state (world frame)
    position: Tuple[float, float, float]  # (x, y, z) metres
    rotation: Tuple[float, float, float]  # (pitch, yaw, roll) degrees
    velocity: Tuple[float, float, float]  # (vx, vy, vz) m/s

    # Traffic
    traffic_light_state: int  # 0=None, 1=Green, 2=Yellow, 3=Red
    traffic_light_distance: float  # metres (−1 if none)
    traffic_light_actor_id: int = -1  # actor id of matched TL (−1 if none)
    traffic_light_stop_line_position: Tuple[float, float, float] = (
        -1.0,
        -1.0,
        -1.0,
    )  # matched stop-line xyz (−1s if unknown)
    stop_sign_distance: float = -1.0  # metres (−1 if none)
    yield_sign_distance: float = -1.0  # metres (−1 if none)

    # Nearby actors (serialisable lists of dicts)
    nearby_vehicles: List[Dict[str, Any]] = field(default_factory=list)
    nearby_pedestrians: List[Dict[str, Any]] = field(default_factory=list)

    # Corridor diagnostics (Option D) — lightweight per-frame metrics
    # for post-hoc verification that the SAT corridor is healthy.
    n_corridor_segments: int = 0  # segments built by _build_route_corridor
    wp_buffer_size: int = 0  # current waypoint buffer length
    wp_buffer_head_dist: float = -1.0  # distance (m) from ego to buffer[0]
    sat_detections: int = 0  # vehicles with obb_in_lane=True
    lateral_detections: int = 0  # vehicles that lateral check would catch

    # Junction gap-fill RPC tracking — cumulative wp.next() calls.
    # Collectors can monitor this to decide when to end an episode.
    gap_fill_rpcs: int = 0
    gap_fill_diverged: int = 0  # cumulative gaps that diverged (discarded)

    # Waypoint planner policy diagnostics.
    planner_policy_reason: int = 0
    planner_policy_exception: int = 0
    dist_to_next_junction: float = -1.0
    pre_junction_lane_change_violation: int = 0
    consecutive_lane_change_violation: int = 0
    straight_exit_lane_violation: int = 0
    turn_lane_legality_violation: int = 0
    roundabout_lane_change_violation: int = 0
    legality_wrong_way_filtered: int = 0
    legality_do_not_enter_filtered: int = 0
    legality_turn_mismatch_filtered: int = 0
    legality_continuity_kept: int = 0
    route_watchdog_triggered: int = 0


# ---------------------------------------------------------------------------
# PID controller
# ---------------------------------------------------------------------------


class PIDController:
    """Discrete PID with anti-windup clamping."""

    def __init__(self, kp: float, ki: float, kd: float, integral_limit: float = 10.0) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral = 0.0
        self._prev_error = 0.0
        self._limit = integral_limit

    def step(self, error: float, dt: float = 0.05) -> float:
        """Compute PID output for a single time-step."""
        self._integral = max(-self._limit, min(self._limit, self._integral + error * dt))
        derivative = (error - self._prev_error) / max(dt, 1e-6)
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0

    def reset_integral(self) -> None:
        """Clear accumulated integral without resetting prev_error.

        Use this instead of ``reset()`` when transitioning between
        modes to avoid derivative kicks caused by zeroing prev_error
        while the actual error is large.
        """
        self._integral = 0.0


# ---------------------------------------------------------------------------
# Main expert class
# ---------------------------------------------------------------------------


class PrivilegedExpert:
    """Full custom expert agent for CARLA data collection.

    Produces smooth, predictable driving demonstrations by combining:
        - A* route from ``GlobalRoutePlanner``
        - PID lateral / longitudinal controllers
        - Privileged hazard detection (traffic lights, vehicles, pedestrians)

    Usage::

        expert = PrivilegedExpert(vehicle, world, route_planner)
        expert.set_destination(start_loc, end_loc)

        while not expert.done():
            world.tick()
            control, info = expert.run_step()
            vehicle.apply_control(control)
    """

    def __init__(
        self,
        vehicle: "carla.Vehicle",
        world: "carla.World",
        route_planner: GlobalRoutePlanner,
        config: ExpertConfig | None = None,
    ) -> None:
        if not CARLA_AVAILABLE:
            raise RuntimeError("carla package not installed")

        self.vehicle = vehicle
        self.world = world
        self.map = world.get_map()
        self.route_planner = route_planner
        self.config = config or ExpertConfig()

        # Cache ego vehicle half-length so we can offset stop distances
        # to account for the bumper extending ahead of the centre of mass.
        self._ego_half_length: float = self.vehicle.bounding_box.extent.x

        # Controllers
        self.speed_pid = PIDController(
            self.config.speed_kp,
            self.config.speed_ki,
            self.config.speed_kd,
            integral_limit=40.0,
        )
        self.steer_pid = PIDController(
            self.config.steer_kp,
            self.config.steer_ki,
            self.config.steer_kd,
        )

        # Route state
        self._route: List[Tuple["carla.Waypoint", RoadOption]] = []
        self._route_length: int = 0
        self._route_index: int = 0
        self._wp_buffer: deque = deque(maxlen=self.config.waypoint_buffer_size)

        # Driving state
        self.mode = DrivingMode.FOLLOW_LANE
        self._tl_waiting: int | None = None  # actor-id of TL we're watching/waiting for
        self._tl_stop_loc = None  # stored carla.Location of the stop-line waypoint
        self._tl_proceed_committed: bool = False  # yellow dilemma commitment
        self._route_tl_candidates: set[int] = set()  # Tier-3 pre-associated TL ids

        # Smoothed curvature estimate (EMA)
        self._smoothed_curvature: float = 0.0

        # Corridor polygon for debug drawing (set each tick)
        self._corridor: np.ndarray | None = None
        self._corridor_3d_segments: list[np.ndarray] = []
        self._cached_half_lane: float = self.config.lane_half_width

        # Number of corridor segments built (for diagnostics)
        self._n_corridor_segments: int = 0

        # Distance to the upcoming turn waypoint (set by _get_target)
        self._turn_distance: float | None = None
        self._prev_lane_wp = None

        # Stop sign state
        self._cleared_stop_signs: set[int] = set()

        # Cumulative wp.next() RPC counter for gap filling budget
        self._gap_fill_rpcs: int = 0
        self._gap_fill_diverged: int = 0
        self._gap_fill_budget_exhausted: bool = False
        self._gap_fill_rpc_tried: set[tuple[int, int]] = set()
        self._gap_fill_bezier_tried: set[tuple[int, int]] = set()
        self._gap_fill_rpc_attempts: dict[tuple[int, int], int] = {}
        self._gap_fill_bezier_attempts: dict[tuple[int, int], int] = {}
        # Bezier arc buffer — populated by _fill_junction_gaps, consumed by
        # _advance_route.  Kept separate from _wp_buffer so arc traversal
        # never modifies the route tracking state.
        self._arc_buffer: deque = deque()
        self._stop_sign_wait_counter: int = 0
        self._stop_sign_active: int | None = None  # actor ID

        # Yield sign state
        self._cleared_yield_signs: set[int] = set()
        self._yield_active: int | None = None  # actor ID of yield we're currently handling
        self._yield_wait_counter: int = 0  # frames waited after conflict cleared
        self._yield_has_conflict: bool = False  # cross-traffic detected this tick

        # Rate limiter — previous target for smooth transitions
        self._prev_target: float = self.config.target_speed

        # Mode-transition PID reset — track previous mode so we can
        # reset the speed PID integral when transitioning out of a
        # stop/hold mode.  Prevents stale integral from causing
        # overshoot on resumption.
        self._prev_mode: DrivingMode = DrivingMode.FOLLOW_LANE
        self._RESET_FROM_MODES: frozenset = frozenset(
            {
                DrivingMode.TRAFFIC_LIGHT_STOP,
                DrivingMode.VEHICLE_FOLLOWING,
                DrivingMode.PEDESTRIAN_STOP,
                DrivingMode.EMERGENCY_STOP,
                DrivingMode.STOP_SIGN_STOP,
                DrivingMode.YIELD_STOP,
            }
        )

        # P3-D: track consecutive TURNING frames so we can reset the
        # speed PID only after a real turn (>0.5 s), not a brief curve.
        self._turn_frame_count: int = 0

        # Waypoint planner policy diagnostics and cumulative counters.
        self._planner_policy_reason: PlannerPolicyReason = PlannerPolicyReason.NONE
        self._planner_policy_exception: PlannerPolicyException = PlannerPolicyException.NONE
        self._dist_to_next_junction: float = -1.0
        self._frame_pre_junction_lane_change_violation: int = 0
        self._frame_consecutive_lane_change_violation: int = 0
        self._frame_straight_exit_lane_violation: int = 0
        self._frame_turn_lane_legality_violation: int = 0
        self._frame_roundabout_lane_change_violation: int = 0
        self._frame_route_watchdog_triggered: int = 0
        self._legality_wrong_way_filtered: int = 0
        self._legality_do_not_enter_filtered: int = 0
        self._legality_turn_mismatch_filtered: int = 0
        self._legality_continuity_kept: int = 0
        self._route_watchdog_consecutive_frames: int = 0
        self._route_watchdog_progress_stall_frames: int = 0
        self._route_watchdog_prev_route_completion: float | None = None
        self._policy_distance_traveled: float = 0.0
        self._policy_last_location: Any | None = None
        self._policy_last_lane_change_distance: float = -1e9
        self._policy_last_roundabout_lane_change_distance: float = -1e9

    # ------------------------------------------------------------------
    # Route management
    # ------------------------------------------------------------------

    def _project_route_gap_samples(
        self,
        wp_prev: "carla.Waypoint",
        wp_next: "carla.Waypoint",
        cmd_next: RoadOption,
        gap: float,
        step: float,
        diverge_limit: float,
        seen_ids: set[int],
    ) -> List[Tuple["carla.Waypoint", RoadOption]]:
        """Project linear samples of a sparse route gap back onto map waypoints."""
        origin = wp_prev.transform.location
        target = wp_next.transform.location
        n_samples = max(2, int(gap / step))
        projected_fills: List[Tuple["carla.Waypoint", RoadOption]] = []
        last_target_dist = origin.distance(target)

        for sample_idx in range(1, n_samples):
            t = sample_idx / n_samples
            sx = float(origin.x) + (float(target.x) - float(origin.x)) * t
            sy = float(origin.y) + (float(target.y) - float(origin.y)) * t
            sz = float(getattr(origin, "z", 0.0)) + (
                float(getattr(target, "z", 0.0)) - float(getattr(origin, "z", 0.0))
            ) * t
            if CARLA_AVAILABLE:
                probe_loc = carla.Location(x=sx, y=sy, z=sz)
            else:
                probe_loc = SimpleNamespace(x=sx, y=sy, z=sz)

            try:
                projected_wp = self.map.get_waypoint(probe_loc)
            except (AttributeError, RuntimeError, TypeError):
                return []
            if projected_wp is None:
                return []

            proj_loc = projected_wp.transform.location
            if proj_loc.distance(origin) > diverge_limit:
                return []

            target_dist = proj_loc.distance(target)
            if target_dist > last_target_dist + step:
                return []
            last_target_dist = target_dist

            projected_id = int(getattr(projected_wp, "id", -1))
            if projected_id != -1 and projected_id in seen_ids:
                continue
            if projected_id != -1:
                seen_ids.add(projected_id)
            projected_fills.append((projected_wp, cmd_next))

        if projected_fills:
            end_dist = projected_fills[-1][0].transform.location.distance(target)
            if end_dist <= max(step * 2.0, 2.0):
                return projected_fills
        return []

    def _densify_route_transitions(
        self,
        route: List[Tuple["carla.Waypoint", RoadOption]],
    ) -> List[Tuple["carla.Waypoint", RoadOption]]:
        """Densify sparse transitions once at route-load time.

        This removes dependence on per-frame gap repair for most junction
        transitions and keeps runtime gap filling as a safety net.
        """
        if not route or len(route) < 2:
            return list(route)

        cfg = self.config
        if not cfg.route_densify_enabled:
            return list(route)

        step = max(0.5, float(cfg.route_densify_step))
        gap_threshold = max(float(cfg.route_densify_gap_threshold), step * 2.0)

        densified: List[Tuple["carla.Waypoint", RoadOption]] = [route[0]]

        for idx in range(1, len(route)):
            wp_prev, _ = route[idx - 1]
            wp_next, cmd_next = route[idx]
            gap = wp_prev.transform.location.distance(wp_next.transform.location)

            if gap <= gap_threshold:
                densified.append((wp_next, cmd_next))
                continue

            max_steps = (
                int(gap / step * cfg.route_densify_max_factor) + 1
                if math.isfinite(cfg.route_densify_max_factor)
                else float("inf")
            )
            diverge_limit = gap * float(cfg.route_densify_diverge_factor)
            origin = wp_prev.transform.location
            target = wp_next.transform.location

            inserted: List[Tuple["carla.Waypoint", RoadOption]] = []
            seen_ids = {int(getattr(wp_prev, "id", -1)), int(getattr(wp_next, "id", -1))}
            converged = False
            current_wp = wp_prev
            steps = 0

            try:
                nexts = current_wp.next(step)
                while nexts and steps < max_steps:
                    # Choose the branch that best approaches the intended target.
                    next_wp = min(
                        nexts,
                        key=lambda cand: cand.transform.location.distance(target),
                    )
                    next_loc = next_wp.transform.location

                    if next_loc.distance(target) < step:
                        converged = True
                        break
                    if next_loc.distance(origin) > diverge_limit:
                        break

                    wp_id = int(getattr(next_wp, "id", -1))
                    if wp_id != -1 and wp_id in seen_ids:
                        break
                    if wp_id != -1:
                        seen_ids.add(wp_id)

                    inserted.append((next_wp, cmd_next))
                    current_wp = next_wp
                    nexts = current_wp.next(step)
                    steps += 1
            except (AttributeError, RuntimeError):
                inserted = []

            if not converged and cfg.route_densify_projection_fallback:
                inserted = self._project_route_gap_samples(
                    wp_prev=wp_prev,
                    wp_next=wp_next,
                    cmd_next=cmd_next,
                    gap=gap,
                    step=step,
                    diverge_limit=diverge_limit,
                    seen_ids=seen_ids,
                )

            densified.extend(inserted)
            densified.append((wp_next, cmd_next))

        return densified

    @staticmethod
    def _wrap_angle_deg(angle: float) -> float:
        """Wrap an angle to [-180, 180] degrees."""
        return (angle + 180.0) % 360.0 - 180.0

    def _transition_signed_angle_deg(self, wp_prev: Any, wp_next: Any) -> float | None:
        """Signed angle from waypoint heading to transition direction."""
        try:
            prev_loc = wp_prev.transform.location
            next_loc = wp_next.transform.location
            yaw_prev = float(wp_prev.transform.rotation.yaw)
        except (AttributeError, TypeError):
            return None

        dx = float(next_loc.x) - float(prev_loc.x)
        dy = float(next_loc.y) - float(prev_loc.y)
        if math.hypot(dx, dy) < 1e-6:
            return None

        transition_yaw = math.degrees(math.atan2(dy, dx))
        return float(self._wrap_angle_deg(transition_yaw - yaw_prev))

    @staticmethod
    def _contains_do_not_enter_landmark(wp: Any) -> bool:
        """Heuristic check for do-not-enter landmarks near a waypoint."""
        try:
            landmarks = wp.get_landmarks(10.0, True)
        except (AttributeError, RuntimeError, TypeError):
            return False

        for lm in landmarks or []:
            lm_blob = " ".join(
                [
                    str(getattr(lm, "type", "")),
                    str(getattr(lm, "name", "")),
                    str(getattr(lm, "id", "")),
                ]
            ).lower()
            if any(token in lm_blob for token in ("do_not_enter", "do-not-enter", "no_entry")):
                return True
        return False

    def _is_route_transition_legal(
        self,
        wp_prev: Any,
        wp_next: Any,
        cmd_next: RoadOption,
        *,
        count_violation: bool,
    ) -> bool:
        """Check one route transition against hard legality filters."""
        cfg = self.config
        if not (cfg.planner_policy_enabled and cfg.planner_legality_filter_enabled):
            return True

        def _flag(reason: str) -> bool:
            if not count_violation:
                return False
            self._planner_policy_reason = PlannerPolicyReason.LEGALITY_FILTERED
            if reason == "wrong_way":
                self._legality_wrong_way_filtered += 1
            elif reason == "do_not_enter":
                self._legality_do_not_enter_filtered += 1
            elif reason == "turn_mismatch":
                self._legality_turn_mismatch_filtered += 1
            return False

        signed = self._transition_signed_angle_deg(wp_prev, wp_next)
        if signed is not None and abs(signed) > float(cfg.planner_wrong_way_max_yaw_diff_deg):
            return _flag("wrong_way")

        prev_lane_id = int(getattr(wp_prev, "lane_id", 0))
        next_lane_id = int(getattr(wp_next, "lane_id", 0))
        same_road = getattr(wp_prev, "road_id", None) == getattr(wp_next, "road_id", None)
        lane_sign_flipped = (
            same_road
            and prev_lane_id != 0
            and next_lane_id != 0
            and ((prev_lane_id > 0) != (next_lane_id > 0))
        )
        if lane_sign_flipped and not cfg.planner_allow_lane_id_sign_flip:
            return _flag("do_not_enter")

        if self._contains_do_not_enter_landmark(wp_next):
            return _flag("do_not_enter")

        if signed is not None:
            threshold = max(5.0, float(cfg.planner_turn_mismatch_threshold_deg))
            if cmd_next == RoadOption.LEFT and signed < -threshold:
                return _flag("turn_mismatch")
            if cmd_next == RoadOption.RIGHT and signed > threshold:
                return _flag("turn_mismatch")
            if cmd_next == RoadOption.STRAIGHT and abs(signed) > max(45.0, threshold * 2.5):
                return _flag("turn_mismatch")

        return True

    def _enforce_route_legality(
        self,
        route: List[Tuple["carla.Waypoint", RoadOption]],
    ) -> List[Tuple["carla.Waypoint", RoadOption]]:
        """Apply hard legality filtering to planned route transitions."""
        if len(route) < 2:
            return list(route)
        if not (self.config.planner_policy_enabled and self.config.planner_legality_filter_enabled):
            return list(route)

        filtered: List[Tuple["carla.Waypoint", RoadOption]] = [route[0]]
        continuity_kept = 0
        for idx in range(1, len(route)):
            wp_prev, _ = filtered[-1]
            wp_next, cmd_next = route[idx]
            if self._is_route_transition_legal(
                wp_prev,
                wp_next,
                cmd_next,
                count_violation=True,
            ):
                filtered.append((wp_next, cmd_next))
                continue

            # Keep continuity when filtering this waypoint would create an
            # excessive skip gap in the active route geometry.
            max_skip_gap = float(self.config.planner_legality_max_skip_gap_m)
            if max_skip_gap > 0.0:
                try:
                    skip_gap = wp_prev.transform.location.distance(wp_next.transform.location)
                except (AttributeError, TypeError):
                    skip_gap = 0.0
                if skip_gap > max_skip_gap:
                    filtered.append((wp_next, cmd_next))
                    continuity_kept += 1

        self._legality_continuity_kept += continuity_kept

        if len(filtered) < 2:
            logger.warning(
                "Legality filter removed too many route transitions (%d -> %d). Falling back to original route.",
                len(route),
                len(filtered),
            )
            return list(route)

        if len(filtered) != len(route):
            logger.info(
                "Legality filter removed %d route transitions (%d -> %d)",
                len(route) - len(filtered),
                len(route),
                len(filtered),
            )
        if continuity_kept > 0:
            logger.info(
                "Legality continuity guard kept %d transitions due to large skip gaps",
                continuity_kept,
            )
        return filtered

    def set_route(self, route: List[Tuple["carla.Waypoint", RoadOption]]) -> None:
        """Load a pre-computed route (list of (waypoint, RoadOption))."""
        route_waypoints = list(route)

        self._planner_policy_reason = PlannerPolicyReason.NONE
        self._planner_policy_exception = PlannerPolicyException.NONE
        self._dist_to_next_junction = -1.0
        self._frame_pre_junction_lane_change_violation = 0
        self._frame_consecutive_lane_change_violation = 0
        self._frame_straight_exit_lane_violation = 0
        self._frame_turn_lane_legality_violation = 0
        self._frame_roundabout_lane_change_violation = 0
        self._frame_route_watchdog_triggered = 0
        self._frame_turn_lane_legality_violation = 0
        self._frame_roundabout_lane_change_violation = 0
        self._frame_route_watchdog_triggered = 0
        self._legality_wrong_way_filtered = 0
        self._legality_do_not_enter_filtered = 0
        self._legality_turn_mismatch_filtered = 0
        self._legality_continuity_kept = 0
        self._route_watchdog_consecutive_frames = 0
        self._route_watchdog_progress_stall_frames = 0
        self._route_watchdog_prev_route_completion = None
        self._policy_distance_traveled = 0.0
        self._policy_last_location = None
        self._policy_last_lane_change_distance = -1e9
        self._policy_last_roundabout_lane_change_distance = -1e9

        route_waypoints = self._densify_route_transitions(route_waypoints)
        route_waypoints = self._enforce_route_legality(route_waypoints)
        self._route = route_waypoints
        self._route_length = len(route_waypoints)
        self._route_index = 0
        self._wp_buffer.clear()
        # Reset gap-fill state BEFORE _fill_buffer so the initial fill starts
        # with a clean slate and the tried-set is valid after fill returns.
        # Also clear any stale arc from a previous route (new expert is created
        # per episode in the collector, but belt-and-suspenders here).
        self._arc_buffer.clear()
        self._gap_fill_rpcs = 0
        self._gap_fill_diverged = 0
        self._gap_fill_budget_exhausted = False
        self._gap_fill_rpc_tried = set()
        self._gap_fill_bezier_tried = set()
        self._gap_fill_rpc_attempts = {}
        self._gap_fill_bezier_attempts = {}
        self._fill_buffer()
        self.speed_pid.reset()
        self.steer_pid.reset()
        self.mode = DrivingMode.FOLLOW_LANE
        self._tl_waiting = None
        self._tl_stop_loc = None
        self._route_tl_candidates.clear()
        self._cleared_stop_signs.clear()
        self._stop_sign_wait_counter = 0
        self._stop_sign_active = None
        self._cleared_yield_signs.clear()
        self._yield_active = None
        self._yield_wait_counter = 0
        self._yield_has_conflict = False
        self._prev_lane_wp = None

        self._planner_policy_reason = PlannerPolicyReason.NONE
        self._planner_policy_exception = PlannerPolicyException.NONE
        self._dist_to_next_junction = -1.0
        self._frame_pre_junction_lane_change_violation = 0
        self._frame_consecutive_lane_change_violation = 0
        self._frame_straight_exit_lane_violation = 0
        self._frame_turn_lane_legality_violation = 0
        self._frame_roundabout_lane_change_violation = 0
        self._frame_route_watchdog_triggered = 0
        self._route_watchdog_progress_stall_frames = 0
        self._route_watchdog_prev_route_completion = None

        try:
            self._refresh_route_tl_candidates(self.vehicle.get_transform())
        except RuntimeError:
            # World not ready yet — cache will be refreshed in run_step.
            self._route_tl_candidates.clear()

    def set_destination(self, start: "carla.Location", end: "carla.Location") -> None:
        """Plan and load a route from *start* to *end*."""
        route = self.route_planner.trace_route(start, end)
        self.set_route(route)

    def done(self) -> bool:
        """``True`` when the route has been consumed."""
        if not self._route:
            return True
        return self._route_index >= self._route_length and len(self._wp_buffer) <= 1

    @property
    def route_completion(self) -> float:
        """Fraction of route completed (0.0 – 1.0)."""
        if self._route_length == 0:
            return 1.0
        return min(1.0, self._route_index / self._route_length)

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def run_step(self, dt: float = 0.05) -> Tuple["carla.VehicleControl", FrameInfo]:
        """Compute controls for the current simulation tick.

        Returns
        -------
        control : carla.VehicleControl
            Expert driving command.
        info : FrameInfo
            Privileged metadata for this frame.
        """
        # --- 1.  Ego state ---
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_kmh = speed_ms * 3.6

        # Per-frame policy diagnostics are reset at the start of each tick.
        self._planner_policy_reason = PlannerPolicyReason.NONE
        self._planner_policy_exception = PlannerPolicyException.NONE
        self._dist_to_next_junction = -1.0
        self._frame_pre_junction_lane_change_violation = 0
        self._frame_consecutive_lane_change_violation = 0
        self._frame_straight_exit_lane_violation = 0

        # Dynamic speed limit from road signs (falls back to config)
        cfg = self.config
        road_limit = self.vehicle.get_speed_limit()  # km/h
        if road_limit is not None and road_limit > 0:
            base_target = min(
                road_limit,
                cfg.max_speed,
            )  # or min(road_limit * cfg.speed_limit_margin, cfg.max_speed)
        else:
            base_target = cfg.target_speed

        # --- 2.  Route progress ---
        self._advance_route(transform)
        target_wp, nav_label, nav_command = self._get_target(transform)
        ego_wp = self.map.get_waypoint(transform.location)

        if self._policy_last_location is not None:
            self._policy_distance_traveled += transform.location.distance(self._policy_last_location)
        self._policy_last_location = transform.location

        nav_label = self._apply_lane_change_policy(nav_label, ego_wp, transform.location)
        if (
            (self._frame_turn_lane_legality_violation > 0 or self._frame_roundabout_lane_change_violation > 0)
            and nav_command in (RoadOption.LEFT, RoadOption.RIGHT)
        ):
            nav_command = RoadOption.LANEFOLLOW

        buf_len = len(self._wp_buffer)
        buf_head_dist = -1.0
        if self._wp_buffer:
            wp0_loc = self._wp_buffer[0][0].transform.location
            buf_head_dist = transform.location.distance(wp0_loc)
        heading_error_deg = self._target_heading_error_deg(transform, target_wp)

        # --- 3.  Privileged perception ---
        tl_state, tl_distance = self._check_traffic_light(transform)
        ss_distance, ss_committed = self._check_stop_signs(transform, speed_kmh)
        ys_distance, ys_has_conflict = self._check_yield_signs(transform, speed_kmh)
        nearby_vehicles = self._detect_vehicles(transform)
        nearby_pedestrians = self._detect_pedestrians(transform)
        lead_distance, lead_speed, lead_lateral = self._lead_vehicle_info(nearby_vehicles, speed_kmh)
        ped_dist = self._closest_pedestrian_distance(nearby_pedestrians, transform)

        # --- 4.  Target speed & driving mode ---
        target_speed, mode = self._plan_speed(
            speed_kmh,
            nav_command,
            tl_state,
            tl_distance,
            ss_distance,
            ss_committed,
            lead_distance,
            lead_speed,
            lead_lateral,
            ped_dist,
            base_target,
            ys_distance,
            ys_has_conflict,
        )
        target_speed, mode = self._apply_route_continuity_watchdog(
            buf_head_dist=buf_head_dist,
            target_speed=target_speed,
            mode=mode,
            speed_kmh=speed_kmh,
            route_completion=self.route_completion,
            heading_error_deg=heading_error_deg,
            tl_state=tl_state,
            tl_distance=tl_distance,
        )

        # Reset speed PID integral when leaving a stop/hold mode.
        # This prevents stale integral accumulated during long stops
        # (e.g. red lights) from causing overshoot on resumption.
        if self._prev_mode in self._RESET_FROM_MODES and mode not in self._RESET_FROM_MODES:
            self.speed_pid.reset()

        # Reset steer PID when exiting a turn so the integral wound up
        # during the turn does not bias the post-turn straight.
        # Also snap _smoothed_curvature to 0 so follow-distance and
        # gain-blending immediately return to straight-line values
        # instead of decaying over ~15 frames (EMA lag).
        if self._prev_mode == DrivingMode.TURNING and mode != DrivingMode.TURNING:
            self.steer_pid.reset()
            self._smoothed_curvature = 0.0
            # P3-D: clear stale speed integral from the turn-regime
            # gains if the turn lasted >10 frames (0.5 s at 20 Hz).
            # Uses reset_integral() instead of reset() to avoid a
            # derivative kick from zeroing _prev_error while the
            # actual speed error is large (~15 km/h).
            if self._turn_frame_count > 10:
                self.speed_pid.reset_integral()

        # P3-D: maintain the turn duration counter.
        if mode == DrivingMode.TURNING:
            self._turn_frame_count += 1
        else:
            self._turn_frame_count = 0

        self._prev_mode = mode

        self.mode = mode
        self._apply_gain_blending()

        # --- 5.  Control ---
        steer = self._lateral_control(transform, target_wp)
        # Inject small Gaussian noise to prevent overfitting to
        # pixel-perfect expert trajectories (D7).
        if cfg.steering_noise_sigma > 0:
            steer += float(np.random.normal(0.0, cfg.steering_noise_sigma))
        throttle, brake = self._longitudinal_control(speed_kmh, target_speed, dt)

        control = carla.VehicleControl(
            throttle=float(np.clip(throttle, 0.0, 1.0)),
            steer=float(np.clip(steer, -1.0, 1.0)),
            brake=float(np.clip(brake, 0.0, 1.0)),
            hand_brake=False,
            manual_gear_shift=False,
        )

        # --- 6.  Privileged info ---
        sat_hits = sum(1 for v in nearby_vehicles if v.get("obb_in_lane", False))
        dynamic_half = self._dynamic_half_lane(transform)
        lat_hits = sum(
            1
            for v in nearby_vehicles
            if v["is_ahead"] and v.get("same_direction", True) and abs(v["lateral"]) < dynamic_half
        )

        info = FrameInfo(
            nav_command=int(nav_label),
            route_completion=self.route_completion,
            target_speed=target_speed,
            current_speed=speed_kmh,
            driving_mode=int(mode),
            position=(transform.location.x, transform.location.y, transform.location.z),
            rotation=(transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll),
            velocity=(velocity.x, velocity.y, velocity.z),
            traffic_light_state=self._encode_tl(tl_state),
            traffic_light_distance=(tl_distance if tl_distance is not None else -1.0),
            traffic_light_actor_id=(self._tl_waiting if self._tl_waiting is not None else -1),
            traffic_light_stop_line_position=(
                (
                    self._tl_stop_loc.x,
                    self._tl_stop_loc.y,
                    self._tl_stop_loc.z,
                )
                if self._tl_stop_loc is not None
                else (-1.0, -1.0, -1.0)
            ),
            stop_sign_distance=(ss_distance if ss_distance is not None else -1.0),
            yield_sign_distance=(ys_distance if ys_distance is not None else -1.0),
            nearby_vehicles=nearby_vehicles,
            nearby_pedestrians=nearby_pedestrians,
            n_corridor_segments=self._n_corridor_segments,
            wp_buffer_size=buf_len,
            wp_buffer_head_dist=round(buf_head_dist, 2),
            sat_detections=sat_hits,
            lateral_detections=lat_hits,
            gap_fill_rpcs=self._gap_fill_rpcs,
            gap_fill_diverged=self._gap_fill_diverged,
            planner_policy_reason=int(self._planner_policy_reason),
            planner_policy_exception=int(self._planner_policy_exception),
            dist_to_next_junction=float(self._dist_to_next_junction),
            pre_junction_lane_change_violation=self._frame_pre_junction_lane_change_violation,
            consecutive_lane_change_violation=self._frame_consecutive_lane_change_violation,
            straight_exit_lane_violation=self._frame_straight_exit_lane_violation,
            turn_lane_legality_violation=self._frame_turn_lane_legality_violation,
            roundabout_lane_change_violation=self._frame_roundabout_lane_change_violation,
            legality_wrong_way_filtered=self._legality_wrong_way_filtered,
            legality_do_not_enter_filtered=self._legality_do_not_enter_filtered,
            legality_turn_mismatch_filtered=self._legality_turn_mismatch_filtered,
            legality_continuity_kept=self._legality_continuity_kept,
            route_watchdog_triggered=self._frame_route_watchdog_triggered,
        )

        return control, info

    # ------------------------------------------------------------------
    # Internal — route buffer
    # ------------------------------------------------------------------

    def _fill_buffer(self, transform: "carla.Transform" | None = None) -> None:
        """Refill the waypoint look-ahead buffer from the route.

        When the planned route is exhausted but the buffer is below
        half capacity, extend it by querying the map for waypoints
        ahead of the last buffered waypoint.  This prevents corridor
        degeneration in the final ~200 frames of an episode where the
        buffer would otherwise shrink to 1–2 entries.
        """
        buffer_maxlen = self._wp_buffer.maxlen or 40
        while len(self._wp_buffer) < buffer_maxlen and self._route_index < self._route_length:
            self._wp_buffer.append(self._route[self._route_index])
            self._route_index += 1

        # --- Junction gap fill ---
        # The GlobalRoutePlanner emits only two waypoints for junction
        # (turn) edges — the last approach waypoint and a point ~10 m
        # past the junction exit.  This creates a single ~17 m chord
        # segment in the corridor that produces a visible ±45° spike
        # and degrades SAT coverage, curvature estimation, and follow-
        # distance adaptation.  Detect these large gaps and fill them
        # with intermediate waypoints queried from the HD map.
        self._fill_junction_gaps(transform)

        # --- Route-end extension ---
        # If the planned route is consumed and the buffer is running
        # low, synthesise continuation waypoints from the map so that
        # the SAT corridor stays healthy until the episode ends.
        half_cap = (self._wp_buffer.maxlen or 40) // 2
        if (
            self._route_index >= self._route_length
            and len(self._wp_buffer) < half_cap
            and self._wp_buffer
        ):
            last_wp, _ = self._wp_buffer[-1]
            try:
                nexts = last_wp.next(2.0)  # 2 m step
                while nexts and len(self._wp_buffer) < buffer_maxlen:
                    legal_nexts = [
                        cand
                        for cand in nexts
                        if self._is_route_transition_legal(
                            last_wp,
                            cand,
                            RoadOption.LANEFOLLOW,
                            count_violation=True,
                        )
                    ]
                    if not legal_nexts:
                        break
                    nxt = legal_nexts[0]  # follow the first legal keep-lane option
                    self._wp_buffer.append((nxt, RoadOption.LANEFOLLOW))
                    last_wp = nxt
                    nexts = nxt.next(2.0)
            except (AttributeError, RuntimeError):
                pass  # graceful degradation if map query fails

        if transform is not None:
            self._refresh_tl_stop_loc(transform)
            self._refresh_route_tl_candidates(transform)

    def _route_waypoint_sources(self, lookahead: int = 200) -> list[tuple[Any, RoadOption]]:
        """Return the active buffer plus a capped lookahead from the planned route."""
        sources = list(self._wp_buffer)
        if self._route and self._route_index < self._route_length:
            end = min(self._route_index + lookahead, self._route_length)
            sources.extend(self._route[self._route_index : end])
        return sources

    def _fill_junction_gaps(self, transform: "carla.Transform" | None = None) -> None:
        """Insert intermediate waypoints where the buffer has large gaps.

        The ``GlobalRoutePlanner`` emits only two waypoints for
        junction (turn) edges — the last approach waypoint and a
        point ~10 m past the junction exit.  The resulting ~15–20 m
        chord degrades the corridor (±45° spike), curvature
        estimation, and adaptive follow-distance.

        This method scans consecutive waypoint pairs in the buffer.
        When a gap exceeds ``_GAP_THRESHOLD`` metres, it first attempts
        a smooth Bezier interpolation between entry and exit waypoints
        (projected back to map waypoints). If projection quality is
        poor, it falls back to the legacy ``wp.next(step)`` chaining.

        The buffer's ``maxlen`` is respected — filling stops when
        full, and the deque naturally evicts the oldest entries.

        Safety bounds (configurable via ``ExpertConfig``):

        1. **Per-gap iteration cap** (``gap_fill_max_factor``):  max
           iterations = ``gap / fill_step * factor``.  Prevents
           runaway loops when ``wp_prev`` and ``wp_next`` lie on
           different roads at a junction — in that case ``wp.next()``
           follows *wp_prev*'s road forward and never converges on
           *wp_next*.  Set to ``float('inf')`` for unbounded
           (NOT RECOMMENDED — the unbounded loop leaked ~19 GB of
           server-side memory and triggered OOM).

        2. **Distance-from-origin guard** (``gap_fill_diverge_factor``):
           breaks if the interpolated waypoint has traveled further
           from the gap's starting point than ``gap * factor``.  This
           catches divergence even with a generous iteration cap.

        3. **Cumulative RPC budget** (``gap_fill_rpc_budget``):  total
           ``wp.next()`` calls across all frames for this episode.
           When exhausted, gap filling is disabled for the rest of
           the episode with a warning.  Default 50,000 RPCs is
           generous for normal driving (~2 RPCs per junction × ~200
           junctions encountered in a long episode) while capping
           server-side memory at a predictable level.  Set to 0
           for unlimited.
        """
        _GAP_THRESHOLD = 4.0  # metres — normal spacing is ~2 m
        _FILL_STEP = 2.0  # metres — interpolation step / fallback step
        cfg = self.config
        bezier_step = max(0.25, float(cfg.gap_fill_bezier_step))

        def _loc_dist(a: Any, b: Any) -> float:
            dx = float(a.x) - float(b.x)
            dy = float(a.y) - float(b.y)
            dz = float(getattr(a, "z", 0.0)) - float(getattr(b, "z", 0.0))
            return math.sqrt(dx * dx + dy * dy + dz * dz)

        def _make_loc(x: float, y: float, z: float) -> Any:
            if CARLA_AVAILABLE:
                try:
                    return carla.Location(x=x, y=y, z=z)
                except (AttributeError, TypeError):
                    pass
            return SimpleNamespace(x=x, y=y, z=z)

        def _bezier_xyz(
            t: float,
            p0: tuple[float, float, float],
            p1: tuple[float, float, float],
            p2: tuple[float, float, float],
            p3: tuple[float, float, float],
        ) -> tuple[float, float, float]:
            omt = 1.0 - t
            omt2 = omt * omt
            t2 = t * t
            bx = (
                omt2 * omt * p0[0]
                + 3.0 * omt2 * t * p1[0]
                + 3.0 * omt * t2 * p2[0]
                + t2 * t * p3[0]
            )
            by = (
                omt2 * omt * p0[1]
                + 3.0 * omt2 * t * p1[1]
                + 3.0 * omt * t2 * p2[1]
                + t2 * t * p3[1]
            )
            bz = (
                omt2 * omt * p0[2]
                + 3.0 * omt2 * t * p1[2]
                + 3.0 * omt * t2 * p2[2]
                + t2 * t * p3[2]
            )
            return bx, by, bz

        # Gap filling disabled by config or budget exhausted
        if cfg.gap_fill_max_factor == 0:
            return
        if self._gap_fill_budget_exhausted:
            return
        if len(self._wp_buffer) < 2:
            return

        # Work on a snapshot; rebuild with fills inserted.
        # NOTE: Bezier arc fills are stored in self._arc_buffer (separate from
        # the route buffer) so they never cause overflow or displace route
        # waypoints.  Only legacy RPC fills go into rebuilt here.
        items = list(self._wp_buffer)
        rebuilt: list[tuple] = [items[0]]
        rpcs_this_call = 0

        for j in range(1, len(items)):
            wp_prev, _ = items[j - 1]
            wp_next, cmd_next = items[j]
            gap = wp_prev.transform.location.distance(wp_next.transform.location)

            if gap > _GAP_THRESHOLD:
                pair_key = (wp_prev.id, wp_next.id)
                bezier_retries = max(1, int(cfg.gap_fill_bezier_retry_limit))
                rpc_retries = max(1, int(cfg.gap_fill_rpc_retry_limit))
                bezier_attempts = self._gap_fill_bezier_attempts.get(pair_key, 0)
                rpc_attempts = self._gap_fill_rpc_attempts.get(pair_key, 0)
                bezier_allowed = (
                    cfg.gap_fill_bezier_enabled
                    and not self._arc_buffer
                    and j <= max(1, int(cfg.gap_fill_bezier_max_pair_index))
                    and bezier_attempts < bezier_retries
                )
                rpc_allowed = rpc_attempts < rpc_retries

                if not bezier_allowed and not rpc_allowed:
                    rebuilt.append((wp_next, cmd_next))
                    continue

                # Cap iterations per gap.  For factor=3.0 and a 10 m
                # gap with 2 m steps, this allows 15 iterations.
                # float('inf') gives unbounded (NOT RECOMMENDED).
                max_fills = (
                    int(gap / _FILL_STEP * cfg.gap_fill_max_factor) + 1
                    if math.isfinite(cfg.gap_fill_max_factor)
                    else float("inf")
                )
                diverge_limit = gap * cfg.gap_fill_diverge_factor
                origin = wp_prev.transform.location

                # Try smooth geometric interpolation first for junction gaps.
                # Only generate an arc for j==1 (the gap immediately ahead of
                # _wp_buffer[0]).  Generating arcs for far-ahead gaps (j>1)
                # freezes _wp_buffer prematurely — ego can't consume arc points
                # that are 50-100 m away, causing wpd to grow to 100m+.
                # Skip if a Bezier arc is already active — don't start a second.
                bezier_fills: list[tuple[Any, RoadOption]] | None = None
                if bezier_allowed:
                    self._gap_fill_bezier_tried.add(pair_key)
                    self._gap_fill_bezier_attempts[pair_key] = bezier_attempts + 1
                    target_loc = wp_next.transform.location
                    yaw0 = math.radians(float(wp_prev.transform.rotation.yaw))
                    yaw3 = math.radians(float(wp_next.transform.rotation.yaw))
                    control_len = max(
                        _FILL_STEP,
                        min(
                            gap * max(0.0, cfg.gap_fill_bezier_tangent_factor),
                            gap * 0.5,
                        ),
                    )

                    p0 = (float(origin.x), float(origin.y), float(getattr(origin, "z", 0.0)))
                    p3 = (
                        float(target_loc.x),
                        float(target_loc.y),
                        float(getattr(target_loc, "z", 0.0)),
                    )
                    p1 = (
                        p0[0] + control_len * math.cos(yaw0),
                        p0[1] + control_len * math.sin(yaw0),
                        p0[2],
                    )
                    p2 = (
                        p3[0] - control_len * math.cos(yaw3),
                        p3[1] - control_len * math.sin(yaw3),
                        p3[2],
                    )

                    n_samples = max(2, int(gap / bezier_step))
                    proj_error_limit = max(_FILL_STEP, float(cfg.gap_fill_bezier_max_project_error))
                    tentative_curve: list[tuple[Any, RoadOption]] = []
                    seen_ids = {int(getattr(wp_prev, "id", -1)), int(getattr(wp_next, "id", -1))}
                    last_target_dist = _loc_dist(origin, target_loc)
                    bezier_ok = True

                    for idx in range(1, n_samples):
                        t = idx / n_samples
                        bx, by, bz = _bezier_xyz(t, p0, p1, p2, p3)
                        probe_loc = _make_loc(bx, by, bz)
                        try:
                            projected_wp = self.map.get_waypoint(probe_loc)
                        except (AttributeError, RuntimeError, TypeError):
                            bezier_ok = False
                            break
                        if projected_wp is None:
                            bezier_ok = False
                            break

                        proj_loc = projected_wp.transform.location
                        if _loc_dist(proj_loc, probe_loc) > proj_error_limit:
                            bezier_ok = False
                            break
                        if _loc_dist(proj_loc, origin) > diverge_limit:
                            bezier_ok = False
                            break

                        target_dist = _loc_dist(proj_loc, target_loc)
                        if target_dist > last_target_dist + _FILL_STEP:
                            bezier_ok = False
                            break
                        last_target_dist = target_dist

                        wp_id = int(getattr(projected_wp, "id", -1))
                        if wp_id != -1 and wp_id in seen_ids:
                            continue
                        if wp_id != -1:
                            seen_ids.add(wp_id)
                        tentative_curve.append((projected_wp, cmd_next))

                    if bezier_ok and tentative_curve:
                        end_dist = _loc_dist(tentative_curve[-1][0].transform.location, target_loc)
                        if end_dist <= max(_FILL_STEP * 2.0, bezier_step * 2.0):
                            bezier_fills = tentative_curve

                if bezier_fills:
                    # Store arc in the dedicated arc buffer — do NOT insert
                    # into rebuilt.  This keeps _wp_buffer free of arc points
                    # so route tracking is never corrupted by overflow.
                    # junction_exit (wp_next) stays in rebuilt via the normal
                    # append below, so _wp_buffer always has the correct exit.
                    self._arc_buffer = deque(bezier_fills)
                    # (fall through to wp_next append; skip RPC fallback)
                elif rpc_allowed:
                    self._gap_fill_rpc_tried.add(pair_key)
                    self._gap_fill_rpc_attempts[pair_key] = rpc_attempts + 1
                    # Accumulate fills tentatively — only commit if they
                    # actually converge toward wp_next.  If the loop exits
                    # via the divergence guard or iteration cap WITHOUT
                    # reaching the target, ALL tentative fills are discarded.
                    # This prevents wrong-road waypoints from corrupting the
                    # buffer and triggering cascading re-fills on subsequent
                    # frames.
                    tentative: list[tuple] = []
                    converged = False
                    try:
                        nexts = wp_prev.next(_FILL_STEP)
                        rpcs_this_call += 1
                        fills = 0
                        while nexts and fills < max_fills:
                            nxt = nexts[0]
                            dist_to_target = nxt.transform.location.distance(
                                wp_next.transform.location
                            )
                            if dist_to_target < _FILL_STEP:
                                converged = True
                                break  # close enough to the next original wp
                            # Divergence guard
                            dist_from_origin = nxt.transform.location.distance(origin)
                            if dist_from_origin > diverge_limit:
                                break
                            tentative.append((nxt, cmd_next))
                            nexts = nxt.next(_FILL_STEP)
                            rpcs_this_call += 1
                            fills += 1
                    except (AttributeError, RuntimeError):
                        pass  # graceful degradation

                    # Only insert fills that actually converged on the target.
                    # Diverging fills (wrong road) are silently discarded —
                    # the original gap is preserved, which is harmless
                    # (the corridor just has one longer segment there).
                    if converged and tentative:
                        rebuilt.extend(tentative)
                    elif tentative:
                        # Gap diverged — fills were on a different road.
                        # Discard them to avoid corrupting the waypoint
                        # buffer and causing cascading re-fills.
                        self._gap_fill_diverged += 1

            rebuilt.append((wp_next, cmd_next))

        # Track cumulative RPCs and check budget
        self._gap_fill_rpcs += rpcs_this_call
        budget = cfg.gap_fill_rpc_budget
        if budget > 0 and self._gap_fill_rpcs >= budget:
            self._gap_fill_budget_exhausted = True
            logger.warning(
                "Junction gap-fill RPC budget exhausted (%d/%d). "
                "Gap filling disabled for remainder of episode. "
                "Increase ExpertConfig.gap_fill_rpc_budget if needed.",
                self._gap_fill_rpcs,
                budget,
            )

        # Only rebuild if we actually inserted anything (RPC fills only —
        # Bezier arc fills now go to _arc_buffer and never reach rebuilt).
        if len(rebuilt) != len(items):
            self._wp_buffer.clear()
            maxlen = self._wp_buffer.maxlen or 60
            for entry in rebuilt[:maxlen]:
                self._wp_buffer.append(entry)

        if transform is not None:
            self._refresh_tl_stop_loc(transform)

    def _advance_route(self, transform: "carla.Transform") -> None:
        """Pop consumed waypoints via distance and segment progress.

        When a Bezier arc is active (_arc_buffer non-empty), the arc is
        advanced instead of the route buffer.  The route buffer is frozen
        during arc traversal so junction_exit stays as _wp_buffer[0] and
        becomes the natural handoff point the moment the arc empties.

        Waypoint advancement is position-based (progress along
        wp[0]->wp[1]) rather than heading-based so yaw rotation in
        turns does not trigger premature mass popping.
        """
        location = transform.location

        def _progress_past(buf: deque) -> bool:
            """Return True and pop buf[0] if ego is past its midpoint."""
            wp_curr, _ = buf[0]
            wp_next, _ = buf[1]
            curr_loc = wp_curr.transform.location
            next_loc = wp_next.transform.location
            dist = location.distance(curr_loc)
            seg_dx = float(next_loc.x - curr_loc.x)
            seg_dy = float(next_loc.y - curr_loc.y)
            seg_len = math.hypot(seg_dx, seg_dy)
            past_midpoint = False
            if seg_len > 1e-6:
                ego_dx = float(location.x - curr_loc.x)
                ego_dy = float(location.y - curr_loc.y)
                progress = (seg_dx * ego_dx + seg_dy * ego_dy) / seg_len
                past_midpoint = progress > (0.5 * seg_len)
            return dist < self.config.min_follow_distance or past_midpoint

        if self._arc_buffer:
            # Arc is active — advance the arc, freeze the route buffer.
            while len(self._arc_buffer) > 1 and _progress_past(self._arc_buffer):
                self._arc_buffer.popleft()
            if len(self._arc_buffer) <= 1:
                # Arc exhausted — clear it so _get_target falls back to
                # _wp_buffer.  The route buffer still holds junction_exit
                # as its front entry; normal catchup popping handles it.
                self._arc_buffer.clear()
            # Still call _fill_buffer so route waypoints reload normally,
            # but _fill_junction_gaps will skip new Bezier generation while
            # _arc_buffer is non-empty (no double-arc).
            self._fill_buffer(transform)
            return

        # Normal route buffer advancement.
        while len(self._wp_buffer) > 1 and _progress_past(self._wp_buffer):
            self._wp_buffer.popleft()
        self._fill_buffer(transform)

    def _get_target(
        self,
        transform: "carla.Transform",
    ) -> Tuple["carla.Waypoint", RoadOption, RoadOption]:
        """Return the target waypoint and the current nav command.

        Navigation command logic:
            Scan ahead in the buffer for the first non-LANEFOLLOW command
            within ``turn_anticipation_distance``.  If found, report that
            command so the model (and the speed planner) can anticipate
            the upcoming turn.  Otherwise report LANEFOLLOW.
        """
        ego_loc = transform.location
        ego_wp = self.map.get_waypoint(ego_loc)

        if not self._wp_buffer:
            wp = ego_wp if ego_wp is not None else self.map.get_waypoint(transform.location)
            self._prev_lane_wp = wp
            return wp, RoadOption.LANEFOLLOW, RoadOption.LANEFOLLOW

        # --- Determine nav command with anticipation ---
        # Two separate outputs:
        #   nav_label   — first actionable command for HDF5 recording
        #                 (includes STRAIGHT, CHANGELANELEFT, CHANGELANERIGHT)
        #   nav_command — LEFT/RIGHT for turn-speed ramp (STRAIGHT doesn't
        #                 need deceleration), used by _plan_speed()
        #
        # Scan for the first non-LANEFOLLOW/VOID command.  Record it
        # as nav_label.  If it's LEFT/RIGHT, also use it for turn
        # anticipation.  If it's STRAIGHT or lane-change, continue
        # scanning for a LEFT/RIGHT behind it (a STRAIGHT followed
        # by LEFT means the turn is just past the junction entry).
        nav_label = RoadOption.LANEFOLLOW
        nav_command = RoadOption.LANEFOLLOW
        self._turn_distance = None
        for wp, cmd in self._wp_buffer:
            if cmd in (RoadOption.LANEFOLLOW, RoadOption.VOID):
                continue
            dist = ego_loc.distance(wp.transform.location)
            if cmd in (
                RoadOption.STRAIGHT,
                RoadOption.CHANGELANELEFT,
                RoadOption.CHANGELANERIGHT,
            ):
                # Record as nav label if it's the first actionable command
                if nav_label == RoadOption.LANEFOLLOW:
                    if dist < self.config.turn_anticipation_distance:
                        nav_label = cmd
                # Keep scanning for LEFT/RIGHT behind it
                continue
            # cmd is LEFT or RIGHT
            if dist < self.config.turn_anticipation_distance:
                nav_command = cmd
                self._turn_distance = dist
                # If no earlier actionable command was recorded, this
                # LEFT/RIGHT is the nav label too
                if nav_label == RoadOption.LANEFOLLOW:
                    nav_label = cmd
            break

        if nav_command in (RoadOption.LEFT, RoadOption.RIGHT) and nav_label in (
            RoadOption.CHANGELANELEFT,
            RoadOption.CHANGELANERIGHT,
        ):
            nav_label = RoadOption.LANEFOLLOW

        # F12: physical lane-change override from map lane IDs.
        # Route labels alone can miss unplanned physical lane shifts.
        if ego_wp is not None and self._prev_lane_wp is not None:
            same_road = ego_wp.road_id == self._prev_lane_wp.road_id
            outside_junction = (not ego_wp.is_junction) and (not self._prev_lane_wp.is_junction)
            if same_road and outside_junction and ego_wp.lane_id != self._prev_lane_wp.lane_id:
                moved_left = False
                moved_right = False
                try:
                    left_wp = self._prev_lane_wp.get_left_lane()
                    right_wp = self._prev_lane_wp.get_right_lane()
                    if (
                        left_wp is not None
                        and left_wp.road_id == ego_wp.road_id
                        and left_wp.lane_id == ego_wp.lane_id
                    ):
                        moved_left = True
                    if (
                        right_wp is not None
                        and right_wp.road_id == ego_wp.road_id
                        and right_wp.lane_id == ego_wp.lane_id
                    ):
                        moved_right = True
                except RuntimeError:
                    pass

                if moved_left:
                    nav_label = RoadOption.CHANGELANELEFT
                elif moved_right:
                    nav_label = RoadOption.CHANGELANERIGHT

        # --- Pick target waypoint at desired follow distance ---
        # Adapt follow distance smoothly using EMA-smoothed curvature.
        # Curved roads tagged LANEFOLLOW still need a shorter look-ahead
        # to prevent corner-cutting.
        curve_angle = self._route_curvature()
        # EMA update: alpha=0.3 gives ~5-frame half-life at 20 Hz
        self._smoothed_curvature += 0.3 * (curve_angle - self._smoothed_curvature)
        cur = self._smoothed_curvature

        # Interpolate between target and turn follow distance
        cfg = self.config
        blend = min(cur / 45.0, 1.0)  # 0→45° maps to 0→1
        follow_dist = cfg.target_follow_distance + blend * (
            cfg.turn_follow_distance - cfg.target_follow_distance
        )

        # When a Bezier arc is active use the arc buffer for the steering
        # target so ego follows the smooth junction path.  Nav command
        # scanning always uses _wp_buffer (route commands, not arc points).
        steer_buf = self._arc_buffer if self._arc_buffer else self._wp_buffer
        target_wp, _ = steer_buf[0]
        for wp, _ in steer_buf:
            if ego_loc.distance(wp.transform.location) >= follow_dist:
                target_wp = wp
                break

        self._prev_lane_wp = ego_wp if ego_wp is not None else target_wp
        return target_wp, nav_label, nav_command

    def _distance_to_next_junction(self, ego_loc: Any) -> float:
        """Approximate route distance from ego to the next junction waypoint."""
        if not self._wp_buffer:
            return -1.0

        accum = 0.0
        prev_loc = ego_loc
        for wp, _ in self._wp_buffer:
            wp_loc = wp.transform.location
            accum += prev_loc.distance(wp_loc)
            if bool(getattr(wp, "is_junction", False)):
                return float(accum)
            prev_loc = wp_loc
        return -1.0

    def _target_heading_error_deg(
        self,
        transform: "carla.Transform",
        target_wp: Any,
    ) -> float | None:
        """Return absolute ego-to-target heading mismatch in degrees."""
        if target_wp is None:
            return None

        ego_loc = transform.location
        target_loc = target_wp.transform.location
        dx = float(target_loc.x - ego_loc.x)
        dy = float(target_loc.y - ego_loc.y)
        if math.hypot(dx, dy) < 1e-6:
            return None

        target_yaw = math.degrees(math.atan2(dy, dx))
        ego_yaw = float(transform.rotation.yaw)
        return abs(self._wrap_angle_deg(target_yaw - ego_yaw))

    def _apply_route_continuity_watchdog(
        self,
        *,
        buf_head_dist: float,
        target_speed: float,
        mode: DrivingMode,
        speed_kmh: float,
        route_completion: float,
        heading_error_deg: float | None,
        tl_state: Any,
        tl_distance: float | None,
    ) -> tuple[float, DrivingMode]:
        """Emergency-stop on persistent route-health failure signatures.

        Two independent trip paths are monitored:
        1. Legacy corridor-collapse signature (small corridor + far target).
        2. Composite stall signature (moving + stalled route progress +
           off-route geometry), with traffic-light hold exemptions.
        """
        cfg = self.config
        if not cfg.route_watchdog_enabled:
            self._route_watchdog_consecutive_frames = 0
            self._route_watchdog_progress_stall_frames = 0
            self._route_watchdog_prev_route_completion = route_completion
            return target_speed, mode

        collapsed = self._n_corridor_segments <= int(cfg.route_watchdog_min_corridor_segments)
        far_head = buf_head_dist >= float(cfg.route_watchdog_wp_buffer_head_dist_m)
        if collapsed and far_head:
            self._route_watchdog_consecutive_frames += 1
        else:
            self._route_watchdog_consecutive_frames = 0

        prev_completion = self._route_watchdog_prev_route_completion
        if prev_completion is None:
            progress_delta = 0.0
        else:
            progress_delta = max(0.0, route_completion - prev_completion)
        self._route_watchdog_prev_route_completion = route_completion

        moving = speed_kmh >= float(cfg.route_watchdog_min_moving_speed_kmh)
        stalled = progress_delta <= float(cfg.route_watchdog_progress_epsilon)
        heading_mismatch = (
            heading_error_deg is not None
            and heading_error_deg >= float(cfg.route_watchdog_heading_mismatch_deg)
        )
        offroute = far_head or heading_mismatch

        tl_forced_hold = False
        if tl_distance is not None and 0.0 < tl_distance < cfg.traffic_light_brake_distance:
            try:
                tl_forced_hold = tl_state in (
                    carla.TrafficLightState.Red,
                    carla.TrafficLightState.Yellow,
                )
            except (AttributeError, NameError):
                # Test/runtime fallback where carla enums are unavailable.
                tl_forced_hold = tl_state in (2, 3)

        exempt_hold = (
            tl_forced_hold and speed_kmh <= float(cfg.route_watchdog_tl_hold_exempt_speed_kmh)
        )

        if moving and stalled and offroute and not exempt_hold:
            self._route_watchdog_progress_stall_frames += 1
        else:
            self._route_watchdog_progress_stall_frames = 0

        min_frames = max(1, int(cfg.route_watchdog_min_frames))
        stall_min_frames = max(1, int(cfg.route_watchdog_stall_min_frames))
        legacy_trip = self._route_watchdog_consecutive_frames >= min_frames
        stall_trip = self._route_watchdog_progress_stall_frames >= stall_min_frames
        if not (legacy_trip or stall_trip):
            return target_speed, mode

        self._frame_route_watchdog_triggered = 1
        if self._planner_policy_reason == PlannerPolicyReason.NONE:
            self._planner_policy_reason = PlannerPolicyReason.ROUTE_CONTINUITY_WATCHDOG
        return 0.0, DrivingMode.EMERGENCY_STOP

    @staticmethod
    def _is_lane_change_command(cmd: RoadOption) -> bool:
        return cmd in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT)

    def _lane_supports_turn_command(self, cmd: RoadOption, ego_loc: Any) -> bool:
        """Return True when near-term route topology supports the requested turn.

        The old immediate ``ego_wp.next(2.0)`` check under-reported valid turns
        on approach lanes because CARLA often exposes only straight successors
        until the ego is almost inside the junction connector.  Here we validate
        against short route lookahead only when a junction is imminent.
        """
        if cmd not in (RoadOption.LEFT, RoadOption.RIGHT):
            return True

        cfg = self.config
        window_m = max(0.0, float(cfg.planner_lane_change_freeze_distance_m))
        if window_m <= 0.0:
            return True

        # Distance-gated: do not enforce turn-lane legality far from the turn.
        if self._dist_to_next_junction <= 0.0 or self._dist_to_next_junction > window_m:
            return True

        # Scan short route lookahead for the first explicit turn command.
        first_turn: RoadOption | None = None
        accum = 0.0
        prev_loc = ego_loc
        preview_limit = window_m + 5.0
        for rwp, rcmd in self._route_waypoint_sources(lookahead=120):
            wp_loc = rwp.transform.location
            try:
                accum += prev_loc.distance(wp_loc)
            except (AttributeError, RuntimeError, TypeError):
                return True
            prev_loc = wp_loc
            if accum > preview_limit:
                break
            if rcmd in (RoadOption.LEFT, RoadOption.RIGHT):
                first_turn = rcmd
                break

        # Fail open when route preview is ambiguous.
        if first_turn is None:
            return True
        return first_turn == cmd

    def _is_mandatory_exit_lane_change(self, ego_loc: Any) -> bool:
        """Heuristic: lane change is mandatory only within a local lookahead window."""
        window_m = max(0.0, float(self.config.planner_mandatory_exit_window_m))
        if window_m <= 0.0:
            return False

        seen_lane_change = False
        accum = 0.0
        prev_loc = ego_loc
        for _wp, cmd in self._wp_buffer:
            wp_loc = _wp.transform.location
            try:
                accum += prev_loc.distance(wp_loc)
            except (AttributeError, TypeError):
                return False
            prev_loc = wp_loc
            if accum > window_m:
                break

            if cmd in (RoadOption.LANEFOLLOW, RoadOption.VOID):
                continue
            if self._is_lane_change_command(cmd):
                seen_lane_change = True
                continue
            if seen_lane_change:
                return cmd in (RoadOption.LEFT, RoadOption.RIGHT)
            return False
        return False

    def _is_exit_only_lane_for_straight(self, ego_wp: Any) -> bool:
        """Heuristic exit-only check: no straight continuation from current lane."""
        try:
            nexts = ego_wp.next(2.0)
        except (AttributeError, RuntimeError, TypeError):
            return False
        if not nexts:
            return False

        signed_angles = [
            self._transition_signed_angle_deg(ego_wp, cand)
            for cand in nexts
        ]
        valid_angles = [a for a in signed_angles if a is not None]
        if not valid_angles:
            return False

        has_straight = any(abs(a) <= 30.0 for a in valid_angles)
        has_turn = any(abs(a) > 35.0 for a in valid_angles)
        return has_turn and not has_straight

    def _is_roundabout_context(self, ego_wp: Any) -> bool:
        """Heuristic roundabout context using local branching near junctions."""
        if ego_wp is None:
            return False

        cfg = self.config
        near_junction = bool(getattr(ego_wp, "is_junction", False))
        if not near_junction:
            near_junction = (
                self._dist_to_next_junction > 0.0
                and self._dist_to_next_junction <= float(cfg.planner_roundabout_context_window_m)
            )
        if not near_junction:
            return False

        try:
            nexts = ego_wp.next(2.0)
        except (AttributeError, RuntimeError, TypeError):
            return bool(getattr(ego_wp, "is_junction", False))
        if not nexts:
            return bool(getattr(ego_wp, "is_junction", False))

        signed_angles = [self._transition_signed_angle_deg(ego_wp, cand) for cand in nexts]
        valid_angles = [a for a in signed_angles if a is not None]
        if not valid_angles:
            return bool(getattr(ego_wp, "is_junction", False))

        has_left = any(a > 35.0 for a in valid_angles)
        has_right = any(a < -35.0 for a in valid_angles)
        has_straight = any(abs(a) <= 30.0 for a in valid_angles)
        return (has_left and has_right) or (has_straight and (has_left or has_right))

    def _apply_lane_change_policy(self, nav_label: RoadOption, ego_wp: Any, ego_loc: Any) -> RoadOption:
        """Apply BR-16/17/18 lane-change policy guardrails to nav labels."""
        cfg = self.config
        self._dist_to_next_junction = self._distance_to_next_junction(ego_loc)
        if not cfg.planner_policy_enabled:
            return nav_label

        out_label = nav_label
        mandatory_exit = self._is_lane_change_command(nav_label) and self._is_mandatory_exit_lane_change(
            ego_loc
        )
        if mandatory_exit:
            self._planner_policy_exception = PlannerPolicyException.MANDATORY_EXIT

        in_freeze_window = (
            self._dist_to_next_junction > 0.0
            and self._dist_to_next_junction <= cfg.planner_lane_change_freeze_distance_m
        )

        if self._is_lane_change_command(out_label) and in_freeze_window and not mandatory_exit:
            self._planner_policy_reason = PlannerPolicyReason.PRE_JUNCTION_LANE_CHANGE_BLOCKED
            self._frame_pre_junction_lane_change_violation = 1
            out_label = RoadOption.LANEFOLLOW

        dist_since_last_lane_change = self._policy_distance_traveled - self._policy_last_lane_change_distance
        if self._is_lane_change_command(out_label) and not mandatory_exit:
            if dist_since_last_lane_change < cfg.planner_lane_change_consecutive_window_m:
                self._planner_policy_reason = PlannerPolicyReason.CONSECUTIVE_LANE_CHANGE_BLOCKED
                self._frame_consecutive_lane_change_violation = 1
                out_label = RoadOption.LANEFOLLOW

        if self._is_lane_change_command(out_label):
            self._policy_last_lane_change_distance = self._policy_distance_traveled

        if (
            cfg.planner_roundabout_commitment_enabled
            and self._is_lane_change_command(out_label)
            and self._is_roundabout_context(ego_wp)
        ):
            dist_since_roundabout_lane_change = (
                self._policy_distance_traveled - self._policy_last_roundabout_lane_change_distance
            )
            if dist_since_roundabout_lane_change < cfg.planner_roundabout_lane_change_cooldown_m:
                self._planner_policy_reason = PlannerPolicyReason.ROUNDABOUT_LANE_CHANGE_BLOCKED
                self._frame_roundabout_lane_change_violation = 1
                out_label = RoadOption.LANEFOLLOW
            else:
                self._policy_last_roundabout_lane_change_distance = self._policy_distance_traveled

        if (
            cfg.planner_turn_lane_validation_enabled
            and out_label in (RoadOption.LEFT, RoadOption.RIGHT)
            and ego_wp is not None
            and not self._lane_supports_turn_command(out_label, ego_loc)
        ):
            self._planner_policy_reason = PlannerPolicyReason.TURN_LANE_LEGALITY_BLOCKED
            self._frame_turn_lane_legality_violation = 1
            out_label = RoadOption.LANEFOLLOW

        in_exit_window = (
            out_label == RoadOption.STRAIGHT
            and self._dist_to_next_junction > 0.0
            and self._dist_to_next_junction <= cfg.planner_exit_lane_leave_distance_m
        )
        if in_exit_window and ego_wp is not None and self._is_exit_only_lane_for_straight(ego_wp):
            if self._planner_policy_reason == PlannerPolicyReason.NONE:
                self._planner_policy_reason = PlannerPolicyReason.STRAIGHT_EXIT_LANE_WARNING
            self._frame_straight_exit_lane_violation = 1

        return out_label

    # ------------------------------------------------------------------
    # Internal — speed planning
    # ------------------------------------------------------------------

    def _plan_speed(
        self,
        speed_kmh: float,
        nav_command: RoadOption,
        tl_state: Any,
        tl_distance: float | None,
        ss_distance: float | None,
        ss_committed: bool,
        lead_distance: float | None,
        lead_speed: float | None,
        lead_lateral: float | None,
        ped_dist: float | None,
        base_target: float | None = None,
        ys_distance: float | None = None,
        ys_has_conflict: bool = False,
    ) -> Tuple[float, DrivingMode]:
        """Determine target speed and driving mode from hazards."""
        cfg = self.config
        target = base_target if base_target is not None else cfg.target_speed
        mode = DrivingMode.FOLLOW_LANE

        # --- Curvature (route bends) ---
        curve_angle = self._route_curvature()
        if curve_angle > cfg.curvature_threshold_deg:
            # Scale reduction with curvature intensity
            curvature_factor = max(
                cfg.curvature_speed_factor,
                1.0 - (curve_angle / 90.0) * 0.5,
            )
            target = min(target, target * curvature_factor)

        # --- Upcoming turn ---
        # Use a kinematic deceleration profile so the car arrives at
        # the junction ENTRY at the desired turn speed, rather than
        # slamming speed down the instant the turn is anticipated.
        #   turn_target = base_target × turn_speed_factor
        # Outside the junction, the target speed is the *higher* of
        # the kinematic profile v=sqrt(2·a·d) and the turn target.
        # Inside the junction, the car simply holds the turn target
        # (no further reduction) and accelerates normally on exit.
        ego_wp = self.map.get_waypoint(
            self.vehicle.get_transform().location,
        )
        if nav_command in (RoadOption.LEFT, RoadOption.RIGHT):
            turn_target = target * cfg.turn_speed_factor
            at_junction = ego_wp is not None and ego_wp.is_junction

            if at_junction:
                # Inside junction — use kinematic ramp from remaining
                # turn distance so target transitions smoothly from
                # the APPROACHING_TURN speed to turn_target instead
                # of a cliff-drop to turn_target in a single frame.
                mode = DrivingMode.TURNING
                turn_dist = self._turn_distance
                if turn_dist is not None and turn_dist > 0:
                    v_turn_ms = turn_target / 3.6
                    d = max(turn_dist, 0.0)
                    v_ms = math.sqrt(
                        2.0 * cfg.turn_comfort_decel * d + v_turn_ms**2,
                    )
                    target = min(target, v_ms * 3.6)
                else:
                    target = min(target, turn_target)
            else:
                # Approaching — kinematic ramp to arrive at turn_target
                mode = DrivingMode.APPROACHING_TURN
                turn_dist = self._turn_distance
                if turn_dist is not None and turn_dist > 0:
                    # v = sqrt(2·a·d + v_turn²) — arrive at turn_target
                    # Uses turn_comfort_decel (gentler than TL) so braking
                    # starts earlier and produces a smooth ramp.
                    v_turn_ms = turn_target / 3.6
                    d = max(turn_dist, 0.0)
                    v_ms = math.sqrt(
                        2.0 * cfg.turn_comfort_decel * d + v_turn_ms**2,
                    )
                    kinematic = v_ms * 3.6

                    # Linear-blend ceiling: guarantees target decreases
                    # from base down to turn_target as distance shrinks
                    # from anticipation_distance to 0.  Without this,
                    # the kinematic ramp stays above base_target for
                    # most of the approach range (e.g. 33 km/h at 25 m
                    # on a 27 km/h road), so the car never actually
                    # slows during APPR.
                    blend_frac = min(d / cfg.turn_anticipation_distance, 1.0)
                    linear_cap = turn_target + (target - turn_target) * blend_frac

                    target = min(target, kinematic, linear_cap)
                else:
                    target = min(target, turn_target)

        # --- Traffic light ---
        # TL stop takes priority when the ramp demands a speed lower
        # than whatever the turn section already set.  Without this,
        # a turn target of ~15 km/h masks the TL ramp (which is still
        # above 15 km/h at 20-40 m distance), so the car approaches
        # a red TL at turn speed and has to slam at the last moment.
        speed_ms = speed_kmh / 3.6
        # F9b: a full stop invalidates prior yellow dilemma commitment.
        # Once speed reaches zero, treat TL handling as a fresh decision.
        if self._tl_proceed_committed and speed_kmh < 0.5:
            self._tl_proceed_committed = False
        if tl_state is not None and tl_distance is not None:
            if tl_state == carla.TrafficLightState.Green:
                # TL is green — clear any yellow commitment.
                self._tl_proceed_committed = False
            elif tl_state in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow):
                if 0 < tl_distance < cfg.traffic_light_brake_distance:
                    effective_tl_stop = cfg.traffic_light_stop_distance + self._ego_half_length

                    # Yellow dilemma zone: commit to proceed only if
                    # (a) the car cannot stop comfortably AND
                    # (b) it can clear past the stop line + margin
                    #     at current speed.  Without (b), a car that
                    #     "can't stop comfortably" at 30+ m from the
                    #     line would proceed into the junction and
                    #     stop mid-intersection when the light turns
                    #     red. The clearance check ensures we only
                    #     commit when we'll actually make it through.
                    proceed_yellow = False
                    if tl_state == carla.TrafficLightState.Yellow:
                        d_to_stop = max(tl_distance - effective_tl_stop, 0.0)
                        min_brake_dist = speed_ms**2 / (2.0 * cfg.tl_commit_decel)
                        junction_clearance = self._estimate_junction_clearance_distance()
                        min_clearance = max(
                            cfg.tl_clearance_margin,
                            cfg.tl_half_junction_clearance_factor * junction_clearance,
                        )
                        within_commit_window = d_to_stop <= cfg.tl_commit_max_stop_distance
                        # Can we clear past the stop line at current speed?
                        can_clear = speed_ms > 0.5 and tl_distance + min_clearance < min_brake_dist
                        if within_commit_window and d_to_stop < min_brake_dist and can_clear:
                            proceed_yellow = True
                            self._tl_proceed_committed = True

                    # Honour a previous yellow commitment through the
                    # red phase — the car already passed the point of
                    # no return and should not stop mid-intersection.
                    if self._tl_proceed_committed:
                        proceed_yellow = True

                    if not proceed_yellow:
                        if tl_distance < effective_tl_stop:
                            mode = DrivingMode.TRAFFIC_LIGHT_STOP
                            target = 0.0
                        else:
                            # Kinematic profile: v = sqrt(2 * a * d)
                            d = max(tl_distance - effective_tl_stop, 0.0)
                            v_ms = math.sqrt(2.0 * cfg.tl_comfort_decel * d)
                            tl_target = v_ms * 3.6  # m/s → km/h
                            # Keep traffic-light arbitration active in
                            # red/yellow brake window even when turn logic
                            # already requested a lower speed.
                            mode = DrivingMode.TRAFFIC_LIGHT_STOP
                            target = min(target, tl_target)

        # --- Stop sign ---
        if ss_committed:
            mode = DrivingMode.STOP_SIGN_STOP
            target = 0.0
        elif ss_distance is not None and 0 < ss_distance < cfg.stop_sign_brake_distance:
            mode = DrivingMode.STOP_SIGN_STOP
            effective_ss_stop = cfg.stop_sign_stop_distance + self._ego_half_length
            d = max(ss_distance - effective_ss_stop, 0.0)
            v_ms = math.sqrt(2.0 * cfg.stop_sign_comfort_decel * d)
            # Floor at creep speed so the car rolls into the trigger
            # volume instead of stopping short of it (Phase 1 → Phase 2
            # handoff).  Once trigger_volume.contains() fires, Phase 2
            # commits to target=0 for a full stop.
            ss_target = max(v_ms * 3.6, cfg.stop_sign_creep_speed)
            target = min(target, ss_target)

        # --- Yield sign ---
        # Unlike stop signs, yield does not require a full stop when the
        # intersection is clear.  The ego slows to yield_slow_speed while
        # approaching, and stops only if cross-traffic conflict exists.
        if ys_distance is not None and 0 < ys_distance < cfg.yield_scan_distance:
            if ys_has_conflict:
                # Conflict detected — kinematic ramp to 0, same as stop sign
                mode = DrivingMode.YIELD_STOP
                d = max(ys_distance - self._ego_half_length, 0.0)
                v_ms = math.sqrt(2.0 * cfg.yield_comfort_decel * d)
                target = min(target, v_ms * 3.6)
            else:
                # No conflict — slow to creep speed and proceed
                d = max(ys_distance - self._ego_half_length, 0.0)
                v_ms = math.sqrt(2.0 * cfg.yield_comfort_decel * d)
                ys_target = max(v_ms * 3.6, cfg.yield_slow_speed)
                if ys_target < target:
                    mode = DrivingMode.YIELD_STOP
                    target = ys_target

        # --- Lead vehicle ---
        # Scale safety distances with current speed (kinematic braking
        # envelope).  At low speed the fixed minimums apply; at high
        # speed the distances stretch to preserve reaction time.
        dyn_hazard = max(
            cfg.vehicle_hazard_distance,
            speed_ms * cfg.vehicle_reaction_time,
        )
        dyn_emergency = max(
            cfg.vehicle_emergency_distance,
            speed_ms * cfg.vehicle_emergency_time,
        )
        tl_forces_stop = (
            tl_state in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow)
            and tl_distance is not None
            and 0 < tl_distance < cfg.traffic_light_brake_distance
            and not self._tl_proceed_committed
        )
        turning_block_active = False

        if lead_distance is not None:
            if mode == DrivingMode.TURNING and lead_speed is not None and lead_speed < 0.5:
                # During turns, only treat near-centerline stopped vehicles as blockers.
                # Off-axis stopped vehicles (adjacent lane/shoulder) should not deadlock us.
                ego_half_width = self.vehicle.bounding_box.extent.y
                turning_block_lateral = max(
                    ego_half_width + 0.35,
                    self._cached_half_lane,
                )
                if lead_lateral is not None and abs(lead_lateral) > turning_block_lateral:
                    lead_distance = None
                else:
                    turning_block_active = True

        if lead_distance is not None:
            if turning_block_active and not tl_forces_stop:
                mode = DrivingMode.EMERGENCY_STOP
                target = 0.0
            elif (
                tl_forces_stop
                and lead_speed is not None
                and lead_speed < 0.5
                and lead_distance >= dyn_emergency
            ):
                # A stopped lead vehicle ahead of a red/yellow light should
                # not escalate to EMERGENCY_STOP when we're still outside
                # the emergency range; the traffic light owns the stop.
                pass
            elif lead_distance < dyn_emergency:
                mode = DrivingMode.EMERGENCY_STOP
                target = 0.0
            elif tl_forces_stop:
                # Red/yellow traffic lights outrank lead-vehicle following.
                # Keep the traffic-light stop target already computed above.
                pass
            elif lead_distance < dyn_hazard:
                mode = DrivingMode.VEHICLE_FOLLOWING
                # Kinematic braking profile: v = sqrt(2 * a * d)
                # Same physics-based curve used for traffic-light stops.
                # Produces smooth, progressive braking that matches how
                # a professional driver modulates brake pressure.
                d = max(lead_distance - dyn_emergency, 0.0)
                v_ms = math.sqrt(2.0 * cfg.vehicle_comfort_decel * d)
                target = min(target, v_ms * 3.6)  # m/s → km/h

        # --- Pedestrians ---
        if ped_dist is not None:
            ped_emergency_dist = cfg.pedestrian_emergency_distance + 5.0
            if ped_dist < ped_emergency_dist:
                mode = DrivingMode.EMERGENCY_STOP
                target = 0.0
            elif ped_dist < cfg.pedestrian_hazard_distance:
                mode = DrivingMode.PEDESTRIAN_STOP
                target = min(target, 8.0)  # creep at 8 km/h

        # --- Rate limiter ---
        # Prevent target from dropping more than max_target_drop_per_frame
        # per tick.  This smooths mode-transition jumps (e.g. APPR→TL)
        # so the overshoot override never fires during normal driving.
        # Emergency stops (target ≈ 0) bypass the limiter entirely.
        if target > 0.0:
            max_drop = cfg.max_target_drop_per_frame
            target = max(target, self._prev_target - max_drop)
        self._prev_target = target

        return max(target, 0.0), mode

    def _route_curvature(self) -> float:
        """Estimate curvature (degrees) over the next few waypoints."""
        if len(self._wp_buffer) < 3:
            return 0.0
        wps = [wp for wp, _ in list(self._wp_buffer)[:6]]
        total_angle = 0.0
        for i in range(len(wps) - 2):
            a = wps[i].transform.location
            b = wps[i + 1].transform.location
            c = wps[i + 2].transform.location
            v1 = (b.x - a.x, b.y - a.y)
            v2 = (c.x - b.x, c.y - b.y)
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            angle = abs(math.degrees(math.atan2(cross, dot)))
            total_angle += angle
        return total_angle

    # ------------------------------------------------------------------
    # Internal — controllers
    # ------------------------------------------------------------------

    def _apply_gain_blending(self) -> None:
        """Smoothly blend PID gains between straight and turn presets.

        Uses the EMA-smoothed curvature as a blend factor instead of a
        binary switch.  This avoids the abrupt gain jumps that cause
        jerky transitions between straight and turn behaviour.

        The blend ramp (0–45°) matches the curvature range used by the
        adaptive follow-distance logic so both respond in sync.
        """
        cfg = self.config
        blend = min(self._smoothed_curvature / 45.0, 1.0)

        def _lerp(a: float, b: float) -> float:
            return a + blend * (b - a)

        self.steer_pid.kp = _lerp(cfg.steer_kp, cfg.turn_steer_kp)
        self.steer_pid.ki = _lerp(cfg.steer_ki, cfg.turn_steer_ki)
        self.steer_pid.kd = _lerp(cfg.steer_kd, cfg.turn_steer_kd)
        self.speed_pid.kp = _lerp(cfg.speed_kp, cfg.turn_speed_kp)
        self.speed_pid.ki = _lerp(cfg.speed_ki, cfg.turn_speed_ki)
        self.speed_pid.kd = _lerp(cfg.speed_kd, cfg.turn_speed_kd)

    def _lateral_control(
        self,
        transform: "carla.Transform",
        target_wp: "carla.Waypoint",
    ) -> float:
        """PID steering on heading error.

        Cross-track correction was previously added here but removed
        because the raw 2-D cross product is unbounded and jumps
        discontinuously when the target waypoint pops, causing
        compounding steering spikes that lead to collision within
        ~80 frames.  A proper Stanley-style controller would fix
        this but is out of scope for phase 1.
        """
        ego = transform.location
        tgt = target_wp.transform.location

        # Desired heading
        target_yaw = math.degrees(math.atan2(tgt.y - ego.y, tgt.x - ego.x))

        # Heading error wrapped to [−180, 180]
        error = target_yaw - transform.rotation.yaw
        error = (error + 180) % 360 - 180

        # Normalise so ±90° → ±1.0
        steer = self.steer_pid.step(error / 90.0)
        return max(-1.0, min(1.0, steer))

    def _longitudinal_control(
        self,
        speed_kmh: float,
        target_speed: float,
        dt: float,
    ) -> Tuple[float, float]:
        """PID throttle / brake on the speed error.

        Produces proportional brake in the final 0-5 km/h before a
        full stop, giving the training signal a smooth gradient
        instead of a binary 0/1 cliff.  Above 5 km/h (and for
        overshoot situations) the behaviour is identical to the
        original binary overrides to preserve braking authority.
        """
        # P4-C: raw km/h error — no normalisation.  Gains are tuned
        # for this scale (kp ≈ 0.05) so the car can reach target speed.
        error = target_speed - speed_kmh
        output = self.speed_pid.step(error, dt)

        if output >= 0:
            throttle = min(output, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(abs(output), 1.0)

        # --- Override 1: target ≈ 0 (red light, stop sign, emergency) ---
        # Above 5 km/h: full brake (same as original, decisive stop).
        # Below 5 km/h: linear ramp so training data gets a gradient.
        if target_speed < 1.0:
            throttle = 0.0
            if speed_kmh > 5.0:
                brake = 1.0
            elif speed_kmh < 0.5:
                brake = 0.5  # firm hold when already stopped
            else:
                brake = speed_kmh / 5.0  # gradient in 0.5–5 km/h transition

        # --- Override 2: large overshoot safety net ---
        # Threshold kept at 10 km/h (original value).  Proportional
        # response so training data is not pure binary; full brake
        # at 15+ km/h over target.
        elif speed_kmh > target_speed + 10.0:
            overshoot = speed_kmh - target_speed
            throttle = 0.0
            brake = min(overshoot / 15.0, 1.0)

        return throttle, brake

    # ------------------------------------------------------------------
    # Internal — privileged perception
    # ------------------------------------------------------------------

    def _stored_tl_stop_plausibility(
        self,
        transform: "carla.Transform",
        stop_loc: Any,
    ) -> tuple[bool, float | None, float, float]:
        """Check whether a stored TL stop-line still plausibly matches ego lane.

        Returns ``(is_plausible, yaw_diff_deg, lateral_offset_m, lateral_limit_m)``.
        ``yaw_diff_deg`` can be ``None`` when stop-line waypoint lookup is unavailable.
        """
        ego_loc = transform.location
        dx = stop_loc.x - ego_loc.x
        dy = stop_loc.y - ego_loc.y

        right = transform.get_right_vector()
        lateral_offset = float(right.x * dx + right.y * dy)

        lane_width = float(self.config.lane_half_width * 2.0)
        try:
            ego_wp = self.map.get_waypoint(ego_loc)
        except RuntimeError:
            ego_wp = None
        if ego_wp is not None:
            lane_width = float(getattr(ego_wp, "lane_width", lane_width))

        lateral_limit = (
            0.5 * lane_width + self.config.traffic_light_stored_stop_lateral_margin
        )

        yaw_diff: float | None = None
        try:
            stop_wp = self.map.get_waypoint(stop_loc)
        except RuntimeError:
            stop_wp = None
        if stop_wp is not None:
            stop_yaw = float(stop_wp.transform.rotation.yaw)
            ego_yaw = float(transform.rotation.yaw)
            yaw_diff = abs((stop_yaw - ego_yaw + 180.0) % 360.0 - 180.0)

        yaw_ok = (
            yaw_diff is None
            or yaw_diff <= self.config.traffic_light_stored_stop_max_yaw_diff_deg
        )
        lateral_ok = abs(lateral_offset) <= lateral_limit
        return yaw_ok and lateral_ok, yaw_diff, lateral_offset, lateral_limit

    def _check_traffic_light(
        self,
        transform: "carla.Transform",
    ) -> Tuple[Any, float | None]:
        """Return (state, distance-to-stop-line) of the relevant TL.

        Three-tier strategy:
        1. **Persistent memory** — if we were already waiting for a
           specific red/yellow light, keep checking it until it turns
           green.  Prevents "forgetting" about it when the ego's
           road_id changes at the junction boundary.
        2. **Fast path** — CARLA's built-in ``is_at_traffic_light()``.
        3. **Junction-aware scan** — stop-line distance via
              ``get_stop_waypoints()`` (preferred) then
              ``get_affected_lane_waypoints()`` for early braking.
        """
        ego_loc = transform.location

        # ---- Tier 1: persistent TL memory ----
        if self._tl_waiting is not None:
            tl_actor = self.world.get_actor(self._tl_waiting)
            if tl_actor is None:
                logger.debug("T1: actor %d gone, clearing", self._tl_waiting)
                self._tl_waiting = None
                self._tl_stop_loc = None
                self._tl_proceed_committed = False
            else:
                tl_loc = tl_actor.get_transform().location
                fwd = transform.get_forward_vector()

                # Use stop-line geometry whenever available. Pole-based
                # stale clear can flip incorrectly on curved approaches.
                if self._tl_stop_loc is not None:
                    stop_dist = ego_loc.distance(self._tl_stop_loc)
                    sdx = self._tl_stop_loc.x - ego_loc.x
                    sdy = self._tl_stop_loc.y - ego_loc.y
                    stop_behind = (fwd.x * sdx + fwd.y * sdy) <= 0
                    stale_far = stop_dist > self.config.traffic_light_brake_distance * 1.5
                    plausible, yaw_diff, lat_off, lat_lim = self._stored_tl_stop_plausibility(
                        transform,
                        self._tl_stop_loc,
                    )
                    yaw_txt = f"{yaw_diff:.1f}" if yaw_diff is not None else "None"
                    if stale_far or stop_behind or not plausible:
                        # Curved approaches can transiently violate the
                        # lateral plausibility gate even when the same TL
                        # still controls us.  Before clearing memory,
                        # re-derive stop-line geometry from the same TL.
                        recovered_stop_dist, recovered_stop_loc = None, None
                        if not stale_far and not stop_behind:
                            recovered_stop_dist, recovered_stop_loc = self._stop_line_distance(
                                tl_actor,
                                transform,
                            )
                        if recovered_stop_loc is not None and recovered_stop_dist is not None:
                            self._tl_stop_loc = recovered_stop_loc
                            logger.debug(
                                "T1-STORED: recovered id=%d state=%s stop_dist=%.1f "
                                "old_stop_dist=%.1f yaw_diff=%s lat=%.2f lat_lim=%.2f",
                                self._tl_waiting,
                                tl_actor.state,
                                recovered_stop_dist,
                                stop_dist,
                                yaw_txt,
                                lat_off,
                                lat_lim,
                            )
                            return tl_actor.state, recovered_stop_dist

                        # Grace-hold memory through short geometry lapses on
                        # red/yellow while the stored stop line is still ahead.
                        if (
                            not stale_far
                            and not stop_behind
                            and tl_actor.state
                            in (
                                carla.TrafficLightState.Red,
                                carla.TrafficLightState.Yellow,
                            )
                            and stop_dist <= self.config.traffic_light_brake_distance
                        ):
                            logger.debug(
                                "T1-STORED: grace hold id=%d state=%s stop_dist=%.1f "
                                "plausible=%s yaw_diff=%s lat=%.2f lat_lim=%.2f",
                                self._tl_waiting,
                                tl_actor.state,
                                stop_dist,
                                plausible,
                                yaw_txt,
                                lat_off,
                                lat_lim,
                            )
                            return tl_actor.state, stop_dist

                        logger.debug(
                            "T1-STORED: stale clear id=%d stop_dist=%.1f "
                            "behind=%s plausible=%s yaw_diff=%s lat=%.2f lat_lim=%.2f",
                            self._tl_waiting,
                            stop_dist,
                            stop_behind,
                            plausible,
                            yaw_txt,
                            lat_off,
                            lat_lim,
                        )
                        self._tl_waiting = None
                        self._tl_stop_loc = None
                        self._tl_proceed_committed = False
                        # Fall through to Tier 2 / Tier 3.
                    else:
                        logger.debug(
                            "T1-STORED: id=%d state=%s stop_dist=%.1f pole_dist=%.1f "
                            "yaw_diff=%s lat=%.2f lat_lim=%.2f",
                            self._tl_waiting,
                            tl_actor.state,
                            stop_dist,
                            ego_loc.distance(tl_loc),
                            yaw_txt,
                            lat_off,
                            lat_lim,
                        )
                        return tl_actor.state, stop_dist
                else:
                    # Recover stop-line location if possible.
                    recovered_stop_dist, stop_loc = self._stop_line_distance(
                        tl_actor,
                        transform,
                    )
                    if stop_loc is not None and recovered_stop_dist is not None:
                        self._tl_stop_loc = stop_loc
                        sdx = stop_loc.x - ego_loc.x
                        sdy = stop_loc.y - ego_loc.y
                        stop_behind = (fwd.x * sdx + fwd.y * sdy) <= 0
                        stale_far = (
                            recovered_stop_dist > self.config.traffic_light_brake_distance * 1.5
                        )
                        plausible, yaw_diff, lat_off, lat_lim = self._stored_tl_stop_plausibility(
                            transform,
                            stop_loc,
                        )
                        yaw_txt = f"{yaw_diff:.1f}" if yaw_diff is not None else "None"
                        if stale_far or stop_behind or not plausible:
                            if (
                                not stale_far
                                and not stop_behind
                                and tl_actor.state
                                in (
                                    carla.TrafficLightState.Red,
                                    carla.TrafficLightState.Yellow,
                                )
                                and recovered_stop_dist <= self.config.traffic_light_brake_distance
                            ):
                                logger.debug(
                                    "T1-RECOVERED: grace hold id=%d state=%s stop_dist=%.1f "
                                    "plausible=%s yaw_diff=%s lat=%.2f lat_lim=%.2f",
                                    self._tl_waiting,
                                    tl_actor.state,
                                    recovered_stop_dist,
                                    plausible,
                                    yaw_txt,
                                    lat_off,
                                    lat_lim,
                                )
                                return tl_actor.state, recovered_stop_dist

                            logger.debug(
                                "T1-RECOVERED: stale clear id=%d stop_dist=%.1f "
                                "behind=%s plausible=%s yaw_diff=%s lat=%.2f lat_lim=%.2f",
                                self._tl_waiting,
                                recovered_stop_dist,
                                stop_behind,
                                plausible,
                                yaw_txt,
                                lat_off,
                                lat_lim,
                            )
                            self._tl_waiting = None
                            self._tl_stop_loc = None
                            self._tl_proceed_committed = False
                        else:
                            logger.debug(
                                "T1-RECOVERED: id=%d state=%s stop_dist=%.1f "
                                "yaw_diff=%s lat=%.2f lat_lim=%.2f",
                                self._tl_waiting,
                                tl_actor.state,
                                recovered_stop_dist,
                                yaw_txt,
                                lat_off,
                                lat_lim,
                            )
                            return tl_actor.state, recovered_stop_dist
                    else:
                        # Stop-line unavailable: do not drive stop behavior
                        # from pole distance (can place stop on crosswalk).
                        # Keep memory and defer to Tier 2/3 each tick.
                        dist = ego_loc.distance(tl_loc)
                        dx = tl_loc.x - ego_loc.x
                        dy = tl_loc.y - ego_loc.y
                        tl_behind = (fwd.x * dx + fwd.y * dy) < 0
                        if dist > self.config.traffic_light_brake_distance * 1.5 or tl_behind:
                            logger.debug(
                                "T1-POLE-FALLBACK: stale clear id=%d dist=%.1f behind=%s",
                                self._tl_waiting,
                                dist,
                                tl_behind,
                            )
                            self._tl_waiting = None
                            self._tl_stop_loc = None
                            self._tl_proceed_committed = False
                        else:
                            logger.debug(
                                "T1-POLE-FALLBACK: id=%d no stop-line, "
                                "deferring to T2/T3 pole_dist=%.1f",
                                self._tl_waiting,
                                dist,
                            )
                            # Fall through to Tier 2 / Tier 3.

        # ---- Tier 2: CARLA built-in (close range) ----
        if self.vehicle.is_at_traffic_light():
            tl = self.vehicle.get_traffic_light()
            if tl is not None:
                # Lane guard: only react if this TL actually controls
                # our lane.  Without this, cross-street TLs whose
                # trigger volumes extend into the junction cause false
                # RED detection during turns.
                stop_dist, stop_loc = self._stop_line_distance(tl, transform)
                logger.debug(
                    "T2: is_at_tl=True tl_id=%d state=%s " "stop_dist=%s",
                    tl.id,
                    tl.state,
                    f"{stop_dist:.1f}" if stop_dist is not None else "None",
                )
                if stop_dist is not None:
                    self._tl_waiting = tl.id
                    self._tl_stop_loc = stop_loc
                    return tl.state, stop_dist
                # TL doesn't match our lane — fall through to Tier 3.

        # ---- Tier 3: junction-aware scan ----
        ego_wp = self.map.get_waypoint(ego_loc)
        if ego_wp is None:
            return None, None

        fwd = transform.get_forward_vector()
        # Use a generous pre-filter on pole distance (pole can be far
        # from stop line); the real threshold is applied to the stop-line.
        pole_cutoff = self.config.traffic_light_brake_distance * 2

        best_state: Any = None
        best_dist: float | None = None
        best_tl_id: int | None = None
        best_stop_loc = None

        # Collect ALL road IDs along the planned route so we can
        # detect TLs whose stop line sits on a road segment the
        # ego hasn't reached yet (typically 30-60 m ahead).  The
        # previous single-element look-ahead only matched TLs on
        # the ego's current or next road segment (~4-12 m range).
        route_road_ids: set[int] = {ego_wp.road_id}
        route_lane_keys: set[tuple[int, int]] = {(ego_wp.road_id, ego_wp.lane_id)}
        route_sources = self._route_waypoint_sources()
        for _bwp, _bcmd in route_sources:
            route_road_ids.add(_bwp.road_id)
            route_lane_keys.add((_bwp.road_id, _bwp.lane_id))

        match_dist = self.config.traffic_light_route_match_distance
        match_yaw = self.config.traffic_light_route_match_yaw_deg

        if self._route_tl_candidates:
            candidate_tls = [self.world.get_actor(tl_id) for tl_id in self._route_tl_candidates]
            tls = [tl for tl in candidate_tls if tl is not None]
        else:
            tls = self.world.get_actors().filter("*traffic_light*")

        for tl in tls:
            tl_loc = tl.get_transform().location
            pole_dist = ego_loc.distance(tl_loc)
            if pole_dist > pole_cutoff:
                continue

            # Ahead of us? (positive dot product with forward vector)
            dx = tl_loc.x - ego_loc.x
            dy = tl_loc.y - ego_loc.y
            if fwd.x * dx + fwd.y * dy <= 0:
                continue

            # ---- Lane matching via affected waypoints ----
            affects_us = False
            stop_line_dist: float | None = None
            stop_loc_candidate = None

            try:
                for awp in tl.get_affected_lane_waypoints():
                    if awp.road_id not in route_road_ids:
                        continue

                    lane_semantic_match = (
                        (awp.road_id, awp.lane_id) in route_lane_keys
                        or (awp.road_id, awp.lane_id - 1) in route_lane_keys
                        or (awp.road_id, awp.lane_id + 1) in route_lane_keys
                    )
                    if not lane_semantic_match:
                        continue

                    # Verify the stop line is AHEAD of us, not behind.
                    awp_loc = awp.transform.location
                    sdx = awp_loc.x - ego_loc.x
                    sdy = awp_loc.y - ego_loc.y
                    if fwd.x * sdx + fwd.y * sdy <= 0:
                        continue  # stop line is behind us

                    # Route-proximity + heading check.
                    awp_yaw = awp.transform.rotation.yaw
                    on_route = any(
                        awp_loc.distance(rwp.transform.location) < match_dist
                        and abs((awp_yaw - rwp.transform.rotation.yaw + 180) % 360 - 180)
                        < match_yaw
                        for rwp, _ in route_sources
                    )
                    if not on_route:
                        continue

                    affects_us = True
                    stop_line_dist = ego_loc.distance(awp_loc)
                    stop_loc_candidate = awp_loc
                    break
            except (AttributeError, RuntimeError):
                # Fallback for older CARLA builds.
                tl_wp = self.map.get_waypoint(tl_loc)
                if tl_wp is not None and tl_wp.road_id in route_road_ids:
                    affects_us = True

            if not affects_us:
                continue

            # Prefer stop-line distance; fall back to pole distance.
            effective_dist = stop_line_dist if stop_line_dist is not None else pole_dist
            if effective_dist > self.config.traffic_light_brake_distance:
                continue

            if best_dist is None or effective_dist < best_dist:
                best_state = tl.state
                best_dist = effective_dist
                best_tl_id = tl.id
                best_stop_loc = stop_loc_candidate

        if best_tl_id is not None:
            prev_tl_id = self._tl_waiting
            self._tl_waiting = best_tl_id
            # Keep a previously valid stop-line location when a repeated
            # match for the same TL only has pole-level fallback geometry.
            if best_stop_loc is not None or prev_tl_id != best_tl_id:
                self._tl_stop_loc = best_stop_loc
            logger.debug(
                "T3: matched tl_id=%d state=%s dist=%.1f",
                best_tl_id,
                best_state,
                best_dist,
            )

        return best_state, best_dist

    def _stop_line_distance(
        self,
        tl: Any,
        transform: "carla.Transform",
    ) -> tuple[float | None, Any]:
        """Distance from ego to the traffic-light's stop line.

        Returns ``(distance, stop_location)`` or ``(None, None)`` if
        the API is unavailable or no stop waypoint matches our lane.
        """
        try:
            ego_wp = self.map.get_waypoint(transform.location)
            if ego_wp is None:
                return None, None
            # Use the vehicle's ACTUAL heading, not the map waypoint
            # yaw.  Inside junctions ego_wp can snap to a cross-street
            # internal road whose yaw is ~90° off from the vehicle's
            # true direction of travel.
            ego_yaw = transform.rotation.yaw

            route_sources = self._route_waypoint_sources()
            route_lane_keys: set[tuple[int, int]] = {(ego_wp.road_id, ego_wp.lane_id)}
            route_refs: list[tuple[Any, float]] = [(ego_wp, ego_wp.transform.rotation.yaw)]
            for rwp, _ in route_sources:
                route_lane_keys.add((rwp.road_id, rwp.lane_id))
                route_refs.append((rwp, rwp.transform.rotation.yaw))

            match_dist = self.config.traffic_light_route_match_distance
            match_yaw = self.config.traffic_light_route_match_yaw_deg
            waypoint_groups: list[tuple[int, list[Any]]] = []
            try:
                stop_wps = list(tl.get_stop_waypoints())
            except (AttributeError, RuntimeError):
                stop_wps = []
            if stop_wps:
                waypoint_groups.append((0, stop_wps))

            try:
                affected_wps = list(tl.get_affected_lane_waypoints())
            except (AttributeError, RuntimeError):
                affected_wps = []
            if affected_wps:
                waypoint_groups.append((1, affected_wps))

            best_match: tuple[int, float, Any] | None = None
            for source_rank, waypoints in waypoint_groups:
                for awp in waypoints:
                    lane_semantic_match = (
                        (awp.road_id, awp.lane_id) in route_lane_keys
                        or (awp.road_id, awp.lane_id - 1) in route_lane_keys
                        or (awp.road_id, awp.lane_id + 1) in route_lane_keys
                    )

                    # Heading gate: reject cross-street TLs whose
                    # candidate waypoint yaw differs by ~90° from ours.
                    awp_yaw = awp.transform.rotation.yaw
                    ego_yaw_diff = abs((awp_yaw - ego_yaw + 180) % 360 - 180)
                    if ego_yaw_diff > 40.0:
                        continue

                    awp_loc = awp.transform.location
                    on_route = any(
                        awp_loc.distance(rwp.transform.location) < match_dist
                        and abs((awp_yaw - rwp_yaw + 180) % 360 - 180) < match_yaw
                        for rwp, rwp_yaw in route_refs
                    )
                    if not on_route:
                        continue

                    # Preserve strong semantic preference but allow very
                    # close geometric fallback at junction seams.
                    near_route = any(
                        awp_loc.distance(rwp.transform.location) < (0.5 * match_dist)
                        for rwp, _ in route_refs
                    )
                    if not (lane_semantic_match or near_route):
                        continue

                    # Ensure the stop line is ahead of the ego.
                    fwd = transform.get_forward_vector()
                    sdx = awp_loc.x - transform.location.x
                    sdy = awp_loc.y - transform.location.y
                    if fwd.x * sdx + fwd.y * sdy <= 0:
                        continue

                    cand_dist = transform.location.distance(awp_loc)
                    if best_match is None or (source_rank, cand_dist) < (
                        best_match[0],
                        best_match[1],
                    ):
                        best_match = (source_rank, cand_dist, awp_loc)

            if best_match is not None:
                return best_match[1], best_match[2]
        except (AttributeError, RuntimeError):
            pass
        return None, None

    def _refresh_tl_stop_loc(self, transform: "carla.Transform") -> None:
        """Refresh the stored stop-line location after buffer changes."""
        if self._tl_waiting is None:
            return

        tl_actor = self.world.get_actor(self._tl_waiting)
        if tl_actor is None:
            self._tl_stop_loc = None
            return

        _, stop_loc = self._stop_line_distance(tl_actor, transform)
        if stop_loc is not None:
            self._tl_stop_loc = stop_loc

    def _refresh_route_tl_candidates(self, transform: "carla.Transform") -> None:
        """Pre-associate traffic lights that plausibly affect the route buffer.

        Tier-3 detection still does full ahead/distance/state checks; this cache
        is only a candidate pre-filter to reduce misses on left-turn geometries.
        """
        ego_wp = self.map.get_waypoint(transform.location)
        if ego_wp is None:
            self._route_tl_candidates.clear()
            return

        route_refs: list[tuple[Any, float]] = []
        route_lane_keys: set[tuple[int, int]] = set()
        route_refs.append((ego_wp, ego_wp.transform.rotation.yaw))
        route_lane_keys.add((ego_wp.road_id, ego_wp.lane_id))
        for rwp, _ in self._route_waypoint_sources():
            route_refs.append((rwp, rwp.transform.rotation.yaw))
            route_lane_keys.add((rwp.road_id, rwp.lane_id))

        max_dist = self.config.traffic_light_route_match_distance
        max_yaw = self.config.traffic_light_route_match_yaw_deg
        candidates: set[int] = set()

        for tl in self.world.get_actors().filter("*traffic_light*"):
            try:
                awps = tl.get_affected_lane_waypoints()
            except (AttributeError, RuntimeError):
                continue

            matched = False
            for awp in awps:
                lane_semantic_match = (
                    (awp.road_id, awp.lane_id) in route_lane_keys
                    or (awp.road_id, awp.lane_id - 1) in route_lane_keys
                    or (awp.road_id, awp.lane_id + 1) in route_lane_keys
                )
                awp_loc = awp.transform.location
                awp_yaw = awp.transform.rotation.yaw
                for rwp, rwp_yaw in route_refs:
                    yaw_diff = abs((awp_yaw - rwp_yaw + 180) % 360 - 180)
                    if yaw_diff >= max_yaw:
                        continue
                    # Prefer lane-semantic matches, but allow strict geometric
                    # fallback to preserve TL association across lane-id churn
                    # around junction seams.
                    route_dist = awp_loc.distance(rwp.transform.location)
                    geom_fallback = route_dist < (0.5 * max_dist)
                    if route_dist < max_dist and (lane_semantic_match or geom_fallback):
                        candidates.add(tl.id)
                        matched = True
                        break
                if matched:
                    break

        self._route_tl_candidates = candidates

    # --- Stop sign detection --------------------------------------------

    def _check_stop_signs(
        self,
        transform: "carla.Transform",
        speed_kmh: float,
    ) -> tuple[float | None, bool]:
        """Return (approach_dist, is_committed) for hybrid stop sign tracking.

        Uses a two-phase hybrid approach:
        1. **Phase 1 (Horizon Scan)**: Smoothly brake for any stop sign
           ahead on our route up to ``stop_sign_brake_distance``.
        2. **Phase 2 (Trigger Commitment)**: Hard commit to 0.0 target speed
           only when the ego's bounding box intersects the sign's true
           ``trigger_volume``.

        A stop sign is "cleared" once the ego has come to a full stop
        within its trigger volume for at least ``stop_sign_wait_ticks``
        consecutive ticks.
        """
        cfg = self.config
        ego_loc = transform.location
        ego_wp = self.map.get_waypoint(ego_loc)
        if ego_wp is None:
            return None, False

        fwd = transform.get_forward_vector()

        # ---- Active stop sign: waiting at it ----
        if self._stop_sign_active is not None:
            ss_actor = self.world.get_actor(self._stop_sign_active)
            if ss_actor is None:
                # Sign was destroyed — clear
                self._stop_sign_active = None
                self._stop_sign_wait_counter = 0
            else:
                ss_loc = ss_actor.get_transform().location
                dx = ss_loc.x - ego_loc.x
                dy = ss_loc.y - ego_loc.y
                behind = (fwd.x * dx + fwd.y * dy) < -2.0  # Give 2m grace

                if behind:
                    # Drove past it — add to cleared
                    self._cleared_stop_signs.add(self._stop_sign_active)
                    self._stop_sign_active = None
                    self._stop_sign_wait_counter = 0
                elif speed_kmh < 1.0:
                    self._stop_sign_wait_counter += 1
                    if self._stop_sign_wait_counter >= cfg.stop_sign_wait_ticks:
                        # Waited long enough — clear the sign
                        self._cleared_stop_signs.add(self._stop_sign_active)
                        self._stop_sign_active = None
                        self._stop_sign_wait_counter = 0
                        return None, False
                    return 0.0, True
                else:
                    self._stop_sign_wait_counter = 0
                    return 0.0, True

        # ---- Scan for new stop signs ----
        best_dist: float | None = None

        for ss in self.world.get_actors().filter("*stop*"):
            if ss.id in self._cleared_stop_signs:
                continue

            ss_loc = ss.get_transform().location
            dist = ego_loc.distance(ss_loc)
            if dist > cfg.stop_sign_brake_distance:
                continue

            # Compute trigger volume center in world space.
            # The TV center sits on (or very near) the road surface
            # in both Town03 (road-painted stops) and Town07
            # (pole-mounted signs with road-level trigger volumes).
            ss_tf = ss.get_transform()
            tv = ss.trigger_volume
            tv_world = ss_tf.transform(tv.location)

            # Route-proximity check — 2D horizontal distance from the
            # trigger volume center to our planned route.  Using the
            # TV center (not the pole) and 2D (ignoring z) avoids
            # false positives from adjacent-lane signs and false
            # rejections from curb/elevation z-offsets.
            #
            # This check runs BEFORE Phase 2 commitment so that
            # adjacent-lane signs (e.g. opposite-direction road-painted
            # stops in Town03) cannot trigger a full-stop commitment.
            sign_near_route = False
            for rwp, _ in self._wp_buffer:
                rwp_loc = rwp.transform.location
                dx_t = tv_world.x - rwp_loc.x
                dy_t = tv_world.y - rwp_loc.y
                tv_dist_2d = math.sqrt(dx_t * dx_t + dy_t * dy_t)
                if tv_dist_2d < cfg.stop_sign_route_proximity:
                    sign_near_route = True
                    break
            if not sign_near_route:
                continue

            # Phase 2: Commitment — ego is close to the stop line.
            # Two complementary checks (either triggers commitment):
            #  1. trigger_volume.contains() — works for road-surface
            #     stops (Town03) where the car drives over the TV box.
            #  2. 2D distance from ego to TV center — works for pole-
            #     mounted signs (Town07) where the TV box is too small
            #     for contains() but the center is on the road surface.
            ego_to_tv_2d = math.sqrt((ego_loc.x - tv_world.x) ** 2 + (ego_loc.y - tv_world.y) ** 2)
            if ego_to_tv_2d < cfg.stop_sign_commit_radius or ss.trigger_volume.contains(
                ego_loc, ss_tf
            ):
                self._stop_sign_active = ss.id
                self._stop_sign_wait_counter = 0
                return 0.0, True

            # Phase 1: Horizon Scan (Smooth approach)
            # Ahead check — only approach signs in front of us.
            dx = ss_loc.x - ego_loc.x
            dy = ss_loc.y - ego_loc.y
            if fwd.x * dx + fwd.y * dy <= 0:
                continue

            if best_dist is None or dist < best_dist:
                best_dist = dist

        return best_dist, False

    # --- Yield sign handling -----------------------------------------------

    def _check_yield_signs(
        self,
        transform: "carla.Transform",
        speed_kmh: float,
    ) -> tuple[float | None, bool]:
        """Return (approach_dist, has_conflict) for yield sign handling.

        Yield signs require slowing and checking for cross-traffic, but
        NOT a mandatory full stop when the intersection is clear.

        State machine:
        1. **Scan**: Find nearest yield sign ahead on our route within
           ``yield_scan_distance``.
        2. **Approach**: Kinematic ramp to ``yield_slow_speed``.
        3. **Conflict check**: Scan for vehicles whose heading diverges
           from ours by more than ``yield_conflict_angle_deg`` within
           ``yield_conflict_distance``.  If conflict → ramp to 0.
        4. **Clear**: Once ego passes the sign, add to cleared set.

        Returns:
            (distance_to_yield, has_conflict):
            - distance is None when no yield sign is relevant.
            - has_conflict is True when cross-traffic blocks our path.
        """
        cfg = self.config
        ego_loc = transform.location
        fwd = transform.get_forward_vector()

        # ---- Active yield sign: check if we've passed it ----
        if self._yield_active is not None:
            ys_actor = self.world.get_actor(self._yield_active)
            if ys_actor is None:
                # Destroyed — clear
                self._yield_active = None
                self._yield_wait_counter = 0
                self._yield_has_conflict = False
            else:
                ys_loc = ys_actor.get_transform().location
                dx = ys_loc.x - ego_loc.x
                dy = ys_loc.y - ego_loc.y
                behind = (fwd.x * dx + fwd.y * dy) < -cfg.yield_clear_behind_m

                if behind:
                    # Drove past — clear
                    self._cleared_yield_signs.add(self._yield_active)
                    self._yield_active = None
                    self._yield_wait_counter = 0
                    self._yield_has_conflict = False
                    return None, False

                dist = ego_loc.distance(ys_loc)
                has_conflict = self._yield_conflict_check(transform)
                self._yield_has_conflict = has_conflict

                if has_conflict:
                    self._yield_wait_counter = 0
                    return dist, True

                # No conflict — but wait a short period before proceeding
                # to avoid darting into a gap that's too small.
                self._yield_wait_counter += 1
                if self._yield_wait_counter < cfg.yield_wait_ticks:
                    return dist, False
                # Waited long enough with no conflict — proceed
                return dist, False

        # ---- Scan for new yield signs ----
        best_dist: float | None = None
        best_id: int | None = None

        for ys in self.world.get_actors().filter("*yield*"):
            if ys.id in self._cleared_yield_signs:
                continue

            ys_loc = ys.get_transform().location
            dist = ego_loc.distance(ys_loc)
            if dist > cfg.yield_scan_distance:
                continue

            # Ahead check
            dx = ys_loc.x - ego_loc.x
            dy = ys_loc.y - ego_loc.y
            if fwd.x * dx + fwd.y * dy <= 0:
                continue

            # Route-proximity check — use trigger volume center (road-
            # level) instead of pole location, same approach as stop
            # signs.  Yield poles sit 2.5–5.4 m from the road in Town07,
            # so pole-based proximity misses many signs.
            ys_tf = ys.get_transform()
            tv = ys.trigger_volume
            tv_world = ys_tf.transform(tv.location)

            sign_near_route = False
            for rwp, _ in self._wp_buffer:
                rwp_loc = rwp.transform.location
                dx_r = tv_world.x - rwp_loc.x
                dy_r = tv_world.y - rwp_loc.y
                horiz_dist = math.sqrt(dx_r * dx_r + dy_r * dy_r)
                if horiz_dist < cfg.yield_route_proximity:
                    sign_near_route = True
                    break
            if not sign_near_route:
                continue

            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_id = ys.id

        # Activate the nearest yield sign if found
        if best_id is not None:
            self._yield_active = best_id
            self._yield_wait_counter = 0
            has_conflict = self._yield_conflict_check(transform)
            self._yield_has_conflict = has_conflict
            return best_dist, has_conflict

        return None, False

    def _yield_conflict_check(self, transform: "carla.Transform") -> bool:
        """Check for cross-traffic that would conflict with our path.

        Scans for nearby vehicles whose heading diverges from ours by
        more than ``yield_conflict_angle_deg``, indicating they are on
        a crossing or merging path.  Only considers vehicles within
        ``yield_conflict_distance`` that are ahead or to the side.

        Returns:
            True if a conflicting vehicle is detected.
        """
        cfg = self.config
        ego_loc = transform.location
        ego_fwd = transform.get_forward_vector()
        ego_heading_rad = math.atan2(ego_fwd.y, ego_fwd.x)

        threshold_rad = math.radians(cfg.yield_conflict_angle_deg)

        for veh in self.world.get_actors().filter("*vehicle*"):
            if veh.id == self.vehicle.id:
                continue

            veh_loc = veh.get_transform().location
            dist = ego_loc.distance(veh_loc)
            if dist > cfg.yield_conflict_distance:
                continue

            # Only check vehicles that are roughly ahead or to our side
            dx = veh_loc.x - ego_loc.x
            dy = veh_loc.y - ego_loc.y
            dot_ahead = ego_fwd.x * dx + ego_fwd.y * dy
            if dot_ahead < -5.0:
                # Well behind us — ignore
                continue

            # Check heading divergence
            veh_fwd = veh.get_transform().get_forward_vector()
            veh_heading_rad = math.atan2(veh_fwd.y, veh_fwd.x)
            angle_diff = abs(ego_heading_rad - veh_heading_rad)
            # Normalise to [0, π]
            angle_diff = angle_diff % (2.0 * math.pi)
            if angle_diff > math.pi:
                angle_diff = 2.0 * math.pi - angle_diff

            # Cross-traffic: heading differs significantly from ours
            # (not same direction, not oncoming on a straight road)
            if threshold_rad < angle_diff < (math.pi - threshold_rad):
                # This vehicle is crossing our path — check if it's
                # actually moving (stationary vehicles aren't a conflict)
                veh_vel = veh.get_velocity()
                veh_speed = math.sqrt(veh_vel.x**2 + veh_vel.y**2)
                if veh_speed > 0.5:  # m/s — moving
                    return True

        return False

    # --- Curve-aware corridor -------------------------------------------

    def _build_route_corridor(
        self,
        ego_loc: "carla.Location",
        cutoff: float,
        half_lane: float,
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Build a lane corridor that follows the planned route.

        Instead of projecting a straight rectangle ahead of the ego,
        this samples waypoints from ``_wp_buffer`` up to *cutoff*
        metres and offsets each by ± *half_lane* perpendicular to the
        waypoint's heading.

        Returns ``(polygon, segments, segments_3d)``:

        * **polygon** — full corridor outline for debug drawing.
        * **segments** — list of convex quadrilaterals (one per
          consecutive waypoint pair).  SAT overlap must be tested
          per-segment because the full polygon is non-convex on
          curved roads.
                * **segments_3d** — list of corridor quads with per-point
                    world Z for debug drawing on slopes.

        Falls back to a single rectangle when no route waypoints
        are available.
        """
        # Collect centre-line samples from the route
        samples: list[tuple[np.ndarray, np.ndarray, float]] = []
        ego_xy = np.array([ego_loc.x, ego_loc.y])

        for wp, _cmd in self._wp_buffer:
            wloc = wp.transform.location
            if ego_loc.distance(wloc) > cutoff:
                break
            wp_fwd = wp.transform.get_forward_vector()
            fwd_2d = np.array([wp_fwd.x, wp_fwd.y])
            norm = np.linalg.norm(fwd_2d)
            if norm < 1e-6:
                continue
            fwd_2d /= norm
            right_2d = np.array([-fwd_2d[1], fwd_2d[0]])
            centre = np.array([wloc.x, wloc.y])
            samples.append((centre, right_2d, float(wloc.z)))

        if not samples:
            # Fallback: straight rectangle using ego heading
            transform = self.vehicle.get_transform()
            fwd = transform.get_forward_vector()
            fwd_2d = np.array([fwd.x, fwd.y])
            right_2d = np.array([-fwd_2d[1], fwd_2d[0]])
            rect = np.array(
                [
                    ego_xy - right_2d * half_lane,
                    ego_xy + right_2d * half_lane,
                    ego_xy + fwd_2d * cutoff + right_2d * half_lane,
                    ego_xy + fwd_2d * cutoff - right_2d * half_lane,
                ]
            )
            z = float(ego_loc.z) + 0.3
            rect_3d = np.array(
                [
                    [rect[0][0], rect[0][1], z],
                    [rect[1][0], rect[1][1], z],
                    [rect[2][0], rect[2][1], z],
                    [rect[3][0], rect[3][1], z],
                ]
            )
            return rect, [rect], [rect_3d]

        # Build corridor as individual convex quad segments + full polygon.
        # Each segment is a perfect rectangle aligned to the vector
        # connecting consecutive centre-line samples.  This guarantees
        # convexity even on very tight inner turns where per-waypoint
        # normal offsets would create self-intersecting "bowtie" quads.
        prev_center = ego_xy
        left_boundary: list[np.ndarray] = []
        right_boundary: list[np.ndarray] = []
        segments: list[np.ndarray] = []
        segments_3d: list[np.ndarray] = []
        prev_z = float(ego_loc.z)

        for centre, _right_2d, centre_z in samples:
            seg_vec = centre - prev_center
            seg_len = np.linalg.norm(seg_vec)
            if seg_len < 1e-6:
                continue
            seg_dir = seg_vec / seg_len
            seg_right = np.array([-seg_dir[1], seg_dir[0]])

            prev_l = prev_center - seg_right * half_lane
            prev_r = prev_center + seg_right * half_lane
            cur_l = centre - seg_right * half_lane
            cur_r = centre + seg_right * half_lane

            # Perfectly convex rectangle aligned to segment direction
            segments.append(np.array([prev_l, prev_r, cur_r, cur_l]))
            z_prev = prev_z + 0.3
            z_cur = float(centre_z) + 0.3
            segments_3d.append(
                np.array(
                    [
                        [prev_l[0], prev_l[1], z_prev],
                        [prev_r[0], prev_r[1], z_prev],
                        [cur_r[0], cur_r[1], z_cur],
                        [cur_l[0], cur_l[1], z_cur],
                    ]
                )
            )
            left_boundary.append(prev_l)
            right_boundary.append(prev_r)
            prev_center = centre
            prev_z = float(centre_z)

        # Close out the last segment's far edge for the debug polygon
        if segments:
            last_seg = segments[-1]
            left_boundary.append(last_seg[3])  # cur_l
            right_boundary.append(last_seg[2])  # cur_r

        # Full polygon for debug visualisation only
        if left_boundary:
            polygon = np.array(left_boundary + right_boundary[::-1])
        else:
            polygon = np.array([ego_xy])
        return polygon, segments, segments_3d

    def _detect_vehicles(
        self,
        transform: "carla.Transform",
    ) -> List[Dict[str, Any]]:
        """Return up to 10 nearest vehicles with privileged info.

        Distances are computed between **bounding-box centroids** (not
        actor transform origins, which sit at the rear axle for most
        CARLA vehicles).  This is critical for large vehicles (buses,
        trucks) where the centre-of-mass can be several metres from
        the rear axle.

        A bumper-to-bumper ``distance`` is also computed by subtracting
        each vehicle's forward half-extent from the centroid distance.

        Each entry includes an ``obb_in_lane`` flag computed via
        Oriented Bounding-Box (OBB) projection + SAT overlap.
        """
        ego_loc = transform.location
        fwd = transform.get_forward_vector()
        right = carla.Vector3D(-fwd.y, fwd.x, 0)
        cutoff = self.config.vehicle_hazard_distance * 2
        half_lane = self._dynamic_half_lane(transform)

        # Ego bounding-box centroid in world space.
        # IMPORTANT: carla.Transform.transform() mutates the input
        # Location in-place, which corrupts actor.bounding_box.location
        # for subsequent reads.  Always copy before transforming.
        ego_bb = self.vehicle.bounding_box
        ego_bb_loc = carla.Location(ego_bb.location.x, ego_bb.location.y, ego_bb.location.z)
        ego_bb_center = transform.transform(ego_bb_loc)
        ego_half_len = ego_bb.extent.x

        # Build a curve-aware lane corridor by sampling waypoints
        # along the planned route.  This follows the actual path
        # through turns instead of projecting a straight rectangle.
        corridor_poly, corridor_segs, corridor_segs_3d = self._build_route_corridor(
            ego_loc,
            cutoff,
            half_lane,
        )
        self._corridor = corridor_poly  # expose for debug drawing
        self._corridor_3d_segments = corridor_segs_3d  # slope-aware debug rendering
        self._n_corridor_segments = len(corridor_segs)  # diagnostic

        nearby: list[dict[str, Any]] = []
        for v in self.world.get_actors().filter("*vehicle*"):
            if v.id == self.vehicle.id:
                continue

            v_transform = v.get_transform()
            v_loc = v_transform.location

            # Quick cull on actor-root distance (cheap)
            if ego_loc.distance(v_loc) > cutoff:
                continue

            # Bounding-box centroid in world space.
            # Copy location to avoid in-place mutation by transform().
            v_bb = v.bounding_box
            v_bb_loc = carla.Location(v_bb.location.x, v_bb.location.y, v_bb.location.z)
            v_bb_center = v_transform.transform(v_bb_loc)
            center_dist = ego_bb_center.distance(v_bb_center)

            dx = v_bb_center.x - ego_bb_center.x
            dy = v_bb_center.y - ego_bb_center.y
            longitudinal = fwd.x * dx + fwd.y * dy  # positive = ahead
            lateral = right.x * dx + right.y * dy  # positive = right

            v_vel = v.get_velocity()
            v_speed = math.sqrt(v_vel.x**2 + v_vel.y**2 + v_vel.z**2) * 3.6

            # Heading similarity — tolerate up to ~120° of heading
            # difference so that vehicles completing a turn ahead of
            # the ego are not falsely classified as oncoming traffic.
            # True oncoming vehicles (opposite lane) have dot ≈ -1.0.
            v_fwd = v_transform.get_forward_vector()
            heading_dot = fwd.x * v_fwd.x + fwd.y * v_fwd.y

            # Bumper-to-bumper distance
            v_half_len = v.bounding_box.extent.x
            bumper_gap = max(center_dist - ego_half_len - v_half_len, 0.0)

            # OBB overlap check via SAT — test each convex corridor
            # segment individually (the full polygon is non-convex on
            # curved roads, which breaks SAT).
            v_corners = self._get_actor_obb_2d(v)
            obb_in_lane = any(self._sat_overlap(seg, v_corners) for seg in corridor_segs)
            # Reject SAT overlaps with large vertical separation
            # (e.g. overpass/underpass lanes that overlap in XY).
            vertical_sep = abs(v_bb_center.z - ego_bb_center.z)
            if obb_in_lane and vertical_sep > self.config.corridor_max_vertical_separation:
                obb_in_lane = False

            # Diagnostic: warn if SAT misses a vehicle that the
            # lateral-distance heuristic would catch.
            if (
                not obb_in_lane
                and longitudinal > 0
                and heading_dot > 0
                and abs(lateral) < half_lane
                and bumper_gap < self.config.vehicle_hazard_distance
            ):
                logger.debug(
                    "SAT miss: bumper=%.1fm long=%.1fm lat=%.1fm " "segs=%d obb=%s ego=(%.1f,%.1f)",
                    bumper_gap,
                    longitudinal,
                    lateral,
                    len(corridor_segs),
                    v_corners.tolist(),
                    ego_loc.x,
                    ego_loc.y,
                )

            nearby.append(
                {
                    "distance": round(bumper_gap, 2),
                    "center_distance": round(center_dist, 2),
                    "longitudinal": round(longitudinal, 2),
                    "lateral": round(lateral, 2),
                    "speed_kmh": round(v_speed, 1),
                    "world_x": round(v_bb_center.x, 2),
                    "world_y": round(v_bb_center.y, 2),
                    "world_z": round(v_bb_center.z, 2),
                    "vertical_sep": round(vertical_sep, 2),
                    "is_ahead": longitudinal > 0,
                    "same_direction": heading_dot > -0.5,
                    "obb_in_lane": obb_in_lane,
                }
            )

        nearby.sort(key=lambda x: x["distance"])
        return nearby[:10]

    # --- OBB helpers ------------------------------------------------

    @staticmethod
    def _get_actor_obb_2d(actor: Any) -> np.ndarray:
        """Return the 4 XY corners of *actor*'s oriented bounding box.

        Uses ``actor.bounding_box`` (half-extents in local frame) and
        the actor's world transform to produce a (4, 2) float array
        of world-space 2-D corner coordinates.

        The corner offsets are relative to ``bounding_box.location``
        (the centre of the bounding box in actor-local space), **not**
        the actor origin.  For most CARLA vehicles the actor origin
        sits at the rear axle, ~1–2 m behind the BB centre.

        .. note::

           Uses pure-Python rotation instead of
           ``carla.Transform.transform()`` because the CARLA API
           mutates the input ``carla.Location`` **in-place**, which
           silently corrupts ``bounding_box.location`` for any
           subsequent access on the same actor.
        """
        bb = actor.bounding_box
        ext = bb.extent  # half-extents (x=length/2, y=width/2)
        # Read local-space values as Python floats immediately so
        # they are immune to any in-place mutation by CARLA.
        cx = float(bb.location.x)
        cy = float(bb.location.y)
        ex = float(ext.x)
        ey = float(ext.y)

        t = actor.get_transform()
        yaw = math.radians(t.rotation.yaw)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        tx = t.location.x
        ty = t.location.y

        # 4 local-frame corners relative to actor origin,
        # centred on BB location (cx, cy).
        local_offsets = [
            (cx + ex, cy + ey),
            (cx + ex, cy - ey),
            (cx - ex, cy - ey),
            (cx - ex, cy + ey),
        ]

        # Rotate by yaw and translate to world space.
        corners = np.empty((4, 2), dtype=np.float64)
        for i, (lx, ly) in enumerate(local_offsets):
            corners[i, 0] = cos_y * lx - sin_y * ly + tx
            corners[i, 1] = sin_y * lx + cos_y * ly + ty

        return corners

    @staticmethod
    def _sat_overlap(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
        """Separating Axis Theorem overlap test for two convex polygons.

        Each polygon is a (N, 2) array of vertices in CCW or CW order.
        Returns ``True`` if the polygons overlap.
        """

        def _project(vertices: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
            dots = vertices @ axis
            return dots.min(), dots.max()

        for poly in (poly_a, poly_b):
            n = len(poly)
            for i in range(n):
                edge = poly[(i + 1) % n] - poly[i]
                # Perpendicular (normal) to the edge
                axis = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(axis)
                if norm < 1e-8:
                    continue
                axis /= norm

                min_a, max_a = _project(poly_a, axis)
                min_b, max_b = _project(poly_b, axis)

                if max_a < min_b or max_b < min_a:
                    return False  # separating axis found → no overlap
        return True

    def _detect_pedestrians(
        self,
        transform: "carla.Transform",
    ) -> List[Dict[str, Any]]:
        """Return up to 5 nearest pedestrians with privileged info.

        Each entry includes:
        - ``lateral``: offset so ``_closest_pedestrian_distance`` can
          ignore pavement walkers.
        - ``predicted_lateral``: lateral offset of the pedestrian's
          predicted position (velocity-extrapolated 1.0 s ahead).
          Enables early braking for pedestrians about to step into
          the lane.
        - ``predicted_distance``: distance to the predicted position.
        """
        ego_loc = transform.location
        fwd = transform.get_forward_vector()
        right = carla.Vector3D(-fwd.y, fwd.x, 0)
        cutoff = self.config.pedestrian_hazard_distance * 2
        predict_horizon = 1.0  # seconds

        nearby: list[dict[str, Any]] = []
        for w in self.world.get_actors().filter("*walker*"):
            w_transform = w.get_transform()
            w_loc = w_transform.location
            dist = ego_loc.distance(w_loc)
            if dist > cutoff:
                continue

            dx = w_loc.x - ego_loc.x
            dy = w_loc.y - ego_loc.y
            longitudinal = fwd.x * dx + fwd.y * dy
            lateral = right.x * dx + right.y * dy

            # Velocity extrapolation for path prediction
            w_vel = w.get_velocity()
            pred_x = w_loc.x + w_vel.x * predict_horizon
            pred_y = w_loc.y + w_vel.y * predict_horizon
            pred_dx = pred_x - ego_loc.x
            pred_dy = pred_y - ego_loc.y
            pred_longitudinal = fwd.x * pred_dx + fwd.y * pred_dy
            pred_lateral = right.x * pred_dx + right.y * pred_dy
            pred_dist = math.sqrt(pred_dx**2 + pred_dy**2)

            nearby.append(
                {
                    "distance": round(dist, 2),
                    "longitudinal": round(longitudinal, 2),
                    "lateral": round(lateral, 2),
                    "is_ahead": longitudinal > 0,
                    "predicted_distance": round(pred_dist, 2),
                    "predicted_lateral": round(pred_lateral, 2),
                    "predicted_ahead": pred_longitudinal > 0,
                }
            )

        nearby.sort(key=lambda x: x["distance"])
        return nearby[:5]

    def _lead_vehicle_info(
        self,
        nearby: List[Dict[str, Any]],
        ego_speed_kmh: float,
    ) -> Tuple[float | None, float | None, float | None]:
        """Return distance, speed, and lateral offset of closest vehicle ahead.

        A vehicle is considered "in lane" if its OBB/SAT corridor
        overlap test passes.  When ``vehicle_allow_lateral_fallback``
        is enabled, a centre-point lateral check is also accepted to
        reduce false negatives on segments where SAT geometry is noisy.

        Oncoming vehicles use a stricter conflict gate (lateral,
        longitudinal, and TTC constraints) so adjacent-lane overlap
        artefacts do not trigger false emergency stops.
        """
        for v in nearby:
            if not v["is_ahead"]:
                continue
            obb_flag = v.get("obb_in_lane", False)
            if not v.get("same_direction", True):
                if not obb_flag:
                    continue
                if not self._passes_oncoming_conflict_gate(v, ego_speed_kmh):
                    continue
                in_lane = True
            elif self.config.vehicle_allow_lateral_fallback:
                # Union of SAT overlap and lateral-distance check — either
                # one is sufficient to flag the vehicle as in-lane.
                lateral_flag = abs(v["lateral"]) < self._cached_half_lane
                in_lane = bool(obb_flag or lateral_flag)
            else:
                # SAT-only mode for ablation/testing.
                in_lane = bool(obb_flag)

            if in_lane:
                return v["distance"], v.get("speed_kmh"), v.get("lateral")
        return None, None, None

    def _passes_oncoming_conflict_gate(self, vehicle_info: Dict[str, Any], ego_speed_kmh: float) -> bool:
        """Return True only for immediate oncoming conflict candidates."""
        cfg = self.config
        if not cfg.vehicle_oncoming_conflict_gate_enabled:
            return True

        lateral = abs(float(vehicle_info.get("lateral", 999.0)))
        longitudinal = max(0.0, float(vehicle_info.get("longitudinal", 999.0)))
        distance = max(0.0, float(vehicle_info.get("distance", 999.0)))
        if lateral > float(cfg.oncoming_lead_lateral_max_m):
            return False
        if longitudinal > float(cfg.oncoming_lead_longitudinal_max_m):
            return False

        other_speed_kmh = max(0.0, float(vehicle_info.get("speed_kmh", 0.0)))
        closing_kmh = max(
            float(cfg.oncoming_lead_min_closing_speed_kmh),
            max(0.0, ego_speed_kmh) + other_speed_kmh,
        )
        if distance <= 0.0:
            return True
        ttc_s = distance / max(closing_kmh / 3.6, 1e-3)
        return ttc_s <= float(cfg.oncoming_lead_ttc_max_s)

    def _dynamic_half_lane(self, transform: "carla.Transform") -> float:
        """Compute corridor half-width from the current lane width.

        Uses ``ego_wp.lane_width / 2`` plus a margin that scales
        linearly between ``corridor_margin_min`` (narrow lanes like
        Town03 @ 2.6 m) and ``corridor_margin_max`` (wide lanes like
        Town01 @ 3.5 m).  The result is cached on ``_cached_half_lane``
        so that ``_lead_vehicle_distance`` can reuse it without an
        extra ``get_waypoint`` call.
        """
        cfg = self.config
        ego_wp = self.map.get_waypoint(transform.location)
        if ego_wp is None:
            self._cached_half_lane = cfg.lane_half_width
            return float(cfg.lane_half_width)
        lw = ego_wp.lane_width
        # Linear interpolation: 2.6m lane → margin_min, 3.5m lane → margin_max
        t = max(0.0, min(1.0, (lw - 2.6) / 0.9))
        margin = cfg.corridor_margin_min + t * (cfg.corridor_margin_max - cfg.corridor_margin_min)
        min_half = float(
            self.vehicle.bounding_box.extent.y + cfg.corridor_vehicle_half_width_margin,
        )
        half = float(max(lw / 2.0 + margin, min_half))
        self._cached_half_lane = half
        return half

    def _estimate_junction_clearance_distance(self) -> float:
        """Estimate distance to clear the upcoming junction from stop line.

        Uses contiguous junction waypoints in route lookahead. Falls
        back to ``tl_clearance_margin`` when no reliable segment exists.
        """
        cfg = self.config
        route_sources = self._route_waypoint_sources(lookahead=120)
        if not route_sources:
            return cfg.tl_clearance_margin

        in_junction = False
        first_wp = None
        last_wp = None
        for wp, _ in route_sources:
            if wp.is_junction:
                if not in_junction:
                    in_junction = True
                    first_wp = wp
                    last_wp = wp
                else:
                    last_wp = wp
            elif in_junction:
                break

        if first_wp is None or last_wp is None:
            return cfg.tl_clearance_margin

        clearance = first_wp.transform.location.distance(last_wp.transform.location)
        return float(max(clearance, cfg.tl_clearance_margin))

    def _closest_pedestrian_distance(
        self,
        nearby: List[Dict[str, Any]],
        transform: "carla.Transform",
    ) -> float | None:
        """Distance to the closest pedestrian hazard ahead **in our lane corridor**.

        A pedestrian is considered a hazard if:
        - They are currently in the lane (lateral < pedestrian_lane_half_width), **or**
        - Their predicted position (1s ahead via velocity extrapolation)
          will be in the lane (``predicted_lateral`` < half-width).

        This enables early braking for pedestrians about to step off
        the pavement into the vehicle's path.
        """
        # Hybrid width: keep map-adaptive lane sizing but never below
        # the fixed 2.5 m floor used by prior tuned behavior.
        half_w = max(self._dynamic_half_lane(transform), self.config.lane_half_width)
        for p in nearby:
            current_in_lane = p["is_ahead"] and abs(p["lateral"]) < half_w
            predicted_in_lane = (
                p.get("predicted_ahead", False) and abs(p.get("predicted_lateral", 999.0)) < half_w
            )
            if current_in_lane or predicted_in_lane:
                # Use whichever distance is shorter (current or predicted)
                pred_dist = float(p.get("predicted_distance", p["distance"]))
                return float(min(p["distance"], pred_dist))
        return None

    @staticmethod
    def _encode_tl(state: Any) -> int:
        """Encode a ``carla.TrafficLightState`` to an integer."""
        if state is None:
            return 0
        if not CARLA_AVAILABLE:
            return 0
        mapping = {
            carla.TrafficLightState.Green: 1,
            carla.TrafficLightState.Yellow: 2,
            carla.TrafficLightState.Red: 3,
        }
        return mapping.get(state, 0)
