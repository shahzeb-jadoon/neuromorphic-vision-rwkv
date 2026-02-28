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
from typing import Any, Dict, List, Optional, Tuple

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
    speed_limit_margin: float = 0.90  # drive at 90% of posted limit

    # --- PID: longitudinal (speed) ---
    speed_kp: float = 0.8
    speed_ki: float = 0.05
    speed_kd: float = 0.01

    # --- PID: lateral (steering) — straight-line defaults ---
    steer_kp: float = 1.2
    steer_ki: float = 0.05
    steer_kd: float = 0.15

    # --- PID gains for turns (blended smoothly via curvature) ---
    turn_speed_kp: float = 1.0
    turn_speed_ki: float = 0.1
    turn_speed_kd: float = 0.05
    turn_steer_kp: float = 1.5
    turn_steer_ki: float = 0.05
    turn_steer_kd: float = 0.3

    # --- Waypoint following ---
    min_follow_distance: float = 3.0  # metres — pop threshold
    target_follow_distance: float = 5.0  # metres — desired look-ahead
    turn_follow_distance: float = 2.0  # metres — shorter look-ahead in turns
    max_follow_distance: float = 20.0  # metres — upper bound
    waypoint_buffer_size: int = 40

    # --- Safety thresholds ---
    traffic_light_brake_distance: float = 60.0  # metres — start braking
    traffic_light_stop_distance: float = 4.0  # metres — full stop
    tl_comfort_decel: float = 3.0  # m/s² — comfortable deceleration for TL
    turn_comfort_decel: float = 1.5  # m/s² — gentler braking for turn approach
    vehicle_hazard_distance: float = 25.0  # metres — start slowing (min)
    vehicle_emergency_distance: float = 5.0  # metres — emergency (min)
    vehicle_reaction_time: float = 2.0  # seconds — scale hazard dist with speed
    vehicle_emergency_time: float = 1.2  # seconds — scale emergency with speed
    vehicle_comfort_decel: float = 3.0  # m/s² — comfortable braking for following
    pedestrian_hazard_distance: float = 15.0
    pedestrian_emergency_distance: float = 5.0
    lane_half_width: float = 2.5  # metres — lateral check for "in-lane"
    pedestrian_lane_half_width: float = 3.0  # metres — wider lateral band for pedestrians

    # --- Stop signs ---
    stop_sign_brake_distance: float = 30.0  # metres — start braking
    stop_sign_stop_distance: float = 4.0  # metres — full stop
    stop_sign_comfort_decel: float = 3.0  # m/s² — comfortable deceleration
    stop_sign_wait_ticks: int = 40  # ~2 s at 20 Hz — minimum pause

    # --- Turn behaviour ---
    turn_speed_factor: float = 0.45  # reduce speed to 45% at turns
    junction_turn_factor: float = 0.65  # additional reduction inside junctions
    turn_anticipation_distance: float = 30.0  # start slowing this far ahead

    # --- Speed limits / curvature ---
    curvature_speed_factor: float = 0.7  # reduce by 30% on curves
    curvature_threshold_deg: float = 20.0  # angle change between successive wps

    # --- Noise injection (for training data diversity) ---
    steering_noise_sigma: float = 1e-3  # Gaussian noise std on steering

    # --- Rate limiter ---
    max_target_drop_per_frame: float = 2.0  # km/h — prevents target jumps


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
    stop_sign_distance: float = -1.0  # metres (−1 if none)

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

        # Controllers
        self.speed_pid = PIDController(
            self.config.speed_kp,
            self.config.speed_ki,
            self.config.speed_kd,
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
        self._tl_waiting: int | None = None  # actor-id of TL we're waiting for

        # Smoothed curvature estimate (EMA)
        self._smoothed_curvature: float = 0.0

        # Corridor polygon for debug drawing (set each tick)
        self._corridor: np.ndarray | None = None

        # Number of corridor segments built (for diagnostics)
        self._n_corridor_segments: int = 0

        # Distance to the upcoming turn waypoint (set by _get_target)
        self._turn_distance: float | None = None

        # Stop sign state
        self._cleared_stop_signs: set[int] = set()
        self._stop_sign_wait_counter: int = 0
        self._stop_sign_active: int | None = None  # actor ID

        # Rate limiter — previous target for smooth transitions
        self._prev_target: float = self.config.target_speed

    # ------------------------------------------------------------------
    # Route management
    # ------------------------------------------------------------------

    def set_route(self, route: List[Tuple["carla.Waypoint", RoadOption]]) -> None:
        """Load a pre-computed route (list of (waypoint, RoadOption))."""
        self._route = list(route)
        self._route_length = len(route)
        self._route_index = 0
        self._wp_buffer.clear()
        self._fill_buffer()
        self.speed_pid.reset()
        self.steer_pid.reset()
        self.mode = DrivingMode.FOLLOW_LANE
        self._tl_waiting = None
        self._cleared_stop_signs.clear()
        self._stop_sign_wait_counter = 0
        self._stop_sign_active = None

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

        # Dynamic speed limit from road signs (falls back to config)
        cfg = self.config
        road_limit = self.vehicle.get_speed_limit()  # km/h
        if road_limit is not None and road_limit > 0:
            base_target = min(road_limit * cfg.speed_limit_margin, cfg.max_speed)
        else:
            base_target = cfg.target_speed

        # --- 2.  Route progress ---
        self._advance_route(transform)
        target_wp, nav_command = self._get_target(transform)

        # --- 3.  Privileged perception ---
        tl_state, tl_distance = self._check_traffic_light(transform)
        ss_distance = self._check_stop_signs(transform, speed_kmh)
        nearby_vehicles = self._detect_vehicles(transform)
        nearby_pedestrians = self._detect_pedestrians(transform)
        lead_dist = self._lead_vehicle_distance(nearby_vehicles)
        ped_dist = self._closest_pedestrian_distance(nearby_pedestrians)

        # --- 4.  Target speed & driving mode ---
        target_speed, mode = self._plan_speed(
            speed_kmh,
            nav_command,
            tl_state,
            tl_distance,
            ss_distance,
            lead_dist,
            ped_dist,
            base_target,
        )
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
        # Corridor diagnostics (Option D)
        n_segs = len(self._corridor) if isinstance(self._corridor, list) else 0
        # _corridor is the polygon (np.ndarray), segments are not stored
        # separately — count via the corridor builder's output.  We
        # re-derive n_segs from the number of vehicles that passed SAT.
        # Actually, count from the corridor_segs stored on the instance.
        buf_len = len(self._wp_buffer)
        buf_head_dist = -1.0
        if self._wp_buffer:
            wp0_loc = self._wp_buffer[0][0].transform.location
            buf_head_dist = transform.location.distance(wp0_loc)

        sat_hits = sum(1 for v in nearby_vehicles if v.get("obb_in_lane", False))
        lat_hits = sum(
            1
            for v in nearby_vehicles
            if v["is_ahead"]
            and v.get("same_direction", True)
            and abs(v["lateral"]) < cfg.lane_half_width
        )

        info = FrameInfo(
            nav_command=int(nav_command),
            route_completion=self.route_completion,
            target_speed=target_speed,
            current_speed=speed_kmh,
            driving_mode=int(mode),
            position=(transform.location.x, transform.location.y, transform.location.z),
            rotation=(transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll),
            velocity=(velocity.x, velocity.y, velocity.z),
            traffic_light_state=self._encode_tl(tl_state),
            traffic_light_distance=(tl_distance if tl_distance is not None else -1.0),
            stop_sign_distance=(ss_distance if ss_distance is not None else -1.0),
            nearby_vehicles=nearby_vehicles,
            nearby_pedestrians=nearby_pedestrians,
            n_corridor_segments=self._n_corridor_segments,
            wp_buffer_size=buf_len,
            wp_buffer_head_dist=round(buf_head_dist, 2),
            sat_detections=sat_hits,
            lateral_detections=lat_hits,
        )

        return control, info

    # ------------------------------------------------------------------
    # Internal — route buffer
    # ------------------------------------------------------------------

    def _fill_buffer(self) -> None:
        """Refill the waypoint look-ahead buffer from the route.

        When the planned route is exhausted but the buffer is below
        half capacity, extend it by querying the map for waypoints
        ahead of the last buffered waypoint.  This prevents corridor
        degeneration in the final ~200 frames of an episode where the
        buffer would otherwise shrink to 1–2 entries.
        """
        while (
            len(self._wp_buffer) < self._wp_buffer.maxlen and self._route_index < self._route_length
        ):
            self._wp_buffer.append(self._route[self._route_index])
            self._route_index += 1

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
                while nexts and len(self._wp_buffer) < self._wp_buffer.maxlen:
                    nxt = nexts[0]  # follow the first (keep-lane) option
                    self._wp_buffer.append((nxt, RoadOption.LANEFOLLOW))
                    nexts = nxt.next(2.0)
            except (AttributeError, RuntimeError):
                pass  # graceful degradation if map query fails

    def _advance_route(self, transform: "carla.Transform") -> None:
        """Pop consumed waypoints (closer than *min_follow_distance* or behind ego)."""
        location = transform.location
        fwd = transform.get_forward_vector()

        while len(self._wp_buffer) > 1:
            wp, _ = self._wp_buffer[0]
            wp_loc = wp.transform.location
            dist = location.distance(wp_loc)

            dx = wp_loc.x - location.x
            dy = wp_loc.y - location.y
            is_behind = (fwd.x * dx + fwd.y * dy) < 0

            if dist < self.config.min_follow_distance or is_behind:
                self._wp_buffer.popleft()
            else:
                break
        self._fill_buffer()

    def _get_target(
        self,
        transform: "carla.Transform",
    ) -> Tuple["carla.Waypoint", RoadOption]:
        """Return the target waypoint and the current nav command.

        Navigation command logic:
            Scan ahead in the buffer for the first non-LANEFOLLOW command
            within ``turn_anticipation_distance``.  If found, report that
            command so the model (and the speed planner) can anticipate
            the upcoming turn.  Otherwise report LANEFOLLOW.
        """
        if not self._wp_buffer:
            wp = self.map.get_waypoint(transform.location)
            return wp, RoadOption.LANEFOLLOW

        ego_loc = transform.location

        # --- Determine nav command with anticipation ---
        nav_command = RoadOption.LANEFOLLOW
        self._turn_distance = None
        for wp, cmd in self._wp_buffer:
            if cmd in (
                RoadOption.LANEFOLLOW,
                RoadOption.VOID,
                RoadOption.CHANGELANELEFT,
                RoadOption.CHANGELANERIGHT,
            ):
                continue
            dist = ego_loc.distance(wp.transform.location)
            if dist < self.config.turn_anticipation_distance:
                nav_command = cmd
                self._turn_distance = dist
            # break is intentional: we only consider the *first*
            # actionable command (LEFT/RIGHT/STRAIGHT).  If it is
            # beyond turn_anticipation_distance we keep LANEFOLLOW,
            # but either way we stop scanning here.
            break

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

        target_wp, _ = self._wp_buffer[0]
        for wp, _ in self._wp_buffer:
            if ego_loc.distance(wp.transform.location) >= follow_dist:
                target_wp = wp
                break

        return target_wp, nav_command

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
        lead_dist: float | None,
        ped_dist: float | None,
        base_target: float | None = None,
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
                    target = min(target, v_ms * 3.6)
                else:
                    target = min(target, turn_target)

        # --- Traffic light ---
        # TL stop takes priority when the ramp demands a speed lower
        # than whatever the turn section already set.  Without this,
        # a turn target of ~15 km/h masks the TL ramp (which is still
        # above 15 km/h at 20-40 m distance), so the car approaches
        # a red TL at turn speed and has to slam at the last moment.
        if tl_state is not None and tl_distance is not None:
            if tl_state in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow):
                if 0 < tl_distance < cfg.traffic_light_brake_distance:
                    if tl_distance < cfg.traffic_light_stop_distance:
                        mode = DrivingMode.TRAFFIC_LIGHT_STOP
                        target = 0.0
                    else:
                        # Kinematic profile: v = sqrt(2 * a * d)
                        d = max(tl_distance - cfg.traffic_light_stop_distance, 0.0)
                        v_ms = math.sqrt(2.0 * cfg.tl_comfort_decel * d)
                        tl_target = v_ms * 3.6  # m/s → km/h
                        if tl_target < target:
                            mode = DrivingMode.TRAFFIC_LIGHT_STOP
                            target = tl_target

        # --- Stop sign ---
        if ss_distance is not None and 0 < ss_distance < cfg.stop_sign_brake_distance:
            mode = DrivingMode.STOP_SIGN_STOP
            if ss_distance < cfg.stop_sign_stop_distance:
                target = 0.0
            else:
                d = max(ss_distance - cfg.stop_sign_stop_distance, 0.0)
                v_ms = math.sqrt(2.0 * cfg.stop_sign_comfort_decel * d)
                target = min(target, v_ms * 3.6)

        # --- Lead vehicle ---
        # Scale safety distances with current speed (kinematic braking
        # envelope).  At low speed the fixed minimums apply; at high
        # speed the distances stretch to preserve reaction time.
        speed_ms = speed_kmh / 3.6
        dyn_hazard = max(
            cfg.vehicle_hazard_distance,
            speed_ms * cfg.vehicle_reaction_time,
        )
        dyn_emergency = max(
            cfg.vehicle_emergency_distance,
            speed_ms * cfg.vehicle_emergency_time,
        )

        if lead_dist is not None:
            if lead_dist < dyn_emergency:
                mode = DrivingMode.EMERGENCY_STOP
                target = 0.0
            elif lead_dist < dyn_hazard:
                mode = DrivingMode.VEHICLE_FOLLOWING
                # Kinematic braking profile: v = sqrt(2 * a * d)
                # Same physics-based curve used for traffic-light stops.
                # Produces smooth, progressive braking that matches how
                # a professional driver modulates brake pressure.
                d = max(lead_dist - dyn_emergency, 0.0)
                v_ms = math.sqrt(2.0 * cfg.vehicle_comfort_decel * d)
                target = min(target, v_ms * 3.6)  # m/s → km/h

        # --- Pedestrians ---
        if ped_dist is not None:
            if ped_dist < cfg.pedestrian_emergency_distance:
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
        """PID steering on the signed heading error to the target waypoint."""
        ego = transform.location
        tgt = target_wp.transform.location

        # Desired heading
        target_yaw = math.degrees(math.atan2(tgt.y - ego.y, tgt.x - ego.x))

        # Heading error wrapped to [−180, 180]
        error = target_yaw - transform.rotation.yaw
        error = (error + 180) % 360 - 180

        # Normalise so ±90° → ±1.0
        return self.steer_pid.step(error / 90.0)

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
        error = (target_speed - speed_kmh) / max(self.config.target_speed, 10.0)
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
                brake = 0.5           # firm hold when already stopped
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
           ``get_affected_lane_waypoints()`` for early braking.
        """
        ego_loc = transform.location

        # ---- Tier 1: persistent TL memory ----
        if self._tl_waiting is not None:
            tl_actor = self.world.get_actor(self._tl_waiting)
            if tl_actor is not None and tl_actor.state in (
                carla.TrafficLightState.Red,
                carla.TrafficLightState.Yellow,
            ):
                tl_loc = tl_actor.get_transform().location
                dist = ego_loc.distance(tl_loc)

                # Clear stale memory if:
                #   (a) we drove far past the light (distance cutoff), OR
                #   (b) the light is behind us (dot-product with forward).
                # The behind-ego check clears within 1–2 frames of
                # passing, avoiding the 100+ frame stale blocks that
                # pollute traffic_light_state in the HDF5 data.
                fwd = transform.get_forward_vector()
                dx = tl_loc.x - ego_loc.x
                dy = tl_loc.y - ego_loc.y
                tl_behind = (fwd.x * dx + fwd.y * dy) < 0

                if dist > self.config.traffic_light_brake_distance * 1.5 or tl_behind:
                    self._tl_waiting = None
                else:
                    # Use stop-line distance if available
                    stop_dist = self._stop_line_distance(
                        tl_actor,
                        transform,
                    )
                    return tl_actor.state, (stop_dist if stop_dist is not None else dist)
            else:
                # Light turned green or was destroyed — clear memory.
                self._tl_waiting = None

        # ---- Tier 2: CARLA built-in (close range) ----
        if self.vehicle.is_at_traffic_light():
            tl = self.vehicle.get_traffic_light()
            if tl is not None:
                if tl.state == carla.TrafficLightState.Green:
                    return None, None  # green → no action needed
                self._tl_waiting = tl.id  # remember red/yellow
                stop_dist = self._stop_line_distance(tl, transform)
                dist = (
                    stop_dist
                    if stop_dist is not None
                    else ego_loc.distance(tl.get_transform().location)
                )
                return tl.state, dist

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

        for tl in self.world.get_actors().filter("*traffic_light*"):
            if tl.state == carla.TrafficLightState.Green:
                continue

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
            stop_dist: float | None = None

            # Predict upcoming road ID via buffer to detect TL early
            upcoming_road_id = self._wp_buffer[0][0].road_id if self._wp_buffer else ego_wp.road_id

            try:
                for awp in tl.get_affected_lane_waypoints():
                    if awp.lane_id == ego_wp.lane_id and (
                        awp.road_id == ego_wp.road_id or awp.road_id == upcoming_road_id
                    ):
                        # Verify the stop line is AHEAD of us, not behind.
                        # Without this check, a cross-street TL whose stop
                        # line shares a lane_id at the junction boundary
                        # can be falsely matched, causing braking at green.
                        awp_loc = awp.transform.location
                        sdx = awp_loc.x - ego_loc.x
                        sdy = awp_loc.y - ego_loc.y
                        if fwd.x * sdx + fwd.y * sdy <= 0:
                            continue  # stop line is behind us
                        affects_us = True
                        stop_dist = ego_loc.distance(awp_loc)
                        break
            except (AttributeError, RuntimeError):
                # Fallback for older CARLA builds.
                tl_wp = self.map.get_waypoint(tl_loc)
                if tl_wp is not None and (
                    tl_wp.road_id == ego_wp.road_id or tl_wp.road_id == upcoming_road_id
                ):
                    affects_us = True

            if not affects_us:
                continue

            # Prefer stop-line distance; fall back to pole distance.
            effective_dist = stop_dist if stop_dist is not None else pole_dist
            if effective_dist > self.config.traffic_light_brake_distance:
                continue

            if best_dist is None or effective_dist < best_dist:
                best_state = tl.state
                best_dist = effective_dist
                best_tl_id = tl.id

        if best_tl_id is not None:
            self._tl_waiting = best_tl_id

        return best_state, best_dist

    def _stop_line_distance(
        self,
        tl: Any,
        transform: "carla.Transform",
    ) -> float | None:
        """Distance from ego to the traffic-light's stop line.

        Returns ``None`` if the API is unavailable or no stop waypoint
        matches our lane.
        """
        try:
            ego_wp = self.map.get_waypoint(transform.location)
            for awp in tl.get_affected_lane_waypoints():
                if awp.road_id == ego_wp.road_id and awp.lane_id == ego_wp.lane_id:
                    return transform.location.distance(
                        awp.transform.location,
                    )
        except (AttributeError, RuntimeError):
            pass
        return None

    # --- Stop sign detection --------------------------------------------

    def _check_stop_signs(
        self,
        transform: "carla.Transform",
        speed_kmh: float,
    ) -> float | None:
        """Return distance to the nearest uncleared stop sign, or ``None``.

        Uses a three-stage approach inspired by TransFuser:

        1. **Lane matching** — only react to stop signs whose trigger
           volume waypoint matches the ego's road and lane.
        2. **Ahead check** — ignore signs that are behind the ego.
        3. **Distance gating** — only return signs within
           ``stop_sign_brake_distance``.

        A stop sign is "cleared" once the ego has come to a full stop
        within ``stop_sign_stop_distance`` for at least
        ``stop_sign_wait_ticks`` consecutive ticks.  Once cleared, the
        sign is remembered so the ego doesn't brake for it again.
        """
        cfg = self.config
        ego_loc = transform.location
        ego_wp = self.map.get_waypoint(ego_loc)
        if ego_wp is None:
            return None

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
                dist = ego_loc.distance(ss_loc)

                # Check if behind us (we've driven past)
                dx = ss_loc.x - ego_loc.x
                dy = ss_loc.y - ego_loc.y
                behind = (fwd.x * dx + fwd.y * dy) < 0

                if behind or dist > cfg.stop_sign_brake_distance * 1.5:
                    # Drove past it — add to cleared
                    self._cleared_stop_signs.add(self._stop_sign_active)
                    self._stop_sign_active = None
                    self._stop_sign_wait_counter = 0
                elif speed_kmh < 0.5 and dist < cfg.stop_sign_stop_distance * 2:
                    self._stop_sign_wait_counter += 1
                    if self._stop_sign_wait_counter >= cfg.stop_sign_wait_ticks:
                        # Waited long enough — clear the sign
                        self._cleared_stop_signs.add(self._stop_sign_active)
                        self._stop_sign_active = None
                        self._stop_sign_wait_counter = 0
                        return None
                    return dist
                else:
                    self._stop_sign_wait_counter = 0
                    return dist

        # ---- Scan for new stop signs ----
        upcoming_road_id = (
            self._wp_buffer[0][0].road_id if self._wp_buffer else ego_wp.road_id
        )

        best_dist: float | None = None
        best_id: int | None = None

        for ss in self.world.get_actors().filter("*stop*"):
            if ss.id in self._cleared_stop_signs:
                continue

            ss_loc = ss.get_transform().location
            dist = ego_loc.distance(ss_loc)
            if dist > cfg.stop_sign_brake_distance:
                continue

            # Ahead check
            dx = ss_loc.x - ego_loc.x
            dy = ss_loc.y - ego_loc.y
            if fwd.x * dx + fwd.y * dy <= 0:
                continue

            # Lane matching — stop sign should be on our road
            ss_wp = self.map.get_waypoint(ss_loc)
            if ss_wp is None:
                continue
            if ss_wp.road_id != ego_wp.road_id and ss_wp.road_id != upcoming_road_id:
                continue

            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_id = ss.id

        if best_id is not None:
            self._stop_sign_active = best_id
            self._stop_sign_wait_counter = 0

        return best_dist

    # --- Curve-aware corridor -------------------------------------------

    def _build_route_corridor(
        self,
        ego_loc: "carla.Location",
        cutoff: float,
        half_lane: float,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Build a lane corridor that follows the planned route.

        Instead of projecting a straight rectangle ahead of the ego,
        this samples waypoints from ``_wp_buffer`` up to *cutoff*
        metres and offsets each by ± *half_lane* perpendicular to the
        waypoint's heading.

        Returns ``(polygon, segments)``:

        * **polygon** — full corridor outline for debug drawing.
        * **segments** — list of convex quadrilaterals (one per
          consecutive waypoint pair).  SAT overlap must be tested
          per-segment because the full polygon is non-convex on
          curved roads.

        Falls back to a single rectangle when no route waypoints
        are available.
        """
        # Collect centre-line samples from the route
        samples: list[tuple[np.ndarray, np.ndarray]] = []
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
            samples.append((centre, right_2d))

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
            return rect, [rect]

        # Build corridor as individual convex quad segments + full polygon.
        # Each segment is a perfect rectangle aligned to the vector
        # connecting consecutive centre-line samples.  This guarantees
        # convexity even on very tight inner turns where per-waypoint
        # normal offsets would create self-intersecting "bowtie" quads.
        prev_center = ego_xy
        left_boundary: list[np.ndarray] = []
        right_boundary: list[np.ndarray] = []
        segments: list[np.ndarray] = []

        for centre, _right_2d in samples:
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
            left_boundary.append(prev_l)
            right_boundary.append(prev_r)
            prev_center = centre

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
        return polygon, segments

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
        half_lane = self.config.lane_half_width

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
        corridor_poly, corridor_segs = self._build_route_corridor(
            ego_loc,
            cutoff,
            half_lane,
        )
        self._corridor = corridor_poly  # expose for debug drawing
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

        def _project(vertices: np.ndarray, axis: np.ndarray):
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

    def _lead_vehicle_distance(
        self,
        nearby: List[Dict[str, Any]],
    ) -> float | None:
        """Distance to the closest vehicle ahead in our lane.

        A vehicle is considered "in lane" if **either** the OBB/SAT
        corridor overlap test says so, **or** its centre-point lateral
        offset is within ``lane_half_width``.  Using both checks as a
        union avoids false negatives from the SAT on road segments
        where the corridor geometry is unreliable, while the SAT
        still provides accurate detection on curved roads where the
        lateral-only heuristic would miss offset vehicles.

        Oncoming vehicles (``same_direction == False``) are excluded
        so that head-on traffic on narrow roads or during turns does
        not trigger false lead-vehicle braking.
        """
        for v in nearby:
            if not v["is_ahead"]:
                continue
            # Skip oncoming traffic
            if not v.get("same_direction", True):
                continue
            # Union of SAT overlap and lateral-distance check — either
            # one is sufficient to flag the vehicle as in-lane.
            obb_flag = v.get("obb_in_lane", False)
            lateral_flag = abs(v["lateral"]) < self.config.lane_half_width
            if obb_flag or lateral_flag:
                return v["distance"]
        return None

    def _closest_pedestrian_distance(
        self,
        nearby: List[Dict[str, Any]],
    ) -> float | None:
        """Distance to the closest pedestrian hazard ahead **in our lane corridor**.

        A pedestrian is considered a hazard if:
        - They are currently in the lane (lateral < pedestrian_lane_half_width), **or**
        - Their predicted position (1s ahead via velocity extrapolation)
          will be in the lane (``predicted_lateral`` < half-width).

        This enables early braking for pedestrians about to step off
        the pavement into the vehicle's path.
        """
        half_w = self.config.pedestrian_lane_half_width
        for p in nearby:
            current_in_lane = p["is_ahead"] and abs(p["lateral"]) < half_w
            predicted_in_lane = p.get("predicted_ahead", False) and abs(
                p.get("predicted_lateral", 999.0)
            ) < half_w
            if current_in_lane or predicted_in_lane:
                # Use whichever distance is shorter (current or predicted)
                pred_dist = p.get("predicted_distance", p["distance"])
                return min(p["distance"], pred_dist)
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
