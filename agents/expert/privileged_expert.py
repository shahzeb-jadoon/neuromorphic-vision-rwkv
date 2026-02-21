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

import math
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
    speed_limit_margin: float = 0.90  # drive at 90 % of posted limit

    # --- PID: longitudinal (speed) — straight-line defaults ---
    speed_kp: float = 0.8
    speed_ki: float = 0.05
    speed_kd: float = 0.01

    # --- PID: lateral (steering) — straight-line defaults ---
    steer_kp: float = 1.2
    steer_ki: float = 0.0
    steer_kd: float = 0.15
    cross_track_gain: float = 2.0  # Stanley k for cross-track correction

    # --- Speed-adaptive PID gains (active during turns) ---
    turn_speed_kp: float = 1.0
    turn_speed_ki: float = 0.1
    turn_speed_kd: float = 0.05
    turn_steer_kp: float = 1.5
    turn_steer_ki: float = 0.05
    turn_steer_kd: float = 0.3
    turn_cross_track_gain: float = 3.0  # tighter CTE correction in turns

    # --- Waypoint following ---
    min_follow_distance: float = 3.0  # metres — pop threshold
    target_follow_distance: float = 8.0  # metres — desired look-ahead
    turn_follow_distance: float = 3.0  # metres — shorter look-ahead in turns
    max_follow_distance: float = 20.0  # metres — upper bound
    waypoint_buffer_size: int = 40

    # --- Safety thresholds ---
    traffic_light_brake_distance: float = 35.0  # metres — start braking
    traffic_light_stop_distance: float = 4.0  # metres — full stop
    tl_comfort_decel: float = 3.0  # m/s² — comfortable deceleration for TL
    vehicle_hazard_distance: float = 25.0  # metres — start slowing (min)
    vehicle_emergency_distance: float = 5.0  # metres — emergency (min)
    vehicle_reaction_time: float = 2.0  # seconds — scale hazard dist with speed
    vehicle_emergency_time: float = 0.8  # seconds — scale emergency with speed
    pedestrian_hazard_distance: float = 15.0
    pedestrian_emergency_distance: float = 5.0
    lane_half_width: float = 2.5  # metres — lateral check for "in-lane"
    pedestrian_lane_half_width: float = 3.0  # metres — wider lateral band for pedestrians

    # --- Turn behaviour ---
    turn_speed_factor: float = 0.45  # reduce speed to 45% at turns
    junction_turn_factor: float = 0.65  # additional reduction inside junctions
    turn_anticipation_distance: float = 30.0  # start slowing this far ahead

    # --- Speed limits / curvature ---
    curvature_speed_factor: float = 0.7  # reduce by 30% on curves
    curvature_threshold_deg: float = 15.0  # angle change between successive wps


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

    # Nearby actors (serialisable lists of dicts)
    nearby_vehicles: List[Dict[str, Any]] = field(default_factory=list)
    nearby_pedestrians: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PID controller
# ---------------------------------------------------------------------------


class PIDController:
    """Discrete PID with anti-windup clamping."""

    def __init__(self, kp: float, ki: float, kd: float,
                 integral_limit: float = 10.0) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral = 0.0
        self._prev_error = 0.0
        self._limit = integral_limit

    def step(self, error: float, dt: float = 0.05) -> float:
        """Compute PID output for a single time-step."""
        self._integral = max(-self._limit,
                             min(self._limit, self._integral + error * dt))
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

    # ------------------------------------------------------------------
    # Route management
    # ------------------------------------------------------------------

    def set_route(
        self, route: List[Tuple["carla.Waypoint", RoadOption]]
    ) -> None:
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

    def set_destination(
        self, start: "carla.Location", end: "carla.Location"
    ) -> None:
        """Plan and load a route from *start* to *end*."""
        route = self.route_planner.trace_route(start, end)
        self.set_route(route)

    def done(self) -> bool:
        """``True`` when the route has been consumed."""
        if not self._route:
            return True
        return (self._route_index >= self._route_length
                and len(self._wp_buffer) <= 1)

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
        speed_ms = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        speed_kmh = speed_ms * 3.6

        # Dynamic speed limit from road signs (falls back to config)
        cfg = self.config
        road_limit = self.vehicle.get_speed_limit()  # km/h
        if road_limit is not None and road_limit > 0:
            base_target = min(road_limit * cfg.speed_limit_margin, cfg.max_speed)
        else:
            base_target = cfg.target_speed

        # --- 2.  Route progress ---
        self._advance_route(transform.location)
        target_wp, nav_command = self._get_target(transform)

        # --- 3.  Privileged perception ---
        tl_state, tl_distance = self._check_traffic_light(transform)
        nearby_vehicles = self._detect_vehicles(transform)
        nearby_pedestrians = self._detect_pedestrians(transform)
        lead_dist = self._lead_vehicle_distance(nearby_vehicles)
        ped_dist = self._closest_pedestrian_distance(nearby_pedestrians)

        # --- 4.  Target speed & driving mode ---
        target_speed, mode = self._plan_speed(
            speed_kmh, nav_command, tl_state, tl_distance,
            lead_dist, ped_dist, base_target,
        )
        self.mode = mode
        self._apply_adaptive_gains()

        # --- 5.  Control ---
        steer = self._lateral_control(transform, target_wp)
        throttle, brake = self._longitudinal_control(speed_kmh, target_speed, dt)

        control = carla.VehicleControl(
            throttle=float(np.clip(throttle, 0.0, 1.0)),
            steer=float(np.clip(steer, -1.0, 1.0)),
            brake=float(np.clip(brake, 0.0, 1.0)),
            hand_brake=False,
            manual_gear_shift=False,
        )

        # --- 6.  Privileged info ---
        info = FrameInfo(
            nav_command=int(nav_command),
            route_completion=self.route_completion,
            target_speed=target_speed,
            current_speed=speed_kmh,
            driving_mode=int(mode),
            position=(transform.location.x,
                      transform.location.y,
                      transform.location.z),
            rotation=(transform.rotation.pitch,
                      transform.rotation.yaw,
                      transform.rotation.roll),
            velocity=(velocity.x, velocity.y, velocity.z),
            traffic_light_state=self._encode_tl(tl_state),
            traffic_light_distance=(tl_distance if tl_distance is not None
                                    else -1.0),
            nearby_vehicles=nearby_vehicles,
            nearby_pedestrians=nearby_pedestrians,
        )

        return control, info

    # ------------------------------------------------------------------
    # Internal — route buffer
    # ------------------------------------------------------------------

    def _fill_buffer(self) -> None:
        """Refill the waypoint look-ahead buffer from the route."""
        while (len(self._wp_buffer) < self._wp_buffer.maxlen
               and self._route_index < self._route_length):
            self._wp_buffer.append(self._route[self._route_index])
            self._route_index += 1

    def _advance_route(self, location: "carla.Location") -> None:
        """Pop consumed waypoints (closer than *min_follow_distance*)."""
        while len(self._wp_buffer) > 1:
            wp, _ = self._wp_buffer[0]
            if location.distance(wp.transform.location) < self.config.min_follow_distance:
                self._wp_buffer.popleft()
            else:
                break
        self._fill_buffer()

    def _get_target(
        self, transform: "carla.Transform",
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
        for wp, cmd in self._wp_buffer:
            if cmd in (RoadOption.LANEFOLLOW, RoadOption.VOID,
                       RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
                continue
            dist = ego_loc.distance(wp.transform.location)
            if dist < self.config.turn_anticipation_distance:
                nav_command = cmd
            # break is intentional: we only consider the *first*
            # actionable command (LEFT/RIGHT/STRAIGHT).  If it is
            # beyond turn_anticipation_distance we keep LANEFOLLOW,
            # but either way we stop scanning here.
            break

        # --- Pick target waypoint at desired follow distance ---
        # Use route curvature to adapt follow distance regardless of
        # nav command.  Curved roads tagged LANEFOLLOW still need a
        # shorter look-ahead to prevent corner-cutting.
        curve_angle = self._route_curvature()
        if curve_angle > self.config.curvature_threshold_deg:
            follow_dist = self.config.turn_follow_distance
        elif nav_command in (RoadOption.LEFT, RoadOption.RIGHT):
            follow_dist = self.config.turn_follow_distance
        else:
            follow_dist = self.config.target_follow_distance

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
        if nav_command in (RoadOption.LEFT, RoadOption.RIGHT):
            mode = DrivingMode.APPROACHING_TURN
            target = min(target, target * cfg.turn_speed_factor)

        # At a junction and actively turning → even slower
        ego_wp = self.map.get_waypoint(self.vehicle.get_transform().location)
        if ego_wp is not None and ego_wp.is_junction:
            if nav_command in (RoadOption.LEFT, RoadOption.RIGHT):
                mode = DrivingMode.TURNING
                target = min(target, target * cfg.junction_turn_factor)

        # --- Traffic light ---
        if tl_state is not None and tl_distance is not None:
            if tl_state in (carla.TrafficLightState.Red,
                            carla.TrafficLightState.Yellow):
                if 0 < tl_distance < cfg.traffic_light_brake_distance:
                    mode = DrivingMode.TRAFFIC_LIGHT_STOP
                    if tl_distance < cfg.traffic_light_stop_distance:
                        target = 0.0
                    else:
                        # Kinematic profile: v = sqrt(2 * a * d)
                        # Produces a smooth, constant-deceleration stop
                        # at the stop line — no oscillation.
                        d = max(tl_distance - cfg.traffic_light_stop_distance, 0.0)
                        v_ms = math.sqrt(2.0 * cfg.tl_comfort_decel * d)
                        target = min(target, v_ms * 3.6)  # m/s → km/h

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
                ratio = lead_dist / dyn_hazard
                target = min(target, target * ratio)

        # --- Pedestrians ---
        if ped_dist is not None:
            if ped_dist < cfg.pedestrian_emergency_distance:
                mode = DrivingMode.EMERGENCY_STOP
                target = 0.0
            elif ped_dist < cfg.pedestrian_hazard_distance:
                mode = DrivingMode.PEDESTRIAN_STOP
                target = min(target, 8.0)  # creep at 8 km/h

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

    def _apply_adaptive_gains(self) -> None:
        """Switch PID gains based on driving mode (turn vs straight).

        Turns need tighter, more aggressive steering and speed control
        to track the curved path accurately.  On straights, softer gains
        produce smoother, more natural demonstrations.
        """
        cfg = self.config
        if self.mode in (
            DrivingMode.APPROACHING_TURN,
            DrivingMode.TURNING,
        ):
            self.steer_pid.kp = cfg.turn_steer_kp
            self.steer_pid.ki = cfg.turn_steer_ki
            self.steer_pid.kd = cfg.turn_steer_kd
            self.speed_pid.kp = cfg.turn_speed_kp
            self.speed_pid.ki = cfg.turn_speed_ki
            self.speed_pid.kd = cfg.turn_speed_kd
        else:
            self.steer_pid.kp = cfg.steer_kp
            self.steer_pid.ki = cfg.steer_ki
            self.steer_pid.kd = cfg.steer_kd
            self.speed_pid.kp = cfg.speed_kp
            self.speed_pid.ki = cfg.speed_ki
            self.speed_pid.kd = cfg.speed_kd

    def _lateral_control(
        self, transform: "carla.Transform", target_wp: "carla.Waypoint",
    ) -> float:
        """Stanley-style steering: heading error + cross-track error.

        Uses the **Stanley controller** formulation from Stanford's 2005
        DARPA Grand Challenge entry:

            δ = heading_error + arctan(k * cross_track_error / v)

        This corrects both the *direction* the car is pointing (heading)
        **and** how far the car has drifted laterally from the path.
        Without the cross-track term, curved LANEFOLLOW roads cause
        systematic inside-cutting because heading error alone is small
        when the car is pointed roughly forward but offset sideways.

        References:
            Thrun et al., "Stanley: The Robot that Won the DARPA Grand
            Challenge", Journal of Field Robotics, 2006.
        """
        ego_loc = transform.location
        tgt_loc = target_wp.transform.location

        # ---- 1. Heading error (PID-filtered) ----
        target_yaw = math.degrees(math.atan2(
            tgt_loc.y - ego_loc.y, tgt_loc.x - ego_loc.x,
        ))
        heading_err = target_yaw - transform.rotation.yaw
        heading_err = (heading_err + 180) % 360 - 180

        # PID on normalised heading error (same as before)
        pid_out = self.steer_pid.step(heading_err / 90.0)

        # ---- 2. Cross-track error (Stanley term) ----
        # Nearest point on the segment between the first two buffer wps.
        # Cross-track error = signed perpendicular distance from ego to
        # the route segment.  Positive = ego is to the *right* of path.
        cte = self._cross_track_error(transform)

        # Stanley gain — inversely proportional to speed so the
        # correction is strong at low speed (parking-lot precision)
        # and gentle at high speed (highway stability).
        speed_ms = math.sqrt(
            self.vehicle.get_velocity().x ** 2
            + self.vehicle.get_velocity().y ** 2
        )
        if self.mode in (DrivingMode.APPROACHING_TURN, DrivingMode.TURNING):
            k = self.config.turn_cross_track_gain
        else:
            k = self.config.cross_track_gain

        cte_steer = math.atan2(k * cte, max(speed_ms, 1.0))
        # Convert radians → normalised [−1, 1] (90° = 1.0)
        cte_steer_norm = cte_steer / (math.pi / 2)

        return pid_out + cte_steer_norm

    def _cross_track_error(self, transform: "carla.Transform") -> float:
        """Signed perpendicular distance from ego to the nearest route segment.

        Positive → ego is to the RIGHT of the path.
        Negative → ego is to the LEFT.
        Zero     → ego is on the path.
        """
        if len(self._wp_buffer) < 2:
            return 0.0

        # Segment: first two waypoints in the look-ahead buffer
        a = self._wp_buffer[0][0].transform.location
        b = self._wp_buffer[1][0].transform.location
        p = transform.location

        # Vector AB and AP
        abx, aby = b.x - a.x, b.y - a.y
        apx, apy = p.x - a.x, p.y - a.y

        # Signed cross product: positive when P is to the right of AB
        cross = abx * apy - aby * apx
        seg_len = math.sqrt(abx * abx + aby * aby)
        if seg_len < 1e-6:
            return 0.0

        return cross / seg_len

    def _longitudinal_control(
        self, speed_kmh: float, target_speed: float, dt: float,
    ) -> Tuple[float, float]:
        """PID throttle / brake on the speed error."""
        error = (target_speed - speed_kmh) / max(self.config.target_speed, 10.0)
        output = self.speed_pid.step(error, dt)

        if output >= 0:
            throttle = min(output, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(abs(output), 1.0)

        # When target ≈ 0 (red light, emergency), always full-brake
        # and zero throttle.  This prevents the PID integral from
        # producing tiny positive outputs that cause creeping.
        if target_speed < 1.0:
            throttle = 0.0
            brake = 1.0

        # Hard brake when well above target — safety net for normal
        # driving.  The threshold is 10 km/h (not 5) to avoid
        # oscillation during gradual deceleration profiles.
        elif speed_kmh > target_speed + 10.0:
            throttle = 0.0
            brake = 1.0

        return throttle, brake

    # ------------------------------------------------------------------
    # Internal — privileged perception
    # ------------------------------------------------------------------

    def _check_traffic_light(
        self, transform: "carla.Transform",
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
                dist = ego_loc.distance(tl_actor.get_transform().location)
                # Use stop-line distance if available
                stop_dist = self._stop_line_distance(
                    tl_actor, transform,
                )
                return tl_actor.state, (stop_dist if stop_dist is not None
                                        else dist)
            # Light turned green (or was destroyed) — clear memory.
            self._tl_waiting = None

        # ---- Tier 2: CARLA built-in (close range) ----
        if self.vehicle.is_at_traffic_light():
            tl = self.vehicle.get_traffic_light()
            if tl is not None:
                if tl.state != carla.TrafficLightState.Green:
                    self._tl_waiting = tl.id  # remember it
                stop_dist = self._stop_line_distance(tl, transform)
                dist = (stop_dist if stop_dist is not None
                        else ego_loc.distance(tl.get_transform().location))
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
            try:
                for awp in tl.get_affected_lane_waypoints():
                    if (awp.road_id == ego_wp.road_id
                            and awp.lane_id == ego_wp.lane_id):
                        affects_us = True
                        stop_dist = ego_loc.distance(
                            awp.transform.location,
                        )
                        break
            except (AttributeError, RuntimeError):
                # Fallback for older CARLA builds.
                tl_wp = self.map.get_waypoint(tl_loc)
                if tl_wp is not None and tl_wp.road_id == ego_wp.road_id:
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
                if (awp.road_id == ego_wp.road_id
                        and awp.lane_id == ego_wp.lane_id):
                    return transform.location.distance(
                        awp.transform.location,
                    )
        except (AttributeError, RuntimeError):
            pass
        return None

    def _detect_vehicles(
        self, transform: "carla.Transform",
    ) -> List[Dict[str, Any]]:
        """Return up to 10 nearest vehicles with privileged info.

        Distances are measured **bumper-to-bumper**: the raw centre-to-
        centre distance minus both vehicles' longitudinal half-extents.
        This is critical for large vehicles (buses, trucks) where the
        centre-of-mass can be 5–6 m from the rear bumper.

        Each entry also includes an ``obb_in_lane`` flag computed via
        Oriented Bounding-Box (OBB) projection + SAT overlap.
        """
        ego_loc = transform.location
        fwd = transform.get_forward_vector()
        right = carla.Vector3D(-fwd.y, fwd.x, 0)
        cutoff = self.config.vehicle_hazard_distance * 2
        half_lane = self.config.lane_half_width

        # Ego half-extent along its forward axis
        ego_half_len = self.vehicle.bounding_box.extent.x

        # Build ego lane corridor polygon (rectangle ahead of ego)
        ego_xy = np.array([ego_loc.x, ego_loc.y])
        fwd_2d = np.array([fwd.x, fwd.y])
        right_2d = np.array([right.x, right.y])
        corridor = np.array([
            ego_xy - right_2d * half_lane,
            ego_xy + right_2d * half_lane,
            ego_xy + fwd_2d * cutoff + right_2d * half_lane,
            ego_xy + fwd_2d * cutoff - right_2d * half_lane,
        ])

        nearby: list[dict[str, Any]] = []
        for v in self.world.get_actors().filter("*vehicle*"):
            if v.id == self.vehicle.id:
                continue
            v_loc = v.get_transform().location
            center_dist = ego_loc.distance(v_loc)
            if center_dist > cutoff:
                continue

            dx = v_loc.x - ego_loc.x
            dy = v_loc.y - ego_loc.y
            longitudinal = fwd.x * dx + fwd.y * dy  # positive = ahead
            lateral = right.x * dx + right.y * dy  # positive = right

            v_vel = v.get_velocity()
            v_speed = math.sqrt(v_vel.x ** 2 + v_vel.y ** 2 + v_vel.z ** 2) * 3.6

            # Heading similarity: positive = same direction
            v_fwd = v.get_transform().get_forward_vector()
            heading_dot = fwd.x * v_fwd.x + fwd.y * v_fwd.y

            # Bumper-to-bumper distance: subtract both half-extents
            v_half_len = v.bounding_box.extent.x
            bumper_gap = center_dist - ego_half_len - v_half_len

            # OBB overlap check via SAT
            v_corners = self._get_actor_obb_2d(v)
            obb_in_lane = self._sat_overlap(corridor, v_corners)

            nearby.append({
                "distance": round(max(bumper_gap, 0.0), 2),
                "center_distance": round(center_dist, 2),
                "longitudinal": round(longitudinal, 2),
                "lateral": round(lateral, 2),
                "speed_kmh": round(v_speed, 1),
                "is_ahead": longitudinal > 0,
                "same_direction": heading_dot > 0,
                "obb_in_lane": obb_in_lane,
            })

        nearby.sort(key=lambda x: x["distance"])
        return nearby[:10]

    # --- OBB helpers ------------------------------------------------

    @staticmethod
    def _get_actor_obb_2d(actor: Any) -> np.ndarray:
        """Return the 4 XY corners of *actor*'s oriented bounding box.

        Uses ``actor.bounding_box`` (half-extents in local frame) and
        the actor's world transform to produce a (4, 2) float array
        of world-space 2-D corner coordinates.
        """
        bb = actor.bounding_box
        ext = bb.extent  # half-extents (x=length/2, y=width/2)
        t = actor.get_transform()

        # 4 corners in local frame (XY plane, Z ignored)
        local_corners = [
            carla.Location(x=+ext.x, y=+ext.y, z=0),
            carla.Location(x=+ext.x, y=-ext.y, z=0),
            carla.Location(x=-ext.x, y=-ext.y, z=0),
            carla.Location(x=-ext.x, y=+ext.y, z=0),
        ]
        world_corners = np.array([
            [t.transform(c).x, t.transform(c).y]
            for c in local_corners
        ])
        return world_corners

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
        self, transform: "carla.Transform",
    ) -> List[Dict[str, Any]]:
        """Return up to 5 nearest pedestrians with privileged info.

        Each entry now includes ``lateral`` offset so that
        ``_closest_pedestrian_distance`` can ignore pedestrians that
        are on the pavement and not in the vehicle's lane corridor.
        """
        ego_loc = transform.location
        fwd = transform.get_forward_vector()
        right = carla.Vector3D(-fwd.y, fwd.x, 0)
        cutoff = self.config.pedestrian_hazard_distance * 2

        nearby: list[dict[str, Any]] = []
        for w in self.world.get_actors().filter("*walker*"):
            w_loc = w.get_transform().location
            dist = ego_loc.distance(w_loc)
            if dist > cutoff:
                continue

            dx = w_loc.x - ego_loc.x
            dy = w_loc.y - ego_loc.y
            longitudinal = fwd.x * dx + fwd.y * dy
            lateral = right.x * dx + right.y * dy

            nearby.append({
                "distance": round(dist, 2),
                "longitudinal": round(longitudinal, 2),
                "lateral": round(lateral, 2),
                "is_ahead": longitudinal > 0,
            })

        nearby.sort(key=lambda x: x["distance"])
        return nearby[:5]

    def _lead_vehicle_distance(
        self, nearby: List[Dict[str, Any]],
    ) -> float | None:
        """Distance to the closest vehicle ahead in our lane.

        Uses the OBB-based ``obb_in_lane`` flag when available (set by
        ``_detect_vehicles``).  Falls back to centre-point lateral
        check for backwards compatibility.

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
            # Prefer OBB overlap flag if present
            in_lane = v.get("obb_in_lane",
                            abs(v["lateral"]) < self.config.lane_half_width)
            if in_lane:
                return v["distance"]
        return None

    def _closest_pedestrian_distance(
        self, nearby: List[Dict[str, Any]],
    ) -> float | None:
        """Distance to the closest pedestrian ahead **in our lane corridor**.

        Uses ``pedestrian_lane_half_width`` (default 3.0 m) to ignore
        pedestrians on the pavement who are not an imminent hazard.
        """
        half_w = self.config.pedestrian_lane_half_width
        for p in nearby:
            if p["is_ahead"] and abs(p["lateral"]) < half_w:
                return p["distance"]
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
