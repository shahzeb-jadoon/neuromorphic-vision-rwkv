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

from agents.navigation.local_planner import RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ExpertConfig:
    """Tunable parameters for the privileged expert.

    All speeds are in **km/h** unless stated otherwise.
    All distances are in **metres**.
    """

    # Target cruising speed
    target_speed: float = 30.0
    max_speed: float = 50.0

    # --- PID: longitudinal (speed) ---
    speed_kp: float = 0.8
    speed_ki: float = 0.05
    speed_kd: float = 0.01

    # --- PID: lateral (steering) ---
    steer_kp: float = 1.2
    steer_ki: float = 0.0
    steer_kd: float = 0.15

    # --- Waypoint following ---
    min_follow_distance: float = 3.0  # metres — pop threshold
    target_follow_distance: float = 5.0  # metres — desired look-ahead
    max_follow_distance: float = 20.0  # metres — upper bound
    waypoint_buffer_size: int = 40

    # --- Safety thresholds ---
    traffic_light_brake_distance: float = 20.0
    traffic_light_stop_distance: float = 4.0
    vehicle_hazard_distance: float = 20.0
    vehicle_emergency_distance: float = 5.0
    pedestrian_hazard_distance: float = 15.0
    pedestrian_emergency_distance: float = 5.0
    lane_half_width: float = 2.5  # metres — lateral check for "in-lane"
    pedestrian_lane_half_width: float = 3.0  # metres — wider lateral band for pedestrians

    # --- Turn behaviour ---
    turn_speed_factor: float = 0.55  # reduce speed to 55% at turns
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
            lead_dist, ped_dist,
        )
        self.mode = mode

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
        target_wp, _ = self._wp_buffer[0]
        for wp, _ in self._wp_buffer:
            if ego_loc.distance(wp.transform.location) >= self.config.target_follow_distance:
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
    ) -> Tuple[float, DrivingMode]:
        """Determine target speed and driving mode from hazards."""
        cfg = self.config
        target = cfg.target_speed
        mode = DrivingMode.FOLLOW_LANE

        # --- Curvature (route bends) ---
        curve_angle = self._route_curvature()
        if curve_angle > cfg.curvature_threshold_deg:
            target = min(target, cfg.target_speed * cfg.curvature_speed_factor)

        # --- Upcoming turn ---
        if nav_command in (RoadOption.LEFT, RoadOption.RIGHT):
            mode = DrivingMode.APPROACHING_TURN
            target = min(target, cfg.target_speed * cfg.turn_speed_factor)

        # At a junction and actively turning → even slower
        ego_wp = self.map.get_waypoint(self.vehicle.get_transform().location)
        if ego_wp is not None and ego_wp.is_junction:
            if nav_command in (RoadOption.LEFT, RoadOption.RIGHT):
                mode = DrivingMode.TURNING
                target = min(target, cfg.target_speed * cfg.turn_speed_factor * 0.8)

        # --- Traffic light ---
        if tl_state is not None and tl_distance is not None:
            if tl_state in (carla.TrafficLightState.Red,
                            carla.TrafficLightState.Yellow):
                if 0 < tl_distance < cfg.traffic_light_brake_distance:
                    mode = DrivingMode.TRAFFIC_LIGHT_STOP
                    if tl_distance < cfg.traffic_light_stop_distance:
                        target = 0.0
                    else:
                        # Smooth deceleration proportional to distance
                        ratio = tl_distance / cfg.traffic_light_brake_distance
                        target = min(target, speed_kmh * ratio)

        # --- Lead vehicle ---
        if lead_dist is not None:
            if lead_dist < cfg.vehicle_emergency_distance:
                mode = DrivingMode.EMERGENCY_STOP
                target = 0.0
            elif lead_dist < cfg.vehicle_hazard_distance:
                mode = DrivingMode.VEHICLE_FOLLOWING
                ratio = lead_dist / cfg.vehicle_hazard_distance
                target = min(target, cfg.target_speed * ratio)

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

    def _lateral_control(
        self, transform: "carla.Transform", target_wp: "carla.Waypoint",
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

        # Hard brake when significantly over target
        if speed_kmh > target_speed + 15.0:
            throttle = 0.0
            brake = 1.0

        return throttle, brake

    # ------------------------------------------------------------------
    # Internal — privileged perception
    # ------------------------------------------------------------------

    def _check_traffic_light(
        self, transform: "carla.Transform",
    ) -> Tuple[Any, float | None]:
        """Return (state, distance) of the relevant traffic light, if any.

        Uses a two-tier strategy:
        1. **Fast path** — CARLA's built-in ``is_at_traffic_light()``
           (only fires when the vehicle is very close to the stop line).
        2. **Junction-aware scan** — for each non-green traffic light
           within ``traffic_light_brake_distance``, check whether any of
           the light's ``get_affected_lane_waypoints()`` shares both
           ``road_id`` and ``lane_id`` with the ego waypoint.  This
           correctly handles multi-road junctions where the naïve
           ``road_id`` comparison fails.
        """
        # Fast path: CARLA's own detector
        if self.vehicle.is_at_traffic_light():
            tl = self.vehicle.get_traffic_light()
            if tl is not None:
                dist = transform.location.distance(tl.get_transform().location)
                return tl.state, dist

        # Junction-aware scan
        ego_wp = self.map.get_waypoint(transform.location)
        if ego_wp is None:
            return None, None

        fwd = transform.get_forward_vector()
        threshold = self.config.traffic_light_brake_distance

        best_state, best_dist = None, None

        for tl in self.world.get_actors().filter("*traffic_light*"):
            if tl.state == carla.TrafficLightState.Green:
                continue

            tl_loc = tl.get_transform().location
            dist = transform.location.distance(tl_loc)
            if dist > threshold:
                continue

            # Ahead of us?  (positive dot product with forward vector)
            dx = tl_loc.x - transform.location.x
            dy = tl_loc.y - transform.location.y
            if fwd.x * dx + fwd.y * dy <= 0:
                continue

            # Junction-aware lane matching: check whether *any* of the
            # lanes this traffic light controls matches our road+lane.
            affects_us = False
            try:
                affected_wps = tl.get_affected_lane_waypoints()
                for awp in affected_wps:
                    if (awp.road_id == ego_wp.road_id
                            and awp.lane_id == ego_wp.lane_id):
                        affects_us = True
                        break
            except AttributeError:
                # Fallback for older CARLA builds that lack the method:
                # revert to the simple road_id comparison.
                tl_wp = self.map.get_waypoint(tl_loc)
                if tl_wp is not None and tl_wp.road_id == ego_wp.road_id:
                    affects_us = True

            if not affects_us:
                continue

            if best_dist is None or dist < best_dist:
                best_state = tl.state
                best_dist = dist

        return best_state, best_dist

    def _detect_vehicles(
        self, transform: "carla.Transform",
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

        # Ego bounding-box centroid in world space
        ego_bb_center = transform.transform(self.vehicle.bounding_box.location)
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

            v_transform = v.get_transform()
            v_loc = v_transform.location

            # Quick cull on actor-root distance (cheap)
            if ego_loc.distance(v_loc) > cutoff:
                continue

            # Bounding-box centroid in world space
            v_bb_center = v_transform.transform(v.bounding_box.location)
            center_dist = ego_bb_center.distance(v_bb_center)

            dx = v_bb_center.x - ego_bb_center.x
            dy = v_bb_center.y - ego_bb_center.y
            longitudinal = fwd.x * dx + fwd.y * dy  # positive = ahead
            lateral = right.x * dx + right.y * dy  # positive = right

            v_vel = v.get_velocity()
            v_speed = math.sqrt(v_vel.x ** 2 + v_vel.y ** 2 + v_vel.z ** 2) * 3.6

            # Heading similarity: positive = same direction
            v_fwd = v_transform.get_forward_vector()
            heading_dot = fwd.x * v_fwd.x + fwd.y * v_fwd.y

            # Bumper-to-bumper distance
            v_half_len = v.bounding_box.extent.x
            bumper_gap = max(center_dist - ego_half_len - v_half_len, 0.0)

            # OBB overlap check via SAT
            v_corners = self._get_actor_obb_2d(v)
            obb_in_lane = self._sat_overlap(corridor, v_corners)

            nearby.append({
                "distance": round(bumper_gap, 2),
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
        """
        for v in nearby:
            if not v["is_ahead"]:
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
