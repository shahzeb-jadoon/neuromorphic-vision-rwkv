"""Expert data collector for CARLA autonomous driving episodes.

Drives a custom **PrivilegedExpert** agent through pre-planned routes
while recording per-frame observations (3-camera panoramic, controls,
navigation commands, ego state, nearby-actor metadata) to HDF5 files.

Schema version: 3 (see ``src/data/hdf5_schema.py``).

Key differences from the v2 collector:
    - Uses ``PrivilegedExpert`` (custom PID + privileged perception)
      instead of CARLA's TrafficManager autopilot.
    - Spawns 3 RGB cameras (left / centre / right) for panoramic input.
    - Stores ego position, rotation, traffic light state, and
      nearby-actor metadata for future-proofing (GPS, LiDAR, etc.).

Handles three failure modes:
    - CARLA crash recovery (reconnect with retries)
    - Episode termination (completed / collision / timeout)
    - Campaign resumption via collection_manifest.json
"""

from __future__ import annotations

from src.data import hdf5_schema as S

import hashlib
import json
import logging
import math
import queue
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# RoadOption is a pure-Python IntEnum with no carla dependency,
# so import it unconditionally.
from agents.navigation.local_planner import RoadOption

try:
    import carla

    from agents.expert.privileged_expert import (  # noqa: F401
        ExpertConfig,
        FrameInfo,
        PrivilegedExpert,
    )
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    CARLA_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CARLA_AVAILABLE = False

    # Provide a lightweight stub so EpisodeConfig (which references
    # ExpertConfig in its default_factory) can still be defined.
    @dataclass
    class ExpertConfig:  # type: ignore[no-redef]
        """Stub — real config lives in agents.expert.privileged_expert."""

        target_speed: float = 30.0


logger = logging.getLogger(__name__)


@dataclass
class EpisodeConfig:
    """Configuration for a single data collection episode."""

    town: str = "Town01"
    weather: str = "ClearNoon"
    n_vehicles: int = 50
    n_walkers: int = 30
    max_frames: int = 2000
    timeout_speed: float = 0.1  # m/s — below this for timeout_duration → discard
    timeout_duration: float = 30.0  # seconds stuck before early termination
    collision_grace_frames: int = 10  # frames to record after collision, then stop
    min_npc_spawn_distance: float = 50.0  # metres — don't spawn NPCs closer than this to ego
    fps: int = 20

    # 3-camera panoramic setup
    camera_width: int = 640  # per-camera width
    camera_height: int = 320  # per-camera height
    camera_fov: int = 60
    camera_yaws: tuple[float, ...] = (-60.0, 0.0, 60.0)  # left, centre, right

    # Camera mount (vehicle frame, metres)
    camera_x: float = 1.5
    camera_z: float = 2.4

    # Visualization / recording
    spectator_follow: bool = False  # Attach CARLA spectator camera to ego vehicle
    spectator_offset: tuple[float, float, float] = (-8.0, 0.0, 5.0)  # (x, y, z) behind/above
    record_simulation: bool = False  # Use CARLA built-in recorder
    debug_draw: bool = False  # Draw route + trajectory in CARLA viewport

    # Expert agent
    expert_config: ExpertConfig = field(default_factory=ExpertConfig)


@dataclass
class EpisodeResult:
    """Result of a single episode collection."""

    status: str  # "completed", "collision", "crashed", "timeout"
    filename: str | None = None
    episode_hash: str | None = None
    n_frames: int = 0
    town: str = ""
    weather: str = ""
    nav_distribution: dict[str, int] = field(default_factory=dict)
    collision_frame: int | None = None
    error_message: str | None = None
    duration_seconds: float = 0.0
    recording_path: str | None = None


class ExpertDataCollector:
    """Collects expert driving data from CARLA using PrivilegedExpert.

    Drives a fully custom expert agent with PID control and privileged
    hazard perception.  Records 3-camera panoramic images, expert
    controls, navigation commands, ego state, and nearby-actor metadata.

    Usage::

        collector = ExpertDataCollector(host="auto", port=2000,
                                        output_dir="data/collected")
        collector.connect()
        result = collector.collect_episode(config, route_waypoints)

    Args:
        host: CARLA server host ("auto" to detect from WSL2 gateway).
        port: CARLA server port.
        output_dir: Directory to save HDF5 files and manifest.
        crash_retries: Number of reconnection attempts on CARLA crash.
        crash_wait: Seconds to wait between reconnection attempts.
    """

    def __init__(
        self,
        host: str = "auto",
        port: int = 2000,
        output_dir: str | Path = "data/collected",
        crash_retries: int = 3,
        crash_wait: float = 30.0,
    ) -> None:
        if not CARLA_AVAILABLE:
            raise RuntimeError("carla package not installed. Run: pip install carla==0.9.16")

        self.port = port
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.crash_retries = crash_retries
        self.crash_wait = crash_wait

        # Resolve host
        if host == "auto":
            self.host = self._detect_wsl2_host() or "localhost"
        else:
            self.host = host

        self.client: carla.Client | None = None
        self.world: carla.World | None = None

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_wsl2_host() -> str | None:
        """Detect Windows host IP from WSL2 default gateway."""
        import subprocess

        try:
            result = subprocess.run(
                ["ip", "route", "show", "default"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip().split()[2]
        except (subprocess.CalledProcessError, IndexError, FileNotFoundError):
            return None

    def connect(self) -> None:
        """Connect to CARLA server."""
        logger.info("Connecting to CARLA at %s:%d", self.host, self.port)
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(60.0)
        self.world = self.client.get_world()
        logger.info(
            "Connected — server=%s, map=%s",
            self.client.get_server_version(),
            self.world.get_map().name,
        )

    def _reconnect(self) -> bool:
        """Attempt to reconnect after CARLA crash."""
        for attempt in range(1, self.crash_retries + 1):
            logger.warning(
                "Reconnection attempt %d/%d (waiting %ds)...",
                attempt,
                self.crash_retries,
                int(self.crash_wait),
            )
            time.sleep(self.crash_wait)
            try:
                self.connect()
                return True
            except Exception as e:
                logger.error("Reconnection failed: %s", e)
        return False

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _update_spectator(
        world: "carla.World",
        vehicle: "carla.Vehicle",
        offset: tuple[float, float, float] = (-8.0, 0.0, 5.0),
    ) -> None:
        """Move the CARLA spectator camera to follow the ego vehicle.

        The spectator is positioned behind and above the vehicle so
        you can watch it drive in the CARLA viewport in real-time.

        Args:
            world: CARLA world instance.
            vehicle: Ego vehicle to follow.
            offset: (x, y, z) offset in vehicle-local frame.
                    Default ``(-8, 0, 5)`` gives a chase-cam view.
        """
        vt = vehicle.get_transform()
        # Compute world-space offset using vehicle's forward/right/up
        fwd = vt.get_forward_vector()
        right = vt.get_right_vector()
        up = carla.Location(x=0.0, y=0.0, z=1.0)

        cam_loc = vt.location + carla.Location(
            x=fwd.x * offset[0] + right.x * offset[1] + up.x * offset[2],
            y=fwd.y * offset[0] + right.y * offset[1] + up.y * offset[2],
            z=fwd.z * offset[0] + right.z * offset[1] + up.z * offset[2],
        )
        cam_rot = carla.Rotation(
            pitch=-15.0,
            yaw=vt.rotation.yaw,
            roll=0.0,
        )
        world.get_spectator().set_transform(carla.Transform(cam_loc, cam_rot))

    @staticmethod
    def _draw_route_debug(
        world: "carla.World",
        expert: "PrivilegedExpert",
        vehicle: "carla.Vehicle",
        frame_idx: int,
    ) -> None:
        """Draw debug markers in the CARLA viewport.

        Renders three layers so you can see:
          1. **Planned route** — green dots along the remaining route
          2. **Actual position** — orange breadcrumb at current location
          3. **Upcoming target** — cyan sphere at the expert's steering target

        All markers use ``life_time=0.3`` so they refresh each tick without
        accumulating clutter.
        """
        debug = world.debug
        life = 0.3  # seconds — slightly longer than 1 tick so no flicker

        # 1. Planned route — draw every 5th remaining waypoint in green
        route = expert._route
        idx = expert._route_index
        GREEN = carla.Color(0, 255, 0)
        for i in range(idx, min(len(route), idx + 200), 5):
            wp, _opt = route[i]
            loc = wp.transform.location
            debug.draw_point(
                carla.Location(loc.x, loc.y, loc.z + 0.5),
                size=0.08,
                color=GREEN,
                life_time=life,
            )

        # Draw route as connected line segments (every 2nd point)
        for i in range(idx, min(len(route) - 1, idx + 200), 2):
            wp_a = route[i][0].transform.location
            wp_b = route[i + 1][0].transform.location
            debug.draw_line(
                carla.Location(wp_a.x, wp_a.y, wp_a.z + 0.5),
                carla.Location(wp_b.x, wp_b.y, wp_b.z + 0.5),
                thickness=0.06,
                color=GREEN,
                life_time=life,
            )

        # 2. Actual position — orange breadcrumb (longer life for trail)
        ego_loc = vehicle.get_transform().location
        ORANGE = carla.Color(255, 165, 0)
        debug.draw_point(
            carla.Location(ego_loc.x, ego_loc.y, ego_loc.z + 0.5),
            size=0.10,
            color=ORANGE,
            life_time=8.0,  # persist for 8s to form a visible trail
        )

        # 3. Current steering target — cyan sphere
        if expert._wp_buffer:
            target_wp = expert._wp_buffer[0]
            t_loc = target_wp[0].transform.location
            CYAN = carla.Color(0, 255, 255)
            debug.draw_point(
                carla.Location(t_loc.x, t_loc.y, t_loc.z + 1.0),
                size=0.15,
                color=CYAN,
                life_time=life,
            )
            # Line from ego to target
            debug.draw_line(
                carla.Location(ego_loc.x, ego_loc.y, ego_loc.z + 0.5),
                carla.Location(t_loc.x, t_loc.y, t_loc.z + 1.0),
                thickness=0.04,
                color=CYAN,
                life_time=life,
            )

        # 4. Frame counter near vehicle
        debug.draw_string(
            carla.Location(ego_loc.x, ego_loc.y, ego_loc.z + 3.0),
            f"F{frame_idx}",
            draw_shadow=True,
            color=carla.Color(255, 255, 255),
            life_time=life,
        )

    @staticmethod
    def _spawn_cameras(
        world: "carla.World",
        vehicle: "carla.Vehicle",
        config: EpisodeConfig,
    ) -> list[tuple["carla.Actor", queue.Queue]]:
        """Spawn 3 RGB cameras and return (actor, image_queue) pairs.

        Cameras are ordered LEFT, CENTRE, RIGHT so that horizontal
        concatenation produces a seamless panoramic image.
        """
        bp_lib = world.get_blueprint_library()
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(config.camera_width))
        camera_bp.set_attribute("image_size_y", str(config.camera_height))
        camera_bp.set_attribute("fov", str(config.camera_fov))

        cameras: list[tuple[carla.Actor, queue.Queue]] = []

        for yaw in config.camera_yaws:
            cam_transform = carla.Transform(
                carla.Location(x=config.camera_x, z=config.camera_z),
                carla.Rotation(yaw=yaw),
            )
            cam = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)
            img_q: queue.Queue = queue.Queue()
            cam.listen(lambda img, q=img_q: q.put(img))
            cameras.append((cam, img_q))

        return cameras

    @staticmethod
    def _capture_panoramic(
        cameras: list[tuple["carla.Actor", queue.Queue]],
        config: EpisodeConfig,
        timeout: float = 2.0,
    ) -> np.ndarray:
        """Capture one frame from all cameras and stitch horizontally.

        Returns:
            (H, W_total, 3) uint8 RGB panoramic image where
            W_total = n_cameras x per_camera_width.
        """
        panels: list[np.ndarray] = []

        for _cam, img_q in cameras:
            try:
                carla_image = img_q.get(timeout=timeout)
                raw = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
                raw = raw.reshape(config.camera_height, config.camera_width, 4)
                rgb = raw[:, :, :3][:, :, ::-1].copy()  # BGRA → RGB
                panels.append(rgb)
            except queue.Empty:
                panels.append(
                    np.zeros(
                        (config.camera_height, config.camera_width, 3),
                        dtype=np.uint8,
                    )
                )

        return np.concatenate(panels, axis=1)  # (H, W_total, 3)

    # ------------------------------------------------------------------
    # Episode collection
    # ------------------------------------------------------------------

    def collect_episode(
        self,
        config: EpisodeConfig,
        route_waypoints: list | None = None,
    ) -> EpisodeResult:
        """Collect a single episode of expert driving data.

        Uses the custom PrivilegedExpert agent for driving.  Navigation
        commands come directly from the route planner — no heuristic
        inference needed.

        Args:
            config: Episode configuration (cameras, expert, etc.).
            route_waypoints: Pre-computed route as
                ``[(carla.Transform, RoadOption_str), ...]``.
                If ``None``, a random route is generated.

        Returns:
            EpisodeResult with status and metadata.
        """
        if self.client is None:
            self.connect()

        start_time = time.time()
        actors: list = []
        npc_vehicles: list = []
        walkers: list = []
        walker_controllers: list = []
        original_settings = None
        cameras: list[tuple[carla.Actor, queue.Queue]] = []

        # Data buffers
        images_buf: list[np.ndarray] = []
        steering_buf: list[float] = []
        throttle_buf: list[float] = []
        brake_buf: list[float] = []
        nav_cmd_buf: list[int] = []
        # Privileged / future-proofing buffers
        position_buf: list[np.ndarray] = []
        rotation_buf: list[np.ndarray] = []
        velocity_buf: list[np.ndarray] = []
        traffic_light_buf: list[int] = []
        target_speed_buf: list[float] = []
        driving_mode_buf: list[int] = []
        nearby_vehicles_buf: list[str] = []  # JSON strings
        nearby_pedestrians_buf: list[str] = []  # JSON strings
        collision_frame: int | None = None

        try:
            assert self.world is not None
            assert self.client is not None

            # --- Setup world ---
            if self.world.get_map().name.split("/")[-1] != config.town:
                logger.info("Loading map: %s", config.town)
                self.client.set_timeout(120.0)
                self.client.load_world(config.town)
                self.world = self.client.get_world()
                self.client.set_timeout(60.0)
                time.sleep(5.0)

            carla_map = self.world.get_map()
            original_settings = self.world.get_settings()

            # Enable synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / config.fps
            self.world.apply_settings(settings)

            # Set weather
            weather_preset = getattr(carla.WeatherParameters, config.weather, None)
            if weather_preset is not None:
                self.world.set_weather(weather_preset)

            # --- Spawn ego vehicle ---
            bp_lib = self.world.get_blueprint_library()
            vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
            spawn_points = carla_map.get_spawn_points()

            # When a pre-planned route is provided, teleport the vehicle
            # directly to the route’s first waypoint transform.  This
            # guarantees the car starts on the route facing the correct
            # direction (same approach TransFuser / Leaderboard uses).
            # Track ego spawn location for NPC distance filtering.
            ego_spawn_loc: Any = None

            if route_waypoints:
                # Lift spawn 0.5m above road surface so the vehicle's
                # physics bounding-box doesn't clip the ground (route
                # waypoint transforms sit at exact road level).
                raw = route_waypoints[0][0]  # carla.Transform
                spawn_transform = carla.Transform(
                    carla.Location(
                        x=raw.location.x,
                        y=raw.location.y,
                        z=raw.location.z + 0.5,
                    ),
                    raw.rotation,
                )
                vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
                if vehicle is None:
                    # Fallback: exact location may be obstructed; try
                    # closest spawn point instead.
                    route_start_loc = raw.location
                    distances = [sp.location.distance(route_start_loc) for sp in spawn_points]
                    spawn_idx = int(np.argmin(distances))
                    vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_points[spawn_idx])
                    ego_spawn_loc = spawn_points[spawn_idx].location
                    logger.warning(
                        "Teleport spawn failed; fell back to spawn point %d "
                        "(%.1fm from route start)",
                        spawn_idx,
                        distances[spawn_idx],
                    )
                else:
                    ego_spawn_loc = spawn_transform.location
                    logger.info(
                        "Teleported ego to route start at (%.1f, %.1f, %.1f)",
                        spawn_transform.location.x,
                        spawn_transform.location.y,
                        spawn_transform.location.z,
                    )
            else:
                spawn_idx = np.random.randint(len(spawn_points))
                vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_points[spawn_idx])
                ego_spawn_loc = spawn_points[spawn_idx].location

            if vehicle is None:
                return EpisodeResult(
                    status="crashed",
                    error_message="Failed to spawn vehicle",
                    duration_seconds=time.time() - start_time,
                )

            actors.append(vehicle)

            # --- Spawn NPC traffic (driven by TrafficManager) ---
            traffic_manager = self.client.get_trafficmanager(8000)
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_global_distance_to_leading_vehicle(2.5)

            if config.n_vehicles > 0:
                npc_bps = bp_lib.filter("vehicle.*")
                npc_bps = [bp for bp in npc_bps if int(bp.get_attribute("number_of_wheels")) == 4]
                available_spawns = [
                    sp
                    for sp in spawn_points
                    if sp.location.distance(ego_spawn_loc) > config.min_npc_spawn_distance
                ]
                np.random.shuffle(available_spawns)
                n_to_spawn = min(config.n_vehicles, len(available_spawns))

                for i in range(n_to_spawn):
                    npc_bp = np.random.choice(npc_bps)
                    npc = self.world.try_spawn_actor(npc_bp, available_spawns[i])
                    if npc is not None:
                        npc.set_autopilot(True, traffic_manager.get_port())
                        npc_vehicles.append(npc)

                logger.info(
                    "Spawned %d NPC vehicles (requested %d)",
                    len(npc_vehicles),
                    config.n_vehicles,
                )

            # --- Spawn NPC walkers with AI controllers ---
            walkers = []
            walker_controllers = []
            if config.n_walkers > 0:
                walker_bps = bp_lib.filter("walker.pedestrian.*")
                # Gather random spawn points on sidewalks
                walker_spawn_points: list[Any] = []
                for _ in range(config.n_walkers * 2):
                    loc = self.world.get_random_location_from_navigation()
                    if loc is not None:
                        wp = carla.Transform(location=loc)
                        walker_spawn_points.append(wp)
                    if len(walker_spawn_points) >= config.n_walkers:
                        break

                n_walkers_to_spawn = min(config.n_walkers, len(walker_spawn_points))
                for i in range(n_walkers_to_spawn):
                    w_bp = np.random.choice(walker_bps)
                    # Enable crossing for ~30 % of walkers
                    if w_bp.has_attribute("is_invincible"):
                        w_bp.set_attribute("is_invincible", "false")
                    walker = self.world.try_spawn_actor(w_bp, walker_spawn_points[i])
                    if walker is not None:
                        walkers.append(walker)

                # Tick so walkers materialise before attaching controllers
                self.world.tick()

                # Attach WalkerAIController to each walker
                controller_bp = bp_lib.find("controller.ai.walker")
                for walker in walkers:
                    ctrl = self.world.spawn_actor(
                        controller_bp, carla.Transform(), attach_to=walker
                    )
                    walker_controllers.append(ctrl)

                # Tick again so controllers are ready
                self.world.tick()

                # Start walking: random destinations + varied speeds
                for ctrl in walker_controllers:
                    dest = self.world.get_random_location_from_navigation()
                    if dest is not None:
                        ctrl.start()
                        ctrl.go_to_location(dest)
                        # Vary speed: 1.0–2.5 m/s (walking to jogging)
                        ctrl.set_max_speed(1.0 + np.random.random() * 1.5)

                logger.info(
                    "Spawned %d NPC walkers (requested %d)",
                    len(walkers),
                    config.n_walkers,
                )

            # --- Attach 3 cameras ---
            cameras = self._spawn_cameras(self.world, vehicle, config)
            for cam, _ in cameras:
                actors.append(cam)

            # --- Attach collision sensor ---
            collision_bp = bp_lib.find("sensor.other.collision")
            collision_sensor = self.world.spawn_actor(
                collision_bp, carla.Transform(), attach_to=vehicle
            )
            actors.append(collision_sensor)
            collision_detected = [False]

            def on_collision(event: Any) -> None:
                collision_detected[0] = True

            collision_sensor.listen(on_collision)

            # --- Setup PrivilegedExpert ---
            route_planner = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)
            expert = PrivilegedExpert(vehicle, self.world, route_planner, config.expert_config)

            # Build route
            if route_waypoints:
                # route_waypoints = [(transform, road_option_str), ...]
                route = []
                for transform, road_opt_name in route_waypoints:
                    wp = carla_map.get_waypoint(transform.location)
                    opt = self._parse_road_option(str(road_opt_name))
                    route.append((wp, opt))
                expert.set_route(route)
                logger.info(
                    "Set custom route with %d waypoints for expert",
                    len(route),
                )
            else:
                # Random destination
                dest_idx = np.random.randint(len(spawn_points))
                while dest_idx == spawn_idx:
                    dest_idx = np.random.randint(len(spawn_points))
                expert.set_destination(
                    spawn_points[spawn_idx].location,
                    spawn_points[dest_idx].location,
                )
                logger.info("Set random route for expert")

            # Warm-up ticks
            for _ in range(10):
                self.world.tick()

            # Drain warm-up images from all cameras
            for _cam, img_q in cameras:
                while not img_q.empty():
                    try:
                        img_q.get_nowait()
                    except queue.Empty:
                        break

            # --- Start CARLA recorder if requested ---
            recording_path: str | None = None
            if config.record_simulation:
                rec_name = f"episode_{int(time.time())}.log"
                recording_path = rec_name
                self.client.start_recorder(rec_name, True)
                logger.info("CARLA recorder started: %s", rec_name)

            # --- Tick loop ---
            stuck_start: float | None = None
            distance_traveled = 0.0
            prev_location = vehicle.get_transform().location
            dt = 1.0 / config.fps

            for frame_idx in range(config.max_frames):
                # Check if expert finished route
                if expert.done():
                    logger.info("Expert completed route at frame %d", frame_idx)
                    break

                self.world.tick()

                # --- Expert step ---
                control, info = expert.run_step(dt)
                vehicle.apply_control(control)

                # --- Spectator follow (real-time visualization) ---
                if config.spectator_follow:
                    self._update_spectator(self.world, vehicle, config.spectator_offset)

                # --- In-CARLA debug drawing (route + trajectory) ---
                if config.debug_draw:
                    self._draw_route_debug(self.world, expert, vehicle, frame_idx)

                # --- Capture panoramic image ---
                panoramic = self._capture_panoramic(cameras, config)
                images_buf.append(panoramic)

                # --- Store control signals ---
                steering_buf.append(float(control.steer))
                throttle_buf.append(float(control.throttle))
                brake_buf.append(float(control.brake))

                # --- Navigation command (from expert route) ---
                nav_cmd_buf.append(info.nav_command)

                # --- Ego state ---
                position_buf.append(np.array(info.position, dtype=np.float32))
                rotation_buf.append(np.array(info.rotation, dtype=np.float32))
                velocity_buf.append(np.array(info.velocity, dtype=np.float32))

                # --- Privileged data ---
                traffic_light_buf.append(info.traffic_light_state)
                target_speed_buf.append(info.target_speed)
                driving_mode_buf.append(info.driving_mode)
                nearby_vehicles_buf.append(json.dumps(info.nearby_vehicles))
                nearby_pedestrians_buf.append(json.dumps(info.nearby_pedestrians))

                # --- Distance tracking ---
                cur_loc = vehicle.get_transform().location
                dx = cur_loc.x - prev_location.x
                dy = cur_loc.y - prev_location.y
                distance_traveled += math.sqrt(dx**2 + dy**2)
                prev_location = cur_loc

                # --- Collision check ---
                if collision_detected[0] and collision_frame is None:
                    collision_frame = frame_idx
                    logger.warning("Collision detected at frame %d", frame_idx)

                # Stop recording shortly after collision (grace period
                # captures the impact context for debugging).
                if (
                    collision_frame is not None
                    and frame_idx >= collision_frame + config.collision_grace_frames
                ):
                    logger.info(
                        "Stopping episode %d frames after collision",
                        config.collision_grace_frames,
                    )
                    break

                # --- Timeout (stuck) check ---
                speed_ms = math.sqrt(
                    info.velocity[0] ** 2 + info.velocity[1] ** 2 + info.velocity[2] ** 2
                )
                if speed_ms < config.timeout_speed:
                    if stuck_start is None:
                        stuck_start = time.time()
                    elif time.time() - stuck_start > config.timeout_duration:
                        logger.warning(
                            "Timeout: vehicle stuck for >%ds",
                            config.timeout_duration,
                        )
                        break
                else:
                    stuck_start = None

                # --- Progress logging ---
                if (frame_idx + 1) % 500 == 0:
                    logger.info(
                        "  Frame %d/%d — speed=%.1f km/h, dist=%.0fm, "
                        "steer=%.3f, mode=%s, route=%.0f%%",
                        frame_idx + 1,
                        config.max_frames,
                        info.current_speed,
                        distance_traveled,
                        control.steer,
                        info.driving_mode,
                        info.route_completion * 100,
                    )

            # --- Determine status ---
            timed_out = stuck_start is not None and (
                time.time() - stuck_start > config.timeout_duration
            )
            if timed_out:
                status = "timeout"
            elif collision_frame is not None:
                status = "collision"
            else:
                status = "completed"

            # --- Save to HDF5 ---
            if len(images_buf) > 0:
                result = self._save_episode(
                    images=np.stack(images_buf),
                    steering=np.array(steering_buf, dtype=np.float32),
                    throttle=np.array(throttle_buf, dtype=np.float32),
                    brake=np.array(brake_buf, dtype=np.float32),
                    nav_command=np.array(nav_cmd_buf, dtype=np.int32),
                    position=np.stack(position_buf),
                    rotation=np.stack(rotation_buf),
                    velocity=np.stack(velocity_buf),
                    traffic_light_state=np.array(traffic_light_buf, dtype=np.int32),
                    target_speed=np.array(target_speed_buf, dtype=np.float32),
                    driving_mode=np.array(driving_mode_buf, dtype=np.int32),
                    nearby_vehicles=nearby_vehicles_buf,
                    nearby_pedestrians=nearby_pedestrians_buf,
                    config=config,
                    status=status,
                    collision_frame=collision_frame,
                )
                result.duration_seconds = time.time() - start_time
                result.recording_path = recording_path
                return result
            else:
                return EpisodeResult(
                    status="crashed",
                    error_message="No frames collected",
                    duration_seconds=time.time() - start_time,
                )

        except Exception as e:
            logger.error("Episode crashed: %s\n%s", e, traceback.format_exc())

            # Save partial episode if we have data
            if len(images_buf) > 10:
                try:
                    result = self._save_episode(
                        images=np.stack(images_buf),
                        steering=np.array(steering_buf, dtype=np.float32),
                        throttle=np.array(throttle_buf, dtype=np.float32),
                        brake=np.array(brake_buf, dtype=np.float32),
                        nav_command=np.array(nav_cmd_buf, dtype=np.int32),
                        position=np.stack(position_buf),
                        rotation=np.stack(rotation_buf),
                        velocity=np.stack(velocity_buf),
                        traffic_light_state=np.array(traffic_light_buf, dtype=np.int32),
                        target_speed=np.array(target_speed_buf, dtype=np.float32),
                        driving_mode=np.array(driving_mode_buf, dtype=np.int32),
                        nearby_vehicles=nearby_vehicles_buf,
                        nearby_pedestrians=nearby_pedestrians_buf,
                        config=config,
                        status="crashed",
                        collision_frame=collision_frame,
                    )
                    result.error_message = str(e)
                    result.duration_seconds = time.time() - start_time
                    return result
                except Exception:
                    pass

            return EpisodeResult(
                status="crashed",
                error_message=str(e),
                n_frames=len(images_buf),
                town=config.town,
                weather=config.weather,
                duration_seconds=time.time() - start_time,
            )

        finally:
            # Stop CARLA recorder if it was started
            if config.record_simulation and self.client is not None:
                try:
                    self.client.stop_recorder()
                    logger.info("CARLA recorder stopped")
                except Exception:
                    pass

            # Stop sensors first
            for actor in actors:
                try:
                    if actor.is_alive and hasattr(actor, "stop"):
                        actor.stop()
                except Exception:
                    pass

            # Destroy NPC vehicles
            for npc in npc_vehicles:
                try:
                    if npc.is_alive:
                        npc.set_autopilot(False)
                        npc.destroy()
                except Exception:
                    pass

            # Destroy walker controllers then walkers
            for ctrl in walker_controllers:
                try:
                    if ctrl.is_alive:
                        ctrl.stop()
                        ctrl.destroy()
                except Exception:
                    pass
            for w in walkers:
                try:
                    if w.is_alive:
                        w.destroy()
                except Exception:
                    pass

            # Destroy ego actors (sensors, collision, vehicle)
            for actor in reversed(actors):
                try:
                    if actor.is_alive:
                        actor.destroy()
                except Exception:
                    pass

            # Restore world settings
            if original_settings is not None:
                try:
                    assert self.world is not None
                    self.world.apply_settings(original_settings)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save_episode(
        self,
        images: np.ndarray,
        steering: np.ndarray,
        throttle: np.ndarray,
        brake: np.ndarray,
        nav_command: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray,
        velocity: np.ndarray,
        traffic_light_state: np.ndarray,
        target_speed: np.ndarray,
        driving_mode: np.ndarray,
        nearby_vehicles: list[str],
        nearby_pedestrians: list[str],
        config: EpisodeConfig,
        status: str,
        collision_frame: int | None,
    ) -> EpisodeResult:
        """Save collected episode data to HDF5 (schema v3)."""
        # Hash-based filename
        timestamp_bytes = int(time.time() * 1000).to_bytes(8, "big")
        hash_input = timestamp_bytes + images[:4].tobytes()[:4096]
        episode_hash = hashlib.sha256(hash_input).hexdigest()[:8]
        filename = f"episode_{episode_hash}.h5"
        filepath = self.output_dir / filename

        pano_width = config.camera_width * len(config.camera_yaws)

        with h5py.File(filepath, "w") as f:
            # --- Images: panoramic (N, H, W_total, 3) ---
            f.create_dataset(
                S.IMAGES,
                data=images,
                compression="gzip",
                compression_opts=1,
                chunks=(1, config.camera_height, pano_width, 3),
            )

            # --- Controls ---
            f.create_dataset(S.STEERING, data=steering)
            f.create_dataset(S.THROTTLE, data=throttle)
            f.create_dataset(S.BRAKE, data=brake)

            # --- Navigation ---
            f.create_dataset(S.NAV_COMMAND, data=nav_command)

            # --- Ego state ---
            f.create_dataset(S.POSITION, data=position)
            f.create_dataset(S.ROTATION, data=rotation)
            f.create_dataset(S.VELOCITY, data=velocity)

            # --- Privileged data ---
            f.create_dataset(S.TRAFFIC_LIGHT_STATE, data=traffic_light_state)
            f.create_dataset(S.TARGET_SPEED, data=target_speed)
            f.create_dataset(S.DRIVING_MODE, data=driving_mode)

            # Variable-length JSON strings for actor lists
            dt_str = h5py.string_dtype()
            f.create_dataset(
                S.NEARBY_VEHICLES,
                data=np.array(nearby_vehicles, dtype=object),
                dtype=dt_str,
            )
            f.create_dataset(
                S.NEARBY_PEDESTRIANS,
                data=np.array(nearby_pedestrians, dtype=object),
                dtype=dt_str,
            )

            # --- Metadata ---
            meta = f.create_group(S.METADATA_GROUP)
            meta.attrs[S.META_SCHEMA_VERSION] = S.SCHEMA_VERSION
            meta.attrs[S.META_TOWN] = config.town
            meta.attrs[S.META_WEATHER] = config.weather
            meta.attrs[S.META_FPS] = config.fps
            meta.attrs[S.META_N_FRAMES] = len(images)
            meta.attrs[S.META_STATUS] = status
            meta.attrs[S.META_CAMERA_WIDTH] = config.camera_width
            meta.attrs[S.META_CAMERA_HEIGHT] = config.camera_height
            meta.attrs[S.META_N_CAMERAS] = len(config.camera_yaws)
            meta.attrs[S.META_CAMERA_FOV] = config.camera_fov
            meta.attrs[S.META_CAMERA_YAWS] = list(config.camera_yaws)
            meta.attrs[S.META_EXPERT_TYPE] = "privileged_expert"
            meta.attrs[S.META_EXPERT_TARGET_SPEED] = config.expert_config.target_speed
            if collision_frame is not None:
                meta.attrs[S.META_COLLISION_FRAME] = collision_frame

        # Nav distribution (using RoadOption integer values)
        nav_names = {
            int(RoadOption.LANEFOLLOW): "LaneFollow",
            int(RoadOption.LEFT): "Left",
            int(RoadOption.RIGHT): "Right",
            int(RoadOption.STRAIGHT): "Straight",
        }
        nav_dist = {}
        for cmd_id, cmd_name in nav_names.items():
            nav_dist[cmd_name] = int(np.sum(nav_command == cmd_id))

        logger.info(
            "Saved %s — %d frames (%dx%d panoramic), status=%s, town=%s",
            filename,
            len(images),
            pano_width,
            config.camera_height,
            status,
            config.town,
        )

        return EpisodeResult(
            status=status,
            filename=filename,
            episode_hash=episode_hash,
            n_frames=len(images),
            town=config.town,
            weather=config.weather,
            nav_distribution=nav_dist,
            collision_frame=collision_frame,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_road_option(name: str) -> RoadOption:
        """Parse a road option string to RoadOption enum."""
        mapping = {
            "LaneFollow": RoadOption.LANEFOLLOW,
            "Left": RoadOption.LEFT,
            "Right": RoadOption.RIGHT,
            "Straight": RoadOption.STRAIGHT,
            "RoadOption.LANEFOLLOW": RoadOption.LANEFOLLOW,
            "RoadOption.LEFT": RoadOption.LEFT,
            "RoadOption.RIGHT": RoadOption.RIGHT,
            "RoadOption.STRAIGHT": RoadOption.STRAIGHT,
            "LANEFOLLOW": RoadOption.LANEFOLLOW,
            "LEFT": RoadOption.LEFT,
            "RIGHT": RoadOption.RIGHT,
            "STRAIGHT": RoadOption.STRAIGHT,
            "CHANGELANELEFT": RoadOption.CHANGELANELEFT,
            "CHANGELANERIGHT": RoadOption.CHANGELANERIGHT,
            "ChangeLaneLeft": RoadOption.CHANGELANELEFT,
            "ChangeLaneRight": RoadOption.CHANGELANERIGHT,
        }
        return mapping.get(name, RoadOption.LANEFOLLOW)


class CollectionManifest:
    """Manages the collection_manifest.json for campaign resumption.

    Tracks every episode: hash, town, weather, frame count, turn
    distribution, status, and timestamp.
    """

    def __init__(self, manifest_path: str | Path) -> None:
        self.path = Path(manifest_path)
        self.data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        """Load existing manifest or create empty."""
        if self.path.exists():
            with open(self.path) as f:
                data: dict[str, Any] = json.load(f)
                return data
        return {
            "created": datetime.now(timezone.utc).isoformat(),
            "total_episodes": 0,
            "total_frames": 0,
            "episodes": [],
        }

    def add_episode(self, result: EpisodeResult) -> None:
        """Add an episode result to the manifest."""
        entry = {
            "filename": result.filename,
            "hash": result.episode_hash,
            "town": result.town,
            "weather": result.weather,
            "n_frames": result.n_frames,
            "status": result.status,
            "nav_distribution": result.nav_distribution,
            "collision_frame": result.collision_frame,
            "duration_seconds": result.duration_seconds,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if result.error_message:
            entry["error"] = result.error_message

        self.data["episodes"].append(entry)
        self.data["total_episodes"] = len(self.data["episodes"])
        self.data["total_frames"] = sum(e["n_frames"] for e in self.data["episodes"])
        self.data["last_updated"] = datetime.now(timezone.utc).isoformat()

    def save(self) -> None:
        """Write manifest to disk."""
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def get_completed_configs(self) -> set[tuple[str, str]]:
        """Return set of (town, weather) combos already completed."""
        return {
            (e["town"], e["weather"]) for e in self.data["episodes"] if e["status"] == "completed"
        }

    def get_completed_counts(self) -> dict[tuple[str, str], int]:
        """Return count of completed episodes per (town, weather) combo."""
        counts: dict[tuple[str, str], int] = {}
        for e in self.data["episodes"]:
            if e["status"] == "completed":
                key = (e["town"], e["weather"])
                counts[key] = counts.get(key, 0) + 1
        return counts

    def summary(self) -> str:
        """Return human-readable summary."""
        total_hrs = self.data["total_frames"] / (20 * 3600)
        status_counts: dict[str, int] = {}
        for e in self.data["episodes"]:
            s = e["status"]
            status_counts[s] = status_counts.get(s, 0) + 1

        lines = [
            f"Episodes: {self.data['total_episodes']}",
            f"Frames:   {self.data['total_frames']}" f" ({total_hrs:.1f} hours at 20 FPS)",
            f"Statuses: {status_counts}",
        ]
        return "\n".join(lines)
