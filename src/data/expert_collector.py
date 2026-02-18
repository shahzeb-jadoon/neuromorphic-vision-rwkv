"""Expert data collector for CARLA autonomous driving episodes.

Uses CARLA's built-in autopilot (TrafficManager) to drive expert routes
while recording per-frame observations (camera, controls, navigation)
to HDF5 files.

NOTE: We use set_autopilot(True) + TrafficManager rather than
BehaviorAgent because the `agents` Python module ships only with the
CARLA server distribution, not the pip package (`carla==0.9.16`).

Handles three failure modes:
    - CARLA crash recovery (reconnect with retries)
    - Episode termination (completed / collision / timeout)
    - Campaign resumption via collection_manifest.json
"""

from __future__ import annotations

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

try:
    import carla

    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False

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
    timeout_duration: float = 180.0  # seconds stuck
    fps: int = 20
    camera_width: int = 224
    camera_height: int = 224
    camera_fov: int = 100


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


class ExpertDataCollector:
    """Collects expert driving data from CARLA using TrafficManager autopilot.

    Uses vehicle.set_autopilot(True) which delegates driving to CARLA's
    built-in TrafficManager. Navigation commands are inferred by comparing
    the vehicle's forward vector with the direction to the next waypoint.

    Usage:
        collector = ExpertDataCollector(
            host="auto",
            port=2000,
            output_dir="data/collected",
        )
        collector.connect()
        result = collector.collect_episode(config)

    Args:
        host: CARLA server host ("auto" to detect from WSL2 gateway).
        port: CARLA server port.
        output_dir: Directory to save HDF5 files and manifest.
        crash_retries: Number of reconnection attempts on CARLA crash.
        crash_wait: Seconds to wait between reconnection attempts.
    """

    # Navigation command mapping
    NAV_MAP = {
        "LaneFollow": 0,
        "Left": 1,
        "Right": 2,
        "Straight": 3,
    }

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
        self.client.set_timeout(30.0)
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

    @staticmethod
    def _infer_nav_command(
        vehicle_transform: Any,
        waypoint: Any,
        carla_map: Any,
        lookahead: float = 10.0,
    ) -> int:
        """Infer navigation command from vehicle heading vs waypoint direction.

        Compares the vehicle's forward vector with the direction to a
        lookahead waypoint. The angle between them determines the nav command:
            - |angle| < 15°  → LaneFollow (0)
            - angle < -15°   → Left (1)
            - angle > 15°    → Right (2)
            - Near junction with small angle → Straight (3)

        Args:
            vehicle_transform: Current vehicle transform.
            waypoint: Current waypoint the vehicle is on.
            carla_map: CARLA map instance.
            lookahead: Distance in meters to look ahead.

        Returns:
            Navigation command ID (0-3).
        """
        # Get next waypoints along the road
        next_wps = waypoint.next(lookahead)
        if not next_wps:
            return 0  # LaneFollow

        target_wp = next_wps[0]

        # Vehicle forward vector (2D)
        fwd = vehicle_transform.get_forward_vector()
        fwd_2d = np.array([fwd.x, fwd.y])

        # Direction to target
        veh_loc = vehicle_transform.location
        tgt_loc = target_wp.transform.location
        dir_2d = np.array([tgt_loc.x - veh_loc.x, tgt_loc.y - veh_loc.y])

        # Compute signed angle
        fwd_norm = np.linalg.norm(fwd_2d)
        dir_norm = np.linalg.norm(dir_2d)
        if fwd_norm < 1e-6 or dir_norm < 1e-6:
            return 0

        fwd_2d = fwd_2d / fwd_norm
        dir_2d = dir_2d / dir_norm

        # Cross product gives signed angle
        cross = fwd_2d[0] * dir_2d[1] - fwd_2d[1] * dir_2d[0]
        dot = np.dot(fwd_2d, dir_2d)
        angle_deg = math.degrees(math.atan2(cross, dot))

        # Check if at junction
        at_junction = waypoint.is_junction

        # Classify
        threshold = 15.0
        if abs(angle_deg) < threshold:
            if at_junction:
                return 3  # Straight (through junction)
            return 0  # LaneFollow
        elif angle_deg < -threshold:
            return 1  # Left
        else:
            return 2  # Right

    def collect_episode(
        self,
        config: EpisodeConfig,
        route_waypoints: list | None = None,
    ) -> EpisodeResult:
        """Collect a single episode of expert driving data.

        Uses CARLA's TrafficManager autopilot for expert driving.
        Navigation commands are inferred from heading-vs-waypoint angle.

        Args:
            config: Episode configuration.
            route_waypoints: Optional pre-computed route (currently unused
                with autopilot; reserved for future BehaviorAgent support).

        Returns:
            EpisodeResult with status and metadata.
        """
        if self.client is None:
            self.connect()

        start_time = time.time()
        actors: list = []
        npc_vehicles: list = []
        original_settings = None
        image_queue: queue.Queue = queue.Queue()

        # Data buffers
        images_buf: list[np.ndarray] = []
        steering_buf: list[float] = []
        throttle_buf: list[float] = []
        brake_buf: list[float] = []
        velocity_buf: list[np.ndarray] = []
        nav_cmd_buf: list[int] = []
        collision_frame: int | None = None

        try:
            assert self.world is not None
            assert self.client is not None

            # --- Setup world ---
            if self.world.get_map().name.split("/")[-1] != config.town:
                logger.info("Loading map: %s", config.town)
                self.client.load_world(config.town)
                self.world = self.client.get_world()
                time.sleep(2.0)  # Wait for map load

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

            # --- Setup TrafficManager ---
            traffic_manager = self.client.get_trafficmanager(8000)
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_global_distance_to_leading_vehicle(2.5)

            # --- Spawn ego vehicle ---
            bp_lib = self.world.get_blueprint_library()
            vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
            spawn_points = carla_map.get_spawn_points()
            spawn_idx = np.random.randint(len(spawn_points))
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_points[spawn_idx])

            if vehicle is None:
                return EpisodeResult(
                    status="crashed",
                    error_message="Failed to spawn vehicle",
                    duration_seconds=time.time() - start_time,
                )

            actors.append(vehicle)

            # Enable autopilot via TrafficManager
            vehicle.set_autopilot(True, traffic_manager.get_port())

            # Configure ego driving behavior
            traffic_manager.auto_lane_change(vehicle, True)
            traffic_manager.ignore_lights_percentage(vehicle, 0.0)  # Obey traffic lights
            traffic_manager.vehicle_percentage_speed_difference(vehicle, -20.0)  # 20% below limit

            # --- Spawn NPC traffic ---
            if config.n_vehicles > 0:
                npc_bps = bp_lib.filter("vehicle.*")
                npc_bps = [bp for bp in npc_bps if int(bp.get_attribute("number_of_wheels")) == 4]
                available_spawns = [sp for i, sp in enumerate(spawn_points) if i != spawn_idx]
                n_to_spawn = min(config.n_vehicles, len(available_spawns))

                for i in range(n_to_spawn):
                    npc_bp = np.random.choice(npc_bps)
                    npc = self.world.try_spawn_actor(npc_bp, available_spawns[i])
                    if npc is not None:
                        npc.set_autopilot(True, traffic_manager.get_port())
                        npc_vehicles.append(npc)

                logger.info("Spawned %d NPC vehicles (requested %d)", len(npc_vehicles), config.n_vehicles)

            # --- Attach camera ---
            camera_bp = bp_lib.find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", str(config.camera_width))
            camera_bp.set_attribute("image_size_y", str(config.camera_height))
            camera_bp.set_attribute("fov", str(config.camera_fov))

            cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            camera = self.world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)
            actors.append(camera)
            camera.listen(lambda img: image_queue.put(img))

            # --- Attach collision sensor ---
            collision_bp = bp_lib.find("sensor.other.collision")
            collision_sensor = self.world.spawn_actor(
                collision_bp, carla.Transform(), attach_to=vehicle
            )
            actors.append(collision_sensor)
            collision_detected = [False]  # Mutable for closure

            def on_collision(event):
                collision_detected[0] = True

            collision_sensor.listen(on_collision)

            # Let world settle (a few warm-up ticks)
            for _ in range(10):
                self.world.tick()

            # Drain any queued warm-up images
            while not image_queue.empty():
                try:
                    image_queue.get_nowait()
                except queue.Empty:
                    break

            # --- Tick loop ---
            stuck_start: float | None = None
            distance_traveled = 0.0
            prev_location = vehicle.get_transform().location

            for frame_idx in range(config.max_frames):
                self.world.tick()

                # Get camera image
                try:
                    carla_image = image_queue.get(timeout=2.0)
                    raw = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
                    raw = raw.reshape(config.camera_height, config.camera_width, 4)
                    rgb = raw[:, :, :3][:, :, ::-1].copy()  # BGRA → RGB
                    images_buf.append(rgb)
                except queue.Empty:
                    logger.warning("No camera image at frame %d, using black frame", frame_idx)
                    images_buf.append(
                        np.zeros(
                            (config.camera_height, config.camera_width, 3),
                            dtype=np.uint8,
                        )
                    )

                # Get controls applied by autopilot
                control = vehicle.get_control()
                steering_buf.append(float(control.steer))
                throttle_buf.append(float(control.throttle))
                brake_buf.append(float(control.brake))

                # Velocity
                vel = vehicle.get_velocity()
                velocity_buf.append(np.array([vel.x, vel.y, vel.z], dtype=np.float32))

                # Navigation command (inferred from heading vs waypoint direction)
                veh_transform = vehicle.get_transform()
                current_wp = carla_map.get_waypoint(veh_transform.location)
                nav_id = self._infer_nav_command(veh_transform, current_wp, carla_map)
                nav_cmd_buf.append(nav_id)

                # Track distance
                cur_loc = veh_transform.location
                dx = cur_loc.x - prev_location.x
                dy = cur_loc.y - prev_location.y
                distance_traveled += (dx**2 + dy**2) ** 0.5
                prev_location = cur_loc

                # Check collision
                if collision_detected[0] and collision_frame is None:
                    collision_frame = frame_idx
                    logger.warning("Collision detected at frame %d", frame_idx)

                # Check timeout (stuck)
                speed = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5
                if speed < config.timeout_speed:
                    if stuck_start is None:
                        stuck_start = time.time()
                    elif time.time() - stuck_start > config.timeout_duration:
                        logger.warning("Timeout: vehicle stuck for >%ds", config.timeout_duration)
                        # Still save the data collected
                        break
                else:
                    stuck_start = None

                # Progress logging every 500 frames
                if (frame_idx + 1) % 500 == 0:
                    logger.info(
                        "  Frame %d/%d — speed=%.1f m/s, distance=%.0fm, steer=%.3f",
                        frame_idx + 1,
                        config.max_frames,
                        speed,
                        distance_traveled,
                        control.steer,
                    )

            # --- Determine status ---
            timed_out = stuck_start is not None and (time.time() - stuck_start > config.timeout_duration)
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
                    velocity=np.stack(velocity_buf),
                    nav_command=np.array(nav_cmd_buf, dtype=np.int32),
                    config=config,
                    status=status,
                    collision_frame=collision_frame,
                )
                result.duration_seconds = time.time() - start_time
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
                        velocity=np.stack(velocity_buf),
                        nav_command=np.array(nav_cmd_buf, dtype=np.int32),
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
            # Disable autopilot before cleanup to avoid TM race conditions
            try:
                if 'vehicle' in dir() and vehicle is not None and vehicle.is_alive:
                    vehicle.set_autopilot(False)
            except Exception:
                pass

            # Stop sensors first
            for actor in actors:
                try:
                    if actor.is_alive and hasattr(actor, "stop"):
                        actor.stop()
                except Exception:
                    pass

            # Destroy NPC vehicles (before ego, to avoid TM issues)
            for npc in npc_vehicles:
                try:
                    if npc.is_alive:
                        npc.set_autopilot(False)
                        npc.destroy()
                except Exception:
                    pass

            # Destroy ego actors (sensors, collision, vehicle)
            for actor in reversed(actors):
                try:
                    if actor.is_alive:
                        actor.destroy()
                except Exception:
                    pass

            # Restore settings
            if original_settings is not None:
                try:
                    assert self.world is not None
                    self.world.apply_settings(original_settings)
                except Exception:
                    pass

    def _save_episode(
        self,
        images: np.ndarray,
        steering: np.ndarray,
        throttle: np.ndarray,
        brake: np.ndarray,
        velocity: np.ndarray,
        nav_command: np.ndarray,
        config: EpisodeConfig,
        status: str,
        collision_frame: int | None,
    ) -> EpisodeResult:
        """Save collected episode data to HDF5."""
        # Hash-based filename (include timestamp for uniqueness)
        timestamp_bytes = int(time.time() * 1000).to_bytes(8, "big")
        hash_input = timestamp_bytes + images[:4].tobytes()[:4096]
        episode_hash = hashlib.sha256(hash_input).hexdigest()[:8]
        filename = f"episode_{episode_hash}.h5"
        filepath = self.output_dir / filename

        with h5py.File(filepath, "w") as f:
            f.create_dataset(
                "images",
                data=images,
                compression="gzip",
                compression_opts=1,  # Level 1 = fastest decompression, ~same ratio
                chunks=(1, config.camera_height, config.camera_width, 3),
            )
            f.create_dataset("steering", data=steering)
            f.create_dataset("throttle", data=throttle)
            f.create_dataset("brake", data=brake)
            f.create_dataset("velocity", data=velocity)
            f.create_dataset("nav_command", data=nav_command)

            meta = f.create_group("metadata")
            meta.attrs["town"] = config.town
            meta.attrs["weather"] = config.weather
            meta.attrs["fps"] = config.fps
            meta.attrs["n_frames"] = len(images)
            meta.attrs["status"] = status
            if collision_frame is not None:
                meta.attrs["collision_frame"] = collision_frame

        # Nav distribution
        nav_names = {0: "LaneFollow", 1: "Left", 2: "Right", 3: "Straight"}
        nav_dist = {}
        for cmd_id, cmd_name in nav_names.items():
            nav_dist[cmd_name] = int(np.sum(nav_command == cmd_id))

        logger.info(
            "Saved %s — %d frames, status=%s, town=%s",
            filename,
            len(images),
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
                return json.load(f)
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
        self.data["total_frames"] = sum(
            e["n_frames"] for e in self.data["episodes"]
        )
        self.data["last_updated"] = datetime.now(timezone.utc).isoformat()

    def save(self) -> None:
        """Write manifest to disk."""
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def get_completed_configs(self) -> set[tuple[str, str]]:
        """Return set of (town, weather) combos already completed.

        Only counts episodes with status='completed'.
        """
        return {
            (e["town"], e["weather"])
            for e in self.data["episodes"]
            if e["status"] == "completed"
        }

    def summary(self) -> str:
        """Return human-readable summary."""
        total_hrs = self.data["total_frames"] / (20 * 3600)
        status_counts: dict[str, int] = {}
        for e in self.data["episodes"]:
            s = e["status"]
            status_counts[s] = status_counts.get(s, 0) + 1

        lines = [
            f"Episodes: {self.data['total_episodes']}",
            f"Frames:   {self.data['total_frames']} ({total_hrs:.1f} hours at 20 FPS)",
            f"Statuses: {status_counts}",
        ]
        return "\n".join(lines)
