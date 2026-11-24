"""Learned driving agent for CARLA deployment.

Wraps the trained VisionRWKVDriver as a CARLA-compatible agent.  The model
receives 3-camera panoramic RGB frames (stitched left|centre|right) and a
navigation command from the GlobalRoutePlanner, and outputs steering and a
merged acceleration value that splits into throttle / brake.

The RWKV recurrent state is preserved across ticks, giving the model temporal
context over its entire driving history — the key architectural advantage over
single-frame CNN policies.

Camera setup mirrors the data collection configuration exactly:
    3 × 640×320 cameras at yaws (-60°, 0°, +60°), mounted at (x=1.5, z=2.4).
    Stitched panoramic: 320×1920 → resized to 160×960 for inference.

Control translation (documented in src/models/policy_head.py):
    steer    = controls[:, :, 0]          ∈ [-1, 1] — direct CARLA mapping
    accel    = controls[:, :, 1]          ∈ [-1, 1]
    throttle = max(0, accel)
    brake    = max(0, -accel)

Copyright (c) 2025–2026 Shahzeb Jadoon.  Licensed under the Apache License 2.0
(see LICENSE).
"""

from __future__ import annotations

import queue
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False

from src.data.driving_dataset import MAX_SPEED_MS
from src.data import labels
from src.data.labels import ROUTE_RESOLUTION, route_progress, target_on_route
from src.data.schema_adapter import town_junctions
from src.models import VisionRWKVDriver
from src.training.config import load_training_config
from src.utils.settings import setting

# ---------------------------------------------------------------------------
# Preprocessing constants — must match src/data/driving_dataset.py exactly
# ---------------------------------------------------------------------------

_RESIZE_H = 160
_RESIZE_W = 960

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# LANEFOLLOW, 0-indexed as in training
_NAV_LANEFOLLOW = labels.LANEFOLLOW - 1

# ---------------------------------------------------------------------------
# Checkpoint loader — handles torch.compile _orig_mod key prefix
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[VisionRWKVDriver, object]:
    """Load a VisionRWKVDriver from a training checkpoint.

    Handles checkpoints saved with torch.compile enabled, which prefixes all
    encoder keys with ``encoder._orig_mod.`` instead of ``encoder.``.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Target device for the model.

    Returns:
        (model, config) tuple — model is in eval mode on device.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    run_dir = Path(checkpoint_path).parent.parent
    config_path = run_dir / "config_resolved.yaml"
    cfg = load_training_config(str(config_path))

    model = VisionRWKVDriver(cfg.model).to(device).eval()

    state_dict = ckpt["model"]
    # Strip _orig_mod. prefix inserted by torch.compile when saving
    state_dict = {k.replace("encoder._orig_mod.", "encoder."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)

    epoch = ckpt.get("epoch", "?")
    best = ckpt.get("best_metric", float("nan"))
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[LearnedAgent] Loaded checkpoint: epoch={epoch}, best_val_loss={best:.4f}, params={n_params:.1f}M")

    return model, cfg


# ---------------------------------------------------------------------------
# Camera utilities
# ---------------------------------------------------------------------------

def _camera_callback(image: "carla.Image", q: queue.Queue) -> None:
    """Drop any queued frame and enqueue the latest one (non-blocking)."""
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    q.put_nowait(image)


def _carla_image_to_rgb(image: "carla.Image", h: int, w: int) -> np.ndarray:
    """Convert a CARLA BGRA image to an RGB uint8 numpy array (H, W, 3)."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((h, w, 4))
    return arr[:, :, 2::-1].copy()  # BGRA → RGB


# ---------------------------------------------------------------------------
# Learned Agent
# ---------------------------------------------------------------------------

class LearnedAgent:
    """CARLA driving agent backed by the trained VisionRWKVDriver model.

    Args:
        vehicle:         The ego CARLA vehicle actor.
        checkpoint_path: Path to the training checkpoint (.pt file).
        device:          PyTorch device string ('cuda' or 'cpu').
        min_speed_kmh:   Minimum speed to enforce via a light throttle nudge.
                         Prevents the model from getting stuck at intersections
                         where expert data has many stationary frames.
    """

    # --- Camera config (must match expert_collector.py EpisodeConfig) ---
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 320
    CAMERA_FOV: int = 60
    CAMERA_YAWS: tuple = (-60.0, 0.0, 60.0)  # left, centre, right
    CAMERA_X: float = 1.5   # forward offset (m)
    CAMERA_Z: float = 2.4   # height (m)

    def __init__(
        self,
        vehicle: "carla.Vehicle",
        checkpoint_path: str,
        device: str = "cuda",
        min_speed_kmh: float = 5.0,
    ) -> None:
        if not CARLA_AVAILABLE:
            raise ImportError("carla package not found — install it or run inside a CARLA Python env")

        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.map = self.world.get_map()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.min_speed_kmh = min_speed_kmh

        # Model
        self.model, self.cfg = load_checkpoint(checkpoint_path, self.device)
        self.state = self.model.initial_state(batch_size=1, device=self.device)

        # Route planner, at the CARLA leaderboard's spacing
        self._planner = GlobalRoutePlanner(self.map, sampling_resolution=ROUTE_RESOLUTION)
        self._route: List[Tuple] = []
        self._route_xy = np.zeros((0, 2))
        self._route_distance = np.zeros(0)
        self._route_command = np.zeros(0, np.int64)
        self._progress = 0

        # Camera queues and actors
        self._cam_queues: List[queue.Queue] = [queue.Queue(maxsize=1) for _ in range(3)]
        self._cameras: List["carla.Actor"] = []
        self._setup_cameras()

        self._tick = 0
        # Diagnostic attributes — updated every run_step() call for external logging.
        self.last_steer: float = 0.0
        self.last_accel: float = 0.0
        self.last_nav_cmd: int = _NAV_LANEFOLLOW
        print(f"[LearnedAgent] Ready on {self.device}. Cameras spawned.")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_cameras(self) -> None:
        """Spawn 3 RGB cameras mirroring the data-collection configuration."""
        bp_lib = self.world.get_blueprint_library()
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.CAMERA_WIDTH))
        cam_bp.set_attribute("image_size_y", str(self.CAMERA_HEIGHT))
        cam_bp.set_attribute("fov", str(self.CAMERA_FOV))

        for i, yaw in enumerate(self.CAMERA_YAWS):
            transform = carla.Transform(
                carla.Location(x=self.CAMERA_X, z=self.CAMERA_Z),
                carla.Rotation(yaw=float(yaw)),
            )
            cam = self.world.spawn_actor(cam_bp, transform, attach_to=self.vehicle)
            cam.listen(lambda img, q=self._cam_queues[i]: _camera_callback(img, q))
            self._cameras.append(cam)

    # ------------------------------------------------------------------
    # Route & navigation
    # ------------------------------------------------------------------

    def set_destination(self, destination: "carla.Location") -> None:
        """Plan a route from the current vehicle position to destination."""
        self.set_route([self.vehicle.get_location(), destination])

    def set_route(self, keypoints: List["carla.Location"]) -> None:
        """Plan the route through `keypoints` in order, as the CARLA leaderboard does."""
        route: List[Tuple] = []
        for a, b in zip(keypoints, keypoints[1:]):
            part = self._planner.trace_route(a, b)
            if route and part and part[0][0].transform.location.distance(
                route[-1][0].transform.location
            ) < 0.5 * ROUTE_RESOLUTION:
                part = part[1:]
            route += part
        self._route = route
        self._route_xy = np.array(
            [[wp.transform.location.x, wp.transform.location.y] for wp, _ in route]
        ).reshape(-1, 2)
        steps = np.linalg.norm(np.diff(self._route_xy, axis=0), axis=1)
        self._route_distance = np.r_[0.0, np.cumsum(steps)]
        # Commands as training derives them from a path, with the same junction outlines.
        town = self.map.name.split("/")[-1].removesuffix("_Opt")
        outlines, _ = town_junctions(town, setting("DATA_DIR"))
        yaw = np.array([wp.transform.rotation.yaw for wp, _ in route])
        junction = labels.junction_ids(self._route_xy, outlines)
        self._route_command = labels.route(self._route_xy, yaw, junction).command
        self._progress = 0
        print(f"[LearnedAgent] Route planned: {len(route)} waypoints.")

    def _update_progress(self) -> None:
        """Move the route position to the point nearest the vehicle ahead of the last one."""
        if self._route:
            loc = self.vehicle.get_location()
            position = np.array([loc.x, loc.y])
            self._progress = route_progress(
                self._route_xy, self._route_distance, self._progress, position
            )

    def _get_nav_command(self) -> int:
        """The 0-indexed command of the route where the vehicle is, derived as in training."""
        if not self._route:
            return _NAV_LANEFOLLOW
        return int(self._route_command[self._progress]) - 1

    def _get_target_point(self) -> torch.Tensor:
        """The route point TARGET_DISTANCE ahead in the vehicle's frame, (1, 1, 2) metres."""
        transform = self.vehicle.get_transform()
        point = target_on_route(
            np.array([transform.location.x, transform.location.y]),
            transform.rotation.yaw,
            self._route_xy[self._progress + 1 :],
        )
        return torch.from_numpy(point)[None, None].to(self.device)

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------

    def _get_panoramic_frame(self) -> Optional[np.ndarray]:
        """Collect one frame from each camera and stitch into a panoramic.

        Returns:
            (H, W_total, 3) uint8 numpy array, or None if cameras aren't ready.
        """
        frames = []
        for q in self._cam_queues:
            try:
                img = q.get(timeout=0.05)
            except queue.Empty:
                return None
            frames.append(_carla_image_to_rgb(img, self.CAMERA_HEIGHT, self.CAMERA_WIDTH))

        return np.concatenate(frames, axis=1)  # (320, 1920, 3)

    def _preprocess(self, panoramic: np.ndarray) -> torch.Tensor:
        """Preprocess a panoramic frame to match the training pipeline exactly.

        Steps (must mirror DrivingSequenceDataset.__getitem__):
            1. uint8 (H, W, 3) → float32 (3, H, W) / 255
            2. Bilinear resize to (_RESIZE_H, _RESIZE_W)
            3. ImageNet mean/std normalization
            4. Add batch and time dimensions → (1, 1, 3, H, W)
        """
        img = torch.from_numpy(panoramic).permute(2, 0, 1).float().div(255.0)  # (3, H, W)
        img = F.interpolate(
            img.unsqueeze(0),
            size=(_RESIZE_H, _RESIZE_W),
            mode="bilinear",
            align_corners=False,
        )  # (1, 3, H, W)
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        return img.unsqueeze(1).to(self.device)  # (1, 1, 3, H, W)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def run_step(self) -> "carla.VehicleControl":
        """Execute one inference step and return a CARLA VehicleControl.

        Called once per simulator tick (target: 20 Hz, matching training FPS).
        The RWKV state (self.state) is updated in-place, carrying temporal
        context across ticks.

        Returns:
            carla.VehicleControl with steer, throttle, and brake set.
        """
        panoramic = self._get_panoramic_frame()
        if panoramic is None:
            # Cameras haven't produced a frame yet — coast forward gently
            return carla.VehicleControl(throttle=0.2, steer=0.0, brake=0.0)

        self._update_progress()
        nav_cmd_idx = self._get_nav_command()
        images = self._preprocess(panoramic)                              # (1, 1, 3, H, W)
        nav_cmd = torch.tensor([[nav_cmd_idx]], device=self.device)       # (1, 1)

        vel = self.vehicle.get_velocity()
        speed_norm = float(np.hypot(vel.x, vel.y)) / MAX_SPEED_MS
        speed = torch.tensor([[speed_norm]], device=self.device)          # (1, 1)

        target_point = self._get_target_point()

        controls, self.state, _ = self.model(images, nav_cmd, speed, target_point, self.state)

        steer = float(controls[0, 0, 0])
        accel = float(controls[0, 0, 1])

        throttle = max(0.0, accel)
        brake    = max(0.0, -accel)

        # Minimum speed nudge: prevent getting permanently stuck.
        # The model was trained on expert data that never sat still deliberately,
        # so it may output near-zero accel when stationary at green lights.
        speed_kmh = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5 * 3.6
        if speed_kmh < self.min_speed_kmh and brake < 0.05:
            throttle = max(throttle, 0.15)

        self.last_steer = float(np.clip(steer, -1.0, 1.0))
        self.last_accel = accel
        self.last_nav_cmd = nav_cmd_idx
        self._tick += 1

        return carla.VehicleControl(
            throttle=float(np.clip(throttle, 0.0, 1.0)),
            steer=self.last_steer,
            brake=float(np.clip(brake, 0.0, 1.0)),
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        """Stop listening cameras and destroy all spawned CARLA actors."""
        for cam in self._cameras:
            try:
                cam.stop()
                cam.destroy()
            except Exception:
                pass
        self._cameras.clear()
        print("[LearnedAgent] Cameras destroyed.")
