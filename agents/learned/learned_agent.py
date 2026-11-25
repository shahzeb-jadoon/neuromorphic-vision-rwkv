"""Learned driving agent for CARLA deployment.

Wraps the trained VisionRWKVDriver as a CARLA agent in a synchronous world with a fixed
time step. Every tick it is given the frame id that `world.tick()` returned. At the rate the
model trained at (`dataset.fps`), it takes the three cameras' images of that frame, stitches
them left|centre|right, prepares them exactly as training reads them from its cache, and
steps the model once in fp32 with the state carried from the frame before. The ticks between
model frames repeat its last control. Nothing overrides the model's control.

Camera setup mirrors the data collection configuration exactly:
    3 × 640×320 cameras at yaws (-60°, 0°, +60°), mounted at (x=1.5, z=2.4).

Control translation (documented in src/models/policy_head.py):
    steer    = controls[:, 0]          ∈ [-1, 1] — direct CARLA mapping
    accel    = controls[:, 1]          ∈ [-1, 1]
    throttle = max(0, accel)
    brake    = max(0, -accel)

Copyright (c) 2025–2026 Shahzeb Jadoon.  Licensed under the Apache License 2.0
(see LICENSE).
"""

from __future__ import annotations

import queue
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

try:
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False

from src.data import labels, schema_adapter
from src.data.driving_dataset import IMAGENET_MEAN, IMAGENET_STD, MAX_SPEED_MS
from src.data.labels import ROUTE_RESOLUTION, route_progress, target_on_route
from src.data.schema_adapter import town_junctions
from src.models import VisionRWKVDriver
from src.models.temporal import TemporalState
from src.training.config import load_training_config
from src.utils.settings import setting

# LANEFOLLOW, 0-indexed as in training
_NAV_LANEFOLLOW = labels.LANEFOLLOW - 1
CAMERA_TIMEOUT_S = 10.0

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
# Frames
# ---------------------------------------------------------------------------

def ticks_per_frame(settings: "carla.WorldSettings", fps: int) -> int:
    """Simulator ticks per model frame: the world must step synchronously, a whole number of
    times per frame of the rate the model trained at."""
    delta = settings.fixed_delta_seconds
    if not settings.synchronous_mode or not delta:
        raise ValueError("the agent needs a synchronous world with a fixed time step")
    ticks = 1.0 / (delta * fps)
    if round(ticks) < 1 or abs(ticks - round(ticks)) > 1e-6:
        raise ValueError(f"a {delta} s time step does not divide the model's {fps} Hz frames")
    return round(ticks)


def frame_image(images: "queue.Queue[carla.Image]", frame: int) -> "carla.Image":
    """The camera's image of simulator frame `frame`, dropping earlier ones. A camera that
    skipped the frame or stays silent stops the drive rather than feeding it another frame."""
    while True:
        try:
            image = images.get(timeout=CAMERA_TIMEOUT_S)
        except queue.Empty:
            raise RuntimeError(f"no camera image of frame {frame} in {CAMERA_TIMEOUT_S} s") from None
        if image.frame == frame:
            return image
        if image.frame > frame:
            raise RuntimeError(f"a camera skipped frame {frame} (its next image is {image.frame})")


def model_image(panorama: np.ndarray, size: Tuple[int, int]) -> torch.Tensor:
    """(H, W, 3) uint8 stitched cameras as the model trained on them, (3, *size) float32:
    resized and JPEG-encoded as the training cache stores them, then normalised as the
    dataset normalises them."""
    pixels = schema_adapter.as_cached(panorama, size)
    image = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float().div_(255.0)
    return image.sub_(IMAGENET_MEAN).div_(IMAGENET_STD)


def _carla_image_to_rgb(image: "carla.Image") -> np.ndarray:
    """Convert a CARLA BGRA image to an RGB uint8 numpy array (H, W, 3)."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    return arr[:, :, 2::-1]  # BGRA → RGB


# ---------------------------------------------------------------------------
# Learned Agent
# ---------------------------------------------------------------------------

class LearnedAgent:
    """CARLA driving agent backed by the trained VisionRWKVDriver model.

    Args:
        vehicle:         The ego CARLA vehicle actor, in a synchronous world.
        checkpoint_path: Path to the training checkpoint (.pt file).
        device:          PyTorch device string ('cuda' or 'cpu').
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
    ) -> None:
        if not CARLA_AVAILABLE:
            raise ImportError("carla package not found — install it or run inside a CARLA Python env")

        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.map = self.world.get_map()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Model, in fp32: clip and step outputs agree only to rounding under bf16
        self.model, self.cfg = load_checkpoint(checkpoint_path, self.device)
        self.state: Optional[TemporalState] = None
        self._size = (self.cfg.dataset.resize_height, self.cfg.dataset.resize_width)
        self._ticks_per_frame = ticks_per_frame(self.world.get_settings(), self.cfg.dataset.fps)

        # Route planner, at the CARLA leaderboard's spacing
        self._planner = GlobalRoutePlanner(self.map, sampling_resolution=ROUTE_RESOLUTION)
        self._route: List[Tuple] = []
        self._route_xy = np.zeros((0, 2))
        self._route_distance = np.zeros(0)
        self._route_command = np.zeros(0, np.int64)
        self._route_sparse = np.zeros(0, np.int64)
        self._progress = 0

        # Every image each camera sends, until the tick that needs it
        self._cam_queues: List["queue.Queue[carla.Image]"] = [queue.Queue() for _ in range(3)]
        self._cameras: List["carla.Actor"] = []
        self._setup_cameras()

        self._tick = 0
        self._control = carla.VehicleControl()
        # Diagnostics of the last run_step() call, for external logging.
        self.stepped: bool = False
        self.camera_ms: float = float("nan")
        self.model_ms: float = float("nan")
        self.last_steer: float = 0.0
        self.last_accel: float = 0.0
        self.last_nav_cmd: int = _NAV_LANEFOLLOW
        print(
            f"[LearnedAgent] Ready on {self.device}: model at {self.cfg.dataset.fps} Hz, "
            f"every {self._ticks_per_frame} ticks. Cameras spawned."
        )

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

        for q, yaw in zip(self._cam_queues, self.CAMERA_YAWS):
            transform = carla.Transform(
                carla.Location(x=self.CAMERA_X, z=self.CAMERA_Z),
                carla.Rotation(yaw=float(yaw)),
            )
            cam = self.world.spawn_actor(cam_bp, transform, attach_to=self.vehicle)
            cam.listen(q.put)
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
        planned = labels.route(self._route_xy, yaw, labels.junction_ids(self._route_xy, outlines))
        self._route_command = planned.command
        self._route_sparse = labels.sparse_points(self._route_distance, planned.in_junction)
        self._progress = 0
        print(f"[LearnedAgent] Route planned: {len(route)} waypoints.")

    @property
    def route_completion(self) -> float:
        """Share of the planned route's length behind the vehicle."""
        if not self._route:
            return 0.0
        return float(self._route_distance[self._progress] / max(self._route_distance[-1], 1e-6))

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
        """The target point the model trained with, in the vehicle's frame, (1, 2) metres:
        the route TARGET_DISTANCE ahead, or the next sparse route point."""
        transform = self.vehicle.get_transform()
        position = np.array([transform.location.x, transform.location.y])
        yaw = transform.rotation.yaw
        if self._route and self.cfg.dataset.target_point == "sparse":
            here = self._route_distance[self._progress : self._progress + 1]
            point = labels.sparse_target(
                self._route_xy, self._route_distance, self._route_sparse, here, position[None], yaw
            )[0]
        else:
            point = target_on_route(position, yaw, self._route_xy[self._progress + 1 :])
        return torch.from_numpy(point)[None].to(self.device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def run_step(self, frame: int) -> "carla.VehicleControl":
        """The control for simulator frame `frame`, the id `world.tick()` returned.

        The model steps on the first tick and every `ticks_per_frame`-th tick after it,
        carrying its state from the frame before; the ticks between repeat its control.
        """
        self.stepped = self._tick % self._ticks_per_frame == 0
        self._tick += 1
        if not self.stepped:
            return self._control

        start = time.perf_counter()
        images = [frame_image(q, frame) for q in self._cam_queues]
        panorama = np.concatenate([_carla_image_to_rgb(image) for image in images], axis=1)
        self.camera_ms = (time.perf_counter() - start) * 1e3

        start = time.perf_counter()
        self._update_progress()
        nav_cmd_idx = self._get_nav_command()
        vel = self.vehicle.get_velocity()
        controls, self.state, _ = self.model.step(
            model_image(panorama, self._size)[None].to(self.device),
            torch.tensor([nav_cmd_idx], device=self.device),
            torch.tensor([float(np.hypot(vel.x, vel.y)) / MAX_SPEED_MS], device=self.device),
            self._get_target_point(),
            self.state,
        )
        steer, accel = controls[0].tolist()
        self.model_ms = (time.perf_counter() - start) * 1e3

        self.last_steer = float(np.clip(steer, -1.0, 1.0))
        self.last_accel = accel
        self.last_nav_cmd = nav_cmd_idx
        self._control = carla.VehicleControl(
            throttle=float(np.clip(accel, 0.0, 1.0)),
            steer=self.last_steer,
            brake=float(np.clip(-accel, 0.0, 1.0)),
        )
        return self._control

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
