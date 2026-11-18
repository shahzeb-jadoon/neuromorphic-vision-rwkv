"""Create a toy dataset for pipeline validation.

Generates synthetic episodes in the same HDF5 schema as the real CARLA
data collector, allowing the DataLoader and training pipeline to be tested
before any real data collection.

Usage:
    python scripts/create_toy_dataset.py --output data/toy/ --episodes 10 --frames 100
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from src.data import hdf5_schema as S
from src.data.schema_adapter import JUNCTIONS_FILE

# Navigation commands matching CARLA RoadOption enum (v3 schema)
NAV_COMMANDS = {
    1: "Left",
    2: "Right",
    3: "Straight",
    4: "LaneFollow",
    5: "ChangeLeft",
    6: "ChangeRight",
}

# Target distribution for nav commands (Left, Right, Straight, LaneFollow, ChangeLeft, ChangeRight)
NAV_WEIGHTS = [0.15, 0.15, 0.10, 0.55, 0.025, 0.025]

# Label codes, taken from the schema so toy data matches what the expert writes.
_MODE = {name: code for code, name in S.DRIVING_MODE_NAMES.items()}
_TL = {name: code for code, name in S.TRAFFIC_LIGHT_STATE_NAMES.items()}
MODE_FOLLOW_LANE = _MODE["FollowLane"]
MODE_TURNING = _MODE["Turning"]
MODE_TRAFFIC_LIGHT_STOP = _MODE["TrafficLightStop"]
TL_NONE, TL_YELLOW, TL_RED = _TL["None"], _TL["Yellow"], _TL["Red"]
TARGET_SPEED_KMH = 30.0


def generate_episode(
    episode_idx: int,
    n_frames: int,
    img_h: int = 320,
    img_w: int = 1920,
    n_cameras: int = 3,
    rng: np.random.Generator | None = None,
) -> dict:
    """Generate a single synthetic episode (schema v3 — panoramic).

    Images use gradient patterns (not pure noise) so visual inspection
    is meaningful. Controls follow sinusoidal patterns.

    Args:
        episode_idx: Index for deterministic seed offset.
        n_frames: Number of frames per episode.
        img_h: Panoramic image height.
        img_w: Panoramic image width (n_cameras * per_camera_width).
        n_cameras: Number of cameras (for metadata only).
        rng: Numpy random generator.

    Returns:
        Dictionary with all episode arrays and metadata.
    """
    if rng is None:
        rng = np.random.default_rng(42 + episode_idx)

    # Images: horizontal gradient + per-frame color shift (uint8)
    # This creates visually distinct frames that aren't pure noise
    base_gradient = np.linspace(0, 255, img_w, dtype=np.float32)
    base_gradient = np.tile(base_gradient, (img_h, 1))  # (H, W)

    images = np.zeros((n_frames, img_h, img_w, 3), dtype=np.uint8)
    for t in range(n_frames):
        phase = 2.0 * np.pi * t / n_frames
        r_shift = int(80 * np.sin(phase))
        g_shift = int(80 * np.sin(phase + 2.0))
        b_shift = int(80 * np.sin(phase + 4.0))
        images[t, :, :, 0] = np.clip(base_gradient + r_shift, 0, 255).astype(np.uint8)
        images[t, :, :, 1] = np.clip(base_gradient + g_shift, 0, 255).astype(np.uint8)
        images[t, :, :, 2] = np.clip(base_gradient + b_shift, 0, 255).astype(np.uint8)

    # Steering: sinusoidal with slight noise (simulates turns)
    t_axis = np.linspace(0, 4 * np.pi, n_frames)
    steering = (0.3 * np.sin(t_axis) + 0.05 * rng.standard_normal(n_frames)).astype(np.float32)
    steering = np.clip(steering, -1.0, 1.0)

    # Throttle: mostly constant with slight variation
    throttle = (0.5 + 0.1 * np.sin(t_axis * 0.5) + 0.02 * rng.standard_normal(n_frames)).astype(
        np.float32
    )
    throttle = np.clip(throttle, 0.0, 1.0)

    # Brake: mostly zero, occasional braking
    brake = np.zeros(n_frames, dtype=np.float32)
    brake_events = rng.choice(n_frames, size=max(1, n_frames // 20), replace=False)
    brake[brake_events] = rng.uniform(0.3, 1.0, size=len(brake_events)).astype(np.float32)

    # Heading (radians) integrates steering; position and velocity follow it.
    heading = np.cumsum(steering) * 0.1

    # Speed (m/s): correlated with throttle and brake
    speed_ms = np.zeros(n_frames, dtype=np.float32)
    speed = 5.0  # initial speed m/s
    for t in range(n_frames):
        speed = speed + (throttle[t] - 0.3) * 0.5 - brake[t] * 2.0
        speed = np.clip(speed, 0.0, 15.0)
        speed_ms[t] = speed

    # Velocity: world frame (vx, vy, vz), as the expert records it
    velocity = np.stack(
        [speed_ms * np.cos(heading), speed_ms * np.sin(heading), np.zeros(n_frames)], axis=1
    ).astype(np.float32)

    # Navigation commands: weighted random distribution (RoadOption 1–4)
    nav_command = rng.choice(list(NAV_COMMANDS.keys()), size=n_frames, p=NAV_WEIGHTS).astype(
        np.int32
    )

    # Position: integrate world-frame velocity at 20 Hz (dt = 0.05 s)
    position = np.zeros((n_frames, 3), dtype=np.float32)
    position[:, :2] = np.cumsum(velocity[:, :2] * 0.05, axis=0)

    # Rotation: (pitch, yaw, roll) in degrees; yaw matches the trajectory heading
    rotation = np.zeros((n_frames, 3), dtype=np.float32)
    rotation[:, 1] = np.degrees(heading)

    # Traffic light state: no light nearby mostly, with yellow-then-red episodes
    traffic_light_state = np.full(n_frames, TL_NONE, dtype=np.int32)
    traffic_light_distance = np.full(n_frames, -1.0, dtype=np.float32)
    red_starts = rng.choice(max(1, n_frames - 20), size=max(1, n_frames // 50), replace=False)
    for rs in red_starts:
        traffic_light_state[rs : min(rs + 15, n_frames)] = TL_RED
        traffic_light_state[max(0, rs - 3) : rs] = TL_YELLOW
        lit = slice(max(0, rs - 3), min(rs + 15, n_frames))
        traffic_light_distance[lit] = np.linspace(20.0, 8.0, lit.stop - lit.start)

    # Target speed (km/h, as the expert records it): 0 while braking
    target_speed = np.full(n_frames, TARGET_SPEED_KMH, dtype=np.float32)
    target_speed[brake > 0.1] = 0.0

    # Driving mode (1-indexed DrivingMode codes): follow lane mostly
    driving_mode = np.full(n_frames, MODE_FOLLOW_LANE, dtype=np.int32)
    driving_mode[traffic_light_state == TL_RED] = MODE_TRAFFIC_LIGHT_STOP
    driving_mode[np.abs(steering) > 0.2] = MODE_TURNING

    # Stop sign distance: -1 means no stop sign nearby
    stop_sign_distance = np.full(n_frames, -1.0, dtype=np.float32)
    # Simulate a few stop sign encounters
    ss_starts = rng.choice(max(1, n_frames - 30), size=max(1, n_frames // 80), replace=False)
    for ss in ss_starts:
        for dt in range(min(30, n_frames - ss)):
            stop_sign_distance[ss + dt] = max(0.0, 25.0 - dt * 0.8)

    # Stationary flag: speed < 0.5 km/h
    speed_kmh = np.linalg.norm(velocity, axis=1) * 3.6
    stationary = speed_kmh < 0.5

    # Corridor diagnostics (synthetic but plausible)
    n_corridor_segments = np.full(n_frames, 20, dtype=np.int32)
    wp_buffer_size = np.full(n_frames, 40, dtype=np.int32)
    wp_buffer_head_dist = rng.uniform(2.0, 6.0, size=n_frames).astype(np.float32)
    sat_detections = rng.integers(0, 3, size=n_frames).astype(np.int32)
    lateral_detections = rng.integers(0, 4, size=n_frames).astype(np.int32)

    # Nearby vehicles/pedestrians as JSON strings
    nearby_vehicles = []
    nearby_pedestrians = []
    for t in range(n_frames):
        n_veh = rng.integers(0, 4)
        vehs = [
            {"distance": float(rng.uniform(5, 50)), "speed": float(rng.uniform(0, 15))}
            for _ in range(n_veh)
        ]
        nearby_vehicles.append(json.dumps(vehs))
        n_ped = rng.integers(0, 3)
        peds = [{"distance": float(rng.uniform(3, 30))} for _ in range(n_ped)]
        nearby_pedestrians.append(json.dumps(peds))

    # Metadata
    towns = ["Town01", "Town03", "Town04", "Town05"]
    weathers = ["ClearNoon", "CloudyNoon", "WetNoon", "SoftRainNoon"]
    town = towns[episode_idx % len(towns)]
    weather = weathers[episode_idx % len(weathers)]

    return {
        "images": images,
        "steering": steering,
        "throttle": throttle,
        "brake": brake,
        "velocity": velocity,
        "nav_command": nav_command,
        "position": position,
        "rotation": rotation,
        "traffic_light_state": traffic_light_state,
        "traffic_light_distance": traffic_light_distance,
        "stop_sign_distance": stop_sign_distance,
        "target_speed": target_speed,
        "driving_mode": driving_mode,
        "stationary": stationary,
        "n_corridor_segments": n_corridor_segments,
        "wp_buffer_size": wp_buffer_size,
        "wp_buffer_head_dist": wp_buffer_head_dist,
        "sat_detections": sat_detections,
        "lateral_detections": lateral_detections,
        "nearby_vehicles": nearby_vehicles,
        "nearby_pedestrians": nearby_pedestrians,
        "metadata": {
            "town": town,
            "weather": weather,
            "fps": 20,
            "n_frames": n_frames,
            "n_cameras": n_cameras,
            "camera_fov": 60,
            "camera_yaws": "[-60, 0, 60]",
            "expert_type": "privileged_expert",
            "expert_target_speed": 30.0,
            "status": "completed",
        },
    }


def save_episode_hdf5(
    episode_data: dict, output_dir: Path, episode_idx: int = 0
) -> tuple[str, str]:
    """Save episode to HDF5 with hash-based filename.

    Args:
        episode_data: Dictionary from generate_episode().
        output_dir: Directory to save the HDF5 file.
        episode_idx: Episode index for unique hash generation.

    Returns:
        Tuple of (filename, sha256_hash).
    """
    # Compute hash from image data + episode index for unique naming
    hash_input = episode_idx.to_bytes(4, "big") + episode_data["images"].tobytes()[:4096]
    episode_hash = hashlib.sha256(hash_input).hexdigest()[:8]
    filename = f"episode_{episode_hash}.h5"
    filepath = output_dir / filename

    with h5py.File(filepath, "w") as f:
        # Data arrays
        f.create_dataset(S.IMAGES, data=episode_data["images"], compression="gzip", chunks=True)
        f.create_dataset(S.STEERING, data=episode_data["steering"])
        f.create_dataset(S.THROTTLE, data=episode_data["throttle"])
        f.create_dataset(S.BRAKE, data=episode_data["brake"])
        f.create_dataset(S.VELOCITY, data=episode_data["velocity"])
        f.create_dataset(S.NAV_COMMAND, data=episode_data["nav_command"])

        # Schema v3 fields
        f.create_dataset(S.POSITION, data=episode_data["position"])
        f.create_dataset(S.ROTATION, data=episode_data["rotation"])
        f.create_dataset(S.TRAFFIC_LIGHT_STATE, data=episode_data["traffic_light_state"])
        f.create_dataset(S.TRAFFIC_LIGHT_DISTANCE, data=episode_data["traffic_light_distance"])
        f.create_dataset(S.TARGET_SPEED, data=episode_data["target_speed"])
        f.create_dataset(S.DRIVING_MODE, data=episode_data["driving_mode"])

        # Schema v3.1 fields
        f.create_dataset(S.STOP_SIGN_DISTANCE, data=episode_data["stop_sign_distance"])
        f.create_dataset(S.STATIONARY, data=episode_data["stationary"])
        f.create_dataset(S.N_CORRIDOR_SEGMENTS, data=episode_data["n_corridor_segments"])
        f.create_dataset(S.WP_BUFFER_SIZE, data=episode_data["wp_buffer_size"])
        f.create_dataset(S.WP_BUFFER_HEAD_DIST, data=episode_data["wp_buffer_head_dist"])
        f.create_dataset(S.SAT_DETECTIONS, data=episode_data["sat_detections"])
        f.create_dataset(S.LATERAL_DETECTIONS, data=episode_data["lateral_detections"])

        # Variable-length string datasets for JSON blobs
        str_dt = h5py.string_dtype()
        n = len(episode_data["nearby_vehicles"])
        ds_veh = f.create_dataset(S.NEARBY_VEHICLES, (n,), dtype=str_dt)
        ds_veh[:] = episode_data["nearby_vehicles"]
        ds_ped = f.create_dataset(S.NEARBY_PEDESTRIANS, (n,), dtype=str_dt)
        ds_ped[:] = episode_data["nearby_pedestrians"]

        # Metadata as attributes on a group
        meta = f.create_group(S.METADATA_GROUP)
        meta.attrs[S.META_SCHEMA_VERSION] = S.SCHEMA_VERSION
        for key, value in episode_data["metadata"].items():
            meta.attrs[key] = value

    # Toy towns have no junctions, but the labels need an outline list for every town.
    junctions = output_dir / JUNCTIONS_FILE
    towns = json.loads(junctions.read_text()) if junctions.exists() else {}
    towns.setdefault(episode_data["metadata"]["town"], [])
    junctions.write_text(json.dumps(towns))
    return filename, episode_hash


def create_manifest(
    episodes_info: list[dict],
    output_dir: Path,
) -> None:
    """Create collection_manifest.json tracking all episodes.

    Args:
        episodes_info: List of per-episode metadata dicts.
        output_dir: Directory to save manifest.
    """
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "generator": "create_toy_dataset.py",
        "total_episodes": len(episodes_info),
        "total_frames": sum(e["n_frames"] for e in episodes_info),
        "episodes": episodes_info,
    }

    manifest_path = output_dir / "collection_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest saved: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create toy dataset for pipeline validation")
    parser.add_argument("--output", type=str, default="data/toy", help="Output directory")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--frames", type=int, default=100, help="Frames per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating toy dataset: {args.episodes} episodes × {args.frames} frames")
    print(f"Output: {output_dir.resolve()}")
    print("Panoramic images: 320 × 1920 (3 cameras × 640)")
    print()

    episodes_info = []
    total_nav_counts = {name: 0 for name in NAV_COMMANDS.values()}

    for i in range(args.episodes):
        episode_data = generate_episode(i, args.frames, rng=np.random.default_rng(args.seed + i))
        filename, episode_hash = save_episode_hdf5(episode_data, output_dir, episode_idx=i)

        # Count nav command distribution
        nav_counts = {}
        for cmd_id, cmd_name in NAV_COMMANDS.items():
            count = int(np.sum(episode_data["nav_command"] == cmd_id))
            nav_counts[cmd_name] = count
            total_nav_counts[cmd_name] += count

        episode_info = {
            "filename": filename,
            "hash": episode_hash,
            "town": episode_data["metadata"]["town"],
            "weather": episode_data["metadata"]["weather"],
            "n_frames": args.frames,
            "status": "completed",
            "nav_distribution": nav_counts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        episodes_info.append(episode_info)

        town = episode_data["metadata"]["town"]
        print(f"  [{i + 1}/{args.episodes}] {filename} — {args.frames} frames, {town}")

    # Create manifest
    create_manifest(episodes_info, output_dir)

    # Summary
    total_frames = args.episodes * args.frames
    print()
    print(f"Done! {args.episodes} episodes, {total_frames} total frames")
    print(f"Storage: {sum(f.stat().st_size for f in output_dir.glob('*.h5')) / 1e6:.1f} MB")
    print()
    print("Navigation command distribution:")
    for name, count in total_nav_counts.items():
        pct = 100 * count / total_frames
        print(f"  {name:12s}: {count:5d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
