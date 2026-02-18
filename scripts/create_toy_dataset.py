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
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

# Ensure src package is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import hdf5_schema as S  # noqa: E402

# Navigation commands matching CARLA RoadOption enum
NAV_COMMANDS = {
    0: "LaneFollow",
    1: "Left",
    2: "Right",
    3: "Straight",
}

# Target distribution for nav commands
NAV_WEIGHTS = [0.60, 0.15, 0.15, 0.10]  # LaneFollow, Left, Right, Straight


def generate_episode(
    episode_idx: int,
    n_frames: int,
    img_h: int = 224,
    img_w: int = 224,
    rng: np.random.Generator | None = None,
) -> dict:
    """Generate a single synthetic episode.

    Images use gradient patterns (not pure noise) so visual inspection
    is meaningful. Controls follow sinusoidal patterns.

    Args:
        episode_idx: Index for deterministic seed offset.
        n_frames: Number of frames per episode.
        img_h: Image height.
        img_w: Image width.
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

    # Velocity: correlated with throttle
    velocity = np.zeros((n_frames, 3), dtype=np.float32)
    speed = 5.0  # initial speed m/s
    for t in range(n_frames):
        speed = speed + (throttle[t] - 0.3) * 0.5 - brake[t] * 2.0
        speed = np.clip(speed, 0.0, 15.0)
        # Forward velocity with slight lateral component from steering
        velocity[t] = [speed, speed * steering[t] * 0.1, 0.0]

    # Navigation commands: weighted random distribution
    nav_command = rng.choice(4, size=n_frames, p=NAV_WEIGHTS).astype(np.int32)

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
        "metadata": {
            "town": town,
            "weather": weather,
            "fps": 20,
            "n_frames": n_frames,
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

        # Metadata as attributes on a group
        meta = f.create_group(S.METADATA_GROUP)
        meta.attrs[S.META_SCHEMA_VERSION] = S.SCHEMA_VERSION
        for key, value in episode_data["metadata"].items():
            meta.attrs[key] = value

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
