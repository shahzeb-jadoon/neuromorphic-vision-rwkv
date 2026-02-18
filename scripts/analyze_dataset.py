"""Analyze a collected driving dataset.

Provides summary statistics, turn distribution analysis,
and visual spot-checks for quality verification.

Usage:
    python scripts/analyze_dataset.py data/collected/
    python scripts/analyze_dataset.py data/toy/ --save-samples 5
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

NAV_NAMES = {0: "LaneFollow", 1: "Left", 2: "Right", 3: "Straight"}


def analyze_dataset(data_dir: Path, save_samples: int = 0) -> None:
    """Print comprehensive dataset analysis."""
    episodes = sorted(data_dir.glob("episode_*.h5"))
    if not episodes:
        print(f"No episode files found in {data_dir}")
        return

    print(f"{'='*60}")
    print(f"Dataset Analysis: {data_dir.resolve()}")
    print(f"{'='*60}")
    print()

    # Aggregate stats
    total_frames = 0
    total_size_bytes = 0
    nav_counts: defaultdict[int, int] = defaultdict(int)
    town_counts: defaultdict[str, int] = defaultdict(int)
    weather_counts: defaultdict[str, int] = defaultdict(int)
    status_counts: defaultdict[str, int] = defaultdict(int)
    steering_all = []
    throttle_all = []
    episode_lengths = []

    for ep_path in episodes:
        total_size_bytes += ep_path.stat().st_size

        with h5py.File(ep_path, "r") as f:
            n_frames = f["images"].shape[0]
            total_frames += n_frames
            episode_lengths.append(n_frames)

            # Nav distribution
            nav = f["nav_command"][:]
            for cmd_id in range(4):
                nav_counts[cmd_id] += int(np.sum(nav == cmd_id))

            # Controls
            steering_all.append(f["steering"][:])
            throttle_all.append(f["throttle"][:])

            # Metadata
            meta = f["metadata"]
            town_counts[meta.attrs["town"]] += 1
            weather_counts[meta.attrs["weather"]] += 1
            status_counts[meta.attrs["status"]] += 1

    # Summary
    fps = 20  # Assumed
    total_hours = total_frames / (fps * 3600)
    size_gb = total_size_bytes / 1e9

    print(f"Episodes:     {len(episodes)}")
    print(f"Total frames: {total_frames:,} ({total_hours:.1f} hours at {fps} FPS)")
    print(f"Storage:      {size_gb:.2f} GB")
    print(
        f"Avg episode:  {np.mean(episode_lengths):.0f} frames ({np.mean(episode_lengths)/fps:.1f}s)"
    )
    print()

    # Status breakdown
    print("Episode status:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status:12s}: {count}")
    print()

    # Town breakdown
    print("Town distribution:")
    for town, count in sorted(town_counts.items()):
        print(f"  {town:12s}: {count} episodes")
    print()

    # Weather breakdown
    print("Weather distribution:")
    for weather, count in sorted(weather_counts.items()):
        print(f"  {weather:20s}: {count} episodes")
    print()

    # Navigation command distribution
    print("Navigation commands:")
    for cmd_id in sorted(nav_counts.keys()):
        name = NAV_NAMES.get(cmd_id, f"Unknown({cmd_id})")
        count = nav_counts[cmd_id]
        pct = 100 * count / total_frames if total_frames > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {name:12s}: {count:8d} ({pct:5.1f}%) {bar}")

    # Check balance criterion
    print()
    for cmd_id in [1, 2, 3]:  # Left, Right, Straight
        pct = 100 * nav_counts[cmd_id] / total_frames if total_frames > 0 else 0
        name = NAV_NAMES[cmd_id]
        status = "✓" if pct >= 10.0 else "✗ BELOW 10%"
        print(f"  {name} balance check: {pct:.1f}% {status}")
    print()

    # Steering/throttle statistics
    all_steering = np.concatenate(steering_all)
    all_throttle = np.concatenate(throttle_all)

    print("Control statistics:")
    print(
        f"  Steering:  mean={all_steering.mean():+.4f}, std={all_steering.std():.4f}, "
        f"min={all_steering.min():+.4f}, max={all_steering.max():+.4f}"
    )
    print(
        f"  Throttle:  mean={all_throttle.mean():.4f}, std={all_throttle.std():.4f}, "
        f"min={all_throttle.min():.4f}, max={all_throttle.max():.4f}"
    )
    print()

    # Save random sample frames
    if save_samples > 0:
        _save_sample_frames(episodes, save_samples, data_dir)

    # Check manifest
    manifest_path = data_dir / "collection_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"Manifest: {manifest.get('total_episodes', '?')} episodes tracked")
    else:
        print("WARNING: No collection_manifest.json found")


def _save_sample_frames(
    episodes: list[Path],
    n_samples: int,
    data_dir: Path,
) -> None:
    """Save random frames as PNGs for visual inspection."""
    try:
        from PIL import Image
    except ImportError:
        print("Pillow not installed — skipping sample frame saves")
        return

    sample_dir = data_dir / "samples"
    sample_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(42)
    selected = rng.choice(len(episodes), size=min(n_samples, len(episodes)), replace=False)

    for i, ep_idx in enumerate(selected):
        with h5py.File(episodes[int(ep_idx)], "r") as f:
            n_frames = f["images"].shape[0]
            frame_idx = rng.integers(n_frames)
            img = f["images"][int(frame_idx)]

        save_path = sample_dir / f"sample_{i:03d}_{episodes[int(ep_idx)].stem}_f{frame_idx}.png"
        Image.fromarray(img).save(save_path)
        print(f"  Saved: {save_path.name}")

    print(f"  {len(selected)} sample frames saved to {sample_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze collected driving dataset")
    parser.add_argument("data_dir", type=str, help="Path to dataset directory")
    parser.add_argument("--save-samples", type=int, default=0, help="Save N random frames as PNGs")
    args = parser.parse_args()

    analyze_dataset(Path(args.data_dir), args.save_samples)


if __name__ == "__main__":
    main()
