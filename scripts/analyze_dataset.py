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

from src.data import hdf5_schema as S


def analyze_dataset(data_dir: Path, save_samples: int = 0) -> None:
    """Print comprehensive dataset analysis."""
    episodes = sorted(data_dir.glob("episode_*.h5"))
    if not episodes:
        print(f"No episode files found in {data_dir}")
        return

    print(f"{'=' * 60}")
    print(f"Dataset Analysis: {data_dir.resolve()}")
    print(f"{'=' * 60}")
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
    policy_episodes = 0
    policy_frames = 0
    ev03_violations = 0
    ev03_episodes = 0
    ev03_frames = 0
    ev08_violations = 0
    ev09_violations = 0
    ev10_violations = 0

    for ep_path in episodes:
        total_size_bytes += ep_path.stat().st_size

        with h5py.File(ep_path, "r") as f:
            n_frames = f["images"].shape[0]
            total_frames += n_frames
            episode_lengths.append(n_frames)

            # Nav distribution
            nav = f[S.NAV_COMMAND][:]
            for cmd_id in S.VALID_NAV_COMMANDS:
                nav_counts[cmd_id] += int(np.sum(nav == cmd_id))

            # Controls
            steering_all.append(f[S.STEERING][:])
            throttle_all.append(f[S.THROTTLE][:])

            # Metadata
            meta = f[S.METADATA_GROUP]
            town_counts[meta.attrs[S.META_TOWN]] += 1
            weather_counts[meta.attrs[S.META_WEATHER]] += 1
            status_counts[meta.attrs[S.META_STATUS]] += 1

            policy_fields = (
                S.PRE_JUNCTION_LANE_CHANGE_VIOLATION,
                S.CONSECUTIVE_LANE_CHANGE_VIOLATION,
                S.STRAIGHT_EXIT_LANE_VIOLATION,
            )
            if all(field in f for field in policy_fields):
                policy_episodes += 1
                policy_frames += n_frames
                if S.TURN_LANE_LEGALITY_VIOLATION in f:
                    ev03_episodes += 1
                    ev03_frames += n_frames
                    ev03_violations += int(np.sum(f[S.TURN_LANE_LEGALITY_VIOLATION][:] > 0))
                ev08_violations += int(np.sum(f[S.PRE_JUNCTION_LANE_CHANGE_VIOLATION][:] > 0))
                ev09_violations += int(np.sum(f[S.CONSECUTIVE_LANE_CHANGE_VIOLATION][:] > 0))
                ev10_violations += int(np.sum(f[S.STRAIGHT_EXIT_LANE_VIOLATION][:] > 0))

    # Summary
    fps = 20  # Assumed
    total_hours = total_frames / (fps * 3600)
    size_gb = total_size_bytes / 1e9

    print(f"Episodes:     {len(episodes)}")
    print(f"Total frames: {total_frames:,} ({total_hours:.1f} hours at {fps} FPS)")
    print(f"Storage:      {size_gb:.2f} GB")
    avg_len = np.mean(episode_lengths)
    print(f"Avg episode:  {avg_len:.0f} frames ({avg_len / fps:.1f}s)")
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
        name = S.NAV_COMMAND_NAMES.get(cmd_id, f"Unknown({cmd_id})")
        count = nav_counts[cmd_id]
        pct = 100 * count / total_frames if total_frames > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {name:12s}: {count:8d} ({pct:5.1f}%) {bar}")

    # Check balance criterion
    print()
    for cmd_id in [1, 2, 3]:  # Left, Right, Straight
        pct = 100 * nav_counts[cmd_id] / total_frames if total_frames > 0 else 0
        name = S.NAV_COMMAND_NAMES[cmd_id]
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

    print("Waypoint policy evaluation (EV-08..EV-10):")
    if policy_episodes == 0 or policy_frames == 0:
        print("  Policy telemetry not present in this dataset")
    else:

        def _ev_line(label: str, count: int) -> None:
            rate = 1000.0 * count / max(policy_frames, 1)
            status = "PASS" if count == 0 else "FAIL"
            print(f"  {label}: {count} violations ({rate:.2f} / 1k frames) -> {status}")

        def _ev_line_cov(label: str, count: int, frames: int, episodes_with_field: int) -> None:
            rate = 1000.0 * count / max(frames, 1)
            status = "PASS" if count == 0 else "FAIL"
            print(
                f"  {label}: {count} violations ({rate:.2f} / 1k frames) -> {status} "
                f"[{episodes_with_field}/{policy_episodes} episodes]"
            )

        print(f"  Episodes with policy telemetry: {policy_episodes}/{len(episodes)}")
        if ev03_episodes > 0:
            _ev_line_cov("EV-03 turn-lane legality", ev03_violations, ev03_frames, ev03_episodes)
        else:
            print("  EV-03 turn-lane legality: telemetry not present")
        _ev_line("EV-08 pre-junction lane-change", ev08_violations)
        _ev_line("EV-09 consecutive lane-change", ev09_violations)
        _ev_line("EV-10 straight-in-exit-lane", ev10_violations)
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
            n_frames = f[S.IMAGES].shape[0]
            frame_idx = rng.integers(n_frames)
            img = f[S.IMAGES][int(frame_idx)]

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
