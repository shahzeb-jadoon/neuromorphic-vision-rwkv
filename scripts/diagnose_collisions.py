#!/usr/bin/env python3
"""Diagnose collision and timeout episodes from HDF5 data.

Reads the privileged metadata stored per-frame (nearby vehicles, driving
mode, traffic light state, target speed, controls) and prints a detailed
timeline of what the expert saw and did in the frames leading up to each
collision or timeout.

Usage:
    python scripts/diagnose_collisions.py data/test_collection
    python scripts/diagnose_collisions.py data/test_collection --window 60
    python scripts/diagnose_collisions.py data/test_collection/episode_XYZ.h5
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np


# DrivingMode values (must match privileged_expert.py IntEnum)
MODE_NAMES = {
    1: "FOLLOW_LANE",
    2: "APPROACH_TURN",
    3: "TURNING",
    4: "TL_STOP",
    5: "VEH_FOLLOW",
    6: "PED_STOP",
    7: "EMERGENCY",
}

TL_NAMES = {0: "None", 1: "Green", 2: "Yellow", 3: "Red"}

NAV_NAMES = {1: "Left", 2: "Right", 3: "Straight", 4: "LaneFollow"}


def speed_from_velocity(vx: float, vy: float, vz: float) -> float:
    """m/s → km/h."""
    return math.sqrt(vx**2 + vy**2 + vz**2) * 3.6


def analyze_episode(filepath: str, window: int = 40) -> None:
    """Print detailed per-frame analysis around the failure point."""
    path = Path(filepath)
    print(f"\n{'='*72}")
    print(f"  {path.name}")
    print(f"{'='*72}")

    with h5py.File(filepath, "r") as f:
        meta = f["metadata"]
        status = meta.attrs.get("status", "unknown")
        town = meta.attrs.get("town", "?")
        n_frames = meta.attrs.get("n_frames", 0)
        collision_frame = meta.attrs.get("collision_frame", None)

        print(f"  Status: {status}")
        print(f"  Town: {town}")
        print(f"  Frames: {n_frames}")
        if collision_frame is not None:
            print(f"  Collision at frame: {collision_frame}")
        print()

        # Load arrays
        velocity = f["velocity"][:]
        target_speed = f["target_speed"][:]
        driving_mode = f["driving_mode"][:]
        tl_state = f["traffic_light_state"][:]
        steering = f["steering"][:]
        throttle = f["throttle"][:]
        brake = f["brake"][:]
        nav_cmd = f["nav_command"][:]
        nearby_vehicles_raw = f["nearby_vehicles"][:]

        # Parse nearby vehicles JSON
        nearby_vehicles = []
        for raw in nearby_vehicles_raw:
            try:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                nearby_vehicles.append(json.loads(raw))
            except (json.JSONDecodeError, ValueError):
                nearby_vehicles.append([])

    # Determine analysis window
    if status == "collision" and collision_frame is not None:
        event_frame = int(collision_frame)
        event_label = "COLLISION"
    else:
        # For timeout, find the last frame where the car was moving
        speeds = np.array([speed_from_velocity(*velocity[i]) for i in range(len(velocity))])
        moving_frames = np.where(speeds > 1.0)[0]
        if len(moving_frames) > 0:
            event_frame = int(moving_frames[-1])
        else:
            event_frame = len(velocity) - 1
        event_label = "TIMEOUT / STUCK"

    start = max(0, event_frame - window)
    end = min(len(velocity), event_frame + 5)  # a few frames after

    print(f"  Event: {event_label} at frame {event_frame}")
    print(f"  Showing frames {start}–{end-1}")
    print()

    # ── Overview: mode distribution ──
    mode_counts: dict[str, int] = {}
    for m in driving_mode:
        name = MODE_NAMES.get(int(m), f"?{m}")
        mode_counts[name] = mode_counts.get(name, 0) + 1
    print("  Mode distribution (full episode):")
    for name, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(driving_mode)
        print(f"    {name:20s}: {count:5d} ({pct:5.1f}%)")
    print()

    # ── TL distribution ──
    tl_counts: dict[str, int] = {}
    for t in tl_state:
        name = TL_NAMES.get(int(t), f"?{t}")
        tl_counts[name] = tl_counts.get(name, 0) + 1
    print("  Traffic light distribution (full episode):")
    for name, count in sorted(tl_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(tl_state)
        print(f"    {name:20s}: {count:5d} ({pct:5.1f}%)")
    print()

    # ── Frame-by-frame timeline ──
    print(
        f"  {'Frame':>6s}  {'Speed':>6s}  {'Target':>6s}  {'Thr':>5s}  "
        f"{'Brk':>5s}  {'Steer':>6s}  {'Mode':>14s}  {'TL':>6s}  "
        f"{'Nav':>10s}  {'Lead':>8s}  {'Lead info'}"
    )
    print(
        f"  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*5}  {'─'*6}  "
        f"{'─'*14}  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*30}"
    )

    for i in range(start, end):
        spd = speed_from_velocity(*velocity[i])
        tgt = target_speed[i]
        thr = throttle[i]
        brk = brake[i]
        steer = steering[i]
        mode = MODE_NAMES.get(int(driving_mode[i]), f"?{driving_mode[i]}")
        tl = TL_NAMES.get(int(tl_state[i]), f"?{tl_state[i]}")
        nav = NAV_NAMES.get(int(nav_cmd[i]), f"?{nav_cmd[i]}")

        # Lead vehicle info
        vehs = nearby_vehicles[i]
        lead_info = ""
        lead_dist_str = "—"
        if vehs:
            # Find first ahead + same_direction + in_lane vehicle
            for v in vehs:
                if v.get("is_ahead") and v.get("same_direction", True):
                    in_lane = v.get("obb_in_lane", abs(v.get("lateral", 99)) < 2.5)
                    if in_lane:
                        lead_dist_str = f"{v['distance']:6.1f}m"
                        lead_info = (
                            f"gap={v['distance']:.1f}m "
                            f"spd={v.get('speed_kmh', 0):.0f}km/h "
                            f"lat={v.get('lateral', 0):.1f}"
                        )
                        break

        marker = ""
        if i == event_frame:
            marker = "  ◄◄◄ " + event_label

        print(
            f"  {i:6d}  {spd:5.1f}  {tgt:6.1f}  {thr:5.2f}  "
            f"{brk:5.2f}  {steer:+6.3f}  {mode:>14s}  {tl:>6s}  "
            f"{nav:>10s}  {lead_dist_str:>8s}  {lead_info}{marker}"
        )

    # ── Summary of what happened ──
    print()
    print(f"  {'─'*72}")

    if status == "collision":
        # Analyze the approach
        print("\n  ── Collision Analysis ──")

        # Find when lead vehicle first appeared within hazard distance
        first_detection = None
        for i in range(max(0, event_frame - 200), event_frame + 1):
            vehs = nearby_vehicles[i]
            for v in vehs:
                if v.get("is_ahead") and v.get("same_direction", True):
                    in_lane = v.get("obb_in_lane", abs(v.get("lateral", 99)) < 2.5)
                    if in_lane and v["distance"] < 25.0:
                        if first_detection is None:
                            first_detection = i
                        break

        if first_detection is not None:
            first_speed = speed_from_velocity(*velocity[first_detection])
            first_vehs = nearby_vehicles[first_detection]
            first_gap = None
            for v in first_vehs:
                if v.get("is_ahead") and v.get("same_direction", True):
                    in_lane = v.get("obb_in_lane", abs(v.get("lateral", 99)) < 2.5)
                    if in_lane:
                        first_gap = v["distance"]
                        break
            print(f"  Lead vehicle first in hazard zone at frame {first_detection}")
            print(f"    Frames before collision: {event_frame - first_detection}")
            print(f"    Ego speed at detection:  {first_speed:.1f} km/h")
            if first_gap is not None:
                print(f"    Gap at first detection:  {first_gap:.1f} m")

        collision_speed = speed_from_velocity(*velocity[event_frame])
        print(f"  Ego speed at collision:    {collision_speed:.1f} km/h")
        print(f"  Mode at collision:         {MODE_NAMES.get(int(driving_mode[event_frame]), '?')}")

        # Check: was there NO lead vehicle detected before collision?
        frames_with_lead = 0
        frames_without_lead = 0
        for i in range(max(0, event_frame - 40), event_frame + 1):
            vehs = nearby_vehicles[i]
            found = False
            for v in vehs:
                if v.get("is_ahead") and v.get("same_direction", True):
                    in_lane = v.get("obb_in_lane", abs(v.get("lateral", 99)) < 2.5)
                    if in_lane:
                        found = True
                        break
            if found:
                frames_with_lead += 1
            else:
                frames_without_lead += 1

        window_frames = min(40, event_frame + 1)
        print(f"\n  In the last {window_frames} frames before collision:")
        print(f"    Lead vehicle detected: {frames_with_lead} frames")
        print(f"    NO lead detected:      {frames_without_lead} frames")
        if frames_without_lead > frames_with_lead:
            print("    ⚠  Lead vehicle was NOT detected for most of the approach!")
            print(
                "    → Possible cause: vehicle detection failing"
                " (not in lane, oncoming filter, etc.)"
            )

    elif status == "timeout":
        # Analyze what mode we were stuck in
        stuck_window = driving_mode[max(0, event_frame - 200) : event_frame + 1]
        mode_in_stuck: dict[str, int] = {}
        for m in stuck_window:
            name = MODE_NAMES.get(int(m), f"?{m}")
            mode_in_stuck[name] = mode_in_stuck.get(name, 0) + 1
        print("\n  ── Timeout Analysis ──")
        print("  Modes during stuck period:")
        for name, count in sorted(mode_in_stuck.items(), key=lambda x: -x[1]):
            print(f"    {name:20s}: {count}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose collision/timeout episodes")
    parser.add_argument("path", help="Directory or single .h5 file")
    parser.add_argument(
        "--window", type=int, default=40, help="Frames before event to show (default: 40)"
    )
    parser.add_argument(
        "--all", action="store_true", help="Show all episodes, including completed ones"
    )
    args = parser.parse_args()

    path = Path(args.path)
    if path.is_file():
        files = [str(path)]
    elif path.is_dir():
        files = sorted(glob.glob(str(path / "episode_*.h5")))
    else:
        print(f"ERROR: {path} not found")
        sys.exit(1)

    if not files:
        print(f"No episode files found in {path}")
        sys.exit(1)

    # Filter to collision/timeout episodes unless --all
    filtered = []
    for fp in files:
        with h5py.File(fp, "r") as f:
            status = f["metadata"].attrs.get("status", "unknown")
        if args.all or status in ("collision", "timeout"):
            filtered.append(fp)

    if not filtered:
        print("No collision or timeout episodes found. Use --all to see all episodes.")
        sys.exit(0)

    print(f"\nAnalyzing {len(filtered)} episode(s) from {path}\n")

    for fp in filtered:
        analyze_episode(fp, window=args.window)


if __name__ == "__main__":
    main()
