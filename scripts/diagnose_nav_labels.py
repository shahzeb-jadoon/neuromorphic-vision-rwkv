#!/usr/bin/env env python
"""Diagnostic: visualize what the planner labels look like per-frame in an episode.

Usage:
    python scripts/diagnose_nav_labels.py data/test_512/episode_*.h5
    python scripts/diagnose_nav_labels.py data/test_512/episode_e0f9b0f0.h5  # specific
"""

from __future__ import annotations

import sys

import h5py
import numpy as np

NAV_NAMES = {0: "LF", 1: "L ", 2: "R ", 3: "ST"}
NAV_CHARS = {0: ".", 1: "L", 2: "R", 3: "S"}


def diagnose_episode(path: str) -> None:
    """Print frame-by-frame label timeline for one episode."""
    with h5py.File(path, "r") as f:
        meta = f["metadata"]
        town = str(meta.attrs.get("town", "?"))
        n_frames = int(meta.attrs["n_frames"])
        status = str(meta.attrs.get("status", "?"))
        cam_w = int(meta.attrs.get("camera_width", 0))

        planner = f["nav_command_planner"][:]
        heuristic = f["nav_command_heuristic"][:]
        velocity = f["velocity"][:]  # (N, 3)

        speeds = np.sqrt(np.sum(velocity**2, axis=1))

    filename = path.split("/")[-1]
    print(f"\n{'=' * 80}")
    print(f"Episode: {filename}  |  {town}  |  {n_frames} frames  |  {status}  |  {cam_w}×{cam_w}")
    print(f"{'=' * 80}")

    # --- Overall distribution ---
    print("\n  PLANNER label distribution:")
    for cmd_id, name in [(0, "LaneFollow"), (1, "Left"), (2, "Right"), (3, "Straight")]:
        count = int(np.sum(planner == cmd_id))
        pct = count / n_frames * 100
        bar = "#" * int(pct / 2)
        print(f"    {name:12s}: {count:5d}/{n_frames} ({pct:5.1f}%)  {bar}")

    print("\n  HEURISTIC label distribution:")
    for cmd_id, name in [(0, "LaneFollow"), (1, "Left"), (2, "Right"), (3, "Straight")]:
        count = int(np.sum(heuristic == cmd_id))
        pct = count / n_frames * 100
        bar = "#" * int(pct / 2)
        print(f"    {name:12s}: {count:5d}/{n_frames} ({pct:5.1f}%)  {bar}")

    agree = int(np.sum(planner == heuristic))
    print(f"\n  Agreement: {agree}/{n_frames} ({agree / n_frames * 100:.1f}%)")

    # --- Speed profile (average per 100 frames) ---
    print("\n  Speed profile (avg m/s per 100-frame chunk):")
    chunks = n_frames // 100
    print("  Frame:  ", end="")
    for i in range(min(chunks, 20)):
        print(f" {i * 100:4d}", end="")
    print()
    print("  Speed:  ", end="")
    for i in range(min(chunks, 20)):
        avg_speed = float(np.mean(speeds[i * 100 : (i + 1) * 100]))
        print(f" {avg_speed:4.1f}", end="")
    print()

    # --- Timeline: one char per 10 frames ---
    # Shows planner label as a character: .=LF, L=Left, R=Right, S=Straight
    print("\n  Planner label timeline (1 char = 10 frames):")
    step = 10
    n_chars = n_frames // step
    line = ""
    for i in range(n_chars):
        chunk = planner[i * step : (i + 1) * step]
        # Use the most common label in this 10-frame chunk
        counts = np.bincount(chunk, minlength=4)
        dominant = int(np.argmax(counts))
        line += NAV_CHARS[dominant]

    # Print in rows of 100 chars (= 1000 frames)
    print(f"  {'':>8s}  |{'Frame 0':.<50s}{'Frame 1000':.<50s}|")
    for row_start in range(0, len(line), 100):
        row = line[row_start : row_start + 100]
        frame_start = row_start * step
        print(f"  {frame_start:>6d}:  |{row}|")

    # Same for heuristic
    print("\n  Heuristic label timeline (1 char = 10 frames):")
    line_h = ""
    for i in range(n_chars):
        chunk = heuristic[i * step : (i + 1) * step]
        counts = np.bincount(chunk, minlength=4)
        dominant = int(np.argmax(counts))
        line_h += NAV_CHARS[dominant]

    print(f"  {'':>8s}  |{'Frame 0':.<50s}{'Frame 1000':.<50s}|")
    for row_start in range(0, len(line_h), 100):
        row = line_h[row_start : row_start + 100]
        frame_start = row_start * step
        print(f"  {frame_start:>6d}:  |{row}|")

    # --- Transition log: when does the planner label change? ---
    print("\n  Planner label transitions (frame → new label):")
    prev = planner[0]
    transitions = [(0, int(prev))]
    for i in range(1, n_frames):
        if planner[i] != prev:
            transitions.append((i, int(planner[i])))
            prev = planner[i]

    if len(transitions) <= 40:
        for frame, cmd in transitions:
            name = ["LaneFollow", "Left", "Right", "Straight"][cmd]
            speed_at = float(speeds[frame]) if frame < len(speeds) else 0.0
            print(f"    Frame {frame:5d}: → {name:12s}  (speed={speed_at:.1f} m/s)")
    else:
        print(f"    ({len(transitions)} transitions — showing first 20 and last 10)")
        for frame, cmd in transitions[:20]:
            name = ["LaneFollow", "Left", "Right", "Straight"][cmd]
            speed_at = float(speeds[frame]) if frame < len(speeds) else 0.0
            print(f"    Frame {frame:5d}: → {name:12s}  (speed={speed_at:.1f} m/s)")
        print("    ...")
        for frame, cmd in transitions[-10:]:
            name = ["LaneFollow", "Left", "Right", "Straight"][cmd]
            speed_at = float(speeds[frame]) if frame < len(speeds) else 0.0
            print(f"    Frame {frame:5d}: → {name:12s}  (speed={speed_at:.1f} m/s)")

    # --- Stuck analysis ---
    stuck_threshold = 0.5  # m/s
    stuck_frames = int(np.sum(speeds < stuck_threshold))
    stuck_pct = stuck_frames / n_frames * 100

    # What labels does planner assign during stuck frames?
    stuck_mask = speeds < stuck_threshold
    if np.any(stuck_mask):
        stuck_labels = planner[stuck_mask]
        print(f"\n  Stuck analysis (speed < {stuck_threshold} m/s):")
        print(f"    Stuck frames: {stuck_frames}/{n_frames} ({stuck_pct:.1f}%)")
        print("    Planner labels during stuck frames:")
        for cmd_id, name in [(0, "LaneFollow"), (1, "Left"), (2, "Right"), (3, "Straight")]:
            count = int(np.sum(stuck_labels == cmd_id))
            if count > 0:
                pct = count / stuck_frames * 100
                print(f"      {name:12s}: {count:5d} ({pct:.1f}%)")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/diagnose_nav_labels.py <episode.h5> [episode2.h5] ...")
        sys.exit(1)

    import glob

    paths = []
    for arg in sys.argv[1:]:
        paths.extend(sorted(glob.glob(arg)))

    if not paths:
        print(f"No files matched: {sys.argv[1:]}")
        sys.exit(1)

    for path in paths:
        diagnose_episode(path)

    print()


if __name__ == "__main__":
    main()
