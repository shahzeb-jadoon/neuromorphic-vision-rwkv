"""Retroactively label STRAIGHT and lane-change nav commands from trajectory geometry.

Existing HDF5 episodes only contain LEFT=1, RIGHT=2, LANEFOLLOW=4 because the
expert skips STRAIGHT and CHANGELANELEFT/CHANGELANERIGHT.  This script infers
those missing labels from position + rotation data already in each episode.

Detection logic
---------------
**STRAIGHT (3):**
  At junction traversals (where the expert recorded LEFT or RIGHT for a nearby
  waypoint, but the ego actually went straight), detect sequences where:
  - The current label is LANEFOLLOW (4)
  - A heading-change spike indicates a junction (other vehicles turning, road
    geometry change) but the ego heading stays nearly constant (< 15°)
  - Position data shows the ego crossed from one road segment to another
    (significant displacement without heading change)

  Simpler heuristic: look at consecutive LANEFOLLOW frames where the ego
  passes through a junction (detected via speed dip + heading stability).
  Since the expert labels everything as LANEFOLLOW when going straight through
  a junction, we need to identify these junctions from trajectory features.

**CHANGELANELEFT (5) / CHANGELANERIGHT (6):**
  Detect lateral displacement perpendicular to heading:
  - Compute frame-to-frame lateral offset relative to heading direction
  - If cumulative lateral displacement > 2m over a short window (20 frames)
    while heading stays roughly constant (< 5° change), label as lane change
  - Direction determined by sign of lateral displacement relative to heading

Usage
-----
    python scripts/extrapolate_nav_labels.py --data-dir /mnt/data/carla_data/ --dry-run
    python scripts/extrapolate_nav_labels.py --data-dir /mnt/data/carla_data/ --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRAIGHT_HEADING_THRESH_DEG = 15.0  # Max heading change to qualify as "straight"
LANE_CHANGE_LATERAL_THRESH_M = 2.0  # Min lateral displacement for lane change
LANE_CHANGE_HEADING_THRESH_DEG = 5.0  # Max heading change during lane change
LANE_CHANGE_WINDOW = 20  # Frames to accumulate lateral displacement
JUNCTION_SPEED_DIP_THRESH_KMH = 15.0  # Speed dip suggesting junction approach
JUNCTION_WINDOW = 40  # Frames to look for junction pattern


def _heading_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """Extract yaw (heading) in radians from rotation array.

    Args:
        rotation: (N, 3) degrees — (pitch, yaw, roll)

    Returns:
        (N,) heading in radians, unwrapped for continuity.
    """
    yaw_deg = rotation[:, 1]
    yaw_rad = np.deg2rad(yaw_deg)
    return np.unwrap(yaw_rad)


def _lateral_displacement(position: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """Compute per-frame lateral displacement relative to heading direction.

    For each consecutive pair of frames, project the displacement vector
    onto the direction perpendicular to the heading.  Positive = leftward.

    Args:
        position: (N, 3) world position (x, y, z).
        heading: (N,) heading in radians (unwrapped).

    Returns:
        (N,) cumulative lateral displacement.  First element is 0.
    """
    dx = np.diff(position[:, 0])
    dy = np.diff(position[:, 1])
    # Perpendicular to heading: (-sin(h), cos(h))
    mid_heading = (heading[:-1] + heading[1:]) / 2.0
    lat = -np.sin(mid_heading) * dx + np.cos(mid_heading) * dy
    return np.concatenate([[0.0], np.cumsum(lat)])


def _detect_straight(
    nav_cmd: np.ndarray,
    heading: np.ndarray,
    velocity: np.ndarray,
) -> np.ndarray:
    """Detect STRAIGHT-through-junction frames currently labeled LANEFOLLOW.

    Heuristic: scan for regions where:
      1. nav_cmd == 4 (LANEFOLLOW)
      2. Speed dips below threshold (junction approach — braking/slowing)
      3. Heading stays within ±STRAIGHT_HEADING_THRESH_DEG over the junction window

    These are frames where the ego went straight through a junction but was
    labeled LANEFOLLOW because the expert skips STRAIGHT.

    Returns:
        Boolean mask (N,) — True where LANEFOLLOW should become STRAIGHT.
    """
    n = len(nav_cmd)
    is_straight = np.zeros(n, dtype=bool)

    # Compute speed from velocity (horizontal plane)
    speed_kmh = np.linalg.norm(velocity[:, :2], axis=1) * 3.6

    half_win = JUNCTION_WINDOW // 2

    for i in range(half_win, n - half_win):
        # Only consider frames currently labeled LANEFOLLOW
        if nav_cmd[i] != 4:
            continue

        window = slice(i - half_win, i + half_win)

        # Check for speed dip in the window (junction approach)
        min_speed = speed_kmh[window].min()
        if min_speed > JUNCTION_SPEED_DIP_THRESH_KMH:
            continue

        # Check heading stability — small heading change means straight
        heading_window = heading[window]
        heading_range = heading_window.max() - heading_window.min()
        heading_range_deg = np.rad2deg(heading_range)

        if heading_range_deg < STRAIGHT_HEADING_THRESH_DEG:
            # Check that we're near a junction: look for LEFT/RIGHT labels
            # within a broader neighbourhood (±80 frames)
            broad = slice(max(0, i - 80), min(n, i + 80))
            has_turn_nearby = np.any((nav_cmd[broad] == 1) | (nav_cmd[broad] == 2))
            if has_turn_nearby:
                is_straight[window] = nav_cmd[window] == 4

    return is_straight


def _detect_lane_changes(
    nav_cmd: np.ndarray,
    heading: np.ndarray,
    lateral: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect lane change frames from lateral displacement patterns.

    Scans for windows where:
      1. nav_cmd == 4 (LANEFOLLOW)
      2. Lateral displacement accumulates > threshold
      3. Heading stays roughly constant

    Returns:
        (is_left, is_right) — two boolean masks (N,).
    """
    n = len(nav_cmd)
    is_left = np.zeros(n, dtype=bool)
    is_right = np.zeros(n, dtype=bool)

    heading_deg = np.rad2deg(heading)

    i = 0
    while i < n - LANE_CHANGE_WINDOW:
        # Only consider LANEFOLLOW frames
        if nav_cmd[i] != 4:
            i += 1
            continue

        window = slice(i, i + LANE_CHANGE_WINDOW)

        # Check heading stability
        h_win = heading_deg[window]
        h_range = h_win.max() - h_win.min()
        if h_range > LANE_CHANGE_HEADING_THRESH_DEG:
            i += 1
            continue

        # Check lateral displacement
        lat_win = lateral[window]
        lat_change = lat_win[-1] - lat_win[0]

        if abs(lat_change) > LANE_CHANGE_LATERAL_THRESH_M:
            mask = nav_cmd[window] == 4
            if lat_change > 0:
                is_left[window] = mask
            else:
                is_right[window] = mask
            i += LANE_CHANGE_WINDOW  # Skip past this detected change
        else:
            i += 1

    return is_left, is_right


def process_episode(path: Path, *, apply: bool = False) -> dict[str, str | int | float | bool]:
    """Process a single HDF5 episode.

    Args:
        path: Path to episode HDF5 file.
        apply: If True, write updated nav_command back to HDF5.

    Returns:
        Dict with counts of detected labels.
    """
    mode = "r+" if apply else "r"
    try:
        f = h5py.File(path, mode)
    except Exception as e:
        return {"error": str(e)}

    try:
        # Check schema version
        meta = f.get("metadata")
        if meta is None:
            return {"error": "no metadata group"}
        version = meta.attrs.get("schema_version", 0)
        if version < 3:
            return {"error": f"schema_version={version}, need >=3"}

        nav_cmd = f["nav_command"][:]
        position = f["position"][:]
        rotation = f["rotation"][:]
        velocity = f["velocity"][:]

        n = len(nav_cmd)
        if n < JUNCTION_WINDOW * 2:
            return {"skipped": True, "reason": "too short", "n_frames": n}

        heading = _heading_from_rotation(rotation)
        lateral = _lateral_displacement(position, heading)

        # Detect STRAIGHT
        is_straight = _detect_straight(nav_cmd, heading, velocity)

        # Detect lane changes
        is_left, is_right = _detect_lane_changes(nav_cmd, heading, lateral)

        # Resolve conflicts: STRAIGHT takes priority over lane change
        is_left = is_left & ~is_straight
        is_right = is_right & ~is_straight

        n_straight = int(is_straight.sum())
        n_left_lc = int(is_left.sum())
        n_right_lc = int(is_right.sum())

        if apply and (n_straight + n_left_lc + n_right_lc) > 0:
            updated = nav_cmd.copy()
            updated[is_straight] = 3  # STRAIGHT
            updated[is_left] = 5  # CHANGELANELEFT
            updated[is_right] = 6  # CHANGELANERIGHT
            f["nav_command"][...] = updated

        return {
            "n_frames": n,
            "n_straight": n_straight,
            "n_left_lc": n_left_lc,
            "n_right_lc": n_right_lc,
            "pct_straight": round(100 * n_straight / n, 1) if n else 0,
            "pct_lane_change": round(100 * (n_left_lc + n_right_lc) / n, 1) if n else 0,
        }
    finally:
        f.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrapolate STRAIGHT + lane-change nav labels from trajectory data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/mnt/data/carla_data/"),
        help="Directory containing episode HDF5 files.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write updated labels back to HDF5 (default: dry-run).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Only report what would change (default).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N episodes (0 = all).",
    )
    args = parser.parse_args()

    if args.apply:
        args.dry_run = False

    episodes = sorted(args.data_dir.glob("*.h5"))
    if args.limit > 0:
        episodes = episodes[: args.limit]

    if not episodes:
        print(f"No .h5 files found in {args.data_dir}")
        sys.exit(1)

    print(f"{'DRY RUN' if not args.apply else 'APPLYING CHANGES'}")
    print(f"Processing {len(episodes)} episodes from {args.data_dir}\n")

    totals = {"n_straight": 0, "n_left_lc": 0, "n_right_lc": 0, "n_frames": 0}
    errors = 0

    for i, ep in enumerate(episodes):
        result = process_episode(ep, apply=args.apply)

        if "error" in result:
            errors += 1
            if i < 5:
                print(f"  ERROR {ep.name}: {result['error']}")
            continue

        if "skipped" in result:
            continue

        for key in totals:
            totals[key] += int(result.get(key, 0))

        ns = int(result["n_straight"])
        nl = int(result["n_left_lc"])
        nr = int(result["n_right_lc"])
        if i < 10 or (ns + nl + nr) > 0:
            print(
                f"  {ep.name}: {ns} STRAIGHT, "
                f"{nl} LC-L, {nr} LC-R "
                f"({result['pct_straight']}% / {result['pct_lane_change']}%)"
            )

    n = totals["n_frames"]
    print(f"\n{'=' * 60}")
    print(f"TOTAL: {len(episodes)} episodes, {n} frames")
    print(
        f"  STRAIGHT:        {totals['n_straight']:>8} ({100 * totals['n_straight'] / n:.1f}%)"
        if n
        else ""
    )
    print(
        f"  CHANGELANELEFT:  {totals['n_left_lc']:>8} ({100 * totals['n_left_lc'] / n:.1f}%)"
        if n
        else ""
    )
    print(
        f"  CHANGELANERIGHT: {totals['n_right_lc']:>8} ({100 * totals['n_right_lc'] / n:.1f}%)"
        if n
        else ""
    )
    if errors:
        print(f"  ERRORS: {errors}")
    if not args.apply:
        print("\nDry run — no files modified. Use --apply to write changes.")


if __name__ == "__main__":
    main()
