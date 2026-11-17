#!/usr/bin/env python3
"""Check driving mode distribution at SAT miss vs SAT hit frames.

If misses cluster at APPROACHING_TURN / TURNING, the junction theory holds.
Also compare high-SAT and low-SAT episodes structurally.
"""

import sys
import glob
import json
from collections import Counter

import h5py
import numpy as np
from src.utils.settings import setting

# DrivingMode enum values (from privileged_expert.py)
MODE_NAMES = {
    0: "FOLLOW_LANE",
    1: "APPROACHING_TURN",
    2: "TURNING",
    3: "TRAFFIC_LIGHT_STOP",
    4: "VEHICLE_FOLLOWING",
    5: "EMERGENCY_STOP",
}


def analyze(data_dir: str, max_episodes: int = 100) -> None:
    paths = sorted(glob.glob(f"{data_dir}/episode_*.h5"))
    if len(paths) > max_episodes:
        indices = np.linspace(0, len(paths) - 1, max_episodes, dtype=int)
        paths = [paths[i] for i in indices]

    miss_modes: Counter[int] = Counter()
    hit_modes: Counter[int] = Counter()
    miss_nav_cmds: Counter[int] = Counter()
    hit_nav_cmds: Counter[int] = Counter()

    # Per-episode: track (sat_rate, mode_distribution, nav_distribution)
    episode_profiles: list[dict] = []

    for path in paths:
        try:
            with h5py.File(path, "r") as f:
                n = f["images"].shape[0]
                ep_name = path.split("/")[-1].replace(".h5", "")
                modes = f["driving_mode"][:]
                nav_cmds = f["nav_command"][:]
                pos = f["position"][:]

                ep_miss = 0
                ep_hit = 0
                ep_total_ahead = 0

                for i in range(n):
                    vlist = json.loads(f["nearby_vehicles"][i])
                    mode = int(modes[i])
                    nav = int(nav_cmds[i])

                    for v in vlist:
                        if not v.get("is_ahead", False):
                            continue
                        if not v.get("same_direction", True):
                            continue

                        obb = v.get("obb_in_lane", False)
                        lat_in = abs(v.get("lateral", 999)) < 2.5

                        if not (obb or lat_in):
                            continue  # neither detects, skip

                        ep_total_ahead += 1

                        if obb:
                            hit_modes[mode] += 1
                            hit_nav_cmds[nav] += 1
                            ep_hit += 1
                        else:
                            # lat_in but not obb → SAT miss
                            miss_modes[mode] += 1
                            miss_nav_cmds[nav] += 1
                            ep_miss += 1

                if ep_total_ahead > 0:
                    # Trajectory analysis: how curvy is the route?
                    positions = pos[:, :2]  # XY
                    # Compute total heading changes
                    diffs = np.diff(positions, axis=0)
                    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
                    angle_changes = np.abs(np.diff(angles))
                    # Wrap to [-pi, pi]
                    angle_changes = np.minimum(angle_changes, 2 * np.pi - angle_changes)
                    total_heading_change = np.sum(angle_changes)
                    n_turns = int((angle_changes > 0.05).sum())  # >~3° per frame

                    # Mode distribution
                    mode_dist = Counter(modes.tolist())

                    episode_profiles.append(
                        {
                            "name": ep_name,
                            "sat_rate": ep_hit / ep_total_ahead * 100,
                            "n_total": ep_total_ahead,
                            "total_heading_change_deg": np.degrees(total_heading_change),
                            "n_turns": n_turns,
                            "pct_follow_lane": mode_dist.get(0, 0) / n * 100,
                            "pct_turning": (mode_dist.get(1, 0) + mode_dist.get(2, 0)) / n * 100,
                            "pct_vehicle_follow": mode_dist.get(4, 0) / n * 100,
                        }
                    )
        except Exception as e:
            print(f"Error reading {path}: {e}")

    # Report
    print("=" * 70)
    print("DRIVING MODE AT SAT MISS FRAMES")
    print("=" * 70)
    total_miss = sum(miss_modes.values())
    for mode_id in sorted(set(list(miss_modes.keys()) + list(hit_modes.keys()))):
        mode_name = MODE_NAMES.get(mode_id, f"UNKNOWN_{mode_id}")
        mc = miss_modes.get(mode_id, 0)
        hc = hit_modes.get(mode_id, 0)
        mp = mc / max(total_miss, 1) * 100
        total_hit = sum(hit_modes.values())
        hp = hc / max(total_hit, 1) * 100
        print(f"  {mode_name:20s}: MISS={mc:6d} ({mp:5.1f}%)  HIT={hc:6d} ({hp:5.1f}%)")

    print(f"\n{'=' * 70}")
    print("NAV COMMAND AT SAT MISS vs HIT FRAMES")
    print("=" * 70)
    # RoadOption values: 0=VOID, 1=LEFT, 2=RIGHT, 3=STRAIGHT,
    # 4=LANEFOLLOW, 5=CHANGELANELEFT, 6=CHANGELANERIGHT
    NAV_NAMES = {
        0: "VOID",
        1: "LEFT",
        2: "RIGHT",
        3: "STRAIGHT",
        4: "LANEFOLLOW",
        5: "CHANGELANELEFT",
        6: "CHANGELANERIGHT",
    }
    for nav_id in sorted(set(list(miss_nav_cmds.keys()) + list(hit_nav_cmds.keys()))):
        nav_name = NAV_NAMES.get(nav_id, f"CMD_{nav_id}")
        mc = miss_nav_cmds.get(nav_id, 0)
        hc = hit_nav_cmds.get(nav_id, 0)
        mp = mc / max(total_miss, 1) * 100
        hp = hc / max(sum(hit_modes.values()), 1) * 100
        print(f"  {nav_name:20s}: MISS={mc:6d} ({mp:5.1f}%)  HIT={hc:6d} ({hp:5.1f}%)")

    print(f"\n{'=' * 70}")
    print("EPISODE SAT RATE vs CURVINESS CORRELATION")
    print("=" * 70)
    if episode_profiles:
        # Sort by SAT rate
        sorted_eps = sorted(episode_profiles, key=lambda x: x["sat_rate"], reverse=True)

        # Top 5
        print("  TOP 5 SAT rate episodes:")
        for ep in sorted_eps[:5]:
            print(
                f"    {ep['name']}: SAT={ep['sat_rate']:.1f}%  "
                f"heading_chg={ep['total_heading_change_deg']:.0f}°  "
                f"turns={ep['n_turns']}  "
                f"follow={ep['pct_follow_lane']:.0f}%  "
                f"turning={ep['pct_turning']:.0f}%  "
                f"veh_follow={ep['pct_vehicle_follow']:.0f}%"
            )

        # Bottom 5
        print("\n  BOTTOM 5 SAT rate episodes (n>10):")
        bottom = [e for e in sorted_eps if e["n_total"] > 10]
        for ep in bottom[-5:]:
            print(
                f"    {ep['name']}: SAT={ep['sat_rate']:.1f}%  "
                f"heading_chg={ep['total_heading_change_deg']:.0f}°  "
                f"turns={ep['n_turns']}  "
                f"follow={ep['pct_follow_lane']:.0f}%  "
                f"turning={ep['pct_turning']:.0f}%  "
                f"veh_follow={ep['pct_vehicle_follow']:.0f}%"
            )

        # Correlation
        rates = np.array([e["sat_rate"] for e in episode_profiles])
        headings = np.array([e["total_heading_change_deg"] for e in episode_profiles])
        turning_pcts = np.array([e["pct_turning"] for e in episode_profiles])

        if len(rates) > 5:
            corr_heading = np.corrcoef(rates, headings)[0, 1]
            corr_turning = np.corrcoef(rates, turning_pcts)[0, 1]
            print(f"\n  Correlation(SAT_rate, heading_change): {corr_heading:.3f}")
            print(f"  Correlation(SAT_rate, turning_%):      {corr_turning:.3f}")
            print("  (Negative = more turns → worse SAT)")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else setting("DATA_DIR")
    max_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    analyze(data_dir, max_ep)
