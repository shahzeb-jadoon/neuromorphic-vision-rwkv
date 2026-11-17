#!/usr/bin/env python3
"""
Forensic analysis of dual-buffer behaviour across all 9 test_collection episodes.

Diagnoses:
  1. wp_buffer_head_dist large jumps (>10m in one frame)  -> premature pop / freeze catchup
  2. wp_buffer_head_dist growing while moving              -> buffer frozen / arc traversal
  3. route_completion regressions                          -> ego going backwards
  4. Long stalls on same waypoint while ego is moving      -> stuck / wrong-target
  5. Mode transitions around anomalies
"""

import os
import math
from typing import cast

import h5py
import numpy as np

DATA_DIR = "data/test_collection"

MODE_NAMES = {
    1: "FOLLOW_LANE",
    2: "APPROACHING_TURN",
    3: "TURNING",
    4: "TL_STOP",
    5: "VEH_FOLLOWING",
    6: "PED_STOP",
    7: "EMERGENCY",
    8: "STOP_SIGN",
    9: "YIELD_STOP",
}

NAV_NAMES = {
    -1: "VOID",
    1: "LEFT",
    2: "RIGHT",
    3: "STRAIGHT",
    4: "LANEFOLLOW",
    5: "CHNG_L",
    6: "CHNG_R",
}

EPISODES = [
    ("1f723746", "Town07"),
    ("239d65e7", "Town07"),
    ("25372e09", "Town07"),
    ("4cd458df", "Town10HD"),
    ("6309d8f8", "Town10HD"),
    ("7acac7f8", "Town10HD"),
    ("a7f5279c", "Town06"),
    ("b497c070", "Town06"),
    ("b8ab0276", "Town06"),
]


def speed_mps(vel: np.ndarray) -> np.ndarray:
    """Compute speed (m/s) from (vx, vy, vz) array."""
    return cast(np.ndarray, np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2 + vel[:, 2] ** 2))


def yaw_to_fwd(yaw_deg: float) -> tuple[float, float]:
    """CARLA: yaw=0 → East (+x), yaw=90 → South (+y)"""
    r = math.radians(yaw_deg)
    return math.cos(r), math.sin(r)


def analyse_episode(ep_id: str, town: str) -> None:
    h5path = os.path.join(DATA_DIR, f"episode_{ep_id}.h5")
    if not os.path.exists(h5path):
        print(f"  MISSING: {h5path}")
        return

    with h5py.File(h5path, "r") as h:
        wpd = h["wp_buffer_head_dist"][:]  # (N,)
        wps = h["wp_buffer_size"][:]  # (N,)
        rc = h["route_completion"][:]  # (N,)
        mode = h["driving_mode"][:]  # (N,)
        nav = h["nav_command"][:]  # (N,)
        vel = h["velocity"][:]  # (N,3)
        gfd = h["gap_fill_diverged"][:]  # (N,)

    N = len(wpd)
    spd = speed_mps(vel)

    print(f"\n{'='*70}")
    print(f"EPISODE {ep_id}  [{town}]  N={N} frames")
    print(f"{'='*70}")
    print(f"  route_completion: {rc[0]:.3f} → {rc[-1]:.3f}")
    print(
        f"  speed:            min={spd.min():.2f}  max={spd.max():.2f}  mean={spd.mean():.2f} m/s"
    )
    print(
        f"  wp_head_dist:     min={wpd.min():.2f}  " f"max={wpd.max():.2f}  mean={wpd.mean():.2f} m"
    )
    print(f"  wp_buffer_size:   min={wps.min()}  max={wps.max()}")
    print(f"  gap_fill_diverged: {gfd.sum()} frames ({100*gfd.mean():.1f}%)")

    # --- 1. Route completion regressions ---
    rc_diff = np.diff(rc)
    rc_reg = np.where(rc_diff < -0.005)[0]  # frames where RC dropped noticeably
    if len(rc_reg):
        print(f"\n  [REGRESSION] route_completion went BACKWARD at {len(rc_reg)} frames:")
        for f in rc_reg[:10]:
            print(
                f"    f{f:5d}: rc {rc[f]:.4f} → {rc[f+1]:.4f}  "
                f"mode={MODE_NAMES.get(mode[f], '?')}  spd={spd[f]:.2f}"
            )
    else:
        print("\n  [OK] route_completion monotonically increasing")

    # --- 2. Large wpd jumps (anomalous pops or freeze catchup) ---
    wpd_diff = np.diff(wpd.astype(float))
    # Jump UP (dist suddenly large) → arc or pop left something far ahead
    jump_up = np.where(wpd_diff > 10.0)[0]
    # Jump DOWN (dist suddenly small) → caught up after freeze
    jump_dn = np.where(wpd_diff < -10.0)[0]

    if len(jump_up):
        print(
            f"\n  [JUMP-UP >10m] wp_head_dist jumps UP at {len(jump_up)} frames "
            "(waypoint suddenly far):"
        )
        for f in jump_up[:15]:
            print(
                f"    f{f:5d}: {wpd[f]:.1f}m → {wpd[f+1]:.1f}m  Δ={wpd_diff[f]:+.1f}  "
                f"mode={MODE_NAMES.get(mode[f], '?')}  nav={NAV_NAMES.get(nav[f], '?')}  "
                f"spd={spd[f]:.2f}"
            )
    else:
        print("\n  [OK] no large wp_head_dist jump-up")

    if len(jump_dn):
        print(f"\n  [JUMP-DOWN >10m] wp_head_dist jumps DOWN at {len(jump_dn)} frames (catchup):")
        for f in jump_dn[:15]:
            print(
                f"    f{f:5d}: {wpd[f]:.1f}m → {wpd[f+1]:.1f}m  Δ={wpd_diff[f]:+.1f}  "
                f"mode={MODE_NAMES.get(mode[f], '?')}  nav={NAV_NAMES.get(nav[f], '?')}  "
                f"spd={spd[f]:.2f}"
            )
    else:
        print("\n  [OK] no large wp_head_dist jump-down (catchup)")

    # --- 3. Frozen buffer while moving: wpd grows >2m over a window while spd>0.5 ---
    FREEZE_WINDOW = 40  # frames
    FREEZE_GROWTH = 5.0  # metres
    freeze_events = []
    i = 0
    while i < N - FREEZE_WINDOW:
        window_spd = spd[i : i + FREEZE_WINDOW].mean()
        growth = wpd[i + FREEZE_WINDOW] - wpd[i]
        if window_spd > 0.5 and growth > FREEZE_GROWTH:
            freeze_events.append((i, i + FREEZE_WINDOW, growth, window_spd))
            i += FREEZE_WINDOW  # skip this window
        else:
            i += 1

    if freeze_events:
        print(
            f"\n  [FREEZE?] wpd growing >5m over 40-frame window while moving: "
            f"{len(freeze_events)} events"
        )
        for f0, f1, grow, avg_spd in freeze_events[:10]:
            print(
                f"    f{f0:5d}–f{f1:5d}: "
                f"wpd {wpd[f0]:.1f}→{wpd[f1]:.1f}m (+{grow:.1f}m) "
                f"avg_spd={avg_spd:.2f}m/s  "
                f"mode={MODE_NAMES.get(mode[f0], '?')}..{MODE_NAMES.get(mode[f1], '?')}  "
                f"nav={NAV_NAMES.get(nav[f0],'?')}"
            )
    else:
        print("\n  [OK] no sustained wpd growth while moving (no frozen-buffer signature)")

    # --- 4. Waypoint "behind ego" estimation ---
    # Estimate direction: if ego yaw points east (yaw=0), forward is +x.
    # wp direction relative to ego: dot(forward, ego→wp) > 0 means ahead.
    # We have wp_head_dist but NOT wp position. Infer: is ego moving toward or away from wp?
    # Proxy: if spd > 0.5 and wpd increases frame-over-frame by >0.5m, ego is moving AWAY.
    wpd_d = np.diff(wpd.astype(float))
    moving_away = (spd[:-1] > 0.5) & (wpd_d > 0.5)
    n_moving_away = moving_away.sum()
    # Find runs of moving-away frames
    runs = []
    in_run = False
    run_start = 0
    for i, ma in enumerate(moving_away):
        if ma and not in_run:
            in_run = True
            run_start = i
        elif not ma and in_run:
            in_run = False
            runs.append((run_start, i))
    if in_run:
        runs.append((run_start, len(moving_away)))

    long_runs = [(s, e) for s, e in runs if e - s >= 10]
    if long_runs:
        print(
            "\n  [AWAY] ego moving AWAY from wp_head for >=10 frames: "
            f"{len(long_runs)} runs ({n_moving_away} total frames)"
        )
        for s, e in long_runs[:8]:
            dur = e - s
            print(
                f"    f{s:5d}–f{e:5d} ({dur}fr)  wpd: {wpd[s]:.1f}→{wpd[e]:.1f}m  "
                f"spd={spd[s:e].mean():.2f}m/s  mode={MODE_NAMES.get(mode[s], '?')}  "
                f"nav={NAV_NAMES.get(nav[s], '?')}"
            )
    else:
        print("\n  [OK] no sustained runs of ego moving away from waypoint head")

    # --- 5. mode timeline summary ---
    mode_blocks = []
    prev_m = mode[0]
    blk_start = 0
    for i in range(1, N):
        if mode[i] != prev_m:
            mode_blocks.append((blk_start, i - 1, prev_m))
            blk_start = i
            prev_m = mode[i]
    mode_blocks.append((blk_start, N - 1, prev_m))
    print(f"\n  MODE timeline ({len(mode_blocks)} blocks):")
    for s, e, m in mode_blocks:
        dur = e - s + 1
        label = MODE_NAMES.get(m, f"?{m}")
        rc_s, rc_e = rc[s], rc[e]
        print(
            f"    f{s:5d}–f{e:5d} ({dur:4d}fr) {label:<18} "
            f"rc={rc_s:.3f}→{rc_e:.3f}  wpd={wpd[s]:.1f}→{wpd[e]:.1f}"
        )

    # --- 6. arc-buffer freeze detection via gap_fill_diverged ---
    # gap_fill_diverged=1 means arc was active (we logged divergence from WP head)
    gfd_diff = np.diff(gfd.astype(float))
    arc_starts = np.where(gfd_diff > 0)[0]
    arc_ends = np.where(gfd_diff < 0)[0]
    if len(arc_starts):
        print("\n  ARC EVENTS (gap_fill_diverged transitions):")
        for idx, fs in enumerate(arc_starts[:10]):
            fe = arc_ends[arc_ends > fs][0] if len(arc_ends[arc_ends > fs]) else N - 1
            dur = fe - fs
            print(
                f"    arc#{idx+1}: f{fs:5d}→f{fe:5d} ({dur}fr)  wpd: {wpd[fs]:.1f}→{wpd[fe]:.1f}m  "
                f"mode_start={MODE_NAMES.get(mode[fs], '?')}  spd_avg={spd[fs:fe].mean():.2f}"
            )

    # --- 7. Final sanity: did it finish? ---
    final_rc = rc[-1]
    final_mode = mode[-1]
    print(
        f"\n  SUMMARY: final rc={final_rc:.4f}  final_mode={MODE_NAMES.get(final_mode,'?')}  "
        f"last_wpd={wpd[-1]:.1f}m  last_spd={spd[-1]:.2f}m/s"
    )


def main() -> None:
    print("DUAL-BUFFER FORENSIC ANALYSIS")
    print("data/test_collection — all 9 episodes")

    for ep_id, town in EPISODES:
        analyse_episode(ep_id, town)

    print("\n\nDONE.")


if __name__ == "__main__":
    main()
