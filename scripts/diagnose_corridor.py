#!/usr/bin/env python3
"""Diagnose SAT corridor health from HDF5 diagnostic fields (Option D).

Usage:
    python scripts/diagnose_corridor.py data/test_collection
"""

import sys
import glob
import json

import h5py
import numpy as np


def diagnose(data_dir: str) -> None:
    paths = sorted(glob.glob(f"{data_dir}/episode_*.h5"))
    if not paths:
        print(f"No episodes found in {data_dir}")
        return

    # Aggregate totals
    agg_sat = 0
    agg_lat = 0
    agg_miss = 0
    agg_veh_frames = 0
    agg_stationary = 0
    agg_total = 0

    for path in paths:
        with h5py.File(path) as f:
            name = path.split("/")[-1]
            n = f["images"].shape[0]
            agg_total += n

            segs = f["n_corridor_segments"][:]
            buf = f["wp_buffer_size"][:]
            buf_d = f["wp_buffer_head_dist"][:]
            dm = f["driving_mode"][:]
            tl = f["traffic_light_state"][:]
            vel = f["velocity"][:]
            speed = np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2 + vel[:, 2] ** 2) * 3.6

            n_with_vehicles = 0
            total_sat_true = 0
            total_lat_true = 0
            sat_miss_frames = 0
            # Sample a few miss cases for debugging
            miss_samples: list[str] = []
            for i in range(n):
                vlist = json.loads(f["nearby_vehicles"][i])
                ahead_same = [v for v in vlist if v["is_ahead"] and v.get("same_direction", True)]
                if ahead_same:
                    n_with_vehicles += 1
                    any_sat = any(v.get("obb_in_lane", False) for v in ahead_same)
                    any_lat = any(abs(v["lateral"]) < 2.5 for v in ahead_same)
                    if any_sat:
                        total_sat_true += 1
                    if any_lat:
                        total_lat_true += 1
                    if any_lat and not any_sat:
                        sat_miss_frames += 1
                        if len(miss_samples) < 3:
                            best = min(ahead_same, key=lambda v: v["distance"])
                            miss_samples.append(
                                f"    F{i}: dist={best['distance']:.1f}m "
                                f"lat={best['lateral']:.2f}m "
                                f"long={best['longitudinal']:.1f}m "
                                f"segs={segs[i]} buf_head={buf_d[i]:.1f}m"
                            )

            red_frames = int((tl == 3).sum())
            red_follow = int(((tl == 3) & (dm == 1)).sum())
            stationary = int((speed < 0.5).sum())

            agg_sat += total_sat_true
            agg_lat += total_lat_true
            agg_miss += sat_miss_frames
            agg_veh_frames += n_with_vehicles
            agg_stationary += stationary

            print(f"\n{'=' * 60}")
            print(f"{name} ({n} frames)")
            print(f"{'=' * 60}")
            print(
                f"  Corridor segments: mean={segs.mean():.1f}, min={segs.min()}, max={segs.max()}"
            )
            print(f"  WP buffer size:    mean={buf.mean():.1f}, min={buf.min()}, max={buf.max()}")
            print(
                f"  WP buffer head:    mean={buf_d.mean():.1f}m, "
                f"min={buf_d.min():.1f}m, max={buf_d.max():.1f}m"
            )
            print(
                f"  Frames w/ ahead+same vehicles: "
                f"{n_with_vehicles}/{n} ({100 * n_with_vehicles / n:.1f}%)"
            )
            pct_s = 100 * total_sat_true / max(n_with_vehicles, 1)
            pct_l = 100 * total_lat_true / max(n_with_vehicles, 1)
            print(f"    SAT detected:     {total_sat_true}/{n_with_vehicles} ({pct_s:.1f}%)")
            print(f"    Lateral detected: {total_lat_true}/{n_with_vehicles} ({pct_l:.1f}%)")
            print(f"    SAT misses:       {sat_miss_frames}")
            if miss_samples:
                print("    Sample miss frames:")
                for s in miss_samples:
                    print(s)
            print(f"  TL: red={red_frames}, phantom (red+FOLLOW_LANE)={red_follow}")
            print(f"  Stationary (<0.5 km/h): {stationary}/{n} ({100 * stationary / n:.1f}%)")

    # Aggregate summary
    print(f"\n{'=' * 60}")
    print("AGGREGATE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total frames: {agg_total}")
    print(f"  Frames with ahead vehicles: {agg_veh_frames}")
    pct_s = 100 * agg_sat / max(agg_veh_frames, 1)
    pct_l = 100 * agg_lat / max(agg_veh_frames, 1)
    print(f"  SAT detection rate:     {agg_sat}/{agg_veh_frames} ({pct_s:.1f}%)")
    print(f"  Lateral detection rate: {agg_lat}/{agg_veh_frames} ({pct_l:.1f}%)")
    print(f"  SAT misses (lat catches, SAT doesn't): {agg_miss}")
    print(
        f"  Total stationary frames: {agg_stationary}/{agg_total} "
        f"({100 * agg_stationary / agg_total:.1f}%)"
    )


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/test_collection"
    diagnose(data_dir)
