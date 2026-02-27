#!/usr/bin/env python3
"""Deep SAT miss analysis: distance distribution + corridor geometry check.

Answers: Are SAT misses happening at dangerous close range, or only at
far distances where the union safety net handles them?
"""

import sys
import glob
import json

import h5py
import numpy as np


def deep_analysis(data_dir: str) -> None:
    paths = sorted(glob.glob(f"{data_dir}/episode_*.h5"))

    # Collect all SAT miss distances
    all_miss_dists: list[float] = []
    all_miss_details: list[tuple[str, int, float, float, float, int, float]] = []
    close_miss_count = 0  # within 25m (hazard distance)
    very_close_miss = 0  # within 10m (emergency zone)

    for path in paths:
        with h5py.File(path) as f:
            name = path.split("/")[-1].replace(".h5", "")
            n = f["images"].shape[0]
            segs = f["n_corridor_segments"][:]
            buf_d = f["wp_buffer_head_dist"][:]

            for i in range(n):
                vlist = json.loads(f["nearby_vehicles"][i])
                ahead_same = [v for v in vlist if v["is_ahead"] and v.get("same_direction", True)]
                for v in ahead_same:
                    lat_in = abs(v["lateral"]) < 2.5
                    sat_in = v.get("obb_in_lane", False)
                    if lat_in and not sat_in:
                        d = v["distance"]
                        all_miss_dists.append(d)
                        if d < 25:
                            close_miss_count += 1
                        if d < 10:
                            very_close_miss += 1
                        if len(all_miss_details) < 20 and d < 25:
                            all_miss_details.append(
                                (
                                    name,
                                    i,
                                    d,
                                    v["lateral"],
                                    v["longitudinal"],
                                    int(segs[i]),
                                    float(buf_d[i]),
                                )
                            )

    if not all_miss_dists:
        print("No SAT misses found!")
        return

    dists = np.array(all_miss_dists)
    print("=" * 60)
    print("SAT MISS DISTANCE DISTRIBUTION")
    print("=" * 60)
    print(f"Total SAT miss instances: {len(dists)}")
    print(f"  0-5m:   {(dists < 5).sum()}")
    print(f"  5-10m:  {((dists >= 5) & (dists < 10)).sum()}")
    print(f"  10-15m: {((dists >= 10) & (dists < 15)).sum()}")
    print(f"  15-25m: {((dists >= 15) & (dists < 25)).sum()}")
    print(f"  25-35m: {((dists >= 25) & (dists < 35)).sum()}")
    print(f"  35-50m: {((dists >= 35) & (dists < 50)).sum()}")
    print(f"  Mean: {dists.mean():.1f}m, Median: {np.median(dists):.1f}m")
    print(f"\n  Close-range misses (<25m): {close_miss_count}")
    print(f"  Emergency-zone misses (<10m): {very_close_miss}")

    if all_miss_details:
        print(f"\n{'='*60}")
        print("CLOSE-RANGE MISS SAMPLES (<25m)")
        print(f"{'='*60}")
        for ep, frame, d, lat, lng, seg, bh in sorted(all_miss_details, key=lambda x: x[2]):
            print(
                f"  {ep} F{frame}: dist={d:.1f}m lat={lat:.2f}m "
                f"long={lng:.1f}m segs={seg} buf_head={bh:.1f}m"
            )

    # Also check: in the episodes where SAT works (53662066),
    # what distances are the successful SAT detections at?
    print(f"\n{'='*60}")
    print("SAT SUCCESS DISTANCE DISTRIBUTION (episode_53662066)")
    print(f"{'='*60}")
    sat_path = f"{data_dir}/episode_53662066.h5"
    try:
        with h5py.File(sat_path) as f:
            n = f["images"].shape[0]
            sat_dists = []
            for i in range(n):
                vlist = json.loads(f["nearby_vehicles"][i])
                for v in vlist:
                    if (
                        v["is_ahead"]
                        and v.get("same_direction", True)
                        and v.get("obb_in_lane", False)
                    ):
                        sat_dists.append(v["distance"])
            if sat_dists:
                sd = np.array(sat_dists)
                print(f"  SAT success instances: {len(sd)}")
                print(f"  0-5m:   {(sd < 5).sum()}")
                print(f"  5-10m:  {((sd >= 5) & (sd < 10)).sum()}")
                print(f"  10-15m: {((sd >= 10) & (sd < 15)).sum()}")
                print(f"  15-25m: {((sd >= 15) & (sd < 25)).sum()}")
                print(f"  25-50m: {((sd >= 25) & (sd < 50)).sum()}")
                print(f"  Mean: {sd.mean():.1f}m, Median: {np.median(sd):.1f}m")
    except Exception as e:
        print(f"  Could not analyze: {e}")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/test_collection"
    deep_analysis(data_dir)
