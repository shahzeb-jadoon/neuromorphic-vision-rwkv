#!/usr/bin/env python3
"""Deep SAT pipeline audit — examine every assumption.

This script does NOT require CARLA. It works purely on collected HDF5 data,
extracting the stored nearby_vehicles JSON to understand the patterns of
SAT success vs failure.

Assumptions under test:
  A1: Corridor geometry — are there enough segments?
  A2: Vehicle lateral position — where are the misses?
  A3: Distance distribution — are misses close or far?
  A4: Per-town patterns — is it geometry-dependent?
  A5: Per-episode consistency — does SAT work in *some* episodes?
  A6: Corridor width — do misses cluster at specific lateral offsets?
"""

import sys
import glob
import json
from collections import defaultdict

import h5py
import numpy as np


def audit(data_dir: str, max_episodes: int = 100) -> None:
    paths = sorted(glob.glob(f"{data_dir}/episode_*.h5"))
    if not paths:
        print(f"No episodes found in {data_dir}")
        return

    if len(paths) > max_episodes:
        # Sample evenly
        indices = np.linspace(0, len(paths) - 1, max_episodes, dtype=int)
        paths = [paths[i] for i in indices]

    print(f"Analyzing {len(paths)} episodes from {data_dir}")
    print("=" * 70)

    # Accumulators
    total_vehicle_obs = 0  # all vehicle observations (any direction)
    total_ahead_same = 0  # ahead + same direction
    sat_hits = 0
    lat_only = 0  # lateral detects, SAT doesn't (= sat_misses)
    sat_only = 0  # SAT detects, lateral doesn't
    both_detect = 0
    neither = 0

    # Distributions
    miss_distances = []
    miss_laterals = []
    miss_longitudinals = []
    hit_distances = []
    hit_laterals = []

    miss_n_segs = []
    miss_buf_head = []
    hit_n_segs = []

    # Per-episode SAT rate
    episode_sat_rates = []

    # Per-town
    town_stats: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"ahead_same": 0, "sat": 0, "lat": 0, "both": 0}
    )

    # Corridor segment distribution
    seg_counts = []

    for path in paths:
        try:
            with h5py.File(path, "r") as f:
                n = f["images"].shape[0]
                ep_name = path.split("/")[-1].replace(".h5", "")

                # Get town from manifest or config
                town = "unknown"
                if "town" in f.attrs:
                    town = f.attrs["town"]
                elif "config" in f.attrs:
                    try:
                        cfg = json.loads(f.attrs["config"])
                        town = cfg.get("town", "unknown")
                    except Exception:
                        pass

                has_segs = "n_corridor_segments" in f
                has_buf = "wp_buffer_head_dist" in f

                segs_arr = f["n_corridor_segments"][:] if has_segs else np.zeros(n)
                buf_arr = f["wp_buffer_head_dist"][:] if has_buf else np.zeros(n)

                ep_ahead_same = 0
                ep_sat = 0

                for i in range(n):
                    vlist = json.loads(f["nearby_vehicles"][i])
                    n_segs = int(segs_arr[i]) if has_segs else -1
                    buf_head = float(buf_arr[i]) if has_buf else -1.0

                    seg_counts.append(n_segs)

                    for v in vlist:
                        total_vehicle_obs += 1

                        is_ahead = v.get("is_ahead", False)
                        same_dir = v.get("same_direction", True)

                        if not (is_ahead and same_dir):
                            continue

                        total_ahead_same += 1
                        ep_ahead_same += 1

                        obb = v.get("obb_in_lane", False)
                        lat = abs(v.get("lateral", 999)) < 2.5  # lane_half_width
                        dist = v.get("distance", 999)
                        lateral_val = v.get("lateral", 0)
                        longi_val = v.get("longitudinal", 0)

                        # Classification
                        if obb and lat:
                            both_detect += 1
                            ep_sat += 1
                            hit_distances.append(dist)
                            hit_laterals.append(lateral_val)
                            hit_n_segs.append(n_segs)
                        elif obb and not lat:
                            sat_only += 1
                            ep_sat += 1
                            hit_distances.append(dist)
                            hit_laterals.append(lateral_val)
                            hit_n_segs.append(n_segs)
                        elif lat and not obb:
                            lat_only += 1
                            miss_distances.append(dist)
                            miss_laterals.append(lateral_val)
                            miss_longitudinals.append(longi_val)
                            miss_n_segs.append(n_segs)
                            miss_buf_head.append(buf_head)
                        else:
                            neither += 1

                        town_stats[town]["ahead_same"] += 1
                        if obb:
                            town_stats[town]["sat"] += 1
                        if lat:
                            town_stats[town]["lat"] += 1
                        if obb and lat:
                            town_stats[town]["both"] += 1

                if ep_ahead_same > 0:
                    episode_sat_rates.append(
                        (ep_name, ep_sat / ep_ahead_same * 100, ep_ahead_same, town)
                    )
        except Exception as e:
            print(f"  Error reading {path}: {e}")
            continue

    sat_hits = both_detect + sat_only

    print(f"\n{'=' * 70}")
    print("A0: OVERALL SAT vs LATERAL DETECTION")
    print(f"{'=' * 70}")
    print(f"Total vehicle observations (all dirs):  {total_vehicle_obs}")
    print(f"Ahead + same direction:                 {total_ahead_same}")
    denom = max(total_ahead_same, 1)
    both_pct = both_detect / denom * 100
    sat_only_pct = sat_only / denom * 100
    lat_only_pct = lat_only / denom * 100
    neither_pct = neither / denom * 100
    sat_pct = sat_hits / denom * 100
    lat_total = both_detect + lat_only
    lat_total_pct = lat_total / denom * 100
    print(f"  Both SAT+LAT detect:  {both_detect} ({both_pct:.1f}%)")
    print(f"  SAT only (lat misses):{sat_only} ({sat_only_pct:.1f}%)")
    print(f"  LAT only (SAT misses):{lat_only} ({lat_only_pct:.1f}%)")
    print(f"  Neither detects:      {neither} ({neither_pct:.1f}%)")
    print(f"  SAT total:            {sat_hits} ({sat_pct:.1f}%)")
    print(f"  LAT total:            {lat_total} ({lat_total_pct:.1f}%)")

    print(f"\n{'=' * 70}")
    print("A1: CORRIDOR SEGMENT COUNT DISTRIBUTION")
    print(f"{'=' * 70}")
    seg_arr = np.array(seg_counts)
    if len(seg_arr) > 0:
        print(f"  Mean segments: {seg_arr.mean():.1f}")
        print(f"  Median:        {np.median(seg_arr):.0f}")
        print(f"  Min:           {seg_arr.min()}")
        print(f"  Max:           {seg_arr.max()}")
        zero_ct = int((seg_arr == 0).sum())
        one_ct = int((seg_arr == 1).sum())
        seg_n = len(seg_arr)
        print(f"  0 segments:    {zero_ct} ({zero_ct / seg_n * 100:.1f}%)")
        print(f"  1 segment:     {one_ct} ({one_ct / seg_n * 100:.1f}%)")
        print(f"  2-5 segs:      {((seg_arr >= 2) & (seg_arr <= 5)).sum()}")
        print(f"  6-10 segs:     {((seg_arr >= 6) & (seg_arr <= 10)).sum()}")
        print(f"  11-20 segs:    {((seg_arr >= 11) & (seg_arr <= 20)).sum()}")
        print(f"  21+ segs:      {(seg_arr >= 21).sum()}")

    print("\n  SAT MISS frames - segment counts:")
    if miss_n_segs:
        ms = np.array(miss_n_segs)
        print(f"    Mean: {ms.mean():.1f}, Median: {np.median(ms):.0f}")
        print(f"    0 segs: {(ms == 0).sum()}, 1 seg: {(ms == 1).sum()}")
        print(f"    2-5: {((ms >= 2) & (ms <= 5)).sum()}")
        print(f"    6-20: {((ms >= 6) & (ms <= 20)).sum()}")
        print(f"    21+: {(ms >= 21).sum()}")

    print("\n  SAT HIT frames - segment counts:")
    if hit_n_segs:
        hs = np.array(hit_n_segs)
        print(f"    Mean: {hs.mean():.1f}, Median: {np.median(hs):.0f}")

    print(f"\n{'=' * 70}")
    print("A2: LATERAL POSITION OF SAT MISSES")
    print(f"{'=' * 70}")
    if miss_laterals:
        ml = np.array(miss_laterals)
        print("  Misses within |lat| < 2.5m (by definition):")
        print(f"    Mean |lateral|: {np.abs(ml).mean():.2f}m")
        print(f"    Median |lateral|: {np.median(np.abs(ml)):.2f}m")
        print(f"    0.0-0.5m: {(np.abs(ml) < 0.5).sum()}")
        print(f"    0.5-1.0m: {((np.abs(ml) >= 0.5) & (np.abs(ml) < 1.0)).sum()}")
        print(f"    1.0-1.5m: {((np.abs(ml) >= 1.0) & (np.abs(ml) < 1.5)).sum()}")
        print(f"    1.5-2.0m: {((np.abs(ml) >= 1.5) & (np.abs(ml) < 2.0)).sum()}")
        print(f"    2.0-2.5m: {((np.abs(ml) >= 2.0) & (np.abs(ml) < 2.5)).sum()}")
        print(f"    Left (neg): {(ml < 0).sum()}, Right (pos): {(ml >= 0).sum()}")

    if hit_laterals:
        hl = np.array(hit_laterals)
        print("\n  SAT HITS lateral distribution:")
        print(f"    Mean |lateral|: {np.abs(hl).mean():.2f}m")
        print(f"    Median |lateral|: {np.median(np.abs(hl)):.2f}m")

    print(f"\n{'=' * 70}")
    print("A3: DISTANCE DISTRIBUTION — MISSES vs HITS")
    print(f"{'=' * 70}")
    if miss_distances:
        md = np.array(miss_distances)
        print("  SAT MISSES (lat detects, SAT doesn't):")
        print(f"    Mean: {md.mean():.1f}m, Median: {np.median(md):.1f}m")
        print(f"    0-5m:   {(md < 5).sum()}")
        print(f"    5-10m:  {((md >= 5) & (md < 10)).sum()}")
        print(f"    10-15m: {((md >= 10) & (md < 15)).sum()}")
        print(f"    15-25m: {((md >= 15) & (md < 25)).sum()}")
        print(f"    25-50m: {((md >= 25) & (md < 50)).sum()}")

    if hit_distances:
        hd = np.array(hit_distances)
        print("\n  SAT HITS:")
        print(f"    Mean: {hd.mean():.1f}m, Median: {np.median(hd):.1f}m")
        print(f"    0-5m:   {(hd < 5).sum()}")
        print(f"    5-10m:  {((hd >= 5) & (hd < 10)).sum()}")
        print(f"    10-15m: {((hd >= 10) & (hd < 15)).sum()}")
        print(f"    15-25m: {((hd >= 15) & (hd < 25)).sum()}")
        print(f"    25-50m: {((hd >= 25) & (hd < 50)).sum()}")

    print(f"\n{'=' * 70}")
    print("A4: PER-TOWN SAT RATES")
    print(f"{'=' * 70}")
    for town, stats in sorted(town_stats.items()):
        n = stats["ahead_same"]
        sr = stats["sat"] / max(n, 1) * 100
        lr = stats["lat"] / max(n, 1) * 100
        br = stats["both"] / max(n, 1) * 100
        print(f"  {town}: SAT={sr:.1f}% LAT={lr:.1f}% BOTH={br:.1f}% (n={n})")

    print(f"\n{'=' * 70}")
    print("A5: PER-EPISODE SAT RATE DISTRIBUTION")
    print(f"{'=' * 70}")
    if episode_sat_rates:
        rates = [r[1] for r in episode_sat_rates]
        ra = np.array(rates)
        print(f"  Episodes with vehicle obs: {len(episode_sat_rates)}")
        print(f"  Mean SAT rate:   {ra.mean():.1f}%")
        print(f"  Median SAT rate: {np.median(ra):.1f}%")
        print(f"  0% SAT rate:     {(ra == 0).sum()} episodes")
        print(f"  0-5% SAT rate:   {(ra < 5).sum()} episodes")
        print(f"  5-20%:           {((ra >= 5) & (ra < 20)).sum()}")
        print(f"  20-50%:          {((ra >= 20) & (ra < 50)).sum()}")
        print(f"  50-90%:          {((ra >= 50) & (ra < 90)).sum()}")
        print(f"  90-100%:         {(ra >= 90).sum()}")

        # Show top 10 and bottom 10
        sorted_rates = sorted(episode_sat_rates, key=lambda x: x[1], reverse=True)
        print("\n  TOP 10 SAT rate episodes:")
        for name, rate, n, town in sorted_rates[:10]:
            print(f"    {name}: {rate:.1f}% (n={n}, {town})")
        print("\n  BOTTOM 10 SAT rate episodes (with >10 obs):")
        bottom = [r for r in sorted_rates if r[2] > 10]
        for name, rate, n, town in bottom[-10:]:
            print(f"    {name}: {rate:.1f}% (n={n}, {town})")

    print(f"\n{'=' * 70}")
    print("A6: LONGITUDINAL POSITION OF SAT MISSES")
    print(f"{'=' * 70}")
    if miss_longitudinals:
        mlong = np.array(miss_longitudinals)
        print(f"  Mean longitudinal: {mlong.mean():.1f}m")
        print(f"  Median:            {np.median(mlong):.1f}m")
        print(f"  0-5m:   {(mlong < 5).sum()}")
        print(f"  5-10m:  {((mlong >= 5) & (mlong < 10)).sum()}")
        print(f"  10-15m: {((mlong >= 10) & (mlong < 15)).sum()}")
        print(f"  15-25m: {((mlong >= 15) & (mlong < 25)).sum()}")
        print(f"  25-50m: {((mlong >= 25) & (mlong < 50)).sum()}")

    # A7: Check if miss buf_head correlates with misses
    print(f"\n{'=' * 70}")
    print("A7: BUFFER HEAD DISTANCE AT SAT MISSES")
    print(f"{'=' * 70}")
    if miss_buf_head:
        mbh = np.array(miss_buf_head)
        valid = mbh[mbh >= 0]
        if len(valid):
            print(f"  Mean buf_head dist: {valid.mean():.1f}m")
            print(f"  Median:             {np.median(valid):.1f}m")
            print(f"  0-2m:   {(valid < 2).sum()}")
            print(f"  2-5m:   {((valid >= 2) & (valid < 5)).sum()}")
            print(f"  5-10m:  {((valid >= 5) & (valid < 10)).sum()}")
            print(f"  10+m:   {(valid >= 10).sum()}")

    # A8: Cross-tabulate: for SAT misses at close range (<10m),
    # what are the corridor segments and lateral offsets?
    print(f"\n{'=' * 70}")
    print("A8: CLOSE-RANGE SAT MISSES (<10m) — DETAIL")
    print(f"{'=' * 70}")
    close_misses = [
        (d, lat, lo, s, b)
        for d, lat, lo, s, b in zip(
            miss_distances, miss_laterals, miss_longitudinals, miss_n_segs, miss_buf_head
        )
        if d < 10
    ]
    print(f"  Total close-range SAT misses: {len(close_misses)}")
    if close_misses:
        cd = np.array([x[0] for x in close_misses])
        cl = np.array([x[1] for x in close_misses])
        clo = np.array([x[2] for x in close_misses])
        cs = np.array([x[3] for x in close_misses])
        print(f"  Distance: mean={cd.mean():.1f}m median={np.median(cd):.1f}m")
        print(f"  |Lateral|: mean={np.abs(cl).mean():.2f}m median={np.median(np.abs(cl)):.2f}m")
        print(f"  Longitudinal: mean={clo.mean():.1f}m median={np.median(clo):.1f}m")
        print(f"  Segments: mean={cs.mean():.1f} median={np.median(cs):.0f}")
        print(f"  Segments=0: {(cs == 0).sum()}")
        print(f"  Segments 1-5: {((cs >= 1) & (cs <= 5)).sum()}")
        print(f"  Segments 6+: {(cs >= 6).sum()}")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/carla_data"
    max_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    audit(data_dir, max_ep)
