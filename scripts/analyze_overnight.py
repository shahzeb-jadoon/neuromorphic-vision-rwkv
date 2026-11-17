#!/usr/bin/env python3
"""Comprehensive overnight collection analysis.

Analyzes:
1. SAT corridor diagnostics across all episodes
2. Deep collision analysis (what happened in frames before each collision)
3. Stationary frame statistics
4. TL phantom detection
5. Hours shortfall analysis

Usage:
    python scripts/analyze_overnight.py /mnt/data/carla_data
"""

import sys
import glob
import json
from collections import Counter, defaultdict

import h5py
import numpy as np


def analyze_all(data_dir: str) -> None:
    paths = sorted(glob.glob(f"{data_dir}/episode_*.h5"))
    if not paths:
        print(f"No episodes found in {data_dir}")
        return

    # Load manifest for status info
    manifest_path = f"{data_dir}/collection_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    status_map = {e["filename"]: e for e in manifest["episodes"]}

    # Aggregates
    total_frames = 0
    total_stationary = 0
    total_sat_veh_frames = 0
    total_sat_hits = 0
    total_lat_hits = 0
    total_sat_misses = 0
    total_phantom_tl = 0
    total_red_frames = 0
    collision_details = []
    town_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"frames": 0, "sat_hits": 0, "lat_hits": 0, "veh_frames": 0, "stationary": 0}
    )

    print(f"Analyzing {len(paths)} episodes from {data_dir}...")
    print(f"(This may take a while for {len(paths)} episodes)\n")

    for idx, path in enumerate(paths):
        fname = path.split("/")[-1]
        ep_info = status_map.get(fname, {})
        town = ep_info.get("town", "?")
        weather = ep_info.get("weather", "?")
        status = ep_info.get("status", "?")
        col_frame = ep_info.get("collision_frame")

        with h5py.File(path, "r") as f:
            n = f["images"].shape[0]
            total_frames += n

            vel = f["velocity"][:]
            speed = np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2 + vel[:, 2] ** 2) * 3.6
            stat = int((speed < 0.5).sum())
            total_stationary += stat

            dm = f["driving_mode"][:]
            tl = f["traffic_light_state"][:]
            red = int((tl == 3).sum())
            phantom = int(((tl == 3) & (dm == 1)).sum())
            total_red_frames += red
            total_phantom_tl += phantom

            # Corridor diagnostics
            has_diag = "n_corridor_segments" in f
            if has_diag:
                segs = f["n_corridor_segments"][:]
                buf_d = f["wp_buffer_head_dist"][:]

            # Per-frame vehicle analysis
            ep_sat = 0
            ep_lat = 0
            ep_veh = 0
            ep_miss = 0

            for i in range(n):
                vlist = json.loads(f["nearby_vehicles"][i])
                ahead_same = [v for v in vlist if v["is_ahead"] and v.get("same_direction", True)]
                if ahead_same:
                    ep_veh += 1
                    any_sat = any(v.get("obb_in_lane", False) for v in ahead_same)
                    any_lat = any(abs(v["lateral"]) < 2.5 for v in ahead_same)
                    if any_sat:
                        ep_sat += 1
                    if any_lat:
                        ep_lat += 1
                    if any_lat and not any_sat:
                        ep_miss += 1

            total_sat_veh_frames += ep_veh
            total_sat_hits += ep_sat
            total_lat_hits += ep_lat
            total_sat_misses += ep_miss

            ts = town_stats[town]
            ts["frames"] += n
            ts["sat_hits"] += ep_sat
            ts["lat_hits"] += ep_lat
            ts["veh_frames"] += ep_veh
            ts["stationary"] += stat

            # === COLLISION DEEP DIVE ===
            if status == "collision" and col_frame is not None:
                # Analyze 100 frames before collision
                start = max(0, col_frame - 100)

                # What was the driving mode at collision?
                col_mode = int(dm[col_frame]) if col_frame < n else -1
                col_speed = float(speed[col_frame]) if col_frame < n else -1

                # What vehicles were nearby at collision?
                col_vehicles = json.loads(f["nearby_vehicles"][min(col_frame, n - 1)])
                closest_ahead = None
                for v in col_vehicles:
                    if v["is_ahead"] and v.get("same_direction", True):
                        closest_ahead = v
                        break

                # Was there a lead vehicle detected?
                lead_detected = closest_ahead is not None
                lead_dist = closest_ahead["distance"] if closest_ahead else None
                lead_sat = closest_ahead.get("obb_in_lane", False) if closest_ahead else False
                lead_lat = abs(closest_ahead["lateral"]) < 2.5 if closest_ahead else False

                # Check frames leading up: was there ever a vehicle detected?
                had_lead_before = False
                lost_lead_frame = None
                for i in range(start, col_frame):
                    vl = json.loads(f["nearby_vehicles"][i])
                    ahead = [v for v in vl if v["is_ahead"] and v.get("same_direction", True)]
                    in_lane = [
                        v for v in ahead if v.get("obb_in_lane", False) or abs(v["lateral"]) < 2.5
                    ]
                    if in_lane:
                        had_lead_before = True
                    elif had_lead_before and not in_lane:
                        lost_lead_frame = i
                        had_lead_before = False  # track last loss

                # Position and heading at collision
                pos = f["position"][col_frame] if col_frame < n else [0, 0, 0]
                rot = f["rotation"][col_frame] if col_frame < n else [0, 0, 0]

                # Corridor health at collision
                col_segs = int(segs[col_frame]) if has_diag and col_frame < n else -1
                col_buf_d = float(buf_d[col_frame]) if has_diag and col_frame < n else -1

                # Speed profile: what happened in last 3 seconds (60 frames)?
                last_60 = speed[max(0, col_frame - 60) : col_frame + 1]

                collision_details.append(
                    {
                        "file": fname,
                        "town": town,
                        "weather": weather,
                        "col_frame": col_frame,
                        "total_frames": n,
                        "col_speed": round(col_speed, 1),
                        "col_mode": col_mode,
                        "col_segs": col_segs,
                        "col_buf_d": round(col_buf_d, 1),
                        "lead_detected": lead_detected,
                        "lead_dist": round(lead_dist, 1) if lead_dist else None,
                        "lead_sat": lead_sat,
                        "lead_lat": lead_lat,
                        "had_lead_before": had_lead_before or (lost_lead_frame is not None),
                        "lost_lead_frame": lost_lead_frame,
                        "speed_3s_avg": round(float(last_60.mean()), 1),
                        "speed_3s_max": round(float(last_60.max()), 1),
                        "pos": [round(float(x), 1) for x in pos],
                        "yaw": round(float(rot[1]), 1),
                        "n_vehicles_at_col": len(col_vehicles),
                    }
                )

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(paths)} episodes...")

    # =====================================================================
    # PRINT RESULTS
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("OVERNIGHT COLLECTION — FULL DIAGNOSTIC REPORT")
    print(f"{'=' * 70}")

    print("\n--- Overall Stats ---")
    print(f"  Total frames: {total_frames:,} ({total_frames / 20 / 3600:.1f} hours)")
    print(
        f"  Stationary frames: {total_stationary:,} ({100 * total_stationary / total_frames:.1f}%)"
    )
    active = total_frames - total_stationary
    print(f"  Active (moving) frames: {active:,} ({active / 20 / 3600:.1f} hours)")

    print("\n--- SAT Corridor Health ---")
    pct_s = 100 * total_sat_hits / max(total_sat_veh_frames, 1)
    pct_l = 100 * total_lat_hits / max(total_sat_veh_frames, 1)
    print(f"  Frames with ahead+same vehicles: {total_sat_veh_frames:,}")
    print(f"  SAT detection rate:     {total_sat_hits:,}/{total_sat_veh_frames:,} ({pct_s:.1f}%)")
    print(f"  Lateral detection rate: {total_lat_hits:,}/{total_sat_veh_frames:,} ({pct_l:.1f}%)")
    print(f"  SAT misses (lat catches, SAT doesn't): {total_sat_misses:,}")

    print("\n--- SAT by Town ---")
    for town in sorted(town_stats):
        ts = town_stats[town]
        s_pct = 100 * ts["sat_hits"] / max(ts["veh_frames"], 1)
        l_pct = 100 * ts["lat_hits"] / max(ts["veh_frames"], 1)
        stat_pct = 100 * ts["stationary"] / max(ts["frames"], 1)
        print(
            f"  {town}: SAT={s_pct:.1f}% Lat={l_pct:.1f}% "
            f"({ts['veh_frames']:,} veh-frames) Stationary={stat_pct:.1f}%"
        )

    print("\n--- Traffic Light ---")
    print(f"  Total red frames: {total_red_frames:,}")
    print(f"  Phantom (red+FOLLOW_LANE): {total_phantom_tl}")

    # =====================================================================
    # COLLISION ANALYSIS
    # =====================================================================
    print(f"\n{'=' * 70}")
    print(f"COLLISION ANALYSIS ({len(collision_details)} episodes)")
    print(f"{'=' * 70}")

    if collision_details:
        # Categorize collisions
        categories: Counter[str] = Counter()
        for c in collision_details:
            if c["col_speed"] < 1.0:
                categories["stationary_collision"] += 1
            elif not c["lead_detected"] and c["n_vehicles_at_col"] == 0:
                categories["no_vehicle_nearby"] += 1
            elif c["lead_detected"] and c["lead_dist"] is not None and c["lead_dist"] < 5:
                if c["lead_sat"] or c["lead_lat"]:
                    categories["detected_but_too_close"] += 1
                else:
                    categories["undetected_close_vehicle"] += 1
            elif c["had_lead_before"] and not c["lead_detected"]:
                categories["lost_tracking"] += 1
            else:
                categories["other"] += 1

        print("\n  Collision Categories:")
        for cat, cnt in sorted(categories.items(), key=lambda x: -x[1]):
            print(f"    {cat}: {cnt}")

        # Mode distribution at collision
        from agents.expert.privileged_expert import DrivingMode

        mode_counts: Counter[str] = Counter()
        for c in collision_details:
            try:
                mode_counts[DrivingMode(c["col_mode"]).name] += 1
            except ValueError:
                mode_counts[f"mode_{c['col_mode']}"] += 1

        print("\n  Driving mode at collision:")
        for mode, cnt in sorted(mode_counts.items(), key=lambda x: -x[1]):
            print(f"    {mode}: {cnt}")

        # Speed distribution at collision
        speeds = [c["col_speed"] for c in collision_details]
        print("\n  Speed at collision:")
        print(f"    Mean: {np.mean(speeds):.1f} km/h")
        print(f"    Median: {np.median(speeds):.1f} km/h")
        print(f"    <5 km/h: {sum(1 for s in speeds if s < 5)}")
        print(f"    5-15 km/h: {sum(1 for s in speeds if 5 <= s < 15)}")
        print(f"    15-25 km/h: {sum(1 for s in speeds if 15 <= s < 25)}")
        print(f"    >25 km/h: {sum(1 for s in speeds if s >= 25)}")

        # Print each collision detail
        print("\n  Individual collision details:")
        for c in sorted(collision_details, key=lambda x: x["col_speed"], reverse=True):
            lead_info = ""
            if c["lead_detected"]:
                sat_flag = "Y" if c["lead_sat"] else "N"
                lat_flag = "Y" if c["lead_lat"] else "N"
                lead_info = f"lead={c['lead_dist']}m SAT={sat_flag} LAT={lat_flag}"
            else:
                lead_info = "no_lead_detected"

            print(
                f"    {c['file']}: {c['town']}/{c['weather']} "
                f"F{c['col_frame']} speed={c['col_speed']}km/h "
                f"mode={DrivingMode(c['col_mode']).name if c['col_mode'] > 0 else '?'} "
                f"segs={c['col_segs']} {lead_info} "
                f"3s_avg={c['speed_3s_avg']}km/h nveh={c['n_vehicles_at_col']}"
            )


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/carla_data"
    analyze_all(data_dir)
