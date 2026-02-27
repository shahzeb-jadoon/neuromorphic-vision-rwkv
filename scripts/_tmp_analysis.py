import glob
import json

import h5py
import numpy as np

files = sorted(
    glob.glob(
        "~/projects/neuromorphic-vision-rwkv/data/test_collection/episode_*.h5"
    )
)

print("=" * 70)
print("PART 1: RED LIGHT DETAIL")
print("=" * 70)

for f in files:
    ep_name = f.split("/")[-1].replace(".h5", "")
    with h5py.File(f, "r") as h:
        vel = h["velocity"][:]  # (N, 3) velocity vector
        speed = np.linalg.norm(vel, axis=1)  # scalar speed m/s
        n = speed.shape[0]
        tl_state = h["traffic_light_state"][:]

        # No traffic_light_distance in this schema
        has_tl_dist = "traffic_light_distance" in h
        if has_tl_dist:
            tl_dist = h["traffic_light_distance"][:]

        red_frames = np.where(tl_state == 3)[0]
        if len(red_frames) == 0:
            continue

        print(f"\n--- {ep_name} ---")
        print(f"  Red frames: {len(red_frames)}")

        if has_tl_dist:
            red_not_detected = np.where((tl_state == 3) & (tl_dist < 0))[0]
            red_detected = np.where((tl_state == 3) & (tl_dist >= 0))[0]
            print(f"  Red + detected (dist>=0): {len(red_detected)}")
            print(f"  Red + NOT detected (dist<0): {len(red_not_detected)}")
            if len(red_detected) > 0:
                print(
                    f"  TL distance when detected: "
                    f"min={tl_dist[red_detected].min():.1f}m, "
                    f"max={tl_dist[red_detected].max():.1f}m, "
                    f"mean={tl_dist[red_detected].mean():.1f}m"
                )
        else:
            print("  (no traffic_light_distance field in this dataset)")

        # Show speed stats during red
        red_speeds = speed[red_frames]
        print(
            f"  Speed during red: min={red_speeds.min():.2f}, "
            f"max={red_speeds.max():.2f}, mean={red_speeds.mean():.2f} m/s"
        )
        red_moving = np.where((tl_state == 3) & (speed > 1.0))[0]
        print(f"  Red + moving (speed>1 m/s): {len(red_moving)} frames")

        if len(red_moving) > 0:
            # Find first cluster
            first = red_moving[0]
            start = max(first - 20, 0)
            end = min(first + 40, n)
            print(f"\n  First red-running starts at frame {first}, context [{start}-{end}]:")
            for fr in range(start, end):
                spd = float(speed[fr])
                tls = int(tl_state[fr])
                tl_name = {0: "None", 1: "Green", 2: "Yellow", 3: "Red"}[tls]
                tgt_spd = float(h["target_speed"][fr]) if "target_speed" in h else -1
                mode = int(h["driving_mode"][fr]) if "driving_mode" in h else -1
                mode_names = {
                    1: "FOLLOW_LANE",
                    2: "APPROACHING_TURN",
                    3: "TURNING",
                    4: "TRAFFIC_LIGHT_STOP",
                    5: "VEHICLE_FOLLOWING",
                    6: "PEDESTRIAN_STOP",
                    7: "EMERGENCY_STOP",
                }
                mn = mode_names.get(mode, str(mode))
                marker = " <-- RED+MOVING" if tls == 3 and spd > 1.0 else ""
                print(
                    f"    F{fr:4d}: spd={spd:5.2f} tgt={tgt_spd:5.2f} "
                    f"TL={tl_name:6s} mode={mn}{marker}"
                )

print("\n")
print("=" * 70)
print("PART 2: SAT CORRIDOR FAILURE DETAIL")
print("=" * 70)

for f in files:
    ep_name = f.split("/")[-1].replace(".h5", "")
    with h5py.File(f, "r") as h:
        n = h["velocity"].shape[0]

        if "nearby_vehicles" not in h:
            continue

        nv_data = h["nearby_vehicles"]

        # Find example frames with SAT misses
        examples_found = 0
        for frame_idx in range(0, n, 5):
            if examples_found >= 3:
                break
            try:
                raw = nv_data[frame_idx]
                if isinstance(raw, bytes):
                    raw = raw.decode()
                vehicles = json.loads(raw)
                for v in vehicles:
                    if (
                        v.get("is_ahead", False)
                        and v.get("same_direction", True)
                        and not v.get("obb_in_lane", False)
                        and abs(v.get("lateral", 999)) < 2.5
                        and v.get("distance", 999) < 25
                    ):

                        pos = h["position"][frame_idx]
                        rot = h["rotation"][frame_idx]
                        print(f"\n  {ep_name} Frame {frame_idx}:")
                        print(f"    Ego pos=({pos[0]:.1f}, {pos[1]:.1f}), yaw={rot[1]:.1f}")
                        print(
                            f"    Vehicle: dist={v['distance']:.1f}m, "
                            f"long={v['longitudinal']:.1f}m, "
                            f"lat={v['lateral']:.1f}m"
                        )
                        print(
                            f"    SAT={v['obb_in_lane']}, "
                            f"ahead={v['is_ahead']}, "
                            f"same_dir={v['same_direction']}"
                        )
                        if "speed_kmh" in v:
                            print(f"    Vehicle speed: {v['speed_kmh']:.1f} km/h")
                        examples_found += 1
                        break
            except Exception:
                pass

        if examples_found == 0:
            # Check if there are ANY obb_in_lane=True at all
            sat_true_count = 0
            for frame_idx in range(0, n, 20):
                try:
                    raw = nv_data[frame_idx]
                    if isinstance(raw, bytes):
                        raw = raw.decode()
                    vehicles = json.loads(raw)
                    for v in vehicles:
                        if v.get("obb_in_lane", False):
                            sat_true_count += 1
                except Exception:
                    pass
            print(
                f"\n  {ep_name}: No SAT misses with lat<2.5m found. "
                f"SAT=True count (sampled): {sat_true_count}"
            )

print("\nDone.")
