"""Analyze latest test_collection episodes for red light and SAT issues."""

import glob
import json

import h5py
import numpy as np

files = sorted(glob.glob("data/test_collection/episode_*.h5"))

print("=" * 70)
print("PART 1: RED LIGHT DETAIL")
print("=" * 70)

for f in files:
    ep_name = f.split("/")[-1].replace(".h5", "")
    with h5py.File(f, "r") as h:
        vel = h["velocity"][:]
        n = vel.shape[0]
        speed = np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2 + vel[:, 2] ** 2) * 3.6  # km/h
        tl_state = h["traffic_light_state"][:]
        driving_mode = h["driving_mode"][:]
        target_spd = h["target_speed"][:]

        red_frames = np.where(tl_state == 3)[0]
        if len(red_frames) == 0:
            print(f"\n--- {ep_name}: NO red lights ---")
            continue

        print(f"\n--- {ep_name} ({n} frames) ---")

        # Red + still moving
        red_moving = np.where((tl_state == 3) & (speed > 1.0))[0]
        # Red + driving_mode is TL_STOP (4)
        red_tl_mode = np.where((tl_state == 3) & (driving_mode == 4))[0]
        # Red + NOT in TL_STOP mode (=expert doesn't know)
        red_not_tl = np.where((tl_state == 3) & (driving_mode != 4))[0]

        print(f"  Red frames: {len(red_frames)}")
        print(f"  Red + moving (>1 km/h): {len(red_moving)}")
        print(f"  Red + mode=TL_STOP: {len(red_tl_mode)}")
        print(f"  Red + mode!=TL_STOP: {len(red_not_tl)} (expert missed it)")

        if len(red_not_tl) > 0:
            # What modes was the expert in when it missed reds?
            modes_during_miss = driving_mode[red_not_tl]
            mode_names = {
                1: "FOLLOW_LANE",
                2: "APPROACH_TURN",
                3: "TURNING",
                4: "TL_STOP",
                5: "VEH_FOLLOW",
                6: "PED_STOP",
                7: "EMERG_STOP",
            }
            unique, counts = np.unique(modes_during_miss, return_counts=True)
            print("  Modes when red was missed:")
            for u, c in zip(unique, counts):
                print(f"    {mode_names.get(int(u), str(u))}: {c}")

        if len(red_moving) > 0:
            # Show first cluster
            first = red_moving[0]
            start = max(first - 15, 0)
            end = min(first + 30, n)
            print(f"\n  First red-running at frame {first}:")
            for fr in range(start, end):
                spd = float(speed[fr])
                tls = int(tl_state[fr])
                tl_name = {0: "None", 1: "Grn", 2: "Yel", 3: "Red"}[tls]
                mode = int(driving_mode[fr])
                mn = mode_names.get(mode, str(mode))
                tgt = float(target_spd[fr])
                marker = " <<<" if tls == 3 and spd > 1.0 else ""
                print(
                    f"    F{fr:4d}: spd={spd:5.1f} tgt={tgt:5.1f} TL={tl_name:4s} mode={mn}{marker}"
                )

print()
print("=" * 70)
print("PART 2: SAT FAILURE EXAMPLES")
print("=" * 70)

for f in files:
    ep_name = f.split("/")[-1].replace(".h5", "")
    with h5py.File(f, "r") as h:
        n = h["velocity"].shape[0]
        if "nearby_vehicles" not in h:
            continue
        nv_data = h["nearby_vehicles"]
        examples = 0
        for frame_idx in range(0, n, 5):
            if examples >= 3:
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
                        print(f"\n  {ep_name} F{frame_idx}:")
                        print(f"    Ego: pos=({pos[0]:.1f},{pos[1]:.1f}) yaw={rot[1]:.1f}")
                        print(
                            f"    Veh: d={v['distance']:.1f}m "
                            f"long={v['longitudinal']:.1f}m "
                            f"lat={v['lateral']:.1f}m "
                            f"SAT={v['obb_in_lane']} "
                            f"spd={v.get('speed_kmh', '?')}"
                        )
                        examples += 1
                        break
            except Exception:
                pass

        # Count total
        sat_t = 0
        sat_f_inlane = 0
        total = 0
        for fr in range(0, n, 10):
            try:
                raw = nv_data[fr]
                if isinstance(raw, bytes):
                    raw = raw.decode()
                vs = json.loads(raw)
                for v in vs:
                    if v.get("is_ahead") and v.get("same_direction", True):
                        total += 1
                        if v.get("obb_in_lane"):
                            sat_t += 1
                        elif abs(v.get("lateral", 999)) < 2.5:
                            sat_f_inlane += 1
            except Exception:
                pass
        print(
            f"  {ep_name} totals: ahead_same={total} "
            f"sat_true={sat_t} sat_false_inlane={sat_f_inlane}"
        )

print("\nDone.")
