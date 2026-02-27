import glob
import json

import h5py
import numpy as np

files = sorted(
    glob.glob(
        "~/projects/neuromorphic-vision-rwkv/data/test_collection/episode_*.h5"
    )
)

for f in files:
    ep_name = f.split("/")[-1]
    with h5py.File(f, "r") as h:
        vel = h["velocity"][:]  # shape (N, 3)
        speed = np.linalg.norm(vel, axis=1) * 3.6  # m/s -> km/h
        n = speed.shape[0]
        tl_state = h["traffic_light_state"][:]

        # Red light running: frames where TL=Red and speed > 1.0 km/h
        red_moving = np.where((tl_state == 3) & (speed > 1.0))[0]
        red_frames = np.where(tl_state == 3)[0]
        yellow_frames = np.where(tl_state == 2)[0]

        print(f"\n{'='*60}")
        print(f"Episode: {ep_name} ({n} frames)")
        print(f"  Red light frames: {len(red_frames)}")
        print(f"  Yellow light frames: {len(yellow_frames)}")
        print(f"  Moving through red (speed>1 km/h): {len(red_moving)}")

        if len(red_moving) > 0:
            # Show clusters of red-light-running
            gaps = np.diff(red_moving)
            cluster_starts = [red_moving[0]]
            for i, g in enumerate(gaps):
                if g > 5:
                    cluster_starts.append(red_moving[i + 1])

            for cs in cluster_starts[:5]:
                end = min(cs + 10, n)
                print(f"\n  Red-running cluster starting frame {cs}:")
                for fr in range(max(cs - 5, 0), min(cs + 20, n)):
                    spd = float(speed[fr])
                    tls = int(tl_state[fr])
                    tl_name = {0: "None", 1: "Green", 2: "Yellow", 3: "Red"}[tls]
                    print(f"    Frame {fr}: speed={spd:.1f} km/h, TL={tl_name}")

        # Vehicle detection stats
        if "nearby_vehicles" in h:
            nv_data = h["nearby_vehicles"]
            sat_true = 0
            sat_false_in_lane = 0
            total_ahead_same = 0

            for frame_idx in range(0, n, 10):  # sample every 10th frame
                try:
                    raw = nv_data[frame_idx]
                    if isinstance(raw, bytes):
                        raw = raw.decode()
                    vehicles = json.loads(raw)
                    for v in vehicles:
                        if v.get("is_ahead", False) and v.get("same_direction", True):
                            total_ahead_same += 1
                            if v.get("obb_in_lane", False):
                                sat_true += 1
                            elif abs(v.get("lateral", 999)) < 2.5:
                                sat_false_in_lane += 1
                except Exception:
                    pass

            print("\n  Vehicle detection (sampled every 10 frames):")
            print(f"    Ahead+SameDir vehicles: {total_ahead_same}")
            print(f"    SAT=True (obb_in_lane): {sat_true}")
            print(f"    SAT=False but lateral<2.5m: {sat_false_in_lane}")
            if total_ahead_same > 0:
                pct = sat_true / total_ahead_same * 100
                print(f"    SAT success rate: {pct:.1f}%")

print("\n\nDone.")
