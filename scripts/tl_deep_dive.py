"""Deep-dive on Red+FOLLOW_LANE traffic light pattern."""

import h5py
import numpy as np

f = "data/test_collection/episode_1a618295.h5"
with h5py.File(f, "r") as h:
    vel = h["velocity"][:]
    n = vel.shape[0]
    speed = np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2 + vel[:, 2] ** 2) * 3.6
    tl_state = h["traffic_light_state"][:]
    dm = h["driving_mode"][:]
    pos = h["position"][:]

    red_fl = np.where((tl_state == 3) & (dm == 1))[0]
    print(f"Red+FOLLOW_LANE: {len(red_fl)} frames")

    if len(red_fl) > 0:
        gaps = np.diff(red_fl)
        starts = [red_fl[0]]
        ends = []
        for i, g in enumerate(gaps):
            if g > 2:
                ends.append(red_fl[i])
                starts.append(red_fl[i + 1])
        ends.append(red_fl[-1])

        print(f"\n{len(starts)} clusters:")
        for cs, ce in zip(starts, ends):
            sz = ce - cs + 1
            ps, pe = pos[cs], pos[ce]
            print(
                f"  F{cs}-{ce} ({sz}f): "
                f"({ps[0]:.0f},{ps[1]:.0f})->({pe[0]:.0f},{pe[1]:.0f}) "
                f"spd={speed[cs]:.0f}->{speed[ce]:.0f} km/h"
            )
            if cs > 0:
                print(f"    Before: TL={int(tl_state[cs-1])} mode={int(dm[cs-1])}")
            if ce < n - 1:
                print(f"    After:  TL={int(tl_state[ce+1])} mode={int(dm[ce+1])}")

    # All red blocks
    red_frames = np.where(tl_state == 3)[0]
    gaps = np.diff(red_frames)
    bs_ = [red_frames[0]]
    be_ = []
    for i, g in enumerate(gaps):
        if g > 2:
            be_.append(red_frames[i])
            bs_.append(red_frames[i + 1])
    be_.append(red_frames[-1])

    mn = {1: "FL", 2: "AT", 3: "TN", 4: "TL", 5: "VF", 6: "PS", 7: "ES"}
    print("\nAll Red blocks:")
    for s, e in zip(bs_, be_):
        modes = dm[s : e + 1]
        um, cm = np.unique(modes, return_counts=True)
        ms = " ".join([f"{mn.get(int(m),str(m))}={c}" for m, c in zip(um, cm)])
        print(f"  F{s}-{e} ({e-s+1}f): {ms}")
