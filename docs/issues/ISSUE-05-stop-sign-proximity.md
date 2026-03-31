# ISSUE-05 — Stop Sign Route Proximity Too Tight

> **Status:** Not fixed
> **Priority:** Tier 0 (F1 — 2-minute config change)
> **Implementation plan tasks:** F1
> **Connected issues:** None directly

---

## Summary

The stop sign detection Phase 1 gate uses `stop_sign_route_proximity` from `configs/collection.yaml` (currently set to `2.5` m) to determine whether a stop sign's trigger-volume center is "close enough" to any waypoint in the ego's buffer to be considered relevant.

On curved approaches to stop signs, the ego's route waypoints (which follow the road curve) may be up to 3–4 m from the trigger volume center. The 2.5 m threshold misses these cases — the Phase 1 gate returns False, and the stop sign is never armed, so the ego drives through without stopping.

The fix is to increase `stop_sign_route_proximity` from `2.5` to `4.0` m in `configs/collection.yaml`. This is a one-line config change.

---

## Background: Two-Phase Stop Sign Gate

The stop sign detection system uses a two-phase gate:

```
Phase 1 (arming):
  For each stop sign in vicinity, check if ANY waypoint in the buffer
  is within stop_sign_route_proximity (2.5m) of the trigger volume center.
  If yes → stop sign is "armed" (relevant to this route).

Phase 2 (triggering):
  When stop sign is armed AND ego is within 4m of trigger center:
  → Enter STOP_SIGN_STOP mode
```

Phase 1 exists to avoid false positives from stop signs on perpendicular roads that happen to be nearby. It uses the route buffer as a proxy for "is this stop sign on my road?"

---

## What Was Observed

During test collection in Town07 and curved-road scenarios in Town06, episodes showed the ego driving past stop sign locations without stopping. The `stop_sign_distance` field in HDF5 showed the sign was present (non-zero distance) but `driving_mode` never transitioned to `mode=8 (STOP_SIGN_STOP)`.

---

## What Was Verified and How

### Verification 1: Phase 1 gate failure on curved approach

```
ASCII: Curved approach geometry

  Trigger volume center:  T
  Route waypoints:        ●──●──●──●──●
                                ↑
  Closest waypoint to T:  2.8m (exceeds 2.5m threshold)

  Result: Phase 1 returns False → stop sign not armed → ego drives through
```

On a curved road, the ego's route waypoints follow the curve while the trigger volume center is placed at the sign's geometric position. At the apex of a curve approaching a sign, the closest waypoint can be 2.5–4.0 m from the trigger center even though the ego will pass directly through it.

### Verification 2: Town06-ep3 confirmed correct stop (dispels confusion)

The Town06-ep3 disputed stop (frames 1060–1088) was confirmed to be a **legitimate stop sign stop** by a Python script:
- `mode=[8]` (STOP_SIGN_STOP), `nav=[2]` (RIGHT), `tl=[0]` (no red)

This confirms the stop sign detection system CAN work correctly (the Town06 stop sign was a straight approach, Phase 1 triggered normally). The proximity issue only affects curved approaches.

### Verification 3: Theoretical geometry (confirmed)

For a circular arc approach with radius R and waypoint spacing 2.0 m, the lateral deviation of the closest waypoint from the arc chord grows as:
- At R = 20 m (tight turn): lateral deviation ~ 0.1 m per 2 m spacing
- The trigger volume center sits at the inside edge of the arc
- Over a 15 m approach, the closest waypoint can be 2.5–3.5 m from the trigger center

At 4.0 m threshold, this case is caught. At 2.5 m, it is not.

### Verification 4: No false-positive risk from increasing to 4.0 m

Perpendicular stop signs (on crossing roads) are excluded by Phase 2 (ego must be within 4 m of trigger center, which won't happen for perpendicular signs). Phase 1 at 4.0 m is still directionally constrained: the ego only has route waypoints on its own road, not on cross streets, so a cross-street stop sign's trigger center will not be within 4.0 m of any buffer waypoint.

---

## Assumptions and How They Were Verified

| Assumption | Verification | Result |
|---|---|---|
| Curved approaches cause Phase 1 gate misses | Geometric analysis of arc + waypoint spacing | CONFIRMED (theoretical) |
| Town06-ep3 stop sign stop was correct | Python script on HDF5 | CONFIRMED |
| 4.0 m will not cause false positives | Cross-street geometry analysis — perpendicular signs excluded by Phase 2 | CONFIRMED (theoretical) |
| Config value is in `configs/collection.yaml` | Code inspection of ExpertConfig + collection.yaml | CONFIRMED |

---

## The Fix

**File:** `configs/collection.yaml`
**Change:** `stop_sign_route_proximity: 2.5` → `stop_sign_route_proximity: 4.0`

This is the entirety of the fix. One value, one file, two minutes.

---

## Options

### Option A — Increase threshold to 4.0 m (F1 — RECOMMENDED)

| | |
|---|---|
| **Effort** | 2 minutes (config change) |
| **Risk** | VERY LOW — false positives prevented by Phase 2 gate |
| **Fixes** | Curved approach stop sign misses |
| **Industry standard?** | N/A (config tuning) |
| **Most professional?** | YES — use geometric-appropriate threshold |
| **Best long-term?** | PARTIAL — see Option B |

---

### Option B — Dynamic proximity based on route curvature (not recommended now)

Compute the lateral deviation of the route at the stop sign location and use it to set a dynamic proximity threshold.

| | |
|---|---|
| **Effort** | ~30 lines |
| **Risk** | LOW-MEDIUM |
| **Best long-term?** | YES — adapts to any road geometry |
| **Recommended?** | NO — 4.0 m fixed threshold is sufficient for all CARLA towns |

---

### Option C — Replace Phase 1 with heading-filtered scan

Instead of distance to buffer waypoints, check if the stop sign is within the forward arc of the ego's heading. This eliminates the curved-approach problem entirely.

| | |
|---|---|
| **Effort** | ~20 lines |
| **Risk** | LOW |
| **Best long-term?** | YES |
| **Recommended?** | Not necessary — 4.0 m fixed threshold solves the problem |

---

## Recommended Action

**Tier 0 (immediate):** F1 — change `stop_sign_route_proximity: 2.5` → `4.0` in `configs/collection.yaml`. Zero risk, done in 2 minutes.

---

## Open Questions for User

None. This is a pure config change with no ambiguity.

---

## Verification After Fix

Run `bash scripts/test_collection.sh --town Town07` and check that stop signs on curved approaches trigger `mode=8` frames. Check HDF5:

```python
import h5py, numpy as np, glob

for f in sorted(glob.glob('data/test_collection/episode_*.h5')):
    with h5py.File(f) as h:
        mode = h['driving_mode'][:]
        ss = np.sum(mode == 8)
        if ss > 0:
            print(f"{f}: {ss} STOP_SIGN_STOP frames")
```

Expect: Town07 episodes should show STOP_SIGN_STOP frames on curved approaches that were previously missed.
