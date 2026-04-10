# ISSUE-02 — Traffic Light State Machine Failures

> **Status:** Partially fixed
> **Priority:** Tier 1 (VEH_FOLLOW fixes), Tier 2 (TL stop-line stale reference)
> **Implementation plan tasks:** F8, F9, F9b, F13
> **Connected issues:** [ISSUE-01](ISSUE-01-junction-waypoints.md) (bad waypoints during turns corrupt TL detection — F13 is partly caused by ISSUE-01)

---

## Update (2026-04-10)

The original forensic sections below are kept for historical traceability.
Current code status is:

- **DONE:** F8 (turning stopped-lead lateral gate)
- **DONE:** F9 (red/yellow TL stop intent outranks vehicle-follow targeting)
- **DONE:** F13 (`_tl_stop_loc` refresh after buffer/gap-fill updates)
- **PENDING:** F9b (clear `_tl_proceed_committed` after full stop)

This means Bug A/B are substantially mitigated in code, while the
yellow-commit reset edge case remains open.

---

## Summary

Three distinct bugs in the traffic light detection and driving mode state machine were confirmed during the test collection run. They are grouped here because they all involve the TL detection system or the VEHICLE_FOLLOWING mode interacting incorrectly with traffic lights.

An additional follow-up bug (F9b) was later identified: committed-yellow
state should be cleared if ego fully stops before the stop line.

### Bug A — TL Stop-Line Position Goes Stale During Turns (Town05 partial crossing)
When gap fill fires and refills the waypoint buffer with junction arc waypoints, the `_tl_stop_loc` (the position of the stop line, stored during APPROACHING_TURN) is not re-evaluated. The TL distance calculation uses the stale reference, falls back to the TL pole distance (~79.9 m), and loses the "am I behind the stop line?" check. Result: the ego enters the junction before fully stopping, then detects the red light again partway through the crossing.

### Bug B — VEH_FOLLOW Follows Lead Vehicle Through Red Light
When the ego is in VEHICLE_FOLLOWING mode and the lead vehicle runs a red light, the ego follows. The TL state check is not entered during VEHICLE_FOLLOWING mode — the ego simply matches the lead vehicle's speed. The TL memory may return None (cleared) or GREEN (if the TL pole is now behind ego), allowing the ego to cross.

### Bug C — VEH_FOLLOW Deadlock When Light Turns Green but Lead Doesn't Move
When a traffic light turns GREEN but the lead vehicle remains stationary (parked car, NPC spawn glitch, or the lead is also stuck), the ego stays in VEHICLE_FOLLOWING mode with target_speed=0. The TL state in memory is cleared (GREEN → no active stop), so TRAFFIC_LIGHT_STOP mode is no longer triggered. Nothing breaks the VEH_FOLLOW lock. The episode times out.

### Bug D — Yellow-Commit Flag Not Cleared on Full Stop (F9b)
If `_tl_proceed_committed` is set during yellow dilemma handling, an
intervening full stop before the stop line should clear that commitment.
Current code clears the flag on green or stale TL-memory clears, but does
not explicitly clear it on full stop. This can preserve stale proceed
intent across stop/resume transitions.

---

## What Was Observed

**Bug A (Town05 ep1 frames 3000–3200):**
- Frame range appeared to show ego crossing a red light going "straight"
- Whether this was a violation is disputed
- The actual behavior was mixed: partial crossing occurred before ego stopped

**Bug B (multiple episodes):**
- Lead vehicle accelerates through a red light
- Ego in VEH_FOLLOW mode matches lead vehicle's speed
- TL state in HDF5 shows `tl=3` (RED) while `speed > 1 km/h`
- Pattern appeared in 5+ episodes across the test collection

**Bug C (Town03 ep2 and ep3):**
- Episodes timed out without completing their routes
- Review of final frames showed ego stationary with mode=VEH_FOLLOW
- TL state had transitioned to GREEN (or None) but lead vehicle was not moving
- Ego had no mechanism to break out of VEH_FOLLOW

---

## What Was Verified and How

### Bug A — Verification

**A Python script was run** against `episode_6bb50371.h5` (Town05 ep1):

```python
with h5py.File('data/test_collection/episode_6bb50371.h5', 'r') as f:
    nav = f['nav_command'][3000:3200]
    tl = f['traffic_light_state'][3000:3200]
    speed = np.linalg.norm(f['velocity'][3000:3200], axis=1) * 3.6
    mode = f['driving_mode'][3000:3200]
```

Script output:
```
Frames 3000–3054:  tl=3 (RED)  speed=2.1–4.8 km/h  mode=3 (TURNING)    ← VIOLATION
Frames 3055–3139:  tl=3 (RED)  speed≈0             mode=4 (TL_STOP)    ← correct stop
Frames 3140–3200:  tl=1 (GREEN) speed rising        mode=1 (FOLLOW_LANE) ← correct resume
```

This is **MIXED** behavior: 55 frames of movement through red light (Bug A), followed by detection and correct stopping, followed by correct green-light resumption.

**Direction:** Nav = 3 (STRAIGHT), not 1 (LEFT). The original observation of "going left on red" was wrong on both the direction and the stop behavior.

**Mechanism:** The 55-frame violation is the TL detection lag during the TURNING → TL_STOP transition. The `_tl_stop_loc` reference was stale after gap fill refilled the buffer.

### Bug B — Verification

Searched HDF5 frames across all 18 test episodes for the pattern `tl=3 AND speed > 1 km/h` outside of frames where mode=3 (TURNING). Found 5+ episode segments where this pattern appeared with mode=5 (VEHICLE_FOLLOWING). In all cases, the pattern was preceded by a lead vehicle that had a non-zero speed in the prior frame window.

### Bug C — Verification

Town03 ep2 and ep3 were the only two episodes that timed out (did not reach route completion). Examined final 100 frames:

```
ep2 final frames: mode=[5,5,5,...,5], target_speed=0, tl=[1,1,1,...] (GREEN), speed=0
ep3 final frames: mode=[5,5,5,...,5], target_speed=0, tl=[0,0,0,...] (NONE), speed=0
```

Both: `mode=5 (VEH_FOLLOW)`, `target_speed=0`, `speed=0`, episode ends at timeout (75 s).

---

## Assumptions and How They Were Verified

| Assumption | Verification | Result |
|---|---|---|
| Bug A is caused by stale `_tl_stop_loc` | TL distance during TURNING mode showed 79.9 m (pole at scan boundary, not stop line) | CONFIRMED — TL distance ~79.9 m is the pole-fallback signature |
| Bug B pattern is VEH_FOLLOW + lead-runs-red | HDF5 `tl=3 + speed>1 + mode=5` across multiple episodes | CONFIRMED in 5+ episodes |
| Bug C caused Town03 timeouts | Final frame data for ep2 and ep3 | CONFIRMED — mode=5, target=0 at timeout |
| TL check is bypassed in VEH_FOLLOW | Code inspection: `_compute_target_speed()` in VEH_FOLLOW branch does not call TL check | CONFIRMED |
| `_tl_stop_loc` is stored pre-gap-fill | Code inspection: stored during APPROACHING_TURN, gap fill fires after mode transition | CONFIRMED |

---

## Connected Issues

- **[ISSUE-01](ISSUE-01-junction-waypoints.md):** Bug A (stale TL stop-line) is a downstream consequence of gap fill divergence. When gap fill refills the buffer with junction arc waypoints, `_tl_stop_loc` (computed before gap fill) no longer references the correct position. F13 (Bug A fix) is partially redundant if F14 (Bezier arcs) is implemented first, because Bezier arcs would provide correct waypoints that the TL distance calculation can use. However, F13 should still be implemented to make the "behind stop-line" check robust regardless of how the buffer was filled.

---

## Options

### Bug A — Options for Stale TL Stop-Line (F13)

#### Option A1 — Re-evaluate `_tl_stop_loc` when buffer changes (recommended)

After gap fill runs and the waypoint buffer changes, re-run the stop-line position lookup and update `_tl_stop_loc`. The check would find the stop-line waypoint closest to the new buffer waypoints.

| | |
|---|---|
| **Effort** | ~20 lines |
| **Risk** | MEDIUM — TL detection state machine is complex; needs testing across all TL towns |
| **Fixes** | Bug A directly |
| **Industry standard?** | YES — stop-line re-evaluation is standard in professional planners |
| **Most professional?** | YES |
| **Best long-term?** | YES |

**Pros:**
- Correct mechanism — updates stale reference at the right time
- Works independently of whether Bezier arcs are implemented

**Cons:**
- Need to handle edge cases: what if no stop-line found after gap fill?
- TL state machine is non-trivial; wrong placement of re-evaluation can cause oscillation

#### Option A2 — Use ego position relative to stop-line tile (alternative)

Instead of tracking a stored waypoint position, check whether the ego's current map tile is "past" the stop-line tile using `carla.Map.get_waypoint()` and road segment comparison.

| | |
|---|---|
| **Effort** | ~30 lines |
| **Risk** | MEDIUM |
| **Fixes** | Bug A + all future stale-reference variants |
| **Industry standard?** | YES |
| **Most professional?** | YES |
| **Best long-term?** | BETTER — more robust to buffer changes |

**Pros:**
- No stored reference that can go stale
- Works correctly even after buffer refills

**Cons:**
- Requires map API call per frame (small overhead)
- "Past stop-line tile" definition depends on CARLA road topology

---

### Bug B — Options for VEH_FOLLOW + Red Light (F9)

#### Option B1 — TL check inside VEH_FOLLOW mode (recommended)

Before computing target speed in VEHICLE_FOLLOWING mode, check the persistent TL memory. If RED, override to TRAFFIC_LIGHT_STOP regardless of lead vehicle speed.

```python
# In _compute_driving_mode() or equivalent:
if current_mode == DrivingMode.VEHICLE_FOLLOWING:
    if self._tl_state == carla.TrafficLightState.Red:
        return DrivingMode.TRAFFIC_LIGHT_STOP
```

| | |
|---|---|
| **Effort** | ~10 lines |
| **Risk** | LOW — conservative addition; ego stops more often |
| **Fixes** | Bug B confirmed violation pattern |
| **Industry standard?** | YES |
| **Most professional?** | YES |
| **Best long-term?** | YES |

**Pros:**
- Simple, safe (errs on side of stopping)
- Cannot cause the ego to run a red

**Cons:**
- If TL state memory has spurious RED readings, ego stops unnecessarily behind lead vehicle

#### Option B2 — Explicit lead-vehicle-ran-red detection

Detect whether the lead vehicle crossed the stop line while TL was RED. If so, do NOT follow.

| | |
|---|---|
| **Effort** | ~40 lines |
| **Risk** | MEDIUM — complex lead vehicle state tracking |
| **Fixes** | Bug B exactly |
| **Industry standard?** | NO — overly specific |
| **Best long-term?** | NO — Option B1 is simpler and sufficient |

**Recommendation: Option B1.**

---

### Bug C — Options for VEH_FOLLOW Deadlock (F8)

#### Option C1 — N-frame escape with hysteresis counter (recommended)

If mode=VEHICLE_FOLLOWING AND target_speed=0 AND TL state is GREEN (or None) for N consecutive frames (e.g., N=60 = 3 seconds), break out of VEHICLE_FOLLOWING to FOLLOW_LANE.

```python
# In mode update loop:
if (mode == VEH_FOLLOW and target_speed < 0.1
        and tl_state != RED and self._veh_follow_stall_counter > 60):
    mode = FOLLOW_LANE
    self._veh_follow_stall_counter = 0
else:
    if mode == VEH_FOLLOW and target_speed < 0.1:
        self._veh_follow_stall_counter += 1
    else:
        self._veh_follow_stall_counter = 0
```

| | |
|---|---|
| **Effort** | ~15 lines |
| **Risk** | LOW-MEDIUM — if N is too small, ego changes lanes too aggressively behind slow vehicles |
| **Fixes** | Bug C confirmed Town03 timeouts |
| **Industry standard?** | YES — timeout escape is standard in imitation learning experts |
| **Most professional?** | YES |
| **Best long-term?** | YES |

**Pros:**
- Handles all deadlock variants (parked car, NPC freeze, signal-after-green)
- Hysteresis prevents flapping on slow-moving leads

**Cons:**
- N must be tuned. Too small → ego overtakes slow-moving vehicles. Too large → ego still times out in short episodes.
- Suggested N = 60 frames = 3 seconds at 20 Hz

#### Option C2 — Check if lead vehicle is parked (has `is_static_actor` flag)

| | |
|---|---|
| **Effort** | ~10 lines |
| **Risk** | LOW |
| **Fixes** | Subset of Bug C only (parked cars) |
| **Best long-term?** | NO — doesn't handle NPC freeze, only parked actors |

**Recommendation: Option C1 (hysteresis counter).** Option C2 can be added as a complement.

---

## Verification Plan After Fixes

After implementing F8, F9, F13:

1. Run `bash scripts/test_collection.sh` with `--town Town03 Town05`
2. For Bug A fix (F13): Check Town05 episodes for `tl=3 + speed>1km/h + mode=TURNING`. Should drop to 0 frames.
3. For Bug B fix (F9): Check that `tl=3 + speed>1km/h + mode=VEH_FOLLOW` no longer appears.
4. For Bug C fix (F8): Check that Town03 episodes complete without timeout.

Script to verify:
```python
import h5py, glob, numpy as np

for f in sorted(glob.glob('data/test_collection/episode_*.h5')):
    with h5py.File(f) as h:
        tl = h['traffic_light_state'][:]
        spd = np.linalg.norm(h['velocity'][:], axis=1) * 3.6
        mode = h['driving_mode'][:]

        # Bug A/B: moving through red
        red_moving = np.where((tl == 3) & (spd > 1.0))[0]
        if len(red_moving):
            print(f"{f}: {len(red_moving)} frames red+moving at {red_moving[:5]}")

        # Bug C: VEH_FOLLOW stall
        vf_stall = np.where((mode == 5) & (spd < 0.1))[0]
        if len(vf_stall) > 60:
            print(f"{f}: {len(vf_stall)} VEH_FOLLOW stall frames")
```

---

## Open Questions for User

None — the remaining work is scoped. Implement F9b next, then validate
with a focused replay covering yellow-commit interrupted by full stop.
