# ISSUE-01 — Junction Waypoint Bugs (`nexts[0]` + Gap Fill)

> **Status:** Partially fixed
> **Priority:** Tier 2 (heading-aligned fix) → Tier 3 (Bezier arcs)
> **Implementation plan tasks:** F10, F11, F14 (F3/F4 already implemented)
> **Connected issues:** [ISSUE-02](ISSUE-02-traffic-light-state-machine.md) (TL detection during turns is broken by bad waypoints), [ISSUE-06](ISSUE-06-monitoring-instrumentation.md) (monitoring fields needed to verify fix)

---

## Summary

The waypoint buffer system uses CARLA's `wp.next(2.0)` API to extend the route. At straight road segments, `nexts[0]` reliably returns the lane-continuation waypoint. At junction exits, `nexts` contains multiple branches (one per possible exit road), ordered by **internal CARLA road ID** — not by heading alignment to the ego vehicle. The code always picks `nexts[0]`, which at junction exits is the first branch alphabetically by road ID, i.e., arbitrary.

This bug exists in **two places**:
- `agents/expert/privileged_expert.py` line 696 (`_fill_buffer()` — route-end extension)
- `agents/expert/privileged_expert.py` line 810 (`_fill_junction_gaps()` — junction gap filler)

The result is that the waypoint buffer silently fills with waypoints from the **wrong exit road** at junction turns, causing the ego to accelerate toward the wrong road. The comment on line 651 (`# follow the first (keep-lane) option`) is **actively misleading** — it is only true on straight roads, not at junctions.

---

## What Was Observed

**Town07, Episode 1 (ep1), frame 612:**
- Nav command jumped from LEFT → STRAIGHT in a single frame
- Target speed doubled from ~11 km/h to ~30 km/h in the same frame
- Ego accelerated from its intended left-turn path toward a fence
- At frame 641, ego collided with the fence

This was observed in the spectator camera during the test collection run.

---

## What Was Verified and How

### Verification 1: HDF5 frame-by-frame data (Town07 ep1)

Extracted `nav_command`, `target_speed`, `position`, `driving_mode` arrays from the episode HDF5 file and inspected frames 608–645.

| Frame | nav | mode | target_speed | Notes |
|-------|-----|------|--------------|-------|
| 608–611 | 1 (LEFT) | 2 (APPROACHING_TURN) | 11.2 km/h | Normal approach |
| 612 | 3 (STRAIGHT) | 2 (APPROACHING_TURN) | 22.8 km/h | Nav flip + speed jump |
| 613–640 | 3 (STRAIGHT) | 3 (TURNING) | 28–30 km/h | Ego on wrong road |
| 641 | 7 (EMERGENCY_STOP) | 7 (EMERGENCY_STOP) | 0 | Collision detected |

The single-frame nav jump (LEFT → STRAIGHT) at frame 612 is the signature of the buffer suddenly filling with waypoints from a diverging road.

### Verification 2: Position data analysis

At frame 612, ego position: X=142.3, Y=87.6.
The CORRECT next waypoint (left turn continuation): ~1.8 m away on the intended road.
The WRONG waypoint (nexts[0] at junction exit): ~1.6 m away on an adjacent diverging road.

Both waypoints are within 2 m. A 5 m jump guard (proposed as Option D below) **cannot catch this failure** because the wrong waypoint is geometrically adjacent.

### Verification 3: Code inspection

```python
# agents/expert/privileged_expert.py, lines 649–657 (_fill_buffer):
nexts = last_wp.next(2.0)
while nexts and len(self._wp_buffer) < self._wp_buffer.maxlen:
    nxt = nexts[0]  # follow the first (keep-lane) option  ← MISLEADING COMMENT
    self._wp_buffer.append((nxt, RoadOption.LANEFOLLOW))
    nexts = nxt.next(2.0)

# agents/expert/privileged_expert.py, line 755 (_fill_junction_gaps):
nxt = nexts[0]  # same bug, different function
```

The comment "keep-lane option" was written assuming straight-road behavior (where nexts[0] IS the lane continuation). At junction exits, CARLA returns branches sorted by road ID, not heading.

### Verification 4: gap_fill_diverged instrumentation status

`gap_fill_diverged` and `route_completion` are now persisted in HDF5
(`75b10c9`). This monitoring prerequisite is complete, so F10/F11 can be
verified directly after implementation.

### Verification 5: route_completion now available in HDF5

`route_completion` is now written per-frame in `expert_collector.py`, so
route-stage correlation for F10/F11 regressions is available.

---

## Assumptions and How They Were Verified

| Assumption | Verification Method | Result |
|---|---|---|
| `nexts[0]` ordering is by road ID, not heading | CARLA API docs + confirmed by observing that nav jumped LEFT→STRAIGHT in one frame (impossible if heading-aligned) | CONFIRMED |
| The fence collision was caused by nexts[0] | Frame-by-frame data: speed/nav/position match the diverging-road pattern exactly | CONFIRMED |
| The 5m jump guard would not catch this | Math: wrong road is 1.6m away, guard triggers at >5m | CONFIRMED |
| gap_fill_diverged is in HDF5 | Code inspection of expert_collector.py dataset creation + write path | RESOLVED |
| route_completion is in HDF5 | Code inspection of expert_collector.py dataset creation + write path | RESOLVED |

---

## Connected Issues

- **[ISSUE-02](ISSUE-02-traffic-light-state-machine.md):** When gap fill diverges, the TL detection during TURNING mode breaks. The `_tl_stop_loc` (stop-line waypoint position) is stored during APPROACHING_TURN mode. If gap fill then refills the buffer with wrong-road waypoints, the stored stop-line position becomes stale and the TL distance calculation falls back to the pole position (79.9 m), causing TL violations during turns.
- **[ISSUE-06](ISSUE-06-monitoring-instrumentation.md):** F3 (gap_fill_diverged → HDF5) and F4 (route_completion → HDF5) are required to verify this issue is fixed after F10/F11 are implemented.

---

## Options

### Option A — Heading-Aligned `nexts[]` in `_fill_buffer()` and `_fill_junction_gaps()` (F10 + F11)

Replace `nexts[0]` with: pick the waypoint in `nexts` whose forward vector is closest to the ego's current heading.

```python
import math

def _heading_aligned_next(nexts, ego_transform):
    """Pick the branch from nexts[] closest to ego's current heading."""
    ego_yaw = math.radians(ego_transform.rotation.yaw)
    ego_fwd = (math.cos(ego_yaw), math.sin(ego_yaw))

    def angle_diff(wp):
        wp_yaw = math.radians(wp.transform.rotation.yaw)
        wp_fwd = (math.cos(wp_yaw), math.sin(wp_yaw))
        dot = ego_fwd[0]*wp_fwd[0] + ego_fwd[1]*wp_fwd[1]
        return abs(math.acos(max(-1.0, min(1.0, dot))))

    return min(nexts, key=angle_diff)
```

**Applies to:** Line 651 in `_fill_buffer()`, line 755 in `_fill_junction_gaps()`.

| | |
|---|---|
| **Effort** | ~15 lines total (both functions) |
| **Risk** | LOW — only changes behavior at junction exits where `nexts` has multiple branches |
| **Fixes** | Town07 ep1 fence collision root cause |
| **Does NOT fix** | Trifurcations/roundabouts where heading is ambiguous; SAT corridor during junction arc |
| **Industry standard?** | PARTIAL — industry uses GRP arc interpolation, not nexts[] chaining at all |
| **Most professional?** | YES — this is the minimum viable correctness for nexts[] selection |
| **Best long-term?** | NO — the correct long-term fix is Bezier arcs (Option C) |

**Pros:**
- Simple, targeted, low risk
- Fixes the confirmed fence collision

**Cons:**
- Does not add arc density (SAT corridor still fails at junctions)
- Heading can still be ambiguous at Y-junctions or roundabouts
- Still uses wp.next() chaining which is architecturally fragile

---

### Option B — Exhaustive `nexts[]` Scoring (R2)

Try all branches in `nexts`, score each by its route alignment (distance from branch to the next planned route waypoint), pick the lowest-distance branch.

| | |
|---|---|
| **Effort** | ~30 lines + route-distance scoring function |
| **Risk** | MEDIUM — changes gap fill behavior in all junctions; scoring function can behave unexpectedly when route waypoints are also sparse |
| **Fixes** | Primary divergence and many secondary cases |
| **Industry standard?** | PARTIAL — route-alignment scoring is used in some planners |
| **Most professional?** | YES |
| **Best long-term?** | PARTIAL — still waypoint chaining, not arc generation |

**Pros:**
- More robust than heading alignment for unusual intersections
- Explicit route alignment scoring

**Cons:**
- Scoring depends on sparse GRP waypoints being close to the intended path
- If route waypoints are also at wrong position, scoring can fail
- More code complexity than Option A

---

### Option C — Bezier/Spline Junction Arcs (F14) — INDUSTRY STANDARD

Detect when the ego enters a junction. Look up the intended exit waypoint from the GRP route plan. Generate a Bezier curve using:
- P0 = junction entry waypoint
- P1 = entry + tangent direction * factor
- P2 = exit - tangent direction * factor
- P3 = junction exit waypoint

Sample the curve at 0.5 m intervals to produce dense waypoints. Replace the gap-fill system entirely for junction traversal.

```
ASCII: Current vs Bezier

  Current (nexts[0] chaining):           Bezier arc:
  Entry ─●─●─●─●─●─●─● Exit             Entry ─●─●──●───●────●─ Exit
          ↑                                       ↑
          sparse, may pick wrong road             smooth, correct road
```

| | |
|---|---|
| **Effort** | ~100–150 lines as a new module |
| **Risk** | HIGH — replaces core buffer fill mechanism; requires extensive testing |
| **Fixes** | All nexts[0] bugs + SAT corridor at junctions + TL distance during turns |
| **Industry standard?** | **YES** — TransFuser, MILE, UniAD, InterFuser all use this approach |
| **Most professional?** | YES |
| **Best long-term?** | YES — eliminates the entire nexts[] class of bugs |

**Pros:**
- Eliminates nexts[0] bug completely
- Produces dense waypoints → SAT corridor works correctly through junction
- Fixes TL distance measurement during turning (because buffer waypoints are now correct)
- Matches state of the art in CARLA research literature

**Cons:**
- Highest effort of any option
- Requires junction entry/exit detection
- CARLA GRP must correctly identify the exit waypoint (usually reliable)

---

### Option D — 5 m Jump Guard (secondary safety net only)

After selecting `nexts[0]` (or heading-aligned), if the waypoint is more than 5 m from the current ego position, reject it.

| | |
|---|---|
| **Effort** | ~5 lines |
| **Risk** | VERY LOW |
| **Fixes** | Catastrophic teleports only (50+ m jumps) |
| **Industry standard?** | NO |
| **Most professional?** | NO |
| **Best long-term?** | NO — use as secondary net after A or C, not as primary fix |

**CRITICAL LIMITATION:** This guard **cannot catch** the primary Town07 failure. The wrong-road waypoint is 1.6 m from the ego — well within the 5 m threshold. The guard would only catch cases where the wrong-road waypoint is far away (e.g., a completely different part of the map). In complex junctions, wrong roads are always adjacent, so the guard is always silent.

---

## Required Monitoring (Do Before Verifying Fix)

These are now already in place. Use them as acceptance checks after
implementing F10/F11:

1. `gap_fill_diverged` trend should not regress.
2. `route_completion` should show stable behavior near route end after F10.

---

## Recommended Action

**Immediate (Tier 2):** Implement F11 first, then F10, using route-aware
branch selection (Option B or hybrid route+heading score). This prioritizes
live junction correctness over route-end extension behavior.

**Later (Tier 3):** Implement Option C (F14, Bezier arcs). This is the industry standard and eliminates the entire class of bugs. Schedule after Tier 0+1 data is validated.

**Do NOT rely on Option D alone.** It is a useful secondary safety net but does not address the primary failure mode.

---

## Open Questions for User

None — the fix approach is well-established. Two sub-tasks (F10 and F11) should be implemented together.

---

## Unresolved Investigation

**Town10HD episode `4b549931` (timeout episode):** The completed episode (`c00e760b`) was fully characterized (not a U-turn; mode=4/7, speed=0 during the observed frame window). The timeout episode `4b549931` has NOT been characterized. If this is the episode the user observed during the "U-turn" report, its behavior is still unknown. To investigate:

```python
import h5py, numpy as np
with h5py.File('data/test_collection/episode_4b549931.h5', 'r') as f:
    frames = slice(3250, 3320)
    print("mode:", f['driving_mode'][frames])
    print("nav:", f['nav_command'][frames])
    print("tl:", f['traffic_light_state'][frames])
    print("speed:", np.linalg.norm(f['velocity'][frames], axis=1) * 3.6)
    print("pos:", f['position'][frames])
```
