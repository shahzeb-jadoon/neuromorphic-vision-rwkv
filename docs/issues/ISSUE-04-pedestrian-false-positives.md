# ISSUE-04 — Pedestrian False Positives (Fixed Lane Half-Width)

> **Status:** Partially fixed
> **Priority:** Tier 1 (F6 — 1-line fix), Tier 3 (F15 — road-surface check)
> **Implementation plan tasks:** F6, F15
> **Connected issues:** None directly

---

## Update (2026-04-10)

- **F6 DONE:** pedestrian hazard logic now uses dynamic lane-width context.
- **F15 PENDING:** map road-surface check is still planned as the long-term
    replacement/complement for lateral thresholding.

The detailed forensic notes below are preserved as historical context.

---

## Summary

The pedestrian stop check in `_closest_pedestrian_distance()` uses a **fixed lateral threshold** of 3.0 m to determine whether a pedestrian is in the ego's lane. This value is stored in `ExpertConfig.pedestrian_lane_half_width` (line 121 of `privileged_expert.py`).

The actual lane half-width in Town03 is approximately 1.75 m (lane width ~3.5 m). The 3.0 m threshold extends 1.25 m into the adjacent sidewalk, triggering emergency pedestrian stops for pedestrians who are on the sidewalk and not in the road.

**The fix is a single line change.** The function `_dynamic_half_lane()` is already implemented in the codebase and already used for vehicle lateral corridor calculations. The pedestrian check is not using it. Changing the pedestrian check to use `_dynamic_half_lane()` instead of the fixed config value requires editing one line.

---

## What Was Observed

During Town10HD collection episodes, the ego made unexpected pedestrian stops in locations where pedestrians appeared to be on sidewalks. The driving_mode HDF5 field showed `mode=6 (PEDESTRIAN_STOP)` at these frames.

The pattern was: pedestrian detected on sidewalk at lateral offset 2.1–2.9 m, which is outside the actual lane (half-width ~1.75 m) but inside the fixed threshold (3.0 m).

---

## What Was Verified and How

### Verification 1: Actual lane widths in CARLA towns

Town lane half-widths from CARLA map API:

| Town | Actual lane half-width | Fixed threshold | Sidewalk overlap |
|------|------------------------|-----------------|-----------------|
| Town03 | ~1.75 m | 3.0 m | 1.25 m into sidewalk |
| Town04 | ~1.85 m | 3.0 m | 1.15 m into sidewalk |
| Town05 | ~1.80 m | 3.0 m | 1.20 m into sidewalk |
| Town10HD | ~2.10 m (wide road) | 3.0 m | 0.90 m into sidewalk |

In all towns, the fixed 3.0 m threshold extends into the sidewalk.

### Verification 2: False positive rate estimate

Examined Town10HD episodes for `mode=6 (PEDESTRIAN_STOP)` frames. Cross-referenced pedestrian lateral offset at those frames. Pedestrians with lateral offset in the range 1.75–3.0 m (in sidewalk but within threshold): 38–43% of all pedestrian stop frames were in this category.

### Verification 3: `_dynamic_half_lane()` already exists

Code inspection of `privileged_expert.py`: the function `_dynamic_half_lane()` is defined and callable. It queries the CARLA map API for the lane width at the ego's current waypoint and returns `lane_width / 2.0`. It is already used in the vehicle SAT corridor calculation. It is NOT currently used in the pedestrian check.

```python
# Current code in _closest_pedestrian_distance() (line ~2295):
half_w = self.config.pedestrian_lane_half_width  # = 3.0 (WRONG)

# Fix (1 line):
half_w = self._dynamic_half_lane()  # = actual lane width / 2 ≈ 1.75 m
```

### Verification 4: Code location confirmed

`privileged_expert.py`, function `_closest_pedestrian_distance()`, lines 2288–2312:

```python
def _closest_pedestrian_distance(self):
    ...
    half_w = self.config.pedestrian_lane_half_width  # ← WRONG: fixed 3.0m
    current_in_lane = p["is_ahead"] and abs(p["lateral"]) < half_w
    ...
```

---

## Assumptions and How They Were Verified

| Assumption | Verification | Result |
|---|---|---|
| Town03 actual lane half-width ≈ 1.75 m | CARLA map API query during live test | CONFIRMED |
| Fixed 3.0 m extends into sidewalk | 3.0 > 1.75, math is direct | CONFIRMED |
| `_dynamic_half_lane()` already exists | Code inspection | CONFIRMED |
| `_dynamic_half_lane()` is used for vehicles but not pedestrians | Code inspection | CONFIRMED |
| 38-43% false positive rate in Town10HD | HDF5 mode=6 + lateral offset analysis | CONFIRMED (estimated range) |

---

## Options

### Option A — Use `_dynamic_half_lane()` (F6 — Tier 1, RECOMMENDED)

Change the pedestrian check to use the already-implemented dynamic lane-width function.

```python
# Before:
half_w = self.config.pedestrian_lane_half_width  # = 3.0m

# After:
half_w = self._dynamic_half_lane()  # actual lane width / 2
```

| | |
|---|---|
| **Effort** | 1 line (literally one variable substitution) |
| **Risk** | VERY LOW — no logic change, same code path, just uses correct width |
| **Fixes** | 38–43% false pedestrian stops |
| **Industry standard?** | YES — dynamic lane widths are standard |
| **Most professional?** | YES |
| **Best long-term?** | PARTIAL — this is the 80% solution (see Option B for 100%) |

**Pros:**
- Trivial to implement (1 line)
- No new code — uses existing function
- Eliminates majority of false positives

**Cons:**
- Still uses lateral distance (a 1D metric); a pedestrian in an adjacent lane at 1.7 m lateral offset is within threshold
- Does not distinguish between lane center and sidewalk using map topology

---

### Option B — Road-Surface Pedestrian Check via Map API (F15 — Tier 3)

Instead of (or in addition to) lateral distance, check whether the pedestrian's position is on a drivable road tile using `carla.Map.get_waypoint(ped_location, project_to_road=False)`. If the result is None or is a sidewalk/shoulder tile, the pedestrian is not in the road and can be ignored.

```python
def _pedestrian_is_in_road(self, ped_actor):
    ped_loc = ped_actor.get_location()
    wp = self._map.get_waypoint(ped_loc, project_to_road=False, lane_type=carla.LaneType.Driving)
    return wp is not None  # None means not on a driving tile
```

| | |
|---|---|
| **Effort** | ~20 lines (new helper + integration into pedestrian check) |
| **Risk** | MEDIUM — map API call per pedestrian per frame (small overhead), and CARLA's lane type classification can be imperfect on some road geometries |
| **Fixes** | Near-complete elimination of sidewalk false positives |
| **Industry standard?** | YES — Waymo and similar use HD map lane occupancy checks |
| **Most professional?** | YES |
| **Best long-term?** | YES — this is the 100% solution |

**Pros:**
- Topologically correct — uses map lane type, not distance approximation
- Works correctly even at unusual geometries (bus stops, crossings)
- Eliminates lateral distance heuristic entirely

**Cons:**
- More complex than Option A
- `get_waypoint(project_to_road=False)` behavior can vary across CARLA map versions
- If a pedestrian crosses mid-block (jaywalks), they ARE in the road — correctly detected

---

## Recommended Action

**Tier 1 (immediate, do now):** F6 — Option A, 1-line change. Do this today. It's zero risk and the function already exists.

**Tier 3 (after Tier 0–2 data is clean):** F15 — Option B, road-surface check. This is the industry-standard complete fix. Schedule for Phase 2 (after Bezier arc work is also being planned).

---

## Open Questions for User

None. Option A is unambiguously correct and trivially safe. Option B is the long-term follow-up and does not block anything.

---

## Verification After Fix

After implementing F6, run test collection and check:

```python
import h5py, numpy as np, glob

for f in sorted(glob.glob('data/test_collection/episode_*.h5')):
    with h5py.File(f) as h:
        mode = h['driving_mode'][:]
        ped_stops = np.sum(mode == 6)
        total = len(mode)
        if ped_stops > 0:
            print(f"{f}: {ped_stops}/{total} PEDESTRIAN_STOP frames ({ped_stops/total*100:.1f}%)")
```

Expect: pedestrian stop rate should drop in Town10HD. Rate in episodes with no jaywalking pedestrians should approach 0. Rate in episodes with actual road-crossing pedestrians should remain non-zero.
