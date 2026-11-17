# ISSUE-09 — Collisions with Untracked Static Obstacles

> **Status:** Open
> **Priority:** Tier 2
> **Implementation plan tasks:** None yet
> **Connected issues:** [ISSUE-04](ISSUE-04-pedestrian-false-positives.md) (pedestrian detection uses lateral distance — same gap)

---

## Summary

The expert agent's hazard detection covers vehicles (SAT/OBB + lateral fallback), pedestrians (lateral distance), and traffic lights. It has no mechanism for detecting static map objects such as utility poles, jersey barriers, fence posts, or parked construction equipment that are not CARLA `vehicle` or `walker` actors. These objects can be hit at full speed with no braking.

---

## What Was Observed

**Town07 ep2:** Ego collided with a roadside pole near the end of the episode. The pole was not a CARLA vehicle or walker actor. No hazard was raised. Ego was traveling at normal following speed at time of impact.

---

## Options

### Option A — CARLA static obstacle actors

CARLA exposes static props as `carla.Actor` with type `static.*`. Query `world.get_actors().filter('static.*')` each tick, build OBBs, run SAT against ego corridor.

| | |
|---|---|
| **Effort** | ~40 lines |
| **Risk** | LOW |
| **Coverage** | Props placed by the CARLA simulator (poles, barriers, signs) |
| **Does NOT cover** | Geometry embedded in the map mesh (wall curbs, building edges) |

### Option B — Map raycasting

Cast rays along the planned path using CARLA's `world.cast_ray()`. Stop if a non-driveable surface is hit within a threshold distance.

| | |
|---|---|
| **Effort** | ~60 lines |
| **Risk** | MEDIUM — ray API is slow; needs throttling |
| **Coverage** | Any geometry in the scene |
| **Best long-term?** | YES |

### Option C — Driveable-tile check ahead

Use `carla.Map.get_waypoint()` at points 2–10 m ahead of ego. If no driveable waypoint is returned, trigger EMERGENCY_STOP.

| | |
|---|---|
| **Effort** | ~20 lines |
| **Risk** | LOW |
| **Coverage** | Detects ego going off-road, but not mid-road poles |

**Recommendation:** Option A as immediate fix (covers most CARLA static props). Option B as Tier 3 improvement.

---

## Open Questions

- Is this a data quality issue (avoid polluting training labels with freak collisions) or a safety issue (improve expert driving score)?
- Option A requires iterating `static.*` actors each tick — may be slow in dense towns.
