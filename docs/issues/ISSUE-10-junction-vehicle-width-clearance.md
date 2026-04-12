# ISSUE-10 — No Vehicle-Width Clearance Check for Stopped Vehicles in Junctions

> **Status:** Open
> **Priority:** Tier 2
> **Implementation plan tasks:** None yet
> **Connected issues:** [ISSUE-02](ISSUE-02-traffic-light-state-machine.md) (VEHICLE_FOLLOWING mode interaction)

---

## Summary

When ego and another stopped vehicle are both inside a junction and the junction is narrow, the existing hazard system does not account for the combined widths of both vehicles when deciding whether to proceed past the other vehicle. Ego may attempt to pass with insufficient lateral clearance and collide side-on.

The SAT corridor and OBB hazard detection work correctly for longitudinal following (vehicle ahead on the same path). They do not handle the case where a perpendicular or oblique stopped vehicle partially overlaps the ego's intended lateral clearance margin.

---

## What Was Observed

**Town07 ep3, end of episode:** Ego and a lead vehicle were both stopped inside a junction. The lead vehicle was stopped at an angle (partially blocking ego's intended path). Ego attempted to proceed alongside it without accounting for the width of its own bounding box relative to the lateral gap available. Collision occurred.

---

## Root Cause

The vehicle hazard check uses OBB-SAT intersection between ego's corridor polygon and the lead vehicle's OBB. This correctly detects vehicles directly in the path. However, it does not:

1. Check whether the lateral gap between ego's OBB and the adjacent vehicle's OBB is sufficient to pass safely
2. Apply a minimum clearance margin (ego half-width + other half-width + buffer) for oblique-to-path vehicles

---

## Options

### Option A — Lateral gap clearance check

For any vehicle within N metres of ego in all directions (not just in corridor): compute the lateral distance between the closest edges of the two OBBs. If < `min_lateral_clearance` (e.g., 0.3 m), trigger EMERGENCY_STOP or reduce speed to near-zero.

| | |
|---|---|
| **Effort** | ~30 lines |
| **Risk** | MEDIUM — may false-positive on adjacent-lane vehicles at stop lines |
| **Best long-term?** | PARTIAL |

### Option B — Widen the SAT corridor inside junctions

When in TURNING or TRAFFIC_LIGHT_STOP mode inside a junction, widen the corridor laterally by ego's half-width + a clearance margin. This ensures any vehicle partially overlapping the widened corridor is treated as a blocker.

| | |
|---|---|
| **Effort** | ~20 lines |
| **Risk** | LOW — conservative (stops more often) |
| **Best long-term?** | YES as first step |

**Recommendation:** Option B first (widen corridor in junctions). Add Option A as a complement if B still misses cases.

---

## Open Questions

- Does this happen often enough to affect route completion metrics (is it a Tier 1 priority)?
- The lateral gate in F8 (stopped-lead handling) partially addresses this — needs review of whether gap in F8 logic is exploitable here.
