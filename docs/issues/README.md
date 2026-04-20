# Phase-1 Test Collection Issues

> **Branch:** `feature/waypoint-planner-policy`
> **Latest verified baseline:** `5a1371f` — `feat(collection): add route continuation chaining`
> **Source:** All issues discovered and verified during a 6-town, 18-episode test collection run.
> **If revert needed:** check out a known-good commit from the branch log.

---

## Issue Index

| File | Issue | Priority | Tasks | Status |
|------|-------|----------|-------|--------|
| [ISSUE-01](ISSUE-01-junction-waypoints.md) | Junction branch selection and gap-filling stability | Tier 2 (fix) / Tier 3 (Bezier arcs) | F10, F11, F14 | Partially fixed (F3/F4 done; F10/F11 refactor in code, runtime closure evidence pending) |
| [ISSUE-02](ISSUE-02-traffic-light-state-machine.md) | TL/VEH_FOLLOW interactions: stale stop-line, follow-through red, deadlock, yellow-commit reset | Tier 1/Tier 2 | F8, F9, F9b, F13 | Mostly fixed (F8/F9/F9b/F13 implemented; keep regression watch in long runs) |
| [ISSUE-03](ISSUE-03-lane-change-mislabeling.md) | 775-frame CHANGELEFT mislabeling + manifest hides nav 5+6 | Tier 0 (F5) / Tier 1 (F7) / Tier 2 (F12) | F5, F7, F12 | Partially fixed (F5/F7 done; F12 partial) |
| [ISSUE-04](ISSUE-04-pedestrian-false-positives.md) | Fixed 3.0 m pedestrian threshold too wide; `_dynamic_half_lane()` already exists | Tier 1 (F6, 1-line) / Tier 3 (F15) | F6, F15 | Partially fixed (F6 done; F15 pending) |
| [ISSUE-05](ISSUE-05-stop-sign-proximity.md) | `stop_sign_route_proximity: 2.5` too tight for curved approaches | Tier 0 (F1) | F1 | Deferred by user |
| [ISSUE-06](ISSUE-06-monitoring-instrumentation.md) | Monitoring and instrumentation gaps from first test run | Tier 0 (F2–F5) | F2, F3, F4, F5 | Mostly fixed |
| [ISSUE-12](ISSUE-12-exhaustive-route-atlas-coverage-planning.md) | Offline exhaustive route atlas coverage planning + background-safe execution | New planning track | RP1 | Open (planning started; script implementation pending) |

---

## Implementation Tiers

```
Tier 0 — Zero risk (config + instrumentation, no logic changes)
  F1  stop_sign_route_proximity 2.5 → 4.0          [ISSUE-05]
  F2  timeout_duration 75 → 80 → 120                [ISSUE-06]  DONE
  F3  gap_fill_diverged + gap_fill_rpcs → HDF5      [ISSUE-06]  DONE
  F4  route_completion → HDF5                        [ISSUE-06]  DONE
  F5  manifest nav_distribution: include 5+6         [ISSUE-03] / [ISSUE-06]  DONE

Tier 1 — Low risk (targeted logic changes)
  F6  pedestrian threshold → _dynamic_half_lane()   [ISSUE-04]  DONE
  F7  suppress LC labels during TURNING             [ISSUE-03]  DONE
  F8  turning stopped-lead lateral gate             [ISSUE-02]  DONE
  F9  VEH_FOLLOW: red TL override                   [ISSUE-02]  DONE
  F9b clear _tl_proceed_committed on full stop      [ISSUE-02]  DONE

Tier 2 — Medium risk (waypoint pipeline + TL state machine)
  F10 heading-aligned branch selection in _fill_buffer() [ISSUE-01]  IN CODE, VALIDATE
  F11 heading-aligned branch selection in _fill_junction_gaps() [ISSUE-01]  IN CODE, VALIDATE
  F12 ground-truth lane_id lane-change detection     [ISSUE-03]  PARTIAL
  F13 _tl_stop_loc re-evaluation after gap fill      [ISSUE-02]  DONE

Tier 3 — High complexity (architectural change)
  F14 Bezier/spline junction arcs                    [ISSUE-01] ← industry standard
  F15 road-surface pedestrian check via map API      [ISSUE-04]

New planning track (parallelizable with collection runs)
  RP1 exhaustive route atlas coverage planner script [ISSUE-12]  OPEN
```

---

## Open Design Decisions (Need User Input Before Implementing)

| ID | Decision | Issue | Recommendation |
|----|----------|-------|----------------|
| OD-1 | Lane-change label window: 775-frame run vs. precise 60-frame crossing | [ISSUE-03](ISSUE-03-lane-change-mislabeling.md) | Precise 60-frame labels |
| OD-2 | Blind-spot check: full privileged vs. 180° FOV restricted | [ISSUE-03](ISSUE-03-lane-change-mislabeling.md) | Keep privileged + annotate `blind_spot_blocked` per frame |

---

## Connected Issues Map

```
ISSUE-01 (nexts[0])
    └── causes → ISSUE-02 (Bug A: stale TL stop-line during turns)
    └── requires monitoring → ISSUE-06 (F3: gap_fill_diverged, F4: route_completion)

ISSUE-02 (TL state machine)
    └── partly caused by → ISSUE-01 (bad junction waypoints → stale TL reference)

ISSUE-03 (lane change mislabeling)
    └── manifest problem also tracked in → ISSUE-06 (F5)
    └── has open decisions → OD-1, OD-2

ISSUE-04 (pedestrian threshold)
    └── standalone; fix already exists in codebase

ISSUE-05 (stop sign proximity)
    └── standalone config fix

ISSUE-06 (monitoring)
    └── supports verification of → ISSUE-01 (F3, F4)
    └── supports visibility of → ISSUE-03 (F5)

ISSUE-12 (route atlas planning)
  └── uses coverage metrics from topology + route sidecars
  └── intended as offline/background precompute path
```

---

## Verified Observations from Test Collection

All issues have been verified through one or more of the following methods:
- Python scripts run against actual HDF5 files
- Direct code inspection with line-number citations
- CARLA map API queries during live test runs
- Frame-by-frame HDF5 data examination

Issues marked "confirmed" have HDF5 data or code evidence. Issues marked "theoretical" have geometric/logical proof but no direct HDF5 measurement.

| Observation | Status | Evidence |
|---|---|---|
| nexts[0] → fence collision Town07 ep1 frame 641 | CONFIRMED | HDF5 nav/speed/position frame-by-frame |
| Town05 ep1 partial red-light crossing (55 frames) | CONFIRMED | Python script |
| Town06 ep3 stop was legitimate (not spurious) | CONFIRMED | Python script |
| Town10HD ep2 c00e760b: not a U-turn | CONFIRMED | Python script |
| Town10HD ep2 c00e760b: 775-frame CHANGELEFT run | CONFIRMED | Python script |
| Manifest drops nav 5+6 | CONFIRMED | Direct manifest vs HDF5 comparison |
| gap_fill_diverged not in HDF5 | CONFIRMED | Code inspection |
| route_completion not in HDF5 | CONFIRMED | Code inspection |
| Pedestrian threshold 3.0 m > Town03 lane half-width 1.75 m | CONFIRMED | CARLA map API + code inspection |
| _dynamic_half_lane() already exists, not used for pedestrians | CONFIRMED | Code inspection |
| VEH_FOLLOW + red TL follow-through (5+ episodes) | CONFIRMED | HDF5 pattern: tl=3 + speed>1 + mode=5 |
| Town03 ep2+ep3 deadlock timeouts | CONFIRMED | HDF5 final frames: mode=5, speed=0, tl=GREEN |
| Town10HD ep2 4b549931 (timeout episode) | UNRESOLVED | Not yet analyzed |
