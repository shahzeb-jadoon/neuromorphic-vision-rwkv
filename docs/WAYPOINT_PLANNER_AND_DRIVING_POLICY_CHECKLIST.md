# Waypoint Planner and Driving Policy Checklist

Status: Drafted on top of `feature/issue-01-bezier-arc`
Scope: Proactive route legality + reactive trained-driving evaluation

## Why this exists

Recent collection results improved significantly after route densification and dual-buffer changes.
This checklist prevents regressions in route legality and defines how to verify the trained model still drives safely and lawfully.

## Brainstorm Rule Backlog (Planner Documentation Candidates)

| ID | Rule idea | Scope | Why it matters | Suggested planner behavior | Verification signal | Priority |
|---|---|---|---|---|---|---|
| BR-01 | Lane lock inside junction | Any junction traversal | Prevent mid-junction lane hopping and side-swipe risk | Record entry lane id at junction ingress; forbid lane change until junction exit unless lane is blocked | Count mid-junction lane-id changes per episode | Critical |
| BR-02 | Exit in corresponding lane | Junction entry/exit pair | Preserve legal turn geometry and realism | Map entry lane index to legal exit lane index; reject connectors that cross multiple lanes at once | Exit-lane mismatch rate | Critical |
| BR-03 | Roundabout lane commitment | Roundabout segments | Avoid weave behavior that creates conflicts | Select lane at roundabout entry based on intended exit; hold lane until a defined pre-exit window | Lane oscillation count inside roundabout polygon | Critical |
| BR-04 | Roundabout anti-zigzag hysteresis | Roundabout only | Prevent inner-outer-inner ping-pong | After one lane change in roundabout, apply cooldown distance before another change | Consecutive lane-change chain length in roundabout | High |
| BR-05 | Junction lane-change hard ban | Junction and connector lanes | Enforce conservative safety baseline | Disallow lane-change commands when waypoint is in junction topology | Number of lane-change nav commands in junction | Critical |
| BR-06 | Turn legality by lane markings | Junction approaches | Prevent illegal turn selection | Validate lane markings and lane links before assigning LEFT/RIGHT/STRAIGHT command | Illegal turn attempts from lane permissions | Critical |
| BR-07 | One-way direction consistency | All route segments | Prevent wrong-way driving | Reject candidate waypoints with heading opposite lane travel direction | Wrong-way frame count | Critical |
| BR-08 | Do-not-enter shielding | Lane entries and ramps | Prevent severe rule violations | Pre-filter successors connected to do-not-enter controlled approaches | Do-not-enter entry events | Critical |
| BR-09 | Route continuity floor | Any sparse topology gap | Prevent skipped connectors and chord-cutting | Require max inter-waypoint distance and legal graph continuity after densification | Max waypoint head distance spikes | High |
| BR-10 | Minimum hold distance after lane change | Non-junction roads | Reduce oscillatory planning | Enforce distance/time lockout before the next lane-change decision | Lane-change oscillation rate | High |
| BR-11 | Dynamic turn-balance compensation | Episode scheduling across campaign | Keep long-run left/right mix near target even with per-route constraints | Track cumulative left/right deficit; adjust next-route target (e.g., 20% left if left is 5% behind) with cap and timeout fallback | Running campaign left/right percentages | High |
| BR-12 | Turn-balance failsafe fallback | Route generation timeout or infeasible map | Preserve throughput when balancing is expensive | If solver exceeds time budget, revert to baseline 15/15 target for that route and log deficit carryover | Route generation latency + fallback counter | Medium |
| BR-13 | Turn quota confidence bounds | Campaign-level routing | Avoid over-correction instability | Apply bounded correction window (for example +/-5%) and exponential smoothing of deficits | Quota overshoot/undershoot variance | Medium |
| BR-14 | Lane-change budget per km | Whole route | Control aggressiveness globally | Define max lane changes per km and penalize routes exceeding budget | Lane changes per km histogram | Medium |
| BR-15 | Blocked-lane exception policy | Active blockage/incident | Keep safety rules practical under obstacles | Permit controlled exception to lane-lock only with explicit blockage reason and recovery plan | Exception count with reason code | Medium |
| BR-16 | Pre-junction lane-change freeze with mandatory-exit exception (25-30 m) | Junction/exit approach corridor | Prevent late weave before turns while preserving required exit positioning | Forbid lane-change initiation inside configured freeze window (default 30 m), except when a required exit would otherwise be missed within 30 m from road entry and straight-through command is not intended | Violations of freeze window without exception reason code | Critical |
| BR-17 | No consecutive lane changes except mandatory-exit recovery | Consecutive lane-change decisions | Prevent zig-zag and multi-lane jumping | Disallow consecutive lane changes unless route requires immediate exit positioning within <=30 m and no single-change option exists | Consecutive lane changes without mandatory-exit justification | Critical |
| BR-18 | Exit-lane discipline for straight-through traffic | Exit lane segments | Prevent staying in exit lane then jumping left to continue straight | If nav command is straight-through, ego must leave exit-only lane >=30 m before exit gore when feasible; never jump multiple lanes in one decision cycle | Straight-command frames spent in exit-only lane near exit; multi-lane jump count | High |

Notes:
- BR-11 to BR-13 should be implemented as optional scheduler policy toggles so collection throughput is not blocked.
- BR-01 to BR-08 and BR-16 to BR-18 should be treated as non-optional legality and safety constraints.

---

## Proactive Waypoint Planner Guardrails

| ID | Guardrail | Why it matters | Planner-side check (must pass) | Priority |
|---|---|---|---|---|
| WP-01 | Never route through `Do Not Enter` approaches | Prevent illegal and unsafe entries | Validate lane direction and sign semantics before committing waypoints | Critical |
| WP-02 | Never enter one-way roads in the opposite direction | Prevent wrong-way driving and collisions | Ensure heading aligns with lane travel direction on every segment | Critical |
| WP-03 | Obey lane-level turn permissions | Prevent illegal maneuvers | If lane allows `Straight/Right`, planner must not output `Left`; same for all lane-marking constraints | Critical |
| WP-04 | Limit consecutive lane changes | Avoid unstable zig-zag behavior | Reject plans that perform more than 2 immediate lane shifts in sequence | High |
| WP-05 | U-turn from the nearest valid lane | Reduce multi-lane crossing risk | Select U-turn lane with minimal lateral crossing cost and legal turn permission | High |
| WP-06 | No rapid lane oscillation | Improve stability and realism | Apply cooldown or hysteresis after lane-change decisions | High |
| WP-07 | Preserve route continuity through junction exits | Prevent sparse-gap skips and chord-cutting | Densify transitions and verify legal successor lane after each junction | Critical |
| WP-08 | No lane change 25-30 m before junction entry (except mandatory exit) | Stabilize turn approach and avoid late merge errors | Enforce configurable freeze window (default 30 m) and allow only required-exit exception with explicit reason code | Critical |
| WP-09 | No consecutive lane changes unless required for imminent exit | Reduce zig-zag and lane-jump behavior | Consecutive lane changes require mandatory-exit proof within <=30 m and cannot be used for straight-through continuation | Critical |
| WP-10 | Do not remain in exit lane when going straight | Avoid unsafe late lane-jump from exit lane | If route command is straight-through, enforce transition out of exit-only lane >=30 m before exit where feasible | High |

---

## Reactive Trained-Driving Evaluation Checks

| ID | Evaluation check | Measurement idea | Pass criterion |
|---|---|---|---|
| EV-01 | Wrong-way driving frames | Count frames where ego heading opposes lane direction | 0 violations on validation routes |
| EV-02 | Do-not-enter violations | Count entries into prohibited lane segments | 0 violations |
| EV-03 | Lane-turn rule violations | Compare executed turn vs lane permission at junction approach | 0 violations |
| EV-04 | Consecutive lane-change burst | Count lane-change chains with length > 2 | 0 bursts |
| EV-05 | U-turn lane correctness | Check whether U-turn initiated from closest legal lane | 100% compliance on U-turn scenarios |
| EV-06 | Junction continuity | Track `wp_buffer_head_dist` spikes and route regressions | No route regressions; no catastrophic head-distance spikes |
| EV-07 | Sign compliance under pressure | Stress scenarios with nearby actors and mixed signs | No illegal turn or wrong-way events |
| EV-08 | Pre-junction lane-change violations | Count lane-change starts within 0-30 m of upcoming junction entry | 0 violations except mandatory-exit or blockage exception codes |
| EV-09 | Consecutive lane-change legality | Count consecutive lane changes lacking mandatory-exit reason within <=30 m | 0 violations |
| EV-10 | Straight-in-exit-lane violations | Count straight-command frames in exit-only lane inside 30 m to exit | 0 avoidable violations |

---

## Research and Industry Pattern Matrix (Proactive vs Reactive)

Based on prior analysis notes in this repository and publicly disclosed patterns.

| Group | Proactive measures | Reactive measures |
|---|---|---|
| TransFuser-style rule experts | Dense route following, junction-aware speed policy, future trajectory extrapolation horizon (longer in junctions) | Hazard-triggered braking (often binary), safety box checks |
| TCP / early LBC-lineage | Command/angle-based slowdown heuristics | Cone/ray hazard checks with binary brake |
| InterFuser | Lane-aware filtering, junction-specific detection modes, richer actor semantics | Still mostly binary safety overrides despite richer detection |
| Roach / MILE / ThinkTwice family | Learn behavior policy from BEV + reward shaping, route-conditioned latent planning | Rule-based safety override layers in data collection/eval pipelines |
| Waymo/Cruise-style (publicly described) | HD map constraints, hierarchical stack (perception -> prediction -> planning -> control), lane-occupancy legality checks | Independent safety monitors, collision-imminent fallback behavior |
| Tesla-style (publicly described) | End-to-end occupancy/trajectory planning with lane/topology constraints and continuous replanning | Runtime intervention layers (e.g., emergency braking/fallback controls) |

---

## Branch Notes for Planning Decisions

| Branch | What it adds | Known gap relative to current baseline | Recommendation |
|---|---|---|---|
| `feature/issue-01-bezier-arc` (current) | Dual-buffer Bezier + route-time densification + strong recent collection behavior | Runtime `nexts[0]` still present in two fallback paths | Keep as primary branch |
| `f10/f11-fixes` | Heading/lane-order branch scoring for route-end and junction fallback | Does not include current route densification architecture and includes older gap-fill flow | Mine selectively; do not wholesale switch |
| `archive/expert-stanley-cte` | Stanley/CTE steering ideas and older control variants | Lacks current gap-fill/dual-buffer/densification stack and many newer safety diagnostics | Treat as reference only |

---

## Implementation Priority Guidance

| Priority | Next action |
|---|---|
| P0 | Add planner legality validator for one-way, do-not-enter, and lane-turn permissions |
| P0 | Add evaluation metrics EV-01..EV-05 to validation reports |
| P1 | Decide whether to port heading-scored fallback from `f10/f11-fixes` into current branch safety-net paths |
| P1 | Add explicit tests for illegal-lane and illegal-turn route generation |
| P2 | Investigate selective Stanley/CTE concepts from archive branch only if needed after P0/P1 |

---

## Acceptance Gate Before Major Re-collection

A collection run is approved only if:
1. Planner guardrails WP-01..WP-05 and WP-08..WP-10 are enforced.
2. Evaluation checks EV-01..EV-05 and EV-08..EV-10 report zero violations.
3. Junction continuity (EV-06) remains stable across multi-town runs.
