# Hybrid Implementation and Forensics (2026-04-19)

## Scope

This note records:

1. The pre-hybrid backup point (commit + branch).
2. What was implemented in the robust long-term hybrid sequence and why.
3. Forensic analysis of issues reported in the latest `data/test_collection` run.
4. A proposed fix plan that requires explicit approval before implementation.

No fixes are implemented from the forensic section in this document.

---

## Pre-Hybrid Backup Reference

- Branch before hybrid implementation steps: `feature/waypoint-planner-policy`
- Commit immediately before hybrid sequence started: `d115607`
- Commit subject: `(test) expert-collector: lock route fail-fast safety defaults`

This is the requested stable backup point before C+H, B, D, and G+F work.

---

## Atomic Hybrid Implementation Record

Implemented in requested order:

1. `b8c8b35` - `feat(routes): enforce waypoint budgets and wire balance config`
   - C+H route controls and actual balance controls.
2. `3059817` - `fix(planner): prevent junction command spillover`
   - B label-source correctness.
3. `33a53bb` - `fix(expert): smooth TL queue following before emergency`
   - D behavior smoothness/safety realism.
4. `6f22477` - `feat(audit): add local-oracle junction-window nav label checks`
   - G ongoing data audit and F local oracle checks around junction windows.

---

## Latest Run Forensics

Data source:
- `data/test_collection/collection_manifest.json`
- `data/test_collection/test_collection.log`
- Episode sidecars/HDF5 in `data/test_collection/`

### 1) Town06 Episode 2: End-of-episode turn-back/wrong-way behavior

Episode:
- `episode_dcccc372` (`Town06`, traffic=50)

Observed pipeline evidence:
- Lane-graph path miss with GRP fallback in generation (`test_collection.log`).
- Legality filter removed many transitions (`630 -> 479`).
- Continuity guard explicitly kept `3` large skip-gap transitions.
- Sanitizer then flagged `3 severe` transitions with max gap `30.68m`.
- Episode ended in collision with `static.pole` at frame `1159`.

Route-sidecar structural evidence (`episode_dcccc372.route.json`):
- Severe route-tail continuity gaps at transitions `475->476`, `476->477`, `477->478`.
- Gap magnitudes are ~`30m` each.
- Tail also includes lane-id shift (`lane 4 -> lane 3`) at the first severe transition.

HDF5 behavior evidence (`episode_dcccc372.h5`):
- `route_completion` reaches `1.0` while the run continues to frame 1159.
- Heading/position near tail shows unstable end behavior before collision.

Conclusion:
- The end-of-episode wrong-way/turn-back behavior is consistent with a malformed route tail (severe discontinuities) combined with continued driving after route completion.
- This is not a random one-off control blip; the route artifact is explicit in sidecar sanitizer diagnostics.

### 2) Frequent emergency stops from adjacent-lane large vehicles in SAT corridor

You asked whether adjacent-lane large vehicles are being captured by SAT corridor and causing emergency stops.

Findings from latest run (all episodes):
- Total emergency-stop frames: `3423`.
- Emergency-stop frames with a SAT-selected ahead lead: `3360`.
- Of those, lead lateral offset `> 2.5m`: `2087` frames.
- Lead lateral offset `> 3.5m`: `1427` frames.
- Lead vertical separation `> 1.0m`: `2163` frames (still below current hard reject threshold `2.5m`).

Interpretation:
- A large share of emergency stops are associated with SAT-selected leads that are far laterally offset from ego centerline.
- This is consistent with your observation that wide adjacent-lane actors can overlap corridor geometry and be treated as in-lane hazards.

Current implementation details (important for tuning):
- Corridor half-width is computed as:
  - `max(lane_half + margin, ego_half_width + corridor_vehicle_half_width_margin)`
- In defaults, lane margin is fixed at `-0.1m` and ego half-width margin is `+0.2m`.
- Vertical separation rejection is only applied when separation exceeds `2.5m`.

Practical implication:
- In dense maps with closely spaced lanes and wide vehicles, SAT overlap can still occur for adjacent lanes and trigger lead hazard logic.

### 3) Town05 Episode 3 emergency stops around frames 500-900

Episode:
- `episode_c25c38bc` (`Town05`, traffic=100)

Emergency-stop intervals in this episode:
- `426-437`
- `687-734`
- `3981-3999`

For the requested range (`500-900`):
- The relevant emergency-stop block is `687-734`.
- Frame-level hazard reconstruction indicates this block is pedestrian-driven, not lead-vehicle emergency-distance driven.
- Example at frame ~710:
  - Pedestrian predicted distance ~`4.09m` and predicted lateral near center (`0.05m`), which is below emergency threshold behavior.

About Z-plane verification for pedestrians:
- Current `nearby_pedestrians` telemetry does not log world `z` or vertical separation fields.
- Because those fields are absent in stored episode data, we cannot conclusively prove same-z-plane for pedestrians post hoc from this dataset alone.

### 4) Approaching-turn/turning when heading appears unchanged after junction

Reported windows:
- Town03 episode 1: frames `1300-2000`
- Town03 episode 2: frames `1450-2600`
- Town05 episode 2: frames `1300-1700`
- Town05 episode 3: frames `900-1500`

What was found:
- In these windows, driving modes are heavily `APPROACHING_TURN`/`TURNING` with `nav_command` mostly `LEFT` or `RIGHT`.
- Many of those turn-labeled frames map to sidecar waypoints where `is_junction=false`.
  - Town03 ep1 turn-mode frames in the requested window: `701`, with `0%` on junction waypoints.
  - Town05 ep2 turn-mode frames in the requested window: `385`, with `0%` on junction waypoints.

Additional planner diagnostic signal:
- Route sidecars show very high lane-graph `junction_segment_count` ratios:
  - Town03 routes: `469/503` (~`93.2%`)
  - Town05 routes: `536/604` (~`88.7%`)

Conclusion:
- The symptom is real and consistent with over-propagation of turn semantics from lane-graph segment/junction classification into long non-junction stretches.
- This explains why you can see approach/turn behavior even when overall before/after heading appears same-direction at the trajectory level.

---

## Analysis and Conclusion

1. The hybrid implementation sequence itself is in place and committed atomically.
2. The biggest new behavioral regression signal is route quality degradation in specific fallback/sanitized routes (Town06 ep2 severe tail discontinuities).
3. Emergency-stop behavior has two distinct contributors:
   - True/likely pedestrian emergencies (confirmed in Town05 ep3 frame window).
   - SAT lead over-triggering with high lateral offsets (likely adjacent-lane overlap sensitivity).
4. Turn-mode overactivation is primarily label-source/segment-classification quality, not just controller tuning.

---

## Proposed Fixes (Approval Required Before Any Implementation)

### A) Route-tail discontinuity hardening (Town06 ep2 class)

- Enforce fail/regenerate when sanitizer reports severe transition gaps above threshold (instead of accepting route with continuity-guard exceptions at tail).
- Optionally trim unsafe tail transitions if route still satisfies min length and legality constraints.

### B) SAT corridor target shape update (as requested)

- Replace corridor half-width computation with:
  - `half_w = max(ego_half_width, 0.95 * lane_width / 2)`
  - apply configurable additive margin only if explicitly enabled.
- Keep/optionally tighten vertical-separation rejection.

### C) Lead hazard gating for adjacent lanes

- Add a stricter lateral gate for emergency lead selection even when SAT overlap is true.
- Keep SAT for awareness, but require tighter centerline/lane consistency for emergency-stop trigger path.

### D) Pedestrian forensic observability

- Extend `nearby_pedestrians` telemetry to include `world_x/world_y/world_z` and `vertical_sep`.
- This enables definitive post-run same-z-plane verification.

### E) Turn-mode semantics correction for non-junction stretches

- Tighten lane-graph segment `is_junction` assignment so long non-junction segments are not marked junction by sparse sampled-point contamination.
- Add route-label guard: if route waypoint is non-junction and heading-change proxy remains below threshold, downgrade turn command toward lane-follow/straight semantics.

No code changes from A-E have been applied yet. Approval is required before implementation.
