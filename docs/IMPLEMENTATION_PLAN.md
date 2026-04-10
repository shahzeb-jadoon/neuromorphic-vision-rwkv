# Implementation Plan — Remaining Work

> **Last updated:** 2026-04-10
> **Schema:** v3.1 — 3-camera panoramic (3×640×320, 60° FOV, yaws −60/0/+60°),
>   PrivilegedExpert agent, RoadOption nav commands, privileged metadata + corridor diagnostics per frame.
> **Current branch:** `feature/issue-01-bezier-arc`
> **Latest verified baseline commit:** `1bda16c` — `feat(debug): add route-scoped overlays and visualization tooling`
> **If revert needed:** use a commit from the verified log below.

### Relevant commit log (April sync window)

```
017b71a  fix(config): increase timeout_duration 75→80 seconds
75b10c9  feat(data): write gap_fill_diverged and route_completion to HDF5
3baf485  fix(manifest): include nav classes 5–6 in distribution
77ed724  fix(data): include nav commands 5/6 in manifest distribution
4b5c1e0  fix(expert): use dynamic lane width for pedestrian hazards
fbf0939  fix(expert): suppress lane-change labels during turn approach
f0e4549  fix(expert): ignore stopped lead vehicles while turning
a24e48a  fix(expert): refine stopped-lead handling with lateral gate
2c8b0e5  fix(expert): let red traffic lights outrank vehicle-follow mode
ef3a25e  fix(expert): refresh TL stop line after buffer changes
aabf657  fix(expert): harden TL persistence and hazard arbitration
1bda16c  feat(debug): add route-scoped overlays and visualization tooling
```

---

## Current Baseline

```
109 offline tests passing (5 online deselected in CI)
87.5% coverage on testable code (threshold: 80%)
Schema v3.1 with corridor diagnostics
Test collection run: 18 episodes across 6 towns
  Towns: Town03, Town04, Town05, Town06, Town07, Town10HD
  Remaining blockers are documented below.

Recent hotfixes outside the original F1-F15 table:
- Town06 TL-drop mitigation in Tier-1 TL memory persistence/recovery logic.
- Stop/yield approach-phase control visualization before activation.
- Junction overlay reduced to current/upcoming route junctions only.
```

---

## Phase 0 — Fix Test Collection Issues Before Re-Collection

> **All issues in this phase were discovered during the multi-town test collection run.**
> **Detailed documentation for each issue is in [`docs/issues/`](issues/).**
> **Re-collection is BLOCKED until Phase 0 is complete.**

The issues are grouped by risk level. Work through tier by tier, running
`bash scripts/test_collection.sh` (smoke test) after each tier completes.

### Status Snapshot (2026-04-10)

| State | Items |
|---|---|
| Done | F2, F3, F4, F5, F6, F7, F8, F9, F13 |
| Partially done | F12 (non-junction same-road lane_id override implemented; policy decisions OD-1/OD-2 still open) |
| Pending | F9b, F10, F11, F14, F15 |
| Deferred | F1 (user-requested defer) |

---

### Tier 0 — Zero-Risk: Config + Pure Instrumentation
*No logic changes. Pure additions and config values. Cannot regress behavior.*

| # | Task | Effort | Issue File | Description |
|---|------|--------|------------|-------------|
| F1 | Fix `stop_sign_route_proximity` | 5 min | [ISSUE-05](issues/ISSUE-05-stop-sign-proximity.md) | **User-deferred (2026-04-02):** keep `2.5` to avoid adjacent-lane false stops observed when value > 3.0. |
| F2 | Fix `timeout_duration` | 2 min | [ISSUE-06](issues/ISSUE-06-monitoring-instrumentation.md) | **DONE** (`017b71a`): `timeout_duration` is `80` in `configs/collection.yaml`. |
| F3 | Write `gap_fill_diverged` to HDF5 | 15 min | [ISSUE-01](issues/ISSUE-01-junction-waypoints.md) | **DONE** (`75b10c9`): HDF5 dataset creation + per-frame write are present in `expert_collector.py`. |
| F4 | Write `route_completion` to HDF5 | 10 min | [ISSUE-01](issues/ISSUE-01-junction-waypoints.md) | **DONE** (`75b10c9`): persisted per frame in HDF5. |
| F5 | Fix manifest `nav_distribution` | 15 min | [ISSUE-03](issues/ISSUE-03-lane-change-mislabeling.md) | **DONE** (`3baf485`, `77ed724`): nav classes 5/6 are included in distribution output. |

---

### Tier 1 — Low-Risk: Logic Fixes (Targeted, Conservative)
*Each change affects a single, isolated behavior. No state machine restructuring.*

| # | Task | Effort | Issue File | Description |
|---|------|--------|------------|-------------|
| F6 | Fix pedestrian lane half-width | 5 min | [ISSUE-04](issues/ISSUE-04-pedestrian-false-positives.md) | **DONE** (`4b5c1e0`): pedestrian hazard logic uses dynamic lane width. |
| F7 | Suppress lane-change labels during TURNING | 30 min | [ISSUE-03](issues/ISSUE-03-lane-change-mislabeling.md) | **DONE** (`fbf0939`): TURNING transition behavior no longer emits persistent lane-change labels. |
| F8 | Turning stopped-lead deadlock fix | 30 min | [ISSUE-02](issues/ISSUE-02-traffic-light-state-machine.md) | **DONE** (`f0e4549`, `a24e48a`, `ea5587d`): implemented lateral-gated blocker logic (not blanket lead skipping). |
| F9 | VEH_FOLLOW red-light TL override | 45 min | [ISSUE-02](issues/ISSUE-02-traffic-light-state-machine.md) | **DONE** (`2c8b0e5`): red/yellow TL stop intent outranks vehicle-follow targeting. |
| F9b | Clear yellow-commit on full stop | 20 min | [ISSUE-02](issues/ISSUE-02-traffic-light-state-machine.md) | **PENDING:** add explicit reset of `_tl_proceed_committed` when ego fully stops before the stop line. |

---

### Tier 2 — Medium-Risk: Waypoint and State Machine Changes
*Each change touches the waypoint buffer pipeline or TL state machine. Test thoroughly.*

| # | Task | Effort | Issue File | Description |
|---|------|--------|------------|-------------|
| F10 | Replace `nexts[0]` in `_fill_buffer()` | 2 hr | [ISSUE-01](issues/ISSUE-01-junction-waypoints.md) | **PENDING:** route-end extension still uses `nexts[0]` (`privileged_expert.py` line 696). |
| F11 | Replace `nexts[0]` in `_fill_junction_gaps()` | 1 hr | [ISSUE-01](issues/ISSUE-01-junction-waypoints.md) | **PENDING (higher urgency than F10):** live junction path still uses `nexts[0]` (`privileged_expert.py` line 810). |
| F12 | Ground-truth `lane_id` lane-change detection | 2 hr | [ISSUE-03](issues/ISSUE-03-lane-change-mislabeling.md) | **PARTIAL:** non-junction, same-road lane_id crossing override exists. OD-1/OD-2 still required for final labeling policy. |
| F13 | Fix `_tl_stop_loc` stale reference during TURNING | 3 hr | [ISSUE-02](issues/ISSUE-02-traffic-light-state-machine.md) | **DONE** (`ef3a25e`): stop-line reference refresh is called after buffer and gap-fill updates. |

---

### Tier 3 — High-Risk: Architecture Change (Phase 2 Junction Fix)
*Full replacement of gap-fill system. Do only after Tier 0–2 are stable.*

| # | Task | Effort | Issue File | Description |
|---|------|--------|------------|-------------|
| F14 | Bezier/spline junction arcs | 2–3 days | [ISSUE-01](issues/ISSUE-01-junction-waypoints.md) | Replace `wp.next()` chaining with dense curve samples through junction interior. This approximates dense HD-map lane geometry and eliminates the `nexts[]` class of bugs. |
| F15 | Road-surface pedestrian check | 3 hr | [ISSUE-04](issues/ISSUE-04-pedestrian-false-positives.md) | Use `carla.Map.get_waypoint(ped_pos)` to check if pedestrian is on a drivable tile. Complete replacement for lateral-distance threshold. Do after F6 is confirmed working. |

---

### Open Design Decisions (Need User Confirmation Before Implementing)

These two decisions require explicit user input. See issue files for full context.

| # | Decision | Issue File | Options |
|---|----------|------------|---------|
| OD-1 | Lane-change label window: 775-frame run vs. precise 60-frame crossing | [ISSUE-03](issues/ISSUE-03-lane-change-mislabeling.md) | Current 775-frame labels inflate CHANGELEFT by ~700 frames per episode. Precise 60-frame labels require re-collection but produce correct signal. **Recommendation: precise 60-frame labels.** |
| OD-2 | Blind-spot lane-change check: full privileged perception vs. 180° FOV restricted | [ISSUE-03](issues/ISSUE-03-lane-change-mislabeling.md) | Privileged = safer expert, imitation gap at inference. 180° = model can replicate, expert crashes more during collection. **Recommendation: keep privileged + annotate frames where blind-spot data was used.** |

---

### Post-Phase-0: Re-Collection Run

Tier 0 is complete. Tier 1 is complete except F9b.
Finish F9b, smoke-test, then proceed with Tier 2 (F11 first, then F10).

| Task | Description | Time |
|------|-------------|------|
| Smoke test | `bash scripts/test_collection.sh` after Tier 0 + Tier 1 | 10–20 min |
| **Full re-collect** | All training towns, multiple weathers. Target: 10–15 hrs driving time. | 4–6 hrs wall (overnight) |
| **Validate** | `make validate` — schema integrity, SAT rate, nav distribution | 15 min |
| **Check monitoring fields** | Verify `gap_fill_diverged`, `route_completion` appear in HDF5 | 10 min |
| **Verify improvements** | Check pedestrian stop rate, LC label durations, TL violation count | 30 min |

---

## Phase 1 Remaining — Expert Fixes & Re-Collection

> Carries forward from previous plan. Phase 0 must complete first.

### P0 — Must fix before re-collection

| # | Task | Effort | Description |
|---|------|--------|-------------|
| D5 | Route-end corridor fix | 2–3 hrs | Waypoint buffer degenerates in last ~200 frames. Fix buffer recycling / re-planning when buffer head distance drops. |
| D2 | SAT corridor debugging | 3–4 hrs | SAT detection rate is 2.1%. Diagnose: is corridor too narrow? OBB orientation bug? Polygon winding issue? Use diagnostic fields + `diagnose_sat_misses.py` output. |

### P1 — Config & data preparation

| # | Task | Effort | Description |
|---|------|--------|-------------|
| D10 | Expand town coverage | 15 min | Add Town02, 06, 07, 10, 12, 15 to `collection.yaml`. Town02 reserved as eval-only. |
| D8 | Velocity stats (initial) | 30 min | Compute `VELOCITY_MEAN`/`VELOCITY_STD` from overnight data. Update `driving_dataset.py`. |
| — | Regenerate toy data | 5 min | Run `make toy-data` with CARLA for v3.1 toy dataset. |

---

## Phase 2a — Architecture & Training Infrastructure

> **Weeks 5–6.** Vision-RWKV architecture + training pipeline.

| # | Task | Priority | Description |
|---|------|----------|-------------|
| D3 | RWKV-7 implementation | P0 | **1-week timebox.** If CUDA kernels fail, pivot to RWKV-6. Infra is version-agnostic. |
| D4 | Vision backbone | P0 | Start with frozen CLIP patch embeddings. Target: from-scratch PatchEmbed after initial results. |
| 6 | Experiment configs | P1 | `configs/experiments/*.yaml` per run. |
| — | Trainer class | P0 | TBPTT, gradient accumulation, AMP, state carry-over. |
| — | Checkpointing | P1 | Save/resume with optimizer, LR schedule, RWKV state. |
| — | W&B integration | P1 | Loss, LR, gradients, image samples. |

---

## Phase 2b — Training & Results

> **Weeks 6–8.** Train model on collected data.

| # | Task | Priority | Description |
|---|------|----------|-------------|
| D6 | Stationary filtering | P0 | Filter clips where >50% of frames are stationary (speed < 0.1 m/s). Prevents "always stop" policy. |
| D7 | Steering noise (training) | P1 | Gaussian noise σ=1e-3 already injected at collection time. Verify in training data. |
| 21 | Multi-resolution ablation | P2 | 224/256/384/512 — FLOPs, latency, driving metrics. |
| — | 3-stage training | P0 | Frozen CLIP backbone → unfreeze → fine-tune with from-scratch PatchEmbed. |
| D4 | PatchEmbed ablation | P1 | Compare frozen CLIP vs from-scratch PatchEmbed on driving metrics. |

---

## Phase 3 — Evaluation

> **Weeks 8–10.** Baselines, ablations, closed-loop.

| # | Task | Priority | Description |
|---|------|----------|-------------|
| 8 | Nav command ablation | P1 | Remove nav input, retrain, compare. |
| — | Baseline comparisons | P0 | ResNet-18, ViT-Small, LSTM. |
| — | Closed-loop eval | P0 | Route Completion >70%, Collision <0.5/km. Town02 held out. |
| 4 | DAgger (stretch) | P2 | Expert corrections → retrain. |

---

## Phase 4 — Documentation & Deployment

> **Weeks 11–12.** Write-up, Jetson, demo.

| # | Task | Priority | Description |
|---|------|----------|-------------|
| 20 | Lit review → LaTeX | P0 | Convert markdown docs to a paper chapter. 7 analysis docs available. |
| 22 | η_AF grounding | P1 | Literature-ground neuromorphic activation fraction metric. |
| — | Jetson deployment | P1 | ONNX/TensorRT, <50ms latency target. |
| — | Write-up + demo | P0 | All chapters, results, figures, video. |

---

## Low-Priority / Optional

| # | Notes |
|---|-------|
| 15 | Pin mypy exactly — only if a mypy upgrade breaks CI. |
| 16 | `make lock` target — low value for conda-managed solo project. |
| 12r | `sys.path.insert` hack in validate_dataset.py — editable install handles it. |

---

## Risk Register

| Risk | Mitigation |
|------|------------|
| RWKV-7 CUDA kernels fail | 1-week timebox, pivot to RWKV-6. Infra is version-agnostic. |
| SAT rate stays low after debugging | Lateral fallback catches 97.8% — model training can proceed. Debug improves quality, not gates progress. |
| Re-collection takes too long | Overnight runs. Target ~15 hrs driving at 20× speed ≈ 4–6 hrs wall. |
| Velocity placeholder stats | Compute from overnight data first (D8), recompute after re-collection. |
| Training instability | Short TBPTT segments (T=32), W&B gradient monitoring. |
| Distribution drift | Steering noise (D7) + DAgger (stretch). |
| Phase 0 Tier 2 regressions | Run `test_collection.sh` smoke test after each tier. Revert to `514a2b8` if needed. |
| nexts[0] fix incomplete | Verify using new `gap_fill_diverged` HDF5 field after F10/F11. |
| Lane-change label redesign (OD-1) | Decision needed before F12. Existing data may need relabeling. |
| Blind-spot imitation gap (OD-2) | Decision needed before F12. Affects training alignment. |

---

## Decision Summary

Phase 0 adds the following decisions from test collection findings:

| ID | Decision | Status | Reference |
|----|----------|--------|-----------|
| OD-1 | Lane-change label window size | **Needs user input** | [ISSUE-03](issues/ISSUE-03-lane-change-mislabeling.md) |
| OD-2 | Blind-spot check: privileged vs. 180° FOV | **Needs user input** | [ISSUE-03](issues/ISSUE-03-lane-change-mislabeling.md) |
| OD-3 | Bezier arcs vs. heading-aligned nexts[] | Defer to Tier 3 (Phase 0) | [ISSUE-01](issues/ISSUE-01-junction-waypoints.md) |
| OD-4 | Pedestrian: dynamic half-width vs. road-surface map check | Use dynamic now, road-surface later | [ISSUE-04](issues/ISSUE-04-pedestrian-false-positives.md) |
