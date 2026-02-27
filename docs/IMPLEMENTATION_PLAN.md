# Implementation Plan — Remaining Work

> **Last updated:** 2026-02-28
> **Schema:** v3.1 — 3-camera panoramic (3×640×320, 60° FOV, yaws −60/0/+60°),
>   PrivilegedExpert agent, RoadOption nav commands, privileged metadata + corridor diagnostics per frame.
> **Branch:** `phase-1/data-pipeline`

---

## Current Baseline

```
109 offline tests passing (5 online deselected in CI)
87.5% coverage on testable code (threshold: 80%)
6 atomic commits on phase-1/data-pipeline
Schema v3.1 with corridor diagnostics
1,125 episodes collected (overnight run) — to be re-collected after fixes
```

---

## Phase 1 Remaining — Expert Fixes & Re-Collection

> **Target: 1–2 days.** Fix known expert driver issues, then do a clean full re-collection.

### P0 — Must fix before re-collection

| # | Task | Effort | Description |
|---|------|--------|-------------|
| D5 | Route-end corridor fix | 2–3 hrs | Waypoint buffer degenerates in last ~200 frames. Fix buffer recycling / re-planning when buffer head distance drops. |
| D2 | SAT corridor debugging | 3–4 hrs | SAT detection rate is 2.1%. Diagnose: is corridor too narrow? OBB orientation bug? Polygon winding issue? Use diagnostic fields + `diagnose_sat_misses.py` output. |
| D11 | Stop sign handling | 2–3 hrs | Add `_check_stop_signs()` using OBB trigger boxes (same approach as TransFuser). CARLA has stop signs in Town03 and Town05. |
| D12 | Pedestrian trajectory prediction | 2–3 hrs | Use velocity extrapolation to predict pedestrian positions 1–2s ahead. Brake for predicted path crossings, not just current position. |
| D7 | Steering noise injection | 30 min | Add Gaussian noise σ=1e-3 to expert steering commands during collection. Prevents overfitting to pixel-perfect trajectories. |

### P1 — Config & data preparation

| # | Task | Effort | Description |
|---|------|--------|-------------|
| D10 | Expand town coverage | 15 min | Add Town02, 06, 07, 10, 12, 15 to `collection.yaml`. Town02 reserved as eval-only; remaining available for training. |
| D8 | Velocity stats (initial) | 30 min | Compute `VELOCITY_MEAN`/`VELOCITY_STD` from overnight data. Update `driving_dataset.py`. Will recompute after re-collection. |
| — | Regenerate toy data | 5 min | Run `make toy-data` with CARLA for v3.1 toy dataset. |
| 19 | Phase 2a/2b in the project plan | 5 min | Split "Phase II" row into "Phase IIa" and "Phase IIb". |

### P2 — Full re-collection (Decision D1)

| Task | Description | Time |
|------|-------------|------|
| **Full re-collect** | All training towns, multiple weathers. Target: 10–15 hrs driving time. | 4–6 hrs wall (overnight) |
| **Validate** | `make validate` — schema integrity, SAT rate, nav distribution | 15 min |
| **Recompute velocity stats** | Update constants from new data (D8) | 10 min |
| **Verify improvements** | SAT rate should be significantly higher; route-end corridor healthy; stop signs handled | 30 min |

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
| Stop signs untested | Only in Town03/Town05. Verify with `test_carla_integration.py` in those towns. |

---

## Decision Summary

Key design decisions (recorded 2026-02-28):

- **Full re-collect** over filtering existing data (D1)
- **Debug SAT** to improve corridor quality (D2)
- **RWKV-7 first** with 1-week timebox → RWKV-6 fallback (D3)
- **Frozen CLIP → from-scratch PatchEmbed** progression (D4)
- **All available CARLA towns** for maximum route diversity (D10)
- **Stop sign handling + pedestrian prediction** added to expert (D11, D12)
