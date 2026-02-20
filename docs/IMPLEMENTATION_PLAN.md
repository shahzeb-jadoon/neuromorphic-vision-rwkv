# Implementation Plan — Remaining Work

> **Last updated:** 2026-02-20
> **Schema:** v3 — 3-camera panoramic (3×640×320, 60° FOV, yaws −60/0/+60°),
>   PrivilegedExpert agent, RoadOption nav commands, privileged metadata per frame.
> **Branch:** `feat/privileged-expert-v3`

---

## Current Baseline

```
103 offline tests passing (5 online deselected in CI)
87.5% coverage on testable code (threshold: 80%)
7 logical commits on feat/privileged-expert-v3
Schema v3 pipeline fully operational
All old v1/v2 data deleted (local + SSD)
```

---

## Resolved Issues (35 of 35 + 6 second-pass)

All Phase 1 issues are resolved.

| Range | Summary |
|-------|---------|
| 1–2, 7, 9–10, 23–25 | Schema v3 migration: panoramic cameras, route-based nav commands, balanced routes, position data, weighted sampler, augmentation order |
| 26–29 | PrivilegedExpert fixes: junction-aware traffic lights, OBB/SAT collision detection, pedestrian lateral filtering, break logic comment |
| 30–35 | Pipeline cleanup: dead test deleted, old data purged, schema gate, walker spawning, coverage config, package discovery |
| 2A–2F | Second-pass: config key mismatch, camera yaws, HDF5 caching, velocity normalisation, README qualifier, commit split |
| 13–14 | CI: guards removed, lint scope expanded, offline/online split |
| 17–18 | Coverage threshold, README qualifier |

---

## Immediate — Low-Effort Cleanup

| # | Issue | Priority | Effort | Description |
|---|-------|----------|--------|-------------|
| 12r | `sys.path.insert` hack | Low | 1 min | Remove `sys.path.insert(0, ...)` from `scripts/validate_dataset.py` L30. Editable install handles imports. |
| 19 | Phase 2a/2b in the project plan | Medium | 5 min | Split "Phase II" row into "Phase IIa" and "Phase IIb" in the project plan. |
| — | Velocity stats recomputation | Medium | 10 min | After first real collection: update `VELOCITY_MEAN`/`VELOCITY_STD`/`VELOCITY_STATS_VERSION` in `driving_dataset.py`. |
| — | Regenerate toy data | Medium | 5 min | Run `make toy-data` with CARLA to create v3 toy dataset. |

---

## Phase 1 Completion — Data Collection

Before Phase 2, collect the training dataset:

| Task | Description | Time |
|------|-------------|------|
| **First collection** | `python scripts/collect_data.py --spectator` — Town01/02/03, multiple weathers. Target: 5–10 hrs driving. | 2–4 hrs wall |
| **Validate** | `make validate` — schema integrity, image quality, nav distribution | 10 min |
| **Recompute velocity stats** | `scripts/analyze_dataset.py` → update constants | 10 min |
| **Verify nav distribution** | Confirm ≥15% LEFT, ≥15% RIGHT | Part of analyze |

---

## Phase 2a — Architecture & Training Infrastructure

> **Weeks 5–6.** Vision-RWKV architecture + training pipeline.

| # | Task | Priority | Description |
|---|------|----------|-------------|
| 5 | RWKV-7 implementation | P0 | 1-week timebox. Fallback: RWKV-6. |
| 6 | Experiment configs | P1 | `configs/experiments/*.yaml` per run. |
| — | Trainer class | P0 | TBPTT, gradient accumulation, AMP, state carry-over. |
| — | Checkpointing | P1 | Save/resume with optimizer, LR schedule, RWKV state. |
| — | W&B integration | P1 | Loss, LR, gradients, image samples. |

---

## Phase 2b — Training & Results

> **Weeks 6–8.** Train model on collected data.

| # | Task | Priority | Description |
|---|------|----------|-------------|
| 4 | Noise injection | P1 | Gaussian noise on steering (σ=0.01→0.05). |
| 21 | Multi-resolution ablation | P2 | 224/256/384/512 — FLOPs, latency, driving metrics. |
| — | 3-stage training | P0 | Frozen backbone → unfreeze → fine-tune. |

---

## Phase 3 — Evaluation

> **Weeks 8–10.** Baselines, ablations, closed-loop.

| # | Task | Priority | Description |
|---|------|----------|-------------|
| 8 | Nav command ablation | P1 | Remove nav input, retrain, compare. |
| — | Baseline comparisons | P0 | ResNet-18, ViT-Small, LSTM. |
| — | Closed-loop eval | P0 | Route Completion >70%, Collision <0.5/km. |
| 4+ | DAgger (stretch) | P2 | Expert corrections → retrain. |

---

## Phase 4 — Documentation & Deployment

> **Weeks 11–12.** Write-up, Jetson, demo.

| # | Task | Priority | Description |
|---|------|----------|-------------|
| 20 | Lit review → LaTeX | P0 | Convert markdown to a paper chapter. |
| 22 | η_AF grounding | P1 | Literature-ground neuromorphic metric. |
| — | Jetson deployment | P1 | ONNX/TensorRT, <50ms latency. |
| — | Write-up + demo | P0 | All chapters, results, figures, video. |

---

## Low-Priority / Optional

| # | Notes |
|---|-------|
| 15 | Pin mypy exactly — only if a mypy upgrade breaks CI. |
| 16 | `make lock` target — low value for conda-managed solo project. |

---

## Risk Register

| Risk | Mitigation |
|------|------------|
| RWKV-7 CUDA kernels fail | 1-week pivot to RWKV-6. Infra is version-agnostic. |
| Collection takes too long | Overnight runs. 50 hrs at 20× ≈ 2.5 hrs wall. |
| Velocity placeholder stats | `VELOCITY_STATS_VERSION` flag. Recompute after collection. |
| Training instability | Short TBPTT segments (T=32), W&B gradient monitoring. |
| Distribution drift | DAgger (stretch) + noise injection (Issue 4). |
