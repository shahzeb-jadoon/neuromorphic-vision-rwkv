# Phase III: Evaluation & Benchmarking

> **Timeline:** Weeks 8–10
> **Goal:** Quantitative comparison against baselines, ablation studies, neuromorphic analysis
> **Depends on:** Phase II (trained models)

---

## Overview

This phase shifts from building to measuring. You will:
1. Evaluate driving performance in CARLA (closed-loop)
2. Run computational efficiency benchmarks
3. Compute the neuromorphic alignment metric (η_AF)
4. Run ablation studies on key design choices
5. Generate all tables and figures for the final report

---

## Steps

### 3.1 — Closed-Loop CARLA Evaluation (Week 8)

**Objective:** Evaluate all models by actually driving in CARLA (not just prediction accuracy).

**Tasks:**
- [ ] Implement `evaluate.py`: load checkpoint, run agent in CARLA, record metrics
- [ ] Evaluation protocol:
  - Town02 (held-out, never seen during training)
  - 25+ routes of varying difficulty (short, medium, long)
  - 3 weather conditions: Clear, Rain, Sunset
  - Each route run 3× for statistical significance
- [ ] Record per-route metrics:
  - Route completion (0-100%)
  - Collisions (count + type: vehicle, pedestrian, static)
  - Red light violations
  - Stuck time (seconds stopped when should be moving)
  - Route time (total time to complete)
- [ ] Implement real-time state visualization: plot recurrent state activations as the agent drives

**Key code artifact:** `src/evaluation/closed_loop_eval.py`

**Closed-loop evaluation is the most important metric.** Prediction MSE on held-out data does not capture compounding errors — a model that is 95% accurate per frame can crash within 5 seconds. Only closed-loop evaluation reveals true driving capability.

### 3.2 — Baseline Comparison (Week 8–9)

**Objective:** Run the same evaluation protocol on all baselines.

**Tasks:**
- [ ] Evaluate CNN baseline (ResNet-18 + MLP) on identical routes
- [ ] Evaluate Transformer baseline (ViT-B/16 + sliding window) on identical routes
- [ ] Evaluate LSTM baseline (ResNet-18 + LSTM) on identical routes
- [ ] Compile results into comparison table:

| Metric | CNN | ViT (Window) | LSTM | RWKV (Ours) |
|---|---|---|---|---|
| Route Completion (%) | ? | ? | ? | ? |
| Collision Rate (/km) | ? | ? | ? | ? |
| Red Light Violations | ? | ? | ? | ? |
| Intersection Success (%) | ? | ? | ? | ? |

### 3.3 — Computational Efficiency Profiling (Week 9)

**Objective:** Benchmark inference speed, memory, and how they scale with context length.

**Tasks:**
- [ ] Profile on the 16 GB laptop GPU and the 24 GB workstation GPU:
  - Inference latency per frame (ms)
  - Peak GPU memory (MB)
  - Throughput (FPS)
- [ ] **Critical test — context length scaling:**
  - Run inference for 10, 50, 100, 500, 1000 frames and measure latency per frame
  - RWKV should show O(1) — flat latency regardless of history length
  - Transformer should show increasing latency (sliding window helps but still)
- [ ] Generate scaling plot: latency vs. context length for all 4 models
- [ ] Profile FLOPs using `torch.profiler` or `fvcore`
- [ ] Memory breakdown: model parameters vs. activations vs. state

**Key code artifact:** `src/evaluation/efficiency_profiler.py`

**This is the strongest selling point of the project.** If you can show a flat line for RWKV latency vs. an increasing line for Transformer, that's the core visual proof of the O(1) inference claim.

### 3.4 — Long-Horizon Context Tests (Week 9)

**Objective:** Demonstrate that RWKV's recurrent state actually retains useful information over time.

**Tasks:**
- [ ] Design context-dependent driving scenarios:
  - **GPS-denied navigation:** Give a turn command 30+ seconds before the intersection. The model must remember the instruction when it arrives.
  - **Command persistence:** Issue "turn left" and drive straight for extended periods. Does the model execute the turn when it reaches an intersection?
  - **Obstacle memory:** Show an obstacle, occlude it, then approach. Does the model still slow down?
- [ ] Compare RWKV vs. Transformer (limited window) on these scenarios
- [ ] Visualize state activations over time to show information retention

### 3.5 — Neuromorphic Alignment Analysis (Week 10)

**Objective:** Compute the Accumulate-and-Fire Efficiency (η_AF) metric and analyze state sparsity.

**Tasks:**
- [ ] Implement η_AF computation:
  ```
  η_AF = (1 / (T × L × C)) × Σ_{t,l,c} 𝟙(|K·V·σ(g)| < ε)
  ```
  where ε = 10⁻⁴, T = timesteps, L = layers, C = channels
- [ ] Run η_AF on evaluation episodes, report mean ± std
- [ ] Target: η_AF > 60% (from the plan)
- [ ] Generate state activation heatmaps: layer × time visualizations showing which state channels are "silent" (below threshold)
- [ ] Analyze per-layer sparsity: do deeper layers have higher or lower η_AF?
- [ ] Discuss implications for neuromorphic conversion: high η_AF means many state updates can be skipped on spiking hardware

**Key code artifact:** `src/evaluation/neuromorphic_analysis.py`

### 3.6 — Ablation Studies (Week 10)

**Objective:** Isolate the contribution of each component.

**Tasks:**
- [ ] **Q-Shift ablation:** RWKV with vs. without Q-Shift spatial mixing
  - Does Q-Shift improve over naive raster-scan ordering?
- [ ] **Burn-in length ablation:** K=0, K=10, K=20, K=30
  - What is the minimum burn-in for stable performance?
- [ ] **State decay initialization:** w=0.5, w=0.9, w=0.99, w=learned
  - How sensitive is driving performance to the decay constant?
- [ ] **Model size ablation:** 6 layers vs. 12 layers vs. 18 layers
  - Diminishing returns analysis
- [ ] Compile ablation table:

| Variant | Route Completion | Latency | Memory |
|---|---|---|---|
| Full model (12L, Q-Shift, K=20) | ? | ? | ? |
| No Q-Shift | ? | ? | ? |
| K=0 (no burn-in) | ? | ? | ? |
| 6 layers | ? | ? | ? |

---

## Completion Criteria

All of the following must be true before starting Phase IV:

1. Closed-loop evaluation completed on Town02 for all 4 models
2. RWKV route completion ≥ 70% (primary target)
3. Efficiency profiling shows O(1) inference scaling for RWKV
4. η_AF computed and documented (target > 60%)
5. At least 3 ablation studies completed
6. All comparison tables and key figures generated
7. Results are reproducible (evaluation scripts committed with fixed seeds)
