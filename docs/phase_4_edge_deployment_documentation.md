# Phase IV: Edge Deployment & Documentation

> **Timeline:** Weeks 11–12
> **Goal:** Jetson Orin Nano deployment, final report, demo video, open-source release
> **Depends on:** Phase III (evaluation complete, best model identified)

---

## Overview

This is the final phase. You will:
1. Optimize and deploy the best model to the Jetson Orin Nano
2. Profile real-hardware inference (latency, memory, power)
3. Write the final technical report
4. Record a demo video
5. Prepare the open-source release

---

## Steps

### 4.1 — Model Optimization for Edge (Week 11)

**Objective:** Reduce model size and precision for Jetson deployment.

**Tasks:**
- [ ] **FP16 conversion:** Convert trained FP32 model to FP16
  - Most straightforward — typically ~2× memory reduction with minimal accuracy loss
  - Verify: run eval on desktop in FP16 to confirm no significant degradation
- [ ] **TorchScript export:** `torch.jit.script` or `torch.jit.trace` the model
  - Required for optimized inference without Python overhead
  - The recurrent state makes tracing tricky — script may be more reliable
- [ ] **TensorRT conversion (optional stretch):**
  - Convert ONNX → TensorRT for maximum Jetson performance
  - TensorRT handles fusion, kernel selection, quantization automatically
  - INT8 calibration with representative dataset for maximum throughput
- [ ] **Model pruning (if needed):**
  - If memory > 4 GB on Jetson, prune less-important channels
  - Structured pruning preferred (remove entire channels, not individual weights)

**Key code artifact:** `scripts/export_model.py`

### 4.2 — Jetson Orin Nano Setup (Week 11)

**Objective:** Prepare the Jetson for inference.

**Tasks:**
- [ ] Flash JetPack 6.x on Jetson Orin Nano Super
- [ ] Install PyTorch for Jetson (NVIDIA provides pre-built wheels)
- [ ] Clone repo and install dependencies
- [ ] Transfer optimized model checkpoint
- [ ] Verify model loads and runs on Jetson (even if CARLA isn't connected)

### 4.3 — Jetson Inference Profiling (Week 11)

**Objective:** Measure real-hardware performance against the planned targets.

**Tasks:**
- [ ] **Latency:** Run inference for 1000 frames, report mean ± std per frame
  - Target: < 50ms (stretch: < 33ms for 30 FPS)
- [ ] **Memory:** Report peak GPU memory during inference
  - Target: < 4 GB (stretch: < 2 GB)
- [ ] **Power:** Use `tegrastats` to monitor power consumption
  - Target: < 20W (stretch: < 15W)
  - Test at different power modes (7W, 15W, 25W)
- [ ] **Throughput:** Sustained FPS over 60-second window
- [ ] Compare FP32 vs. FP16 vs. INT8 (if available) on Jetson
- [ ] Generate hardware comparison table:

| Platform | Precision | Latency (ms) | Memory (GB) | Power (W) | FPS |
|---|---|---|---|---|---|
| workstation GPU | FP32 | ? | ? | — | ? |
| workstation GPU | FP16 | ? | ? | — | ? |
| laptop GPU | FP16 | ? | ? | — | ? |
| Jetson Orin Nano | FP16 | ? | ? | ? | ? |
| Jetson Orin Nano | INT8 | ? | ? | ? | ? |

**Key code artifact:** `src/evaluation/jetson_profiler.py`

### 4.4 — Jetson + CARLA Demo (Week 11–12, stretch goal)

**Objective:** Run closed-loop driving with Jetson as the inference device.

**Tasks:**
- [ ] CARLA server runs on desktop, sends camera frames to Jetson via network
- [ ] Jetson runs inference, sends controls back to CARLA
- [ ] Record real-time driving video with overlay showing:
  - Camera feed
  - Predicted steering (visual indicator)
  - Recurrent state heatmap
  - Latency counter
  - Power consumption
- [ ] This becomes the demo video

### 4.5 — Technical Report (Week 12)

**Objective:** Complete the final project report.

**Tasks:**
- [ ] Write/update all chapters:
  - Abstract (revise with actual results)
  - Introduction (no major changes)
  - Background (no major changes)
  - Methodology (update with actual implementation details, any deviations from the plan)
  - Results & Discussion (new chapter — all tables, figures, analysis)
  - Jetson Deployment (new section or chapter)
  - Conclusion (actual conclusions from results)
  - Future Work (neuromorphic conversion, Loihi deployment, real vehicle integration)
- [ ] All figures publication-quality (vector where possible, high-DPI raster)
- [ ] All tables formatted consistently
- [ ] Bibliography complete and accurate
- [ ] Proofread for grammar, clarity, consistency

### 4.6 — Demo Video (Week 12)

**Objective:** 3-5 minute video showcasing the project.

**Contents:**
1. Problem statement (15 seconds)
2. Architecture overview diagram (30 seconds)
3. Training visualization — loss curves, prediction examples (30 seconds)
4. Driving demo — split screen: CARLA view + agent predictions (60 seconds)
5. Long-horizon memory test — show context retention (30 seconds)
6. Efficiency comparison — latency scaling chart (30 seconds)
7. Jetson deployment — live demo or profiling results (30 seconds)
8. Key results summary (15 seconds)

### 4.7 — Open-Source Release (Week 12)

**Objective:** Prepare repository for public release.

**Tasks:**
- [ ] Clean up code: remove dead code, TODOs, personal paths
- [ ] Add docstrings to all modules and public functions
- [ ] Write installation instructions in README
- [ ] Add usage examples for: training, evaluation, data collection, Jetson inference
- [ ] Add `requirements.txt` with pinned versions
- [ ] Verify a fresh clone → install → train cycle works
- [ ] Add pre-trained model weights to GitHub Releases or HuggingFace
- [ ] Consider making repo public (change visibility on GitHub)
- [ ] Add citation instructions (BibTeX entry)

---

## Completion Criteria (Project Done)

1. Best model deployed to Jetson Orin Nano with measured latency < 50ms
2. Final report complete with all results
3. Demo video recorded
4. Code repository clean and documented
5. All planned deliverables completed (or documented reasons for deviation)
6. Final review
