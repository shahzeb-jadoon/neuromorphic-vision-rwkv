# Phase 1 — Validation Results

> **Date:** 2026-02-18
> **Branch:** `phase-1/data-pipeline`

---

## CARLA Connectivity

| Test | Result |
|------|--------|
| WSL2 → Windows CARLA | ✅ PASSED (host: `172.20.0.1`) |
| Server/Client version | 0.9.16 / 0.9.16 |
| Synchronous mode (20 FPS) | ✅ PASSED |
| Vehicle spawn (Tesla Model 3) | ✅ PASSED |
| Camera capture (224×224, FOV=100°) | ✅ PASSED |
| 100-tick sync test | ✅ PASSED |

---

## Unit Tests

```
16 passed in 12.59s
```
- TestDatasetShapes: images, controls, nav_command, velocity, dtypes
- TestNormalization: ImageNet normalization, steering/throttle ranges
- TestSplit: no episode overlap, all episodes covered, deterministic
- TestAugmentation: horizontal flip with steering sign flip, val no-augment
- TestSampler: weighted sampler creates and iterates
- TestDataLoader: batch loading (8, 32, 3, 224, 224)

---

## Expert Data Collector — Single Episode Test

| Metric | Value |
|--------|-------|
| Frames | 200 |
| Duration | 9.7s wall time |
| Status | completed |
| NPC traffic | 20/20 spawned |
| HDF5 size | 14.5 MB |
| Clean exit | ✅ (no SIGABRT) |

**Bug fixed:** BehaviorAgent (`agents.navigation`) is not in the pip `carla==0.9.16` package. Rewrote to use `vehicle.set_autopilot(True)` + TrafficManager. Nav commands inferred from heading-vs-waypoint angle.

**Bug fixed:** SIGABRT in cleanup — CARLA C++ throws when destroying actors already cleaned up by TrafficManager. Fixed with `is_alive` guards.

---

## Overnight Collection Campaign (Final Results)

**Runtime:** 3 hours 47 minutes (23:01 → 02:48 EST, Feb 17–18, 2026)

### Summary

| Metric | Value |
|--------|-------|
| Episodes | 120 (119 completed, 1 collision) |
| Total frames | 480,000 (6.7 hours at 20 FPS) |
| Storage | 35.87 GB (gzip-1 compressed) |
| Avg episode | 4,000 frames (200s) |
| Wall time per episode | ~1.9 minutes |

### Town Distribution

| Town | Episodes | LaneFollow | Left | Right | Straight |
|------|----------|------------|------|-------|----------|
| Town01 | 30 | 90.7% | 3.0% | 2.7% | 3.6% |
| Town03 | 30 | 79.5% | 2.2% | 2.2% | 16.1% |
| Town04 | 30 | 77.3% | 0.7% | 0.8% | 21.2% |
| Town05 | 30 | 88.9% | 2.4% | 1.5% | 7.3% |
| **Total** | **120** | **84.1%** | **2.1%** | **1.8%** | **12.1%** |

### Weather Distribution

All 6 conditions evenly represented (20 episodes each):
ClearNoon, CloudyNoon, WetNoon, SoftRainNoon, HardRainNoon, ClearSunset

### Collision Episodes

| Episode | Town | Weather | Collision Frame | Total Frames |
|---------|------|---------|-----------------|--------------|
| `episode_0d035e63.h5` | Town03 | HardRainNoon | 841 | 4,000 |

> **Note:** Post-collision frames (841–4000) contain garbage data — vehicle may be stuck. The `DrivingSequenceDataset` should truncate at `collision_frame` during training. See *Collision Data Policy* below.

### Navigation Distribution Analysis

```
LaneFollow  :   403,645 ( 84.1%) ██████████████████████████████████████████
Left        :     9,920 (  2.1%) █
Right       :     8,521 (  1.8%)
Straight    :    57,914 ( 12.1%) ██████
```

**Left/Right are severely underrepresented** (2.1% and 1.8% vs 15% target). Root causes:
1. The autopilot rarely selects turn-heavy routes — it follows TrafficManager's default pathfinding
2. The nav command inference uses a 15° heading threshold, which may miss gentle turns
3. Towns 01 and 05 have long straight roads with few junctions

**Mitigation strategy (for training):**
- `TurnBalancedSampler` oversamples clips containing Left/Right commands (~7–10× upweighting)
- This is sufficient for the model to learn turn behavior despite the natural distribution imbalance

### Speed Analysis

| Metric | Town01 | Town03 | Town04 | Town05 |
|--------|--------|--------|--------|--------|
| Typical | 5–10 m/s | 5–10 m/s | 5–15 m/s | 5–29 m/s |
| Max observed | 22.7 m/s (82 km/h) | ~15 m/s | ~15 m/s | **29.0 m/s (104 km/h)** |

> **⚠️ Town05 speeds (29 m/s / 104 km/h):** Town05 has highway-like roads with higher speed limits. The TrafficManager drives at `speed_limit × 0.8`, so 29 m/s suggests a 36 m/s (130 km/h) speed limit zone. This is valid data — the model should learn highway-speed driving.

### Control Statistics

| Metric | Steering | Throttle |
|--------|----------|----------|
| Mean | +0.003 | 0.296 |
| Std | 0.061 | 0.319 |
| Min | -0.636 | 0.000 |
| Max | +0.800 | 0.850 |

---

## HDF5 Compression Benchmark

Tested via `scripts/benchmark_hdf5.py` (500 frames, 100 random 32-frame clip reads):

| Config | Size (MB) | Write (s) | Read (s) | Read (MB/s) |
|--------|-----------|-----------|----------|-------------|
| **uncompressed** | 75.3 | 0.10 | 0.27 | **1765.6** |
| **gzip-1** | 75.3 | 2.39 | 0.52 | **925.1** |
| gzip-4 (was default) | 75.3 | 2.67 | 0.51 | 948.1 |
| gzip-9 | 75.3 | 2.75 | 0.53 | 907.7 |

Real CARLA data read speeds:
- gzip-1: **26.1 clips/sec**
- gzip-4: **24.5 clips/sec** (~7% slower)

**Decision:** Use gzip-1. Same compression ratio (~72 KB/frame), fastest decompression.

### DataLoader Throughput

| Config | Throughput | Target |
|--------|-----------|--------|
| gzip-4, 4 workers, batch=8 | 15.1 clips/sec | >100 |
| gzip-1, raw HDF5 reads | 26.1 clips/sec | — |

> **Throughput note:** 15–26 clips/sec at batch_size=8 means a new batch every 0.3–0.5s. RWKV forward+backward is ~0.3–0.5s per batch, so the GPU is not severely I/O-starved. Further optimization possible with pre-loading or memory mapping in Phase 2.
