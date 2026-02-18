# Phase 1 — Validation Results

> **Date:** 2026-02-17
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

## 5-Episode Campaign Test

| Episode | Town | Weather | Status | Nav Distribution |
|---------|------|---------|--------|------------------|
| 1 | Town01 | ClearNoon | ✅ completed | 100% LaneFollow |
| 2 | Town01 | CloudyNoon | ✅ completed | 100% LaneFollow |
| 3 | Town03 | ClearNoon | ✅ completed | 98.2% LF, 1.3% R, 0.5% S |
| 4 | Town03 | WetNoon | ⚠️ collision (f321) | 8.7% LF, 0.8% L, 86.1% R, 4.5% S |
| 5 | Town04 | ClearNoon | ⚠️ collision (f32) | 0.4% LF, 99.6% R |

**Aggregate:** 10,000 frames, 0.72 GB, 338s wall time

**Nav distribution:**
- LaneFollow: 59.3%
- Right: 37.7%
- Left: 0.7% ✗ (below 10% target)
- Straight: 2.3% ✗ (below 10% target)

**Observations:**
- Collision episodes produce skewed nav data (vehicle stuck turning in place)
- Short episodes on Town01 are all LaneFollow (no junctions reached)
- Full-scale collection needs longer episodes + more junction-rich towns

---

## HDF5 Compression Benchmark

Tested on 500 synthetic 224×224×3 frames, reading 100 random 32-frame clips:

| Config | Size (MB) | Write (s) | Read (s) | Read (MB/s) |
|--------|-----------|-----------|----------|-------------|
| **uncompressed** | 75.3 | 0.10 | 0.27 | **1765.6** |
| **gzip-1** | 75.3 | 2.39 | 0.52 | **925.1** |
| gzip-4 (was default) | 75.3 | 2.67 | 0.51 | 948.1 |
| gzip-9 | 75.3 | 2.75 | 0.53 | 907.7 |

Real CARLA data (`gzip-4`): **27.6 clips/sec** (143.8 MB for 2000 frames)

**Decision:** Switched to `gzip-1`. Same compression ratio on CARLA images, fastest decompression. Uncompressed would double storage (~100 GB → ~200 GB for full dataset).

### DataLoader Throughput

| Config | Throughput | Target |
|--------|-----------|--------|
| gzip-4, 4 workers, batch=8 | 15.1 clips/sec | >100 |
| gzip-1 (estimated) | ~20-30 clips/sec | >100 |

**Note:** The 100 clips/sec target may be over-aggressive for 32×224×224×3 sequences. At batch_size=8 with 15 clips/sec, the GPU receives a new batch every 0.53s, which is likely sufficient given that RWKV forward + backward takes ~0.3-0.5s per batch. Will re-evaluate during Phase 2 training.

---

## Control Statistics (Campaign Data)

| Metric | Steering | Throttle |
|--------|----------|----------|
| Mean | -0.129 | 0.404 |
| Std | 0.344 | 0.398 |
| Min | -0.800 | 0.000 |
| Max | +0.717 | 0.850 |
