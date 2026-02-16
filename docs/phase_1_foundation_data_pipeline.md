# Phase I: Foundation & Data Pipeline

> **Timeline:** Weeks 1–4
> **Goal:** CARLA running reliably, data collector built, 10+ hours of expert driving data collected
> **Depends on:** Phase 0 (environment ready)

---

## Overview

This phase sets the foundation for everything. You will:
1. Get CARLA running reliably with synchronous mode
2. Build a data collection pipeline that captures expert demonstrations
3. Collect a large, balanced dataset of expert driving
4. Build the data loading infrastructure for training

The quality of your data pipeline directly determines training success. Spend the time to get this right.

---

## Steps

### 1.1 — CARLA Server Setup & Validation (Week 1)

**Objective:** CARLA server running in synchronous mode with deterministic traffic.

**Tasks:**
- [ ] Install CARLA 0.9.15 server
- [ ] Verify server starts and accepts Python client connections
- [ ] Implement basic spawn script: spawn ego vehicle, attach RGB camera, drive manually
- [ ] Configure synchronous mode (server waits for client tick — critical for data consistency)
- [ ] Test with basic `BasicAgent` autopilot to confirm agent API works
- [ ] Explore towns: Town01, Town03, Town04, Town05 (training), Town02 (held-out eval)

**Key code artifact:** `scripts/test_carla_connection.py`

```python
# Minimal connection test structure (pseudocode):
# 1. Connect to CARLA server
# 2. Set synchronous mode, fixed time step (0.05s = 20 FPS)
# 3. Spawn ego vehicle + RGB camera (224x224)
# 4. Run 100 ticks, save one frame to verify
# 5. Clean up and disconnect
```

**Synchronous mode is critical** — without it, the server runs at variable FPS and your data timestamps will be inconsistent. Always use:
```python
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05  # 20 FPS
world.apply_settings(settings)
```

### 1.2 — Expert Data Collector (Week 2–3)

**Objective:** Build a robust data collection pipeline using CARLA's autopilot.

**Tasks:**
- [ ] Implement `ExpertDataCollector` class
- [ ] Use `BasicAgent` or `BehaviorAgent` for goal-directed expert driving
- [ ] Capture per-frame: RGB image (224×224, uint8), steering, throttle, brake, velocity, navigation command (RoadOption)
- [ ] Save episodes to HDF5 format for efficient random access during training
- [ ] Implement route generation: random start/end points with waypoint planning
- [ ] Add episode metadata: town, weather, length, route info

**Key code artifact:** `src/data/expert_collector.py`

**HDF5 Dataset Structure:**
```
episode_0001.h5
├── images/          # (N, 224, 224, 3) uint8
├── steering/        # (N,) float32
├── throttle/        # (N,) float32
├── brake/           # (N,) float32
├── velocity/        # (N, 3) float32 [vx, vy, vz]
├── nav_command/     # (N,) int32 [RoadOption enum]
└── metadata/
    ├── town         # string
    ├── weather      # string
    ├── fps          # int
    └── n_frames     # int
```

**Navigation Commands (RoadOption):**
- 0: LaneFollow
- 1: Left
- 2: Right
- 3: Straight

### 1.3 — Data Collection Campaign (Week 3–4)

**Objective:** Collect 10+ hours of balanced expert driving data.

**Tasks:**
- [ ] Run collection across Towns 01, 03, 04, 05
- [ ] Vary weather conditions (Clear, Rain, Fog, Sunset, etc.)
- [ ] Vary traffic density (sparse, medium, dense)
- [ ] Monitor and balance turn distribution (Left/Right are underrepresented)
- [ ] Aim for **720,000+ frames** (~10 hrs at 20 FPS)
- [ ] Validate data quality: spot-check images, verify steering distributions, check for stuck episodes

**Collection strategy for balance:**
```
Target distribution:
  LaneFollow:  ~60%
  Left turns:  ~15%
  Right turns: ~15%
  Straight:    ~10%
```

If turns are underrepresented (they will be), run additional collection with route plans that force more intersections.

**Storage estimate:**
- 720K frames × 224×224×3 bytes = ~103 GB (raw uint8)
- With HDF5 chunked compression (gzip level 4): ~30-50 GB
- Plus controls/metadata: negligible

### 1.4 — Dataset Loader & Preprocessing (Week 4)

**Objective:** Build PyTorch `Dataset` and `DataLoader` for sequential training.

**Tasks:**
- [ ] Implement `DrivingSequenceDataset` — returns sequences of (frames, controls) for TBPTT
- [ ] Sequence length: 32 frames per clip
- [ ] ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- [ ] Data augmentation: color jitter, slight random crop, horizontal flip (with steering sign flip)
- [ ] Implement balanced sampling: weight episodes by turn density to equalize distribution
- [ ] Benchmark data loading speed — ensure GPU is not I/O-starved
- [ ] Train/val split: 80/20 by episode (not by frame, to avoid leaking sequence info)

**Key code artifact:** `src/data/driving_dataset.py`

**Critical design decision:** The `Dataset` must return **sequences**, not individual frames. Each item should be a 32-frame clip with corresponding control signals. This is because RWKV needs sequential input for its recurrent state.

```python
# Pseudocode for a single training item:
class DrivingSequenceDataset(Dataset):
    def __getitem__(self, idx):
        # Returns:
        #   images: (seq_len, 3, 224, 224) float32
        #   controls: (seq_len, 2) float32 [steer, throttle]
        #   nav_cmd: (seq_len,) int64
        #   velocity: (seq_len, 3) float32
```

---

## Completion Criteria

All of the following must be true before starting Phase II:

1. CARLA runs reliably in synchronous mode
2. Expert data collector produces consistent, valid HDF5 episodes
3. 10+ hours of driving data collected across 4 towns with varied conditions
4. Turn distribution is reasonably balanced (no class < 10%)
5. `DrivingSequenceDataset` loads data correctly and feeds batches to GPU at >100 samples/sec
6. Train/val split is determined and documented
7. All data collection code is committed and documented
