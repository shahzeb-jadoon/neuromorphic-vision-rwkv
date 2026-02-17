# Phase I: Foundation & Data Pipeline

> **Timeline:** Weeks 1–4
> **Goal:** CARLA running reliably, data collector built, 10+ hours of expert driving data collected
> **Depends on:** Phase 0 (environment ready)

---

## Overview

This phase sets the foundation for everything. You will:

- **CARLA integration** — Get CARLA 0.9.16 running reliably with synchronous mode
- **Pipeline validation** — Create a toy dataset to validate the HDF5 schema and DataLoader before collecting real data
- **Data collection** — Build a robust expert data collector and collect 10+ hours of balanced driving data
- **Training infrastructure** — Build the data loading pipeline for sequential RWKV training

The quality of your data pipeline directly determines training success. Spend the time to get this right.

---

## Steps

### 1.1 — CARLA Server Setup & Validation (Week 1)

**Objective:** CARLA server running in synchronous mode with deterministic traffic.

**Tasks:**
- [ ] Install CARLA 0.9.16 server
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

### 1.1b — Toy Dataset for Pipeline Validation (Week 1)

**Objective:** Create a small synthetic dataset to validate the HDF5 schema, DataLoader, and training pipeline before collecting real CARLA data.

**Tasks:**
- [ ] Implement `scripts/create_toy_dataset.py`: generate 10 fake episodes, 100 frames each, with synthetic patterns (e.g., sinusoidal steering)
- [ ] Verify HDF5 schema matches the structure defined in Step 1.2
- [ ] Test `DrivingSequenceDataset` (Step 1.4) loads toy data correctly
- [ ] Verify model can overfit toy data (training sanity check — confirms gradient flow end-to-end)

**Key code artifact:** `scripts/create_toy_dataset.py`

**Why this matters:** Discovering a schema mismatch or DataLoader bug after collecting 10 hours of real data is a waste of time. This step catches pipeline bugs before they cost real effort.

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
episode_{sha256[:8]}.h5
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
    ├── n_frames     # int
    └── status       # string: "completed" | "collision" | "crashed" | "timeout"
```

**Navigation Commands (RoadOption):**
- 0: LaneFollow
- 1: Left
- 2: Right
- 3: Straight

#### Robustness & Error Handling

The data collector must handle three failure modes gracefully:

**CARLA crash recovery:**
- Wrap the tick loop in try/except. On exception, save the partial episode with `status: crashed` in metadata, log the crash type and stack trace, and move to the next episode.
- If CARLA server disconnects entirely, wait 30 seconds and attempt reconnection (3 retries).

**Episode termination policy:**
| Outcome | Condition | Action |
|---------|-----------|--------|
| `completed` | Route finished, no collisions | Save episode normally |
| `collision` | Ego vehicle collides with another actor or static object | Save episode, tag `status: collision` in metadata. Include collision frame index. |
| `timeout` | Vehicle stuck for >3 minutes (speed < 0.1 m/s continuously) | Discard episode. Log timeout location for debugging. |

**Interrupted campaign resumption:**
- Maintain a `collection_manifest.json` tracking every episode: hash, town, weather, frame count, turn distribution, status, timestamp.
- On restart, read the manifest and skip already-completed town/weather combinations. Resume from the last incomplete configuration.
- Manifest enables exact dataset reconstruction months later (data versioning).

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

#### Turn-Balanced Route Generation

CARLA's autopilot drives naturally (mostly straight), producing ~80% LaneFollow data. To achieve the target distribution:

- [ ] Implement `BalancedRouteGenerator` using CARLA's topology API (`world.get_map().get_topology()`)
- [ ] Find intersections with desired exit options (left/right turns) and chain waypoints to force the autopilot through them
- [ ] CLI flags: `--route-strategy balanced --target-left 0.15 --target-right 0.15`
- [ ] Post-collection: run `scripts/analyze_dataset.py` to verify actual distribution matches targets

**Key code artifact:** `src/data/route_generator.py`

#### Data Versioning

- Episode files use hash-based naming: `episode_{sha256[:8]}.h5`
- `collection_manifest.json` tracks per-episode: hash, collection date, town, weather, frame count, turn distribution
- Enables exact dataset identification and reconstruction months later

**Storage estimate:**
- 720K frames × 224×224×3 bytes = ~103 GB (raw uint8)
- With HDF5 chunked compression (gzip level 4): ~30-50 GB
- Plus controls/metadata: negligible
- **Storage location:** laptop SSD for active collection; a network home directory on the workstations for backup

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
2. Toy dataset validates HDF5 schema the and DataLoader pipeline
3. Expert data collector handles crashes, collisions, and timeouts gracefully
4. 10+ hours of driving data collected across 4 towns with varied conditions
5. Turn distribution is reasonably balanced (no class < 10%)
6. `collection_manifest.json` tracks all episodes with hashes
7. `DrivingSequenceDataset` loads data correctly and feeds batches to GPU at >100 samples/sec
8. Train/val split is determined and documented
9. All data collection code is committed and documented
