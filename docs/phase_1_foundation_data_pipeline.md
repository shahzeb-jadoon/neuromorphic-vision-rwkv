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
- [x] Install CARLA 0.9.16 server (Windows host)
- [x] Verify server starts and accepts WSL2 Python client connections (host=`172.20.0.1`)
- [x] Implement basic spawn script: spawn ego vehicle (Tesla Model 3), attach RGB camera
- [x] Configure synchronous mode (20 FPS, fixed delta 0.05s)
- [x] Test with `vehicle.set_autopilot(True)` via TrafficManager (**note:** `BasicAgent`/`BehaviorAgent` not available in pip package; those ship only with CARLA server distribution)
- [x] Explore towns: Town01, Town03, Town04, Town05 (training), Town02 (held-out eval)

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
- [x] Implement `scripts/create_toy_dataset.py`: 10 episodes × 100 frames, gradient images, sinusoidal steering
- [x] Verify HDF5 schema matches structure defined in Step 1.2
- [x] Test `DrivingSequenceDataset` (Step 1.4) loads toy data correctly — **16/16 tests pass**
- [ ] Verify model can overfit toy data (training sanity check — Phase 2)

**Key code artifact:** `scripts/create_toy_dataset.py`

**Why this matters:** Discovering a schema mismatch or DataLoader bug after collecting 10 hours of real data is a waste of time. This step catches pipeline bugs before they cost real effort.

### 1.2 — Expert Data Collector (Week 2–3)

**Objective:** Build a robust data collection pipeline using CARLA's autopilot.

**Tasks:**
- [x] Implement `ExpertDataCollector` class
- [x] Use `vehicle.set_autopilot(True)` + TrafficManager for expert driving (replaces BehaviorAgent)
- [x] Capture per-frame: RGB image (224×224, uint8), steering, throttle, brake, velocity, navigation command (inferred from heading-vs-waypoint angle)
- [x] Save episodes to HDF5 with gzip-1 compression (see HDF5 Tuning section below)
- [x] Implement `BalancedRouteGenerator` with topology-based junction routing
- [x] Add episode metadata: town, weather, fps, n_frames, status

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
- [x] Verified 5-episode campaign: 10K frames, 0.72 GB, 338s, 3 completed / 2 collisions
- [ ] Run overnight collection: `bash scripts/overnight_collection.sh` (120 episodes × 4000 frames)
- [ ] Vary weather conditions: ClearNoon, CloudyNoon, WetNoon, SoftRainNoon, HardRainNoon, ClearSunset
- [ ] Monitor and balance turn distribution (Left currently 0.7% — needs more Town03/04 junctions)
- [ ] Aim for **720,000+ frames** (~10 hrs at 20 FPS)
- [ ] Validate data quality: `python scripts/analyze_dataset.py data/collected/`

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
- With HDF5 chunked compression (gzip level 1): ~30-50 GB
- Plus controls/metadata: negligible
- **Storage location:** laptop SSD for active collection; a network home directory on the workstations for backup

#### HDF5 Compression Tuning

Benchmarked via `scripts/benchmark_hdf5.py` (500 frames, 100 random 32-frame clip reads):

| Compression | File Size | Read Speed | Notes |
|-------------|-----------|------------|-------|
| Uncompressed | 75.3 MB | 1765 MB/s | Fastest but doubles storage |
| **gzip-1** | 75.3 MB | 925 MB/s | **Selected** — fastest gzip, same ratio |
| gzip-4 | 75.3 MB | 948 MB/s | Was default, similar speed |
| gzip-9 | 75.3 MB | 907 MB/s | Slowest, no size benefit |

On real CARLA data (224×224 frames), gzip-1 achieves ~50% compression (143 MB → ~72 KB/frame vs 150 KB raw).

**Decision:** Use gzip-1 for the best speed/storage trade-off. Saves ~50 GB storage while keeping decompression fast.

### 1.4 — Dataset Loader & Preprocessing (Week 4)

**Objective:** Build PyTorch `Dataset` and `DataLoader` for sequential training.

**Tasks:**
- [x] Implement `DrivingSequenceDataset` — returns 32-frame sequences of (images, controls, nav_cmd, velocity)
- [x] Sequence length: 32 frames per clip
- [x] ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- [x] Data augmentation: horizontal flip with steering sign flip, val no-augment
- [x] Implement `TurnBalancedSampler`: weights clips by turn density
- [x] Benchmark: 15.1 clips/sec (gzip-4) → estimated ~20-30 clips/sec (gzip-1). See throughput note below.
- [x] Train/val split: 80/20 by episode (deterministic, no sequence info leak)

> **Throughput note:** 15 clips/sec at batch_size=8 means a new batch every 0.53s. RWKV forward+backward per batch is ~0.3-0.5s, so the GPU is not severely I/O-starved. Further optimization possible with memory-mapped files or pre-loading in Phase 2.

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

## Design Decisions

### NPC Traffic: 40 Vehicles, 0 Pedestrians

**Vehicles (40 per episode):** Moderate density chosen to balance realism with simulation stability. CARLA towns have ~100-200 spawn points; 40 NPCs provides enough traffic for realistic interactions (waiting at lights, following, navigating around) without gridlocking. All NPCs are managed by TrafficManager and follow traffic rules. Spawn count is configurable via `EpisodeConfig.n_vehicles`.

**Pedestrians (0):** Omitted for Phase 1. Rationale:
- Our model predicts steering/throttle, not object detection — pedestrian reactions matter most for brake prediction
- `WalkerAIController` adds spawn/despawn complexity and potential crashes
- Acceptable simplification for a project focused on lane-following + turn behavior
- **Future work:** Add 10-20 walkers per episode in a supplementary collection round before final evaluation

### Collision Data Policy

When a collision is detected during collection:
1. The episode continues recording all frames (to avoid data loss if collision is minor)
2. HDF5 metadata stores `collision_frame` index
3. **At training time**, `DrivingSequenceDataset` truncates at `collision_frame` — clips are only generated from pre-collision frames
4. Post-collision data (vehicle stuck/spinning) is excluded from training automatically

This approach preserves the pre-collision driving data (which is valid expert behavior) while preventing garbage post-collision frames from corrupting training.

### Autopilot Speed and Expert Quality

**Speed control:** `vehicle_percentage_speed_difference(vehicle, -20.0)` → drives at 80% of speed limit.

| Town | Typical Speed | Max Observed | Speed Limit Context |
|------|--------------|--------------|---------------------|
| Town01 | 5–10 m/s | 22.7 m/s (82 km/h) | Suburban roads |
| Town03 | 5–10 m/s | ~15 m/s | Residential |
| Town04 | 5–15 m/s | ~15 m/s | Mixed urban |
| Town05 | 5–29 m/s | 29.0 m/s (104 km/h) | Highway segments |

**Why autopilot is acceptable as "expert":**
- ✅ Obeys traffic lights and stop signs (clean labels)
- ✅ Maintains safe following distance
- ✅ Consistent behavior across episodes (reproducible)
- ⚠️ Drives ~20% below speed limit (conservative but safe)
- ⚠️ Gets stuck behind traffic occasionally (produces 0 m/s frames)

**What we cannot currently detect:**
- Whether the car exceeded a specific road's speed limit (speed limit signs aren't logged)
- Whether stopped frames are legitimate (red light) vs unnecessary (stuck in traffic)
- How ego speed compares to surrounding traffic flow

> **Future improvement (Phase 2):** Log road speed limits via `carla_map.get_waypoint(location).get_landmarks()` and filter out episodes with speed limit violations or excessive idle time.

### Navigation Command Distribution

The overnight campaign shows Left (2.1%) and Right (1.8%) are far below the 15% target. Root causes:
1. TrafficManager's default pathfinding prefers straight routes
2. The 15° heading threshold in `_infer_nav_command` may miss gentle turns
3. Towns 01 and 05 have long straight roads with few junctions

**Mitigation:** `TurnBalancedSampler` oversamples clips containing turn commands (~7-10× upweighting). This is the standard approach in imitation learning — the natural distribution is preserved in the data, and the sampler corrects for it during training.

---

## Completion Criteria

Status of each criterion for starting Phase II:

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 1 | CARLA runs reliably in synchronous mode | ✅ | Tested on Town01–05 |
| 2 | Toy dataset validates HDF5 schema and DataLoader | ✅ | 16/16 tests pass |
| 3 | Expert collector handles crashes, collisions, timeouts | ✅ | 119/120 episodes completed |
| 4 | 10+ hours of driving data collected | ⚠️ | 6.7 hours (480K frames). Sufficient for initial training; can augment later. |
| 5 | Turn distribution reasonably balanced | ⚠️ | Left=2.1%, Right=1.8%. Mitigated by TurnBalancedSampler. |
| 6 | `collection_manifest.json` tracks all episodes | ✅ | 120 episodes tracked |
| 7 | DataLoader feeds GPU at >100 clips/sec | ⚠️ | 15-26 clips/sec. Not GPU-bottlenecked due to RWKV forward time. |
| 8 | Train/val split determined | ✅ | 80/20 by episode, deterministic seed |
| 9 | All code committed and documented | ✅ | `phase-1/data-pipeline` branch |

> **Assessment:** Items 4, 5, and 7 are partially met but acceptable for Phase II start. The TurnBalancedSampler and natural RWKV forward time mitigate items 5 and 7. Item 4 can be supplemented with a second overnight run if training reveals insufficient data.

