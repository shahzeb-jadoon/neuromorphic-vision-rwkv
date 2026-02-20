# Implementation Plan: All Issues & Decisions

> **Last updated:** Post-review, pre-commit sprint
> **Schema:** v3 — 3-camera panoramic (3×640×320, 60° FOV, yaws −60/0/+60°),
>   PrivilegedExpert agent, RoadOption nav commands, privileged metadata per frame.
> **Branch:** `feat/schema-v2-dual-nav-labels` (name outdated — will rename during commit split)

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Resolved / implemented |
| 🔧 | In progress this sprint |
| 🔲 | Planned, not yet started |

---

## Decision Summary

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| 1 | Turn distribution (2% → 15%) | ✅ | BalancedRouteGenerator ensures 15% L/R. Old 180-ep v2 data on SSD to be deleted. |
| 2 | Nav command labels (15.1% error) | ✅ | Single `nav_command` from PrivilegedExpert route (RoadOption enum). Dual labels removed. |
| 3 | Velocity normalization | 🔧 **Issue 2D** | Z-score normalize velocity in `DrivingSequenceDataset.__getitem__`. |
| 4 | Collision handling | 🔲 Phase 2b/3 | Noise injection + DAgger. |
| 5 | RWKV-7 vs RWKV-6 | 🔲 Phase 2a | RWKV-7 first, 1-week timebox, fallback to RWKV-6. |
| 6 | Experiment config structure | 🔲 Phase 2a | `configs/experiments/*.yaml` per run. |
| 7 | Camera resolution | ✅ | 3-camera panoramic: 3×640×320 → 1920×320 stitched. |
| 8 | Nav oracle ablation | 🔲 Phase 3 | Remove nav_command, retrain, compare metrics. |
| 9 | Color jitter ordering | ✅ | Jitter applied BEFORE ImageNet normalization in `_augment()`. |
| 10 | Weighted sampler per-clip | ✅ | Per-clip weighting by nav_command inverse-LaneFollow frequency. |
| 11 | HDF5 per-worker caching | 🔧 **Issue 2C** | `worker_init_fn` to keep h5py.File handles open per worker. |
| 12 | pyproject.toml package discovery | ✅ Issue 35 | `include = ["src*", "agents*", "scripts*"]` added. |
| 13 | CI defensive if-guards | 🔧 | Remove if-guards, fail loudly. |
| 14 | CI lint scope | 🔧 | Flake8 + black on `src/`, `scripts/`, `agents/`. |
| 15 | Pre-commit mypy pin | 🔲 Low | Pin mypy version in pre-commit and CI. |
| 16 | Lock file (`make lock`) | 🔲 Low | Auto-generate `requirements-lock.txt`. |
| 17 | Coverage threshold | ✅ Issue 34 | `fail_under = 80` on testable code; CARLA-only and model stubs omitted. |
| 18 | README test qualifier | 🔧 **Issue 2E** | Say "103 offline unit tests" (not bare "103 tests"). |
| 19 | Phase 2 timeline split | 🔲 | Split into Phase 2a (arch) and Phase 2b (training). |
| 20 | Literature review → LaTeX | 🔲 Phase 4 | Convert to LaTeX during the write-up. |
| 21 | Multi-resolution ablation | 🔲 Phase 2b/3 | Resolution scaling curves: 224, 256, 384, 512. |
| 22 | Neuromorphic claim framing | 🔲 Phase 4 | Literature-ground η_AF, frame as design-time property. |
| 23 | GPS / position in HDF5 | ✅ | Position, rotation, privileged metadata stored per-frame in schema v3. |
| 24 | Forward verification strategy | ✅ | PrivilegedExpert provides ground-truth nav from route. |
| 25 | Cascading bug documentation | ✅ | Heuristic removed; PrivilegedExpert replaces TrafficManager autopilot. |
| 26 | TL junction-aware matching | ✅ | `_check_traffic_light()` uses `get_affected_lane_waypoints()`. |
| 27 | OBB/SAT vehicle collision | ✅ | Full oriented bounding-box collision detection. |
| 28 | Pedestrian lateral filtering | ✅ | `pedestrian_lane_half_width=3.0m` filter. |
| 29 | Break logic clarification | ✅ | Multi-line comment in `_get_target()`. |
| 30 | Delete obsolete skipped test | ✅ | `tests/test_infer_nav_command.py` deleted. |
| 31 | Delete old v1/v2 local data | ✅ | All `.h5` files removed from `data/` subdirs. |
| 32 | Schema version gate | ✅ | `DrivingSequenceDataset` requires schema ≥ 3. |
| 33 | Walker spawning | ✅ | `collect_episode()` spawns walkers with AI controllers. |
| 34 | Coverage configuration | ✅ | `[tool.coverage]` omits CARLA-only / model stubs; `fail_under = 80`. |
| 35 | Package discovery | ✅ | `include = ["src*", "agents*", "scripts*"]` in pyproject.toml. |

---

## Current Sprint: Issues 2A–2F + Infrastructure

### Issue 2A (P0): `collect_data.py` Config Key Mismatch 🔧

**Problem:** `build_campaign()` reads `cfg.get("camera", {})` but the YAML key is
`cameras` (plural).  Sub-keys use `"width"` / `"height"` but the YAML has
`"per_camera_width"` / `"per_camera_height"`.  Defaults silently to 512×512 FOV=100
instead of the panoramic config.

**File:** `scripts/collect_data.py`, lines 85–90

**Fix:**
```python
camera_width=cfg.get("cameras", {}).get("per_camera_width", 640),
camera_height=cfg.get("cameras", {}).get("per_camera_height", 320),
camera_fov=cfg.get("cameras", {}).get("fov", 60),
```

---

### Issue 2B (P0): Missing `camera_yaws` Override 🔧

**Problem:** `EpisodeConfig.camera_yaws` is never populated from the YAML's
`cameras.yaw_offsets`.  Works only because the default matches, but any
YAML override is silently ignored.

**File:** `scripts/collect_data.py`, `build_campaign()`

**Fix:** Add `camera_yaws=tuple(cfg.get("cameras", {}).get("yaw_offsets", [-60, 0, 60]))`.

---

### Issue 2C (P1): HDF5 Per-Worker Caching 🔧

**Problem:** `__getitem__` opens/closes `h5py.File` on every call.

**File:** `src/data/driving_dataset.py`

**Fix:** Add `_h5_cache` dict, `_get_h5()` method, and module-level
`_worker_init_fn()` for fork-safe cache reset.

---

### Issue 2D (P1): Velocity Normalization 🔧

**Problem:** Velocity returned raw as `(T, 3)` m/s — not normalised.

**File:** `src/data/driving_dataset.py`

**Fix:** Z-score normalisation with placeholder constants (recompute after
next real collection run).

---

### Issue 2E (P2): README Test Qualifier 🔧

**Problem:** "103 passing" without "offline" qualifier is misleading.

**File:** `README.md`

**Fix:** "103 offline unit tests passing".

---

### Issue 2F (P0): Uncommitted Changes 🔧

**Problem:** 20+ files modified on stale branch `feat/schema-v2-dual-nav-labels`.

**Plan:** Split into logical commits by function:

1. Schema v3 definition (`hdf5_schema.py`)
2. PrivilegedExpert agent (`agents/expert/`)
3. ExpertCollector updates (`expert_collector.py`)
4. Driving dataset updates (`driving_dataset.py`)
5. Collection config + scripts
6. Test updates + conftest
7. Build/CI/docs
8. CARLA integration tests (new)

---

## Infrastructure Tasks (This Sprint)

### CI: Remove Defensive If-Guards (Issue 13)

**File:** `.github/workflows/ci.yml`

Remove `if [ -d src ]` guards.  Lint and test all relevant directories directly.

### Test Split: Offline vs Online

- Add `@pytest.mark.online` marker for CARLA integration tests
- Register marker in `pyproject.toml`
- CI runs: `pytest tests/ -m "not online"` (offline only)
- `Makefile`: `make test` (offline), `make test-online` (CARLA required)

### CARLA Integration Tests

Create `tests/test_carla_integration.py` with `@pytest.mark.online`:

1. Connection test: verify server version ≥ 0.9.16
2. Map loading: load Town01, verify world
3. Expert instantiation: run PrivilegedExpert for 10 ticks
4. Short collection: ~50 frames, verify HDF5 schema v3 output

### SSD Cleanup

`/mnt/data/carla_data/` — 119 GB, 180 episodes, all schema v2.
Delete all `.h5` files and manifest.  Incompatible with v3 pipeline.

---

## Future Phases

### Phase 2a — Architecture & Training Infrastructure

| Task | Description |
|------|-------------|
| Issue 5 | RWKV-7 (1-week timebox, RWKV-6 fallback) |
| Issue 6 | Experiment config structure |
| Infra | Trainer, TBPTT, checkpointing, W&B |

### Phase 2b — Training & Results

| Task | Description |
|------|-------------|
| Issue 4 | Noise injection |
| Issue 21 | Multi-resolution ablation |
| Training | 3-stage: frozen → unfrozen → fine-tune |

### Phase 3 — Evaluation

| Task | Description |
|------|-------------|
| Issue 8 | Nav command ablation |
| Baselines | ResNet-18, ViT, LSTM comparisons |

### Phase 4 — Documentation & Deployment

| Task | Description |
|------|-------------|
| Issue 20 | Literature review → LaTeX |
| Issue 22 | η_AF literature grounding |
| Deploy | Jetson Orin Nano, write-up, demo |

---

## Resolved Issues — Details

<details>
<summary>Issues 26–35: PrivilegedExpert & Pipeline Fixes (click to expand)</summary>

#### Issue 26: Traffic Light Matching — Junction-Aware ✅

Rewrote `_check_traffic_light()` to use `tl.get_affected_lane_waypoints()`.
Includes `AttributeError` fallback for older CARLA builds.

**File:** `agents/expert/privileged_expert.py`

#### Issue 27: OBB/SAT Vehicle Collision ✅

`_get_actor_obb_2d()` + `_sat_overlap()` for oriented bounding-box collision
detection.  Ego lane corridor polygon tested via SAT against each vehicle's OBB.

**File:** `agents/expert/privileged_expert.py`

#### Issue 28: Pedestrian Lateral Filtering ✅

Lateral offset filtering with `pedestrian_lane_half_width=3.0m`.

**File:** `agents/expert/privileged_expert.py`

#### Issue 29: Break Logic Clarification ✅

Multi-line comment explaining intentional `break` in `_get_target()`.

#### Issue 30: Delete Obsolete Skipped Test ✅

Deleted `tests/test_infer_nav_command.py`.

#### Issue 31: Delete Old v1/v2 Local Data ✅

All `.h5` files removed from `data/test_512/`, `data/test_collector/`, etc.

#### Issue 32: Schema Version Gate ✅

`DrivingSequenceDataset.__init__()` raises `ValueError` if schema < 3.

#### Issue 33: Walker Spawning ✅

Walkers with AI controllers, configurable crossing probability, varied speeds.

**File:** `src/data/expert_collector.py`

#### Issue 34: Coverage Configuration ✅

`[tool.coverage]` omits `expert_collector.py`, `route_generator.py`, `src/models/*`.

#### Issue 35: Package Discovery ✅

`include = ["src*", "agents*", "scripts*"]` in pyproject.toml.

</details>

<details>
<summary>Issues 1–2, 7, 9–10, 23–25: Schema v3 Migration (click to expand)</summary>

#### Issue 1: Turn Distribution ✅

`BalancedRouteGenerator` ensures 15% L/R target distribution.

#### Issue 2: Nav Command Labels ✅

Single `nav_command` from PrivilegedExpert route planner (RoadOption enum).
`_infer_nav_command()` and dual-label machinery removed.

#### Issue 7: Camera Resolution ✅

3-camera panoramic: 3×640×320 at 60° FOV, yaws [-60, 0, 60]°.

#### Issue 9: Color Jitter Ordering ✅

Augmentation operates on [0, 1] pixel values BEFORE normalization.

#### Issue 10: Weighted Sampler ✅

Per-clip weighting by inverse LaneFollow fraction.

#### Issue 23: GPS/Position ✅

Position, rotation, privileged metadata per-frame in schema v3.

#### Issue 24: Forward Verification ✅

Ground-truth nav from route planner — no heuristic to verify.

#### Issue 25: Cascading Bug ✅

Heuristic removed. PrivilegedExpert replaces TrafficManager.

</details>
