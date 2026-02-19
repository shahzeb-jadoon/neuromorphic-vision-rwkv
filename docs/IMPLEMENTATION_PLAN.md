# Implementation Plan: All Issues & Decisions

> **Generated:** Based on exhaustive code review + interactive decision-making session
> **Status:** All 22 issues reviewed, decisions made, ready for implementation

---

## Decision Summary

| # | Issue | Decision | Priority | Phase |
|---|-------|----------|----------|-------|
| 1 | Turn distribution (2% vs 15%) | Re-collect from scratch with 1M+ frames; verify with sample run first | **P0** | 1.5 (re-collection) |
| 2 | Nav command labels (15.1% error) | Store BOTH planner + heuristic labels at collection time | **P0** | 1.5 |
| 3 | Velocity normalization | Z-score normalize velocity input in Dataset | **P1** | Now |
| 4 | Collision episode scarcity | Noise injection in Phase 2 + DAgger in Phase 3 | P2 | 2 / 3 |
| 5 | RWKV-7 vs RWKV-6 | RWKV-7 first, 1-week timebox, fallback to RWKV-6 | **P0** | 2a |
| 6 | embed_dim 384 vs 768 | Separate experiment configs per run (no coupling to pretrained flag) | P1 | 2a |
| 7 | Camera resolution | Collect at 1024×1024; resize to 224 (pretrained) or 256 (from-scratch) in transforms | **P0** | 1.5 |
| 8 | Nav oracle problem | Conditional IL primary path; remove-nav-command ablation as later study | P2 | 3 |
| 9 | Color jitter after normalize | Fix: move ColorJitter before Normalize | **P0** | Now |
| 10 | Weighted sampler per-episode | Fix: per-clip weighting by nav_command | **P1** | Now |
| 11 | HDF5 per-access file opens | Per-worker caching via worker_init_fn | **P1** | Now |
| 12 | sys.path.insert hack | Fix pyproject.toml package discovery + audit imports | P1 | Now |
| 13 | CI defensive guards | Remove if-guards, fail loudly | P1 | Now |
| 14 | CI lint scope | Flake8 + mypy on src/, scripts/, tests/ | P1 | Now |
| 15 | Pre-commit mypy divergence | Pin mypy version in both pre-commit and CI | P2 | Now |
| 16 | requirements.txt confusion | Auto-generate lock file via `make lock` | P2 | Now |
| 17 | Coverage 61% vs 80% claim | Lower threshold to 60%, enforce in CI, ratchet up | P1 | Now |
| 18 | README "112 tests" claim | Add qualifier: "92 functional, 20 model interface contracts" | P2 | Now |
| 19 | Phase 2 timeline (3 weeks) | Split into Phase 2a (arch + infra) and Phase 2b (training + results) | P1 | Now |
| 20 | Literature review in Markdown | Convert to LaTeX in Phase 4 (deferred) | P3 | 4 |
| 21 | Paper angle (spatial vs temporal) | Temporal efficiency framing + multi-resolution ablation (collect at 1024) | P1 | 2b / 3 |
| 22 | Neuromorphic claim | Keep title, literature-ground η_AF, transparently frame as design-time property | P2 | 4 |
| 23 | GPS / position in HDF5 | NOT needed — store planner RoadOption labels directly at collection time | — | 1.5 |
| 24 | Forward verification strategy | 4-point ongoing label quality monitoring (see below) | P1 | 1.5+ |
| 25 | Cascading bug documentation | Document the two-part bug chain (agents missing → heuristic exposed) in Phase 1 results | P2 | Now |

---

## Implementation Details by Phase

### Batch 0: Immediate Code Fixes (Do Now, Before Re-collection)

These can all be done in the current codebase before any re-collection.

---

#### Fix 9: Color Jitter Ordering

**File:** `src/data/driving_dataset.py`

**What to change:** Move `ColorJitter` to BEFORE `Normalize` in the transform pipeline.

**Current (broken):**
```python
transforms.Compose([
    transforms.ToTensor(),          # 0-255 → 0.0-1.0
    transforms.Normalize(...),       # → z-scores (-2.1 to +2.6)
    transforms.ColorJitter(...),     # ← WRONG: distorts z-scores
])
```

**Fixed:**
```python
transforms.Compose([
    transforms.ToTensor(),          # 0-255 → 0.0-1.0
    transforms.ColorJitter(...),     # ← CORRECT: jitters real pixel values
    transforms.Normalize(...),       # → z-scores
])
```

**Tests to update:** Any test that checks transform ordering or augmentation output ranges.

---

#### Fix 10: Weighted Sampler — Per-Clip by Nav Command

**File:** `src/data/driving_dataset.py`

**What to change:** Replace per-episode weighting with per-clip weighting based on nav_command.

**Implementation sketch:**
```python
def _build_sample_weights(self) -> torch.Tensor:
    """Weight each clip inversely proportional to its nav_command frequency."""
    # Count clips per nav_command
    command_counts: dict[int, int] = {}
    clip_commands: list[int] = []
    for idx in range(len(self)):
        # Get the nav_command for this clip's last frame (the target frame)
        ep_idx, start = self._clip_index[idx]
        ep = self._episodes[ep_idx]
        target_frame = start + self.seq_len - 1
        cmd = ep["nav_command"][target_frame]
        clip_commands.append(int(cmd))
        command_counts[int(cmd)] = command_counts.get(int(cmd), 0) + 1

    # Weight = 1 / count_of_command
    total = len(clip_commands)
    weights = []
    for cmd in clip_commands:
        weights.append(total / (len(command_counts) * command_counts[cmd]))
    return torch.tensor(weights, dtype=torch.float64)
```

**Note:** This is compute-intensive at Dataset init time (iterates all clips). Consider caching the result or computing it lazily. For datasets < 1M clips, it's fast enough.

---

#### Fix 3: Velocity Z-Score Normalization

**File:** `src/data/driving_dataset.py`

**What to change:** Add z-score normalization for velocity input.

**Implementation:**
1. Compute dataset-wide velocity statistics (mean, std) at init time (or use hardcoded values from analysis)
2. Normalize in `__getitem__`: `velocity = (velocity - self.vel_mean) / (self.vel_std + 1e-8)`

**Two approaches:**
- **Offline stats:** Run `scripts/analyze_dataset.py` to compute mean/std across all episodes. Hardcode as constants (e.g., `VEL_MEAN = 7.2, VEL_STD = 3.1`). Simpler, faster init.
- **Online stats:** Compute at Dataset `__init__` by scanning all episodes. More accurate but slower startup.

**Recommended:** Offline stats hardcoded as module-level constants, with a script to regenerate them if the dataset changes. Add a `VELOCITY_STATS_VERSION` constant to detect stale values.

---

#### Fix 11: HDF5 Per-Worker File Handle Caching

**File:** `src/data/driving_dataset.py`

**What to change:** Open HDF5 files once per DataLoader worker instead of per `__getitem__` call.

**Implementation:**
```python
class DrivingDataset(Dataset):
    def __init__(self, ...):
        ...
        self._h5_handles: dict[str, h5py.File] = {}  # Populated by worker_init_fn

    def _open_handles(self) -> None:
        """Open all HDF5 file handles. Called once per worker."""
        for ep_path in self._episode_paths:
            self._h5_handles[ep_path] = h5py.File(ep_path, "r")

    def _close_handles(self) -> None:
        """Close all HDF5 file handles."""
        for h in self._h5_handles.values():
            h.close()
        self._h5_handles.clear()

    def __getitem__(self, idx: int) -> dict:
        ep_path = self._get_episode_path(idx)
        h5f = self._h5_handles.get(ep_path)
        if h5f is None:
            # Fallback for num_workers=0 (main process)
            h5f = h5py.File(ep_path, "r")
            self._h5_handles[ep_path] = h5f
        # ... read data from h5f ...

    def __del__(self) -> None:
        self._close_handles()


def worker_init_fn(worker_id: int) -> None:
    """Called once per DataLoader worker to open HDF5 handles."""
    dataset = torch.utils.data.get_worker_info().dataset
    dataset._open_handles()
```

**Usage in DataLoader:**
```python
DataLoader(dataset, num_workers=4, worker_init_fn=worker_init_fn)
```

---

#### Fix 12: pyproject.toml Package Discovery + Import Audit

**File:** `pyproject.toml`, all Python files with `from src.` imports

**What to change:** Your pyproject.toml already has `package-dir = {"" = "src"}`, which means `src/` is the package root. Imports should be:
- ✅ `from data.expert_collector import ExpertCollector`
- ❌ `from src.data.expert_collector import ExpertCollector`

**Steps:**
1. Search all files for `from src.` or `import src.` patterns
2. Remove the `src.` prefix (since `src/` IS the package root per pyproject.toml)
3. Remove all `sys.path.insert` hacks
4. Verify with `python -c "from data.driving_dataset import DrivingDataset"` in the conda env

**Caveat:** If scripts/ are run standalone (not via `python -m`), they need the package to be installed. The editable install (`pip install -e .`) handles this, but developers must remember to install first.

---

#### Fix 13: Remove CI Defensive Guards

**File:** `.github/workflows/ci.yml` (or equivalent)

**Change:**
```yaml
# BEFORE (bad — silently skips)
- run: |
    if [ -d src ]; then
        mypy src/
    fi

# AFTER (good — fails loudly if something is wrong)
- run: mypy src/ scripts/ tests/
```

---

#### Fix 14: Expand CI Lint Scope

**File:** `.github/workflows/ci.yml`

**Change:** Add `scripts/` and `tests/` to flake8 and mypy commands:
```yaml
- run: flake8 src/ scripts/ tests/
- run: mypy src/ scripts/ tests/
```

**Note (user chose B: mypy everything):** mypy on test files often produces false positives from pytest fixtures and mock objects. You may need to add some `# type: ignore` annotations or adjust mypy config for the tests directory. Consider adding to pyproject.toml:
```toml
[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false
```

---

#### Fix 15: Pin mypy Version

**Files:** `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `requirements.txt`

**Steps:**
1. Check current mypy version: `mypy --version`
2. Pin in CI: `pip install mypy==X.Y.Z`
3. Pin in pre-commit config (rev field of the mypy hook)
4. Pin in requirements.txt

---

#### Fix 16: Auto-Generate Lock File

**File:** `Makefile`

**Add target:**
```makefile
.PHONY: lock
lock:
	pip freeze > requirements-lock.txt
	@echo "Lock file updated: requirements-lock.txt"
```

**Workflow:**
- `requirements.txt` = human-maintained, loose version specs (e.g., `torch>=2.6`)
- `requirements-lock.txt` = auto-generated via `make lock`, exact versions for reproducibility
- `environment.yml` = conda environment definition (primary dev setup)

---

#### Fix 17: Coverage Threshold

**File:** `Makefile`

**Change:**
```makefile
# BEFORE
pytest --cov=src --cov-fail-under=80

# AFTER
pytest --cov=src --cov-fail-under=60
```

Also add `--cov-fail-under=60` to CI if not already there. Ratchet up to 65, 70, 75, 80 as you add tests.

---

#### Fix 18: README Test Count Qualifier

**File:** `README.md`

**Change the test count claim to something like:**
```markdown
**112 tests** (92 data-pipeline functional tests, 20 model-interface contract tests)
```

---

#### Fix 19: Split Phase 2 Timeline

**File:** the project plan

**Change Phase 2 row in table to:**
```markdown
| **IIa** | 5–6 | Architecture & Training Infrastructure | RWKV encoder, tokenizer, fusion, policy head implemented; training loop with TBPTT, checkpointing, logging |
| **IIb** | 7–9 | Training & Initial Results | 3-stage training (frozen → unfrozen → fine-tune); initial driving metrics on validation episodes |
```

**Key point:** Phase 2a is "can the model forward-pass without crashing?" Phase 2b is "can the model learn to drive?" Separate checkpoints let you catch problems early.

---

### Batch 1: Data Re-Collection (Phase 1.5)

After all code fixes are applied and tested.

---

#### Fix 1: Turn Distribution Re-Collection

**Goal:** Re-collect from scratch with 1M+ frames, balanced turn distribution.

**Steps:**
1. **Verification run first:** Collect 2-3 short episodes (~5 min each), immediately run `scripts/analyze_dataset.py` to check turn distribution. Target: >10% turns.
2. **Fix BalancedRouteGenerator:** With vendored agents module (already fixed), routes through junctions should now work. Verify junction coverage in the verification run.
3. **Full collection:** Once verified, run overnight collection for 50+ hours per the plan.
4. **Post-collection validation:** Run `scripts/validate_dataset.py` + `scripts/analyze_dataset.py` on full dataset. Document distribution in a results file.

---

#### Fix 2: Dual Nav Command Labels

**File:** `src/data/expert_collector.py`

**What to change:** At each frame, store BOTH:
- `nav_command_planner`: The RoadOption from the CARLA route planner (ground truth)
- `nav_command_heuristic`: The heuristic-inferred label (current approach, for comparison)

**Implementation:**
```python
# In _save_frame() or equivalent
nav_planner = self._current_road_option      # From planner
nav_heuristic = self._infer_nav_command(...)  # Current heuristic

# Store both in HDF5
nav_commands_planner[frame_idx] = nav_planner.value
nav_commands_heuristic[frame_idx] = nav_heuristic
```

**HDF5 schema update:** Add `nav_command_planner` and `nav_command_heuristic` fields to `hdf5_schema.py`. Bump `SCHEMA_VERSION` to 2.

**Training decision:** Use `nav_command_planner` as the training label (ground truth). Keep `nav_command_heuristic` for post-hoc error analysis and potential ensemble/comparison studies.

---

#### Decision 23: GPS / Position Data — NOT Needed

**Context:** The question was whether to store the vehicle's GPS/world position (x, y, z) per frame in HDF5, so that navigation labels could be computed post-hoc by re-running the route planner.

**Decision: No GPS needed.** Since we are storing the planner's `RoadOption` label directly at collection time (Fix 2), there is no need to store raw position and re-derive labels later. The planner already has the correct answer — we just write it into the HDF5 at the point of collection.

**If GPS is ever needed later** (e.g., for trajectory visualisation, distance-to-goal metrics, or waypoint-conditioned IL), it would be a trivial addition: `position = vehicle.get_transform().location` gives `(x, y, z)` at each frame. Three `float32` values per frame add negligible storage (~12 bytes/frame vs ~68,000 bytes/frame for the image).

---

#### Fix 24: Forward Verification Strategy (4-Point Plan)

Ongoing label quality monitoring to catch regressions after re-collection or between campaigns.

**Point 1 — At collection time (ongoing):**
Both planner and heuristic labels are stored (Fix 2). After each collection session, compare them. If they diverge massively on a new town, investigate — the route planner may have produced an unexpected graph.

**Point 2 — Per-campaign validation:**
After each overnight collection run, run `scripts/quantify_label_error.py` on the same towns used in the original analysis (Town01, Town03, etc.) to check if the error rate has changed. New CARLA versions or new maps could alter junction topology.

**Point 3 — During training (Phase 2):**
Monitor the `nav_command` class distribution in each training batch. If any single class exceeds 90% in a batch, log a warning. This catches silent regression to the all-LaneFollow failure mode.

```python
# In training loop:
cmd_dist = batch["nav_command"].bincount(minlength=4)
if cmd_dist.max() / cmd_dist.sum() > 0.9:
    logger.warning(f"Nav command imbalance: {cmd_dist.tolist()}")
```

**Point 4 — Cross-town generalisation (Phase 3):**
The label error rate varied by town (Town01: 10%, Town03: 24.7%). When evaluating the trained model, test on towns it was NOT trained on. If performance degrades specifically at intersections, it may indicate that label noise in training data from high-error towns hurt generalisation.

---

#### Fix 25: Document the Cascading Bug

**File:** `docs/phase_1_results.md`

**What to add:** A clear write-up of the two-part bug chain for future reference and documentation:

> **Bug 1 (root cause):** The `agents` module (containing `GlobalRoutePlanner`) ships with the CARLA server distribution, not the `carla` pip package. Every call to `BalancedRouteGenerator._trace_route()` silently failed and fell back to `_extend_route()`, which extends routes in a straight line. Result: 120 episodes with ~84% LaneFollow and only ~2% turns.
>
> **Fix:** Vendored the agents module (MIT licensed) into the project.
>
> **Bug 2 (exposed by fixing Bug 1):** Even with real balanced routes, `_infer_nav_command()` uses steering angle + waypoint geometry to guess the navigation command. This heuristic has a 15.1% error rate — it almost never invents a turn, but it misses the vast majority of real turns (Left precision: 1.2%, Right precision: 2.4%).
>
> **Fix (Phase 2):** Use the `RoadOption` label from the route planner directly, bypassing the heuristic entirely.
>
> The first bug made the second bug invisible — there were almost no turns to mislabel.

---

#### Fix 7: Camera Resolution — Collect at 1024×1024

**File:** `src/data/expert_collector.py`, `configs/collection.yaml`

**What to change:**
1. Set CARLA camera sensor to capture at 1024×1024
2. Store full 1024×1024 frames in HDF5

**Storage estimate (calibrated from real data):**

Measured gzip-1 compression on existing 224×224 CARLA images: **~2.15× compression ratio** (~68 KB per compressed frame). Extrapolating to higher resolutions (compression ratio is roughly constant for same image content):

| Resolution | Raw/frame | Compressed/frame (÷2.15) | 50 hrs (3.6M frames) | vs 224 |
|---|---|---|---|---|
| 224×224 | 147 KB | ~68 KB | **~233 GB** | 1× |
| 512×512 | 768 KB | ~357 KB | **~1.22 TB** | 5.2× |
| 1024×1024 | 3,072 KB | ~1,429 KB | **~4.9 TB** | 21× |

**Important:** Verify you have sufficient disk space before starting. Consider:
- Lab NFS has 64 PB pool — 4.9 TB is trivial, this is where the 1024 data should live
- Laptop with 925 GB — can hold 224 or a subset at 512, NOT sufficient for full 1024 dataset
- **512×512 compromise (~1.2 TB):** fits on a large external drive, still supports ablation at 224/256/384/512. Only loses the ability to test at 768+, which is an edge case
- For multi-resolution ablation, all you need is the source resolution to be ≥ your largest target. If 512 is the max ablation target, 512 source data is sufficient

**DataLoader changes:** In `DrivingDataset`, add resolution-aware transforms:
```python
if self.config.use_pretrained_vision:
    resize_target = 224  # Match pretrained ViT input
else:
    resize_target = 256  # From-scratch path

self.transform = transforms.Compose([
    transforms.Resize(resize_target),
    transforms.ToTensor(),
    transforms.ColorJitter(...),    # Before normalize!
    transforms.Normalize(...),
])
```

**For multi-resolution ablation:** At experiment time, override resize_target to 224, 256, 384, 512 to build scaling curves. The 1024 source data downsizes cleanly to all of these.

---

### Batch 2: Phase 2a — Architecture & Training Infrastructure

After re-collection is running/complete.

---

#### Issue 5: RWKV-7 Implementation (1-Week Timebox)

**Plan:**
1. Week 1, Days 1-2: Source RWKV-7 reference implementation. Get the custom CUDA kernels compiled on workstation GPU.
2. Week 1, Days 3-5: Adapt for 2D vision (patch tokens instead of text tokens). Get a single forward pass working with dummy input.
3. Week 1, Days 6-7: Integration test — plug RWKV-7 encoder into VisionRWKVDriver, verify gradient flow.

**Pivot criteria:** If by end of day 7, you cannot get a forward+backward pass through the full model (tokenizer → RWKV-7 encoder → fusion → policy head → loss.backward()), switch to RWKV-6.

**RWKV-6 fallback:** Use the well-documented implementation from the Vision-RWKV ICLR paper. The architecture is identical except for the decay mechanism (learnable but static instead of data-dependent).

---

#### Issue 6: Experiment Config Structure

**Files:** Create `configs/experiments/` directory

**Structure:**
```
configs/
├── collection.yaml          # Data collection settings
└── experiments/
    ├── rwkv6_384.yaml        # RWKV-6, embed_dim=384, from-scratch
    ├── rwkv6_768.yaml        # RWKV-6, embed_dim=768, from-scratch
    ├── rwkv6_768_clip.yaml   # RWKV-6, embed_dim=768, CLIP pretrained
    ├── rwkv7_384.yaml        # RWKV-7, embed_dim=384, from-scratch
    └── baseline_resnet.yaml  # ResNet-18 baseline
```

Each config specifies all hyperparameters for a single experiment. No ambiguity about which embed_dim to use — it's in the config file name.

---

### Batch 3: Phase 2b — Training & Results

---

#### Issue 4: Collision Handling (Noise Injection + DAgger)

**Phase 2b — Noise injection:**
Add small random perturbations to autopilot actions during later training stages:
```python
# During training, with probability p:
steer_noisy = steer_clean + torch.randn_like(steer_clean) * noise_std
# The model learns to recover from slightly-off trajectories
```
Start with `noise_std=0.01`, increase to `0.05` as training stabilizes.

**Phase 3 — DAgger (Dataset Aggregation):**
1. Run the trained model in CARLA closed-loop
2. Record the expert's (autopilot's) corrective actions at each step
3. Add these correction episodes to the training set
4. Retrain — the model now has examples of recovering from its own mistakes

---

#### Issue 21: Multi-Resolution Ablation

**Setup (Phase 2b/3):**
1. Train primary model at 256×256 (from-scratch) or 224×224 (pretrained)
2. For ablation, train identical architecture at 224, 256, 384, 512
3. Resize from 1024 source data in DataLoader transforms
4. Measure: inference FLOPs, latency (ms), Route Completion Rate, Collision Rate
5. Plot: resolution vs. efficiency (RWKV) compared to transformer baseline
6. Expected result: RWKV scales linearly while transformer scales quadratically with token count

---

### Batch 4: Phase 3 — Evaluation

---

#### Issue 8: Nav Command Ablation Study

**After conditional IL model is trained and evaluated:**
1. Remove nav_command input from model
2. Retrain with identical hyperparameters
3. Compare driving metrics (Route Completion, Collision Rate)
4. Discuss in the write-up: "Removing the navigation oracle degrades performance by X%, confirming the model relies on route guidance rather than learning autonomous navigation. This is consistent with the conditional IL literature."

---

### Batch 5: Phase 4 — Documentation & Deployment

---

#### Issue 20: Literature Review LaTeX Conversion

Convert `docs/literature_review.md` to `paper/chapters/literature_review.tex` during Phase 4 writing.

---

#### Issue 22: η_AF Metric Literature Grounding

**Steps:**
1. Survey neuromorphic computing metrics in literature:
   - **SynOps** (synaptic operations per inference) — Merolla et al.
   - **Spike rate** / activation sparsity — standard in SNN literature
   - **Memory bandwidth** — critical for neuromorphic chips with limited SRAM
   - **Fan-in/fan-out** constraints — Akida hardware limits
2. Map each component of η_AF to a cited metric
3. Position η_AF as: "We synthesize existing measures into a single scalar for cross-architecture comparison"
4. Add transparency statement: "η_AF evaluates architectural properties; deployment on neuromorphic hardware (Akida) is future work"

---

## Implementation Order (Recommended)

```
Week 0 (Now):
├── Fix 9:  Color jitter ordering ............... (30 min)
├── Fix 10: Weighted sampler per-clip ........... (1-2 hours)
├── Fix 3:  Velocity z-score normalization ...... (1-2 hours)
├── Fix 11: HDF5 worker_init_fn caching ......... (2-3 hours)
├── Fix 12: pyproject.toml + import audit ....... (1-2 hours)
├── Fix 13: Remove CI guards .................... (15 min)
├── Fix 14: Expand CI lint scope ................ (30 min)
├── Fix 15: Pin mypy version .................... (30 min)
├── Fix 16: Make lock target .................... (15 min)
├── Fix 17: Coverage threshold → 60% ........... (5 min)
├── Fix 18: README test qualifier ............... (5 min)
└── Fix 19: Split Phase 2 timeline .............. (30 min)
├── Fix 25: Document cascading bug in phase_1_results.md (30 min)
                                       Total: ~1 day

├── Fix 25: Document cascading bug in phase_1_results.md (30 min)
                                       Total: ~1 day

Week 1 (Re-collection Prep):
├── Fix 2:  Dual nav command labels ............. (2-3 hours)
├── Fix 7:  Camera resolution → 1024 or 512 .... (1-2 hours)
├── Fix 24: Forward verification strategy hooks . (1-2 hours)
├── Fix 1:  Verification collection run ......... (2-3 hours)
└── Fix 1:  Full re-collection begins ........... (runs overnight/multi-day)

Week 2-3 (Phase 2a — Architecture):
├── Issue 5:  RWKV-7 implementation (1-week timebox)
├── Issue 6:  Experiment config structure
└── Training infrastructure (Trainer, TBPTT, checkpointing, W&B)

Week 4-6 (Phase 2b — Training):
├── Issue 4:  Noise injection during training
├── Issue 21: Multi-resolution ablation setup
└── 3-stage training (frozen → unfrozen → fine-tune)

Week 7-9 (Phase 3 — Evaluation):
├── Issue 8:  Nav command ablation
├── Issue 21: Multi-resolution ablation runs
└── Baseline comparisons (ResNet-18, ViT, LSTM)

Week 10-12 (Phase 4 — Documentation & Deployment):
├── Issue 20: Literature review → LaTeX
├── Issue 22: η_AF literature grounding
└── Jetson deployment, write-up, demo video
```

---

## Risk Mitigations Added

| New Risk | Mitigation |
|----------|------------|
| 1024×1024 data exceeds storage | 50 hrs at 1024² ≈ 4.9 TB (measured 2.15× gzip-1 ratio). Store on lab NFS (64 PB pool). Fallback: 512×512 (1.22 TB). Laptop: 224 subset or 512 subset only. |
| RWKV-7 CUDA kernels won't compile on workstation GPU | Hard 1-week pivot to RWKV-6. No time wasted on architecture; training infra is RWKV-version-agnostic. |
| mypy on tests/ produces false positives | Add `[[tool.mypy.overrides]]` for tests/ with relaxed settings. |
| Re-collection takes too long | Start collection early in the week; run overnight/weekend. 50 hours at 20× real-time takes ~2.5 hours wall time with CARLA acceleration. |
| Z-score velocity stats become stale after re-collection | Add `VELOCITY_STATS_VERSION` constant. Recompute stats after re-collection and update. |
