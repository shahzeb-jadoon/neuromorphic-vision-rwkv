# Atlas Continuation Findings and Fixes (2026-04-20)

## Scope
This note documents continuation-route failures observed during atlas batch runs and the fixes applied on 2026-04-20.

Primary batch command (correct executable path):

```bash
~/miniconda3/envs/rwkv/bin/python scripts/run_atlas_route_episode.py \
  --atlas-root data/atlas_cpp/test_collection_towns \
  --town-filter Town03 Town05 Town06 Town10HD \
  --solver cpp-approx \
  --episodes-per-town 3 \
  --route-order random \
  --seed 42 \
  --host 172.20.0.1 \
  --port 2000 \
  --output-dir data/test_collection_atlas_batch \
  --weather ClearNoon \
  --vehicles 20 \
  --walkers 10 \
  --max-frames 4000 \
  --spectator \
  --debug-draw \
  --continuation \
  --continuation-mode config \
  --reject-multi-lane-jumps
```

## Where Data Is Stored
### Input atlas data
- Root: `data/atlas_cpp/test_collection_towns`
- Per-town route files:
  - `data/atlas_cpp/test_collection_towns/<Town>/cpp-approx/routes/route_*.json`
  - `data/atlas_cpp/test_collection_towns/<Town>/cpp-exact/routes/route_*.json` (if used)
- Atlas run state cache:
  - `data/atlas_cpp/test_collection_towns/atlas_run_state.json`

### Output run data (from --output-dir)
- Output root: `data/test_collection_atlas_batch`
- Per-episode files:
  - `episode_<hash>.h5`
  - `episode_<hash>.route.json`
- Batch manifest:
  - `collection_manifest.json`
- Batch reports:
  - `atlas_route_batch_report.json`
  - `atlas_route_batch_report.csv`

## Observed Failure Signature (Town03 and continuation-heavy runs)
- Continuation could accept a candidate route whose first waypoint was not a safe local handoff from ego.
- In practice this looked like a handoff to a different road/lane context, producing unstable target behavior immediately after continuation.
- Independently of continuation, several atlas route files contain severe spatial jumps between consecutive waypoints (e.g., 30m-200m+), which can trigger route watchdog fail-fast and wrong-way recovery behavior very early in an episode.

## Fixes Applied
### 1) Safer continuation start projection
File: `scripts/collect_data.py`
- `_LazyAtlasContinuationPlanner.next_route()` now projects start waypoint with `project_to_road=True` and driving-lane preference before fallback.

### 2) Anchor continuity gate for continuation candidates
File: `scripts/run_atlas_route_episode.py`
- Added continuation anchor validation before accepting generated continuation route:
  - start distance from ego to candidate[0]
  - heading difference between ego lane and candidate start lane
  - same-road lane delta guard when applicable
- New CLI controls:
  - `--reject-anchor-mismatch/--no-reject-anchor-mismatch`
  - `--anchor-max-start-distance-m`
  - `--anchor-max-heading-diff-deg`
  - `--anchor-max-lane-delta`

### 3) Early continuation preload near route end
Files:
- `src/data/expert_collector.py`
- `scripts/collect_data.py`
- `scripts/run_atlas_route_episode.py`
- `configs/collection.yaml`

Behavior:
- Continuation can now be requested before full exhaustion when remaining waypoints are below a threshold.
- New continuation config field:
  - `routes.continuation.preload_remaining_waypoints`
- Default set to `40` in `configs/collection.yaml` as a safer memory/performance choice for long runs.

### 4) Reporting additions
File: `scripts/run_atlas_route_episode.py`
- Batch report rows now include `continuation_preload_remaining_waypoints`.

### 5) Pre-run severe route-gap gate (atlas route quality)
File: `scripts/run_atlas_route_episode.py`
- Added `_route_gap_report()` to compute warn/severe transition counts from route geometry before launching an episode.
- Added route-level skip gate so malformed atlas files are rejected before ego spawn/rollout.
- New CLI controls:
  - `--reject-severe-route-gaps/--no-reject-severe-route-gaps`
  - `--route-gap-warn-m`
  - `--route-gap-severe-m`
  - `--route-gap-max-severe-transitions`
  - `--route-gap-max-issues`
- CSV/JSON reports now include route-gap metrics:
  - `route_gap_warn_transition_count`
  - `route_gap_severe_transition_count`
  - `route_gap_max_gap_m`
  - `route_gap_missing_waypoint_projections`

## OOM/Resource Notes
- Preload threshold does not duplicate route buffers in memory; it swaps route context earlier.
- Threshold 40 is selected as a conservative default for long episodes.
- If behavior is still too late, raise to 50 with CLI override:

```bash
--continuation-preload-remaining-waypoints 50
```

## Expected Post-Fix Behavior
- Continuation routes with poor local handoff are rejected and regenerated (up to attempt limit).
- Atlas routes with severe geometric discontinuities are skipped before runtime, preventing immediate route-invalid/wrong-way failure loops.
- Route handoff happens before final-route starvation region, reducing delayed stop-sign/traffic-light reaction near route end.
- Batch report and manifest in output directory capture episode-level outcomes for review.
