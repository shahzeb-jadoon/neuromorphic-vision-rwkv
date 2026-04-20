# C2 CPP Atlas Plan (Validated) - 2026-04-19

## 1) Why this document exists
This document captures the validated findings from live code checks and CARLA runtime probes, and defines how C2 should be implemented as a standalone script before any integration into collection runtime.

The goal is to support route-bank precomputation for all towns, with explicit resource expectations and operational constraints.

## 2) What is already verified

### 2.1 Lane graph segment-key behavior (branch-drop concern)
- Segment IDs are keyed by composite waypoint pair: `start_id:end_id`.
- Dedup skips only repeated identical segment IDs.
- Result: old start-only branch-drop claim is stale for current code.

Code evidence:
- `agents/navigation/lane_graph_planner.py` (`_segment_id_for_topology_pair`, dedup in build loop)

### 2.2 Multi-town topology probe (live CARLA)
Measured driving topology counts from CARLA:

| Town | Driving pairs | Unique pair IDs |
|---|---:|---:|
| Town01 | 160 | 160 |
| Town03 | 503 | 503 |
| Town04 | 559 | 559 |
| Town05 | 604 | 604 |
| Town06 | 490 | 490 |
| Town07 | 360 | 360 |
| Town10HD | 200 | 200 |
| Town12 | 10415 | 10415 |
| Town15 | 1038 | 1037 |
| **Total** | **14329** | **14328** |

Town15 has one duplicate composite pair ID. Further check showed:
- duplicate pair ID count: 1
- true collisions (same pair ID mapping to different lane keys): 0

Interpretation:
- There is one exact duplicate edge in Town15 topology under pair-ID projection.
- This does **not** imply distinct-branch loss in current dedup behavior.

### 2.3 Continuation vs tail-extension lifecycle
- Continuation is applied when `expert.done()` is true.
- Tail extension is explicitly disabled when continuation is enabled.
- This resolves the historical paradox where tail extension could prevent continuation from triggering.

Code evidence:
- `scripts/collect_data.py` (tail-extension disable when continuation on)
- `src/data/expert_collector.py` (done-time continuation loop)

### 2.4 atlas_chain reality today
- Current `atlas_chain` is continuation scaffolding, not a full coverage-led atlas.
- It attempts anchor destinations, and falls back to `generate_route_from_location(..., strategy="balanced")`.
- No executed-coverage ledger is consulted in continuation route choice.

Code evidence:
- `scripts/collect_data.py` (`_LazyAtlasContinuationPlanner.next_route`)

### 2.5 Branch choice correctness (tail extension + gap fill)
- Tail extension uses scored successor selection, not first candidate.
- Junction gap fill uses closest-to-target next waypoint selection.

Code evidence:
- `agents/expert/privileged_expert.py` (`_route_end_branch_score`, `min(...)`)
- `agents/expert/privileged_expert.py` (`_pick_best_next` and call site)

### 2.6 Vertical conflict gate status (current branch)
- Current branch uses shared vertical conflict gate logic with 2.0 m defaults.
- The previously discussed 1.0 m vs 2.5 m mismatch is not present in current code/config defaults.

Code/config evidence:
- `agents/expert/privileged_expert.py` (`vehicle_vertical_conflict_max_m`, shared gate function)
- `configs/collection.yaml` (`vehicle_vertical_conflict_max_m: 2.0`)

### 2.7 Campaign turn balance status
- `campaign_turn_balance.enabled` is currently `false`.

Config evidence:
- `configs/collection.yaml`

### 2.8 Executed telemetry for coverage ledgering
- `wp_buffer_head_index` and `route_completion` are in schema and writer.
- In `data/test_collection`, both fields exist and are populated in all 12 files.

Observed artifacts:
- 12/12 files contain both datasets.
- 11/12 episodes reached max route completion 1.0.

## 3) Option comparison (for project planning)

| Option | Summary | Strengths | Weaknesses |
|---|---|---|---|
| A | Current continuation + balanced routing | Already working, low risk | No explicit spatial coverage targeting |
| B | Executed coverage ledger (post-episode) | Visibility and measurable coverage progress | Adds bookkeeping pipeline |
| C1 | Heuristic deficit-targeted scheduler | Big practical gains with moderate complexity | Not globally optimal |
| C2 | Full CPP/RPP optimizer | Best theoretical route-efficiency | Highest complexity, needs careful scaling |
| D | Benchmark route slice for evaluation | Reproducible comparisons | Not sufficient alone for training coverage |

Recommended long-term architecture:
- A + B + C1 in near term
- D for evaluation comparability
- C2 as optional advanced optimizer once B/C1 are stable

## 4) C2 as standalone script (pre-integration)

### 4.1 Script intent
Build a precomputed route bank from map topology using CPP-style optimization (exact or approximate), independent from runtime collection code.

### 4.2 Proposed script name
- `scripts/build_route_atlas_cpp.py`

### 4.3 Proposed outputs
- `data/atlas_cpp/<town>/atlas_manifest.json`
- `data/atlas_cpp/<town>/routes/*.json`
- Optional diagnostics:
  - node imbalance report
  - deadhead overhead report
  - coverage verification report

### 4.4 Suggested CLI shape
- `--town` (repeatable)
- `--all-towns`
- `--max-routes-per-town`
- `--target-coverage`
- `--chunk-min-waypoints`
- `--chunk-max-waypoints`
- `--solver` (`heuristic`, `cpp-approx`, `cpp-exact`)
- `--resume`
- `--output`

### 4.5 Operational mode
- One town at a time (sequential) for stability and memory control.
- Checkpoint after each route chunk.

## 5) Resource expectations for C2 standalone

## 5.1 Storage estimate
Depends on output format and route density.

Typical range for all towns in JSON route-bank form:
- Route chunks only: ~50 MB to ~400 MB
- With rich diagnostics and checkpoints: ~0.5 GB to ~2.0 GB

Main drivers:
- Number of chunks
- Waypoints per chunk
- Whether transforms + metadata are stored per waypoint
- Whether dense solver diagnostics are persisted

## 5.2 Memory estimate
Expected process RSS if towns are handled sequentially:
- Small/medium towns: ~300 MB to ~1.5 GB
- Town12 worst phase (graph + optimization): ~1.5 GB to ~4.5 GB

If naive all-pairs-heavy methods are used on Town12 without pruning, spikes can be higher.

## 5.3 Compute estimate
Very algorithm-dependent:
- Fast heuristic / approximate CPP passes: ~1 to ~3 hours total for all towns
- Heavier near-exact CPP with richer diagnostics: ~3 to ~10+ hours total

Town12 dominates runtime.

## 5.4 CARLA requirement
For the current intended implementation path (using CARLA topology and map APIs):
- CARLA server must be running for the full precompute run.
- No fixed map must be preloaded manually; script can load towns sequentially.

## 5.5 Can C2 run while test_collection is running?
On the same CARLA server/port:
- Not recommended.
- Both jobs would fight over `load_world` and world state.

Safe concurrent patterns:
- Run separate CARLA instances on different ports.
- Pin C2 to one CARLA server and collection to another.
- Ensure host RAM/VRAM and CPU headroom before parallel runs.

## 6) OOM and stability notes
- Existing collector has per-frame RSS guard and bounded episode loops.
- C2 should copy that philosophy: bounded chunks, checkpoint/resume, per-town sequencing.
- Avoid global all-town in-memory graph construction.

## 7) Practical recommendation
For your goals (robust, long-term, no band-aid):
1. Keep collection runtime stable with current continuation behavior.
2. Build C2 as standalone script now (your request), with checkpoint/resume and town-sequential processing.
3. Keep B ledger generation in parallel so executed coverage can validate C2 outputs.
4. After validation, integrate selected C2 outputs into runtime scheduler.

This gives you both:
- immediate route-bank generation capability
- a safe path to eventual fully coverage-led runtime behavior

## 8) Open decisions before implementation
The following choices should be confirmed before coding C2 standalone:
- Solver mode default: `cpp-approx` vs `cpp-exact`
- Chunk size defaults (min/max waypoints)
- Route file format and metadata level
- Whether to include Town12/Town15 in first run by default (user requested yes)
- Whether to include diagnostic exports by default

## 9) Implementation status update (same day)

The standalone C2 script has now been added at:

- `scripts/build_route_atlas_cpp.py`

Current implemented capabilities:

- Solver modes: `cpp-approx`, `cpp-exact`, and `both` (default = `both`).
- Input modes: `auto`, `carla-live`, `topology-cache`.
- Output format: JSON route files + JSON manifest.
- Topology cache read/write support for repeated runs.
- Town-sequential execution.
- Route chunk controls and max-routes-per-town cap.

Smoke-test result (Town01):

- `cpp-approx` run succeeded.
- `cpp-exact` run succeeded.
- Both produced route bank artifacts and manifest metrics.

Example smoke outputs:

- `/tmp/atlas_cpp_smoke/Town01/cpp-approx/atlas_manifest.json`
- `/tmp/atlas_cpp_smoke_exact/Town01/cpp-exact/atlas_manifest.json`
