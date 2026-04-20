# C2 Atlas Runbook and Resume Workflow (2026-04-19)

## Purpose
This runbook captures the validated all-town execution plan for C2 atlas generation, including safe concurrent CARLA setup, and documents the implemented resume behavior so reruns continue from the last incomplete town/solver combination.

## What changed in C2 script
Script: `scripts/build_route_atlas_cpp.py`

Implemented capabilities:
- Town selection now supports list-style input, for example:
  - `--town Town03 Town05 Town06 Town10HD`
  - You can still repeat `--town` if desired.
- Resume is enabled by default.
- Completed town/solver combinations are skipped automatically on reruns.
- Incomplete or missing-manifest combinations are resumed.
- Persistent state is stored at:
  - `<output>/atlas_run_state.json`
- You can disable resume and force rebuild with:
  - `--no-resume`

Combination identity used by resume logic:
- `<town>::<solver>`
- Example: `Town06::cpp-exact`

## Exact validated runbook command plan

### 1) Start two CARLA instances on separate ports
Collection CARLA:

```bash
CarlaUE4.exe -carla-rpc-port=2000
```

C2 CARLA:

```bash
CarlaUE4.exe -carla-rpc-port=2100
```

Headless option (recommended for C2 to reduce render load):

```bash
CarlaUE4.exe -RenderOffScreen -carla-rpc-port=2100
```

Linux equivalent:

```bash
./CarlaUE4.sh -RenderOffScreen -carla-rpc-port=2100
```

Notes:
- Headless mode reduces graphics/viewport overhead (mainly VRAM/GPU pressure).
- C2 still loads full map topology and solves graph optimization, so Town12/Town15 can still dominate CPU/RAM.

### 2) Prepare environment

```bash
cd ~/projects/neuromorphic-vision-rwkv
conda activate rwkv
mkdir -p data/atlas_cpp/logs
```

### 3) Optional connectivity probe for both ports

```bash
~/miniconda3/envs/rwkv/bin/python - <<'PY'
import carla
host = "172.20.0.1"
for port in (2000, 2100):
    c = carla.Client(host, port)
    c.set_timeout(15.0)
    print(port, c.get_world().get_map().name)
PY
```

### 4) Run full all-town C2 build on dedicated C2 CARLA (port 2100)

```bash
~/miniconda3/envs/rwkv/bin/python scripts/build_route_atlas_cpp.py \
  --all-towns \
  --input-mode carla-live \
  --carla-host 172.20.0.1 \
  --carla-port 2100 \
  --write-topology-cache \
  --cache-dir data/atlas_cpp/topology_cache \
  --output data/atlas_cpp/all_towns_live \
  --solver both \
  --chunk-min-waypoints 120 \
  --chunk-max-waypoints 260 \
  --max-routes-per-town 150 \
  --log-level INFO | tee data/atlas_cpp/logs/c2_live_all_towns.log
```

### 5) Run collection concurrently on separate CARLA (port 2000)

Wrapper route:

```bash
bash scripts/test_collection.sh
```

Explicit host/port route:

```bash
~/miniconda3/envs/rwkv/bin/python scripts/collect_data.py \
  --config configs/collection.yaml \
  --output data/test_collection \
  --town Town03 Town05 Town06 Town10HD \
  --weather ClearNoon \
  --episodes 1 \
  --spectator \
  --debug-draw \
  --route-strategy balanced \
  --target-left 0.15 \
  --target-right 0.15 \
  --host 172.20.0.1 \
  --port 2000
```

### 6) Optional cache-only fallback (no live CARLA dependency)

```bash
~/miniconda3/envs/rwkv/bin/python scripts/build_route_atlas_cpp.py \
  --all-towns \
  --input-mode topology-cache \
  --cache-dir data/atlas_cpp/topology_cache \
  --output data/atlas_cpp/all_towns_cache \
  --solver both \
  --chunk-min-waypoints 120 \
  --chunk-max-waypoints 260 \
  --max-routes-per-town 150 \
  --log-level INFO | tee data/atlas_cpp/logs/c2_cache_all_towns.log
```

## Solver selection and differences

Use one solver only:

cpp-approx only:

```bash
~/miniconda3/envs/rwkv/bin/python scripts/build_route_atlas_cpp.py \
  --all-towns \
  --input-mode topology-cache \
  --cache-dir data/atlas_cpp/topology_cache \
  --output data/atlas_cpp/all_towns_approx \
  --solver cpp-approx \
  --max-routes-per-town 150 \
  --log-level INFO
```

cpp-exact only:

```bash
~/miniconda3/envs/rwkv/bin/python scripts/build_route_atlas_cpp.py \
  --all-towns \
  --input-mode topology-cache \
  --cache-dir data/atlas_cpp/topology_cache \
  --output data/atlas_cpp/all_towns_exact \
  --solver cpp-exact \
  --max-routes-per-town 150 \
  --log-level INFO
```

Practical difference:
- `cpp-approx`:
  - Uses a greedy balancing assignment.
  - Faster and lighter for large maps.
  - May add more duplicate/deadhead traversals than exact mode.
- `cpp-exact`:
  - Uses min-cost flow (`network_simplex`) for balancing.
  - Usually lower deadhead overhead in modeled costs.
  - Slower and heavier on large maps (especially Town12).
  - Falls back to approx if exact balancing cannot be resolved for a component.

Recommendation:
- Use `cpp-approx` for rapid iteration or repeated operational runs.
- Use `cpp-exact` when you want higher route-efficiency and can afford longer runtime.

### 7) Post-run verification

```bash
find data/atlas_cpp/all_towns_live -name atlas_manifest.json | wc -l
```

```bash
~/miniconda3/envs/rwkv/bin/python - <<'PY'
from pathlib import Path
import json
root = Path("data/atlas_cpp/all_towns_live")
manifests = sorted(root.glob("Town*/cpp-*/atlas_manifest.json"))
print("manifest_count", len(manifests))
bad = []
for p in manifests:
    m = json.loads(p.read_text())
    if m.get("route_count") != 150:
        bad.append((str(p), m.get("route_count")))
print("route_count_mismatches", len(bad))
for row in bad[:20]:
    print(row[0], row[1])
PY
```

## Resume behavior examples

### A) Start only test_collection towns

```bash
~/miniconda3/envs/rwkv/bin/python scripts/build_route_atlas_cpp.py \
  --town Town03 Town05 Town06 Town10HD \
  --input-mode carla-live \
  --carla-host 172.20.0.1 \
  --carla-port 2100 \
  --write-topology-cache \
  --cache-dir data/atlas_cpp/topology_cache \
  --output data/atlas_cpp/test_collection_towns \
  --solver both \
  --max-routes-per-town 150 \
  --log-level INFO
```

If CARLA crashes mid-run, run the same command again.
Behavior on rerun:
- Completed combinations (for example `Town03::cpp-approx`, `Town03::cpp-exact`) are skipped.
- Incomplete combinations continue.
- If a completed combination is missing its `atlas_manifest.json`, that combo is rebuilt.

### B) Default run after prior subset completion
If `Town03` is already completed in the same output directory state, rerunning default/all-town mode skips `Town03` and advances to the next unresolved combinations.

### B2) Resume behavior when running one solver only
Resume tracks combinations as `<town>::<solver>`.

If you run only one solver (for example `--solver cpp-approx`):
- Only `TownX::cpp-approx` combinations are created/updated.
- Rerunning `--solver cpp-approx` skips completed approx combinations and resumes incomplete approx combinations.
- `cpp-exact` combinations are untouched.

If later you switch solver mode:
- Running `--solver cpp-exact` on the same output directory starts/continues exact combinations independently.
- Running `--solver both` will skip already completed combinations from either solver and only run missing/incomplete ones.

### C) Force rebuild everything

```bash
~/miniconda3/envs/rwkv/bin/python scripts/build_route_atlas_cpp.py \
  --all-towns \
  --input-mode topology-cache \
  --cache-dir data/atlas_cpp/topology_cache \
  --output data/atlas_cpp/all_towns_live \
  --solver both \
  --max-routes-per-town 150 \
  --no-resume \
  --log-level INFO
```

### D) How to interpret "Skipping town ... solver=... (already completed)"

This message is expected when resume mode is active (default) and that
`<town>::<solver>` combination is already marked complete with a present
manifest.

Relevant behavior in script:
- State key lives in `<output>/atlas_run_state.json`.
- Completion is synchronized from `atlas_manifest.json`.
- Completed combinations are skipped automatically.

So, in a mixed run like:
- Town03/Town05 skipped
- Town06 cpp-approx rebuilt
- Town06 cpp-exact skipped

the script is correctly resuming only missing/incomplete combinations.

### E) Validation workflow for `data/atlas_cpp/test_collection_towns`

Use this after running:

```bash
~/miniconda3/envs/rwkv/bin/python scripts/build_route_atlas_cpp.py \
  --town Town03 Town05 Town06 Town10HD \
  --input-mode carla-live \
  --carla-host 172.20.0.1 \
  --carla-port 2100 \
  --write-topology-cache \
  --cache-dir data/atlas_cpp/topology_cache \
  --output data/atlas_cpp/test_collection_towns \
  --solver both \
  --max-routes-per-town 150 \
  --log-level INFO
```

#### E1) Check completion matrix + validation gate status

```bash
cd ~/projects/neuromorphic-vision-rwkv
~/miniconda3/envs/rwkv/bin/python - <<'PY'
from pathlib import Path
import json

root = Path("data/atlas_cpp/test_collection_towns")
towns = ["Town03", "Town05", "Town06", "Town10HD"]
solvers = ["cpp-approx", "cpp-exact"]

for t in towns:
  for s in solvers:
    m = root / t / s / "atlas_manifest.json"
    if not m.exists():
      print(t, s, "MISSING")
      continue
    obj = json.loads(m.read_text())
    vg = obj.get("validation_gates", {})
    print(
      t,
      s,
      "routes=", obj.get("route_count"),
      "combo_cov=", obj.get("combo_stats", {}).get("coverage_ratio"),
      "errors=", vg.get("errors"),
      "gates=", vg.get("gates"),
    )
PY
```

Expected healthy result:
- all manifests present
- `errors=[]`
- all gate booleans true

#### E2) Verify route waypoint contract bounds from route files

```bash
cd ~/projects/neuromorphic-vision-rwkv
~/miniconda3/envs/rwkv/bin/python - <<'PY'
from pathlib import Path
import json
import glob

root = Path("data/atlas_cpp/test_collection_towns")
towns = ["Town03", "Town05", "Town06", "Town10HD"]
solvers = ["cpp-approx", "cpp-exact"]

for t in towns:
  for s in solvers:
    m = root / t / s / "atlas_manifest.json"
    if not m.exists():
      continue
    obj = json.loads(m.read_text())
    mn = int(obj["settings"]["chunk_min_waypoints"])
    mx = int(obj["settings"]["chunk_max_waypoints"])
    routes = sorted(glob.glob(str(root / t / s / "routes" / "route_*.json")))
    under = 0
    over = 0
    for rf in routes:
      ro = json.loads(Path(rf).read_text())
      w = int(ro.get("waypoint_count", 0))
      if w < mn:
        under += 1
      if mx > 0 and w > mx:
        over += 1
    print(t, s, "under=", under, "over=", over)
PY
```

Expected healthy result:
- `under=0`
- `over=0`

#### E3) Verify route schema + command distribution sanity

```bash
cd ~/projects/neuromorphic-vision-rwkv
~/miniconda3/envs/rwkv/bin/python - <<'PY'
from pathlib import Path
import json
import glob
from collections import Counter

root = Path("data/atlas_cpp/test_collection_towns")
towns = ["Town03", "Town05", "Town06", "Town10HD"]
solvers = ["cpp-approx", "cpp-exact"]
required = {"x", "y", "z", "yaw", "road_option"}

for t in towns:
  for s in solvers:
    routes = sorted(glob.glob(str(root / t / s / "routes" / "route_*.json")))
    missing = 0
    labels = Counter()
    for rf in routes:
      ro = json.loads(Path(rf).read_text())
      wps = ro.get("waypoints", [])
      if not wps or not required.issubset(wps[0].keys()):
        missing += 1
      for wp in wps:
        labels[str(wp.get("road_option"))] += 1
    print(t, s, "bad_routes=", missing, "labels=", dict(labels))
PY
```

Expected healthy result:
- `bad_routes=0`
- label mix is not degenerate (not all one class).

#### E4) Force clean rebuild test (disable resume)

```bash
cd ~/projects/neuromorphic-vision-rwkv
~/miniconda3/envs/rwkv/bin/python scripts/build_route_atlas_cpp.py \
  --town Town03 Town05 Town06 Town10HD \
  --input-mode topology-cache \
  --cache-dir data/atlas_cpp/topology_cache \
  --output /tmp/atlas_cpp_test_collection_rebuild \
  --solver both \
  --max-routes-per-town 150 \
  --no-resume \
  --strict-validation-gates \
  --log-level INFO
```

Use this when you want to validate current code behavior from scratch, ignoring
any prior run-state.

### F) Common shell pitfall (here-doc indentation)

If a `python - <<'PY'` command fails with syntax/indentation errors, check that:
- each `for` block is properly indented,
- `PY` terminator appears alone on its own line,
- no accidental pasted fragments were inserted mid-line.

Safe template:

```bash
cd ~/projects/neuromorphic-vision-rwkv
~/miniconda3/envs/rwkv/bin/python - <<'PY'
print("ok")
PY
```

## Safe execution checklist
1. Verify AdditionalMaps availability before all-town launch (Town06, Town07, Town12, Town15).
2. Keep strict CARLA port isolation:
   - Collection on 2000
   - C2 on 2100
3. Do not run collection and replay against the same CARLA instance.
4. Keep separate output roots for C2 and collection.
5. Expect Town12 to dominate runtime.
6. If dual-instance resource pressure occurs, stop C2 live mode and use cache-only mode.
7. If run interrupts, rerun the same command and let resume skip completed combinations.
8. Confirm expected manifest count after completion.
9. Archive logs and `atlas_run_state.json` for traceability.
10. For repeated large-map campaigns (especially Town12/Town15), run one live pass with `--write-topology-cache`, then prefer `--input-mode topology-cache` for improved memory stability.

## Validation notes from implementation session
- Multi-town selection with one `--town` flag was validated.
- Immediate rerun skip behavior was validated.
- Partial recovery was validated by removing one manifest (`Town06/cpp-exact`) and rerunning; only that missing combination was rebuilt.
