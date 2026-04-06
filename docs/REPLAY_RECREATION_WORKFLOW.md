# Replay and Recreation Workflow Scripts

This document explains the automation scripts added for targeted episode recreation,
interactive replay with overlays, and automatic TL mismatch regression checks.

## Why these scripts exist

The main goals are:

- avoid fragile multiline shell commands,
- recreate only the exact towns/episode counts you care about,
- capture enough artifacts to replay episodes exactly,
- inspect runs with pause/free-fly + TL overlays,
- automate mismatch checks at anchor frames.

## Script 1: targeted recreation

File:

- `scripts/recreate_target_episodes.py`

What it does:

- recreates only requested towns and episode counts (for example `Town10HD:1 Town06:2`),
- infers weather/traffic profile from a reference manifest,
- writes per-town generated config files under output,
- runs collection sequentially (one town command at a time),
- writes logs and a replay plan JSON.

Recommended command for your current target:

```bash
conda activate rwkv
python scripts/recreate_target_episodes.py \
  --town-episodes Town10HD:1 Town06:2 \
  --reference-manifest data/test_collection/collection_manifest.json \
  --output-dir data/replay_targets \
  --seed 42 \
  --record --debug-draw --spectator
```

Optional stop-position tuning:

```bash
python scripts/recreate_target_episodes.py \
  --town-episodes Town10HD:1 Town06:2 \
  --reference-manifest data/test_collection/collection_manifest.json \
  --output-dir data/replay_targets \
  --traffic-light-stop-distance 0.8
```

`--traffic-light-stop-distance` is ego-center distance (meters) to matched TL stop-line at full stop.

Tuning direction:

- smaller value -> later stop (closer to junction-side edge target),
- larger value -> earlier stop.

Practical sweep values: `0.8`, `0.5`, `0.3`.

## Script 2: interactive replay with overlay

File:

- `scripts/replay_with_overlay.py`

What it does:

- replays a CARLA recorder `.log`,
- optionally draws TL overlays using an episode `.h5`,
- supports interactive controls directly from terminal.

Interactive controls:

- `p`: pause/resume replay,
- `f`: toggle ego follow camera,
- `h`: help,
- `q`: quit.

When paused, free-fly with CARLA viewport controls:

- click viewport,
- hold right mouse button,
- move with `W/A/S/D`, vertical `Q/E`.

Example:

```bash
python scripts/replay_with_overlay.py episode_8b7e10e7.log \
  --episode data/replay_targets/episode_8b7e10e7.h5 \
  --town Town10HD
```

## Script 3: anchor-frame mismatch regression check

File:

- `scripts/check_tl_mismatch_frames.py`

What it does:

- checks anchor frames and windows for wrong-direction TL mismatch signatures,
- prints PASS/FAIL and exits non-zero on mismatch,
- can export JSON summary.

Default anchors:

- `413 500 1580 1976`

If an episode ends early (collision/timeout), out-of-range anchors are skipped by default and reported.

Example:

```bash
python scripts/check_tl_mismatch_frames.py \
  data/replay_targets/episode_8b7e10e7.h5 \
  --town Town10HD \
  --anchors 413 500 1580 1976 \
  --window 25 \
  --json-out data/replay_targets/mismatch_8b7e10e7.json
```

Exit behavior:

- exit `0`: PASS,
- exit `2`: FAIL (one or more anchor frames flagged).

## Captured reproducibility artifacts

Collector now stores replay-critical side artifacts per episode:

- `episode_<hash>.h5` (dataset),
- `episode_<hash>.log` (CARLA recorder),
- `episode_<hash>.route.json` (route sidecar),
- manifest entry fields: `episode_hash`, `recording_path`, `route_path`.

This enables exact episode replays for newly collected runs.

## Run order (important)

Run scripts sequentially:

1. recreate target episodes,
2. replay selected episode with overlay,
3. run mismatch regression checker.

Do not run collection and replay scripts simultaneously against the same CARLA server instance.
