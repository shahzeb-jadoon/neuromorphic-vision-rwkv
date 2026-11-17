# Town06 Traffic-Light Incident Trace Notes

This document tracks the Town06 incidents requested for follow-up analysis using the same visualization workflow as Town10HD.

## Scope

- Incident A: episode 1 around `f2098`
- Incident B: episode 2 around `f3600`

In this collection manifest, the Town06 episodes are:

1. `data/test_collection/episode_126a4655.h5` (Town06 ep1)
2. `data/test_collection/episode_51be5089.h5` (Town06 ep2)
3. `data/test_collection/episode_f82a64c0.h5` (Town06 ep3)

## Important Setup Constraint

The visualizer binds episode stop-lines to **live** CARLA traffic lights. You must load the same town first.

```bash
~/miniconda3/envs/rwkv/bin/python - <<'PY'
import carla
client = carla.Client('172.20.0.1', 2000)
client.set_timeout(120.0)
client.load_world('Town06')
print(client.get_world().get_map().name)
PY
```

### Minimal auto command (Town profile mode)

Town profile mode now auto-loads Town06 and auto-selects episode/TL IDs:

```bash
python scripts/visualize_cf183fc1_tl_stops.py --town Town06
```

Applied defaults:

- episode: `data/test_collection/episode_51be5089.h5`
- TL IDs: `3114 3127`
- host/port: `172.20.0.1:2000`
- hud side: `left`
- initial view: enabled

Default drawing behavior now hides TLs with no near-stop frame in that episode.

- For this Town06 profile, `TL3127` is hidden by default because it has no near-stop event.
- Use `--show-no-stop-tls` if you want those TLs drawn anyway (they appear with `F?`).

## Incident A: Town06 ep1 around f2098

Episode file:

- `data/test_collection/episode_126a4655.h5`

Frame check at `f2098`:

- `tlid=3114`
- `tl_state=RED (3)`
- `tl_dist=8.22 m`
- `mode=TL_STOP (4)`
- `speed=0.00 km/h`

Window scan result (`f2098 ± 220`):

- dominant red/yellow TL id: `3114`
- first near-stop for `3114`: `f2093`

Visualizer command:

```bash
~/miniconda3/envs/rwkv/bin/python scripts/visualize_cf183fc1_tl_stops.py \
  --host 172.20.0.1 \
  --port 2000 \
  --town Town06 \
  --episode data/test_collection/episode_126a4655.h5 \
  --tl-ids 3114 \
  --hud-side left \
  --set-initial-view
```

Observed mapping from short smoke run:

- `TL3114 -> live_id 106` (`bind=nearest-stopwp`, `bind_dist=12.43 m`)
- zone width 100%: `7.00 m`
- center-stop error vs yellow exit-edge center at this stop: `longitudinal=-18.34 m`, `lateral=-7.91 m`

## Incident B: Town06 ep2 around f3600

Episode file:

- `data/test_collection/episode_51be5089.h5`

Frame check at `f3600`:

- `tlid=3114`
- `tl_state=RED (3)`
- `tl_dist=8.27 m`
- `mode=TL_STOP (4)`
- `speed=0.00 km/h`

Window scan result (`f3600 ± 220`):

- dominant red/yellow TL ids: `3114` (dominant), `3127` (minor)
- first near-stop for `3114`: `f3589`
- no near-stop event for `3127` in this pass

Visualizer command:

```bash
~/miniconda3/envs/rwkv/bin/python scripts/visualize_cf183fc1_tl_stops.py \
  --host 172.20.0.1 \
  --port 2000 \
  --town Town06 \
  --episode data/test_collection/episode_51be5089.h5 \
  --tl-ids 3114 3127 \
  --hud-side left \
  --set-initial-view
```

Observed mapping from short smoke run:

- `TL3114 -> live_id 106` (`bind=nearest-stopwp`, `bind_dist=12.43 m`)
- `TL3127` has no near-stop frame in this pass (`first_stop_frame=none`), so it is hidden by default.
- use `--show-no-stop-tls` to render `TL3127` anyway for inspection.
- center-stop error for `TL3114`: `longitudinal=-18.38 m`, `lateral=-7.94 m`

## Cross-Episode Verification: f2098 (ep1) vs f3589 (ep2)

Direct numeric check in Town06 (same logic as visualizer):

- `ep1 f2098`: `tlid=3114`, `state=RED (3)`
- `ep2 f3589`: `tlid=3114`, `state=RED (3)`
- ego position distance between those two frames: `0.050 m`
- first stop for `TL3114`:
  - ep1: `f2093`
  - ep2: `f3589`
- distance between those `TL3114` stop positions: `0.049 m`
- distance between `TL3114` stop-line medians (ep1 vs ep2): `0.000 m`
- live binding in current Town06 world:
  - ep1 `TL3114 -> live_id 390`
  - ep2 `TL3114 -> live_id 390`

Conclusion: the stop around `f2098` in ep1 and stop around `f3589` in ep2 are the same physical stop at the same TL (`3114`).

Note: live actor IDs (for example `390`) can change after CARLA world reloads; the geometric match result is the stable evidence.

## Current Visualization Semantics

- Magenta box: full-lane stop zone (100%).
- Yellow edge: junction-side exit edge of the stop zone.
- Yellow point/text (`CENTER TARGET`): desired center-stop point.
- White line: actual stop point to matched live TL actor anchor.
- HUD: location bar + heading only (compass disabled).
- Width overlays are disabled by default. Re-enable with:
  - `--width-overlays 0.75 0.65`
