# Episode cf183fc1 Trace Notes

This document stores verified findings as we inspect the incident step-by-step.

## Step 1: Spawn Pose, Initial Heading, and J1 Relation

### Verified from HDF5 (frame 0)

- File: `data/test_collection/episode_cf183fc1.h5`
- Frame 0 position: `(-113.35, -25.39)`
- Frame 0 rotation vector: `[0.0, 94.33519, 0.0]`
- Frame 0 mode: `TL_STOP (4)`
- Frame 0 TL actor id: `2789`
- Frame 0 TL state: `RED (3)`
- Frame 0 TL distance: `24.79 m`
- Frame 0 matched stop line: `(-110.23, -0.80)`

### Heading interpretation

The yaw component in this dataset is the **second** rotation entry (`rotation[:, 1]`).

- At spawn, yaw is approximately `94°`, which means ego is facing roughly south in CARLA's Town10HD frame.
- Early frame motion vectors (frames 9-25) show positive Y movement and heading angles around 80-86°, consistent with southbound approach toward the first stop line.

### J1 relation to ego at spawn

Using your naming where TL `2789` is J1:

- Ego starts north of J1's stop line and approaches it southbound.
- TL `2789` is already active red at frame 0.

This confirms the first visible interaction is approach/hold for J1 red, not a post-J1 event.

## Step 1b: How many TL-controlled junction signatures appear before f1580?

### Unique red/yellow TL signatures up to frame 1580

Extracted unique `(tl_id, stop_line_xy)` where TL state is yellow or red:

1. `2789` -> `(-110.23, -0.80)`
2. `2778` -> `(-88.82, 16.44)`
3. `2779` -> `(83.68, 28.42)`
4. `2782` -> `(106.09, 40.59)`

There is also a green TL actor observed in this range:

- `2792` -> `(-64.09, 24.51)` (green-only in this window)

### Answer to "only J1/J2/J3 until f1580?"

From telemetry signatures alone, this stage observed four red/yellow TL IDs before `f1580`.

- We see four distinct red/yellow TL actors (`2789`, `2778`, `2779`, `2782`) before frame 1580.
- The map-context grouping was then resolved and is documented in Step 1c.

## Step 1c: Verified Junction Grouping and TL Direction Catalog

For this investigation, we now use the verified grouping below:

- J1 contains `TL2789` and `TL2778`.
- J3 contains `TL2779` and `TL2782`.

Stop-line and direction references used for all mismatch checks:

| TL ID | Stop-line coordinate (x, y) | Waypoint yaw (deg) | Guided traffic direction |
|---:|---|---:|---|
| 2789 | (-110.232, -0.797) | 90.64 | Southbound |
| 2778 | (-88.824, 16.437) | -179.84 | Westbound |
| 2779 | (83.684, 28.417) | -359.84 (equiv. to 0.16) | Eastbound |
| 2782 | (106.089, 40.589) | -89.61 | Northbound |

## Step 2: Frame-Level Mismatch Verification (Wrong-Direction TL Association)

### Key check

For a valid same-lane stop-line match, we expect:

- modest heading difference between ego and stop-line lane direction,
- lateral offset near lane center,
- stop line ahead of ego in ego-forward axis.

### Verified metrics (live Town10HD waypoint yaw + episode telemetry)

| frame | TL id | state | yaw diff (deg) | ego->stop lateral (m) | ego lane width (m) |
|---:|---:|---:|---:|---:|---:|
| 413 | 2778 | 3 (red) | 178.58 | -8.00 | 3.50 |
| 500 | 2778 | 3 (red) | 178.99 | -7.97 | 3.50 |
| 1580 | 2782 | 3 (red) | 159.27 | -5.27 | 3.50 |
| 1976 | 2782 | 1 (green) | 173.58 | -6.67 | 3.50 |

Interpretation:

- Lateral offsets (5-8 m) are multiple lane-width halves away from ego centerline.
- Yaw differences (~159-179 deg) indicate opposite-direction lane alignment.
- This is consistent with a wrong-direction stop-line association, not a same-lane stop target.

## Step 3: Stop Order for Requested TL IDs in cf183fc1

Using first **stop-after-motion** criterion (`speed <= 0.8 km/h` with prior motion on same TL):

1. `TL2789` -> `Stop 1` (stop-after-motion frame: `f94`, raw first near-stop was `f0`)
2. `TL2778` -> `Stop 2` (first near-stop frame: `f413`)
3. `TL2779` -> `Stop 3` (first near-stop frame: `f1204`)
4. `TL2782` -> `Stop 4` (first near-stop frame: `f1549`)

## Step 3b: Verified Failure Narrative (What Happens in Sequence)

This sequence is now explicitly captured as verified behavior for `episode_cf183fc1.h5`:

1. Ego starts approaching J1 and correctly tracks `TL2789`; first stop-after-motion is at `f94`.
2. When `TL2789` turns green, ego enters J1 and starts turning east.
3. Inside J1, ego associates with `TL2778` (red), which guides westbound traffic in this junction phase, and incorrectly stops for it at `f413` while ego is turning east.
4. Ego then continues toward J3 (passing J2 in between) and correctly tracks `TL2779`; it slows and performs an emergency stop behind an already-stopped lead vehicle (this part is expected behavior).
5. When `TL2779` turns green and the lead vehicle proceeds, ego enters J3 and turns south.
6. Inside J3, ego associates with `TL2782` (red), and incorrectly stops for it on the crosswalk/pedestrian-crossing area while in the turn.

This timeline matches the geometric mismatch signature in Step 2 for both problematic associations (`TL2778` and `TL2782`).

## Step 4: Live CARLA Visual Confirmation (Free-Roam + Overlay)

### Goal

Show, in live Town10HD:

- a **magenta box** around each requested TL (`2778`, `2789`, `2779`, `2782`),
- TL label text (`TL2778`, etc),
- `Stop 1..Stop 4` markers where ego first near-stopped for that TL in this episode.

### Important runtime note

In a fresh CARLA world, actor IDs are often reallocated, so episode TL IDs may not exist directly.

To keep labels faithful to the episode, the visualizer:

1. tries direct actor lookup by requested ID,
2. falls back to nearest live traffic light using that TL's episode stop-line position against **live stop-waypoints** (not pole location).

Why this matters: traffic-light pole coordinates can be tens of meters away from their stop lines in Town10HD. Stop-waypoint-based binding avoids swapped TL labels inside shared junctions.

### Script added

- `scripts/visualize_cf183fc1_tl_stops.py`

### Run command

```bash
~/miniconda3/envs/rwkv/bin/python scripts/visualize_cf183fc1_tl_stops.py \
	--host 172.20.0.1 \
	--port 2000 \
	--town Town10HD \
	--episode data/test_collection/episode_cf183fc1.h5 \
	--tl-ids 2778 2789 2779 2782 \
	--highlight-live-ids 15 14 21 22 \
	--hud-side left \
	--set-initial-view
```

### Minimal auto command (Town profile mode)

The script now supports town-only profile mode and auto-loads the map when needed.

```bash
python scripts/visualize_cf183fc1_tl_stops.py --town Town10HD
```

This automatically applies profile defaults:

- episode: `data/test_collection/episode_cf183fc1.h5`
- TL IDs: `2778 2789 2779 2782`
- host/port: `172.20.0.1:2000`
- hud side: `left`
- initial view: enabled

### Live location bar overlay

The visualizer now renders a spectator HUD that updates every refresh tick:

- Location bar: current spectator `x, y`, `yaw`, and heading label (`northbound/southbound/eastbound/westbound`).

Flags:

- `--hud-side left|right`: side for location bar.
- `--compass-side left|right|opposite`: deprecated and ignored (kept for compatibility).
- `--highlight-live-ids <id...>`: draw selected live TL actors in green (for direct map-side inspection).

### Stop-frame selection policy

Default behavior now prefers the first **stop-after-motion** per TL (avoids counting spawn hold at frame 0 as the stop point).

- If an initial hold exists at spawn (e.g., `raw:F0`) and later motion-stop exists, the later frame is used for `Stop N` labeling.
- To force the old behavior, add `--allow-initial-hold-stop`.

### Controls for free roam

In CARLA UE viewport:

- click viewport to focus,
- hold right mouse button and use mouse look,
- move with `W/A/S/D`, vertical with `Q/E`.

### Overlay semantics

- Magenta box: lane-local stop zone derived from live stop-waypoints for the matched TL.
- Yellow exit edge: junction-side edge of that stop zone.
- Yellow center marker/text (`CENTER TARGET`): desired ego-center stop point on the yellow exit edge.
- TL label: `TL<episode_id>` plus `Stop <order>`.
- TL state shown inline: `[RED]`, `[YELLOW]`, `[GREEN]`, etc.
- White point/text: ego first near-stop position for that TL in cf183fc1.
- White line: matched live TL actor anchor -> that stop point.
- Green box: explicitly requested live TL IDs (raw live world actors).

Operational interpretation used during review:

- If the white stop marker for the currently relevant TL lands near that TL's yellow center target, stop placement is lane-consistent.
- If the stop marker is far from the relevant TL's yellow center target and aligns with a cross-direction TL zone, this is treated as a wrong-direction TL association candidate.

Width comparison overlays (inside magenta zone):

- `75%` width = **ORANGE**
- `65%` width = **SKY**

Default overlay state:

- width overlays are **disabled by default**.
- re-enable explicitly with: `--width-overlays 0.75 0.65`

### When done

Stop the overlay script with `Ctrl+C` in the terminal.

## Step 5: TL Logic Fix Plan (Next Steps)

The evidence above indicates we need a planner-side correction, not just visualization.

1. Add Tier-1 plausibility gate before reusing stored TL stop data.
	- File: `agents/expert/privileged_expert.py`
	- Area: `_check_traffic_light` Tier-1 stored-stop path around the `self._tl_stop_loc` reuse block.
	- Change: before returning `(tl_actor.state, stop_dist)`, re-check heading/lateral plausibility for the stored stop location and clear memory when implausible.

2. Prefer stop-waypoint candidates over affected-lane candidates in stop-line matching.
	- File: `agents/expert/privileged_expert.py`
	- Area: `_stop_line_distance` candidate iteration.
	- Change: try `tl.get_stop_waypoints()` first, then fallback to `tl.get_affected_lane_waypoints()`.

3. Keep ahead/behind checks but add explicit lateral guard tied to lane width.
	- Suggested rule: reject candidate when lateral offset is greater than half-lane width plus tolerance.

4. Add mismatch debug telemetry for validation runs.
	- Log per-frame: ego yaw, candidate stop yaw, yaw delta, longitudinal offset, lateral offset, selected TL id.
	- This makes regressions detectable without manual frame-by-frame inspection.

5. Validate on known failure frames and expected non-failure sequence.
	- Known mismatch frames to clear: `f413`, `f500`, `f1580`, `f1976`.
	- Keep expected behavior: `TL2789` stop at `f94`, `TL2779` lead-vehicle constrained stop behavior before entering J3.

6. Re-run visual confirmation with magenta/yellow overlays after planner patch.
	- Pass condition: stop markers should align with the relevant TL's yellow center target, and no in-junction cross-direction TL latch should persist.
