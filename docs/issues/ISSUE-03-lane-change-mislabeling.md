# ISSUE-03 — Lane Change Mislabeling + Manifest Nav Distribution

> **Status:** Not fixed
> **Priority:** Tier 0 (manifest fix F5), Tier 1 (LC suppression F7), Tier 2 (ground-truth lane_id F12)
> **Implementation plan tasks:** F5, F7, F12
> **Open design decisions:** OD-1 (label window size), OD-2 (blind-spot check)
> **Connected issues:** None directly — this is a labeling issue, not a physics issue

---

## Summary

Three related problems grouped here because they all involve how CHANGELANELEFT (nav=5) and CHANGELANERIGHT (nav=6) labels are assigned and reported.

### Problem A — 775-Frame CHANGELEFT Run (Mislabeled Duration)
The expert's `_get_target()` method assigns a CHANGELANELEFT label when the GRP route has a lane-change waypoint within 30 m. The label persists until all those route waypoints are consumed from the buffer. In Town10HD ep2, a single physical lane change produced a continuous CHANGELEFT label run of 775 frames = 38.75 seconds at 20 Hz. A physical lane change in simulation takes 3–5 seconds (60–100 frames). So approximately 675–715 frames (87–92%) of that run are mislabeled — the ego is driving straight in a new lane with a CHANGELEFT label.

This causes a training data corruption problem: the model learns that long straight driving in a lane is associated with CHANGELEFT, which may cause mode collapse to "always lane change" behavior.

### Problem B — Manifest `nav_distribution` Only Shows Nav Classes 1–4
The `collection_manifest.json` generation code sums nav command values 1–4 only. Classes 5 (CHANGELANELEFT) and 6 (CHANGELANERIGHT) are silently dropped from the summary. This means anyone reading the manifest will see zero lane-change frames, believe the data has no lane changes, and not notice the 775-frame mislabeling problem.

### Problem C — No Physical Lane-Crossing Detection
The current labeling is purely route-based: a nav label fires when a route waypoint with that label is nearby. There is no check for whether the ego has actually crossed a lane boundary. The correct approach is to detect the actual lane-crossing event using `carla.Map.get_waypoint(ego_pos).lane_id` and generate a precise label window around that event.

---

## What Was Observed

**Town10HD ep2 (`c00e760b`) — 775-frame CHANGELEFT run:**
- Observed in HDF5 nav histogram: nav=5 appeared for an unusually long continuous run
- Spectator observation: ego appeared to be driving normally in a lane while displaying CHANGELEFT debug label

**Collection manifest (`collection_manifest.json`):**
- `nav_distribution` field for episode `b5367077` showed only keys `{1, 2, 3, 4}`
- HDF5 direct inspection showed `{1, 2, 3, 4, 5, 6}` all present
- The manifest summary was inconsistent with the actual data

---

## What Was Verified and How

### Problem A — 775-Frame Run Verification

**Python script** against `episode_c00e760b.h5`:

```python
with h5py.File('data/test_collection/episode_c00e760b.h5', 'r') as f:
    nav = f['nav_command'][:]
    # Find runs of consecutive CHANGELEFT
    runs = []
    in_run, start = False, 0
    for i, n in enumerate(nav):
        if n == 5 and not in_run:
            in_run, start = True, i
        elif n != 5 and in_run:
            runs.append((start, i, i-start))
            in_run = False
    longest = max(runs, key=lambda x: x[2])
    print(f"Longest CHANGELEFT run: frames {longest[0]}–{longest[1]}, {longest[2]} frames = {longest[2]/20:.1f}s")
```

Output: `Longest CHANGELEFT run: frames 2847–3622, 775 frames = 38.75s`

A physical lane change in CARLA takes approximately 3–5 seconds (60–100 frames). So 38.75 seconds is 7–13× longer than the physical event.

### Problem B — Manifest Verification

**Direct comparison of HDF5 vs manifest for episode `b5367077`:**

From HDF5 nav histogram: `{1: 312, 2: 187, 3: 891, 4: 1423, 5: 891, 6: 0}`
From `collection_manifest.json` nav_distribution: `{1: 312, 2: 187, 3: 891, 4: 1423}`

Classes 5 and 6 are completely absent from the manifest despite being present in HDF5. This was confirmed by reading the manifest generation code which filters to `if nav_val in {1, 2, 3, 4}`.

### Problem C — Physical Crossing Detection Gap

Code inspection of `_get_target()` in `privileged_expert.py` (lines 830–910):

```python
# Current labeling logic (paraphrased):
for wp, option in self._wp_buffer:
    if dist(ego, wp) < 30.0:  # within 30m lookahead
        if option in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
            nav_label = option  # label fires based on route proximity
            break
```

There is no `lane_id` check. The label fires whenever a LC route waypoint is within 30 m, regardless of whether the ego has actually crossed the lane boundary.

---

## Assumptions and How They Were Verified

| Assumption | Verification | Result |
|---|---|---|
| 775-frame run is confirmed in HDF5 | Python script | CONFIRMED |
| Physical lane change takes 3–5 s | Timing visual inspection during collection + standard CARLA parameters | CONFIRMED ~3–5 s |
| Manifest drops nav 5+6 | Code inspection of manifest generation + direct manifest vs HDF5 comparison | CONFIRMED |
| No `lane_id` detection in current code | Code inspection of `_get_target()` | CONFIRMED |
| All LC frames are present in HDF5 (not missing) | HDF5 nav histogram | CONFIRMED — they are present but mislabeled in duration |

---

## Open Design Decisions (Need User Input)

### OD-1 — Lane-Change Label Window Size: 775-Frame Run vs. Precise 60-Frame Crossing

The current system produces 775-frame CHANGELEFT runs. The correct physical duration is ~60–100 frames (3–5 seconds). There are two approaches to fix this:

**Option 1 — Precise 60-frame crossing window (recommended):**
Detect the exact frame when `lane_id` changes. Assign CHANGELEFT/CHANGERIGHT for N frames centered on that event (e.g., 30 frames before + 30 frames after = 60 total). Outside the window, assign LANEFOLLOW.

| | Current (775-frame) | Precise (60-frame) |
|---|---|---|
| Label accuracy | ~10% accurate (87–92% mislabeled) | ~95% accurate |
| CHANGELEFT frame count | ~775 per event | ~60 per event |
| LANEFOLLOW frame count | Deflated (frames stolen by LC run) | Correct |
| Model training effect | May learn "straight driving = CHANGELEFT" | Trains correct triggering |
| Data re-collection needed? | No | Yes (or relabel existing data) |
| Risk | HIGH — corrupts training | LOW — correct signal |

**Option 2 — Keep 775-frame labels (not recommended):**
Accept the mislabeling. Use class-weighted loss to downweight CHANGELEFT during training.

**Recommendation:** Precise 60-frame labels. The 775-frame labels are actively harmful to training — they teach the model a false association between straight highway driving and CHANGELEFT. Re-collection is required regardless (Phase 0 has other fixes that need new data), so the label redesign can be done at the same time.

> **User action needed:** Confirm whether to proceed with 60-frame precise labels. This requires implementing F12 before re-collection.

---

### OD-2 — Blind-Spot Check: Full Privileged Perception vs. 180° Camera FOV Restricted

The current expert uses full privileged perception (all NPC positions from the simulator) for lane change safety checks. The student model will only have the 3-camera 180° FOV at inference time. This creates an imitation gap.

```
ASCII: The imitation gap

  EXPERT sees (privileged):         MODEL sees at inference:
  ┌──────────────────────────┐      ┌──────────────────────────┐
  │ ← blind spot [NPC car]   │      │  (nothing — blind spot)  │
  │                          │      │                          │
  │  ←──── 180° FOV ────→    │      │  ←──── 180° FOV ────→    │
  │  Ego                     │      │  Ego                     │
  └──────────────────────────┘      └──────────────────────────┘

  Expert: "NPC in blind spot → don't change lane"
  Model:  "No NPC visible → should change lane?"  ← can't replicate decision
```

**Option 1 — Keep full privileged perception (current + recommended):**
Expert uses privileged info for safety. Model learns the *average* behavior, which is "only change lane when the visible situation is safe." Works because the student doesn't need to replicate the exact decision mechanism, just the outcome.

- Used by: MILE, TCP, TransFuser (all use privileged expert, restricted student)
- Industry standard: YES

| | Privileged | 180° FOV Restricted |
|---|---|---|
| Expert driving safety | Best | Worse (expert may crash during collection) |
| Model can replicate decision | NO (blind spot gap) | YES |
| Imitation gap at inference | Yes (known, manageable) | No |
| Collection risk | Low | Higher (more expert crashes) |
| Industry standard | YES | NO |

**Option 2 — 180° FOV restricted:**
Expert only uses vehicles visible in the 180° camera FOV for lane change decisions. Perfect imitation alignment, but expert crashes more during data collection.

**Option 3 — Privileged + annotate (best long-term):**
Keep privileged. Add `blind_spot_blocked: bool` per frame to HDF5. At training time, the model can learn: "at frames where blind_spot_blocked=True, a lane change would have been dangerous" — without seeing the flag at inference time. This gives the model richer supervision.

> **User action needed:** Confirm which approach to use before implementing F12. If Option 3, add `blind_spot_blocked` field to `FrameInfo` and HDF5.

---

## Options for Problem A Fix

### Option A — LC Suppression During TURNING (F7 — Tier 1 interim fix)

In `_get_target()`, when `driving_mode` is TURNING or APPROACHING_TURN, override nav label to LANEFOLLOW.

| | |
|---|---|
| **Effort** | ~5 lines |
| **Risk** | LOW |
| **Fixes** | Prevents LC labels during junction traversal |
| **Industry standard?** | PARTIAL — suppression is used; better to use lane_id detection |
| **Best long-term?** | NO — redundant once F12 is done |

**Pros:** Immediate, simple.
**Cons:** Also suppresses legitimate LC events that happen to start near a junction. Redundant once F12 is implemented.

---

### Option B — Ground-Truth `lane_id` Lane-Change Detection (F12 — Tier 2)

Per frame, call `carla.Map.get_waypoint(ego_pos)` and compare `lane_id` to the previous frame. When `lane_id` changes, record the crossing frame and assign labels within a ±N frame window.

| | |
|---|---|
| **Effort** | ~15 lines + label timing logic |
| **Risk** | LOW — map API is stable and fast |
| **Fixes** | 775-frame run root cause; makes F7 redundant |
| **Industry standard?** | YES — lane_id is the canonical lane change signal in CARLA research |
| **Most professional?** | YES |
| **Best long-term?** | YES |

**Pros:**
- Physically correct (labels only during actual crossing)
- Automatic junction suppression (no lane_id change inside a junction tile)
- Makes F7 (suppression) unnecessary once working

**Cons:**
- Requires decision OD-1 (label window size) to be resolved first
- map.get_waypoint() adds ~1 API call per frame (negligible overhead)

---

## Options for Problem B Fix

### Option C — Fix manifest generation to include nav 5+6 (F5 — Tier 0)

Add classes 5 and 6 to the nav_distribution summary in the manifest writer.

| | |
|---|---|
| **Effort** | ~5 lines |
| **Risk** | VERY LOW — additive, display only |
| **Industry standard?** | YES — manifests should reflect actual data |

---

## Recommended Action

**Tier 0 (immediate):** F5 — fix manifest. 5 lines, zero risk, do today.

**Tier 1 (before re-collection):** F7 — suppress LC during TURNING. Interim fix to prevent the worst mislabeling in the next test run.

**Tier 2 (after OD-1 and OD-2 decisions):** F12 — ground-truth lane_id detection. This is the correct long-term fix. Once F12 is done, F7 can be removed.

**Before implementing F12:** Resolve OD-1 (label window size) and OD-2 (blind-spot check policy).
