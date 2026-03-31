# ISSUE-06 — Monitoring & Instrumentation Gaps

> **Status:** Not fixed
> **Priority:** Tier 0 (F2, F3, F4, F5 — all trivial)
> **Implementation plan tasks:** F2, F3, F4, F5
> **Connected issues:** [ISSUE-01](ISSUE-01-junction-waypoints.md) (F3/F4 are required to verify ISSUE-01 fixes), [ISSUE-03](ISSUE-03-lane-change-mislabeling.md) (F5 makes manifest accurate for lane-change monitoring)

---

## Summary

Four small instrumentation and configuration problems grouped here because they are all Tier 0 (zero risk, no logic changes) and should all be done in a single sitting before any other work.

| # | Fix | File | What's wrong | What to change |
|---|-----|------|-------------|----------------|
| F1 | `stop_sign_route_proximity` | `configs/collection.yaml` | 2.5 m too tight for curved approaches | Change to 4.0 m |
| F2 | `timeout_duration` | `configs/collection.yaml` | 75 s causes spurious timeouts | Change to 80 s |
| F3 | `gap_fill_diverged` → HDF5 | `src/data/expert_collector.py` | Computed in `FrameInfo`, never written to HDF5 | Add 2 lines to dataset writer |
| F4 | `route_completion` → HDF5 | `src/data/expert_collector.py` | Computed as property, never persisted | Add scalar dataset per episode |
| F5 | Manifest `nav_distribution` | manifest writer | Only sums nav 1–4, hides classes 5+6 | Add nav 5 and 6 to the sum |

*(F1 is documented in full in [ISSUE-05](ISSUE-05-stop-sign-proximity.md). Included in the table above for completeness.)*

---

## F2 — Timeout Duration (75 s → 80 s)

### What was observed

Town03 ep2 and ep3 timed out during the test collection run. The timeout episodes contributed 0 route-completion data and were excluded from the nav distribution stats. The current `timeout_duration: 75` in `configs/collection.yaml` is used as the per-episode wall-clock timeout.

### Root cause

The Town03 deadlock bug (documented in [ISSUE-02](ISSUE-02-traffic-light-state-machine.md), Bug C) caused the ego to stall at an intersection. Even without the deadlock, 75 s is tight for longer routes in towns with dense traffic. Increasing to 80 s gives more margin before the episode is killed.

### Verification

Two Town03 episodes timed out in the 18-episode test run. No other towns timed out. The deadlock bug (ISSUE-02 Bug C, F8) is the primary fix — but increasing the timeout ensures marginal cases (near-route-end stalls, pedestrian stops on long routes) don't also hit the limit.

### Fix

`configs/collection.yaml`: change `timeout_duration: 75` → `timeout_duration: 80`.

### Options

| Option | Effort | Risk | Notes |
|--------|--------|------|-------|
| Increase to 80 s | 1 min | VERY LOW | Sufficient margin |
| Increase to 90 s | 1 min | VERY LOW | More margin; episodes take slightly longer |
| Dynamic timeout by town | ~10 lines | LOW | Overkill for this issue |

**Recommendation:** 80 s. If timeouts persist after F8 (deadlock fix), increase further.

---

## F3 — Write `gap_fill_diverged` and `gap_fill_rpcs` to HDF5

### What was observed

After the Town07 ep1 fence collision (documented in [ISSUE-01](ISSUE-01-junction-waypoints.md)), there was no way to check from HDF5 how often gap fill diverged across all episodes. The monitoring field exists in code but was never wired to the HDF5 output.

### Code locations

`FrameInfo` dataclass in `src/data/expert_collector.py`:
```python
@dataclass
class FrameInfo:
    ...
    gap_fill_diverged: bool = False   # ← exists, not written to HDF5
    gap_fill_rpcs: int = 0            # ← exists, not written to HDF5
    ...
```

The HDF5 dataset creation loop in `expert_collector.py` does not include these fields. They are computed per frame by the expert but silently discarded.

### Why it matters

Without `gap_fill_diverged` in HDF5, you cannot:
- Check post-hoc how often junction gap fill diverged in a 1,000+ episode overnight run
- Verify that the nexts[] heading-alignment fix (F10/F11) actually reduced divergence
- Identify which episodes and frames had divergences for targeted investigation

**This field is a prerequisite to verifying F10/F11 (ISSUE-01 Tier 2 fixes).** Write it first.

### Fix

In `src/data/expert_collector.py`, in the dataset creation loop, add:
```python
# gap fill diagnostics
f.create_dataset('gap_fill_diverged', data=np.array([fi.gap_fill_diverged for fi in frames], dtype=bool))
f.create_dataset('gap_fill_rpcs', data=np.array([fi.gap_fill_rpcs for fi in frames], dtype=np.int32))
```

### Verification

After adding:
```python
import h5py
with h5py.File('data/test_collection/episode_XXXX.h5', 'r') as f:
    print(list(f.keys()))  # should include 'gap_fill_diverged', 'gap_fill_rpcs'
    print(f['gap_fill_diverged'][:20])  # should be bool array
```

---

## F4 — Write `route_completion` to HDF5

### What was observed

The expert's `route_completion` property (lines 450–454 in `privileged_expert.py`) computes `_route_index / _route_length` as a float 0.0–1.0. This value is never written to HDF5. Without it:
- Cannot determine at what route stage the nexts[0] bug fires
- Cannot correlate episode timeouts with route progress
- Cannot filter training clips by route completion (e.g., skip the last 5% where buffer degenerates)

### Fix

In `src/data/expert_collector.py`, add a scalar dataset per episode (since route_completion is per-frame):
```python
f.create_dataset('route_completion',
    data=np.array([fi.route_completion for fi in frames], dtype=np.float32))
```

*If `FrameInfo` does not already have `route_completion` as a field, it must be added there first.*

### Verification

```python
with h5py.File('data/test_collection/episode_XXXX.h5', 'r') as f:
    rc = f['route_completion'][:]
    print(f"Route completion: {rc[0]:.3f} → {rc[-1]:.3f}")  # should go from ~0 to ~1
```

---

## F5 — Fix Manifest `nav_distribution` (Include Classes 5+6)

### What was observed

`collection_manifest.json` has a `nav_distribution` field for each episode. This field is the primary artifact users check to verify nav command balance. It only includes nav values 1–4. Classes 5 (CHANGELANELEFT) and 6 (CHANGELANERIGHT) are absent, even in episodes where HDF5 data clearly shows they are present.

### Verification

For episode `b5367077`:
- HDF5 nav histogram: `{1: 312, 2: 187, 3: 891, 4: 1423, 5: 891, 6: 0}`
- Manifest `nav_distribution`: `{1: 312, 2: 187, 3: 891, 4: 1423}`

Classes 5 and 6 are silently dropped, making the manifest unreliable for monitoring lane-change data balance.

### Fix

In the manifest generation code, change the nav distribution loop from filtering `if nav_val in {1, 2, 3, 4}` to include all valid nav values (or iterate over actual unique values):

```python
# Before (wrong):
nav_dist = {k: int(v) for k, v in zip(*np.unique(nav, return_counts=True))
            if k in {1, 2, 3, 4}}

# After (correct):
nav_dist = {int(k): int(v) for k, v in zip(*np.unique(nav, return_counts=True))}
```

### Verification

After fix, re-run manifest generation on existing test episodes and check that nav=5 and nav=6 appear in `nav_distribution` where HDF5 data has those values.

---

## All Tier 0 Fixes in One Sitting

These five items can be done in a single sitting before any other work. Total estimated time: 45–60 minutes including verification.

| Fix | File | Change | Estimate |
|-----|------|--------|----------|
| F1 | `configs/collection.yaml` | `stop_sign_route_proximity: 2.5` → `4.0` | 2 min |
| F2 | `configs/collection.yaml` | `timeout_duration: 75` → `80` | 2 min |
| F3 | `src/data/expert_collector.py` | Write `gap_fill_diverged` + `gap_fill_rpcs` | 15 min |
| F4 | `src/data/expert_collector.py` | Write `route_completion` | 10 min |
| F5 | manifest writer | Include nav 5+6 in distribution | 10 min |
| — | Verify all changes | Run `bash scripts/test_collection.sh --town Town03` | 15–20 min |

---

## Open Questions for User

None. All five changes are unambiguous and zero-risk.
