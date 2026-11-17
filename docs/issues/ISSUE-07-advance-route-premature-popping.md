# ISSUE-07 — `_advance_route` Premature Waypoint Popping

> **Status:** Fixed
> **Priority:** Tier 2
> **Implementation plan tasks:** (not yet in plan — discovered during Bezier arc testing)
> **Connected issues:** [ISSUE-01](ISSUE-01-junction-waypoints.md) (premature popping was the original Town07 fence failure, separate from nexts[0] bug)

---

## Summary

`_advance_route` used a heading-relative `is_behind` check to decide when to pop `_wp_buffer[0]`:

```python
is_behind = (fwd.x * dx + fwd.y * dy) < 0
if dist < min_follow_distance or is_behind:
    self._wp_buffer.popleft()
```

During a heading rotation (yield sign turn, junction exit, U-turn), the ego's forward vector rotates. Waypoints that were geometrically ahead now project as "behind" the new heading even though ego has not physically passed them. This causes a mass-pop event: all near-buffer waypoints get consumed in one tick, leaving only a distant route waypoint as `buffer[0]`.

---

## Observed Failure

**Town07 ep1, frame 612 (originally attributed to `nexts[0]` but actually this bug):**
- HDF5 forensics: `wphead` teleports from 3.2 m → 54.9 m in one frame
- `nav` jumps STRAIGHT → LANEFOLLOW in the same frame (not LEFT → STRAIGHT, ruling out nexts[0])
- No single-frame yaw change > 8° — heading rotation was gradual, over the prior ~20 frames
- `route[113]` at (−126.8, −36.4) was 54.9 m from ego at frame 633, confirmed behind ego by dot product = −35.6

This confirmed the pattern: gradual heading rotation → accumulated `is_behind` true → mass pop → distant waypoint.

---

## Fix

Replaced `is_behind` with a **segment progress** check:

```python
# Pop wp[0] when ego is past the midpoint of the wp[0] → wp[1] segment.
seg_dx = next_loc.x - curr_loc.x
seg_dy = next_loc.y - curr_loc.y
progress = (seg_dx * ego_dx + seg_dy * ego_dy) / seg_len
past_midpoint = progress > (0.5 * seg_len)
if dist < min_follow_distance or past_midpoint:
    self._wp_buffer.popleft()
```

This is heading-independent. A waypoint is only consumed when ego has physically crossed the midpoint toward the next waypoint, regardless of which way ego is facing.

**Committed:** `60b432d` (approximately — see git log for exact hash)

---

## Status

Fixed. No regression. Test: `test_advance_route_uses_segment_progress_even_when_heading_would_disagree`.
