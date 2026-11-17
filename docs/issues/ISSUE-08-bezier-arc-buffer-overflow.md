# ISSUE-08 — Bezier Arc Buffer Overflow (Waypoint Loss After Junction)

> **Status:** Fixed
> **Priority:** Tier 2
> **Implementation plan tasks:** Part of F14 (Bezier arc implementation)
> **Connected issues:** [ISSUE-01](ISSUE-01-junction-waypoints.md)

---

## Summary

When `_fill_junction_gaps` inserted Bezier arc waypoints into `_wp_buffer`, the `rebuilt` list overflowed beyond `maxlen`. The truncation `rebuilt[:maxlen]` kept the near-ego approach waypoints but silently dropped post-junction route waypoints from the back — permanently, because `_route_index` had already advanced past them. After the junction, `_fill_buffer` loaded from `_route_index`, which now pointed 30–60 waypoints beyond the exit. The next available waypoint was 60–100 m away, causing:

- SAT corridor collapsing to a single 40 m block (only one distant waypoint)
- Ego steering toward a point 100 m away in the wrong lane
- Subsequent wrong-direction driving to reach missed waypoints

---

## Failed Intermediate Fix (Rewind Approach)

An intermediate fix rewound `_route_index` by the count of dropped original items. This had three failure modes:

1. **Timing mismatch**: `items[j]` (junction exit) reloaded after ego had already physically passed it. `_advance_route` immediately popped it. Ego then steered to `items[j+1]` in the wrong direction.
2. **Cascading secondary fills**: The second-stage gap fill started from a mid-arc projected waypoint whose yaw was the map road yaw, not the arc tangent. The secondary arc deviated from the intended path.
3. **`_gap_fill_tried` stale pair**: The pair `(items[j-1].id, items[j].id)` was added to `_gap_fill_tried` during the first fill. After the rewind, when `items[j]` reloaded, the pair was already tried — no arc generated — ego went straight into a fence (Town10HD ep3).

---

## Root Cause (Deeper)

The fundamental problem was that `_wp_buffer` served two roles simultaneously:
1. **Route tracking** — which waypoints ego has passed, what's next
2. **Arc interpolation** — dense curve samples for smooth junction navigation

These two roles conflict when the arc overflows the buffer's capacity.

---

## Fix — Dual Buffer Architecture

Separated the two roles:

- **`_wp_buffer`**: pure route tracking, never modified by Bezier arc fills. Junction exit (`wp_next`) always stays in `_wp_buffer`. No overflow possible from arc generation.
- **`_arc_buffer`**: dedicated `deque` for the current Bezier arc. Populated by `_fill_junction_gaps`. Consumed by `_advance_route` during junction traversal. Cleared when exhausted.

Behavior:
- `_fill_junction_gaps`: when Bezier fills are generated, store in `_arc_buffer` instead of `rebuilt`. Fall through so junction exit is added to `rebuilt` (and thus `_wp_buffer`) via the normal bottom-of-loop append.
- `_advance_route`: if `_arc_buffer` non-empty → pop from arc; when empty → clear it, fall back to `_wp_buffer` (which still has junction exit as front).
- `_get_target`: if `_arc_buffer` non-empty → scan arc for steering target; always scan `_wp_buffer` for nav command.
- No new arc is started while one is active (`if not self._arc_buffer:` guard in bezier section).

Also increased `waypoint_buffer_size` default from 40 to 60 (compensates for the larger buffer needed for longer routes and adds margin).

---

## What Was Observed

- **Town10HD ep1**: After first TL junction, Bezier arc led ego down a different path. Next waypoint was 100+ m away. Ego drove in wrong traffic lane (backwards on one-way road) trying to reach a missed waypoint.
- **Town10HD ep3**: No arc generated at final junction (pair already in `_gap_fill_tried` from aborted first fill). Ego went straight through junction into fence.
- **Town06 ep2, frame ~1100**: Arc overshot junction exit, junction exit waypoint popped prematurely, ego reversed direction, drove wrong way on one-way road.

---

## Status

Fixed. Tests:
- `test_fill_junction_gaps_bezier_goes_to_arc_buffer_not_wp_buffer`
- `test_fill_junction_gaps_does_not_start_second_arc_while_one_is_active`
- `test_fill_junction_gaps_prefers_bezier_for_junction_gap` (updated to check `_arc_buffer`)
