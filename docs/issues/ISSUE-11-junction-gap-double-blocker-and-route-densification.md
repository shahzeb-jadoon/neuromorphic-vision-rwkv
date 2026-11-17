# ISSUE-11 — Junction Gap Double-Blocker and Route Densification Plan

> Status: Open (analysis-complete)
> Priority: Tier 1
> Branch baseline: `feature/issue-01-bezier-arc` @ commit `82ac87b`
> Related: ISSUE-01, ISSUE-08

---

## Why this issue exists

Town06 episode `d7a26d97` revealed two independent blockers that can prevent Bezier generation for the same gap:

1. **Tried-set poisoning**: large far-ahead gaps are marked as tried after RPC divergence.
2. **Strict `j == 1` timing gate**: the critical pair can remain at `j == 2` and be popped before ever reaching `j == 1`.

This means fixing only one blocker is insufficient in some episodes.

---

## Verified incident chain (current state)

- Gap `[195 -> 196]` is 24.06 m.
- Divergence increments at frames `695/698/704/711`.
- For `[195 -> 196]`, `ridx=255` never appears; route index jumps `254 -> 256`.
- At frame 974, duplicated route points (`194` and `195` have identical XY) trigger a double-pop in one cycle.
- Result: pair `[195 -> 196]` is removed before strict `j == 1` can be evaluated.

---

## Short-term fix proposal

**Option 6 (combined):**

1. Split tried state by method (`rpc_tried_pairs` vs `bezier_tried_pairs`).
2. Keep `j == 1` safe path and allow `j == 2` only with a proximity gate (distance from ego to `items[j-1]` under threshold).

Rationale:
- Prevents poisoning from blocking near-front Bezier.
- Avoids the old ISSUE-08 failure mode (far-ahead arc freeze) by requiring proximity for `j == 2`.

---

## Long-term robust solution

Move from per-tick gap repair to **route-time densification**:

1. Preprocess the route once in `set_route`.
2. For any large transition gap, insert intermediate waypoints using map-aware interpolation.
3. Keep runtime `_fill_junction_gaps` only as a safety net.

Expected benefits:
- Removes dependence on frame timing (`j` race conditions).
- Reduces repeated CARLA RPC pressure.
- Improves determinism and replayability.

---

## OOM safety notes

- Do **not** clear tried sets every frame (high repeated RPC cost).
- Keep bounded guards (`max_factor`, `diverge_factor`, `rpc_budget`) active.
- Route-time densification is one-time per episode and materially safer than repeated per-tick gap recovery.

---

## Acceptance criteria

1. No `ridx`-skip incident should block junction smoothing when ego is near the gap.
2. No `wp_buffer_head_dist` runaway growth from premature far-ahead arc activation.
3. No increase in gap-fill RPC budget exhaustion frequency.
4. Route sidecar should reflect densified route geometry used in collection.
