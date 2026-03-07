# Memory Leak Investigation: CARLA Expert Agent OOM

## Executive Summary

Systematic investigation disproved the hypothesis that CARLA's C++ `BufferPool` leaks memory.
The actual root cause was an **unbounded `while` loop** in
`PrivilegedExpert._fill_junction_gaps()` that issued thousands of CARLA server RPCs per
frame at road junctions, leaking ~19 GB of **server-side** memory and triggering OOM kills.

Three successive fixes were required:

| Fix | Problem | Solution |
|-----|---------|----------|
| **Bounded iterations** (§4.1) | Unbounded `while nexts:` loop → infinite RPCs | Per-gap iteration cap + divergence guard + cumulative RPC budget |
| **Tentative fills** (§4.5) | Diverging fills committed to buffer → cascading corruption → 25 RPCs/frame → budget exhaustion at ~2000 frames | Accumulate fills in temp list; only commit if they converge on `wp_next` |
| **Pair-key cache** (§4.6) | Known-diverging gaps re-probed every frame (~10 RPCs each) | Cache `(wp_prev.id, wp_next.id)` in `_gap_fill_tried` set; skip on re-encounter |

After the final fix, the expert runs 5000+ ticks across multiple junction-heavy Town03 routes
with only 20–33 total RPCs per episode, zero corridor corruption, and MemAvailable stable.

---

## 1. Initial Hypothesis (Disproved)

**Claim:** libcarla's C++ `BufferPool` leaks at 40–100 MB/s because `Pop()` loses
`shared_ptr` references.

**Analysis of CARLA 0.9.16 C++ Source:**

| File | Finding |
|------|---------|
| `LibCarla/source/carla/Buffer.h` | `ReuseThisBuffer()` stores a `weak_ptr` (not `shared_ptr`) back to the pool — no circular reference possible |
| `LibCarla/source/carla/streaming/detail/tcp/Client.cpp` | Each `Buffer` is `Pop()`'d from the pool, filled, dispatched. If the pool is empty, a fresh `Buffer` is allocated — no leak |
| `LibCarla/source/carla/BufferPool.h` | Pool is a fixed-size `std::vector<shared_ptr<BufferPool>>` behind a `mutex`. `Push()` puts consumed buffers back. Lifecycle is sound |

**Verdict:** The C++ buffer pool is correctly implemented. The leak is elsewhere.

---

## 2. Systematic Test Results

All tests ran on **Town03** (the map where OOM was observed).

### 2.1 Isolation Tests (One Variable at a Time)

| Test | Description | Ticks | RSS Start→End | Verdict |
|------|-------------|-------|---------------|---------|
| T1 | `world.tick()` only | 5,000 | 97→97 MB | **CLEAN** |
| T2 | 1 RGB camera | 5,000 | 99→99 MB | **CLEAN** |
| T3 | 3 RGB cameras | 5,000 | 100→100 MB | **CLEAN** |
| T4 | 20 NPCs + Traffic Manager | 5,000 | 99→99 MB | **CLEAN** |
| T5 | `get_actors()` ×4/frame | 5,000 | 94→94 MB | **CLEAN** |
| T6 | `get_waypoint()` ×5/frame | 5,000 | 94→94 MB | **CLEAN** |
| T7 | 3 cams + 20 NPCs + queries | 5,000 | 105→105 MB | **CLEAN** |

**Conclusion:** CARLA's core systems (rendering, sensors, traffic manager, map queries)
do not leak.

### 2.2 Expert Agent Tests

| Test | Description | Ticks | RSS | Verdict |
|------|-------------|-------|-----|---------|
| T8 | Full `PrivilegedExpert` | — | — | **OOM KILLED** |
| T8a | Expert init only (no `run_step`) | 3,000 | 133→133 MB | **CLEAN** |
| T8b | Traffic-light scan (`get_affected_lane_waypoints`) | 5,000 | 95→95 MB | **CLEAN** |

**Conclusion:** The leak is in `PrivilegedExpert.run_step()`, not in initialization
or traffic-light queries.

### 2.3 Binary Isolation (Narrowing the Leak)

| Test | Description | Ticks | RSS | VmmemWSL | Verdict |
|------|-------------|-------|-----|----------|---------|
| T11 | Expert with per-tick monitoring | 40 | 138 MB (flat) | 19 GB | **OOM KILLED** |
| T12 | Driving + all queries, no expert object | 5,000 | 98→102 MB | stable | **CLEAN** |
| T13 | Expert, detection disabled, vehicle driving | — | — | 19 GB | **OOM KILLED** |
| T14 | Expert, full detection, NO `apply_control` (vehicle stationary) | 5,000 | 139→139 MB | stable | **CLEAN** |
| T15 | Pure driving, no expert | 5,000 | 97→97 MB | stable | **CLEAN** |
| T16 | Expert + driving, `_fill_junction_gaps` **disabled** | 5,000 | 137→139 MB | stable | **CLEAN** |

### 2.4 Root Cause Confirmed

The leak occurs **if and only if** all three conditions are met:
1. `PrivilegedExpert` is active
2. The vehicle is **moving** (triggering `_advance_route()` → `_fill_buffer()` → `_fill_junction_gaps()`)
3. `_fill_junction_gaps()` is **enabled**

Disabling any one of these three conditions eliminates the OOM.

---

## 3. Root Cause Analysis

### 3.1 What `_fill_junction_gaps()` Does

The expert agent maintains a waypoint buffer for its planned corridor.
`GlobalRoutePlanner` sometimes produces waypoints that are >4 m apart at junctions
(normal spacing is ~2 m). `_fill_junction_gaps()` interpolates additional waypoints
to fill these gaps by calling `wp.next(2.0)` repeatedly.

### 3.2 The Bug

```python
# BEFORE (simplified) — the unbounded version
while nexts:
    nxt = nexts[0]
    if nxt.transform.location.distance(wp_next.transform.location) < _FILL_STEP:
        break
    rebuilt.append((nxt, cmd_next))
    nexts = nxt.next(_FILL_STEP)  # ← CARLA server RPC
```

At junctions, `wp_prev` and `wp_next` can sit on **different roads**.
`wp_prev.next(2.0)` follows `wp_prev`'s road forward — it never converges on
`wp_next` because `wp_next` is on a diverging road.

**Result:** The `while nexts:` loop runs indefinitely. Each `nxt.next(_FILL_STEP)`
is a **CARLA server RPC** that allocates server-side waypoint serialization data.
With no iteration limit:

- Thousands of RPCs per gap, per frame
- The CARLA server (running on Windows) accumulates ~19 GB of waypoint data
- VmmemWSL (the Windows process for the WSL2 VM) balloons → OOM kill

### 3.3 Why Python RSS Stayed Flat

The leak is **server-side**, not client-side. Python receives small waypoint
response objects (~200 bytes each) that are promptly garbage-collected.
The CARLA server keeps expanding its internal waypoint cache.
This is why `tracemalloc` and Python RSS monitoring never showed the leak.

### 3.4 Why It Only Triggers When the Vehicle Moves

When the vehicle is stationary (T14), `_advance_route()` never pops waypoints,
so `_fill_buffer()` has nothing new to process — `_fill_junction_gaps()` returns
immediately. As soon as the vehicle moves, it consumes waypoints, triggering
continuous gap-fills at every junction it passes through.

---

## 4. The Fix

### 4.1 Three-Level Bounding

```python
# AFTER — bounded version (agents/expert/privileged_expert.py)
max_fills = int(gap / _FILL_STEP * cfg.gap_fill_max_factor) + 1
diverge_limit = gap * cfg.gap_fill_diverge_factor

while nexts and fills < max_fills:
    nxt = nexts[0]
    if dist_to_target < _FILL_STEP:
        break
    if dist_from_origin > diverge_limit:
        break  # divergence guard
    rebuilt.append((nxt, cmd_next))
    nexts = nxt.next(_FILL_STEP)
    fills += 1
```

| Guard | Config Parameter | Default | Purpose |
|-------|-----------------|---------|---------|
| Per-gap iteration cap | `ExpertConfig.gap_fill_max_factor` | 3.0 | Max iterations = `gap / fill_step × factor`. A 10 m gap with 2 m steps allows 15 iterations |
| Distance-from-origin | `ExpertConfig.gap_fill_diverge_factor` | 2.0 | Break if traveled > `gap × factor` from the gap's start |
| Cumulative RPC budget | `ExpertConfig.gap_fill_rpc_budget` | 50,000 | Total `wp.next()` calls per episode. Disables gap filling when reached |

### 4.2 Monitoring & Graceful Degradation

- **`self._gap_fill_rpcs`**: Cumulative counter, exposed in `FrameInfo.gap_fill_rpcs`
  so the data collector can monitor it per frame.
- **`self._gap_fill_diverged`**: Counter for gaps where the fill diverged from the
  target waypoint, exposed in `FrameInfo.gap_fill_diverged`.
- **Budget exhaustion**: When the RPC budget is reached, a `logger.warning` is emitted
  and gap filling is disabled for the remainder of the episode — the expert continues
  driving normally, just without junction corridor interpolation.
- **`set_route()` reset**: All gap-fill state (`_gap_fill_rpcs`, `_gap_fill_diverged`,
  `_gap_fill_budget_exhausted`, `_gap_fill_tried`) is reset when a new route is set,
  so each episode starts fresh.

### 4.3 Configurable Parameters

```python
from agents.expert.privileged_expert import ExpertConfig

# Default (recommended) — bounded and safe
config = ExpertConfig()

# More generous gap filling for systems with >32 GB RAM
config = ExpertConfig(
    gap_fill_max_factor=5.0,
    gap_fill_diverge_factor=3.0,
    gap_fill_rpc_budget=100_000,
)

# Disable gap filling entirely (fastest, least corridor quality)
config = ExpertConfig(gap_fill_max_factor=0)

# Unbounded (NOT RECOMMENDED — will OOM on Town03 junctions)
config = ExpertConfig(
    gap_fill_max_factor=float('inf'),
    gap_fill_rpc_budget=0,
)
```

### 4.4 Using `FrameInfo` Gap-Fill Fields in the Collector

```python
info: FrameInfo = expert.run_step(...)

# Monitor RPC usage — gracefully end episode before OOM
if info.gap_fill_rpcs > expert.config.gap_fill_rpc_budget * 0.8:
    logger.warning("Approaching RPC budget — consider ending episode")

# Monitor divergence — high counts may indicate route issues
if info.gap_fill_diverged > 20:
    logger.warning("Many diverging junctions — corridor may have gaps")

# Log for post-hoc analysis
writer.add_scalar("gap_fill_rpcs", info.gap_fill_rpcs, step)
writer.add_scalar("gap_fill_diverged", info.gap_fill_diverged, step)
```

### 4.5 Bug 2: Cascading Corridor Corruption (Fixed 2026-03-04)

The bounded-iteration fix (§4.1) was necessary but not sufficient. It introduced
a **cascading corruption** bug:

**Problem:** When a gap between `wp_prev` and `wp_next` sat on different roads
(diverging junction), the fill would follow `wp_prev`'s road forward. The
iteration cap or divergence guard would eventually fire, but by then,
**diverging waypoints had already been inserted into `_wp_buffer`**. These
misplaced waypoints created *new* gaps to the *next* valid waypoint, triggering
re-fills on subsequent frames — a cascade:

```
Frame N:   gap(A→B) → fill diverges → inserts [X, Y, Z] (wrong road)
Frame N+1: gap(Z→B) → fill diverges again → inserts [X', Y', Z'] (still wrong)
Frame N+2: gap(Z'→B) → ...etc → ~25 RPCs/frame × 2000 frames = 50,000 RPCs
```

**Impact:**
- RPC budget exhausted at ~2000 frames instead of lasting the full episode
- SAT corridor (built from `_wp_buffer`) followed the wrong road at junctions,
  causing the corridor to "stop following the car" → missed detections → collisions
- Memory: 50K RPCs ≈ 50K `wp.next()` server calls ≈ server-side waypoint cache growth

**Fix — Tentative Fill Accumulation:**

```python
# AFTER — tentative fill (agents/expert/privileged_expert.py)
tentative = []               # <── don't touch _wp_buffer yet
while nexts and fills < max_fills:
    nxt = nexts[0]
    dist_to_target = nxt.transform.location.distance(wp_next.transform.location)
    if dist_to_target < _FILL_STEP:
        break                # converged — will commit
    dist_from_origin = nxt.transform.location.distance(origin)
    if dist_from_origin > diverge_limit:
        break                # diverging — will NOT commit
    tentative.append((nxt, cmd_next))
    nexts = nxt.next(_FILL_STEP)
    fills += 1

# Only commit fills that actually converged on wp_next
converged = (fills > 0 and dist_to_target < _FILL_STEP)
if converged:
    rebuilt.extend(tentative)  # safe to insert
else:
    self._gap_fill_diverged += 1  # count for diagnostics
```

The key insight: **never modify `_wp_buffer` with waypoints that don't converge.**
Diverging fills are accumulated in `tentative[]` and simply discarded when the
divergence guard fires. The gap remains as-is (one longer segment), which the
lateral controller handles without issues.

### 4.6 Bug 3: Re-probe Waste (Fixed 2026-03-04)

Even with tentative fills, each known-diverging gap was re-probed on every
subsequent frame (the gap persists since we didn't fill it). Each probe costs
~10 RPCs before the divergence guard fires.

**Fix — Pair-Key Cache:**

```python
# In __init__:
self._gap_fill_tried: set[tuple[int, int]] = set()

# In _fill_junction_gaps(), before probing:
pair_key = (wp_prev.id, wp_next.id)
if pair_key in self._gap_fill_tried:
    rebuilt.append((wp_next, cmd_next))  # skip fill, keep gap as-is
    continue
self._gap_fill_tried.add(pair_key)

# If fill diverges, pair_key stays in the set → never re-probed
# If fill converges, pair_key stays but doesn't matter (gap is filled)
```

**Impact:** After the first pass through a junction, all diverging gaps are
cached. Subsequent frames see 0 RPCs for those gaps.

### 4.7 Complete Fix Chain Summary

| Fix | When | Effect on RPCs/episode | Effect on corridor |
|-----|------|----------------------|-------------------|
| Original (unbounded) | — | ∞ (OOM) | Infinite waypoints on wrong road |
| §4.1 Bounded iterations | 2026-03-04 | 50,000 (budget cap) | Corrupted — diverging fills inserted |
| §4.5 Tentative fills | 2026-03-04 | ~500–1000 (re-probes every frame) | Clean — diverging fills discarded |
| §4.6 Pair-key cache | 2026-03-04 | 20–33 per episode | Clean — known gaps skipped |

---

## 5. Impact on Driving Quality

### What Changes for Longer Episodes

The final fix (tentative fills + pair-key cache) **does not degrade driving quality**.
For **converging gaps** (normal junctions), fills work exactly as before.
For **diverging junctions** (the bug trigger), the gap is left as-is — one
longer segment (~4–13 m instead of 2 m spacing). The expert's lateral controller
handles these gaps without issue.

**The first fix (bounded iterations only) DID cause problems:** diverging fills
were committed to the buffer, corrupting the SAT corridor. This manifested as the
corridor "stopping following the car" before collisions. The tentative-fill approach
(§4.5) eliminated this corruption entirely.

### Memory Footprint

| Scenario | Server RPCs | Server Memory | Python RSS |
|----------|-------------|---------------|------------|
| Before fix (Town03 junction) | ∞ | ~19 GB → OOM | 138 MB (stable) |
| Fix §4.1 only (bounded) | 50,000/ep (budget cap) | ~50 MB growth | 139 MB |
| Fix §4.1+§4.5+§4.6 (final) | 20–33/ep | negligible | 139 MB |

---

## 6. Verification Results (2026-03-04)

### 6.1 `diagnose_gap_fill.py` Results

Three targeted tests on junction-heavy Town03 routes (spawn 20→100: 618 wps, 128 turns).

| Test | Description | Ticks | Gap-Fill RPCs | Diverged | MemAvail Start→End | Verdict |
|------|-------------|-------|--------------|----------|-------------------|---------|
| T1 `gap_rpc` | Single episode RPC tracking | 5,000 | 20 | 2 | 17,516→17,535 MB | **CLEAN** |
| T2 `multi_ep` | 3 sequential episodes | 3×2,500 | 33 total | 5 | 17,538→17,528 MB (10 MB drain) | **CLEAN** |
| T3 `corridor` | Buffer integrity check | 5,000 | — | — | — | **CLEAN** (0 corruption frames, max gap 13 m) |

**T1 details:** 20 RPCs total across 5000 ticks. 2 gaps diverged (discarded via tentative fill).
Budget usage: 20/50,000 = 0.04%. MemAvailable actually *increased* by 19 MB (noise).

**T2 details:** 33 RPCs across 3 episodes (11 per episode average). `set_route()` reset
correctly between episodes. 10 MB MemAvailable drain across all 3 episodes — negligible.

**T3 details:** Corridor integrity verified every frame. Zero corruption (no waypoints
on wrong road). Maximum gap between consecutive waypoints: 13 m at diverging junctions.
Lateral controller handled all gaps cleanly.

### 6.2 Comparison with Previous Results

| Test | Before Fix | After §4.1 | After Final Fix |
|------|-----------|------------|----------------|
| T8 (5000 ticks) | OOM KILLED | Clean (RSS stable) | Clean (20 RPCs total) |
| test_collection.sh (Town03 eps) | OOM KILLED | Budget exhausted at ~2000 frames | Not yet re-run |
| Corridor integrity | N/A (OOM) | Corrupted (diverging fills) | Clean (0 corruption) |

---

## 7. Test Environment

| Component | Value |
|-----------|-------|
| CARLA | 0.9.16 |
| Python | 3.11 |
| OS | WSL2 on Windows (AlienX) |
| WSL2 Memory | 20 GB |
| WSL2 Swap | 16 GB |
| Physical RAM | 32 GB |
| CARLA Server | Windows-side at 172.20.0.1:2000 |
| Conda Env | `rwkv` |
| Test Map | Town03 |
| Diagnostic Scripts | `scripts/diagnose_tick_leak.py` (isolation tests T1–T16), `scripts/diagnose_gap_fill.py` (gap-fill validation T1–T3) |

---

## 8. Key Takeaway

**The leak was not in CARLA's C++ buffer pool.** It was an application-level bug
in our `PrivilegedExpert` agent: an unbounded loop that hammered the CARLA server
with RPC calls at road junctions, causing the server (not the client) to accumulate
~19 GB of waypoint data.

The fix required three rounds:
1. **Bounded iterations** stopped the infinite loop but still inserted diverging
   waypoints into the buffer, causing cascading re-fills (~25 RPCs/frame).
2. **Tentative fills** prevented corridor corruption by only committing fills that
   converge on the target waypoint.
3. **Pair-key caching** eliminated redundant re-probing of known-diverging gaps.

The final fix reduces gap-fill RPCs from ∞ (OOM) to 20–33 per episode, with zero
corridor corruption. Verified clean across multiple junction-heavy Town03 routes.

## 9. VmmemWSL Page Cache Investigation (2026-03-05)

After fixing the gap-fill OOM, VmmemWSL still climbed to ~19 GB during data
collection. Dual-monitor investigation (PowerShell on Windows + bash on WSL2)
identified **Linux page cache from HDF5 writes** as the sole cause — not a leak.

- **97.8%** of VmmemWSL growth is page cache from `_StreamingEpisodeWriter` HDF5 writes
- Cache is **reclaimable** — `MemAvailable` stayed at ~16 GB even with `MemFree` at 191 MB
- The kernel handles memory pressure correctly; collection completed without OOM
- `posix_fadvise(POSIX_FADV_DONTNEED)` eliminates 100% of cache bloat (+2 MB vs +16 GB)
- The fix is a **Windows usability improvement**, not an OOM prevention

See [MEMORY_DIAGNOSTICS_FINDINGS.md §15](MEMORY_DIAGNOSTICS_FINDINGS.md) for the
full dual-monitor correlation analysis, write rate measurements, and data tables.
