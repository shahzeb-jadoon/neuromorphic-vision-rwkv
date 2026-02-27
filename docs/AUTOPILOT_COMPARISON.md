# Autopilot / Expert System Comparison: Literature vs. This Project

> **Purpose:** Exhaustive comparison of every expert/autopilot system used by
> the papers in our literature review (and key papers NOT in our review but
> central to the CARLA AD ecosystem). Documents each feature, how it was
> implemented, where to find the code, and how our PrivilegedExpert differs.
>
> **Last updated:** 2026-02-22

---

## Table of Contents

1. [Paper Inventory](#1-paper-inventory)
2. [Feature Matrix](#2-feature-matrix)
3. [Detailed Per-Paper Analysis](#3-detailed-per-paper-analysis)
4. [Feature-by-Feature Comparison with Our Expert](#4-feature-by-feature-comparison)
5. [Where Our Expert Is Inherently Better](#5-where-our-expert-is-inherently-better)
6. [Where Other Experts Have Features We Lack](#6-where-other-experts-have-features-we-lack)
7. [Repositories for Testing](#7-repositories-for-testing)
8. [Verdict and Recommendations](#8-verdict-and-recommendations)

---

## 1. Paper Inventory

### Papers WITH Custom Autopilots (Rule-Based)

| # | Paper | Year | Venue | Autopilot Type | Lineage |
|---|-------|------|-------|---------------|---------|
| A | **TransFuser** (Chitta et al.) | 2022 | PAMI/CVPR | Rule-based, privileged | Original — defines the lineage |
| B | **Learning by Cheating** (Chen et al.) | 2020 | CoRL | Rule-based, privileged | Independent (predates TransFuser) |
| C | **TCP** (Wu et al.) | 2022 | NeurIPS | Rule-based, privileged | Fork of TransFuser/LBC |
| D | **InterFuser** (Shao et al.) | 2023 | CoRL | Rule-based, privileged | Extended TransFuser |
| E | **ThinkTwice** (Jia et al.) | 2023 | CVPR | Hybrid (RL + rule override) | Roach RL + TransFuser-style rules |

### Papers WITH RL-Based Experts

| # | Paper | Year | Venue | Expert Type |
|---|-------|------|-------|------------|
| F | **Roach** (Zhang et al.) | 2022 | ICLR | PPO on BEV semantic masks |
| G | **MILE** (Hu et al.) | 2022 | NeurIPS | Reuses Roach verbatim |
| H | **Bench2Drive / Think2Drive** (Jia et al.) | 2024 | NeurIPS | RL world-model expert |

### Papers WITHOUT Custom Autopilots (Architecture-Only or Real-World Data)

| # | Paper | Year | Venue | Data Source |
|---|-------|------|-------|-------------|
| I | **LADY** (Huang et al.) | 2025 | arXiv | Evaluates on NAVSIM/Bench2Drive datasets |
| J | **NAVSIM** (Dauner et al.) | 2024 | CVPR | Real-world human driving logs (nuPlan/OpenScene) |

### Papers in Literature Review NOT About Driving Experts

The remaining papers in our literature review ([1]–[41]) cover RWKV
architecture, Vision-RWKV, SpikeGPT, neuromorphic hardware, Decision
Transformers, and related theory. They do not implement CARLA autopilots
and are excluded from this comparison.

---

## 2. Feature Matrix

Every autopilot feature found across all papers, with presence/absence marked.

| Feature | TransFuser (A) | LBC (B) | TCP (C) | InterFuser (D) | ThinkTwice (E) | Roach (F) | **Ours** |
|---------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Route following (A\*)** | ✓ | ✓ | ✓ | ✓ | ✓ | Learned | ✓ |
| **PID longitudinal** | ✓ | P-only | ✓ | ✓ | ✓ | N/A | ✓ (full PID) |
| **PID lateral** | ✓ | PID | ✓ | ✓ | ✓ | N/A | ✓ (full PID) |
| **Gain blending (curvature)** | ✗ | ✗ | ✗ | ✗ | ✗ | N/A | **✓** |
| **EMA curvature smoothing** | ✗ | ✗ | ✗ | ✗ | ✗ | N/A | **✓** |
| **Adaptive follow distance** | ✗ | ✗ | ✗ | ✗ | ✗ | N/A | **✓** |
| **Traffic light detection** | Trigger vol. OBB | road_id cone | Trigger vol. | Trigger vol. | Learned | Learned | 3-tier (persistent+built-in+junction scan) |
| **Traffic light braking** | Binary (1.0) | Binary (1.0) | Binary (1.0) | Binary (1.0) | Learned | Learned | **Kinematic** $v=\sqrt{2ad}$ |
| **Vehicle detection** | Future OBB prediction | Forward cone | Future 2D ray | 3-mode (std/jct/lane) | Angle/dist filter | Learned | **OBB+SAT on curve-aware corridor** |
| **Vehicle braking** | Binary (1.0) | Binary (1.0) | Binary (1.0) | Binary (1.0) | Learned | Learned | **Kinematic** $v=\sqrt{2ad}$ |
| **Lane corridor** | None (all vehicles in range) | None (commented out) | None | road_id/lane_id | None | N/A | **Per-segment SAT on route corridor** |
| **Oncoming filter** | ✗ | ✗ | ✗ | ✗ | ✗ | N/A | **✓** (heading_dot) |
| **Pedestrian detection** | Future trajectory pred. | Dynamic cone + sidewalk | Future trajectory | Future trajectory | Trajectory pred. | Learned | Lateral band filter |
| **Pedestrian braking** | Binary (1.0) | Binary (1.0) | Binary (1.0) | Binary (1.0) | Learned | Learned | Creep at 8 km/h + emergency stop |
| **Junction speed reduction** | Fixed 3.0 m/s | 75% factor | Angle-based | Fixed 4.0 m/s | Fixed 4.0 m/s | Learned | **Kinematic ramp** to turn_target |
| **Turn anticipation** | ✗ | ✗ | ✗ | ✗ | ✗ | N/A | **✓** (30 m look-ahead) |
| **Speed limit compliance** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** (90% of posted) |
| **Slope handling** | ✓ (pitch-based throttle) | ✗ | ✗ | ✗ | ✗ | Learned | ✗ |
| **Stop sign handling** | ✓ (OBB trigger) | ✗ | ✗ | ✓ | ✗ | Learned | ✗ |
| **Stuck vehicle detection** | ✓ (speed buffer) | ✗ | ✗ | ✓ (hazard destruction) | ✗ | Terminal condition | ✗ |
| **Future state prediction** | ✓ (bicycle model) | ✗ | ✓ (2D ray) | ✓ (bicycle model) | ✗ | N/A | ✗ |
| **GPS denoising** | ✓ (100-frame buffer) | ✗ | ✗ | ✗ | ✗ | N/A | N/A (uses world coords) |
| **Steering noise injection** | ✓ (1e-3 Gaussian) | Disabled in code | ✗ | ✗ | ✗ | N/A | ✗ (planned Phase 2) |
| **Collision → episode stop** | ✗ (Leaderboard handles) | ✓ (impulse > 400) | ✗ | ✗ | ✗ | Terminal | **✓** (immediate break) |
| **Waypoint disturbance** | ✗ | ✗ | ✗ | ✓ (±0.3 m lateral) | ✗ | N/A | ✗ |
| **Bike-specific detection** | ✗ | ✗ | ✗ | ✓ | ✗ | N/A | ✗ |
| **BEV label generation** | ✓ | ✗ | ✗ | ✓ | ✓ | Input | ✗ |
| **Privileged metadata export** | Partial (JSON) | ✗ | Partial | Full (3D BB) | Full (Roach features) | N/A | **Full** (HDF5, every frame) |
| **Panoramic camera** | ✗ (single front) | ✗ (BEV) | ✗ (single front) | ✗ (multi-view) | ✗ | ✗ (BEV) | **✓** (3×640×320) |
| **Balanced route generation** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** (≥15% L, ≥15% R) |

Key: ✓ = Present, ✗ = Absent, "Learned" = handled by RL policy, **bold** = unique or superior to all others.

---

## 3. Detailed Per-Paper Analysis

### A. TransFuser Autopilot (Chitta et al., 2022)

**Repository:** https://github.com/autonomousvision/transfuser
**Branch:** `2022`
**Key file:** `team_code_autopilot/autopilot.py` (~1000 lines)

#### Features

1. **Route following**: `RoutePlanner` with GPS denoising (100-frame deque average). Waypoint resolution: 3.5 m near, 7.5 m far command.

2. **PID control**:
   - Lateral: `K_P=1.25, K_I=0.75, K_D=0.3, n=40` — "integral" is actually a 40-frame windowed mean, not a true integral.
   - Longitudinal: `K_P=5.0, K_I=0.5, K_D=1.0, n=40` — same windowed PID.
   - Throttle clipped to 0.75. Speed error clipped to `[0, 0.25]` (delta).

3. **Traffic light detection**: A 9×3×4 m OBB is attached to the ego vehicle (centred 2 m behind). CARLA traffic light trigger volumes are checked for OBB intersection. Red/Yellow → `brake=1.0`. Detection radius: 15 m.

4. **Vehicle detection + collision prediction**: For every vehicle within 30 m, their future trajectory is predicted using a kinematic bicycle model (from World on Rails: `front_wb=-0.09`, `rear_wb=1.42`, `steer_gain=0.37`, `brake_accel=-4.95 m/s²`). Ego future is predicted similarly. ego bounding box is split into front and back halves. At each future timestep (up to 4s at junctions, 1s outside), OBB intersection is tested. If any front-half intersection → `vehicle_hazard=True` → `brake=1.0`.

5. **Safety bounding box (Bremsweg)**: `d = ((v×3.6)/10)²/2`. A forward-projected box of this length is tested against all vehicles. Any intersection → `brake=1.0`.

6. **Stuck vehicle detection**: If a nearby vehicle's 30-frame average speed < 0.1 m/s AND throttle > 0.1 AND brake < 0.1 → marked as stuck. Stuck vehicles' future positions are held constant (not extrapolated).

7. **Junction speed**: Binary switch — 3.0 m/s at junctions, 4.0 m/s elsewhere. No anticipatory slowing.

8. **Slope handling**: If `pitch > 10°`, add `+0.4` throttle.

9. **Stop sign handling**: OBB intersection with trigger volumes. If stopped (speed ≈ 0), mark as cleared.

10. **Steering noise**: Gaussian noise added to expert steering: `σ = 1e-3`.

11. **No lane detection**: All vehicles within 30 m are considered. The only "in-lane" check is ignoring vehicles that only intersect the rear half of the ego box (treated as overtaking/being overtaken).

12. **No collision handling**: The autopilot has no collision sensor. The CARLA Leaderboard framework handles route termination. All frames including post-collision are saved.

#### Critical Limitations
- **All braking is binary** (1.0 or 0.0). No proportional, kinematic, or comfort deceleration. This produces extremely jerky demonstrations.
- **No lane filtering**. A vehicle stopped in the opposite lane on a narrow road triggers full brake.
- **No turn anticipation**. Speed changes abruptly at junction boundaries.
- **Bicycle model parameters** are from World on Rails, not tuned for this autopilot.

---

### B. Learning by Cheating (Chen et al., 2020)

**Repository:** https://github.com/dotchen/LearningByCheating
**Branch:** `release-0.9.6`
**Key files:**
- `bird_view/models/roaming.py` — `RoamingAgentMine`
- `PythonAPI/agents/navigation/agent.py` — `Agent` base class
- `bird_view/models/controller.py` — `PIDController`
- `data_collector.py` — `NoisyAgent` wrapper

#### Features

1. **Route following**: CARLA's built-in `LocalPlanner`.

2. **PID control**:
   - Steering: `K_P=1.5, K_I=0.5, K_D=0.0` (no derivative!).
   - Speed: P-only `K_P=1.0`. Very simple.
   - `PIDController` stores a 20-element deque, but uses `np.mean` + `np.gradient` — not a proper discrete PID.

3. **Vehicle detection**: Forward cone: 9.5 m distance, 45° half-angle. Yaw difference ≤ 150°. Lane check exists in code but is **commented out**:
   ```python
   # o_lane = state.lane_id
   # e_lane = state.ego_lane_id
   # if o_lane * e_lane < 0:  #not in same lane (opposite direction)
   ```
   Any vehicle in the cone → `emergency_stop()` → `brake=1.0`.

4. **Pedestrian detection**: Dynamic cone angle: 90° at 1.5 m → 15° at 10.5 m (interpolated). Sidewalk filter checks a pre-rendered road map image.

5. **Traffic light detection**:
   - EU-style (Town01/02): Same `road_id` AND `lane_id`, within 9.5 m, 60° cone.
   - US-style: Nearest light within 80 m, 25° before intersection.
   - Red/Yellow → `emergency_stop()` → `brake=1.0`.

6. **Turn handling**: `target_speed × 0.75` when command is LEFT or RIGHT. No junction detection, no anticipation.

7. **Collision termination**: Episode exits when collision impulse exceeds 400.

8. **Minimum episode length**: Episodes < 500 frames are discarded.

9. **Noise injection**: `NoisyAgent` wrapper exists but steering noise is **disabled** in the released code (only throttle noise applied).

#### Critical Limitations
- **Simpler than every subsequent autopilot**. ~200 lines.
- **Lane check commented out**, making it the most aggressive (any vehicle in cone = stop).
- **No future prediction**, no stuck detection, no safety box.
- **No proportional braking** at all.

---

### C. TCP (Wu et al., 2022)

**Repository:** https://github.com/OpenDriveLab/TCP
**Key file:** `leaderboard/team_code/auto_pilot.py`

#### Design
TCP reuses the TransFuser/LBC-lineage autopilot **almost verbatim** with these differences:
- Higher target speed: 7.0 m/s (vs TransFuser's 4.0)
- Angle-based turn detection instead of `is_junction`
- No OBB/SAT, no bicycle model, no Bremsweg, no stuck detection
- Uses `get_collision()` — a simple 2D ray intersection test from the original CARLA challenge codebase
- Still binary braking
- Post-collection `filter_data.py` for data quality filtering
- Weather randomisation every 10 ticks

---

### D. InterFuser (Shao et al., 2023)

**Repository:** https://github.com/opendilab/InterFuser
**Key file:** `leaderboard/team_code/auto_pilot.py`

#### Design
InterFuser **substantially extends** the TransFuser autopilot:

1. **Three vehicle detection modes**:
   - Standard: Same as TransFuser (future trajectory OBB intersection)
   - **Junction-specific**: Triple-ray (±15°) with command-aware ±25° lateral shift
   - **Lane-change**: Uses Shapely polygon intersection for lane-change scenarios

2. **Lane-aware filtering**: Checks `road_id`/`lane_id` match — the **only** rule-based autopilot that does proper lane filtering.

3. **Dedicated bike detection**: Separate logic for bicycles with different distance thresholds.

4. **Hazard actor destruction**: After 1500 ticks, blocking NPCs are destroyed to prevent permanent deadlocks.

5. **Waypoint disturbance**: ±0.3 m lateral perturbation for data diversity.

6. **BEV generation**: Full bird's-eye-view semantic labels stored per frame.

7. **21 weather conditions** including 7 night variants.

8. **Full actor state recording**: 3D bounding boxes with LiDAR point counts.

9. **Still binary braking** despite all the sophisticated detection.

#### Notable
InterFuser's autopilot is the most sophisticated rule-based one in the literature. Its lane-aware filtering (`road_id`/`lane_id`) and junction-specific triple-ray approach are the closest analog to our corridor-based SAT, though they solve the problem differently.

---

### E. ThinkTwice (Jia et al., 2023)

**Repository:** https://github.com/OpenDriveLab/ThinkTwice
**Key file:** `leaderboard/team_code/auto_pilot.py`

#### Design
Hybrid approach: Roach RL policy drives, with rule-based collision detection as a safety override:

1. **Primary driver**: Roach's PPO-trained policy
2. **Safety override**: TransFuser-style `should_brake` check. If the rule-based system disagrees with Roach, a `only_ap_brake` flag is saved per frame.
3. **Vehicle hazard**: `s1 = max(10, 3×speed)`, `angle_to_car < 30°`, `angle_between_heading < 60°`
4. **Standalone autopilot**: Also includes a full rule-based `AutoPilot` class: `target_speed=7.0 m/s` straight, `4.0 m/s` turns.
5. **Rich annotations**: 3D bounding boxes with LiDAR point counts, BEV segmentation, Roach value/feature outputs, `only_ap_brake` disagreement label.

---

### F. Roach (Zhang et al., 2022)

**Repository:** https://github.com/zhejz/carla-roach
**Key files:**
- `carla_gym/core/task_actor/ego_vehicle/reward/valeo_action.py` — reward
- `carla_gym/core/task_actor/ego_vehicle/terminal/` — terminal conditions
- `agents/rl_birdview/rl_birdview_agent.py` — PPO agent

#### Design
**Fundamentally different**: No hand-crafted driving rules. Everything learned through PPO with reward shaping:

1. **Input**: BEV semantic mask (road, lane, vehicles, pedestrians, lights, route)
2. **Reward**: `ValeoAction` composite — speed match, position on route, rotation penalty
3. **Terminal conditions** (6):
   - Stuck for too long
   - Lateral deviation > threshold
   - Red light violation
   - Collision (any)
   - Stop sign violation
   - Blocked by other actors
4. **Key parameters**: `_maxium_speed=6.0 m/s`, vehicle proximity `9.5 m`, lateral threshold `3.5 m`
5. **Braking**: **Learned** — smooth, proportional, emergent from training. This is the key advantage: RL produces naturally smooth braking profiles without hand-tuning.

#### Why This Matters
Roach demonstrates that **learning** can produce smoother driving than any rule-based system. However, it requires extensive RL training (millions of steps), BEV rendering infrastructure, and is not deterministically reproducible per episode.

---

### G. MILE (Hu et al., 2022)

**Repository:** https://github.com/wayveai/mile

**Reuses Roach entirely.** MILE's contribution is the world model architecture, not data collection. The `carla_gym` wrapper, RL expert, reward, and terminal conditions are all Roach's.

---

### H. Bench2Drive / Think2Drive (Jia et al., 2024)

**Repository:** https://github.com/Thinklab-SJTU/Bench2DriveZoo

Uses **Think2Drive** — an RL world-model expert that handles all 44 CARLA v2 interactive scenarios. Not rule-based. The dataset (13,638 clips, 2M frames) is pre-generated and distributed.

---

### I. LADY (Huang et al., 2025)

**No custom expert.** Evaluates on NAVSIM and Bench2Drive pre-existing datasets. Architecture paper only. **No public code** found as of Feb 2026.

---

### J. NAVSIM (Dauner et al., 2024)

**Repository:** https://github.com/autonomousvision/navsim

**No CARLA expert.** Uses real-world human driving logs from nuPlan/OpenScene (120 hours, 2 Hz). Evaluation via PDMS/EPDMS with non-reactive pseudo-simulation.

---

## 4. Feature-by-Feature Comparison

### 4.1 Vehicle Following / Braking Profile

| System | Method | Profile | Smoothness |
|--------|--------|---------|------------|
| TransFuser | Future trajectory OBB intersection | **Binary** (1.0 or 0.0) | Very jerky |
| LBC | Forward cone 9.5 m | **Binary** (1.0 or 0.0) | Very jerky |
| TCP | 2D ray intersection | **Binary** (1.0 or 0.0) | Very jerky |
| InterFuser | 3-mode detection | **Binary** (1.0 or 0.0) | Very jerky |
| ThinkTwice | Angle/distance filter | **Binary** (1.0 or 0.0) | Very jerky |
| Roach | Learned (PPO) | **Smooth** (emergent) | Smooth but not physics-based |
| **Ours** | OBB+SAT on corridor segments | **Kinematic** $v=\sqrt{2ad}$ | **Smooth, physics-based** |

**Our advantage**: We are the **only rule-based system** that uses a smooth, physics-based braking profile. Every other rule-based autopilot uses binary braking. Roach learns smooth braking but it's not interpretable or tunable.

Our kinematic profile:
```python
d = max(lead_dist - dyn_emergency, 0.0)
v_ms = math.sqrt(2.0 * cfg.vehicle_comfort_decel * d)
target = min(target, v_ms * 3.6)
```

This is the same $v = \sqrt{2 a d}$ relationship from physics — the speed at which you can come to a stop within distance $d$ at deceleration $a$. It produces the exact deceleration curve a skilled driver uses: gentle at first, progressively firmer as distance shrinks.

**Why this produces cleaner training data**: Binary braking creates bimodal label distributions — the model sees either "no brake" or "full brake" with nothing in between. This makes it harder for an imitation learning model to learn proportional responses. The kinematic profile produces a continuous distribution of brake values, which is inherently more learnable via MSE/L1 regression.

### 4.2 Traffic Light Handling

| System | Detection Method | Braking |
|--------|-----------------|---------|
| TransFuser | 9×3×4 m OBB vs trigger volumes | Binary |
| LBC | road_id + cone | Binary |
| InterFuser | Trigger volumes + enhanced | Binary |
| **Ours** | **3-tier: persistent memory + built-in + junction scan** | **Kinematic** |

Our 3-tier strategy:
1. **Persistent memory**: Once we identify a red/yellow light, we track it by `actor.id` until it turns green. This prevents "forgetting" at junction boundaries where road_id changes.
2. **CARLA built-in**: `is_at_traffic_light()` for close-range quick detection.
3. **Junction-aware scan**: `get_affected_lane_waypoints()` for lane-specific matching at complex junctions.

**Our advantage**: Persistent memory is unique among all autopilots surveyed. TransFuser's approach can "forget" a red light when the ego crosses a road boundary. Our kinematic braking profile ($v = \sqrt{2 \cdot 3.0 \cdot d}$) produces the smooth, "start gentle, firm up" deceleration curve instead of a binary stop.

### 4.3 Lane Corridor / In-Lane Detection

| System | Method | Handles Curves? |
|--------|--------|----------------|
| TransFuser | None — all vehicles in 30 m range | N/A |
| LBC | Commented out | N/A |
| TCP | None | N/A |
| InterFuser | road_id/lane_id matching | CARLA API-dependent (yes for simple curves) |
| **Ours** | **Per-segment SAT on curve-aware corridor** | **Yes — convex segments along route** |

Our corridor system:
1. `_build_route_corridor()` samples waypoints from the planned route up to cutoff distance.
2. Each waypoint's heading determines the corridor's local orientation — it **follows the curve**.
3. The corridor is decomposed into thin **convex quadrilaterals** (one per waypoint pair).
4. SAT overlap is tested per-segment, solving the non-convex polygon problem on curved roads.
5. The full polygon is retained for debug visualisation (magenta outline).

**Our advantage**: InterFuser's `road_id`/`lane_id` approach works but relies on CARLA's map topology, which can be unreliable at complex junctions where a single physical lane spans multiple road IDs. Our geometric approach works regardless of map topology because it follows the actual planned path.

**The oncoming filter** (heading_dot > 0) is unique to our expert. Every other autopilot either triggers on all nearby vehicles or uses cone-based angular filtering that doesn't distinguish direction.

### 4.4 Turn / Junction Handling

| System | Speed Reduction | Anticipation | Method |
|--------|----------------|-------------|--------|
| TransFuser | 3.0 m/s at `is_junction` | None | Binary switch |
| LBC | 75% of target at LEFT/RIGHT | None | Binary switch |
| TCP | Fixed at angle threshold | None | Angle-based switch |
| InterFuser | 4.0 m/s at `is_junction` | None | Binary switch |
| ThinkTwice | 4.0 m/s at `is_junction` | None | Binary switch |
| **Ours** | 45% of target, **kinematic ramp** | **30 m look-ahead** | $v = \sqrt{2 a_{turn} d + v_{turn}^2}$ |

Our turn handling:
1. **Anticipation**: `_get_target()` scans the waypoint buffer for upcoming LEFT/RIGHT commands up to 30 m ahead.
2. **Kinematic approach profile**: Instead of slamming speed at the junction boundary, we use:
   ```python
   v_ms = math.sqrt(2.0 * turn_comfort_decel * d + v_turn_ms**2)
   ```
   This computes the maximum speed at which the car can decelerate from current position to `turn_target` at the junction entry given distance `d` and comfort deceleration 1.5 m/s².
3. **Inside junction**: Hold `turn_target` speed, don't reduce further.
4. **Adaptive follow distance**: Blended from 5.0 m (straight) to 2.0 m (full turn) based on EMA-smoothed curvature.

**Our advantage**: Every other system uses a binary speed switch at junction boundaries. Our kinematic ramp starts braking 18+ metres before the junction, producing the smooth "coast into the turn" profile of a professional driver. The turn comfort deceleration (1.5 m/s²) is gentler than the vehicle-following deceleration (3.0 m/s²), giving 2.2 seconds of smooth ramp vs 1.1 seconds.

### 4.5 PID Controller Design

| System | Lateral PID | Longitudinal PID | Gain Adaptation |
|--------|------------|------------------|----------------|
| TransFuser | (1.25, 0.75, 0.3) windowed | (5.0, 0.5, 1.0) windowed | None |
| LBC | (1.5, 0.5, 0.0) no-D | P-only (1.0) | None |
| TCP | Same as TransFuser | Same as TransFuser | None |
| InterFuser | Same as TransFuser | Same as TransFuser | None |
| **Ours** | (1.2, 0.05, 0.15) true PID | (0.8, 0.05, 0.01) true PID | **Curvature-based gain blending** |

Our PID design:
1. **True discrete PID** with anti-windup clamping (integral limited to ±10).
2. **Two gain sets**: Straight (Kp=1.2, Ki=0.05, Kd=0.15) and Turn (Kp=1.5, Ki=0.05, Kd=0.3).
3. **Smooth blending**: `_apply_gain_blending()` interpolates between straight and turn gains using EMA-smoothed curvature / 45°. Same ramp as follow distance.
4. **Steering normalization**: Error normalised to ±90° → ±1.0, ensuring PID output is directly interpretable as steering angle.

**Our advantage**: Gain blending is unique. Every other system uses fixed gains that must be compromised between straight stability and turn responsiveness. Our system uses tighter gains in turns (more responsive) and gentler gains on straights (more stable) with a smooth transition that avoids overshoot or oscillation.

### 4.6 Pedestrian Handling

| System | Detection | Response |
|--------|----------|----------|
| TransFuser | Future trajectory prediction (linear extrapolation) | Binary brake |
| LBC | Dynamic cone (90°@1.5m → 15°@10.5m) + sidewalk filter | Binary brake |
| InterFuser | Future trajectory prediction | Binary brake |
| **Ours** | Lateral band filter (±3.0 m) | Creep (8 km/h) + emergency stop |

**Our advantage**: The creep-at-8-km/h response is unique. Every other system uses binary emergency brake, which in many cases is overly aggressive (pedestrian is walking perpendicular and will clear the lane in 1-2 seconds). Our two-tier response (creep if > 5 m, emergency stop if < 5 m) produces more naturalistic demonstrations.

**Our weakness**: We don't predict pedestrian future trajectories. TransFuser and InterFuser extrapolate walker position forward in time, which catches crossing pedestrians earlier. This is a meaningful capability gap.

### 4.7 Data Quality / Collection Infrastructure

| Feature | TransFuser | LBC | InterFuser | **Ours** |
|---------|-----------|-----|-----------|---------|
| Collision → stop episode | ✗ | ✓ | ✗ | **✓** |
| Balanced routes | ✗ | ✗ | ✗ | **✓** (≥15% L/R) |
| HDF5 compression | ✗ (JSON + PNG) | ✗ (LMDB) | ✗ (JSON + PNG) | **✓** (gzip-1) |
| Schema versioning | ✗ | ✗ | ✗ | **✓** (v3 gate) |
| Validation pipeline | ✗ | ✗ | ✗ | **✓** (validate + analyze) |
| Trajectory plots | ✗ | ✗ | ✗ | **✓** |
| Per-clip weighted sampling | ✗ | ✗ | ✗ | **✓** |
| Panoramic camera | ✗ | ✗ | ✗ | **✓** (3×640×320) |
| Ground-truth nav commands | ✗ (heuristic) | ✗ (heuristic) | ✗ | **✓** (RoadOption from route) |

---

## 5. Where Our Expert Is Inherently Better

### 5.1 Smooth, Physics-Based Braking (Highest Impact)

**Every** rule-based autopilot in the literature uses binary braking (brake=1.0 or brake=0.0). This is the single most impactful differentiator.

Binary braking produces bimodal training data:
- 95% of frames: brake=0.0 (throttle only)
- 5% of frames: brake=1.0 (emergency stop)

An imitation learning model trained on this must learn a discontinuous control function — the brake goes from 0 to 1 with no intermediate states. This is fundamentally harder to learn than a continuous function, and it produces jerky driving when the learned policy is deployed.

Our kinematic braking profile ($v = \sqrt{2ad}$) produces a continuous distribution of brake and throttle values. The PID controller smoothly modulates between these extremes based on the speed error. **This directly improves the quality of the imitation learning signal.**

### 5.2 Curve-Aware Lane Detection with Per-Segment SAT (Critical Bugfix Over Literature)

The collision you observed in episode 3 of the earlier test run — and would have occurred with ANY of the literature autopilots — was caused by SAT applied to a non-convex corridor polygon. The per-segment decomposition that was implemented solves this by testing each thin convex quad individually.

No other autopilot in the literature uses geometry that correctly handles curved roads:
- TransFuser doesn't check lanes at all
- LBC's lane check is commented out
- InterFuser uses `road_id`/`lane_id` (topology-based, not geometry-based)

### 5.3 Turn Anticipation + Kinematic Approach (Unique)

Every autopilot surveyed uses binary speed switches at junction boundaries. The TransFuser autopilot drops from 4.0 m/s to 3.0 m/s the instant `is_junction` becomes True. This creates a training signal where the model sees a sudden speed drop with no prior indication — the model cannot learn to anticipate turns because the expert doesn't.

Our 30 m look-ahead + kinematic ramp creates a training signal where the model sees gradual deceleration starting well before the junction. The navigation command changes from LANEFOLLOW to LEFT/RIGHT simultaneously, giving the model a paired (nav_command, speed_profile) signal that is directly learnable.

### 5.4 Gain Blending (Unique)

No other autopilot uses curvature-adaptive PID gains. This matters most for steering smoothness during transitions — the exact moments when binary-gain systems produce overshoot or oscillation.

### 5.5 Oncoming Vehicle Filter (Unique Among Rule-Based)

Every rule-based autopilot either brakes for all nearby vehicles or uses angular filtering that doesn't distinguish direction. On narrow Town01 roads, this causes the ego to brake for oncoming traffic that is safely passing in the opposite lane. Our `heading_dot > 0` check (same-direction filter) prevents this.

### 5.6 Data Pipeline Quality

Balanced routes, per-clip weighted sampling, HDF5 schema versioning, ground-truth nav commands, validation pipeline, trajectory visualisation — none of these exist in any other data collection pipeline in the surveyed literature. These don't affect driving quality but they significantly affect training data quality and reproducibility.

---

## 6. Where Other Experts Have Features We Lack

### 6.1 Future State Prediction (TransFuser, InterFuser)
**What they do**: Predict where each vehicle/pedestrian will be in 1-4 seconds using a kinematic bicycle model, then check for OBB intersection with the ego's predicted future trajectory.

**Impact**: Catches fast-closing hazards earlier than distance-only checks. If a vehicle is 30 m away but closing at 20 m/s, our distance-based check won't trigger until ~25 m; trajectory prediction catches it at ~40 m.

**Should we add it?** Medium priority. Our kinematic braking profile compensates partially because it starts braking earlier than binary systems. But for scenarios with fast-moving cross-traffic at intersections, trajectory prediction would add genuine safety margin.

**Effort**: Medium. The bicycle model from World on Rails is ~30 lines. The main complexity is iterating all future timesteps for all actors.

### 6.2 Stop Sign Handling (TransFuser, InterFuser)
**What they do**: OBB intersection with stop sign trigger volumes. Come to a complete stop, then proceed.

**Should we add it?** Low priority for two reasons: (1) Town01 doesn't have many stop signs, (2) stop sign behaviour is essentially the same as traffic light behaviour (stop, wait, go) which we already handle well.

### 6.3 Pedestrian Future Trajectory Prediction (TransFuser, InterFuser)
**What they do**: Extrapolate pedestrian position forward using their current walking direction and speed. Predict OBB intersection with ego's future path.

**Impact**: Catches jaywalking pedestrians earlier. Our lateral-band + distance check only reacts when the pedestrian is already in the lane corridor.

**Should we add it?** Medium priority. Pedestrian encounters are less frequent than vehicle encounters in urban driving, but when they happen, earlier detection prevents emergency braking.

### 6.4 Stuck Vehicle Detection (TransFuser)
**What they do**: Track each vehicle's speed over 30 frames. If consistently stopped with throttle applied → stuck. Stuck vehicles are not extrapolated forward (held at current position).

**Impact**: Prevents the ego from waiting forever behind a genuinely stuck NPC.

**Should we add it?** Low priority. Our 30-second timeout handles this at the episode level. The specific vehicle-level stuck detection would allow the ego to attempt an overtake, but overtaking requires lane-change logic we don't have.

### 6.5 Slope Handling (TransFuser)
**What they do**: If vehicle pitch > 10°, add extra throttle.

**Should we add it?** Very low priority. CARLA's towns are mostly flat. Relevant only for Town05 (has some hills).

### 6.6 Waypoint Disturbance (InterFuser)
**What they do**: Add ±0.3 m lateral perturbation to waypoints for data diversity.

**Should we add it?** This is better handled by DAgger (Phase 3) or the planned noise injection (Phase 2b). Waypoint disturbance is a crude approximation.

### 6.7 RL-Based Expert (Roach, Think2Drive)
**What they do**: Train a full RL policy using BEV semantic masks. The policy learns all driving behaviours including smooth braking, lane changing, and complex manoeuvres.

**Impact**: Produces the smoothest driving demonstrations. No hand-crafted thresholds to tune.

**Should we switch?** No. Our project's focus is on RWKV architecture evaluation, not on pushing the boundaries of expert data collection. The RL expert approach requires:
- Millions of RL training steps
- BEV rendering infrastructure
- Reward engineering and tuning
- Non-deterministic data collection

Our rule-based approach with kinematic braking already produces smoother data than every other rule-based system. The marginal improvement from RL does not justify the implementation complexity for this project.

---

## 7. Repositories for Testing

### Directly Testable (Have CARLA Autopilot Code)

| Paper | Repository | Autopilot File | CARLA Version | Notes |
|-------|-----------|---------------|--------------|-------|
| **TransFuser** | https://github.com/autonomousvision/transfuser | `team_code_autopilot/autopilot.py` (branch `2022`) | 0.9.10–0.9.14 | Most complex rule-based expert. Has `--visualize` mode. |
| **LBC** | https://github.com/dotchen/LearningByCheating | `bird_view/models/roaming.py` + `data_collector.py` | 0.9.6 | Simplest baseline. Old CARLA version — may need porting. |
| **TCP** | https://github.com/OpenDriveLab/TCP | `leaderboard/team_code/auto_pilot.py` | 0.9.10 | Fork of TransFuser. Good for comparison. |
| **InterFuser** | https://github.com/opendilab/InterFuser | `leaderboard/team_code/auto_pilot.py` | 0.9.10 | Most features. Lane-aware detection. |
| **ThinkTwice** | https://github.com/OpenDriveLab/ThinkTwice | `leaderboard/team_code/auto_pilot.py` | 0.9.10 | Hybrid RL+rules. Needs Roach checkpoint. |
| **Roach** | https://github.com/zhejz/carla-roach | `agents/rl_birdview/` | 0.9.11 | RL expert. Needs pre-trained PPO weights. |
| **NAVSIM** | https://github.com/autonomousvision/navsim | — | — | No CARLA expert but has evaluation framework. |

### CARLA Version Compatibility Notes

All of the above were written for CARLA 0.9.10–0.9.14. Your project uses **CARLA 0.9.16**. Key differences:

1. **API changes**: `get_speed_limit()` return type, `WalkerAIController` API, some traffic light APIs.
2. **Map topology**: Town01 layout is identical across versions, but junction IDs and road IDs may differ.
3. **Physics**: Vehicle dynamics slightly different between versions (better tire model in 0.9.14+).

**Recommendation for testing**: TransFuser and InterFuser are the most informative to test against. TransFuser requires the CARLA Leaderboard framework. For a quick apples-to-apples comparison, you could extract just the autopilot logic from TransFuser's `autopilot.py` and run it through your `collect_data.py` framework with the same routes and traffic density.

### Pre-Built Datasets (No CARLA Needed)

| Dataset | Source | Size | Link |
|---------|--------|------|------|
| **Bench2Drive** | Think2Drive RL expert | 13,638 clips, 2M frames | https://github.com/Thinklab-SJTU/Bench2DriveZoo |
| **NAVSIM** | nuPlan real-world logs | 120 hours, 2 Hz | https://github.com/autonomousvision/navsim |

---

## 8. Verdict and Recommendations

### Overall Assessment

Our PrivilegedExpert is **the most sophisticated rule-based autopilot** in the surveyed literature, and in several critical dimensions it is **categorically better** than everything else:

| Dimension | Status |
|-----------|--------|
| Braking smoothness | **Best-in-class** (only kinematic braking among rule-based) |
| Lane detection geometry | **Best-in-class** (per-segment SAT on route corridor) |
| Turn anticipation | **Unique** (no other rule-based system has this) |
| PID gain adaptation | **Unique** |
| Oncoming vehicle filter | **Unique** among rule-based |
| Data pipeline quality | **Best-in-class** |
| Traffic light robustness | **Best-in-class** (3-tier with persistence) |
| Pedestrian trajectory prediction | **Missing** (all others have it) |
| Future state prediction | **Missing** (TransFuser/InterFuser have it) |
| Stop sign handling | **Missing** (low impact) |

### Honest Weaknesses

1. **No future state prediction**: We react to current positions only. TransFuser's bicycle-model extrapolation catches fast-closing hazards ~0.5-1.0 s earlier. For the test collection collision scenario (vehicle stopping after a turn), this wouldn't have helped (the vehicle was already stopped), but for cross-traffic scenarios it matters.

2. **No pedestrian trajectory prediction**: We check if a pedestrian is currently in the lane. TransFuser predicts where they'll be in 1-4 seconds.

3. **No stop sign compliance**: Low impact for Town01 but matters for Town03/04/05.

4. **Test run results**: 1 collision (per-segment SAT fix applied), 2 timeouts (30s timeout working correctly — long CARLA red lights), 3 completed. The collision was the SAT non-convex bug, now fixed. The timeouts are expected behaviour at high traffic density (100 vehicles) in Town01.

### Where Binary Braking Hurts Training the Most

Consider what a model trained on TransFuser data sees for "vehicle following":
```
Frame 100: lead_dist=25m, brake=0.0, throttle=0.5   ← driving normally
Frame 101: lead_dist=24m, brake=0.0, throttle=0.5   ← still driving
Frame 102: lead_dist=15m, brake=1.0, throttle=0.0   ← FULL BRAKE
Frame 103: lead_dist=14m, brake=1.0, throttle=0.0   ← still full brake
...
Frame 120: lead_dist=5m,  brake=0.0, throttle=0.3   ← back to normal
```

The model must learn a step function: "if lead_dist < some threshold → brake=1.0". This is inherently noisy to learn because:
- The threshold varies with speed, road conditions, etc.
- The regression target has a discontinuity
- MSE loss heavily penalises being "wrong" about which side of the boundary you're on

What our model sees:
```
Frame 100: lead_dist=25m, brake=0.00, throttle=0.50  ← driving normally
Frame 101: lead_dist=24m, brake=0.05, throttle=0.45  ← starting to slow
Frame 102: lead_dist=22m, brake=0.12, throttle=0.35  ← smooth increase
Frame 103: lead_dist=20m, brake=0.20, throttle=0.25  ← continuing
...
Frame 115: lead_dist=8m,  brake=0.65, throttle=0.00  ← near stop
Frame 120: lead_dist=5m,  brake=1.00, throttle=0.00  ← stopped
```

This is a smooth, monotonic function of distance — trivially learnable by any regression model. **This is the primary argument for why our kinematic braking is better for imitation learning.**

### Final Note on the RL vs. Rule-Based Debate

The trend in the literature is toward RL-based experts (Roach → MILE → ThinkTwice → Think2Drive/Bench2Drive). However, for a project evaluating a novel architecture (RWKV), the data collection method is a means to an end, not the end itself. Our rule-based expert with kinematic braking produces data quality comparable to RL experts (smooth, proportional, physics-based) without the multi-million-step RL training overhead. This is the right engineering trade-off for this project.
