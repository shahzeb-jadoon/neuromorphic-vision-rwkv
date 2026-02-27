# LADY, Bench2Drive & NAVSIM — Expert/Autopilot & Benchmark Analysis

**Date**: 2025-02-22
**Purpose**: Research notes on modern CARLA-era autonomous driving papers and how
they differ from the older TransFuser/LBC/Roach expert-data-collection paradigm.

---

## Executive Summary

| Paper | Expert/Autopilot | Data Source | Eval Protocol | CARLA Version |
|-------|-----------------|-------------|---------------|---------------|
| **LADY** (Huang 2025) | None — *consumer* of existing benchmarks | NAVSIM (nuPlan/OpenScene) + Bench2Drive datasets | PDMS on NAVSIM `navtest`; Driving Score + SR on Bench2Drive 220 routes | N/A for NAVSIM; CARLA 0.9.15 for Bench2Drive |
| **Bench2Drive** (Jia 2024) | **Think2Drive** — RL world-model expert (custom) | Collected in CARLA v2 (0.9.15) by Think2Drive | 220 short routes, 44 scenarios, Driving Score + Success Rate + Multi-Ability + Efficiency + Smoothness | CARLA 0.9.15 (Leaderboard v2 scenarios) |
| **NAVSIM** (Dauner 2024) | **Human driver** (real-world logs) — no CARLA expert needed | nuPlan / OpenScene (real-world dataset, 120 h, 2 Hz) | PDMS / EPDMS (non-reactive pseudo-simulation); v2 adds pseudo closed-loop two-stage | N/A — real-world data |

**Key paradigm shift**: Unlike TransFuser/LBC/Roach which all build custom
rule-based CARLA autopilots to collect training data, the newer generation either
(a) uses an RL-trained expert (Bench2Drive), (b) uses real-world human driving
logs (NAVSIM), or (c) is a pure model architecture paper that consumes existing
benchmark datasets (LADY).

---

## 1. LADY — Linear Attention for Autonomous Driving Efficiency

**Paper**: Huang et al., "LADY: Linear Attention for Autonomous Driving
Efficiency without Transformers," arXiv:2512.15038, Dec 2025.
**Authors**: Jihao Huang, Xi Xia, Zhiyuan Li, Tianle Liu, Jingke Wang, Junbo Chen, Tengju Ye (Udeer AI / Zhejiang Univ.)
**Status**: Under review.
**GitHub**: **No public repository found** (as of Feb 2026).

### 1.1 Expert/Autopilot

LADY does **not** have its own expert or autopilot. It is a **model
architecture** paper that evaluates on two existing benchmarks:

1. **NAVSIM** — uses the OpenScene/nuPlan real-world dataset with the PDMS
   evaluation protocol. No CARLA autopilot involved.
2. **Bench2Drive** — uses the official Bench2Drive training set (collected by
   Think2Drive RL expert) and evaluates in CARLA closed-loop.

LADY reuses whatever data the benchmark provides; it does not collect its own.

### 1.2 Architecture (Relevant to RWKV)

LADY is **the first fully linear-attention-based end-to-end autonomous driving
model**, built on **RWKV-7 ("Goose")**. This is directly relevant to this
project:

- **Encoder**: ResNet-34 backbone (same as TransFuser) extracts multi-frame
  camera and LiDAR features.
- **Temporal fusion**: Stacked RWKV-7 blocks fuse multi-frame features. During
  training, parallel mode; during inference, sequential/RNN mode with constant
  time and memory regardless of history length.
- **Cross-attention**: Custom **LICA** (Lightweight Linear Cross-Attention)
  mechanism — concatenates BEV+ego features with query tokens, passes through
  a second RWKV-7 block, and extracts the last M tokens. Avoids quadratic
  softmax attention entirely.
- **Decoder**: Truncated diffusion policy (from DiffusionDrive) generates
  multi-modal trajectory candidates. 2 denoising steps at inference from 100
  clustered anchor trajectories.
- **Scoring**: Confidence scores distilled from NAVSIM's rule-based PDM
  scoring; highest-scoring trajectory selected at inference.

### 1.3 Data & Training Details

| Parameter | Value |
|-----------|-------|
| Backbone | ResNet-34 |
| Input frames | 10 (current + 9 history, 5 s at 2 Hz) |
| Camera input | 3 cropped forward-facing cameras → 1024×256 stitched |
| LiDAR input | Rasterized BEV |
| Ego status | Velocity, acceleration, driving commands (left/straight/right) |
| Output | 8 waypoints over 4 s at 2 Hz (x, y, heading) |
| Training | 100 epochs on `navtrain` split, Adam, lr=1e-4, ReduceLROnPlateau |
| GPUs | 4× NVIDIA RTX 4090 (48 GB), batch size 4 |
| Diffusion | 2 cascaded decoder layers, truncated to 50/1000 steps |
| Edge deployment | NVIDIA Jetson AGX Orin (32 GB) |

### 1.4 Benchmark Results

**NAVSIM `navtest`** (open-loop, PDMS):

| Method | NC | DAC | TTC | Comf. | EP | PDMS |
|--------|-----|-----|-----|-------|-----|------|
| Transfuser | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| UniAD | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| DiffusionDrive | 97.6 | 96.0 | 92.5 | 100 | 85.3 | 88.1 |
| **LADY (15 frames)** | **98.0** | **97.3** | **94.0** | **100** | **88.6** | **90.9** |

**Bench2Drive** (closed-loop, 220 routes):

| Method | DS ↑ | SR (%) | Efficiency | Smoothness |
|--------|------|--------|------------|------------|
| UniAD-Tiny | 40.73 | 13.2 | 123.9 | 47.0 |
| VAD | 42.35 | 15.0 | 157.9 | 46.0 |
| DriveTransformer | 63.46 | 35.0 | 100.6 | 20.8 |
| iPad (w/ distill) | 70.07 | 42.7 | — | — |
| **LADY (∞ frames)** | **71.17** | **43.2** | — | — |

### 1.5 Key Takeaways for This Project

- **RWKV-7 is proven effective for AD temporal fusion** — LADY shows that
  RWKV's RNN-mode inference gives constant time/memory while matching or
  exceeding transformer performance.
- The model is validated on **NVIDIA Jetson AGX Orin** — directly relevant
  to edge deployment goals.
- LADY uses the **same ResNet-34 backbone as TransFuser** — architecture
  choices are comparable to this project.
- Multi-frame fusion with RWKV yields measurable gains: PDMS increases from
  86.8 (1 frame) to 90.9 (15 frames).

---

## 2. Bench2Drive — Jia et al., 2024

**Paper**: Jia et al., "Bench2Drive: Towards Multi-Ability Benchmarking of
Closed-Loop End-to-End Autonomous Driving," NeurIPS 2024 Datasets and
Benchmarks Track. arXiv:2406.03877.
**Authors**: Xiaosong Jia, Zhenjie Yang, Qifeng Li, Zhiyuan Zhang, Junchi Yan
(Shanghai Jiao Tong University).
**GitHub**:
- Benchmark: <https://github.com/Thinklab-SJTU/Bench2Drive>
- Model Zoo: <https://github.com/Thinklab-SJTU/Bench2DriveZoo>
- Key files:
  - `tools/data_collect.py` — data collection code
  - `leaderboard/` — evaluation infrastructure (forked CARLA Leaderboard v2)
  - `scenario_runner/` — scenario definitions
  - `tools/merge_route_json.py` — metric aggregation
  - `tools/ability_benchmark.py` — multi-ability scoring
  - `tools/efficiency_smoothness_benchmark.py` — efficiency/smoothness metrics

### 2.1 Expert/Autopilot: Think2Drive (RL World-Model Expert)

Bench2Drive uses **Think2Drive** (Li et al., ECCV 2024, arXiv:2402.16720) as
its data collection expert. This is fundamentally different from
TransFuser/LBC/Roach rule-based autopilots:

| Aspect | TransFuser Autopilot | Think2Drive |
|--------|---------------------|-------------|
| **Type** | Rule-based (PID + OBB collision checks) | Model-based RL with latent world model |
| **Training** | Hand-crafted rules, no training | RL training in latent world model (~3 days on 1× A6000) |
| **Input** | Privileged CARLA info (GT bounding boxes, traffic light states) | Privileged CARLA info (GT states encoded) |
| **Capabilities** | Basic: lane following, traffic lights, vehicle/ped avoidance | Handles all 44 CARLA v2 interactive scenarios (cut-in, overtaking, give way, etc.) |
| **Scenario coverage** | ~6 basic skills (Longest6 benchmarks) | 44 interactive scenarios |
| **CARLA version** | v1 (0.9.10–0.9.13) | v2 (0.9.15, Leaderboard 2.0) |

Think2Drive was specifically chosen because **it is the only expert model able
to solve all 44 Leaderboard v2 scenarios** at the time of Bench2Drive's
construction. The rule-based expert PDM-Lite (from Beißwenger 2024) was later
open-sourced as an alternative.

**Think2Drive key details:**
- **Architecture**: Learns a latent world model of the CARLA environment, then
  trains a planner via RL within that latent space.
- **Efficiency**: Reaches expert-level proficiency within 3 days of training on
  a single A6000 GPU.
- **Performance**: 100% route completion on CARLA v2 scenarios — no previously
  published method achieved this.
- **Outputs provided to student models**: In addition to sensor data, Bench2Drive
  provides Think2Drive's value estimations and features for knowledge
  distillation (used by TCP, ThinkTwice, DriveAdapter baselines).

### 2.2 Data Collection Details

| Parameter | Value |
|-----------|-------|
| **Expert agent** | Think2Drive (RL) |
| **CARLA version** | 0.9.15 (Leaderboard v2) |
| **Total frames** | ~2 million fully annotated |
| **Total clips** | 13,638 short clips |
| **Clip length** | ~150 meters per clip, single scenario each |
| **Sampling rate** | 10 Hz |
| **Scenarios** | 44 interactive scenarios |
| **Weathers** | 23 weather conditions (sunny, foggy, rainy, etc.) |
| **Towns** | 12 towns (urban, village, university, etc.) — dominated by Town12/13 |
| **Data subsets** | Mini (10 clips), Base (1,000 clips, ~400 GB), Full (10,000 clips, ~4 TB) |
| **Quality filtering** | Only infraction-free runs kept; buggy renderings manually filtered |

**Sensor configuration** (mirrors nuScenes):

| Sensor | Details |
|--------|---------|
| LiDAR × 1 | 64 channels, 85 m range, 600K points/sec |
| Camera × 6 | Surround coverage, 900×1600, JPEG quality=20 |
| Radar × 5 | 100 m range, 30° horizontal/vertical FoV |
| IMU + GNSS | Location, yaw, speed, acceleration, angular velocity |
| BEV Camera | For debugging/visualization/remote sensing |
| HD-Map | Lanes, centerlines, topology, dynamic light states, trigger volumes |

**Annotations per frame**: 3D bounding boxes for vehicles/pedestrians/traffic
signs/lights, depth maps, semantic segmentation, instance segmentation, ego
control signals (throttle/steer/brake), expert value estimations, and expert
features.

### 2.3 Benchmark/Evaluation Protocol

**220 Test Routes**: 44 scenarios × 5 routes each, different weather/town
per route. Each route is ~150 m with a single scenario.

**Metrics**:

1. **Success Rate (SR)**: Proportion of routes completed without any
   infraction, within the time limit.

2. **Driving Score (DS)**: CARLA Leaderboard v2 metric:
   $$DS = \frac{1}{n_{total}} \sum_{i=1}^{n_{total}} \text{RouteCompletion}_i \times \prod_{j=1}^{n_{penalty}} p_{i,j}$$
   where penalty scores are multiplicative (e.g., pedestrian collision = 0.50,
   vehicle collision = 0.60, red light = 0.70, etc.).

3. **Multi-Ability Results**: Scores disaggregated across 5 skill categories:
   - **Merging** (16 scenarios): Cut-in, lane change, junction turns, highway
     exit, etc.
   - **Overtaking** (9 scenarios): Accidents, construction, parked obstacles,
     door opening.
   - **Emergency Brake** (12 scenarios): Dynamic crossing, hard brake, blocked
     intersection, control loss, etc.
   - **Give Way** (2 scenarios): Invading turn, yield to emergency vehicle.
   - **Traffic Sign** (18 scenarios): Signalized/non-signalized turns,
     stop signs, red/green lights, T-junction, etc.

4. **Driving Efficiency**: Speed percentage relative to nearby vehicles,
   checked at 20 checkpoints (every 5% of route length). Excluded from driving
   score calculation.

5. **Driving Smoothness/Comfortness**: Segment-based (20-frame segments)
   evaluation of longitudinal acceleration, lateral acceleration, yaw rate,
   yaw acceleration, jerk — compared against nuPlan expert thresholds:
   - Longitudinal accel: [-4.05, 2.40] m/s²
   - Lateral accel: [-4.89, 4.89] m/s²
   - Yaw rate: [-0.95, 0.95] rad/s
   - Yaw accel: [-1.93, 1.93] rad/s²
   - Longitudinal jerk: [-4.13, 4.13] m/s³
   - Jerk magnitude: [-8.37, 8.37] m/s³

**Infraction penalties** (from CARLA Leaderboard v2):

| Infraction | Penalty | Notes |
|------------|---------|-------|
| Pedestrian Collision | ×0.50 | Per infraction |
| Vehicle Collision | ×0.60 | Per infraction |
| Other Collision | ×0.65 | Per infraction |
| Running Red Light | ×0.70 | Per infraction |
| Scenario Timeout | ×0.70 | Fail to pass scenario in 4 min |
| Too Slow | ×0.70 | Speed check vs. nearby vehicles |
| No Give Way | ×0.70 | Fail to yield to emergency vehicle |
| Off-Road | — | Not counted in route completion |
| Route Deviation | — | >30 m deviation → immediate shutdown |
| Agent Blocked | — | No action for 180 s → shutdown |
| Route Timeout | — | Exceeded time limit → shutdown |

### 2.4 Key Differences from TransFuser/LBC Approach

| Aspect | TransFuser/LBC Era | Bench2Drive |
|--------|-------------------|-------------|
| **CARLA version** | v1 (0.9.10–0.9.13), ~6 basic scenarios | v2 (0.9.15), 44 interactive scenarios |
| **Expert type** | Hand-crafted rule-based autopilot | RL-trained world-model expert (Think2Drive) |
| **Route length** | Long routes (7–10 km, Longest6/Town05Long) | Short routes (~150 m each) |
| **Data sharing** | Each paper collects its own data | Official standardized dataset |
| **Evaluation** | Driving Score on long routes (high variance) | DS + SR + Multi-Ability + Efficiency + Smoothness on 220 short routes |
| **Scenario complexity** | Lane following, basic turns, traffic lights | Cut-in, construction, accidents, emergency vehicles, door opening, etc. |
| **NPC behavior** | Simple traffic manager | Rule-based cautious/normal/aggressive NPCs with detailed parameters |
| **Annotation richness** | Limited (waypoints, basic labels) | Full annotation: 3D boxes, depth, semantic seg, expert features |

### 2.5 NPC Agent Behavior (Bench2Drive)

Bench2Drive documents the NPC behavior parameters:

| Parameter | Cautious | Normal | Aggressive |
|-----------|----------|--------|------------|
| max_speed (km/h) | 40 | 50 | 70 |
| speed_lim_dist (km/h) | 6 | 3 | 1 |
| speed_decrease (km/h) | 12 | 10 | 8 |
| safety_time (s) | 3 | 3 | 3 |
| min_proximity_threshold (m) | 12 | 10 | 8 |
| braking_distance (m) | 6 | 5 | 4 |
| tailgate_counter | 0 | 0 | -1 |

NPC decision algorithm (simplified): red light / stop sign → emergency stop;
pedestrian within braking distance → emergency stop; vehicle nearby within
braking distance → emergency stop, else → car following; at intersection turning
→ adjust speed + PID; normal driving → PID to target speed.

---

## 3. NAVSIM — Dauner et al., 2024

**Paper**: Dauner et al., "NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle
Simulation and Benchmarking," NeurIPS 2024 Datasets and Benchmarks.
arXiv:2406.15349.
**v2 Paper**: Cao et al., "Pseudo-Simulation for Autonomous Driving," CoRL
2025. arXiv:2506.04218.
**Authors**: Daniel Dauner, Marcel Hallgarten, Tianyu Li, et al. (University of
Tübingen, Tübingen AI Center, NVIDIA Research, Robert Bosch GmbH).
**GitHub**: <https://github.com/autonomousvision/navsim>
- Key files:
  - `navsim/agents/` — agent implementations (abstract_agent, constant_velocity,
    ego_status_mlp, transfuser/)
  - `navsim/common/dataclasses.py` — SensorConfig, Trajectory, AgentInput
  - `navsim/planning/` — training and evaluation infrastructure
  - `docs/metrics.md` — PDMS / EPDMS definition
  - `docs/agents.md` — agent interface specification
  - `docs/splits.md` — dataset splits documentation
  - `scripts/evaluation/` — evaluation scripts

### 3.1 Expert/Autopilot: No CARLA Expert — Uses Real-World Human Driving

NAVSIM fundamentally differs from all CARLA-based approaches:

- **No CARLA simulator** is used for data collection.
- **No autopilot/expert** is used — the data comes from **nuPlan**, a
  real-world autonomous driving dataset recorded by human drivers.
- **OpenScene** is a compact redistribution of nuPlan at lower frequency (2 Hz).
- The dataset contains **120 hours of real-world driving logs** from actual
  urban driving.

The "expert" in NAVSIM is simply the **recorded human driver trajectory** — the
`Human` agent that replays what the human actually did. This serves as the
upper-bound reference for evaluation.

### 3.2 Data Details

| Parameter | Value |
|-----------|-------|
| **Source** | nuPlan / OpenScene (real-world) |
| **Duration** | 120 hours of driving logs |
| **Frequency** | 2 Hz (downsampled from nuPlan's original rate) |
| **Sensors** | 8 cameras + 1 merged LiDAR point cloud (from 5 LiDARs) |
| **History** | 2 seconds into the past at 2 Hz (4 frames) |
| **Target** | 4-second future trajectory at 10 Hz evaluation |
| **Ego status** | Pose, velocity, acceleration + discrete driving command (left/straight/right/unknown) |

**Dataset splits**:

| Split | Description | Logs | Sensors |
|-------|-------------|------|---------|
| `trainval` | Large training/validation, regular driving | 14 GB | >2000 GB |
| `test` | Small test, regular driving | 1 GB | 217 GB |
| `mini` | Demo | 1 GB | 151 GB |
| `navtrain` | Filtered for non-trivial scenes (standard training) | 14 GB | 445 GB |
| `navtest` | Filtered for non-trivial scenes (standard test, v1) | 983 MB | 223 GB |
| `navhard_two_stage` | Real + synthetic scenes (v2 test) | 892 MB | 31 GB |

The `navtrain` and `navtest` splits are **filtered** to increase the proportion
of interesting/challenging driving scenarios (not just straight-line cruising).

### 3.3 Benchmark/Evaluation Protocol

#### NAVSIM v1: Predictive Driver Model Score (PDMS)

Non-reactive simulation: background actors follow recorded trajectories; ego
moves based on LQR controller for 4 seconds.

$$PDMS = \left(\prod_{m \in \{NC, DAC\}} score_m\right) \times \left(\frac{\sum_{w \in \{EP, TTC, C\}} weight_w \times score_w}{\sum_{w \in \{EP, TTC, C\}} weight_w}\right)$$

**Multiplicative penalties** (zero out the score if violated):
- **NC** (No at-fault Collision): {0, ½, 1}
- **DAC** (Drivable Area Compliance): {0, 1}

**Weighted sub-scores**:
- **EP** (Ego Progress along route center): weight=5, range [0, 1]
- **TTC** (Time to Collision within bound): weight=5, {0, 1}
- **C** (Comfort — acceleration/jerk thresholds): weight=2, {0, 1}

#### NAVSIM v2: Extended PDMS (EPDMS)

Adds new metrics and false-positive filtering via the human agent:

| Metric | Type | Range | New in v2 |
|--------|------|-------|-----------|
| NC (No at-fault Collision) | multiplier | {0, ½, 1} | No |
| DAC (Drivable Area Compliance) | multiplier | {0, 1} | No |
| **DDC** (Driving Direction Compliance) | multiplier | {0, ½, 1} | **Yes** |
| **TLC** (Traffic Light Compliance) | multiplier | {0, 1} | **Yes** |
| EP (Ego Progress) | weighted (5) | [0, 1] | No |
| TTC (Time to Collision) | weighted (5) | {0, 1} | No |
| **LK** (Lane Keeping) | weighted (2) | {0, 1} | **Yes** |
| HC (History Comfort) | weighted (2) | {0, 1} | Updated |
| **EC** (Extended Comfort) | weighted (2) | {0, 1} | **Yes** |

$$EPDMS = \left(\prod_{m \in \{NC, DAC, DDC, TLC\}} filter_m(agent, human)\right) \cdot \left(\frac{\sum_{m \in \{TTC, EP, HC, LK, EC\}} w_m \cdot filter_m(agent, human)}{\sum w_m}\right)$$

Where: $filter_m(agent, human) = 1.0$ if $m(human) = 0$, else $m(agent)$.

This false-positive filtering means: if the human driver also violated a rule
(e.g., entered opposing lane to overtake), the agent is not penalized for the
same violation.

#### NAVSIM v2: Pseudo Closed-Loop (Two-Stage Aggregation)

1. **Stage 1**: Evaluate initial scene over 4 seconds using EPDMS.
2. **Stage 2**: Multiple pre-computed follow-up scenes branch from the initial
   scene. Each follow-up starts from a *different* state (lateral offset,
   different speed, etc.) simulating what would happen after different first-stage
   plans.
3. **Aggregation**: Follow-up scenes weighted by Gaussian kernel based on
   proximity to where the submitted planner actually ended in Stage 1. Final
   score = Stage 1 × weighted Stage 2.

This approximates closed-loop behavior (error recovery, compounding errors)
while maintaining open-loop computational efficiency (~6× less compute than
closed-loop simulation).

### 3.4 Baseline Agents

| Agent | Type | Description |
|-------|------|-------------|
| `ConstantVelocityAgent` | Naive | Maintains constant speed/heading → straight line |
| `EgoStatusMLPAgent` | Blind | MLP on ego velocity/acceleration/command only (no sensors) |
| `TransfuserAgent` | Sensor | ResNet-34 CNN on front camera + LiDAR BEV, transformer fusion, BEV segmentation + DETR detection aux tasks |
| `LatentTransfuserAgent` | Sensor (no LiDAR) | TransFuser variant with positional encoding replacing LiDAR |
| `Human` | Reference | Replays recorded human trajectory |

### 3.5 Agent Interface

All agents implement `AbstractAgent` with:
- `get_sensor_config()` → `SensorConfig` specifying which sensors/history frames
- `compute_trajectory()` → `Trajectory` (array of BEV poses: x, y, heading)
- For learned agents: `forward()`, `compute_loss()`, `get_optimizers()`,
  `get_feature_builders()`, `get_target_builders()`

Output: trajectory of BEV poses for 4 seconds. `TrajectorySampling` config
handles interpolation if output frequency ≠ evaluation frequency (10 Hz).

### 3.6 Key Differences from TransFuser/LBC Approach

| Aspect | TransFuser/LBC/Roach | NAVSIM |
|--------|---------------------|--------|
| **Data** | Simulated (CARLA) | Real-world (nuPlan) |
| **Expert** | Rule-based CARLA autopilot | Human driver recordings |
| **Evaluation** | Closed-loop in CARLA (reactive) | Non-reactive pseudo-simulation on real data |
| **Metrics** | Driving Score (route completion × infraction penalties) | PDMS/EPDMS (multi-faceted scoring) |
| **Domain gap** | Train and test in same simulator | Train on real data, evaluate on real scenarios |
| **Scalability** | Limited by CARLA rendering speed | Highly scalable (no simulator required) |
| **Reactivity** | Full closed-loop interaction | Non-reactive (v1), pseudo closed-loop (v2) |
| **Competition** | CARLA Leaderboard | CVPR 2024 (143 teams, 463 submissions); ICCV/AGC 2025 |

---

## 4. Comparison: How Modern Benchmarks Differ from TransFuser/LBC Era

### 4.1 Expert/Data Collection Evolution

```
  TransFuser (2021-2023)        Bench2Drive (2024)         NAVSIM (2024)
  ========================     ====================       ================
  Rule-based autopilot         RL Expert (Think2Drive)    Human Drivers
  - PID controllers            - World-model RL           - Real-world logs
  - OBB collision checks       - Handles 44 scenarios     - 120 hours
  - Hand-tuned params          - 3 days A6000 training    - nuPlan dataset
  - ~6 basic skills            - 100% route completion    - No CARLA needed
  - Each paper collects own    - Standardized dataset     - Standardized
  - CARLA v1 (0.9.10-0.9.13)  - CARLA v2 (0.9.15)       - Real world
```

### 4.2 Evaluation Protocol Evolution

```
  Longest6 / Town05Long        Bench2Drive 220            NAVSIM PDMS/EPDMS
  ========================     ====================       ==================
  Long routes (7-10 km)        Short routes (~150 m)      4-second scenarios
  1 aggregate Driving Score    DS + SR + 5 abilities      Multi-metric score
  High variance (exp decay)    Low variance (isolated)    Standardized scoring
  Route completion focus       Skill-based assessment     Progress + safety
  Closed-loop (reactive)       Closed-loop (reactive)     Non-reactive / pseudo CL
  Same sim for train+eval      Same sim for train+eval    Real data eval
```

### 4.3 CARLA Leaderboard v1 vs v2

| Aspect | Leaderboard v1 (TransFuser era) | Leaderboard v2 (Bench2Drive era) |
|--------|-------------------------------|----------------------------------|
| CARLA version | 0.9.10–0.9.13 | 0.9.15 |
| Scenarios | ~6 basic (lane follow, turns, lights, collisions) | 44 interactive (cut-in, overtaking, construction, emergency vehicles, etc.) |
| Routes | Long (7–10 km), 6 routes (Longest6) | Short (~150 m), 220 routes |
| Maps | Town01–05 | Town01–13 (including large city-scale maps) |
| NPC behavior | Basic traffic manager | Detailed cautious/normal/aggressive behavior models |
| Expert availability | Multiple rule-based autopilots (TransFuser, Roach, LBC) | Think2Drive (RL), PDM-Lite (rule-based, released later) |
| Standard dataset | None (each paper collects its own) | Official: 2M frames, 13K clips |

### 4.4 Implications for This Project

1. **RWKV is validated for AD**: LADY proves that RWKV-7 linear attention is
   effective for autonomous driving temporal fusion, achieving SOTA on NAVSIM
   and competitive results on Bench2Drive.

2. **TransFuser autopilot remains valid**: The TransFuser-style autopilot
   (our current expert) covers basic driving skills. For more complex scenarios
   (cut-in, emergency vehicles, etc.), Think2Drive or PDM-Lite would be needed.

3. **Evaluation options**:
   - For CARLA-based eval: Use Bench2Drive's 220-route protocol with our
     trained model.
   - For real-world-aligned eval: Port model to NAVSIM framework and evaluate
     with PDMS/EPDMS on nuPlan data.

4. **Architecture alignment**: LADY uses the same ResNet-34 backbone as
   TransFuser and the same RWKV-7 architecture we're targeting. Their
   implementation details (10-frame history, 4s prediction horizon, diffusion
   decoder) provide a direct reference.

5. **Edge deployment**: LADY demonstrates RWKV-based AD running on Jetson AGX
   Orin with constant time/memory — validates our edge deployment feasibility.

---

## 5. GitHub Repositories & Key File Paths Summary

### LADY
- **Paper**: arXiv:2512.15038
- **Code**: Not publicly available (as of Feb 2026)
- **Reference architecture**: Flash Linear Attention (FLA) library for RWKV-7
  integration: <https://github.com/sustcsonglin/flash-linear-attention>

### Bench2Drive
- **Benchmark repo**: <https://github.com/Thinklab-SJTU/Bench2Drive>
  - `tools/data_collect.py` — Data collection with Think2Drive expert
  - `tools/visualize.py` — Dataset visualization
  - `tools/merge_route_json.py` — Merge evaluation JSONs
  - `tools/ability_benchmark.py` — Multi-ability scoring
  - `tools/efficiency_smoothness_benchmark.py` — Efficiency/smoothness metrics
  - `tools/gen_hdmap.py` — HD-Map generation
  - `leaderboard/` — CARLA Leaderboard v2 eval infrastructure
  - `leaderboard/data/` — Route XML files (220 routes)
  - `leaderboard/scripts/run_evaluation_debug.sh` — Debug eval
  - `leaderboard/scripts/run_evaluation_multi_uniad.sh` — Multi-GPU eval
  - `scenario_runner/` — CARLA v2 scenario definitions
  - `docs/anno.md` — Annotation format documentation
- **Model Zoo**: <https://github.com/Thinklab-SJTU/Bench2DriveZoo>
  - Branches: `uniad/vad`, `tcp/admlp`
  - `adzoo/` — Model configs
  - `team_code/` — CARLA agent implementations
  - `team_code/vad_b2d_agent_visualize.py` — Visualization agent
- **Dataset**: <https://huggingface.co/datasets/rethinklab/Bench2Drive>
- **Think2Drive paper**: arXiv:2402.16720

### NAVSIM
- **Main repo**: <https://github.com/autonomousvision/navsim>
  - `navsim/agents/abstract_agent.py` — Agent interface
  - `navsim/agents/constant_velocity_agent.py` — CV baseline
  - `navsim/agents/ego_status_mlp_agent.py` — MLP baseline
  - `navsim/agents/transfuser/` — TransFuser baseline
  - `navsim/common/dataclasses.py` — SensorConfig, Trajectory, etc.
  - `navsim/planning/` — Training/evaluation framework
  - `docs/metrics.md` — PDMS/EPDMS definitions
  - `docs/agents.md` — Agent interface docs
  - `docs/splits.md` — Dataset split docs
  - `docs/cache.md` — Data format docs
  - `docs/traffic_agents.md` — Traffic simulation docs
  - `scripts/evaluation/` — Evaluation scripts
  - `download/` — Dataset download scripts
  - `tutorial/` — Jupyter notebook tutorials
- **NAVSIM v1 paper**: arXiv:2406.15349
- **NAVSIM v2 paper**: arXiv:2506.04218
- **Related repos**:
  - CARLA Garage: <https://github.com/autonomousvision/carla_garage>
  - TransFuser: <https://github.com/autonomousvision/transfuser>
  - SLEDGE: <https://github.com/autonomousvision/sledge>

---

## 6. Unique Aspects of Each Data Pipeline / Expert System

### LADY
- **No custom expert or data pipeline** — purely an architecture paper.
- Unique: **First fully linear-attention AD model** using RWKV-7.
- Unique: Demonstrates **constant-time multi-frame fusion** during inference
  via RWKV hidden state propagation (unlike transformers which grow quadratically).
- Unique: **LICA mechanism** — cross-attention via concatenation + RWKV block
  instead of softmax attention.

### Bench2Drive
- **RL expert instead of rules**: Think2Drive learns to drive via RL in a
  latent world model, handling 44 scenarios that no rule-based expert could
  solve at the time.
- **Standardized training data**: Eliminates the "each paper collects its own
  data" problem that plagued TransFuser/LBC era comparisons.
- **Multi-ability evaluation**: Disaggregates performance into 5 skill
  categories, providing actionable insights rather than a single aggregate score.
- **Short-route evaluation**: ~150 m routes reduce variance from exponential
  penalty decay in long routes.
- **Expert feature distillation**: Think2Drive features provided alongside
  sensor data, enabling knowledge distillation from expert to student models
  (TCP, ThinkTwice, DriveAdapter all benefit significantly).
- **nuScenes-like sensor config**: 6 cameras + LiDAR + radar matching nuScenes
  layout, easing transfer of existing methods.

### NAVSIM
- **Real-world data**: No simulation domain gap — uses actual human driving
  from nuPlan.
- **Non-reactive simulation**: Evaluates planned trajectories without running a
  full simulator. Background actors follow recorded paths.
- **Pseudo closed-loop (v2)**: Two-stage evaluation approximates closed-loop
  behavior (error recovery assessment) without interactive simulation.
- **False-positive filtering**: If the human driver also violated a rule, the
  evaluated agent is not penalized — reduces unfair penalization.
- **Driving command**: Based solely on intended route (left/straight/right),
  not entangled with obstacle/traffic information (unlike nuScenes commands).
- **Massive competition participation**: 143 teams, 463 submissions at CVPR
  2024 — most active AD benchmark competition.
- **LiDAR emphasis**: Explicitly encourages LiDAR use to reduce compute
  requirements vs. high-res surround cameras.

---

## 7. Summary Table

| Feature | TransFuser (our current) | Bench2Drive | NAVSIM | LADY |
|---------|------------------------|-------------|--------|------|
| Expert type | Rule-based autopilot | RL (Think2Drive) | Human drivers | N/A (uses benchmarks) |
| Data collection | CARLA v1 simulation | CARLA v2 simulation | Real-world (nuPlan) | N/A |
| Custom expert? | Yes | Yes (Think2Drive) | No (human logs) | No |
| Vehicle following | PID + OBB checks | RL-learned behavior | N/A | N/A |
| Traffic lights | OBB intersection + SAT | RL-learned | N/A | N/A |
| Braking profiles | Binary brake/no-brake | RL-learned (continuous) | N/A | N/A |
| Scenarios covered | ~6 basic | 44 interactive | Real-world scenarios | N/A |
| Evaluation type | Closed-loop (CARLA) | Closed-loop (CARLA) | Non-reactive/pseudo CL | Uses both |
| Primary metric | Driving Score | DS + SR + Multi-Ability | PDMS / EPDMS | PDMS + DS |
| Dataset standard | No (each paper makes own) | Yes (official) | Yes (nuPlan/OpenScene) | Uses both |
| RWKV relevance | Backbone target | Eval benchmark | Eval benchmark | **Uses RWKV-7** |
