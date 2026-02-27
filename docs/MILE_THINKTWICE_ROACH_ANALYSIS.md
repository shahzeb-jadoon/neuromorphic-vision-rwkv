# MILE, ThinkTwice & Roach — Expert/Autopilot Analysis

**Date**: February 2026

---

## Table of Contents

1. [Roach (RL Coach) — Zhang et al., ICCV 2021 / ICLR 2022](#1-roach-rl-coach)
2. [MILE — Hu et al., NeurIPS 2022](#2-mile)
3. [ThinkTwice — Jia et al., CVPR 2023](#3-thinktwice)
4. [Cross-Comparison Table](#4-cross-comparison-table)
5. [RL Expert vs Rule-Based Expert](#5-rl-expert-vs-rule-based-expert)

---

## 1. Roach (RL Coach)

**Paper**: "End-to-End Urban Driving by Imitating a Reinforcement Learning Coach"
— Zhang, Liniger, Dai, Yu, Van Gool (ICCV 2021)

**Repository**: <https://github.com/zhejz/carla-roach>

**Key insight**: Roach trains a **PPO-based reinforcement learning agent** on
a **bird's-eye-view (BEV) representation** as the privileged expert, rather
than using hand-crafted rules. This RL expert is then used as the "coach"
for imitation learning via DAgger.

### 1.1 Expert/Autopilot Features

Roach has **two** expert systems:

| # | System | Type | File |
|---|--------|------|------|
| 1 | **Roach (PPO)** — primary expert | RL-trained policy | `agents/rl_birdview/rl_birdview_agent.py` |
| 2 | **LBC Roaming Agent** — baseline autopilot | Rule-based | `agents/expert/lbc_roaming_agent.py` |

**The RL expert (Roach)** consumes:
- **BEV semantic masks** rendered by `carla_gym/core/obs_manager/birdview/` —
  channels for road, route, lane markings, vehicles (4 timesteps), walkers
  (4 timesteps), and traffic lights (red/yellow/green).
- **State vector** containing: current control (throttle, steer, brake, gear),
  velocity `vel_xy`, and optionally speed, acceleration.

**The rule-based LBC autopilot** handles hazards via:

| Feature | How Triggered |
|---------|---------------|
| Vehicle avoidance | `lbc_hazard_vehicle()` — heading + distance check |
| Pedestrian avoidance | `lbc_hazard_walker()` — on-sidewalk filter + distance ahead check |
| Traffic light | `input_data['traffic_light']['at_red_light']` flag |
| Stop sign | `input_data['stop_sign']['at_stop_sign']` flag |
| Negative speed | `forward_speed < -1.0` |

All hazards trigger **binary emergency stop**: `throttle=0, steer=0, brake=1`.

### 1.2 Vehicle Following (Roach RL Expert)

The RL expert does **not** use any explicit vehicle-following model (no IDM, no
kinematic model, no proportional braking). Instead, vehicle awareness is
**learned implicitly** through:

1. **BEV observation**: Vehicles are rendered on 4 time-separated channels in
   the BEV mask, giving the policy implicit motion information.
2. **Reward shaping**: The `ValeoAction` reward function
   (`carla_gym/core/task_actor/ego_vehicle/reward/valeo_action.py`) computes
   a **desired speed** based on detected hazards:

```python
# valeo_action.py — desired speed from vehicle hazard
hazard_vehicle_loc = lbc_hazard_vehicle(obs_vehicle, proximity_threshold=9.5)
if hazard_vehicle_loc is not None:
    dist_veh = max(0.0, np.linalg.norm(hazard_vehicle_loc[0:2]) - 8.0)
    desired_spd_veh = self._maxium_speed * np.clip(dist_veh, 0.0, 5.0) / 5.0
```

- `proximity_threshold = 9.5 m` for vehicle detection
- Effective braking distance: 8.0 m offset + 5.0 m linear ramp
- `_maxium_speed = 6.0 m/s` (~21.6 km/h)
- `r_speed = 1.0 - |ev_speed - desired_speed| / _maxium_speed`

This means the **reward penalises** deviations from the dynamically-computed
desired speed, teaching the RL agent proportional braking behaviour **emergently**
without hardcoded rules.

**The LBC rule-based autopilot** uses a simpler approach:

```python
# hazard_actor.py — lbc_hazard_vehicle()
def lbc_hazard_vehicle(obs_surrounding_vehicles, ev_speed=None, proximity_threshold=9.5):
    for i, is_valid in enumerate(obs_surrounding_vehicles['binary_mask']):
        sv_yaw = obs_surrounding_vehicles['rotation'][i][2]
        same_heading = abs(sv_yaw) <= 150  # heading filter
        sv_loc = obs_surrounding_vehicles['location'][i]
        with_distance_ahead = is_within_distance_ahead(sv_loc, proximity_threshold, up_angle_th=45)
        if same_heading and with_distance_ahead:
            return sv_loc
    return None
```

- **Binary**: Returns location or None → triggers full brake/no brake.
- `same_heading`: Other vehicle heading within ±150° of ego.
- `up_angle_th=45°`: Only considers vehicles within ±45° forward cone.

### 1.3 Traffic Light Handling

**RL Expert**: Traffic lights are encoded in the BEV observation as separate
channels for red, yellow, and green lights. The RL agent learns to stop at red
lights through the reward function:

```python
# valeo_action.py
light_state, light_loc, _ = TrafficLightHandler.get_light_state(
    self._ego_vehicle.vehicle, offset=self._tl_offset, dist_threshold=18.0)

if light_state in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow):
    dist_rl = max(0.0, np.linalg.norm(light_loc[0:2]) - 5.0)
    desired_spd_rl = self._maxium_speed * np.clip(dist_rl, 0.0, 5.0) / 5.0
```

- `dist_threshold = 18.0 m` for traffic light detection radius.
- `tl_offset = -0.8 * ego_extent.x` (offset behind ego centre for detection point).
- **5.0 m offset** + 5.0 m linear speed ramp to zero.

**TrafficLightHandler** (`carla_gym/utils/traffic_light.py`):
- Pre-processes all traffic lights at episode start via `reset()`.
- Computes stopline waypoints by advancing the trigger volume waypoint until
  it reaches the intersection.
- `get_light_state()` checks if the ego vehicle is on the same `road_id` and
  `lane_id` as any traffic light's stopline waypoints.

**Terminal condition**: Running a red light (`c_run_rl`) triggers episode
termination with penalty `terminal_reward -= ev_speed` (speed-dependent penalty).

**LBC autopilot**: Reads `input_data['traffic_light']['at_red_light']` directly
from the observation manager — binary flag → emergency stop.

### 1.4 Turn/Junction Handling

**RL Expert**: No explicit turn-speed reduction. The BEV contains route
information, and the agent learns turns through experience. The action space is
`[acc, steer]` where `acc ∈ [-1, 1]` (negative = brake).

**LBC autopilot**: Uses a `LocalPlanner` with PID controllers. The
`_hack_throttle` flag reduces throttle proportionally to steering:
```python
if self._hack_throttle:
    throttle *= max((1.0 - abs(steer)), 0.25)
```
This gives a minimum 25% throttle multiplier during sharp turns.

### 1.5 Lane Detection / In-Lane Check

**For reward/terminal computation** (applies to both RL and rule-based):

Lateral distance from route:
```python
# valeo_action.py and valeo.py
d_vec = ev_transform.location - wp_transform.location
np_wp_unit_right = np.array([-wp_unit_forward.y, wp_unit_forward.x])
lateral_distance = abs(np.dot(np_wp_unit_right, np_d_vec))
r_position = -1.0 * (lateral_distance / 2.0)
```

Terminal condition for lateral deviation (`valeo.py`):
- `_min_thresh_lat_dist = 3.5 m`.
- Dynamic threshold: if `lat_dist - last_lat_dist > 0.8`, threshold = `lat_dist + 0.5`.
- Episode terminates if `lat_dist > thresh_lat_dist`.

**For vehicle-in-lane filtering** (`hazard_actor.py`):
- `lbc_hazard_vehicle`: Uses heading filter (`|sv_yaw| <= 150°`) and
  distance-ahead cone (`up_angle_th=45°`). Does **not** use explicit
  `road_id/lane_id` matching.
- `behavior_hazard_vehicle`: More sophisticated — matches `road_id` and
  `lane_id` between ego and target, with `lane_offset` support.

### 1.6 Pedestrian Handling

**RL Expert**: Learned from reward. Desired speed computed as:
```python
hazard_ped_loc = lbc_hazard_walker(obs_pedestrian, proximity_threshold=9.5)
if hazard_ped_loc is not None:
    dist_ped = max(0.0, np.linalg.norm(hazard_ped_loc[0:2]) - 6.0)
    desired_spd_ped = self._maxium_speed * np.clip(dist_ped, 0.0, 5.0) / 5.0
```
- 6.0 m offset + 5.0 m ramp (vs 8.0 m for vehicles — more conservative clearance).

**LBC pedestrian detection** (`hazard_actor.py`):
```python
def lbc_hazard_walker(obs_surrounding_pedestrians, proximity_threshold=9.5):
    if int(obs_surrounding_pedestrians['on_sidewalk'][i]) == 1:
        continue  # skip pedestrians on sidewalk
    dist = np.linalg.norm(ped_loc)
    degree = 162 / (np.clip(dist, 1.5, 10.5) + 0.3)
    # dynamic angle — wider cone at close range, narrower at distance
```

- `on_sidewalk` filter: Ignores pedestrians on the sidewalk.
- **Adaptive angular threshold**: `162 / (distance + 0.3)` degrees:
  - At 1.5 m: ~90° cone
  - At 5.0 m: ~30° cone
  - At 10.0 m: ~15° cone

### 1.7 Key Parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| `_maxium_speed` (reward) | 6.0 m/s (~21.6 km/h) | `valeo_action.py` |
| Vehicle proximity | 9.5 m | `hazard_actor.py` |
| Vehicle dist offset | 8.0 m | `valeo_action.py` |
| Pedestrian dist offset | 6.0 m | `valeo_action.py` |
| Traffic light dist | 18.0 m, 5.0 m offset | `valeo_action.py` |
| Lateral dist threshold | 3.5 m | `valeo.py` |
| Vehicle stuck counter | 100 steps | `valeo.py` |
| PPO action space | `[-1,1]×[-1,1]` (acc, steer) | `rl_birdview_wrapper.py` |
| BEV resolution | Configurable | Config YAML |
| Eval timeout | 1200 sec | `valeo.py` |
| Speed queue size | 10 samples | `valeo.py` |

### 1.8 Collision / Episode Termination

The `Valeo` terminal handler (`carla_gym/core/task_actor/ego_vehicle/terminal/valeo.py`)
implements **6 termination conditions**:

| # | Condition | Variable | Details |
|---|-----------|----------|---------|
| 1 | **Vehicle stuck** | `c_vehicle_stuck` | Mean speed < 1.0 m/s for 100 steps AND no hazard ahead |
| 2 | **Lateral deviation** | `c_lat_dist` | > dynamic threshold (min 3.5 m) |
| 3 | **Red light violation** | `c_run_rl` | CARLA criteria detection |
| 4 | **Collision** | `c_collision` | CARLA collision sensor |
| 5 | **Stop sign violation** | `c_run_stop` | CARLA criteria detection |
| 6 | **Vehicle blocked** | `c_blocked` | CARLA criteria detection |

**Terminal reward** penalties:
- Any termination: `terminal_reward = -1.0`
- Red light / collision / stop sign: additional `terminal_reward -= ev_speed`
  (speed-proportional penalty — crashing faster is worse).

**Exploration suggestions** (unique RL feature):
- Stuck/blocked → suggest `('go', '')` for 100 steps.
- Lateral deviation → suggest `('', 'turn')`.
- Collision/red light → suggest `('stop', '')`.

**During data collection** (not RL training):
- `no_collision=True`: Early-stop episode on collision (error-free datasets).
- `no_run_rl=False`: Default doesn't early-stop on red-light violations.
- `max_time`: Maximum episode duration in seconds.
- Noise injection (`inject_noise=True`): Triangular noise on steering/throttle.

### 1.9 Unique Features vs TransFuser/LBC

| Feature | Roach | TransFuser | LBC |
|---------|-------|------------|-----|
| **Expert type** | RL-trained (PPO) | Rule-based | Rule-based |
| **Observation** | BEV semantic masks | Privileged CARLA APIs | Privileged CARLA APIs |
| **Vehicle awareness** | Learned from BEV + reward | Kinematic bicycle model + OBB | Distance + heading check |
| **Action distribution** | Beta distribution (continuous) | N/A (direct PID) | N/A (direct PID) |
| **Proportional braking** | Emergent from reward shaping | Proportional (hazard model) | Binary (stop/go) |
| **DAgger support** | Built-in coach/student framework | Not in autopilot | Not in autopilot |
| **Value function** | Outputs value estimate per step | N/A | N/A |
| **Feature distillation** | IL agent can learn Roach features | N/A | N/A |
| **Exploration guidance** | Terminal suggests (go/stop/turn) | N/A | N/A |
| **Multi-env training** | Gym wrapper with configurable maps | Single route | Single route |

### 1.10 Key File Paths

```
carla-roach/
├── agents/
│   ├── expert/
│   │   ├── lbc_roaming_agent.py          # Rule-based LBC autopilot
│   │   └── utils/
│   │       └── local_planner.py          # PID-based local planner
│   ├── rl_birdview/
│   │   ├── rl_birdview_agent.py          # RL expert agent interface
│   │   ├── models/
│   │   │   └── ppo_policy.py             # PPO policy network
│   │   └── utils/
│   │       └── rl_birdview_wrapper.py    # Gym wrapper (obs/action processing)
│   └── cilrs/                            # IL student agent (CILRS)
├── carla_gym/
│   ├── core/
│   │   ├── obs_manager/
│   │   │   └── birdview/                 # BEV rendering
│   │   └── task_actor/ego_vehicle/
│   │       ├── reward/valeo_action.py    # RL reward function
│   │       └── terminal/valeo.py         # Episode termination conditions
│   └── utils/
│       ├── hazard_actor.py               # Vehicle/pedestrian hazard detection
│       └── traffic_light.py              # Traffic light handler
├── config/
│   └── agent/
│       ├── ppo.yaml                      # RL expert config
│       └── roaming.yaml                  # Rule-based autopilot config
├── data_collect.py                       # Data collection script
├── train_rl.py                           # RL training script
└── train_il.py                           # IL (DAgger) training script
```

---

## 2. MILE

**Paper**: "Model-Based Imitation Learning for Urban Driving"
— Hu, Corrado, Griffiths, Murez, Gurau, Yeo, Kendall, Cipolla, Shotton (NeurIPS 2022)

**Repository**: <https://github.com/wayveai/mile>

**Key insight**: MILE does **not** implement its own expert/autopilot. It
**directly uses the Roach (PPO RL) expert** for data collection, inheriting
Roach's `carla_gym` wrapper verbatim. MILE's contribution is its world model
architecture and planning, not the data collection agent.

### 2.1 Expert/Autopilot Features

MILE's data collection reuses Roach's entire infrastructure:

| Component | Source | MILE Path |
|-----------|--------|-----------|
| Gym environment | Roach `carla_gym` | `carla_gym/` (adapted copy) |
| RL expert (Roach) | Roach checkpoint | `config/agent/ppo.yaml` → `agents.rl_birdview` |
| Observation wrapper | Roach | `mile/agents/rl_birdview/utils/rl_birdview_wrapper.py` |
| Reward function | Roach `valeo_action.py` | Same as Roach |
| Terminal conditions | Roach `valeo.py` | Same as Roach |
| BEV observation | Roach birdview manager | Same as Roach |

From `config/agent/mile.yaml`:
```yaml
mile:
  entry_point: agents.mile.mile_agent:MileAgent
  ckpt: ''
  online_deployment: false
  env_wrapper:
    entry_point: agents.mile.mile_wrapper:MileWrapper
```

**MILE credits Roach explicitly** in their README:
> "Thanks to the authors of End-to-End Urban Driving by Imitating a
> Reinforcement Learning Coach for providing a gym wrapper around CARLA
> making it easy to use, as well as an RL expert to collect data."

### 2.2 Vehicle Following

Identical to Roach RL expert. See [Section 1.2](#12-vehicle-following-roach-rl-expert).

### 2.3 Traffic Light Handling

Identical to Roach. See [Section 1.3](#13-traffic-light-handling).

### 2.4 Turn/Junction Handling

Identical to Roach. See [Section 1.4](#14-turnjunction-handling).

### 2.5 Lane Detection

Identical to Roach. See [Section 1.5](#15-lane-detection--in-lane-check).

### 2.6 Pedestrian Handling

Identical to Roach. See [Section 1.6](#16-pedestrian-handling).

### 2.7 Key Parameters

Same as Roach (Section 1.7). Additional MILE-specific parameters:

| Parameter | Value | Location |
|-----------|-------|----------|
| CARLA version | 0.9.11 | `environment.yml` |
| Model stride | Configurable (sec) | `cfg.DATASET.STRIDE_SEC` |
| Receptive field | Model-dependent | `trainer.model.receptive_field` |
| Input buffer | `(receptive_field - 1) * n_image_per_stride + 1` | `mile_agent.py` |

### 2.8 Collision / Episode Termination

Identical to Roach — uses `valeo.py` terminal conditions.

The `data_collect.py` script additionally:
- Calls `data_writer.close()` with `remove_final_steps` flag to discard
  frames around termination events.
- Validates episodes — retries if `valid=False` (e.g., CARLA crash).
- Logs per-episode statistics to Weights & Biases.

### 2.9 Unique Features vs TransFuser/LBC

| Feature | MILE | TransFuser | LBC |
|---------|------|------------|-----|
| **Expert source** | Roach RL expert | Own rule-based autopilot | Own rule-based autopilot |
| **World model** | Yes — latent dynamics model | No | No |
| **Imagination driving** | Can plan in latent space | No | No |
| **BEV prediction** | Predicts future BEV segmentation | No | No |
| **Driving policy** | Learned from world model outputs | Direct IL from camera | Direct IL from camera |
| **Data format** | H5 with BEV labels | H5/PNG per frame | LmDB |
| **Evaluation** | Online closed-loop (CARLA) | Online closed-loop | Online closed-loop |

### 2.10 Key File Paths

```
mile/
├── carla_gym/                                  # Adapted from Roach
│   ├── core/
│   │   ├── obs_manager/                        # BEV, vehicle, pedestrian observers
│   │   └── task_actor/ego_vehicle/
│   │       ├── reward/valeo_action.py          # Same reward as Roach
│   │       └── terminal/valeo.py               # Same terminal as Roach
│   └── utils/
│       ├── hazard_actor.py                     # Same hazard detection
│       └── traffic_light.py                    # Same TL handler
├── mile/
│   ├── agents/
│   │   ├── mile/
│   │   │   ├── mile_agent.py                   # MILE evaluation agent
│   │   │   └── mile_wrapper.py                 # Action processing
│   │   └── rl_birdview/
│   │       └── utils/rl_birdview_wrapper.py    # Roach obs/action wrapper
│   ├── trainer.py                              # WorldModelTrainer
│   └── configs/mile.yml                        # Model config
├── config/
│   └── agent/
│       ├── mile.yaml                           # MILE agent config
│       └── ppo.yaml                            # Roach RL config (for data collection)
├── data_collect.py                             # Data collection entry point
├── train.py                                    # Training entry point
└── evaluate.py                                 # Evaluation entry point
```

---

## 3. ThinkTwice

**Paper**: "Think Twice before Driving: Towards Scalable Decoders for
End-to-End Autonomous Driving" — Jia, Wu, Chen, Xie, He, Yan, Li (CVPR 2023)

**Repository**: <https://github.com/OpenDriveLab/ThinkTwice>

**Key insight**: ThinkTwice uses a **hybrid approach** for data collection:
a **Roach RL expert** provides the **driving actions**, while a **TransFuser-style
rule-based autopilot** provides **collision detection as a safety override**. The
data collection agent (`ROACHAgent`) runs the Roach policy forward, then applies
rule-based emergency braking on top.

### 3.1 Expert/Autopilot Features

The data collection agent (`leaderboard/team_code/roach_ap_agent_data_collection.py`,
class `ROACHAgent`) combines:

| Component | Source | Purpose |
|-----------|--------|---------|
| **Roach RL policy** | Pre-trained checkpoint (`roach/log/ckpt_11833344.pth`) | Primary driving control |
| **Collision detection** | Rule-based (`_is_vehicle_hazard`, `_is_walker_hazard`) | Emergency brake override |
| **BEV observation** | Roach's `ObsManager` | Input to RL policy |
| **Stop sign criteria** | `roach.criteria.run_stop_sign` | Stop sign monitoring |
| **Traffic light handler** | `roach.utils.traffic_light.TrafficLightHandler` | TL state tracking |

There is **also** a standalone rule-based autopilot (`leaderboard/team_code/auto_pilot.py`,
class `AutoPilot`) that can be used independently. This autopilot is nearly
identical to TransFuser/LBC's approach.

### 3.2 Vehicle Following

**Roach-based data collection** (`ROACHAgent.collision_detect()`):

The `_is_vehicle_hazard()` method uses a **collision prediction** approach nearly
identical to TransFuser's LBC autopilot:

```python
def _is_vehicle_hazard(self, vehicle_list):
    o1 = _orientation(self._ego_vehicle.get_transform().rotation.yaw)
    p1 = _numpy(self._ego_vehicle.get_location())
    s1 = max(10, 3.0 * np.linalg.norm(_numpy(self._ego_vehicle.get_velocity())))
    v1_hat = o1
    v1 = s1 * v1_hat

    for target_vehicle in vehicle_list:
        o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
        p2 = _numpy(target_vehicle.get_location())
        s2 = max(5.0, 2.0 * np.linalg.norm(_numpy(target_vehicle.get_velocity())))
        v2_hat = o2
        v2 = s2 * v2_hat

        p2_p1 = p2 - p1
        distance = np.linalg.norm(p2_p1)
        p2_p1_hat = p2_p1 / (distance + 1e-4)

        angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
        angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))

        angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
        angle_between_heading = min(angle_between_heading, 360.0 - angle_between_heading)

        if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
            continue
        elif angle_to_car > 30.0:
            continue
        elif distance > s1:
            continue
        return target_vehicle
```

**Parameters**:
- `s1 = max(10, 3.0 × ego_speed)` — dynamic detection range, minimum 10 m.
- `s2 = max(5.0, 2.0 × target_speed)` — target projection distance, minimum 5 m.
- `angle_to_car > 30°` → ignored (not ahead).
- `angle_between_heading > 60°` → ignored unless very close and directly ahead
  (`angle_to_car < 15°` AND `distance < s1`).
- **Binary outcome**: Returns target vehicle or None → full brake.

**Braking is overridden on top of Roach output**:
```python
should_brake = self.collision_detect()
only_ap_brake = True if (control.brake <= 0 and should_brake) else False
if should_brake:
    control.steer = control.steer * 0.5   # reduce steer by 50%
    control.throttle = 0.0
    control.brake = 1.0
```

**Standalone AutoPilot** (`auto_pilot.py`): Uses identical vehicle hazard
detection as above. Braking is also binary — any hazard → `brake=True`.

### 3.3 Traffic Light Handling

**ROACHAgent**: The Roach RL policy handles traffic lights implicitly through
BEV channels. The `TrafficLightHandler` is initialised but used primarily for
the BEV observation rendering, not for explicit rule-based braking.

**Standalone AutoPilot** (`auto_pilot.py`):
```python
def _is_light_red(self, lights_list):
    if self._vehicle.get_traffic_light_state() != carla.TrafficLightState.Green:
        affecting = self._vehicle.get_traffic_light()
        for light in self._traffic_lights:
            if light.id == affecting.id:
                return affecting
    return None
```

- Uses CARLA's built-in `get_traffic_light_state()` — checks if the ego
  vehicle's currently-affecting traffic light is not green.
- **Binary response**: Red/Yellow → brake, Green → proceed.
- No distance-based proportional braking.

### 3.4 Turn/Junction Handling

**Standalone AutoPilot** (`auto_pilot.py`):
```python
def _get_control(self, target, far_target, tick_data):
    angle_unnorm = self._get_angle_to(pos, theta, target)
    angle = angle_unnorm / 90

    angle_far_unnorm = self._get_angle_to(pos, theta, far_target)
    should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
    target_speed = 4.0 if should_slow else 7.0
```

- **Two speed modes**: 4.0 m/s (~14.4 km/h) for turns, 7.0 m/s (~25.2 km/h) straight.
- `should_slow` if: far waypoint angle > 45° OR near waypoint angle > 5°.
- PID controllers:
  - Turn: `K_P=1.25, K_I=0.75, K_D=0.3, n=40`
  - Speed: `K_P=5.0, K_I=0.5, K_D=1.0, n=40`
- Throttle clamped to `[0.0, 0.75]`, delta clamped to `[0.0, 0.25]`.

**ROACHAgent**: Roach RL policy handles turns implicitly. Steering noise added:
```python
control.steer = steer + 1e-2 * np.random.randn()  # small random noise
```

### 3.5 Lane Detection / In-Lane Check

**Vehicle hazard detection** (both agents): Uses heading + angle + distance
filtering (see Section 3.2). No explicit `road_id/lane_id` check — relies on
angular cone-based filtering.

### 3.6 Pedestrian Handling

**ROACHAgent** (`_is_walker_hazard`):
```python
def _is_walker_hazard(self, walkers_list):
    p1 = _numpy(self._ego_vehicle.get_location())
    v1 = 10.0 * _orientation(self._ego_vehicle.get_transform().rotation.yaw)

    for walker in walkers_list:
        v2_hat = _orientation(walker.get_transform().rotation.yaw)
        s2 = np.linalg.norm(_numpy(walker.get_velocity()))
        if s2 < 0.05:
            v2_hat *= s2   # nearly-stationary → shrink velocity vector
        p2 = -3.0 * v2_hat + _numpy(walker.get_location())
        v2 = 8.0 * v2_hat

        collides, collision_point = get_collision(p1, v1, p2, v2)
        if collides:
            return walker
```

- **Collision prediction** using linear trajectory intersection (`get_collision()`).
- Ego projection: `10.0 × orientation` (10 m lookahead).
- Pedestrian: `p2 = walker_loc - 3.0 × velocity_hat` (3 m backwards offset),
  `v2 = 8.0 × velocity_hat` (8 m projection).
- Stationary pedestrians (`speed < 0.05`): velocity scaled by speed
  (effectively makes them a point obstacle).
- **Binary**: Collision predicted → full brake.

**Standalone AutoPilot**: Identical implementation.

### 3.7 Key Parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| Ego detection range | `max(10, 3.0 × speed)` m | `roach_ap_agent_data_collection.py` |
| Target detection range | `max(5.0, 2.0 × speed)` m | `roach_ap_agent_data_collection.py` |
| Vehicle heading filter | 60° between headings | `_is_vehicle_hazard` |
| Vehicle angle filter | 30° cone ahead | `_is_vehicle_hazard` |
| Pedestrian ego projection | 10.0 m | `_is_walker_hazard` |
| Pedestrian velocity projection | 8.0 m | `_is_walker_hazard` |
| Pedestrian backwards offset | 3.0 m | `_is_walker_hazard` |
| Roach checkpoint | `roach/log/ckpt_11833344.pth` | `roach_ap_agent_data_collection.py` |
| Normal speed (AutoPilot) | 7.0 m/s | `auto_pilot.py` |
| Turn speed (AutoPilot) | 4.0 m/s | `auto_pilot.py` |
| Steering PID (AutoPilot) | K_P=1.25, K_I=0.75, K_D=0.3, n=40 | `auto_pilot.py` |
| Speed PID (AutoPilot) | K_P=5.0, K_I=0.5, K_D=1.0, n=40 | `auto_pilot.py` |
| Stop sign proximity | 30.0 m | `auto_pilot.py` |
| Stop speed threshold | 0.1 m/s | `auto_pilot.py` |
| Steer reduction on brake | × 0.5 | `roach_ap_agent_data_collection.py` |
| 3D BB detection distance | 50 m | `_get_3d_bbs` |

### 3.8 Collision / Episode Termination

**During data collection**, the `ROACHAgent` does **not** terminate on collision.
Instead:
- Collisions are detected each step by `collision_detect()` → emergency brake
  applied, episode continues.
- The `should_brake` flag is saved in measurements JSON alongside each frame,
  along with `only_ap_brake` (true if collision detection braked but Roach didn't).
- No frame is discarded due to collision — all data is saved.

**Standalone AutoPilot**: Does not self-terminate; the leaderboard evaluator
handles episode management.

### 3.9 Unique Features vs TransFuser/LBC

| Feature | ThinkTwice | TransFuser | LBC |
|---------|------------|------------|-----|
| **Expert combination** | Roach RL + rule-based override | Pure rule-based | Pure rule-based |
| **Sensors collected** | 4 cameras (front/left/right/back) + LiDAR + depth + segmentation | 1-4 cameras + LiDAR | 1 camera |
| **BEV segmentation labels** | Saved from Roach obs manager (road/route/lane/vehicle/walker/TL channels) | Not saved by autopilot | Not available |
| **3D bounding boxes** | Saved per frame with LiDAR point counts | Not saved | Not saved |
| **Dual-frame LiDAR** | Previous + current LiDAR merged via transform | Single sweep | N/A |
| **`should_brake` annotation** | Explicit label per frame from rule-based override | Not explicit | Not explicit |
| **`only_ap_brake`** | Tracks when rule-based overrides RL (disagreement label) | N/A | N/A |
| **Weather randomisation** | Changes weather every 10 steps during AutoPilot collection | Fixed per episode | Fixed per episode |
| **Camera FOV** | 150° (very wide) | 100° typically | 100° |
| **Camera resolution** | 1600×900 | Varies (400×300 typical for autopilot) | 256×256 |
| **Stop sign handling** | Full state-machine (affected → completed → reset) | OBB intersection | Binary |
| **Value/feature labels** | Saves Roach's policy values, mus, sigmas, features | N/A | N/A |
| **Decoder architecture** | Multi-scale BEV decoder (think-twice refinement) | Transformer fusion | Direct waypoint |

### 3.10 Key File Paths

```
ThinkTwice/
├── leaderboard/
│   ├── team_code/
│   │   ├── roach_ap_agent_data_collection.py  # Primary: Roach + rule-based override
│   │   ├── auto_pilot.py                      # Standalone rule-based autopilot
│   │   ├── base_agent.py                      # Base agent interface
│   │   ├── map_agent.py                       # Map-based agent base class
│   │   ├── pid_controller.py                  # PID controller implementation
│   │   ├── planner.py                         # RoutePlanner
│   │   └── thinktwice_agent.py                # ThinkTwice model inference agent
│   ├── scripts/
│   │   └── evaluation_town05long.sh           # Evaluation launcher
│   └── leaderboard/
│       └── leaderboard_evaluator.py           # Evaluation harness
├── roach/
│   ├── rl_birdview_agent.py                   # Roach RL agent adapter
│   ├── config/                                # Roach configuration
│   ├── criteria/
│   │   └── run_stop_sign.py                   # Stop sign criteria
│   ├── models/                                # Roach PPO models
│   ├── obs_manager/
│   │   └── birdview/chauffeurnet.py           # BEV observation rendering
│   └── utils/
│       ├── traffic_light.py                   # Traffic light handler
│       └── expert_noiser.py                   # Noise injection
├── open_loop_training/
│   ├── code/
│   │   ├── datasets/                          # Data loading & preprocessing
│   │   └── model_code/
│   │       ├── backbones/                     # Encoder modules
│   │       └── dense_heads/                   # Decoder & loss modules
│   └── train.py                               # Training entry point
└── dataset/tools/                             # Data preparation scripts
```

---

## 4. Cross-Comparison Table

| Feature | Roach | MILE | ThinkTwice | TransFuser | LBC |
|---------|-------|------|------------|------------|-----|
| **Expert type** | PPO RL agent | Roach (reused) | Roach + rule-based safety | Rule-based | Rule-based |
| **Vehicle following** | Learned (reward-shaped) | Learned (Roach) | Collision lines + binary brake | Kinematic bicycle + OBB | Distance + heading |
| **Traffic lights** | Learned from BEV + reward | Learned (Roach) | Roach + CARLA API check | OBB trigger volume intersection | OBB / CARLA API |
| **Turn handling** | Learned from BEV | Learned (Roach) | Angle-based `should_slow` | Junction flag → reduced speed | Angle-based |
| **Lane check** | BEV + lateral dist reward | BEV (Roach) | Heading/angle filter | road_id/lane_id | Heading filter |
| **Pedestrians** | Learned + sidewalk filter | Learned (Roach) | Line intersection prediction | Linear extrap + OBB | Distance + heading |
| **Braking model** | Continuous (emergent) | Continuous (Roach) | Binary override on RL output | Proportional (hazard-distance) | Binary |
| **Speed targets** | 6.0 m/s (reward) | 6.0 m/s (Roach) | 7.0 / 4.0 m/s (AP) | 4.0 / 7.0 m/s | 4.0 / 7.0 m/s |
| **Collision handling** | Terminal + speed penalty | Terminal (Roach) | Emergency brake, no termination | Frame discard possible | Episode restart |
| **Stop signs** | Learned + terminal penalty | Learned (Roach) | State machine | OBB intersection | Simplified |
| **Noise injection** | Triangular (configurable) | Triangular (Roach) | Gaussian (`1e-2 × randn`) | Gaussian | Lateral noise |
| **BEV encoding** | Core observation | Core observation | For RL policy + saved labels | Not used by autopilot | Not used |

---

## 5. RL Expert vs Rule-Based Expert

### Why Roach Uses RL Instead of Rules

The fundamental difference between Roach and traditional experts (TransFuser/LBC
autopilots) is that Roach replaces **hand-crafted driving rules** with a
**policy learned through reinforcement learning**.

### Architecture of the RL Expert

```
Input:
  ├── BEV Semantic Masks (192×192, ~15 channels)
  │    ├── Road mask
  │    ├── Route mask
  │    ├── Lane markings (broken lanes = 0.5)
  │    ├── Vehicles × 4 timesteps
  │    ├── Walkers × 4 timesteps
  │    └── Traffic lights (red/yellow/green) × 4 timesteps
  └── State Vector
       ├── throttle, steer, brake, gear/5.0
       └── vel_x, vel_y

CNN Feature Extractor → features (256-dim)
  ├── Policy Head (MLP 256→256→256, ReLU) → μ, σ → Beta distribution → [acc, steer]
  └── Value Head (MLP 256→256→256→1, ReLU) → V(s)
```

- **Action space**: `[-1, 1] × [-1, 1]` where `acc ∈ [-1,1]` (negative = brake),
  `steer ∈ [-1,1]`.
- **Distribution**: Beta distribution (bounded support, better for bounded action spaces).
- **PPO training**: Standard Proximal Policy Optimisation.

### How RL Data Collection Differs

| Aspect | Rule-Based Expert | RL Expert (Roach) |
|--------|-------------------|-------------------|
| **Decision process** | If-else rules on privileged state | Neural network forward pass on BEV |
| **Tuning** | Manual parameter tuning | Reward function design + training |
| **Generalization** | Brittle at edge cases | Better generalisation to unseen scenarios |
| **Naturalness** | Jerky (binary brake/go) | Smoother (continuous action distribution) |
| **Speed** | Real-time | Real-time (single forward pass) |
| **Debugging** | Transparent (inspect rules) | Opaque (inspect reward/value) |
| **Failure modes** | Known (gaps in rules) | Unknown (reward hacking, distribution shift) |
| **Supervision quality** | Limited by rule quality | Limited by reward quality + training |

### Reward Function Components

The `ValeoAction` reward (`valeo_action.py`) has 4 components per step:

$$r_\text{total} = r_\text{speed} + r_\text{position} + r_\text{rotation} + r_\text{action} + r_\text{terminal}$$

Where:

| Component | Formula | Range |
|-----------|---------|-------|
| $r_\text{speed}$ | $1 - \|v_\text{ego} - v_\text{desired}\| / v_\text{max}$ | $[-\infty, 1]$ |
| $r_\text{position}$ | $-d_\text{lateral} / 2$ | $(-\infty, 0]$ |
| $r_\text{rotation}$ | $-\|\theta_\text{ego} - \theta_\text{route}\|$ (radians) | $[-\pi, 0]$ |
| $r_\text{action}$ | $-0.1$ if $\|\Delta\text{steer}\| > 0.01$, else $0$ | $\{-0.1, 0\}$ |
| $r_\text{terminal}$ | $-1$ on any termination; additional $-v_\text{ego}$ for violations | $[-\infty, 0]$ |

The **desired speed** ($v_\text{desired}$) is the minimum of speed targets from
all hazard sources (vehicles, pedestrians, traffic lights, stop signs), each
computed as a **linear ramp**:

$$v_\text{desired,hazard} = v_\text{max} \times \frac{\text{clip}(d_\text{hazard} - d_\text{offset}, 0, 5)}{5}$$

### Benefits for Imitation Learning

When the RL expert produces data for an IL student:

1. **Smoother trajectories**: The Beta distribution produces continuous actions,
   avoiding the jerky stop/go of rule-based experts.
2. **Value labels**: Each frame comes with a value estimate $V(s)$ that can be
   used as an additional supervision signal (value distillation loss $L_V$).
3. **Feature labels**: The intermediate features can be distilled to the student
   (feature imitation loss $L_F$).
4. **Action distribution labels**: The $(\mu, \sigma)$ parameters of the Beta
   distribution can supervise the student's action distribution (KL loss).
5. **DAgger with RL coach**: For on-policy data collection, the student drives
   while the RL expert labels each frame with what *it* would have done,
   providing richer supervision than rule-based experts.

### Comparison Summary

```
Rule-Based Expert:                    RL Expert (Roach):
┌─────────────────────┐              ┌─────────────────────┐
│ Privileged CARLA    │              │ BEV Semantic Masks  │
│ APIs (groundtruth   │              │ (rendered from CARLA│
│ positions, states)  │              │ groundtruth)        │
│         │           │              │         │           │
│    ┌────▼────┐      │              │    ┌────▼────┐      │
│    │ if-else │      │              │    │  CNN +  │      │
│    │  rules  │      │              │    │   MLP   │      │
│    └────┬────┘      │              │    └────┬────┘      │
│         │           │              │         │           │
│    throttle/steer/  │              │   acc, steer (μ,σ)  │
│    brake (discrete) │              │   value, features   │
└─────────────────────┘              └─────────────────────┘

  Output: action only                Output: action + value +
                                            features + μ,σ
```

---

## References

1. Zhang, Z., Liniger, A., Dai, D., Yu, F., & Van Gool, L. (2021).
   *End-to-End Urban Driving by Imitating a Reinforcement Learning Coach*.
   ICCV 2021. <https://arxiv.org/abs/2108.08265>

2. Hu, A., Corrado, G., Griffiths, N., Murez, Z., Gurau, C., Yeo, H.,
   Kendall, A., Cipolla, R., & Shotton, J. (2022). *Model-Based Imitation
   Learning for Urban Driving*. NeurIPS 2022. <https://arxiv.org/abs/2210.07729>

3. Jia, X., Wu, P., Chen, L., Xie, J., He, C., Yan, J., & Li, H. (2023).
   *Think Twice before Driving: Towards Scalable Decoders for End-to-End
   Autonomous Driving*. CVPR 2023. <https://arxiv.org/abs/2305.06242>
