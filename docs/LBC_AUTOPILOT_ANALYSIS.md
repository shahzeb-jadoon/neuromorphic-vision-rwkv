# Learning by Cheating (LBC) Expert Autopilot — Detailed Analysis

**Source**: Dian Chen, Brady Zhou, Vladlen Koltun, Philipp Krähenbühl,
"Learning by Cheating" (CoRL 2019), UT Austin / Intel Labs

**Paper**: <https://arxiv.org/abs/1912.12294>

**Repository**: <https://github.com/dotchen/LearningByCheating> (branch `release-0.9.6`)

**CARLA version**: 0.9.6

**Related work**: World on Rails (Chen et al., ICCV 2021) — covered in §11.

---

## 1. Expert/Autopilot Features — Complete List

| # | Feature | How Triggered |
|---|---------|---------------|
| 1 | **Route / lane following** | Bearing angle to next route waypoint → PID steering (always active) |
| 2 | **Traffic light handling** | Road/lane ID match + distance-ahead check → emergency stop |
| 3 | **Vehicle collision avoidance** | Yaw-difference + forward-cone distance check → emergency stop |
| 4 | **Pedestrian / walker avoidance** | Dynamic-angle forward-cone + sidewalk filter → emergency stop |
| 5 | **Turn / junction speed reduction** | Command ≠ STRAIGHT/LANEFOLLOW → 0.75× target speed |
| 6 | **Steering-coupled throttle reduction** | `throttle *= max(1.0 - abs(steer), 0.25)` in data collector |
| 7 | **Steering noise injection** | Optional uniform noise `[-0.25, 0.25]` for data augmentation |

### What It Does **Not** Have (comparison with TransFuser autopilot)

| Feature | LBC | TransFuser |
|---------|-----|------------|
| Stop sign handling | ✗ | ✓ (OBB intersection) |
| Stuck vehicle detection | ✗ | ✓ (30-frame buffer) |
| Slope / pitch handling | ✗ | ✓ (extra throttle) |
| Safety bounding box (Bremsweg) | ✗ | ✓ (emergency braking box) |
| Forward kinematic extrapolation | ✗ | ✓ (bicycle model) |
| GPS denoising | ✗ | ✓ (100-frame rolling avg) |
| Waypoint future-state prediction | ✗ | ✓ (for training labels) |
| Proportional braking | ✗ (binary) | ✓ (distance-proportional) |

---

## 2. Code Architecture & File Map

```
LearningByCheating/  (branch release-0.9.6)
├── data_collector.py                         # NoisyAgent wrapper, episode loop
├── benchmark_agent.py                        # Benchmarking entry point
├── benchmark/
│   ├── __init__.py                           # Suite registration (traffic density configs)
│   └── goal_suite.py                         # PointGoalSuite — episode termination logic
├── bird_view/
│   ├── models/
│   │   ├── roaming.py                        # RoamingAgentMine — THE autopilot
│   │   └── controller.py                     # PIDController class
│   └── utils/
│       ├── carla_utils.py                    # CarlaWrapper, TrafficTracker, PedestrianTracker
│       └── map_utils.py                      # BEV rendering (road, lane, traffic, vehicle, ped)
└── PythonAPI/agents/
    ├── navigation/
    │   ├── agent.py                          # Agent base — _is_vehicle_hazard, _is_light_red, etc.
    │   ├── controller.py                     # VehiclePIDController (lateral + longitudinal)
    │   └── local_planner.py                  # LocalPlannerNew, LocalPlannerOld, RoadOption
    └── tools/
        └── misc.py                           # is_within_distance_ahead, compute_yaw_difference
```

### Class Hierarchy

```
Agent                              (PythonAPI/agents/navigation/agent.py)
  └── RoamingAgentMine             (bird_view/models/roaming.py)
        └── NoisyAgent             (data_collector.py)  — used for data collection
```

---

## 3. Implementation Details — Per Feature

### 3.1 Route / Lane Following

**File**: `bird_view/models/roaming.py` → `RoamingAgentMine.run_step()`
**Planner**: `LocalPlannerNew` (`PythonAPI/agents/navigation/local_planner.py`)

#### Route Planning (LocalPlannerNew)
- Uses `GlobalRoutePlanner` + `GlobalRoutePlannerDAO` to trace a route from start to target.
- Route stored as a deque of `(waypoint, RoadOption)` tuples.
- **Resolution**: configurable (default=15 for `LocalPlannerNew`; 1 for `NoisyAgent`).
- **Max skip**: 20 waypoints to prevent misplanning when route spans both lanes.

#### Waypoint Consumption
- Two thresholds control when waypoints are consumed:
  - `threshold_before` (default 2.5; NoisyAgent uses 7.5): distance threshold
    when checkpoint is LANEFOLLOW and next command is different.
  - `threshold_after` (default 5.0; NoisyAgent uses 5.0): distance threshold
    for all other transitions.
- Waypoints within threshold distance are popped from the queue.

#### Steering Computation
```python
# In RoamingAgentMine.run_step():
# 1. Get vehicle forward vector as rotation matrix
ox = self._vehicle.get_transform().get_forward_vector().x
oy = self._vehicle.get_transform().get_forward_vector().y
rot = np.array([[ox, oy], [-oy, ox]])

# 2. Transform target to vehicle-relative coordinates
target = self._local_planner.target[0].transform.location
diff = rot.dot(target_array - pos_array)

# 3. Compute angle between [diff_x, diff_y, 0] and [1, 0, 0]
theta = np.arccos(np.dot(u, v) / np.linalg.norm(u))
theta = theta if np.cross(u, v)[2] < 0 else -theta

# 4. PID on angular error
steer = self.turn_control.step(theta)
```

- Steering is clipped to **[-1.0, 1.0]**.
- This is a **pure pursuit**-style approach: aim at the next waypoint.

### 3.2 PID Controllers

**File**: `bird_view/models/controller.py` → `PIDController`

```python
class PIDController:
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, fps=10, n=30):
        self._dt = 1.0 / fps
        self._window = deque(maxlen=n)     # 30-sample history

    def step(self, error):
        integral = sum(self._window) * self._dt
        derivative = (self._window[-1] - self._window[-2]) / self._dt
        return K_P * error + K_I * integral + K_D * derivative
```

#### PID Gain Values

| Controller | Context | K_P | K_I | K_D | fps | Notes |
|------------|---------|-----|-----|-----|-----|-------|
| **Turn PID** (TURNING_PID) | `RoamingAgentMine` default | 1.5 | 0.5 | 0.0 | 10 | Steering control |
| **Speed PID** | `RoamingAgentMine` default | 1.0 | 0.0 | 0.0 | 10 | P-only speed controller |
| **Turn PID** | `NoisyAgent` override | 0.75 | 1.0/20=0.05 | 0.0 | 10 | Softer steering for data collection |
| **Speed PID** | `NoisyAgent` override | 0.5 | 0.5/20=0.025 | 0.1 | 10 | Full PID for smoother speed |

The `VehiclePIDController` in `PythonAPI/agents/navigation/controller.py` (from CARLA)
is **not** directly used by the LBC autopilot — `RoamingAgentMine` uses its own custom
PID logic. The CARLA `VehiclePIDController` is only used by CARLA's standard `LocalPlanner`.

### 3.3 Target Speed and Throttle

```python
# In RoamingAgentMine.run_step():
target_speed = 6.0                          # m/s ≈ 21.6 km/h

# Reduce speed for turns (LEFT=1, RIGHT=2 are non-straight/follow commands)
if int(self._local_planner.target[1]) not in [3, 4]:  # not STRAIGHT or LANEFOLLOW
    target_speed *= 0.75                    # → 4.5 m/s ≈ 16.2 km/h

delta = target_speed - speed
throttle = self.speed_control.step(delta)

control.throttle = np.clip(throttle, 0.0, 1.0)
control.brake = 0.0                         # NO brake unless emergency stop
```

In the `NoisyAgent` data collector, an additional throttle reduction is applied:
```python
real_control.throttle *= max((1.0 - abs(real_control.steer)), 0.25)
```
This ensures throttle is reduced during sharp turns (minimum 25% of computed value).

---

## 4. Vehicle Following / Collision Avoidance

**File**: `PythonAPI/agents/navigation/agent.py` → `Agent._is_vehicle_hazard()`

### Detection Method
1. **Skip ego**: `if target_vehicle.id == self._vehicle.id: continue`
2. **Yaw alignment check**: `compute_yaw_difference(ego_yaw, target_yaw) <= 150°`
   - Ensures the target vehicle is going in roughly the same direction
   - This is a **very loose** filter (allows vehicles up to 150° from ego heading)
3. **Forward cone check**: `is_within_distance_ahead(loc, ego_loc, ego_yaw, 9.5, degree=45)`
   - Target must be within **9.5 meters** ahead
   - Within a **45° half-angle cone** from ego heading

### Lane Corridor
- **Lane checking is commented out** in the code:
  ```python
  # if not ego_vehicle_waypoint.is_intersection and
  #     target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
  #     continue
  ```
- Result: **any vehicle** within the 9.5m / 45° cone is treated as a hazard,
  regardless of lane. This is a deliberate simplification.

### Braking Strategy
- **Binary**: hazard detected → `emergency_stop()` (full brake=1.0, throttle=0.0, steer=0.0)
- **No proportional braking**: no distance-dependent deceleration
- **No following distance**: either full stop or full speed
- No speed matching with the lead vehicle

```python
def emergency_stop(self):
    control = carla.VehicleControl()
    control.steer = 0.0
    control.throttle = 0.0
    control.brake = 1.0
    control.hand_brake = False
    return control
```

### Limitations
- No bicycle model / forward extrapolation (unlike TransFuser)
- No OBB (oriented bounding box) collision prediction
- Binary response means jerky following behavior
- No consideration of relative velocity

---

## 5. Traffic Light Handling

**File**: `PythonAPI/agents/navigation/agent.py` → `Agent._is_light_red()`

### Routing by Map
```python
def _is_light_red(self, lights_list):
    if self._map.name == 'Town01' or self._map.name == 'Town02':
        return self._is_light_red_europe_style(lights_list)
    else:
        return self._is_light_red_us_style(lights_list)
```

### Europe-Style (Town01, Town02)
1. Get ego vehicle's waypoint (`road_id`, `lane_id`).
2. For each traffic light:
   - Get light's waypoint → filter by **same `road_id` AND `lane_id`**.
   - Must be within `_proximity_threshold=9.5m` ahead in a **60° cone**.
   - Must be `TrafficLightState.Red`.
3. If matched → return `(True, traffic_light)`.

### US-Style (Other Towns)
1. If ego is **already in intersection**: return `(False, None)` — too late to stop.
2. If target waypoint **is an intersection**:
   - Find nearest traffic light within **80m** and **25° angle**.
   - Remember last light (sticky reference).
   - If last light is Red → trigger.
3. Otherwise → no light detected.

### Response
- Emergency stop: `brake=1.0, throttle=0.0, steer=0.0`

### Traffic Tracker (Separate from Autopilot)
In `CarlaWrapper` (`carla_utils.py`), a `TrafficTracker` monitors whether the ego
ran a red light using **line-line intersection** between the ego's movement vector
and a perpendicular line through the traffic light (width = 5.0m).

---

## 6. Turn / Junction Handling

- **Detection**: Based on the navigation command from `LocalPlannerNew`:
  - `RoadOption.LEFT = 1`
  - `RoadOption.RIGHT = 2`
  - `RoadOption.STRAIGHT = 3`
  - `RoadOption.LANEFOLLOW = 4`

- **Speed reduction**:
  ```python
  if int(self._local_planner.target[1]) not in [3, 4]:
      target_speed *= 0.75  # 6.0 → 4.5 m/s
  ```

- **No explicit junction detection** (unlike TransFuser which checks `is_junction`).
- **No steering adjustment** beyond the PID aiming at the next waypoint.
- The NoisyAgent additionally reduces throttle proportional to steering magnitude.

---

## 7. Pedestrian / Walker Handling

**File**: `PythonAPI/agents/navigation/agent.py` → `Agent._is_walker_hazard()`

### Detection
```python
def _is_walker_hazard(self, walkers_list):
    for walker in walkers_list:
        loc = walker.get_location()
        dist = loc.distance(ego_vehicle_location)

        # 1. Dynamic cone angle — wider for closer pedestrians
        degree = 162 / (np.clip(dist, 1.5, 10.5) + 0.3)

        # 2. Skip pedestrians on sidewalk
        if self._is_point_on_sidewalk(loc):
            continue

        # 3. Forward cone check
        if is_within_distance_ahead(loc, ego_loc, ego_yaw,
                                    self._proximity_threshold, degree=degree):
            return (True, walker)
```

### Dynamic Cone Angle

| Distance (m) | Cone Half-Angle |
|--------------|-----------------|
| 1.5 | 162 / 1.8 = **90.0°** |
| 3.0 | 162 / 3.3 = **49.1°** |
| 5.0 | 162 / 5.3 = **30.6°** |
| 7.0 | 162 / 7.3 = **22.2°** |
| 9.5 | 162 / 9.8 = **16.5°** |
| 10.5 | 162 / 10.8 = **15.0°** |

This creates a **wide field of awareness for close pedestrians** (nearly 180°
for pedestrians at 1.5m) and a **narrow forward focus for distant ones**.

### Sidewalk Filter
- Uses a pre-rendered road map image (`Town01.png` / `Town02.png`).
- Converts walker world location to pixel coordinates.
- If pixel value == 0 (off-road / sidewalk) → skip walker.
- `PIXELS_PER_METER = 5`, with hard-coded world offsets per town.

### Response
- Emergency stop: `brake=1.0, throttle=0.0, steer=0.0`
- No evasive steering or speed proportional to distance.

---

## 8. Key Parameters — Complete Reference

### Distances and Speeds

| Parameter | Value | Location |
|-----------|-------|----------|
| `_proximity_threshold` | **9.5 m** | `roaming.py:RoamingAgentMine.__init__()` |
| Target speed | **6.0 m/s** (21.6 km/h) | `roaming.py:RoamingAgentMine.run_step()` |
| Turn speed multiplier | **0.75×** | `roaming.py:RoamingAgentMine.run_step()` |
| Turn speed | **4.5 m/s** (16.2 km/h) | Computed: 6.0 × 0.75 |
| Success distance | **5.0 m** | `goal_suite.py:PointGoalSuite.__init__()` |
| Max ticks per episode | **10,000** | `goal_suite.py:PointGoalSuite.is_failure()` |

### PID Gains

| Controller | K_P | K_I | K_D | fps | Window |
|------------|-----|-----|-----|-----|--------|
| Turn (RoamingAgentMine) | 1.5 | 0.5 | 0.0 | 10 | 30 |
| Speed (RoamingAgentMine) | 1.0 | 0.0 | 0.0 | 10 | 30 |
| Turn (NoisyAgent) | 0.75 | 0.05 | 0.0 | 10 | 30 |
| Speed (NoisyAgent) | 0.5 | 0.025 | 0.1 | 10 | 30 |

### Hazard Detection Cones

| Hazard Type | Max Distance | Cone Angle |
|-------------|-------------|------------|
| Vehicle | 9.5 m | 45° |
| Traffic light (Europe) | 9.5 m | 60° |
| Traffic light (US) | 80.0 m | 25° |
| Pedestrian | 9.5 m | 15°–90° (dynamic) |

### Collision Detection

| Parameter | Value | Location |
|-----------|-------|----------|
| `col_threshold` | **400** (impulse norm) | `carla_utils.py:CarlaWrapper.__init__()` |
| `fixed_delta_seconds` | **0.1 s** (10 FPS) | `carla_utils.py:set_sync_mode()` |

### Route Planner (LocalPlannerNew)

| Parameter | RoamingAgentMine | NoisyAgent | PointGoalSuite |
|-----------|-----------------|------------|----------------|
| Resolution | — | 1 | 2.5 |
| `threshold_before` | — | 7.5 | 9.0 |
| `threshold_after` | — | 5.0 | 1.5 |
| `_max_skip` | 20 | 20 | 20 |

### Data Collection (NoisyAgent)

| Parameter | Default | Location |
|-----------|---------|----------|
| `n_vehicles` | 100 | `data_collector.py` CLI |
| `n_pedestrians` | 250 | `data_collector.py` CLI |
| `n_episodes` | 50 | `data_collector.py` CLI |
| `frames_per_episode` | 4000 | `data_collector.py` CLI |
| `frame_skip` | 1 | `data_collector.py` CLI |
| Noise state duration | 100 frames (drive only) | `NoisyAgent.__init__()` |
| Noise steering range | **[-0.25, 0.25]** uniform | `NoisyAgent.__init__()` |

---

## 9. Data Collection & Episode Termination

### Data Collection Loop (`data_collector.py`)

```python
def get_episode(env, params):
    # Select random start/target pose pair
    start, target = env.pose_tasks[np.random.randint(len(env.pose_tasks))]

    # Initialize environment
    env.init(weather=random_weather, start=start, target=target,
             n_pedestrians=params.n_pedestrians, n_vehicles=params.n_vehicles)

    agent = NoisyAgent(env)
    agent.set_route(env._start_pose.location, env._target_pose.location)

    # Main loop — runs until any termination condition
    while len(data) < params.frames_per_episode \
          and not env.is_success() \
          and not env.collided:
        # frame_skip inner loop
        for _ in range(params.frame_skip):
            env.tick()
            observations = env.get_observations()
            control, command, status, real_control = agent.run_step(observations)
            env.apply_control(control)

        processed = cu.process(observations)
        data.append(processed)

    # Quality gate: discard short or incomplete episodes
    if (not env.is_success() and not env.collided) or len(data) < 500:
        return None

    return data
```

### Collision Handling
- **Collision sensor** attached to ego vehicle (`CarlaWrapper._setup_sensors()`).
- Only registered if impulse norm > `col_threshold=400`.
- On collision: `self.collided = True` → episode loop exits.
- **Episode is NOT automatically discarded on collision** — the `get_episode()`
  function keeps it if `len(data) >= 500`.
- In NoCrash benchmarks: `col_is_failure=True` → collision = benchmark failure.

### Episode Termination Conditions

| Condition | Data Collection | Benchmarking |
|-----------|----------------|--------------|
| Reached target (≤5m) | ✓ (success) | ✓ (success) |
| Collision (impulse > 400) | ✓ (exit loop) | ✓ if `col_is_failure` |
| Max frames reached | ✓ (4000 default) | ✓ (10,000 ticks) |
| Timeout | ✗ | ✓ (distance-based) |
| Episode too short (<500) | Discarded + retry | N/A |

### NoisyAgent Noise Injection

The `NoisyAgent` wraps `RoamingAgentMine` and can inject steering noise:
```python
class NoisyAgent(RoamingAgentMine):
    def __init__(self, env, noise=None):
        # State machine: 'drive' → 'drive' (noise is disabled by default!)
        self.params = {'drive': (100, 'drive')}
        self.noise_func = noise if noise else lambda: np.random.uniform(-0.25, 0.25)
```
Note: In the released code, noise is effectively **disabled** — the state machine
stays in `'drive'` mode permanently. The `'noise'` state (10-frame random steering)
is commented out.

### What Gets Saved Per Frame

```python
def process(observations):
    result = {
        'rgb':          observations['rgb'],          # (160, 384, 3) uint8
        'birdview':     observations['birdview'],     # (H, W, 7) uint8
        'collided':     observations['collided'],     # bool
        'control':      [steer, throttle, brake],     # float32
        'measurements': [position(3), orientation(2), velocity(3),
                         acceleration(3), command(1), steer, throttle,
                         brake, manual_gear_shift, gear]   # float32
    }
```

### Bird's-Eye View Channels
```python
# From carla_utils.py:visualize_birdview()
# Channel 0: road        → gray   (102, 102, 102)
# Channel 1: lane        → yellow (253, 253, 17)
# Channel 2: red light   → red    (204, 6, 5)
# Channel 3: yellow light→ orange (250, 210, 1)
# Channel 4: green light → green  (39, 232, 51)
# Channel 5: vehicle     → blue   (0, 0, 142)
# Channel 6: pedestrian  → crimson(220, 20, 60)
```

---

## 10. Comparison: LBC vs TransFuser Autopilot

| Aspect | LBC (2019) | TransFuser (2022) |
|--------|-----------|-------------------|
| **CARLA version** | 0.9.6 | 0.9.10+ |
| **Steering** | Bearing angle to next waypoint + PID | Same, with `angle_search_range` |
| **Speed control** | Simple PID (P-only default) | Separate longitudinal PID |
| **Target speed** | 6.0 m/s fixed | 4.0 m/s (configurable) |
| **Vehicle detection** | Yaw diff + forward cone (9.5m, 45°) | Bicycle model extrapolation + OBB |
| **Pedestrian detection** | Dynamic cone + sidewalk filter | Linear extrapolation + OBB |
| **Traffic lights** | Road/lane ID + cone check | OBB trigger volume intersection |
| **Stop signs** | ✗ Not handled | ✓ OBB intersection |
| **Braking** | Binary emergency stop only | Proportional + emergency stop |
| **Stuck detection** | ✗ Not handled | ✓ 30-frame buffer analysis |
| **GPS noise** | ✗ Not handled | ✓ 100-frame rolling average |
| **Slope handling** | ✗ Not handled | ✓ Extra throttle on pitch |
| **Safety box** | ✗ Not handled | ✓ Bremsweg emergency box |
| **Data noise** | Uniform steering noise (disabled) | Gaussian steering noise |
| **Collision handling** | Continue if <500 frames, else keep | Episode terminated + filter |
| **Complexity** | ~200 lines total | ~800+ lines |

---

## 11. World on Rails Expert (Chen et al., ICCV 2021)

**Repository**: <https://github.com/dotchen/WorldOnRails> (branch `release`)

### Fundamental Difference from LBC

World on Rails uses a **completely different expert paradigm**. Instead of a
rule-based autopilot with PID controllers, it uses:

1. **Bellman Q-value table**: Pre-computes action-values over a discretized
   action space using a factorized world model.
2. **QCollector** (`autoagents/collector_agents/q_collector.py`): Samples
   actions from the Q-value distribution with temperature-based exploration.
3. **Ego dynamics model**: A learned vehicle dynamics model (`ego_model.th`)
   predicts future states given actions.

### Key Architecture

```python
# From q_collector.py:
action_values, _ = BellmanUpdater.get_action(
    delta_locs, delta_yaws, next_spds,
    waypoint_rews[..., cmd_value], brak_rews, stop_rews, free,
    torch.zeros((self.num_plan, 2)).to(device),
    extract=(
        torch.tensor([[0., 0.]]),    # current location
        torch.tensor([0.]),          # current yaw
        torch.tensor([spd]),         # current speed
    )
)
action = int(Categorical(logits=action_values / self.temperature).sample())
steer, throt, brake = BellmanUpdater._actions[action]
```

### Differences from LBC Expert

| Aspect | LBC Expert | World on Rails |
|--------|-----------|----------------|
| **Expert type** | Rule-based PID autopilot | Bellman Q-value optimization |
| **Action space** | Continuous (steer, throttle, brake) | Discretized action table |
| **Hazard handling** | Explicit vehicle/ped/light checks | Implicit via reward maps |
| **Reward signals** | N/A | waypoint_rews, stop_rews, brak_rews, free |
| **Exploration** | Steering noise (disabled) | Ornstein-Uhlenbeck noise + temperature sampling |
| **Collision response** | Continue episode | `raise Exception` — flush data + terminate |
| **World model** | None | Learned ego dynamics model |
| **Observation** | BEV (7-channel) + RGB (384×160) | Semantic map label + multi-camera RGB+Seg |
| **FPS** | 10 | 20 |
| **Sensors** | 1 RGB camera | Multi-camera (wide 60° FOV + narrow 50° FOV) + semantic seg |

### WoR Data Collection
- Also has a `RandomCollector` for Phase 0: random actions to train the ego dynamics model.
- Collision → `flush_data()` + `raise Exception('Collector has collided!!')`.
- Steering multiplied by 1.2× for non-lane-follow commands (indices 4, 5).

### WoR Also Bundles LBC
- The WoR repository includes an `lbc/` directory with updated LBC training code,
  and an `autoagents/lbc_agent.py` for running LBC within WoR's evaluation framework.

---

## 12. Utility Functions — Reference

### `is_within_distance_ahead()` (`PythonAPI/agents/tools/misc.py`)
Core spatial check used by all hazard detectors:
```python
def is_within_distance_ahead(target_location, current_location,
                             orientation, max_distance, degree=60):
    u = np.array([target.x - current.x, target.y - current.y])
    distance = np.linalg.norm(u)
    if distance > max_distance:
        return False
    v = np.array([cos(radians(orientation)), sin(radians(orientation))])
    angle = degrees(acos(dot(u, v) / distance))
    return angle < degree
```

### `compute_yaw_difference()` (`PythonAPI/agents/tools/misc.py`)
Used for vehicle hazard yaw alignment:
```python
def compute_yaw_difference(yaw1, yaw2):
    u = [cos(radians(yaw1)), sin(radians(yaw1))]
    v = [cos(radians(yaw2)), sin(radians(yaw2))]
    angle = degrees(acos(clip(dot(u, v), -1, 1)))
    return angle  # 0° = same direction, 180° = opposite
```

### `_is_point_on_sidewalk()` (`PythonAPI/agents/navigation/agent.py`)
Used to filter out pedestrians standing on sidewalks:
```python
def _is_point_on_sidewalk(self, loc):
    pixel_x, pixel_y = self._world_to_pixel(loc)
    point = self._road_map[pixel_y, pixel_x, 0]
    return point == 0  # 0 = not road (sidewalk/offroad)
```

---

## 13. Summary — Suitability as Expert for Our Project

### Strengths of LBC Expert
1. **Simplicity**: ~200 lines of clear, auditable expert logic.
2. **Proven results**: 100% success rate on CoRL benchmarks.
3. **7-channel BEV**: Rich privileged observation encoding.
4. **PID baseline**: Easy to understand and tune.

### Weaknesses / Risks for Our Use
1. **Binary braking**: No smooth deceleration — causes jerky demonstrations.
2. **No lane filtering for vehicles**: May brake for vehicles in adjacent lanes.
3. **No forward prediction**: Reactive only — detects hazards at current instant.
4. **No stop sign handling**: Missing for CARLA 0.9.10+ which has stop signs.
5. **No slope handling**: May struggle on hilly terrain.
6. **Limited traffic light logic**: Only handles Town01/02 well; US-style is brittle.
7. **Old CARLA API**: Uses 0.9.6 API (`start_dtcrowd`, etc.) — needs porting for newer CARLA.

### Recommendation
The **TransFuser autopilot is significantly more sophisticated** and is the better
choice as a base for our expert. However, the LBC/WoR approach of using a BEV
representation with 7 semantic channels is valuable for our bird's-eye-view model
architecture. We should:
- Use TransFuser-style autopilot for data collection (proportional braking,
  OBB checks, stuck detection).
- Adopt LBC's BEV channel design as inspiration for our observation space.
- Consider WoR's Q-value approach for offline reinforcement learning experiments.
