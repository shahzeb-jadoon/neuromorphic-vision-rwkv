# TransFuser Expert Autopilot — Detailed Analysis

**Source**: Chitta et al., "TransFuser: Imitation with Transformer-Based Sensor
Fusion for Autonomous Driving" (PAMI 2023, CVPR 2021)

**Repository**: <https://github.com/autonomousvision/transfuser> (branch `2022`)

**Code file**: `team_code_autopilot/autopilot.py`

**Additional reference**: [Bernhard Jaeger's Master Thesis](https://kait0.github.io/assets/pdf/master_thesis_bernhard_jaeger.pdf)
— provides deeper analysis of the autopilot building blocks.

---

## 1. Expert/Autopilot Features — Complete List

| # | Feature | How Triggered |
|---|---------|---------------|
| 1 | **Lane / route following** | PID steering towards dense route waypoints (always active) |
| 2 | **Traffic light handling** | OBB intersection between detector box and light trigger volume |
| 3 | **Stop sign handling** | OBB intersection (can be disabled via `ignore_stop_signs=True`) |
| 4 | **Vehicle collision avoidance** | Forward kinematic bicycle model extrapolation + OBB checks |
| 5 | **Pedestrian / walker avoidance** | Forward linear extrapolation + OBB checks |
| 6 | **Stuck vehicle detection** | Speed/throttle/brake buffer analysis over 30 frames |
| 7 | **Junction / turn handling** | Reduced target speed when `ego_vehicle_waypoint.is_junction` |
| 8 | **Slope handling** | Extra throttle when pitch > threshold |
| 9 | **Safety bounding box (Bremsweg)** | Emergency braking box placed in front of ego |
| 10 | **Waypoint future-state prediction** | Save future ego states for training labels |
| 11 | **GPS denoising** | Rolling average of GPS buffer (100 frames) |
| 12 | **Steering noise injection** | Small Gaussian noise on expert steer for data augmentation |

---

## 2. Implementation Details — Per Feature

### 2.1 Lane / Route Following

- The route is densified via `interpolate_trajectory()` using CARLA's
  `GlobalRoutePlanner` at 1.0 m hop resolution.
- Two `RoutePlanner` instances consume waypoints:
  - **Near planner**: `min_distance=3.5 m`, `max_distance=50 m` — used for
    steering.
  - **Far planner (command)**: `min_distance=7.5 m`, `max_distance=50 m` —
    used for navigation commands (left/right/straight/follow lane).
- Steering angle is computed as the bearing angle to the next route waypoint,
  normalised to `[-1, 1]` corresponding to `[-90°, 90°]`.
- The `angle_search_range` parameter (default `0`) controls how many future
  waypoints are examined to find the smallest steering angle.

### 2.2 PID Controllers

There are **four** PID controller instances (two for normal, two for
extrapolation):

| Controller | K_P | K_I | K_D | Window (n) | Purpose |
|---|---|---|---|---|---|
| **Turn (steering)** | 1.25 | 0.75 | 0.3 | 40 | Lateral control |
| **Speed (throttle)** | 5.0 | 0.5 | 1.0 | 40 | Longitudinal control |

PID formula:
```
output = K_P * error + K_I * mean(window) + K_D * (window[-1] - window[-2])
```
- The integral term is the **mean** of the last `n` errors (not a traditional sum).
- The derivative is just the **single-step difference**, no dt normalisation.

### 2.3 Traffic Light Handling

1. Filter all traffic lights within `light_radius = 15.0 m` of the vehicle.
2. Create a **detector bounding box** attached to the ego vehicle:
   - Centre: `(-2.0, 0.0, 0.0)` in ego-local coords (i.e. 2 m behind centre)
   - Half-extents: `(4.5, 1.5, 2.0)` m → full box is 9×3×4 m
3. For each nearby traffic light, compute its **trigger volume** (the box on
   the road surface where CARLA activates the light).
4. Check **OBB (oriented bounding box) intersection** between the detector box
   and the trigger volume using the **Separating Axis Theorem (SAT)** with 15
   axis tests.
5. If intersection is detected AND the light is **Red or Yellow** →
   `light_hazard = True` → full brake.
6. **No proportional braking** — it's a binary decision: brake or don't.

### 2.4 Stop Sign Handling

- Default: **disabled** (`self.ignore_stop_signs = True`).
- When enabled, uses OBB intersection of the vehicle's own bounding box with
  stop sign trigger volumes.
- Maintains a `cleared_stop_signs` list so it only stops once per sign.
- A stop sign is "cleared" when the vehicle is inside the trigger box AND
  `speed == 0`.

### 2.5 Vehicle Collision Avoidance (Vehicle Following)

This is the most complex subsystem. **There is no explicit "vehicle following"
mode** — the autopilot uses collision prediction:

1. **Detection radius**: `30.0 m` — only vehicles within this distance are
   considered.
2. **Future prediction**:
   - Each nearby vehicle's trajectory is extrapolated forward using a
     **kinematic bicycle model** (`EgoModel`) with the vehicle's current
     steer/throttle/brake held constant.
   - The ego vehicle is also extrapolated forward assuming it **does NOT brake**
     (using `target_speed` throttle and current steering).
   - **Junction extrapolation**: `4.0 seconds` into the future.
   - **Non-junction extrapolation**: `1.0 seconds` into the future.
   - At the frame rate of 20 Hz, this means **80 future frames** at junctions,
     **20 outside junctions**.
3. **OBB intersection check**: At each future timestep, check if the ego's
   predicted bounding box overlaps any other actor's predicted bounding box
   using SAT.
4. **Ego box is split in half**: Front half and back half are checked
   separately. If only the back half intersects (i.e. the other car is
   overtaking from behind), it is **ignored** (`back_only_vehicle_id` list).
5. **Stuck vehicle handling**: If a nearby vehicle has:
   - Mean velocity < `0.1 m/s`
   - Mean throttle > `0.1`
   - Mean brake < `0.1`
   …over the last 30 frames, it is marked "stuck" and its bounding box is
   frozen at timestep 0 (not extrapolated forward).

#### Braking Profile

**Binary brake only** — there is **no proportional/IDM/kinematic braking
profile**. The `_get_brake()` method returns either `True` or `False`.

When `brake == True`:
- `throttle = 0.0`
- `control.brake = 1.0` (full brake)
- Steering is damped by `steer_damping = 0.5` (multiplied)

When `brake == False`:
- Normal PID throttle control.

#### Safety Bounding Box ("Bremsweg")

An additional safety check places a box directly in front of the ego:

```python
bremsweg = ((speed * 3.6) / 10.0)**2 / 2.0   # German braking distance formula
safety_x = np.clip(bremsweg + 1.0, a_min=2.0, a_max=4.0)
```

This box is checked against **current** positions of all actors (no
extrapolation). If anything is in this box → immediate `vehicle_hazard = True`.

### 2.6 Pedestrian / Walker Avoidance

1. Walkers within `detection_radius = 30.0 m` are checked.
2. Each walker is extrapolated forward using **constant velocity linear
   motion**: each frame advances `position += direction * speed * (1/frame_rate)`.
3. The same OBB intersection check as vehicle detection is used (ego front-half
   box vs walker bounding box).
4. Same junction/non-junction time horizon applies.
5. Binary brake, no proportional response.

### 2.7 Junction / Turn Handling

Very simple two-speed system:
- **At junction** (`ego_vehicle_waypoint.is_junction` is True):
  `target_speed = 3.0 m/s` ≈ `10.8 km/h`
- **Outside junction**: `target_speed = 4.0 m/s` ≈ `14.4 km/h`

No explicit turn anticipation, curvature-based speed reduction, or turn type
differentiation. The speed reduction happens **only once inside the junction**.

### 2.8 Slope Handling

```python
if vehicle.get_transform().rotation.pitch > self.slope_pitch:  # 10.0 degrees
    throttle += self.slope_throttle  # adds 0.4
```

### 2.9 Steering

- Angle to next waypoint: `atan2(aim[1], aim[0])` normalised to `[-1, 1]`.
- PID controller output is clipped to `[-1.0, 1.0]`.
- When braking, steer is damped: `steer *= 0.5`.
- A small noise `N(0, 0.001)` is added for data diversity.
- Steering is averaged over a buffer of size 1 (no smoothing by default).
- Maximum speed error (delta for throttle PID): `clip_delta = 0.25`.
- Maximum throttle: `clip_throttle = 0.75`.

---

## 3. Key Parameters — Complete Reference

### Speed Control
| Parameter | Value | Unit |
|---|---|---|
| `target_speed_slow` (junction) | 3.0 | m/s |
| `target_speed_fast` (outside junction) | 4.0 | m/s |
| `clip_delta` | 0.25 | m/s |
| `clip_throttle` | 0.75 | 0–1 |
| `slope_pitch` | 10.0 | degrees |
| `slope_throttle` | 0.4 | 0–1 |

### Steering Control
| Parameter | Value | Unit |
|---|---|---|
| `steer_buffer_size` | 1 | frames |
| `steer_damping` | 0.5 | multiplier |
| `steer_noise` | 1e-3 | std dev |
| `angle_search_range` | 0 | waypoints |

### PID Gains
| Controller | K_P | K_I | K_D | Window |
|---|---|---|---|---|
| Turn (lateral) | 1.25 | 0.75 | 0.3 | 40 |
| Speed (longitudinal) | 5.0 | 0.5 | 1.0 | 40 |

### Traffic Light Detection
| Parameter | Value | Unit |
|---|---|---|
| `light_radius` | 15.0 | m |
| `center_bb_light_x` | -2.0 | m (ego-local) |
| `center_bb_light_y` | 0.0 | m |
| `center_bb_light_z` | 0.0 | m |
| `extent_bb_light_x` | 4.5 | m (half-extent) |
| `extent_bb_light_y` | 1.5 | m (half-extent) |
| `extent_bb_light_z` | 2.0 | m (half-extent) |

### Obstacle Detection
| Parameter | Value | Unit |
|---|---|---|
| `detection_radius` | 30.0 | m |
| `extrapolation_seconds` (junction) | 4.0 | s |
| `extrapolation_seconds_no_junction` | 1.0 | s |
| `frame_rate` (dynamics model) | 20 | Hz |

### Stuck Vehicle Detection
| Parameter | Value | Unit |
|---|---|---|
| `stuck_buffer_size` | 30 | frames |
| `stuck_vel_threshold` | 0.1 | m/s |
| `stuck_throttle_threshold` | 0.1 | 0–1 |
| `stuck_brake_threshold` | 0.1 | 0–1 |

### Route Planning
| Planner | `min_distance` | `max_distance` |
|---|---|---|
| Near waypoint | 3.5 m | 50 m |
| Far command | 7.5 m | 50 m |
| Trajectory hop resolution | 1.0 m | — |

### Kinematic Bicycle Model (EgoModel)
| Parameter | Value | Source |
|---|---|---|
| `front_wb` | -0.090769015 | World on Rails tuned |
| `rear_wb` | 1.4178275 | World on Rails tuned |
| `steer_gain` | 0.36848336 | World on Rails tuned |
| `brake_accel` | -4.952399 m/s² | World on Rails tuned |
| `throt_accel` | 0.5633837 m/s² | World on Rails tuned |
| `dt` | 0.05 s (1/20) | — |

---

## 4. Vehicle Following — Detailed Analysis

TransFuser's autopilot has **no dedicated vehicle-following mode** (no IDM,
no car-following model, no gap-keeping). Instead:

1. **Binary collision check**: If the extrapolated ego trajectory (assuming no
   braking) would intersect with any extrapolated vehicle trajectory, the
   autopilot applies **full brake** (`brake = 1.0`).
2. When the hazard clears, it resumes normal throttle to `target_speed`.
3. This results in a **bang-bang** braking profile: full brake → release → full
   brake, which creates somewhat jerky following behaviour at low speeds.
4. The "Bremsweg" (braking distance) safety box provides a speed-dependent
   minimum following distance: `((speed_kmh / 10)² / 2) + 1` metres, clamped
   to [2, 4] m.

**Implication**: At `target_speed_fast = 4.0 m/s` (14.4 km/h), the Bremsweg
safety distance is ~2.0 m. This is quite close.

---

## 5. Traffic Light Handling — Detailed

1. CARLA provides `trigger_volume` for each traffic light — a box on the road
   indicating where the light "applies".
2. The autopilot creates a detector box in front of the ego vehicle (centred
   2 m behind the car centre, extending 4.5 m forward, 1.5 m to each side).
3. When this detector box intersects a red/yellow light's trigger volume →
   full brake, `light_hazard = True`.
4. **No proportional braking**: no gradual deceleration as you approach. It's
   full-brake-or-nothing once the OBB test fires.
5. Green lights: no action (treated as no hazard).

---

## 6. Turn / Junction Handling — Detailed

**Extremely simple**:
- Junction detection: `self.world_map.get_waypoint(self._vehicle.get_location()).is_junction`
- Speed: `3.0 m/s` (≈11 km/h) inside junction, `4.0 m/s` (≈14.4 km/h) outside.
- **No turn anticipation**: The speed only changes once the ego is already on
  the junction waypoint, not beforehand.
- **No differentiation by turn type**: Left turn, right turn, U-turn, and
  going straight through a junction all get the same 3.0 m/s target.
- **No curvature-based steering adaptation**.
- Collision extrapolation horizon increases to 4 s at junctions (vs 1 s outside).

---

## 7. Collision Handling During Data Collection

The autopilot itself **does not handle collisions or recovery**. It has no
collision sensor and no post-collision logic. Key points:

- The `destroy()` method is a no-op (`pass`).
- The CARLA **Leaderboard** framework handles route termination — if there are
  too many infractions (collisions, red-light violations, etc.), the route is
  terminated and the next route begins.
- The data collection script (`leaderboard/scripts/datagen.sh`) iterates over
  routes and restarts the agent for each one.
- **There is no explicit "discard this episode" or "restart after collision"
  logic** in the autopilot code. The saved data includes collision frames
  without any special handling.
- Data quality control would need to be done in post-processing (e.g.
  examining the `vehicle_hazard`, `walker_hazard` fields in measurements).

---

## 8. Route Corridor / Lane Detection

**There is no explicit lane detection or route corridor**. The autopilot
does NOT check whether a vehicle is "in its lane". Instead:

- **All vehicles within `detection_radius = 30.0 m`** are considered,
  regardless of what lane they are in.
- The OBB collision prediction is purely geometric — if extrapolated
  trajectories overlap, it brakes.
- Oncoming traffic, vehicles in adjacent lanes, and vehicles behind are only
  ignored via the **back-half** check: if only the rear half of the ego's
  bounding box intersects, it's assumed the other vehicle is overtaking and
  is ignored.
- There is no waypoint-based or lane-boundary-based filtering to determine
  whether a vehicle is actually in the ego's path.

**Consequence**: The autopilot may unnecessarily brake for vehicles in adjacent
lanes whose bounding boxes are close but not actually in the ego's lane. This
is mitigated by the low target speeds (3–4 m/s) and the relatively small
bounding boxes.

---

## 9. GitHub Repository Confirmation

| Item | URL |
|---|---|
| **Main repository** | <https://github.com/autonomousvision/transfuser> |
| **Branch** | `2022` |
| **Autopilot code** | `team_code_autopilot/autopilot.py` |
| **Navigation planner** | `team_code_autopilot/nav_planner.py` |
| **Data agent wrapper** | `team_code_autopilot/data_agent.py` |
| **Utilities** | `team_code_autopilot/utils/` |
| **Data generation script** | `leaderboard/scripts/datagen.sh` |

---

## 10. Known Limitations and Issues

### From the Code

1. **Very low target speeds**: 3.0–4.0 m/s (11–14.4 km/h). Driving at
   highway speeds or realistic urban speeds is not attempted.
2. **Binary braking only**: No proportional/smooth braking. Full brake or no
   brake. Results in jerky stop-and-go behaviour.
3. **No lane-awareness**: All nearby vehicles trigger collision checks
   regardless of lane, causing phantom braking.
4. **No turn anticipation**: Speed reduction only happens once already inside
   a junction.
5. **No speed-limit awareness**: Fixed target speeds, ignores CARLA speed limit
   signs.
6. **Simulation bug handling**: `if math.isnan(compass): compass = 0.0` — CARLA
   sometimes returns NaN for compass readings.
7. **Stop signs disabled by default**: `self.ignore_stop_signs = True`.
8. **No explicit collision recovery**: If a collision occurs, data collection
   continues without reset.
9. **Constant-action extrapolation**: Vehicles are predicted assuming they
   continue their current steer/throttle/brake forever. No anticipation of
   other agents changing behaviour.
10. **Steer noise is tiny**: `1e-3` std dev adds negligible variation.

### From GitHub Issues

- **Issue #77** ("Weird vehicle intersection prediction"): Users reported
  unexpected braking for vehicles not in the ego's path — consistent with
  the lack of lane filtering.
- **Issue #170** ("Question about collision avoidance"): Community questions
  about the collision avoidance mechanism.
- **Issue #17** ("About data generation"): Questions about data quality and
  generation process.

### From the Paper (Section 4.3)

- The expert is described as having "access to the complete state of the
  environment". Its performance is an **upper bound** for the learned agent.
- The paper notes that the expert is imperfect — it accumulates infractions
  over long routes.

---

## 11. OBB Intersection — Separating Axis Theorem

The core collision algorithm is the **Separating Axis Theorem (SAT)** for 3D
oriented bounding boxes. The implementation checks 15 potential separating
axes:

- 3 face normals of box A (forward, right, up)
- 3 face normals of box B (forward, right, up)
- 9 cross-products of pairs of edges

If NO separating axis exists → boxes **intersect** → hazard detected.

Function: `check_obb_intersection(obb1, obb2)` with helper
`get_separating_plane(rPos, plane, obb1, obb2)`.

---

## 12. Data Saved Per Frame

Every `save_freq` frames (default: every 10 frames = 0.5 s), the autopilot
saves a JSON measurement file containing:

```
x, y, theta, speed, target_speed, x_command, y_command, command,
waypoints (future ego positions), steer, throttle, brake, junction,
vehicle_hazard, light_hazard, walker_hazard, stop_sign_hazard,
angle, ego_matrix
```

---

## 13. Comparison with Our Expert

Our `PrivilegedExpert` in `agents/expert/privileged_expert.py` improves upon
the TransFuser autopilot in several key areas:

| Feature | TransFuser | Ours |
|---|---|---|
| Target speed | 14.4 km/h fixed | 30 km/h, speed-limit aware |
| Braking profile | Binary (0 or 1) | Proportional (kinematic/comfort decel) |
| Vehicle following | Collision-or-not | Speed-dependent hazard distances |
| Lane awareness | None (30 m radius) | Frenet frame lateral filtering |
| Turn handling | Junction flag only | Curvature-based + anticipation |
| Traffic lights | Binary brake | Distance-proportional deceleration |
| PID | Windowed mean integral | Anti-windup clamped integral |
| Driving modes | None (implicit) | Explicit state machine |
