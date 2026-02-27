# TCP & InterFuser Expert Autopilot — Detailed Analysis

---

## Part 1: TCP (Trajectory-guided Control Prediction)

**Source**: Penghao Wu, Xiaosong Jia, Li Chen, Junchi Yan, Hongyang Li, Yu Qiao,
"Trajectory-guided Control Prediction for End-to-end Autonomous Driving: A Simple
yet Strong Baseline" (NeurIPS 2022)

**Repository**: <https://github.com/OpenDriveLab/TCP>

**Code files**:
- `leaderboard/team_code/auto_pilot.py` — expert autopilot
- `leaderboard/team_code/map_agent.py` — sensor handling / data saving
- `leaderboard/team_code/base_agent.py` — base sensors, save, tick logic
- `leaderboard/team_code/pid_controller.py` — PID implementation
- `leaderboard/team_code/planner.py` — route planner

**CARLA version**: 0.9.10.1

---

### 1. Expert/Autopilot Features — Complete List

| # | Feature | How Triggered |
|---|---------|---------------|
| 1 | **Route / lane following** | PID steering towards near waypoint, PID throttle towards target speed |
| 2 | **Traffic light handling** | `_vehicle.get_traffic_light_state()` CARLA API + proximity filter |
| 3 | **Stop sign handling** | Trigger-volume waypoint-step proximity test (always enabled) |
| 4 | **Vehicle collision avoidance** | `get_collision()` time-to-collision linear projection |
| 5 | **Pedestrian / walker avoidance** | `get_collision()` linear extrapolation |
| 6 | **Two-speed system (junction/open road)** | `should_slow` based on angle thresholds |
| 7 | **Weather randomisation** | Randomly changes weather every 10 steps during data collection |
| 8 | **Steering noise injection** | `steer + 1e-2 * np.random.randn()` |

### What TCP Does **NOT** Have

| Feature | TCP | TransFuser |
|---------|-----|------------|
| Kinematic bicycle model extrapolation | ✗ (linear only) | ✓ (`EgoModel`) |
| OBB / SAT collision checking | ✗ (ray intersection) | ✓ (full 3D OBB) |
| Safety bounding box (Bremsweg) | ✗ | ✓ |
| Stuck vehicle detection | ✗ | ✓ (30-frame buffer) |
| Slope / pitch handling | ✗ | ✓ (`slope_throttle`) |
| GPS denoising | ✗ | ✓ (100-frame rolling avg) |
| Waypoint future-state prediction | ✗ | ✓ |
| Lane change handling | ✗ | ✗ |
| Bike-specific handling | ✗ | ✗ |
| Junction vehicle detection | ✗ | ✗ |
| Hazard actor destruction | ✗ | ✗ |

---

### 2. Does TCP Reuse the TransFuser Autopilot?

**Yes — TCP's autopilot is essentially a stripped-down copy of the TransFuser
LBC-lineage autopilot.** TCP acknowledges TransFuser in its repository
README. Key evidence:

- Identical `get_collision()` function (ray-intersection, not OBB/SAT).
- Identical `_is_vehicle_hazard()`, `_is_walker_hazard()`, `_is_light_red()`,
  `_is_stop_sign_hazard()` methods — same logic, same parameter values.
- Identical PID controller gains and structure.
- Same `MapAgent` → `BaseAgent` inheritance.
- Same two-speed system with same angle thresholds.
- **Missing**: TransFuser's 2022 branch has OBB/bicycle-model collision checking
  and Bremsweg box; TCP uses the simpler 2020/LBC-era collision approach (linear
  ray intersection). This suggests TCP forked from an earlier TransFuser commit
  or from `2020_CARLA_challenge`.

---

### 3. Vehicle Following Implementation

TCP has **no explicit vehicle-following mode**. The collision avoidance serves
as de facto vehicle following:

#### `_is_vehicle_hazard()` — Vehicle Detection

```python
def _is_vehicle_hazard(self, vehicle_list):
    z = self._vehicle.get_location().z
    o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
    p1 = _numpy(self._vehicle.get_location())
    s1 = max(10, 3.0 * np.linalg.norm(_numpy(self._vehicle.get_velocity())))
    v1_hat = o1
    v1 = s1 * v1_hat

    for target_vehicle in vehicle_list:
        if target_vehicle.id == self._vehicle.id:
            continue

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

        # Filter rules:
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

#### Braking Profile: **Binary**

- `_should_brake()` returns `True`/`False`. When true: `throttle = 0.0`,
  `brake = True` (full brake), steering damped by `steer *= 0.5`.
- No proportional/IDM/kinematic braking profile.
- The detection distance `s1` **does scale with ego speed**: `max(10, 3.0 * ego_speed)`,
  so there is a speed-dependent look-ahead, but the response is still binary.
- The other vehicle's "beam" length is `s2 = max(5.0, 2.0 * other_speed)`.

#### Lane Corridor / In-Lane Detection

**No lane filtering**. All vehicles within the angular cone and distance are
considered hazards regardless of lane. The filtering is purely geometric:
- `angle_to_car <= 30°` (within a ±30° forward cone)
- `angle_between_heading <= 60°` (headings roughly aligned)
- `distance <= s1` (within speed-scaled detection range)

There is no waypoint, road ID, or lane ID filtering.

#### `get_collision()` — Ray Intersection

```python
def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1
    if abs(np.linalg.det(A)) < 1e-3:
        return False, None
    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1)
    return collides, p1 + x[0] * v1
```

This is a **2D ray intersection test**: it checks whether two line segments
(`p1→p1+v1` and `p2→p2+v2`) intersect. Both `x[0]` and `x[1]` must be in
`[0, 1]` for a collision, meaning the intersection happens within the "time
horizon" of both rays. The ego's ray length is speed-dependent (`s1 *
ego_heading`), giving implicit time scaling.

---

### 4. Traffic Light Handling

```python
def _is_light_red(self, lights_list):
    if self._vehicle.get_traffic_light_state() != carla.libcarla.TrafficLightState.Green:
        affecting = self._vehicle.get_traffic_light()
        for light in self._traffic_lights:
            if light.id == affecting.id:
                return affecting
    return None
```

- Uses **CARLA's built-in** `get_traffic_light_state()` API, which automatically
  determines which traffic light affects the ego vehicle based on lane
  registry.
- `self._traffic_lights` is a proximity-filtered list (via `get_nearby_lights()`
  which filters lights whose trigger volume is within bounding-box-diagonal
  distance).
- **Binary brake**: If the state is not green (red OR yellow) → `brake = True`.
- **No proportional braking**, no distance-based deceleration. The CARLA API
  only triggers when the vehicle is near the stop line, so the brake is
  effectively already at a short distance.

---

### 5. Turn / Junction Handling

TCP does **not** use `is_junction` directly. Instead it uses angle thresholds:

```python
angle_far_unnorm = self._get_angle_to(pos, theta, far_target)
should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
target_speed = 4.0 if should_slow else 7.0
```

| Condition | Target Speed |
|-----------|-------------|
| `abs(angle_far) > 45°` or `abs(angle_near) > 5°` | 4.0 m/s (14.4 km/h) |
| Otherwise | 7.0 m/s (25.2 km/h) |

This is slightly different from TransFuser's junction-based speed reduction:
- **TCP uses angle-to-waypoint** as a proxy for turn detection.
- When steering towards a sharp turn (far waypoint > 45° offset), it slows
  down.
- When steering even slightly (near waypoint > 5° offset), it also slows.
- **No junction-specific logic** — turns on straight roads with slight
  bends trigger the same behaviour.

---

### 6. Pedestrian Handling

```python
def _is_walker_hazard(self, walkers_list):
    z = self._vehicle.get_location().z
    p1 = _numpy(self._vehicle.get_location())
    v1 = 10.0 * _orientation(self._vehicle.get_transform().rotation.yaw)

    for walker in walkers_list:
        v2_hat = _orientation(walker.get_transform().rotation.yaw)
        s2 = np.linalg.norm(_numpy(walker.get_velocity()))
        if s2 < 0.05:
            v2_hat *= s2
        p2 = -3.0 * v2_hat + _numpy(walker.get_location())
        v2 = 8.0 * v2_hat
        collides, collision_point = get_collision(p1, v1, p2, v2)
        if collides:
            return walker
    return None
```

- **Ego ray**: Fixed length of 10.0 units (not speed-dependent, unlike
  vehicle detection).
- **Walker ray**: Starts 3.0 units behind the walker's current position
  (`p2 = -3.0 * v2_hat + walker_location`), extends 8.0 units in the
  walker's heading.
- **Stationary walkers**: If `speed < 0.05`, the velocity vector is scaled
  down (effectively zero-length ray), so the collision check only triggers
  if the ego's ray directly hits the walker's offset position.
- Binary brake on collision detection.

---

### 7. Key Parameters

#### Speed Control
| Parameter | Value | Unit |
|---|---|---|
| `target_speed` (slowing) | 4.0 | m/s |
| `target_speed` (normal) | 7.0 | m/s |
| `clip_delta` | 0.25 | m/s |
| `clip_throttle` | 0.75 | 0–1 |
| `should_slow angle_far` | 45.0 | degrees |
| `should_slow angle_near` | 5.0 | degrees |

#### PID Gains
| Controller | K_P | K_I | K_D | Window (n) |
|---|---|---|---|---|
| Turn (steering) | 1.25 | 0.75 | 0.3 | 40 |
| Speed (throttle) | 5.0 | 0.5 | 1.0 | 40 |

#### Vehicle Detection
| Parameter | Value | Notes |
|---|---|---|
| Ego ray length `s1` | `max(10, 3.0 * ego_speed)` | Speed-dependent |
| Other ray length `s2` | `max(5.0, 2.0 * other_speed)` | Speed-dependent |
| Max `angle_to_car` | 30° | Forward cone |
| Max `angle_between_heading` | 60° | Heading alignment |

#### Walker Detection
| Parameter | Value |
|---|---|
| Ego ray length | 10.0 (fixed) |
| Walker ray length | 8.0 |
| Walker ray offset behind | -3.0 |
| Stationary threshold | 0.05 m/s |

#### Stop Sign Detection
| Parameter | Value |
|---|---|
| `PROXIMITY_THRESHOLD` | 30.0 m |
| `SPEED_THRESHOLD` | 0.1 m/s |
| `WAYPOINT_STEP` | 1.0 m |
| `multi_step` | 20 steps |

#### Route Planning
| Planner | `min_distance` | `max_distance` |
|---|---|---|
| Near waypoint | 4.0 m | 50 m |
| Far command | 7.5 m | 25 m |

#### Steer Noise
| Parameter | Value |
|---|---|
| Noise type | Gaussian |
| Noise scale | `1e-2` std dev |

#### Weather Randomisation
| Parameter | Value |
|---|---|
| Change frequency | Every 10 steps |
| Weather pool | ClearNoon, ClearSunset, WetNoon, HardRainNoon (4 weathers) |

---

### 8. Collision Handling During Data Collection

- **No collision sensor** in the autopilot.
- **No explicit collision recovery** — if a collision occurs, the autopilot
  continues driving.
- The CARLA **Leaderboard** framework handles route termination on excessive
  infractions.
- Data is saved every 10 steps unconditionally. Collision frames are included
  in the dataset without special marking.
- **Post-hoc filtering**: TCP provides `tools/filter_data.py` to filter out
  invalid data after collection (likely removes frames with collisions or
  routing errors), and `tools/gen_data.py` to pack data for training.

---

### 9. Data Saved Per Frame

Saved every 10 frames:
```
x, y, theta, speed, target_speed, x_command, y_command, command,
steer, throttle, brake, weather, weather_id,
near_node_x, near_node_y, far_node_x, far_node_y,
is_vehicle_present, is_pedestrian_present,
is_red_light_present, is_stop_sign_present,
should_slow, should_brake,
angle, angle_unnorm, angle_far_unnorm
```
Plus RGB front image (`400×300` sensor data, but saved as `900×256` for the
front camera based on `base_agent.py`).

---

### 10. Unique Features (vs TransFuser/LBC)

1. **Higher normal target speed**: 7.0 m/s vs TransFuser's 4.0 m/s.
2. **Angle-based turn detection** instead of `is_junction` flag.
3. **Weather randomisation** during collection (changes every 10 steps).
4. **Fewer weathers**: Only 4 weathers vs TransFuser's 14.
5. **Post-collection data filtering**: `tools/filter_data.py` pipeline.
6. **Larger steering noise**: `1e-2` vs TransFuser's `1e-3`.
7. **Minimal sensor suite for data collection**: Only front RGB + GPS + speed +
   IMU (all other sensors commented out in `base_agent.py`).

---
---

## Part 2: InterFuser (Safety-Enhanced Autonomous Driving)

**Source**: Hao Shao, Letian Wang, RuoBing Chen, Hongsheng Li, Yu Liu,
"Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion
Transformer" (CoRL 2022)

**Repository**: <https://github.com/opendilab/InterFuser>

**Code files**:
- `leaderboard/team_code/auto_pilot.py` — expert autopilot (significantly extended)
- `leaderboard/team_code/map_agent.py` — sensor handling / tick
- `leaderboard/team_code/base_agent.py` — base sensors, save, tick, YAML config
- `leaderboard/team_code/pid_controller.py` — PID implementation
- `leaderboard/team_code/planner.py` — route planner
- `data_collection/` — batch data generation scripts
- `tools/data/` — post-processing utilities

**CARLA version**: 0.9.10.1

---

### 1. Expert/Autopilot Features — Complete List

| # | Feature | How Triggered |
|---|---------|---------------|
| 1 | **Route / lane following** | PID steering towards near waypoint, PID throttle towards target speed |
| 2 | **Traffic light handling** | CARLA API + custom `_find_closest_valid_traffic_light()` waypoint-based search |
| 3 | **Stop sign handling** | Trigger-volume waypoint-step proximity test (always enabled) |
| 4 | **Vehicle collision avoidance** | `get_collision()` ray intersection + angle/heading/distance filtering |
| 5 | **Lane-change vehicle avoidance** | `_is_lane_vehicle_hazard()` — dedicated lane-change collision check |
| 6 | **Junction vehicle avoidance** | `_is_junction_vehicle_hazard()` — junction-specific multi-ray check |
| 7 | **Bicycle / motorbike handling** | `_is_bike_hazard()` — dedicated bike detection with wider angle |
| 8 | **Pedestrian / walker avoidance** | `get_collision()` linear extrapolation |
| 9 | **Two-speed angle-based system** | `should_slow` based on angle thresholds |
| 10 | **Waypoint disturbance** | Configurable lateral waypoint perturbation for training diversity |
| 11 | **Bird's-eye-view generation** | `BirdViewProducer` renders semantic BEV at each step |
| 12 | **Multi-view sensor suite** | Front/left/right/rear RGB + seg + depth + LiDAR |
| 13 | **Actor data collection** | Saves all nearby vehicle/walker/traffic-light world states |
| 14 | **Hazard actor destruction** | `_detect_and_destory_hazard_actors()` removes permanently stuck actors |
| 15 | **Multi-weather + night** | 21 weather conditions including 7 custom night weathers |
| 16 | **Configurable save skip frames** | YAML-configurable frame skip rate |
| 17 | **Junction flag recording** | Records `is_junction` in measurements |
| 18 | **Future waypoints recording** | Saves 50 future waypoints for supervision |
| 19 | **Traffic light ID recording** | Saves `affected_light_id` for the closest relevant light |

---

### 2. Does InterFuser Reuse the TransFuser Autopilot?

**Partially — InterFuser forks from the same TransFuser/LBC autopilot lineage
but makes substantial extensions**. InterFuser explicitly acknowledges TransFuser
and `2020_CARLA_challenge` in its README.

**Shared with TCP/TransFuser baseline**:
- Same `get_collision()` 2D ray intersection function.
- Same `_is_vehicle_hazard()` core logic (angle/heading/distance filtering).
- Same `_is_walker_hazard()` logic.
- Same PID gains.
- Same `_point_inside_boundingbox()` and `_is_actor_affected_by_stop()`.
- Same `MapAgent` → `BaseAgent` inheritance.

**InterFuser additions beyond the baseline**:
- `_is_junction_vehicle_hazard()` — entirely new method.
- `_is_lane_vehicle_hazard()` — entirely new method.
- `_is_bike_hazard()` — entirely new method.
- `_find_closest_valid_traffic_light()` — custom traffic light search using
  road/lane IDs.
- `_detect_and_destory_hazard_actors()` — removes stuck actors from simulation.
- `collect_actor_data()` — saves full world state of nearby actors.
- `disturb_waypoints()` — waypoint perturbation for data augmentation.
- `BirdViewProducer` integration — BEV rendering.
- YAML-based configuration system.
- Multi-view camera suite with full segmentation/depth.
- Night weather support (7 custom night conditions).

---

### 3. Vehicle Following Implementation

InterFuser has **three** separate vehicle detection methods (vs TCP/TransFuser's
one):

#### 3a. `_is_vehicle_hazard()` — Standard Forward Detection

Almost identical to TCP's but with **additional road-aware filtering**:

```python
# Added in InterFuser: skip vehicles stopped on different roads
if (
    not w2.is_junction
    and angle_between_heading > 45.0
    and s2a < 0.5
    and distance > 4
):
    if w1.road_id != w2.road_id:
        continue

# Added: skip same-road different-lane vehicles unless lane changing
if (angle_between_heading < 15
    and w1.road_id == w2.road_id
    and w1.lane_id != w2.lane_id
    and command != RoadOption.CHANGELANELEFT
    and command != RoadOption.CHANGELANERIGHT
):
    continue
```

**Key addition**: InterFuser uses `w1.road_id`, `w1.lane_id`, `w2.road_id`,
`w2.lane_id` from CARLA's waypoint system to filter out vehicles that are:
- On a different road (and stopped, heading differently)
- On the same road but different lane (unless actively changing lanes)

This is **significant** — TCP and TransFuser have no lane filtering at all.

#### 3b. `_is_junction_vehicle_hazard()` — Junction-Specific

This is **entirely new** in InterFuser:

```python
def _is_junction_vehicle_hazard(self, vehicle_list, command):
    # Uses bounding box extent for more accurate position
    x1 = self._vehicle.bounding_box.extent.x
    p1 = self._vehicle.get_location() + x1 * self._vehicle.get_transform().get_forward_vector()

    # Ego ray direction adjusted by turn command:
    if command == RoadOption.RIGHT:
        shift_angle = 25
    elif command == RoadOption.LEFT:
        shift_angle = -25
    else:
        shift_angle = 0
    v1 = (4 * s1 + 5) * _orientation(ego_yaw + shift_angle)

    for target_vehicle in vehicle_list:
        # Skip if both ego and target are NOT in a junction
        if w1.is_junction == False and w2.is_junction == False:
            continue
        # Skip nearly-aligned or anti-aligned vehicles
        if angle_between_heading < 15.0 or angle_between_heading > 165:
            continue
        # Skip vehicles > 20m away
        if self._vehicle.get_location().distance(p2) > 20:
            continue

        # Triple ray check: straight, left-angled, right-angled
        v2 = (4 * s2 + 2 * x2 + 6) * o2
        v2_left = (4 * s2 + 2 * x2 + 6) * o2_left    # yaw - 15°
        v2_right = (4 * s2 + 2 * x2 + 6) * o2_right   # yaw + 15°

        # Also checks if target has a red/green light
        light = self._find_closest_valid_traffic_light(target_vehicle.get_location(), min_dis=10)
        if light is not None and light.state != Green:
            continue  # target is stopped at red, don't brake for them
```

**Key features**:
- Ego ray is **shifted by ±25°** based on planned turn direction.
- Target vehicles use **triple rays** (±15° fan) for wider collision detection.
- Ray lengths incorporate **bounding box extent**: `4 * speed + 2 * bbox_x + 6`.
- **Traffic light awareness**: If the conflicting vehicle has a red light,
  it's assumed to be stopped and not a hazard.
- Only triggers inside/near junctions (at least one vehicle must be at a
  junction waypoint).
- **Max distance**: 20 m (tighter than the standard 30 m).

#### 3c. `_is_lane_vehicle_hazard()` — Lane Change Detection

Another **InterFuser-only** method:

```python
def _is_lane_vehicle_hazard(self, vehicle_list, command):
    # Only active during lane changes
    if command != RoadOption.CHANGELANELEFT and command != RoadOption.CHANGELANERIGHT:
        return []

    # Compute lane boundaries using waypoint lane width
    lane_width = w1.lane_width
    lft_shift = 0.5
    rgt_shift = 0.5
    if command == RoadOption.CHANGELANELEFT:
        rgt_shift += 1
    else:
        lft_shift += 1

    # Create lane boundary line from left to right edge
    lft_lane_wp = rotate_point(Vector3D(lft_shift * lane_width, ...), yaw + 90)
    rgt_lane_wp = rotate_point(Vector3D(rgt_shift * lane_width, ...), yaw - 90)

    for target_vehicle in vehicle_list:
        # Filter: same road, adjacent lane
        if w1.road_id != w2.road_id or w1.lane_id * w2.lane_id < 0:
            continue

        # Detection distance scales with speed + bounding box
        s2_value = max(12, 2 + 2 * x2 + 3.0 * other_speed)

        # Check if target vehicle's projected future path crosses lane boundary
        if self._are_vehicles_crossing_future(p2_hat, s2, lft_lane_wp, rgt_lane_wp):
            res.append(target_vehicle)
```

Uses `shapely.geometry.LineString.intersection()` to check if a target vehicle's
future trajectory (3× velocity vector) crosses the ego's lane boundary.

#### Braking Profile: **Binary**

Despite the much more sophisticated detection, InterFuser still uses **binary
braking**: `brake = True` → `throttle = 0.0`, `steer *= 0.5`. No proportional/
IDM/kinematic braking.

---

### 4. Traffic Light Handling

InterFuser has a **dual detection system**:

#### 4a. CARLA API Detection (same as TCP)

```python
if self._vehicle.get_traffic_light_state() != carla.libcarla.TrafficLightState.Green:
    affecting = self._vehicle.get_traffic_light()
    for light in self._traffic_lights:
        if light.id == affecting.id:
            return [light]
```

#### 4b. Custom Waypoint-Based Detection (InterFuser-only)

```python
light = self._find_closest_valid_traffic_light(self._vehicle.get_location(), min_dis=8)
if light is not None and light.state != carla.libcarla.TrafficLightState.Green:
    return [light]
```

This method uses a **pre-built traffic light dictionary** (`_dict_traffic_lights`)
that maps waypoints to traffic lights:

```python
def _find_closest_valid_traffic_light(self, loc, min_dis):
    wp = self._map.get_waypoint(loc)
    min_distance = min_dis
    for waypoint in self._list_traffic_waypoints:
        if waypoint.road_id != wp.road_id or waypoint.lane_id * wp.lane_id < 0:
            continue  # Must be same road AND same direction
        dis = loc.distance(waypoint.transform.location)
        if dis <= min_distance:
            min_distance = dis
            min_wp = waypoint
    if min_wp is None:
        return None
    else:
        return self._dict_traffic_lights[min_wp][0]
```

**Key features**:
- **Road ID and lane direction filtering**: Only lights on the same road AND
  in the same direction (sign of `lane_id`) are considered.
- Detection range: `min_dis=8 m` for ego braking, `min_dis=50 m` for recording
  the `affected_light_id`, `min_dis=10 m` for checking other vehicles at
  junctions.
- **Traffic light dictionary built at init**: At `_init()`, all traffic lights
  are processed with `get_traffic_light_waypoints()` which discretizes each
  light's trigger volume, finds the corresponding road waypoints, and advances
  them to the intersection boundary.
- **Binary brake**: Still full-brake-or-nothing.

---

### 5. Turn / Junction Handling

Same angle-based approach as TCP but with **slightly lower normal speed**:

```python
should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
target_speed = 4.0 if should_slow else 6.5  # TCP uses 7.0
```

| Condition | Target Speed |
|-----------|-------------|
| `abs(angle_far) > 45°` or `abs(angle_near) > 5°` | 4.0 m/s (14.4 km/h) |
| Otherwise | 6.5 m/s (23.4 km/h) |

Additionally, the junction vehicle hazard detection adjusts the ego ray by
±25° based on the planned turn direction (see §3b above).

The `is_junction` flag is **recorded** in measurements but not directly used
for speed control.

---

### 6. Pedestrian Handling

Identical to TCP:

```python
def _is_walker_hazard(self, walkers_list):
    p1 = _numpy(self._vehicle.get_location())
    v1 = 10.0 * _orientation(self._vehicle.get_transform().rotation.yaw)
    for walker in walkers_list:
        v2_hat = _orientation(walker.get_transform().rotation.yaw)
        s2 = np.linalg.norm(_numpy(walker.get_velocity()))
        if s2 < 0.05:
            v2_hat *= s2
        p2 = -3.0 * v2_hat + _numpy(walker.get_location())
        v2 = 8.0 * v2_hat
        collides, collision_point = get_collision(p1, v1, p2, v2)
        if collides:
            res.append(walker)
    return res
```

Same parameters: ego ray = 10.0, walker ray = 8.0, walker offset = -3.0,
stationary threshold = 0.05 m/s.

---

### 7. Bike Handling (InterFuser-only)

InterFuser adds a dedicated `_is_bike_hazard()` method that's absent from
TCP and TransFuser:

```python
def _is_bike_hazard(self, bikes_list):
    o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
    p1 = _numpy(self._vehicle.get_location())
    v1 = 10.0 * o1

    for bike in bikes_list:
        o2 = _orientation(bike.get_transform().rotation.yaw)
        p2 = _numpy(bike.get_location())

        # Geometric filtering
        if distance > 20:        # Max range 20m (vs 30m for vehicles)
            continue
        if angle_to_car > 30:    # Same ±30° cone
            continue
        if angle_between_heading < 80 and angle_between_heading > 100:
            continue             # Only detect near-perpendicular bikes

        # Ray offset behind bike
        p2_hat = -2.0 * v2_hat + _numpy(bike.get_location())
        v2 = 7.0 * v2_hat       # Shorter ray than walkers (7 vs 8)

        collides, collision_point = get_collision(p1, v1, p2_hat, v2)
```

**Key differences from vehicle detection**:
- Shorter detection range: **20 m** (vs `max(10, 3*speed)` for vehicles)
- Filters bikes with `angle_between_heading` between 80°–100° — only
  considers bikes crossing at near-perpendicular angles
- Shorter ray offset: **-2.0** (vs -3.0 for walkers)
- Shorter projection: **7.0** (vs 8.0 for walkers)

Note: This filters the `*vehicle*` actor list (bikes are CARLA vehicles), so
a bike can trigger both `_is_vehicle_hazard()` and `_is_bike_hazard()`.

---

### 8. Key Parameters

#### Speed Control
| Parameter | Value | Unit |
|---|---|---|
| `target_speed` (slowing) | 4.0 | m/s |
| `target_speed` (normal) | 6.5 | m/s |
| `clip_delta` | 0.25 | m/s |
| `clip_throttle` | 0.75 | 0–1 |
| `should_slow angle_far` | 45.0 | degrees |
| `should_slow angle_near` | 5.0 | degrees |

#### PID Gains
| Controller | K_P | K_I | K_D | Window (n) |
|---|---|---|---|---|
| Turn (steering) | 1.25 | 0.75 | 0.3 | 40 |
| Speed (throttle) | 5.0 | 0.5 | 1.0 | 40 |

#### Vehicle Detection
| Parameter | Value | Notes |
|---|---|---|
| Ego ray `s1` | `max(10, 3.0 * ego_speed)` | Speed-dependent |
| Other ray `s2` | `max(5.0, 2.0 * other_speed)` | Speed-dependent |
| Max `angle_to_car` | 30° | Forward cone |
| Max `angle_between_heading` | 60° | Heading filter |

#### Junction Vehicle Detection (InterFuser-only)
| Parameter | Value | Notes |
|---|---|---|
| Ego ray `v1` | `(4 * s1 + 5) * heading` | With ±25° shift |
| Other ray `v2` | `(4 * s2 + 2*bbox_x + 6) * heading` | Triple-ray ±15° |
| Max distance | 20 m | |
| Heading filter | < 15° or > 165° excluded | Reject aligned/anti-aligned |
| Turn shift angle | ±25° based on command | |

#### Lane Change Vehicle Detection (InterFuser-only)
| Parameter | Value | Notes |
|---|---|---|
| Detection distance | `max(12, 2 + 2*bbox_x + 3*speed)` | |
| Crossing check | `shapely.LineString.intersection()` | |

#### Bike Detection (InterFuser-only)
| Parameter | Value |
|---|---|
| Max distance | 20 m |
| Max `angle_to_car` | 30° |
| `angle_between_heading` filter | 80°–100° only |
| Bike ray offset | -2.0 |
| Bike ray length | 7.0 |

#### Walker Detection
| Parameter | Value |
|---|---|
| Ego ray length | 10.0 (fixed) |
| Walker ray length | 8.0 |
| Walker ray offset behind | -3.0 |
| Stationary threshold | 0.05 m/s |

#### Traffic Light Detection (InterFuser-only additions)
| Parameter | Value |
|---|---|
| CARLA API proximity | `get_nearby_lights()` bounding-box diagonal |
| Custom search `min_dis` (braking) | 8 m |
| Custom search `min_dis` (recording) | 50 m |
| Custom search `min_dis` (other vehicles) | 10 m |
| Waypoint advance step | 0.5 m |

#### Stop Sign Detection
| Parameter | Value |
|---|---|
| `PROXIMITY_THRESHOLD` | 30.0 m |
| `SPEED_THRESHOLD` | 0.1 m/s |
| `WAYPOINT_STEP` | 1.0 m |
| `multi_step` | 20 |

#### Route Planning
| Planner | `min_distance` | `max_distance` |
|---|---|---|
| Near waypoint | 4.0 m | 50 m |
| Far command | 7.5 m | 25 m |

#### Waypoint Disturbance
| Parameter | Default | Config key |
|---|---|---|
| `waypoint_disturb` | 0 | `waypoint_disturb` |
| `waypoint_disturb_seed` | 2021 | `waypoint_disturb_seed` |
| Applied to | LANEFOLLOW / STRAIGHT only | |
| Max perturbation | `0.8 * waypoint_disturb * rand()` | |
| Direction | Random left or right (perpendicular) | |

#### Hazard Actor Destruction
| Parameter | Default | Config key |
|---|---|---|
| `destory_hazard_actors` | `True` | `destory_hazard_actors` |
| Stuck threshold | 1500 ticks at speed < 0.05 m/s | |

#### Data Collection
| Parameter | Default | Config key |
|---|---|---|
| `save_skip_frames` | 10 | `save_skip_frames` |
| Actor data save radius | 50 m (vehicles/walkers), 70 m (lights) | |
| Steer noise | `1e-2 * np.random.randn()` | |

#### Sensors
| Sensor | Position | Resolution | FOV |
|---|---|---|---|
| `rgb_front` | (1.3, 0, 2.3) yaw=0 | 800×600 | 100° |
| `rgb_rear` | (-1.3, 0, 2.3) yaw=180 | 800×600 | 100° |
| `rgb_left` | (1.3, 0, 2.3) yaw=-60 | 800×600 | 100° |
| `rgb_right` | (1.3, 0, 2.3) yaw=60 | 800×600 | 100° |
| `seg_front/left/right` | same as rgb | 400×300 | 100° |
| `depth_front/left/right` | same as rgb | 400×300 | 100° |
| `lidar` | (1.3, 0, 2.5) yaw=-90 | — | — |
| `map` (overhead) | (0, 0, 100) pitch=-90 | 512×512 | 50° |
| IMU | (0, 0, 0) | — | 20 Hz |
| GPS | (0, 0, 0) | — | 100 Hz |
| Speedometer | — | — | 20 Hz |

Note: With `rgb_only=True` (default), only RGB cameras + GPS + IMU + speed are
active. Segmentation, depth, and LiDAR require `rgb_only=False`.

#### Weather
| Type | Count | Details |
|---|---|---|
| Day weathers | 14 | Standard CARLA presets |
| Night weathers | 7 | Custom `WeatherParameters(...)` |
| Total | 21 | |

---

### 9. Collision Handling During Data Collection

- **No collision sensor** in the autopilot code.
- **No explicit collision recovery**.
- **Hazard actor destruction**: If any vehicle or walker is stationary
  (speed < 0.05 m/s) for more than **1500 ticks** (≈75 seconds at 20 Hz),
  it is `destroy()`ed from the simulation. This prevents permanently stuck
  NPCs from blocking data collection indefinitely.
- The Leaderboard framework handles route termination on excessive infractions.
- **Post-hoc tools**: InterFuser provides utilities in `tools/data/`:
  - `batch_stat_blocked_data.py` — find frames where ego is blocked
  - `batch_recollect_blocked_data.py` — delete blocked frames and recollect
  - `batch_merge_data.py` — merge fragmented data items
  - `batch_stat_data.py` — statistical analysis of collected data
  - `batch_preload.py` — precompute waypoints for all frames

---

### 10. Data Saved Per Frame

Saved every `save_skip_frames` frames (default 10):

**Measurements JSON**:
```
gps_x, gps_y, x, y, theta, speed, target_speed,
x_command, y_command, command, gt_command,
steer, throttle, brake,
weather, weather_id,
near_node_x, near_node_y, far_node_x, far_node_y,
is_junction,
is_vehicle_present, is_bike_present,
is_lane_vehicle_present, is_junction_vehicle_present,
is_pedestrian_present, is_red_light_present, is_stop_sign_present,
should_slow, should_brake,
future_waypoints (50 waypoints),
affected_light_id
```

**Actors Data JSON** (separate file):
```
For each actor within 50m (vehicles/walkers) or 70m (traffic lights):
  - loc: [x, y, z]
  - ori: [forward_x, forward_y, forward_z]
  - box: [extent_x, extent_y]
  - vel: [vx, vy, vz]
  - tpe: 0=vehicle, 1=walker, 2=traffic_light
  For traffic lights additionally:
  - sta: int(state)
  - trigger_loc, trigger_ori, trigger_box
```

**Images**: 4× RGB (front/rear/left/right), optionally 3× seg, 3× depth,
LiDAR, BEV birdview, topdown semantic.

---

### 11. Unique Features Not Found in TransFuser or LBC

| # | Feature | Description |
|---|---------|-------------|
| 1 | **Junction vehicle detection** | Dedicated `_is_junction_vehicle_hazard()` with command-aware ray shifting (±25°) and triple-ray (±15°) target projection |
| 2 | **Lane-change vehicle detection** | `_is_lane_vehicle_hazard()` using Shapely geometry for trajectory-lane-boundary intersection |
| 3 | **Bike-specific detection** | `_is_bike_hazard()` tuned for perpendicular crossings |
| 4 | **Road/lane ID filtering** | Uses `waypoint.road_id` and `waypoint.lane_id` to filter irrelevant vehicles |
| 5 | **Custom traffic light search** | `_find_closest_valid_traffic_light()` using pre-built waypoint-to-light dictionary with road/lane filtering |
| 6 | **Traffic-light-aware junction detection** | Ignores conflicting vehicles that have a red light (they should be stopped) |
| 7 | **Hazard actor destruction** | Removes permanently stuck NPCs after 1500 ticks |
| 8 | **Waypoint disturbance** | Lateral perturbation of route waypoints for data diversity |
| 9 | **BEV generation** | `BirdViewProducer` rendering during data collection |
| 10 | **Full actor state recording** | Saves position/velocity/bbox of ALL nearby actors per frame |
| 11 | **Night weather support** | 7 custom night weather conditions |
| 12 | **YAML configuration** | Configurable collection parameters via YAML file |
| 13 | **`gt_command` recording** | Records ground-truth route command alongside computed command |
| 14 | **Future waypoints** | Saves 50 future route waypoints per frame |
| 15 | **`is_alive` check** | Checks `target_vehicle.is_alive` before processing (prevents crashes from destroyed actors) |
| 16 | **Post-collection blocked data tools** | Utilities to detect and recollect frames where ego was stuck |

---

## Part 3: Cross-Comparison — All Four Autopilots

| Feature | LBC | TransFuser | TCP | InterFuser |
|---------|-----|------------|-----|------------|
| **Normal target speed** | ~5.0 m/s | 4.0 m/s | 7.0 m/s | 6.5 m/s |
| **Junction target speed** | ~3.75 m/s | 3.0 m/s | 4.0 m/s | 4.0 m/s |
| **Braking profile** | Binary | Binary | Binary | Binary |
| **Vehicle following model** | None (cone+distance) | None (OBB collision) | None (ray intersection) | None (ray intersection) |
| **Lane filtering** | ✗ | ✗ | ✗ | ✓ (road_id + lane_id) |
| **Junction vehicle detection** | ✗ | ✓ (longer extrapolation) | ✗ | ✓ (dedicated method + triple rays) |
| **Lane change detection** | ✗ | ✗ | ✗ | ✓ (Shapely intersection) |
| **Bike detection** | ✗ | ✗ | ✗ | ✓ (dedicated method) |
| **Collision model** | Yaw+cone | Kinematic bicycle + OBB/SAT | 2D ray intersection | 2D ray intersection |
| **Traffic light detection** | CARLA API + lane match | OBB intersection | CARLA API | CARLA API + waypoint dict |
| **Stop signs** | ✗ | ✓ (optional) | ✓ | ✓ |
| **Stuck vehicle handling** | ✗ | ✓ (30-frame buffer) | ✗ | ✓ (1500-tick destroy) |
| **Slope handling** | ✗ | ✓ | ✗ | ✗ |
| **Safety box (Bremsweg)** | ✗ | ✓ | ✗ | ✗ |
| **GPS denoising** | ✗ | ✓ | ✗ | ✗ |
| **Weather support** | N/A | 14 day | 4 day | 21 (14 day + 7 night) |
| **Waypoint perturbation** | ✗ | ✗ | ✗ | ✓ |
| **BEV generation** | ✗ | ✗ | ✗ | ✓ |
| **Actor state recording** | ✗ | ✗ | ✗ | ✓ (full world state) |
| **Future waypoint recording** | ✗ | ✓ | ✗ | ✓ |
| **Steering noise** | Uniform ±0.25 | Gaussian 1e-3 | Gaussian 1e-2 | Gaussian 1e-2 |
| **Post-collection filtering** | ✗ | ✗ | ✓ (filter_data.py) | ✓ (blocked data tools) |
| **CARLA version** | 0.9.6 | 0.9.10.1 | 0.9.10.1 | 0.9.10.1 |

---

## Part 4: Implications for Our Expert

Our `PrivilegedExpert` already improves on all four in several areas:

| Feature | Best Other | Ours |
|---------|-----------|------|
| Braking | Binary (all four) | Proportional (kinematic/comfort) |
| Vehicle following | Ray intersection / OBB | Speed-dependent hazard distances |
| Lane awareness | InterFuser road_id/lane_id | Frenet frame lateral filtering |
| Junction handling | InterFuser triple-ray | Curvature-based + anticipation |
| Traffic lights | InterFuser waypoint dict | Distance-proportional deceleration |
| Target speed | 7.0 m/s (TCP, highest) | 30 km/h, speed-limit aware |

**Worth adopting from InterFuser**:
1. **Traffic-light-aware junction detection** — skipping vehicles that have a
   red light is a clever optimisation.
2. **Actor state recording** — saving full world state enables richer offline
   analysis and supervision signals.
3. **Hazard actor destruction** — preventing permanently stuck NPCs from
   blocking data collection.
4. **Waypoint disturbance** — lateral perturbation for training diversity.
5. **Future waypoints** — saving planned waypoints for trajectory supervision.

**Not worth adopting**:
1. Binary braking (we already have proportional).
2. The simple ray intersection collision model (we have better).
3. `shapely` dependency for lane change detection (we use Frenet directly).
