"""HDF5 schema constants for the CARLA driving dataset.

Centralizes field names so that schema changes propagate automatically
to both the writer (ExpertDataCollector) and reader (DrivingSequenceDataset).

Bump SCHEMA_VERSION when adding, removing, or renaming any field.

Schema history
--------------
v1  Single camera, heuristic nav label only.
v2  Added dual nav labels (planner + heuristic).
v3  3-camera panoramic, custom expert (PrivilegedExpert), GPS/position
    data, traffic light state, nearby-actor bounding info.  Dropped
    legacy dual labels in favour of a single expert-derived nav command.
"""

SCHEMA_VERSION = 3

# --- Image data ---
# Schema v3: panoramic image stitched from 3 cameras (H, W_total, 3) uint8.
# W_total = 3 × per-camera width (left | centre | right).
IMAGES = "images"

# --- Control signals ---
STEERING = "steering"
THROTTLE = "throttle"
BRAKE = "brake"

# --- Navigation ---
# Single ground-truth nav command from the PrivilegedExpert's route.
# Values match agents.navigation.local_planner.RoadOption:
#   LEFT=1, RIGHT=2, STRAIGHT=3, LANEFOLLOW=4
NAV_COMMAND = "nav_command"

# Valid nav command IDs for schema v3 (used by validation / analysis scripts).
# 1-4 are from RoadOption enum; 5-6 added for lane change support.
VALID_NAV_COMMANDS = {1, 2, 3, 4, 5, 6}

# Human-readable names for reporting and visualisation.
NAV_COMMAND_NAMES: dict[int, str] = {
    1: "Left",
    2: "Right",
    3: "Straight",
    4: "LaneFollow",
    5: "ChangeLaneLeft",
    6: "ChangeLaneRight",
}

# Legacy aliases kept for backward-compatible readers of v2 files.
NAV_COMMAND_PLANNER = "nav_command_planner"
NAV_COMMAND_HEURISTIC = "nav_command_heuristic"

# --- Ego-vehicle state (new in v3) ---
VELOCITY = "velocity"  # (N, 3)  float32  m/s  (vx, vy, vz)
POSITION = "position"  # (N, 3)  float32  metres  (x, y, z) world
ROTATION = "rotation"  # (N, 3)  float32  degrees (pitch, yaw, roll)

# --- Privileged / future-proofing (new in v3) ---
TRAFFIC_LIGHT_STATE = "traffic_light_state"  # (N,)    int32  0=None 1=G 2=Y 3=R
TRAFFIC_LIGHT_DISTANCE = "traffic_light_distance"  # (N,)  float32  metres (-1 if none)
STOP_SIGN_DISTANCE = "stop_sign_distance"  # (N,)    float32  metres (-1 if none)
YIELD_SIGN_DISTANCE = "yield_sign_distance"  # (N,)   float32  metres (-1 if none)
TARGET_SPEED = "target_speed"  # (N,)    float32  km/h
DRIVING_MODE = "driving_mode"  # (N,)    int32  DrivingMode enum
NEARBY_VEHICLES = "nearby_vehicles"  # JSON string per frame
NEARBY_PEDESTRIANS = "nearby_pedestrians"  # JSON string per frame

# --- Corridor diagnostics (new in v3.1) ---
# Lightweight per-frame metrics for verifying SAT corridor health.
STATIONARY = "stationary"  # (N,) bool  speed < 0.5 km/h

N_CORRIDOR_SEGMENTS = "n_corridor_segments"  # (N,) int32
WP_BUFFER_SIZE = "wp_buffer_size"  # (N,) int32
WP_BUFFER_HEAD_DIST = "wp_buffer_head_dist"  # (N,) float32 metres
SAT_DETECTIONS = "sat_detections"  # (N,) int32
LATERAL_DETECTIONS = "lateral_detections"  # (N,) int32
GAP_FILL_DIVERGED = "gap_fill_diverged"  # (N,) int32 count of junction gaps that diverged
ROUTE_COMPLETION = "route_completion"  # (N,) float32 route progress 0.0–1.0

# --- Metadata group and attribute names ---
METADATA_GROUP = "metadata"
META_TOWN = "town"
META_WEATHER = "weather"
META_FPS = "fps"
META_N_FRAMES = "n_frames"
META_STATUS = "status"
META_COLLISION_FRAME = "collision_frame"
META_SCHEMA_VERSION = "schema_version"
META_CAMERA_WIDTH = "camera_width"
META_CAMERA_HEIGHT = "camera_height"
META_N_CAMERAS = "n_cameras"
META_CAMERA_FOV = "camera_fov"
META_CAMERA_YAWS = "camera_yaws"  # e.g. [-60, 0, 60]
META_EXPERT_TYPE = "expert_type"  # "privileged_expert"
META_EXPERT_TARGET_SPEED = "expert_target_speed"
