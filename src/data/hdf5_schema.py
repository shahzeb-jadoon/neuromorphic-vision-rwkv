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
v4  Added explicit nav telemetry channels (route-raw and speed-planning
    commands) while keeping NAV_COMMAND as the primary training label.
v5  Added explicit wp_buffer head waypoint telemetry (route index, road id,
    lane id) for one-glance route-head forensics.
v6  Added arc/policy interaction telemetry channels for policy-vs-steering
    diagnostics (arc activity, policy block streak, steering source).
"""

SCHEMA_VERSION = 6

# --- Image data ---
# Schema v3: panoramic image stitched from 3 cameras (H, W_total, 3) uint8.
# W_total = 3 × per-camera width (left | centre | right).
IMAGES = "images"

# --- Control signals ---
STEERING = "steering"
THROTTLE = "throttle"
BRAKE = "brake"

# --- Navigation ---
# Primary training label after planner-policy filtering.
# Values match agents.navigation.local_planner.RoadOption:
#   LEFT=1, RIGHT=2, STRAIGHT=3, LANEFOLLOW=4
NAV_COMMAND = "nav_command"

# Extra telemetry channels for diagnostics (v4):
# NAV_COMMAND_ROUTE is the raw route-local command before policy filtering.
# NAV_COMMAND_SPEED is the command used by speed planning after safety guards.
NAV_COMMAND_ROUTE = "nav_command_route"
NAV_COMMAND_SPEED = "nav_command_speed"

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
TRAFFIC_LIGHT_ACTOR_ID = "traffic_light_actor_id"  # (N,) int32 TL actor id (-1 if none)
TRAFFIC_LIGHT_STOP_LINE_POSITION = (
    "traffic_light_stop_line_position"  # (N,3) float32 stop-line xyz (-1s if unknown)
)
STOP_SIGN_DISTANCE = "stop_sign_distance"  # (N,)    float32  metres (-1 if none)
YIELD_SIGN_DISTANCE = "yield_sign_distance"  # (N,)   float32  metres (-1 if none)
TARGET_SPEED = "target_speed"  # (N,)    float32  km/h
DRIVING_MODE = "driving_mode"  # (N,)    int32  DrivingMode enum
NEARBY_VEHICLES = "nearby_vehicles"  # JSON string per frame
NEARBY_PEDESTRIANS = "nearby_pedestrians"  # JSON string per frame

# Codes written by the expert (agents.expert.privileged_expert): DrivingMode
# members in declaration order, and PrivilegedExpert._encode_tl.
DRIVING_MODE_NAMES: dict[int, str] = {
    1: "FollowLane",
    2: "ApproachingTurn",
    3: "Turning",
    4: "TrafficLightStop",
    5: "VehicleFollowing",
    6: "PedestrianStop",
    7: "EmergencyStop",
    8: "StopSignStop",
    9: "YieldStop",
}
VALID_DRIVING_MODES = set(DRIVING_MODE_NAMES)
TRAFFIC_LIGHT_STATE_NAMES: dict[int, str] = {0: "None", 1: "Green", 2: "Yellow", 3: "Red"}

# --- Corridor diagnostics (new in v3.1) ---
# Lightweight per-frame metrics for verifying SAT corridor health.
STATIONARY = "stationary"  # (N,) bool  speed < 0.5 km/h

N_CORRIDOR_SEGMENTS = "n_corridor_segments"  # (N,) int32
WP_BUFFER_SIZE = "wp_buffer_size"  # (N,) int32
WP_BUFFER_HEAD_DIST = "wp_buffer_head_dist"  # (N,) float32 metres
WP_BUFFER_HEAD_INDEX = "wp_buffer_head_index"  # (N,) int32 route index (-1 if synthetic/unknown)
WP_BUFFER_HEAD_ROAD_ID = "wp_buffer_head_road_id"  # (N,) int32
WP_BUFFER_HEAD_LANE_ID = "wp_buffer_head_lane_id"  # (N,) int32
ARC_ACTIVE = "arc_active"  # (N,) int8 1 if Bezier arc buffer active
ARC_SIZE = "arc_size"  # (N,) int32 number of waypoints in _arc_buffer
POLICY_BLOCK_STREAK = "policy_block_streak"  # (N,) int32 consecutive policy blocks
STEER_FROM_ARC = "steer_from_arc"  # (N,) int8 1 if steering target came from arc
SAT_DETECTIONS = "sat_detections"  # (N,) int32
LATERAL_DETECTIONS = "lateral_detections"  # (N,) int32
GAP_FILL_RPCS = "gap_fill_rpcs"  # (N,) int32 cumulative gap-fill RPC attempts
GAP_FILL_DIVERGED = "gap_fill_diverged"  # (N,) int32 count of junction gaps that diverged
ROUTE_COMPLETION = "route_completion"  # (N,) float32 route progress 0.0–1.0

# --- Planner policy diagnostics (waypoint checklist instrumentation) ---
PLANNER_POLICY_REASON = "planner_policy_reason"  # (N,) int32 PlannerPolicyReason enum
PLANNER_POLICY_EXCEPTION = "planner_policy_exception"  # (N,) int32 PlannerPolicyException enum
DIST_TO_NEXT_JUNCTION = "dist_to_next_junction"  # (N,) float32 metres (-1 if unknown)
PRE_JUNCTION_LANE_CHANGE_VIOLATION = "pre_junction_lane_change_violation"  # (N,) int8
CONSECUTIVE_LANE_CHANGE_VIOLATION = "consecutive_lane_change_violation"  # (N,) int8
STRAIGHT_EXIT_LANE_VIOLATION = "straight_exit_lane_violation"  # (N,) int8
TURN_LANE_LEGALITY_VIOLATION = "turn_lane_legality_violation"  # (N,) int8
ROUNDABOUT_LANE_CHANGE_VIOLATION = "roundabout_lane_change_violation"  # (N,) int8
LEGALITY_WRONG_WAY_FILTERED = "legality_wrong_way_filtered"  # (N,) int32 cumulative
LEGALITY_DO_NOT_ENTER_FILTERED = "legality_do_not_enter_filtered"  # (N,) int32 cumulative
LEGALITY_TURN_MISMATCH_FILTERED = "legality_turn_mismatch_filtered"  # (N,) int32 cumulative
LEGALITY_CONTINUITY_KEPT = "legality_continuity_kept"  # (N,) int32 cumulative
ROUTE_WATCHDOG_TRIGGERED = "route_watchdog_triggered"  # (N,) int8 frame-level watchdog trip

# --- Training caches (scripts/build_testbed_cache.py) ---
# A cache keeps every signal of its source episode at full rate and one JPEG image every
# META_IMAGE_STRIDE source frames; src.data.schema_adapter reads it as a lower-rate episode.
IMAGES_JPEG = "images_jpeg"  # (N_images,) vlen uint8, one JPEG per image frame
META_CACHE_FORMAT = "cache_format"  # e.g. "jpeg-q90-960x160-stride2"
META_IMAGE_STRIDE = "image_stride"  # source frames per cached image
META_SOURCE_FILE = "source_file"

# A golden-subset excerpt (scripts/extract_golden.py) names its source episode and the
# first source frame it copies.
META_GOLDEN_SOURCE_FILE = "golden_source_file"
META_GOLDEN_SOURCE_START = "golden_source_start"

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
