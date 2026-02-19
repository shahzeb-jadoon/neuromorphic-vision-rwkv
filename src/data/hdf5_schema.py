"""HDF5 schema constants for the CARLA driving dataset.

Centralizes field names so that schema changes propagate automatically
to both the writer (ExpertDataCollector) and reader (DrivingSequenceDataset).

Bump SCHEMA_VERSION when adding, removing, or renaming any field.
"""

SCHEMA_VERSION = 2

# --- Dataset field names ---
IMAGES = "images"
STEERING = "steering"
THROTTLE = "throttle"
BRAKE = "brake"
VELOCITY = "velocity"
NAV_COMMAND = "nav_command"  # Legacy (schema v1): heuristic-inferred label
NAV_COMMAND_PLANNER = "nav_command_planner"  # Schema v2: ground-truth from route planner
NAV_COMMAND_HEURISTIC = "nav_command_heuristic"  # Schema v2: heuristic for comparison

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
