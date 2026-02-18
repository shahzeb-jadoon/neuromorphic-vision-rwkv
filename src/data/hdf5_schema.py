"""HDF5 schema constants for the CARLA driving dataset.

Centralizes field names so that schema changes propagate automatically
to both the writer (ExpertDataCollector) and reader (DrivingSequenceDataset).

Bump SCHEMA_VERSION when adding, removing, or renaming any field.
"""

SCHEMA_VERSION = 1

# --- Dataset field names ---
IMAGES = "images"
STEERING = "steering"
THROTTLE = "throttle"
BRAKE = "brake"
VELOCITY = "velocity"
NAV_COMMAND = "nav_command"

# --- Metadata group and attribute names ---
METADATA_GROUP = "metadata"
META_TOWN = "town"
META_WEATHER = "weather"
META_FPS = "fps"
META_N_FRAMES = "n_frames"
META_STATUS = "status"
META_COLLISION_FRAME = "collision_frame"
META_SCHEMA_VERSION = "schema_version"
