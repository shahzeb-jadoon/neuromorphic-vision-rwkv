#!/usr/bin/env bash
# ==============================================================================
# Quick Smoke-Test Collection (Schema v3)
# ==============================================================================
#
# Runs a minimal collection across 2 towns (1 episode per traffic config)
# to verify:
#   1. CARLA connection works
#   2. 3-camera panoramic captures at 1920×320 (3×640×320)
#   3. RoadOption nav commands saved (1=Left, 2=Right, 3=Straight, 4=LaneFollow)
#   4. Turn distribution is ~15% left, ~15% right (from balanced routes)
#   5. Metadata: position, rotation, traffic light, nearby actors
#
# Usage:
#   conda activate rwkv
#   bash scripts/test_collection.sh
#
# Expected runtime: ~5-10 minutes (depends on CARLA frame rate)
# Expected storage:  ~500 MB
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [ -n "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
else
    export PYTHONPATH="${REPO_ROOT}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-data/test_collection}"
CONFIG_PATH="${CONFIG_PATH:-configs/collection.yaml}"
LOG_FILE="${OUTPUT_DIR}/test_collection.log"

# Clean previous test data
if [ -d "${OUTPUT_DIR}" ]; then
    echo "Removing previous test data at ${OUTPUT_DIR}..."
    rm -rf "${OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}"

echo "=================================================================="
echo "Smoke-Test Collection (Schema v3 — 3-Camera Panoramic)"
echo "Started: $(date)"
echo "Output:  ${OUTPUT_DIR}"
echo "=================================================================="
echo ""

# --- Run a smoke collection: 7 towns, 1 weather, 1 episode per traffic config ---
# collect_data.py iterates towns × weathers × traffic_configs × episodes,
# so 7 towns × 1 weather × 3 traffic densities × 1 episode = 21 episodes total.
python scripts/collect_data.py \
    --config "${CONFIG_PATH}" \
    --output "${OUTPUT_DIR}" \
    --town Town01 Town03 Town04 Town05 Town06 Town07 Town10HD \
    --weather ClearNoon \
    --episodes 1 \
    --spectator \
    --debug-draw \
    --route-strategy balanced \
    --target-left 0.15 \
    --target-right 0.15 \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "=================================================================="
echo "Collection complete. Running validation + analysis..."
echo "=================================================================="
echo ""

# --- Validate dataset integrity ---
python scripts/validate_dataset.py "${OUTPUT_DIR}"

echo ""

# --- Analyze the collected data ---
python scripts/analyze_dataset.py "${OUTPUT_DIR}" --save-samples 5

echo ""

# --- Visualize trajectories ---
echo "=================================================================="
echo "Generating trajectory visualizations..."
echo "=================================================================="
python scripts/visualize_routes.py "${OUTPUT_DIR}" --save "${OUTPUT_DIR}/plots"

echo ""
echo "=================================================================="
echo "Verification Checklist"
echo "=================================================================="
echo ""
echo "Check the output above for:"
echo "  1. Camera: 3-camera panoramic, 1920×320 (3×640×320)"
echo "  2. Turn distribution: Left > 10%, Right > 10%"
echo "  3. Nav commands: 1=Left, 2=Right, 3=Straight, 4=LaneFollow"
echo "  4. Schema version: 3"
echo "  5. No validation errors"
echo ""

# --- Verify HDF5 schema v3 fields ---
echo "Verifying HDF5 schema v3 fields..."
python -c "
import h5py, glob, sys, numpy as np
from scripts.create_toy_dataset import NAV_COMMANDS

files = sorted(glob.glob('${OUTPUT_DIR}/episode_*.h5'))
if not files:
    print('ERROR: No episode files found!')
    sys.exit(1)

NAV_NAMES = dict(NAV_COMMANDS)
all_ok = True

for f in files:
    with h5py.File(f, 'r') as h:
        meta = h['metadata']
        schema_v = meta.attrs.get('schema_version', '?')
        n_frames = meta.attrs.get('n_frames', '?')
        cam_w = meta.attrs.get('camera_width', '?')
        cam_h = meta.attrs.get('camera_height', '?')
        town = meta.attrs.get('town', '?')

        print(f'--- {f} ---')
        print(f'  schema_version: {schema_v}')
        print(f'  per-camera: {cam_w}x{cam_h}')
        print(f'  n_frames: {n_frames}')
        print(f'  town: {town}')

        # Check images shape
        img_shape = h['images'].shape
        print(f'  images shape: {img_shape}')
        if len(img_shape) == 4 and img_shape[2] == int(cam_w) * 3:
            print(f'  panoramic width: {img_shape[2]} = 3 x {cam_w} ✓')
        else:
            print(f'  WARNING: unexpected image shape!')
            all_ok = False

        # Check nav commands (schema v3: single nav_command, values defined by NAV_COMMANDS)
        if 'nav_command' in h:
            nav = h['nav_command'][:]
            unique = set(nav.tolist())
            valid = set(NAV_NAMES.keys())
            if unique <= valid:
                print(f'  nav_command values: {sorted(unique)} ✓')
            else:
                print(f'  WARNING: unexpected nav values: {unique - valid}')
                all_ok = False

            total = len(nav)
            print(f'  nav distribution:')
            for cmd_id in sorted(NAV_NAMES.keys()):
                count = int(np.sum(nav == cmd_id))
                pct = count / total * 100
                print(f'    {NAV_NAMES[cmd_id]}: {count}/{total} ({pct:.1f}%)')
        else:
            print(f'  WARNING: nav_command dataset missing!')
            all_ok = False

        # Check privileged metadata
        for key in ['position', 'rotation', 'velocity', 'traffic_light_state',
                     'stop_sign_distance', 'driving_mode', 'target_speed',
                     'stationary', 'n_corridor_segments', 'wp_buffer_size',
                     'wp_buffer_head_dist', 'sat_detections', 'lateral_detections']:
            status = '✓' if key in h else 'MISSING!'
            print(f'  {key}: {status}')
            if key not in h:
                all_ok = False

        print()

if all_ok:
    print('All checks passed! ✓')
else:
    print('Some checks failed — review output above.')
    sys.exit(1)
"

echo ""
echo "If everything looks good, delete this test data and start the real collection:"
echo "  rm -rf ${OUTPUT_DIR}"
echo "  bash scripts/overnight_collection.sh"
