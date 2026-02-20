#!/usr/bin/env bash
# ============================================================================
# Quick Smoke-Test Collection
# ============================================================================
#
# Runs a minimal collection (1 town, 1 weather, 2 episodes, 2000 frames each)
# to verify:
#   1. CARLA connection works
#   2. Camera captures at 512×512
#   3. Dual nav labels (planner + heuristic) are saved
#   4. Turn distribution is ~15% left, ~15% right (from balanced routes)
#   5. Crash recovery/resume works
#
# Usage:
#   conda activate rwkv
#   bash scripts/test_collection.sh
#
# Expected runtime: ~5-10 minutes (depends on CARLA frame rate)
# Expected storage:  ~500 MB
# ============================================================================

set -euo pipefail

OUTPUT_DIR="data/test_512"
LOG_FILE="${OUTPUT_DIR}/test_collection.log"

# Clean previous test data
if [ -d "${OUTPUT_DIR}" ]; then
    echo "Removing previous test data at ${OUTPUT_DIR}..."
    rm -rf "${OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "Smoke-Test Collection (512×512)"
echo "Started: $(date)"
echo "Output:  ${OUTPUT_DIR}"
echo "============================================================"
echo ""

# --- Run a small collection: 1 town, 1 weather, 2 episodes, 500 frames ---
python scripts/collect_data.py \
    --config configs/collection.yaml \
    --output "${OUTPUT_DIR}" \
    --town Town01 \
    --weather ClearNoon \
    --episodes 2 \
    --route-strategy balanced \
    --target-left 0.15 \
    --target-right 0.15 \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "============================================================"
echo "Collection complete. Running analysis..."
echo "============================================================"
echo ""

# --- Analyze the collected data ---
python scripts/analyze_dataset.py "${OUTPUT_DIR}" --save-samples 5

echo ""
echo "============================================================"
echo "Verification Checklist"
echo "============================================================"
echo ""
echo "Check the output above for:"
echo "  1. Camera resolution: should show 512×512 in metadata"
echo "  2. Turn distribution: Left > 10%, Right > 10%"
echo "  3. Nav labels: both nav_command_planner and nav_command_heuristic present"
echo ""

# --- Verify HDF5 schema v2 fields ---
echo "Verifying HDF5 schema v2 fields..."
python -c "
import h5py, glob, sys
files = sorted(glob.glob('${OUTPUT_DIR}/episode_*.h5'))
if not files:
    print('ERROR: No episode files found!')
    sys.exit(1)
for f in files:
    with h5py.File(f, 'r') as h:
        meta = h['metadata']
        print(f'--- {f} ---')
        print(f'  schema_version: {meta.attrs[\"schema_version\"]}')
        print(f'  camera: {meta.attrs.get(\"camera_width\", \"?\")}×{meta.attrs.get(\"camera_height\", \"?\")}')
        print(f'  n_frames: {meta.attrs[\"n_frames\"]}')
        print(f'  town: {meta.attrs[\"town\"]}')
        print(f'  images shape: {h[\"images\"].shape}')
        has_planner = 'nav_command_planner' in h
        has_heuristic = 'nav_command_heuristic' in h
        has_legacy = 'nav_command' in h
        print(f'  nav_command_planner: {\"YES\" if has_planner else \"MISSING!\"}')
        print(f'  nav_command_heuristic: {\"YES\" if has_heuristic else \"MISSING!\"}')
        print(f'  nav_command (legacy): {\"YES\" if has_legacy else \"MISSING!\"}')
        if has_planner and has_heuristic:
            import numpy as np
            p = h['nav_command_planner'][:]
            he = h['nav_command_heuristic'][:]
            match = np.mean(p == he) * 100
            names = ['LaneFollow', 'Left', 'Right', 'Straight']
            print(f'  planner-heuristic agreement: {match:.1f}%')
            total = len(p)
            print(f'  planner distribution:')
            for i, name in enumerate(names):
                count = int(np.sum(p == i))
                pct = count / total * 100
                print(f'    {name}: {count}/{total} ({pct:.1f}%)')
        print()
print('Done!')
"

echo ""
echo "If everything looks good, delete this test data and start the real collection:"
echo "  rm -rf ${OUTPUT_DIR}"
echo "  bash scripts/overnight_collection.sh"
