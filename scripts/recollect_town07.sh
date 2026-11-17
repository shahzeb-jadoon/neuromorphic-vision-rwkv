#!/usr/bin/env bash
# ==============================================================================
# Recollect Town07 Data Only
# ==============================================================================
#
# Runs collection for Town07 only, using the stop-sign-trigger-fix branch
# expert code that correctly handles Town07's pole-mounted stop signs.
#
# IMPORTANT: Run this from the phase-1/stop-sign-trigger-fix branch!
#   git checkout phase-1/stop-sign-trigger-fix
#   conda activate rwkv
#   bash scripts/recollect_town07.sh
#
# This script will NOT overwrite existing non-Town07 data.
# New Town07 episodes get unique filenames via the manifest system.
#
# Expected runtime: ~3-5 hours
# Expected storage: ~300-500 GB
# ==============================================================================

set -uo pipefail  # Don't use -e: allow retries without aborting

# --- Verify we're on the correct branch ---
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
if [[ "${CURRENT_BRANCH}" != "phase-1/stop-sign-trigger-fix" ]]; then
    echo "WARNING: Current branch is '${CURRENT_BRANCH}'"
    echo "Expected: phase-1/stop-sign-trigger-fix"
    echo "Town07 stop sign fix is only on that branch."
    read -rp "Continue anyway? [y/N] " response
    if [[ "${response}" != "y" && "${response}" != "Y" ]]; then
        echo "Aborted. Run: git checkout phase-1/stop-sign-trigger-fix"
        exit 1
    fi
fi

# --- Storage location ---
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data/carla_data}"
LOG_FILE="${OUTPUT_DIR}/town07_recollection.log"

# Verify mount
# Verify Town07 backup was removed (or warn)
TOWN07_COUNT=$(python3 -c "
import h5py, glob
count = 0
for f in glob.glob('${OUTPUT_DIR}/episode_*.h5'):
    try:
        with h5py.File(f, 'r') as h:
            town = h['metadata'].attrs.get('town', b'').decode() if isinstance(h['metadata'].attrs.get('town', b''), bytes) else h['metadata'].attrs.get('town', '')
            if town == 'Town07':
                count += 1
    except: pass
print(count)
" 2>/dev/null || echo "0")

if [[ "${TOWN07_COUNT}" -gt 0 ]]; then
    echo "WARNING: Found ${TOWN07_COUNT} existing Town07 episodes in ${OUTPUT_DIR}"
    echo "Run scripts/remove_town07_data.py first to avoid duplicates."
    read -rp "Continue anyway? [y/N] " response
    if [[ "${response}" != "y" && "${response}" != "Y" ]]; then
        echo "Aborted. Run: python scripts/remove_town07_data.py ${OUTPUT_DIR}"
        exit 1
    fi
fi

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "Town07 Recollection Campaign"
echo "Started: $(date)"
echo "Branch:  ${CURRENT_BRANCH}"
echo "Output:  ${OUTPUT_DIR}"
echo "Log:     ${LOG_FILE}"
echo "============================================================"

# --- Configuration ---
# Match overnight_collection.sh settings exactly
EPISODES_PER_CONFIG=6
MAX_FRAMES=4000
ROUTE_STRATEGY="balanced"
TARGET_LEFT=0.15
TARGET_RIGHT=0.15

N_WEATHERS=10
N_DENSITIES=3
TOTAL_EPISODES=$(( N_WEATHERS * N_DENSITIES * EPISODES_PER_CONFIG ))

echo ""
echo "Plan: Town07 × ${N_WEATHERS} weathers × ${N_DENSITIES} densities × ${EPISODES_PER_CONFIG} episodes"
echo "Total: ${TOTAL_EPISODES} episodes"
echo "Frames per episode: ${MAX_FRAMES} ($(( MAX_FRAMES / 20 ))s at 20 FPS)"
echo ""

# --- Run Collection ---
echo "[$(date +%H:%M:%S)] Starting Town07 collection..." | tee -a "${LOG_FILE}"

if python scripts/collect_data.py \
    --config configs/collection.yaml \
    --output "${OUTPUT_DIR}" \
    --town "Town07" \
    --episodes "${EPISODES_PER_CONFIG}" \
    --max-frames "${MAX_FRAMES}" \
    --route-strategy "${ROUTE_STRATEGY}" \
    --target-left "${TARGET_LEFT}" \
    --target-right "${TARGET_RIGHT}" \
    --resume \
    2>&1 | tee -a "${LOG_FILE}"; then
    echo "[$(date +%H:%M:%S)] Town07 collection complete!" | tee -a "${LOG_FILE}"
else
    echo "[$(date +%H:%M:%S)] Town07 collection FAILED" | tee -a "${LOG_FILE}"
    echo "Check log: ${LOG_FILE}"
    exit 1
fi

echo ""
echo "============================================================"
echo "Town07 Recollection finished: $(date)"
echo "============================================================"

# --- Verify the new data ---
echo ""
echo "Verifying new Town07 episodes..."
python3 -c "
import h5py, glob

files = sorted(glob.glob('${OUTPUT_DIR}/episode_*.h5'))
town07 = []
for f in files:
    try:
        with h5py.File(f, 'r') as h:
            town = h['metadata'].attrs.get('town', b'')
            if isinstance(town, bytes):
                town = town.decode()
            if town == 'Town07':
                town07.append(f)
    except:
        pass

print(f'Town07 episodes: {len(town07)}')
print(f'Total episodes in dataset: {len(files)}')

if town07:
    import numpy as np
    # Check a sample episode for stop sign data
    with h5py.File(town07[0], 'r') as h:
        if 'stop_sign_distance' in h:
            ssd = h['stop_sign_distance'][:]
            stops = np.sum(ssd < 100)
            print(f'Sample episode stop sign frames: {stops}/{len(ssd)}')
        print(f'Schema version: {h[\"metadata\"].attrs.get(\"schema_version\", \"?\")}')
"

echo ""
echo "Done! Town07 data recollected with stop-sign-trigger-fix branch."
echo "You can now merge or use this data alongside the rest of the dataset."
