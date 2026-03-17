#!/usr/bin/env bash
# ============================================================================
# Overnight CARLA Data Collection Campaign
# ============================================================================
#
# Run this script with CARLA server running on Windows:
#   conda activate rwkv
#   bash scripts/overnight_collection.sh
#
# Campaign layout (derived from configs/collection.yaml):
#   9 towns × 10 weathers × 3 traffic densities × 6 episodes = 1,620 episodes
#   Each episode: 4000 frames (200 s at 20 FPS)
#   Total: ~90 hours of driving data (~11.7 M frames)
#
# Expected runtime: ~24-48 hours (depends on CARLA frame rate)
# Expected storage: ~1-2 TB at 1920×320 panoramic (gzip-1)
#
# Monitor progress: tail -f data/collected/collection.log
# Resume after interruption: just re-run the script (manifest-based resume)
# Move data to NFS and re-run to continue on larger storage.
# ============================================================================

set -uo pipefail  # Don't use -e: allow per-town failures without aborting

# --- Storage location ---
# Default: the dataset directory below
# Override: OUTPUT_DIR=/path/to/dir bash scripts/overnight_collection.sh
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data/carla_data}"
LOG_FILE="${OUTPUT_DIR}/collection.log"
MANIFEST="${OUTPUT_DIR}/collection_manifest.json"

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "CARLA Overnight Collection Campaign"
echo "Started: $(date)"
echo "Output:  ${OUTPUT_DIR}"
echo "Log:     ${LOG_FILE}"
echo "============================================================"

# --- Configuration ---
# Towns to collect from (must match configs/collection.yaml towns list).
# Town02 is held out for evaluation — do NOT add it here.
# Town10HD is the correct CARLA 0.9.16 name for Town10.
# Town06, Town07, Town12, Town15 require the AdditionalMaps package.
TOWNS=("Town01" "Town03" "Town04" "Town05" "Town06" "Town07" "Town10HD" "Town12" "Town15")

# Episodes per (town, weather, traffic-density) combination.
# Weathers (10) and traffic densities (3) come from collection.yaml
# and are NOT overridden here.
# Total: 9 towns × 10 weathers × 3 densities × 6 episodes = 1,620 episodes (~90 hrs)
EPISODES_PER_CONFIG=6

# Each episode = 4000 frames (200 seconds at 20 FPS)
MAX_FRAMES=4000

# Route generation strategy: balanced turns
ROUTE_STRATEGY="balanced"
TARGET_LEFT=0.15
TARGET_RIGHT=0.15

N_WEATHERS=10   # from collection.yaml (for display only)
N_DENSITIES=3   # from collection.yaml (for display only)
TOTAL_EPISODES=$(( ${#TOWNS[@]} * N_WEATHERS * N_DENSITIES * EPISODES_PER_CONFIG ))
echo ""
echo "Plan: ${#TOWNS[@]} towns × ${N_WEATHERS} weathers × ${N_DENSITIES} densities × ${EPISODES_PER_CONFIG} episodes"
echo "Total: ${TOTAL_EPISODES} episodes"
echo "Frames per episode: ${MAX_FRAMES} ($(( MAX_FRAMES / 20 ))s at 20 FPS)"
echo "Expected total: $(( TOTAL_EPISODES * MAX_FRAMES )) frames (~$(( TOTAL_EPISODES * MAX_FRAMES / 20 / 3600 )) hrs)"
echo "Route strategy: ${ROUTE_STRATEGY} (left=${TARGET_LEFT}, right=${TARGET_RIGHT})"
echo ""

# --- Run Collection via collect_data.py (uses route generator) ---
# Each town runs as its own collect_data.py invocation for isolation.
# Weathers come from configs/collection.yaml (not overridden here).
# Resume is automatic: manifest tracks completed episodes.
FAILED_TOWNS=()
for town in "${TOWNS[@]}"; do
    echo "[$(date +%H:%M:%S)] Starting: ${town} (${EPISODES_PER_CONFIG} episodes per weather×density)" | tee -a "${LOG_FILE}"

    if python scripts/collect_data.py \
        --config configs/collection.yaml \
        --output "${OUTPUT_DIR}" \
        --town "${town}" \
        --episodes "${EPISODES_PER_CONFIG}" \
        --max-frames "${MAX_FRAMES}" \
        --route-strategy "${ROUTE_STRATEGY}" \
        --target-left "${TARGET_LEFT}" \
        --target-right "${TARGET_RIGHT}" \
        --resume \
        2>&1 | tee -a "${LOG_FILE}"; then
        echo "[$(date +%H:%M:%S)] ✓ ${town} complete" | tee -a "${LOG_FILE}"
    else
        echo "[$(date +%H:%M:%S)] ✗ ${town} FAILED — continuing with next town" | tee -a "${LOG_FILE}"
        FAILED_TOWNS+=("${town}")
    fi

    # Pause between towns to let CARLA settle after map change
    sleep 10
done

echo ""
echo "============================================================"
echo "Campaign finished: $(date)"
if [ ${#FAILED_TOWNS[@]} -gt 0 ]; then
    echo "FAILED TOWNS: ${FAILED_TOWNS[*]}"
    echo "Re-run this script to retry (resume is automatic)."
fi
echo "============================================================"

# Run analysis
echo ""
echo "Running dataset analysis..."
python scripts/analyze_dataset.py "${OUTPUT_DIR}" --save-samples 10

echo ""
echo "Done! Check ${OUTPUT_DIR} for results."
echo "Manifest: ${MANIFEST}"
