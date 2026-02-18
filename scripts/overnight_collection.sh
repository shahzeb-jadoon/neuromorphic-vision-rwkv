#!/usr/bin/env bash
# ============================================================================
# Overnight CARLA Data Collection Campaign
# ============================================================================
#
# Run this script with CARLA server running on Windows:
#   conda activate rwkv
#   bash scripts/overnight_collection.sh
#
# Expected runtime: ~6-10 hours
# Expected output:  ~10+ hours of driving data (720K+ frames)
# Expected storage: ~50-70 GB (gzip-1 compressed)
#
# Monitor progress: tail -f data/collected/collection.log
# Resume after interruption: just re-run the script (manifest-based resume)
# ============================================================================

set -euo pipefail

OUTPUT_DIR="data/collected"
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
# Towns to collect from (Town02 is held out for evaluation)
TOWNS=("Town01" "Town03" "Town04" "Town05")

# Weather conditions
WEATHERS=("ClearNoon" "CloudyNoon" "WetNoon" "SoftRainNoon" "HardRainNoon" "ClearSunset")

# Episodes per (town, weather) combination = 4×6×6 = 144 episodes
EPISODES_PER_CONFIG=6

# Each episode = 5000 frames (250 seconds at 20 FPS)
MAX_FRAMES=5000

# Traffic
N_VEHICLES=40
N_WALKERS=0

# Route generation strategy: balanced turns
ROUTE_STRATEGY="balanced"
TARGET_LEFT=0.15
TARGET_RIGHT=0.15

# Total expected: 144 episodes × 5000 frames = 720,000 frames (≈10 hours)
echo ""
echo "Plan: ${#TOWNS[@]} towns × ${#WEATHERS[@]} weathers × ${EPISODES_PER_CONFIG} episodes"
echo "Total: $(( ${#TOWNS[@]} * ${#WEATHERS[@]} * ${EPISODES_PER_CONFIG} )) episodes"
echo "Frames per episode: ${MAX_FRAMES} ($(( MAX_FRAMES / 20 ))s at 20 FPS)"
echo "Expected total: $(( ${#TOWNS[@]} * ${#WEATHERS[@]} * ${EPISODES_PER_CONFIG} * MAX_FRAMES )) frames"
echo "Route strategy: ${ROUTE_STRATEGY} (left=${TARGET_LEFT}, right=${TARGET_RIGHT})"
echo ""

# --- Run Collection via collect_data.py (uses route generator) ---
for town in "${TOWNS[@]}"; do
    echo "[$(date +%H:%M:%S)] Starting: ${town} (${EPISODES_PER_CONFIG} episodes per weather)" | tee -a "${LOG_FILE}"

    python scripts/collect_data.py \
        --config configs/collection.yaml \
        --output "${OUTPUT_DIR}" \
        --town "${town}" \
        --episodes "${EPISODES_PER_CONFIG}" \
        --route-strategy "${ROUTE_STRATEGY}" \
        --target-left "${TARGET_LEFT}" \
        --target-right "${TARGET_RIGHT}" \
        --resume \
        2>&1 | tee -a "${LOG_FILE}"

    # Small pause between towns to let CARLA settle
    sleep 5
done

echo ""
echo "============================================================"
echo "Campaign finished: $(date)"
echo "============================================================"

# Run analysis
echo ""
echo "Running dataset analysis..."
python scripts/analyze_dataset.py "${OUTPUT_DIR}" --save-samples 10

echo ""
echo "Done! Check ${OUTPUT_DIR} for results."
echo "Manifest: ${MANIFEST}"
