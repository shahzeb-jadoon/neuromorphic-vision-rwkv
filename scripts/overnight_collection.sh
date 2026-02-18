#!/usr/bin/env bash
# ============================================================================
# Overnight CARLA Data Collection Campaign
# ============================================================================
#
# Run this script with CARLA server running on Windows:
#   conda activate rwkv
#   bash scripts/overnight_collection.sh
#
# Expected runtime: ~4-6 hours
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

# Episodes per (town, weather) combination = 4×6×5 = 120 episodes
EPISODES_PER_CONFIG=5

# Each episode = 4000 frames (200 seconds at 20 FPS)
MAX_FRAMES=4000

# Traffic
N_VEHICLES=40
N_WALKERS=0

# Total expected: 120 episodes × 4000 frames = 480,000 frames (≈6.7 hours)
# With additional runs or more episodes_per_config, aim for 720K+ frames
echo ""
echo "Plan: ${#TOWNS[@]} towns × ${#WEATHERS[@]} weathers × ${EPISODES_PER_CONFIG} episodes"
echo "Total: $(( ${#TOWNS[@]} * ${#WEATHERS[@]} * ${EPISODES_PER_CONFIG} )) episodes"
echo "Frames per episode: ${MAX_FRAMES} ($(( MAX_FRAMES / 20 ))s at 20 FPS)"
echo "Expected total: $(( ${#TOWNS[@]} * ${#WEATHERS[@]} * ${EPISODES_PER_CONFIG} * MAX_FRAMES )) frames"
echo ""

# --- Run Collection ---
for town in "${TOWNS[@]}"; do
    for weather in "${WEATHERS[@]}"; do
        echo "[$(date +%H:%M:%S)] Starting: ${town} / ${weather} (${EPISODES_PER_CONFIG} episodes)" | tee -a "${LOG_FILE}"

        python -c "
import logging, time, sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('${LOG_FILE}', mode='a'),
        logging.StreamHandler(sys.stdout),
    ]
)

from src.data.expert_collector import ExpertDataCollector, EpisodeConfig, CollectionManifest
from pathlib import Path

collector = ExpertDataCollector(host='auto', port=2000, output_dir='${OUTPUT_DIR}')
collector.connect()

manifest = CollectionManifest(Path('${MANIFEST}'))

# Check if this (town, weather) is already done
completed = manifest.get_completed_configs()
if ('${town}', '${weather}') in completed:
    print('Skipping ${town}/${weather} — already completed')
else:
    for ep in range(${EPISODES_PER_CONFIG}):
        config = EpisodeConfig(
            town='${town}',
            weather='${weather}',
            max_frames=${MAX_FRAMES},
            n_vehicles=${N_VEHICLES},
            fps=20,
        )
        result = collector.collect_episode(config)
        manifest.add_episode(result)
        manifest.save()
        print(f'  Episode {ep+1}/${EPISODES_PER_CONFIG}: {result.status}, {result.n_frames} frames, nav={result.nav_distribution}')

        # If crashed, try to reconnect
        if result.status == 'crashed' and result.error_message:
            print('  Attempting reconnect after crash...')
            if not collector._reconnect():
                print('  FATAL: Could not reconnect. Skipping remaining episodes.')
                break
"
        # Small pause between town/weather combos to let CARLA settle
        sleep 2
    done
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
