#!/usr/bin/env bash
# ============================================================
# WSL2-side memory monitor — runs alongside tests to capture
# page cache, process RSS, and MemAvailable over time.
#
# Usage:
#   # In one WSL2 terminal (start BEFORE running your test):
#   bash scripts/monitor_wsl_memory.sh [interval_seconds] [output_csv]
#
#   # In another WSL2 terminal:
#   bash scripts/test_collection.sh
#
#   # Press Ctrl+C when done. CSV has all the data.
# ============================================================

INTERVAL=${1:-5}
OUTPUT=${2:-"data/wsl_memory_monitor.csv"}

mkdir -p "$(dirname "$OUTPUT")"

echo "============================================================"
echo "  WSL2 Memory Monitor"
echo "============================================================"
echo "  Output:   $OUTPUT"
echo "  Interval: ${INTERVAL}s"
echo "  Ctrl+C to stop"
echo ""

# CSV header
echo "timestamp,elapsed_s,mem_total_mb,mem_free_mb,mem_available_mb,cached_mb,buffers_mb,active_mb,inactive_mb,dirty_mb,python_rss_mb,page_cache_pct" > "$OUTPUT"

START=$(date +%s)

printf "%-20s %8s %8s %8s %8s %8s %8s\n" "Timestamp" "MemAvail" "Cached" "Dirty" "PyRSS" "Cache%" "FreeMB"
printf "%-20s %8s %8s %8s %8s %8s %8s\n" "--------------------" "--------" "--------" "--------" "--------" "--------" "--------"

cleanup() {
    echo ""
    echo "============================================================"
    echo "  Monitor stopped. CSV saved to: $OUTPUT"
    echo "  Samples: $SAMPLE_COUNT"
    echo "============================================================"

    # Quick summary
    if command -v awk &>/dev/null && [ -f "$OUTPUT" ]; then
        echo ""
        awk -F',' 'NR>1 {
            if ($5+0 > max_cache) max_cache=$5;
            if ($5+0 < min_cache || min_cache==0) min_cache=$5;
            if ($9+0 > max_rss) max_rss=$9;
            total_cache+=$5; count++
        } END {
            printf "  Peak page cache:   %.0f MB\n", max_cache
            printf "  Min page cache:    %.0f MB\n", min_cache
            printf "  Avg page cache:    %.0f MB\n", total_cache/count
            printf "  Peak Python RSS:   %.0f MB\n", max_rss
        }' "$OUTPUT" 2>/dev/null
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM

SAMPLE_COUNT=0

while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START))
    TS=$(date "+%Y-%m-%d %H:%M:%S")

    # Parse /proc/meminfo (values in kB)
    eval "$(awk '
        /^MemTotal:/     { printf "MT=%d;", $2 }
        /^MemFree:/      { printf "MF=%d;", $2 }
        /^MemAvailable:/ { printf "MA=%d;", $2 }
        /^Buffers:/      { printf "BF=%d;", $2 }
        /^Cached:/       { printf "CA=%d;", $2 }
        /^Active:/       { if (!seen_active) { printf "AC=%d;", $2; seen_active=1 } }
        /^Inactive:/     { if (!seen_inactive) { printf "IN=%d;", $2; seen_inactive=1 } }
        /^Dirty:/        { printf "DI=%d;", $2 }
    ' /proc/meminfo)"

    # Convert kB to MB
    MT_MB=$((MT / 1024))
    MF_MB=$((MF / 1024))
    MA_MB=$((MA / 1024))
    CA_MB=$((CA / 1024))
    BF_MB=$((BF / 1024))
    AC_MB=$((AC / 1024))
    IN_MB=$((IN / 1024))
    DI_MB=$((DI / 1024))

    # Page cache as % of total
    if [ "$MT" -gt 0 ]; then
        CACHE_PCT=$(awk "BEGIN { printf \"%.1f\", ($CA + $BF) / $MT * 100 }")
    else
        CACHE_PCT="0.0"
    fi

    # Sum RSS of all python processes (kB → MB)
    PY_RSS_KB=$(ps -C python,python3 -o rss= 2>/dev/null | awk '{s+=$1} END {print s+0}')
    PY_RSS_MB=$((PY_RSS_KB / 1024))

    # Color coding for terminal
    if [ "$CA_MB" -gt 10000 ]; then
        COLOR="\033[31m"  # red
    elif [ "$CA_MB" -gt 5000 ]; then
        COLOR="\033[33m"  # yellow
    else
        COLOR="\033[32m"  # green
    fi
    RESET="\033[0m"

    printf "${COLOR}%-20s %7dM %7dM %7dM %7dM %6s%% %7dM${RESET}\n" \
        "$TS" "$MA_MB" "$CA_MB" "$DI_MB" "$PY_RSS_MB" "$CACHE_PCT" "$MF_MB"

    # CSV line
    echo "$TS,$ELAPSED,$MT_MB,$MF_MB,$MA_MB,$CA_MB,$BF_MB,$AC_MB,$IN_MB,$DI_MB,$PY_RSS_MB,$CACHE_PCT" >> "$OUTPUT"

    SAMPLE_COUNT=$((SAMPLE_COUNT + 1))
    sleep "$INTERVAL"
done
