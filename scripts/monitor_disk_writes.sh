#!/usr/bin/env bash
# ============================================================
# WSL2 Disk Write Monitor — tracks per-process disk I/O rates
# to identify which processes are writing the most data.
#
# Reads /proc/[pid]/io for write_bytes per process and
# /proc/diskstats for system-wide disk writes. Reports the
# top N writers each interval with MB/s write rates.
#
# Usage:
#   # Start BEFORE running your workload:
#   bash scripts/monitor_disk_writes.sh [interval_seconds] [output_csv] [top_n]
#
#   # Press Ctrl+C to stop. CSV has all the data.
#
# Notes:
#   - /proc/[pid]/io requires same-user or root access
#   - CARLA (Windows-side) writes are tracked by the companion
#     PowerShell monitor_vmmemwsl.ps1 script
# ============================================================

INTERVAL=${1:-8}
OUTPUT=${2:-"data/disk_write_monitor.csv"}
TOP_N=${3:-5}

mkdir -p "$(dirname "$OUTPUT")"

echo "============================================================"
echo "  WSL2 Disk Write Monitor"
echo "============================================================"
echo "  Output:      $OUTPUT"
echo "  Interval:    ${INTERVAL}s"
echo "  Top writers: ${TOP_N}"
echo "  Ctrl+C to stop"
echo ""

# CSV header — fixed columns for top writers
# Each top-N writer gets: name, pid, write_total_mb, write_rate_mbs
HEADER="timestamp,elapsed_s,sys_write_total_mb,sys_write_rate_mbs,dirty_mb,writeback_mb"
for i in $(seq 1 "$TOP_N"); do
    HEADER="${HEADER},top${i}_name,top${i}_pid,top${i}_total_mb,top${i}_rate_mbs"
done
echo "$HEADER" > "$OUTPUT"

START=$(date +%s)
SAMPLE_COUNT=0

# Store previous write_bytes per PID for rate calculation
declare -A PREV_WRITE
PREV_SYS_WRITE=0

# System-wide write bytes from /proc/diskstats
# Sum write_sectors (column 10, 0-indexed) across all block devices × 512 bytes
get_sys_write_bytes() {
    awk '{ total += $10 * 512 } END { printf "%.0f", total }' /proc/diskstats 2>/dev/null
}

# Collect per-process write_bytes and compute rates.
# Writes output to $TOP_WRITERS_FILE (one line per top writer).
# MUST be called in the main shell (not a subshell) so PREV_WRITE persists.
# Output format per line: PID COMM TOTAL_MB RATE_MBS RAW_RATE_BYTES
TOP_WRITERS_FILE=$(mktemp)

collect_process_writes() {
    local tmpfile
    tmpfile=$(mktemp)

    for io_file in /proc/[0-9]*/io; do
        local pid_dir
        pid_dir=$(dirname "$io_file")
        local pid
        pid=$(basename "$pid_dir")

        # Read write_bytes from /proc/[pid]/io
        local wb
        wb=$(awk -F': ' '/^write_bytes:/ { print $2 }' "$io_file" 2>/dev/null)
        [ -z "$wb" ] && continue

        # Get process name (comm)
        local comm
        comm=$(cat "${pid_dir}/comm" 2>/dev/null || echo "unknown")

        # Calculate rate if we have a previous sample
        local rate_bytes=0
        local prev=${PREV_WRITE[$pid]:-0}
        if [ "$prev" -gt 0 ] && [ "$wb" -ge "$prev" ]; then
            rate_bytes=$(( (wb - prev) ))
        fi

        # Store current for next iteration (persists because we're in main shell)
        PREV_WRITE[$pid]=$wb

        # Only include processes that have written something
        if [ "$wb" -gt 0 ]; then
            local total_mb
            total_mb=$(awk "BEGIN { printf \"%.2f\", $wb / 1048576 }")
            local rate_mbs
            rate_mbs=$(awk "BEGIN { printf \"%.3f\", $rate_bytes / 1048576 / $INTERVAL }")
            echo "$pid $comm $total_mb $rate_mbs $rate_bytes" >> "$tmpfile"
        fi
    done

    # Sort by rate (column 5 = raw rate_bytes) descending, take top N
    sort -t' ' -k5 -nr "$tmpfile" | head -n "$TOP_N" > "$TOP_WRITERS_FILE"
    rm -f "$tmpfile"
}

# Print header
printf "%-20s %8s %8s %8s | %-16s %6s %10s %8s\n" \
    "Timestamp" "SysWrMB" "SysRate" "DirtyMB" "Top Writer" "PID" "TotalMB" "Rate"
printf "%-20s %8s %8s %8s | %-16s %6s %10s %8s\n" \
    "--------------------" "--------" "--------" "--------" \
    "----------------" "------" "----------" "--------"

cleanup() {
    echo ""
    echo "============================================================"
    echo "  Disk Write Monitor stopped. CSV saved to: $OUTPUT"
    echo "  Samples: $SAMPLE_COUNT"
    echo "============================================================"

    # Summary from CSV
    if command -v awk &>/dev/null && [ -f "$OUTPUT" ]; then
        echo ""
        awk -F',' 'NR>1 {
            if ($4+0 > max_rate) max_rate=$4;
            total_written=$3;
            count++
        } END {
            printf "  Final sys writes:  %.1f MB\n", total_written
            printf "  Peak sys rate:     %.3f MB/s\n", max_rate
            printf "  Samples:           %d\n", count
        }' "$OUTPUT" 2>/dev/null
    fi
    rm -f "$TOP_WRITERS_FILE"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Initial system write bytes reading
PREV_SYS_WRITE=$(get_sys_write_bytes)
sleep "$INTERVAL"

while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START))
    TS=$(date "+%Y-%m-%d %H:%M:%S")

    # System-wide disk writes
    CUR_SYS_WRITE=$(get_sys_write_bytes)
    SYS_DELTA=$((CUR_SYS_WRITE - PREV_SYS_WRITE))
    SYS_TOTAL_MB=$(awk "BEGIN { printf \"%.1f\", $CUR_SYS_WRITE / 1048576 }")
    SYS_RATE_MBS=$(awk "BEGIN { printf \"%.3f\", $SYS_DELTA / 1048576 / $INTERVAL }")
    PREV_SYS_WRITE=$CUR_SYS_WRITE

    # Dirty and writeback from /proc/meminfo (kB)
    DIRTY_KB=$(awk '/^Dirty:/ { print $2 }' /proc/meminfo)
    WB_KB=$(awk '/^Writeback:/ { print $2 }' /proc/meminfo)
    DIRTY_MB=$((DIRTY_KB / 1024))
    WB_MB=$((WB_KB / 1024))

    # Per-process top writers (called in main shell so PREV_WRITE persists)
    collect_process_writes
    mapfile -t TOP_WRITERS < "$TOP_WRITERS_FILE"

    # Build CSV line
    CSV_LINE="$TS,$ELAPSED,$SYS_TOTAL_MB,$SYS_RATE_MBS,$DIRTY_MB,$WB_MB"

    # Print system-level line
    printf "\033[36m%-20s %7sM %7s/s %7dM\033[0m" "$TS" "$SYS_TOTAL_MB" "$SYS_RATE_MBS" "$DIRTY_MB"

    # Add each top writer to the CSV and console
    for i in $(seq 0 $((TOP_N - 1))); do
        if [ $i -lt ${#TOP_WRITERS[@]} ]; then
            read -r w_pid w_name w_total w_rate _ <<< "${TOP_WRITERS[$i]}"
            CSV_LINE="${CSV_LINE},${w_name},${w_pid},${w_total},${w_rate}"

            # Print first writer on same line, rest on continuation lines
            if [ $i -eq 0 ]; then
                # Color code by write rate
                if (( $(echo "$w_rate > 10" | bc -l 2>/dev/null || echo 0) )); then
                    COLOR="\033[31m"  # red >10 MB/s
                elif (( $(echo "$w_rate > 1" | bc -l 2>/dev/null || echo 0) )); then
                    COLOR="\033[33m"  # yellow >1 MB/s
                else
                    COLOR="\033[32m"  # green
                fi
                printf " | ${COLOR}%-16s %6s %9sMB %7s/s\033[0m\n" "$w_name" "$w_pid" "$w_total" "$w_rate"
            else
                printf "%-52s | %-16s %6s %9sMB %7s/s\n" "" "$w_name" "$w_pid" "$w_total" "$w_rate"
            fi
        else
            CSV_LINE="${CSV_LINE},,,,"
            if [ $i -eq 0 ]; then
                printf " | (no writers)\n"
            fi
        fi
    done

    echo "$CSV_LINE" >> "$OUTPUT"
    SAMPLE_COUNT=$((SAMPLE_COUNT + 1))
    sleep "$INTERVAL"
done
