#!/usr/bin/env bash
# ----------------------------------------------------------------
# Verification tests for Town03 OOM hypothesis
#
# Purpose: Systematically test assumptions about the CARLA memory
# leak by isolating one variable at a time.
#
# Usage:
#   # Run a specific test:
#   bash scripts/verify_town03_leak.sh T1
#
#   # Run all tests sequentially:
#   bash scripts/verify_town03_leak.sh ALL
#
# Each test writes its own log + CSV to data/test_collection/.
# The memory monitor runs for 600s (10 min) per test; tests that
# don't OOM should finish well within that window.
#
# Prerequisites:
#   - CARLA 0.9.16 server running and reachable
#   - conda env 'rwkv' activated or available
# ----------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$PROJECT_DIR/data/test_collection"
CONFIG="$PROJECT_DIR/configs/collection.yaml"
MEMMON_DURATION=600  # 10 minutes per test

# Each test uses 1 episode, max_frames=4000, traffic=20 (first
# config from YAML). Leak shows up within 3 min if it exists.
# Override to 1 episode via --episodes 1.

run_test() {
    local name="$1"
    shift
    local log_file="$OUT_DIR/${name}.log"
    local csv_file="$OUT_DIR/memory_log_${name}.csv"

    echo ""
    echo "================================================================"
    echo "  TEST: $name"
    echo "  Args: $@"
    echo "  Log:  $log_file"
    echo "  CSV:  $csv_file"
    echo "================================================================"
    echo ""

    # Start background memory monitor
    conda run -n rwkv python "$PROJECT_DIR/scripts/diagnose_memory.py" \
        --interval 10 \
        --output "$csv_file" \
        --duration "$MEMMON_DURATION" &
    local memmon_pid=$!

    # Give monitor time to start
    sleep 2

    # Run collection
    conda run -n rwkv python "$PROJECT_DIR/scripts/collect_data.py" \
        --config "$CONFIG" \
        --output "$OUT_DIR" \
        --episodes 1 \
        --weather ClearNoon \
        --memory-log \
        "$@" \
        2>&1 | tee "$log_file"

    local exit_code=${PIPESTATUS[0]}

    # Kill memory monitor if still running
    kill "$memmon_pid" 2>/dev/null || true
    wait "$memmon_pid" 2>/dev/null || true

    echo ""
    echo "--- $name finished (exit code: $exit_code) ---"
    echo ""

    return 0  # Don't abort the script on OOM
}

show_usage() {
    echo "Usage: $0 <TEST_NAME>"
    echo ""
    echo "Available tests:"
    echo "  T1  - Town03-only, 1 episode, traffic=20 (is it Town03-specific?)"
    echo "  T2  - Town04-only, 1 episode (do other large towns leak?)"
    echo "  T3  - Town03, 0 traffic   (is it NPC-related?)"
    echo "  T4  - Town01 → Town02     (does a non-Town03 map change leak?)"
    echo "  ALL - Run T1, T2, T4 sequentially (skip T3 — needs config hack)"
    echo ""
    echo "Each test runs for up to ~10 minutes with memory monitoring."
}

# ----------------------------------------------------------------
# T1: Town03-only, no prior map change
#
# Tests: Is the leak Town03-specific, or does it require a prior
#        map change to trigger?
# Expected: OOM within 3-6 min (confirms Town03 inherently leaks)
# ----------------------------------------------------------------
test_T1() {
    run_test "T1_town03_only" \
        --town Town03
}

# ----------------------------------------------------------------
# T2: Town04-only (a different large town)
#
# Tests: Do other towns also leak, or is it Town03-specific?
# Expected if Town03-specific: Stable at ~500 MB
# Expected if widespread: Also OOM
# ----------------------------------------------------------------
test_T2() {
    run_test "T2_town04_only" \
        --town Town04
}

# ----------------------------------------------------------------
# T3: Town03 with 0 traffic
#
# Tests: Is the leak related to NPC physics/rendering or just the
#        map + ego cameras?
# NOTE: This requires a temporary YAML override or CLI flag for
#        traffic.  For now, run manually with a modified config.
# ----------------------------------------------------------------
test_T3() {
    echo "T3 requires a zero-traffic config override."
    echo "Create a temporary config with traffic.vehicles: [0] and"
    echo "traffic.walkers: [0], then run:"
    echo ""
    echo "  conda run -n rwkv python scripts/collect_data.py \\"
    echo "      --config configs/collection_zero_traffic.yaml \\"
    echo "      --output data/test_collection \\"
    echo "      --town Town03 --episodes 1 --weather ClearNoon \\"
    echo "      --memory-log 2>&1 | tee data/test_collection/T3_town03_zero_traffic.log"
}

# ----------------------------------------------------------------
# T4: Town01 → Town02 (eval town, not Town03)
#
# Tests: Does a non-Town03 map change leak?
# Expected: Both towns stable, no OOM
# ----------------------------------------------------------------
test_T4() {
    run_test "T4_town01_town02" \
        --town Town01 Town02
}

# Main dispatch
case "${1:-}" in
    T1) test_T1 ;;
    T2) test_T2 ;;
    T3) test_T3 ;;
    T4) test_T4 ;;
    ALL)
        test_T1
        echo ""
        echo "Waiting 30s between tests for system recovery..."
        sleep 30
        test_T2
        echo ""
        echo "Waiting 30s between tests for system recovery..."
        sleep 30
        test_T4
        echo ""
        echo "================================================================"
        echo "  All tests complete. Analyze logs in: $OUT_DIR/"
        echo "================================================================"
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
