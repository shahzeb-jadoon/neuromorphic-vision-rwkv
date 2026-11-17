#!/usr/bin/env python3
"""Memory diagnostics monitor for data collection.

Runs alongside data collection to continuously monitor system and process
memory, writing timestamped samples to a CSV for post-hoc analysis.

Usage (standalone — run in a second terminal while collecting):
    python scripts/diagnose_memory.py --interval 5 --output data/memory_log.csv

Usage (one-shot — print current state):
    python scripts/diagnose_memory.py --once

The CSV columns:
    timestamp, wall_clock, system_total_mb, system_available_mb,
    system_used_mb, system_buffers_mb, system_cached_mb,
    wsl_committed_mb, python_procs (JSON list of {pid, rss_mb, cmdline})

On WSL2, also reads committed memory from /proc/meminfo (CommitLimit,
Committed_AS) which is the real OOM trigger — the WSL2 kernel OOM-kills
based on committed anonymous memory, not RSS or page cache.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Single point-in-time memory measurement."""

    timestamp: str
    wall_clock: float

    # System-wide (from /proc/meminfo)
    system_total_mb: float = 0.0
    system_available_mb: float = 0.0
    system_used_mb: float = 0.0
    system_buffers_mb: float = 0.0
    system_cached_mb: float = 0.0
    system_slab_mb: float = 0.0
    commit_limit_mb: float = 0.0
    committed_as_mb: float = 0.0
    swap_total_mb: float = 0.0
    swap_free_mb: float = 0.0

    # Per-process info (best-effort scan of /proc)
    python_procs: list[dict[str, Any]] = field(default_factory=list)

    def system_used_pct(self) -> float:
        if self.system_total_mb == 0:
            return 0.0
        return 100.0 * self.system_used_mb / self.system_total_mb

    def commit_pct(self) -> float:
        if self.commit_limit_mb == 0:
            return 0.0
        return 100.0 * self.committed_as_mb / self.commit_limit_mb

    def swap_used_mb(self) -> float:
        return self.swap_total_mb - self.swap_free_mb


def read_meminfo() -> dict[str, int]:
    """Parse /proc/meminfo into a dict of field -> kB values."""
    result: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]  # "12345 kB" -> "12345"
                    result[key] = int(val)
    except (OSError, ValueError):
        pass
    return result


def scan_python_processes() -> list[dict[str, Any]]:
    """Find Python processes by scanning /proc/*/exe and reading their RSS.

    Returns list of {pid, rss_mb, vm_mb, cmdline} dicts.
    """
    procs: list[dict[str, Any]] = []
    proc_dir = Path("/proc")
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            exe = (entry / "exe").resolve().name
            if "python" not in exe.lower():
                continue
            # Read VmRSS and VmSize from /proc/<pid>/status
            status_data: dict[str, str] = {}
            with open(entry / "status") as f:
                for line in f:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        status_data[parts[0].strip()] = parts[1].strip()
            rss_kb = int(status_data.get("VmRSS", "0").split()[0])
            vm_kb = int(status_data.get("VmSize", "0").split()[0])
            # Read cmdline
            with open(entry / "cmdline", "rb") as fc:
                cmdline = fc.read().decode("utf-8", errors="replace").replace("\0", " ").strip()
            procs.append(
                {
                    "pid": pid,
                    "rss_mb": round(rss_kb / 1024, 1),
                    "vm_mb": round(vm_kb / 1024, 1),
                    "cmdline": cmdline[:200],  # Truncate for readability
                }
            )
        except (OSError, ValueError, KeyError):
            continue
    return procs


def take_snapshot(start_time: float) -> MemorySnapshot:
    """Collect a complete memory snapshot."""
    now = time.monotonic()
    meminfo = read_meminfo()

    total_kb = meminfo.get("MemTotal", 0)
    available_kb = meminfo.get("MemAvailable", 0)
    buffers_kb = meminfo.get("Buffers", 0)
    cached_kb = meminfo.get("Cached", 0)
    slab_kb = meminfo.get("Slab", 0)
    used_kb = total_kb - available_kb  # Kernel's definition of "used"

    snap = MemorySnapshot(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        wall_clock=round(now - start_time, 1),
        system_total_mb=round(total_kb / 1024, 1),
        system_available_mb=round(available_kb / 1024, 1),
        system_used_mb=round(used_kb / 1024, 1),
        system_buffers_mb=round(buffers_kb / 1024, 1),
        system_cached_mb=round(cached_kb / 1024, 1),
        system_slab_mb=round(slab_kb / 1024, 1),
        commit_limit_mb=round(meminfo.get("CommitLimit", 0) / 1024, 1),
        committed_as_mb=round(meminfo.get("Committed_AS", 0) / 1024, 1),
        swap_total_mb=round(meminfo.get("SwapTotal", 0) / 1024, 1),
        swap_free_mb=round(meminfo.get("SwapFree", 0) / 1024, 1),
        python_procs=scan_python_processes(),
    )
    return snap


def format_snapshot(snap: MemorySnapshot) -> str:
    """Human-readable single-line summary."""
    total_py_rss = sum(p.get("rss_mb", 0.0) for p in snap.python_procs)
    swap_used = snap.swap_used_mb()
    lines = [
        f"[{snap.timestamp}] +{snap.wall_clock:.0f}s",
        f"  System: {snap.system_used_mb:.0f}/{snap.system_total_mb:.0f} MB used"
        f" ({snap.system_used_pct():.1f}%)"
        f" | avail={snap.system_available_mb:.0f} MB"
        f" | buffers={snap.system_buffers_mb:.0f}"
        f" | cached={snap.system_cached_mb:.0f}"
        f" | slab={snap.system_slab_mb:.0f}",
        f"  Commit: {snap.committed_as_mb:.0f}/{snap.commit_limit_mb:.0f} MB"
        f" ({snap.commit_pct():.1f}%)"
        f" | Swap: {swap_used:.0f}/{snap.swap_total_mb:.0f} MB",
        f"  Python: {len(snap.python_procs)} procs, total RSS={total_py_rss:.0f} MB",
    ]
    for p in snap.python_procs:
        lines.append(
            f"    PID {p['pid']:>6}: RSS={p['rss_mb']:>7.1f} MB"
            f" VM={p['vm_mb']:>8.1f} MB  {p['cmdline']}"
        )
    return "\n".join(lines)


CSV_FIELDS = [
    "timestamp",
    "wall_clock",
    "system_total_mb",
    "system_available_mb",
    "system_used_mb",
    "system_buffers_mb",
    "system_cached_mb",
    "system_slab_mb",
    "commit_limit_mb",
    "committed_as_mb",
    "swap_total_mb",
    "swap_free_mb",
    "swap_used_mb",
    "n_python_procs",
    "total_python_rss_mb",
    "python_procs_json",
]


def snapshot_to_csv_row(snap: MemorySnapshot) -> dict[str, object]:
    """Convert snapshot to CSV-writable dict."""
    total_py_rss = sum(p.get("rss_mb", 0.0) for p in snap.python_procs)
    return {
        "timestamp": snap.timestamp,
        "wall_clock": snap.wall_clock,
        "system_total_mb": snap.system_total_mb,
        "system_available_mb": snap.system_available_mb,
        "system_used_mb": snap.system_used_mb,
        "system_buffers_mb": snap.system_buffers_mb,
        "system_cached_mb": snap.system_cached_mb,
        "system_slab_mb": snap.system_slab_mb,
        "commit_limit_mb": snap.commit_limit_mb,
        "committed_as_mb": snap.committed_as_mb,
        "swap_total_mb": snap.swap_total_mb,
        "swap_free_mb": snap.swap_free_mb,
        "swap_used_mb": snap.swap_used_mb(),
        "n_python_procs": len(snap.python_procs),
        "total_python_rss_mb": round(total_py_rss, 1),
        "python_procs_json": json.dumps(snap.python_procs),
    }


def log_memory_snapshot(label: str = "") -> MemorySnapshot:
    """Take and log a snapshot. Importable helper for use in collect_data.py.

    Returns the snapshot for programmatic use.
    """
    snap = take_snapshot(0.0)
    prefix = f"[{label}] " if label else ""
    logger.info("%s%s", prefix, format_snapshot(snap))
    return snap


def run_monitor(interval: float, output_path: Path | None, duration: float | None) -> None:
    """Run continuous monitoring loop.

    Args:
        interval: Seconds between samples.
        output_path: If set, write CSV here.
        duration: If set, stop after this many seconds.
    """
    start = time.monotonic()
    stop_event = False

    def _handle_signal(signum: int, frame: object) -> None:
        nonlocal stop_event
        stop_event = True
        logger.info("Caught signal %d, stopping...", signum)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    csv_file = None
    csv_writer = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(output_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        csv_writer.writeheader()
        logger.info("Writing CSV to %s", output_path)

    logger.info(
        "Monitoring every %.1fs%s — Ctrl+C to stop",
        interval,
        f" for {duration:.0f}s" if duration else "",
    )

    try:
        sample = 0
        while not stop_event:
            snap = take_snapshot(start)
            sample += 1

            print(f"\n{'=' * 70}")
            print(f"Sample {sample}")
            print(format_snapshot(snap))

            if csv_writer is not None:
                csv_writer.writerow(snapshot_to_csv_row(snap))
                csv_file.flush()  # type: ignore[union-attr]

            # Check duration limit
            if duration is not None and (time.monotonic() - start) >= duration:
                logger.info("Duration limit (%.0fs) reached.", duration)
                break

            time.sleep(interval)
    finally:
        if csv_file is not None:
            csv_file.close()
            logger.info("CSV saved to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M11: Monitor system & Python memory during data collection"
    )
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between samples")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV output path (default: data/memory_log.csv)",
    )
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    parser.add_argument("--duration", type=float, default=None, help="Max monitoring duration (s)")
    args = parser.parse_args()

    if args.once:
        snap = take_snapshot(0.0)
        print(format_snapshot(snap))
        return

    output = Path(args.output) if args.output else Path("data/memory_log.csv")
    run_monitor(args.interval, output, args.duration)


if __name__ == "__main__":
    main()
