"""Resident memory of this process and of a DataLoader's worker processes.

Read from /proc, so only on Linux, where training runs inside WSL's memory limit;
elsewhere there are no readings.
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader


def rss_mib(pid: int | str = "self") -> float | None:
    """Resident set size of process `pid` in MiB, or None where /proc is unavailable."""
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def loader_rss_mib(loader: DataLoader[Any]) -> dict[str, float]:
    """RSS of this process (`rss_main_mib`) and of each worker of a loader with persistent
    workers (`rss_worker<i>_mib`), as far as they can be read."""
    # DataLoader exposes its worker processes only through its private iterator.
    workers = getattr(getattr(loader, "_iterator", None), "_workers", [])
    readings = {"rss_main_mib": rss_mib()}
    readings |= {f"rss_worker{i}_mib": rss_mib(w.pid) for i, w in enumerate(workers)}
    return {k: v for k, v in readings.items() if v is not None}
