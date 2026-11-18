#!/usr/bin/env python3
"""Check that an environment smoke-training run finished healthily.

Usage:
    python scripts/check_smoke_run.py runs/smoke

Fails (exit code 1) when the run directory is missing, when any logged loss is
not finite, when every logged gradient norm is zero (the failure mode of the
April 2026 run), when step times were not logged, when no validation pass was
logged, or when no checkpoint was written.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

LOSS_FIELDS = ("total_loss", "control_loss", "speed_loss", "mode_loss", "waypoint_loss")
TIME_FIELDS = ("data_wait_s", "train_step_s")


def check_run(run_dir: Path) -> list[str]:
    """Return a list of problems found in one run directory (empty if healthy)."""
    problems: list[str] = []
    metrics_path = run_dir / "logs" / "metrics.csv"
    if not metrics_path.exists():
        return [f"missing {metrics_path}"]

    with metrics_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    if not train:
        problems.append("no training rows logged")
    if not val:
        problems.append("no validation rows logged")

    for row in rows:
        for name in LOSS_FIELDS:
            value = float(row[name])
            if not math.isfinite(value):
                problems.append(f"non-finite {row['split']} {name} at step {row['step']}")

    for row in train:
        for name in TIME_FIELDS:
            value = float(row[name])
            if not (math.isfinite(value) and value >= 0.0):
                problems.append(f"bad {name} {row[name]!r} at step {row['step']}")

    grad_norms = [float(r["grad_norm"]) for r in train]
    if grad_norms and not any(g > 0.0 for g in grad_norms if math.isfinite(g)):
        problems.append("every logged gradient norm is zero or non-finite")

    checkpoints = list((run_dir / "checkpoints").glob("*.pt"))
    if not checkpoints:
        problems.append("no checkpoint written")
    return problems


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    root = Path(sys.argv[1])
    run_dirs = sorted(p for p in root.iterdir() if (p / "logs").is_dir()) if root.is_dir() else []
    if not run_dirs:
        print(f"FAIL: no run directory under {root}")
        sys.exit(1)

    run_dir = run_dirs[-1]
    problems = check_run(run_dir)
    if problems:
        print(f"FAIL: {run_dir}")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    print(f"OK: {run_dir} (finite losses, non-zero gradients, validation and checkpoint present)")


if __name__ == "__main__":
    main()
