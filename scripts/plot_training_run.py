#!/usr/bin/env python3
"""Generate a run dashboard plot from CSV logs."""

from __future__ import annotations

import argparse

from src.training.plotting import plot_run_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Path to run directory (contains logs/metrics.csv)")
    args = parser.parse_args()

    out_path = plot_run_dashboard(args.run_dir)
    print(f"[plot] wrote {out_path}")


if __name__ == "__main__":
    main()
