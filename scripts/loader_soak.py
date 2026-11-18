"""Run a training config's training loader for a while and watch its memory.

Builds the training DataLoader as the trainer does and iterates it, epoch after epoch,
for --minutes, reading the resident memory (RSS) of the main process and of each worker
every --every seconds. A worker's RSS rises while it holds a batch and falls back between
batches, so a leak shows as a rising floor: memory counts as flat when, after the --warmup
minutes, no process's lowest RSS over the last third of the readings is 5% or more above its
lowest over the first third. RSS is read from /proc (Linux).

Usage:
    python scripts/loader_soak.py --config configs/vrwkv_base.yaml --data-dir data/golden
"""

from __future__ import annotations

import argparse
import time

from src.training.config import load_training_config
from src.training.trainer import build_loader
from src.utils.memory import loader_rss_mib

GROWTH_LIMIT = 0.05


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="configs/vrwkv_base.yaml")
    p.add_argument("--data-dir", help="Episode directory (default: the config's, else DATA_DIR)")
    p.add_argument("--minutes", type=float, default=30)
    p.add_argument("--warmup", type=float, default=5, help="Minutes before memory must stay flat")
    p.add_argument("--every", type=float, default=60, help="Seconds between readings")
    args = p.parse_args()

    cfg = load_training_config(args.config)
    if args.data_dir:
        cfg.dataset.data_dir = args.data_dir
    d = cfg.dataset
    loader = build_loader(d, cfg.run.seed, "train")
    print(
        f"{d.num_workers} workers, batches of {d.batch_size} x {d.seq_len} frames at "
        f"{d.resize_height}x{d.resize_width}, {len(loader)} batches per epoch",
        flush=True,
    )

    start, end, batches = time.time(), args.minutes * 60, 0
    readings: list[tuple[float, dict[str, float]]] = []
    while time.time() - start < end:
        for _ in loader:
            batches += 1
            elapsed = time.time() - start
            if elapsed >= len(readings) * args.every:
                rss = loader_rss_mib(loader)
                if not rss:
                    raise SystemExit("cannot read RSS here: it is read from /proc")
                readings.append((elapsed, rss))
                cells = "  ".join(f"{k[4:-4]} {v:,.0f}" for k, v in rss.items())
                print(f"{elapsed / 60:5.1f} min  {batches:6d} batches  {cells}  MiB", flush=True)
            if elapsed >= end:
                break

    settled = [rss for t, rss in readings if t >= args.warmup * 60]
    if len(settled) < 2:
        raise SystemExit("too few readings after the warm-up")
    third = max(1, len(settled) // 3)
    growth = {
        k: min(r[k] for r in settled[-third:]) / min(r[k] for r in settled[:third]) - 1
        for k in settled[0]
    }
    print(f"\n{batches} batches in {(time.time() - start) / 60:.1f} min")
    print("growth after warm-up: " + ", ".join(f"{k[4:-4]} {g:+.1%}" for k, g in growth.items()))
    flat = max(growth.values()) < GROWTH_LIMIT
    print("memory flat" if flat else f"memory grew by {GROWTH_LIMIT:.0%} or more")
    raise SystemExit(0 if flat else 1)


if __name__ == "__main__":
    main()
