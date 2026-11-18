"""How much of a training split each clip sampler can reach, and reaches in a run.

Simulates a training config's run on its training split and prints, for each way of
choosing clips: the share of usable frames it can draw at all (reachable), the share it
draws at least once over the run's epochs, and the share of drawn frames that turn.

Usage:
    python scripts/sampling_report.py --config configs/vrwkv_base.yaml --stride 512
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable

import numpy as np
import torch

from src.data.driving_dataset import ClipKey, DrivingSequenceDataset, EpisodeOrderedSampler
from src.training.config import load_training_config
from src.utils.settings import setting


def frame_counts(ds: DrivingSequenceDataset, keys: Iterable[ClipKey]) -> np.ndarray:
    """How many of the clips `keys` hold each usable frame, episodes concatenated."""
    first = np.cumsum([0] + ds.episode_frames)
    episodes, starts = (np.asarray(a, np.int64) for a in zip(*keys))
    begin = first[episodes] + starts
    edges = np.bincount(begin, minlength=first[-1] + 1)
    edges -= np.bincount(begin + ds.seq_len, minlength=first[-1] + 1)
    return np.cumsum(edges[:-1])


def row(
    name: str, ds: DrivingSequenceDataset, drawn: np.ndarray, reachable: np.ndarray | None
) -> str:
    turning = np.average(np.concatenate(ds.turning), weights=drawn)
    reach = "—" if reachable is None else f"{np.mean(reachable > 0):.2%}"
    return f"| {name} | {reach} | {np.mean(drawn > 0):.2%} | {turning:.1%} |"


def drawn_in_run(ds: DrivingSequenceDataset, sampler: Iterable[ClipKey], epochs: int) -> np.ndarray:
    drawn = np.zeros(sum(ds.episode_frames), np.int64)
    for _ in range(epochs):
        drawn += frame_counts(ds, sampler)
    return drawn


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="configs/vrwkv_base.yaml")
    p.add_argument("--stride", type=int, help="Override dataset.stride")
    p.add_argument("--epochs", type=int, help="Override optimization.epochs")
    p.add_argument("--data-dir", help="Episode directory (default: DATA_DIR)")
    args = p.parse_args()

    cfg = load_training_config(args.config)
    torch.manual_seed(cfg.run.seed)
    d = cfg.dataset
    epochs = args.epochs or cfg.optimization.epochs
    ds = DrivingSequenceDataset(
        args.data_dir or d.data_dir or setting("DATA_DIR"),
        seq_len=d.seq_len,
        stride=args.stride or d.stride,
        val_fraction=d.val_fraction,
        seed=cfg.run.seed,
        augment=False,
    )
    usable = sum(ds.episode_frames)
    print(
        f"Training split: {len(ds.episode_files)} episodes, {usable:,} usable frames "
        f"({np.concatenate(ds.turning).mean():.1%} turning); {ds.seq_len}-frame clips, "
        f"{len(ds):,} per epoch, {epochs} epochs\n"
    )
    print("| Clips | Reachable | Drawn in the run | Turning share of drawn frames |")
    print("| --- | ---: | ---: | ---: |")
    fixed = frame_counts(ds, ds.clips)
    print(row(f"Fixed, every {ds.stride} frames", ds, fixed, fixed))
    for weighted, name in ((False, "Random starts"), (True, "Random starts, turn-weighted")):
        sampler = ds.get_sampler(weighted=weighted)
        reachable = frame_counts(ds, sampler.keys(np.flatnonzero(sampler.weights.numpy() > 0)))
        print(row(name, ds, drawn_in_run(ds, sampler, epochs), reachable))
    ordered = EpisodeOrderedSampler(ds.episode_frames, ds.seq_len, d.batch_size)
    name = f"Episode-ordered, batch {d.batch_size}, {len(ordered):,} per epoch"
    print(row(name, ds, drawn_in_run(ds, ordered, epochs), None))
    ds.close()


if __name__ == "__main__":
    main()
