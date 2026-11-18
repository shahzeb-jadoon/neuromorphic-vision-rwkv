"""Read every usable frame of a training config's dataset once, as training does, and check it.

Builds both splits with clips of seq_len frames placed back to back (stride = seq_len) and
reads every clip through a DataLoader. Each sample is checked for shapes, finite values and
valid label codes; an exception while reading a clip is recorded, not raised. Prints a
summary and every problem, and exits 1 if there is any.

Usage:
    python scripts/dataset_pass.py --config configs/vrwkv_base.yaml --data-dir DIR
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.driving_dataset import (
    WAYPOINT_OFFSETS,
    DrivingSequenceDataset,
    driving_worker_init_fn,
)
from src.data.labels import SITUATIONS
from src.models.vision_rwkv_driver import N_DRIVING_MODES
from src.training.config import load_training_config
from src.utils.settings import setting


class Guarded(Dataset):
    """A dataset whose items are (sample, None) or (None, the error reading it)."""

    def __init__(self, ds: DrivingSequenceDataset) -> None:
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> tuple[dict[str, torch.Tensor] | None, str | None]:
        try:
            return self.ds[i], None
        except Exception as e:  # noqa: BLE001 — every unreadable clip is reported
            return None, f"{type(e).__name__}: {e}"


def problems(sample: dict[str, torch.Tensor], seq_len: int) -> list[str]:
    t, k = seq_len, len(WAYPOINT_OFFSETS)
    shapes = {
        "controls": (t, 2),
        "nav_command": (t,),
        "velocity": (t, 3),
        "speed": (t,),
        "loss_weight": (t,),
        "target_speed": (t,),
        "driving_mode": (t,),
        "future_waypoints": (t, k, 2),
        "waypoint_mask": (t, k),
        "target_point": (t, 2),
        "situation": (t, len(SITUATIONS)),
    }
    found = [
        f"{name} shape {tuple(sample[name].shape)}"
        for name, shape in shapes.items()
        if tuple(sample[name].shape) != shape
    ]
    if sample["images"].ndim != 4 or sample["images"].shape[:2] != (t, 3):
        found.append(f"images shape {tuple(sample['images'].shape)}")
    for name, value in sample.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            found.append(f"{name} not finite")
    if sample["controls"].abs().max() > 1:
        found.append("controls outside [-1, 1]")
    if not ((sample["nav_command"] >= 0) & (sample["nav_command"] < 6)).all():
        found.append(f"nav_command {sample['nav_command'].tolist()}")
    if not ((sample["driving_mode"] >= 0) & (sample["driving_mode"] < N_DRIVING_MODES)).all():
        found.append(f"driving_mode {sample['driving_mode'].tolist()}")
    if not torch.isin(
        sample["loss_weight"],
        torch.tensor([0.0, DrivingSequenceDataset.STATIONARY_LOSS_WEIGHT, 1.0]),
    ).all():
        found.append("loss_weight not 0, 0.1 or 1")
    return found


def run_split(ds: DrivingSequenceDataset, workers: int) -> dict[str, Any]:
    loader = DataLoader(
        Guarded(ds),
        batch_size=8,
        num_workers=workers,
        collate_fn=list,
        worker_init_fn=driving_worker_init_fn,
    )
    found, frames, start = [], 0, time.time()
    for b, batch in enumerate(loader):
        for j, (sample, error) in enumerate(batch):
            ep_idx, first = ds.clips[b * 8 + j]
            where = f"{ds.episode_files[ep_idx].name} frames {first}-{first + ds.seq_len - 1}"
            if error:
                found.append(f"{where}: {error}")
                continue
            assert sample is not None
            frames += ds.seq_len
            found += [f"{where}: {p}" for p in problems(sample, ds.seq_len)]
    return {
        "episodes": len(ds.episode_files),
        "clips": len(ds),
        "frames": frames,
        "usable": sum(ds.episode_frames),
        "minutes": (time.time() - start) / 60,
        "problems": found,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="configs/vrwkv_base.yaml")
    p.add_argument("--data-dir", help="Episode directory (default: the config's, else DATA_DIR)")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    cfg = load_training_config(args.config)
    d = cfg.dataset
    failed = False
    for split in ("train", "val"):
        ds = DrivingSequenceDataset(
            args.data_dir or d.data_dir or setting("DATA_DIR"),
            seq_len=d.seq_len,
            stride=d.seq_len,
            fps=d.fps,
            split=split,
            val_fraction=d.val_fraction,
            seed=cfg.run.seed,
            augment=False,
            resize_height=d.resize_height,
            resize_width=d.resize_width,
        )
        r = run_split(ds, args.workers)
        ds.close()
        print(
            f"{split}: {r['episodes']} episodes, {r['clips']:,} clips, {r['frames']:,} of "
            f"{r['usable']:,} usable frames read ({r['frames'] / max(r['usable'], 1):.1%}) in "
            f"{r['minutes']:.1f} min, {len(r['problems'])} problems",
            flush=True,
        )
        for line in r["problems"][:50]:
            print("  " + line)
        failed |= bool(r["problems"])
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
