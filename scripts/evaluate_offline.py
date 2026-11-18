"""Evaluate a checkpoint on its run's validation split, next to trivial baselines.

Two protocols are reported:
- logged: the trainer's validation metric (frame-weighted, burn-in frames excluded,
  averaged over the first `val_max_batches` batches), to check against the run's log;
- pooled: every validation clip, frames after burn-in, unweighted, for the model and the
  always-zero, always-mean and per-command-mean baselines, fitted on the training split.

Usage:
    python scripts/evaluate_offline.py --checkpoint runs_fast/<run>/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from agents.learned.learned_agent import load_checkpoint
from src.data import schema_adapter
from src.data.driving_dataset import DrivingSequenceDataset, driving_worker_init_fn
from src.eval import metrics as M
from src.models import VisionRWKVDriver
from src.utils.device import select_device
from src.utils.settings import setting

LEFT, RIGHT = 0, 1  # nav command indices in the sample format (RoadOption - 1)


def collect(
    model: VisionRWKVDriver, loader: DataLoader[Any], burn_in: int, device: torch.device
) -> dict[str, np.ndarray]:
    """Per-frame predictions, on plain and mirrored panoramas, and targets, in loader order."""
    parts: dict[str, list[np.ndarray]] = {
        k: [] for k in ("pred", "mirrored", "target", "weight", "nav")
    }
    with torch.no_grad():
        for batch in tqdm(loader, desc="validation batches"):
            images = batch["images"].to(device)
            nav = batch["nav_command"].to(device)
            pred, _, _ = model(images, nav)
            mirrored, _, _ = model(images.flip(-1), nav)  # also swaps the left and right panels
            weight = batch["loss_weight"].clone()
            weight[:, :burn_in] = 0.0
            parts["pred"].append(pred.cpu().numpy())
            parts["mirrored"].append(mirrored.cpu().numpy())
            parts["target"].append(batch["controls"].numpy())
            parts["weight"].append(weight.numpy())
            parts["nav"].append(batch["nav_command"].numpy())
    return {k: np.concatenate(v) for k, v in parts.items()}


def logged_protocol(c: dict[str, np.ndarray], batch_size: int, max_batches: int) -> dict[str, Any]:
    n_batches = math.ceil(len(c["pred"]) / batch_size)
    if max_batches > 0:
        n_batches = min(n_batches, max_batches)
    per_batch: dict[str, list[float]] = {"steer_mae": [], "accel_mae": []}
    for b in range(n_batches):
        s = slice(b * batch_size, (b + 1) * batch_size)
        for i, name in enumerate(per_batch):
            per_batch[name].append(
                M.weighted_mae(c["pred"][s, :, i], c["target"][s, :, i], c["weight"][s])
            )
    return {"batches": n_batches, **{k: float(np.mean(v)) for k, v in per_batch.items()}}


def train_labels(episodes: list[Path], fps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steer, accel, nav = [], [], []
    for path in tqdm(episodes, desc="training labels"):
        with h5py.File(path, "r") as f:
            episode = schema_adapter.read_labels(f, fps)
        kept = ~episode.excluded
        steer.append(episode.steering[kept])
        accel.append(episode.acceleration[kept])
        nav.append(episode.command[kept] - 1)
    return np.concatenate(steer), np.concatenate(accel), np.concatenate(nav)


def scores(
    pred_steer: np.ndarray,
    pred_accel: np.ndarray,
    steer: np.ndarray,
    accel: np.ndarray,
    weight: np.ndarray | None,
) -> dict[str, float]:
    turns = M.turn_frames(steer)
    w_turns = None if weight is None else weight[turns]
    return {
        "steer_mae": M.weighted_mae(pred_steer, steer, weight),
        "accel_mae": M.weighted_mae(pred_accel, accel, weight),
        "turn_steer_mae": M.weighted_mae(pred_steer[turns], steer[turns], w_turns),
        "turn_sign_accuracy": M.sign_accuracy(pred_steer[turns], steer[turns], w_turns),
        "steer_std": float(np.std(pred_steer)),
        "steer_corr": M.pearson(pred_steer, steer),
    }


def mean_on(values: np.ndarray, mask: np.ndarray, weight: np.ndarray | None) -> float:
    w = np.ones(mask.sum()) if weight is None else weight[mask]
    return float((values[mask] * w).sum() / max(w.sum(), 1e-6))


def finite(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    return None if isinstance(x, float) and not math.isfinite(x) else x


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", help="Episode directory (default: DATA_DIR)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", help="Report path (default: <run>/eval/<checkpoint>_offline.json)")
    args = p.parse_args()

    device = select_device("auto")
    model, cfg = load_checkpoint(args.checkpoint, device)
    d, burn_in = cfg.dataset, cfg.loss.burn_in
    common = dict(
        data_dir=args.data_dir or setting("DATA_DIR"),
        seq_len=d.seq_len,
        stride=d.stride,
        fps=d.fps,
        val_fraction=d.val_fraction,
        seed=cfg.run.seed,
        resize_height=d.resize_height,
        resize_width=d.resize_width,
    )
    val = DrivingSequenceDataset(split="val", augment=False, **common)
    train = DrivingSequenceDataset(split="train", augment=False, **common)
    loader = DataLoader(
        val,
        batch_size=d.batch_size,
        shuffle=False,
        num_workers=args.workers,
        worker_init_fn=driving_worker_init_fn,
    )
    c = collect(model, loader, burn_in, device)
    logged = logged_protocol(c, d.batch_size, int(cfg.logging.val_max_batches))

    after = np.s_[:, burn_in:]
    frames = {
        "steer": c["target"][after][..., 0].ravel(),
        "accel": c["target"][after][..., 1].ravel(),
        "nav": c["nav"][after].ravel(),
        "weight": c["weight"][after].ravel(),
        "model_steer": c["pred"][after][..., 0].ravel(),
        "model_accel": c["pred"][after][..., 1].ravel(),
        "mirrored_steer": c["mirrored"][after][..., 0].ravel(),
    }
    steer, accel, nav = frames["steer"], frames["accel"], frames["nav"]
    t_steer, t_accel, t_nav = train_labels(train.episode_files, d.fps)
    mean_s, mean_a = float(t_steer.mean()), float(t_accel.mean())
    zeros = np.zeros_like(steer)
    baselines = {
        "model": (frames["model_steer"], frames["model_accel"]),
        "always_zero": (zeros, zeros),
        "always_mean": (zeros + mean_s, zeros + mean_a),
        "per_command_mean": (
            M.predict_by_command(M.command_means(t_steer, t_nav), nav, mean_s),
            M.predict_by_command(M.command_means(t_accel, t_nav), nav, mean_a),
        ),
    }
    weightings = {"training_weights": frames["weight"], "all_frames": None}
    turns = M.turn_frames(steer)
    report = {
        "checkpoint": args.checkpoint,
        "protocol": {
            "seq_len": d.seq_len,
            "stride": d.stride,
            "burn_in": burn_in,
            "batch_size": d.batch_size,
            "resize": [d.resize_height, d.resize_width],
            "val_fraction": d.val_fraction,
            "seed": cfg.run.seed,
        },
        "data": {
            "val_episodes": len(val.episode_files),
            "val_clips": len(val),
            "pooled_frames": int(steer.size),
            "train_episodes": len(train.episode_files),
            "train_frames": int(t_steer.size),
        },
        "logged_protocol": logged,
        "pooled": {
            w_name: {
                "turn_frame_share": mean_on(turns.astype(float), np.ones_like(turns), w),
                **{name: scores(ps, pa, steer, accel, w) for name, (ps, pa) in baselines.items()},
                "mean_steer_on_left": {
                    "expert": mean_on(steer, nav == LEFT, w),
                    "model": mean_on(frames["model_steer"], nav == LEFT, w),
                },
                "mean_steer_on_right": {
                    "expert": mean_on(steer, nav == RIGHT, w),
                    "model": mean_on(frames["model_steer"], nav == RIGHT, w),
                },
            }
            for w_name, w in weightings.items()
        },
        "model_behaviour": {
            "camera_swap_score": M.camera_swap_score(
                frames["model_steer"], frames["mirrored_steer"]
            ),
            "expert_steer_std": float(np.std(steer)),
        },
    }
    out = (
        Path(args.out)
        if args.out
        else Path(args.checkpoint).parent.parent
        / "eval"
        / f"{Path(args.checkpoint).stem}_offline.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(finite(report), indent=2))
    np.savez_compressed(out.with_suffix(".frames.npz"), **cast(dict[str, Any], frames))

    print(f"\nlogged protocol ({logged['batches']} batches): ", end="")
    print(f"steer MAE {logged['steer_mae']:.4f}, accel MAE {logged['accel_mae']:.4f}")
    for w_name, table in report["pooled"].items():
        share = table["turn_frame_share"]
        print(f"\npooled, {w_name}: {steer.size:,} frames, turn frames {share:.1%}")
        print(
            f"{'':18}{'steer MAE':>10}{'turn MAE':>10}{'turn sign':>10}{'accel MAE':>10}{'corr':>8}"
        )
        for name in baselines:
            s = table[name]
            print(
                f"{name:18}{s['steer_mae']:10.4f}{s['turn_steer_mae']:10.4f}"
                f"{s['turn_sign_accuracy']:10.3f}{s['accel_mae']:10.4f}{s['steer_corr']:8.3f}"
            )
    print(
        f"\ncamera-swap score {report['model_behaviour']['camera_swap_score']:.3f}; report: {out}"
    )


if __name__ == "__main__":
    main()
