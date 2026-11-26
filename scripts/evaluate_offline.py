"""Evaluate a checkpoint on its run's validation split, next to trivial baselines.

Two protocols are reported:
- logged: the trainer's validation metric (frame-weighted, burn-in frames excluded,
  averaged over the first `val_max_batches` batches), to check against the run's log;
- pooled: every validation clip, frames after burn-in, unweighted, for the model and the
  always-zero, always-mean, per-command-mean and two target-point baselines, fitted on the
  training split. Those see no images: one reads the bearing of the target point, the other
  averages the 8 training frames whose target points lie nearest.

The model is also run on mirrored panoramas, and on another clip's panoramas with its own
command, speed and target point: a model that steers from the images gets worse with them.
A situation head is scored by its AUROC per cause, on frames after burn-in that count in
training; answering the same for every frame scores 0.5.

The waypoint head is scored by its displacement errors on moving frames, against driving
straight on at the current speed and against the 8 training frames nearest in target point
and distance covered in 2 s; and, at standstill, by how well its predicted 2 s distance tells
frames that pull away from frames that wait, next to the control head's acceleration.

The clips that start an episode run from an empty state, as a drive starts: their
acceleration until the car first moves is reported next to the expert's.
Intervals are 95% over episode bootstrap draws.

Usage:
    python scripts/evaluate_offline.py --checkpoint runs_fast/<run>/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
from functools import partial
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from agents.learned.learned_agent import load_checkpoint
from src.data import labels, schema_adapter
from src.data.driving_dataset import (
    MAX_SPEED_MS,
    WAYPOINT_OFFSETS,
    DrivingSequenceDataset,
    driving_worker_init_fn,
)
from src.eval import metrics as M
from src.models import VisionRWKVDriver
from src.utils.device import select_device
from src.utils.settings import setting

LEFT, RIGHT = 0, 1  # nav command indices in the sample format (RoadOption - 1)
WAITING_M = 0.5  # labelled distance in 2 s of a standing frame that waits
PULLING_AWAY_M = 2.0  # labelled distance in 2 s of a standing frame that pulls away
BRAKE_SPEED_MS = 0.4  # TransFuser's controller brakes below this desired speed
START_ACCEL = 0.4  # held from rest, moves the expert's car within 1 s in CARLA (0.12: 6 s)


def collect(
    model: VisionRWKVDriver, loader: DataLoader[Any], burn_in: int, device: torch.device
) -> dict[str, np.ndarray]:
    """Per-frame predictions and targets, in loader order.

    Predictions are made on the plain and the mirrored panoramas, and on the panoramas of
    the clip before in the batch.
    """
    parts: dict[str, list[np.ndarray]] = {
        k: []
        for k in (
            "pred",
            "mirrored",
            "other_clip",
            "target",
            "weight",
            "nav",
            "target_point",
            "waypoints",
            "waypoint_target",
            "waypoint_mask",
            "speed",
            "episode",
            "start",
        )
    }
    with torch.no_grad():
        for batch in tqdm(loader, desc="validation batches"):
            images = batch["images"].to(device)
            nav = batch["nav_command"].to(device)
            speed = batch["speed"].to(device)
            target_point = batch["target_point"].to(device)
            pred, _, aux = model(images, nav, speed, target_point)
            # flipping also swaps the left and right panels
            mirrored, _, _ = model(images.flip(-1), nav, speed, target_point)
            other_clip, _, _ = model(images.roll(1, dims=0), nav, speed, target_point)
            weight = batch["loss_weight"].clone()
            weight[:, :burn_in] = 0.0
            parts["pred"].append(pred.cpu().numpy())
            parts["mirrored"].append(mirrored.cpu().numpy())
            parts["other_clip"].append(other_clip.cpu().numpy())
            parts["target"].append(batch["controls"].numpy())
            parts["weight"].append(weight.numpy())
            parts["nav"].append(batch["nav_command"].numpy())
            parts["target_point"].append(batch["target_point"].numpy())
            parts["waypoints"].append(aux["future_waypoints"].float().cpu().numpy())
            parts["waypoint_target"].append(batch["future_waypoints"].numpy())
            parts["waypoint_mask"].append(batch["waypoint_mask"].numpy())
            parts["speed"].append(batch["speed"].numpy() * MAX_SPEED_MS)
            parts["episode"].append(batch["clip"][:, :1].expand_as(batch["speed"]).numpy())
            parts["start"].append(batch["clip"][:, 1:].expand_as(batch["speed"]).numpy())
            if "situation_logits" in aux:
                probability = torch.sigmoid(aux["situation_logits"].float()).cpu().numpy()
                causes = batch["situation"][..., labels.CAUSE_SITUATIONS].any(-1).numpy()
                parts.setdefault("situation", []).append(probability)
                parts.setdefault("causes", []).append(causes)
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


def train_labels(episodes: list[Path], fps: int, sparse: bool = False) -> dict[str, np.ndarray]:
    """Labels of the training frames kept in the loss, which of them are moving, and their
    speeds and waypoints (valid where every horizon lies within the recording)."""
    steer, accel, nav, point, speed, waypoints, valid = [], [], [], [], [], [], []
    for path in tqdm(episodes, desc="training labels"):
        with h5py.File(path, "r") as f:
            episode = schema_adapter.read_labels(f, fps)
            n, lookahead = len(episode.steering), max(WAYPOINT_OFFSETS)
            w = schema_adapter.read_window(f, 0, n, lookahead, fps=fps, images=False)
        kept = ~episode.excluded
        steer.append(episode.steering[kept])
        accel.append(episode.acceleration[kept])
        nav.append(episode.command[kept] - 1)
        point.append((episode.sparse_target_point if sparse else episode.target_point)[kept])
        speed.append(episode.speed[kept])
        waypoints.append(labels.ego_waypoints(w.position, w.yaw, WAYPOINT_OFFSETS)[kept])
        valid.append((np.arange(n) + lookahead < len(w.position))[kept])
    return {
        "steer": np.concatenate(steer),
        "accel": np.concatenate(accel),
        "nav": np.concatenate(nav),
        "target_point": np.concatenate(point),
        "moving": np.concatenate(speed) >= labels.STANDING_SPEED,
        "speed": np.concatenate(speed),
        "waypoints": np.concatenate(waypoints),
        "waypoints_valid": np.concatenate(valid),
    }


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


def weighted_mean(values: np.ndarray, mask: np.ndarray, weight: np.ndarray) -> float:
    ok = mask & np.isfinite(values)
    return float((values[ok] * weight[ok]).sum() / max(float(weight[ok].sum()), 1e-9))


def mean_at(values: np.ndarray, mask: np.ndarray, weight: np.ndarray, i: np.ndarray) -> float:
    return weighted_mean(values[i], mask[i], weight[i])


def waypoint_report(
    frames: dict[str, np.ndarray], fitted: dict[str, np.ndarray], fps: int
) -> dict[str, Any]:
    """The waypoint head against image-free baselines, and as a go/stop signal at standstill."""
    horizons = np.asarray(WAYPOINT_OFFSETS) / fps
    speed, truth, valid = frames["speed"], frames["waypoints"], frames["waypoint_mask"]
    weight, episode = frames["weight"], frames["episode"]
    ok = fitted["waypoints_valid"]
    neighbours = M.predict_by_neighbours(
        fitted["waypoints"][ok].reshape(int(ok.sum()), -1),
        np.c_[fitted["target_point"][ok], horizons[-1] * fitted["speed"][ok]],
        np.c_[frames["target_point"], horizons[-1] * speed],
    ).reshape(truth.shape)
    straight = np.stack([speed[:, None] * horizons, np.zeros_like(truth[..., 0])], axis=-1)
    predictions = {
        "model": frames["model_waypoints"],
        "constant_velocity": straight,
        "neighbours": neighbours,
    }
    errors = {name: M.displacement_errors(p, truth, valid) for name, p in predictions.items()}
    moving = (weight > 0) & (speed >= labels.STANDING_SPEED) & valid.any(axis=-1)
    turns = moving & M.turn_frames(frames["steer"])
    table: dict[str, Any] = {
        name: {
            "ade_moving": weighted_mean(ade, moving, weight),
            "fde_moving": weighted_mean(fde, moving, weight),
            "ade_turns": weighted_mean(ade, turns, weight),
        }
        for name, (ade, fde) in errors.items()
    }
    distance = np.linalg.norm(frames["model_waypoints"] - truth, axis=-1)
    table["model"]["error_per_horizon"] = {
        f"{h:.1f}s": weighted_mean(distance[:, k], moving & valid[:, k], weight)
        for k, h in enumerate(horizons)
    }
    for name in ("constant_velocity", "neighbours"):
        diff = errors["model"][0] - errors[name][0]
        table["model"][f"ade_minus_{name}_ci"] = M.episode_bootstrap(
            partial(mean_at, diff, moving, weight), episode
        )

    standing = (weight > 0) & (speed < labels.STANDING_SPEED) & valid[:, -1]
    labelled = np.linalg.norm(truth[:, -1], axis=-1)
    going, waiting = standing & (labelled >= PULLING_AWAY_M), standing & (labelled < WAITING_M)
    decided = going | waiting
    predicted = np.linalg.norm(frames["model_waypoints"][:, -1], axis=-1)
    below = predicted < BRAKE_SPEED_MS * horizons[-1]
    everything = np.arange(len(speed))

    def auroc(score: np.ndarray, i: np.ndarray) -> float:
        d = decided[i]
        y = going[i][d]
        return float(roc_auc_score(y, score[i][d])) if 0 < y.sum() < len(y) else float("nan")

    def share(mask: np.ndarray, condition: np.ndarray, i: np.ndarray) -> float:
        return float(condition[i][mask[i]].mean()) if mask[i].any() else float("nan")

    table["at_standstill"] = {
        "pulling_away_frames": int(going.sum()),
        "waiting_frames": int(waiting.sum()),
        "stop_distance_m": BRAKE_SPEED_MS * horizons[-1],
        "auroc": auroc(predicted, everything),
        "auroc_ci": M.episode_bootstrap(lambda i: auroc(predicted, i), episode),
        "control_head_auroc": auroc(frames["model_accel"], everything),
        "control_head_auroc_ci": M.episode_bootstrap(
            lambda i: auroc(frames["model_accel"], i), episode
        ),
        "waiting_below_stop_distance": share(waiting, below, everything),
        "waiting_below_stop_distance_ci": M.episode_bootstrap(
            lambda i: share(waiting, below, i), episode
        ),
        "pulling_away_above_stop_distance": share(going, ~below, everything),
    }
    return table


def cold_start_report(c: dict[str, np.ndarray]) -> dict[str, Any]:
    """The acceleration of the clips that start an episode, run from an empty state as a drive
    starts, on their frames until the car first moves."""
    resting = np.logical_and.accumulate(c["speed"] < labels.STANDING_SPEED, axis=1)
    cold = resting & (c["start"] == 0)
    if not cold.any():
        return {"frames": 0}
    model, expert, episode = c["pred"][..., 1][cold], c["target"][..., 1][cold], c["episode"][cold]

    def median(i: np.ndarray) -> float:
        return float(np.median(model[i]))

    return {
        "episodes": len(np.unique(episode)),
        "frames": int(cold.sum()),
        "expert_accel_median": float(np.median(expert)),
        "model_accel_median": median(np.arange(len(model))),
        "model_accel_median_ci": M.episode_bootstrap(median, episode),
        "model_accel_at_least_start_accel": float(np.mean(model >= START_ACCEL)),
    }


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
        target_point=d.target_point,
    )
    val = DrivingSequenceDataset(split="val", augment=False, **common)
    train = DrivingSequenceDataset(split="train", augment=False, **common)
    # Each batch takes its clips from across the split, so the clip before is from another
    # episode; the results are put back in split order for the logged protocol.
    n_batches = math.ceil(len(val) / d.batch_size)
    batches = [list(range(b, len(val), n_batches)) for b in range(n_batches)]
    loader = DataLoader(
        val,
        batch_sampler=batches,
        num_workers=args.workers,
        worker_init_fn=driving_worker_init_fn,
    )
    c = collect(model, loader, burn_in, device)
    split_order = np.argsort(np.concatenate(batches))
    c = {k: v[split_order] for k, v in c.items()}
    logged = logged_protocol(c, d.batch_size, int(cfg.logging.val_max_batches))

    after = np.s_[:, burn_in:]
    frames = {
        "steer": c["target"][after][..., 0].ravel(),
        "accel": c["target"][after][..., 1].ravel(),
        "nav": c["nav"][after].ravel(),
        "target_point": c["target_point"][after].reshape(-1, 2),
        "weight": c["weight"][after].ravel(),
        "model_steer": c["pred"][after][..., 0].ravel(),
        "model_accel": c["pred"][after][..., 1].ravel(),
        "mirrored_steer": c["mirrored"][after][..., 0].ravel(),
        "other_clip_steer": c["other_clip"][after][..., 0].ravel(),
        "other_clip_accel": c["other_clip"][after][..., 1].ravel(),
    }
    if "situation" in c:
        frames["situation"] = c["situation"][after].reshape(-1, len(labels.CAUSES))
        frames["causes"] = c["causes"][after].reshape(-1, len(labels.CAUSES))
    k = len(WAYPOINT_OFFSETS)
    frames["speed"] = c["speed"][after].ravel()
    frames["episode"] = c["episode"][after].ravel()
    frames["waypoints"] = c["waypoint_target"][after].reshape(-1, k, 2)
    frames["waypoint_mask"] = c["waypoint_mask"][after].reshape(-1, k)
    frames["model_waypoints"] = c["waypoints"][after].reshape(-1, k, 2)
    steer, accel, nav = frames["steer"], frames["accel"], frames["nav"]
    fitted = train_labels(train.episode_files, d.fps, d.target_point == "sparse")
    t_steer, t_accel, t_nav = fitted["steer"], fitted["accel"], fitted["nav"]
    mean_s, mean_a = float(t_steer.mean()), float(t_accel.mean())
    moving, t_point = fitted["moving"], fitted["target_point"][fitted["moving"]]
    point_steer = M.fit_target_lookup(t_steer[moving], t_point)
    point_accel = M.fit_target_lookup(t_accel[moving], t_point)
    nearest = M.predict_by_neighbours(
        np.c_[t_steer[moving], t_accel[moving]], t_point, frames["target_point"]
    )
    zeros = np.zeros_like(steer)
    baselines = {
        "model": (frames["model_steer"], frames["model_accel"]),
        "other_clip_images": (frames["other_clip_steer"], frames["other_clip_accel"]),
        "always_zero": (zeros, zeros),
        "always_mean": (zeros + mean_s, zeros + mean_a),
        "per_command_mean": (
            M.predict_by_command(M.command_means(t_steer, t_nav), nav, mean_s),
            M.predict_by_command(M.command_means(t_accel, t_nav), nav, mean_a),
        ),
        "target_point_only": (
            M.predict_by_target(point_steer, frames["target_point"]),
            M.predict_by_target(point_accel, frames["target_point"]),
        ),
        "target_point_knn": (nearest[:, 0], nearest[:, 1]),
    }
    weightings = {"training_weights": frames["weight"], "all_frames": None}
    turns = M.turn_frames(steer)
    pooled: dict[str, dict[str, Any]] = {
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
    }
    trained = pooled["training_weights"]
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
        "pooled": pooled,
        "model_behaviour": {
            "camera_swap_score": M.camera_swap_score(
                frames["model_steer"], frames["mirrored_steer"]
            ),
            "other_clip_turn_error_ratio": trained["other_clip_images"]["turn_steer_mae"]
            / trained["model"]["turn_steer_mae"],
            "expert_steer_std": float(np.std(steer)),
        },
    }
    report["waypoints"] = waypoint_report(frames, fitted, d.fps)
    report["cold_start"] = cold_start_report(c)
    if "situation" in frames:
        counted = frames["weight"] > 0
        truth, probability = frames["causes"][counted], frames["situation"][counted]
        report["model_behaviour"]["situation_auroc"] = {
            cause: float(roc_auc_score(truth[:, i], probability[:, i]))
            if 0 < truth[:, i].sum() < len(truth)
            else float("nan")
            for i, cause in enumerate(labels.CAUSES)
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
    behaviour = report["model_behaviour"]
    print(
        f"\ncamera-swap score {behaviour['camera_swap_score']:.3f}; turn error with other clips' "
        f"images x{behaviour['other_clip_turn_error_ratio']:.2f}; report: {out}"
    )
    if "situation_auroc" in behaviour:
        aurocs = ", ".join(f"{k} {v:.3f}" for k, v in behaviour["situation_auroc"].items())
        print(f"situation AUROC (a constant answer scores 0.5): {aurocs}")
    wp = report["waypoints"]
    print(f"\nwaypoints on moving frames (metres): {'ADE':>8}{'FDE 2 s':>9}{'turn ADE':>9}")
    for name in ("model", "constant_velocity", "neighbours"):
        r = wp[name]
        print(f"  {name:34}{r['ade_moving']:8.3f}{r['fde_moving']:9.3f}{r['ade_turns']:9.3f}")
    m, st = wp["model"], wp["at_standstill"]

    def ci(pair: tuple[float, float], spec: str = ".3f") -> str:
        return f"[{pair[0]:{spec}}, {pair[1]:{spec}}]"

    print(
        f"  model minus constant velocity, 95% CI {ci(m['ade_minus_constant_velocity_ci'])}; "
        f"minus neighbours {ci(m['ade_minus_neighbours_ci'])}"
    )
    print(
        "  error per horizon: "
        + ", ".join(f"{h} {e:.3f}" for h, e in m["error_per_horizon"].items())
    )
    print(
        f"at standstill ({st['pulling_away_frames']:,} pulling away, "
        f"{st['waiting_frames']:,} waiting): go/stop AUROC of the 2 s distance "
        f"{st['auroc']:.3f} {ci(st['auroc_ci'])}, of the control head's acceleration "
        f"{st['control_head_auroc']:.3f} {ci(st['control_head_auroc_ci'])}"
    )
    print(
        f"  under {st['stop_distance_m']:.1f} m on {st['waiting_below_stop_distance']:.1%} of "
        f"waiting frames {ci(st['waiting_below_stop_distance_ci'], '.1%')}, over it on "
        f"{st['pulling_away_above_stop_distance']:.1%} of frames pulling away"
    )
    cold = report["cold_start"]
    if cold["frames"]:
        print(
            f"\ncold start ({cold['frames']} frames of {cold['episodes']} episode starts, before "
            f"the car moves): acceleration median {cold['model_accel_median']:.3f} "
            f"{ci(cold['model_accel_median_ci'])}, expert {cold['expert_accel_median']:.2f}; "
            f"{START_ACCEL} or more on {cold['model_accel_at_least_start_accel']:.1%}"
        )


if __name__ == "__main__":
    main()
