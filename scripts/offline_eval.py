"""Offline evaluation: run the trained model on val-split HDF5 episodes.

Does NOT require a running CARLA server.  Loads saved episodes, runs the
model frame-by-frame (maintaining RWKV state across each episode), and
reports steer_mae, accel_mae, and total loss — the same metrics logged
during training validation.

Usage:
    python scripts/offline_eval.py \
        --checkpoint runs_fast/fastsafe_20260424_093210/checkpoints/best.pt \
        --data-dir /mnt/data/carla_data

    # Use latest.pt instead of best.pt
    python scripts/offline_eval.py \
        --checkpoint runs_fast/fastsafe_20260424_093210/checkpoints/latest.pt \
        --data-dir /mnt/data/carla_data \
        --split val \
        --max-episodes 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from agents.learned.learned_agent import load_checkpoint
from src.data.driving_dataset import DrivingSequenceDataset
from src.data.driving_dataset import IMAGENET_MEAN, IMAGENET_STD


def run_offline_eval(
    checkpoint_path: str,
    data_dir: str,
    split: str = "val",
    max_episodes: int | None = None,
    seq_len: int = 32,
    resize_h: int = 160,
    resize_w: int = 960,
    device: str = "cuda",
) -> dict:
    """Evaluate the model on saved HDF5 episodes.

    Runs the model autoregressively: RWKV state is reset at the start of
    each episode and propagated across all frames within the episode.
    This is the closed-loop equivalent for offline data — temporally
    consistent, unlike the training loop which uses random clip batches.

    Args:
        checkpoint_path: Path to .pt checkpoint.
        data_dir:        Directory containing episode_*.h5 files.
        split:           'val' or 'train'.
        max_episodes:    Limit evaluation to this many episodes (None = all).
        seq_len:         Clip length for the dataset reader (only affects
                         how clips are indexed; inference uses full episodes).
        resize_h/w:      Must match the training resize dimensions.
        device:          PyTorch device.

    Returns:
        Dict of aggregate metrics.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(checkpoint_path, dev)

    dataset = DrivingSequenceDataset(
        data_dir=data_dir,
        seq_len=seq_len,
        stride=seq_len,  # non-overlapping clips for fair eval
        split=split,
        augment=False,
        resize_height=resize_h,
        resize_width=resize_w,
    )

    episode_files = dataset.episode_files
    if max_episodes is not None:
        episode_files = episode_files[:max_episodes]

    all_steer_ae: list[float] = []
    all_accel_ae: list[float] = []
    all_total_loss: list[float] = []

    import h5py
    from src.data import hdf5_schema as S

    print(f"\nEvaluating {len(episode_files)} {split} episodes on {dev}...")

    for ep_path in tqdm(episode_files, desc="Episodes"):
        # Load scalar arrays (small) eagerly; keep images dataset open for lazy access
        with h5py.File(ep_path, "r") as f:
            steering = f[S.STEERING][:]  # (N,) float32
            throttle = f[S.THROTTLE][:]  # (N,)
            brake = f[S.BRAKE][:]  # (N,)
            nav_cmd = f[S.NAV_COMMAND][:]  # (N,) values 1-6
            n_frames = len(steering)

        # Remap nav: 1-6 → 0-5 (match training)
        nav_cmd = np.clip(nav_cmd - 1, 0, 5)
        accel_gt = throttle - brake  # ∈ [-1, 1]

        # Reset RWKV state at the start of each episode
        state = model.initial_state(batch_size=1, device=dev)

        ep_steer_ae: list[float] = []
        ep_accel_ae: list[float] = []

        # Re-open file for frame-by-frame image access — avoids loading the
        # full episode (~GB) into RAM. h5py slices are lazily fetched from disk.
        with h5py.File(ep_path, "r") as f, torch.inference_mode():
            images_ds = f[S.IMAGES]  # kept open; individual [t] reads one frame
            for t in range(n_frames):
                img = torch.from_numpy(images_ds[t]).permute(2, 0, 1).float().div(255.0)
                img = F.interpolate(
                    img.unsqueeze(0),
                    size=(resize_h, resize_w),
                    mode="bilinear",
                    align_corners=False,
                )
                img = (img - IMAGENET_MEAN) / IMAGENET_STD
                img = img.unsqueeze(1).to(dev)  # (1, 1, 3, H, W)

                nav = torch.tensor([[int(nav_cmd[t])]], device=dev)  # (1, 1)

                controls, state, _ = model(img, nav, state)

                steer_pred = float(controls[0, 0, 0])
                accel_pred = float(controls[0, 0, 1])

                ep_steer_ae.append(abs(steer_pred - float(steering[t])))
                ep_accel_ae.append(abs(accel_pred - float(accel_gt[t])))

        all_steer_ae.extend(ep_steer_ae)
        all_accel_ae.extend(ep_accel_ae)
        all_total_loss.append(float(np.mean(ep_steer_ae) + np.mean(ep_accel_ae)))

    results = {
        "n_episodes": len(episode_files),
        "n_frames": len(all_steer_ae),
        "steer_mae": float(np.mean(all_steer_ae)),
        "accel_mae": float(np.mean(all_accel_ae)),
        "steer_mae_p95": float(np.percentile(all_steer_ae, 95)),
        "accel_mae_p95": float(np.percentile(all_accel_ae, 95)),
        "avg_episode_loss": float(np.mean(all_total_loss)),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data-dir", required=True, help="Directory with episode_*.h5 files")
    parser.add_argument("--split", default="val", choices=["val", "train"])
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--resize-h", type=int, default=160)
    parser.add_argument("--resize-w", type=int, default=960)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    results = run_offline_eval(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        max_episodes=args.max_episodes,
        resize_h=args.resize_h,
        resize_w=args.resize_w,
        device=args.device,
    )

    ckpt_name = Path(args.checkpoint).name
    print(f"\n{'=' * 55}")
    print(f"Offline Eval — {ckpt_name}  ({args.split} split)")
    print(f"{'=' * 55}")
    print(f"Episodes evaluated:  {results['n_episodes']}")
    print(f"Frames evaluated:    {results['n_frames']:,}")
    print(f"Steer  MAE (mean):   {results['steer_mae']:.4f}")
    print(f"Steer  MAE (p95):    {results['steer_mae_p95']:.4f}")
    print(f"Accel  MAE (mean):   {results['accel_mae']:.4f}")
    print(f"Accel  MAE (p95):    {results['accel_mae_p95']:.4f}")
    print(f"Avg episode loss:    {results['avg_episode_loss']:.4f}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
