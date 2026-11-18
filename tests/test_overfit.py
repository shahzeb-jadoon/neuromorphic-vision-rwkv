"""GOLD: the real training step must be able to memorise one recorded window.

Up to an hour on a GPU. Run with `make overfit`.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pytest
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import default_collate

from src.data import schema_adapter
from src.data.driving_dataset import DrivingSequenceDataset
from src.eval.metrics import TURN_THRESHOLD
from src.training.config import load_training_config
from src.training.trainer import Trainer

pytestmark = [pytest.mark.gold, pytest.mark.overfit]

MAX_STEPS = 4000
CHECK_EVERY = 100
BATCH = 4
CLIP_STRIDE = 24  # each clip's burn-in frames are scored in the clip before it
LEARNING_RATE = 1e-4  # decays along a cosine to a tenth at MAX_STEPS
SPIKE_WINDOW = 5  # frames in the running median that removes single-frame steering spikes
MAX_STEERING_MAE = 0.02
MIN_TURN_SIGN = 0.95

Sample = dict[str, torch.Tensor]


def overfit_trainer(golden_dir: Path, out: Path) -> Trainer:
    """The short-run setup, with everything that resists memorising turned off."""
    cfg = load_training_config("configs/short_run.yaml")
    cfg.dataset.data_dir = str(golden_dir)
    cfg.dataset.num_workers = 0
    cfg.dataset.pin_memory = False
    cfg.dataset.augment_train = False
    cfg.run.output_root = str(out)
    cfg.run.run_name = "overfit"
    cfg.model.encoder.dropout = 0.0
    cfg.model.fusion.dropout = 0.0
    cfg.model.fusion.speed_dropout = 0.0
    cfg.model.policy_head.dropout = 0.0
    cfg.optimization.weight_decay = 0.0
    cfg.optimization.learning_rate = LEARNING_RATE
    cfg.scheduler.name = "cosine"
    cfg.scheduler.warmup_steps = 0
    cfg.scheduler.min_lr_ratio = 0.1
    trainer = Trainer(cfg)
    trainer.scheduler = trainer._build_scheduler(MAX_STEPS)
    return trainer


def window_with_most_turns(dataset: DrivingSequenceDataset) -> list[Sample]:
    """Clips across the training window with the most turns, with its steering spikes removed.

    Single-frame spikes that the images do not explain (docs/LABELS.md) would hold even a
    perfect fit to the rest of this window's steering at an error of 0.025.
    """
    episode = max(range(len(dataset.episode_files)), key=lambda e: int(dataset.turning[e].sum()))
    with h5py.File(dataset.episode_files[episode], "r") as f:
        steering = schema_adapter.read_labels(f, dataset.fps).steering
    padded = np.pad(steering, SPIKE_WINDOW // 2, mode="edge")
    smooth = torch.from_numpy(np.median(sliding_window_view(padded, SPIKE_WINDOW), axis=1))
    clips = []
    for start in range(0, dataset.episode_frames[episode] - dataset.seq_len + 1, CLIP_STRIDE):
        clip = dataset[(episode, start)]
        clip["controls"][:, 0] = smooth[start : start + dataset.seq_len]
        clips.append(clip)
    return clips


def score(trainer: Trainer, clips: list[Sample]) -> tuple[float, float]:
    """Steering MAE of the moving frames after burn-in, and the turn sign accuracy on them."""
    trainer.model.eval()
    errors, signs = [], []
    with torch.no_grad():
        for first in range(0, len(clips), BATCH):
            batch = default_collate(clips[first : first + BATCH])
            inputs = [
                batch[k].to(trainer.device)
                for k in ("images", "nav_command", "speed", "target_point")
            ]
            with trainer._autocast():
                controls, _, _ = trainer.model(*inputs)
            steering = controls[..., 0].float().cpu()
            expert = batch["controls"][..., 0]
            moving = batch["loss_weight"] > DrivingSequenceDataset.STATIONARY_LOSS_WEIGHT
            moving[:, : trainer.cfg.loss.burn_in] = False
            turning = moving & (expert.abs() > TURN_THRESHOLD)
            errors.append((steering - expert).abs()[moving])
            signs.append(steering[turning].sign() == expert[turning].sign())
    trainer.model.train()
    return float(torch.cat(errors).mean()), float(torch.cat(signs).float().mean())


def test_the_model_can_memorise_one_recorded_window(golden_dir: Path, tmp_path: Path) -> None:
    trainer = overfit_trainer(golden_dir, tmp_path)
    if trainer.device.type == "cpu":
        trainer.logger.close()
        pytest.skip("needs a GPU")
    clips = window_with_most_turns(cast(DrivingSequenceDataset, trainer.train_loader.dataset))
    order = torch.Generator().manual_seed(0)
    mae, sign = float("inf"), 0.0
    try:
        for step in range(1, MAX_STEPS + 1):
            rows = torch.randperm(len(clips), generator=order)[:BATCH].tolist()
            assert trainer._train_step(default_collate([clips[r] for r in rows])) is not None
            if step % CHECK_EVERY == 0:
                mae, sign = score(trainer, clips)
                print(f"step {step}: steering MAE {mae:.4f}, turn sign {sign:.1%}", flush=True)
                if mae < MAX_STEERING_MAE and sign > MIN_TURN_SIGN:
                    return
    finally:
        trainer.logger.close()
    pytest.fail(f"after {MAX_STEPS} steps: steering MAE {mae:.4f}, turn sign {sign:.1%}")
