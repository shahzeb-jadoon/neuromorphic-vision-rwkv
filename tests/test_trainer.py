"""The training step: bf16 autocast and the guard against non-finite steps."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
import torch

from src.data.driving_dataset import DrivingSequenceDataset
from src.training.config import load_training_config
from src.training.trainer import NonFiniteTraining, Trainer


def make_trainer(data_dir: Path, out: Path, device: str = "cpu") -> Trainer:
    cfg = load_training_config("configs/smoke_toy.yaml")
    cfg.dataset.data_dir = str(data_dir)
    cfg.run.output_root = str(out)
    cfg.run.run_name = "guard"
    cfg.run.device = device
    cfg.optimization.max_nonfinite_steps = 3
    return Trainer(cfg)


@pytest.fixture
def trainer(toy_data_dir: Path, tmp_path: Path) -> Iterator[Trainer]:
    t = make_trainer(toy_data_dir, tmp_path)
    yield t
    t.logger.close()


def weights(t: Trainer) -> list[torch.Tensor]:
    return [p.detach().clone() for p in t.model.parameters()]


def unchanged(t: Trainer, before: list[torch.Tensor]) -> bool:
    return all(torch.equal(a, b) for a, b in zip(before, t.model.parameters()))


def events(t: Trainer, kind: str) -> list[dict[str, str]]:
    with t.logger.events_path.open(encoding="utf-8", newline="") as f:
        return [row for row in csv.DictReader(f) if row["event_type"] == kind]


def test_a_non_finite_loss_skips_the_step_and_names_its_clips(trainer: Trainer) -> None:
    batch = next(iter(trainer.train_loader))
    poisoned = {**batch, "controls": batch["controls"].clone()}
    poisoned["controls"][0, 8, 0] = float("nan")
    before, position = weights(trainer), trainer.scheduler.last_epoch

    assert trainer._train_step(poisoned) is None
    assert unchanged(trainer, before) and trainer.scheduler.last_epoch == position
    assert not trainer.optimizer.state

    event = events(trainer, "nonfinite_step")[0]
    episode, start = batch["clip"][0].tolist()
    files = cast(DrivingSequenceDataset, trainer.train_loader.dataset).episode_files
    assert "control_loss" in event["detail"]
    assert f"{files[episode].name}:{start}" in event["value"]

    assert trainer._train_step(batch) is not None
    assert not unchanged(trainer, before) and trainer.nonfinite_in_a_row == 0


def test_a_non_finite_gradient_skips_the_step_and_names_the_parameter(trainer: Trainer) -> None:
    batch = next(iter(trainer.train_loader))
    name, parameter = next(iter(trainer.model.named_parameters()))
    hook = parameter.register_hook(lambda g: g * float("nan"))
    before = weights(trainer)

    assert trainer._train_step(batch) is None
    assert unchanged(trainer, before)
    assert events(trainer, "nonfinite_step")[0]["detail"] == f"gradient of {name}"

    hook.remove()
    assert trainer._train_step(batch) is not None


def test_training_aborts_after_too_many_non_finite_steps_in_a_row(trainer: Trainer) -> None:
    batch = next(iter(trainer.train_loader))
    poisoned = {**batch, "controls": batch["controls"] * float("nan")}
    trainer._train_step(poisoned)
    trainer._train_step(poisoned)
    assert trainer._train_step(batch) is not None  # a good step starts the count again
    trainer._train_step(poisoned)
    trainer._train_step(poisoned)
    with pytest.raises(NonFiniteTraining, match="3 steps in a row"):
        trainer._train_step(poisoned)
    assert all(p.isfinite().all() for p in trainer.model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 autocast is used on CUDA only")
def test_training_runs_in_bf16(toy_data_dir: Path, tmp_path: Path) -> None:
    t = make_trainer(toy_data_dir, tmp_path, "cuda")
    batch = next(iter(t.train_loader))
    images, nav = batch["images"].cuda(), batch["nav_command"].cuda()
    with t._autocast():
        controls, _, _ = t.model(images, nav)
    assert controls.dtype == torch.bfloat16
    assert t._train_step(batch) is not None
    t.logger.close()
