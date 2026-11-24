"""The training step: bf16 autocast and the guard against non-finite steps."""

from __future__ import annotations

import csv
import hashlib
import math
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from src.data.driving_dataset import DrivingSequenceDataset
from src.data.labels import CAUSE_SITUATIONS
from src.training.config import (
    SchedulerConfig,
    TrainingConfig,
    apply_dotlist_overrides,
    load_training_config,
)
from src.training.trainer import (
    ControlError,
    NonFiniteTraining,
    StepTimes,
    Trainer,
    learning_rate_factor,
    token_spread,
)


def make_config(data_dir: Path, out: Path, device: str = "cpu") -> TrainingConfig:
    cfg = load_training_config("configs/smoke_toy.yaml")
    cfg.dataset.data_dir = str(data_dir)
    cfg.run.output_root = str(out)
    cfg.run.run_name = "guard"
    cfg.run.device = device
    cfg.optimization.max_nonfinite_steps = 3
    return cfg


def make_trainer(data_dir: Path, out: Path, device: str = "cpu") -> Trainer:
    return Trainer(make_config(data_dir, out, device))


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
    before, position = weights(trainer), trainer.optimizer_steps

    assert trainer._train_step(poisoned) is None
    assert unchanged(trainer, before) and trainer.optimizer_steps == position
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


def test_training_gives_the_policy_the_speed_and_target_point(trainer: Trainer) -> None:
    batch = next(iter(trainer.train_loader))
    seen: list[tuple[torch.Tensor, ...]] = []
    trainer.model.fusion.register_forward_pre_hook(lambda _, inputs: seen.append(inputs))
    trainer._compute_losses(batch)
    assert torch.equal(seen[0][2], batch["speed"])
    assert torch.equal(seen[0][3], batch["target_point"])


def test_the_backbone_trains_at_a_fraction_of_the_learning_rate(trainer: Trainer) -> None:
    rest, backbone = trainer.optimizer.param_groups
    scale = trainer.cfg.optimization.backbone_lr_scale
    assert scale < 1.0 and backbone["lr"] == pytest.approx(rest["lr"] * scale)
    in_backbone = {id(p) for p in trainer.model.tokenizer.backbone.parameters()}
    assert {id(p) for p in backbone["params"]} == in_backbone
    assert len(rest["params"]) + len(in_backbone) == len(list(trainer.model.parameters()))


def test_a_situation_head_trains_only_when_its_loss_counts(
    trainer: Trainer, toy_data_dir: Path, tmp_path: Path
) -> None:
    batch = next(iter(trainer.train_loader))
    assert trainer.model.situation_head is None
    assert trainer._compute_losses(batch)[1]["situation_loss"] == 0.0

    cfg = make_config(toy_data_dir, tmp_path / "situation")
    apply_dotlist_overrides(cfg, {"loss.situation_weight": 0.2})
    t = Trainer(cfg)
    assert t.model.situation_head is not None and t.situation_pos_weight is not None
    train = cast(DrivingSequenceDataset, t.train_loader.dataset)
    share = np.concatenate(train.situations)[:, CAUSE_SITUATIONS].any(-1).mean(0)
    assert int(t.situation_pos_weight.argmax()) == int(share.argmin())  # the rarest weighs most
    assert t._compute_losses(batch)[1]["situation_loss"] > 0.0
    t.logger.close()


def test_the_learning_rate_follows_the_config_and_the_step_count() -> None:
    cosine = SchedulerConfig(name="cosine", warmup_steps=10, min_lr_ratio=0.1)
    assert learning_rate_factor(0, 110, cosine) == pytest.approx(0.1)
    assert learning_rate_factor(10, 110, cosine) == pytest.approx(1.0)
    assert learning_rate_factor(60, 110, cosine) == pytest.approx(0.55)
    assert learning_rate_factor(110, 110, cosine) == pytest.approx(0.1)
    assert learning_rate_factor(5, 110, SchedulerConfig(name="constant")) == 1.0


def test_a_resumed_run_continues_with_its_weights_steps_and_rate(
    toy_data_dir: Path, tmp_path: Path
) -> None:
    first = make_trainer(toy_data_dir, tmp_path)
    batch = next(iter(first.train_loader))
    for _ in range(3):
        first._train_step(batch)
    first._save_checkpoint(tag="step")
    first.logger.close()

    cfg = make_config(toy_data_dir, tmp_path)
    cfg.run.resume = "auto"
    cfg.dataset.num_workers += 1  # where and how a run is executed may change
    second = Trainer(cfg)
    assert second.optimizer_steps == 3 and unchanged(second, weights(first))
    first._train_step(batch)
    second._train_step(batch)
    assert second._current_lr() == first._current_lr()
    second.logger.close()


def test_a_resume_that_would_change_the_training_is_refused(
    toy_data_dir: Path, tmp_path: Path
) -> None:
    first = make_trainer(toy_data_dir, tmp_path)
    first._save_checkpoint(tag="step")
    first.logger.close()
    record = (first.run_dir / "config_resolved.yaml").read_text()

    cfg = make_config(toy_data_dir, tmp_path)
    cfg.run.resume = "auto"
    cfg.optimization.learning_rate *= 10
    with pytest.raises(ValueError, match="optimization.learning_rate"):
        Trainer(cfg)
    assert (first.run_dir / "config_resolved.yaml").read_text() == record


def test_a_warm_start_begins_a_new_run_from_another_runs_weights(
    toy_data_dir: Path, tmp_path: Path
) -> None:
    parent = make_trainer(toy_data_dir, tmp_path)
    parent._train_step(next(iter(parent.train_loader)))
    checkpoint = parent._save_checkpoint(tag="step")
    parent.logger.close()

    cfg = make_config(toy_data_dir, tmp_path)
    cfg.run.run_name = "child"
    cfg.run.init_from = str(checkpoint)
    cfg.optimization.learning_rate *= 0.1  # a new run may change anything
    child = Trainer(cfg)
    assert unchanged(child, weights(parent))
    assert child.optimizer_steps == 0 and not child.optimizer.state
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert events(child, "warm_start")[0]["value"] == f"sha256:{digest}"
    child.logger.close()


def test_token_spread_is_zero_only_when_every_frame_gets_the_same_token() -> None:
    assert token_spread(torch.ones(2, 3, 4)) == 0.0
    assert token_spread(torch.randn(2, 3, 4)) > 0.1


def test_step_times_average_the_steps_since_the_last_report() -> None:
    times = StepTimes()
    times.add(1.0, 3.0)
    times.add(3.0, 5.0)
    assert times.report() == {"data_wait_s": 2.0, "train_step_s": 4.0}
    times.add(0.5, 0.25)
    assert times.report() == {"data_wait_s": 0.5, "train_step_s": 0.25}


def test_control_error_pools_the_frames_that_count() -> None:
    target = torch.tensor([[[0.0, 0.0], [0.5, 0.0], [0.5, 0.0], [0.5, 0.0]]])
    errors = torch.tensor([[[0.0, 0.2], [0.4, -0.8], [0.9, 0.9], [0.9, 0.9]]])
    # moving straight, moving turn, stationary turn, burn-in or excluded
    weight = torch.tensor([[1.0, 1.0, 0.1, 0.0]])

    error = ControlError()
    error.add(target + errors, target, weight)
    error.add(torch.tensor([[[0.2, 0.8]]]), torch.zeros(1, 1, 2), torch.ones(1, 1))

    metrics = error.metrics()
    assert metrics["moving_control_mae"] == pytest.approx((0.1 + 0.6 + 0.5) / 3)
    assert metrics["turn_control_mae"] == pytest.approx(0.6)
    assert metrics["control_score"] == pytest.approx(0.5)


def test_control_score_without_turns_is_the_moving_error() -> None:
    error = ControlError()
    error.add(torch.full((1, 2, 2), 0.2), torch.zeros(1, 2, 2), torch.ones(1, 2))
    metrics = error.metrics()
    assert math.isnan(metrics["turn_control_mae"])
    assert metrics["control_score"] == pytest.approx(0.2)


def test_a_score_that_is_not_a_number_never_becomes_the_best(trainer: Trainer) -> None:
    error = ControlError()
    error.add(torch.zeros(1, 2, 2), torch.zeros(1, 2, 2), torch.full((1, 2), 0.1))
    score = error.metrics()["control_score"]

    assert math.isnan(score) and not trainer._is_better(score)
    trainer.best_metric = 0.5
    assert trainer._is_better(0.4) and not trainer._is_better(0.6)


def test_validation_logs_the_score_that_selects_checkpoints(trainer: Trainer) -> None:
    monitored = trainer.cfg.checkpoint.monitor_metric.split("/", maxsplit=1)[-1]
    metrics = trainer.evaluate()
    assert monitored == "control_score" and math.isfinite(metrics[monitored])

    trainer.logger.log_metrics("val", 1, 0, metrics)
    with trainer.logger.metrics_path.open(encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    assert float(row["control_score"]) == pytest.approx(metrics["control_score"])
    assert metrics["token_spread"] > 0.0
    assert float(row["token_spread"]) == pytest.approx(metrics["token_spread"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 autocast is used on CUDA only")
def test_training_runs_in_bf16(toy_data_dir: Path, tmp_path: Path) -> None:
    t = make_trainer(toy_data_dir, tmp_path, "cuda")
    batch = next(iter(t.train_loader))
    inputs = [batch[k].cuda() for k in ("images", "nav_command", "speed", "target_point")]
    with t._autocast():
        controls, _, _ = t.model(*inputs)
    assert controls.dtype == torch.bfloat16
    assert t._train_step(batch) is not None
    t.logger.close()
