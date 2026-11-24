"""Tests for Phase-2 training configuration and naming utilities."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.training.config import (
    apply_dotlist_overrides,
    load_training_config,
    resume_conflicts,
    training_config_to_dict,
)
from src.training.trainer import build_run_name


def test_load_training_config_and_model_alignment() -> None:
    cfg = load_training_config("configs/vrwkv_base.yaml")
    assert cfg.model.seq_len == cfg.dataset.seq_len
    assert cfg.model.batch_size == cfg.dataset.batch_size
    assert cfg.model.learning_rate == cfg.optimization.learning_rate


def test_resize_overrides_propagate_to_model(tmp_path: Path) -> None:
    with Path("configs/vrwkv_base.yaml").open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    data["dataset"]["resize_height"] = 160
    data["dataset"]["resize_width"] = 960
    cfg_path = tmp_path / "cfg.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    cfg = load_training_config(cfg_path)
    assert cfg.model.tokenizer.img_height == 160
    assert cfg.model.tokenizer.img_width == 960


def test_run_name_contains_key_parameters() -> None:
    cfg = load_training_config("configs/vrwkv_base.yaml")
    run_name = build_run_name(cfg)
    assert cfg.run.name_prefix in run_name
    assert f"_e{cfg.model.encoder.n_layers}d{cfg.model.encoder.embed_dim}" in run_name
    assert f"_s{cfg.dataset.seq_len}b{cfg.dataset.batch_size}" in run_name


def test_dotlist_override_updates_nested_fields() -> None:
    cfg = load_training_config("configs/vrwkv_base.yaml")
    apply_dotlist_overrides(
        cfg,
        {
            "optimization.learning_rate": 5e-5,
            "loss.waypoint_weight": 0.75,
            "run.run_name": "unit_test_run",
        },
    )
    assert cfg.optimization.learning_rate == 5e-5
    assert cfg.loss.waypoint_weight == 0.75
    assert cfg.run.run_name == "unit_test_run"


def test_a_rate_written_without_a_decimal_point_is_a_number(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("optimization:\n  learning_rate: 1e-4\n", encoding="utf-8")
    cfg = load_training_config(path)
    assert cfg.optimization.learning_rate == 1e-4

    apply_dotlist_overrides(cfg, {"optimization.learning_rate": yaml.safe_load("1e-5")})
    assert cfg.optimization.learning_rate == 1e-5 and cfg.model.learning_rate == 1e-5


def test_older_runs_load_and_resume_with_retired_and_later_settings(tmp_path: Path) -> None:
    data = yaml.safe_load(Path("configs/short_run.yaml").read_text(encoding="utf-8"))
    data["runtime_overrides"] = {"enabled": True, "poll_every_steps": 100}
    old = tmp_path / "config_resolved.yaml"
    old.write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg = load_training_config(old)

    saved = training_config_to_dict(cfg) | {"runtime_overrides": data["runtime_overrides"]}
    del saved["run"]["init_from"]  # added after the run was saved: counts as its default
    assert resume_conflicts(saved, cfg) == []
    cfg.scheduler.warmup_steps += 1
    assert resume_conflicts(saved, cfg) == ["scheduler.warmup_steps: 50 -> 51"]


def test_config_file_exists() -> None:
    assert Path("configs/vrwkv_base.yaml").exists()
