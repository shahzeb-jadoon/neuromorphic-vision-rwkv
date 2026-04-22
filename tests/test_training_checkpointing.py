"""Tests for checkpoint manager behavior."""

from __future__ import annotations

from pathlib import Path

from src.training.checkpointing import CheckpointManager
from src.training.config import CheckpointConfig


def test_checkpoint_save_latest_and_prune(tmp_path: Path) -> None:
    cfg = CheckpointConfig(keep_last_n=2)
    manager = CheckpointManager(tmp_path / "run", cfg)

    for step in [10, 20, 30]:
        manager.save(payload={"model": {"w": step}}, step=step, epoch=1)

    step_files = sorted((tmp_path / "run" / "checkpoints").glob("step_*.pt"))
    assert len(step_files) == 2
    assert manager.latest_path.exists()


def test_checkpoint_resolve_resume(tmp_path: Path) -> None:
    cfg = CheckpointConfig()
    manager = CheckpointManager(tmp_path / "run", cfg)

    assert manager.resolve_resume_path("auto") is None
    manager.save(payload={"model": {"w": 1}}, step=1, epoch=0)

    auto_path = manager.resolve_resume_path("auto")
    assert auto_path is not None
    assert auto_path.name == "latest.pt"


def test_checkpoint_resolve_explicit_run_dir(tmp_path: Path) -> None:
    cfg = CheckpointConfig()
    manager = CheckpointManager(tmp_path / "run", cfg)
    manager.save(payload={"model": {"w": 1}}, step=1, epoch=0)

    resolved = manager.resolve_resume_path(str(tmp_path / "run"))
    assert resolved is not None
    assert resolved.name == "latest.pt"
