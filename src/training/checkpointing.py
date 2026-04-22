"""Checkpoint and RNG state utilities for crash-safe training."""

from __future__ import annotations

import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from src.training.config import CheckpointConfig


def capture_rng_state() -> dict[str, Any]:
    """Capture Python/NumPy/Torch RNG state for exact resume."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state if present in a checkpoint."""
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch_cpu = state["torch_cpu"]
        if isinstance(torch_cpu, torch.Tensor):
            torch_cpu = torch_cpu.detach().cpu().to(dtype=torch.uint8)
        torch.set_rng_state(torch_cpu)
    if "torch_cuda" in state and torch.cuda.is_available():
        cuda_states = state["torch_cuda"]
        normalized: list[torch.Tensor] = []
        for tensor in cuda_states:
            if isinstance(tensor, torch.Tensor):
                normalized.append(tensor.detach().cpu().to(dtype=torch.uint8))
        torch.cuda.set_rng_state_all(normalized)


class CheckpointManager:
    """Manages periodic, latest, and best checkpoints with retention."""

    def __init__(self, run_dir: Path, cfg: CheckpointConfig) -> None:
        self.run_dir = run_dir
        self.cfg = cfg
        self.ckpt_dir = run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    @property
    def latest_path(self) -> Path:
        return self.ckpt_dir / "latest.pt"

    @property
    def best_path(self) -> Path:
        return self.ckpt_dir / "best.pt"

    def _atomic_save(self, payload: dict[str, Any], dest: Path) -> None:
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, dest)

    def save(
        self,
        payload: dict[str, Any],
        *,
        step: int,
        epoch: int,
        is_best: bool = False,
        tag: str | None = None,
    ) -> Path:
        """Save a step checkpoint and update latest/best aliases."""
        suffix = f"_{tag}" if tag else ""
        step_path = self.ckpt_dir / f"step_{step:08d}_epoch_{epoch:04d}{suffix}.pt"

        meta = {
            "saved_at_unix": time.time(),
            "step": step,
            "epoch": epoch,
            "tag": tag,
        }
        payload = {**payload, "checkpoint_meta": meta}

        self._atomic_save(payload, step_path)
        shutil.copy2(step_path, self.latest_path)

        if is_best:
            shutil.copy2(step_path, self.best_path)

        self._prune_old_step_checkpoints()
        return step_path

    def _prune_old_step_checkpoints(self) -> None:
        """Keep only the most recent N step checkpoints."""
        keep = max(1, int(self.cfg.keep_last_n))
        step_files = sorted(self.ckpt_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
        if len(step_files) <= keep:
            return
        for old in step_files[:-keep]:
            try:
                old.unlink()
            except FileNotFoundError:
                pass

    def resolve_resume_path(self, resume: str) -> Path | None:
        """Resolve resume policy to an actual checkpoint path."""
        policy = resume.strip()
        if policy == "never":
            return None
        if policy == "auto":
            return self.latest_path if self.latest_path.exists() else None

        candidate = Path(policy)
        if candidate.is_dir():
            candidate = candidate / "checkpoints" / "latest.pt"
        if not candidate.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {candidate}")
        return candidate

    @staticmethod
    def load(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        """Load a checkpoint payload."""
        return cast(dict[str, Any], torch.load(path, map_location=map_location, weights_only=False))
