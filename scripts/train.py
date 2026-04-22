#!/usr/bin/env python3
"""Train Vision-RWKV with crash-safe checkpoints and rich experiment logging."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.training.config import apply_dotlist_overrides, load_training_config
from src.training.trainer import Trainer


def _parse_value(raw: str) -> Any:
    return yaml.safe_load(raw)


def _parse_dot_overrides(pairs: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid override '{pair}'. Expected KEY=VALUE")
        key, value = pair.split("=", maxsplit=1)
        out[key] = _parse_value(value)
    return out


def _derive_run_location_from_resume(resume: str) -> tuple[str | None, str | None]:
    path = Path(resume)
    if path.is_dir():
        return path.name, str(path.parent)
    # Expect .../<run>/checkpoints/*.pt
    if path.is_file() and path.parent.name == "checkpoints":
        run_dir = path.parent.parent
        return run_dir.name, str(run_dir.parent)
    return None, None


def _derive_run_dir_from_resume(resume: str) -> Path | None:
    path = Path(resume)
    if path.is_dir():
        return path
    if path.is_file() and path.parent.name == "checkpoints":
        return path.parent.parent
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/vrwkv_base.yaml",
        help="Path to training YAML config (ignored if resume target has config_resolved.yaml)",
    )
    parser.add_argument("--run-name", default=None, help="Optional explicit run name")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume policy/path override: auto|never|/path/to/run_or_checkpoint",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted-key overrides, e.g. optimization.learning_rate=5e-5",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Optional cap for smoke-runs/tuning",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if args.resume is not None:
        resume_run_dir = _derive_run_dir_from_resume(args.resume)
        if resume_run_dir is not None:
            resolved_cfg = resume_run_dir / "config_resolved.yaml"
            if resolved_cfg.exists():
                cfg_path = resolved_cfg

    cfg = load_training_config(cfg_path)

    dot_overrides = _parse_dot_overrides(args.set)
    if dot_overrides:
        apply_dotlist_overrides(cfg, dot_overrides)

    if args.resume is not None:
        cfg.run.resume = args.resume
        run_name, output_root = _derive_run_location_from_resume(args.resume)
        if cfg.run.run_name is None and args.run_name is None and run_name is not None:
            cfg.run.run_name = run_name
        if output_root is not None:
            cfg.run.output_root = output_root

    if args.run_name is not None:
        cfg.run.run_name = args.run_name

    trainer = Trainer(cfg, run_name=cfg.run.run_name)
    run_dir = trainer.fit(max_epochs=args.max_epochs)

    if cfg.quantization.export_dynamic_int8:
        quant_path = trainer.export_dynamic_quantized_model()
        print(f"[train] quantized export: {quant_path}")

    print(f"[train] completed run_dir={run_dir}")


if __name__ == "__main__":
    main()
