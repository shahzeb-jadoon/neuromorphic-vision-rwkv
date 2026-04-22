#!/usr/bin/env python3
"""Optuna-based hyperparameter tuning for Vision-RWKV training."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import yaml

from src.training.config import apply_dotlist_overrides, load_training_config
from src.training.trainer import Trainer


def _read_best_val_loss(run_dir: Path) -> float:
    metrics_path = run_dir / "logs" / "metrics.csv"
    best = float("inf")
    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") != "val":
                continue
            value = row.get("total_loss", "")
            if value == "":
                continue
            best = min(best, float(value))
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Base training config YAML")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--trial-epochs", type=int, default=3)
    parser.add_argument("--study-name", default="vrwkv_tuning")
    parser.add_argument(
        "--storage",
        default="runs/optuna_study.db",
        help="Optuna storage URI or sqlite file path",
    )
    args = parser.parse_args()

    try:
        import optuna
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "optuna is required for tuning. Install with: pip install optuna"
        ) from exc

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    storage = args.storage
    if "://" not in storage:
        storage_path = Path(storage).expanduser().resolve()
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{storage_path}"

    def objective(trial: "optuna.Trial") -> float:
        cfg = load_training_config(args.config)
        cfg.run.resume = "never"
        cfg.logging.use_wandb = False
        cfg.logging.plot_on_finish = False
        cfg.logging.val_max_batches = 20

        overrides = {
            "optimization.learning_rate": trial.suggest_float(
                "optimization.learning_rate", 1e-5, 5e-4, log=True
            ),
            "optimization.weight_decay": trial.suggest_float(
                "optimization.weight_decay", 1e-5, 1e-1, log=True
            ),
            "dataset.batch_size": trial.suggest_categorical("dataset.batch_size", [2, 4, 8]),
            "loss.speed_weight": trial.suggest_float("loss.speed_weight", 0.05, 0.5),
            "loss.waypoint_weight": trial.suggest_float("loss.waypoint_weight", 0.1, 1.0),
            "model.fusion.dropout": trial.suggest_float("model.fusion.dropout", 0.0, 0.2),
            "model.policy_head.dropout": trial.suggest_float("model.policy_head.dropout", 0.0, 0.2),
            "model.encoder.dropout": trial.suggest_float("model.encoder.dropout", 0.0, 0.2),
        }
        apply_dotlist_overrides(cfg, overrides)

        cfg.run.output_root = str(Path(cfg.run.output_root) / "tuning" / args.study_name)
        cfg.run.run_name = f"trial_{trial.number:04d}_{timestamp}"

        trainer = Trainer(cfg, run_name=cfg.run.run_name)
        run_dir = trainer.fit(max_epochs=args.trial_epochs)
        score = _read_best_val_loss(Path(run_dir))
        return score

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("[tune] best_value=", study.best_value)
    print("[tune] best_params=")
    print(yaml.safe_dump(study.best_params, sort_keys=True))

    out_path = Path("runs") / f"best_params_{args.study_name}_{timestamp}.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(study.best_params, f, sort_keys=True)
    print(f"[tune] wrote {out_path}")


if __name__ == "__main__":
    main()
