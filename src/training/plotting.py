"""Run-level plotting utilities for glanceable experiment dashboards."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float_array(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value == "":
            values.append(np.nan)
        else:
            values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def _exp_smooth(x: np.ndarray, alpha: float = 0.08) -> np.ndarray:
    if x.size == 0:
        return x
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        val = x[i]
        if np.isnan(val):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * val + (1.0 - alpha) * out[i - 1]
    return out


def _draw_event_lines(ax: Any, events: list[dict[str, str]]) -> None:
    for event in events:
        if event.get("event_type") != "override":
            continue
        step = float(event["step"])
        detail = event.get("detail", "override")
        ax.axvline(step, color="#d62728", linestyle="--", linewidth=0.8, alpha=0.8)
        ax.text(step, ax.get_ylim()[1], detail, rotation=90, va="top", ha="right", fontsize=7)


def plot_run_dashboard(run_dir: str | Path) -> Path:
    """Create a 4-panel run dashboard PNG from CSV logs."""
    run_dir = Path(run_dir)
    logs_dir = run_dir / "logs"
    metrics_rows = _read_csv_rows(logs_dir / "metrics.csv")
    events_rows = _read_csv_rows(logs_dir / "events.csv")

    if not metrics_rows:
        raise FileNotFoundError(f"No metrics found at {logs_dir / 'metrics.csv'}")

    train_rows = [r for r in metrics_rows if r.get("split") == "train"]
    val_rows = [r for r in metrics_rows if r.get("split") == "val"]

    train_steps = _to_float_array(train_rows, "step")
    val_steps = _to_float_array(val_rows, "step")

    train_total = _to_float_array(train_rows, "total_loss")
    val_total = _to_float_array(val_rows, "total_loss")

    train_control = _to_float_array(train_rows, "control_loss")
    val_control = _to_float_array(val_rows, "control_loss")

    train_wp = _to_float_array(train_rows, "waypoint_loss")
    val_wp = _to_float_array(val_rows, "waypoint_loss")

    lr = _to_float_array(train_rows, "lr")
    grad_norm = _to_float_array(train_rows, "grad_norm")

    steer_mae_train = _to_float_array(train_rows, "steer_mae")
    accel_mae_train = _to_float_array(train_rows, "accel_mae")
    steer_mae_val = _to_float_array(val_rows, "steer_mae")
    accel_mae_val = _to_float_array(val_rows, "accel_mae")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=150)

    ax = axes[0, 0]
    ax.plot(
        train_steps, _exp_smooth(train_total), label="train total", color="#1f77b4", linewidth=1.8
    )
    if len(val_rows) > 0:
        ax.plot(val_steps, val_total, label="val total", color="#ff7f0e", marker="o", linewidth=1.2)
    ax.set_title("Total Loss")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    _draw_event_lines(ax, events_rows)

    ax = axes[0, 1]
    ax.plot(
        train_steps,
        _exp_smooth(train_control),
        label="train control",
        color="#2ca02c",
        linewidth=1.6,
    )
    ax.plot(
        train_steps, _exp_smooth(train_wp), label="train waypoint", color="#9467bd", linewidth=1.2
    )
    if len(val_rows) > 0:
        ax.plot(
            val_steps, val_control, label="val control", color="#17becf", marker="o", linewidth=1.0
        )
        ax.plot(val_steps, val_wp, label="val waypoint", color="#8c564b", marker="o", linewidth=1.0)
    ax.set_title("Primary Components")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    _draw_event_lines(ax, events_rows)

    ax = axes[1, 0]
    ax.plot(train_steps, lr, label="learning rate", color="#d62728", linewidth=1.5)
    ax2 = ax.twinx()
    ax2.plot(train_steps, _exp_smooth(grad_norm), label="grad norm", color="#7f7f7f", linewidth=1.2)
    ax.set_title("Optimization Health")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("LR")
    ax2.set_ylabel("Grad Norm")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    _draw_event_lines(ax, events_rows)

    ax = axes[1, 1]
    ax.plot(train_steps, _exp_smooth(steer_mae_train), label="train steer MAE", color="#1f77b4")
    ax.plot(train_steps, _exp_smooth(accel_mae_train), label="train accel MAE", color="#ff7f0e")
    if len(val_rows) > 0:
        ax.plot(val_steps, steer_mae_val, label="val steer MAE", color="#2ca02c", marker="o")
        ax.plot(val_steps, accel_mae_val, label="val accel MAE", color="#d62728", marker="o")
    ax.set_title("Action Error")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Absolute Error")
    ax.legend(loc="upper right")
    _draw_event_lines(ax, events_rows)

    fig.suptitle(f"Training Dashboard: {run_dir.name}", fontsize=14)
    fig.tight_layout()

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / "dashboard.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
