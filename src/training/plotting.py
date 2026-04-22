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
    """Create a 6-panel run dashboard PNG from CSV logs.

    Panels:
        [0,0] Total Loss          — train (smoothed) + val dots
        [0,1] Action Error (MAE)  — steer/accel MAE, train (smoothed) + val dots
        [1,0] Control & Waypoint  — per-component losses, train + val
        [1,1] Auxiliary Losses    — speed loss + mode loss, train + val
        [2,0] Optimization        — LR (left axis) + grad norm (right axis)
        [2,1] Distillation Loss   — distill loss if enabled, else mode logit loss
    """
    run_dir = Path(run_dir)
    logs_dir = run_dir / "logs"
    metrics_rows = _read_csv_rows(logs_dir / "metrics.csv")
    events_rows = _read_csv_rows(logs_dir / "events.csv")

    if not metrics_rows:
        raise FileNotFoundError(f"No metrics found at {logs_dir / 'metrics.csv'}")

    train_rows = [r for r in metrics_rows if r.get("split") == "train"]
    val_rows = [r for r in metrics_rows if r.get("split") == "val"]

    tr_steps = _to_float_array(train_rows, "step")
    va_steps = _to_float_array(val_rows, "step")
    has_val = len(val_rows) > 0

    # ── Losses ──────────────────────────────────────────────────────────────
    tr_total = _to_float_array(train_rows, "total_loss")
    va_total = _to_float_array(val_rows, "total_loss")

    tr_control = _to_float_array(train_rows, "control_loss")
    va_control = _to_float_array(val_rows, "control_loss")

    tr_wp = _to_float_array(train_rows, "waypoint_loss")
    va_wp = _to_float_array(val_rows, "waypoint_loss")

    tr_speed = _to_float_array(train_rows, "speed_loss")
    va_speed = _to_float_array(val_rows, "speed_loss")

    tr_mode = _to_float_array(train_rows, "mode_loss")
    va_mode = _to_float_array(val_rows, "mode_loss")

    tr_distill = _to_float_array(train_rows, "distill_loss")
    va_distill = _to_float_array(val_rows, "distill_loss")

    # ── Optimisation ────────────────────────────────────────────────────────
    lr = _to_float_array(train_rows, "lr")
    grad_norm = _to_float_array(train_rows, "grad_norm")

    # ── Action error ────────────────────────────────────────────────────────
    tr_steer_mae = _to_float_array(train_rows, "steer_mae")
    va_steer_mae = _to_float_array(val_rows, "steer_mae")
    tr_accel_mae = _to_float_array(train_rows, "accel_mae")
    va_accel_mae = _to_float_array(val_rows, "accel_mae")

    # ── Layout ──────────────────────────────────────────────────────────────
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(18, 14), dpi=150)

    val_kw = dict(marker="o", markersize=4, linewidth=1.2, alpha=0.85)

    # ── [0,0] Total loss ────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(
        tr_steps, _exp_smooth(tr_total), label="train (smoothed)", color="#1f77b4", linewidth=1.8
    )
    ax.plot(tr_steps, tr_total, color="#1f77b4", linewidth=0.4, alpha=0.25)
    if has_val:
        ax.plot(va_steps, va_total, label="val", color="#ff7f0e", **val_kw)
    ax.set_title("Total Loss")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    _draw_event_lines(ax, events_rows)

    # ── [0,1] Action error ──────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(
        tr_steps, _exp_smooth(tr_steer_mae), label="train steer MAE", color="#1f77b4", linewidth=1.6
    )
    ax.plot(
        tr_steps, _exp_smooth(tr_accel_mae), label="train accel MAE", color="#ff7f0e", linewidth=1.6
    )
    if has_val:
        ax.plot(va_steps, va_steer_mae, label="val steer MAE", color="#2ca02c", **val_kw)
        ax.plot(va_steps, va_accel_mae, label="val accel MAE", color="#d62728", **val_kw)
    ax.set_title("Action Error (MAE)")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Mean Absolute Error")
    ax.legend(loc="upper right")
    _draw_event_lines(ax, events_rows)

    # ── [1,0] Control + waypoint losses ─────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(
        tr_steps, _exp_smooth(tr_control), label="train control", color="#2ca02c", linewidth=1.6
    )
    ax.plot(tr_steps, _exp_smooth(tr_wp), label="train waypoint", color="#9467bd", linewidth=1.2)
    if has_val:
        ax.plot(va_steps, va_control, label="val control", color="#17becf", **val_kw)
        ax.plot(va_steps, va_wp, label="val waypoint", color="#8c564b", **val_kw)
    ax.set_title("Control & Waypoint Loss")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    _draw_event_lines(ax, events_rows)

    # ── [1,1] Auxiliary losses (speed + mode) ────────────────────────────────
    ax = axes[1, 1]
    ax.plot(tr_steps, _exp_smooth(tr_speed), label="train speed", color="#e377c2", linewidth=1.6)
    ax.plot(tr_steps, _exp_smooth(tr_mode), label="train mode", color="#7f7f7f", linewidth=1.2)
    if has_val:
        ax.plot(va_steps, va_speed, label="val speed", color="#bcbd22", **val_kw)
        ax.plot(va_steps, va_mode, label="val mode", color="#17becf", **val_kw)
    ax.set_title("Auxiliary Losses (Speed + Driving Mode)")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    _draw_event_lines(ax, events_rows)

    # ── [2,0] LR + grad norm ─────────────────────────────────────────────────
    ax = axes[2, 0]
    ax.plot(tr_steps, lr, label="learning rate", color="#d62728", linewidth=1.5)
    ax2 = ax.twinx()
    ax2.plot(tr_steps, _exp_smooth(grad_norm), label="grad norm", color="#7f7f7f", linewidth=1.2)
    ax.set_title("Optimisation Health")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("LR")
    ax2.set_ylabel("Grad Norm")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    _draw_event_lines(ax, events_rows)

    # ── [2,1] Distillation loss ──────────────────────────────────────────────
    ax = axes[2, 1]
    distill_nonzero = not np.all(np.isnan(tr_distill) | (tr_distill == 0.0))
    if distill_nonzero:
        ax.plot(
            tr_steps, _exp_smooth(tr_distill), label="train distill", color="#1f77b4", linewidth=1.6
        )
        if has_val:
            ax.plot(va_steps, va_distill, label="val distill", color="#ff7f0e", **val_kw)
        ax.set_title("Distillation Loss")
    else:
        # If no distillation, reuse panel for a train/val total loss overlay (log scale)
        ax.semilogy(
            tr_steps,
            _exp_smooth(tr_total),
            label="train total (log)",
            color="#1f77b4",
            linewidth=1.6,
        )
        if has_val:
            ax.semilogy(va_steps, va_total, label="val total (log)", color="#ff7f0e", **val_kw)
        ax.set_title("Total Loss (log scale)")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Loss")
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
