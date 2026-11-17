"""Metrics for offline evaluation of driving policies, on flat per-frame numpy arrays."""

from __future__ import annotations

import numpy as np

TURN_THRESHOLD = 0.1  # |expert steering| above this counts as a turning frame


def weighted_mae(pred: np.ndarray, target: np.ndarray, weight: np.ndarray | None = None) -> float:
    err = np.abs(np.asarray(pred, np.float64) - np.asarray(target, np.float64))
    if weight is None:
        return float(err.mean())
    w = np.asarray(weight, np.float64)
    return float((err * w).sum() / max(w.sum(), 1e-6))


def turn_frames(steering: np.ndarray, threshold: float = TURN_THRESHOLD) -> np.ndarray:
    return np.asarray(np.abs(steering) > threshold, dtype=bool)


def sign_accuracy(pred: np.ndarray, target: np.ndarray, weight: np.ndarray | None = None) -> float:
    """Share of frames where the predicted steering has the expert's sign."""
    match = (np.sign(pred) == np.sign(target)).astype(np.float64)
    if weight is None:
        return float(match.mean())
    w = np.asarray(weight, np.float64)
    return float((match * w).sum() / max(w.sum(), 1e-6))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def camera_swap_score(pred: np.ndarray, pred_mirrored: np.ndarray) -> float:
    """Correlation of the steering with the negated steering on mirrored frames.

    +1: the policy mirrors its steering when the scene is mirrored (it uses the cameras).
    -1: mirroring changes nothing (the steering does not depend on the image layout).
    """
    return pearson(pred, -np.asarray(pred_mirrored))


def command_means(values: np.ndarray, commands: np.ndarray) -> dict[int, float]:
    return {int(c): float(values[commands == c].mean()) for c in np.unique(commands)}


def predict_by_command(
    means: dict[int, float], commands: np.ndarray, fallback: float
) -> np.ndarray:
    return np.array([means.get(int(c), fallback) for c in commands], dtype=np.float64)
