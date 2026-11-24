"""Metrics for offline evaluation of driving policies, on flat per-frame numpy arrays."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

TURN_THRESHOLD = 0.1  # |expert steering| above this counts as a turning frame
TARGET_BINS = 60  # bearings of the target point told apart by the target point baseline
TARGET_NEIGHBOURS = 8  # training frames averaged by the nearest-neighbour target baseline

TargetLookup = tuple[np.ndarray, np.ndarray]  # bin edges of the bearing, the mean of each bin


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


def _bearing(points: np.ndarray) -> np.ndarray:
    return np.asarray(np.arctan2(points[:, 1], points[:, 0]))


def fit_target_lookup(
    values: np.ndarray, points: np.ndarray, bins: int = TARGET_BINS
) -> TargetLookup:
    """The mean of `values` per bearing of the (N, 2) target points, in bins of equal share."""
    bearing = _bearing(points)
    edges = np.unique(np.quantile(bearing, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    index = np.searchsorted(edges, bearing)
    counts = np.bincount(index, minlength=len(edges) + 1)
    totals = np.bincount(index, weights=values, minlength=len(edges) + 1)
    means = np.full(len(counts), float(np.mean(values)))
    means[counts > 0] = totals[counts > 0] / counts[counts > 0]
    return edges, means


def predict_by_target(lookup: TargetLookup, points: np.ndarray) -> np.ndarray:
    """What an image-free policy that sees only the target point would output."""
    edges, means = lookup
    return means[np.searchsorted(edges, _bearing(points))]


def predict_by_neighbours(
    values: np.ndarray, points: np.ndarray, query: np.ndarray, k: int = TARGET_NEIGHBOURS
) -> np.ndarray:
    """The mean of `values` (N, C) at the k training target points (N, 2) nearest each query
    point: an image-free policy that reads the whole target point, not only its bearing."""
    _, nearest = cKDTree(points).query(query, k=k)
    return np.asarray(values[nearest.reshape(len(query), -1)].mean(axis=1))
