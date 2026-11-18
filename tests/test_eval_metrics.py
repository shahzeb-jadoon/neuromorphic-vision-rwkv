"""Evaluation metrics against hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest

from src.eval import metrics as M


def test_weighted_mae() -> None:
    pred, target = np.array([1.0, -1.0, 0.5]), np.zeros(3)
    assert M.weighted_mae(pred, target) == pytest.approx(2.5 / 3)
    assert M.weighted_mae(pred, target, np.array([1.0, 0.0, 1.0])) == pytest.approx(0.75)
    assert M.weighted_mae(pred, target, np.array([0.1, 1.0, 1.0])) == pytest.approx(1.6 / 2.1)


def test_turn_frames_and_sign_accuracy() -> None:
    target = np.array([0.5, -0.5, -0.2, 0.05, 0.1])
    pred = np.array([0.2, -0.3, 0.1, -0.4, 0.3])
    turns = M.turn_frames(target)
    assert turns.tolist() == [True, True, True, False, False]
    assert M.sign_accuracy(pred[turns], target[turns]) == pytest.approx(2 / 3)
    weight = np.array([1.0, 0.1, 1.0])
    assert M.sign_accuracy(pred[turns], target[turns], weight) == pytest.approx(1.1 / 2.1)


def test_pearson_and_camera_swap_score() -> None:
    pred = np.array([0.1, -0.2, 0.3, 0.0])
    assert M.pearson(pred, 2 * pred + 1) == pytest.approx(1.0)
    assert M.camera_swap_score(pred, -pred) == pytest.approx(1.0)
    assert M.camera_swap_score(pred, pred) == pytest.approx(-1.0)
    assert np.isnan(M.pearson(pred, np.ones(4)))


def test_target_point_baseline_reads_the_bearing_of_the_point() -> None:
    bearings = np.repeat([-0.5, 0.0, 0.5], 10)
    points = 30.0 * np.column_stack([np.cos(bearings), np.sin(bearings)])
    steering = np.repeat([-0.4, 0.0, 0.4], 10)
    lookup = M.fit_target_lookup(steering, points, bins=3)

    query = np.array([[26.3, -14.4], [30.0, 0.0], [26.3, 14.4], [13.2, 7.2]])
    assert M.predict_by_target(lookup, query) == pytest.approx([-0.4, 0.0, 0.4, 0.4])


def test_a_target_point_bin_without_frames_predicts_the_overall_mean() -> None:
    edges, means = M.fit_target_lookup(
        np.array([0.2, 0.4]), np.array([[1.0, -1.0], [1.0, 1.0]]), 60
    )
    assert len(means) == len(edges) + 1
    assert means.min() == pytest.approx(0.2) and means.max() == pytest.approx(0.4)


def test_per_command_baseline() -> None:
    steer = np.array([0.1, 0.3, -0.2, -0.4])
    means = M.command_means(steer, np.array([1, 1, 2, 2]))
    assert means == pytest.approx({1: 0.2, 2: -0.3})
    pred = M.predict_by_command(means, np.array([2, 1, 3]), fallback=-0.05)
    assert pred == pytest.approx([-0.3, 0.2, -0.05])
