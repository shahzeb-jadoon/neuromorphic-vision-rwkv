"""Training targets computed from an episode's raw signals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def ego_waypoints(position: np.ndarray, yaw: np.ndarray, offsets: Sequence[int]) -> np.ndarray:
    """Positions at frames t + offsets, in the vehicle's frame at t.

    position: (T + lookahead, 3) world positions in metres; frames past its end are
    clamped to the last one. yaw: (T,) CARLA yaw in degrees, clockwise seen from above.
    Returns (T, K, 2) float32 offsets in metres: x forward, y to the right.
    """
    t = len(yaw)
    future = np.minimum(np.arange(t)[:, None] + np.asarray(offsets), len(position) - 1)
    d = position[future, :2].astype(np.float64) - position[:t, None, :2]
    psi = np.radians(yaw.astype(np.float64))[:, None]
    cos, sin = np.cos(psi), np.sin(psi)
    forward = cos * d[..., 0] + sin * d[..., 1]
    right = -sin * d[..., 0] + cos * d[..., 1]
    return np.stack([forward, right], axis=-1).astype(np.float32)
