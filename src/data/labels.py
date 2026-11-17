"""Training targets computed from an episode's raw signals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

PRE_COLLISION_S = 3.0  # the expert was already failing this close to a collision
STUCK_SPEED = 0.1  # m/s; the collector times an episode out after standing below it


def usable_end(speed: np.ndarray, fps: int, status: str, collision_frame: int) -> int:
    """End (exclusive) of the recorded frames that make valid targets.

    speed: (N,) speed of every recorded frame, m/s. Frames from PRE_COLLISION_S before a
    collision on are dropped. An episode that timed out loses its final stop: the
    vehicle stood still until the collector gave up, so those frames show it stuck.
    """
    end = len(speed)
    if collision_frame > 0:
        end = min(end, collision_frame - round(PRE_COLLISION_S * fps))
    if status == "timeout":
        moving = np.flatnonzero(speed[: max(end, 0)] >= STUCK_SPEED)
        end = int(moving[-1]) + 1 if moving.size else 0
    return max(end, 0)


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
