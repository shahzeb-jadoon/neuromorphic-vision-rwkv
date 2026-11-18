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


# RoadOption values of the derived commands.
LEFT, RIGHT, LANEFOLLOW = 1, 2, 4
# A turn is a stretch of the path curving the same way by at least CURVE_START that peaks
# above TURN_PEAK (a radius under 19 m) and turns the heading by TURN_ANGLE or more. Road
# curves stay below TURN_PEAK; junction turns in v3 peak near 6 deg/m.
CURVE_START = 1.0  # deg/m
TURN_PEAK = 3.0  # deg/m
TURN_ANGLE = 45.0  # deg
TURN_APPROACH = 30.0  # m: how far ahead of a turn its command starts
TURN_GAP = 10.0  # m: turns the same way whose commands come closer than this read as one
PATH_STEP = 0.5  # m: resolution of the path's heading


def path_commands(position: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """RoadOption of every row from the path itself: LEFT or RIGHT from TURN_APPROACH
    metres before a turn until it ends, LANEFOLLOW elsewhere. Turns the same way less than
    TURN_GAP apart share one command.

    position: (N, 2+) world positions in metres; yaw: (N,) CARLA yaw in degrees, clockwise
    seen from above, so a right turn increases it. Measured along the distance travelled,
    so a vehicle waiting before a turn keeps its command; works on a driven trajectory and
    on a planned route alike.
    """
    xy = position[:, :2].astype(np.float64)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    command = np.full(len(s), LANEFOLLOW)
    if s[-1] < 4 * PATH_STEP:
        return command
    heading = np.degrees(np.unwrap(np.radians(yaw.astype(np.float64))))
    moving = np.r_[True, np.diff(s) > 1e-3]
    grid = np.arange(0.0, s[-1], PATH_STEP)
    h = np.interp(grid, s[moving], heading[moving])
    curvature = (np.interp(grid + 1, grid, h) - np.interp(grid - 1, grid, h)) / 2
    side = np.sign(curvature) * (np.abs(curvature) >= CURVE_START)
    cut = np.flatnonzero(np.diff(side)) + 1
    turns: list[tuple[float, float, int]] = []  # (from, to, command) in path order
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(side)]):
        turn = h[b - 1] - h[a]
        if side[a] == 0 or abs(turn) < TURN_ANGLE or np.abs(curvature[a:b]).max() < TURN_PEAK:
            continue
        start, end, way = grid[a] - TURN_APPROACH, grid[b - 1], RIGHT if turn > 0 else LEFT
        if turns and turns[-1][2] == way and start - turns[-1][1] < TURN_GAP:
            turns[-1] = (turns[-1][0], end, way)
        else:
            turns.append((start, end, way))
    # Latest turn first, so an earlier turn keeps its rows from the next one's approach.
    for start, end, way in reversed(turns):
        command[(s >= start) & (s <= end)] = way
    return command
