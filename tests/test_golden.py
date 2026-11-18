"""GOLD: label behaviour on real recorded windows (data/golden). Run with `make gold`."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from src.data import labels, schema_adapter
from src.data.driving_dataset import FPS, WAYPOINT_OFFSETS

pytestmark = pytest.mark.gold

LEFT, RIGHT = 1, 2  # RoadOption


def test_turn_waypoints_bend_the_way_of_the_command(golden_dir: Path) -> None:
    lateral, nav = [], []
    for path in sorted(golden_dir.glob("episode_*.h5")):
        with h5py.File(path, "r") as f:
            n = schema_adapter.usable_frames(f, FPS)
            w = schema_adapter.read_window(f, 0, n, max(WAYPOINT_OFFSETS), fps=FPS, images=False)
        far = labels.ego_waypoints(w.position, w.yaw, WAYPOINT_OFFSETS)[:, -1, 1]
        keep = (np.abs(w.steering) > 0.1) & (np.arange(n) + max(WAYPOINT_OFFSETS) < len(w.position))
        lateral.append(far[keep])
        nav.append(w.nav_command[keep])
    y, command = np.concatenate(lateral), np.concatenate(nav)
    left, right = (y[command == LEFT] < 0).mean(), (y[command == RIGHT] > 0).mean()
    assert left >= 0.95 and right >= 0.95, f"LEFT y<0 {left:.1%}, RIGHT y>0 {right:.1%}"


def test_path_commands_agree_with_recorded_turns(golden_dir: Path) -> None:
    derived, recorded = [], []
    for path in sorted(golden_dir.glob("episode_*.h5")):
        with h5py.File(path, "r") as f:
            n = schema_adapter.usable_frames(f, FPS)
            w = schema_adapter.read_window(f, 0, n, 0, fps=FPS, images=False)
            hard = (np.abs(w.steering) >= 0.5) & (np.linalg.norm(w.velocity[:, :2], axis=1) >= 1)
            keep = hard & np.isin(w.nav_command, (LEFT, RIGHT))
            derived.append(schema_adapter.commands(f, FPS)[keep])
            recorded.append(w.nav_command[keep])
    agree = np.mean(np.concatenate(derived) == np.concatenate(recorded))
    assert agree >= 0.95, f"derived equals recorded on {agree:.1%} of hard turning frames"
