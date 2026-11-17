"""Toy data must follow the same conventions as data written by the expert.

The toy generator feeds the test fixtures and the environment smoke run. When its
label codes drifted from the schema (0-indexed driving modes), training crashed
with a CUDA assert while every existing test still passed.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.create_toy_dataset import generate_episode
from src.data import hdf5_schema as S


@pytest.fixture(scope="module")
def episode() -> dict:
    return generate_episode(episode_idx=0, n_frames=200, img_h=32, img_w=96)


def test_label_codes_follow_schema(episode: dict) -> None:
    assert set(np.unique(episode["nav_command"])) <= S.VALID_NAV_COMMANDS
    assert set(np.unique(episode["driving_mode"])) <= S.VALID_DRIVING_MODES
    assert set(np.unique(episode["traffic_light_state"])) <= set(S.TRAFFIC_LIGHT_STATE_NAMES)


def test_target_speed_is_km_per_hour(episode: dict) -> None:
    speeds = episode["target_speed"]
    assert speeds.min() >= 0.0
    assert speeds.max() == pytest.approx(30.0)


def test_yaw_matches_direction_of_travel(episode: dict) -> None:
    """Rotation is (pitch, yaw, roll); yaw must point along the driven path."""
    step = np.diff(episode["position"][:, :2], axis=0)
    moving = np.linalg.norm(step, axis=1) > 1e-3
    travel = np.arctan2(step[moving, 1], step[moving, 0])
    yaw = np.radians(episode["rotation"][1:, 1][moving])
    wrapped = np.angle(np.exp(1j * (travel - yaw)))
    assert np.abs(wrapped).max() < 1e-3


def test_schema_mode_codes_match_expert_enum() -> None:
    expert = pytest.importorskip("agents.expert.privileged_expert")
    enum_codes = {member.value: member.name for member in expert.DrivingMode}
    assert set(enum_codes) == S.VALID_DRIVING_MODES
    for code, name in S.DRIVING_MODE_NAMES.items():
        assert enum_codes[code].replace("_", "").lower() == name.lower()
