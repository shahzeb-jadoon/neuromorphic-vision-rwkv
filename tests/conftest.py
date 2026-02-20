"""Shared pytest fixtures for the test suite.

Provides a session-scoped toy dataset that is generated once and reused
across all test modules, avoiding redundant subprocess calls and
ensuring consistent test data.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

# Re-use the toy dataset generator directly (no subprocess shell-out)
from scripts.create_toy_dataset import generate_episode, save_episode_hdf5

_TOY_DIR = Path("data/test_fixtures")


@pytest.fixture(scope="session")
def toy_data_dir() -> Iterator[Path]:
    """Generate a toy dataset once per test session.

    Creates 6 episodes × 50 frames in ``data/test_fixtures/``.
    Cleaned up after the session ends.
    """
    _TOY_DIR.mkdir(parents=True, exist_ok=True)

    rng_base_seed = 42
    n_episodes = 6
    n_frames = 50

    for i in range(n_episodes):
        ep = generate_episode(i, n_frames, rng=np.random.default_rng(rng_base_seed + i))
        save_episode_hdf5(ep, _TOY_DIR, episode_idx=i)

    yield _TOY_DIR

    # Cleanup
    if _TOY_DIR.exists():
        shutil.rmtree(_TOY_DIR)
