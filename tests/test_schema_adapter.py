"""v3 and v6 episodes read into one sample format; the v3 read is unchanged byte for byte."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.create_toy_dataset import generate_episode, save_episode_hdf5
from src.data import hdf5_schema as S
from src.data import schema_adapter
from src.data.driving_dataset import DrivingSequenceDataset
from src.models.vision_rwkv_driver import N_DRIVING_MODES


def make_episode(directory: Path, version: int, idx: int = 0, n_frames: int = 40) -> Path:
    ep = generate_episode(idx, n_frames, img_h=16, img_w=48, rng=np.random.default_rng(idx))
    filename, _ = save_episode_hdf5(ep, directory, episode_idx=idx)
    path = directory / filename
    with h5py.File(path, "r+") as f:
        f[S.METADATA_GROUP].attrs[S.META_SCHEMA_VERSION] = version
    return path


@pytest.mark.parametrize("version", [3, 6])
def test_window_equals_the_stored_fields(tmp_path: Path, version: int) -> None:
    path = make_episode(tmp_path, version)
    with h5py.File(path, "r") as f:
        window = schema_adapter.read_window(f, 5, 21, lookahead=20)
        stored = {
            "images": f[S.IMAGES][5:21],
            "steering": f[S.STEERING][5:21],
            "throttle": f[S.THROTTLE][5:21],
            "brake": f[S.BRAKE][5:21],
            "nav_command": f[S.NAV_COMMAND][5:21],
            "velocity": f[S.VELOCITY][5:21],
            "target_speed": f[S.TARGET_SPEED][5:21],
            "driving_mode": f[S.DRIVING_MODE][5:21],
            "position": f[S.POSITION][5:40],
        }
    for name, expected in stored.items():
        got = getattr(window, name)
        assert got.dtype == expected.dtype and np.array_equal(got, expected), name


def test_yield_stop_folds_into_stop_sign_stop(tmp_path: Path) -> None:
    path = make_episode(tmp_path, version=6)
    with h5py.File(path, "r+") as f:
        f[S.DRIVING_MODE][10:15] = 9
    with h5py.File(path, "r") as f:
        modes = schema_adapter.read_window(f, 0, 20, lookahead=0).driving_mode
    assert set(modes[10:15]) == {8} and 9 not in modes


def test_every_written_mode_is_a_model_class() -> None:
    for mode in S.VALID_DRIVING_MODES:
        assert 0 <= schema_adapter.MODE_FOLD.get(mode, mode) - 1 < N_DRIVING_MODES


@pytest.mark.parametrize("version", [2, 7])
def test_unsupported_schema_names_the_file(tmp_path: Path, version: int) -> None:
    path = make_episode(tmp_path, version)
    with (
        h5py.File(path, "r") as f,
        pytest.raises(ValueError, match=rf"{path.name}: schema v{version}"),
    ):
        schema_adapter.check_supported(f, path.name)


def test_collision_frame_limits_usable_frames(tmp_path: Path) -> None:
    path = make_episode(tmp_path, version=6, n_frames=40)
    with h5py.File(path, "r+") as f:
        f[S.METADATA_GROUP].attrs[S.META_COLLISION_FRAME] = 25
    with h5py.File(path, "r") as f:
        assert schema_adapter.usable_frames(f) == 25
        assert len(schema_adapter.nav_commands(f)) == 25


def test_dataset_mixes_v3_and_v6_episodes(tmp_path: Path) -> None:
    make_episode(tmp_path, version=3, idx=0)
    v6 = make_episode(tmp_path, version=6, idx=1)
    with h5py.File(v6, "r+") as f:
        f[S.DRIVING_MODE][:] = 9
    files, modes = set(), set()
    for split in ("train", "val"):
        with DrivingSequenceDataset(tmp_path, seq_len=16, stride=8, split=split) as ds:
            files |= set(ds.episode_files)
            modes |= {int(m) for i in range(len(ds)) for m in ds[i]["driving_mode"]}
    assert len(files) == 2 and max(modes) < N_DRIVING_MODES
