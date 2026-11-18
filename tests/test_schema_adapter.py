"""v3 and v6 episodes read into one sample format; the v3 read is unchanged byte for byte."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import torch

from scripts.create_toy_dataset import generate_episode, save_episode_hdf5
from src.data import hdf5_schema as S
from src.data import schema_adapter
from src.data.driving_dataset import IMAGENET_MEAN, IMAGENET_STD, DrivingSequenceDataset
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
            "yaw": f[S.ROTATION][5:21, 1],
        }
        signals_only = schema_adapter.read_window(f, 5, 21, lookahead=20, images=False)
    for name, expected in stored.items():
        got = getattr(window, name)
        assert got.dtype == expected.dtype and np.array_equal(got, expected), name
    assert signals_only.images.size == 0 and np.array_equal(signals_only.yaw, window.yaw)


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


def test_usable_frames_end_three_seconds_before_a_collision(tmp_path: Path) -> None:
    path = make_episode(tmp_path, version=6, n_frames=100)
    with h5py.File(path, "r+") as f:
        f[S.METADATA_GROUP].attrs[S.META_COLLISION_FRAME] = 85  # 20 fps: 60 frames before
    with h5py.File(path, "r") as f:
        assert schema_adapter.usable_frames(f) == 25
        assert len(schema_adapter.nav_commands(f)) == 25


def test_timeout_loses_its_final_stop_unless_an_excerpt(tmp_path: Path) -> None:
    path = make_episode(tmp_path, version=6, n_frames=60)
    with h5py.File(path, "r+") as f:
        f[S.METADATA_GROUP].attrs[S.META_STATUS] = "timeout"
        f[S.VELOCITY][...] = 3.0
        f[S.VELOCITY][45:] = 0.0
    with h5py.File(path, "r") as f:
        assert schema_adapter.usable_frames(f) == 45
    with h5py.File(path, "r+") as f:
        f[S.METADATA_GROUP].attrs[S.META_GOLDEN_SOURCE_FILE] = "episode_source.h5"
    with h5py.File(path, "r") as f:
        assert schema_adapter.usable_frames(f) == 60


def test_an_episode_without_usable_frames_reads_empty(tmp_path: Path) -> None:
    from scripts.build_testbed_cache import build_episode

    src = make_episode(tmp_path, version=3, n_frames=54)
    with h5py.File(src, "r+") as f:
        f[S.METADATA_GROUP].attrs[S.META_COLLISION_FRAME] = 53
    cache = tmp_path / "cache" / src.name
    cache.parent.mkdir()
    build_episode(src, cache, height=8, width=24, quality=90, stride=2)
    for path in (src, cache):
        with h5py.File(path, "r") as f:
            assert [len(a) for a in schema_adapter.read_labels(f, fps=10)] == [0, 0, 0]


def test_invalid_nav_command_is_rejected(tmp_path: Path) -> None:
    path = make_episode(tmp_path, version=3)
    with h5py.File(path, "r+") as f:
        f[S.NAV_COMMAND][7] = 9
    with h5py.File(path, "r") as f, pytest.raises(ValueError, match="nav command 9 at frame 7"):
        schema_adapter.read_labels(f)
    with pytest.raises(ValueError, match=path.name):
        DrivingSequenceDataset(tmp_path, seq_len=16, split="val")


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


def test_cache_reads_as_a_lower_rate_episode(tmp_path: Path) -> None:
    from scripts.build_testbed_cache import build_episode

    src = make_episode(tmp_path, version=3, n_frames=101)
    with h5py.File(src, "r+") as f:
        f[S.METADATA_GROUP].attrs[S.META_COLLISION_FRAME] = 85  # usable source frames 0-24
    cache = tmp_path / "cache" / src.name
    cache.parent.mkdir()
    build_episode(src, cache, height=8, width=24, quality=95, stride=2)
    with h5py.File(src, "r") as f, h5py.File(cache, "r") as c:
        assert schema_adapter.is_cache(c) and len(c[S.IMAGES_JPEG]) == 51
        assert schema_adapter.usable_frames(c) == 13  # source frames 0, 2, ..., 24
        window = schema_adapter.read_window(c, 2, 6, lookahead=3)
        assert np.array_equal(window.steering, f[S.STEERING][4:12:2])
        assert np.array_equal(window.position, f[S.POSITION][4:18:2])
        assert np.array_equal(window.yaw, f[S.ROTATION][4:12:2, 1])
        source = f[S.IMAGES][4:12:2].astype(float)
        small = source.reshape(4, 8, 2, 24, 2, 3).mean(axis=(2, 4))
        assert window.images.shape == (4, 8, 24, 3)
        assert np.abs(window.images - small).mean() < 6
        half = schema_adapter.read_window(c, 2, 6, lookahead=3, size=(4, 12)).images
        smaller = small.reshape(4, 4, 2, 12, 2, 3).mean(axis=(2, 4))
        assert half.shape == (4, 4, 12, 3) and np.abs(half - smaller).mean() < 6


def test_a_cache_sample_equals_the_raw_sample(tmp_path: Path) -> None:
    from scripts.build_testbed_cache import build_episode

    (tmp_path / "cache").mkdir()
    for idx in range(3):
        src = make_episode(tmp_path, version=3, idx=idx, n_frames=160)
        with h5py.File(src, "r+") as f:
            f[S.METADATA_GROUP].attrs[S.META_COLLISION_FRAME] = 150
        build_episode(src, tmp_path / "cache" / src.name, height=8, width=24, quality=90, stride=2)
    options: dict[str, Any] = dict(
        seq_len=8, stride=4, augment=False, resize_height=8, resize_width=24
    )
    with (
        DrivingSequenceDataset(tmp_path, **options) as raw,
        DrivingSequenceDataset(tmp_path / "cache", **options) as cache,
    ):
        assert raw.clips == cache.clips and len(raw) > 10
        for i in range(len(raw)):
            a, b = raw[i], cache[i]
            assert all(torch.equal(a[k], b[k]) for k in a if k != "images"), i
            pixels = [(s["images"] * IMAGENET_STD + IMAGENET_MEAN) * 255 for s in (a, b)]
            mse = float(((pixels[0] - pixels[1]) ** 2).mean())
            assert 10 * np.log10(255**2 / mse) >= 38, i


def test_dataset_loads_a_cache(tmp_path: Path) -> None:
    from scripts.build_testbed_cache import build_episode

    (tmp_path / "cache").mkdir()
    for idx in range(3):
        src = make_episode(tmp_path, version=3, idx=idx, n_frames=60)
        build_episode(src, tmp_path / "cache" / src.name, height=8, width=24, quality=90, stride=2)
    with DrivingSequenceDataset(tmp_path / "cache", seq_len=16, stride=8, split="train") as ds:
        assert ds[0]["images"].shape == (16, 3, 8, 24)
