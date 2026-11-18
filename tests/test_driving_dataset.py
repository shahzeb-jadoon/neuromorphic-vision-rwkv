"""Tests for DrivingSequenceDataset.

These tests use the session-scoped toy dataset fixture from conftest.py,
then verify the DataLoader produces correctly shaped and normalized outputs.

Run with: pytest tests/test_driving_dataset.py -v
"""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from src.data import labels, schema_adapter
from src.data.driving_dataset import (
    TURN_WEIGHT,
    WAYPOINT_OFFSETS,
    DrivingSequenceDataset,
    EpisodeOrderedSampler,
    RandomClipSampler,
    mirror,
)


@pytest.fixture
def train_dataset(toy_data_dir: Path) -> DrivingSequenceDataset:
    """Create a train split dataset."""
    return DrivingSequenceDataset(
        data_dir=toy_data_dir,
        seq_len=16,
        stride=8,
        split="train",
        val_fraction=0.34,  # 2 val episodes out of 6
        seed=42,
        augment=False,
    )


@pytest.fixture
def val_dataset(toy_data_dir: Path) -> DrivingSequenceDataset:
    """Create a val split dataset."""
    return DrivingSequenceDataset(
        data_dir=toy_data_dir,
        seq_len=16,
        stride=8,
        split="val",
        val_fraction=0.34,
        seed=42,
        augment=False,
    )


class TestDatasetShapes:
    """Verify output tensor shapes are correct."""

    def test_images_shape(self, train_dataset: DrivingSequenceDataset) -> None:
        item = train_dataset[0]
        assert item["images"].shape == (16, 3, 320, 1920)

    def test_controls_shape(self, train_dataset: DrivingSequenceDataset) -> None:
        item = train_dataset[0]
        assert item["controls"].shape == (16, 2)

    def test_nav_command_shape(self, train_dataset: DrivingSequenceDataset) -> None:
        item = train_dataset[0]
        assert item["nav_command"].shape == (16,)

    def test_velocity_shape(self, train_dataset: DrivingSequenceDataset) -> None:
        item = train_dataset[0]
        assert item["velocity"].shape == (16, 3)

    def test_images_shape_with_resize(self, toy_data_dir: Path) -> None:
        ds = DrivingSequenceDataset(
            data_dir=toy_data_dir,
            seq_len=16,
            stride=8,
            split="train",
            val_fraction=0.34,
            seed=42,
            augment=False,
            resize_height=160,
            resize_width=960,
        )
        item = ds[0]
        assert item["images"].shape == (16, 3, 160, 960)

    def test_resize_with_default_augment_path(self, toy_data_dir: Path) -> None:
        ds = DrivingSequenceDataset(
            data_dir=toy_data_dir,
            seq_len=16,
            stride=8,
            split="val",
            val_fraction=0.34,
            seed=42,
            augment=None,
            resize_height=160,
            resize_width=960,
        )
        item = ds[0]
        assert item["images"].shape[-2:] == (160, 960)

    def test_dtypes(self, train_dataset: DrivingSequenceDataset) -> None:
        item = train_dataset[0]
        assert item["images"].dtype == torch.float32
        assert item["controls"].dtype == torch.float32
        assert item["nav_command"].dtype == torch.int64
        assert item["velocity"].dtype == torch.float32


class TestNormalization:
    """Verify ImageNet normalization is applied."""

    def test_images_not_in_0_1_range(self, train_dataset: DrivingSequenceDataset) -> None:
        """After ImageNet normalization, values should NOT be in [0, 1]."""
        item = train_dataset[0]
        images = item["images"]
        # ImageNet-normalized images typically have values in [-2.1, 2.6]
        assert images.min() < 0.0, "Normalized images should have negative values"

    def test_steering_range(self, train_dataset: DrivingSequenceDataset) -> None:
        """Steering should be in [-1, 1]."""
        item = train_dataset[0]
        steering = item["controls"][:, 0]
        assert steering.min() >= -1.0
        assert steering.max() <= 1.0

    def test_acceleration_range(self, train_dataset: DrivingSequenceDataset) -> None:
        """Acceleration (throttle - brake) should be in [-1, 1]."""
        item = train_dataset[0]
        acceleration = item["controls"][:, 1]
        assert acceleration.min() >= -1.0
        assert acceleration.max() <= 1.0

    def test_class_targets_are_valid_indices(self, train_dataset: DrivingSequenceDataset) -> None:
        """Every clip's nav and mode targets must index the model's embedding and logits.

        Out-of-range values (e.g. a 0-indexed mode code shifted to -1) crash training
        with a CUDA device-side assert instead of a readable error.
        """
        for idx in range(len(train_dataset)):
            item = train_dataset[idx]
            nav, mode = item["nav_command"], item["driving_mode"]
            assert 0 <= int(nav.min()) and int(nav.max()) < 6, f"clip {idx}: nav {nav.tolist()}"
            assert 0 <= int(mode.min()) and int(mode.max()) < 8, f"clip {idx}: mode {mode.tolist()}"

    def test_waypoints_are_in_the_vehicle_frame(
        self, train_dataset: DrivingSequenceDataset
    ) -> None:
        ep_idx, start = train_dataset.clips[-1]
        with h5py.File(train_dataset.episode_files[ep_idx], "r") as f:
            w = schema_adapter.read_window(
                f,
                start,
                start + train_dataset.seq_len,
                max(WAYPOINT_OFFSETS),
                fps=train_dataset.fps,
                images=False,
            )
        expected = labels.ego_waypoints(w.position, w.yaw, WAYPOINT_OFFSETS)
        got = train_dataset[len(train_dataset) - 1]["future_waypoints"]
        assert got.dtype == torch.float32 and np.array_equal(got.numpy(), expected)


class TestSplit:
    """Verify train/val split properties."""

    def test_no_episode_overlap(
        self, train_dataset: DrivingSequenceDataset, val_dataset: DrivingSequenceDataset
    ) -> None:
        """Train and val should have no shared episodes."""
        train_files = set(p.name for p in train_dataset.episode_files)
        val_files = set(p.name for p in val_dataset.episode_files)
        assert train_files.isdisjoint(val_files), "Train and val share episodes!"

    def test_all_episodes_covered(
        self,
        train_dataset: DrivingSequenceDataset,
        val_dataset: DrivingSequenceDataset,
        toy_data_dir: Path,
    ) -> None:
        """Train + val should cover all episodes."""
        all_files = set(f.name for f in toy_data_dir.glob("episode_*.h5"))
        split_files = set(p.name for p in train_dataset.episode_files) | set(
            p.name for p in val_dataset.episode_files
        )
        assert all_files == split_files, "Not all episodes assigned to a split"

    def test_deterministic_split(self, toy_data_dir: Path) -> None:
        """Same seed should produce same split."""
        ds1 = DrivingSequenceDataset(
            toy_data_dir, seq_len=16, split="train", seed=42, augment=False
        )
        ds2 = DrivingSequenceDataset(
            toy_data_dir, seq_len=16, split="train", seed=42, augment=False
        )
        assert [p.name for p in ds1.episode_files] == [p.name for p in ds2.episode_files]


class TestAugmentation:
    """Augmentation keeps every label consistent with the image."""

    def test_val_no_augmentation(self, toy_data_dir: Path) -> None:
        """Validation split should not augment by default."""
        ds = DrivingSequenceDataset(toy_data_dir, seq_len=16, split="val", seed=42)
        assert not ds.augment

    def test_mirroring_twice_changes_nothing(self, train_dataset: DrivingSequenceDataset) -> None:
        sample = train_dataset[0]
        twice = mirror(mirror(sample))
        assert twice.keys() == sample.keys()
        assert all(torch.equal(twice[k], sample[k]) for k in sample)

    def test_mirror_swaps_sides(self) -> None:
        g = torch.Generator().manual_seed(0)
        sample = {
            "images": torch.rand(6, 3, 2, 6, generator=g),
            "controls": torch.randn(6, 2, generator=g),
            "nav_command": torch.arange(6),
            "future_waypoints": torch.randn(6, 4, 2, generator=g),
            "velocity": torch.randn(6, 3, generator=g),
        }
        m = mirror(sample)
        assert torch.equal(m["images"], sample["images"].flip(-1))
        assert torch.equal(m["controls"][:, 0], -sample["controls"][:, 0])
        assert torch.equal(m["controls"][:, 1], sample["controls"][:, 1])
        assert m["nav_command"].tolist() == [1, 0, 2, 3, 5, 4]  # LEFT<->RIGHT, lane changes
        assert torch.equal(m["future_waypoints"][..., 0], sample["future_waypoints"][..., 0])
        assert torch.equal(m["future_waypoints"][..., 1], -sample["future_waypoints"][..., 1])
        assert torch.equal(m["velocity"], sample["velocity"])

    def test_mirror_moves_each_side_camera_to_the_other_panel(self) -> None:
        panels = torch.rand(3, 3, 4, 8)  # left, centre, right camera
        image = torch.cat(list(panels), dim=-1)[None]
        flipped = mirror(
            {
                "images": image,
                "controls": torch.zeros(1, 2),
                "nav_command": torch.zeros(1, dtype=torch.long),
                "future_waypoints": torch.zeros(1, 4, 2),
            }
        )["images"][0]
        for camera, panel in ((0, 2), (1, 1), (2, 0)):
            assert torch.equal(flipped[..., panel * 8 : (panel + 1) * 8], panels[camera].flip(-1))

    def test_augmented_clip_is_the_clip_or_its_mirror(
        self, toy_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(DrivingSequenceDataset, "_jitter", staticmethod(lambda images: images))
        plain = DrivingSequenceDataset(toy_data_dir, seq_len=16, seed=42, augment=False)
        augmented = DrivingSequenceDataset(toy_data_dir, seq_len=16, seed=42, augment=True)
        torch.manual_seed(0)
        seen = set()
        for i in range(len(plain)):
            original, got = plain[i], augmented[i]
            for name, candidate in (("plain", original), ("mirrored", mirror(original))):
                if all(torch.equal(got[k], candidate[k]) for k in original):
                    seen.add(name)
                    break
            else:
                pytest.fail(f"clip {i} is neither the clip nor its mirror")
        assert seen == {"plain", "mirrored"}


class TestSampler:
    """The training sampler draws clips that may start at any usable frame."""

    def test_random_starts_reach_every_usable_frame(self, toy_data_dir: Path) -> None:
        ds = DrivingSequenceDataset(
            toy_data_dir, seq_len=16, stride=32, val_fraction=0.34, seed=42, augment=False
        )
        fixed = [np.zeros(n, bool) for n in ds.episode_frames]
        for e, s in ds.clips:
            fixed[e][s : s + ds.seq_len] = True
        seen = [np.zeros(n, bool) for n in ds.episode_frames]
        sampler = ds.get_sampler()
        torch.manual_seed(0)
        for _ in range(2500):
            for e, s in sampler:
                seen[e][s : s + ds.seq_len] = True
        assert len(sampler) == len(ds) and not all(f.all() for f in fixed)
        assert all(s.all() for s in seen)

    def test_clips_are_weighted_by_their_turning_frames(
        self, train_dataset: DrivingSequenceDataset
    ) -> None:
        turning, n = train_dataset.turning[0], train_dataset.seq_len
        starts = range(len(turning) - n + 1)
        expected = [1 + TURN_WEIGHT * turning[s : s + n].mean() for s in starts]
        weights = train_dataset.get_sampler().weights
        assert np.allclose(weights[: len(expected)], expected) and len(set(expected)) > 1
        assert set(train_dataset.get_sampler(weighted=False).weights.tolist()) == {1.0}

    def test_episodes_too_short_for_a_clip_are_skipped(self) -> None:
        sampler = RandomClipSampler([np.ones(2), np.ones(0), np.ones(3)], num_samples=200)
        assert set(sampler) == {(0, 0), (0, 1), (2, 0), (2, 1), (2, 2)}

    def test_a_clip_key_reads_that_clip(self, train_dataset: DrivingSequenceDataset) -> None:
        by_index = train_dataset[3]
        by_key = train_dataset[train_dataset.clips[3]]
        assert all(torch.equal(by_key[k], by_index[k]) for k in by_index)
        overrun = train_dataset.episode_frames[0] - train_dataset.seq_len + 1
        with pytest.raises(IndexError, match="overruns"):
            train_dataset[(0, overrun)]


def test_episode_ordered_rows_continue_their_episodes() -> None:
    frames, n, lanes = [100, 37, 64, 20, 81, 10], 16, 4
    sampler = EpisodeOrderedSampler(frames, seq_len=n, batch_size=lanes)
    seen = [np.zeros(f, bool) for f in frames]
    torch.manual_seed(0)
    for _ in range(200):
        keys = list(sampler)
        assert len(keys) == len(sampler) == 16
        for lane in range(lanes):
            row = keys[lane::lanes]
            for (e0, s0), (e1, s1) in zip(row, row[1:]):
                assert (e1, s1) == (e0, s0 + n) or (e1 != e0 and s1 < n)
        for e, s in keys:
            assert s + n <= frames[e]
            seen[e][s : s + n] = True
    assert all(s.all() for s in seen[:5]) and not seen[5].any()


class TestDataLoader:
    """Verify DataLoader integration."""

    def test_batch_loading(self, train_dataset: DrivingSequenceDataset) -> None:
        loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=2, sampler=train_dataset.get_sampler(), num_workers=0
        )
        batch = next(iter(loader))
        assert batch["images"].shape == (2, 16, 3, 320, 1920)
        assert batch["controls"].shape == (2, 16, 2)
        assert batch["nav_command"].shape == (2, 16)
        assert batch["velocity"].shape == (2, 16, 3)
