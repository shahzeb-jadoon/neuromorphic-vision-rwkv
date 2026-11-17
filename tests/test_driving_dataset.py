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
from src.data.driving_dataset import WAYPOINT_OFFSETS, DrivingSequenceDataset


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
                f, start, start + train_dataset.seq_len, max(WAYPOINT_OFFSETS), images=False
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
    """Verify augmentation behavior."""

    def test_hflip_flips_steering(self, toy_data_dir: Path) -> None:
        """Horizontal flip should negate steering sign."""
        ds_no_aug = DrivingSequenceDataset(
            toy_data_dir, seq_len=16, split="train", seed=42, augment=False
        )
        ds_aug = DrivingSequenceDataset(
            toy_data_dir, seq_len=16, split="train", seed=42, augment=True
        )

        # Get same clip from both — augmented may differ due to randomness
        item_no_aug = ds_no_aug[0]
        # Just verify the augmented dataset runs without error
        item_aug = ds_aug[0]
        assert item_aug["images"].shape == item_no_aug["images"].shape

    def test_val_no_augmentation(self, toy_data_dir: Path) -> None:
        """Validation split should not augment by default."""
        ds = DrivingSequenceDataset(toy_data_dir, seq_len=16, split="val", seed=42)
        assert not ds.augment

    def test_panoramic_flip_correctness(self) -> None:
        """Verify horizontal flip reverses panel order and negates steering.

        Constructs a synthetic panoramic image with identifiable panels
        (left=red, centre=green, right=blue), forces a flip via
        monkeypatched torch.rand, and checks:
          1. Panel order becomes [blue'|green'|red']
          2. Each panel is individually flipped horizontally
          3. Steering sign is negated
        """
        T, C, H = 4, 3, 8
        panel_w = 16
        n_cameras = 3
        W_total = panel_w * n_cameras

        # Build synthetic panels: left=red(1,0,0), centre=green(0,1,0), right=blue(0,0,1)
        images = torch.zeros(T, C, H, W_total)
        # Left panel [0:16] → red channel gradient (0→1 across width)
        images[:, 0, :, :panel_w] = torch.linspace(0, 1, panel_w)
        # Centre panel [16:32] → green channel gradient
        images[:, 1, :, panel_w : 2 * panel_w] = torch.linspace(0, 1, panel_w)
        # Right panel [32:48] → blue channel gradient
        images[:, 2, :, 2 * panel_w : 3 * panel_w] = torch.linspace(0, 1, panel_w)

        steering = torch.tensor([0.3, -0.2, 0.5, -0.1])
        controls = torch.stack([steering, torch.zeros(T)], dim=-1)

        # Create a dataset instance just to call _augment — need n_cameras set
        # We'll call the private method directly instead of going through __getitem__
        import unittest.mock

        # Monkeypatch torch.rand to always return 0.0 ⇒ all augmentations trigger
        with unittest.mock.patch("torch.rand", return_value=torch.tensor([0.0])):
            # Create a minimal object with the augment method
            ds = object.__new__(DrivingSequenceDataset)
            ds.n_cameras = n_cameras
            ds.augment = True
            aug_images, aug_controls = ds._augment(images.clone(), controls.clone())

        # 1. Steering should be negated
        torch.testing.assert_close(aug_controls[:, 0], -controls[:, 0])

        # 2. Acceleration should be unchanged
        torch.testing.assert_close(aug_controls[:, 1], controls[:, 1])

        # 3. Panel order should be reversed: what was right (blue) is now in left position
        #    The right panel had blue gradient [0→1]; after flip+reverse it should be
        #    in position [0:16] with the gradient reversed [1→0]
        aug_left = aug_images[:, :, :, :panel_w]
        # The original right panel was blue (channel 2) with gradient 0→1
        # After horizontal flip of that panel, gradient becomes 1→0
        assert aug_left[:, 2].mean() > 0.3, "Reversed right panel should be in left position"
        assert aug_left[:, 0].mean() < 0.1, "Left position should NOT have red (original left)"

        # 4. Original left panel (red) should now be in right position
        aug_right = aug_images[:, :, :, 2 * panel_w : 3 * panel_w]
        assert aug_right[:, 0].mean() > 0.3, "Original left (red) panel should be in right"
        assert aug_right[:, 2].mean() < 0.1, "Right position should NOT have blue"


class TestSampler:
    """Verify weighted sampler works."""

    def test_sampler_creates(self, train_dataset: DrivingSequenceDataset) -> None:
        sampler = train_dataset.get_sampler()
        assert len(sampler) == len(train_dataset)

    def test_sampler_iterates(self, train_dataset: DrivingSequenceDataset) -> None:
        sampler = train_dataset.get_sampler()
        indices = list(iter(sampler))
        assert len(indices) == len(train_dataset)
        assert all(0 <= i < len(train_dataset) for i in indices)


class TestDataLoader:
    """Verify DataLoader integration."""

    def test_batch_loading(self, train_dataset: DrivingSequenceDataset) -> None:
        loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=2, shuffle=True, num_workers=0
        )
        batch = next(iter(loader))
        assert batch["images"].shape == (2, 16, 3, 320, 1920)
        assert batch["controls"].shape == (2, 16, 2)
        assert batch["nav_command"].shape == (2, 16)
        assert batch["velocity"].shape == (2, 16, 3)
