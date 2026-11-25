"""The learned agent feeds the model what it trained on, from the frame it is driving."""

from __future__ import annotations

import queue
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest
import torch

from agents.learned import learned_agent
from agents.learned.learned_agent import frame_image, model_image, ticks_per_frame
from src.data import hdf5_schema as S
from src.data import schema_adapter
from src.data.driving_dataset import DrivingSequenceDataset
from tests.test_schema_adapter import make_episode


def test_agent_images_equal_the_training_samples(tmp_path: Path) -> None:
    from scripts.build_testbed_cache import build_episode

    (tmp_path / "cache").mkdir()
    for idx in range(3):
        src = make_episode(tmp_path, version=3, idx=idx, n_frames=60)
        build_episode(
            src,
            tmp_path / "cache" / src.name,
            height=8,
            width=24,
            quality=schema_adapter.CACHE_QUALITY,
            stride=2,
        )
    options = dict(seq_len=8, stride=8, augment=False, resize_height=8, resize_width=24)
    with DrivingSequenceDataset(tmp_path / "cache", **options) as ds:  # type: ignore[arg-type]
        for i, (ep_idx, start) in enumerate(ds.clips):
            with h5py.File(tmp_path / ds.episode_files[ep_idx].name, "r") as f:
                recorded = f[S.IMAGES][2 * start : 2 * (start + 8) : 2]
            agent = torch.stack([model_image(frame, (8, 24)) for frame in recorded])
            assert torch.equal(agent, ds[i]["images"]), i


def test_frame_image_takes_the_frame_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(learned_agent, "CAMERA_TIMEOUT_S", 0.01)
    images: queue.Queue[SimpleNamespace] = queue.Queue()
    for frame in (5, 6, 8):
        images.put(SimpleNamespace(frame=frame))
    assert frame_image(images, 6).frame == 6
    with pytest.raises(RuntimeError, match="skipped frame 7"):
        frame_image(images, 7)
    with pytest.raises(RuntimeError, match="no camera image of frame 9"):
        frame_image(images, 9)


def test_the_model_steps_at_the_rate_it_trained_at() -> None:
    def world(delta: float | None, synchronous: bool = True) -> SimpleNamespace:
        return SimpleNamespace(fixed_delta_seconds=delta, synchronous_mode=synchronous)

    assert ticks_per_frame(world(0.05), fps=10) == 2
    assert ticks_per_frame(world(0.1), fps=10) == 1
    for settings in (world(0.05, synchronous=False), world(None), world(0.03)):
        with pytest.raises(ValueError):
            ticks_per_frame(settings, fps=10)
