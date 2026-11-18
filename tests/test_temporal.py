"""Temporal cores: how a sequence is cut into calls must not change the output."""

from __future__ import annotations

import pytest
import torch

from src.models.config import (
    FusionConfig,
    ModelConfig,
    PolicyHeadConfig,
    RWKVEncoderConfig,
    TokenizerConfig,
)
from src.models.temporal import GRUCore, TemporalCore
from src.models.vision_rwkv_driver import VisionRWKVDriver, build_temporal_core

D = 32
FRAMES = 32
TOLERANCE = 1e-5


def core_config(name: str) -> RWKVEncoderConfig:
    return RWKVEncoderConfig(core=name, embed_dim=D, n_layers=2, ffn_dim=64, dropout=0.1)


@pytest.fixture(params=["gru", "rwkv"])
def core_name(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def core(core_name: str) -> TemporalCore:
    torch.manual_seed(0)
    return build_temporal_core(core_config(core_name)).eval()


@pytest.fixture
def driver(core_name: str) -> VisionRWKVDriver:
    torch.manual_seed(0)
    config = ModelConfig(
        tokenizer=TokenizerConfig(img_height=32, img_width=64, embed_dim=D),
        encoder=core_config(core_name),
        fusion=FusionConfig(visual_dim=D, nav_embed_dim=8, fused_dim=D),
        policy_head=PolicyHeadConfig(input_dim=D, hidden_dims=(16,)),
    )
    return VisionRWKVDriver(config).eval()


def test_one_call_equals_stepping_every_frame(core: TemporalCore) -> None:
    x = torch.randn(3, FRAMES, D)
    whole, whole_state = core(x)

    state = None
    stepped = []
    for t in range(FRAMES):
        features, state = core.step(x[:, t], state)
        stepped.append(features)

    assert state is not None
    assert (whole - torch.stack(stepped, dim=1)).abs().max() < TOLERANCE
    assert all((a - b).abs().max() < TOLERANCE for a, b in zip(whole_state, state))


def test_chunks_of_any_length_equal_one_call(core: TemporalCore) -> None:
    x = torch.randn(3, FRAMES, D)
    whole, _ = core(x)

    state = None
    chunks = []
    for chunk in x.split([5, 1, 11, 15], dim=1):
        features, state = core(chunk, state)
        chunks.append(features)

    assert (whole - torch.cat(chunks, dim=1)).abs().max() < TOLERANCE


def test_memory_changes_the_output(core: TemporalCore) -> None:
    x = torch.randn(3, FRAMES, D)
    _, state = core(x)
    with_memory, _ = core.step(x[:, 0], state)
    without_memory, _ = core.step(x[:, 0])
    assert (with_memory - without_memory).abs().max() > 1e-3


def test_driver_one_call_equals_stepping_every_frame(driver: VisionRWKVDriver) -> None:
    images = torch.rand(2, FRAMES, 3, 32, 64)
    commands = torch.randint(0, 6, (2, FRAMES))
    controls, _, aux = driver(images, commands)

    state = None
    for t in range(FRAMES):
        step_controls, state, step_aux = driver.step(images[:, t], commands[:, t], state)
        assert (controls[:, t] - step_controls).abs().max() < TOLERANCE
        assert all((aux[k][:, t] - v).abs().max() < TOLERANCE for k, v in step_aux.items())


def test_unknown_core_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown temporal core 'lstm'"):
        build_temporal_core(RWKVEncoderConfig(core="lstm"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gru_trains_under_bf16_autocast() -> None:
    core = GRUCore(core_config("gru")).cuda()
    x = torch.randn(2, FRAMES, D, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        features, state = core(x)
    features.sum().backward()
    assert features.isfinite().all()
    assert all(s.dtype == torch.float32 for s in state)
    assert all(p.grad is not None and p.grad.isfinite().all() for p in core.parameters())
