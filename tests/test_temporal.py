"""Temporal cores: how a sequence is cut into calls must not change the output."""

from __future__ import annotations

from typing import cast

import pytest
import torch

from src.models.config import (
    FusionConfig,
    ModelConfig,
    PolicyHeadConfig,
    RWKVEncoderConfig,
    TokenizerConfig,
)
from src.models.rwkv_encoder import NO_HISTORY, RWKVBlock, WKVMemory, wkv_step
from src.models.temporal import TemporalCore
from src.models.vision_rwkv_driver import VisionRWKVDriver, build_temporal_core

D = 32
FRAMES = 32
TOLERANCE = 1e-5
BF16_TOLERANCE = 0.05  # a few bf16 rounding steps on features of size 3


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


@pytest.mark.parametrize("bf16", [False, True])
def test_chunks_of_any_length_equal_one_call(core: TemporalCore, bf16: bool) -> None:
    x = torch.randn(3, FRAMES, D)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        whole, _ = core(x)

        state = None
        chunks = []
        for chunk in x.split([5, 1, 11, 15], dim=1):
            features, state = core(chunk, state)
            chunks.append(features)

    limit = BF16_TOLERANCE if bf16 else TOLERANCE
    assert (whole - torch.cat(chunks, dim=1)).float().abs().max() < limit


def test_bf16_agrees_with_fp32_and_keeps_memory_in_fp32(core: TemporalCore) -> None:
    x = torch.randn(3, FRAMES, D).to(torch.bfloat16)
    exact, _ = core(x.float())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        rounded, state = core(x)

    assert (exact - rounded.float()).abs().max() < BF16_TOLERANCE
    assert all(part.dtype == torch.float32 for part in state)


def test_memory_changes_the_output(core: TemporalCore) -> None:
    x = torch.randn(3, FRAMES, D)
    _, state = core(x)
    with_memory, _ = core.step(x[:, 0], state)
    without_memory, _ = core.step(x[:, 0])
    assert (with_memory - without_memory).abs().max() > 1e-3


def test_driver_one_call_equals_stepping_every_frame(driver: VisionRWKVDriver) -> None:
    images = torch.rand(2, FRAMES, 3, 32, 64)
    commands = torch.randint(0, 6, (2, FRAMES))
    speed = torch.rand(2, FRAMES)
    target = torch.randn(2, FRAMES, 2) * 30
    controls, _, aux = driver(images, commands, speed, target)

    state = None
    for t in range(FRAMES):
        step_controls, state, step_aux = driver.step(
            images[:, t], commands[:, t], speed[:, t], target[:, t], state
        )
        assert (controls[:, t] - step_controls).abs().max() < TOLERANCE
        assert all((aux[k][:, t] - v).abs().max() < TOLERANCE for k, v in step_aux.items())


def test_unknown_core_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown temporal core 'lstm'"):
        build_temporal_core(RWKVEncoderConfig(core="lstm"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_core_trains_under_bf16_autocast_on_the_gpu(core_name: str) -> None:
    core = build_temporal_core(core_config(core_name)).cuda()
    x = torch.randn(2, FRAMES, D, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        features, state = core(x)
    features.float().sum().backward()
    assert features.isfinite().all()
    assert all(part.dtype == torch.float32 for part in state)
    assert all(p.grad is not None and p.grad.isfinite().all() for p in core.parameters())


def empty_memory(channels: int, dtype: torch.dtype = torch.float32) -> WKVMemory:
    zeros = torch.zeros(channels, dtype=dtype)
    return zeros, zeros, torch.full((channels,), NO_HISTORY, dtype=dtype)


def test_wkv_equals_the_sums_written_out() -> None:
    frames, channels = 40, 6
    k, v = torch.randn(frames, channels).double(), torch.randn(frames, channels).double()
    log_decay = -torch.rand(frames, channels).double()
    bonus = torch.randn(channels).double()

    memory = empty_memory(channels, torch.float64)
    a, b = torch.zeros(channels).double(), torch.zeros(channels).double()
    for t in range(frames):
        average, memory = wkv_step(k[t], v[t], log_decay[t], bonus, memory)
        weight = k[t].exp()
        now = (bonus + k[t]).exp()
        assert (average - (a + now * v[t]) / (b + now)).abs().max() < 1e-9
        a = log_decay[t].exp() * a + weight * v[t]
        b = log_decay[t].exp() * b + weight


@pytest.mark.parametrize("pattern", ["random", "alternating"])
def test_wkv_stays_finite_and_differentiable_for_keys_up_to_50(pattern: str) -> None:
    frames, channels = 2000, 8
    if pattern == "random":
        k = (torch.rand(frames, channels) * 100 - 50).requires_grad_()
    else:
        k = (50.0 * (-1.0) ** torch.arange(frames)[:, None].repeat(1, channels)).requires_grad_()
    v = torch.randn(frames, channels).requires_grad_()
    log_decay = -torch.rand(frames, channels)

    memory = empty_memory(channels)
    averages = []
    for t in range(frames):
        average, memory = wkv_step(k[t], v[t], log_decay[t], torch.zeros(channels), memory)
        averages.append(average)
    result = torch.stack(averages)
    result.sum().backward()

    assert result.isfinite().all()
    assert k.grad is not None and k.grad.isfinite().all()
    assert v.grad is not None and v.grad.isfinite().all()
    assert result.abs().max() <= v.abs().max() + 1e-3  # an average stays within its values


def test_block_gradients_stay_finite_when_keys_and_decay_rates_explode() -> None:
    block = RWKVBlock(RWKVEncoderConfig(embed_dim=D, ffn_dim=64, dropout=0.0), 0)
    with torch.no_grad():
        block.W_k.weight.mul_(1e3)
        block.W_w.weight.mul_(1e3)

    output, _ = block(torch.randn(2, FRAMES, D))
    output.sum().backward()

    assert output.isfinite().all()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in block.parameters())


def test_gradients_through_the_carried_state_match_numerical_ones() -> None:
    torch.manual_seed(0)
    config = RWKVEncoderConfig(embed_dim=4, ffn_dim=8, dropout=0.0)
    block = RWKVBlock(config, 0).double().eval()
    x = torch.randn(1, 6, 4, dtype=torch.float64, requires_grad=True)

    def two_calls(frames: torch.Tensor) -> torch.Tensor:
        _, state = block(frames[:, :3])
        return cast(torch.Tensor, block(frames[:, 3:], state)[0])

    assert torch.autograd.gradcheck(two_calls, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)
