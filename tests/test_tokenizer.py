"""The visual stem must keep what the old one lost: where things are in the panorama."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from src.models.config import TokenizerConfig
from src.models.tokenizer import BACKBONE_CHANNELS, BACKBONE_STRIDE, VisionTokenizer

H, W, D = 64, 192, 32
MIN_CHANGE = 1e-3  # a stem blind to position changes tokens by 1e-7


@pytest.fixture
def tokenizer() -> VisionTokenizer:
    torch.manual_seed(0)
    return VisionTokenizer(TokenizerConfig(img_height=H, img_width=W, embed_dim=D)).eval()


@pytest.fixture
def cellwise(tokenizer: VisionTokenizer) -> VisionTokenizer:
    """A backbone that sees each cell alone: moving cells only reorders the tokens."""
    tokenizer.backbone = nn.Conv2d(3, BACKBONE_CHANNELS, BACKBONE_STRIDE, BACKBONE_STRIDE)
    return tokenizer


def change(tokenizer: VisionTokenizer, frames: torch.Tensor, other: torch.Tensor) -> float:
    with torch.no_grad():
        a, b = tokenizer(frames[:, None]), tokenizer(other[:, None])
    return float(((a - b).norm(dim=-1) / a.norm(dim=-1)).min())


def test_one_token_per_frame(tokenizer: VisionTokenizer) -> None:
    assert tokenizer(torch.randn(2, 3, 3, H, W)).shape == (2, 3, D)


def test_image_size_must_fit_the_backbone() -> None:
    with pytest.raises(ValueError, match="multiples of 32"):
        VisionTokenizer(TokenizerConfig(img_height=100, img_width=W, embed_dim=D))


def test_swapping_the_side_cameras_changes_the_token(cellwise: VisionTokenizer) -> None:
    frames = torch.randn(2, 3, H, W)
    left, centre, right = frames.split(W // 3, dim=-1)
    assert change(cellwise, frames, torch.cat([right, centre, left], dim=-1)) > MIN_CHANGE


def test_shuffling_image_cells_changes_the_token(cellwise: VisionTokenizer) -> None:
    frames = torch.randn(2, 3, H, W)
    grid = (H // BACKBONE_STRIDE, W // BACKBONE_STRIDE)
    cells = F.pixel_unshuffle(frames, BACKBONE_STRIDE).flatten(2)
    shuffled = cells[:, :, torch.randperm(cells.shape[2])].unflatten(2, grid)
    assert change(cellwise, frames, F.pixel_shuffle(shuffled, BACKBONE_STRIDE)) > MIN_CHANGE


def test_training_does_not_change_batch_norm(tokenizer: VisionTokenizer) -> None:
    buffers = {k: v.clone() for k, v in tokenizer.named_buffers()}
    tokenizer.train()
    tokenizer(torch.randn(2, 2, 3, H, W))
    assert buffers and all(torch.equal(v, buffers[k]) for k, v in tokenizer.named_buffers())


def test_only_the_last_frames_train_the_backbone(tokenizer: VisionTokenizer) -> None:
    images = torch.randn(1, 6, 3, H, W, requires_grad=True)
    tokenizer(images).sum().backward()
    assert images.grad is not None
    reach = images.grad.abs().flatten(2).sum(dim=2)[0]
    last = tokenizer.config.grad_frames
    assert (reach[:-last] == 0).all() and (reach[-last:] > 0).all()


def test_early_frames_still_train_the_pooling(tokenizer: VisionTokenizer) -> None:
    tokenizer(torch.randn(1, 6, 3, H, W))[:, 0].sum().backward()
    assert tokenizer.to_frame.weight.grad is not None
    assert tokenizer.to_frame.weight.grad.abs().sum() > 0
    backbone_grad = next(tokenizer.backbone.parameters()).grad
    assert backbone_grad is None or not backbone_grad.any()


def test_limiting_gradients_keeps_the_tokens(tokenizer: VisionTokenizer) -> None:
    images = torch.randn(2, 6, 3, H, W)
    with torch.no_grad():
        whole = tokenizer(images)
    assert (tokenizer(images) - whole).abs().max() < 1e-5


def test_pretrained_weights_reach_the_backbone(
    tokenizer: VisionTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = resnet18().state_dict()
    monkeypatch.setattr(
        type(ResNet18_Weights.IMAGENET1K_V1), "get_state_dict", lambda *_, **__: source
    )
    tokenizer.load_pretrained()
    loaded = tokenizer.backbone.state_dict()
    assert torch.equal(loaded["conv1.weight"], source["conv1.weight"])
    assert torch.equal(loaded["layer4.1.bn2.running_mean"], source["layer4.1.bn2.running_mean"])
