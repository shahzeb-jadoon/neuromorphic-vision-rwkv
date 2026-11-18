"""Vision tokenizer: a pretrained CNN, then attention pooling into one token per frame.

Input:  (B, T, 3, H, W) ImageNet-normalised frames, H and W multiples of 32
Output: (B, T, embed_dim) frame tokens
"""

from __future__ import annotations

from collections import OrderedDict
from typing import cast

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.ops.misc import FrozenBatchNorm2d

from src.models.config import TokenizerConfig

BACKBONE_STRIDE = 32
BACKBONE_CHANNELS = 512


def grid_positions(rows: int, cols: int, dim: int) -> torch.Tensor:
    """Sine-cosine codes of a grid's cells, row by row: (rows * cols, dim), half per axis."""

    def axis(cells: int) -> torch.Tensor:
        frequencies = 10000.0 ** -torch.linspace(0.0, 1.0, dim // 4)
        angles = torch.arange(cells)[:, None] * frequencies
        return torch.cat([angles.sin(), angles.cos()], dim=1)

    row_codes = axis(rows)[:, None].expand(-1, cols, -1)
    col_codes = axis(cols)[None].expand(rows, -1, -1)
    return torch.cat([row_codes, col_codes], dim=-1).flatten(0, 1)


class AttentionPool(nn.Module):
    """Learned queries, each gathering what it needs from a set of tokens."""

    def __init__(self, dim: int, n_queries: int, n_heads: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, dim))
        self.attention = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """(N, M, D) tokens to (N, n_queries, D) pooled tokens."""
        queries = self.queries.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attention(queries, tokens, tokens, need_weights=False)
        return cast(torch.Tensor, self.norm(queries + pooled))


class VisionTokenizer(nn.Module):
    """ResNet-18 feature map, projected, positioned and pooled into a frame token.

    Batch norm is frozen: clips are too correlated to estimate it, and a frame must give
    the same token however it is batched.

    Args:
        config: Tokenizer configuration.
    """

    def __init__(self, config: TokenizerConfig) -> None:
        super().__init__()
        if config.img_height % BACKBONE_STRIDE or config.img_width % BACKBONE_STRIDE:
            raise ValueError(
                f"Image size {config.img_height}x{config.img_width} must be multiples of "
                f"{BACKBONE_STRIDE}"
            )
        self.config = config
        layers = resnet18(norm_layer=FrozenBatchNorm2d).named_children()
        self.backbone: nn.Module = nn.Sequential(
            OrderedDict((name, m) for name, m in layers if name not in ("avgpool", "fc"))
        )
        dim = config.embed_dim
        self.project = nn.Conv2d(BACKBONE_CHANNELS, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        rows, cols = config.img_height // BACKBONE_STRIDE, config.img_width // BACKBONE_STRIDE
        self.position = nn.Parameter(grid_positions(rows, cols, dim)[None])
        self.pool = AttentionPool(dim, config.n_queries, config.n_heads)
        self.to_frame = nn.Linear(config.n_queries * dim, dim)

    def load_pretrained(self) -> None:
        """Load the ImageNet weights into the backbone, downloading them on first use."""
        weights = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=False)
        self.backbone.load_state_dict({k: v for k, v in weights.items() if not k.startswith("fc.")})

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Tokenize a sequence of images.

        Args:
            images: (B, T, C, H, W) image sequence tensor.

        Returns:
            (B, T, embed_dim) frame tokens.
        """
        if images.ndim != 5:
            raise ValueError(f"VisionTokenizer expected 5D input, got shape {tuple(images.shape)}")

        batch, steps = images.shape[:2]
        return self.pool_features(self.backbone(images.flatten(0, 1))).view(batch, steps, -1)

    def pool_features(self, features: torch.Tensor) -> torch.Tensor:
        """(N, 512, rows, cols) backbone features to (N, embed_dim) frame tokens."""
        tokens = self.norm(self.project(features).flatten(2).transpose(1, 2)) + self.position
        return cast(torch.Tensor, self.to_frame(self.pool(tokens).flatten(1)))
