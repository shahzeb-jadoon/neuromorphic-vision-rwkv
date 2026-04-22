"""Vision tokenizer: Patch embedding + Q-Shift spatial mixing.

Converts raw image frames into a sequence of visual tokens for the
RWKV encoder. Each image is divided into non-overlapping patches,
linearly projected, and optionally spatially mixed via Q-Shift.

Input:  (B, T, 3, H, W) — batch of image sequences
        Default panoramic: H=320, W=1920 (3 cameras × 640).
Output: (B, T, N_patches, embed_dim) — patch token sequences

where N_patches = (H / patch_size) * (W / patch_size)
      Default: (320/16) * (1920/16) = 20 × 120 = 2400 patches.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from src.models.config import TokenizerConfig


class PatchEmbedding(nn.Module):
    """Convert images to patch token sequences via linear projection.

    Splits each image into non-overlapping patches and projects them
    to the embedding dimension using a strided convolution.

    Args:
        config: Tokenizer configuration.
    """

    def __init__(self, config: TokenizerConfig) -> None:
        super().__init__()
        self.config = config
        self.n_patches_h = config.img_height // config.patch_size
        self.n_patches_w = config.img_width // config.patch_size
        self.n_patches = self.n_patches_h * self.n_patches_w

        # Strided conv = patch extraction + linear projection
        self.projection = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

        # Learnable position embedding
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.n_patches, config.embed_dim) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch embeddings from images.

        Args:
            x: (B, C, H, W) image tensor, float32, ImageNet-normalized.

        Returns:
            (B, N_patches, embed_dim) patch tokens with position encoding.
        """
        if x.ndim == 5:
            batch, steps, channels, height, width = x.shape
            x = x.reshape(batch * steps, channels, height, width)
        if x.ndim != 4:
            raise ValueError(f"PatchEmbedding expected 4D or 5D input, got shape {tuple(x.shape)}")

        tokens = self.projection(x)  # (B, D, H/P, W/P)
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, N_patches, D)
        return cast(torch.Tensor, tokens + self.position_embedding)


class QShift(nn.Module):
    """Q-Shift spatial token mixing for 2D patch grids.

    Shifts portions of the token embedding channels in 4 spatial
    directions (up, down, left, right) to enable cross-patch
    information flow without attention. See RWKV-v5/v6 papers.

    Args:
        embed_dim: Token embedding dimension.
        n_patches_h: Number of patches along height.
        n_patches_w: Number of patches along width.
    """

    def __init__(self, embed_dim: int, n_patches_h: int, n_patches_w: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_patches_h = n_patches_h
        self.n_patches_w = n_patches_w

        # Each direction gets 1/4 of the channels
        assert embed_dim % 4 == 0, f"embed_dim ({embed_dim}) must be divisible by 4 for Q-Shift"
        self.shift_size = embed_dim // 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Q-Shift spatial mixing.

        Args:
            x: (B, N_patches, embed_dim) patch tokens.

        Returns:
            (B, N_patches, embed_dim) spatially mixed tokens.
        """
        if x.ndim != 3:
            raise ValueError(f"QShift expected 3D input, got shape {tuple(x.shape)}")
        batch, n_patches, embed_dim = x.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"QShift embed_dim mismatch: expected {self.embed_dim}, got {embed_dim}"
            )
        expected_patches = self.n_patches_h * self.n_patches_w
        if n_patches != expected_patches:
            raise ValueError(
                f"QShift patch count mismatch: expected {expected_patches}, got {n_patches}"
            )

        grid = x.view(batch, self.n_patches_h, self.n_patches_w, embed_dim)
        shift = self.shift_size
        chunk_up = grid[..., :shift]
        chunk_down = grid[..., shift : 2 * shift]
        chunk_left = grid[..., 2 * shift : 3 * shift]
        chunk_right = grid[..., 3 * shift :]

        up = torch.zeros_like(chunk_up)
        up[:, :-1, :, :] = chunk_up[:, 1:, :, :]

        down = torch.zeros_like(chunk_down)
        down[:, 1:, :, :] = chunk_down[:, :-1, :, :]

        left = torch.zeros_like(chunk_left)
        left[:, :, :-1, :] = chunk_left[:, :, 1:, :]

        right = torch.zeros_like(chunk_right)
        right[:, :, 1:, :] = chunk_right[:, :, :-1, :]

        mixed = torch.cat([up, down, left, right], dim=-1)
        return mixed.view(batch, n_patches, embed_dim)


class VisionTokenizer(nn.Module):
    """Complete vision tokenizer: PatchEmbedding + optional Q-Shift.

    Processes a sequence of images into visual token sequences
    suitable for the RWKV temporal encoder.

    Args:
        config: Tokenizer configuration.
    """

    def __init__(self, config: TokenizerConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(config)

        if config.use_qshift:
            self.qshift: QShift | None = QShift(
                embed_dim=config.embed_dim,
                n_patches_h=self.patch_embed.n_patches_h,
                n_patches_w=self.patch_embed.n_patches_w,
            )
        else:
            self.qshift = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize a sequence of images.

        Args:
            x: (B, T, C, H, W) image sequence tensor.

        Returns:
            (B, T, N_patches, embed_dim) visual token sequence.
        """
        if x.ndim != 5:
            raise ValueError(f"VisionTokenizer expected 5D input, got shape {tuple(x.shape)}")

        batch, steps, channels, height, width = x.shape
        flat = x.reshape(batch * steps, channels, height, width)
        tokens = self.patch_embed(flat)  # (B*T, N_patches, D)
        if self.qshift is not None:
            tokens = self.qshift(tokens)
        return cast(
            torch.Tensor,
            tokens.reshape(batch, steps, self.patch_embed.n_patches, self.config.embed_dim),
        )
