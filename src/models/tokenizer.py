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
        raise NotImplementedError("PatchEmbedding.forward — implement in Phase 2")


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
        raise NotImplementedError("QShift.forward — implement in Phase 2")


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
        raise NotImplementedError("VisionTokenizer.forward — implement in Phase 2")
