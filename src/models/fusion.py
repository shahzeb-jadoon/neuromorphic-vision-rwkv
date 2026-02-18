"""LICA (Linear Cross-Attention) fusion module.

Fuses visual features from the RWKV encoder with navigation
command embeddings to produce context-aware driving features.

The fusion is lightweight and linear to maintain the O(1)
inference cost guarantee of the RWKV pipeline.

Input:  visual_features (B, T, visual_dim)
        nav_command     (B, T) int64
Output: fused_features  (B, T, fused_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.config import FusionConfig


class LICA(nn.Module):
    """Linear Cross-Attention fusion for visual + navigation features.

    Embeds the discrete navigation command into a continuous vector,
    then fuses with visual features via gated linear projection.

    Args:
        config: Fusion configuration.
    """

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        self.config = config

        # Navigation command embedding
        self.nav_embedding = nn.Embedding(config.n_nav_commands, config.nav_embed_dim)

        # Concatenate visual + nav, then project to fused dim
        combined_dim = config.visual_dim + config.nav_embed_dim
        self.projection = nn.Linear(combined_dim, config.fused_dim)

        # Gating mechanism
        self.gate = nn.Linear(combined_dim, config.fused_dim)

        self.ln = nn.LayerNorm(config.fused_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        visual_features: torch.Tensor,
        nav_command: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse visual features with navigation command.

        Args:
            visual_features: (B, T, visual_dim) from RWKV encoder.
            nav_command: (B, T) int64, navigation command IDs (0-3).

        Returns:
            (B, T, fused_dim) fused feature tensor.
        """
        raise NotImplementedError("LICA.forward — implement in Phase 2")
