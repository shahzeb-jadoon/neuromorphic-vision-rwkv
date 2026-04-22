"""Navigation fusion module (gated linear fusion).

Fuses visual features from the RWKV encoder with navigation
command embeddings to produce context-aware driving features.

Mechanism: concatenate visual + nav embeddings, project through
a linear layer, and apply a learned sigmoid gate. This is a gated
linear fusion — lightweight and O(1) per token.

Input:  visual_features (B, T, visual_dim)
        nav_command     (B, T) int64
Output: fused_features  (B, T, fused_dim)
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from src.models.config import FusionConfig


class NavigationFusion(nn.Module):
    """Gated linear fusion for visual + navigation features.

    Embeds the discrete navigation command into a continuous vector,
    concatenates with visual features, and applies a gated linear
    projection to produce fused driving features.

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
            nav_command: (B, T) int64, navigation command IDs (0-5).

        Returns:
            (B, T, fused_dim) fused feature tensor.
        """
        if visual_features.ndim != 3:
            raise ValueError(
                f"visual_features must be (B, T, D), got shape {tuple(visual_features.shape)}"
            )
        if nav_command.ndim != 2:
            raise ValueError(f"nav_command must be (B, T), got shape {tuple(nav_command.shape)}")
        if visual_features.shape[:2] != nav_command.shape:
            raise ValueError(
                "visual_features and nav_command batch/time dims must match: "
                f"{tuple(visual_features.shape[:2])} vs {tuple(nav_command.shape)}"
            )

        nav_embed = self.nav_embedding(nav_command)  # (B, T, nav_embed_dim)
        combined = torch.cat([visual_features, nav_embed], dim=-1)
        projected = self.projection(combined)
        gate = torch.sigmoid(self.gate(combined))

        if visual_features.shape[-1] == self.config.fused_dim:
            fused = gate * projected + (1.0 - gate) * visual_features
        else:
            fused = gate * projected

        fused = self.ln(fused)
        return cast(torch.Tensor, self.dropout(fused))
