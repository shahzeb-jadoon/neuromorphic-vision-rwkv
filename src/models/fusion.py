"""Navigation fusion module (gated linear fusion).

Fuses visual features from the temporal core with navigation command, ego speed
and target point embeddings to produce context-aware driving features.

Mechanism: concatenate visual + nav + speed + target embeddings, project through
a linear layer, and apply a learned sigmoid gate. This is a gated
linear fusion — lightweight and O(1) per token.

Input:  visual_features (B, T, visual_dim)
        nav_command     (B, T) int64
        speed           (B, T) normalised ego speed
        target_point    (B, T, 2) route point ahead: x forward, y right, metres
Output: fused_features  (B, T, fused_dim)
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from src.models.config import FusionConfig


class NavigationFusion(nn.Module):
    """Gated linear fusion for visual, navigation, speed and route features.

    Embeds the navigation command, the ego speed and the target point, concatenates them
    with the visual features, and applies a gated linear projection.

    In training, `config.speed_dropout` of the clips see a learned "unknown speed" vector
    for all their frames, so the policy cannot simply continue the motion it measures.

    Args:
        config: Fusion configuration.
    """

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        self.config = config

        # Navigation command embedding
        self.nav_embedding = nn.Embedding(config.n_nav_commands, config.nav_embed_dim)

        speed_dim, target_dim = config.speed_embed_dim, config.target_embed_dim
        self.speed_encoder = nn.Sequential(
            nn.Linear(1, speed_dim), nn.ReLU(), nn.Linear(speed_dim, speed_dim)
        )
        self.unknown_speed = nn.Parameter(torch.zeros(speed_dim))
        self.target_encoder = nn.Sequential(
            nn.Linear(2, target_dim), nn.ReLU(), nn.Linear(target_dim, target_dim)
        )

        # Concatenate visual + nav + speed + target, then project to fused dim
        combined_dim = config.visual_dim + config.nav_embed_dim + speed_dim + target_dim
        self.projection = nn.Linear(combined_dim, config.fused_dim)

        # Gating mechanism
        self.gate = nn.Linear(combined_dim, config.fused_dim)

        self.ln = nn.LayerNorm(config.fused_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        visual_features: torch.Tensor,
        nav_command: torch.Tensor,
        speed: torch.Tensor,
        target_point: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse visual features with the navigation command, ego speed and target point.

        Args:
            visual_features: (B, T, visual_dim) from the temporal core.
            nav_command: (B, T) int64, navigation command IDs (0-5).
            speed: (B, T) ego speed, normalised as in the dataset's `speed` field.
            target_point: (B, T, 2) route point ahead in the vehicle's frame, metres.

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
        if speed.shape != nav_command.shape:
            raise ValueError(
                f"speed must be (B, T) like nav_command, got shape {tuple(speed.shape)}"
            )
        if target_point.shape != (*nav_command.shape, 2):
            raise ValueError(
                f"target_point must be (B, T, 2), got shape {tuple(target_point.shape)}"
            )

        nav_embed = self.nav_embedding(nav_command)  # (B, T, nav_embed_dim)
        target_embed = self.target_encoder(target_point / self.config.target_scale)
        combined = torch.cat(
            [visual_features, nav_embed, self._embed_speed(speed), target_embed], dim=-1
        )
        projected = self.projection(combined)
        gate = torch.sigmoid(self.gate(combined))

        if visual_features.shape[-1] == self.config.fused_dim:
            fused = gate * projected + (1.0 - gate) * visual_features
        else:
            fused = gate * projected

        fused = self.ln(fused)
        return cast(torch.Tensor, self.dropout(fused))

    def _embed_speed(self, speed: torch.Tensor) -> torch.Tensor:
        """(B, T) speed to (B, T, speed_embed_dim), hidden from whole clips in training."""
        embedded = cast(torch.Tensor, self.speed_encoder(speed.unsqueeze(-1)))
        if not self.training:
            return embedded
        hidden = torch.rand(speed.shape[0], device=speed.device) < self.config.speed_dropout
        return torch.where(hidden[:, None, None], self.unknown_speed.to(embedded.dtype), embedded)
