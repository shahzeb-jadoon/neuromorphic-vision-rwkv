"""Policy head: maps fused features to driving control signals.

Outputs [steering, acceleration] predictions per frame.

Design Decision — Merged Acceleration Axis:
    The policy head outputs 2 values: [steering, acceleration].
    Acceleration merges throttle and brake into a single axis:
        acceleration = throttle - brake  ∈ [-1, 1]
    where positive = throttle, negative = brake.

    This avoids the throttle-brake conflict in expert data (the
    CARLA autopilot sometimes applies both simultaneously during
    transitions) and follows standard practice in imitation
    learning literature (Codevilla et al., 2018; Chen et al., 2020).

    At inference time, the predicted acceleration is split back:
        throttle = max(0, acceleration)
        brake    = max(0, -acceleration)

    The training target is computed from expert data:
        target_accel = expert_throttle - expert_brake

Input:  (B, T, input_dim) — fused features
Output: (B, T, 2) — [steering, acceleration] per frame
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.config import PolicyHeadConfig


class PolicyHead(nn.Module):
    """MLP policy head for driving control prediction.

    Architecture: D→256→ReLU→Drop→64→ReLU→Drop→2→tanh (3-layer MLP)
    - Steering:     tanh (output ∈ [-1, 1])
    - Acceleration: tanh (output ∈ [-1, 1]), positive = throttle, negative = brake

    Args:
        config: Policy head configuration.
    """

    def __init__(self, config: PolicyHeadConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        in_dim = config.input_dim
        for h_dim in config.hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                ]
            )
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, config.n_outputs))
        layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict driving controls from fused features.

        Args:
            x: (B, T, input_dim) fused feature tensor.

        Returns:
            (B, T, 2) control predictions:
                [:, :, 0] = steering     ∈ [-1, 1] (tanh)
                [:, :, 1] = acceleration ∈ [-1, 1] (tanh)
                    positive = throttle, negative = brake
        """
        raise NotImplementedError("PolicyHead.forward — implement in Phase 2")
