"""Vision-RWKV driver model: full pipeline from images to control signals.

Composes:
    VisionTokenizer → TemporalCore → NavigationFusion → PolicyHead
                                                      → SpeedHead     (auxiliary)
                                                      → ModeHead      (auxiliary)
                                                      → WaypointHead  (auxiliary)

Input:
    images:      (B, T, 3, H, W)  — RGB camera frames
    nav_command: (B, T)            — navigation command per frame (0-5)
    speed:       (B, T)            — normalised ego speed per frame
    target_point: (B, T, 2)        — route point ahead per frame (x forward, y right, metres)
    state:       optional TemporalState — recurrent hidden state

Output:
    controls:    (B, T, 2)         — [steering, acceleration]
    state:       TemporalState     — updated recurrent state
    aux:         dict of auxiliary predictions (speed, mode, waypoints)

Acceleration merges throttle and brake into a single axis:
    acceleration = throttle - brake  ∈ [-1, 1]
At inference time, split back:
    throttle = max(0, acceleration)
    brake    = max(0, -acceleration)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn

from src.models.config import ModelConfig, RWKVEncoderConfig
from src.models.fusion import NavigationFusion
from src.models.policy_head import PolicyHead
from src.models.rwkv_encoder import RWKVEncoder
from src.models.temporal import GRUCore, TemporalCore, TemporalState
from src.models.tokenizer import VisionTokenizer


def build_temporal_core(config: RWKVEncoderConfig) -> TemporalCore:
    """The temporal core named by `config.core`."""
    cores: dict[str, Callable[[RWKVEncoderConfig], TemporalCore]] = {
        "rwkv": RWKVEncoder,
        "gru": GRUCore,
    }
    if config.core not in cores:
        raise ValueError(f"Unknown temporal core '{config.core}', expected one of {list(cores)}")
    return cores[config.core](config)


# ---------------------------------------------------------------------------
# Auxiliary prediction heads (Phase 2)
# ---------------------------------------------------------------------------

N_DRIVING_MODES = 8  # sample-format classes; src/data/schema_adapter.py folds newer modes
N_WAYPOINT_OFFSETS = 4  # Future position predictions (t+5, t+10, t+15, t+20)


class SpeedHead(nn.Module):
    """Predicts target speed (km/h) from fused features. MSE loss."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) → (B, T, 1) target speed prediction."""
        return cast(torch.Tensor, self.linear(x))


class ModeHead(nn.Module):
    """Predicts driving mode (8-class classification) from fused features. CE loss."""

    def __init__(self, input_dim: int, n_modes: int = N_DRIVING_MODES) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, n_modes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) → (B, T, 8) driving mode logits."""
        return cast(torch.Tensor, self.linear(x))


class WaypointHead(nn.Module):
    """Predicts future (x, y) positions from fused features. L1 loss.

    Outputs K future positions in the vehicle's frame (x forward, y right).
    K defaults to 4 (at frame offsets t+5, t+10, t+15, t+20).
    """

    def __init__(self, input_dim: int, n_offsets: int = N_WAYPOINT_OFFSETS) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_offsets * 2),  # (x, y) per offset
        )
        self.n_offsets = n_offsets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) → (B, T, K, 2) future waypoint predictions."""
        out = self.mlp(x)
        batch, steps, _ = out.shape
        return cast(torch.Tensor, out.view(batch, steps, self.n_offsets, 2))


# Phase 3 ablation auxiliary heads (documented for future implementation):
# - TrafficLightHead: Linear(D, 4) → TL state classification (None/G/Y/R)
# - TrafficLightDistHead: Linear(D, 1) → TL distance regression
# - StopSignDistHead: Linear(D, 1) → stop sign distance regression
# - NearbyActorHead: MLP(D, 128, max_actors*5) → nearby vehicle/ped detection


class VisionRWKVDriver(nn.Module):
    """End-to-end driving model: Vision-RWKV with navigation fusion.

    This is the top-level model that chains:
    1. VisionTokenizer:  images → one token per frame
    2. TemporalCore:     frame tokens → temporal features
    3. NavigationFusion: temporal features + nav + speed + target point → fused features
    4. PolicyHead:       fused features → control predictions
    5. Auxiliary heads:  fused features → speed, mode, waypoint predictions

    Args:
        config: Full model configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.tokenizer = VisionTokenizer(config.tokenizer)
        self.encoder = build_temporal_core(config.encoder)
        self.fusion = NavigationFusion(config.fusion)
        self.policy_head = PolicyHead(config.policy_head)

        # Auxiliary heads (Phase 2)
        fused_dim = config.fusion.fused_dim
        self.speed_head = SpeedHead(fused_dim)
        self.mode_head = ModeHead(fused_dim)
        self.waypoint_head = WaypointHead(fused_dim)

    def forward(
        self,
        images: torch.Tensor,
        nav_command: torch.Tensor,
        speed: torch.Tensor,
        target_point: torch.Tensor,
        state: TemporalState | None = None,
    ) -> tuple[torch.Tensor, TemporalState, dict[str, torch.Tensor]]:
        """End-to-end forward pass from images to controls.

        Args:
            images:      (B, T, 3, H, W) RGB camera frames, normalized [0, 1].
            nav_command: (B, T) integer navigation commands (0=follow, 1=left, etc.).
            speed:       (B, T) ego speed, normalised as in the dataset's `speed` field.
            target_point: (B, T, 2) route point ahead in the vehicle's frame, metres.
            state:       Optional recurrent state from previous forward call.
                         If None, initialized to zeros.

        Returns:
            controls: (B, T, 2) predicted [steering, acceleration].
            state:    Updated TemporalState for the next call.
            aux:      Dict of auxiliary predictions keyed by name:
                          speed           (B, T, 1)
                          mode_logits     (B, T, 8)
                          future_waypoints (B, T, K, 2)
        """
        tokens = self.tokenizer(images)
        features, new_state = self.encoder(tokens, state)
        fused = self.fusion(features, nav_command, speed, target_point)
        controls = self.policy_head(fused)

        aux: dict[str, torch.Tensor] = {
            "speed": self.speed_head(fused),
            "mode_logits": self.mode_head(fused),
            "future_waypoints": self.waypoint_head(fused),
        }

        return controls, new_state, aux

    def step(
        self,
        image: torch.Tensor,
        nav_command: torch.Tensor,
        speed: torch.Tensor,
        target_point: torch.Tensor,
        state: TemporalState | None = None,
    ) -> tuple[torch.Tensor, TemporalState, dict[str, torch.Tensor]]:
        """`forward` for one frame: every input without its time axis, and so are the outputs."""
        controls, state, aux = self(
            image.unsqueeze(1),
            nav_command.unsqueeze(1),
            speed.unsqueeze(1),
            target_point.unsqueeze(1),
            state,
        )
        return controls.squeeze(1), state, {k: v.squeeze(1) for k, v in aux.items()}

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> TemporalState:
        """Create zero-initialized state for the temporal core.

        Args:
            batch_size: Number of samples in the batch.
            device: Target device. Defaults to model's device.

        Returns:
            Zero-initialized TemporalState list.
        """
        if device is None:
            device = next(self.parameters()).device
        return self.encoder.initial_state(batch_size, device)
