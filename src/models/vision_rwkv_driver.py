"""Vision-RWKV driver model: full pipeline from images to control signals.

Composes:
    VisionTokenizer → RWKVEncoder → LICA → PolicyHead

Input:
    images:      (B, T, 3, H, W)  — RGB camera frames
    nav_command: (B, T)            — navigation command per frame (0-4)
    state:       optional RWKVState — recurrent hidden state

Output:
    controls:    (B, T, 2)         — [steering, acceleration]
    state:       RWKVState         — updated recurrent state

Acceleration merges throttle and brake into a single axis:
    acceleration = throttle - brake  ∈ [-1, 1]
At inference time, split back:
    throttle = max(0, acceleration)
    brake    = max(0, -acceleration)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.config import ModelConfig
from src.models.fusion import LICA
from src.models.policy_head import PolicyHead
from src.models.rwkv_encoder import RWKVEncoder, RWKVState
from src.models.tokenizer import VisionTokenizer


class VisionRWKVDriver(nn.Module):
    """End-to-end driving model: Vision-RWKV with LICA fusion.

    This is the top-level model that chains:
    1. VisionTokenizer: images → patch tokens
    2. RWKVEncoder:     patch tokens → temporal features
    3. LICA:            temporal features + nav → fused features
    4. PolicyHead:      fused features → control predictions

    Args:
        config: Full model configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.tokenizer = VisionTokenizer(config.tokenizer)
        self.encoder = RWKVEncoder(config.encoder)
        self.fusion = LICA(config.fusion)
        self.policy_head = PolicyHead(config.policy_head)

    def forward(
        self,
        images: torch.Tensor,
        nav_command: torch.Tensor,
        state: RWKVState | None = None,
    ) -> tuple[torch.Tensor, RWKVState]:
        """End-to-end forward pass from images to controls.

        Args:
            images:      (B, T, 3, H, W) RGB camera frames, normalized [0, 1].
            nav_command: (B, T) integer navigation commands (0=follow, 1=left, etc.).
            state:       Optional recurrent state from previous forward call.
                         If None, initialized to zeros.

        Returns:
            controls: (B, T, 2) predicted [steering, acceleration].
            state:    Updated RWKVState for the next call.
        """
        raise NotImplementedError("VisionRWKVDriver.forward — implement in Phase 2")

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> RWKVState:
        """Create zero-initialized state for the encoder.

        Args:
            batch_size: Number of samples in the batch.
            device: Target device. Defaults to model's device.

        Returns:
            Zero-initialized RWKVState list.
        """
        if device is None:
            device = next(self.parameters()).device
        return self.encoder.initial_state(batch_size, device)
