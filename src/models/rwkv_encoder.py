"""RWKV temporal encoder for sequential visual token processing.

Processes sequences of visual tokens through stacked RWKV blocks,
maintaining a recurrent state that captures temporal context across
frames. Supports RWKV-7 (primary) and RWKV-6 (fallback).

Input:  (B, T, N_patches, embed_dim) — visual token sequences
Output: (B, T, embed_dim) — per-frame feature vectors
State:  list of per-layer state tensors (carry across sequences)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.config import RWKVEncoderConfig

# Type alias for the recurrent state: one tensor per layer
RWKVState = list[torch.Tensor]


class RWKVBlock(nn.Module):
    """Single RWKV block: time-mixing + channel-mixing.

    Each block performs:
    1. Time-mixing (recurrent attention with decay)
    2. Channel-mixing (feed-forward with gating)

    The block maintains per-layer state for recurrent inference.

    Args:
        config: RWKV encoder configuration.
        layer_idx: Index of this block in the stack.
    """

    def __init__(self, config: RWKVEncoderConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.embed_dim = config.embed_dim

        # Layer norms
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)

        # Placeholder linear layers — actual RWKV ops implemented in Phase 2
        self.time_mix = nn.Linear(config.embed_dim, config.embed_dim)
        self.channel_mix = nn.Linear(config.embed_dim, config.ffn_dim)
        self.channel_out = nn.Linear(config.ffn_dim, config.embed_dim)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process tokens through this RWKV block.

        Args:
            x: (B, T, embed_dim) input tokens.
            state: (B, state_size) recurrent state from previous call,
                   or None for fresh state.

        Returns:
            Tuple of:
                output: (B, T, embed_dim) processed tokens.
                new_state: (B, state_size) updated recurrent state.
        """
        raise NotImplementedError("RWKVBlock.forward — implement in Phase 2")


class RWKVEncoder(nn.Module):
    """Stacked RWKV blocks for temporal encoding.

    Processes visual token sequences through N RWKV blocks and
    pools the spatial dimension to produce per-frame features.

    Args:
        config: RWKV encoder configuration.
    """

    def __init__(self, config: RWKVEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            [RWKVBlock(config, layer_idx=i) for i in range(config.n_layers)]
        )
        self.ln_out = nn.LayerNorm(config.embed_dim)

    def initial_state(self, batch_size: int, device: torch.device) -> RWKVState:
        """Create zero-initialized recurrent state for all layers.

        Args:
            batch_size: Batch size B.
            device: Target device.

        Returns:
            List of N_layers state tensors, each (B, state_size).
        """
        # Each layer gets a zero state — actual shape depends on RWKV version
        # For now, use embed_dim as a placeholder state size
        return [
            torch.zeros(batch_size, self.config.embed_dim, device=device)
            for _ in range(self.config.n_layers)
        ]

    def forward(
        self,
        x: torch.Tensor,
        state: RWKVState | None = None,
    ) -> tuple[torch.Tensor, RWKVState]:
        """Encode a sequence of visual tokens.

        Args:
            x: (B, T, N_patches, embed_dim) visual token sequence.
            state: List of per-layer states, or None for fresh state.

        Returns:
            Tuple of:
                features: (B, T, embed_dim) per-frame feature vectors
                    (spatial dimension pooled via mean).
                new_state: Updated list of per-layer states.
        """
        raise NotImplementedError("RWKVEncoder.forward — implement in Phase 2")
