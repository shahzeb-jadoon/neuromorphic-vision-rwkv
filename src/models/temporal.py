"""Temporal cores: the model's memory, behind one streaming interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from src.models.config import RWKVEncoderConfig

# Carried memory: every tensor is (B, ...), whatever the core keeps in it.
TemporalState = list[torch.Tensor]


class TemporalCore(nn.Module, ABC):
    """Turns per-frame tokens into per-frame features, carrying memory between calls.

    How a sequence is cut into calls must not change the output, so training on clips
    and driving one frame at a time run the same computation.
    """

    @abstractmethod
    def initial_state(self, batch_size: int, device: torch.device) -> TemporalState:
        """Memory at the start of a sequence."""

    @abstractmethod
    def forward(
        self, x: torch.Tensor, state: TemporalState | None = None
    ) -> tuple[torch.Tensor, TemporalState]:
        """(B, T, D) tokens to (B, T, D) features and the memory after the last frame."""

    def step(
        self, x: torch.Tensor, state: TemporalState | None = None
    ) -> tuple[torch.Tensor, TemporalState]:
        """(B, D) token of one frame to (B, D) features and the memory after it."""
        features, state = self(x.unsqueeze(1), state)
        return features.squeeze(1), state


class GRUCore(TemporalCore):
    """Stacked GRU: the reference that other temporal cores are compared against."""

    def __init__(self, config: RWKVEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.gru = nn.GRU(
            config.embed_dim,
            config.embed_dim,
            config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0.0,
        )
        self.ln_out = nn.LayerNorm(config.embed_dim)

    def initial_state(self, batch_size: int, device: torch.device) -> TemporalState:
        size = (batch_size, self.config.embed_dim)
        return [torch.zeros(size, device=device) for _ in range(self.config.n_layers)]

    def forward(
        self, x: torch.Tensor, state: TemporalState | None = None
    ) -> tuple[torch.Tensor, TemporalState]:
        if state is None:
            state = self.initial_state(x.shape[0], x.device)
        with torch.autocast(x.device.type, enabled=False):  # memory is always carried in fp32
            features, last = self.gru(x.float(), torch.stack(state))
        return self.ln_out(features), list(last.unbind(0))
