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
    """GRU layers in the RWKV blocks' residual shape: the reference other cores are compared to.

    Each layer reads normalised features and adds its output to them. A plain stack halves
    the frame-dependent part of its input at every layer, and a large shared offset in the
    input saturates its gates; trained that way it stopped responding to the images.
    """

    def __init__(self, config: RWKVEncoderConfig) -> None:
        super().__init__()
        self.config = config
        dim, layers = config.embed_dim, range(config.n_layers)
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in layers)
        self.layers = nn.ModuleList(nn.GRU(dim, dim, batch_first=True) for _ in layers)
        self.dropout = nn.Dropout(config.dropout)
        self.ln_out = nn.LayerNorm(dim)

    def initial_state(self, batch_size: int, device: torch.device) -> TemporalState:
        size = (batch_size, self.config.embed_dim)
        return [torch.zeros(size, device=device) for _ in range(self.config.n_layers)]

    def forward(
        self, x: torch.Tensor, state: TemporalState | None = None
    ) -> tuple[torch.Tensor, TemporalState]:
        if state is None:
            state = self.initial_state(x.shape[0], x.device)
        new_state: TemporalState = []
        with torch.autocast(x.device.type, enabled=False):  # memory is always carried in fp32
            features = x.float()
            for norm, gru, memory in zip(self.norms, self.layers, state, strict=True):
                out, last = gru(norm(features), memory.unsqueeze(0))
                features = features + self.dropout(out)
                new_state.append(last.squeeze(0))
        return self.ln_out(features), new_state
