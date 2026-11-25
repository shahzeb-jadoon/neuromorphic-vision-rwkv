"""RWKV-7 ("Goose", x070) temporal encoder in plain PyTorch.

Each block follows the official RWKV-LM implementation (RWKV-v7, `RWKV_Tmix_x070` and
`RWKV_CMix_x070`), with its initialisation and parameter names:

  1. Time mixing: token shift into r, w, k, v, a, g; a matrix memory per head of
     `head_size` channels, updated by the generalised delta rule
         S_t = S_{t-1} diag(w_t) - S_{t-1} kk_t (a_t * kk_t)^T + v_t k_t^T,   y_t = S_t r_t
     where w_t is a data-dependent decay, kk_t the normalised key and a_t the in-context
     learning rate: the memory can replace what it holds for a key, not only let it fade.
     Every layer after the first mixes in the first layer's values (value residual).
  2. Channel mixing: squared-ReLU feed-forward with its own token shift.

The memory is accumulated in fp32 one frame at a time, as the reference's plain-PyTorch
operator does; the fast chunked kernels are for long contexts (plan step 7.1).

Carried state per layer, so that how a sequence is cut into calls does not change the output:
  - time-mixing shift: the last normalised input of the previous call, (B, C)
  - WKV memory: (B, H, N, N), N = head_size
  - channel-mixing shift, (B, C)

References:
  RWKV-7: Peng et al., 2025 (arXiv:2503.14456); github.com/BlinkDL/RWKV-LM, RWKV-v7.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.config import RWKVEncoderConfig
from src.models.temporal import TemporalCore, TemporalState

STATE_PARTS = 3  # tensors per layer: time-mixing shift, WKV memory, channel-mixing shift


def _lora_dim(channels: int, factor: float) -> int:
    """The official suggestion for a low-rank projection's width."""
    return max(32, int(round(factor * channels**0.5 / 32) * 32))


def _ortho(rows: int, cols: int, scale: float) -> nn.Parameter:
    weight = torch.zeros(rows, cols)
    gain = math.sqrt(rows / cols) if rows > cols else 1.0
    nn.init.orthogonal_(weight, gain=gain * scale)
    return nn.Parameter(weight)


def wkv7(
    r: torch.Tensor,
    log_decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    memory: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The RWKV-7 memory over T frames, in fp32.

    r, log_decay, k, v, a, b: (B, T, H, N), with decay exp(-exp(log_decay)); memory:
    (B, H, N, N), values by keys. Returns (B, T, H, N) outputs and the memory after the last
    frame: S = S * decay + (S a) b^T + v k^T, then y = S r.
    """
    with torch.autocast(r.device.type, enabled=False):
        r, k, v, a, b = (t.float() for t in (r, k, v, a, b))
        decay = torch.exp(-torch.exp(log_decay.float()))
        memory = memory.float()
        outputs = []
        for t in range(r.shape[1]):
            memory = (
                memory * decay[:, t, :, None, :]
                + (memory @ a[:, t, :, :, None]) @ b[:, t, :, None, :]
                + v[:, t, :, :, None] @ k[:, t, :, None, :]
            )
            outputs.append((memory @ r[:, t, :, :, None]).squeeze(-1))
        return torch.stack(outputs, dim=1), memory


class TimeMix7(nn.Module):
    """RWKV-7 time mixing (`RWKV_Tmix_x070`)."""

    def __init__(self, config: RWKVEncoderConfig, layer_id: int) -> None:
        super().__init__()
        C, N = config.embed_dim, config.head_size
        if C % N:
            raise ValueError(f"embed_dim {C} is not a multiple of head_size {N}")
        H = C // N
        self.heads, self.layer_id = H, layer_id
        ratio_0_to_1 = layer_id / max(1, config.n_layers - 1)
        ratio_1_to_almost0 = 1.0 - layer_id / config.n_layers
        ddd = (torch.arange(C) / C).view(1, 1, C)
        n = torch.arange(C, dtype=torch.float32)
        linear = n / (C - 1) - 0.5
        zigzag = ((n % N) - (N - 1) / 2) / ((N - 1) / 2)
        zigzag = zigzag * zigzag.abs()
        www = -6 + 6 * (n / (C - 1)) ** (1 + ratio_0_to_1**0.3)

        def mix(power: float) -> nn.Parameter:
            return nn.Parameter(1.0 - torch.pow(ddd, power * ratio_1_to_almost0))

        self.x_r, self.x_w, self.x_k = mix(0.2), mix(0.9), mix(0.7)
        self.x_v, self.x_a, self.x_g = mix(0.7), mix(0.9), mix(0.2)

        decay_dim, rate_dim = _lora_dim(C, 2.5), _lora_dim(C, 2.5)
        value_dim, gate_dim = _lora_dim(C, 1.7), _lora_dim(C, 5.0)
        self.w0 = nn.Parameter((www + 0.5 + zigzag * 2.5).view(1, 1, C))
        self.w1 = nn.Parameter(torch.zeros(C, decay_dim))
        self.w2 = _ortho(decay_dim, C, 0.1)
        self.a0 = nn.Parameter((-0.19 + zigzag * 0.3 + linear * 0.4).view(1, 1, C))
        self.a1 = nn.Parameter(torch.zeros(C, rate_dim))
        self.a2 = _ortho(rate_dim, C, 0.1)
        if layer_id > 0:  # the first layer's values are the ones mixed in
            self.v0 = nn.Parameter((0.73 - linear * 0.4).view(1, 1, C))
            self.v1 = nn.Parameter(torch.zeros(C, value_dim))
            self.v2 = _ortho(value_dim, C, 0.1)
        self.g1 = nn.Parameter(torch.zeros(C, gate_dim))
        self.g2 = _ortho(gate_dim, C, 0.1)
        self.k_k = nn.Parameter((0.71 - linear * 0.1).view(1, 1, C))
        self.k_a = nn.Parameter(torch.full((1, 1, C), 1.02))
        self.r_k = nn.Parameter(torch.full((H, N), -0.04))

        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)
        with torch.no_grad():
            self.receptance.weight.uniform_(-0.5 / C**0.5, 0.5 / C**0.5)
            self.key.weight.uniform_(-0.05 / C**0.5, 0.05 / C**0.5)
            self.value.weight.uniform_(-0.5 / C**0.5, 0.5 / C**0.5)
            self.output.weight.zero_()

    def forward(
        self,
        x: torch.Tensor,
        shift: torch.Tensor,
        memory: torch.Tensor,
        v_first: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(B, T, C) normalised input to (B, T, C) output, new shift and memory, v_first."""
        B, T, C = x.shape
        H = self.heads
        xx = torch.cat([shift.unsqueeze(1).to(x.dtype), x[:, :-1]], dim=1) - x
        xr, xw, xk = x + xx * self.x_r, x + xx * self.x_w, x + xx * self.x_k
        xv, xa, xg = x + xx * self.x_v, x + xx * self.x_a, x + xx * self.x_g

        r = self.receptance(xr)
        log_decay = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if v_first is None:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = F.normalize((k * self.k_k).view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, H, -1)

        y, memory = wkv7(
            heads(r), heads(log_decay), heads(k), heads(v), heads(-kk), heads(kk * a), memory
        )
        y = self.ln_x(y.reshape(B * T, C)).view(B, T, C)
        bonus = (heads(r) * heads(k) * self.r_k).sum(dim=-1, keepdim=True) * heads(v)
        y = y + bonus.reshape(B, T, C)
        return self.output(y * g), x[:, -1], memory, v_first


class ChannelMix7(nn.Module):
    """RWKV-7 channel mixing (`RWKV_CMix_x070`)."""

    def __init__(self, config: RWKVEncoderConfig, layer_id: int) -> None:
        super().__init__()
        C = config.embed_dim
        ddd = (torch.arange(C) / C).view(1, 1, C)
        self.x_k = nn.Parameter(1.0 - torch.pow(ddd, (1.0 - layer_id / config.n_layers) ** 4))
        self.key = nn.Linear(C, config.ffn_dim, bias=False)
        self.value = nn.Linear(config.ffn_dim, C, bias=False)
        with torch.no_grad():
            self.key.weight.uniform_(-0.5 / C**0.5, 0.5 / C**0.5)
            self.value.weight.zero_()

    def forward(self, x: torch.Tensor, shift: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T, C) normalised input to (B, T, C) output and the new shift."""
        xx = torch.cat([shift.unsqueeze(1).to(x.dtype), x[:, :-1]], dim=1) - x
        k = torch.relu(self.key(x + xx * self.x_k)) ** 2
        return cast(torch.Tensor, self.value(k)), x[:, -1]


class RWKV7Block(nn.Module):
    """Pre-norm residual block: time mixing, then channel mixing."""

    def __init__(self, config: RWKVEncoderConfig, layer_id: int) -> None:
        super().__init__()
        C = config.embed_dim
        self.ln0 = nn.LayerNorm(C) if layer_id == 0 else None
        self.ln1 = nn.LayerNorm(C)
        self.ln2 = nn.LayerNorm(C)
        self.att = TimeMix7(config, layer_id)
        self.ffn = ChannelMix7(config, layer_id)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, state: TemporalState, v_first: torch.Tensor | None
    ) -> tuple[torch.Tensor, TemporalState, torch.Tensor]:
        if self.ln0 is not None:
            x = self.ln0(x)
        out, att_shift, memory, v_first = self.att(self.ln1(x), state[0], state[1], v_first)
        x = x + self.dropout(out)
        out, ffn_shift = self.ffn(self.ln2(x), state[2])
        x = x + self.dropout(out)
        return x, [att_shift.float(), memory, ffn_shift.float()], v_first


class RWKV7Encoder(TemporalCore):
    """Stacked RWKV-7 blocks over per-frame tokens.

    Args:
        config: Temporal core configuration (embed_dim a multiple of head_size).
    """

    def __init__(self, config: RWKVEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(RWKV7Block(config, i) for i in range(config.n_layers))
        self.ln_out = nn.LayerNorm(config.embed_dim)

    def initial_state(self, batch_size: int, device: torch.device) -> TemporalState:
        C, N = self.config.embed_dim, self.config.head_size
        state: TemporalState = []
        for _ in self.blocks:
            state += [
                torch.zeros(batch_size, C, device=device),
                torch.zeros(batch_size, C // N, N, N, device=device),
                torch.zeros(batch_size, C, device=device),
            ]
        return state

    def forward(
        self, x: torch.Tensor, state: TemporalState | None = None
    ) -> tuple[torch.Tensor, TemporalState]:
        if state is None:
            state = self.initial_state(x.shape[0], x.device)
        v_first: torch.Tensor | None = None
        new_state: TemporalState = []
        for i, block in enumerate(self.blocks):
            x, layer_state, v_first = block(
                x, state[STATE_PARTS * i : STATE_PARTS * (i + 1)], v_first
            )
            new_state += layer_state
        return self.ln_out(x), new_state
