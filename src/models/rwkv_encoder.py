"""RWKV-6 temporal encoder for sequential visual token processing.

Implements a pure-PyTorch RWKV-6 simplified block and stacks N of them to
encode visual token sequences with recurrent temporal memory.

Architecture per block:
  1. Time-mixing (RWKV-6):
     - Token shift: each of r, k, v, w gets its own learned per-channel lerp
       between the current token and the previous-position token.  The
       previous-position token at t=0 is seeded from the carried shift-state.
     - Data-dependent decay: per-token w_t = exp(-exp(W_w(xw) + w_log))
       gives each channel its own learned decay rate that varies with content.
     - WKV recurrence (sequential, pure PyTorch):
         s_t  = w_t * s_{t-1} + exp(k_t) * v_t
         y_t  = r_t * (exp(u) * exp(k_t) * v_t + s_{t-1})
       where s is a per-channel running sum and u is a learned bonus for the
       current position (preventing the model from ignoring the current token).
     - Output: W_o(y_t)
  2. Channel-mixing: squared-ReLU gated FFN with its own token shift.

Carried state per layer, so that how a sequence is cut into calls does not
change the output:
  - time-mixing shift: the last normalised input of the previous call,
  - WKV: the running sum s,
  - channel-mixing shift: the last normalised input of its own token shift.

References:
  RWKV-4: Peng et al., 2023 (arXiv:2305.13048)
  RWKV-6: BlinkDL/RWKV-LM (open source, 2024)
  Vision-RWKV: Duan et al., ICLR 2025 (arXiv:2403.02308)

Input:  (B, T, embed_dim) — per-frame tokens
Output: (B, T, embed_dim) — per-frame feature vectors
State:  list[Tensor(B, embed_dim)] — three tensors per layer
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.config import RWKVEncoderConfig
from src.models.temporal import TemporalCore, TemporalState

# Three (B, D) tensors per encoder layer: time-mixing shift, WKV sum, channel-mixing shift.
RWKVState = TemporalState
STATE_PARTS = 3


class RWKVBlock(nn.Module):
    """Single RWKV-6 block: token-shifted time-mixing + channel-mixing.

    Args:
        config:    RWKV encoder configuration.
        layer_idx: Index of this block in the stack (unused at runtime,
                   kept for identification and future layer-wise settings).
    """

    def __init__(self, config: RWKVEncoderConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        D = config.embed_dim
        F = config.ffn_dim

        # ── Layer norms ──────────────────────────────────────────────
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)

        # ── RWKV-6 time-mixing ───────────────────────────────────────
        # Per-channel lerp coefficients: how much to blend current vs prev token.
        # Separate lerp for each signal (r, k, v, w) — the RWKV-6 feature vs v4.
        self.lerp_r = nn.Parameter(torch.full((D,), 0.5))
        self.lerp_k = nn.Parameter(torch.full((D,), 0.5))
        self.lerp_v = nn.Parameter(torch.full((D,), 0.5))
        self.lerp_w = nn.Parameter(torch.full((D,), 0.5))

        # RKWV projections — no bias, standard convention in all RWKV versions.
        self.W_r = nn.Linear(D, D, bias=False)
        self.W_k = nn.Linear(D, D, bias=False)
        self.W_v = nn.Linear(D, D, bias=False)
        self.W_o = nn.Linear(D, D, bias=False)

        # Data-dependent decay (RWKV-6): a small network maps shifted input
        # to a per-token log-decay, offset by a learnable per-channel baseline.
        # Final decay w_t = exp(-exp(W_w(xw) + w_log)) is always in (0, 1).
        self.W_w = nn.Linear(D, D, bias=False)
        self.w_log = nn.Parameter(torch.full((D,), -1.0))

        # Current-position bonus: prevents the model ignoring the current token.
        # exp(u) scales the key-value contribution at position t before state decay.
        self.u = nn.Parameter(torch.zeros(D))

        # ── Channel-mixing (squared-ReLU gated FFN) ──────────────────
        self.lerp_cm_k = nn.Parameter(torch.full((D,), 0.5))
        self.lerp_cm_r = nn.Parameter(torch.full((D,), 0.5))
        self.W_cm_k = nn.Linear(D, F, bias=False)
        self.W_cm_v = nn.Linear(F, D, bias=False)
        self.W_cm_r = nn.Linear(D, D, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def initial_state(self, batch_size: int, device: torch.device) -> RWKVState:
        """Zeros for the time-mixing shift, the WKV sum and the channel-mixing shift."""
        size = (batch_size, self.config.embed_dim)
        return [torch.zeros(size, device=device) for _ in range(STATE_PARTS)]

    def forward(
        self,
        x: torch.Tensor,
        state: RWKVState | None = None,
    ) -> tuple[torch.Tensor, RWKVState]:
        """Process a clip through this RWKV-6 block.

        Args:
            x:     (B, T, D) token sequence for this clip.
            state: This layer's state after the previous call, or None to start from zeros.

        Returns:
            output:    (B, T, D) processed tokens.
            new_state: This layer's state after the last frame.
        """
        if x.ndim != 3:
            raise ValueError(f"RWKVBlock expected 3D input, got shape {tuple(x.shape)}")

        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        if state is None:
            state = self.initial_state(B, device)
        time_shift, s, channel_shift = state

        # ── Time-mixing ──────────────────────────────────────────────
        h = self.ln1(x)

        # Token shift: prepend the carried token as the virtual t=-1 token.
        h_prev = torch.cat([time_shift.to(h.dtype).unsqueeze(1), h[:, :-1, :]], dim=1)  # (B, T, D)

        # RWKV-6: separate lerp for each projection input.
        # lerp_r ∈ ℝ^D is unconstrained; optimizer keeps it near reasonable values.
        xr = h + (h_prev - h) * self.lerp_r  # (B, T, D)
        xk = h + (h_prev - h) * self.lerp_k
        xv = h + (h_prev - h) * self.lerp_v
        xw = h + (h_prev - h) * self.lerp_w

        r = torch.sigmoid(self.W_r(xr))  # receptance gate  (B, T, D)
        k = self.W_k(xk)  # key              (B, T, D)
        v = self.W_v(xv)  # value            (B, T, D)

        # Data-dependent per-token decay: always ∈ (0, 1).
        # Inner exp maps anything to ℝ+; outer -exp maps that to ℝ-; outer exp → (0,1).
        # Cast to float32 before exp() — fp16 overflows to inf when input > ~11,
        # which causes NaN in subsequent recurrence steps under AMP autocast.
        w = torch.exp(-torch.exp((self.W_w(xw) + self.w_log).float())).to(dtype)  # (B, T, D)

        # WKV recurrence — sequential over the T dimension.
        # s: (B, D) running exponentially-decayed sum, carried across calls.
        # y_t = r_t * (bonus * k_t * v_t + s_{t-1})
        # s_t =  w_t * s_{t-1} + k_t * v_t
        ek = torch.exp(k.float()).to(dtype)  # key: cast to fp32 to prevent overflow
        eu = torch.exp(self.u.float()).to(dtype)  # bonus: same reason
        s = s.to(dtype)
        y_parts: list[torch.Tensor] = []
        for t in range(T):
            wt = w[:, t, :]
            kt = ek[:, t, :]
            vt = v[:, t, :]
            rt = r[:, t, :]
            kv = kt * vt  # (B, D)
            y_parts.append(rt * (eu * kv + s))
            s = wt * s + kv

        time_out = self.W_o(torch.stack(y_parts, dim=1))  # (B, T, D)
        x = x + self.dropout(time_out)

        # ── Channel-mixing ───────────────────────────────────────────
        h2 = self.ln2(x)
        h2_prev = torch.cat([channel_shift.to(h2.dtype).unsqueeze(1), h2[:, :-1, :]], dim=1)

        xk2 = h2 + (h2_prev - h2) * self.lerp_cm_k
        xr2 = h2 + (h2_prev - h2) * self.lerp_cm_r

        k2 = torch.relu(self.W_cm_k(xk2)) ** 2  # squared ReLU (standard RWKV channel-mix)
        r2 = torch.sigmoid(self.W_cm_r(xr2))
        channel_out = r2 * self.W_cm_v(k2)
        x = x + self.dropout(channel_out)

        return x, [h[:, -1, :], s, h2[:, -1, :]]


class RWKVEncoder(TemporalCore):
    """Stacked RWKV-6 blocks for temporal encoding of visual sequences.

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
        """State at the start of a sequence, `STATE_PARTS` tensors per layer."""
        return [part for block in self.blocks for part in block.initial_state(batch_size, device)]

    def forward(
        self,
        x: torch.Tensor,
        state: RWKVState | None = None,
    ) -> tuple[torch.Tensor, RWKVState]:
        """Encode a sequence of per-frame tokens.

        Args:
            x:     (B, T, embed_dim) per-frame tokens.
            state: State after the previous call, or None to start a sequence.

        Returns:
            features:  (B, T, embed_dim) per-frame feature vectors.
            new_state: State after the last frame.
        """
        if x.ndim != 3:
            raise ValueError(f"RWKVEncoder expected 3D input, got shape {tuple(x.shape)}")

        batch, _, embed = x.shape
        if embed != self.config.embed_dim:
            raise ValueError(f"embed_dim mismatch: expected {self.config.embed_dim}, got {embed}")

        if state is None:
            state = self.initial_state(batch, x.device)
        expected = STATE_PARTS * len(self.blocks)
        if len(state) != expected:
            raise ValueError(f"state length mismatch: expected {expected}, got {len(state)}")

        hidden = x
        new_state: RWKVState = []
        for i, block in enumerate(self.blocks):
            hidden, layer_state = block(hidden, state[STATE_PARTS * i : STATE_PARTS * (i + 1)])
            new_state.extend(layer_state)

        return self.ln_out(hidden), new_state
