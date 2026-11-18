"""RWKV-4-style temporal encoder with data-dependent decay.

The memory is one number per channel, as in RWKV-4. RWKV-6 and RWKV-7 keep a matrix per
head instead, so this is neither of them.

Architecture per block:
  1. Time-mixing:
     - Token shift: each of r, k, v, w gets its own learned per-channel lerp
       between the current token and the previous-position token.
     - Data-dependent decay: per-token w_t = exp(-exp(W_w(xw) + w_log))
       gives each channel its own learned decay rate that varies with content.
     - WKV memory: an average of the values seen so far, weighted by their keys.
         a_t  = w_t * a_{t-1} + exp(k_t) * v_t
         b_t  = w_t * b_{t-1} + exp(k_t)
         y_t  = r_t * (a_{t-1} + exp(u + k_t) * v_t) / (b_{t-1} + exp(u + k_t))
       where u is a learned bonus for the current frame. Both sums are stored divided
       by exp(p), with p the largest log weight so far, and accumulated in fp32, so a
       large key cannot overflow them.
     - Output: W_o(y_t)
  2. Channel-mixing: squared-ReLU gated FFN with its own token shift.

Carried state per layer, so that how a sequence is cut into calls does not
change the output:
  - time-mixing shift: the last normalised input of the previous call,
  - WKV memory: (a, b, p),
  - channel-mixing shift: the last normalised input of its own token shift.

References:
  RWKV-4: Peng et al., 2023 (arXiv:2305.13048)
  Vision-RWKV: Duan et al., ICLR 2025 (arXiv:2403.02308)

Input:  (B, T, embed_dim) — per-frame tokens
Output: (B, T, embed_dim) — per-frame feature vectors
State:  list of tensors, three per layer: (B, D) shift, (B, 3, D) WKV memory, (B, D) shift
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.config import RWKVEncoderConfig
from src.models.temporal import TemporalCore, TemporalState

RWKVState = TemporalState
STATE_PARTS = 3  # tensors per layer: time-mixing shift, WKV memory, channel-mixing shift
NO_HISTORY = -1e30  # log weight of an empty WKV memory
MAX_LOG_DECAY_RATE = 10.0  # the memory is already gone at this rate; keeps exp() finite

WKVMemory = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _add_frame(memory: WKVMemory, k: torch.Tensor, v: torch.Tensor) -> WKVMemory:
    """Add a value of log weight `k` to (a, b, p), rescaling to the larger log weight."""
    a, b, p = memory
    top = torch.maximum(p, k)
    old, new = torch.exp(p - top), torch.exp(k - top)
    return old * a + new * v, old * b + new, top


def wkv_step(
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    bonus: torch.Tensor,
    memory: WKVMemory,
) -> tuple[torch.Tensor, WKVMemory]:
    """One frame of the WKV memory: the weighted average now, and the memory for the next frame."""
    a, b, _ = _add_frame(memory, bonus + k, v)
    decayed = (memory[0], memory[1], memory[2] + log_decay)
    return a / b, _add_frame(decayed, k, v)


class RWKVBlock(nn.Module):
    """Single block: token-shifted time-mixing + channel-mixing.

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

        # ── Time-mixing ──────────────────────────────────────────────
        # Per-channel lerp coefficients: how much to blend current vs prev token.
        self.lerp_r = nn.Parameter(torch.full((D,), 0.5))
        self.lerp_k = nn.Parameter(torch.full((D,), 0.5))
        self.lerp_v = nn.Parameter(torch.full((D,), 0.5))
        self.lerp_w = nn.Parameter(torch.full((D,), 0.5))

        # RKWV projections — no bias, standard convention in all RWKV versions.
        self.W_r = nn.Linear(D, D, bias=False)
        self.W_k = nn.Linear(D, D, bias=False)
        self.W_v = nn.Linear(D, D, bias=False)
        self.W_o = nn.Linear(D, D, bias=False)

        # Data-dependent decay: a linear map of the shifted input gives a per-token
        # log-decay rate, offset by a learnable per-channel baseline.
        # Final decay w_t = exp(-exp(W_w(xw) + w_log)) is always in (0, 1).
        self.W_w = nn.Linear(D, D, bias=False)
        self.w_log = nn.Parameter(torch.full((D,), -1.0))

        # Current-position bonus: prevents the model ignoring the current token.
        self.u = nn.Parameter(torch.zeros(D))

        # ── Channel-mixing (squared-ReLU gated FFN) ──────────────────
        self.lerp_cm_k = nn.Parameter(torch.full((D,), 0.5))
        self.lerp_cm_r = nn.Parameter(torch.full((D,), 0.5))
        self.W_cm_k = nn.Linear(D, F, bias=False)
        self.W_cm_v = nn.Linear(F, D, bias=False)
        self.W_cm_r = nn.Linear(D, D, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def initial_state(self, batch_size: int, device: torch.device) -> RWKVState:
        """The state before the first frame: empty shifts and an empty WKV memory."""
        size = (batch_size, self.config.embed_dim)
        memory = torch.zeros((batch_size, 3, self.config.embed_dim), device=device)
        memory[:, 2] = NO_HISTORY
        return [torch.zeros(size, device=device), memory, torch.zeros(size, device=device)]

    def forward(
        self,
        x: torch.Tensor,
        state: RWKVState | None = None,
    ) -> tuple[torch.Tensor, RWKVState]:
        """Process a clip through this block.

        Args:
            x:     (B, T, D) token sequence for this clip.
            state: This layer's state after the previous call, or None to start from zeros.

        Returns:
            output:    (B, T, D) processed tokens.
            new_state: This layer's state after the last frame.
        """
        if x.ndim != 3:
            raise ValueError(f"RWKVBlock expected 3D input, got shape {tuple(x.shape)}")

        B, T, _ = x.shape
        wide = torch.promote_types(x.dtype, torch.float32)  # memory is never kept in bf16

        if state is None:
            state = self.initial_state(B, x.device)
        time_shift, wkv_memory, channel_shift = state

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
        decay_rate = self.W_w(xw)  # log-decay rate   (B, T, D)

        with torch.autocast(x.device.type, enabled=False):
            k, v, r = k.to(wide), v.to(wide), r.to(wide)
            rate = (decay_rate.to(wide) + self.w_log).clamp(max=MAX_LOG_DECAY_RATE)
            log_decay = -torch.exp(rate)  # w_t = exp(log_decay) is always in (0, 1)

            a, b, p = wkv_memory.to(wide).unbind(dim=1)
            memory: WKVMemory = (a, b, p)
            averages: list[torch.Tensor] = []
            for t in range(T):
                average, memory = wkv_step(k[:, t], v[:, t], log_decay[:, t], self.u, memory)
                averages.append(average)
            y = r * torch.stack(averages, dim=1)

        time_out = self.W_o(y)  # (B, T, D)
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

        return x, [h[:, -1, :].to(wide), torch.stack(memory, dim=1), h2[:, -1, :].to(wide)]


class RWKVEncoder(TemporalCore):
    """Stacked RWKV-4-style blocks for temporal encoding of visual sequences.

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
