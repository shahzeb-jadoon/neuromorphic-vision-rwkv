# Phase II: Model Architecture & Training

> **Timeline:** Weeks 5–7
> **Goal:** Vision-RWKV driving agent trained and converging
> **Depends on:** Phase I (dataset ready)

---

## Overview

This is the core ML engineering phase. You will:
1. Implement the Vision-RWKV encoder with Q-Shift
2. Build the full driving pipeline (tokenizer → encoder → fusion → policy head)
3. Implement the TBPTT training loop with burn-in
4. Train on collected data until convergence
5. Implement baseline models for comparison

---

## Steps

### 2.1 — Patch Embedding & Q-Shift Tokenizer (Week 5)

**Objective:** Convert 224×224 RGB images into token sequences with spatial awareness.

**Tasks:**
- [ ] Implement `PatchEmbedding`: Conv2d(3, D, kernel=16, stride=16) → (B, 196, D) where D=768
- [ ] Implement `QShift` (Quad-directional Token Shift): shift tokens by 1 position in 4 spatial directions to inject 2D locality
- [ ] Add positional encoding (learnable 2D positional embeddings for 14×14 grid)
- [ ] Unit test: verify output shapes, gradient flow

**Key code artifact:** `src/models/tokenizer.py`

**Q-Shift explained:** After flattening the 14×14 patch grid into a 196-token sequence, Q-Shift splits the channel dimension into 4 groups. Each group is shifted by 1 position corresponding to up/down/left/right spatial neighbors. This recovers local 2D structure that raster-scan flattening destroys, without adding any parameters.

### 2.2 — RWKV-7 Encoder Block (Week 5)

**Objective:** Implement the core RWKV-7 time-mixing and channel-mixing blocks.

**Tasks:**
- [ ] Implement `TimeMixing` block with data-dependent decay (RWKV-7 formulation)
- [ ] Implement `ChannelMixing` block (feed-forward equivalent)
- [ ] Implement `RWKVBlock`: LayerNorm → TimeMixing → LayerNorm → ChannelMixing
- [ ] Stack L=12 blocks into `RWKVEncoder`
- [ ] Implement state management: `init_state()`, `forward_with_state(x, state) → (out, new_state)`
- [ ] Use official RWKV-7 CUDA kernels for the WKV operator if available, else pure PyTorch fallback
- [ ] Unit test: verify O(1) memory w.r.t. sequence length during inference

**Key code artifact:** `src/models/rwkv_encoder.py`

**State management is the most critical detail.** During training, states propagate within each 32-frame clip. During inference, states persist across frames indefinitely. The encoder must support both modes:

```python
class RWKVEncoder(nn.Module):
    def forward(self, x, state=None):
        """
        x: (B, T, D) — sequence of T token vectors
        state: list of (B, D, D) tensors per layer, or None
        Returns: (B, T, D) output, new_state
        """
```

### 2.3 — Sensor Fusion (LICA) & Policy Head (Week 5–6)

**Objective:** Fuse visual features with navigation commands, output driving controls.

**Tasks:**
- [ ] Implement `LICA` (Linear Cross-Attention): fuses visual features with nav command embedding
  - Nav command → learned embedding (dim 64)
  - Linear cross-attention: visual features attend to nav embedding without quadratic cost
- [ ] Implement `PolicyHead`: MLP that outputs [steer, throttle]
  - Architecture: Linear(D, 256) → ReLU → Linear(256, 64) → ReLU → Linear(64, 2)
  - Steering: tanh activation (range [-1, 1])
  - Throttle: sigmoid activation (range [0, 1])
- [ ] Optional: velocity prediction auxiliary head for improved representation learning

**Key code artifacts:** `src/models/fusion.py`, `src/models/policy_head.py`

### 2.4 — Full Model Assembly (Week 6)

**Objective:** Wire all components into a single end-to-end model.

**Tasks:**
- [ ] Implement `VisionRWKVDriver` that composes: PatchEmbedding → QShift → RWKVEncoder → LICA → PolicyHead
- [ ] Verify forward pass end-to-end: input (B, T, 3, 224, 224) → output (B, T, 2)
- [ ] Profile memory usage and forward pass time
- [ ] Implement model config via YAML (see `configs/`)

**Key code artifact:** `src/models/vision_rwkv_driver.py`

```python
class VisionRWKVDriver(nn.Module):
    def __init__(self, config):
        # config: d_model, n_layers, n_heads, img_size, patch_size, ...

    def forward(self, images, nav_commands, state=None):
        """
        images: (B, T, 3, H, W)
        nav_commands: (B, T) int
        state: optional recurrent state
        Returns: controls (B, T, 2), new_state
        """
```

### 2.5 — Training Loop with TBPTT (Week 6)

**Objective:** Implement training with Truncated Backpropagation Through Time and burn-in.

**Tasks:**
- [ ] Implement `Trainer` class with TBPTT
- [ ] Burn-in: first K=20 frames of each clip run forward-only (no gradient) to warm up state
- [ ] Loss: MSE on steering + throttle predictions (after burn-in frames only)
- [ ] Optimizer: AdamW, lr=1e-4, weight_decay=0.01
- [ ] LR schedule: linear warmup (1000 steps) + cosine decay
- [ ] Gradient clipping: max_norm=1.0
- [ ] Logging: TensorBoard and/or W&B
- [ ] Checkpointing: save every N epochs + best model by val loss
- [ ] Mixed precision (FP16) for faster training on workstation GPU

**Key code artifact:** `src/training/trainer.py`

**TBPTT burn-in explained:**
```
Frame:  [1  2  3  ... 20  21  22  ... 32]
         ←— burn-in —→  ←— loss computed —→
         (state warmup)   (gradients flow)
```

This prevents the model from being penalized on the first frames of a clip where the recurrent state is cold (zero-initialized at clip boundaries).

### 2.6 — Training Run (Week 6–7)

**Objective:** Train to convergence on the expert dataset.

**Tasks:**
- [ ] Run initial training on laptop (small batch, 1-2 epochs) to verify pipeline correctness
- [ ] Transfer to the 24 GB workstation for full training
- [ ] Train for 50+ epochs with early stopping (patience=10 on val loss)
- [ ] Monitor: loss curves, steering/throttle prediction distributions, gradient norms
- [ ] Batch size: 16-32 sequences (each 32 frames) — adjust based on VRAM
- [ ] Expected training time: ~30-50 GPU-hours on workstation GPU (24 GB)

**Training commands:**
```bash
# On the workstation:
conda activate rwkv
cd ~/projects/neuromorphic-vision-rwkv
python -m src.training.trainer --config configs/vrwkv_base.yaml
```

### 2.7 — Baseline Models (Week 7)

**Objective:** Implement comparison models to contextualize RWKV performance.

**Tasks:**
- [ ] **CNN Baseline**: ResNet-18 (pretrained ImageNet) → MLP policy head. No temporal modeling. Frame-by-frame.
- [ ] **Transformer Baseline**: ViT-B/16 with sliding window attention (window=32 frames). Standard self-attention within window.
- [ ] **LSTM Baseline**: ResNet-18 encoder → LSTM (hidden_size=512, 2 layers) → MLP policy head.
- [ ] Train all baselines with same data, optimizer, and schedule for fair comparison
- [ ] Log all baseline metrics for Phase III comparison

**Key code artifact:** `src/models/baselines/`

---

## Model Configurations

Create these YAML configs in `configs/`:

```yaml
# configs/vrwkv_base.yaml
model:
  name: "VisionRWKVDriver"
  d_model: 768
  n_layers: 12
  patch_size: 16
  img_size: 224
  nav_embed_dim: 64

training:
  batch_size: 16
  seq_len: 32
  burn_in: 20
  lr: 1e-4
  weight_decay: 0.01
  max_epochs: 100
  early_stop_patience: 10
  grad_clip: 1.0
  fp16: true

data:
  dataset_dir: "~/datasets/carla_expert"
  train_split: 0.8
  augmentation: true
```

---

## Completion Criteria

All of the following must be true before starting Phase III:

1. Vision-RWKV model trains without errors for 50+ epochs
2. Training loss converges (not diverging, not flat)
3. Val loss shows improvement and early stopping hasn't triggered too early
4. All 3 baselines trained with same protocol
5. Checkpoints saved for best model + all baselines
6. Training curves logged and visually inspected
7. Forward pass verified: (B, T, 3, 224, 224) → (B, T, 2) with correct ranges
