# Phase II: Model Architecture & Training

> **Timeline:** Weeks 5–7
> **Goal:** Vision-RWKV driving agent trained and converging
> **Depends on:** Phase I (dataset ready)

---

## Overview

This is the core ML engineering phase. You will:

- **Vision encoder** — Implement patch tokenization with Q-Shift (3 stages: frozen pretrained → fine-tuned → from-scratch)
- **RWKV encoder** — Build the RWKV-7 encoder with data-dependent decay (attempt-first-then-pivot to RWKV-6)
- **Full pipeline** — Assemble tokenizer → encoder → NavigationFusion → policy head
- **Training** — Implement TBPTT with burn-in, steering noise injection for implicit recovery, crash-safe checkpointing
- **Baselines** — Train CNN, Transformer, and LSTM baselines with identical protocol

### Branching Strategy for Concurrent Experiments

With multiple GPUs available, run experiments in parallel on separate branches:

| Branch | Description | GPU |
|--------|-------------|-----|
| `phase-2/rwkv7-pretrained` | RWKV-7 + frozen CLIP → fine-tuned CLIP (primary path) | GPU 1 |
| `phase-2/rwkv7-scratch` | RWKV-7 + from-scratch PatchEmbedding (concurrent) | GPU 2 |
| `phase-2/rwkv6-fallback` | RWKV-6 fallback if RWKV-7 blocks after 1 week | Created only if needed |

- All branches share the same data pipeline, configs, and evaluation code from `main`
- Merge whichever branch produces best closed-loop results into `main` at end of Phase II
- Unmerged branches remain available for future comparison or resumption

---

## Steps

### 2.1 — Patch Embedding & Q-Shift Tokenizer (Week 5)

**Objective:** Convert 224×224 RGB images into token sequences with spatial awareness.

#### Vision Tokenization Strategy (3 Stages)

All three stages are required — they are run sequentially (or concurrently on separate branches/GPUs):

**Stage 1 — Frozen pretrained encoder (initial validation):**
- Use pretrained CLIP ViT-B/16 or DINOv2 as frozen patch feature extractor
- This validates the RWKV + LICA + PolicyHead architecture works, independent of vision encoder quality
- Fastest to converge; provides a performance baseline for the vision encoder itself

**Stage 2 — Fine-tuned pretrained encoder:**
- Unfreeze the pretrained encoder and fine-tune end-to-end on CARLA data
- Lower LR for encoder (1e-5) vs RWKV head (1e-4) to avoid destroying pretrained features
- Expected to outperform Stage 1 by learning CARLA-specific features (traffic lights, lane markers)

**Stage 3 — From-scratch PatchEmbedding:**
- Replace pretrained encoder with a single `Conv2d(3, D, kernel=16, stride=16)` trained from scratch
- Requires more data (50+ hours recommended) and longer training
- **Preparation during Stage 2:** While Stage 2 trains on GPU 1, simultaneously on a separate branch: (1) implement `PatchEmbedding` module, (2) ensure sufficient CARLA data is collected (extend collection if <50 hours), (3) set up augmentation pipeline more aggressively (heavier color jitter, random erasing), (4) write comparison evaluation script

**Tasks:**
- [ ] Implement `PatchEmbedding`: Conv2d(3, D, kernel=16, stride=16) → (B, 196, D) where D=768
- [ ] Implement pretrained encoder wrapper: load CLIP ViT-B/16 or DINOv2, extract patch tokens
- [ ] Implement `QShift` (Quad-directional Token Shift): shift tokens by 1 position in 4 spatial directions to inject 2D locality
- [ ] Add positional encoding (learnable 2D positional embeddings for 14×14 grid)
- [ ] Config flag: `use_pretrained_vision: true/false` to switch between stages
- [ ] Unit test: verify output shapes, gradient flow for both modes
- [ ] Reference implementation: [OpenGVLab/Vision-RWKV](https://github.com/OpenGVLab/Vision-RWKV) (ICLR 2025)

**Key code artifact:** `src/models/tokenizer.py`

**Q-Shift explained:** After flattening the 14×14 patch grid into a 196-token sequence, Q-Shift splits the channel dimension into 4 groups. Each group is shifted by 1 position corresponding to up/down/left/right spatial neighbors. This recovers local 2D structure that raster-scan flattening destroys, without adding any parameters.

### 2.2 — RWKV-7 Encoder Block (Week 5)

**Objective:** Implement the core RWKV-7 time-mixing and channel-mixing blocks.

#### Attempt-First-Then-Pivot Plan

**Week 1 attempt (RWKV-7):**
- Implement data-dependent decay using official CUDA kernels from `rwkv==0.8.32` (enable via `os.environ["RWKV_V7_ON"] = "1"`)
- Reference training code: `BlinkDL/RWKV-LM/RWKV-v7/train_temp/`
- Sub-tasks: (1) integrate WKV-7 operator, (2) implement state transition with data-dependent `w_t`, (3) test gradient flow through time, (4) benchmark memory vs scalar decay

**Pivot criteria (after 1 week):**
- If CUDA kernel integration fails, gradient instability persists, or RWKV-7 doesn't converge → create `phase-2/rwkv6-fallback` branch
- RWKV-6 uses scalar decay (simpler, well-documented, battle-tested), proceed with full training
- Treat RWKV-6 → 7 comparison as a Phase III ablation study
- RWKV-7 branch is preserved for future resumption

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

### 2.3 — Sensor Fusion (NavigationFusion) & Policy Head (Week 5–6)

**Objective:** Fuse visual features with navigation commands, output driving controls.

**Tasks:**
- [ ] Implement `NavigationFusion` (gated linear fusion): fuses visual features with nav command embedding
  - Nav command → learned embedding (dim 64)
  - Gated linear fusion: concatenate visual + nav embeddings, linear projection with learned sigmoid gate
- [ ] Implement `PolicyHead`: MLP that outputs [steer, acceleration]
  - Architecture: Linear(D, 256) → ReLU → Linear(256, 64) → ReLU → Linear(64, 2)
  - Steering: tanh activation (range [-1, 1])
  - Acceleration: tanh activation (range [-1, 1]), where positive = throttle, negative = brake
  - This merged axis avoids the throttle-brake conflict in expert data and follows standard practice in imitation learning literature
- [ ] Optional: velocity prediction auxiliary head for improved representation learning

**Key code artifacts:** `src/models/fusion.py`, `src/models/policy_head.py`

### 2.4 — Full Model Assembly (Week 6)

**Objective:** Wire all components into a single end-to-end model.

**Tasks:**
- [ ] Implement `VisionRWKVDriver` that composes: PatchEmbedding → QShift → RWKVEncoder → NavigationFusion → PolicyHead
- [ ] Verify forward pass end-to-end: input (B, T, 3, 224, 224) → output (B, T, 2) where output is [steer, acceleration]
- [ ] Implement model config via YAML (see `configs/`)
- [ ] Config flag: `use_pretrained_vision` controls Stage 1/2 vs Stage 3 tokenizer

**Testing note:** The existing `tests/test_model_interfaces.py` currently only verifies that `forward()` raises `NotImplementedError` (stub assertions). As each module is implemented, **rewrite these tests** to verify:
- Output shapes match expected `(B, T, D)` or `(B, T, 2)` dimensions
- Gradient flow (`.backward()` succeeds, `.grad` is not None on parameters)
- State propagation (RWKV hidden state shapes, persistence across calls)
- Numerical behavior (outputs bounded by `tanh`, no NaN/Inf)

**Key code artifact:** `src/models/vision_rwkv_driver.py`

```python
class VisionRWKVDriver(nn.Module):
    def __init__(self, config):
        # config: d_model, n_layers, n_heads, img_size, patch_size, use_pretrained_vision, ...

    def forward(self, images, nav_commands, state=None):
        """
        images: (B, T, 3, H, W)
        nav_commands: (B, T) int
        state: optional recurrent state
        Returns: controls (B, T, 2), new_state
        """
```

### 2.4b — Memory Profiling (Week 6)

**Objective:** Validate batch sizes fit in VRAM before committing to full training.

**Tasks:**
- [ ] Implement `src/training/profile_memory.py`: run forward + backward on dummy data, report peak VRAM
- [ ] Test batch sizes [4, 8, 16, 32] × sequence lengths [16, 32] on target GPU
- [ ] Identify maximum safe batch size with 4 GB headroom (to account for allocation fragmentation)
- [ ] Profile both pretrained and from-scratch encoder modes (different memory profiles)
- [ ] Record results in W&B for reference

**This step prevents wasted GPU hours on OOM configurations.** Run before committing to any multi-day training run.

### 2.5 — Training Loop with TBPTT (Week 6)

**Objective:** Implement training with Truncated Backpropagation Through Time, burn-in, and crash-safe checkpointing.

**Important data notes for training:**
- **Collision episodes:** `DrivingSequenceDataset` already truncates at `collision_frame` metadata
  to exclude post-collision garbage frames. Verify this works correctly before training.
- **Town05 speed range:** Town05 has highway-like roads where the autopilot reaches 29 m/s (104 km/h).
  This is valid data; the model should learn to handle highway-speed driving.
  Consider normalizing velocity inputs if the distribution is very wide.
- **Aggressive traffic (deferred):** A future collection campaign could spawn erratic NPC vehicles
  (varying `ignore_lights_percentage`, `ignore_signs_percentage` in TrafficManager) to expose
  the model to adversarial driving conditions. Not required for initial training.

**Tasks:**
- [ ] Implement `Trainer` class with TBPTT
- [ ] Burn-in: first K=20 frames of each clip run forward-only (no gradient) to warm up state
- [ ] Loss: MSE on steering + acceleration predictions (after burn-in frames only)\n  - Training target: `acceleration = expert_throttle - expert_brake`
- [ ] Optimizer: AdamW, lr=1e-4, weight_decay=0.01
- [ ] LR schedule: linear warmup (1000 steps) + cosine decay
- [ ] Gradient clipping: max_norm=1.0
- [ ] Logging: W&B (primary) + TensorBoard (local fallback)
- [ ] Mixed precision (FP16) for faster training on workstation GPU

**Key code artifact:** `src/training/trainer.py`

**TBPTT burn-in explained:**
```
Frame:  [1  2  3  ... 20  21  22  ... 32]
         ←— burn-in —→  ←— loss computed —→
         (state warmup)   (gradients flow)
```

This prevents the model from being penalized on the first frames of a clip where the recurrent state is cold (zero-initialized at clip boundaries).

#### Training Crash Recovery

Training runs may be interrupted by OOM, hardware issues, or power loss. The checkpoint system must handle this:

- [ ] **Checkpoint every 500 steps** (not just epochs): save model weights, optimizer state, LR scheduler state, epoch counter, step counter, and best val loss
- [ ] `--resume` CLI flag: reload from latest checkpoint and continue training exactly where it left off
- [ ] Log checkpoint metadata to W&B for traceability (step, epoch, val loss at save time)
- [ ] On workstations: save checkpoints to a network home directory (survives local disk issues)

#### W&B Integration Convention

- Project: `neuromorphic-vision-rwkv`, entity: your personal W&B username
- Run naming convention: `{model_name}_{timestamp}` (e.g., `vrwkv_base_20260301_143022`)
- Log: loss components (steer MSE + acceleration MSE), gradient norms, learning rate, epoch, val predictions vs. ground truth
- Tags: `[rwkv-7, carla, imitation-learning]` (update tag to `rwkv-6` if pivoting)

### 2.5b — Steering Noise Injection (During Data Collection)

**Objective:** Mitigate distribution drift without requiring a closed-loop evaluation pipeline (which is built in Phase III).

**Rationale:** Full DAgger requires running the trained agent in CARLA and measuring deviations — this depends on the closed-loop evaluation pipeline (Phase III, Step 3.1). Instead, we use a simpler Phase 2 alternative: inject Gaussian noise into expert steering during data collection. This creates "slightly wrong" states that the expert then corrects from naturally, giving the model implicit recovery demonstrations without needing a running agent.

**Tasks:**
- [ ] Add `steering_noise_std` parameter to `EpisodeConfig` (default σ=0.05)
- [ ] During data collection, apply noise to the recorded steering signal: `steering_noisy = steering + N(0, σ)`, clamped to [-1, 1]
- [ ] Store the *noisy* steering as the "observation" and the *clean* expert steering as the "label" — the model learns to correct back toward the expert action
- [ ] Collect ~20% of episodes with noise enabled, 80% clean
- [ ] Validate that noise magnitude doesn't cause unsafe driving (σ=0.05 ≈ 3° perturbation)

**Key code artifact:** `src/data/expert_collector.py` (noise injection in `collect_episode`)

**Why this works:** Noise injection is a well-established technique in imitation learning (Ross & Bagnell, 2010). It exposes the model to off-policy states without the engineering overhead of DAgger's iterative data collection loop. Full DAgger can be applied in Phase III once closed-loop evaluation exists.

> **Note:** Full DAgger recovery training is deferred to Phase III (Step 3.7) where the closed-loop evaluation pipeline provides the required deviation detection.

### 2.6 — Training Run (Week 6–7)

**Objective:** Train to convergence on the expert dataset.

**Tasks:**
- [ ] Run initial training on laptop (small batch, 1-2 epochs) to verify pipeline correctness
- [ ] Transfer to the 24 GB workstation for full training
- [ ] Train for 50+ epochs with early stopping (patience=10 on val loss)
- [ ] Monitor: loss curves, steering/acceleration prediction distributions, gradient norms
- [ ] Batch size: 16-32 sequences (each 32 frames) — adjust based on memory profiling results (Step 2.4b)
- [ ] Expected training time: ~30-50 GPU-hours on workstation GPU (24 GB)

#### Seed Management

- [ ] Fixed seed convention: `SEED=42` in all config YAMLs (vary for ablation runs)
- [ ] Set all RNG sources on startup:
  ```python
  torch.manual_seed(seed)
  numpy.random.seed(seed)
  random.seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  ```
- [ ] When running ablation studies, document the seed used in each W&B run

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
  d_model: 768           # Explicit: code default is 384 (lightweight for dev/test)
  n_layers: 12
  patch_size: 16
  img_size: 224
  nav_embed_dim: 64
  use_pretrained_vision: true  # Stage 1/2
  vision_encoder: "openai/clip-vit-base-patch16"

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
  checkpoint_every_steps: 500
  seed: 42

data:
  dataset_dir: "./data/collected"  # Or set $DATA_DIR for lab NFS
  train_split: 0.8
  augmentation: true

wandb:
  project: "neuromorphic-vision-rwkv"
  tags: ["rwkv-7", "carla", "imitation-learning"]
```

```yaml
# configs/vrwkv_debug.yaml (for rapid iteration)
model:
  name: "VisionRWKVDriver"
  d_model: 256
  n_layers: 6
  patch_size: 16
  img_size: 224
  nav_embed_dim: 32
  use_pretrained_vision: true

training:
  batch_size: 4
  seq_len: 16
  burn_in: 5
  lr: 1e-3
  max_epochs: 10
  grad_clip: 1.0
  fp16: true
  checkpoint_every_steps: 100
  seed: 42

data:
  dataset_dir: "./data/toy_dataset"
  train_split: 0.8
  augmentation: false
```

---

## Completion Criteria

All of the following must be true before starting Phase III:

1. Vision-RWKV model trains without errors for 50+ epochs
2. Training loss converges (not diverging, not flat)
3. Val loss shows improvement and early stopping hasn't triggered too early
4. All 3 baselines trained with same protocol
5. Checkpointing implemented and tested for model resumption
6. Steering noise injection implemented and validated (σ=0.05, ~20% of episodes)
7. Memory profiling completed — batch size validated for target GPU
8. Training curves logged in W&B and visually inspected
9. Forward pass verified: (B, T, 3, 224, 224) → (B, T, 2) where output = [steer, acceleration ∈ [-1, 1]]
10. At least 2 vision tokenization stages compared (pretrained vs from-scratch)
