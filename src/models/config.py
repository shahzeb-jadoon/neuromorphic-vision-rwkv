"""Model configuration dataclasses for Vision-RWKV driving agent.

All hyperparameters are centralized here so that experiment configs
can be serialized, logged to W&B, and reproduced deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TokenizerConfig:
    """Configuration for the vision tokenizer (pretrained CNN + attention pooling).

    The tokenizer converts raw image frames into one visual token per
    frame for the temporal core.

    Default image size matches the 3-camera panoramic setup:
        3 cameras × 640 px wide × 320 px tall → 1920 × 320 stitched.
    """

    img_height: int = 320  # multiple of 32, the backbone's stride
    img_width: int = 1920  # multiple of 32
    embed_dim: int = 768  # Dimension of each frame token (384 for debug/ablation)
    pretrained: bool = True  # start training from the ImageNet weights
    n_queries: int = 8  # learned queries that pool the feature map
    n_heads: int = 4  # attention heads of the pooling
    grad_frames: int = 4  # last frames of a clip whose gradients reach the backbone


@dataclass
class RWKVEncoderConfig:
    """Configuration for the temporal core.

    Processes sequences of visual tokens, maintaining a recurrent
    state for temporal context.
    """

    core: str = "rwkv"  # rwkv | gru
    embed_dim: int = 768
    n_layers: int = 12
    ffn_dim: int = 3072  # Feed-forward hidden dimension (4x embed)
    dropout: float = 0.1


@dataclass
class FusionConfig:
    """Configuration for the navigation fusion module (gated linear fusion).

    Fuses visual features with navigation command embeddings via
    concatenation, linear projection, and learned sigmoid gating.
    """

    visual_dim: int = 768
    nav_embed_dim: int = 64
    n_nav_commands: int = 6  # LEFT..CHANGELANERIGHT (remapped from RoadOption 1-6 → 0-5)
    fused_dim: int = 768
    dropout: float = 0.1


@dataclass
class PolicyHeadConfig:
    """Configuration for the policy (action) output head.

    Maps fused features to driving control signals.

    Design Decision — Merged Acceleration Axis:
        The policy head outputs 2 values: [steering, acceleration].
        Acceleration merges throttle and brake into a single axis:
            acceleration = throttle - brake  ∈ [-1, 1]
        where positive = throttle, negative = brake.

        This avoids the throttle-brake conflict in expert data (the
        CARLA autopilot sometimes applies both simultaneously during
        transitions) and follows standard practice in imitation
        learning literature. At inference time, the predicted
        acceleration is split back:
            throttle = max(0, acceleration)
            brake    = max(0, -acceleration)
    """

    input_dim: int = 768
    hidden_dims: tuple[int, ...] = (256, 64)  # 3-layer: D→256→64→2
    n_outputs: int = 2  # [steering, acceleration]
    dropout: float = 0.1


@dataclass
class ModelConfig:
    """Top-level model configuration composing all sub-configs.

    Usage:
        config = ModelConfig()  # All defaults
        config = ModelConfig(encoder=RWKVEncoderConfig(n_layers=6))  # Override encoder
    """

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    encoder: RWKVEncoderConfig = field(default_factory=RWKVEncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    policy_head: PolicyHeadConfig = field(default_factory=PolicyHeadConfig)

    # Training hyperparameters
    seq_len: int = 32  # TBPTT sequence length
    burn_in: int = 8  # Initial frames for state warm-up (no gradient)
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 8
    max_epochs: int = 100
    gradient_clip: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration consistency."""
        assert self.tokenizer.embed_dim == self.encoder.embed_dim, (
            f"Tokenizer embed_dim ({self.tokenizer.embed_dim}) must match "
            f"encoder embed_dim ({self.encoder.embed_dim})"
        )
        assert self.encoder.embed_dim == self.fusion.visual_dim, (
            f"Encoder embed_dim ({self.encoder.embed_dim}) must match "
            f"fusion visual_dim ({self.fusion.visual_dim})"
        )
        assert self.fusion.fused_dim == self.policy_head.input_dim, (
            f"Fusion fused_dim ({self.fusion.fused_dim}) must match "
            f"policy_head input_dim ({self.policy_head.input_dim})"
        )
        assert self.burn_in < self.seq_len, (
            f"burn_in ({self.burn_in}) must be less than seq_len ({self.seq_len})"
        )
