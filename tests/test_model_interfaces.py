"""Tests for model interface stubs.

Verifies that:
- All model modules instantiate correctly with default configs.
- Config validation catches dimensional mismatches.
- Forward methods raise NotImplementedError (stubs, not yet implemented).
- initial_state returns correct shapes.
- __all__ exports resolve.
"""

from __future__ import annotations

import pytest
import torch

from src.models.config import (
    FusionConfig,
    ModelConfig,
    PolicyHeadConfig,
    RWKVEncoderConfig,
    TokenizerConfig,
)
from src.models.fusion import NavigationFusion
from src.models.policy_head import PolicyHead
from src.models.rwkv_encoder import RWKVBlock, RWKVEncoder
from src.models.tokenizer import PatchEmbedding, QShift, VisionTokenizer
from src.models.vision_rwkv_driver import VisionRWKVDriver

# ── Config tests ──────────────────────────────────────────────────────


class TestModelConfig:
    """Tests for model configuration dataclasses."""

    def test_default_config_valid(self) -> None:
        """Default ModelConfig should pass all assertions."""
        config = ModelConfig()
        assert config.tokenizer.embed_dim == config.encoder.embed_dim
        assert config.encoder.embed_dim == config.fusion.visual_dim
        assert config.fusion.fused_dim == config.policy_head.input_dim

    def test_config_embed_dim_mismatch(self) -> None:
        """Mismatch between tokenizer and encoder embed_dim should raise."""
        with pytest.raises(AssertionError, match="Tokenizer embed_dim"):
            ModelConfig(
                tokenizer=TokenizerConfig(embed_dim=256),
                encoder=RWKVEncoderConfig(embed_dim=384),
            )

    def test_config_fusion_dim_mismatch(self) -> None:
        """Mismatch between encoder and fusion visual_dim should raise."""
        with pytest.raises(AssertionError, match="Encoder embed_dim"):
            ModelConfig(
                tokenizer=TokenizerConfig(embed_dim=384),
                encoder=RWKVEncoderConfig(embed_dim=384),
                fusion=FusionConfig(visual_dim=256),
            )

    def test_config_policy_dim_mismatch(self) -> None:
        """Mismatch between fusion fused_dim and policy input_dim should raise."""
        with pytest.raises(AssertionError, match="Fusion fused_dim"):
            ModelConfig(
                tokenizer=TokenizerConfig(embed_dim=384),
                encoder=RWKVEncoderConfig(embed_dim=384),
                fusion=FusionConfig(visual_dim=384, fused_dim=384),
                policy_head=PolicyHeadConfig(input_dim=256),
            )

    def test_config_burn_in_exceeds_seq_len(self) -> None:
        """burn_in >= seq_len should raise."""
        with pytest.raises(AssertionError, match="burn_in"):
            ModelConfig(seq_len=32, burn_in=32)

    def test_config_custom_values(self) -> None:
        """Custom config with consistent dimensions should pass."""
        config = ModelConfig(
            tokenizer=TokenizerConfig(embed_dim=192),
            encoder=RWKVEncoderConfig(embed_dim=192, n_layers=6),
            fusion=FusionConfig(visual_dim=192, fused_dim=192),
            policy_head=PolicyHeadConfig(input_dim=192),
        )
        assert config.encoder.n_layers == 6
        assert config.tokenizer.embed_dim == 192


# ── Tokenizer tests ──────────────────────────────────────────────────


class TestVisionTokenizer:
    """Tests for tokenizer module instantiation and interface."""

    def test_patch_embedding_instantiates(self) -> None:
        config = TokenizerConfig()
        module = PatchEmbedding(config)
        assert isinstance(module, torch.nn.Module)

    def test_qshift_instantiates(self) -> None:
        n_h = 320 // 16
        n_w = 1920 // 16
        module = QShift(embed_dim=384, n_patches_h=n_h, n_patches_w=n_w)
        assert isinstance(module, torch.nn.Module)

    def test_vision_tokenizer_instantiates(self) -> None:
        config = TokenizerConfig()
        module = VisionTokenizer(config)
        assert isinstance(module, torch.nn.Module)

    def test_patch_embedding_forward_raises(self) -> None:
        module = PatchEmbedding(TokenizerConfig())
        x = torch.randn(1, 4, 3, 320, 1920)
        with pytest.raises(NotImplementedError, match="PatchEmbedding"):
            module(x)

    def test_qshift_forward_raises(self) -> None:
        n_h = 320 // 16
        n_w = 1920 // 16
        module = QShift(embed_dim=384, n_patches_h=n_h, n_patches_w=n_w)
        n_patches = n_h * n_w
        x = torch.randn(1, 4, n_patches, 384)
        with pytest.raises(NotImplementedError, match="QShift"):
            module(x)

    def test_vision_tokenizer_forward_raises(self) -> None:
        module = VisionTokenizer(TokenizerConfig())
        x = torch.randn(1, 4, 3, 320, 1920)
        with pytest.raises(NotImplementedError, match="VisionTokenizer"):
            module(x)


# ── RWKV Encoder tests ───────────────────────────────────────────────


class TestRWKVEncoder:
    """Tests for RWKV encoder module instantiation and interface."""

    def test_rwkv_block_instantiates(self) -> None:
        config = RWKVEncoderConfig()
        module = RWKVBlock(config, layer_idx=0)
        assert isinstance(module, torch.nn.Module)

    def test_rwkv_encoder_instantiates(self) -> None:
        config = RWKVEncoderConfig()
        module = RWKVEncoder(config)
        assert isinstance(module, torch.nn.Module)

    def test_initial_state_shape(self) -> None:
        config = RWKVEncoderConfig(n_layers=4, embed_dim=128)
        encoder = RWKVEncoder(config)
        state = encoder.initial_state(batch_size=2, device=torch.device("cpu"))
        assert len(state) == 4
        assert state[0].shape == (2, 128)

    def test_encoder_forward_raises(self) -> None:
        config = RWKVEncoderConfig(n_layers=2, embed_dim=64)
        encoder = RWKVEncoder(config)
        x = torch.randn(1, 4, 196, 64)
        with pytest.raises(NotImplementedError, match="RWKVEncoder"):
            encoder(x)


# ── Fusion tests ──────────────────────────────────────────────────────


class TestNavigationFusion:
    """Tests for NavigationFusion module."""

    def test_fusion_instantiates(self) -> None:
        config = FusionConfig()
        module = NavigationFusion(config)
        assert isinstance(module, torch.nn.Module)

    def test_fusion_forward_raises(self) -> None:
        config = FusionConfig()
        module = NavigationFusion(config)
        visual = torch.randn(1, 4, config.visual_dim)
        nav = torch.zeros(1, 4, dtype=torch.long)
        with pytest.raises(NotImplementedError, match="NavigationFusion"):
            module(visual, nav)

    def test_fusion_nav_embedding_size(self) -> None:
        config = FusionConfig(n_nav_commands=5, nav_embed_dim=64)
        module = NavigationFusion(config)
        assert module.nav_embedding.num_embeddings == 5
        assert module.nav_embedding.embedding_dim == 64


# ── Policy Head tests ────────────────────────────────────────────────


class TestPolicyHead:
    """Tests for policy head module."""

    def test_policy_head_instantiates(self) -> None:
        config = PolicyHeadConfig()
        module = PolicyHead(config)
        assert isinstance(module, torch.nn.Module)

    def test_policy_head_forward_raises(self) -> None:
        config = PolicyHeadConfig()
        module = PolicyHead(config)
        x = torch.randn(1, 4, 768)
        with pytest.raises(NotImplementedError, match="PolicyHead"):
            module(x)

    def test_policy_head_output_size(self) -> None:
        config = PolicyHeadConfig(input_dim=64, hidden_dims=(32, 16), n_outputs=2)
        module = PolicyHead(config)
        # Check final layer before tanh is the Linear layer
        final_linear = module.mlp[-2]
        assert isinstance(final_linear, torch.nn.Linear)
        assert final_linear.out_features == 2
        # Check tanh activation is present
        assert isinstance(module.mlp[-1], torch.nn.Tanh)


# ── Top-level model tests ────────────────────────────────────────────


class TestVisionRWKVDriver:
    """Tests for the assembled end-to-end model."""

    def test_model_instantiates(self) -> None:
        config = ModelConfig()
        model = VisionRWKVDriver(config)
        assert isinstance(model, torch.nn.Module)

    def test_model_submodules_exist(self) -> None:
        model = VisionRWKVDriver(ModelConfig())
        assert hasattr(model, "tokenizer")
        assert hasattr(model, "encoder")
        assert hasattr(model, "fusion")
        assert hasattr(model, "policy_head")

    def test_model_forward_raises(self) -> None:
        model = VisionRWKVDriver(ModelConfig())
        images = torch.randn(1, 4, 3, 320, 1920)
        nav = torch.zeros(1, 4, dtype=torch.long)
        with pytest.raises(NotImplementedError, match="VisionRWKVDriver"):
            model(images, nav)

    def test_model_initial_state(self) -> None:
        config = ModelConfig(
            tokenizer=TokenizerConfig(embed_dim=384),
            encoder=RWKVEncoderConfig(n_layers=3, embed_dim=384),
            fusion=FusionConfig(visual_dim=384, fused_dim=384),
            policy_head=PolicyHeadConfig(input_dim=384),
        )
        model = VisionRWKVDriver(config)
        state = model.initial_state(batch_size=2, device=torch.device("cpu"))
        assert len(state) == 3
        assert state[0].shape == (2, 384)

    def test_model_parameter_count(self) -> None:
        """Smoke test: model should have >0 parameters."""
        model = VisionRWKVDriver(ModelConfig())
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0


# ── Package __init__ tests ───────────────────────────────────────────


class TestPackageExports:
    """Tests that the models package exports are valid."""

    def test_all_exports_importable(self) -> None:
        import src.models

        for name in src.models.__all__:
            assert hasattr(src.models, name), f"Missing export: {name}"

    def test_top_level_import(self) -> None:
        from src.models import ModelConfig, VisionRWKVDriver

        assert VisionRWKVDriver is not None
        assert ModelConfig is not None
