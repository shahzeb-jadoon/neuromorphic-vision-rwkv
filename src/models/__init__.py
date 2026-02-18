"""Model package for Vision-RWKV autonomous driving.

Exports the top-level model and all sub-modules for convenient imports::

    from src.models import VisionRWKVDriver, ModelConfig
"""

from src.models.config import (
    FusionConfig,
    ModelConfig,
    PolicyHeadConfig,
    RWKVEncoderConfig,
    TokenizerConfig,
)
from src.models.fusion import LICA
from src.models.policy_head import PolicyHead
from src.models.rwkv_encoder import RWKVBlock, RWKVEncoder, RWKVState
from src.models.tokenizer import PatchEmbedding, QShift, VisionTokenizer
from src.models.vision_rwkv_driver import VisionRWKVDriver

__all__ = [
    # Top-level model
    "VisionRWKVDriver",
    # Sub-modules
    "VisionTokenizer",
    "PatchEmbedding",
    "QShift",
    "RWKVEncoder",
    "RWKVBlock",
    "RWKVState",
    "LICA",
    "PolicyHead",
    # Configs
    "ModelConfig",
    "TokenizerConfig",
    "RWKVEncoderConfig",
    "FusionConfig",
    "PolicyHeadConfig",
]
