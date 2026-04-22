#!/usr/bin/env python3
"""Export a dynamic-int8 quantized model from a training checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from src.models import VisionRWKVDriver
from src.training.config import load_training_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training config YAML used for the model")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Output path for quantized checkpoint")
    args = parser.parse_args()

    cfg = load_training_config(args.config)
    model = VisionRWKVDriver(cfg.model)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    model.eval()

    quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": quantized.state_dict(), "model_config": asdict(cfg.model)}, out_path)
    print(f"[quant] wrote {out_path}")


if __name__ == "__main__":
    main()
