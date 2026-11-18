#!/usr/bin/env python3
"""Verify the rwkv environment and print a diagnostic report.

Run with:  python scripts/verify_env.py
Run in env: conda run -n rwkv python scripts/verify_env.py
"""

from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from collections.abc import Callable


def _check(label: str, fn: Callable[[], str]) -> tuple[bool, str]:
    try:
        result = fn()
        return True, result
    except Exception as e:
        return False, str(e)


def main() -> None:
    ok = "\033[32m✓\033[0m"
    fail = "\033[31m✗\033[0m"
    warn = "\033[33m!\033[0m"

    print(f"\n{'=' * 60}")
    print("  Environment Verification Report")
    print(f"{'=' * 60}\n")

    # ── System ────────────────────────────────────────────────────
    print("System")
    print(f"  Python : {sys.version.split()[0]}")
    print(f"  Platform: {platform.platform()}")
    print()

    # ── CUDA / GPU ────────────────────────────────────────────────
    print("GPU / CUDA")
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        symbol = ok if cuda_ok else warn
        print(f"  {symbol} torch          : {torch.__version__}")
        print(f"  {'  '} cuda available : {cuda_ok}")
        if cuda_ok:
            print(f"       cuda version  : {torch.version.cuda}")
            print(f"       device        : {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"       VRAM          : {vram:.1f} GB")
        else:
            mps_ok = torch.backends.mps.is_available()
            print(f"  {'  '} mps available  : {mps_ok}")
            if not mps_ok:
                print(f"  {warn} No CUDA or MPS — training will run on CPU only")
    except ImportError as e:
        print(f"  {fail} torch not installed: {e}")
    print()

    # ── Critical packages ─────────────────────────────────────────
    print("Critical packages")
    critical = [
        ("h5py", "h5py"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("tqdm", "tqdm"),
        ("yaml", "pyyaml"),
        ("tensorboard", "tensorboard"),
        ("optuna", "optuna"),
        ("einops", "einops"),
        ("scipy", "scipy"),
        ("networkx", "networkx"),
        ("zstandard", "zstandard"),
    ]
    all_ok = True
    for import_name, pip_name in critical:

        def _version_fn(n: str = import_name) -> str:
            return str(importlib.import_module(n).__version__)

        passed, msg = _check(pip_name, _version_fn)
        symbol = ok if passed else fail
        if not passed:
            all_ok = False
        print(f"  {symbol} {pip_name:<20} {msg}")
    print()

    # ── Project import ────────────────────────────────────────────
    print("Project")
    passed, msg = _check(
        "src.models",
        lambda: __import__("src.models", fromlist=["VisionRWKVDriver"]).VisionRWKVDriver.__module__,
    )
    symbol = ok if passed else fail
    if not passed:
        all_ok = False
    print(f"  {symbol} src.models (VisionRWKVDriver) : {msg}")

    passed, msg = _check(
        "src.training",
        lambda: __import__("src.training.trainer", fromlist=["Trainer"]).Trainer.__module__,
    )
    symbol = ok if passed else fail
    if not passed:
        all_ok = False
    print(f"  {symbol} src.training.trainer          : {msg}")
    print()

    # ── Quick smoke test ─────────────────────────────────────────
    print("Smoke test (forward pass)")
    try:
        import torch
        from src.models import ModelConfig, VisionRWKVDriver
        from src.utils.device import select_device

        from src.models.config import (
            RWKVEncoderConfig,
            TokenizerConfig,
            FusionConfig,
            PolicyHeadConfig,
        )

        D = 64
        H, W = 32, 192  # multiples of the backbone's stride of 32
        cfg = ModelConfig(
            tokenizer=TokenizerConfig(img_height=H, img_width=W, embed_dim=D),
            encoder=RWKVEncoderConfig(embed_dim=D, n_layers=2, ffn_dim=256),
            fusion=FusionConfig(visual_dim=D, fused_dim=D),
            policy_head=PolicyHeadConfig(input_dim=D, hidden_dims=(32,)),
            seq_len=4,
            burn_in=1,
            batch_size=2,
        )

        device = select_device("auto")
        model = VisionRWKVDriver(cfg).to(device)

        imgs = torch.randn(2, 4, 3, H, W, device=device)
        nav = torch.zeros(2, 4, dtype=torch.long, device=device)
        out, _, _ = model(imgs, nav, torch.zeros(2, 4, device=device))
        assert out.shape == (2, 4, 2), f"unexpected shape {out.shape}"
        print(f"  {ok} forward pass on {device} — output shape {tuple(out.shape)}")
    except Exception as e:
        print(f"  {fail} forward pass failed: {e}")
        all_ok = False
    print()

    # ── nvidia-smi ───────────────────────────────────────────────
    print("nvidia-smi")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.strip().splitlines():
            print(f"  {ok} {line.strip()}")
    except Exception as e:
        print(f"  {warn} nvidia-smi unavailable: {e}")
    print()

    print("=" * 60)
    if all_ok:
        print(f"  {ok}  All checks passed")
    else:
        print(f"  {fail}  Some checks failed — see above")
    print("=" * 60 + "\n")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
