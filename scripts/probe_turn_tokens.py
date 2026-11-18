"""Can the visual stem tell a left turn from a right turn in episodes it has never seen?

Trains only the pooling head and a linear classifier on backbone features of turning frames,
with mirrored copies so that neither direction is favoured, and scores held-out episodes. A
random backbone is the control. Only fp16 feature maps stay in memory, and the run stops at
--max-gib.

Usage:
    python scripts/probe_turn_tokens.py --episodes 250
"""

from __future__ import annotations

import argparse
import resource
import sys
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import labels, schema_adapter
from src.data.driving_dataset import FPS, IMAGENET_MEAN, IMAGENET_STD
from src.models.config import TokenizerConfig
from src.models.tokenizer import VisionTokenizer
from src.utils.device import select_device
from src.utils.settings import setting

CONFIG = TokenizerConfig(img_height=160, img_width=960, embed_dim=128)
LEFT, RIGHT = labels.LEFT, labels.RIGHT
TRAIN_SHARE = 0.8


def peak_gib(device: torch.device) -> float:
    """Peak memory of this process so far, with what Apple's GPU has allocated."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss *= 1 if sys.platform == "darwin" else 1024
    gpu = torch.mps.driver_allocated_memory() if device.type == "mps" else 0
    return (rss + gpu) / 2**30


def turn_frames(path: Path, per_episode: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Evenly spaced frames of an episode's commanded turns: uint8 images, and which go left."""
    with h5py.File(path, "r") as f:
        usable = schema_adapter.usable_frames(f, FPS)
        if usable < 32:
            return None
        lab = schema_adapter.read_labels(f, FPS)
        turning = lab.situation[:usable, labels.SITUATIONS.index("turning")]
        rows = np.flatnonzero(turning & np.isin(lab.command[:usable], (LEFT, RIGHT)))
        if rows.size == 0:
            return None
        rows = rows[np.linspace(0, rows.size - 1, min(per_episode, rows.size)).astype(int)]
        size = (CONFIG.img_height, CONFIG.img_width)
        images = [
            schema_adapter.read_window(f, int(r), int(r) + 1, 0, fps=FPS, size=size).images[0]
            for r in rows
        ]
    return np.stack(images), lab.command[rows] == LEFT


def balanced_accuracy(predicted: torch.Tensor, left: torch.Tensor) -> float:
    """The mean of the share of left turns and the share of right turns called correctly."""
    return 0.5 * float(predicted[left].float().mean() + (~predicted[~left]).float().mean())


def fit_head(
    maps: torch.Tensor,
    left: torch.Tensor,
    config: TokenizerConfig,
    device: torch.device,
    steps: int = 1500,
    seed: int = 0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Train a pooling head and classifier on feature maps; returns its left-turn predictor."""
    torch.manual_seed(seed)
    head = VisionTokenizer(config)
    head.backbone = nn.Identity()
    head.to(device)
    classifier = nn.Linear(config.embed_dim, 1).to(device)
    parameters = [*head.parameters(), *classifier.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=1e-2)
    for _ in range(steps):
        rows = torch.randint(len(maps), (64,))
        logits = classifier(head.pool_features(maps[rows].float().to(device))).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, left[rows].float().to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    head.eval()

    def predict(query: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            scores = [
                classifier(head.pool_features(part.float().to(device))).squeeze(1).cpu()
                for part in query.split(128)
            ]
        return torch.cat(scores) > 0

    return predict


def backbone(pretrained: bool, device: torch.device) -> nn.Module:
    torch.manual_seed(0)
    tokenizer = VisionTokenizer(CONFIG)
    if pretrained:
        tokenizer.load_pretrained()
    return tokenizer.backbone.to(device).eval()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None, help="episodes (default: DATA_DIR)")
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--per-episode", type=int, default=16)
    parser.add_argument("--min-accuracy", type=float, default=0.8)
    parser.add_argument("--max-gib", type=float, default=12.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)

    def check_memory() -> None:
        if peak_gib(device) > args.max_gib:
            raise SystemExit(f"Stopped: memory passed the limit of {args.max_gib} GiB")

    files = sorted(Path(args.data_dir or setting("DATA_DIR")).glob("episode_*.h5"))
    chosen = np.random.default_rng(0).permutation(len(files))[: args.episodes]
    nets = {"pretrained": backbone(True, device), "random": backbone(False, device)}
    plain: dict[str, list[torch.Tensor]] = {name: [] for name in nets}
    mirrored: dict[str, list[torch.Tensor]] = {name: [] for name in nets}
    directions: list[np.ndarray] = []
    for index in chosen:
        check_memory()
        frames = turn_frames(files[index], args.per_episode)
        if frames is None:
            continue
        images = torch.from_numpy(frames[0]).to(device).permute(0, 3, 1, 2).float() / 255
        images = (images - mean) / std
        with torch.no_grad():
            for name, net in nets.items():
                plain[name].append(net(images).cpu().half())
                mirrored[name].append(net(images.flip(-1)).cpu().half())
        directions.append(frames[1])

    held_out_from = int(len(directions) * TRAIN_SHARE)
    split = sum(len(d) for d in directions[:held_out_from])
    left = torch.from_numpy(np.concatenate(directions))
    print(
        f"{held_out_from} training and {len(directions) - held_out_from} held-out episodes, "
        f"{split} and {len(left) - split} frames"
    )

    scores: dict[str, float] = {}
    for name in nets:
        check_memory()
        maps, flipped = torch.cat(plain[name]), torch.cat(mirrored[name])
        train_maps = torch.cat([maps[:split], flipped[:split]])
        train_left = torch.cat([left[:split], ~left[:split]])
        predict = fit_head(train_maps, train_left, CONFIG, device)
        train = balanced_accuracy(predict(maps[:split]), left[:split])
        scores[name] = balanced_accuracy(predict(maps[split:]), left[split:])
        print(f"{name} backbone: training {train:.1%}, held-out {scores[name]:.1%}")

    passed = scores["pretrained"] >= args.min_accuracy and scores["pretrained"] > scores["random"]
    print(f"{'PASS' if passed else 'FAIL'}, peak memory {peak_gib(device):.1f} GiB")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
