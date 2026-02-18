"""Data collection campaign runner.

Orchestrates multi-episode data collection across towns, weathers,
and traffic densities. Supports campaign resumption via manifest.

Usage:
    python scripts/collect_data.py --config configs/collection.yaml
    python scripts/collect_data.py --town Town01 --weather ClearNoon --episodes 10
    python scripts/collect_data.py --resume   # Continue from manifest
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.expert_collector import (  # noqa: E402
    CollectionManifest,
    EpisodeConfig,
    ExpertDataCollector,
)
from src.data.route_generator import BalancedRouteGenerator, RouteConfig  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML collection config."""
    with open(config_path) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def build_campaign(
    cfg: dict,
    completed: set[tuple[str, str]],
) -> list[EpisodeConfig]:
    """Build list of episode configs, skipping completed (town, weather) combos."""
    episodes = []
    towns = cfg.get("towns", ["Town01"])
    weathers = cfg.get("weathers", ["ClearNoon"])
    traffic_configs = cfg.get("traffic", {}).get("vehicles", [50])

    for town, weather in itertools.product(towns, weathers):
        if (town, weather) in completed:
            logger.info("Skipping %s / %s (already completed)", town, weather)
            continue

        for n_vehicles in traffic_configs:
            ep = EpisodeConfig(
                town=town,
                weather=weather,
                n_vehicles=n_vehicles,
                n_walkers=cfg.get("traffic", {}).get("walkers", [30])[0],
                max_frames=cfg.get("episode", {}).get("max_frames", 2000),
                timeout_speed=cfg.get("episode", {}).get("timeout_speed", 0.1),
                timeout_duration=cfg.get("episode", {}).get("timeout_duration", 180),
                fps=cfg.get("carla", {}).get("fps", 20),
                camera_width=cfg.get("camera", {}).get("width", 224),
                camera_height=cfg.get("camera", {}).get("height", 224),
                camera_fov=cfg.get("camera", {}).get("fov", 100),
            )
            episodes.append(ep)

    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data collection campaign")
    parser.add_argument("--config", type=str, default="configs/collection.yaml")
    parser.add_argument("--output", type=str, default=None, help="Override output dir")
    parser.add_argument("--town", type=str, nargs="+", default=None, help="Specific town(s)")
    parser.add_argument("--weather", type=str, nargs="+", default=None, help="Specific weather(s)")
    parser.add_argument("--episodes", type=int, default=None, help="Episodes per config")
    parser.add_argument("--resume", action="store_true", help="Resume from manifest")
    parser.add_argument("--host", type=str, default="auto", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--route-strategy",
        type=str,
        default="balanced",
        choices=["random", "balanced"],
    )
    parser.add_argument("--target-left", type=float, default=0.15)
    parser.add_argument("--target-right", type=float, default=0.15)
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.town:
        cfg["towns"] = args.town
    if args.weather:
        cfg["weathers"] = args.weather

    output_dir = args.output or cfg.get("output", {}).get("directory", "data/collected")
    manifest_name = cfg.get("output", {}).get("manifest", "collection_manifest.json")
    manifest_path = Path(output_dir) / manifest_name

    # Initialize
    manifest = CollectionManifest(manifest_path)
    completed = manifest.get_completed_configs() if args.resume else set()

    campaign = build_campaign(cfg, completed)
    n_episodes = args.episodes or 1  # Episodes per config

    total_episodes = len(campaign) * n_episodes
    logger.info(
        "Campaign: %d configs × %d episodes = %d total", len(campaign), n_episodes, total_episodes
    )

    if args.resume:
        logger.info("Resuming from manifest:\n%s", manifest.summary())

    # Setup collector
    rng = np.random.default_rng(args.seed)
    collector = ExpertDataCollector(
        host=args.host,
        port=args.port,
        output_dir=output_dir,
    )
    collector.connect()

    # Route config
    route_cfg = RouteConfig(
        strategy=args.route_strategy,
        target_left=args.target_left,
        target_right=args.target_right,
    )

    # Run campaign
    episode_num = 0
    for config in campaign:
        # Setup route generator for this town
        try:
            route_gen = BalancedRouteGenerator(collector.world, route_cfg)
        except Exception as e:
            logger.error("Failed to init route generator for %s: %s", config.town, e)
            route_gen = None

        for ep_idx in range(n_episodes):
            episode_num += 1
            logger.info(
                "=== Episode %d/%d: %s / %s (traffic=%d) ===",
                episode_num,
                total_episodes,
                config.town,
                config.weather,
                config.n_vehicles,
            )

            # Generate route
            route = None
            if route_gen is not None:
                try:
                    route_data = route_gen.generate_route(rng)
                    if route_data:
                        route = route_data
                        dist = route_gen.get_distribution_report(route_data)
                        logger.info("Route: %d waypoints, distribution: %s", len(route_data), dist)
                except Exception as e:
                    logger.warning("Route generation failed, using random: %s", e)

            # Collect
            result = collector.collect_episode(config, route_waypoints=route)
            logger.info(
                "Result: status=%s, frames=%d, duration=%.1fs",
                result.status,
                result.n_frames,
                result.duration_seconds,
            )

            # Update manifest
            manifest.add_episode(result)
            manifest.save()

            # Handle crash → reconnect
            if result.status == "crashed" and result.error_message:
                logger.warning("Episode crashed, attempting reconnect...")
                if not collector._reconnect():
                    logger.error("Failed to reconnect after crash. Aborting.")
                    sys.exit(1)
                # Re-initialize route generator after reconnect
                try:
                    route_gen = BalancedRouteGenerator(collector.world, route_cfg)
                except Exception:
                    route_gen = None

    # Final summary
    logger.info("=" * 60)
    logger.info("Campaign complete!")
    logger.info(manifest.summary())


if __name__ == "__main__":
    main()
