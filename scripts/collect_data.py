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
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.data.expert_collector import (
    CollectionManifest,
    EpisodeConfig,
    ExpertDataCollector,
)
from agents.expert.privileged_expert import ExpertConfig
from src.data.route_generator import BalancedRouteGenerator, RouteConfig

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
    completed_counts: dict[tuple[str, str], int],
    episodes_per_config: int,
) -> list[tuple[EpisodeConfig, int]]:
    """Build list of (episode_config, remaining_count) pairs, accounting for completed.

    Args:
        cfg: Collection config dict.
        completed_counts: Map of (town, weather) -> number of completed episodes.
        episodes_per_config: Total desired episodes per (town, weather) combo.

    Returns:
        List of (EpisodeConfig, n_remaining) where n_remaining > 0.
    """
    episodes = []
    towns = cfg.get("towns", ["Town01"])
    weathers = cfg.get("weathers", ["ClearNoon"])
    traffic_configs = cfg.get("traffic", {}).get("vehicles", [50])

    for town, weather in itertools.product(towns, weathers):
        already_done = completed_counts.get((town, weather), 0)
        remaining = episodes_per_config - already_done
        if remaining <= 0:
            logger.info(
                "Skipping %s / %s (%d/%d already completed)",
                town,
                weather,
                already_done,
                episodes_per_config,
            )
            continue

        for n_vehicles in traffic_configs:
            cam_cfg = cfg.get("cameras", {})

            # Parse expert agent config from YAML (falls back to defaults)
            expert_cfg_dict = cfg.get("expert", {})
            expert_kw: dict[str, Any] = {}
            if "target_speed" in expert_cfg_dict:
                expert_kw["target_speed"] = float(expert_cfg_dict["target_speed"])
            if "max_speed" in expert_cfg_dict:
                expert_kw["max_speed"] = float(expert_cfg_dict["max_speed"])
            if "speed_pid" in expert_cfg_dict:
                kp, ki, kd = expert_cfg_dict["speed_pid"]
                expert_kw.update(speed_kp=float(kp), speed_ki=float(ki), speed_kd=float(kd))
            if "steer_pid" in expert_cfg_dict:
                kp, ki, kd = expert_cfg_dict["steer_pid"]
                expert_kw.update(steer_kp=float(kp), steer_ki=float(ki), steer_kd=float(kd))
            if "traffic_light_brake_distance" in expert_cfg_dict:
                expert_kw["traffic_light_brake_distance"] = float(
                    expert_cfg_dict["traffic_light_brake_distance"]
                )
            if "vehicle_hazard_distance" in expert_cfg_dict:
                expert_kw["vehicle_hazard_distance"] = float(
                    expert_cfg_dict["vehicle_hazard_distance"]
                )
            if "pedestrian_hazard_distance" in expert_cfg_dict:
                expert_kw["pedestrian_hazard_distance"] = float(
                    expert_cfg_dict["pedestrian_hazard_distance"]
                )
            if "turn_speed_factor" in expert_cfg_dict:
                expert_kw["turn_speed_factor"] = float(expert_cfg_dict["turn_speed_factor"])
            expert_config = ExpertConfig(**expert_kw)

            ep = EpisodeConfig(
                town=town,
                weather=weather,
                n_vehicles=n_vehicles,
                n_walkers=cfg.get("traffic", {}).get("walkers", [30])[0],
                max_frames=cfg.get("episode", {}).get("max_frames", 2000),
                timeout_speed=cfg.get("episode", {}).get("timeout_speed", 0.1),
                timeout_duration=cfg.get("episode", {}).get("timeout_duration", 180),
                fps=cfg.get("carla", {}).get("fps", 20),
                camera_width=cam_cfg.get("per_camera_width", 640),
                camera_height=cam_cfg.get("per_camera_height", 320),
                camera_fov=cam_cfg.get("fov", 60),
                camera_yaws=tuple(cam_cfg.get("yaw_offsets", [-60, 0, 60])),
                expert_config=expert_config,
            )
            episodes.append((ep, remaining))

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
    parser.add_argument(
        "--spectator",
        action="store_true",
        help="Follow ego vehicle with CARLA spectator camera (real-time view)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record simulation with CARLA's built-in recorder for later replay",
    )
    parser.add_argument(
        "--debug-draw",
        action="store_true",
        help="Draw route, trajectory, and target in CARLA viewport",
    )
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
    completed_counts = manifest.get_completed_counts() if args.resume else {}

    n_episodes = args.episodes or 1  # Episodes per (town, weather, traffic) config
    campaign = build_campaign(cfg, completed_counts, n_episodes)

    # Apply visualization / recording flags to every episode config
    if args.spectator or args.record or args.debug_draw:
        patched = []
        for ep_cfg, remaining in campaign:
            ep_cfg.spectator_follow = args.spectator
            ep_cfg.record_simulation = args.record
            ep_cfg.debug_draw = args.debug_draw
            patched.append((ep_cfg, remaining))
        campaign = patched

    total_episodes = sum(remaining for _, remaining in campaign)
    logger.info(
        "Campaign: %d configs with %d total remaining episodes",
        len(campaign),
        total_episodes,
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
    current_town: str | None = None
    route_gen: BalancedRouteGenerator | None = None

    for config, n_remaining in campaign:
        # Ensure correct map is loaded BEFORE creating route generator
        if config.town != current_town:
            logger.info("Loading map: %s", config.town)
            assert collector.client is not None
            # Map loading can take 60-120s over WSL2; temporarily increase timeout
            collector.client.set_timeout(120.0)
            try:
                collector.client.load_world(config.town)
                time.sleep(5.0)  # Let the new map fully settle
                collector.world = collector.client.get_world()
            except RuntimeError as e:
                logger.error("Failed to load %s: %s — retrying once", config.town, e)
                time.sleep(10.0)
                try:
                    collector.client.load_world(config.town)
                    time.sleep(5.0)
                    collector.world = collector.client.get_world()
                except RuntimeError as e2:
                    logger.error("Retry failed for %s: %s — skipping town", config.town, e2)
                    collector.client.set_timeout(60.0)
                    current_town = None
                    continue
            collector.client.set_timeout(60.0)
            current_town = config.town

            # (Re-)create route generator on the now-correct map
            try:
                route_gen = BalancedRouteGenerator(collector.world, route_cfg)
            except Exception as e:
                logger.error("Failed to init route generator for %s: %s", config.town, e)
                route_gen = None

        for ep_idx in range(n_remaining):
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
                # Force map reload + route_gen re-init on next iteration
                current_town = None

    # Final summary
    logger.info("=" * 60)
    logger.info("Campaign complete!")
    logger.info(manifest.summary())


if __name__ == "__main__":
    main()
