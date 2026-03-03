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
import os
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.diagnose_memory import log_memory_snapshot

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


def _run_orchestrator(args: argparse.Namespace) -> None:
    """Spawn one subprocess per town for memory isolation.

    CARLA's C++ client library leaks ~100 MB/s of anonymous memory
    after load_world() map changes.  Running each town in a separate
    process ensures the OS reclaims all C-level memory (including the
    leaked buffers) when the subprocess exits.

    Overhead: ~12 s per town for fresh CARLA connection + map load,
    negligible vs. hours of data collection per town.
    """
    cfg = load_config(args.config)
    if args.town:
        cfg["towns"] = args.town
    towns: list[str] = cfg.get("towns", ["Town01"])

    logger.info(
        "F2 memory isolation: spawning %d subprocess(es), one per town",
        len(towns),
    )

    # Build subprocess command: same CLI args + internal --_subprocess-town flag
    # Use resolved __file__ so the call works from any cwd.
    script = str(Path(__file__).resolve())
    base_argv = [a for a in sys.argv[1:] if not a.startswith("--_subprocess-town")]

    failed_towns: list[tuple[str, str]] = []
    for i, town in enumerate(towns, 1):
        logger.info("--- [%d/%d] Spawning worker for %s ---", i, len(towns), town)

        cmd = [sys.executable, script] + base_argv + ["--_subprocess-town", town]
        result = subprocess.run(cmd)

        if result.returncode == 0:
            logger.info("Town %s worker exited cleanly", town)
        elif result.returncode == -9:
            logger.error(
                "Town %s worker OOM-killed (SIGKILL). "
                "Leak may persist within-town for very long runs.",
                town,
            )
            failed_towns.append((town, "OOM-killed"))
        else:
            logger.error(
                "Town %s worker exited with code %d",
                town,
                result.returncode,
            )
            failed_towns.append((town, f"exit code {result.returncode}"))

    # Final summary (read manifest written by workers)
    output_dir = args.output or cfg.get("output", {}).get("directory", "data/collected")
    manifest_name = cfg.get("output", {}).get("manifest", "collection_manifest.json")
    manifest = CollectionManifest(Path(output_dir) / manifest_name)

    logger.info("=" * 60)
    logger.info("Campaign complete!")
    logger.info(manifest.summary())
    if failed_towns:
        logger.warning(
            "Failed towns: %s",
            ", ".join(f"{t} ({r})" for t, r in failed_towns),
        )


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
        "--max-frames",
        type=int,
        default=None,
        help="Override max frames per episode (default: from YAML config)",
    )
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
    parser.add_argument(
        "--tracemalloc",
        action="store_true",
        help="Enable tracemalloc to profile Python memory allocations",
    )
    parser.add_argument(
        "--memory-log",
        action="store_true",
        help="Log system memory at key points (map loads, episode boundaries)",
    )
    parser.add_argument(
        "--_subprocess-town",
        type=str,
        default=None,
        dest="_subprocess_town",
        help=argparse.SUPPRESS,  # Internal: per-town worker spawned by orchestrator
    )
    args = parser.parse_args()

    # --- F2: Per-town subprocess isolation (memory leak mitigation) ---
    # CARLA's C++ client library (libcarla_client.so) leaks ~100 MB/s of
    # anonymous memory after load_world() map changes.  By running each
    # town in a separate subprocess, the OS reclaims all leaked C-level
    # memory when the subprocess exits.
    if not args._subprocess_town:
        _run_orchestrator(args)
        return

    # ---- Below: per-town worker mode (one town per process) ----
    logger.info(
        "Worker process for %s (PID %d)",
        args._subprocess_town,
        os.getpid(),
    )

    # Load config
    cfg = load_config(args.config)

    # Force single town (subprocess isolation — ignore CLI --town)
    cfg["towns"] = [args._subprocess_town]
    if args.weather:
        cfg["weathers"] = args.weather
    if args.max_frames:
        cfg.setdefault("episode", {})["max_frames"] = args.max_frames

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

    # --- M13: tracemalloc profiling (opt-in) ---
    if args.tracemalloc:
        tracemalloc.start(10)  # 10 frames of call stack
        logger.info("tracemalloc started (10-frame stack depth)")

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

            # --- F1: Tear down all actors and flush C++ client state ---
            # CARLA's C++ client library retains sensor data buffers
            # after map changes, leaking ~100 MB/s.  Explicitly destroy
            # all actors and drop the world reference before load_world().
            if current_town is not None:
                collector.teardown_world()

            # --- Town03 / memory logging: snapshot before map load ---
            if args.memory_log:
                log_memory_snapshot(f"PRE load_world({config.town})")
            map_load_start = time.monotonic()

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

            map_load_elapsed = time.monotonic() - map_load_start
            logger.info("Map %s loaded in %.1fs", config.town, map_load_elapsed)

            # --- Town03 / memory logging: snapshot after map load ---
            if args.memory_log:
                log_memory_snapshot(f"POST load_world({config.town}) [{map_load_elapsed:.1f}s]")

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

            # --- Memory snapshot before episode ---
            if args.memory_log:
                log_memory_snapshot(f"PRE episode {episode_num} ({config.town}/{config.weather})")

            # Collect
            result = collector.collect_episode(config, route_waypoints=route)
            logger.info(
                "Result: status=%s, frames=%d, duration=%.1fs",
                result.status,
                result.n_frames,
                result.duration_seconds,
            )

            # --- Memory snapshot after episode ---
            if args.memory_log:
                log_memory_snapshot(
                    f"POST episode {episode_num} status={result.status} "
                    f"frames={result.n_frames}"
                )

            # --- M13: tracemalloc snapshot between episodes ---
            if args.tracemalloc:
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics("lineno")
                logger.info("tracemalloc top 10 after episode %d:", episode_num)
                for stat in top_stats[:10]:
                    logger.info("  %s", stat)

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

    # Worker complete — process exit reclaims all C-level memory
    logger.info(
        "Worker for %s finished (PID %d)",
        args._subprocess_town,
        os.getpid(),
    )


if __name__ == "__main__":
    main()
