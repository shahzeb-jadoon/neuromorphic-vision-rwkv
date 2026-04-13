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
import hashlib
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


def _episode_config_hash(config: EpisodeConfig) -> str:
    """SHA-256 fingerprint of data-affecting EpisodeConfig fields.

    Hashes camera geometry, PID gains, expert hazard distances,
    and timing parameters.  Excludes runtime-only flags (spectator,
    debug_draw, rss_limit) so that re-runs with ``--spectator``
    don't trigger false hash-mismatch warnings.
    """
    ec = config.expert_config
    blob = (
        f"{config.camera_width}x{config.camera_height}"
        f"_fov{config.camera_fov}"
        f"_yaws{config.camera_yaws}"
        f"_fps{config.fps}"
        f"_maxf{config.max_frames}"
        f"_tspd{config.timeout_speed}_tdur{config.timeout_duration}"
        f"_ts{ec.target_speed}_ms{ec.max_speed}"
        f"_spid{ec.speed_kp},{ec.speed_ki},{ec.speed_kd}"
        f"_stpid{ec.steer_kp},{ec.steer_ki},{ec.steer_kd}"
        f"_tlbd{ec.traffic_light_brake_distance}"
        f"_vhd{ec.vehicle_hazard_distance}"
        f"_phd{ec.pedestrian_hazard_distance}"
        f"_tsf{ec.turn_speed_factor}"
        f"_sns{ec.steering_noise_sigma}"
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build_campaign(
    cfg: dict,
    manifest: "CollectionManifest",
    episodes_per_config: int,
) -> list[tuple[EpisodeConfig, int]]:
    """Build remaining (episode_config, 1) pairs using per-key resume.

    Uses ``manifest.get_completed_counts()`` keyed by
    ``(town, weather, n_vehicles, n_walkers)`` to skip only
    truly-completed episodes.  Also warns if the current config
    hash differs from hashes already stored in the manifest for
    the same scenario key (C-4 audit).

    Args:
        cfg: Collection config dict.
        manifest: Previously loaded CollectionManifest.
        episodes_per_config: Target episodes per traffic config.

    Returns:
        List of (EpisodeConfig, 1) for remaining episodes.
    """
    towns = cfg.get("towns", ["Town01"])
    weathers = cfg.get("weathers", ["ClearNoon"])
    traffic_configs = cfg.get("traffic", {}).get("vehicles", [50])
    walkers = cfg.get("traffic", {}).get("walkers", [30])

    completed_counts = manifest.get_completed_counts()

    # Build a single EpisodeConfig template per scenario —
    # duplicating the YAML → EpisodeConfig parsing only once.
    cam_cfg = cfg.get("cameras", {})
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
    if "traffic_light_stop_distance" in expert_cfg_dict:
        expert_kw["traffic_light_stop_distance"] = float(
            expert_cfg_dict["traffic_light_stop_distance"]
        )
    if "vehicle_hazard_distance" in expert_cfg_dict:
        expert_kw["vehicle_hazard_distance"] = float(expert_cfg_dict["vehicle_hazard_distance"])
    if "pedestrian_hazard_distance" in expert_cfg_dict:
        expert_kw["pedestrian_hazard_distance"] = float(
            expert_cfg_dict["pedestrian_hazard_distance"]
        )
    if "turn_speed_factor" in expert_cfg_dict:
        expert_kw["turn_speed_factor"] = float(expert_cfg_dict["turn_speed_factor"])
    if "steering_noise_sigma" in expert_cfg_dict:
        expert_kw["steering_noise_sigma"] = float(expert_cfg_dict["steering_noise_sigma"])
    if "stanley_cte_enabled" in expert_cfg_dict:
        expert_kw["stanley_cte_enabled"] = bool(expert_cfg_dict["stanley_cte_enabled"])
    if "cross_track_gain" in expert_cfg_dict:
        expert_kw["cross_track_gain"] = float(expert_cfg_dict["cross_track_gain"])
    if "turn_cross_track_gain" in expert_cfg_dict:
        expert_kw["turn_cross_track_gain"] = float(expert_cfg_dict["turn_cross_track_gain"])
    if "cross_track_deadband" in expert_cfg_dict:
        expert_kw["cross_track_deadband"] = float(expert_cfg_dict["cross_track_deadband"])
    if "cross_track_speed_floor" in expert_cfg_dict:
        expert_kw["cross_track_speed_floor"] = float(expert_cfg_dict["cross_track_speed_floor"])
    if "cross_track_steer_limit" in expert_cfg_dict:
        expert_kw["cross_track_steer_limit"] = float(expert_cfg_dict["cross_track_steer_limit"])
    expert_config = ExpertConfig(**expert_kw)

    ep_template = EpisodeConfig(
        town="",  # filled per-scenario
        weather="",
        n_vehicles=0,
        n_walkers=0,
        max_frames=cfg.get("episode", {}).get("max_frames", 2000),
        timeout_speed=cfg.get("episode", {}).get("timeout_speed", 0.1),
        timeout_duration=cfg.get("episode", {}).get("timeout_duration", 75),
        fps=cfg.get("carla", {}).get("fps", 20),
        camera_width=cam_cfg.get("per_camera_width", 640),
        camera_height=cam_cfg.get("per_camera_height", 320),
        camera_fov=cam_cfg.get("fov", 60),
        camera_yaws=tuple(cam_cfg.get("yaw_offsets", [-60, 0, 60])),
        expert_config=expert_config,
    )

    remaining: list[tuple[EpisodeConfig, int]] = []

    for town, weather in itertools.product(towns, weathers):
        for n_vehicles in traffic_configs:
            n_walkers_val = walkers[0] if walkers else 30
            key = (town, weather, n_vehicles, n_walkers_val)
            done = completed_counts.get(key, 0)
            needed = episodes_per_config - done

            if needed <= 0:
                logger.info(
                    "Skipping %s/%s (veh=%d, wal=%d): %d already completed",
                    town,
                    weather,
                    n_vehicles,
                    n_walkers_val,
                    done,
                )
                continue

            # C-4: warn if config changed since last run
            import dataclasses

            ep = dataclasses.replace(
                ep_template,
                town=town,
                weather=weather,
                n_vehicles=n_vehicles,
                n_walkers=n_walkers_val,
            )
            current_hash = _episode_config_hash(ep)
            existing_hashes = manifest.get_config_hashes(town, weather, n_vehicles, n_walkers_val)
            if existing_hashes and current_hash not in existing_hashes:
                logger.warning(
                    "Config hash mismatch for %s/%s (veh=%d, wal=%d): "
                    "current=%s, existing=%s — data-affecting params changed!",
                    town,
                    weather,
                    n_vehicles,
                    n_walkers_val,
                    current_hash,
                    existing_hashes,
                )

            if done > 0:
                logger.info(
                    "%s/%s (veh=%d, wal=%d): %d done, %d remaining",
                    town,
                    weather,
                    n_vehicles,
                    n_walkers_val,
                    done,
                    needed,
                )

            for _ in range(needed):
                remaining.append((dataclasses.replace(ep), 1))

    return remaining


def _preload_map(
    town: str,
    host: str,
    port: int,
    max_retries: int = 3,
    base_wait: float = 10.0,
) -> bool:
    """Load a CARLA map in a throwaway subprocess with retry.

    CARLA's C++ client library leaks ~100 MB/s of anonymous memory
    after ``load_world()`` map changes.  By calling ``load_world()``
    in a short-lived subprocess, the OS reclaims all leaked C-level
    memory when the subprocess exits.  The CARLA *server* retains the
    new map, so the next client that connects sees it immediately.

    Uses exponential backoff on failure (10 s → 20 s → 40 s by default)
    to handle transient CARLA server hiccups without losing an entire
    town's data.

    Returns True if the map was loaded successfully.
    """
    # Inline script: connect, load_world, exit.  No sensor callbacks
    # are created, so even if the leak fires it's tiny and dies with
    # this process.
    loader_script = f"""\
import subprocess as sp, sys, time
import carla
host = "{host}"
if host == "auto":
    try:
        r = sp.run(["ip", "route", "show", "default"],
                    capture_output=True, text=True, check=True)
        host = r.stdout.strip().split()[2]
    except Exception:
        host = "localhost"
c = carla.Client(host, {port})
c.set_timeout(120.0)
c.load_world("{town}")
time.sleep(3)
print(f"Map preloaded: {town} on {{host}}")
"""
    for attempt in range(1, max_retries + 1):
        try:
            result = subprocess.run(
                [sys.executable, "-c", loader_script],
                timeout=180,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "Map preload for %s timed out (attempt %d/%d)",
                town,
                attempt,
                max_retries,
            )
            if attempt < max_retries:
                wait = base_wait * (2 ** (attempt - 1))
                logger.info("Retrying in %.0fs...", wait)
                time.sleep(wait)
            continue

        if result.returncode == 0:
            logger.info("Map preloader for %s exited cleanly", town)
            return True

        stderr_tail = result.stderr.strip()[-500:] if result.stderr else "(no stderr)"
        if attempt < max_retries:
            wait = base_wait * (2 ** (attempt - 1))
            logger.warning(
                "Map preload for %s failed (attempt %d/%d, rc=%d): %s " "— retrying in %.0fs",
                town,
                attempt,
                max_retries,
                result.returncode,
                stderr_tail,
                wait,
            )
            time.sleep(wait)
        else:
            logger.error(
                "Map preload for %s failed after %d attempts (rc=%d): %s",
                town,
                max_retries,
                result.returncode,
                stderr_tail,
            )

    return False


def _run_orchestrator(args: argparse.Namespace) -> None:
    """Spawn one subprocess per town for memory isolation.

    For each town the orchestrator:
    1. Spawns a throwaway "map preloader" subprocess that calls
       ``load_world(town)`` then exits — any leaked C-level memory
       is reclaimed by the OS.
    2. Spawns the real collector worker which connects to CARLA
       (server is already on the correct map), calls ``get_world()``,
       and collects all episodes **without ever calling load_world()**.

    This two-phase design ensures the collector process never triggers
    the libcarla_client.so memory leak.
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

    # Resolve host once for the preloader
    carla_host = args.host
    carla_port = args.port

    failed_towns: list[tuple[str, str]] = []
    for i, town in enumerate(towns, 1):
        # --- Phase 1: preload map in throwaway subprocess ---
        logger.info("--- [%d/%d] Preloading map %s ---", i, len(towns), town)
        if not _preload_map(town, carla_host, carla_port):
            logger.error("Skipping %s — map preload failed", town)
            failed_towns.append((town, "preload failed"))
            continue

        # --- Phase 2: spawn collector worker (no load_world) ---
        logger.info("--- [%d/%d] Spawning worker for %s ---", i, len(towns), town)
        cmd = [sys.executable, script] + base_argv + ["--_subprocess-town", town]

        # F3: retry loop — worker exits with code 42 when RSS guard fires.
        # The OS reclaims all leaked C-level memory, then we respawn.
        max_restarts = 10
        for restart in range(max_restarts + 1):
            if restart > 0:
                logger.info(
                    "--- Restarting worker for %s (attempt %d/%d) ---",
                    town,
                    restart + 1,
                    max_restarts + 1,
                )
            result = subprocess.run(cmd)

            if result.returncode == 0:
                logger.info("Town %s worker exited cleanly", town)
                break
            elif result.returncode == 42:
                # F3 RSS guard triggered — worker saved its episode and
                # exited.  The manifest records progress, so the next
                # worker will skip completed episodes.
                logger.info(
                    "Town %s worker hit RSS limit — restarting " "(restart %d/%d)",
                    town,
                    restart + 1,
                    max_restarts,
                )
                # Quick preload to ensure map is still correct
                _preload_map(town, carla_host, carla_port)
                continue
            elif result.returncode == -9:
                logger.error(
                    "Town %s worker OOM-killed (SIGKILL). "
                    "Leak may persist within-town for very long runs.",
                    town,
                )
                failed_towns.append((town, "OOM-killed"))
                break
            else:
                logger.error(
                    "Town %s worker exited with code %d",
                    town,
                    result.returncode,
                )
                failed_towns.append((town, f"exit code {result.returncode}"))
                break
        else:
            # Exhausted all restarts
            logger.error(
                "Town %s exhausted %d RSS-guard restarts",
                town,
                max_restarts,
            )
            failed_towns.append((town, "max RSS restarts exceeded"))

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

    # Count already-recorded episodes for this town (any status) so we
    # can skip them.  This enables both --resume and F3 restart-after-
    # RSS-guard to work correctly.
    done_count = 0
    town = args._subprocess_town
    for e in manifest.data.get("episodes", []):
        if e.get("town", "") == town:
            done_count += 1

    n_episodes = args.episodes or 1  # Episodes per traffic config
    campaign = build_campaign(cfg, manifest, n_episodes)

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
    # F2b: The map was already loaded by the orchestrator's preloader
    # subprocess.  The worker just calls get_world() — NO load_world()
    # — so the libcarla_client.so memory leak is never triggered.
    episode_num = 0
    current_town: str | None = None
    route_gen: BalancedRouteGenerator | None = None

    for config, n_remaining in campaign:
        # Verify correct map (preloaded by orchestrator) — no load_world()!
        if config.town != current_town:
            assert collector.client is not None

            if args.memory_log:
                log_memory_snapshot(f"PRE get_world({config.town})")

            # The orchestrator's _preload_map() already called load_world()
            # in a throwaway subprocess.  We just get_world() here.
            collector.world = collector.client.get_world()
            actual_map = collector.world.get_map().name
            expected_suffix = config.town

            if expected_suffix not in actual_map:
                # Fallback: map mismatch — the preloader may have failed
                # or another client changed the map.  Load it ourselves.
                logger.warning(
                    "Map mismatch: expected %s, got %s — loading directly",
                    config.town,
                    actual_map,
                )
                collector.client.set_timeout(120.0)
                try:
                    collector.client.load_world(config.town)
                    time.sleep(5.0)
                    collector.world = collector.client.get_world()
                except RuntimeError as e:
                    logger.error(
                        "Failed to load %s: %s — skipping town",
                        config.town,
                        e,
                    )
                    collector.client.set_timeout(60.0)
                    current_town = None
                    continue
                collector.client.set_timeout(60.0)
            else:
                logger.info(
                    "Map %s already loaded (preloaded by orchestrator)",
                    actual_map,
                )

            if args.memory_log:
                log_memory_snapshot(f"POST get_world({config.town})")

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
            result.config_hash = _episode_config_hash(config)
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

            # F3: RSS memory guard — exit worker so orchestrator restarts it
            if result.status == "memory_limit":
                logger.warning(
                    "RSS memory guard triggered — exiting worker for restart " "(PID %d, %s)",
                    os.getpid(),
                    args._subprocess_town,
                )
                sys.exit(42)  # Special exit code: "restart me"

    # Worker complete — process exit reclaims all C-level memory
    logger.info(
        "Worker for %s finished (PID %d)",
        args._subprocess_town,
        os.getpid(),
    )


if __name__ == "__main__":
    main()
