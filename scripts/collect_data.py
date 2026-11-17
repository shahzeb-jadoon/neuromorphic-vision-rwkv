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
from collections import deque
from dataclasses import dataclass
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


@dataclass
class RouteContinuationConfig:
    """Continuation policy once a route is truly exhausted (expert.done())."""

    enabled: bool = False
    mode: str = "off"  # off | micro_replan | atlas_chain
    max_chains_per_episode: int = 0
    min_route_length: int = 120
    max_route_length: int = 240
    atlas_active_horizon: int = 4
    atlas_bank_size: int = 32
    atlas_min_destination_distance_m: float = 60.0
    preload_remaining_waypoints: int = 0


class _LazyAtlasContinuationPlanner:
    """Bounded-memory, lazy route-atlas continuation planner.

    Maintains a small queue of destination anchors and only materializes
    routes when continuation is requested. This keeps memory usage bounded
    while enabling long-lived chained routing.
    """

    def __init__(
        self,
        route_gen: BalancedRouteGenerator,
        rng: np.random.Generator,
        *,
        min_waypoints: int,
        max_waypoints: int,
        active_horizon: int,
        bank_size: int,
        min_destination_distance_m: float,
    ) -> None:
        self.route_gen = route_gen
        self.rng = rng
        self.min_waypoints = max(1, int(min_waypoints))
        self.max_waypoints = max(self.min_waypoints, int(max_waypoints))
        self.active_horizon = max(1, int(active_horizon))
        self.bank_size = max(self.active_horizon, int(bank_size))
        self.min_destination_distance_m = max(0.0, float(min_destination_distance_m))
        self._dest_queue: deque[Any] = deque()
        self._spawn_cycle = list(self.route_gen.spawn_points)
        if self._spawn_cycle:
            order = np.arange(len(self._spawn_cycle))
            self.rng.shuffle(order)
            self._spawn_cycle = [self._spawn_cycle[int(i)] for i in order]
        self._spawn_idx = 0

    def _next_destination_anchor(self, current_location: Any) -> Any | None:
        if not self._spawn_cycle:
            return None

        attempts = max(1, len(self._spawn_cycle))
        for _ in range(attempts):
            anchor = self._spawn_cycle[self._spawn_idx]
            self._spawn_idx = (self._spawn_idx + 1) % len(self._spawn_cycle)

            try:
                if current_location.distance(anchor.location) < self.min_destination_distance_m:
                    continue
            except Exception:
                pass
            return anchor
        return None

    def _refill(self, current_location: Any) -> None:
        target_size = min(self.active_horizon, self.bank_size)
        while len(self._dest_queue) < target_size:
            anchor = self._next_destination_anchor(current_location)
            if anchor is None:
                break
            if len(self._dest_queue) < self.bank_size:
                self._dest_queue.append(anchor)

    def next_route(self, current_location: Any) -> list[tuple[Any, str]]:
        self._refill(current_location)

        lane_type_driving: Any | None = None
        try:
            import carla as _carla

            lane_type_driving = _carla.LaneType.Driving
        except Exception:
            lane_type_driving = None

        start_wp = None
        try:
            if lane_type_driving is not None:
                start_wp = self.route_gen.map.get_waypoint(
                    current_location,
                    project_to_road=True,
                    lane_type=lane_type_driving,
                )
            else:
                start_wp = self.route_gen.map.get_waypoint(
                    current_location,
                    project_to_road=True,
                )
        except Exception:
            try:
                start_wp = self.route_gen.map.get_waypoint(current_location)
            except Exception:
                start_wp = None

        if start_wp is None:
            return []

        while self._dest_queue:
            anchor = self._dest_queue.popleft()
            end_wp = self.route_gen._waypoint_from_transform(anchor)
            if end_wp is None:
                continue

            segment = self.route_gen._trace_route(start_wp, end_wp)
            fitted = self.route_gen._fit_route_length(
                segment,
                start_wp,
                self.min_waypoints,
                self.max_waypoints,
            )
            if fitted:
                return fitted

        # Fallback keeps continuation alive even if all current anchors fail.
        return self.route_gen.generate_route_from_location(
            current_location,
            self.rng,
            min_waypoints=self.min_waypoints,
            max_waypoints=self.max_waypoints,
            strategy="balanced",
        )


def parse_route_continuation_config(routes_cfg: dict) -> RouteContinuationConfig:
    """Parse continuation controls from ``routes.continuation`` YAML block."""
    cont_cfg = routes_cfg.get("continuation", {})

    enabled = bool(cont_cfg.get("enabled", False))
    mode = str(cont_cfg.get("mode", "off")).strip().lower()
    if mode not in {"off", "micro_replan", "atlas_chain"}:
        logger.warning("Unknown routes.continuation.mode=%s; falling back to off", mode)
        mode = "off"
    if not enabled:
        mode = "off"
    if mode == "off":
        enabled = False

    min_route_length = int(cont_cfg.get("min_route_length", 120))
    max_route_length = int(cont_cfg.get("max_route_length", 240))
    if max_route_length < min_route_length:
        logger.warning(
            "routes.continuation.max_route_length (%d) < min_route_length (%d); clamping",
            max_route_length,
            min_route_length,
        )
        max_route_length = min_route_length

    return RouteContinuationConfig(
        enabled=enabled,
        mode=mode,
        max_chains_per_episode=max(0, int(cont_cfg.get("max_chains_per_episode", 0))),
        min_route_length=max(1, min_route_length),
        max_route_length=max(1, max_route_length),
        atlas_active_horizon=max(1, int(cont_cfg.get("atlas_active_horizon", 4))),
        atlas_bank_size=max(1, int(cont_cfg.get("atlas_bank_size", 32))),
        atlas_min_destination_distance_m=max(
            0.0,
            float(cont_cfg.get("atlas_min_destination_distance_m", 60.0)),
        ),
        preload_remaining_waypoints=max(
            0,
            int(cont_cfg.get("preload_remaining_waypoints", 0)),
        ),
    )


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
        f"_vvert{ec.vehicle_vertical_conflict_max_m}"
        f"_phd{ec.pedestrian_hazard_distance}"
        f"_tailx{int(bool(ec.route_tail_extension_enabled))}"
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
    if "pedestrian_emergency_distance" in expert_cfg_dict:
        expert_kw["pedestrian_emergency_distance"] = float(
            expert_cfg_dict["pedestrian_emergency_distance"]
        )
    if "turn_speed_factor" in expert_cfg_dict:
        expert_kw["turn_speed_factor"] = float(expert_cfg_dict["turn_speed_factor"])
    if "steering_noise_sigma" in expert_cfg_dict:
        expert_kw["steering_noise_sigma"] = float(expert_cfg_dict["steering_noise_sigma"])
    if "route_planner_backend" in expert_cfg_dict:
        expert_kw["route_planner_backend"] = (
            str(expert_cfg_dict["route_planner_backend"]).strip().lower()
        )
    # V2 shared vertical gate: support both new key and legacy aliases.
    if "vehicle_vertical_conflict_max_m" in expert_cfg_dict:
        expert_kw["vehicle_vertical_conflict_max_m"] = float(
            expert_cfg_dict["vehicle_vertical_conflict_max_m"]
        )
    elif "vehicle_adjacent_sat_vertical_max_m" in expert_cfg_dict:
        expert_kw["vehicle_vertical_conflict_max_m"] = float(
            expert_cfg_dict["vehicle_adjacent_sat_vertical_max_m"]
        )
    elif "corridor_max_vertical_separation" in expert_cfg_dict:
        expert_kw["vehicle_vertical_conflict_max_m"] = float(
            expert_cfg_dict["corridor_max_vertical_separation"]
        )

    float_policy_fields = [
        "gap_fill_max_factor",
        "gap_fill_diverge_factor",
        "gap_fill_bezier_step",
        "gap_fill_bezier_tangent_factor",
        "gap_fill_bezier_max_project_error",
        "planner_wrong_way_max_yaw_diff_deg",
        "planner_turn_mismatch_threshold_deg",
        "planner_lane_change_freeze_distance_m",
        "planner_lane_change_consecutive_window_m",
        "planner_mandatory_exit_window_m",
        "planner_exit_lane_leave_distance_m",
        "planner_legality_max_skip_gap_m",
        "route_watchdog_wp_buffer_head_dist_m",
        "route_watchdog_progress_epsilon",
        "route_watchdog_min_moving_speed_kmh",
        "route_watchdog_heading_mismatch_deg",
        "route_watchdog_tl_hold_exempt_speed_kmh",
        "oncoming_lead_lateral_max_m",
        "oncoming_lead_longitudinal_max_m",
        "oncoming_lead_ttc_max_s",
        "oncoming_lead_min_closing_speed_kmh",
        "planner_roundabout_context_window_m",
        "planner_roundabout_lane_change_cooldown_m",
        "lane_graph_sampling_resolution_m",
        "lane_graph_lane_change_cost",
        "lane_graph_late_lane_change_penalty",
        "lane_graph_turn_threshold_deg",
        "arc_watchdog_progress_epsilon_m",
        "route_tail_extension_step_m",
    ]
    for policy_key in float_policy_fields:
        if policy_key in expert_cfg_dict:
            expert_kw[policy_key] = float(expert_cfg_dict[policy_key])

    bool_policy_fields = [
        "planner_policy_enabled",
        "planner_legality_filter_enabled",
        "planner_allow_lane_id_sign_flip",
        "planner_turn_lane_validation_enabled",
        "route_watchdog_enabled",
        "vehicle_allow_lateral_fallback",
        "vehicle_oncoming_conflict_gate_enabled",
        "planner_roundabout_commitment_enabled",
        "gap_fill_startup_circuit_breaker_enabled",
        "gap_fill_bezier_enabled",
        "planner_policy_steer_consistency_gate_enabled",
        "arc_watchdog_enabled",
        "route_tail_extension_enabled",
    ]
    for policy_key in bool_policy_fields:
        if policy_key in expert_cfg_dict:
            expert_kw[policy_key] = bool(expert_cfg_dict[policy_key])

    int_policy_fields = [
        "gap_fill_rpc_budget",
        "gap_fill_bezier_retry_limit",
        "gap_fill_rpc_retry_limit",
        "gap_fill_bezier_max_pair_index",
        "route_watchdog_min_corridor_segments",
        "route_watchdog_min_frames",
        "route_watchdog_stall_min_frames",
        "lane_graph_max_search_expansions",
        "arc_watchdog_stall_min_frames",
    ]
    for policy_key in int_policy_fields:
        if policy_key in expert_cfg_dict:
            expert_kw[policy_key] = int(expert_cfg_dict[policy_key])

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
                "Map preload for %s failed (attempt %d/%d, rc=%d): %s — retrying in %.0fs",
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
                    "Town %s worker hit RSS limit — restarting (restart %d/%d)",
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
        default=None,
        choices=["random", "balanced"],
        help="Override route strategy (default: from YAML routes.strategy)",
    )
    parser.add_argument(
        "--target-left",
        type=float,
        default=None,
        help=(
            "Override target left-turn fraction "
            "(default: from YAML routes.target_distribution.left)"
        ),
    )
    parser.add_argument(
        "--target-right",
        type=float,
        default=None,
        help=(
            "Override target right-turn fraction "
            "(default: from YAML routes.target_distribution.right)"
        ),
    )
    parser.add_argument(
        "--target-straight",
        type=float,
        default=None,
        help=(
            "Override target straight-turn fraction "
            "(default: from YAML routes.target_distribution.straight)"
        ),
    )
    parser.add_argument(
        "--min-route-length",
        type=int,
        default=None,
        help=(
            "Override minimum route length in waypoints "
            "(default: from YAML routes.min_route_length)"
        ),
    )
    parser.add_argument(
        "--max-route-length",
        type=int,
        default=None,
        help=(
            "Override maximum route length in waypoints "
            "(default: from YAML routes.max_route_length)"
        ),
    )
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

    # Route config (kept consistent with expert planner backend)
    route_backend = "lane_graph"
    lane_graph_sampling_resolution_m = 1.0
    lane_graph_lane_change_cost = 25.0
    lane_graph_late_lane_change_penalty = 300.0
    lane_graph_lane_change_freeze_distance_m = 30.0
    lane_graph_turn_threshold_deg = 35.0
    lane_graph_max_search_expansions = 250_000

    if campaign:
        cfg0 = campaign[0][0].expert_config
        route_backend = str(cfg0.route_planner_backend).strip().lower()
        lane_graph_sampling_resolution_m = float(cfg0.lane_graph_sampling_resolution_m)
        lane_graph_lane_change_cost = float(cfg0.lane_graph_lane_change_cost)
        lane_graph_late_lane_change_penalty = float(cfg0.lane_graph_late_lane_change_penalty)
        lane_graph_lane_change_freeze_distance_m = float(cfg0.planner_lane_change_freeze_distance_m)
        lane_graph_turn_threshold_deg = float(cfg0.lane_graph_turn_threshold_deg)
        lane_graph_max_search_expansions = int(cfg0.lane_graph_max_search_expansions)

    routes_cfg = cfg.get("routes", {})
    target_dist_cfg = routes_cfg.get("target_distribution", {})

    route_strategy = args.route_strategy or str(routes_cfg.get("strategy", "balanced"))
    target_left = (
        float(args.target_left)
        if args.target_left is not None
        else float(target_dist_cfg.get("left", 0.15))
    )
    target_right = (
        float(args.target_right)
        if args.target_right is not None
        else float(target_dist_cfg.get("right", 0.15))
    )
    target_straight = (
        float(args.target_straight)
        if args.target_straight is not None
        else float(target_dist_cfg.get("straight", 0.10))
    )
    min_route_length = (
        int(args.min_route_length)
        if args.min_route_length is not None
        else int(routes_cfg.get("min_route_length", 100))
    )
    max_route_length = (
        int(args.max_route_length)
        if args.max_route_length is not None
        else int(routes_cfg.get("max_route_length", 500))
    )
    campaign_turn_cfg = routes_cfg.get("campaign_turn_balance", {})
    campaign_turn_balance_enabled = bool(campaign_turn_cfg.get("enabled", False))
    campaign_turn_balance_max_adjust = float(campaign_turn_cfg.get("max_adjust", 0.05))
    campaign_turn_balance_smoothing_alpha = float(campaign_turn_cfg.get("smoothing_alpha", 0.35))
    campaign_turn_balance_min_turn_samples = int(campaign_turn_cfg.get("min_turn_samples", 20))

    min_route_length = max(1, min_route_length)
    max_route_length = max(1, max_route_length)
    if max_route_length < min_route_length:
        logger.warning(
            "max_route_length (%d) < min_route_length (%d); clamping max to min",
            max_route_length,
            min_route_length,
        )
        max_route_length = min_route_length

    route_cfg = RouteConfig(
        strategy=route_strategy,
        target_left=target_left,
        target_right=target_right,
        target_straight=target_straight,
        min_waypoints=min_route_length,
        max_waypoints=max_route_length,
        route_planner_backend=route_backend,
        lane_graph_sampling_resolution_m=lane_graph_sampling_resolution_m,
        lane_graph_lane_change_cost=lane_graph_lane_change_cost,
        lane_graph_late_lane_change_penalty=lane_graph_late_lane_change_penalty,
        lane_graph_lane_change_freeze_distance_m=lane_graph_lane_change_freeze_distance_m,
        lane_graph_turn_threshold_deg=lane_graph_turn_threshold_deg,
        lane_graph_max_search_expansions=lane_graph_max_search_expansions,
        campaign_turn_balance_enabled=campaign_turn_balance_enabled,
        campaign_turn_balance_max_adjust=campaign_turn_balance_max_adjust,
        campaign_turn_balance_smoothing_alpha=campaign_turn_balance_smoothing_alpha,
        campaign_turn_balance_min_turn_samples=campaign_turn_balance_min_turn_samples,
    )

    continuation_cfg = parse_route_continuation_config(routes_cfg)
    if continuation_cfg.enabled:
        logger.info(
            "Route continuation enabled: mode=%s chains=%d len=[%d,%d] "
            "atlas_horizon=%d atlas_bank=%d preload_remaining_wp=%d",
            continuation_cfg.mode,
            continuation_cfg.max_chains_per_episode,
            continuation_cfg.min_route_length,
            continuation_cfg.max_route_length,
            continuation_cfg.atlas_active_horizon,
            continuation_cfg.atlas_bank_size,
            continuation_cfg.preload_remaining_waypoints,
        )

    # Run campaign
    # F2b: The map was already loaded by the orchestrator's preloader
    # subprocess.  The worker just calls get_world() — NO load_world()
    # — so the libcarla_client.so memory leak is never triggered.
    episode_num = 0
    current_town: str | None = None
    route_gen: BalancedRouteGenerator | None = None
    atlas_continuation_planner: _LazyAtlasContinuationPlanner | None = None

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

            atlas_continuation_planner = None
            if (
                continuation_cfg.enabled
                and continuation_cfg.mode == "atlas_chain"
                and route_gen is not None
            ):
                atlas_continuation_planner = _LazyAtlasContinuationPlanner(
                    route_gen,
                    rng,
                    min_waypoints=continuation_cfg.min_route_length,
                    max_waypoints=continuation_cfg.max_route_length,
                    active_horizon=continuation_cfg.atlas_active_horizon,
                    bank_size=continuation_cfg.atlas_bank_size,
                    min_destination_distance_m=continuation_cfg.atlas_min_destination_distance_m,
                )

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

            continuation_provider = None
            continuation_max_chains = 0
            if continuation_cfg.enabled and route_gen is not None:
                continuation_max_chains = continuation_cfg.max_chains_per_episode
                if config.expert_config.route_tail_extension_enabled:
                    # Continuation chaining should trigger only on true exhaustion
                    # (expert.done()) instead of synthetic tail extension.
                    config.expert_config.route_tail_extension_enabled = False

                if continuation_cfg.mode == "micro_replan":

                    def _micro_replan_provider(current_location: Any) -> list[tuple[Any, str]]:
                        return route_gen.generate_route_from_location(
                            current_location,
                            rng,
                            min_waypoints=continuation_cfg.min_route_length,
                            max_waypoints=continuation_cfg.max_route_length,
                            strategy=route_strategy,
                        )

                    continuation_provider = _micro_replan_provider

                elif continuation_cfg.mode == "atlas_chain":
                    planner = atlas_continuation_planner
                    if planner is None:
                        planner = _LazyAtlasContinuationPlanner(
                            route_gen,
                            rng,
                            min_waypoints=continuation_cfg.min_route_length,
                            max_waypoints=continuation_cfg.max_route_length,
                            active_horizon=continuation_cfg.atlas_active_horizon,
                            bank_size=continuation_cfg.atlas_bank_size,
                            min_destination_distance_m=(
                                continuation_cfg.atlas_min_destination_distance_m
                            ),
                        )
                        atlas_continuation_planner = planner
                    assert planner is not None
                    planner_nonnull: _LazyAtlasContinuationPlanner = planner

                    def _atlas_chain_provider(
                        current_location: Any,
                        _planner: _LazyAtlasContinuationPlanner = planner_nonnull,
                    ) -> list[tuple[Any, str]]:
                        return _planner.next_route(current_location)

                    continuation_provider = _atlas_chain_provider
            elif continuation_cfg.enabled:
                logger.warning(
                    "Route continuation is enabled but route generator is unavailable; "
                    "continuation disabled for this episode"
                )

            # Collect
            result = collector.collect_episode(
                config,
                route_waypoints=route,
                route_continuation_provider=continuation_provider,
                route_continuation_max_chains=continuation_max_chains,
                route_continuation_preload_remaining_waypoints=(
                    continuation_cfg.preload_remaining_waypoints
                ),
            )
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
                    f"POST episode {episode_num} status={result.status} frames={result.n_frames}"
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
                    "RSS memory guard triggered — exiting worker for restart (PID %d, %s)",
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
