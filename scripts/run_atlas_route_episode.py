#!/usr/bin/env python3
"""Run atlas routes through ExpertDataCollector (single or batch).

Supports three input modes:
- single route file
- one routes directory
- atlas root with per-town selection

Also supports automatic continuation chaining, route lane-jump gating, and
per-route completion/legality reporting.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import logging
from pathlib import Path
import random
from typing import Callable
from typing import Any
from uuid import uuid4

import carla
import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency at runtime
    h5py = None

from scripts.collect_data import (
    _LazyAtlasContinuationPlanner,
    build_campaign,
    load_config,
    parse_route_continuation_config,
)
from src.data.expert_collector import CollectionManifest, EpisodeConfig, ExpertDataCollector
from src.data.route_generator import BalancedRouteGenerator, RouteConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _to_transform(point: dict[str, Any]) -> carla.Transform:
    return carla.Transform(
        carla.Location(
            x=float(point.get("x", 0.0)),
            y=float(point.get("y", 0.0)),
            z=float(point.get("z", 0.0)),
        ),
        carla.Rotation(
            pitch=float(point.get("pitch", 0.0)),
            yaw=float(point.get("yaw", 0.0)),
            roll=float(point.get("roll", 0.0)),
        ),
    )


def _load_route(route_path: Path) -> dict[str, Any]:
    payload = json.loads(route_path.read_text())
    town = payload.get("town")
    raw_waypoints = list(payload.get("waypoints", []))
    if not raw_waypoints:
        raise ValueError(f"Route has no waypoints: {route_path}")

    route_waypoints: list[tuple[carla.Transform, str]] = []
    for item in raw_waypoints:
        transform = _to_transform(item)
        road_option = str(item.get("road_option", "LANEFOLLOW"))
        route_waypoints.append((transform, road_option))

    return {
        "route_path": route_path,
        "route_id": str(payload.get("route_id", route_path.stem)),
        "town": str(town) if town else None,
        "waypoint_count": int(payload.get("waypoint_count", len(route_waypoints))),
        "route_waypoints": route_waypoints,
    }


def _discover_route_files(args: argparse.Namespace, rng: random.Random) -> list[Path]:
    files: list[Path] = []

    if args.atlas_route is not None:
        files = [args.atlas_route]

    elif args.atlas_routes_dir is not None:
        files = sorted(args.atlas_routes_dir.glob("route_*.json"))
        if not files:
            files = sorted(args.atlas_routes_dir.glob("*.json"))
        if args.route_order == "random":
            rng.shuffle(files)

    elif args.atlas_root is not None:
        towns = list(args.town_filter or [])
        if not towns:
            towns = sorted(
                p.name
                for p in args.atlas_root.iterdir()
                if p.is_dir() and p.name.startswith("Town")
            )

        if not towns:
            raise SystemExit(f"ERROR: no towns found under atlas root: {args.atlas_root}")

        solver_names = ["cpp-approx", "cpp-exact"] if args.solver == "both" else [args.solver]
        for town in towns:
            town_routes: list[Path] = []
            for solver in solver_names:
                route_dir = args.atlas_root / town / solver / "routes"
                if not route_dir.exists():
                    continue
                town_routes.extend(sorted(route_dir.glob("route_*.json")))

            if args.route_order == "random":
                rng.shuffle(town_routes)

            files.extend(town_routes)

    if args.max_routes > 0:
        files = files[: int(args.max_routes)]

    if not files:
        raise SystemExit("ERROR: no route files resolved from the selected mode")

    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"ERROR: missing route files: {missing[:5]}")

    return files


def _build_episode_config(
    *,
    cfg: dict[str, Any],
    output_dir: Path,
    town: str,
    weather: str,
    n_vehicles: int,
    n_walkers: int,
    max_frames: int,
    spectator: bool,
    debug_draw: bool,
    record: bool,
) -> EpisodeConfig:
    local_cfg = dict(cfg)
    local_cfg["towns"] = [town]
    local_cfg["weathers"] = [weather]
    local_cfg.setdefault("traffic", {})["vehicles"] = [int(n_vehicles)]
    local_cfg.setdefault("traffic", {})["walkers"] = [int(n_walkers)]

    temp_manifest = CollectionManifest(output_dir / f"._atlas_probe_{uuid4().hex}.json")
    campaign = build_campaign(local_cfg, temp_manifest, episodes_per_config=1)
    if not campaign:
        raise RuntimeError("Unable to create episode config from collection config")

    base_config = campaign[0][0]
    return replace(
        base_config,
        town=town,
        weather=weather,
        n_vehicles=int(n_vehicles),
        n_walkers=int(n_walkers),
        max_frames=int(max_frames),
        spectator_follow=bool(spectator),
        debug_draw=bool(debug_draw),
        record_simulation=bool(record),
    )


def _build_route_config(cfg: dict[str, Any], episode_cfg: EpisodeConfig) -> RouteConfig:
    routes_cfg = cfg.get("routes", {})
    target_dist_cfg = routes_cfg.get("target_distribution", {})
    campaign_turn_cfg = routes_cfg.get("campaign_turn_balance", {})
    expert_cfg = episode_cfg.expert_config

    return RouteConfig(
        strategy=str(routes_cfg.get("strategy", "balanced")),
        target_left=float(target_dist_cfg.get("left", 0.15)),
        target_right=float(target_dist_cfg.get("right", 0.15)),
        target_straight=float(target_dist_cfg.get("straight", 0.10)),
        min_waypoints=max(1, int(routes_cfg.get("min_route_length", 100))),
        max_waypoints=max(1, int(routes_cfg.get("max_route_length", 500))),
        route_planner_backend=str(expert_cfg.route_planner_backend).strip().lower(),
        lane_graph_sampling_resolution_m=float(expert_cfg.lane_graph_sampling_resolution_m),
        lane_graph_lane_change_cost=float(expert_cfg.lane_graph_lane_change_cost),
        lane_graph_late_lane_change_penalty=float(expert_cfg.lane_graph_late_lane_change_penalty),
        lane_graph_lane_change_freeze_distance_m=float(
            expert_cfg.planner_lane_change_freeze_distance_m
        ),
        lane_graph_turn_threshold_deg=float(expert_cfg.lane_graph_turn_threshold_deg),
        lane_graph_max_search_expansions=int(expert_cfg.lane_graph_max_search_expansions),
        campaign_turn_balance_enabled=bool(campaign_turn_cfg.get("enabled", False)),
        campaign_turn_balance_max_adjust=float(campaign_turn_cfg.get("max_adjust", 0.05)),
        campaign_turn_balance_smoothing_alpha=float(campaign_turn_cfg.get("smoothing_alpha", 0.35)),
        campaign_turn_balance_min_turn_samples=int(campaign_turn_cfg.get("min_turn_samples", 20)),
    )


def _ensure_world_for_town(collector: ExpertDataCollector, town: str) -> None:
    if collector.client is None:
        raise RuntimeError("Collector client is not connected")

    world = collector.client.get_world()
    map_name = world.get_map().name.split("/")[-1]
    if map_name != town:
        logger.info("Loading map: %s", town)
        collector.client.set_timeout(120.0)
        collector.client.load_world(town)
        collector.client.set_timeout(60.0)
        collector.world = collector.client.get_world()
    else:
        collector.world = world


def _project_driving_waypoint(carla_map: Any, location: Any) -> Any:
    """Project a location to a drivable waypoint with safe fallbacks."""
    try:
        return carla_map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
    except Exception:
        try:
            return carla_map.get_waypoint(location, project_to_road=True)
        except Exception:
            try:
                return carla_map.get_waypoint(location)
            except Exception:
                return None


def _lane_jump_report(
    carla_map: Any,
    route_waypoints: list[tuple[Any, str]],
    min_spacing_m: float,
) -> dict[str, Any]:
    lane_change_labels = {"CHANGELANELEFT", "CHANGELANERIGHT"}
    projected: list[tuple[int, Any, Any, str]] = []
    missing_projection_count = 0

    for index, (transform, road_opt) in enumerate(route_waypoints):
        wp = _project_driving_waypoint(carla_map, transform.location)
        if wp is None:
            missing_projection_count += 1
            continue
        projected.append((index, wp, transform.location, str(road_opt).upper()))

    multi_lane_steps = 0
    rapid_consecutive_lane_changes = 0
    rapid_lane_change_command_streaks = 0
    issues: list[dict[str, Any]] = []

    path_distance_m = 0.0
    last_lane_change_distance_m: float | None = None
    last_lane_change_command_distance_m: float | None = None
    previous_command: str | None = None

    for prev, curr in zip(projected, projected[1:]):
        prev_idx, prev_wp, prev_loc, prev_cmd = prev
        curr_idx, curr_wp, curr_loc, curr_cmd = curr
        seg_dist = float(prev_loc.distance(curr_loc))
        path_distance_m += max(0.0, seg_dist)

        same_road = int(prev_wp.road_id) == int(curr_wp.road_id)
        same_section = int(prev_wp.section_id) == int(curr_wp.section_id)
        same_sign = int(prev_wp.lane_id) * int(curr_wp.lane_id) > 0
        lane_delta = abs(int(curr_wp.lane_id) - int(prev_wp.lane_id))

        if same_road and same_section and same_sign and lane_delta > 1:
            multi_lane_steps += 1
            if len(issues) < 6:
                issues.append(
                    {
                        "type": "multi_lane_step",
                        "from_index": int(prev_idx),
                        "to_index": int(curr_idx),
                        "road_id": int(curr_wp.road_id),
                        "from_lane": int(prev_wp.lane_id),
                        "to_lane": int(curr_wp.lane_id),
                        "path_distance_m": round(path_distance_m, 3),
                    }
                )

        if same_road and same_section and same_sign and lane_delta >= 1:
            if last_lane_change_distance_m is not None and (
                path_distance_m - last_lane_change_distance_m
            ) < float(min_spacing_m):
                rapid_consecutive_lane_changes += 1
                if len(issues) < 6:
                    issues.append(
                        {
                            "type": "rapid_consecutive_lane_change",
                            "from_index": int(prev_idx),
                            "to_index": int(curr_idx),
                            "road_id": int(curr_wp.road_id),
                            "lane_delta": int(lane_delta),
                            "distance_since_last_change_m": round(
                                path_distance_m - last_lane_change_distance_m,
                                3,
                            ),
                        }
                    )
            last_lane_change_distance_m = float(path_distance_m)

        curr_is_lc = curr_cmd in lane_change_labels
        if (
            curr_is_lc
            and previous_command == curr_cmd
            and last_lane_change_command_distance_m is not None
        ):
            if (path_distance_m - last_lane_change_command_distance_m) < float(min_spacing_m):
                rapid_lane_change_command_streaks += 1
                if len(issues) < 6:
                    issues.append(
                        {
                            "type": "rapid_lane_change_command_streak",
                            "from_index": int(prev_idx),
                            "to_index": int(curr_idx),
                            "command": curr_cmd,
                            "distance_since_last_same_command_m": round(
                                path_distance_m - last_lane_change_command_distance_m,
                                3,
                            ),
                        }
                    )
        if curr_is_lc:
            last_lane_change_command_distance_m = float(path_distance_m)
        previous_command = curr_cmd

    invalid = bool(
        multi_lane_steps > 0
        or rapid_consecutive_lane_changes > 0
        or rapid_lane_change_command_streaks > 0
    )

    return {
        "invalid": invalid,
        "multi_lane_steps": int(multi_lane_steps),
        "rapid_consecutive_lane_changes": int(rapid_consecutive_lane_changes),
        "rapid_lane_change_command_streaks": int(rapid_lane_change_command_streaks),
        "missing_waypoint_projections": int(missing_projection_count),
        "sample_issues": issues,
    }


def _route_gap_report(
    carla_map: Any,
    route_waypoints: list[tuple[Any, str]],
    *,
    warn_gap_m: float,
    severe_gap_m: float,
    max_issues: int,
) -> dict[str, Any]:
    """Report suspicious large route jumps before running an episode."""
    if len(route_waypoints) < 2:
        return {
            "warn_gap_m": float(warn_gap_m),
            "severe_gap_m": float(severe_gap_m),
            "warn_transition_count": 0,
            "severe_transition_count": 0,
            "max_gap_m": 0.0,
            "missing_waypoint_projections": 0,
            "issues": [],
        }

    projected: list[tuple[int, Any, str, Any]] = []
    missing_projection_count = 0
    for index, (transform, road_opt) in enumerate(route_waypoints):
        wp = _project_driving_waypoint(carla_map, transform.location)
        if wp is None:
            missing_projection_count += 1
        projected.append((index, transform.location, str(road_opt).upper(), wp))

    issues: list[dict[str, Any]] = []
    warn_count = 0
    severe_count = 0
    max_gap = 0.0

    for prev, curr in zip(projected, projected[1:]):
        prev_idx, prev_loc, prev_cmd, prev_wp = prev
        curr_idx, curr_loc, curr_cmd, curr_wp = curr

        gap_m = float(prev_loc.distance(curr_loc))
        max_gap = max(max_gap, gap_m)

        same_command = prev_cmd == curr_cmd
        road_changed = False
        if prev_wp is not None and curr_wp is not None:
            road_changed = int(getattr(prev_wp, "road_id", -1)) != int(
                getattr(curr_wp, "road_id", -1)
            )

        if gap_m < float(warn_gap_m) or not (same_command or road_changed):
            continue

        warn_count += 1
        severity = "severe" if gap_m >= float(severe_gap_m) else "warn"
        if severity == "severe":
            severe_count += 1

        if len(issues) < max(1, int(max_issues)):
            issues.append(
                {
                    "from_index": int(prev_idx),
                    "to_index": int(curr_idx),
                    "gap_m": round(gap_m, 3),
                    "severity": severity,
                    "same_command": bool(same_command),
                    "road_option_from": str(prev_cmd),
                    "road_option_to": str(curr_cmd),
                    "road_id_from": (
                        int(getattr(prev_wp, "road_id", -1)) if prev_wp is not None else -1
                    ),
                    "road_id_to": (
                        int(getattr(curr_wp, "road_id", -1)) if curr_wp is not None else -1
                    ),
                    "lane_id_from": (
                        int(getattr(prev_wp, "lane_id", -1)) if prev_wp is not None else -1
                    ),
                    "lane_id_to": (
                        int(getattr(curr_wp, "lane_id", -1)) if curr_wp is not None else -1
                    ),
                }
            )

    return {
        "warn_gap_m": float(warn_gap_m),
        "severe_gap_m": float(severe_gap_m),
        "warn_transition_count": int(warn_count),
        "severe_transition_count": int(severe_count),
        "max_gap_m": round(max_gap, 3),
        "missing_waypoint_projections": int(missing_projection_count),
        "issues": issues,
    }


def _continuation_anchor_report(
    carla_map: Any,
    current_location: Any,
    candidate_route: list[tuple[Any, str]],
    *,
    max_start_distance_m: float,
    max_heading_diff_deg: float,
    max_lane_delta: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "invalid": False,
        "reasons": [],
        "start_distance_m": None,
        "heading_diff_deg": None,
        "current_road_id": None,
        "current_lane_id": None,
        "candidate_road_id": None,
        "candidate_lane_id": None,
        "lane_delta": None,
    }

    if not candidate_route:
        report["invalid"] = True
        report["reasons"].append("empty_candidate")
        return report

    first_transform = candidate_route[0][0]

    try:
        report["start_distance_m"] = float(current_location.distance(first_transform.location))
    except Exception:
        report["start_distance_m"] = None

    current_wp = None
    candidate_wp = None
    try:
        current_wp = _project_driving_waypoint(carla_map, current_location)
    except Exception:
        current_wp = None
    try:
        candidate_wp = _project_driving_waypoint(carla_map, first_transform.location)
    except Exception:
        candidate_wp = None

    if current_wp is not None:
        report["current_road_id"] = int(getattr(current_wp, "road_id", -1))
        report["current_lane_id"] = int(getattr(current_wp, "lane_id", 0))
    if candidate_wp is not None:
        report["candidate_road_id"] = int(getattr(candidate_wp, "road_id", -1))
        report["candidate_lane_id"] = int(getattr(candidate_wp, "lane_id", 0))

    start_distance_m = report["start_distance_m"]
    if isinstance(start_distance_m, float) and start_distance_m > float(max_start_distance_m):
        report["invalid"] = True
        report["reasons"].append(f"start_distance>{float(max_start_distance_m):.1f}m")

    if current_wp is not None and candidate_wp is not None:
        current_yaw = float(current_wp.transform.rotation.yaw)
        candidate_yaw = float(candidate_wp.transform.rotation.yaw)
        yaw_diff = abs((candidate_yaw - current_yaw + 180.0) % 360.0 - 180.0)
        report["heading_diff_deg"] = float(yaw_diff)
        if yaw_diff > float(max_heading_diff_deg):
            report["invalid"] = True
            report["reasons"].append(f"heading_diff>{float(max_heading_diff_deg):.1f}deg")

        same_road = int(getattr(current_wp, "road_id", -1)) == int(
            getattr(candidate_wp, "road_id", -2)
        )
        same_section = int(getattr(current_wp, "section_id", -1)) == int(
            getattr(candidate_wp, "section_id", -2)
        )
        same_sign = (
            int(getattr(current_wp, "lane_id", 0)) * int(getattr(candidate_wp, "lane_id", 0)) > 0
        )
        if same_road and same_section and same_sign:
            lane_delta = abs(
                int(getattr(candidate_wp, "lane_id", 0)) - int(getattr(current_wp, "lane_id", 0))
            )
            report["lane_delta"] = int(lane_delta)
            if lane_delta > int(max_lane_delta):
                report["invalid"] = True
                report["reasons"].append(f"lane_delta>{int(max_lane_delta)}")

    return report


def _effective_continuation_config(base_cfg: Any, args: argparse.Namespace) -> Any:
    cfg = replace(base_cfg)

    if not bool(args.continuation):
        cfg.enabled = False
        cfg.mode = "off"
        cfg.max_chains_per_episode = 0
        return cfg

    override_mode = str(args.continuation_mode).strip().lower()
    if override_mode != "config":
        if override_mode == "off":
            cfg.enabled = False
            cfg.mode = "off"
            cfg.max_chains_per_episode = 0
        else:
            cfg.enabled = True
            cfg.mode = override_mode

    if args.continuation_max_chains is not None:
        cfg.max_chains_per_episode = max(0, int(args.continuation_max_chains))
        if cfg.max_chains_per_episode <= 0:
            cfg.enabled = False
            cfg.mode = "off"

    if args.continuation_preload_remaining_waypoints is not None:
        cfg.preload_remaining_waypoints = max(
            0,
            int(args.continuation_preload_remaining_waypoints),
        )

    return cfg


def _build_continuation_provider(
    *,
    continuation_cfg: Any,
    route_gen: BalancedRouteGenerator | None,
    rng: np.random.Generator,
    route_strategy: str,
    carla_map: Any,
    min_spacing_m: float,
    reject_multi_lane_jumps: bool,
    reject_anchor_mismatch: bool,
    max_anchor_start_distance_m: float,
    max_anchor_heading_diff_deg: float,
    max_anchor_lane_delta: int,
    max_attempts: int,
) -> tuple[Any | None, int]:
    if not continuation_cfg.enabled or route_gen is None:
        return None, 0

    raw_provider: Callable[[Any], list[tuple[Any, str]]]
    if continuation_cfg.mode == "micro_replan":

        def _micro_replan_provider(current_location: Any) -> list[tuple[Any, str]]:
            return route_gen.generate_route_from_location(
                current_location,
                rng,
                min_waypoints=continuation_cfg.min_route_length,
                max_waypoints=continuation_cfg.max_route_length,
                strategy=route_strategy,
            )

        raw_provider = _micro_replan_provider

    elif continuation_cfg.mode == "atlas_chain":
        planner = _LazyAtlasContinuationPlanner(
            route_gen,
            rng,
            min_waypoints=continuation_cfg.min_route_length,
            max_waypoints=continuation_cfg.max_route_length,
            active_horizon=continuation_cfg.atlas_active_horizon,
            bank_size=continuation_cfg.atlas_bank_size,
            min_destination_distance_m=continuation_cfg.atlas_min_destination_distance_m,
        )

        def _atlas_chain_provider(current_location: Any) -> list[tuple[Any, str]]:
            return planner.next_route(current_location)

        raw_provider = _atlas_chain_provider
    else:
        return None, 0

    attempts = max(1, int(max_attempts))

    def _guarded_provider(current_location: Any) -> list[tuple[Any, str]]:
        for _ in range(attempts):
            candidate: list[tuple[Any, str]] = raw_provider(current_location)
            if not candidate:
                return []

            if reject_anchor_mismatch:
                anchor_report = _continuation_anchor_report(
                    carla_map,
                    current_location,
                    candidate,
                    max_start_distance_m=max_anchor_start_distance_m,
                    max_heading_diff_deg=max_anchor_heading_diff_deg,
                    max_lane_delta=max_anchor_lane_delta,
                )
                if bool(anchor_report.get("invalid", False)):
                    logger.warning(
                        "Rejected continuation candidate by anchor gate: %s",
                        json.dumps(anchor_report, ensure_ascii=True),
                    )
                    continue

            report = _lane_jump_report(carla_map, candidate, min_spacing_m)
            if (not reject_multi_lane_jumps) or (not bool(report.get("invalid", False))):
                return candidate
        return []

    return _guarded_provider, int(max(0, continuation_cfg.max_chains_per_episode))


def _read_episode_metrics(output_dir: Path, filename: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "route_completion_max": None,
        "route_completion_final": None,
        "pre_junction_lane_change_violation_sum": None,
        "consecutive_lane_change_violation_sum": None,
        "roundabout_lane_change_violation_sum": None,
    }
    if h5py is None or not filename:
        return out

    path = output_dir / str(filename)
    if not path.exists():
        return out

    with h5py.File(path, "r") as h5:
        if "route_completion" in h5:
            values = np.asarray(h5["route_completion"])
            if values.size > 0:
                out["route_completion_max"] = float(np.max(values))
                out["route_completion_final"] = float(values[-1])

        for key in (
            "pre_junction_lane_change_violation",
            "consecutive_lane_change_violation",
            "roundabout_lane_change_violation",
        ):
            if key in h5:
                out[f"{key}_sum"] = int(np.sum(np.asarray(h5[key])))

    return out


def _build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    town_counts: dict[str, int] = {}

    completion_max_values: list[float] = []
    full_completion_count = 0

    for row in records:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

        town = str(row.get("town", ""))
        if town:
            town_counts[town] = town_counts.get(town, 0) + 1

        completion_max = row.get("route_completion_max")
        if isinstance(completion_max, (float, int)):
            completion_max_values.append(float(completion_max))
            if float(completion_max) >= 0.999:
                full_completion_count += 1

    mean_completion_max = None
    if completion_max_values:
        mean_completion_max = float(sum(completion_max_values) / len(completion_max_values))

    return {
        "total_records": int(len(records)),
        "status_counts": status_counts,
        "town_counts": town_counts,
        "full_route_completion_count": int(full_completion_count),
        "mean_route_completion_max": mean_completion_max,
    }


def _write_reports(
    *,
    output_dir: Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    json_path: Path | None,
    csv_path: Path | None,
) -> tuple[Path, Path]:
    out_json = json_path or (output_dir / "atlas_route_batch_report.json")
    out_csv = csv_path or (output_dir / "atlas_route_batch_report.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps({"summary": summary, "records": records}, indent=2))

    fieldnames = [
        "index",
        "route_file",
        "route_id",
        "town",
        "planned_waypoints",
        "status",
        "n_frames",
        "duration_seconds",
        "route_completion_max",
        "route_completion_final",
        "pre_junction_lane_change_violation_sum",
        "consecutive_lane_change_violation_sum",
        "roundabout_lane_change_violation_sum",
        "route_gap_warn_gap_m",
        "route_gap_severe_gap_m",
        "route_gap_warn_transition_count",
        "route_gap_severe_transition_count",
        "route_gap_max_gap_m",
        "route_gap_missing_waypoint_projections",
        "route_gap_gate_max_severe_transitions",
        "lane_jump_invalid",
        "lane_jump_multi_lane_steps",
        "lane_jump_rapid_consecutive_lane_changes",
        "lane_jump_rapid_lane_change_command_streaks",
        "lane_jump_missing_waypoint_projections",
        "continuation_mode",
        "continuation_max_chains",
        "continuation_preload_remaining_waypoints",
        "error_message",
        "filename",
        "route_path",
    ]

    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({name: row.get(name) for name in fieldnames})

    return out_json, out_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run atlas routes with ego through ExpertDataCollector (single or batch)"
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--atlas-route", type=Path, help="Path to one atlas route JSON")
    source.add_argument(
        "--atlas-routes-dir",
        type=Path,
        help="Directory containing route_*.json files for one batch",
    )
    source.add_argument(
        "--atlas-root",
        type=Path,
        help="Atlas root: <root>/<Town>/<solver>/routes/route_*.json",
    )

    parser.add_argument(
        "--town-override",
        "--town",
        dest="town_override",
        type=str,
        default=None,
        help="Override town embedded in route JSON (single/folder mode)",
    )
    parser.add_argument(
        "--town-filter",
        type=str,
        nargs="+",
        default=None,
        help="Town list to select under --atlas-root mode",
    )
    parser.add_argument(
        "--solver",
        choices=["cpp-approx", "cpp-exact", "both"],
        default="cpp-approx",
        help="Solver subfolder used with --atlas-root",
    )
    parser.add_argument(
        "--episodes-per-town",
        type=int,
        default=0,
        help="With --atlas-root: number of route files to run per town (0 means all)",
    )
    parser.add_argument(
        "--max-routes",
        type=int,
        default=0,
        help="Global cap on route files to run after selection (0 means all)",
    )
    parser.add_argument(
        "--route-order",
        choices=["sorted", "random"],
        default="sorted",
        help="Route selection order",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for random route ordering")

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/collection.yaml"),
        help="Collection config used to build EpisodeConfig defaults",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/test_collection_atlas_route"),
        help="Output directory for HDF5/sidecar/report artifacts",
    )

    parser.add_argument("--weather", type=str, default="ClearNoon")
    parser.add_argument("--vehicles", type=int, default=20)
    parser.add_argument("--walkers", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=4000)
    parser.add_argument("--host", type=str, default="auto")
    parser.add_argument("--port", type=int, default=2000)

    parser.add_argument(
        "--spectator",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable spectator follow",
    )
    parser.add_argument(
        "--debug-draw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable route debug drawing",
    )
    parser.add_argument(
        "--record",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable CARLA recorder",
    )

    parser.add_argument(
        "--continuation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable continuation chaining after route completion",
    )
    parser.add_argument(
        "--continuation-mode",
        choices=["config", "off", "micro_replan", "atlas_chain"],
        default="config",
        help="Override continuation mode (default: use config)",
    )
    parser.add_argument(
        "--continuation-max-chains",
        type=int,
        default=None,
        help="Override max continuation chains per episode",
    )
    parser.add_argument(
        "--continuation-preload-remaining-waypoints",
        type=int,
        default=None,
        help="Override preload trigger: continue when remaining waypoints <= this value",
    )
    parser.add_argument(
        "--continuation-attempts",
        type=int,
        default=4,
        help="Max attempts to get a continuation route that passes lane-jump gate",
    )

    parser.add_argument(
        "--reject-multi-lane-jumps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject routes with multi-lane skip or rapid consecutive lane-change jumps",
    )
    parser.add_argument(
        "--reject-severe-route-gaps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject routes with too many severe geometry gaps before episode start",
    )
    parser.add_argument(
        "--route-gap-warn-m",
        type=float,
        default=None,
        help="Warn threshold for route-gap diagnostics (default: episode config value)",
    )
    parser.add_argument(
        "--route-gap-severe-m",
        type=float,
        default=None,
        help="Severe threshold for route-gap diagnostics (default: episode config value)",
    )
    parser.add_argument(
        "--route-gap-max-severe-transitions",
        type=int,
        default=0,
        help="Maximum allowed severe transitions before route is skipped",
    )
    parser.add_argument(
        "--route-gap-max-issues",
        type=int,
        default=32,
        help="Maximum number of route-gap issue samples stored in report rows",
    )
    parser.add_argument(
        "--reject-anchor-mismatch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject continuation routes whose first waypoint is too far/misaligned from ego",
    )
    parser.add_argument(
        "--anchor-max-start-distance-m",
        type=float,
        default=20.0,
        help="Max allowed distance from ego to continuation route start waypoint",
    )
    parser.add_argument(
        "--anchor-max-heading-diff-deg",
        type=float,
        default=100.0,
        help="Max allowed heading difference between ego lane and continuation start",
    )
    parser.add_argument(
        "--anchor-max-lane-delta",
        type=int,
        default=1,
        help="Max same-road lane delta allowed at continuation start",
    )
    parser.add_argument(
        "--lane-change-min-spacing-m",
        type=float,
        default=None,
        help="Minimum spacing between lane changes (default: expert policy window)",
    )

    parser.add_argument(
        "--report-json", type=Path, default=None, help="Write JSON report to this path"
    )
    parser.add_argument(
        "--report-csv", type=Path, default=None, help="Write CSV report to this path"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve routes and print execution plan only (no CARLA run)",
    )

    args = parser.parse_args()

    if not args.config.exists():
        raise SystemExit(f"ERROR: config not found: {args.config}")
    if args.atlas_root is not None and not args.atlas_root.exists():
        raise SystemExit(f"ERROR: atlas root not found: {args.atlas_root}")
    if args.atlas_routes_dir is not None and not args.atlas_routes_dir.exists():
        raise SystemExit(f"ERROR: routes dir not found: {args.atlas_routes_dir}")
    if args.atlas_route is not None and not args.atlas_route.exists():
        raise SystemExit(f"ERROR: atlas route not found: {args.atlas_route}")

    rng_random = random.Random(int(args.seed))
    route_files = _discover_route_files(args, rng_random)

    quota_mode = bool(args.atlas_root is not None and int(args.episodes_per_town) > 0)
    quota_target = int(args.episodes_per_town) if quota_mode else 0
    target_towns = (
        set(args.town_filter)
        if args.town_filter
        else (
            {route_file.parents[2].name for route_file in route_files}
            if args.atlas_root is not None
            else set()
        )
    )

    if args.dry_run:
        dry_records: list[dict[str, Any]] = []
        dry_counts: dict[str, int] = {}
        for route_file in route_files:
            meta = _load_route(route_file)
            town = str(args.town_override or meta["town"] or "")

            if quota_mode and town in target_towns:
                if int(dry_counts.get(town, 0)) >= quota_target:
                    continue
                dry_counts[town] = int(dry_counts.get(town, 0)) + 1

            dry_records.append(
                {
                    "route_file": str(route_file),
                    "route_id": meta["route_id"],
                    "town": town,
                    "planned_waypoints": meta["waypoint_count"],
                }
            )

            if quota_mode and target_towns:
                if all(
                    int(dry_counts.get(town_name, 0)) >= quota_target for town_name in target_towns
                ):
                    break

        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "selected_route_count": len(dry_records),
                    "quota_mode": quota_mode,
                    "episodes_per_town_target": quota_target if quota_mode else None,
                    "episodes_per_town_preview_counts": dry_counts if quota_mode else None,
                    "routes": dry_records,
                },
                indent=2,
            )
        )
        return

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(args.config))
    routes_cfg = cfg.get("routes", {})
    route_strategy = str(routes_cfg.get("strategy", "balanced"))
    base_continuation_cfg = parse_route_continuation_config(routes_cfg)

    collector = ExpertDataCollector(
        host=str(args.host),
        port=int(args.port),
        output_dir=output_dir,
    )
    collector.connect()

    manifest = CollectionManifest(output_dir / "collection_manifest.json")

    episode_cfg_cache: dict[str, EpisodeConfig] = {}
    route_cfg_cache: dict[str, RouteConfig] = {}
    records: list[dict[str, Any]] = []
    runs_per_town: dict[str, int] = {}

    for idx, route_file in enumerate(route_files, start=1):
        if quota_mode and target_towns:
            if all(
                int(runs_per_town.get(town_name, 0)) >= quota_target for town_name in target_towns
            ):
                logger.info("Reached per-town episode quota for all target towns; stopping early.")
                break

        meta = _load_route(route_file)
        town = str(args.town_override or meta.get("town") or "")

        if quota_mode and town in target_towns and int(runs_per_town.get(town, 0)) >= quota_target:
            continue

        if not town:
            records.append(
                {
                    "index": int(idx),
                    "route_file": str(route_file),
                    "route_id": meta["route_id"],
                    "town": None,
                    "planned_waypoints": int(meta["waypoint_count"]),
                    "status": "skipped_missing_town",
                    "error_message": "town missing in route and no --town-override provided",
                }
            )
            continue

        if town not in episode_cfg_cache:
            episode_cfg_cache[town] = _build_episode_config(
                cfg=cfg,
                output_dir=output_dir,
                town=town,
                weather=str(args.weather),
                n_vehicles=int(args.vehicles),
                n_walkers=int(args.walkers),
                max_frames=int(args.max_frames),
                spectator=bool(args.spectator),
                debug_draw=bool(args.debug_draw),
                record=bool(args.record),
            )
            route_cfg_cache[town] = _build_route_config(cfg, episode_cfg_cache[town])

        episode_cfg = episode_cfg_cache[town]
        route_cfg = route_cfg_cache[town]

        _ensure_world_for_town(collector, town)
        assert collector.world is not None
        carla_map = collector.world.get_map()

        spacing_m = (
            float(args.lane_change_min_spacing_m)
            if args.lane_change_min_spacing_m is not None
            else float(episode_cfg.expert_config.planner_lane_change_consecutive_window_m)
        )

        route_gap_warn_m = (
            float(args.route_gap_warn_m)
            if args.route_gap_warn_m is not None
            else float(episode_cfg.route_sanitizer_warn_gap_m)
        )
        route_gap_severe_m = (
            float(args.route_gap_severe_m)
            if args.route_gap_severe_m is not None
            else float(episode_cfg.route_sanitizer_severe_gap_m)
        )
        route_gap_report = _route_gap_report(
            carla_map,
            meta["route_waypoints"],
            warn_gap_m=route_gap_warn_m,
            severe_gap_m=route_gap_severe_m,
            max_issues=max(1, int(args.route_gap_max_issues)),
        )
        max_severe_allowed = max(0, int(args.route_gap_max_severe_transitions))
        if bool(args.reject_severe_route_gaps) and int(
            route_gap_report["severe_transition_count"]
        ) > int(max_severe_allowed):
            records.append(
                {
                    "index": int(idx),
                    "route_file": str(route_file),
                    "route_id": meta["route_id"],
                    "town": town,
                    "planned_waypoints": int(meta["waypoint_count"]),
                    "status": "skipped_route_gap_gate",
                    "route_gap_warn_gap_m": float(route_gap_report["warn_gap_m"]),
                    "route_gap_severe_gap_m": float(route_gap_report["severe_gap_m"]),
                    "route_gap_warn_transition_count": int(
                        route_gap_report["warn_transition_count"]
                    ),
                    "route_gap_severe_transition_count": int(
                        route_gap_report["severe_transition_count"]
                    ),
                    "route_gap_max_gap_m": float(route_gap_report["max_gap_m"]),
                    "route_gap_missing_waypoint_projections": int(
                        route_gap_report["missing_waypoint_projections"]
                    ),
                    "route_gap_gate_max_severe_transitions": int(max_severe_allowed),
                    "error_message": json.dumps(route_gap_report.get("issues", [])[:3]),
                }
            )
            logger.warning(
                "[%d/%d] Skipping %s due route-gap gate (severe=%d>%d, max_gap=%.2fm)",
                idx,
                len(route_files),
                route_file,
                int(route_gap_report["severe_transition_count"]),
                int(max_severe_allowed),
                float(route_gap_report["max_gap_m"]),
            )
            continue

        lane_report = _lane_jump_report(carla_map, meta["route_waypoints"], spacing_m)
        if bool(args.reject_multi_lane_jumps) and bool(lane_report.get("invalid", False)):
            records.append(
                {
                    "index": int(idx),
                    "route_file": str(route_file),
                    "route_id": meta["route_id"],
                    "town": town,
                    "planned_waypoints": int(meta["waypoint_count"]),
                    "status": "skipped_lane_jump_gate",
                    "route_gap_warn_gap_m": float(route_gap_report["warn_gap_m"]),
                    "route_gap_severe_gap_m": float(route_gap_report["severe_gap_m"]),
                    "route_gap_warn_transition_count": int(
                        route_gap_report["warn_transition_count"]
                    ),
                    "route_gap_severe_transition_count": int(
                        route_gap_report["severe_transition_count"]
                    ),
                    "route_gap_max_gap_m": float(route_gap_report["max_gap_m"]),
                    "route_gap_missing_waypoint_projections": int(
                        route_gap_report["missing_waypoint_projections"]
                    ),
                    "route_gap_gate_max_severe_transitions": int(max_severe_allowed),
                    "lane_jump_invalid": True,
                    "lane_jump_multi_lane_steps": int(lane_report["multi_lane_steps"]),
                    "lane_jump_rapid_consecutive_lane_changes": int(
                        lane_report["rapid_consecutive_lane_changes"]
                    ),
                    "lane_jump_rapid_lane_change_command_streaks": int(
                        lane_report["rapid_lane_change_command_streaks"]
                    ),
                    "lane_jump_missing_waypoint_projections": int(
                        lane_report["missing_waypoint_projections"]
                    ),
                    "error_message": json.dumps(lane_report.get("sample_issues", [])[:3]),
                }
            )
            logger.warning(
                "[%d/%d] Skipping %s due lane-jump gate (multi=%d rapid=%d cmd_rapid=%d)",
                idx,
                len(route_files),
                route_file,
                int(lane_report["multi_lane_steps"]),
                int(lane_report["rapid_consecutive_lane_changes"]),
                int(lane_report["rapid_lane_change_command_streaks"]),
            )
            continue

        run_rng = np.random.default_rng(int(args.seed) + int(idx))

        continuation_cfg = _effective_continuation_config(base_continuation_cfg, args)
        route_gen: BalancedRouteGenerator | None = None
        continuation_provider: Any | None = None
        continuation_max_chains = 0
        continuation_mode = "off"

        if continuation_cfg.enabled:
            if bool(getattr(episode_cfg.expert_config, "route_tail_extension_enabled", False)):
                # Continuation should trigger on true expert.done() rather than synthetic
                # tail extension waypoints.
                episode_cfg.expert_config.route_tail_extension_enabled = False
            try:
                route_gen = BalancedRouteGenerator(collector.world, route_cfg)
                continuation_provider, continuation_max_chains = _build_continuation_provider(
                    continuation_cfg=continuation_cfg,
                    route_gen=route_gen,
                    rng=run_rng,
                    route_strategy=route_strategy,
                    carla_map=carla_map,
                    min_spacing_m=spacing_m,
                    reject_multi_lane_jumps=bool(args.reject_multi_lane_jumps),
                    reject_anchor_mismatch=bool(args.reject_anchor_mismatch),
                    max_anchor_start_distance_m=float(args.anchor_max_start_distance_m),
                    max_anchor_heading_diff_deg=float(args.anchor_max_heading_diff_deg),
                    max_anchor_lane_delta=max(0, int(args.anchor_max_lane_delta)),
                    max_attempts=int(args.continuation_attempts),
                )
                if continuation_provider is not None:
                    continuation_mode = str(continuation_cfg.mode)
            except Exception as e:
                logger.warning("Continuation setup failed for %s: %s", route_file, e)
                continuation_provider = None
                continuation_max_chains = 0
                continuation_mode = "off"

        logger.info(
            "[%d/%d] Running %s (town=%s, continuation=%s chains=%d preload_wp=%d)",
            idx,
            len(route_files),
            route_file,
            town,
            continuation_mode,
            continuation_max_chains,
            int(getattr(continuation_cfg, "preload_remaining_waypoints", 0)),
        )

        try:
            result = collector.collect_episode(
                episode_cfg,
                route_waypoints=meta["route_waypoints"],
                route_continuation_provider=continuation_provider,
                route_continuation_max_chains=continuation_max_chains,
                route_continuation_preload_remaining_waypoints=int(
                    getattr(continuation_cfg, "preload_remaining_waypoints", 0)
                ),
            )
        except Exception as e:
            records.append(
                {
                    "index": int(idx),
                    "route_file": str(route_file),
                    "route_id": meta["route_id"],
                    "town": town,
                    "planned_waypoints": int(meta["waypoint_count"]),
                    "status": "run_exception",
                    "route_gap_warn_gap_m": float(route_gap_report["warn_gap_m"]),
                    "route_gap_severe_gap_m": float(route_gap_report["severe_gap_m"]),
                    "route_gap_warn_transition_count": int(
                        route_gap_report["warn_transition_count"]
                    ),
                    "route_gap_severe_transition_count": int(
                        route_gap_report["severe_transition_count"]
                    ),
                    "route_gap_max_gap_m": float(route_gap_report["max_gap_m"]),
                    "route_gap_missing_waypoint_projections": int(
                        route_gap_report["missing_waypoint_projections"]
                    ),
                    "route_gap_gate_max_severe_transitions": int(max_severe_allowed),
                    "error_message": str(e),
                    "lane_jump_invalid": bool(lane_report.get("invalid", False)),
                    "lane_jump_multi_lane_steps": int(lane_report["multi_lane_steps"]),
                    "lane_jump_rapid_consecutive_lane_changes": int(
                        lane_report["rapid_consecutive_lane_changes"]
                    ),
                    "lane_jump_rapid_lane_change_command_streaks": int(
                        lane_report["rapid_lane_change_command_streaks"]
                    ),
                    "lane_jump_missing_waypoint_projections": int(
                        lane_report["missing_waypoint_projections"]
                    ),
                    "continuation_mode": continuation_mode,
                    "continuation_max_chains": int(continuation_max_chains),
                    "continuation_preload_remaining_waypoints": int(
                        getattr(continuation_cfg, "preload_remaining_waypoints", 0)
                    ),
                }
            )
            continue

        manifest.add_episode(result)
        manifest.save()
        if quota_mode and town in target_towns:
            runs_per_town[town] = int(runs_per_town.get(town, 0)) + 1

        metrics = _read_episode_metrics(output_dir, result.filename)
        records.append(
            {
                "index": int(idx),
                "route_file": str(route_file),
                "route_id": meta["route_id"],
                "town": town,
                "planned_waypoints": int(meta["waypoint_count"]),
                "status": str(result.status),
                "n_frames": int(result.n_frames),
                "duration_seconds": float(result.duration_seconds),
                "route_completion_max": metrics["route_completion_max"],
                "route_completion_final": metrics["route_completion_final"],
                "pre_junction_lane_change_violation_sum": metrics[
                    "pre_junction_lane_change_violation_sum"
                ],
                "consecutive_lane_change_violation_sum": metrics[
                    "consecutive_lane_change_violation_sum"
                ],
                "roundabout_lane_change_violation_sum": metrics[
                    "roundabout_lane_change_violation_sum"
                ],
                "route_gap_warn_gap_m": float(route_gap_report["warn_gap_m"]),
                "route_gap_severe_gap_m": float(route_gap_report["severe_gap_m"]),
                "route_gap_warn_transition_count": int(route_gap_report["warn_transition_count"]),
                "route_gap_severe_transition_count": int(
                    route_gap_report["severe_transition_count"]
                ),
                "route_gap_max_gap_m": float(route_gap_report["max_gap_m"]),
                "route_gap_missing_waypoint_projections": int(
                    route_gap_report["missing_waypoint_projections"]
                ),
                "route_gap_gate_max_severe_transitions": int(max_severe_allowed),
                "lane_jump_invalid": bool(lane_report.get("invalid", False)),
                "lane_jump_multi_lane_steps": int(lane_report["multi_lane_steps"]),
                "lane_jump_rapid_consecutive_lane_changes": int(
                    lane_report["rapid_consecutive_lane_changes"]
                ),
                "lane_jump_rapid_lane_change_command_streaks": int(
                    lane_report["rapid_lane_change_command_streaks"]
                ),
                "lane_jump_missing_waypoint_projections": int(
                    lane_report["missing_waypoint_projections"]
                ),
                "continuation_mode": continuation_mode,
                "continuation_max_chains": int(continuation_max_chains),
                "continuation_preload_remaining_waypoints": int(
                    getattr(continuation_cfg, "preload_remaining_waypoints", 0)
                ),
                "error_message": result.error_message,
                "filename": result.filename,
                "route_path": result.route_path,
            }
        )

    summary = _build_summary(records)
    if quota_mode:
        summary["episodes_per_town_target"] = int(quota_target)
        summary["episodes_per_town_achieved"] = {
            town_name: int(runs_per_town.get(town_name, 0)) for town_name in sorted(target_towns)
        }

    report_json, report_csv = _write_reports(
        output_dir=output_dir,
        records=records,
        summary=summary,
        json_path=args.report_json,
        csv_path=args.report_csv,
    )

    print(
        json.dumps(
            {
                "summary": summary,
                "report_json": str(report_json),
                "report_csv": str(report_csv),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
