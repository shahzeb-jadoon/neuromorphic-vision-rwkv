#!/usr/bin/env python3
"""Recreate selected town episode counts with reproducible collection settings.

This wrapper avoids fragile multiline shell commands and narrows collection to
specific towns + episode counts with a single traffic/weather profile per town.

Example:
    python scripts/recreate_target_episodes.py \
      --town-episodes Town10HD:1 Town06:2 \
      --reference-manifest data/test_collection/collection_manifest.json \
      --output-dir data/replay_targets
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_town_episode_spec(spec: str) -> tuple[str, int]:
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            f"Invalid --town-episodes value '{spec}'. Expected format TownName:Count"
        )
    town, count_str = spec.split(":", 1)
    town = town.strip()
    if not town:
        raise argparse.ArgumentTypeError(f"Invalid town in spec '{spec}'")
    try:
        count = int(count_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid episode count in spec '{spec}': '{count_str}'"
        ) from exc
    if count < 1:
        raise argparse.ArgumentTypeError(f"Episode count must be >= 1 in spec '{spec}'")
    return town, count


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return data


def _read_manifest_profiles(
    manifest_path: Path,
) -> dict[str, tuple[str, int, int]]:
    """Return per-town dominant (weather, n_vehicles, n_walkers) profile."""
    if not manifest_path.exists():
        return {}

    payload = json.loads(manifest_path.read_text())
    rows = payload.get("episodes", [])
    per_town: dict[str, list[tuple[str, int, int]]] = defaultdict(list)

    for row in rows:
        if row.get("status") != "completed":
            continue
        town = row.get("town")
        weather = row.get("weather")
        if not town or not weather:
            continue
        n_vehicles = int(row.get("n_vehicles", 0))
        n_walkers = int(row.get("n_walkers", 0))
        per_town[str(town)].append((str(weather), n_vehicles, n_walkers))

    dominant: dict[str, tuple[str, int, int]] = {}
    for town, combos in per_town.items():
        if not combos:
            continue
        dominant[town] = Counter(combos).most_common(1)[0][0]
    return dominant


def _run_with_tee(cmd: list[str], log_path: Path, dry_run: bool) -> None:
    logger.info("Command: %s", " ".join(cmd))
    logger.info("Log: %s", log_path)
    if dry_run:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recreate selected towns with explicit episode counts",
    )
    parser.add_argument(
        "--town-episodes",
        type=_parse_town_episode_spec,
        nargs="+",
        required=True,
        help="Town:episode_count entries, e.g. Town10HD:1 Town06:2",
    )
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        default=Path("data/test_collection/collection_manifest.json"),
        help="Manifest to infer weather/traffic profile per town",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/collection.yaml"),
        help="Base collection config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/replay_targets"),
        help="Output directory for recreated episodes",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to run scripts/collect_data.py",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--route-strategy",
        choices=["random", "balanced"],
        default="balanced",
    )
    parser.add_argument("--target-left", type=float, default=0.15)
    parser.add_argument("--target-right", type=float, default=0.15)
    parser.add_argument("--max-frames", type=int, default=4000)
    parser.add_argument("--host", type=str, default="auto")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument(
        "--traffic-light-stop-distance",
        type=float,
        default=None,
        help=(
            "Desired ego-center distance (m) to TL stop-line at full stop. "
            "If omitted, use base config/default."
        ),
    )
    parser.add_argument(
        "--record",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CARLA recorder for exact replay",
    )
    parser.add_argument(
        "--debug-draw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable in-world debug drawing",
    )
    parser.add_argument(
        "--spectator",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable spectator follow camera during collection",
    )
    parser.add_argument(
        "--default-weather",
        type=str,
        default="ClearNoon",
        help="Fallback weather if no profile found in reference manifest",
    )
    parser.add_argument(
        "--default-vehicles",
        type=int,
        default=20,
        help="Fallback vehicle count if no profile found",
    )
    parser.add_argument(
        "--default-walkers",
        type=int,
        default=10,
        help="Fallback walker count if no profile found",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated config + command without executing",
    )
    args = parser.parse_args()

    base_cfg = _load_yaml(args.base_config)
    profiles = _read_manifest_profiles(args.reference_manifest)

    logger.info("Reference manifest: %s", args.reference_manifest)
    logger.info("Base config: %s", args.base_config)
    logger.info("Output dir: %s", args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = args.output_dir / "recreate_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    plan: dict[str, Any] = {
        "reference_manifest": str(args.reference_manifest),
        "base_config": str(args.base_config),
        "output_dir": str(args.output_dir),
        "runs": [],
    }

    for town, episode_count in args.town_episodes:
        weather, n_vehicles, n_walkers = profiles.get(
            town,
            (args.default_weather, args.default_vehicles, args.default_walkers),
        )

        cfg = copy.deepcopy(base_cfg)
        cfg["towns"] = [town]
        cfg["weathers"] = [weather]
        cfg.setdefault("traffic", {})["vehicles"] = [n_vehicles]
        cfg.setdefault("traffic", {})["walkers"] = [n_walkers]
        cfg.setdefault("output", {})["directory"] = str(args.output_dir)
        cfg.setdefault("seed", args.seed)
        if args.traffic_light_stop_distance is not None:
            cfg.setdefault("expert", {})["traffic_light_stop_distance"] = float(
                args.traffic_light_stop_distance
            )

        town_cfg = cfg_dir / f"recreate_{town}.yaml"
        town_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))

        cmd = [
            args.python,
            "scripts/collect_data.py",
            "--config",
            str(town_cfg),
            "--output",
            str(args.output_dir),
            "--town",
            town,
            "--weather",
            weather,
            "--episodes",
            str(episode_count),
            "--seed",
            str(args.seed),
            "--route-strategy",
            args.route_strategy,
            "--target-left",
            str(args.target_left),
            "--target-right",
            str(args.target_right),
            "--max-frames",
            str(args.max_frames),
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
        if args.record:
            cmd.append("--record")
        if args.debug_draw:
            cmd.append("--debug-draw")
        if args.spectator:
            cmd.append("--spectator")

        log_path = args.output_dir / f"recreate_{town}.log"
        plan["runs"].append(
            {
                "town": town,
                "episodes": episode_count,
                "weather": weather,
                "n_vehicles": n_vehicles,
                "n_walkers": n_walkers,
                "config_path": str(town_cfg),
                "log_path": str(log_path),
                "command": cmd,
                "traffic_light_stop_distance": args.traffic_light_stop_distance,
            }
        )

        logger.info(
            "Planned run: town=%s episodes=%d weather=%s traffic=%d/%d",
            town,
            episode_count,
            weather,
            n_vehicles,
            n_walkers,
        )
        _run_with_tee(cmd, log_path, args.dry_run)

    plan_path = args.output_dir / "recreate_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    logger.info("Wrote plan: %s", plan_path)
    logger.info("All targeted recreation runs completed.")


if __name__ == "__main__":
    main()
