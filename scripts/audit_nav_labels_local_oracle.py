"""Audit nav labels using a local GRP oracle near junction windows.

This script performs local oracle checks only around junction windows. It is
intended for ongoing quality audits and avoids full-route endpoint relabeling.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import h5py

logger = logging.getLogger(__name__)


def canonical_option_name(option_name: str) -> str:
    normalized = str(option_name).strip().replace("_", "").replace(" ", "").lower()
    mapping = {
        "left": "LEFT",
        "right": "RIGHT",
        "straight": "STRAIGHT",
        "lanefollow": "LANEFOLLOW",
        "changelaneleft": "CHANGELANELEFT",
        "changelaneright": "CHANGELANERIGHT",
        "void": "VOID",
    }
    return mapping.get(normalized, str(option_name).strip().upper())


def load_max_reached_index(h5_path: Path) -> int | None:
    if not h5_path.exists():
        return None

    try:
        with h5py.File(h5_path, "r") as handle:
            if "wp_buffer_head_index" not in handle:
                return None
            head_indices = handle["wp_buffer_head_index"][:]
    except Exception as exc:
        logger.warning("Failed reading %s: %s", h5_path, exc)
        return None

    valid = [int(v) for v in head_indices if int(v) >= 0]
    if not valid:
        return None
    return max(valid)


def compute_junction_window_indices(
    waypoints: list[dict[str, Any]],
    max_index: int,
    radius: int,
    stride: int,
) -> list[int]:
    if not waypoints or max_index < 0:
        return []

    radius = max(0, int(radius))
    stride = max(1, int(stride))

    flagged: set[int] = set()
    for idx in range(0, max_index + 1):
        if bool(waypoints[idx].get("is_junction", False)):
            lo = max(0, idx - radius)
            hi = min(max_index, idx + radius)
            for cand in range(lo, hi + 1):
                flagged.add(cand)

    ordered = sorted(flagged)
    if stride > 1:
        ordered = ordered[::stride]
    return ordered


def audit_episode_waypoints(
    *,
    waypoints: list[dict[str, Any]],
    max_reached_index: int,
    grp: Any,
    location_factory: Callable[[float, float, float], Any],
    lookahead_waypoints: int,
    junction_window_radius: int,
    stride: int,
    max_mismatch_samples: int = 16,
) -> dict[str, Any]:
    if not waypoints:
        return {
            "checked": 0,
            "matches": 0,
            "mismatches": 0,
            "oracle_errors": 0,
            "checked_indices": [],
            "mismatch_samples": [],
        }

    max_reached_index = max(0, min(int(max_reached_index), len(waypoints) - 1))
    lookahead_waypoints = max(1, int(lookahead_waypoints))

    check_indices = compute_junction_window_indices(
        waypoints,
        max_reached_index,
        radius=junction_window_radius,
        stride=stride,
    )

    checked = 0
    matches = 0
    mismatches = 0
    oracle_errors = 0
    mismatch_samples: list[dict[str, Any]] = []

    for idx in check_indices:
        dst_idx = min(max_reached_index, idx + lookahead_waypoints)
        if dst_idx <= idx:
            continue

        origin = waypoints[idx]
        target = waypoints[dst_idx]

        origin_loc = location_factory(
            float(origin["x"]),
            float(origin["y"]),
            float(origin.get("z", 0.0)),
        )
        target_loc = location_factory(
            float(target["x"]),
            float(target["y"]),
            float(target.get("z", 0.0)),
        )

        try:
            local_route = grp.trace_route(origin_loc, target_loc)
        except Exception:
            oracle_errors += 1
            continue

        if not local_route:
            oracle_errors += 1
            continue

        oracle_name = canonical_option_name(
            getattr(local_route[0][1], "name", str(local_route[0][1]))
        )
        recorded_name = canonical_option_name(str(origin.get("road_option", "LANEFOLLOW")))

        checked += 1
        if oracle_name == recorded_name:
            matches += 1
            continue

        mismatches += 1
        if len(mismatch_samples) < max_mismatch_samples:
            mismatch_samples.append(
                {
                    "index": int(idx),
                    "recorded": recorded_name,
                    "oracle": oracle_name,
                    "road_id": int(origin.get("road_id", -1)),
                    "lane_id": int(origin.get("lane_id", 0)),
                    "is_junction": bool(origin.get("is_junction", False)),
                    "lookahead_to_index": int(dst_idx),
                }
            )

    return {
        "checked": checked,
        "matches": matches,
        "mismatches": mismatches,
        "oracle_errors": oracle_errors,
        "checked_indices": check_indices,
        "mismatch_samples": mismatch_samples,
    }


def _resolve_host(host: str) -> str:
    if host != "auto":
        return host

    try:
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            check=True,
            capture_output=True,
            text=True,
        )
        parts = result.stdout.strip().split()
        if len(parts) >= 3:
            return parts[2]
    except Exception:
        pass

    return "localhost"


def _episode_sidecar_path(collection_dir: Path, entry: dict[str, Any]) -> Path:
    route_path = entry.get("route_path")
    if route_path:
        return collection_dir / str(route_path)

    ep_hash = str(entry.get("episode_hash") or entry.get("hash") or "")
    if ep_hash:
        return collection_dir / f"episode_{ep_hash}.route.json"

    filename = str(entry.get("filename", ""))
    if filename.startswith("episode_") and filename.endswith(".h5"):
        return collection_dir / filename.replace(".h5", ".route.json")

    return collection_dir / ""


def _episode_h5_path(collection_dir: Path, entry: dict[str, Any]) -> Path:
    filename = entry.get("filename")
    if filename:
        return collection_dir / str(filename)

    ep_hash = str(entry.get("episode_hash") or entry.get("hash") or "")
    if ep_hash:
        return collection_dir / f"episode_{ep_hash}.h5"

    return collection_dir / ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit nav labels using local GRP windows around junctions",
    )
    parser.add_argument("--collection-dir", type=str, default="data/test_collection")
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Override manifest path (default: <collection-dir>/collection_manifest.json)",
    )
    parser.add_argument("--host", type=str, default="auto")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--lookahead-waypoints", type=int, default=12)
    parser.add_argument("--junction-window", type=int, default=6)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Default: <collection-dir>/analysis/local_oracle_nav_audit.json",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    collection_dir = Path(args.collection_dir)
    manifest_path = (
        Path(args.manifest) if args.manifest else collection_dir / "collection_manifest.json"
    )

    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return 1

    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)

    episodes = list(manifest.get("episodes", []))
    if args.max_episodes and args.max_episodes > 0:
        episodes = episodes[: int(args.max_episodes)]
    if not episodes:
        logger.error("No episodes found in manifest: %s", manifest_path)
        return 1

    try:
        import carla
        from agents.navigation.global_route_planner import GlobalRoutePlanner
    except Exception as exc:
        logger.error("CARLA imports failed: %s", exc)
        return 1

    host = _resolve_host(str(args.host))
    logger.info("Connecting to CARLA at %s:%d", host, int(args.port))

    try:
        client = carla.Client(host, int(args.port))
        client.set_timeout(60.0)
        world = client.get_world()
    except Exception as exc:
        logger.error("Failed to connect to CARLA: %s", exc)
        return 1

    planners: dict[str, Any] = {}
    per_episode: list[dict[str, Any]] = []

    totals = {
        "episodes_processed": 0,
        "episodes_skipped": 0,
        "checked": 0,
        "matches": 0,
        "mismatches": 0,
        "oracle_errors": 0,
    }

    for entry in episodes:
        town = str(entry.get("town", ""))
        if not town:
            totals["episodes_skipped"] += 1
            continue

        sidecar_path = _episode_sidecar_path(collection_dir, entry)
        h5_path = _episode_h5_path(collection_dir, entry)

        if not sidecar_path.exists():
            totals["episodes_skipped"] += 1
            continue

        with open(sidecar_path, encoding="utf-8") as handle:
            sidecar = json.load(handle)
        waypoints = list(sidecar.get("waypoints", []))

        if not waypoints:
            totals["episodes_skipped"] += 1
            continue

        current_map = str(world.get_map().name)
        if town not in current_map:
            logger.info("Loading %s for local-oracle audit", town)
            try:
                client.load_world(town)
                time.sleep(2.0)
                world = client.get_world()
            except Exception as exc:
                logger.warning("Failed to load %s: %s", town, exc)
                totals["episodes_skipped"] += 1
                continue

        map_name = str(world.get_map().name)
        if map_name not in planners:
            planners[map_name] = GlobalRoutePlanner(world.get_map(), sampling_resolution=2.0)
        grp = planners[map_name]

        max_reached = load_max_reached_index(h5_path)
        if max_reached is None:
            max_reached = len(waypoints) - 1
        else:
            max_reached = min(int(max_reached), len(waypoints) - 1)

        report = audit_episode_waypoints(
            waypoints=waypoints,
            max_reached_index=max_reached,
            grp=grp,
            location_factory=lambda x, y, z: carla.Location(x=x, y=y, z=z),
            lookahead_waypoints=int(args.lookahead_waypoints),
            junction_window_radius=int(args.junction_window),
            stride=int(args.stride),
        )

        checked = int(report["checked"])
        matches = int(report["matches"])
        mismatches = int(report["mismatches"])
        oracle_errors = int(report["oracle_errors"])
        mismatch_rate = (float(mismatches) / float(checked)) if checked > 0 else 0.0

        episode_hash = str(entry.get("episode_hash") or entry.get("hash") or "unknown")
        per_episode.append(
            {
                "episode_hash": episode_hash,
                "town": town,
                "checked": checked,
                "matches": matches,
                "mismatches": mismatches,
                "oracle_errors": oracle_errors,
                "mismatch_rate": mismatch_rate,
                "max_reached_index": int(max_reached),
                "checked_window_count": len(report.get("checked_indices", [])),
                "mismatch_samples": report.get("mismatch_samples", []),
            }
        )

        totals["episodes_processed"] += 1
        totals["checked"] += checked
        totals["matches"] += matches
        totals["mismatches"] += mismatches
        totals["oracle_errors"] += oracle_errors

        logger.info(
            "%s (%s): checked=%d mismatches=%d (%.1f%%) oracle_errors=%d",
            episode_hash,
            town,
            checked,
            mismatches,
            mismatch_rate * 100.0,
            oracle_errors,
        )

    overall_checked = int(totals["checked"])
    overall_mismatches = int(totals["mismatches"])
    overall_rate = (
        float(overall_mismatches) / float(overall_checked) if overall_checked > 0 else 0.0
    )

    summary = {
        "collection_dir": str(collection_dir),
        "manifest_path": str(manifest_path),
        "lookahead_waypoints": int(args.lookahead_waypoints),
        "junction_window": int(args.junction_window),
        "stride": int(args.stride),
        "totals": totals,
        "overall_mismatch_rate": overall_rate,
        "episodes": per_episode,
    }

    output_path = (
        Path(args.output_json)
        if args.output_json
        else (collection_dir / "analysis" / "local_oracle_nav_audit.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info(
        "Audit summary: processed=%d checked=%d mismatches=%d (%.2f%%)",
        int(totals["episodes_processed"]),
        overall_checked,
        overall_mismatches,
        overall_rate * 100.0,
    )
    logger.info("Wrote report: %s", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
