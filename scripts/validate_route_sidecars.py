"""Offline validation and optional repair for episode route sidecars.

This tool inspects ``*.route.json`` sidecars emitted by the collector,
reports suspicious route transitions, and can optionally emit repaired
sidecars with interpolated synthetic waypoints for large gaps.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _wrap_angle_deg(angle: float) -> float:
    """Normalize an angle to [-180, 180)."""
    wrapped = (angle + 180.0) % 360.0 - 180.0
    return float(wrapped)


def _distance3(a: dict[str, Any], b: dict[str, Any]) -> float:
    dx = float(a.get("x", 0.0)) - float(b.get("x", 0.0))
    dy = float(a.get("y", 0.0)) - float(b.get("y", 0.0))
    dz = float(a.get("z", 0.0)) - float(b.get("z", 0.0))
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))


def analyze_route_sidecar(
    sidecar: dict[str, Any],
    *,
    warn_gap_m: float,
    severe_gap_m: float,
    wrong_way_yaw_diff_deg: float,
    max_issues: int,
) -> dict[str, Any]:
    """Analyze a route sidecar for suspicious transitions."""
    waypoints = list(sidecar.get("waypoints", []))
    issues: list[dict[str, Any]] = []

    warn_transition_count = 0
    severe_transition_count = 0
    wrong_way_transition_count = 0
    max_gap_m = 0.0

    for idx in range(len(waypoints) - 1):
        wp0 = waypoints[idx]
        wp1 = waypoints[idx + 1]

        gap_m = _distance3(wp0, wp1)
        max_gap_m = max(max_gap_m, gap_m)
        same_command = str(wp0.get("road_option", "")) == str(wp1.get("road_option", ""))
        road_changed = int(wp0.get("road_id", -1)) != int(wp1.get("road_id", -1))

        yaw0 = float(wp0.get("yaw", 0.0))
        yaw1 = float(wp1.get("yaw", 0.0))
        yaw_diff = abs(_wrap_angle_deg(yaw1 - yaw0))
        wrong_way = (
            yaw_diff >= wrong_way_yaw_diff_deg
            and int(wp0.get("road_id", -1)) == int(wp1.get("road_id", -1))
            and int(wp0.get("lane_id", -1)) == int(wp1.get("lane_id", -1))
        )

        flagged_gap = gap_m >= warn_gap_m and (same_command or road_changed)
        if not flagged_gap and not wrong_way:
            continue

        if flagged_gap:
            warn_transition_count += 1
            if gap_m >= severe_gap_m:
                severe_transition_count += 1
        if wrong_way:
            wrong_way_transition_count += 1

        if len(issues) < max(1, int(max_issues)):
            issues.append(
                {
                    "from_index": idx,
                    "to_index": idx + 1,
                    "gap_m": round(gap_m, 3),
                    "yaw_diff_deg": round(yaw_diff, 3),
                    "same_command": bool(same_command),
                    "road_option_from": str(wp0.get("road_option", "")),
                    "road_option_to": str(wp1.get("road_option", "")),
                    "road_id_from": int(wp0.get("road_id", -1)),
                    "road_id_to": int(wp1.get("road_id", -1)),
                    "lane_id_from": int(wp0.get("lane_id", -1)),
                    "lane_id_to": int(wp1.get("lane_id", -1)),
                    "flags": {
                        "gap": bool(flagged_gap),
                        "wrong_way": bool(wrong_way),
                    },
                }
            )

    return {
        "warn_gap_m": float(warn_gap_m),
        "severe_gap_m": float(severe_gap_m),
        "wrong_way_yaw_diff_deg": float(wrong_way_yaw_diff_deg),
        "warn_transition_count": int(warn_transition_count),
        "severe_transition_count": int(severe_transition_count),
        "wrong_way_transition_count": int(wrong_way_transition_count),
        "max_gap_m": round(max_gap_m, 3),
        "issues": issues,
    }


def repair_route_sidecar(
    sidecar: dict[str, Any],
    *,
    repair_gap_m: float,
    repair_step_m: float,
) -> tuple[dict[str, Any], int]:
    """Return a repaired sidecar payload with interpolated synthetic waypoints."""
    waypoints = list(sidecar.get("waypoints", []))
    if len(waypoints) < 2:
        return sidecar, 0

    repaired: list[dict[str, Any]] = []
    inserted = 0
    step = max(0.25, float(repair_step_m))

    for idx in range(len(waypoints) - 1):
        wp0 = dict(waypoints[idx])
        wp1 = dict(waypoints[idx + 1])
        repaired.append(wp0)

        gap_m = _distance3(wp0, wp1)
        if gap_m <= float(repair_gap_m):
            continue

        n_insert = max(1, int(gap_m / step) - 1)
        for k in range(1, n_insert + 1):
            t = k / (n_insert + 1)
            synth = {
                "x": float(wp0.get("x", 0.0))
                + (float(wp1.get("x", 0.0)) - float(wp0.get("x", 0.0))) * t,
                "y": float(wp0.get("y", 0.0))
                + (float(wp1.get("y", 0.0)) - float(wp0.get("y", 0.0))) * t,
                "z": float(wp0.get("z", 0.0))
                + (float(wp1.get("z", 0.0)) - float(wp0.get("z", 0.0))) * t,
                "pitch": float(wp0.get("pitch", 0.0))
                + (float(wp1.get("pitch", 0.0)) - float(wp0.get("pitch", 0.0))) * t,
                "yaw": float(wp0.get("yaw", 0.0))
                + _wrap_angle_deg(float(wp1.get("yaw", 0.0)) - float(wp0.get("yaw", 0.0))) * t,
                "roll": float(wp0.get("roll", 0.0))
                + (float(wp1.get("roll", 0.0)) - float(wp0.get("roll", 0.0))) * t,
                "road_option": str(wp1.get("road_option", "LANEFOLLOW")),
                "road_id": (
                    int(wp0.get("road_id", -1))
                    if int(wp0.get("road_id", -1)) == int(wp1.get("road_id", -1))
                    else -1
                ),
                "lane_id": (
                    int(wp0.get("lane_id", -1))
                    if int(wp0.get("lane_id", -1)) == int(wp1.get("lane_id", -1))
                    else -1
                ),
                "is_junction": bool(wp0.get("is_junction", False) or wp1.get("is_junction", False)),
                "synthetic": True,
                "repair_reason": "large_gap_interpolation",
            }
            repaired.append(synth)
            inserted += 1

    repaired.append(dict(waypoints[-1]))

    for idx, wp in enumerate(repaired):
        wp["index"] = idx

    out = dict(sidecar)
    out["waypoints"] = repaired
    out["n_waypoints"] = len(repaired)
    return out, inserted


def _iter_sidecars(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.glob("*.route.json") if p.is_file())


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate/repair episode route sidecars.")
    parser.add_argument("input", type=Path, help="Path to a .route.json file or directory")
    parser.add_argument("--warn-gap-m", type=float, default=12.0)
    parser.add_argument("--severe-gap-m", type=float, default=30.0)
    parser.add_argument("--wrong-way-yaw-diff-deg", type=float, default=150.0)
    parser.add_argument("--max-issues", type=int, default=64)
    parser.add_argument("--repair", action="store_true", help="Emit repaired sidecar files")
    parser.add_argument("--repair-gap-m", type=float, default=8.0)
    parser.add_argument("--repair-step-m", type=float, default=2.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for repaired sidecars (default: alongside source)",
    )
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sidecars = _iter_sidecars(args.input)
    if not sidecars:
        print("No route sidecars found.")
        return 1

    summaries: list[dict[str, Any]] = []
    total_warn = 0
    total_severe = 0
    total_wrong_way = 0
    total_inserted = 0

    for sidecar_path in sidecars:
        payload = _load_json(sidecar_path)
        report = analyze_route_sidecar(
            payload,
            warn_gap_m=args.warn_gap_m,
            severe_gap_m=args.severe_gap_m,
            wrong_way_yaw_diff_deg=args.wrong_way_yaw_diff_deg,
            max_issues=args.max_issues,
        )
        total_warn += int(report["warn_transition_count"])
        total_severe += int(report["severe_transition_count"])
        total_wrong_way += int(report["wrong_way_transition_count"])

        repaired_path: str | None = None
        inserted = 0
        if args.repair:
            repaired, inserted = repair_route_sidecar(
                payload,
                repair_gap_m=args.repair_gap_m,
                repair_step_m=args.repair_step_m,
            )
            total_inserted += inserted
            if args.output_dir is None:
                out_path = sidecar_path.with_suffix(".repaired.route.json")
            else:
                out_path = args.output_dir / sidecar_path.name.replace(
                    ".route.json", ".repaired.route.json"
                )
            _write_json(out_path, repaired)
            repaired_path = str(out_path)

        summary_entry = {
            "path": str(sidecar_path),
            "episode_hash": payload.get("episode_hash"),
            "n_waypoints": int(payload.get("n_waypoints", len(payload.get("waypoints", [])))),
            "report": report,
            "repair": {
                "enabled": bool(args.repair),
                "inserted_waypoints": int(inserted),
                "output_path": repaired_path,
            },
        }
        summaries.append(summary_entry)

        print(
            f"{sidecar_path.name}: warn={report['warn_transition_count']} "
            f"severe={report['severe_transition_count']} "
            f"wrong_way={report['wrong_way_transition_count']}"
        )

    aggregate = {
        "n_sidecars": len(sidecars),
        "total_warn_transition_count": int(total_warn),
        "total_severe_transition_count": int(total_severe),
        "total_wrong_way_transition_count": int(total_wrong_way),
        "total_inserted_waypoints": int(total_inserted),
        "entries": summaries,
    }

    if args.summary_json is not None:
        _write_json(args.summary_json, aggregate)

    print(
        "Summary: "
        f"sidecars={aggregate['n_sidecars']} "
        f"warn={aggregate['total_warn_transition_count']} "
        f"severe={aggregate['total_severe_transition_count']} "
        f"wrong_way={aggregate['total_wrong_way_transition_count']} "
        f"inserted={aggregate['total_inserted_waypoints']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
