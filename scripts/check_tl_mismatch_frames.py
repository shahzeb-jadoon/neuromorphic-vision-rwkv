#!/usr/bin/env python3
"""Check anchor frames for wrong-direction TL stop-line mismatch signatures.

This automates the geometry checks used in cf183fc1 forensic analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    import carla
except ImportError as exc:  # pragma: no cover
    raise SystemExit("CARLA Python API is required.") from exc


def _detect_wsl2_host() -> str:
    import subprocess

    try:
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().split()[2]
    except Exception:
        return "localhost"


def _angdiff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


@dataclass
class FrameCheck:
    frame: int
    tl_id: int
    tl_state: int
    yaw_diff_deg: float | None
    longitudinal_m: float | None
    lateral_m: float | None
    lane_width_m: float | None
    mismatch: bool


@dataclass
class AnchorCheck:
    anchor_frame: int
    exact: FrameCheck
    window_start: int
    window_end: int
    window_mismatches: int
    window_total: int


def _check_frame(
    frame_idx: int,
    pos: np.ndarray,
    rot: np.ndarray,
    tlid: np.ndarray,
    tls: np.ndarray,
    stop: np.ndarray,
    carla_map: Any,
    yaw_threshold: float,
    lateral_margin: float,
) -> FrameCheck:
    s = stop[frame_idx]
    if np.all(np.isclose(s, np.array([-1.0, -1.0, -1.0]))):
        return FrameCheck(
            frame=frame_idx,
            tl_id=int(tlid[frame_idx]),
            tl_state=int(tls[frame_idx]),
            yaw_diff_deg=None,
            longitudinal_m=None,
            lateral_m=None,
            lane_width_m=None,
            mismatch=False,
        )

    ego = carla.Location(
        float(pos[frame_idx, 0]), float(pos[frame_idx, 1]), float(pos[frame_idx, 2])
    )
    stp = carla.Location(float(s[0]), float(s[1]), float(s[2]))
    ego_wp = carla_map.get_waypoint(ego, project_to_road=True, lane_type=carla.LaneType.Driving)
    stop_wp = carla_map.get_waypoint(stp, project_to_road=True, lane_type=carla.LaneType.Driving)

    lane_w = float(ego_wp.lane_width) if ego_wp is not None else 3.5
    ego_yaw = float(rot[frame_idx, 1])
    stop_yaw = float(stop_wp.transform.rotation.yaw) if stop_wp is not None else ego_yaw
    yaw_diff = _angdiff(ego_yaw, stop_yaw)

    dx = float(s[0] - pos[frame_idx, 0])
    dy = float(s[1] - pos[frame_idx, 1])
    yaw_rad = math.radians(ego_yaw)
    fwd = (math.cos(yaw_rad), math.sin(yaw_rad))
    right = (-math.sin(yaw_rad), math.cos(yaw_rad))
    lon = dx * fwd[0] + dy * fwd[1]
    lat = dx * right[0] + dy * right[1]

    mismatch = yaw_diff > yaw_threshold or abs(lat) > (0.5 * lane_w + lateral_margin) or lon <= 0.0

    return FrameCheck(
        frame=frame_idx,
        tl_id=int(tlid[frame_idx]),
        tl_state=int(tls[frame_idx]),
        yaw_diff_deg=float(yaw_diff),
        longitudinal_m=float(lon),
        lateral_m=float(lat),
        lane_width_m=float(lane_w),
        mismatch=bool(mismatch),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check anchor frames for TL wrong-direction mismatch signatures",
    )
    parser.add_argument("episode", type=Path, help="Episode H5 file")
    parser.add_argument("--host", default="auto")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default=None, help="Expected town, e.g. Town10HD")
    parser.add_argument(
        "--auto-load-town",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--anchors",
        type=int,
        nargs="+",
        default=[413, 500, 1580, 1976],
    )
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--yaw-threshold", type=float, default=60.0)
    parser.add_argument("--lateral-margin", type=float, default=1.0)
    parser.add_argument(
        "--skip-out-of-range-anchors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip anchors beyond episode length instead of exiting with error",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.episode.exists():
        raise SystemExit(f"Episode file not found: {args.episode}")

    host = _detect_wsl2_host() if args.host == "auto" else args.host
    client = carla.Client(host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()

    if args.town is not None:
        map_name = world.get_map().name
        if args.auto_load_town and args.town not in map_name:
            print(f"[info] Current world '{map_name}', loading '{args.town}'")
            client.set_timeout(120.0)
            client.load_world(args.town)
            world = client.get_world()
            client.set_timeout(30.0)
            print(f"[info] Loaded world '{world.get_map().name}'")
        elif args.town not in map_name:
            print(f"[warn] Current world '{map_name}', expected '{args.town}'")

    carla_map = world.get_map()

    with h5py.File(args.episode, "r") as h:
        pos = h["position"][:, :3]
        rot = h["rotation"][:, :3]
        tlid = h["traffic_light_actor_id"][:]
        tls = h["traffic_light_state"][:]
        stop = h["traffic_light_stop_line_position"][:, :3]

    n_frames = int(len(pos))
    results: list[AnchorCheck] = []
    skipped_anchors: list[int] = []

    for anchor in args.anchors:
        if anchor < 0 or anchor >= n_frames:
            if args.skip_out_of_range_anchors:
                skipped_anchors.append(anchor)
                continue
            raise SystemExit(f"Anchor frame {anchor} is out of range for episode length {n_frames}")

        i0 = max(0, anchor - args.window)
        i1 = min(n_frames - 1, anchor + args.window)

        window_checks = [
            _check_frame(
                frame_idx=i,
                pos=pos,
                rot=rot,
                tlid=tlid,
                tls=tls,
                stop=stop,
                carla_map=carla_map,
                yaw_threshold=args.yaw_threshold,
                lateral_margin=args.lateral_margin,
            )
            for i in range(i0, i1 + 1)
        ]
        exact = window_checks[anchor - i0]
        window_mismatches = sum(int(v.mismatch) for v in window_checks)

        result = AnchorCheck(
            anchor_frame=anchor,
            exact=exact,
            window_start=i0,
            window_end=i1,
            window_mismatches=window_mismatches,
            window_total=len(window_checks),
        )
        results.append(result)

    print(f"Episode: {args.episode}")
    print(f"Map: {carla_map.name}")
    if skipped_anchors:
        print(f"Skipped anchors out of range for n_frames={n_frames}: {sorted(skipped_anchors)}")
    print(
        "Thresholds: "
        f"yaw_diff>{args.yaw_threshold:.1f}deg, "
        f"|lat|>(0.5*lane_width+{args.lateral_margin:.2f}), lon<=0"
    )

    if not results:
        raise SystemExit("No valid anchors to evaluate after range filtering.")

    suspect_count = 0
    for result in results:
        ex = result.exact
        if ex.mismatch:
            suspect_count += 1
        print(
            f"anchor f{result.anchor_frame}: tl={ex.tl_id} state={ex.tl_state} "
            f"yaw_diff={ex.yaw_diff_deg} lon={ex.longitudinal_m} "
            f"lat={ex.lateral_m} lane_w={ex.lane_width_m} mismatch={ex.mismatch}"
        )
        print(
            f"  window[{result.window_start},{result.window_end}] "
            f"mismatches={result.window_mismatches}/{result.window_total}"
        )

    if args.json_out is not None:
        payload = {
            "episode": str(args.episode),
            "map": carla_map.name,
            "n_frames": n_frames,
            "skipped_anchors": skipped_anchors,
            "yaw_threshold": args.yaw_threshold,
            "lateral_margin": args.lateral_margin,
            "results": [asdict(r) for r in results],
            "suspect_anchor_count": suspect_count,
        }
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote JSON: {args.json_out}")

    if suspect_count > 0:
        print(f"FAIL: {suspect_count} anchor frame(s) flagged as mismatch.")
        sys.exit(2)

    print("PASS: no anchor frames flagged as mismatch.")


if __name__ == "__main__":
    main()
