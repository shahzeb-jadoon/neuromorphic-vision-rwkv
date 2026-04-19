"""Unit tests for local GRP-oracle nav-label audit helpers."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from scripts.audit_nav_labels_local_oracle import (
    audit_episode_waypoints,
    canonical_option_name,
    compute_junction_window_indices,
    load_max_reached_index,
)


class _FakeRoadOption:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeGRP:
    def __init__(self, command_by_x: dict[int, str], error_x: set[int] | None = None) -> None:
        self.command_by_x = command_by_x
        self.error_x = error_x or set()

    def trace_route(
        self,
        origin: tuple[float, float, float],
        _target: tuple[float, float, float],
    ) -> list[tuple[None, _FakeRoadOption]]:
        origin_x = int(origin[0])
        if origin_x in self.error_x:
            raise RuntimeError("oracle route failed")
        cmd = self.command_by_x.get(origin_x, "LANEFOLLOW")
        return [(None, _FakeRoadOption(cmd))]


def _waypoint(idx: int, road_option: str, is_junction: bool) -> dict[str, object]:
    return {
        "index": idx,
        "x": float(idx),
        "y": 0.0,
        "z": 0.0,
        "road_option": road_option,
        "road_id": 100 + idx,
        "lane_id": 1,
        "is_junction": is_junction,
    }


def test_canonical_option_name_handles_variants() -> None:
    assert canonical_option_name("Left") == "LEFT"
    assert canonical_option_name("LANEFOLLOW") == "LANEFOLLOW"
    assert canonical_option_name("ChangeLaneRight") == "CHANGELANERIGHT"


def test_compute_junction_window_indices_applies_radius_and_stride() -> None:
    waypoints = [_waypoint(i, "LANEFOLLOW", is_junction=(i == 4)) for i in range(10)]

    indices = compute_junction_window_indices(waypoints, max_index=9, radius=2, stride=2)

    assert indices == [2, 4, 6]


def test_load_max_reached_index_reads_dataset(tmp_path: Path) -> None:
    h5_path = tmp_path / "episode.h5"
    with h5py.File(h5_path, "w") as h:
        h.create_dataset("wp_buffer_head_index", data=np.array([0, 2, -1, 7, 6], dtype=np.int32))

    assert load_max_reached_index(h5_path) == 7


def test_audit_episode_waypoints_compares_local_oracle_near_junctions() -> None:
    waypoints = [
        _waypoint(0, "LANEFOLLOW", False),
        _waypoint(1, "LANEFOLLOW", False),
        _waypoint(2, "LEFT", True),
        _waypoint(3, "RIGHT", False),
        _waypoint(4, "LANEFOLLOW", False),
    ]
    fake_grp = _FakeGRP(
        command_by_x={1: "LANEFOLLOW", 2: "LEFT", 3: "LANEFOLLOW"},
    )

    report = audit_episode_waypoints(
        waypoints=waypoints,
        max_reached_index=4,
        grp=fake_grp,
        location_factory=lambda x, y, z: (x, y, z),
        lookahead_waypoints=1,
        junction_window_radius=1,
        stride=1,
    )

    assert report["checked"] == 3
    assert report["matches"] == 2
    assert report["mismatches"] == 1
    assert report["oracle_errors"] == 0
    assert report["mismatch_samples"][0]["index"] == 3
    assert report["mismatch_samples"][0]["recorded"] == "RIGHT"
    assert report["mismatch_samples"][0]["oracle"] == "LANEFOLLOW"


def test_audit_episode_waypoints_counts_oracle_errors() -> None:
    waypoints = [_waypoint(i, "LANEFOLLOW", is_junction=(i == 1)) for i in range(4)]
    fake_grp = _FakeGRP(command_by_x={0: "LANEFOLLOW", 1: "LEFT", 2: "LANEFOLLOW"}, error_x={1})

    report = audit_episode_waypoints(
        waypoints=waypoints,
        max_reached_index=3,
        grp=fake_grp,
        location_factory=lambda x, y, z: (x, y, z),
        lookahead_waypoints=1,
        junction_window_radius=1,
        stride=1,
    )

    assert report["checked"] == 2
    assert report["oracle_errors"] == 1
