"""Regression tests for LaneGraphPlanner._infer_commands()."""

from __future__ import annotations

import pytest

pytest.importorskip("carla")

from agents.navigation.lane_graph_planner import EdgeKind, LaneGraphPlanner, SegmentNode  # noqa: E402
from agents.navigation.local_planner import RoadOption  # noqa: E402


def _segment(seg_id: str, is_junction: bool, road_id: int) -> SegmentNode:
    return SegmentNode(
        segment_id=seg_id,
        road_id=road_id,
        section_id=0,
        lane_id=1,
        is_junction=is_junction,
        entry_xyz=(0.0, 0.0, 0.0),
        exit_xyz=(1.0, 0.0, 0.0),
        entry_yaw=0.0,
        exit_yaw=0.0,
        centerline=((0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        length_m=1.0,
    )


def _planner(segment_path: list[str], junction_flags: dict[str, bool]) -> LaneGraphPlanner:
    planner = LaneGraphPlanner.__new__(LaneGraphPlanner)
    planner._segments = {
        seg_id: _segment(seg_id, is_junction=flag, road_id=idx + 1)
        for idx, (seg_id, flag) in enumerate(junction_flags.items())
    }
    planner._edge_kind_lookup = {
        (segment_path[i], segment_path[i + 1]): EdgeKind.FORWARD
        for i in range(len(segment_path) - 1)
    }
    planner._classify_junction_turn = lambda _path, _idx: RoadOption.LEFT
    return planner


def test_infer_commands_does_not_spill_turn_to_first_non_junction_exit() -> None:
    path = ["approach", "junction_a", "junction_b", "exit"]
    planner = _planner(
        segment_path=path,
        junction_flags={
            "approach": False,
            "junction_a": True,
            "junction_b": True,
            "exit": False,
        },
    )

    commands = planner._infer_commands(path)

    assert commands == [
        RoadOption.LANEFOLLOW,
        RoadOption.LEFT,
        RoadOption.LEFT,
        RoadOption.LANEFOLLOW,
    ]


def test_infer_commands_no_head_copy_from_second_segment() -> None:
    path = ["approach", "junction_a"]
    planner = _planner(
        segment_path=path,
        junction_flags={
            "approach": False,
            "junction_a": True,
        },
    )

    commands = planner._infer_commands(path)

    assert commands[0] == RoadOption.LANEFOLLOW
    assert commands[1] == RoadOption.LEFT


def test_lane_change_command_clears_active_junction_state() -> None:
    path = ["approach", "junction", "lane_change", "post_change"]
    planner = _planner(
        segment_path=path,
        junction_flags={
            "approach": False,
            "junction": True,
            "lane_change": False,
            "post_change": False,
        },
    )
    planner._edge_kind_lookup[("junction", "lane_change")] = EdgeKind.LANE_CHANGE_RIGHT

    commands = planner._infer_commands(path)

    assert commands == [
        RoadOption.LANEFOLLOW,
        RoadOption.LEFT,
        RoadOption.CHANGELANERIGHT,
        RoadOption.LANEFOLLOW,
    ]
