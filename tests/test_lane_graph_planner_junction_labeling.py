"""Regression tests for lane-graph junction labeling and turn classification."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("carla")

from agents.navigation.lane_graph_planner import LaneGraphPlanner, SegmentNode  # noqa: E402
from agents.navigation.local_planner import RoadOption  # noqa: E402


def _node(
    seg_id: str,
    *,
    road_id: int,
    section_id: int,
    lane_id: int,
    is_junction: bool,
    mid_x: float,
    mid_y: float,
) -> SegmentNode:
    return SegmentNode(
        segment_id=seg_id,
        road_id=road_id,
        section_id=section_id,
        lane_id=lane_id,
        is_junction=is_junction,
        entry_xyz=(mid_x, mid_y, 0.0),
        exit_xyz=(mid_x, mid_y, 0.0),
        entry_yaw=0.0,
        exit_yaw=0.0,
        centerline=((mid_x, mid_y, 0.0, 0.0),),
        length_m=1.0,
    )


def test_segment_is_junction_requires_strong_evidence() -> None:
    wp_start = SimpleNamespace(is_junction=False)
    wp_end = SimpleNamespace(is_junction=False)
    sampled = [
        SimpleNamespace(is_junction=True),
        SimpleNamespace(is_junction=False),
        SimpleNamespace(is_junction=False),
        SimpleNamespace(is_junction=False),
    ]

    is_junction = LaneGraphPlanner._segment_is_junction(wp_start, wp_end, sampled)

    assert is_junction is False


def test_segment_is_junction_true_when_sampled_waypoints_are_dominant() -> None:
    wp_start = SimpleNamespace(is_junction=False)
    wp_end = SimpleNamespace(is_junction=False)
    sampled = [
        SimpleNamespace(is_junction=True),
        SimpleNamespace(is_junction=True),
        SimpleNamespace(is_junction=True),
        SimpleNamespace(is_junction=False),
    ]

    is_junction = LaneGraphPlanner._segment_is_junction(wp_start, wp_end, sampled)

    assert is_junction is True


def test_classify_junction_turn_treats_same_lane_entry_exit_as_straight() -> None:
    planner = LaneGraphPlanner.__new__(LaneGraphPlanner)
    planner._turn_threshold_deg = 35.0
    planner._segments = {
        "approach": _node(
            "approach",
            road_id=10,
            section_id=0,
            lane_id=1,
            is_junction=False,
            mid_x=0.0,
            mid_y=0.0,
        ),
        "junction": _node(
            "junction",
            road_id=99,
            section_id=0,
            lane_id=5,
            is_junction=True,
            mid_x=10.0,
            mid_y=0.0,
        ),
        "exit": _node(
            "exit",
            road_id=10,
            section_id=0,
            lane_id=1,
            is_junction=False,
            mid_x=10.0,
            mid_y=10.0,
        ),
    }

    command = planner._classify_junction_turn(["approach", "junction", "exit"], 1)

    assert command == RoadOption.STRAIGHT
