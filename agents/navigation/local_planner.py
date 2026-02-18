# Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
# Vendored from CARLA 0.9.16 PythonAPI/carla/agents/navigation/local_planner.py
# Only the RoadOption enum is included — the full LocalPlanner class is not
# needed for route generation.

"""Minimal local_planner stub providing the RoadOption enum."""

from enum import IntEnum


class RoadOption(IntEnum):
    """
    RoadOption represents the possible topological configurations
    when moving from a segment of lane to another.
    """

    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6
