"""Leaderboard-style scoring against hand-computed examples."""

import numpy as np
import pytest

from src.eval import driving_score as D

STRAIGHT = np.stack([np.arange(101.0), np.zeros(101)], axis=1)  # 100 m east, a point a metre
EAST = np.tile([1.0, 0.0], (101, 1))


def test_a_route_scores_completion_times_each_penalty() -> None:
    hits = [D.Infraction("collision_vehicle", 10, 0, 0), D.Infraction("red_light", 20, 0, 0)]
    score = D.score_route(80.0, hits, outside_lanes=10.0, failure="Agent got blocked")
    assert score.infraction_penalty == pytest.approx(0.6 * 0.7 * 0.9)
    assert score.driving_score == pytest.approx(80 * 0.378)
    assert score.status == "Failed - Agent got blocked"
    assert D.score_route(100.0, [], 0.0).status == "Perfect"
    assert D.score_route(100.0, hits[:1], 0.0).status == "Completed"


def test_the_summary_counts_infractions_per_km_driven() -> None:
    hit = [D.Infraction("collision_vehicle", 1, 0, 0)]
    scores = [D.score_route(100.0, hit, 0.0), D.score_route(50.0, [], 0.0, "Agent timed out")]
    summary = D.summarize(scores, [1000.0, 500.0])
    assert summary["km_driven"] == pytest.approx(1.25)
    assert summary["collision_vehicle_per_km"] == pytest.approx(0.8)
    assert summary["driving_score"] == pytest.approx((60.0 + 50.0) / 2)
    assert summary["timed_out_share"] == 0.5


def test_progress_moves_two_points_a_tick_and_completes_near_the_end() -> None:
    progress = D.RouteProgress(STRAIGHT, EAST)
    assert progress.update(np.array([50.5, 0.0])) == 2.0  # the window holds the next two points
    for x in np.arange(2.5, 50.6, 1.0):
        progress.update(np.array([x, 0.0]))
    assert progress.completion == pytest.approx(50.0)
    for x in np.arange(51.5, 100.6, 1.0):
        progress.update(np.array([x, 0.0]))
    assert progress.done  # over 99% passed, within 10 m of the last point


def test_the_route_ends_30_m_off_it_or_after_30_percent_of_it_passed_15_m_off() -> None:
    assert D.RouteDeviation(STRAIGHT).update(np.array([0.0, 30.0]))
    beside = D.RouteDeviation(STRAIGHT)
    assert not any(beside.update(np.array([x, 14.0])) for x in np.arange(1.0, 100.0))
    away = D.RouteDeviation(STRAIGHT)
    ended = [away.update(np.array([x, 20.0])) for x in np.arange(1.0, 40.0)]
    assert ended.index(True) == 30  # 31 m of a 100 m route passed 20 m away


def test_the_timeout_starts_at_300_s_and_grows_with_progress() -> None:
    timeout = D.RouteTimeout()
    assert not timeout.update(300.0, 0.0, 36.0)
    assert timeout.update(300.1, 0.0, 36.0)
    assert not timeout.update(349.9, 50.0, 36.0)  # 50 m at 36 km/h adds 50 / (0.1 * 10 m/s) s


def test_a_car_is_blocked_after_180_s_below_0_1_m_s() -> None:
    blocked = D.Blocked()
    assert not blocked.update(0.0, 0.0) and not blocked.update(180.0, 0.05)
    assert blocked.update(180.05, 0.0)
    assert not blocked.update(200.0, 0.2) and not blocked.update(379.0, 0.0)


def test_a_collision_counts_once() -> None:
    hits = D.CollisionFilter()
    at = np.array([0.0, 0.0])
    assert hits.count(0.0, at, 5.0, 7, "vehicle.tesla.model3") == "collision_vehicle"
    assert hits.count(1.0, at + 6, 5.0, 7, "vehicle.tesla.model3") is None  # same actor, 1 s on
    assert hits.count(1.0, at + 1, 5.0, 8, "walker.pedestrian.0001") is None  # within 5 m
    hits.update(6.0, at + 6)
    assert hits.count(6.0, at + 6, 0.05, 9, "static.prop") is None  # ego nearly at rest
    assert hits.count(6.0, at + 6, 5.0, 0, "static.prop") == "collision_static"
    assert D.collision_kind("static.sidewalk") is None


def test_leaving_a_stop_sign_without_stopping_runs_it_once() -> None:
    stops = D.StopSigns()
    assert stops.update({3}, 3, 5.0) is None and stops.update({3}, None, 2.0) is None
    assert stops.update(set(), None, 5.0) == 3
    assert stops.update({3}, 3, 5.0) is None and stops.update(set(), None, 5.0) is None
    stopped = D.StopSigns()
    for speed in (5.0, 0.05, 3.0):
        stopped.update({4}, 4, speed)
    assert stopped.update(set(), None, 5.0) is None


def test_a_red_light_runs_when_the_tail_crosses_its_lane_line() -> None:
    line = D.StopLine(1, np.array([10.0, -2.0]), np.array([10.0, 2.0]), 5, -1, np.array([1.0, 0]))
    red = {1: np.array([10.0, 3.0])}
    lights = D.RedLights([line])
    car = np.array([12.5, 0.0])  # tail from 1.92 to 3.4 m behind the centre spans x = 10
    assert lights.update(car, 180.0, 2.4, (5, -1), red) is None  # against the lane
    assert lights.update(car, 0.0, 2.4, (5, 1), red) is None  # another lane
    assert lights.update(car, 0.0, 2.4, (5, -1), {}) is None  # not red
    assert lights.update(car + [0, 15], 0.0, 2.4, (5, -1), red) is None  # light over 15 m away
    assert lights.update(car - [1.0, 0], 0.0, 2.4, (5, -1), red) is None  # tail not there yet
    assert lights.update(car, 0.0, 2.4, (5, -1), red) == 1
    assert lights.update(car + [0.2, 0], 0.0, 2.4, (5, -1), red) is None  # counted once


def test_off_road_and_wrong_way_shares_of_the_route() -> None:
    lanes = D.OutsideLanes()
    on = D.LanePlace(1, -1, False, 0.0, 0.5, 3.5)
    assert not lanes.update(on, 0.0, 10.0)
    assert lanes.update(D.LanePlace(1, -1, False, 0.0, 2.3, 3.5), 0.0, 10.0)  # 2.3 > 1.75 + 0.5
    assert lanes.update(D.LanePlace(1, 1, False, 180.0, 0.5, 3.5), 0.0, 10.0)  # lane flips
    assert not lanes.update(D.LanePlace(1, -1, False, 0.0, 0.5, 3.5), 0.0, 10.0)  # flips back
    assert lanes.percentage == pytest.approx(50.0)
    lanes.update(D.LanePlace(9, 1, True, 90.0, 0.5, 3.5), 0.0, 0.0)
    assert lanes.update(D.LanePlace(2, 1, False, 180.0, 0.5, 3.5), 0.0, 0.0)  # left a junction
