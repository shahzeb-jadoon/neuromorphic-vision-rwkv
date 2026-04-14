from __future__ import annotations

from scripts.validate_route_sidecars import analyze_route_sidecar, repair_route_sidecar


def _sidecar_payload() -> dict:
    return {
        "episode_hash": "abc123",
        "n_waypoints": 3,
        "waypoints": [
            {
                "index": 0,
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "road_option": "LANEFOLLOW",
                "road_id": 1,
                "lane_id": 1,
                "is_junction": False,
            },
            {
                "index": 1,
                "x": 35.0,
                "y": 0.0,
                "z": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "road_option": "LANEFOLLOW",
                "road_id": 1,
                "lane_id": 1,
                "is_junction": False,
            },
            {
                "index": 2,
                "x": 37.0,
                "y": 0.0,
                "z": 0.0,
                "pitch": 0.0,
                "yaw": 179.0,
                "roll": 0.0,
                "road_option": "LEFT",
                "road_id": 1,
                "lane_id": 1,
                "is_junction": True,
            },
        ],
    }


def test_analyze_route_sidecar_flags_gap_and_wrong_way() -> None:
    report = analyze_route_sidecar(
        _sidecar_payload(),
        warn_gap_m=12.0,
        severe_gap_m=30.0,
        wrong_way_yaw_diff_deg=150.0,
        max_issues=10,
    )

    assert report["warn_transition_count"] == 1
    assert report["severe_transition_count"] == 1
    assert report["wrong_way_transition_count"] == 1
    assert report["max_gap_m"] >= 35.0
    assert len(report["issues"]) >= 1


def test_repair_route_sidecar_inserts_synthetic_waypoints() -> None:
    repaired, inserted = repair_route_sidecar(
        _sidecar_payload(),
        repair_gap_m=8.0,
        repair_step_m=2.0,
    )

    assert inserted > 0
    assert repaired["n_waypoints"] > 3
    assert repaired["waypoints"][0]["index"] == 0
    assert repaired["waypoints"][-1]["index"] == repaired["n_waypoints"] - 1

    synthetic = [wp for wp in repaired["waypoints"] if wp.get("synthetic")]
    assert len(synthetic) == inserted
    assert all(wp.get("repair_reason") == "large_gap_interpolation" for wp in synthetic)
