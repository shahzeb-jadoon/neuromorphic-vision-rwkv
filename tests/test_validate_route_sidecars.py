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
        "route_links": {
            "transition_count": 2,
            "max_gap_m": 35.0,
            "link_kind_counts": {
                "continuous": 1,
                "into_junction_connector": 1,
            },
            "junction_connector_transfer_count": 0,
            "junction_connectors": [],
            "links": [
                {
                    "from_index": 0,
                    "to_index": 1,
                    "kind": "continuous",
                    "gap_m": 35.0,
                    "road_option_from": "LANEFOLLOW",
                    "road_option_to": "LANEFOLLOW",
                    "road_id_from": 1,
                    "road_id_to": 1,
                    "lane_id_from": 1,
                    "lane_id_to": 1,
                    "is_junction_from": False,
                    "is_junction_to": False,
                },
                {
                    "from_index": 1,
                    "to_index": 2,
                    "kind": "into_junction_connector",
                    "gap_m": 2.0,
                    "road_option_from": "LANEFOLLOW",
                    "road_option_to": "LEFT",
                    "road_id_from": 1,
                    "road_id_to": 1,
                    "lane_id_from": 1,
                    "lane_id_to": 1,
                    "is_junction_from": False,
                    "is_junction_to": True,
                    "connector_road_id": 1,
                },
            ],
        },
    }


def _junction_connector_payload() -> dict:
    return {
        "episode_hash": "junction123",
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
                "road_id": 10,
                "lane_id": 1,
                "is_junction": False,
            },
            {
                "index": 1,
                "x": 2.0,
                "y": 0.0,
                "z": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "road_option": "LEFT",
                "road_id": 100,
                "lane_id": 1,
                "is_junction": True,
            },
            {
                "index": 2,
                "x": 4.0,
                "y": 1.0,
                "z": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "road_option": "LANEFOLLOW",
                "road_id": 20,
                "lane_id": 1,
                "is_junction": False,
            },
        ],
        "route_links": {
            "transition_count": 2,
            "max_gap_m": 2.236,
            "link_kind_counts": {
                "into_junction_connector": 1,
                "out_of_junction_connector": 1,
            },
            "junction_connector_transfer_count": 1,
            "junction_connectors": [
                {
                    "connector_road_id": 100,
                    "transfer_count": 1,
                    "incoming_roads": [10],
                    "outgoing_roads": [20],
                }
            ],
            "links": [
                {
                    "from_index": 0,
                    "to_index": 1,
                    "kind": "into_junction_connector",
                    "gap_m": 2.0,
                    "road_option_from": "LANEFOLLOW",
                    "road_option_to": "LEFT",
                    "road_id_from": 10,
                    "road_id_to": 100,
                    "lane_id_from": 1,
                    "lane_id_to": 1,
                    "is_junction_from": False,
                    "is_junction_to": True,
                    "connector_road_id": 100,
                },
                {
                    "from_index": 1,
                    "to_index": 2,
                    "kind": "out_of_junction_connector",
                    "gap_m": 2.236,
                    "road_option_from": "LEFT",
                    "road_option_to": "LANEFOLLOW",
                    "road_id_from": 100,
                    "road_id_to": 20,
                    "lane_id_from": 1,
                    "lane_id_to": 1,
                    "is_junction_from": True,
                    "is_junction_to": False,
                    "connector_road_id": 100,
                },
            ],
        },
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


def test_analyze_route_sidecar_validates_route_links() -> None:
    report = analyze_route_sidecar(
        _junction_connector_payload(),
        warn_gap_m=12.0,
        severe_gap_m=30.0,
        wrong_way_yaw_diff_deg=150.0,
        max_issues=10,
    )

    assert report["route_links"]["present"] is True
    assert report["route_links"]["mismatch_count"] == 0
    assert report["route_links"]["junction_connector_transfer_count"] == 1
    assert report["route_links"]["junction_connectors"][0]["connector_road_id"] == 100


def test_analyze_route_sidecar_detects_connector_mismatch() -> None:
    payload = _junction_connector_payload()
    payload["route_links"]["links"][1]["connector_road_id"] = 999

    report = analyze_route_sidecar(
        payload,
        warn_gap_m=12.0,
        severe_gap_m=30.0,
        wrong_way_yaw_diff_deg=150.0,
        max_issues=10,
    )

    assert report["route_links"]["present"] is True
    assert report["route_links"]["mismatch_count"] >= 1
