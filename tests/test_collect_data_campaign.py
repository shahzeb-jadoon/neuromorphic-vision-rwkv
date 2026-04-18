"""Unit tests for YAML -> ExpertConfig mapping in build_campaign()."""

from __future__ import annotations

from pathlib import Path

from scripts.collect_data import build_campaign
from src.data.expert_collector import CollectionManifest


def _base_cfg() -> dict:
    return {
        "towns": ["Town01"],
        "weathers": ["ClearNoon"],
        "traffic": {"vehicles": [20], "walkers": [10]},
        "cameras": {},
        "expert": {},
    }


def test_build_campaign_maps_arc_and_policy_controls(tmp_path: Path) -> None:
    cfg = _base_cfg()
    cfg["expert"] = {
        "gap_fill_max_factor": 2.5,
        "gap_fill_diverge_factor": 1.8,
        "gap_fill_bezier_step": 0.4,
        "gap_fill_bezier_tangent_factor": 0.25,
        "gap_fill_bezier_max_project_error": 1.7,
        "gap_fill_rpc_budget": 321,
        "gap_fill_bezier_retry_limit": 1,
        "gap_fill_rpc_retry_limit": 3,
        "gap_fill_bezier_max_pair_index": 1,
        "gap_fill_bezier_enabled": False,
        "planner_policy_steer_consistency_gate_enabled": False,
    }

    manifest = CollectionManifest(tmp_path / "collection_manifest.json")
    remaining = build_campaign(cfg, manifest, episodes_per_config=1)

    assert len(remaining) == 1
    ep_cfg, _weight = remaining[0]
    expert_cfg = ep_cfg.expert_config

    assert expert_cfg.gap_fill_max_factor == 2.5
    assert expert_cfg.gap_fill_diverge_factor == 1.8
    assert expert_cfg.gap_fill_bezier_step == 0.4
    assert expert_cfg.gap_fill_bezier_tangent_factor == 0.25
    assert expert_cfg.gap_fill_bezier_max_project_error == 1.7
    assert expert_cfg.gap_fill_rpc_budget == 321
    assert expert_cfg.gap_fill_bezier_retry_limit == 1
    assert expert_cfg.gap_fill_rpc_retry_limit == 3
    assert expert_cfg.gap_fill_bezier_max_pair_index == 1
    assert expert_cfg.gap_fill_bezier_enabled is False
    assert expert_cfg.planner_policy_steer_consistency_gate_enabled is False


def test_build_campaign_keeps_steer_consistency_gate_enabled_by_default(tmp_path: Path) -> None:
    cfg = _base_cfg()

    manifest = CollectionManifest(tmp_path / "collection_manifest.json")
    remaining = build_campaign(cfg, manifest, episodes_per_config=1)

    assert len(remaining) == 1
    ep_cfg, _weight = remaining[0]
    assert ep_cfg.expert_config.planner_policy_steer_consistency_gate_enabled is True
