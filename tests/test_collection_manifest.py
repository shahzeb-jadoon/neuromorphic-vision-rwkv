"""Tests for CollectionManifest — pure logic, no CARLA required.

CollectionManifest manages campaign resumption metadata:
adding episodes, saving/reloading JSON, tracking completions,
and generating human-readable summaries.

Run with: pytest tests/test_collection_manifest.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.expert_collector import CollectionManifest, EpisodeResult


@pytest.fixture
def tmp_manifest_path(tmp_path: Path) -> Path:
    """Provide a temporary path for a manifest file."""
    return tmp_path / "collection_manifest.json"


@pytest.fixture
def manifest(tmp_manifest_path: Path) -> CollectionManifest:
    """Create a fresh, empty CollectionManifest."""
    return CollectionManifest(tmp_manifest_path)


def _make_result(
    *,
    status: str = "completed",
    filename: str = "episode_abc12345.h5",
    episode_hash: str = "abc12345",
    n_frames: int = 2000,
    town: str = "Town01",
    weather: str = "ClearNoon",
    nav_distribution: dict[str, int] | None = None,
    collision_frame: int | None = None,
    duration_seconds: float = 100.0,
) -> EpisodeResult:
    """Helper to create EpisodeResult instances for tests."""
    return EpisodeResult(
        status=status,
        filename=filename,
        episode_hash=episode_hash,
        n_frames=n_frames,
        town=town,
        weather=weather,
        nav_distribution=nav_distribution
        or {"LaneFollow": 1680, "Left": 100, "Right": 120, "Straight": 100},
        collision_frame=collision_frame,
        duration_seconds=duration_seconds,
    )


class TestManifestCreation:
    """Test empty manifest creation."""

    def test_creates_empty_manifest(self, manifest: CollectionManifest) -> None:
        """New manifest has 0 episodes and a valid created timestamp."""
        assert manifest.data["total_episodes"] == 0
        assert manifest.data["total_frames"] == 0
        assert "created" in manifest.data
        assert isinstance(manifest.data["episodes"], list)
        assert len(manifest.data["episodes"]) == 0

    def test_created_timestamp_is_iso(self, manifest: CollectionManifest) -> None:
        """Created timestamp is a valid ISO 8601 string."""
        from datetime import datetime

        ts = manifest.data["created"]
        # Should not raise
        datetime.fromisoformat(ts)


class TestAddEpisode:
    """Test adding episodes to the manifest."""

    def test_add_episode_increments_counts(self, manifest: CollectionManifest) -> None:
        """Adding an EpisodeResult increments total_episodes and total_frames."""
        result = _make_result(n_frames=2000)
        manifest.add_episode(result)

        assert manifest.data["total_episodes"] == 1
        assert manifest.data["total_frames"] == 2000
        assert len(manifest.data["episodes"]) == 1

    def test_add_multiple_episodes(self, manifest: CollectionManifest) -> None:
        """Adding multiple episodes accumulates correctly."""
        manifest.add_episode(_make_result(n_frames=2000, filename="ep1.h5", episode_hash="aaa"))
        manifest.add_episode(_make_result(n_frames=1500, filename="ep2.h5", episode_hash="bbb"))
        manifest.add_episode(_make_result(n_frames=1000, filename="ep3.h5", episode_hash="ccc"))

        assert manifest.data["total_episodes"] == 3
        assert manifest.data["total_frames"] == 4500

    def test_episode_entry_has_required_fields(self, manifest: CollectionManifest) -> None:
        """Each episode entry contains expected metadata fields."""
        result = _make_result(town="Town03", weather="WetNoon")
        manifest.add_episode(result)

        entry = manifest.data["episodes"][0]
        assert entry["filename"] == "episode_abc12345.h5"
        assert entry["hash"] == "abc12345"
        assert entry["town"] == "Town03"
        assert entry["weather"] == "WetNoon"
        assert entry["n_frames"] == 2000
        assert entry["status"] == "completed"
        assert "nav_distribution" in entry
        assert "timestamp" in entry

    def test_collision_episode_records_collision_frame(self, manifest: CollectionManifest) -> None:
        """Collision episodes record the collision frame."""
        result = _make_result(status="collision", collision_frame=1500)
        manifest.add_episode(result)

        entry = manifest.data["episodes"][0]
        assert entry["status"] == "collision"
        assert entry["collision_frame"] == 1500

    def test_error_message_recorded(self, manifest: CollectionManifest) -> None:
        """Crashed episodes record the error message."""
        result = _make_result(status="crashed")
        result.error_message = "CARLA server disconnected"
        manifest.add_episode(result)

        entry = manifest.data["episodes"][0]
        assert entry["error"] == "CARLA server disconnected"

    def test_last_updated_set(self, manifest: CollectionManifest) -> None:
        """Adding an episode sets last_updated timestamp."""
        manifest.add_episode(_make_result())
        assert "last_updated" in manifest.data


class TestSaveAndReload:
    """Test manifest persistence."""

    def test_save_and_reload(self, tmp_manifest_path: Path) -> None:
        """Write manifest to disk, reload → same data."""
        m1 = CollectionManifest(tmp_manifest_path)
        m1.add_episode(_make_result(n_frames=2000, town="Town01", weather="ClearNoon"))
        m1.add_episode(
            _make_result(
                n_frames=1500,
                town="Town03",
                weather="WetNoon",
                filename="ep2.h5",
                episode_hash="def456",
            )
        )
        m1.save()

        # Reload from same path
        m2 = CollectionManifest(tmp_manifest_path)
        assert m2.data["total_episodes"] == 2
        assert m2.data["total_frames"] == 3500
        assert len(m2.data["episodes"]) == 2
        assert m2.data["episodes"][0]["town"] == "Town01"
        assert m2.data["episodes"][1]["town"] == "Town03"

    def test_save_creates_valid_json(self, tmp_manifest_path: Path) -> None:
        """Saved file is valid JSON."""
        m = CollectionManifest(tmp_manifest_path)
        m.add_episode(_make_result())
        m.save()

        with open(tmp_manifest_path) as f:
            data = json.load(f)
        assert data["total_episodes"] == 1


class TestGetCompletedConfigs:
    """Test campaign resumption via completed config tracking."""

    def test_get_completed_configs(self, manifest: CollectionManifest) -> None:
        """Only 'completed' episodes appear in get_completed_configs."""
        manifest.add_episode(_make_result(status="completed", town="Town01", weather="ClearNoon"))
        manifest.add_episode(
            _make_result(status="completed", town="Town03", weather="WetNoon", filename="ep2.h5")
        )

        completed = manifest.get_completed_configs()
        assert ("Town01", "ClearNoon") in completed
        assert ("Town03", "WetNoon") in completed
        assert len(completed) == 2

    def test_get_completed_ignores_non_completed(self, manifest: CollectionManifest) -> None:
        """'collision' and 'crashed' status episodes are NOT in completed configs."""
        manifest.add_episode(_make_result(status="completed", town="Town01", weather="ClearNoon"))
        manifest.add_episode(
            _make_result(
                status="collision",
                town="Town03",
                weather="WetNoon",
                filename="ep2.h5",
                collision_frame=1500,
            )
        )
        manifest.add_episode(
            _make_result(
                status="crashed",
                town="Town04",
                weather="SoftRainNoon",
                filename="ep3.h5",
            )
        )

        completed = manifest.get_completed_configs()
        assert ("Town01", "ClearNoon") in completed
        assert ("Town03", "WetNoon") not in completed
        assert ("Town04", "SoftRainNoon") not in completed
        assert len(completed) == 1

    def test_empty_manifest_returns_empty_set(self, manifest: CollectionManifest) -> None:
        """Empty manifest has no completed configs."""
        assert manifest.get_completed_configs() == set()


class TestGetCompletedCounts:
    """Test count-based campaign resumption."""

    def test_counts_completed_only(self, manifest: CollectionManifest) -> None:
        """Only 'completed' episodes are counted."""
        manifest.add_episode(_make_result(status="completed", town="Town01", weather="ClearNoon"))
        manifest.add_episode(
            _make_result(
                status="completed",
                town="Town01",
                weather="ClearNoon",
                filename="ep2.h5",
                episode_hash="bbb",
            )
        )
        manifest.add_episode(
            _make_result(
                status="collision",
                town="Town01",
                weather="ClearNoon",
                filename="ep3.h5",
                episode_hash="ccc",
                collision_frame=100,
            )
        )
        counts = manifest.get_completed_counts()
        assert counts[("Town01", "ClearNoon")] == 2

    def test_counts_multiple_configs(self, manifest: CollectionManifest) -> None:
        """Counts are tracked per (town, weather) pair."""
        manifest.add_episode(_make_result(status="completed", town="Town01", weather="ClearNoon"))
        manifest.add_episode(
            _make_result(
                status="completed",
                town="Town03",
                weather="WetNoon",
                filename="ep2.h5",
                episode_hash="bbb",
            )
        )
        manifest.add_episode(
            _make_result(
                status="completed",
                town="Town01",
                weather="ClearNoon",
                filename="ep3.h5",
                episode_hash="ccc",
            )
        )
        counts = manifest.get_completed_counts()
        assert counts[("Town01", "ClearNoon")] == 2
        assert counts[("Town03", "WetNoon")] == 1

    def test_empty_manifest_returns_empty_dict(self, manifest: CollectionManifest) -> None:
        """Empty manifest has no completed counts."""
        assert manifest.get_completed_counts() == {}


class TestSummary:
    """Test human-readable summary output."""

    def test_summary_contains_episode_count(self, manifest: CollectionManifest) -> None:
        """Summary string mentions episode count."""
        manifest.add_episode(_make_result(n_frames=2000))
        manifest.add_episode(_make_result(n_frames=1500, filename="ep2.h5"))

        summary = manifest.summary()
        assert "2" in summary  # 2 episodes

    def test_summary_contains_hours(self, manifest: CollectionManifest) -> None:
        """Summary string mentions hours of data."""
        # 72000 frames at 20 FPS = 1.0 hours
        manifest.add_episode(_make_result(n_frames=72000))

        summary = manifest.summary()
        assert "1.0" in summary  # 1.0 hours

    def test_summary_contains_status_counts(self, manifest: CollectionManifest) -> None:
        """Summary string mentions status breakdown."""
        manifest.add_episode(_make_result(status="completed"))
        manifest.add_episode(
            _make_result(status="collision", filename="ep2.h5", collision_frame=500)
        )

        summary = manifest.summary()
        assert "completed" in summary
        assert "collision" in summary
