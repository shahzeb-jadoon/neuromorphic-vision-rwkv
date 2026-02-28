"""Unit tests for expert_collector pure-logic methods.

Tests the non-CARLA parts: dataclasses, _save_episode, _detect_wsl2_host,
and hash generation. CARLA is mocked where needed.

Updated for schema v3: panoramic images, PrivilegedExpert, no NAV_MAP.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

from src.data import hdf5_schema as S
from src.data.expert_collector import (
    EpisodeConfig,
    EpisodeResult,
    ExpertDataCollector,
)

# ── Dataclass tests ──────────────────────────────────────────────────


class TestEpisodeConfig:
    """Tests for EpisodeConfig defaults and override."""

    def test_default_values(self) -> None:
        cfg = EpisodeConfig()
        assert cfg.town == "Town01"
        assert cfg.weather == "ClearNoon"
        assert cfg.max_frames == 2000
        assert cfg.fps == 20
        assert cfg.camera_width == 640
        assert cfg.camera_height == 320
        assert cfg.camera_fov == 60
        assert cfg.camera_yaws == (-60.0, 0.0, 60.0)

    def test_override(self) -> None:
        cfg = EpisodeConfig(town="Town03", weather="WetNoon", max_frames=500)
        assert cfg.town == "Town03"
        assert cfg.weather == "WetNoon"
        assert cfg.max_frames == 500


class TestEpisodeResult:
    """Tests for EpisodeResult fields."""

    def test_required_fields(self) -> None:
        result = EpisodeResult(status="completed")
        assert result.status == "completed"
        assert result.filename is None
        assert result.n_frames == 0
        assert result.nav_distribution == {}

    def test_all_fields(self) -> None:
        result = EpisodeResult(
            status="collision",
            filename="episode_abc.h5",
            episode_hash="abc12345",
            n_frames=100,
            town="Town02",
            weather="Rain",
            nav_distribution={"LaneFollow": 80, "Left": 10, "Right": 10},
            collision_frame=95,
            error_message=None,
            duration_seconds=30.5,
        )
        assert result.collision_frame == 95
        assert result.nav_distribution["Left"] == 10
        assert result.duration_seconds == 30.5


# ── _detect_wsl2_host tests ─────────────────────────────────────────


class TestDetectWSL2Host:
    """Tests for WSL2 host detection (mocked subprocess)."""

    def test_returns_gateway_ip(self) -> None:
        mock_result = MagicMock()
        mock_result.stdout = "default via 172.23.48.1 dev eth0"
        with patch("subprocess.run", return_value=mock_result):
            ip = ExpertDataCollector._detect_wsl2_host()
        assert ip == "172.23.48.1"

    def test_returns_none_on_error(self) -> None:
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "ip")):
            ip = ExpertDataCollector._detect_wsl2_host()
        assert ip is None

    def test_returns_none_on_missing_command(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            ip = ExpertDataCollector._detect_wsl2_host()
        assert ip is None

    def test_returns_none_on_empty_output(self) -> None:
        mock_result = MagicMock()
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            ip = ExpertDataCollector._detect_wsl2_host()
        # Empty string split → IndexError → returns None
        assert ip is None


# ── _save_episode tests ─────────────────────────────────────────────


class TestSaveEpisode:
    """Tests for HDF5 episode saving (actual file I/O to tmp dir)."""

    @pytest.fixture()
    def collector(self, tmp_path: Path) -> ExpertDataCollector:
        """Create a collector with mocked CARLA import."""
        with patch.dict("sys.modules", {"carla": MagicMock()}):
            # Need to reload to pick up mock
            with patch("src.data.expert_collector.CARLA_AVAILABLE", True):
                c = ExpertDataCollector.__new__(ExpertDataCollector)
                c.output_dir = tmp_path
                c.host = "localhost"
                c.port = 2000
                c.crash_retries = 0
                c.crash_wait = 0
                c.client = None
                c.world = None
                return c

    @pytest.fixture()
    def sample_data(self) -> dict[str, Any]:
        """Minimal episode data (schema v3 — panoramic)."""
        n = 50
        pano_w = 640 * 3  # 3 cameras
        return {
            "images": np.random.randint(0, 255, (n, 320, pano_w, 3), dtype=np.uint8),
            "steering": np.random.uniform(-1, 1, n).astype(np.float32),
            "throttle": np.random.uniform(0, 1, n).astype(np.float32),
            "brake": np.zeros(n, dtype=np.float32),
            "nav_command": np.full(n, 4, dtype=np.int32),  # LANEFOLLOW=4
            "position": np.random.uniform(-100, 100, (n, 3)).astype(np.float32),
            "rotation": np.random.uniform(-180, 180, (n, 3)).astype(np.float32),
            "velocity": np.random.uniform(0, 10, (n, 3)).astype(np.float32),
            "traffic_light_state": np.zeros(n, dtype=np.int32),
            "stop_sign_distance": np.full(n, -1.0, dtype=np.float32),
            "target_speed": np.full(n, 8.33, dtype=np.float32),
            "driving_mode": np.zeros(n, dtype=np.int32),
            "nearby_vehicles": [json.dumps([]) for _ in range(n)],
            "nearby_pedestrians": [json.dumps([]) for _ in range(n)],
            "n_corridor_segments": np.full(n, 10, dtype=np.int32),
            "wp_buffer_size": np.full(n, 20, dtype=np.int32),
            "wp_buffer_head_dist": np.full(n, 3.5, dtype=np.float32),
            "sat_detections": np.zeros(n, dtype=np.int32),
            "lateral_detections": np.zeros(n, dtype=np.int32),
        }

    def test_save_creates_h5_file(
        self, collector: ExpertDataCollector, sample_data: dict[str, Any]
    ) -> None:
        result = collector._save_episode(
            **sample_data,
            config=EpisodeConfig(),
            status="completed",
            collision_frame=None,
        )
        assert result.status == "completed"
        assert result.filename is not None
        assert (collector.output_dir / result.filename).exists()

    def test_save_h5_contains_required_fields(
        self, collector: ExpertDataCollector, sample_data: dict[str, Any]
    ) -> None:
        result = collector._save_episode(
            **sample_data,
            config=EpisodeConfig(),
            status="completed",
            collision_frame=None,
        )
        filepath = collector.output_dir / result.filename  # type: ignore[operator]
        with h5py.File(filepath, "r") as f:
            assert S.IMAGES in f
            assert S.STEERING in f
            assert S.THROTTLE in f
            assert S.BRAKE in f
            assert S.VELOCITY in f
            assert S.NAV_COMMAND in f
            assert S.POSITION in f
            assert S.ROTATION in f
            assert S.TRAFFIC_LIGHT_STATE in f
            assert S.STOP_SIGN_DISTANCE in f
            assert S.TARGET_SPEED in f
            assert S.DRIVING_MODE in f
            assert S.NEARBY_VEHICLES in f
            assert S.NEARBY_PEDESTRIANS in f
            assert S.METADATA_GROUP in f

    def test_save_metadata_attrs(
        self, collector: ExpertDataCollector, sample_data: dict[str, Any]
    ) -> None:
        result = collector._save_episode(
            **sample_data,
            config=EpisodeConfig(town="Town03", weather="WetNoon"),
            status="completed",
            collision_frame=None,
        )
        filepath = collector.output_dir / result.filename  # type: ignore[operator]
        with h5py.File(filepath, "r") as f:
            meta = f[S.METADATA_GROUP]
            assert meta.attrs[S.META_SCHEMA_VERSION] == S.SCHEMA_VERSION
            assert meta.attrs[S.META_TOWN] == "Town03"
            assert meta.attrs[S.META_WEATHER] == "WetNoon"
            assert meta.attrs[S.META_N_FRAMES] == 50
            assert meta.attrs[S.META_N_CAMERAS] == 3
            assert meta.attrs[S.META_CAMERA_FOV] == 60
            assert meta.attrs[S.META_EXPERT_TYPE] == "privileged_expert"

    def test_save_collision_frame_recorded(
        self, collector: ExpertDataCollector, sample_data: dict[str, Any]
    ) -> None:
        result = collector._save_episode(
            **sample_data,
            config=EpisodeConfig(),
            status="collision",
            collision_frame=42,
        )
        assert result.collision_frame == 42
        filepath = collector.output_dir / result.filename  # type: ignore[operator]
        with h5py.File(filepath, "r") as f:
            meta = f[S.METADATA_GROUP]
            assert meta.attrs[S.META_COLLISION_FRAME] == 42

    def test_save_nav_distribution(
        self, collector: ExpertDataCollector, sample_data: dict[str, Any]
    ) -> None:
        # Set mixed nav commands (RoadOption values)
        sample_data["nav_command"][:30] = 4  # LaneFollow
        sample_data["nav_command"][30:40] = 1  # Left
        sample_data["nav_command"][40:] = 2  # Right
        result = collector._save_episode(
            **sample_data,
            config=EpisodeConfig(),
            status="completed",
            collision_frame=None,
        )
        assert result.nav_distribution["LaneFollow"] == 30
        assert result.nav_distribution["Left"] == 10
        assert result.nav_distribution["Right"] == 10
        assert result.nav_distribution["Straight"] == 0

    def test_save_hash_is_8_chars(
        self, collector: ExpertDataCollector, sample_data: dict[str, Any]
    ) -> None:
        result = collector._save_episode(
            **sample_data,
            config=EpisodeConfig(),
            status="completed",
            collision_frame=None,
        )
        assert result.episode_hash is not None
        assert len(result.episode_hash) == 8
        # Should be hex
        int(result.episode_hash, 16)

    def test_save_uses_gzip_compression(
        self, collector: ExpertDataCollector, sample_data: dict[str, Any]
    ) -> None:
        result = collector._save_episode(
            **sample_data,
            config=EpisodeConfig(),
            status="completed",
            collision_frame=None,
        )
        filepath = collector.output_dir / result.filename  # type: ignore[operator]
        with h5py.File(filepath, "r") as f:
            assert f[S.IMAGES].compression == "gzip"
            assert f[S.IMAGES].compression_opts == 1


# ── hdf5_schema constants test ───────────────────────────────────────


class TestHDF5Schema:
    """Tests that schema constants are consistent."""

    def test_schema_version_is_v3(self) -> None:
        assert S.SCHEMA_VERSION == 3

    def test_required_fields_exist(self) -> None:
        for field in [
            S.IMAGES,
            S.STEERING,
            S.THROTTLE,
            S.BRAKE,
            S.VELOCITY,
            S.NAV_COMMAND,
            S.POSITION,
            S.ROTATION,
            S.TRAFFIC_LIGHT_STATE,
            S.STOP_SIGN_DISTANCE,
            S.TARGET_SPEED,
            S.DRIVING_MODE,
            S.NEARBY_VEHICLES,
            S.NEARBY_PEDESTRIANS,
        ]:
            assert isinstance(field, str)
            assert len(field) > 0

    def test_metadata_fields_exist(self) -> None:
        for field in [
            S.METADATA_GROUP,
            S.META_SCHEMA_VERSION,
            S.META_TOWN,
            S.META_WEATHER,
            S.META_FPS,
            S.META_N_FRAMES,
            S.META_STATUS,
            S.META_CAMERA_WIDTH,
            S.META_CAMERA_HEIGHT,
            S.META_N_CAMERAS,
            S.META_CAMERA_FOV,
            S.META_CAMERA_YAWS,
            S.META_EXPERT_TYPE,
            S.META_EXPERT_TARGET_SPEED,
        ]:
            assert isinstance(field, str)
