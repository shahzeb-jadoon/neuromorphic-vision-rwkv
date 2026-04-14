"""Unit tests for expert_collector pure-logic methods.

Tests the non-CARLA parts: dataclasses, _detect_wsl2_host,
_StreamingEpisodeWriter, and hash generation. CARLA is mocked where needed.

Updated for schema v3: panoramic images, PrivilegedExpert, no NAV_MAP.
"""

from __future__ import annotations

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
    _StreamingEpisodeWriter,
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
            S.TRAFFIC_LIGHT_DISTANCE,
            S.TRAFFIC_LIGHT_ACTOR_ID,
            S.TRAFFIC_LIGHT_STOP_LINE_POSITION,
            S.STOP_SIGN_DISTANCE,
            S.TARGET_SPEED,
            S.DRIVING_MODE,
            S.GAP_FILL_RPCS,
            S.PLANNER_POLICY_REASON,
            S.PLANNER_POLICY_EXCEPTION,
            S.DIST_TO_NEXT_JUNCTION,
            S.PRE_JUNCTION_LANE_CHANGE_VIOLATION,
            S.CONSECUTIVE_LANE_CHANGE_VIOLATION,
            S.STRAIGHT_EXIT_LANE_VIOLATION,
            S.TURN_LANE_LEGALITY_VIOLATION,
            S.LEGALITY_CONTINUITY_KEPT,
            S.ROUTE_WATCHDOG_TRIGGERED,
            S.NEARBY_VEHICLES,
            S.NEARBY_PEDESTRIANS,
            S.ROUNDABOUT_LANE_CHANGE_VIOLATION,
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


# ── _StreamingEpisodeWriter tests ────────────────────────────────────


class _FakeControl:
    """Lightweight stand-in for carla.VehicleControl."""

    def __init__(self, steer: float = 0.0, throttle: float = 0.5, brake: float = 0.0):
        self.steer = steer
        self.throttle = throttle
        self.brake = brake


class _FakeFrameInfo:
    """Lightweight stand-in for FrameInfo with all required fields."""

    def __init__(self, nav_command: int = 4, speed_kmh: float = 30.0):
        self.nav_command = nav_command
        self.position = [10.0, 20.0, 0.5]
        self.rotation = [0.0, 90.0, 0.0]
        self.velocity = [speed_kmh / 3.6, 0.0, 0.0]
        self.traffic_light_state = 0
        self.traffic_light_distance = -1.0
        self.traffic_light_actor_id = -1
        self.traffic_light_stop_line_position = [-1.0, -1.0, -1.0]
        self.stop_sign_distance = -1.0
        self.yield_sign_distance = -1.0
        self.target_speed = 8.33
        self.driving_mode = 0
        self.nearby_vehicles: list[dict[str, Any]] = []
        self.nearby_pedestrians: list[dict[str, Any]] = []
        self.n_corridor_segments = 10
        self.wp_buffer_size = 20
        self.wp_buffer_head_dist = 3.5
        self.sat_detections = 0
        self.lateral_detections = 0
        self.gap_fill_diverged = 0
        self.gap_fill_rpcs = 0
        self.current_speed = speed_kmh
        self.route_completion = 0.5
        self.planner_policy_reason = 0
        self.planner_policy_exception = 0
        self.dist_to_next_junction = -1.0
        self.pre_junction_lane_change_violation = 0
        self.consecutive_lane_change_violation = 0
        self.straight_exit_lane_violation = 0
        self.turn_lane_legality_violation = 0
        self.roundabout_lane_change_violation = 0
        self.legality_wrong_way_filtered = 0
        self.legality_do_not_enter_filtered = 0
        self.legality_turn_mismatch_filtered = 0
        self.legality_continuity_kept = 0
        self.route_watchdog_triggered = 0


class TestStreamingEpisodeWriter:
    """Tests for the streaming HDF5 writer."""

    @pytest.fixture()
    def config(self) -> EpisodeConfig:
        return EpisodeConfig(max_frames=100)

    @pytest.fixture()
    def filepath(self, tmp_path: Path) -> Path:
        return tmp_path / "episode_test1234.h5"

    def test_write_and_finalize(self, filepath: Path, config: EpisodeConfig) -> None:
        """Writer creates a valid HDF5 file with all schema v3 fields."""
        n_frames = 20
        writer = _StreamingEpisodeWriter(filepath, config, max_frames=config.max_frames)

        pano_w = config.camera_width * len(config.camera_yaws)
        for _ in range(n_frames):
            img = np.random.randint(0, 255, (config.camera_height, pano_w, 3), dtype=np.uint8)
            writer.append_frame(img, _FakeControl(), _FakeFrameInfo())

        result = writer.finalize(status="completed", collision_frame=None)

        assert result.status == "completed"
        assert result.n_frames == n_frames
        assert filepath.exists()

        with h5py.File(filepath, "r") as f:
            assert f[S.IMAGES].shape == (n_frames, config.camera_height, pano_w, 3)
            assert f[S.STEERING].shape == (n_frames,)
            assert f[S.POSITION].shape == (n_frames, 3)
            assert f[S.GAP_FILL_RPCS].shape == (n_frames,)
            assert f[S.PLANNER_POLICY_REASON].shape == (n_frames,)
            assert f[S.PRE_JUNCTION_LANE_CHANGE_VIOLATION].shape == (n_frames,)
            assert f[S.TURN_LANE_LEGALITY_VIOLATION].shape == (n_frames,)
            assert f[S.LEGALITY_CONTINUITY_KEPT].shape == (n_frames,)
            assert f[S.ROUTE_WATCHDOG_TRIGGERED].shape == (n_frames,)
            assert f[S.ROUNDABOUT_LANE_CHANGE_VIOLATION].shape == (n_frames,)
            assert S.STATIONARY in f
            assert S.METADATA_GROUP in f
            assert f[S.METADATA_GROUP].attrs[S.META_N_FRAMES] == n_frames
            assert f[S.METADATA_GROUP].attrs[S.META_STATUS] == "completed"

    def test_finalize_with_collision(self, filepath: Path, config: EpisodeConfig) -> None:
        writer = _StreamingEpisodeWriter(filepath, config, max_frames=100)
        pano_w = config.camera_width * len(config.camera_yaws)
        for _ in range(5):
            img = np.random.randint(0, 255, (config.camera_height, pano_w, 3), dtype=np.uint8)
            writer.append_frame(img, _FakeControl(), _FakeFrameInfo())

        result = writer.finalize(status="collision", collision_frame=4)

        assert result.status == "collision"
        assert result.collision_frame == 4
        with h5py.File(filepath, "r") as f:
            assert f[S.METADATA_GROUP].attrs[S.META_COLLISION_FRAME] == 4

    def test_finalize_zero_frames_aborts(self, filepath: Path, config: EpisodeConfig) -> None:
        writer = _StreamingEpisodeWriter(filepath, config, max_frames=100)
        result = writer.finalize(status="completed", collision_frame=None)
        assert result.status == "crashed"
        assert not filepath.exists()  # File deleted

    def test_abort_removes_file(self, filepath: Path, config: EpisodeConfig) -> None:
        writer = _StreamingEpisodeWriter(filepath, config, max_frames=100)
        pano_w = config.camera_width * len(config.camera_yaws)
        img = np.random.randint(0, 255, (config.camera_height, pano_w, 3), dtype=np.uint8)
        writer.append_frame(img, _FakeControl(), _FakeFrameInfo())
        writer.abort()
        assert not filepath.exists()

    def test_compression_is_gzip(self, filepath: Path, config: EpisodeConfig) -> None:
        writer = _StreamingEpisodeWriter(filepath, config, max_frames=100)
        pano_w = config.camera_width * len(config.camera_yaws)
        img = np.random.randint(0, 255, (config.camera_height, pano_w, 3), dtype=np.uint8)
        writer.append_frame(img, _FakeControl(), _FakeFrameInfo())
        writer.finalize(status="completed", collision_frame=None)

        with h5py.File(filepath, "r") as f:
            assert f[S.IMAGES].compression == "gzip"
            assert f[S.IMAGES].compression_opts == 1

    def test_nav_distribution(self, filepath: Path, config: EpisodeConfig) -> None:
        writer = _StreamingEpisodeWriter(filepath, config, max_frames=100)
        pano_w = config.camera_width * len(config.camera_yaws)
        img = np.random.randint(0, 255, (config.camera_height, pano_w, 3), dtype=np.uint8)

        # 10 frames of LaneFollow (nav=4)
        for _ in range(10):
            writer.append_frame(img, _FakeControl(), _FakeFrameInfo(nav_command=4))
        # 5 frames of Left (nav=1)
        for _ in range(5):
            writer.append_frame(img, _FakeControl(), _FakeFrameInfo(nav_command=1))

        result = writer.finalize(status="completed", collision_frame=None)
        assert result.nav_distribution["LaneFollow"] == 10
        assert result.nav_distribution["Left"] == 5
