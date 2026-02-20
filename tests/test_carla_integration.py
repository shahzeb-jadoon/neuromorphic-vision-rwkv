"""CARLA integration tests — require a running CARLA server.

All tests in this module are marked ``@pytest.mark.online`` and are
**excluded** from CI.  Run them locally with::

    pytest tests/test_carla_integration.py -v          # just this file
    pytest tests/ -m online                             # all online tests
    make test-online                                    # Makefile shortcut

Prerequisites:
    - CARLA 0.9.16 server running (Windows host or native)
    - ``pip install carla==0.9.16``
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

try:
    import carla

    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False

from src.data import hdf5_schema as S

# Skip entire module if carla package is not installed
pytestmark = [
    pytest.mark.online,
    pytest.mark.skipif(not CARLA_AVAILABLE, reason="carla package not installed"),
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _get_carla_host() -> str:
    """Detect CARLA host (WSL2 auto-detect or localhost)."""
    import subprocess

    try:
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True,
            text=True,
            check=True,
        )
        gateway = result.stdout.strip().split()[2]
        return gateway
    except (subprocess.CalledProcessError, IndexError, FileNotFoundError):
        return "localhost"


@pytest.fixture(scope="module")
def carla_client() -> "carla.Client":
    """Connect to CARLA once per module; fail fast if server is unreachable."""
    host = _get_carla_host()
    client = carla.Client(host, 2000)
    client.set_timeout(30.0)
    # Verify we can actually talk to the server
    try:
        _ = client.get_server_version()
    except RuntimeError as exc:
        pytest.skip(f"Cannot connect to CARLA at {host}:2000 — {exc}")
    return client


@pytest.fixture(scope="module")
def carla_world(carla_client: "carla.Client") -> "carla.World":
    """Return the current CARLA world, loading Town01 if necessary."""
    carla_client.set_timeout(120.0)
    try:
        carla_client.load_world("Town01")
    except RuntimeError:
        pass  # Town01 may already be loaded
    import time

    time.sleep(3.0)
    carla_client.set_timeout(30.0)
    return carla_client.get_world()


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestCarlaConnection:
    """Basic connectivity and version checks."""

    def test_server_version(self, carla_client: "carla.Client") -> None:
        version = carla_client.get_server_version()
        major, minor = version.split(".")[:2]
        assert int(major) == 0 and int(minor) >= 9, f"Expected CARLA ≥ 0.9.x, got {version}"

    def test_world_not_none(self, carla_world: "carla.World") -> None:
        assert carla_world is not None

    def test_map_loaded(self, carla_world: "carla.World") -> None:
        world_map = carla_world.get_map()
        assert world_map is not None
        assert "Town01" in world_map.name


class TestPrivilegedExpert:
    """PrivilegedExpert instantiation and basic tick."""

    def test_expert_runs_10_ticks(self, carla_world: "carla.World") -> None:
        from agents.expert.privileged_expert import (
            ExpertConfig,
            FrameInfo,
            PrivilegedExpert,
        )
        from agents.navigation.local_planner import RoadOption

        spawn_points = carla_world.get_map().get_spawn_points()
        assert len(spawn_points) > 0

        # Spawn ego vehicle
        bp_lib = carla_world.get_blueprint_library()
        vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        vehicle = carla_world.try_spawn_actor(vehicle_bp, spawn_points[0])
        assert vehicle is not None, "Failed to spawn ego vehicle"

        try:
            expert = PrivilegedExpert(ExpertConfig())

            # Build a short route: just 5 waypoints ahead
            wp = carla_world.get_map().get_waypoint(spawn_points[0].location)
            route = []
            for _ in range(5):
                nxt = wp.next(5.0)
                if not nxt:
                    break
                wp = nxt[0]
                route.append((wp, RoadOption.LANEFOLLOW))

            expert.set_route(route)

            for tick in range(10):
                carla_world.tick()
                transform = vehicle.get_transform()
                velocity = vehicle.get_velocity()
                frame_info = FrameInfo(
                    transform=transform,
                    velocity=velocity,
                    world=carla_world,
                    tick=tick,
                )
                ctrl = expert.run_step(frame_info)
                assert hasattr(ctrl, "steer")
                assert hasattr(ctrl, "throttle")
                assert hasattr(ctrl, "brake")
        finally:
            vehicle.destroy()


class TestShortCollection:
    """Run a mini collection and verify the HDF5 output."""

    def test_collect_50_frames(
        self, carla_client: "carla.Client", carla_world: "carla.World"
    ) -> None:
        from src.data.expert_collector import (
            EpisodeConfig,
            ExpertDataCollector,
        )

        tmpdir = tempfile.mkdtemp(prefix="carla_test_")
        try:
            collector = ExpertDataCollector(
                host=(
                    carla_client._client.host
                    if hasattr(carla_client, "_client")
                    else _get_carla_host()
                ),
                port=2000,
                output_dir=tmpdir,
            )
            collector.client = carla_client
            collector.world = carla_world

            config = EpisodeConfig(
                town="Town01",
                max_frames=50,
                n_vehicles=5,
                n_walkers=5,
            )

            result = collector.collect_episode(config)

            assert result.status in ("completed", "collision", "timeout")
            assert result.n_frames > 0

            if result.filename:
                h5_path = Path(tmpdir) / result.filename
                assert h5_path.exists()

                with h5py.File(h5_path, "r") as f:
                    # Schema version
                    assert S.METADATA_GROUP in f
                    sv = int(f[S.METADATA_GROUP].attrs.get(S.META_SCHEMA_VERSION, 0))
                    assert sv == 3, f"Expected schema v3, got {sv}"

                    # Image shape: (N, H=320, W_total=1920, 3)
                    imgs = f[S.IMAGES]
                    assert imgs.ndim == 4
                    n_frames = imgs.shape[0]
                    assert n_frames > 0
                    assert imgs.shape[1] == config.camera_height  # 320
                    assert imgs.shape[2] == config.camera_width * 3  # 1920
                    assert imgs.shape[3] == 3

                    # Required datasets present
                    for key in [
                        S.STEERING,
                        S.THROTTLE,
                        S.BRAKE,
                        S.NAV_COMMAND,
                        S.VELOCITY,
                    ]:
                        assert key in f, f"Missing dataset: {key}"
                        assert f[key].shape[0] == n_frames

                    # Nav commands in valid range [1, 4] (RoadOption enum)
                    nav = f[S.NAV_COMMAND][:]
                    assert np.all(nav >= 1) and np.all(
                        nav <= 4
                    ), f"Nav commands out of range: min={nav.min()}, max={nav.max()}"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
