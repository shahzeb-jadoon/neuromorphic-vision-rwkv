"""Integration tests for utility scripts.

Tests that scripts execute correctly and produce expected outputs,
using the toy dataset generator (no CARLA required).

Run with: pytest tests/test_scripts.py -v
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.data import hdf5_schema as S
from src.data import schema_adapter
from src.data.driving_dataset import FPS

# Directory for test outputs
TEST_OUTPUT_DIR = Path("/tmp/test_scripts_output")

# Project root — needed so subprocess can find `src` package
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


def _script_env() -> dict[str, str]:
    """Return an env dict with PYTHONPATH set to project root."""
    env = os.environ.copy()
    env["PYTHONPATH"] = _PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return env


@pytest.fixture(scope="module", autouse=True)
def clean_output_dir() -> Iterator[None]:
    """Create fresh output directory before tests, clean up after."""
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield
    # Cleanup after all tests
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)


class TestCreateToyDataset:
    """Test that create_toy_dataset.py produces valid HDF5 files."""

    TOY_DIR = TEST_OUTPUT_DIR / "toy_data"

    @pytest.fixture(scope="class", autouse=True)
    def run_toy_script(self) -> None:
        """Run create_toy_dataset.py once for all tests in this class."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/create_toy_dataset.py",
                "--output",
                str(self.TOY_DIR),
                "--episodes",
                "5",
                "--frames",
                "64",
            ],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            env=_script_env(),
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"

    def test_creates_episode_files(self) -> None:
        """Script creates the expected number of episode HDF5 files."""
        h5_files = list(self.TOY_DIR.glob("episode_*.h5"))
        assert len(h5_files) == 5, f"Expected 5 episodes, got {len(h5_files)}"

    def test_creates_manifest(self) -> None:
        """Script creates a collection manifest."""
        manifest_path = self.TOY_DIR / "collection_manifest.json"
        assert manifest_path.exists(), "Manifest file not found"

        with open(manifest_path) as f:
            manifest = json.load(f)
        assert "episodes" in manifest

    def test_episode_has_required_fields(self) -> None:
        """Each episode HDF5 file has the required dataset fields."""
        h5_files = list(self.TOY_DIR.glob("episode_*.h5"))
        assert len(h5_files) > 0

        required_fields = [S.IMAGES, S.STEERING, S.THROTTLE, S.NAV_COMMAND, S.VELOCITY]

        with h5py.File(h5_files[0], "r") as f:
            for field in required_fields:
                assert field in f, f"Missing field: {field}"

    def test_episode_frame_count(self) -> None:
        """Each episode has the expected number of frames."""
        h5_files = list(self.TOY_DIR.glob("episode_*.h5"))

        with h5py.File(h5_files[0], "r") as f:
            n_frames = f[S.IMAGES].shape[0]
            assert n_frames == 64, f"Expected 64 frames, got {n_frames}"

    def test_episode_image_shape(self) -> None:
        """Images have expected shape (N, H, W, 3)."""
        h5_files = list(self.TOY_DIR.glob("episode_*.h5"))

        with h5py.File(h5_files[0], "r") as f:
            shape = f[S.IMAGES].shape
            assert len(shape) == 4, f"Expected 4D shape, got {shape}"
            assert shape[3] == 3, f"Expected 3 channels, got {shape[3]}"

    def test_metadata_group_exists(self) -> None:
        """Each episode has a metadata group."""
        h5_files = list(self.TOY_DIR.glob("episode_*.h5"))

        with h5py.File(h5_files[0], "r") as f:
            assert S.METADATA_GROUP in f, "Missing metadata group"


class TestAnalyzeDataset:
    """Test that analyze_dataset.py runs successfully on toy data."""

    @pytest.fixture(scope="class")
    def toy_data_dir(self) -> Path:
        """Generate toy data for analysis."""
        toy_dir = TEST_OUTPUT_DIR / "analyze_toy_data"
        if not toy_dir.exists():
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/create_toy_dataset.py",
                    "--output",
                    str(toy_dir),
                    "--episodes",
                    "3",
                ],
                capture_output=True,
                text=True,
                cwd=_PROJECT_ROOT,
                env=_script_env(),
            )
            assert result.returncode == 0, f"Toy data generation failed:\n{result.stderr}"
        return toy_dir

    def test_analyze_runs_without_error(self, toy_data_dir: Path) -> None:
        """analyze_dataset.py completes successfully on toy data."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_dataset.py",
                str(toy_data_dir),
            ],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            env=_script_env(),
        )
        assert result.returncode == 0, f"Analysis failed:\n{result.stderr}"

    def test_analyze_outputs_statistics(self, toy_data_dir: Path) -> None:
        """analyze_dataset.py output includes expected statistics."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_dataset.py",
                str(toy_data_dir),
            ],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            env=_script_env(),
        )
        output = result.stdout.lower()
        # Should mention episodes, frames, or similar stats
        assert "episode" in output or "frame" in output, (
            f"Expected statistics in output, got:\n{result.stdout[:500]}"
        )
        assert "waypoint policy evaluation" in output

    def test_analyze_reports_policy_violation_totals(self, tmp_path: Path) -> None:
        """Policy telemetry is aggregated into explicit violation totals."""
        data_dir = tmp_path / "policy_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        ep_path = data_dir / "episode_policy_test.h5"

        with h5py.File(ep_path, "w") as f:
            f.create_dataset(S.IMAGES, data=np.zeros((4, 8, 8, 3), dtype=np.uint8))
            f.create_dataset(S.NAV_COMMAND, data=np.array([4, 4, 4, 4], dtype=np.int32))
            f.create_dataset(S.STEERING, data=np.zeros((4,), dtype=np.float32))
            f.create_dataset(S.THROTTLE, data=np.zeros((4,), dtype=np.float32))
            f.create_dataset(
                S.PRE_JUNCTION_LANE_CHANGE_VIOLATION,
                data=np.array([0, 1, 0, 1], dtype=np.int8),
            )
            f.create_dataset(
                S.CONSECUTIVE_LANE_CHANGE_VIOLATION,
                data=np.array([0, 1, 0, 0], dtype=np.int8),
            )
            f.create_dataset(
                S.STRAIGHT_EXIT_LANE_VIOLATION,
                data=np.array([0, 0, 0, 1], dtype=np.int8),
            )
            f.create_dataset(
                S.TURN_LANE_LEGALITY_VIOLATION,
                data=np.array([0, 1, 0, 0], dtype=np.int8),
            )

            meta = f.create_group(S.METADATA_GROUP)
            meta.attrs[S.META_TOWN] = "Town01"
            meta.attrs[S.META_WEATHER] = "ClearNoon"
            meta.attrs[S.META_STATUS] = "completed"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_dataset.py",
                str(data_dir),
            ],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            env=_script_env(),
        )

        assert result.returncode == 0, f"Analysis failed:\n{result.stderr}"
        output = result.stdout
        assert "ev-03 turn-lane legality: 1 violations" in output.lower()
        assert "pre-junction lane-change: 2 violations" in output
        assert "consecutive lane-change: 1 violations" in output
        assert "straight-in-exit-lane: 1 violations" in output
        assert output.count("-> FAIL") == 4


class TestLabelAudit:
    def test_report_covers_every_usable_frame(self, toy_data_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "audit"
        result = subprocess.run(
            [sys.executable, "scripts/label_audit.py", str(toy_data_dir), str(out)]
            + ["--workers", "1", "--samples", "2"],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            env=_script_env(),
        )
        assert result.returncode == 0, result.stderr
        usable = 0
        for path in toy_data_dir.glob("episode_*.h5"):
            with h5py.File(path, "r") as f:
                usable += schema_adapter.usable_frames(f, FPS)
        numbers = json.loads((out / "audit.json").read_text())
        assert numbers["usable_frames"] == usable and not numbers["errors"]
        assert any(f["frames"] for f in numbers["findings"])
        assert "<h2>Findings</h2>" in (out / "index.html").read_text()
