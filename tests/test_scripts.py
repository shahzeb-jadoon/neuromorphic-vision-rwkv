"""Integration tests for utility scripts.

Tests that scripts execute correctly and produce expected outputs,
using the toy dataset generator (no CARLA required).

Run with: pytest tests/test_scripts.py -v
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import pytest

from src.data import hdf5_schema as S

# Directory for test outputs
TEST_OUTPUT_DIR = Path("/tmp/test_scripts_output")


@pytest.fixture(scope="module", autouse=True)
def clean_output_dir():
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
    def run_toy_script(self):
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
            cwd=str(Path(__file__).parent.parent),
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
    def toy_data_dir(self):
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
                cwd=str(Path(__file__).parent.parent),
            )
            assert result.returncode == 0, f"Toy data generation failed:\n{result.stderr}"
        return toy_dir

    def test_analyze_runs_without_error(self, toy_data_dir) -> None:
        """analyze_dataset.py completes successfully on toy data."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_dataset.py",
                str(toy_data_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0, f"Analysis failed:\n{result.stderr}"

    def test_analyze_outputs_statistics(self, toy_data_dir) -> None:
        """analyze_dataset.py output includes expected statistics."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_dataset.py",
                str(toy_data_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        output = result.stdout.lower()
        # Should mention episodes, frames, or similar stats
        assert (
            "episode" in output or "frame" in output
        ), f"Expected statistics in output, got:\n{result.stdout[:500]}"
