"""Settings come from the environment first, then .env; a missing one fails with a hint."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from src.utils import settings


@pytest.fixture
def env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    path = tmp_path / ".env"
    monkeypatch.setattr(settings, "ENV_FILE", path)
    monkeypatch.delenv("DATA_DIR", raising=False)
    settings._dotenv.cache_clear()
    yield path
    settings._dotenv.cache_clear()


def test_reads_env_file_and_skips_comments(env_file: Path) -> None:
    env_file.write_text('# DATA_DIR=/commented\nDATA_DIR = "/data/episodes"\n')
    assert settings.setting("DATA_DIR") == "/data/episodes"
    assert settings.setting("CARLA_HOST", "localhost") == "localhost"


def test_environment_overrides_env_file(env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file.write_text("DATA_DIR=/from/file\n")
    monkeypatch.setenv("DATA_DIR", "/from/env")
    assert settings.setting("DATA_DIR") == "/from/env"


def test_missing_setting_names_the_example_file(env_file: Path) -> None:
    with pytest.raises(KeyError, match=r"\.env\.example"):
        settings.setting("DATA_DIR")
