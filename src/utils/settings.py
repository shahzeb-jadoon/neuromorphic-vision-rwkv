"""Machine-specific settings, read from the environment or the git-ignored .env file.

Keys and examples are listed in .env.example at the repository root.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


@cache
def _dotenv() -> dict[str, str]:
    if not ENV_FILE.is_file():
        return {}
    values: dict[str, str] = {}
    for line in ENV_FILE.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.lstrip().startswith("#"):
            values[key.strip()] = os.path.expanduser(value.strip().strip("'\""))
    return values


def setting(name: str, default: str | None = None) -> str:
    """Return `name` from the environment, else from .env, else `default`."""
    value = os.environ.get(name) or _dotenv().get(name) or default
    if value is None:
        raise KeyError(f"{name} is not set: add it to .env (see .env.example) or export it")
    return value
