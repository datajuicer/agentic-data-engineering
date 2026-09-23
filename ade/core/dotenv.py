"""Small project-local reader for runtime values kept in ``.env``."""

from __future__ import annotations

import shlex
from pathlib import Path


def read_dotenv_value(
    name: str,
    *,
    project_root: Path | None = None,
) -> str | None:
    """Read one value without adding dotenv loading to the process environment."""
    root = project_root or Path(__file__).resolve().parents[2]
    try:
        lines = (root / ".env").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        key, separator, value = line.partition("=")
        if separator and key.strip() == name:
            try:
                parsed = shlex.split(value, comments=True)
            except ValueError:
                return None
            return parsed[0] if parsed else None
    return None
