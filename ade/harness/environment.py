"""Load ignored project deployment secrets without publishing them."""

from __future__ import annotations

import os
from pathlib import Path


def load_project_environment(project_root: str | Path) -> None:
    path = Path(project_root).resolve() / ".env"
    if not path.is_file() or path.is_symlink():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not key.replace("_", "a").isalnum():
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)
