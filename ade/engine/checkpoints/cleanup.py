from __future__ import annotations

import errno
import re
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path


def cleanup_checkpoint_directories(
    candidates: Iterable[str | Path],
    *,
    allowed_root: str | Path,
    keep: Iterable[str | Path] = (),
    protected: Iterable[str | Path] = (),
    ready: Callable[[Path], bool] | None = None,
    remove_incomplete: bool = False,
    name_pattern: str | None = None,
) -> list[str]:
    root = Path(allowed_root).resolve()
    retained = {_resolve(path) for path in keep}
    retained.update(_resolve(path) for path in protected)
    removed: list[str] = []
    seen: set[str] = set()

    for candidate in candidates:
        checkpoint = Path(candidate).resolve()
        checkpoint_key = str(checkpoint)
        if checkpoint_key in seen:
            continue
        seen.add(checkpoint_key)
        if checkpoint.parent != root:
            raise ValueError(f"checkpoint path is outside allowed checkpoint root: {checkpoint}")
        if name_pattern is not None and re.fullmatch(name_pattern, checkpoint.name) is None:
            raise ValueError(f"checkpoint name does not match required pattern: {checkpoint}")
        if checkpoint_key in retained or not checkpoint.exists():
            continue
        if ready is not None and not ready(checkpoint) and not remove_incomplete:
            continue
        try:
            shutil.rmtree(checkpoint)
        except OSError as exc:
            if exc.errno == errno.ENOTEMPTY:
                continue
            raise
        removed.append(str(checkpoint))
    return removed


def _resolve(path: str | Path) -> str:
    return str(Path(path).resolve())
