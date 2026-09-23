#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

from ade.tasks.data_selection.selection_contract import validate_selection_source


def fail(message: str) -> None:
    raise SystemExit(message)


def require_non_empty_utf8(path: Path) -> None:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        fail(f"{path}: must be UTF-8 text: {error}")
    if not content.strip():
        fail(f"{path}: must be non-empty")


def main() -> None:
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "output")
    try:
        files = {path.name for path in output.iterdir() if path.is_file()}
    except OSError as error:
        fail(f"{output}: cannot read output directory: {error}")
    if files != {"selection.py", "design.md"}:
        fail("output: require exactly selection.py and design.md")
    try:
        validate_selection_source(
            (output / "selection.py").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, SyntaxError, ValueError) as error:
        fail(f"selection.py: invalid selection contract: {error}")
    require_non_empty_utf8(output / "design.md")


if __name__ == "__main__":
    main()
