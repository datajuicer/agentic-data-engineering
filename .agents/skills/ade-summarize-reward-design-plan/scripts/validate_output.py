#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


root = Path(sys.argv[1] if len(sys.argv) > 1 else "output")
files = {path.name for path in root.iterdir()} if root.is_dir() else set()
try:
    memory = (root / "MEMORY.md").read_text(encoding="utf-8")
except (OSError, UnicodeDecodeError):
    memory = ""
if files != {"MEMORY.md"} or not memory.strip():
    raise SystemExit("expected only non-empty UTF-8 MEMORY.md")
if memory.count("## Trial Conclusions") != 1:
    raise SystemExit("MEMORY.md requires exactly one Trial Conclusions block")
fields = {
    "Realization status: ": {"verified", "deviated", "unverified"},
    "Hypothesis result: ": {"supported", "rejected", "inconclusive"},
    "Portfolio result: ": {"new_best", "not_new_best", "not_comparable"},
}
for prefix, allowed in fields.items():
    values = [
        line.removeprefix(prefix)
        for line in memory.splitlines()
        if line.startswith(prefix)
    ]
    if len(values) != 1 or values[0] not in allowed:
        raise SystemExit(f"MEMORY.md requires one valid {prefix.strip()}")
