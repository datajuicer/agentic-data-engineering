#!/usr/bin/env python3
import ast
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "output")
files = {path.name for path in root.iterdir()} if root.is_dir() else set()
if files != {"curriculum.py", "design.md"} or not (root / "design.md").read_text().strip():
    raise SystemExit("expected only non-empty curriculum.py and design.md")
tree = ast.parse((root / "curriculum.py").read_text(), filename="curriculum.py")
functions = [node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)]
expected = ["candidate_inventory", "total_steps", "prompts_per_step", "judge_batch"]
if len(functions) != 1 or functions[0].name != "build_curriculum" or [arg.arg for arg in functions[0].args.args] != expected:
    raise SystemExit("invalid build_curriculum entrypoint")
