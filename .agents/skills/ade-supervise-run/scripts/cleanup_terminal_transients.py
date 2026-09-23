#!/usr/bin/env python3
"""Remove exact Supervisor-owned preflight directories after a final ADE Run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from ade.engine.storage.atomic import write_json_atomic


FINAL_RUN_STATUSES = {"completed", "failed", "cancelled"}
ALLOWED_PREFLIGHT_PREFIXES = ("local-judge-preflight-",)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean explicitly recorded Supervisor preflight directories."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--path", type=Path, action="append", default=[])
    return parser


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validated_preflight_path(project_root: Path, path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    candidate = candidate.resolve()
    runs_root = (project_root / "runs").resolve()
    if candidate.parent != runs_root:
        raise ValueError(f"preflight path is not directly under {runs_root}: {candidate}")
    if not candidate.name.startswith(ALLOWED_PREFLIGHT_PREFIXES):
        raise ValueError(f"unsupported preflight directory name: {candidate.name}")
    return candidate


def main() -> int:
    args = _parser().parse_args()
    project_root = args.project_root.resolve()
    run_dir = (args.control_root.resolve() / args.run_id).resolve()
    state = _read_json(run_dir / "run.json")
    status = str(state.get("status") or "")
    if status not in FINAL_RUN_STATUSES:
        raise ValueError(
            f"Supervisor transient cleanup requires a final Run status, got {status!r}"
        )

    removed: list[str] = []
    already_absent: list[str] = []
    errors: list[str] = []
    paths = sorted(
        {_validated_preflight_path(project_root, path) for path in args.path},
        key=str,
    )
    for path in paths:
        if not path.exists() and not path.is_symlink():
            already_absent.append(str(path))
            continue
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(str(path))
        except OSError as error:
            errors.append(f"{path}: {error}")

    receipt = {
        "schema_version": "ade.supervisor_transient_cleanup.v1",
        "run_id": args.run_id,
        "run_status": status,
        "status": "complete" if not errors else "incomplete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "requested_paths": [str(path) for path in paths],
        "removed_paths": removed,
        "already_absent_paths": already_absent,
        "errors": errors,
    }
    receipt_path = run_dir / "reports" / "cleanup" / "supervisor-transients.json"
    write_json_atomic(receipt_path, receipt)
    print(json.dumps({**receipt, "receipt_path": str(receipt_path)}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
