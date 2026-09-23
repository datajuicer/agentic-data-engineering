#!/usr/bin/env python3
"""Run one real ADE WorkflowDriver experiment smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ade.harness.workflow_smoke import run_real_workflow_smoke  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_config", type=Path)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-ticks", type=int, default=32)
    return parser


def main() -> int:
    args = _parser().parse_args()
    state, evidence_path = run_real_workflow_smoke(
        project_root=args.project_root,
        experiment_config=args.experiment_config,
        deployment_config=args.deployment,
        run_id=args.run_id,
        resume=args.resume,
        max_ticks=args.max_ticks,
    )
    print(
        json.dumps(
            {
                "run_id": state.run_id,
                "revision": state.revision,
                "status": state.status.value,
                "evidence_path": str(evidence_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
