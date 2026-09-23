from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq


DOMAINS = ("math", "code", "science")
QWEN_MATH_SOURCE_ALIASES = {
    "math__deepscaler_preview": "math_qwen__deepscaler_preview",
    "math__merged_deduped_dapo_or1_dataset": (
        "math_qwen__merged_deduped_dapo_or1_dataset"
    ),
}
REQUIRED_FIELDS = {
    "data_source",
    "prompt",
    "apply_chat_template",
    "reward_model",
    "extra_info",
}


def require_project_local_path(path: str | Path, *, project_root: str | Path) -> Path:
    resolved = Path(path).resolve()
    root = Path(project_root).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"RFT data path must be inside ADE project root: {resolved}")
    return resolved


def import_and_prepare(
    *,
    upstream_root: str | Path,
    project_root: str | Path,
    prompt_paths: Mapping[str, str | Path],
    expected_rows: int = 3000,
) -> dict[str, Any]:
    upstream_root = Path(upstream_root).resolve()
    project_root = Path(project_root).resolve()
    data_root = project_root / "data" / "rft" / "guru_3k"
    source_root = data_root / "source"
    prepared_root = data_root / "prepared" / "plain_step_by_step"
    smoke_root = prepared_root / "smoke"
    source_root.mkdir(parents=True, exist_ok=True)
    prepared_root.mkdir(parents=True, exist_ok=True)
    smoke_root.mkdir(parents=True, exist_ok=True)

    validated: dict[str, tuple[Path, pa.Table, str]] = {}
    for domain in DOMAINS:
        upstream = upstream_root / f"{domain}_3k.parquet"
        table = pq.read_table(upstream)
        _validate_table(table, path=upstream, expected_rows=expected_rows)
        prompt = Path(prompt_paths[domain]).read_text(encoding="utf-8").rstrip("\n")
        if not prompt:
            raise ValueError(f"{domain} system prompt must not be empty")
        validated[domain] = (upstream, table, prompt)

    domains: dict[str, Any] = {}
    for domain, (upstream, table, prompt) in validated.items():
        source = require_project_local_path(
            source_root / f"{domain}_3k.parquet", project_root=project_root
        )
        prepared = require_project_local_path(
            prepared_root / f"{domain}_3k.parquet", project_root=project_root
        )
        shutil.copyfile(upstream, source)
        if _sha256(source) != _sha256(upstream):
            raise RuntimeError(f"source copy digest mismatch for {domain}")

        transformed = _with_system_prompt(table, prompt)
        pq.write_table(transformed, prepared)
        smoke = require_project_local_path(
            smoke_root / f"{domain}_16.parquet", project_root=project_root
        )
        smoke_table = transformed.slice(0, min(16, transformed.num_rows))
        pq.write_table(smoke_table, smoke)
        if domain == "math":
            qwen_outcome = require_project_local_path(
                prepared_root / "math_3k_qwen_outcome.parquet",
                project_root=project_root,
            )
            qwen_outcome_table = with_qwen_math_source_ids(transformed)
            pq.write_table(qwen_outcome_table, qwen_outcome)
        else:
            qwen_outcome = None
            qwen_outcome_table = None
        domains[domain] = {
            "source": _identity(source, row_count=table.num_rows, upstream=upstream),
            "prepared": _identity(prepared, row_count=transformed.num_rows),
            "smoke": _identity(smoke, row_count=smoke_table.num_rows),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "transformation": "prepend_system_message_v1",
        }
        if qwen_outcome is not None and qwen_outcome_table is not None:
            domains[domain]["qwen_outcome"] = _identity(
                qwen_outcome,
                row_count=qwen_outcome_table.num_rows,
            )
            domains[domain]["qwen_outcome_transformation"] = (
                "replace_data_source_for_math_qwen_outcome_v1"
            )

    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expected_rows_per_domain": expected_rows,
        "copy_command": "scripts/prepare_rft_guru_3k.py",
        "domains": domains,
    }
    _write_json(data_root / "manifest.json", manifest)
    _write_json(prepared_root / "manifest.json", manifest)
    return manifest


def _validate_table(table: pa.Table, *, path: Path, expected_rows: int) -> None:
    if table.num_rows != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, found {table.num_rows}")
    missing = sorted(REQUIRED_FIELDS - set(table.column_names))
    if missing:
        raise ValueError(f"{path}: missing required fields: {', '.join(missing)}")
    for index, row in enumerate(table.select(sorted(REQUIRED_FIELDS)).to_pylist()):
        prompt = row["prompt"]
        reward_model = row["reward_model"]
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(f"{path}: row {index} prompt must be a non-empty list")
        if not isinstance(reward_model, dict) or reward_model.get("ground_truth") is None:
            raise ValueError(f"{path}: row {index} reward_model.ground_truth is required")
        if not isinstance(row["extra_info"], dict):
            raise ValueError(f"{path}: row {index} extra_info must be a mapping")
        if row["apply_chat_template"] is not True:
            raise ValueError(f"{path}: row {index} apply_chat_template must be true")


def _with_system_prompt(table: pa.Table, system_prompt: str) -> pa.Table:
    rows = table.to_pylist()
    for row in rows:
        row["prompt"] = [
            {"role": "system", "content": system_prompt},
            *row["prompt"],
        ]
    return pa.Table.from_pylist(rows, schema=table.schema)


def with_qwen_math_source_ids(table: pa.Table) -> pa.Table:
    """Give Guru math rows a distinct identity for the Qwen Math outcome port."""
    rows = table.to_pylist()
    observed = {str(row["data_source"]) for row in rows}
    unknown = sorted(observed - set(QWEN_MATH_SOURCE_ALIASES))
    if unknown:
        raise ValueError(f"unsupported Guru math data_source values: {unknown}")
    for row in rows:
        row["data_source"] = QWEN_MATH_SOURCE_ALIASES[str(row["data_source"])]
    return pa.Table.from_pylist(rows, schema=table.schema)


def _identity(
    path: Path,
    *,
    row_count: int,
    upstream: Path | None = None,
) -> dict[str, Any]:
    table = pq.read_table(path)
    identity = {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
        "row_count": row_count,
        "schema_sha256": hashlib.sha256(str(table.schema).encode("utf-8")).hexdigest(),
    }
    if upstream is not None:
        identity["upstream_path"] = str(upstream)
    return identity


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--math-prompt", required=True)
    parser.add_argument("--code-prompt", required=True)
    parser.add_argument("--science-prompt", required=True)
    args = parser.parse_args()
    manifest = import_and_prepare(
        upstream_root=args.upstream_root,
        project_root=args.project_root,
        prompt_paths={
            "math": args.math_prompt,
            "code": args.code_prompt,
            "science": args.science_prompt,
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
