#!/usr/bin/env python3
"""Build the pinned ADE benchmark catalog into Hugging Face disk datasets."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ade.harness.yaml_config import load_yaml_mapping  # noqa: E402


DEFAULT_CATALOG = PROJECT_ROOT / "configs" / "benchmarks" / "catalog.yaml"
EVALPLUS_RELEASES = {
    "humaneval_plus": (
        "v0.1.10",
        "https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz",
    ),
    "mbpp_plus": (
        "v0.2.0",
        "https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmarks", nargs="*", help="Catalog IDs; defaults to all catalog entries")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--force", action="store_true", help="Replace only the explicitly selected artifact directories")
    parser.add_argument("--check", action="store_true", help="Validate existing artifacts without downloading or writing")
    return parser.parse_args()


def load_catalog(path: Path) -> dict[str, Any]:
    payload = load_yaml_mapping(path)
    if payload.get("schema_version") != 1 or not isinstance(payload.get("benchmarks"), dict):
        raise ValueError(f"unsupported benchmark catalog: {path}")
    return payload


def _dataset_classes():
    from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk

    return Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk


def _artifact_path(project_root: Path, spec: dict[str, Any]) -> Path:
    path = (project_root / spec["artifact"]["path"]).resolve()
    data_root = (project_root / "data").resolve()
    path.relative_to(data_root)
    return path


def _row_id(row: dict[str, Any]) -> str:
    for key in ("uuid", "unique_id", "id", "task_id", "problem_id"):
        if row.get(key) is not None:
            return str(row[key])
    canonical = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _source_revision(spec: dict[str, Any]) -> str:
    source = spec["source"]
    return str(source.get("revision") or source["path"])


def _load_hf(spec: dict[str, Any]):
    _, _, _, load_dataset, _ = _dataset_classes()
    source = spec["source"]
    kwargs: dict[str, Any] = {"revision": source["revision"]}
    config = source.get("config")
    return load_dataset(source["path"], config, **kwargs) if config else load_dataset(source["path"], **kwargs)


def _source_split(dataset: Any, preferred: str) -> Any:
    _, DatasetDict, _, _, _ = _dataset_classes()
    if isinstance(dataset, DatasetDict):
        if preferred in dataset:
            return dataset[preferred]
        for name in ("test", "train", "validation"):
            if name in dataset:
                return dataset[name]
    return dataset


def _save_rows(rows: list[dict[str, Any]], target: Path, split: str) -> None:
    Dataset, DatasetDict, _, _, _ = _dataset_classes()
    DatasetDict({split: Dataset.from_list(rows)}).save_to_disk(str(target))


def _save_dataset(dataset: Any, target: Path, split: str) -> None:
    _, DatasetDict, _, _, _ = _dataset_classes()
    if isinstance(dataset, DatasetDict):
        dataset = DatasetDict({split: _source_split(dataset, split)})
    else:
        dataset = DatasetDict({split: dataset})
    dataset.save_to_disk(str(target))


def _build_evalplus(benchmark_id: str, target: Path, split: str) -> dict[str, Any]:
    release, url = EVALPLUS_RELEASES[benchmark_id]
    with urllib.request.urlopen(url, timeout=180) as response:
        payload = gzip.decompress(response.read()).decode("utf-8")
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    for row in rows:
        row["base_input_json"] = json.dumps(row.pop("base_input"), ensure_ascii=False)
        row["plus_input_json"] = json.dumps(row.pop("plus_input"), ensure_ascii=False)
    _save_rows(rows, target, split)
    return {"release": release, "url": url, "rows": len(rows)}


def _build_math_subset(
    spec: dict[str, Any], source_rows: list[dict[str, Any]], project_root: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    seed = int(spec["build"]["seed"])
    size = int(spec["build"]["size"])
    indices = sorted(random.Random(seed).sample(range(len(source_rows)), size))
    rows = [source_rows[index] for index in indices]
    ids = [_row_id(row) for row in rows]
    _write_id_manifest(project_root / spec["build"]["id_manifest"], ids)
    return rows, ids


def _load_local_rows(path: Path, split: str) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as parquet

        return [dict(row) for row in parquet.read_table(path).to_pylist()]
    _, _, _, _, load_from_disk = _dataset_classes()
    return [dict(row) for row in _source_split(load_from_disk(str(path)), split)]


def _build_derived_subset(
    spec: dict[str, Any],
    parent: dict[str, Any],
    target: Path,
    project_root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    parent_artifact = parent["artifact"]
    parent_path = (project_root / parent_artifact["path"]).resolve()
    rows = _load_local_rows(parent_path, str(parent_artifact["split"]))
    exclude_manifest = spec["build"].get("exclude_id_manifest")
    if exclude_manifest:
        excluded = {
            line.strip()
            for line in (project_root / str(exclude_manifest)).read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        }
        rows = [row for row in rows if _row_id(row) not in excluded]
    seed = int(spec["build"]["seed"])
    size = int(spec["build"]["size"])
    indices = sorted(random.Random(seed).sample(range(len(rows)), size))
    selected = [rows[index] for index in indices]
    ids = [_row_id(row) for row in selected]
    _write_id_manifest(project_root / spec["build"]["id_manifest"], ids)
    if target.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as parquet

        parquet.write_table(pa.Table.from_pylist(selected), target)
    else:
        _save_rows(selected, target, str(spec["artifact"]["split"]))
    return selected, ids


def _write_id_manifest(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def build_one(
    benchmark_id: str,
    spec: dict[str, Any],
    *,
    project_root: Path,
    catalog: dict[str, Any],
) -> dict[str, Any]:
    target = _artifact_path(project_root, spec)
    split = str(spec["artifact"]["split"])
    derived_from = spec.get("derived_from")
    if derived_from:
        parent = catalog.get(str(derived_from))
        if not isinstance(parent, dict):
            raise ValueError(f"{benchmark_id}: unknown derived_from={derived_from!r}")
        rows, ids = _build_derived_subset(
            spec,
            parent,
            target,
            project_root,
        )
        metadata = {
            "rows": len(rows),
            "derived_from": str(derived_from),
            "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        }
    elif benchmark_id in EVALPLUS_RELEASES:
        metadata = _build_evalplus(benchmark_id, target, split)
    else:
        local_source = spec.get("build", {}).get("local_source_artifact")
        if local_source and (project_root / local_source).exists():
            _, _, _, _, load_from_disk = _dataset_classes()
            dataset = load_from_disk(str((project_root / local_source).resolve()))
        else:
            dataset = _load_hf(spec)
        source = _source_split(dataset, split)
        if benchmark_id == "math_100":
            rows, ids = _build_math_subset(spec, [dict(row) for row in source], project_root)
            _save_rows(rows, target, split)
            metadata = {"rows": len(rows), "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}
        elif benchmark_id == "codeforces":
            rows = [dict(row) for row in source if row.get("official_tests") or row.get("examples")]
            _save_rows(rows, target, split)
            metadata = {"rows": len(rows), "runnable_tests_only": True}
        else:
            _save_dataset(dataset, target, split)
            metadata = {"rows": len(_source_split(dataset, split))}
    return {"path": str(target), "revision": _source_revision(spec), **metadata}


def validate_one(benchmark_id: str, spec: dict[str, Any], project_root: Path) -> dict[str, Any]:
    target = _artifact_path(project_root, spec)
    if not target.exists():
        raise FileNotFoundError(f"{benchmark_id}: missing artifact {target}")
    split = str(spec["artifact"]["split"])
    if target.suffix == ".parquet":
        import pyarrow.parquet as parquet

        rows = parquet.read_table(target)
    else:
        _, _, _, _, load_from_disk = _dataset_classes()
        rows = _source_split(load_from_disk(str(target)), split)
    expected = spec["artifact"].get("expected_rows")
    if expected is not None and len(rows) != int(expected):
        raise ValueError(f"{benchmark_id}: expected {expected} rows, found {len(rows)}")
    return {"path": str(target), "rows": len(rows), "revision": _source_revision(spec)}


def main() -> None:
    args = parse_args()
    catalog = load_catalog(args.catalog)
    specs = catalog["benchmarks"]
    selected = args.benchmarks or list(specs)
    unknown = sorted(set(selected) - set(specs))
    if unknown:
        raise SystemExit(f"unknown benchmark IDs: {', '.join(unknown)}")
    manifest_path = (
        args.project_root / "data" / "benchmarks" / "catalog_manifest.json"
    ).resolve()
    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    for benchmark_id in selected:
        spec = specs[benchmark_id]
        target = _artifact_path(args.project_root, spec)
        if args.check:
            manifest[benchmark_id] = validate_one(benchmark_id, spec, args.project_root)
            print(f"ok {benchmark_id}: {manifest[benchmark_id]['rows']} rows")
            continue
        if target.exists() and not args.force:
            manifest[benchmark_id] = validate_one(benchmark_id, spec, args.project_root)
            print(f"skip {benchmark_id}: {target}")
            continue
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        manifest[benchmark_id] = build_one(
            benchmark_id,
            spec,
            project_root=args.project_root,
            catalog=specs,
        )
        validate_one(benchmark_id, spec, args.project_root)
        print(f"built {benchmark_id}: {target}")
    if not args.check:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
