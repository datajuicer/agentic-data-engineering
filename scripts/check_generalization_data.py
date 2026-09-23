#!/usr/bin/env python3
"""Verify the original MATH validation artifact used by generalization evals."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ade.harness.yaml_config import load_yaml_mapping  # noqa: E402


_SFT_WRAPPER = "Return your final response within \\boxed{}. "


def _normalized_text(value: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", value).casefold(),
    ).strip()


def _training_question(value: str) -> str:
    normalized = _normalized_text(value)
    prefix = _normalized_text(_SFT_WRAPPER)
    return normalized[len(prefix) :].strip() if normalized.startswith(prefix) else normalized


def _arrow_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.ipc as ipc

    rows: list[dict[str, Any]] = []
    for shard in sorted(path.glob("*.arrow")):
        with shard.open("rb") as handle:
            rows.extend(ipc.open_stream(handle).read_all().to_pylist())
    if not rows:
        raise ValueError(f"no Arrow rows found under {path}")
    return rows


def _parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as parquet

    return parquet.read_table(path).to_pylist()


def check(project_root: Path) -> dict[str, object]:
    root = project_root.resolve()
    catalog = load_yaml_mapping(root / "configs/benchmarks/catalog.yaml")
    benchmarks = catalog.get("benchmarks")
    if not isinstance(benchmarks, dict):
        raise ValueError("benchmark catalog must contain a benchmarks mapping")
    math_val = benchmarks.get("math_val")
    if not isinstance(math_val, dict):
        raise ValueError("math_val must be cataloged")

    source_rows = _arrow_rows(root / "data/benchmarks/math-val/test")
    validation_path = root / str(math_val["artifact"]["path"])
    validation_rows = _parquet_rows(validation_path)
    if len(source_rows) != 500 or len(validation_rows) != 500:
        raise ValueError("math_val artifacts must both contain 500 rows")
    for index, (source, derived) in enumerate(zip(source_rows, validation_rows)):
        same = (
            source["problem"] == derived["question"] == derived["extra_info"]["question"]
            and source["answer"]
            == derived["answer"]
            == derived["gt_answer"]
            == derived["target"]
            == derived["extra_info"]["answer"]
            and source["subject"] == derived["subject"]
            and source["level"] == derived["level"] == derived["extra_info"]["level"]
            and str(source["unique_id"]).endswith(f"/{derived['unique_id']}")
            and derived["unique_id"] == derived["extra_info"]["index"]
        )
        if not same:
            raise ValueError(f"math_val semantic identity mismatch at row {index}")

    split_manifest = json.loads(
        (root / str(math_val["source"]["split_manifest"])).read_text(encoding="utf-8")
    )
    split_overlap = split_manifest.get("overlap")
    if not isinstance(split_overlap, dict) or any(int(value) for value in split_overlap.values()):
        raise ValueError("math_val overlaps its RFT train split or MATH-500")

    training_path = (
        root / "data/sft/openthoughts/openthoughts_mcs_3840_proportional.jsonl"
    )
    training_questions: set[str] = set()
    training_rows = 0
    with training_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            training_rows += 1
            item = json.loads(line)
            conversations = item.get("conversations")
            if not isinstance(conversations, list) or not conversations:
                raise ValueError("OpenThoughts row has no conversations")
            training_questions.add(_training_question(str(conversations[0]["value"])))
    overlap_ids = [
        str(row["unique_id"])
        for row in validation_rows
        if _normalized_text(str(row["question"])) in training_questions
    ]
    if training_rows != 3840:
        raise ValueError(f"OpenThoughts candidate pool must contain 3840 rows, got {training_rows}")
    if len(overlap_ids) != 13:
        raise ValueError(
            "expected the frozen math_val/OpenThoughts overlap count to be 13, "
            f"got {len(overlap_ids)}"
        )

    return {
        "schema_version": 1,
        "status": "passed",
        "math_val_rows": len(validation_rows),
        "math_val_semantic_identity": True,
        "rft_overlap": split_overlap,
        "openthoughts_rows": training_rows,
        "openthoughts_overlap_rows": len(overlap_ids),
        "sft_math_dataset": "math_val",
        "sft_math_filtering": "none",
        "sft_math_rows_used": len(validation_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    print(json.dumps(check(args.project_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
