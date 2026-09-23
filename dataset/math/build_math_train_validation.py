#!/usr/bin/env python3
"""Build a MATH-train validation split aligned to MATH500."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
SUBJECT_NAMES = {
    "counting_and_probability": "Counting & Probability",
    "intermediate_algebra": "Intermediate Algebra",
}


def _norm(text: Any) -> str:
    return " ".join(str(text or "").split())


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _boxed_answer(solution: str) -> str:
    """Extract the final balanced ``\\boxed{...}`` without symbolic parsing."""
    marker = r"\boxed{"
    start = solution.rfind(marker)
    if start < 0:
        plain_marker = r"\boxed"
        start = solution.rfind(plain_marker)
        if start < 0:
            raise ValueError("MATH solution has no boxed final answer")
        plain = solution[start + len(plain_marker) :].strip().strip("$").strip()
        return plain.rstrip(". ")
    index = start + len(marker)
    depth = 1
    while index < len(solution) and depth:
        if solution[index] == "{":
            depth += 1
        elif solution[index] == "}":
            depth -= 1
        index += 1
    if depth:
        raise ValueError("MATH solution has an unbalanced boxed final answer")
    return solution[start + len(marker) : index - 1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(f"{row['level']}|{row['subject']}" for row in rows).items()))


def _load_original_train(revision: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in MATH_CONFIGS:
        dataset = load_dataset(
            "EleutherAI/hendrycks_math",
            config,
            revision=revision,
            split="train",
        )
        subject = SUBJECT_NAMES.get(config, config.replace("_", " ").title())
        for index, row in enumerate(dataset):
            raw_level = str(row["level"])
            rows.append(
                {
                    "problem": str(row["problem"]),
                    "solution": str(row["solution"]),
                    "subject": subject,
                    # The original source contains two "Level ?" rows. Keep
                    # them in training with an explicit sentinel; they cannot
                    # contribute to the MATH500 level-stratified validation.
                    "level": int(raw_level.split()[-1]) if raw_level.split()[-1].isdigit() else 0,
                    "source_level": raw_level,
                    "unique_id": f"train/{config}/{index}",
                }
            )
    return rows


def _load_math500(path: Path) -> list[dict[str, Any]]:
    dataset = load_from_disk(str(path.resolve()))
    split = dataset["test"] if isinstance(dataset, DatasetDict) else dataset
    return [dict(row) for row in split]


def _largest_remainder(targets: Counter[tuple[int, str]], total: int) -> dict[tuple[int, str], int]:
    denominator = sum(targets.values())
    raw = {key: total * value / denominator for key, value in targets.items()}
    allocation = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(allocation.values())
    order = sorted(
        raw,
        key=lambda key: (raw[key] - allocation[key], key),
        reverse=True,
    )
    for key in order[:remainder]:
        allocation[key] += 1
    return allocation


def _select_validation(
    source: list[dict[str, Any]],
    target: list[dict[str, Any]],
    size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    target_strata = Counter((int(row["level"]), str(row["subject"])) for row in target)
    quotas = _largest_remainder(target_strata, size)
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source:
        groups[(int(row["level"]), str(row["subject"]))].append(row)
    selected: list[dict[str, Any]] = []
    for offset, key in enumerate(sorted(quotas)):
        candidates = sorted(groups[key], key=lambda row: row["unique_id"])
        if len(candidates) < quotas[key]:
            raise ValueError(f"MATH train lacks capacity for stratum {key}: {len(candidates)} < {quotas[key]}")
        rng = np.random.default_rng(seed + offset)
        indices = sorted(int(index) for index in rng.choice(len(candidates), quotas[key], replace=False))
        selected.extend(candidates[index] for index in indices)
    selected.sort(key=lambda row: row["unique_id"])
    return selected, {f"{level}|{subject}": count for (level, subject), count in sorted(quotas.items())}


def _select_training(source: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    if size < 1 or size > len(source):
        raise ValueError(f"invalid MATH training size {size} for {len(source)} candidates")
    ordered = sorted(source, key=lambda row: row["unique_id"])
    rng = np.random.default_rng(seed)
    indices = sorted(int(index) for index in rng.choice(len(ordered), size, replace=False))
    selected = [ordered[index] for index in indices]
    selected.sort(key=lambda row: row["unique_id"])
    return selected


def _to_training_row(row: dict[str, Any], answer: str) -> dict[str, Any]:
    prompt = (
        f"{row['problem']}\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    return {
        "answer": answer,
        "gt_answer": answer,
        "subject": row["subject"],
        "level": row["level"],
        "question": row["problem"],
        "target": answer,
        "data_source": "math",
        "prompt": [{"content": prompt, "role": "user"}],
        "ability": "math",
        "reward_model": {"ground_truth": answer, "style": "rule"},
        "extra_info": {
            "answer": answer,
            "index": row["unique_id"],
            "level": row["level"],
            "question": row["problem"],
            "split": "train",
        },
        "unique_id": row["unique_id"],
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_revision = args.source_revision
    source = _load_original_train(source_revision)
    math500 = _load_math500(args.math500)
    print(f"loaded MATH train={len(source)} MATH500={len(math500)}", flush=True)
    if len(source) != 7500:
        raise ValueError(f"expected 7,500 original MATH train rows, found {len(source)}")
    if len(math500) != args.validation_size:
        raise ValueError(f"expected MATH500 to contain {args.validation_size} rows, found {len(math500)}")

    validation, quotas = _select_validation(source, math500, args.validation_size, args.seed)
    validation_questions = {_norm(row["problem"]) for row in validation}
    math500_questions = {_norm(row["problem"]) for row in math500}
    if validation_questions & math500_questions:
        raise RuntimeError("new validation overlaps MATH500")
    validation_ids = {row["unique_id"] for row in validation}
    train_pool = [row for row in source if row["unique_id"] not in validation_ids]
    if len(train_pool) != len(source) - args.validation_size:
        raise RuntimeError("training/validation split has duplicate or missing IDs")
    train = _select_training(train_pool, args.train_size, args.seed + 100000)
    print(f"selected validation={len(validation)} train_pool={len(train_pool)} train={len(train)}", flush=True)

    # The shared Qwen math extractor is the same reference construction used by
    # the MATH500 evaluator, so train and validation use identical answer semantics.
    def answer(row: dict[str, Any]) -> str:
        return _boxed_answer(row["solution"]).strip().rstrip(".").strip("$").strip()

    train_rows: list[dict[str, Any]] = []
    for index, row in enumerate(train):
        train_rows.append(_to_training_row(row, answer(row)))
        if index and index % 500 == 0:
            print(f"extracted train references={index}", flush=True)
    validation_rows: list[dict[str, Any]] = []
    for row in validation:
        validation_rows.append(
            {
                "problem": row["problem"],
                "solution": row["solution"],
                "answer": answer(row),
                "subject": row["subject"],
                "level": row["level"],
                "unique_id": f"validation/{_slug(str(row['subject']))}/{row['unique_id']}",
            }
        )
    print("extracted references", flush=True)
    validation_train_rows = [
        _to_training_row(row, validation_rows[index]["answer"])
        for index, row in enumerate(validation)
    ]

    args.train_output.mkdir(parents=True, exist_ok=True)
    args.validation_output.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(train_rows), args.train_output / "train.parquet")
    pq.write_table(
        pa.Table.from_pylist(validation_train_rows),
        args.validation_output / "validation.parquet",
    )
    DatasetDict({"test": Dataset.from_list(validation_rows)}).save_to_disk(str(args.validation_output))
    print("wrote artifacts", flush=True)

    manifest = {
        "schema_version": 1,
        "source": "EleutherAI/hendrycks_math",
        "source_revision": source_revision,
        "source_train_rows": len(source),
        "math500": str(args.math500.resolve()),
        "math500_rows": len(math500),
        "seed": args.seed,
        "validation_size": len(validation_rows),
        "training_size": len(train_rows),
        "training_pool_size": len(train_pool),
        "split_method": "MATH500_subject_level_joint_largest_remainder_per_stratum_seeded_sampling_from_MATH_train",
        "validation_target_distribution": _counts(math500),
        "validation_distribution": _counts(validation),
        "training_distribution": _counts(train),
        "overlap": {
            "train_validation_ids": len({row["unique_id"] for row in train} & validation_ids),
            "train_validation_questions": len({_norm(row["problem"]) for row in train} & validation_questions),
            "validation_math500_questions": len(validation_questions & math500_questions),
            "train_math500_questions": len({_norm(row["problem"]) for row in train} & math500_questions),
        },
        "train_path": str((args.train_output / "train.parquet").resolve()),
        "validation_path": str(args.validation_output.resolve()),
        "validation_parquet_path": str((args.validation_output / "validation.parquet").resolve()),
    }
    manifest["train_sha256"] = _sha256(args.train_output / "train.parquet")
    print("computed manifest", flush=True)
    manifest_path = args.train_output / "split.manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"manifest": str(manifest_path), "quotas": quotas, **manifest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-revision", default="21a5633873b6a120296cce3e2df9d5550074f4a3")
    parser.add_argument("--math500", type=Path, default=PROJECT_ROOT / "data/benchmarks/math-500")
    parser.add_argument("--train-output", type=Path, default=PROJECT_ROOT / "data/rft/math/math_train3072_val500_seed42")
    parser.add_argument("--train-size", type=int, default=3072)
    parser.add_argument("--validation-output", type=Path, default=PROJECT_ROOT / "data/benchmarks/math-val")
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.train_output.exists() or args.validation_output.exists():
        raise FileExistsError("refusing to overwrite existing MATH split; remove exact outputs first")
    print(json.dumps(build(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
