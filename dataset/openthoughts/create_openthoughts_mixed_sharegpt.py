#!/usr/bin/env python3
"""Create one proportional OpenThoughts math/code/science ShareGPT mix."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

try:
    import pyarrow.parquet as pq
except (ImportError, OSError):  # pragma: no cover - environment guard
    pq = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover - environment guard
    pd = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data/raw_data/OpenThoughts-114k"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data/sft/openthoughts"
DEFAULT_OUTPUT_NAME = "openthoughts_mcs_3840_proportional.jsonl"
DEFAULT_GROUPS = {
    "math": ("math",),
    "code": ("code",),
    "science": ("biology", "physics", "chemistry"),
}
PROMPT_PREFIXES = (
    "Generate an executable Python function generated from the given prompt. "
    "The function should take stdin as input and print the output. "
    "Simply call the function after the definition.",
    "Generate an executable Python function generated from the given prompt. "
    "Return the function body without invoking it at the final solution.",
    "Return your final response within \\boxed{}.",
)


def _parquet_files(path: Path) -> list[Path]:
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found under {path}")
    return files


def _iter_parquet_records(file_path: Path, columns: list[str]):
    if pq is not None:
        parquet_file = pq.ParquetFile(file_path)
        missing = set(columns).difference(parquet_file.schema_arrow.names)
        if missing:
            raise ValueError(f"{file_path} does not contain columns: {sorted(missing)}")
        for batch in parquet_file.iter_batches(columns=columns):
            batch_dict = batch.to_pydict()
            for offset in range(batch.num_rows):
                yield {column: batch_dict[column][offset] for column in columns}
        return

    if pd is None:
        raise RuntimeError(
            "reading parquet requires pyarrow or pandas. "
            "Run with .unified-vllm-0.19.1-verl-venv/bin/python."
        )

    frame = pd.read_parquet(file_path, columns=columns)
    for record in frame.to_dict(orient="records"):
        yield record


def _normalize_template(text: str) -> str:
    replacements = (
        ("<|begin_of_thought|>", "<think>"),
        ("<|end_of_thought|>", "</think>"),
        ("<|begin_of_solution|>", ""),
        ("<|end_of_solution|>", ""),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text.strip()


def _metadata_key(text: Any) -> str:
    return "" if text is None else str(text).strip()


def _problem_from_user_prompt(text: str) -> str:
    text = text.strip()
    for prefix in PROMPT_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


def _normalize_conversations(conversations: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in conversations:
        role = str(message.get("from", ""))
        value = "" if message.get("value") is None else str(message.get("value"))
        if role == "assistant":
            value = _normalize_template(value)
        normalized.append({"from": role, "value": value})
    return normalized


def _group_for_domain(domain: str) -> str | None:
    for group, domains in DEFAULT_GROUPS.items():
        if domain in domains:
            return group
    return None


def _collect_problem_groups(metadata_files: list[Path]) -> tuple[dict[str, str], dict[str, int], int]:
    problem_to_group: dict[str, str] = {}
    domain_counts: dict[str, int] = {}
    metadata_rows = 0

    for file_path in metadata_files:
        for record in _iter_parquet_records(file_path, ["problem", "domain"]):
            domain = record["domain"]
            key = str(domain) if domain is not None else "<missing>"
            domain_counts[key] = domain_counts.get(key, 0) + 1
            group = _group_for_domain(key)
            if group is not None:
                problem = _metadata_key(record["problem"])
                if problem and problem not in problem_to_group:
                    problem_to_group[problem] = group
            metadata_rows += 1

    return problem_to_group, domain_counts, metadata_rows


def _iter_classified_rows(
    data_files: Iterable[Path],
    problem_to_group: dict[str, str],
) -> tuple[dict[str, int], dict[str, int], Iterable[tuple[str, dict[str, Any]]]]:
    match_counts: dict[str, int] = {"matched": 0, "unmatched": 0}
    prefix_counts: dict[str, int] = {}

    def rows():
        for file_path in data_files:
            for record in _iter_parquet_records(file_path, ["conversations"]):
                conversations = record["conversations"]
                user_value = ""
                if len(conversations) > 0:
                    user_value = "" if conversations[0].get("value") is None else str(conversations[0]["value"])
                stripped = _problem_from_user_prompt(user_value)
                prefix_name = "no_known_prefix" if stripped == user_value.strip() else "known_prefix"
                prefix_counts[prefix_name] = prefix_counts.get(prefix_name, 0) + 1
                group = problem_to_group.get(_metadata_key(stripped))
                if group is None:
                    match_counts["unmatched"] += 1
                    continue
                match_counts["matched"] += 1
                yield group, {"conversations": _normalize_conversations(conversations)}

    return match_counts, prefix_counts, rows()


def _allocate_counts(available_counts: dict[str, int], sample_size: int) -> dict[str, int]:
    total_available = sum(available_counts.values())
    if total_available <= 0:
        raise ValueError("no matched math/code/science rows found")
    if sample_size > total_available:
        raise ValueError(f"sample_size={sample_size} exceeds available rows={total_available}")

    raw = {
        group: sample_size * available / total_available
        for group, available in available_counts.items()
    }
    allocations = {group: int(value) for group, value in raw.items()}
    remaining = sample_size - sum(allocations.values())
    for group, _ in sorted(raw.items(), key=lambda item: (item[1] - int(item[1]), item[0]), reverse=True):
        if remaining == 0:
            break
        allocations[group] += 1
        remaining -= 1

    for group, count in allocations.items():
        if count > available_counts[group]:
            raise ValueError(f"allocated {count} rows for {group}, but only {available_counts[group]} available")
    return allocations


def _sample_rows_by_group(
    classified_rows: Iterable[tuple[str, dict[str, Any]]],
    *,
    allocations: dict[str, int],
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    rngs = {group: random.Random(f"{seed}:{group}") for group in DEFAULT_GROUPS}
    sampled: dict[str, list[dict[str, Any]]] = {group: [] for group in DEFAULT_GROUPS}
    seen_counts: dict[str, int] = {group: 0 for group in DEFAULT_GROUPS}

    for group, row in classified_rows:
        seen_counts[group] += 1
        target = allocations[group]
        if target == 0:
            continue
        reservoir = sampled[group]
        if len(reservoir) < target:
            reservoir.append(row)
            continue
        replace_at = rngs[group].randrange(seen_counts[group])
        if replace_at < target:
            reservoir[replace_at] = row

    for group, target in allocations.items():
        if len(sampled[group]) != target:
            raise RuntimeError(f"sampled {len(sampled[group])} rows for {group}, expected {target}")
    return sampled, seen_counts


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing OpenThoughts data/ and metadata/ parquet subdirectories.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--sample-size", type=int, default=3840)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata_files = _parquet_files(args.dataset_dir / "metadata")
    data_files = _parquet_files(args.dataset_dir / "data")

    problem_to_group, domain_counts, metadata_rows = _collect_problem_groups(metadata_files)
    match_counts, prefix_counts, classified_rows = _iter_classified_rows(data_files, problem_to_group)

    materialized_rows = list(classified_rows)
    available_counts = {group: 0 for group in DEFAULT_GROUPS}
    for group, _ in materialized_rows:
        available_counts[group] += 1
    allocations = _allocate_counts(available_counts, args.sample_size)
    sampled, seen_counts = _sample_rows_by_group(materialized_rows, allocations=allocations, seed=args.seed)

    mixed_rows: list[dict[str, Any]] = []
    for group in DEFAULT_GROUPS:
        mixed_rows.extend(sampled[group])
    random.Random(f"{args.seed}:mixed").shuffle(mixed_rows)

    output_path = args.out_dir / args.output_name
    _write_jsonl(output_path, mixed_rows)

    summary = {
        "dataset_dir": str(args.dataset_dir),
        "out_dir": str(args.out_dir),
        "output_path": str(output_path),
        "metadata_rows": metadata_rows,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "jsonl_format": "sharegpt",
        "jsonl_fields": ["conversations"],
        "system_prompt_included": False,
        "grouping": {
            "math": list(DEFAULT_GROUPS["math"]),
            "code": list(DEFAULT_GROUPS["code"]),
            "science": list(DEFAULT_GROUPS["science"]),
        },
        "available_counts": available_counts,
        "sampled_counts": {group: len(rows) for group, rows in sampled.items()},
        "sampled_ratio": {
            group: len(rows) / len(mixed_rows)
            for group, rows in sampled.items()
        },
        "metadata_problem_keys": len(problem_to_group),
        "data_match_counts": match_counts,
        "data_prefix_counts": prefix_counts,
        "domain_counts": dict(sorted(domain_counts.items())),
        "seen_counts": seen_counts,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {len(mixed_rows)} rows to {output_path}")
    for group in DEFAULT_GROUPS:
        print(
            f"{group}: sampled={summary['sampled_counts'][group]} "
            f"available={available_counts[group]} ratio={summary['sampled_ratio'][group]:.6f}"
        )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
