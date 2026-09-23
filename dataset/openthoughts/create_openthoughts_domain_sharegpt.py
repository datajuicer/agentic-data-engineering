#!/usr/bin/env python3
"""Create OpenThoughts domain JSONL files in ShareGPT format.

The script uses metadata.problem to map each data prompt to its metadata
domain. Output rows do not include the OpenThoughts system prompt.
"""

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


DEFAULT_DATASET_DIR = Path(
    "data/raw_data/OpenThoughts-114k"
)
DEFAULT_OUT_DIR = Path("data/openthoughts")
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
            "Run with .unified-vllm-0.19.1-venv/bin/python."
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


def _sample_rows_by_group(
    classified_rows: Iterable[tuple[str, dict[str, Any]]],
    *,
    sample_size: int,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    rngs = {group: random.Random(f"{seed}:{group}") for group in DEFAULT_GROUPS}
    sampled: dict[str, list[dict[str, Any]]] = {group: [] for group in DEFAULT_GROUPS}
    seen_counts: dict[str, int] = {group: 0 for group in DEFAULT_GROUPS}
    summary: dict[str, dict[str, Any]] = {}

    for group, row in classified_rows:
        seen_counts[group] += 1
        seen = seen_counts[group]
        reservoir = sampled[group]
        if len(reservoir) < sample_size:
            reservoir.append(row)
            continue
        replace_at = rngs[group].randrange(seen)
        if replace_at < sample_size:
            reservoir[replace_at] = row

    for group_name, domains in DEFAULT_GROUPS.items():
        available = seen_counts[group_name]
        sampled_rows = len(sampled[group_name])
        summary[group_name] = {
            "domains": list(domains),
            "requested_rows": sample_size,
            "available_rows": available,
            "sampled_rows": sampled_rows,
            "shortfall_rows": max(0, sample_size - available),
            "with_replacement": False,
        }

    return sampled, summary


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
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for math/code/science JSONL files.",
    )
    parser.add_argument("--sample-size", type=int, default=3840, help="Rows to sample per group.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    metadata_files = _parquet_files(args.dataset_dir / "metadata")
    data_files = _parquet_files(args.dataset_dir / "data")

    problem_to_group, domain_counts, metadata_rows = _collect_problem_groups(metadata_files)
    match_counts, prefix_counts, classified_rows = _iter_classified_rows(data_files, problem_to_group)
    sampled, group_summary = _sample_rows_by_group(
        classified_rows,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    all_rows: list[dict[str, Any]] = []
    output_paths: dict[str, str] = {}
    for group_name, group_rows in sampled.items():
        output_path = args.out_dir / f"{group_name}.jsonl"
        _write_jsonl(output_path, group_rows)
        output_paths[group_name] = str(output_path)
        all_rows.extend(group_rows)

    all_path = args.out_dir / "all.jsonl"
    _write_jsonl(all_path, all_rows)

    summary = {
        "dataset_dir": str(args.dataset_dir),
        "out_dir": str(args.out_dir),
        "metadata_rows": metadata_rows,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "jsonl_format": "sharegpt",
        "jsonl_fields": ["conversations"],
        "system_prompt_included": False,
        "groups": group_summary,
        "metadata_problem_keys": len(problem_to_group),
        "data_match_counts": match_counts,
        "data_prefix_counts": prefix_counts,
        "domain_counts": dict(sorted(domain_counts.items())),
        "outputs": {"all": str(all_path), **output_paths},
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {len(all_rows)} rows to {all_path}")
    for group_name, info in group_summary.items():
        print(
            f"{group_name}: sampled={info['sampled_rows']} "
            f"available={info['available_rows']} domains={','.join(info['domains'])}"
        )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
