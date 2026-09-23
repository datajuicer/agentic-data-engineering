#!/usr/bin/env python3
"""Compute chat-template token length distributions for OpenThoughts JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from transformers import AutoTokenizer


DEFAULT_DATA_DIR = Path("data/openthoughts")
DEFAULT_TOKENIZER = Path("models/Qwen2.5-7B-Instruct")
DEFAULT_OUT_DIR = DEFAULT_DATA_DIR / "token_stats"
DEFAULT_FILES = ("math.jsonl", "code.jsonl", "science.jsonl")
DEFAULT_QUANTILES = (0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
DEFAULT_THRESHOLDS = (4096, 8192, 16384, 32768, 65536)


def _to_chat_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for message in row["conversations"]:
        role = message.get("from")
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        messages.append(
            {
                "role": str(role),
                "content": "" if message.get("value") is None else str(message["value"]),
            }
        )
    return messages


def _quantiles(values: list[int], points: tuple[float, ...]) -> dict[str, int]:
    ordered = sorted(values)
    last = len(ordered) - 1
    return {f"p{int(point * 100):02d}": ordered[round(last * point)] for point in points}


def _length_from_chat_template(tokenizer, messages: list[dict[str, str]]) -> int:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        truncation=False,
    )
    return len(token_ids)


def analyze_file(
    path: Path,
    tokenizer,
    *,
    quantiles: tuple[float, ...],
    thresholds: tuple[int, ...],
) -> tuple[dict[str, Any], list[int]]:
    lengths: list[int] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                messages = _to_chat_messages(row)
                lengths.append(_length_from_chat_template(tokenizer, messages))
            except Exception as exc:
                raise RuntimeError(f"failed to tokenize {path}:{line_number}") from exc

    if not lengths:
        raise ValueError(f"no rows found in {path}")

    threshold_counts = {f"over_{threshold}": sum(length > threshold for length in lengths) for threshold in thresholds}
    summary = {
        "file": str(path),
        "rows": len(lengths),
        "mean": mean(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "quantiles": _quantiles(lengths, quantiles),
        **threshold_counts,
    }
    return summary, lengths


def _write_summary_csv(path: Path, summaries: list[dict[str, Any]], thresholds: tuple[int, ...]) -> None:
    quantile_keys = list(summaries[0]["quantiles"].keys())
    threshold_keys = [f"over_{threshold}" for threshold in thresholds]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "rows", "mean", "min", "max", *quantile_keys, *threshold_keys])
        for item in summaries:
            quantiles = item["quantiles"]
            writer.writerow(
                [
                    Path(item["file"]).name,
                    item["rows"],
                    f"{item['mean']:.2f}",
                    item["min"],
                    item["max"],
                    *[quantiles[key] for key in quantile_keys],
                    *[item[key] for key in threshold_keys],
                ]
            )


def _write_lengths_csv(path: Path, file_to_lengths: dict[str, list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "row_index", "chat_template_tokens"])
        for filename, lengths in file_to_lengths.items():
            for row_index, length in enumerate(lengths):
                writer.writerow([filename, row_index, length])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--files", nargs="*", default=list(DEFAULT_FILES))
    parser.add_argument(
        "--save-lengths",
        action="store_true",
        help="Also write per-row token lengths to token_lengths.csv.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=True,
    )

    summaries: list[dict[str, Any]] = []
    file_to_lengths: dict[str, list[int]] = {}
    for filename in args.files:
        path = args.data_dir / filename
        summary, lengths = analyze_file(
            path,
            tokenizer,
            quantiles=DEFAULT_QUANTILES,
            thresholds=DEFAULT_THRESHOLDS,
        )
        summaries.append(summary)
        file_to_lengths[filename] = lengths

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "tokenizer": str(args.tokenizer),
        "mode": "tokenizer.apply_chat_template(tokenize=True, add_generation_prompt=False)",
        "files": summaries,
    }
    summary_json = args.out_dir / "chat_template_token_length_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_csv = args.out_dir / "chat_template_token_length_summary.csv"
    _write_summary_csv(summary_csv, summaries, DEFAULT_THRESHOLDS)

    if args.save_lengths:
        _write_lengths_csv(args.out_dir / "chat_template_token_lengths.csv", file_to_lengths)

    print(f"summary_json: {summary_json}")
    print(f"summary_csv: {summary_csv}")
    for item in summaries:
        q = item["quantiles"]
        print(
            f"{Path(item['file']).name}: rows={item['rows']} mean={item['mean']:.2f} "
            f"p50={q['p50']} p90={q['p90']} p95={q['p95']} p99={q['p99']} max={item['max']} "
            f">8192={item['over_8192']} >32768={item['over_32768']}"
        )


if __name__ == "__main__":
    main()
