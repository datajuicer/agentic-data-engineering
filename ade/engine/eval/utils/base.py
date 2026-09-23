"""Shared dataset I/O and evaluation task mechanics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..contracts import EvalExample, EvalScore, RolloutRecord


def read_json_or_jsonl(path: str | Path, *, split: str | None = None) -> list[dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        return read_hf_disk_dataset(path, split=split)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    data = json.loads(text) if text.startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(data, list):
        raise ValueError(f"evaluation dataset must be a JSON array or JSONL rows: {path}")
    return [row for row in data if isinstance(row, dict)]


def read_hf_disk_dataset(path: str | Path, *, split: str | None = None) -> list[dict[str, Any]]:
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise ImportError(
            "reading Hugging Face save_to_disk datasets requires the optional 'datasets' package; "
            "install datasets or export the eval data as JSON/JSONL"
        ) from exc

    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict) or (isinstance(dataset, Mapping) and not hasattr(dataset, "features")):
        split_name = split or _default_hf_split(dataset)
        if split_name not in dataset:
            available = ", ".join(str(key) for key in dataset.keys())
            raise ValueError(f"HF disk dataset {path} does not contain split {split_name!r}; available splits: {available}")
        dataset = dataset[split_name]
    elif split:
        raise ValueError(f"HF disk dataset {path} is a single split; remove split={split!r} from the eval config")

    rows: list[dict[str, Any]] = []
    for row in dataset:
        if isinstance(row, dict):
            rows.append(row)
        elif isinstance(row, Mapping):
            rows.append(dict(row))
    return rows


def _default_hf_split(dataset: Mapping[str, Any]) -> str:
    for candidate in ("test", "validation", "val", "dev", "train"):
        if candidate in dataset:
            return candidate
    try:
        return next(iter(dataset.keys()))
    except StopIteration as exc:
        raise ValueError("HF disk DatasetDict has no splits") from exc


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
        if key in meta and meta[key] is not None:
            return meta[key]
    return None


def require_first_present(row: dict[str, Any], keys: tuple[str, ...], *, task_type: str, row_index: int, field_name: str) -> Any:
    value = first_present(row, keys)
    if value is None or value == "":
        raise ValueError(f"{task_type} dataset row {row_index} is missing required {field_name}; expected one of {', '.join(keys)}")
    return value


class EvalTask:
    task_type = "base"
    primary_metric = "accuracy_avg"

    def load_examples(
        self,
        path: str | Path,
        *,
        dataset_name: str | None = None,
        dataset_config: dict[str, Any] | None = None,
    ) -> list[EvalExample]:
        split = (dataset_config or {}).get("split") or (dataset_config or {}).get("hf_split")
        examples = []
        for idx, row in enumerate(read_json_or_jsonl(path, split=split)):
            examples.append(self.row_to_example(row, idx, dataset_name=dataset_name))
        return examples

    def row_to_example(self, row: dict[str, Any], idx: int, *, dataset_name: str | None = None) -> EvalExample:
        raise NotImplementedError

    def score_rollouts(
        self,
        examples: list[EvalExample],
        rollouts_by_repeat: list[list[RolloutRecord]],
        *,
        primary_metric: str | None = None,
    ) -> EvalScore:
        raise NotImplementedError

    @staticmethod
    def std(values: list[float]) -> float:
        if len(values) <= 1:
            return 0.0
        avg = sum(values) / len(values)
        return (sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5

    @classmethod
    def average_score(cls, run_stats: list[dict[str, Any]]) -> tuple[float, float, float]:
        accuracies = [float(item.get("accuracy") or 0.0) for item in run_stats]
        avg = sum(accuracies) / len(accuracies) if accuracies else 0.0
        std = cls.std(accuracies)
        stderr = std / (len(accuracies) ** 0.5) if accuracies else 0.0
        return avg, std, stderr
