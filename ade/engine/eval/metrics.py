from __future__ import annotations

from collections import defaultdict
import re
from typing import Any


AVG_AT_K_METRIC = "avg@k"
AVG_AT_K_RE = re.compile(r"^avg@\d+$")
PASS_AT_K_RE = re.compile(r"^pass@\d+$")
SUPPORTED_PRIMARY_METRICS = {
    AVG_AT_K_METRIC,
    "avg_at_k_score",
    "accuracy_avg",
    "pass@1",
    "global_average",
    "completion_rate",
}


def summarize_details(
    details: list[dict[str, Any]],
    *,
    primary_metric: str = AVG_AT_K_METRIC,
    configured_k: int | None = None,
) -> dict[str, Any]:
    metric_k = configured_k if configured_k is not None else len(details)
    accuracies = [float(detail.get("accuracy") or 0.0) for detail in details]
    accuracy_avg = sum(accuracies) / len(accuracies) if accuracies else 0.0
    if len(accuracies) > 1:
        std = (sum((acc - accuracy_avg) ** 2 for acc in accuracies) / len(accuracies)) ** 0.5
    else:
        std = 0.0
    metrics = {
        "num_total": max((int(detail.get("total_count") or 0) for detail in details), default=0),
        "solved_avg": (
            sum(float(detail.get("correct_count") or 0) for detail in details) / len(details)
            if details
            else 0.0
        ),
        "accuracy_avg": accuracy_avg,
        AVG_AT_K_METRIC: accuracy_avg,
        "avg_at_k_score": accuracy_avg,
        "accuracy_std": std,
        "accuracy_std_err": std / (len(accuracies) ** 0.5) if accuracies else 0.0,
        "num_repeat": len(details),
        "run_stats": [
            {
                "repetition": idx + 1,
                "num_total": int(detail.get("total_count") or 0),
                "num_solved": float(detail.get("correct_count") or 0),
                "accuracy": float(detail.get("accuracy") or 0.0),
            }
            for idx, detail in enumerate(details)
        ],
    }
    _add_sample_metrics(metrics, details, metric_k)
    _add_detail_level_metrics(metrics, details)
    _set_configured_k_metrics(metrics, metric_k)
    metrics["primary_metric"] = primary_metric
    metrics["score"] = selected_score(metrics, primary_metric)
    return metrics


def standardize_eval_metrics(
    metrics: dict[str, Any] | None,
    details: list[dict[str, Any]],
    *,
    primary_metric: str = AVG_AT_K_METRIC,
    configured_k: int | None = None,
) -> dict[str, Any]:
    metric_k = configured_k if configured_k is not None else len(details)
    result = dict(metrics or {})
    avg, std, stderr = avg_at_k_from_details(details)
    result["accuracy_avg"] = avg
    result[AVG_AT_K_METRIC] = avg
    result["avg_at_k_score"] = avg
    result["accuracy_std"] = std
    result["accuracy_std_err"] = stderr
    result.setdefault("num_repeat", len(details))
    _add_sample_metrics(result, details, metric_k)
    _add_detail_level_metrics(result, details)
    _set_configured_k_metrics(result, metric_k)
    result["primary_metric"] = primary_metric
    result["score"] = selected_score(result, primary_metric)
    return result


def selected_score(metrics: dict[str, Any], primary_metric: str | None = None) -> float:
    metric_name = primary_metric or AVG_AT_K_METRIC
    if metric_name not in metrics and metric_name not in SUPPORTED_PRIMARY_METRICS and PASS_AT_K_RE.match(metric_name) is None:
        raise ValueError(f"unsupported primary_metric: {metric_name}")
    if metric_name not in metrics:
        raise ValueError(f"primary_metric {metric_name!r} is not available in eval metrics")
    try:
        return float(metrics.get(metric_name) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def avg_at_k_from_details(details: list[dict[str, Any]]) -> tuple[float, float, float]:
    accuracies = [float(detail.get("accuracy") or 0.0) for detail in details]
    avg = sum(accuracies) / len(accuracies) if accuracies else 0.0
    std = (sum((acc - avg) ** 2 for acc in accuracies) / len(accuracies)) ** 0.5 if len(accuracies) > 1 else 0.0
    stderr = std / (len(accuracies) ** 0.5) if accuracies else 0.0
    return avg, std, stderr


def resolve_primary_metric(dataset: dict[str, Any] | None, request: dict[str, Any] | None, items: list[dict[str, Any]] | None) -> str:
    dataset = dataset or {}
    request = request or {}
    metric_name = str(request.get("primary_metric") or dataset.get("primary_metric") or AVG_AT_K_METRIC).strip()
    if not metric_name:
        metric_name = AVG_AT_K_METRIC
    if metric_name not in SUPPORTED_PRIMARY_METRICS and PASS_AT_K_RE.match(metric_name) is None:
        raise ValueError(f"unsupported primary_metric: {metric_name}")
    return metric_name


def merge_data_parallel_details(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run: dict[int, dict[str, Any]] = {}
    for detail in details:
        run_index = int(detail.get("index", 0))
        merged = by_run.setdefault(
            run_index,
            {
                "index": run_index,
                "seed": detail.get("seed"),
                "correct_count": 0.0,
                "total_count": 0,
                "errors": [],
                "successes": [],
                "row_indices": [],
            },
        )
        merged["correct_count"] += float(detail.get("correct_count") or len(detail.get("successes") or []))
        merged["total_count"] += int(
            detail.get("total_count")
            or (len(detail.get("successes") or []) + len(detail.get("errors") or []))
        )
        merged["errors"].extend(detail.get("errors") or [])
        merged["successes"].extend(detail.get("successes") or [])
        merged["row_indices"].extend(detail.get("row_indices") or [])
    result = []
    for run_index in sorted(by_run):
        detail = by_run[run_index]
        total = int(detail["total_count"])
        detail["accuracy"] = (float(detail["correct_count"]) / total) if total else 0.0
        result.append(detail)
    return result


def _all_samples(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for detail in details:
        for sample in [*(detail.get("successes") or []), *(detail.get("errors") or [])]:
            samples.append(_sample_metric_view(sample))
    return samples


def _sample_metric_view(sample: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(sample, dict):
        return sample
    view = {
        "case_id": sample.get("case_id"),
        "bucket": sample.get("bucket"),
        "index": sample.get("index"),
        "example_id": sample.get("example_id"),
    }
    for key in ("passed", "correctness", "equivalent", "score", "mode", "source_id", "difficulty"):
        if key in sample:
            view[key] = sample[key]
    for section in ("prediction", "diagnostics", "metadata"):
        value = sample.get(section)
        if isinstance(value, dict):
            view.update(value)
    return view


def _add_sample_metrics(metrics: dict[str, Any], details: list[dict[str, Any]], configured_k: int) -> None:
    samples = _all_samples(details)
    if not samples:
        return
    completed = sum(1 for sample in samples if str(sample.get("generation") or sample.get("answer_content") or "").strip())
    metrics["completion_rate"] = completed / len(samples)

    if all("example_id" in sample and _sample_has_correctness(sample) for sample in samples):
        by_example: dict[str, list[bool]] = defaultdict(list)
        by_mode: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
        for sample in samples:
            passed = _sample_passed(sample)
            example_id = str(sample["example_id"])
            by_example[example_id].append(passed)
            if sample.get("mode"):
                by_mode[str(sample["mode"])][str(sample.get("source_id") or example_id)].append(passed)
        totals = [len(values) for values in by_example.values()]
        corrects = [sum(1 for value in values if value) for values in by_example.values()]
        if totals and min(totals) >= configured_k:
            metrics[f"pass@{configured_k}"] = _estimate_pass_at_k(totals, corrects, configured_k)
        for mode, completions in by_mode.items():
            mode_totals = [len(values) for values in completions.values()]
            mode_corrects = [sum(1 for value in values if value) for values in completions.values()]
            if mode_totals:
                metrics[f"{mode}_accuracy_avg"] = sum(mode_corrects) / sum(mode_totals)
                if min(mode_totals) >= configured_k:
                    metrics[f"{mode}_pass@{configured_k}"] = _estimate_pass_at_k(
                        mode_totals,
                        mode_corrects,
                        configured_k,
                    )

    difficulty_totals: dict[str, int] = defaultdict(int)
    difficulty_correct: dict[str, int] = defaultdict(int)
    for sample in samples:
        if sample.get("difficulty") is None:
            continue
        difficulty = str(sample["difficulty"])
        difficulty_totals[difficulty] += 1
        difficulty_correct[difficulty] += int(_sample_passed(sample))
    for difficulty, total in difficulty_totals.items():
        if total:
            metrics[f"accuracy_{difficulty}_avg"] = difficulty_correct[difficulty] / total
            metrics[f"accuracy_{difficulty}_std_err"] = 0.0

    score_samples = [float(sample["score"]) for sample in samples if sample.get("score") is not None]
    if score_samples:
        metrics["global_average"] = _mean(score_samples) * 100
        metrics["accuracy_avg"] = _mean(score_samples)


def _sample_has_correctness(sample: dict[str, Any]) -> bool:
    return any(key in sample for key in ("passed", "correctness", "equivalent", "score"))


def _set_configured_k_metrics(metrics: dict[str, Any], configured_k: int) -> None:
    if isinstance(configured_k, bool) or int(configured_k) < 1:
        raise ValueError("configured_k must be a positive integer")
    metric_k = int(configured_k)
    avg_key = f"avg@{metric_k}"
    pass_key = f"pass@{metric_k}"
    for key in list(metrics):
        if (AVG_AT_K_RE.fullmatch(key) and key != avg_key) or (
            PASS_AT_K_RE.fullmatch(key) and key != pass_key
        ):
            metrics.pop(key)
    metrics["configured_k"] = metric_k
    metrics[avg_key] = metrics[AVG_AT_K_METRIC]
    if pass_key in metrics:
        metrics["pass@k"] = metrics[pass_key]
    else:
        metrics.pop("pass@k", None)


def _sample_passed(sample: dict[str, Any]) -> bool:
    for key in ("passed", "correctness", "equivalent"):
        if key in sample:
            return bool(sample.get(key))
    try:
        return float(sample.get("score") or 0.0) >= 1.0
    except (TypeError, ValueError):
        return False


def _add_detail_level_metrics(metrics: dict[str, Any], details: list[dict[str, Any]]) -> None:
    keys = (
        "strict_prompt_level",
        "strict_instruction_level",
        "loose_prompt_level",
        "loose_instruction_level",
        "prompt-level",
        "instruction-level",
    )
    for key in keys:
        values = [float(detail[key]) for detail in details if detail.get(key) is not None]
        if values:
            metrics[key] = _mean(values)


def _estimate_pass_at_k(num_samples: list[int], num_correct: list[int], k: int) -> float:
    values = []
    for n, c in zip(num_samples, num_correct):
        n = int(n)
        c = int(c)
        if n < k:
            continue
        if n - c < k:
            values.append(1.0)
            continue
        value = 1.0
        for item in range(n - c + 1, n + 1):
            value *= 1.0 - k / item
        values.append(1.0 - value)
    return _mean(values)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
