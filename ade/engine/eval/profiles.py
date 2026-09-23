from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


_AVG_RE = re.compile(r"^avg@(\d+)$")
_PASS_RE = re.compile(r"^pass@(\d+)$")


@dataclass(frozen=True, order=True)
class DatasetEvalProfile:
    num_samples: int
    data_parallel_shards: int


@dataclass(frozen=True)
class DatasetEvalGroup:
    profile: DatasetEvalProfile
    datasets: tuple[dict[str, Any], ...]


def dataset_eval_profile(
    eval_config: dict[str, Any],
    dataset_name: str,
    purpose: str,
) -> DatasetEvalProfile:
    profiles = eval_config.get("dataset_profiles")
    if not isinstance(profiles, dict):
        raise ValueError("eval.dataset_profiles is required")
    dataset_profiles = profiles.get(dataset_name)
    if not isinstance(dataset_profiles, dict):
        raise ValueError(f"eval.dataset_profiles.{dataset_name} is required")
    raw = dataset_profiles.get(purpose)
    if not isinstance(raw, dict):
        raise ValueError(
            f"eval.dataset_profiles.{dataset_name}.{purpose} is required"
        )
    num_samples = _positive_int(
        raw.get("num_samples"), f"{dataset_name}.{purpose}.num_samples"
    )
    default_shards = eval_config.get(f"{purpose}_data_parallel_shards")
    if purpose == "online_eval":
        default_shards = eval_config.get(
            "online_validation_data_parallel_shards", default_shards
        )
    shards = _positive_int(
        raw.get(
            "data_parallel_shards",
            default_shards or eval_config.get("data_parallel_shards", 1),
        ),
        f"{dataset_name}.{purpose}.data_parallel_shards",
    )
    return DatasetEvalProfile(num_samples=num_samples, data_parallel_shards=shards)


def group_datasets_by_profile(
    datasets: list[dict[str, Any]],
    eval_config: dict[str, Any],
    *,
    purpose: str,
) -> list[DatasetEvalGroup]:
    grouped: dict[DatasetEvalProfile, list[dict[str, Any]]] = {}
    for dataset in datasets:
        name = str(dataset.get("name") or "")
        if not name:
            raise ValueError("evaluation dataset name is required")
        profile = dataset_eval_profile(eval_config, name, purpose)
        grouped.setdefault(profile, []).append(dataset)
    return [
        DatasetEvalGroup(profile=profile, datasets=tuple(grouped[profile]))
        for profile in sorted(grouped)
    ]


def attach_dataset_profiles(
    datasets: list[dict[str, Any]],
    eval_config: dict[str, Any],
    *,
    purpose: str,
) -> list[dict[str, Any]]:
    attached: list[dict[str, Any]] = []
    for dataset in datasets:
        name = str(dataset.get("name") or "")
        if not name:
            raise ValueError("evaluation dataset name is required")
        profile = dataset_eval_profile(eval_config, name, purpose)
        attached.append(
            {
                **dataset,
                "eval_profile": {
                    "num_samples": profile.num_samples,
                    "data_parallel_shards": profile.data_parallel_shards,
                },
            }
        )
    return attached


def aggregate_validation_metrics(
    dataset_metrics: dict[str, dict[str, Any]],
    ranking_config: dict[str, Any],
) -> dict[str, Any]:
    primary = str(ranking_config.get("primary") or "")
    secondary = str(ranking_config.get("secondary") or "")
    supported = {
        "macro_avg_at_configured_k",
        "macro_pass_at_configured_k",
    }
    if {primary, secondary} != supported:
        raise ValueError(
            "validation ranking primary and secondary must configure macro pass@K and avg@K exactly once"
        )
    raw_weights = ranking_config.get("weights")
    if not isinstance(raw_weights, dict) or set(raw_weights) != set(dataset_metrics):
        raise ValueError("validation ranking weights must name every dataset exactly once")
    weights = {name: float(raw_weights[name]) for name in dataset_metrics}
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights.values()):
        raise ValueError("validation ranking weights must be positive finite numbers")
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("validation ranking weights must sum to 1")

    avg_score = 0.0
    pass_score = 0.0
    variance = 0.0
    for name, metrics in dataset_metrics.items():
        weight = weights[name]
        avg_score += weight * _concrete_metric(metrics, _AVG_RE, name, "avg")
        pass_score += weight * _concrete_metric(metrics, _PASS_RE, name, "pass")
        std = _finite_number(metrics.get("accuracy_std"), f"{name}.accuracy_std")
        variance += (weight * std) ** 2
    return {
        "ranking_score": (
            pass_score if primary == "macro_pass_at_configured_k" else avg_score
        ),
        "secondary_score": (
            pass_score if secondary == "macro_pass_at_configured_k" else avg_score
        ),
        "avg@k": avg_score,
        "pass@k": pass_score,
        "accuracy_std": math.sqrt(variance),
        "dataset_weights": weights,
        "dataset_metrics": {name: dict(value) for name, value in dataset_metrics.items()},
        "ranking_policy": {
            "primary": primary,
            "secondary": secondary,
        },
    }


def _concrete_metric(
    metrics: dict[str, Any],
    pattern: re.Pattern[str],
    dataset_name: str,
    label: str,
) -> float:
    matches = [
        (key, value)
        for key, value in metrics.items()
        if pattern.match(str(key)) is not None
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{dataset_name} must contain exactly one concrete {label}@K metric"
        )
    key, value = matches[0]
    return _finite_number(value, f"{dataset_name}.{key}")


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be a finite number")
    return parsed
