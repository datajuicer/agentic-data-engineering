"""Deterministic compaction of real training reward-rollout dumps."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Iterator

from ade.engine.storage.atomic import write_json_atomic


REWARD_TRACE_SCHEMA_VERSION = "ade.reward_rollout_trace.v5"
GROUP_CREDIT_TRACE_SCHEMA_VERSION = REWARD_TRACE_SCHEMA_VERSION
_REQUIRED = {
    "segment_id",
    "training_step",
    "source_row_index",
    "sample_uid",
    "data_source",
    "prompt",
    "response",
    "custom_reward_score",
    "effective_training_reward",
    "extracted_answer",
    "response_length_tokens",
    "max_response_length_tokens",
    "group_credit_enabled",
    "group_uid",
    "group_type",
    "outcome_score",
    "process_evidence",
    "artifact_projection",
    "rule_evidence",
    "pre_group_reward",
    "final_training_reward",
    "realized_grpo_advantage",
    "counterfactual_identity_grpo_advantage",
}
_GROUP_CREDIT_REQUIRED = {
    "assigned_training_reward",
    "group_credit_mode",
    "group_credit_source",
    "group_credit_reason",
    "group_credit_evidence_sources",
    "group_credit_process_dimensions",
}


@dataclass(frozen=True)
class RewardTraceSpec:
    trial_uid: str
    reward_function_sha256: str
    reward_entrypoint: str
    raw_paths: tuple[Path, ...]
    output_path: Path
    manifest_path: Path
    reward_transformations_active: bool = False
    snapshot_steps: tuple[int, ...] = ()
    prompt_groups_per_step: int = 256
    responses_per_group: int = 8
    group_credit_enabled: bool = False


def finalize_reward_trace(
    *,
    run_dir: Path,
    trial_uid: str,
    reward_function_sha256: str,
    reward_entrypoint: str = "compute_score",
    reward_transformations_active: bool = False,
    snapshot_steps: tuple[int, ...] = (),
    prompt_groups_per_step: int = 256,
    responses_per_group: int = 8,
    group_credit_enabled: bool = False,
) -> dict[str, Any]:
    raw_root = run_dir / "engine_audit" / "reward_rollouts_raw"
    analysis_root = run_dir / "analysis_sources"
    manifest_path = analysis_root / "reward_trace_manifest.json"
    raw_paths = _compress_raw_segments(raw_root)
    if not raw_paths:
        manifest = {
            "schema_version": _trace_schema(group_credit_enabled),
            "status": "incomplete",
            "reason": "reward_rollout_raw_files_unavailable",
            "trial_uid": trial_uid,
            "reward_function_sha256": reward_function_sha256,
            "ranking_impact": "none",
        }
        write_json_atomic(manifest_path, manifest)
        return manifest
    spec = RewardTraceSpec(
        trial_uid=trial_uid,
        reward_function_sha256=reward_function_sha256,
        reward_entrypoint=reward_entrypoint,
        raw_paths=tuple(raw_paths),
        output_path=analysis_root / "reward_rollout_trace.jsonl",
        manifest_path=manifest_path,
        reward_transformations_active=reward_transformations_active,
        snapshot_steps=snapshot_steps,
        prompt_groups_per_step=prompt_groups_per_step,
        responses_per_group=responses_per_group,
        group_credit_enabled=group_credit_enabled,
    )
    try:
        return compact_reward_trace(spec)
    except Exception as exc:
        manifest = {
            "schema_version": _trace_schema(group_credit_enabled),
            "status": "incomplete",
            "reason": f"{type(exc).__name__}: {exc}",
            "trial_uid": trial_uid,
            "reward_function_sha256": reward_function_sha256,
            "raw_paths": [str(path) for path in raw_paths],
            "compressed_byte_count": sum(path.stat().st_size for path in raw_paths),
            "ranking_impact": "none",
        }
        write_json_atomic(manifest_path, manifest)
        return manifest


def _compress_raw_segments(raw_root: Path) -> list[Path]:
    if not raw_root.exists():
        return []
    for source in sorted(raw_root.rglob("*.jsonl"), key=str):
        target = source.with_suffix(source.suffix + ".gz")
        if target.exists():
            raise ValueError(f"reward trace compressed segment already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with source.open("rb") as input_handle, os.fdopen(fd, "wb") as raw_output:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as output:
                    for chunk in iter(lambda: input_handle.read(1024 * 1024), b""):
                        output.write(chunk)
                raw_output.flush()
                os.fsync(raw_output.fileno())
            os.replace(temporary, target)
            source.unlink()
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
    return sorted(raw_root.rglob("*.jsonl.gz"), key=str)


def compact_reward_trace(spec: RewardTraceSpec) -> dict[str, Any]:
    raw_bytes = sum(path.stat().st_size for path in spec.raw_paths)
    max_step = max(
        (int(row["training_step"]) for row in _iter_normalized_rows(spec)),
        default=0,
    )
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_step_set = set(spec.snapshot_steps)
    snapshot_rows: list[dict[str, Any]] = []
    summary = _new_population_accumulator()
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{spec.output_path.name}.",
        suffix=".tmp",
        dir=spec.output_path.parent,
    )
    row_count = 0
    accepted_segment_ids: set[str] = set()
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            for raw_row in _iter_normalized_rows(spec):
                row = _prepare_row(raw_row, spec=spec)
                row["training_phase"] = _phase(
                    int(row["training_step"]), max_step=max_step
                )
                output.write(
                    json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                row_count += 1
                accepted_segment_ids.add(str(row["segment_id"]))
                _accumulate_population(summary, row)
                if int(row["training_step"]) in snapshot_step_set:
                    snapshot_rows.append(row)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, spec.output_path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    population_summary_path = spec.output_path.with_name("reward_population_summary.json")
    write_json_atomic(population_summary_path, _population_summary_from_accumulator(summary))
    population_summary_digest = _sha256(population_summary_path)
    step_snapshots, snapshot_sources = _write_step_snapshots(snapshot_rows, spec=spec)
    source_files = [
        _source_file(
            "reward-population-summary",
            "reward_population_summary",
            population_summary_path,
            "application/json",
        ),
        *snapshot_sources,
    ]
    manifest = {
        "schema_version": _trace_schema(spec.group_credit_enabled),
        "status": "complete",
        "trial_uid": spec.trial_uid,
        "reward_function_sha256": spec.reward_function_sha256,
        "reward_entrypoint": spec.reward_entrypoint,
        "raw_paths": [str(path) for path in spec.raw_paths],
        "raw_bytes": raw_bytes,
        "compressed_byte_count": raw_bytes,
        "prepared_trace_path": str(spec.output_path),
        "prepared_trace_sha256": _sha256(spec.output_path),
        "population_summary_path": str(population_summary_path),
        "population_summary_sha256": population_summary_digest,
        "raw_row_count": row_count,
        "snapshot_policy": "all_complete_groups_at_artifact_positions.v1",
        "accepted_segment_ids": sorted(accepted_segment_ids),
        "raw_source_file_digests": {str(path): _sha256(path) for path in spec.raw_paths},
        "reward_transformations_active": spec.reward_transformations_active,
        "no_extra_rollout_launched": True,
        "step_snapshots": step_snapshots,
        "source_files": source_files,
    }
    write_json_atomic(spec.manifest_path, manifest)
    return manifest


def _write_step_snapshots(
    rows: list[dict[str, Any]],
    *,
    spec: RewardTraceSpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not spec.snapshot_steps:
        return [], []
    if spec.prompt_groups_per_step < 1 or spec.responses_per_group < 1:
        raise ValueError("reward rollout snapshot dimensions must be positive")
    observed_steps = sorted({int(row["training_step"]) for row in rows})
    snapshots: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for snapshot_step in sorted(set(spec.snapshot_steps)):
        rollout_step = (
            observed_steps[0]
            if snapshot_step == 0 and observed_steps
            else int(snapshot_step)
        )
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            if int(row["training_step"]) == rollout_step:
                grouped.setdefault(str(row["prompt_group_id"]), []).append(row)
        complete_groups = [
            (group_id, sorted(group_rows, key=lambda row: str(row["trace_id"])))
            for group_id, group_rows in grouped.items()
            if len(group_rows) == spec.responses_per_group
        ]
        complete_groups.sort(key=lambda item: item[0])
        chosen_groups = complete_groups
        selected: list[dict[str, Any]] = []
        for group_id, group_rows in chosen_groups:
            for response_index, row in enumerate(
                group_rows
            ):
                item = dict(row)
                item.update(
                    {
                        "snapshot_step": int(snapshot_step),
                        "rollout_training_step": rollout_step,
                        "policy_step_at_generation": max(rollout_step - 1, 0),
                        "response_index": response_index,
                        "prompt_group_id": group_id,
                        "sample_role": "grpo_step_snapshot",
                        "selected_by": "artifact_position_full_population",
                    }
                )
                selected.append(item)
        step_name = f"step-{int(snapshot_step):03d}"
        step_root = spec.output_path.parent / "steps" / step_name
        rollout_path = step_root / "grpo-rollouts.jsonl.gz"
        statistics_path = step_root / "rollout-statistics.json"
        _write_gzip_jsonl_atomic(rollout_path, selected)
        statistics = _snapshot_statistics(
            selected,
            snapshot_step=int(snapshot_step),
            rollout_step=rollout_step,
            expected_group_count=spec.prompt_groups_per_step,
            responses_per_group=spec.responses_per_group,
        )
        write_json_atomic(statistics_path, statistics)
        rollout_source_id = f"grpo-rollouts-{step_name}"
        statistics_source_id = f"rollout-statistics-{step_name}"
        snapshots.append(
            {
                "step": int(snapshot_step),
                "rollout_training_step": rollout_step,
                "policy_step_at_generation": max(rollout_step - 1, 0),
                "status": (
                    "complete"
                    if len(chosen_groups) == spec.prompt_groups_per_step
                    and len(chosen_groups) == len(grouped)
                    else "incomplete"
                ),
                "prompt_group_count": len(chosen_groups),
                "responses_per_group": spec.responses_per_group,
                "response_count": len(selected),
                "rollout_source_id": rollout_source_id,
                "statistics_source_id": statistics_source_id,
                "rollout_path": str(rollout_path),
                "rollout_sha256": _sha256(rollout_path),
                "statistics_path": str(statistics_path),
                "statistics_sha256": _sha256(statistics_path),
            }
        )
        sources.extend(
            (
                _source_file(
                    rollout_source_id,
                    "grpo_rollouts",
                    rollout_path,
                    "application/gzip",
                    step=int(snapshot_step),
                    prompt_group_count=len(chosen_groups),
                    response_count=len(selected),
                ),
                _source_file(
                    statistics_source_id,
                    "rollout_statistics",
                    statistics_path,
                    "application/json",
                    step=int(snapshot_step),
                ),
            )
        )
    return snapshots, sources


def _snapshot_statistics(
    rows: list[dict[str, Any]],
    *,
    snapshot_step: int,
    rollout_step: int,
    expected_group_count: int,
    responses_per_group: int,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["prompt_group_id"]), []).append(row)
    zero_variance = sum(
        len(
            {
                float(row.get("effective_training_reward") or 0.0)
                for row in group_rows
            }
        )
        == 1
        for group_rows in grouped.values()
    )
    incorrect_but_rewarded = sum(
        not _rollout_correct(row)
        and float(row.get("effective_training_reward") or 0.0) > 0.0
        for row in rows
    )
    correct_but_unrewarded = sum(
        _rollout_correct(row)
        and float(row.get("effective_training_reward") or 0.0) == 0.0
        for row in rows
    )
    return {
        "schema_version": _trace_schema(
            any("realized_grpo_advantage" in row for row in rows)
        ),
        "summary_role": "complete_grpo_step_snapshot",
        "snapshot_step": snapshot_step,
        "rollout_training_step": rollout_step,
        "policy_step_at_generation": max(rollout_step - 1, 0),
        "expected_prompt_group_count": expected_group_count,
        "prompt_group_count": len(grouped),
        "responses_per_group": responses_per_group,
        "response_count": len(rows),
        "custom_reward_score": _numeric_summary(
            row.get("custom_reward_score") for row in rows
        ),
        "effective_training_reward": _numeric_summary(
            row.get("effective_training_reward") for row in rows
        ),
        "response_length_tokens": _numeric_summary(
            row.get("response_length_tokens") for row in rows
        ),
        "zero_group_variance": _count_rate(zero_variance, len(grouped)),
        "empty_extraction": _count_rate(
            sum(not str(row.get("extracted_answer") or "").strip() for row in rows),
            len(rows),
        ),
        "length_cutoff": _count_rate(
            sum(_is_length_cutoff(row) for row in rows),
            len(rows),
        ),
        "reward_timeout_or_error": _count_rate(
            sum(bool(row.get("reward_timeout") or row.get("reward_error")) for row in rows),
            len(rows),
        ),
        "incorrect_but_rewarded": _count_rate(incorrect_but_rewarded, len(rows)),
        "correct_but_unrewarded": _count_rate(correct_but_unrewarded, len(rows)),
    }


def _rollout_correct(row: dict[str, Any]) -> bool:
    ground_truth = str(row.get("ground_truth") or "").strip()
    extracted = str(row.get("extracted_answer") or "").strip()
    return bool(ground_truth) and extracted == ground_truth


def _source_file(
    source_id: str,
    kind: str,
    path: Path,
    media_type: str,
    *,
    step: int | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    result = {
        "source_id": source_id,
        "kind": kind,
        "path": str(path),
        "filename": path.name,
        "media_type": media_type,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "visibility": "agent",
    }
    if step is not None:
        result["step"] = step
    result.update(metadata)
    return result


def _write_gzip_jsonl_atomic(
    path: Path, rows: Iterable[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as raw_output:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_output, mtime=0
            ) as output:
                for row in rows:
                    output.write(
                        (
                            json.dumps(
                                row,
                                sort_keys=True,
                                separators=(",", ":"),
                                ensure_ascii=True,
                            )
                            + "\n"
                        ).encode("utf-8")
                    )
            raw_output.flush()
            os.fsync(raw_output.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _new_population_accumulator() -> dict[str, Any]:
    def population_slice() -> dict[str, Any]:
        return {
            "row_count": 0,
            "effective_training_reward": [],
            "response_length_tokens": [],
            "empty_extraction": 0,
            "length_cutoff": 0,
            "reward_timeout_or_error": 0,
        }

    return {
        **population_slice(),
        "custom_reward_score": [],
        "pre_group_reward": [],
        "assigned_training_reward": [],
        "realized_grpo_advantage": [],
        "counterfactual_identity_grpo_advantage": [],
        "prompt_group_ids": set(),
        "group_rows": {},
        "joint_behavior_counts": {},
        "by_phase": {
            phase: population_slice() for phase in ("early", "middle", "late")
        },
    }


def _accumulate_population(accumulator: dict[str, Any], row: dict[str, Any]) -> None:
    empty = not str(row.get("extracted_answer") or "").strip()
    cutoff = _is_length_cutoff(row)
    failed = bool(row.get("reward_timeout") or row.get("reward_error"))
    zero_reward = float(row.get("effective_training_reward") or 0.0) == 0.0

    def add(target: dict[str, Any]) -> None:
        target["row_count"] += 1
        for field in ("effective_training_reward", "response_length_tokens"):
            value = row.get(field)
            if isinstance(value, (int, float)):
                target[field].append(float(value))
        target["empty_extraction"] += int(empty)
        target["length_cutoff"] += int(cutoff)
        target["reward_timeout_or_error"] += int(failed)

    add(accumulator)
    custom_reward = row.get("custom_reward_score")
    if isinstance(custom_reward, (int, float)):
        accumulator["custom_reward_score"].append(float(custom_reward))
    for field in (
        "pre_group_reward",
        "assigned_training_reward",
        "realized_grpo_advantage",
        "counterfactual_identity_grpo_advantage",
    ):
        value = row.get(field)
        if isinstance(value, (int, float)):
            accumulator[field].append(float(value))
    accumulator["prompt_group_ids"].add(str(row["prompt_group_id"]))
    accumulator["group_rows"].setdefault(str(row["prompt_group_id"]), []).append(row)
    joint_key = (empty, cutoff, zero_reward)
    joint = accumulator["joint_behavior_counts"]
    joint[joint_key] = joint.get(joint_key, 0) + 1
    add(accumulator["by_phase"][row["training_phase"]])


def _population_summary_from_accumulator(
    accumulator: dict[str, Any],
) -> dict[str, Any]:
    def population_slice(value: dict[str, Any]) -> dict[str, Any]:
        count = int(value["row_count"])
        return {
            "row_count": count,
            "effective_training_reward": _numeric_summary(
                value["effective_training_reward"]
            ),
            "response_length_tokens": _numeric_summary(
                value["response_length_tokens"]
            ),
            "empty_extraction": _count_rate(value["empty_extraction"], count),
            "length_cutoff": _count_rate(value["length_cutoff"], count),
            "reward_timeout_or_error": _count_rate(
                value["reward_timeout_or_error"], count
            ),
        }

    count = int(accumulator["row_count"])
    by_phase = {
        phase: population_slice(accumulator["by_phase"][phase])
        for phase in ("early", "middle", "late")
    }
    group_rows = accumulator["group_rows"]
    group_credit_summary = _group_credit_population_summary(group_rows)
    return {
        "schema_version": _trace_schema(bool(accumulator["realized_grpo_advantage"])),
        "population_role": "complete_raw_rollout_population",
        "row_count": count,
        "unique_prompt_group_count": len(accumulator["prompt_group_ids"]),
        "phase_counts": {
            phase: summary["row_count"] for phase, summary in by_phase.items()
        },
        "custom_reward_score": _numeric_summary(
            accumulator["custom_reward_score"]
        ),
        "pre_group_reward": _numeric_summary(accumulator["pre_group_reward"]),
        "assigned_training_reward": _numeric_summary(
            accumulator["assigned_training_reward"]
        ),
        "realized_grpo_advantage": _numeric_summary(
            accumulator["realized_grpo_advantage"]
        ),
        "counterfactual_identity_grpo_advantage": _numeric_summary(
            accumulator["counterfactual_identity_grpo_advantage"]
        ),
        **group_credit_summary,
        **{
            key: value
            for key, value in population_slice(accumulator).items()
            if key != "row_count"
        },
        "by_phase": by_phase,
        "joint_behavior_counts": [
            {
                "empty_extraction": key[0],
                "length_cutoff": key[1],
                "zero_effective_reward": key[2],
                "count": value,
            }
            for key, value in sorted(
                accumulator["joint_behavior_counts"].items()
            )
        ],
    }


def _group_credit_population_summary(
    groups: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    enabled_groups = [
        rows for rows in groups.values() if rows and "assigned_training_reward" in rows[0]
    ]
    if not enabled_groups:
        return {}
    type_counts = {name: 0 for name in ("all_wrong", "mixed", "all_correct")}
    outcome_flat = pre_flat = assigned_flat = sign_flips = newly_nonzero = 0
    wrong_positive = fallback_groups = 0
    for rows in enabled_groups:
        group_type = str(rows[0].get("group_type"))
        if group_type in type_counts:
            type_counts[group_type] += 1
        outcome_flat += len({float(row["outcome_score"]) for row in rows}) == 1
        pre_flat += len({float(row["pre_group_reward"]) for row in rows}) == 1
        assigned_flat += len({float(row["assigned_training_reward"]) for row in rows}) == 1
        fallback_groups += rows[0].get("group_credit_source") != "artifact"
        for row in rows:
            actual = float(row["realized_grpo_advantage"])
            counterfactual = float(row["counterfactual_identity_grpo_advantage"])
            sign_flips += (actual > 0) != (counterfactual > 0) or (actual < 0) != (
                counterfactual < 0
            )
            newly_nonzero += actual != 0.0 and counterfactual == 0.0
            wrong_positive += float(row["outcome_score"]) != 1.0 and actual > 0.0
    total = len(enabled_groups)
    row_total = sum(len(rows) for rows in enabled_groups)
    return {
        "group_credit": {
            "group_count": total,
            "group_type_counts": type_counts,
            "outcome_flat_groups": _count_rate(outcome_flat, total),
            "pre_group_flat_groups": _count_rate(pre_flat, total),
            "assigned_reward_flat_groups": _count_rate(assigned_flat, total),
            "fallback_groups": _count_rate(fallback_groups, total),
            "advantage_sign_flips": _count_rate(sign_flips, row_total),
            "newly_nonzero_advantages": _count_rate(newly_nonzero, row_total),
            "verifier_wrong_positive_advantage": _count_rate(
                wrong_positive, row_total
            ),
        }
    }


def _population_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_phase = {
        phase_name: _population_slice(
            [row for row in rows if row["training_phase"] == phase_name]
        )
        for phase_name in ("early", "middle", "late")
    }
    return {
        "schema_version": REWARD_TRACE_SCHEMA_VERSION,
        "population_role": "complete_raw_rollout_population",
        "row_count": len(rows),
        "unique_prompt_group_count": len({str(row["prompt_group_id"]) for row in rows}),
        "phase_counts": {phase_name: summary["row_count"] for phase_name, summary in by_phase.items()},
        "custom_reward_score": _numeric_summary(
            row.get("custom_reward_score") for row in rows
        ),
        "effective_training_reward": _numeric_summary(
            row.get("effective_training_reward") for row in rows
        ),
        "response_length_tokens": _numeric_summary(
            row.get("response_length_tokens") for row in rows
        ),
        "empty_extraction": _count_rate(
            sum(not str(row.get("extracted_answer") or "").strip() for row in rows),
            len(rows),
        ),
        "length_cutoff": _count_rate(
            sum(_is_length_cutoff(row) for row in rows),
            len(rows),
        ),
        "reward_timeout_or_error": _count_rate(
            sum(bool(row.get("reward_timeout") or row.get("reward_error")) for row in rows),
            len(rows),
        ),
        "by_phase": by_phase,
        "joint_behavior_counts": _joint_behavior_counts(rows),
    }


def _population_slice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "effective_training_reward": _numeric_summary(
            row.get("effective_training_reward") for row in rows
        ),
        "response_length_tokens": _numeric_summary(
            row.get("response_length_tokens") for row in rows
        ),
        "empty_extraction": _count_rate(
            sum(not str(row.get("extracted_answer") or "").strip() for row in rows),
            len(rows),
        ),
        "length_cutoff": _count_rate(sum(_is_length_cutoff(row) for row in rows), len(rows)),
        "reward_timeout_or_error": _count_rate(
            sum(bool(row.get("reward_timeout") or row.get("reward_error")) for row in rows),
            len(rows),
        ),
    }


def _numeric_summary(values: Iterable[Any]) -> dict[str, Any]:
    numbers = sorted(float(value) for value in values if isinstance(value, (int, float)))
    if not numbers:
        return {
            "count": 0,
            "zero_count": 0,
            "zero_rate": None,
            "one_count": 0,
            "one_rate": None,
            "other_count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "quantiles": {},
        }
    count = len(numbers)
    zero_count = sum(value == 0.0 for value in numbers)
    one_count = sum(value == 1.0 for value in numbers)
    return {
        "count": count,
        "zero_count": zero_count,
        "zero_rate": zero_count / count,
        "one_count": one_count,
        "one_rate": one_count / count,
        "other_count": count - zero_count - one_count,
        "min": numbers[0],
        "max": numbers[-1],
        "mean": sum(numbers) / count,
        "quantiles": {
            name: _quantile(numbers, fraction)
            for name, fraction in (
                ("p25", 0.25),
                ("p50", 0.50),
                ("p75", 0.75),
                ("p90", 0.90),
                ("p95", 0.95),
                ("p99", 0.99),
            )
        },
    }


def _quantile(numbers: list[float], fraction: float) -> float:
    position = (len(numbers) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(numbers) - 1)
    weight = position - lower
    return numbers[lower] * (1.0 - weight) + numbers[upper] * weight


def _count_rate(count: int, total: int) -> dict[str, Any]:
    return {"count": count, "rate": count / total if total else None}


def _joint_behavior_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[bool, bool, bool], int] = {}
    for row in rows:
        key = (
            not str(row.get("extracted_answer") or "").strip(),
            _is_length_cutoff(row),
            float(row.get("effective_training_reward") or 0.0) == 0.0,
        )
        counts[key] = counts.get(key, 0) + 1
    return [
        {
            "empty_extraction": key[0],
            "length_cutoff": key[1],
            "zero_effective_reward": key[2],
            "count": count,
        }
        for key, count in sorted(counts.items())
    ]


def _iter_normalized_rows(spec: RewardTraceSpec) -> Iterator[dict[str, Any]]:
    for path in sorted(spec.raw_paths, key=str):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"malformed reward rollout row {path}:{index + 1}") from exc
                if not isinstance(row, dict):
                    raise ValueError("reward rollout rows must be objects")
                yield _normalize_raw_row(
                    row, path=path, source_row_index=index
                )


def _prepare_row(row: dict[str, Any], *, spec: RewardTraceSpec) -> dict[str, Any]:
    required = _REQUIRED | (_GROUP_CREDIT_REQUIRED if spec.group_credit_enabled else set())
    missing = sorted(required - row.keys())
    if missing:
        raise ValueError(f"reward rollout row missing fields: {missing}")
    if spec.reward_transformations_active and row.get("effective_training_reward") is None:
        raise ValueError("effective training reward is required when transformations are active")
    identity = "\0".join(
        str(value)
        for value in (
            spec.trial_uid,
            spec.reward_function_sha256,
            row["segment_id"],
            row["training_step"],
            row["source_row_index"],
            row["sample_uid"],
        )
    )
    prepared = dict(row)
    prepared["trace_id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    prepared["trial_uid"] = spec.trial_uid
    prepared["reward_function_sha256"] = spec.reward_function_sha256
    prepared["reward_entrypoint"] = spec.reward_entrypoint
    prepared["prompt_group_id"] = str(
        row.get("group_uid")
        or row.get("prompt_group_id")
        or hashlib.sha256(str(row["prompt"]).encode("utf-8")).hexdigest()
    )
    return prepared


def _normalize_raw_row(
    row: dict[str, Any], *, path: Path, source_row_index: int
) -> dict[str, Any]:
    normalized = dict(row)
    normalized.setdefault("segment_id", path.parent.name)
    normalized.setdefault("training_step", row.get("step"))
    normalized.setdefault("source_row_index", source_row_index)
    normalized.setdefault(
        "sample_uid",
        row.get("request_id") or row.get("uid") or f"{path.name}:{source_row_index}",
    )
    normalized.setdefault("prompt", row.get("input"))
    normalized.setdefault("response", row.get("output"))
    normalized.setdefault("ground_truth", row.get("gts"))
    normalized.setdefault("custom_reward_score", row.get("custom_reward_score", row.get("acc")))
    normalized.setdefault("effective_training_reward", row.get("score"))
    if "assigned_training_reward" in normalized:
        normalized["effective_training_reward"] = normalized[
            "assigned_training_reward"
        ]
    normalized.setdefault("extracted_answer", row.get("extracted_answer", ""))
    if normalized.get("response_length_tokens") is None:
        normalized["response_length_tokens"] = len(str(normalized.get("response") or "").split())
    normalized.setdefault(
        "max_response_length_tokens", normalized.get("response_length_tokens")
    )
    return normalized


def _trace_schema(group_credit_enabled: bool) -> str:
    del group_credit_enabled
    return REWARD_TRACE_SCHEMA_VERSION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _phase(step: int, *, max_step: int) -> str:
    if max_step <= 0 or step * 3 <= max_step:
        return "early"
    if step * 3 <= max_step * 2:
        return "middle"
    return "late"


def _is_length_cutoff(row: dict[str, Any]) -> bool:
    return int(row["response_length_tokens"]) >= int(row["max_response_length_tokens"])
