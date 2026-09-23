"""Deterministic compliance replay for generated reward functions."""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ade.tasks.reward_design.rewards.contracts import (
    enforce_outcome_component,
    validate_reward_result,
    validate_reward_score,
)
from ade.engine.judge_dispatcher_impl import RubricRowUnavailable
from ade.tasks.reward_design.rewards.training_outcome import TrainingOutcome, compute_training_outcome
from ade.tasks.reward_design.group_credit import (
    execute_group_credit,
    grpo_sequence_advantages,
    group_type_for,
)


_COVERAGE_ORDER = (
    "all_wrong",
    "mixed",
    "all_correct",
    "outcome_flat",
    "cutoff",
    "malformed",
    "conflict",
    "process_tie",
    "process_separation",
    "process_unavailable",
)


class RewardComplianceInputError(ValueError):
    """The Harness cannot reconstruct the fixed reward call from a source row."""


def select_baseline_records(
    records_by_step: Mapping[int, Sequence[dict[str, object]]],
    *,
    group_size: int,
    prompt_groups: int,
) -> tuple[dict[str, object], ...]:
    _step, records = select_baseline_position_records(
        records_by_step,
        group_size=group_size,
        prompt_groups=prompt_groups,
    )
    return records


def select_baseline_position_records(
    records_by_step: Mapping[int, Sequence[dict[str, object]]],
    *,
    group_size: int,
    prompt_groups: int,
) -> tuple[int, tuple[dict[str, object], ...]]:
    step, records, _selection = select_baseline_position(
        records_by_step,
        group_size=group_size,
        prompt_groups=prompt_groups,
    )
    return step, records


def select_baseline_position(
    records_by_step: Mapping[int, Sequence[dict[str, object]]],
    *,
    group_size: int,
    prompt_groups: int,
) -> tuple[int, tuple[dict[str, object], ...], dict[str, object]]:
    if not records_by_step:
        raise ValueError("reward compliance requires a baseline rollout step")
    if group_size < 1 or prompt_groups < 1:
        raise ValueError("reward compliance group dimensions must be positive")
    selected_step = None
    complete: list[tuple[str, list[dict[str, object]]]] = []
    for step in sorted(records_by_step, reverse=True):
        candidate = _complete_groups(records_by_step[step], group_size=group_size)
        if len(candidate) >= prompt_groups:
            selected_step = step
            complete = candidate
            break
    if selected_step is None:
        raise ValueError(
            f"reward compliance lacks {prompt_groups} complete rollout groups "
            f"of size {group_size} at one position"
        )
    by_id = dict(complete)
    labels = {group_id: _group_labels(rows) for group_id, rows in complete}
    selected_ids: list[str] = []
    for label in _COVERAGE_ORDER:
        if any(label in labels[group_id] for group_id in selected_ids):
            continue
        match = next(
            (
                group_id
                for group_id, _rows in complete
                if group_id not in selected_ids and label in labels[group_id]
            ),
            None,
        )
        if match is not None:
            selected_ids.append(match)
    selected_ids.extend(
        group_id
        for group_id, _rows in complete
        if group_id not in selected_ids
    )
    selected_ids = selected_ids[:prompt_groups]
    selected_records = tuple(
        row for group_id in selected_ids for row in by_id[group_id]
    )
    return selected_step, selected_records, {
        "position": {"unit": "rl_step", "value": selected_step},
        "selected_group_ids": selected_ids,
        "shape_coverage": {
            label: (
                {
                    "status": "covered",
                    "group_ids": [
                        group_id
                        for group_id in selected_ids
                        if label in labels[group_id]
                    ],
                }
                if any(label in labels[group_id] for group_id, _rows in complete)
                else {"status": "unavailable_in_source", "group_ids": []}
            )
            for label in _COVERAGE_ORDER
        },
    }


def _complete_groups(
    records: Sequence[dict[str, object]],
    *,
    group_size: int,
) -> list[tuple[str, list[dict[str, object]]]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for record in records:
        details = record.get("details")
        group_id = details.get("prompt_group_id") if isinstance(details, dict) else None
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("reward compliance record lacks prompt_group_id")
        groups.setdefault(group_id, []).append(record)
    complete = []
    for group_id, rows in sorted(groups.items()):
        ordered = sorted(rows, key=_record_identity)
        indices = [_response_index(row) for row in ordered]
        if len(rows) == group_size and indices == list(range(group_size)):
            complete.append((group_id, ordered))
    return complete


def _group_labels(rows: Sequence[dict[str, object]]) -> set[str]:
    labels: set[str] = set()
    group_types = {_value(row, "group_type") for row in rows}
    group_types.discard(None)
    if len(group_types) == 1 and next(iter(group_types)) in {
        "all_wrong",
        "mixed",
        "all_correct",
    }:
        labels.add(str(next(iter(group_types))))
    else:
        correctness = [_value(row, "correctness") is True for row in rows]
        labels.add(
            "all_correct"
            if all(correctness)
            else "mixed" if any(correctness) else "all_wrong"
        )
    outcomes = [_value(row, "outcome_score") for row in rows]
    if all(type(value) in {int, float} for value in outcomes) and len(
        {float(value) for value in outcomes}
    ) == 1:
        labels.add("outcome_flat")
    if any(
        type(_value(row, "response_length_tokens")) is int
        and type(_value(row, "max_response_length_tokens")) is int
        and int(_value(row, "response_length_tokens"))
        >= int(_value(row, "max_response_length_tokens"))
        for row in rows
    ):
        labels.add("cutoff")
    if any(_value(row, "malformed") is True for row in rows):
        labels.add("malformed")
    if any(_value(row, "conflict") is True for row in rows):
        labels.add("conflict")
    process = [_value(row, "process_evidence") for row in rows]
    available = [
        value
        for value in process
        if isinstance(value, Mapping) and value.get("status") == "available"
    ]
    if len(available) != len(rows):
        labels.add("process_unavailable")
    else:
        scores = [value.get("projected_score", value.get("value")) for value in available]
        if all(type(value) in {int, float} for value in scores):
            labels.add(
                "process_tie"
                if len({float(value) for value in scores}) == 1
                else "process_separation"
            )
    return labels


def _value(record: Mapping[str, object], key: str) -> object:
    for owner in ("details", "training_context", "assessment"):
        value = record.get(owner)
        if isinstance(value, Mapping) and key in value:
            return value[key]
    return None


async def replay_reward_pair(
    reference: Callable[..., Any],
    candidate: Callable[..., Any],
    fallback: Callable[..., Any],
    records: Sequence[dict[str, object]],
    *,
    max_concurrency: int = 256,
    group_credit_enabled: bool = False,
    reference_group_credit: Callable[..., Any] | None = None,
    candidate_group_credit: Callable[..., Any] | None = None,
) -> dict[str, object]:
    validate_reward_compliance_records(records)
    if max_concurrency < 1 or max_concurrency > 256:
        raise ValueError("reward compliance concurrency must be between one and 256")
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run(index: int, record: dict[str, object]) -> dict[str, object]:
        async with semaphore:
            kwargs, training_outcome = _reward_arguments(record)
            reference_score, reference_outcome, reference_rule = _reference_result(
                await _call(reference, kwargs)
            )
            if not math.isclose(reference_outcome, training_outcome.outcome_score):
                raise ValueError(
                    "reference reward outcome does not match TrainingOutcomeAdapter"
                )
            forced = enforce_outcome_component(
                validate_reward_result(
                    await _call(fallback, kwargs),
                    fallback=True,
                    group_credit_enabled=group_credit_enabled,
                ),
                training_outcome.outcome_score,
            )
            used_fallback = False
            fallback_reason = None
            try:
                result = enforce_outcome_component(
                    validate_reward_result(
                        await _call(candidate, kwargs),
                        fallback=False,
                        group_credit_enabled=group_credit_enabled,
                    ),
                    training_outcome.outcome_score,
                )
            except RubricRowUnavailable as error:
                result = forced
                used_fallback = True
                fallback_reason = str(error)
            forced_outcome = forced["outcome_score"]
            if not math.isclose(float(forced_outcome), reference_outcome):
                raise ValueError("candidate fallback outcome does not preserve reference reward")
            outcome = result["outcome_score"]
            if not math.isclose(float(outcome), reference_outcome):
                raise ValueError("candidate outcome does not preserve reference reward")
            return {
                "index": index,
                "record_id": str(record.get("record_id") or ""),
                "reference_score": reference_score,
                "candidate_score": result["score"],
                "forced_fallback_score": forced["score"],
                "artifact_projection": result["artifact_projection"],
                "rule_evidence": result["rule_evidence"],
                "reference_rule_evidence": reference_rule,
                "training_outcome": training_outcome.to_dict(),
                "extraction_method": training_outcome.extraction_method,
                "judge_fallback": used_fallback,
                "fallback_reason": fallback_reason,
            }

    rows = await asyncio.gather(*(run(index, row) for index, row in enumerate(records)))
    _apply_replay_group_facts(
        rows,
        records,
        reference_group_credit=reference_group_credit,
        candidate_group_credit=candidate_group_credit,
    )
    return {
        "schema_version": "1",
        "record_count": len(rows),
        "normal_count": len(rows),
        "forced_fallback_count": len(rows),
        "fallback_count": sum(bool(row["judge_fallback"]) for row in rows),
        "records": rows,
    }


def _reward_arguments(
    record: dict[str, object],
) -> tuple[dict[str, object], TrainingOutcome]:
    question_prompt, response_content, data_source, ground_truth, details = (
        _validated_reward_inputs(record)
    )
    outcome = compute_training_outcome(data_source, response_content, ground_truth)
    return (
        {
            "question_prompt": question_prompt,
            "response_content": response_content,
            "extracted_answer": outcome.extracted_answer,
            "outcome_score": outcome.outcome_score,
            "response_length_tokens": int(
                details.get("response_length_tokens") or 0
            ),
            "max_response_length_tokens": int(
                details.get("max_response_length_tokens") or 0
            ),
        },
        outcome,
    )


def _validated_reward_inputs(
    record: dict[str, object],
) -> tuple[str, str, str, str, dict[str, object]]:
    details = record.get("details")
    generation = record.get("generation")
    prompt = record.get("prompt")
    if (
        not isinstance(details, dict)
        or not isinstance(generation, dict)
        or not isinstance(prompt, dict)
    ):
        raise RewardComplianceInputError(
            "model-behavior record requires prompt, generation, and details objects"
        )
    inputs = {
        "prompt.text": prompt.get("text"),
        "generation.raw_response": generation.get("raw_response"),
        "details.data_source": details.get("data_source"),
        "details.ground_truth": details.get("ground_truth"),
    }
    missing = sorted(
        field for field, value in inputs.items() if not isinstance(value, str)
    )
    if missing:
        raise RewardComplianceInputError(
            f"reward compliance record lacks string inputs: {missing}"
        )
    return (
        str(inputs["prompt.text"]),
        str(inputs["generation.raw_response"]),
        str(inputs["details.data_source"]),
        str(inputs["details.ground_truth"]),
        details,
    )


def validate_reward_compliance_records(
    records: Sequence[dict[str, object]],
) -> None:
    if not records or len(records) > 256:
        raise RewardComplianceInputError(
            "reward compliance requires between one and 256 records"
        )
    for index, record in enumerate(records):
        try:
            _validated_reward_inputs(record)
        except RewardComplianceInputError as error:
            raise RewardComplianceInputError(
                f"reward compliance record {index} is invalid: {error}"
            ) from error


def _reference_result(value: object) -> tuple[float, float, object]:
    if isinstance(value, dict):
        if set(value) == {
            "score",
            "components",
            "component_weights",
            "judge_fallback",
        }:
            components = value.get("components")
            if not isinstance(components, dict) or "outcome" not in components:
                raise ValueError("legacy baseline reward lacks components.outcome")
            return (
                validate_reward_score(value["score"]),
                validate_reward_score(components["outcome"]),
                {"status": "not_configured", "value": None},
            )
        result = validate_reward_result(value, fallback=False)
        return (
            float(result["score"]),
            float(result["outcome_score"]),
            result["rule_evidence"],
        )
    score = validate_reward_score(value)
    return score, score, {"status": "not_configured", "value": None}


def _apply_replay_group_facts(
    rows: list[dict[str, object]],
    records: Sequence[dict[str, object]],
    *,
    reference_group_credit: Callable[..., Any] | None,
    candidate_group_credit: Callable[..., Any] | None,
) -> None:
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        details = record.get("details")
        group_id = details.get("prompt_group_id") if isinstance(details, dict) else None
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("reward replay record lacks prompt_group_id")
        groups.setdefault(group_id, []).append(index)
    for group_id in sorted(groups):
        indices = sorted(groups[group_id], key=lambda index: _record_identity(records[index]))
        reference_rewards = tuple(float(rows[index]["reference_score"]) for index in indices)
        candidate_rewards = tuple(float(rows[index]["candidate_score"]) for index in indices)
        reference_assigned, reference_advantages = _assigned_group_facts(
            reference_group_credit,
            indices,
            rows,
            records,
            rewards=reference_rewards,
            rule_key="reference_rule_evidence",
        )
        candidate_assigned, candidate_advantages = _assigned_group_facts(
            candidate_group_credit,
            indices,
            rows,
            records,
            rewards=candidate_rewards,
            rule_key="rule_evidence",
        )
        group_type = group_type_for(
            [
                {
                    "is_correct": float(
                        rows[index]["training_outcome"]["outcome_score"]
                    )
                    == 1.0
                }
                for index in indices
            ]
        )
        for offset, index in enumerate(indices):
            rows[index].update(
                {
                    "prompt_group_id": group_id,
                    "group_type": group_type,
                    "reference_assigned_reward": reference_assigned[offset],
                    "candidate_assigned_reward": candidate_assigned[offset],
                    "reference_realized_advantage": reference_advantages[offset],
                    "candidate_realized_advantage": candidate_advantages[offset],
                    "advantage_delta": (
                        candidate_advantages[offset] - reference_advantages[offset]
                    ),
                    "advantage_sign_change": _sign(candidate_advantages[offset])
                    != _sign(reference_advantages[offset]),
                }
            )


def _assigned_group_facts(
    function: Callable[..., Any] | None,
    indices: Sequence[int],
    rows: Sequence[Mapping[str, object]],
    records: Sequence[Mapping[str, object]],
    *,
    rewards: tuple[float, ...],
    rule_key: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if function is None:
        return rewards, grpo_sequence_advantages(rewards)
    group_records = []
    for record_id, index in enumerate(indices):
        outcome = float(rows[index]["training_outcome"]["outcome_score"])
        group_records.append(
            {
                "record_id": record_id,
                "is_correct": outcome == 1.0,
                "outcome_score": outcome,
                "pre_group_reward": rewards[record_id],
                "evidence": {
                    "outcome": {"status": "authoritative", "value": outcome},
                    "process": _value(records[index], "process_evidence")
                    or {"status": "unavailable", "reason": "not_persisted"},
                    "rule_based": rows[index][rule_key],
                    "response_length": {
                        "status": "authoritative",
                        "value": _value(records[index], "response_length_tokens") or 0,
                        "maximum": _value(records[index], "max_response_length_tokens") or 0,
                    },
                },
            }
        )
    group_input = {
        "schema_version": "ade.group_credit_input.v3",
        "group_type": group_type_for(group_records),
        "records": group_records,
    }
    result = execute_group_credit(function, group_input)
    return result.assigned_rewards, result.projected_advantages


def _sign(value: float) -> int:
    return (value > 0.0) - (value < 0.0)


async def _call(function: Callable[..., Any], kwargs: dict[str, object]) -> Any:
    result = function(**kwargs)
    return await result if inspect.isawaitable(result) else result


def _response_index(record: Mapping[str, object]) -> int:
    details = record.get("details")
    response_index = details.get("response_index") if isinstance(details, dict) else None
    if response_index is None:
        response_index = record.get("generation", {}).get("sample_index") if isinstance(record.get("generation"), dict) else None
    return int(response_index or 0)


def _record_identity(record: dict[str, object]) -> tuple[int, str]:
    return _response_index(record), str(record.get("record_id") or "")
