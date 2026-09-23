"""Engine-owned validation and execution for complete-group reward assignment."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import importlib.util
import inspect
import math
import re
from typing import Callable, Mapping, Sequence

from ade.tasks.reward_design.process_evidence import (
    ADAPTER_ID,
    DIMENSIONS,
    SCHEMA_VERSION as PROCESS_SCHEMA_VERSION,
)


INPUT_SCHEMA = "ade.group_credit_input.v3"
OUTPUT_SCHEMA = "ade.group_credit_output.v2"
_GROUP_TYPES = {"all_wrong", "mixed", "all_correct"}
_MODES = {"identity", "abstained", "shaped"}
_SOURCES = {"outcome", "process", "rule_based", "response_length"}
_REASON = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
REWARD_TRACE_COMMON_INFO_KEYS = frozenset(
    {
        "data_source",
        "sample_uid",
        "group_credit_enabled",
        "process_evidence",
        "artifact_projection",
        "rule_evidence",
        "outcome_score",
        "pre_group_reward",
        "final_training_reward",
        "response_length_tokens",
        "max_response_length_tokens",
    }
)
GROUP_CREDIT_REQUIRED_REWARD_INFO_KEYS = REWARD_TRACE_COMMON_INFO_KEYS | frozenset(
    {
        "assigned_training_reward",
        "group_credit_source",
        "group_credit_reason",
        "group_credit_mode",
        "group_credit_evidence_sources",
        "group_credit_process_dimensions",
        "group_type",
        "group_uid",
    }
)


@dataclass(frozen=True)
class GroupCreditResult:
    assigned_rewards: tuple[float, ...]
    projected_advantages: tuple[float, ...]
    source: str
    abstained: bool
    reason: str
    mode: str
    evidence_sources: tuple[str, ...]
    process_dimensions: tuple[str, ...]


def load_group_credit_function(path: str, name: str = "assign_group_credit") -> Callable:
    spec = importlib.util.spec_from_file_location("ade_bound_group_credit", path)
    if spec is None or spec.loader is None:
        raise ValueError("group credit reward module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, name, None)
    _validate_callable(function)
    return function


def apply_group_credit_to_outputs(
    uids: Sequence[object],
    outputs: list[dict[str, object]],
    function: Callable[[dict[str, object]], object],
    *,
    rollout_n: int,
    reward_range: tuple[float, float] = (0.0, 1.0),
) -> list[dict[str, object]]:
    if len(uids) != len(outputs):
        raise ValueError("group credit uid/output lengths differ")
    groups: dict[object, list[int]] = {}
    for index, uid in enumerate(uids):
        groups.setdefault(uid, []).append(index)
    for uid, indices in groups.items():
        if len(indices) != rollout_n:
            raise ValueError("group credit requires one complete rollout group")
        records = []
        for record_id, batch_index in enumerate(indices):
            output = outputs[batch_index]
            info = output.get("reward_extra_info")
            if not isinstance(info, dict):
                raise ValueError("group credit requires reward_extra_info")
            outcome = _finite_number(info.get("outcome_score"), "outcome_score")
            process = _process_evidence_for_group(
                info.get("process_evidence")
            )
            rule = _rule_evidence(info.get("rule_evidence"))
            response_length = _response_length_evidence(info)
            records.append(
                {
                    "record_id": record_id,
                    "is_correct": outcome == 1.0,
                    "outcome_score": outcome,
                    "pre_group_reward": _finite_number(
                        output.get("reward_score"), "pre_group_reward"
                    ),
                    "evidence": {
                        "outcome": {"status": "authoritative", "value": outcome},
                        "process": process,
                        "rule_based": rule,
                        "response_length": response_length,
                    },
                }
            )
        group_input = {
            "schema_version": INPUT_SCHEMA,
            "group_type": group_type_for(records),
            "records": records,
        }
        result = execute_group_credit(function, group_input, reward_range=reward_range)
        for batch_index, assigned in zip(indices, result.assigned_rewards, strict=True):
            output = outputs[batch_index]
            info = output["reward_extra_info"]
            pre_group = float(output["reward_score"])
            output["reward_score"] = assigned
            info.update(
                {
                    "group_credit_enabled": True,
                    "group_uid": str(uid),
                    "group_type": group_input["group_type"],
                    "pre_group_reward": pre_group,
                    "assigned_training_reward": assigned,
                    "final_training_reward": assigned,
                    "group_credit_source": result.source,
                    "group_credit_reason": result.reason,
                    "group_credit_mode": result.mode,
                    "group_credit_evidence_sources": list(result.evidence_sources),
                    "group_credit_process_dimensions": list(result.process_dimensions),
                }
            )
    group_credit_reward_extra_keys(outputs)
    return outputs


def group_credit_reward_extra_keys(
    outputs: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    keys: set[str] = set()
    for row, output in enumerate(outputs):
        info = output.get("reward_extra_info")
        if not isinstance(info, Mapping):
            raise ValueError(
                f"group credit reward metadata row {row} requires reward_extra_info"
            )
        missing = GROUP_CREDIT_REQUIRED_REWARD_INFO_KEYS.difference(info)
        if missing:
            raise ValueError(
                f"group credit reward metadata row {row} missing required fields: "
                f"{sorted(missing)}; available fields: {sorted(info)}"
            )
        _finite_number(info["outcome_score"], "outcome_score")
        keys.update(str(key) for key in info)
    return tuple(sorted(keys))


def admit_group_credit_function(
    function: Callable[[dict[str, object]], object],
    *,
    group_size: int = 8,
    reward_range: tuple[float, float] = (0.0, 1.0),
    observe_process_bank: bool = False,
) -> dict[str, object] | None:
    if group_size < 2:
        raise ValueError("group credit admission requires group_size >= 2")
    ladder = [(index + 1) / (group_size + 1) for index in range(group_size)]
    for outcomes, process_values in (
        ([0.0] * group_size, ladder),
        ([1.0 if index % 2 else 0.0 for index in range(group_size)], ladder),
        ([0.0] * (group_size - 1) + [1.0], [1.0] + [0.0] * (group_size - 1)),
        ([0.0] + [1.0] * (group_size - 1), [0.0, 0.0] + [1.0] * (group_size - 2)),
        ([1.0] * group_size, ladder),
    ):
        group_input = _admission_group(outcomes, process_values=process_values)
        first = execute_group_credit(function, group_input, reward_range=reward_range)
        second = execute_group_credit(function, group_input, reward_range=reward_range)
        if first != second:
            raise ValueError("assign_group_credit must be deterministic")
        base_by_id = {
            record["record_id"]: reward
            for record, reward in zip(group_input["records"], first.assigned_rewards, strict=True)
        }
        permuted = copy.deepcopy(group_input)
        permuted["records"] = list(reversed(permuted["records"]))
        permuted_result = execute_group_credit(function, permuted, reward_range=reward_range)
        permuted_by_id = {
            record["record_id"]: reward
            for record, reward in zip(permuted["records"], permuted_result.assigned_rewards, strict=True)
        }
        if permuted_by_id != base_by_id:
            raise ValueError("assign_group_credit must be invariant to record order")
        relabeled = copy.deepcopy(group_input)
        for record in relabeled["records"]:
            record["record_id"] = group_size - 1 - int(record["record_id"])
        relabeled_result = execute_group_credit(function, relabeled, reward_range=reward_range)
        if relabeled_result.assigned_rewards != first.assigned_rewards:
            raise ValueError("assign_group_credit cannot depend on opaque record IDs")
    if observe_process_bank:
        bank_sensitive = False
        for outcomes in (
            [0.0] * group_size,
            [1.0 if index % 2 else 0.0 for index in range(group_size)],
            [1.0] * group_size,
        ):
            direct = execute_group_credit(
                function,
                _admission_group(
                    outcomes,
                    process_values=[0.0] + [1.0] * (group_size - 1),
                ),
                reward_range=reward_range,
            )
            reversed_bank = execute_group_credit(
                function,
                _admission_group(
                    outcomes,
                    process_values=[1.0] * (group_size - 1) + [0.0],
                ),
                reward_range=reward_range,
            )
            if (
                direct.process_dimensions
                and reversed_bank.process_dimensions
                and direct != reversed_bank
            ):
                bank_sensitive = True
                break
        return {
            "status": "change_observed" if bank_sensitive else "no_change_observed",
            "group_size": group_size,
            "scope": "fixed_synthetic_process_bank_probes",
            "interpretation": (
                "These probes observe declared process-dependent result changes only; "
                "no change does not establish that the policy ignores process evidence. "
                "Interpret against the accepted Plan during reflection."
            ),
        }
    return None


def _admission_group(
    outcomes: Sequence[float],
    *,
    reverse_process: bool = False,
    process_values: Sequence[float] | None = None,
) -> dict[str, object]:
    if process_values is not None and len(process_values) != len(outcomes):
        raise ValueError("admission process bank length differs from outcomes")
    records = []
    for record_id, outcome in enumerate(outcomes):
        if process_values is None:
            process_index = len(outcomes) - record_id if reverse_process else record_id + 1
            process = process_index / (len(outcomes) + 1)
        else:
            process = float(process_values[record_id])
        records.append(
            {
                "record_id": record_id,
                "is_correct": outcome == 1.0,
                "outcome_score": outcome,
                "pre_group_reward": outcome,
                "evidence": {
                    "outcome": {"status": "authoritative", "value": outcome},
                    "process": {
                        "status": "available",
                        "adapter_id": ADAPTER_ID,
                        "schema_version": PROCESS_SCHEMA_VERSION,
                        "dimensions": {name: process for name in DIMENSIONS},
                    },
                    "rule_based": {"status": "available", "value": float(record_id % 2)},
                    "response_length": {
                        "status": "authoritative",
                        "value": 10 + record_id,
                        "maximum": 32,
                    },
                },
            }
        )
    return {"schema_version": INPUT_SCHEMA, "group_type": group_type_for(records), "records": records}


def group_type_for(records: Sequence[Mapping[str, object]]) -> str:
    correctness = [record.get("is_correct") is True for record in records]
    if all(correctness):
        return "all_correct"
    if any(correctness):
        return "mixed"
    return "all_wrong"


def grpo_sequence_advantages(
    rewards: Sequence[float], *, epsilon: float = 1.0e-6
) -> tuple[float, ...]:
    values = [float(value) for value in rewards]
    if len(values) < 2:
        return tuple(values)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = math.sqrt(variance)
    return tuple((value - mean) / (std + epsilon) for value in values)


def execute_group_credit(
    function: Callable[[dict[str, object]], object],
    group_input: Mapping[str, object],
    *,
    reward_range: tuple[float, float] = (0.0, 1.0),
) -> GroupCreditResult:
    records = _validate_group_input(group_input)
    _validate_callable(function)
    output = function(copy.deepcopy(dict(group_input)))
    assigned, mode, sources, process_dimensions, reason = _validate_output(
        output, records=records, reward_range=reward_range
    )
    advantages = grpo_sequence_advantages(assigned)
    _validate_mixed_advantage_direction(records, advantages)
    return GroupCreditResult(
        assigned,
        advantages,
        "artifact",
        mode == "abstained",
        reason,
        mode,
        sources,
        process_dimensions,
    )


def _validate_callable(function: object) -> None:
    if not callable(function) or inspect.iscoroutinefunction(function):
        raise ValueError("assign_group_credit must be a synchronous callable")
    if list(inspect.signature(function).parameters) != ["group_input"]:
        raise ValueError("assign_group_credit must accept exactly group_input")


def _validate_group_input(value: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "group_type", "records"}:
        raise ValueError("group credit input fields are invalid")
    if value.get("schema_version") != INPUT_SCHEMA:
        raise ValueError("group credit input schema is invalid")
    records_value = value.get("records")
    if not isinstance(records_value, list) or not records_value:
        raise ValueError("group credit input requires records")
    records: list[Mapping[str, object]] = []
    ids: set[int] = set()
    for record in records_value:
        if not isinstance(record, Mapping) or set(record) != {
            "record_id", "is_correct", "outcome_score", "pre_group_reward", "evidence"
        }:
            raise ValueError("group credit record fields are invalid")
        record_id = record.get("record_id")
        if type(record_id) is not int or record_id < 0 or record_id >= len(records_value) or record_id in ids:
            raise ValueError("record_id must be a unique in-range integer")
        ids.add(record_id)
        if type(record.get("is_correct")) is not bool:
            raise ValueError("is_correct must be boolean")
        outcome = _finite_number(record.get("outcome_score"), "outcome_score")
        _finite_number(record.get("pre_group_reward"), "pre_group_reward")
        evidence = record.get("evidence")
        if not isinstance(evidence, Mapping) or set(evidence) != {
            "outcome", "process", "rule_based", "response_length"
        }:
            raise ValueError("record evidence fields are invalid")
        outcome_evidence = evidence["outcome"]
        if not isinstance(outcome_evidence, Mapping) or outcome_evidence != {
            "status": "authoritative", "value": outcome
        }:
            raise ValueError("outcome evidence must match authoritative outcome")
        _process_evidence_for_group(evidence["process"])
        _rule_evidence(evidence["rule_based"])
        _response_length_evidence_values(evidence["response_length"])
        records.append(record)
    if value.get("group_type") not in _GROUP_TYPES or value["group_type"] != group_type_for(records):
        raise ValueError("group_type does not match authoritative correctness")
    return tuple(records)


def _validate_output(
    value: object,
    *,
    records: Sequence[Mapping[str, object]],
    reward_range: tuple[float, float],
) -> tuple[tuple[float, ...], str, tuple[str, ...], tuple[str, ...], str]:
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "decision", "assignments"}:
        raise ValueError("group credit output fields are invalid")
    if value.get("schema_version") != OUTPUT_SCHEMA:
        raise ValueError("group credit output schema is invalid")
    decision = value.get("decision")
    if not isinstance(decision, Mapping) or set(decision) != {
        "mode", "evidence_sources", "process_dimensions", "reason_code"
    }:
        raise ValueError("group credit decision fields are invalid")
    mode = decision.get("mode")
    if mode not in _MODES:
        raise ValueError("group credit decision mode is invalid")
    sources = decision.get("evidence_sources")
    if not isinstance(sources, list) or not sources or sources != sorted(set(sources)) or any(
        source not in _SOURCES for source in sources
    ):
        raise ValueError("evidence_sources must be a sorted unique supported list")
    dimensions = decision.get("process_dimensions")
    if not isinstance(dimensions, list) or dimensions != sorted(set(dimensions)) or any(
        dimension not in DIMENSIONS for dimension in dimensions
    ):
        raise ValueError("process_dimensions must be a sorted unique fixed list")
    if ("process" in sources) != bool(dimensions):
        raise ValueError("process evidence source must declare exact process_dimensions")
    if dimensions:
        for record in records:
            process = record["evidence"]["process"]  # type: ignore[index]
            if process["status"] != "available" or any(
                dimension not in process["dimensions"] for dimension in dimensions
            ):
                raise ValueError("declared process dimension is unavailable")
    for source in sources:
        if source == "rule_based" and any(
            record["evidence"]["rule_based"]["status"] != "available"  # type: ignore[index]
            for record in records
        ):
            raise ValueError("declared rule_based evidence is unavailable")
    reason = decision.get("reason_code")
    if not isinstance(reason, str) or _REASON.fullmatch(reason) is None:
        raise ValueError("group credit reason_code is invalid")
    assignments = value.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(records):
        raise ValueError("group credit assignments length is invalid")
    by_id: dict[int, float] = {}
    lower, upper = reward_range
    for assignment in assignments:
        if not isinstance(assignment, Mapping) or set(assignment) != {"record_id", "training_reward"}:
            raise ValueError("group credit assignment fields are invalid")
        record_id = assignment.get("record_id")
        if type(record_id) is not int or record_id in by_id:
            raise ValueError("assignment record_id must be a unique integer")
        reward = _finite_number(assignment.get("training_reward"), "training_reward")
        if reward < lower or reward > upper:
            raise ValueError("training_reward is outside the fixed reward range")
        by_id[record_id] = reward
    input_ids = {int(record["record_id"]) for record in records}
    if set(by_id) != input_ids:
        raise ValueError("assignment identity set does not match input")
    assigned = tuple(by_id[int(record["record_id"])] for record in records)
    outcomes = tuple(float(record["outcome_score"]) for record in records)
    changed = any(reward != outcome for reward, outcome in zip(assigned, outcomes, strict=True))
    if mode in {"identity", "abstained"} and changed:
        raise ValueError(f"{mode} assignment must equal outcome")
    return assigned, str(mode), tuple(sources), tuple(dimensions), str(reason)


def _validate_mixed_advantage_direction(
    records: Sequence[Mapping[str, object]], advantages: Sequence[float]
) -> None:
    if group_type_for(records) != "mixed":
        return
    for record, advantage in zip(records, advantages, strict=True):
        if record["is_correct"] is True and not advantage > 0.0:
            raise ValueError("mixed group correct response must have positive realized advantage")
        if record["is_correct"] is False and advantage > 0.0:
            raise ValueError("mixed group incorrect response cannot have positive realized advantage")


def _process_evidence_for_group(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("process evidence must be an object")
    status = value.get("status")
    if status == "available":
        dimensions = value.get("dimensions")
        if value.get("adapter_id") != ADAPTER_ID or value.get("schema_version") != PROCESS_SCHEMA_VERSION:
            raise ValueError("process evidence adapter/schema is invalid")
        if not isinstance(dimensions, Mapping) or set(dimensions) != set(DIMENSIONS):
            raise ValueError("process evidence dimensions are invalid")
        normalized = {name: _bounded_number(dimensions[name], name) for name in DIMENSIONS}
        return {
            "status": status,
            "adapter_id": ADAPTER_ID,
            "schema_version": PROCESS_SCHEMA_VERSION,
            "dimensions": normalized,
        }
    if status == "unavailable":
        if value.get("adapter_id") != ADAPTER_ID or value.get("schema_version") != PROCESS_SCHEMA_VERSION or value.get("dimensions") is not None:
            raise ValueError("unavailable process evidence is invalid")
        return {
            "status": status,
            "adapter_id": ADAPTER_ID,
            "schema_version": PROCESS_SCHEMA_VERSION,
            "dimensions": None,
        }
    if status == "not_configured":
        if any(value.get(key) is not None for key in ("adapter_id", "schema_version", "dimensions")):
            raise ValueError("not_configured process evidence is invalid")
        return {
            "status": status,
            "adapter_id": None,
            "schema_version": None,
            "dimensions": None,
        }
    raise ValueError("process evidence status is invalid")


def _rule_evidence(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"status", "value"}:
        raise ValueError("rule evidence fields are invalid")
    if value.get("status") == "available":
        return {"status": "available", "value": _bounded_number(value.get("value"), "rule evidence")}
    if value.get("status") == "not_configured" and value.get("value") is None:
        return {"status": "not_configured", "value": None}
    raise ValueError("rule evidence status/value is invalid")


def _response_length_evidence(info: Mapping[str, object]) -> dict[str, object]:
    return _response_length_evidence_values(
        {
            "status": "authoritative",
            "value": info.get("response_length_tokens"),
            "maximum": info.get("max_response_length_tokens"),
        }
    )


def _response_length_evidence_values(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"status", "value", "maximum"}:
        raise ValueError("response length evidence fields are invalid")
    length = value.get("value")
    maximum = value.get("maximum")
    if value.get("status") != "authoritative" or type(length) is not int or type(maximum) is not int or length < 0 or maximum < 1 or length > maximum:
        raise ValueError("response length evidence is invalid")
    return {"status": "authoritative", "value": length, "maximum": maximum}


def _bounded_number(value: object, label: str) -> float:
    number = _finite_number(value, label)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be within [0,1]")
    return number


def _finite_number(value: object, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric")
    return float(value)
