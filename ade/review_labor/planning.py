"""Admission and canonical expansion for Analyzer-authored Review plans."""

from __future__ import annotations

import json
import math
import gzip
from pathlib import Path
from typing import Any, Mapping

from ade.review_labor.service import _validate_rubrics

from ade.review_labor.protocol import ReviewBatch, ReviewCommand
from ade.tasks.contracts import AnalysisReviewPlan


def compile_review_command(
    *,
    plan: AnalysisReviewPlan,
    attempt: Path,
    command_id: str,
    logical_command_id: str,
    attempt_id: str,
    attempt_index: int,
    run_id: str,
    coordinator_id: str,
    plan_id: str,
    trial_id: str,
    basis_revision: int,
) -> ReviewCommand:
    if plan.schema_version != "ade.analysis_review_plan.v2":
        raise ValueError("review plan schema must be ade.analysis_review_plan.v2")
    manifest = json.loads(
        (attempt / "input" / "experiment" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    catalog = json.loads(
        (attempt / "input" / "experiment" / "evidence-catalog.json").read_text(
            encoding="utf-8"
        )
    )
    inventory, requirements = _inventory(attempt, manifest, catalog)
    known_pools = set(inventory)
    batch_ids: set[str] = set()
    selected: set[tuple[str, str]] = set()
    counts = {pool: 0 for pool in inventory}
    selected_groups: dict[str, set[str]] = {pool: set() for pool in inventory}
    batches = []
    for batch in plan.batches:
        if not batch.batch_id.strip() or batch.batch_id in batch_ids:
            raise ValueError("review plan batch IDs must be non-empty and unique")
        batch_ids.add(batch.batch_id)
        if batch.pool not in known_pools:
            raise ValueError(f"unknown review pool: {batch.pool}")
        if not batch.investigation_purpose.strip():
            raise ValueError("review batch investigation purpose is required")
        _validate_rubrics(tuple(dict(item) for item in batch.rubrics))
        units = []
        for identity, canonical in _expand_selection(
            batch.selection, inventory[batch.pool]
        ):
            if identity in selected:
                continue
            selected.add(identity)
            counts[batch.pool] += 1
            group_id = canonical.get("context", {}).get("prompt_group_id")
            if group_id is not None:
                selected_groups[batch.pool].add(str(group_id))
            units.append(dict(canonical))
        if units:
            batches.append(
                ReviewBatch(
                    batch_id=batch.batch_id,
                    pool=batch.pool,
                    investigation_purpose=batch.investigation_purpose,
                    units=tuple(units),
                    rubrics=tuple(dict(item) for item in batch.rubrics),
                )
            )
    for pool, requirement in requirements.items():
        required = int(requirement["required_unique"])
        observed = (
            len(selected_groups[pool])
            if requirement.get("coverage_unit") == "groups"
            else counts[pool]
        )
        if requirement.get("coverage_unit") == "groups":
            for group_id in selected_groups[pool]:
                expected_members = {
                    identity
                    for identity, unit in inventory[pool].items()
                    if str(unit.get("context", {}).get("prompt_group_id")) == group_id
                }
                if not expected_members.issubset(selected):
                    raise ValueError(
                        f"review pool {pool} group {group_id} must include every sibling"
                    )
        if observed < required:
            raise ValueError(
                f"review pool {pool} requires {required} unique units, received {observed}"
            )
    return ReviewCommand(
        command_id=command_id,
        logical_command_id=logical_command_id,
        attempt_id=attempt_id,
        attempt_index=attempt_index,
        run_id=run_id,
        coordinator_id=coordinator_id,
        plan_id=plan_id,
        trial_id=trial_id,
        basis_revision=basis_revision,
        batches=tuple(batches),
        pool_requirements=tuple(requirements.values()),
    )


def coverage_from_packet(
    command: ReviewCommand, packet: Mapping[str, object]
) -> dict[str, object]:
    result_batches = packet.get("batches", ())
    if not isinstance(result_batches, list):
        result_batches = []
    by_id = {
        str(item.get("batch_id")): item
        for item in result_batches
        if isinstance(item, Mapping)
    }
    pools: dict[str, dict[str, object]] = {
        str(item["pool_id"]): {
            **dict(item),
            "requested_unique": 0,
            "completed_unique": 0,
            "failed_unique": 0,
            "fallback_unique": 0,
            "position_distribution": {},
            "group_distribution": {},
        }
        for item in command.pool_requirements
    }
    degraded = packet.get("status") in {"partial", "unavailable"}
    requested_groups: dict[str, set[str]] = {pool: set() for pool in pools}
    completed_groups: dict[str, set[str]] = {pool: set() for pool in pools}
    for batch in command.batches:
        row = by_id.get(batch.batch_id, {})
        entry = pools.setdefault(
            batch.pool,
            {
                "requested_unique": 0,
                "completed_unique": 0,
                "failed_unique": 0,
                "fallback_unique": 0,
                "position_distribution": {},
                "group_distribution": {},
            },
        )
        group_ids = {
            str(unit.get("context", {}).get("prompt_group_id"))
            for unit in batch.units
            if unit.get("context", {}).get("prompt_group_id") is not None
        }
        if entry.get("coverage_unit") == "groups":
            requested_groups.setdefault(batch.pool, set()).update(group_ids)
            if int(row.get("completed_units", 0)) == len(batch.units):
                completed_groups.setdefault(batch.pool, set()).update(group_ids)
        else:
            entry["requested_unique"] = int(entry["requested_unique"]) + len(batch.units)
        entry["completed_unique"] = int(entry["completed_unique"]) + int(
            row.get("completed_units", 0)
        )
        entry["failed_unique"] = int(entry["failed_unique"]) + int(
            row.get("failed_units", len(batch.units))
        )
        entry["fallback_unique"] = int(entry["fallback_unique"]) + int(
            row.get("fallback_units", 0)
        )
        for unit in batch.units:
            context = unit.get("context") if isinstance(unit, Mapping) else None
            if not isinstance(context, Mapping):
                continue
            for field, target in (
                ("artifact_position", "position_distribution"),
                ("prompt_group_id", "group_distribution"),
            ):
                value = context.get(field)
                if value is None:
                    continue
                key = json.dumps(value, sort_keys=True) if isinstance(value, Mapping) else str(value)
                distribution = entry[target]
                distribution[key] = int(distribution.get(key, 0)) + 1
    for entry in pools.values():
        pool_id = str(entry.get("pool_id") or "")
        if entry.get("coverage_unit") == "groups":
            entry["requested_unique"] = len(requested_groups.get(pool_id, set()))
            entry["completed_unique"] = len(completed_groups.get(pool_id, set()))
        eligible = int(entry.get("eligible_unique", 0))
        entry["actual_fraction"] = (
            int(entry["requested_unique"]) / eligible if eligible else 0.0
        )
    return {
        "schema_version": "ade.analysis_review_coverage.v1",
        "command_id": command.command_id,
        "logical_command_id": command.logical_command_id,
        "attempt_id": command.attempt_id,
        "attempt_index": command.attempt_index,
        "scope": {
            "run_id": command.run_id,
            "coordinator_id": command.coordinator_id,
            "plan_id": command.plan_id,
            "trial_id": command.trial_id,
        },
        "subject_ref": (
            f"{command.run_id}/{command.coordinator_id}/"
            f"{command.plan_id}/{command.trial_id}"
        ),
        "status": "accepted_degraded" if degraded else "accepted",
        "passed": True,
        "pools": pools,
        "usage": packet.get("usage", {}),
    }


def validate_review_packet(
    command: ReviewCommand,
    packet: Mapping[str, object],
) -> None:
    expected = {
        "schema_version": "ade.analysis_review_packet.v1",
        "command_id": command.command_id,
        "logical_command_id": command.logical_command_id,
        "attempt_id": command.attempt_id,
        "attempt_index": command.attempt_index,
        "run_id": command.run_id,
        "coordinator_id": command.coordinator_id,
        "plan_id": command.plan_id,
        "trial_id": command.trial_id,
        "scope": {
            "run_id": command.run_id,
            "coordinator_id": command.coordinator_id,
            "plan_id": command.plan_id,
            "trial_id": command.trial_id,
        },
        "subject_ref": (
            f"{command.run_id}/{command.coordinator_id}/"
            f"{command.plan_id}/{command.trial_id}"
        ),
    }
    if any(packet.get(key) != value for key, value in expected.items()):
        raise ValueError("Review packet identity is invalid")
    status = packet.get("status")
    if status not in {"complete", "partial", "unavailable"}:
        raise ValueError("Review packet status is invalid")
    if not isinstance(packet.get("usage"), Mapping):
        raise ValueError("Review packet usage is invalid")
    batches = packet.get("batches")
    if not isinstance(batches, list):
        raise ValueError("Review packet batches are invalid")
    if status == "unavailable" and not batches:
        return
    expected_batches = {batch.batch_id: batch for batch in command.batches}
    seen: set[str] = set()
    for item in batches:
        if not isinstance(item, Mapping):
            raise ValueError("Review packet batch must be an object")
        batch_id = item.get("batch_id")
        if not isinstance(batch_id, str) or batch_id in seen:
            raise ValueError("Review packet batch identity is invalid")
        seen.add(batch_id)
        try:
            batch = expected_batches[batch_id]
        except KeyError as error:
            raise ValueError("Review packet contains an unknown batch") from error
        requested = item.get("requested_units")
        completed = item.get("completed_units")
        failed = item.get("failed_units")
        fallback = item.get("fallback_units", 0)
        if (
            item.get("pool") != batch.pool
            or item.get("investigation_purpose")
            != batch.investigation_purpose
            or requested != len(batch.units)
            or isinstance(completed, bool)
            or not isinstance(completed, int)
            or isinstance(failed, bool)
            or not isinstance(failed, int)
            or completed < 0
            or failed < 0
            or isinstance(fallback, bool)
            or not isinstance(fallback, int)
            or fallback < 0
            or fallback > completed
            or completed + failed != requested
        ):
            raise ValueError("Review packet batch coverage is invalid")
    if seen != set(expected_batches):
        raise ValueError("Review packet does not cover every frozen batch")


def _inventory(
    attempt: Path,
    manifest: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> tuple[
    dict[str, dict[tuple[str, str], dict[str, Any]]],
    dict[str, dict[str, Any]],
]:
    if catalog.get("schema_version") not in {
        "ade.analyzer_evidence_catalog.v1",
        "ade.analyzer_evidence_catalog.v2",
    }:
        raise ValueError("Analyzer evidence catalog schema is invalid")
    if catalog.get("source_manifest_binding") != manifest.get("source_manifest_binding"):
        raise ValueError("Analyzer catalog source manifest binding is invalid")
    manifest_artifacts = {
        str(item["id"]): item
        for item in manifest.get("artifacts", ())
        if isinstance(item, Mapping) and item.get("id")
    }
    pools: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    requirements: dict[str, dict[str, Any]] = {}
    required_line_indexes: dict[str, set[int]] = {}
    for raw_pool in catalog.get("pools", ()):
        if not isinstance(raw_pool, Mapping):
            continue
        for entry in raw_pool.get("records", ()):
            if not isinstance(entry, Mapping):
                continue
            artifact_id = str(entry.get("source_artifact_id") or "")
            line_index = int(entry.get("line_index", -1))
            if artifact_id and line_index >= 0:
                required_line_indexes.setdefault(artifact_id, set()).add(line_index)
    record_cache: dict[str, dict[int, dict[str, Any]]] = {}
    for raw_pool in catalog.get("pools", ()):
        if not isinstance(raw_pool, Mapping):
            raise ValueError("Analyzer evidence catalog pools must be objects")
        pool_id = str(raw_pool.get("pool_id") or "")
        if not pool_id or pool_id in pools:
            raise ValueError("Analyzer evidence catalog pool IDs are invalid")
        review_fields = raw_pool.get("review_fields")
        if not isinstance(review_fields, Mapping):
            raise ValueError(f"review field binding is missing for {pool_id}")
        artifact_paths = {
            str(item["artifact_id"]): str(item["path"])
            for item in raw_pool.get("artifacts", ())
            if isinstance(item, Mapping) and item.get("artifact_id") and item.get("path")
        }
        inventory: dict[tuple[str, str], dict[str, Any]] = {}
        for entry in raw_pool.get("records", ()):
            if not isinstance(entry, Mapping):
                raise ValueError(f"catalog records are invalid for {pool_id}")
            artifact_id = str(entry.get("source_artifact_id") or "")
            record_id = str(entry.get("source_record_id") or "")
            identity = (artifact_id, record_id)
            if not all(identity) or identity in inventory or artifact_id not in artifact_paths:
                raise ValueError(f"catalog identity is invalid for {pool_id}: {identity}")
            artifact = manifest_artifacts.get(artifact_id)
            if artifact is None or artifact.get("path") != artifact_paths[artifact_id]:
                raise ValueError(f"catalog artifact binding is invalid: {artifact_id}")
            path = attempt / "input" / artifact_paths[artifact_id]
            if artifact_id not in record_cache:
                if artifact.get("size_bytes") != path.stat().st_size:
                    raise ValueError(f"artifact binding is invalid: {artifact_id}")
                record_cache[artifact_id] = _stream_records(
                    path,
                    required_line_indexes.get(artifact_id, set()),
                )
            line_index = int(entry.get("line_index", -1))
            records = record_cache[artifact_id]
            if line_index not in records:
                raise ValueError(f"catalog locator is invalid for {identity}")
            record = records[line_index]
            if not isinstance(record, Mapping) or str(record.get("record_id")) != record_id:
                raise ValueError(f"catalog record binding is invalid for {identity}")
            question = _field_value(record, str(review_fields.get("question") or ""))
            response = _field_value(record, str(review_fields.get("response") or ""))
            if not isinstance(question, str) or not question.strip() or not isinstance(response, str) or not response.strip():
                raise ValueError(f"catalog review fields are invalid for {identity}")
            group_id = entry.get("group_id")
            context = dict(entry.get("context") or {})
            context.update(
                {
                    "artifact_position": record.get("artifact_position") or entry.get("position"),
                    "prompt_group_id": group_id,
                    "response_index": entry.get("response_index"),
                    "group_size": entry.get("group_size"),
                }
            )
            inventory[identity] = {
                "unit_id": f"{artifact_id}::{record_id}",
                "source_artifact_id": artifact_id,
                "source_record_id": record_id,
                "locator": {
                    "artifact_path": artifact_paths[artifact_id],
                    "line_index": line_index,
                },
                "question": question,
                "response": response,
                "context": context,
            }
        eligible = int(raw_pool.get("eligible_record_count", -1))
        if eligible != len(inventory):
            raise ValueError(f"catalog eligible count is invalid for {pool_id}")
        coverage = raw_pool.get("coverage")
        if not isinstance(coverage, Mapping) or coverage.get("mode") not in {"all", "fraction"}:
            raise ValueError(f"catalog coverage is invalid for {pool_id}")
        fraction = coverage.get("fraction")
        coverage_unit = str(coverage.get("unit") or "records")
        if coverage_unit not in {"records", "groups"}:
            raise ValueError(f"catalog coverage unit is invalid for {pool_id}")
        eligible_units = (
            int(raw_pool.get("eligible_group_count", -1))
            if coverage_unit == "groups"
            else eligible
        )
        if coverage_unit == "groups":
            raw_group_ids = [
                unit.get("context", {}).get("prompt_group_id")
                for unit in inventory.values()
            ]
            observed_groups = {str(value) for value in raw_group_ids if value is not None}
            if any(value is None for value in raw_group_ids) or eligible_units != len(observed_groups):
                raise ValueError(f"catalog eligible group count is invalid for {pool_id}")
        required = (
            eligible_units
            if coverage["mode"] == "all"
            else math.ceil(eligible_units * float(fraction))
        )
        pools[pool_id] = inventory
        requirements[pool_id] = {
            "pool_id": pool_id,
            "coverage_mode": str(coverage["mode"]),
            "coverage_fraction": fraction,
            "eligible_unique": eligible_units,
            "required_unique": required,
            **({"coverage_unit": "groups"} if coverage_unit == "groups" else {}),
        }
    return pools, requirements


def _stream_records(
    path: Path,
    required_line_indexes: set[int],
) -> dict[int, dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else Path.open
    records: dict[int, dict[str, Any]] = {}
    record_index = 0
    try:
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if record_index in required_line_indexes:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("behavior record must be an object")
                    records[record_index] = record
                record_index += 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"behavior artifact is invalid: {path.name}") from error
    if set(records) != required_line_indexes:
        raise ValueError(f"behavior catalog locator is invalid: {path.name}")
    return records


def _expand_selection(selection, inventory):
    source_ids = set(selection.source_artifact_ids)
    known_artifacts = {identity[0] for identity in inventory}
    unknown_artifacts = sorted(source_ids - known_artifacts)
    if unknown_artifacts:
        raise ValueError(f"selection contains unknown source artifacts: {unknown_artifacts}")
    candidates = [
        (identity, unit)
        for identity, unit in inventory.items()
        if not source_ids or identity[0] in source_ids
    ]
    if selection.mode == "records":
        requested = set(selection.record_ids)
        candidates = [item for item in candidates if item[0][1] in requested]
        missing = sorted(requested - {identity[1] for identity, _ in candidates})
        if missing:
            raise ValueError(f"selection contains unknown record IDs: {missing}")
    elif selection.mode == "groups":
        requested = set(selection.group_ids)
        candidates = [
            item
            for item in candidates
            if str(item[1].get("context", {}).get("prompt_group_id")) in requested
        ]
        found = {
            str(unit.get("context", {}).get("prompt_group_id"))
            for _, unit in candidates
        }
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"selection contains unknown group IDs: {missing}")
    return candidates


def _field_value(record: Mapping[str, Any], path: str) -> Any:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _review_inputs(record: Mapping[str, Any]) -> tuple[str, str] | None:
    generation = record.get("generation")
    if isinstance(generation, Mapping):
        prompt = record.get("prompt")
        question = str(prompt.get("text") or "") if isinstance(prompt, Mapping) else ""
        response = str(generation.get("raw_response") or "")
        return (question, response) if question.strip() and response.strip() else None
    question = str(record.get("question") or "")
    response = str(record.get("response") or "")
    if question.strip() and response.strip():
        return question, response
    instruction = str(record.get("instruction") or "").strip()
    input_text = str(record.get("input") or "").strip()
    response = str(record.get("output") or record.get("response") or "").strip()
    question = f"{instruction}\n\nInput:\n{input_text}" if instruction and input_text else instruction
    return (question, response) if question and response else None
