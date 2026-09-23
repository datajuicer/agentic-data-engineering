"""Load the immutable p000 response suite used for reward compliance."""

from __future__ import annotations

import gzip
import copy
import hashlib
import importlib.util
import asyncio
import inspect
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Any, Mapping
from collections import defaultdict, deque

from ade.tasks.reward_design.rewards.replay import select_baseline_position
from ade.tasks.reward_design.rewards.replay import replay_reward_pair
from ade.tasks.reward_design.rewards.contracts import validate_reward_result
from ade.engine.judge_dispatcher_impl import RubricRowUnavailable
from ade.tasks.reward_design.process_evidence import (
    available_process_evidence,
    not_configured_process_evidence,
    unavailable_process_evidence,
)


_STEP = re.compile(r"rl-step-(\d+)\.jsonl\.gz$")
_MAX_COMPLIANCE_RECORDS = 256
_COMPLIANCE_PROMPT_GROUPS = 32


class DirectReferenceRolloutUnavailable(ValueError):
    pass


def direct_reference_trial(state: object, artifact_ref_id: str):
    matches = tuple(
        item
        for item in getattr(state, "trials", ())
        if item.artifact_ref_id == artifact_ref_id
    )
    return matches[0] if len(matches) == 1 else None


def load_published_baseline_records(
    manifest: dict[str, Any],
    package: Mapping[str, bytes],
    *,
    durable_artifacts_root: Path,
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    if manifest.get("schema_version") != "ade.trial_artifacts.v1":
        raise ValueError("baseline compliance requires a published Trial manifest")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("baseline compliance Trial artifacts are unavailable")
    candidates: list[tuple[int, dict[str, Any], PurePosixPath]] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or artifact.get("kind") != "training_rollout":
            continue
        relative = PurePosixPath(str(artifact.get("path") or ""))
        match = _STEP.search(relative.as_posix())
        if relative.is_absolute() or ".." in relative.parts or match is None:
            raise ValueError("baseline compliance rollout path is invalid")
        step = int(match.group(1))
        candidates.append((step, artifact, relative))
    if not candidates:
        raise DirectReferenceRolloutUnavailable(
            "direct reference Trial has no published training rollout"
        )
    group_size, prompt_groups = _compliance_dimensions(package, artifacts)
    records_by_step: dict[int, tuple[dict[str, object], ...]] = {}
    source_metadata: dict[int, tuple[dict[str, Any], str, int]] = {}
    for step, artifact, _relative in candidates:
        if step in records_by_step:
            raise ValueError(
                "baseline compliance requires one rollout artifact per position"
            )
        rollout_records, digest, size_bytes = _durable_artifact_records(
            artifact,
            durable_artifacts_root,
        )
        records_by_step[step] = rollout_records
        source_metadata[step] = (artifact, digest, size_bytes)
    try:
        selected_step, records, selection = select_baseline_position(
            records_by_step,
            group_size=group_size,
            prompt_groups=prompt_groups,
        )
    except ValueError as error:
        if "complete rollout groups" not in str(error):
            raise
        raise DirectReferenceRolloutUnavailable(str(error)) from error
    artifact, digest, size_bytes = source_metadata[selected_step]
    encoded = encode_records(records)
    return records, {
        "schema_version": "1",
        "purpose": "reward_function_compliance",
        "record_count": len(records),
        "group_size": group_size,
        "prompt_groups": prompt_groups,
        "baseline_step": selected_step,
        "selection": selection,
        "steps": {str(selected_step): len(records)},
        "records_sha256": hashlib.sha256(encoded).hexdigest(),
        "sources": [
            {
                "artifact_id": str(artifact.get("id") or ""),
                "step": selected_step,
                "sha256": digest,
                "size_bytes": size_bytes,
            }
        ],
    }


def _durable_artifact_records(
    artifact: Mapping[str, Any],
    durable_artifacts_root: Path,
) -> tuple[tuple[dict[str, object], ...], str, int]:
    storage = artifact.get("storage")
    prefix = "engine://trial-artifacts/"
    if not isinstance(storage, Mapping) or storage.get("mode") != "durable_reference":
        raise ValueError("baseline compliance rollout is not durably referenced")
    uri = str(storage.get("uri") or "")
    relative = PurePosixPath(uri.removeprefix(prefix))
    if (
        not uri.startswith(prefix)
        or relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) < 6
    ):
        raise ValueError("baseline compliance rollout reference is invalid")
    root = durable_artifacts_root.resolve()
    source = root.joinpath(*relative.parts)
    expected_size = storage.get("size_bytes")
    digest = storage.get("sha256")
    if (
        not source.is_file()
        or source.is_symlink()
        or not source.resolve().is_relative_to(root)
        or type(expected_size) is not int
        or source.stat().st_size != expected_size
        or not isinstance(digest, str)
        or artifact.get("size_bytes") != expected_size
        or artifact.get("sha256") != digest
    ):
        raise ValueError("baseline compliance rollout binding is invalid")
    try:
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            records = tuple(json.loads(line) for line in handle if line.strip())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("baseline compliance rollout is invalid JSONL") from error
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("baseline compliance rollout records are invalid")
    return records, digest, expected_size


def _compliance_dimensions(
    package: Mapping[str, bytes],
    artifacts: list[object],
) -> tuple[int, int]:
    matches = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, dict)
        and artifact.get("kind") == "reward_runtime_config"
    ]
    if len(matches) != 1:
        raise ValueError("baseline compliance requires one reward runtime config")
    artifact = matches[0]
    relative = PurePosixPath(str(artifact.get("path") or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("baseline compliance runtime config path is invalid")
    content = _package_artifact(package, artifact)
    try:
        runtime = json.loads(content)
        rft = runtime["rft"]
        group_size = rft["rollout_n"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("baseline compliance runtime config is invalid") from error
    if (
        type(group_size) is not int
        or group_size < 1
    ):
        raise ValueError("baseline compliance runtime dimensions are invalid")
    prompt_groups = _COMPLIANCE_PROMPT_GROUPS
    if prompt_groups * group_size > _MAX_COMPLIANCE_RECORDS:
        raise ValueError("rollout group exceeds reward compliance record limit")
    return group_size, prompt_groups


def _package_artifact(
    package: Mapping[str, bytes], artifact: Mapping[str, Any]
) -> bytes:
    relative = PurePosixPath(str(artifact.get("path") or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("baseline compliance artifact path is invalid")
    try:
        content = package[relative.as_posix()]
    except KeyError as error:
        raise ValueError("baseline compliance artifact is unavailable") from error
    if (
        artifact.get("size_bytes") != len(content)
        or artifact.get("sha256") != hashlib.sha256(content).hexdigest()
    ):
        raise ValueError("baseline compliance artifact binding is invalid")
    return content


def encode_records(records: tuple[dict[str, object], ...]) -> bytes:
    return b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for record in records
    )


def validate_persisted_judge_batch(
    *,
    workspace_root: Path,
    records: tuple[dict[str, object], ...],
    records_content: bytes,
    rubric: str,
    results_content: bytes,
    manifest: dict[str, object],
) -> tuple[dict[str, object], ...]:
    try:
        rows = tuple(
            json.loads(line) for line in results_content.splitlines() if line
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("persisted LLM Judge results are invalid JSONL") from error
    if len(rows) != len(records) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("persisted LLM Judge result count mismatch")
    completed = sum(row.get("status") == "completed" for row in rows)
    unavailable = sum(row.get("status") == "unavailable" for row in rows)
    results_digest = hashlib.sha256(results_content).hexdigest()
    rubric_digest = hashlib.sha256(rubric.strip().encode("utf-8")).hexdigest()
    required_manifest = {
        "schema_version": "1",
        "protocol": "ade.llm_judge.compliance_batch.v1",
        "requested_units": len(records),
        "completed_units": completed,
        "unavailable_units": unavailable,
        "records_sha256": hashlib.sha256(records_content).hexdigest(),
        "rubric_sha256": rubric_digest,
        "results_sha256": results_digest,
        "results_size_bytes": len(results_content),
    }
    if completed + unavailable != len(records) or any(
        manifest.get(field) != expected
        for field, expected in required_manifest.items()
    ):
        raise ValueError("persisted LLM Judge manifest identity mismatch")
    model = manifest.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("persisted LLM Judge manifest model is invalid")
    workspace = workspace_root.resolve()
    for record, row in zip(records, rows):
        prompt = record.get("prompt")
        generation = record.get("generation")
        if not isinstance(prompt, dict) or not isinstance(generation, dict):
            raise ValueError("persisted LLM Judge source record is invalid")
        question = prompt.get("text")
        response = generation.get("raw_response")
        if not isinstance(question, str) or not isinstance(response, str):
            raise ValueError("persisted LLM Judge source text is invalid")
        expected_row = {
            "record_id": record.get("record_id"),
            "model": model,
            "question_prompt_sha256": hashlib.sha256(
                question.encode("utf-8")
            ).hexdigest(),
            "response_content_sha256": hashlib.sha256(
                response.encode("utf-8")
            ).hexdigest(),
            "rubric_sha256": rubric_digest,
        }
        if any(row.get(field) != expected for field, expected in expected_row.items()):
            raise ValueError("persisted LLM Judge row identity mismatch")
        audit_relative = row.get("audit_path")
        if not isinstance(audit_relative, str):
            raise ValueError("persisted LLM Judge audit path is invalid")
        audit_path = (workspace / audit_relative).resolve()
        if not audit_path.is_relative_to(workspace) or not audit_path.is_file():
            raise ValueError("persisted LLM Judge audit is unavailable")
        audit_content = audit_path.read_bytes()
        if (
            row.get("audit_sha256") != hashlib.sha256(audit_content).hexdigest()
            or row.get("audit_size_bytes") != len(audit_content)
        ):
            raise ValueError("persisted LLM Judge audit digest mismatch")
        try:
            audit = json.loads(audit_content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("persisted LLM Judge audit is invalid") from error
        request = audit.get("request") if isinstance(audit, dict) else None
        result = audit.get("result") if isinstance(audit, dict) else None
        if (
            audit.get("schema_version") != "1"
            or audit.get("protocol") != "ade.llm_judge.v1"
            or audit.get("request_id") != row.get("request_id")
            or audit.get("model") != model
            or not isinstance(request, dict)
            or request.get("question_prompt") != question
            or request.get("response_content") != response
            or request.get("rubric") != rubric.strip()
            or not isinstance(result, dict)
            or result.get("status") != row.get("status")
            or result.get("score") != row.get("score")
        ):
            raise ValueError("persisted LLM Judge audit identity mismatch")
        provider_response = result.get("provider_response")
        if row.get("status") == "completed" and (
            not isinstance(provider_response, dict)
            or not all(
                field in provider_response
                for field in ("reasoning_content", "content", "raw_response")
            )
        ):
            raise ValueError("persisted LLM Judge provider response is missing")
    return rows


def run_reward_compliance(
    *,
    reference_source: bytes,
    candidate_source: bytes,
    records: tuple[dict[str, object], ...],
    judge_results: tuple[dict[str, object], ...],
) -> dict[str, object]:
    if len(judge_results) != len(records):
        raise ValueError("reward compliance Judge result count mismatch")
    judge_results_by_source = defaultdict(list)
    for record, result in zip(records, judge_results):
        if result.get("record_id") != record.get("record_id"):
            raise ValueError("reward compliance Judge result identity mismatch")
        if result.get("status") not in {
            "available",
            "unavailable",
            "not_configured",
        }:
            raise ValueError("reward compliance Judge result status is invalid")
        dimensions = result.get("dimensions")
        if result.get("status") == "available" and (
            not isinstance(dimensions, dict)
        ):
            raise ValueError("reward compliance Judge dimensions are invalid")
        prompt = record.get("prompt")
        generation = record.get("generation")
        if not isinstance(prompt, dict) or not isinstance(generation, dict):
            raise ValueError("reward compliance Judge source record is invalid")
        question = prompt.get("text")
        response = generation.get("raw_response")
        if not isinstance(question, str) or not isinstance(response, str):
            raise ValueError("reward compliance Judge source text is invalid")
        judge_results_by_source[
            (
                hashlib.sha256(question.encode("utf-8")).hexdigest(),
                hashlib.sha256(response.encode("utf-8")).hexdigest(),
            )
        ].append(result)
    with tempfile.TemporaryDirectory(prefix="ade-reward-compliance-") as temporary:
        root = Path(temporary) / "bindings" / "reward" / "candidate" / "module"
        root.mkdir(parents=True)
        reference = _load_reward(root / "reference_reward.py", reference_source)
        candidate = _load_reward(root / "candidate_reward.py", candidate_source)
        reference_score = getattr(reference, "compute_score", None)
        reference_group_credit = getattr(reference, "assign_group_credit", None)
        candidate_score = getattr(candidate, "compute_score", None)
        fallback_score = getattr(candidate, "compute_fallback_score", None)
        group_credit = getattr(candidate, "assign_group_credit", None)
        if not all(callable(item) for item in (reference_score, candidate_score, fallback_score)):
            raise ValueError("reward compliance entrypoints are unavailable")
        process_bank_sensitivity = None
        if group_credit is not None:
            from ade.tasks.reward_design.group_credit import (
                admit_group_credit_function,
            )

            process_bank_sensitivity = admit_group_credit_function(
                group_credit,
                observe_process_bank=any(
                    result.get("status") != "not_configured"
                    for result in judge_results
                ),
            )

        def judge_capability():
            remaining = {
                source: deque(results)
                for source, results in judge_results_by_source.items()
            }

            async def llm_judge(
                question_prompt: str,
                response_content: str,
            ) -> dict[str, object]:
                source = (
                    hashlib.sha256(question_prompt.encode("utf-8")).hexdigest(),
                    hashlib.sha256(response_content.encode("utf-8")).hexdigest(),
                )
                matching_results = remaining.get(source)
                if not matching_results:
                    raise ValueError(
                        "reward compliance Judge source identity mismatch"
                    )
                result = matching_results.popleft()
                expected = {
                    "question_prompt_sha256": hashlib.sha256(
                        question_prompt.encode("utf-8")
                    ).hexdigest(),
                    "response_content_sha256": hashlib.sha256(
                        response_content.encode("utf-8")
                    ).hexdigest(),
                }
                for field, digest in expected.items():
                    if field in result and result[field] != digest:
                        raise ValueError(
                            f"reward compliance Judge {field} mismatch"
                        )
                if result["status"] == "unavailable":
                    raise RubricRowUnavailable(
                        "persisted Rubric Job row is unavailable"
                    )
                if result["status"] == "not_configured":
                    raise ValueError(
                        "reward artifact called Judge while process evidence is not configured"
                    )
                return available_process_evidence(
                    {
                        "status": "completed",
                        "scores_by_dimension": result["dimensions"],
                    }
                )

            return llm_judge

        reference_judge = judge_capability()
        candidate_judge = judge_capability()
        reference_score.__globals__["llm_judge"] = reference_judge

        candidate_for_replay = candidate_score
        engine_acquires_judge = group_credit is not None and any(
            result.get("status") != "not_configured"
            for result in judge_results
        )
        if engine_acquires_judge:
            async def candidate_with_engine_evidence(**kwargs: object) -> object:
                await candidate_judge(
                    str(kwargs["question_prompt"]),
                    str(kwargs["response_content"]),
                )
                value = candidate_score(**kwargs)
                return await value if inspect.isawaitable(value) else value

            candidate_for_replay = candidate_with_engine_evidence
        elif group_credit is None:
            candidate_score.__globals__["llm_judge"] = candidate_judge
        replay_records = []
        for record, judge_result in zip(records, judge_results, strict=True):
            replay_record = copy.deepcopy(record)
            details = dict(replay_record.get("details") or {})
            if judge_result["status"] == "available":
                details["process_evidence"] = available_process_evidence(
                    {
                        "status": "completed",
                        "scores_by_dimension": judge_result["dimensions"],
                    }
                )
            elif judge_result["status"] == "unavailable":
                details["process_evidence"] = unavailable_process_evidence(
                    str(judge_result.get("reason") or "persisted row unavailable")
                )
            else:
                details["process_evidence"] = not_configured_process_evidence()
            replay_record["details"] = details
            replay_records.append(replay_record)
        replay = asyncio.run(
            replay_reward_pair(
                reference_score,
                candidate_for_replay,
                fallback_score,
                replay_records,
                max_concurrency=1,
                group_credit_enabled=group_credit is not None,
                reference_group_credit=(
                    reference_group_credit
                    if callable(reference_group_credit)
                    else None
                ),
                candidate_group_credit=group_credit,
            )
        )
        replay["judge_evidence_statuses"] = sorted(
            {str(result["status"]) for result in judge_results}
        )
        replay["group_credit_enabled"] = group_credit is not None
        if process_bank_sensitivity is not None:
            replay["process_bank_sensitivity"] = process_bank_sensitivity
        return replay


def _load_reward(path: Path, source: bytes):
    path.write_bytes(source)
    module_name = f"_ade_reward_compliance_{hashlib.sha256(source).hexdigest()[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError("reward compliance could not load reward module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module
