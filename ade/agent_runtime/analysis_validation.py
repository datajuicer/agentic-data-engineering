"""Controller-side admission for staged Analyzer Review evidence."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from ade.core.validation import DeliveryViolation, ValidationReport
from ade.review_labor.usage import aggregate_usage
from ade.tasks.contracts import AnalysisReport


class StagedReviewContractError(ValueError):
    """A Harness-owned staged Review input cannot be repaired by the Agent."""


def derive_analysis_sidecars(
    workspace: Path,
    *,
    analysis_content: bytes,
    findings_content: bytes,
) -> tuple[bytes, bytes]:
    """Derive semantic citations from the immutable staged Review packet."""

    packet, coverage = _require_staged_review_inputs(workspace)
    _validate_staged_coverage_catalog(workspace, coverage)
    text = analysis_content.decode("utf-8") + "\n" + findings_content.decode("utf-8")
    review_ids = sorted(set(re.findall(r"\[\[review:([^\]\s]+)\]\]", text)))
    known_review_ids = {
        str(batch["review_id"])
        for batch in packet.get("batches", ())
        if isinstance(batch, dict) and batch.get("review_id")
    }
    unknown = sorted(set(review_ids) - known_review_ids)
    if unknown:
        raise ValueError(f"unknown review IDs: {unknown}")
    artifact_ids = sorted(set(re.findall(r"\[\[artifact:([^\]\s]+)\]\]", text)))
    evidence = {
        "schema_version": "ade.analysis_evidence.v2",
        "command_id": packet["command_id"],
        "artifact_ids": artifact_ids,
        "review_ids": review_ids,
    }
    return (
        (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode(),
        (json.dumps(coverage, indent=2, sort_keys=True) + "\n").encode(),
    )


def validate_analysis_delivery(
    output: AnalysisReport,
    workspace: Path,
    *,
    task_id: str,
    authorized_ids: tuple[str, ...],
) -> ValidationReport:
    del task_id
    violations: list[DeliveryViolation] = []
    try:
        packet, expected_coverage = _require_staged_review_inputs(workspace)
        _validate_staged_coverage_catalog(workspace, expected_coverage)
        evidence = json.loads(output.evidence_content)
        coverage = json.loads(output.review_coverage_content)
        if evidence.get("schema_version") != "ade.analysis_evidence.v2":
            raise ValueError("staged analysis evidence schema is invalid")
        if evidence.get("command_id") != packet.get("command_id"):
            raise ValueError("staged analysis command binding is invalid")
        if coverage != expected_coverage or coverage.get("passed") is not True:
            raise ValueError("staged Review coverage binding is invalid")
        known_review_ids = {
            str(batch["review_id"])
            for batch in packet.get("batches", ())
            if isinstance(batch, dict) and batch.get("review_id")
        }
        if not set(evidence.get("review_ids", ())).issubset(known_review_ids):
            raise ValueError("staged analysis cites an unknown review")
        if not set(evidence.get("artifact_ids", ())).issubset(authorized_ids):
            raise ValueError("staged analysis cites an unauthorized artifact")
    except StagedReviewContractError as error:
        violations.append(
            DeliveryViolation(
                "invalid_staged_review_input",
                str(error),
                path="input/review/packet.json",
                repairable=False,
            )
        )
    except (KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        violations.extend(
            DeliveryViolation(
                "invalid_analysis_evidence",
                str(error),
                path=path,
                repairable=True,
            )
            for path in ("analysis.md", "findings.md")
        )
    return ValidationReport(tuple(violations))


def _require_staged_review_inputs(
    workspace: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    staged = _load_staged_review_inputs(workspace)
    if staged is None:
        raise StagedReviewContractError("staged Review packet and coverage are required")
    return staged


def _load_staged_review_inputs(
    workspace: Path,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    packet_path = workspace / "input" / "review" / "packet.json"
    coverage_path = workspace / "input" / "review" / "coverage.json"
    if not packet_path.is_file() and not coverage_path.is_file():
        return None
    if not packet_path.is_file() or not coverage_path.is_file():
        raise StagedReviewContractError("staged Review packet and coverage must both exist")
    try:
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        manifest = json.loads(
            (workspace / "input" / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StagedReviewContractError(f"staged Review input is unreadable: {error}") from error
    if not isinstance(packet, dict) or not isinstance(coverage, dict):
        raise StagedReviewContractError("staged Review inputs must be JSON objects")
    expected_packet = {
        "schema_version": "ade.analysis_review_packet.v1",
        "run_id": manifest.get("run_id"),
        "trial_id": manifest.get("subject_id"),
    }
    if any(packet.get(key) != value for key, value in expected_packet.items()):
        raise StagedReviewContractError("staged Review packet identity is invalid")
    for field in ("command_id", "coordinator_id", "plan_id"):
        if not isinstance(packet.get(field), str) or not packet[field]:
            raise StagedReviewContractError(f"staged Review packet {field} is missing")
    if packet.get("status") not in {"complete", "partial", "unavailable"}:
        raise StagedReviewContractError("staged Review packet status is invalid")
    if not isinstance(packet.get("batches"), list) or not isinstance(packet.get("usage"), dict):
        raise StagedReviewContractError("staged Review packet schema is invalid")
    if (
        coverage.get("schema_version") != "ade.analysis_review_coverage.v1"
        or coverage.get("command_id") != packet["command_id"]
        or coverage.get("passed") is not True
        or not isinstance(coverage.get("pools"), dict)
        or not isinstance(coverage.get("usage"), dict)
    ):
        raise StagedReviewContractError("staged Review coverage identity is invalid")
    return packet, coverage


def _validate_staged_coverage_catalog(
    workspace: Path, coverage: Mapping[str, Any]
) -> None:
    try:
        catalog = json.loads(
            (workspace / "input" / "experiment" / "evidence-catalog.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StagedReviewContractError(f"Analyzer evidence catalog is unreadable: {error}") from error
    if catalog.get("schema_version") not in {
        "ade.analyzer_evidence_catalog.v1",
        "ade.analyzer_evidence_catalog.v2",
    }:
        raise StagedReviewContractError("Analyzer evidence catalog schema is invalid")
    observed = coverage.get("pools")
    if not isinstance(observed, Mapping):
        raise StagedReviewContractError("staged Review coverage pools are invalid")
    expected_ids: set[str] = set()
    for pool in catalog.get("pools", ()):
        if not isinstance(pool, Mapping):
            raise StagedReviewContractError("Analyzer evidence catalog pools are invalid")
        pool_id = str(pool.get("pool_id") or "")
        expected_ids.add(pool_id)
        coverage_spec = pool.get("coverage")
        if not isinstance(coverage_spec, Mapping):
            raise StagedReviewContractError(f"catalog coverage is missing for {pool_id}")
        mode = coverage_spec.get("mode")
        fraction = coverage_spec.get("fraction")
        unit = str(coverage_spec.get("unit") or "records")
        eligible = int(
            pool.get("eligible_group_count", -1)
            if unit == "groups"
            else pool.get("eligible_record_count", -1)
        )
        required = eligible if mode == "all" else math.ceil(eligible * float(fraction))
        actual = observed.get(pool_id)
        if (
            not isinstance(actual, Mapping)
            or actual.get("coverage_mode") != mode
            or actual.get("coverage_fraction") != fraction
            or actual.get("coverage_unit", "records") != unit
            or actual.get("eligible_unique") != eligible
            or actual.get("required_unique") != required
            or int(actual.get("requested_unique", -1)) < required
        ):
            raise StagedReviewContractError(
                f"staged Review coverage does not satisfy catalog pool {pool_id}"
            )
    if set(observed) != expected_ids:
        raise StagedReviewContractError("staged Review coverage pool set is invalid")


def observed_analysis_failure(workspace: Path) -> dict[str, Any]:
    manifests = sorted(
        (
            *workspace.glob("scratch/reviews/**/manifest.json"),
            *workspace.glob("harness/prior-reviews/**/manifest.json"),
        ),
        key=str,
    )
    usage_rows: list[Mapping[str, Any]] = []
    review_ids: list[str] = []
    for path in manifests:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if value.get("schema_version") == "3":
            review_ids.append(str(value.get("review_id") or ""))
            usage_rows.extend(
                item for item in value.get("unit_usage", ()) if isinstance(item, dict)
            )
    return {
        "review_ids": sorted(item for item in review_ids if item),
        "usage_observed": aggregate_usage(usage_rows),
    }
