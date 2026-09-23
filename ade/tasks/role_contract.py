"""Shared implementation for task-role Agent contracts."""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Callable, Generic, Mapping, TypeVar

from ade.core.agent import AgentRole
from ade.core.plan import PlanRelationKind
from ade.core.validation import DeliveryViolation, ValidationReport
from ade.tasks.contracts import (
    AgentInputRequest,
    AgentRoleOutput,
    AnalysisFinding,
    AnalysisReviewBatch,
    AnalysisReviewPlan,
    AnalysisReviewSelection,
    AnalysisReport,
    ArtifactDelivery,
    PlanSummary,
    PlanningDecision,
    RoleOutputError,
    RunSummary,
    SupportingArtifact,
)

T = TypeVar("T", bound=AgentRoleOutput)
Decoder = Callable[[Path, Mapping[str, object]], T]
Validator = Callable[[T], ValidationReport]

_ACTION_KINDS = {
    AgentRole.COORDINATOR: "coordinate",
    AgentRole.ARTIFACT_BUILDER: "build_artifact",
    AgentRole.ANALYZER: "analyze_trial",
    AgentRole.PLAN_SUMMARIZER: "summarize_plan",
    AgentRole.RUN_SUMMARIZER: "summarize_run",
}
_REQUIRED_CONTEXT_PREFIXES = {
    AgentRole.COORDINATOR: (
        "memory/manifest.json",
        "memory/MEMORY.md",
        "subject/PLAN_CATALOG.md",
        "subject/ranking.json",
        "subject/portfolio-target.json",
    ),
    AgentRole.ARTIFACT_BUILDER: ("snapshots/current-plan/",),
    AgentRole.ANALYZER: ("experiment/manifest.json",),
    AgentRole.PLAN_SUMMARIZER: (
        "memory/MEMORY.md",
        "memory/manifest.json",
        "subject/plan.md",
        "subject/trial/manifest.json",
        "subject/trial/outcome/",
    ),
    AgentRole.RUN_SUMMARIZER: (
        "memory/MEMORY.md",
        "memory/manifest.json",
        "subject/plan-memory/MEMORY.md",
        "subject/plan-memory/manifest.json",
        "subject/PLAN_UPDATE.md",
        "run/manifest.json",
        "run/PLAN_CATALOG.md",
        "run/RANKING.md",
    ),
}


def _canonical_identity(
    scope: object,
    reference: object,
    run_id: str,
) -> bool:
    return (
        isinstance(scope, Mapping)
        and tuple(scope) in {
            ("run_id",),
            ("run_id", "coordinator_id"),
            ("run_id", "coordinator_id", "plan_id"),
            ("run_id", "coordinator_id", "plan_id", "trial_id"),
        }
        and scope.get("run_id") == run_id
        and reference == "/".join(str(scope[name]) for name in scope)
    )


@dataclass(frozen=True)
class BoundRoleContract(Generic[T]):
    task_id: str
    domain: str
    role: AgentRole
    skill_id: str
    output_path: str
    output_kind: str
    decoder: Decoder[T]
    validator: Validator[T]
    auxiliary_outputs: tuple[tuple[str, str], ...] = ()

    @property
    def output_paths(self) -> tuple[str, ...]:
        return (self.output_path, *(path for path, _ in self.auxiliary_outputs))

    def output_paths_for(self, action: Mapping[str, object]) -> tuple[str, ...]:
        del action
        return self.output_paths

    def validate_input(self, request: AgentInputRequest) -> None:
        if request.role is not self.role:
            raise ValueError(f"expected role {self.role}, received {request.role}")
        if request.basis_revision < 0:
            raise ValueError("basis_revision must be non-negative")
        if not request.run_id or not request.subject_id:
            raise ValueError("run_id and subject_id are required")
        if request.task.get("task_id") != self.task_id:
            raise ValueError(f"task payload must identify {self.task_id}")
        if request.task.get("domain") != self.domain:
            raise ValueError(f"task domain must be {self.domain}")
        if request.action.get("kind") != _ACTION_KINDS[self.role]:
            raise ValueError(f"action kind must be {_ACTION_KINDS[self.role]}")
        scope = request.action.get("scope")
        subject_ref = request.action.get("subject_ref")
        if (
            not isinstance(scope, Mapping)
            or tuple(scope) not in {
                ("run_id",),
                ("run_id", "coordinator_id"),
                ("run_id", "coordinator_id", "plan_id"),
                ("run_id", "coordinator_id", "plan_id", "trial_id"),
            }
            or scope.get("run_id") != request.run_id
            or subject_ref
            != "/".join(
                str(scope[name])
                for name in (
                    "run_id",
                    "coordinator_id",
                    "plan_id",
                    "trial_id",
                )
                if name in scope
            )
        ):
            raise ValueError("Agent action requires canonical ScopeKey and SubjectRef")
        owner_scope = request.action.get("owner_scope")
        owner_subject_ref = request.action.get("owner_subject_ref")
        target_scope = request.action.get("target_scope")
        target_subject_ref = request.action.get("target_subject_ref")
        if (
            not _canonical_identity(owner_scope, owner_subject_ref, request.run_id)
            or not _canonical_identity(target_scope, target_subject_ref, request.run_id)
        ):
            raise ValueError(
                "Agent action requires canonical owner and target ScopeKeys/SubjectRefs"
            )
        assert isinstance(owner_scope, Mapping)
        assert isinstance(target_scope, Mapping)
        owner_values = tuple(str(value) for value in owner_scope.values())
        target_values = tuple(str(value) for value in target_scope.values())
        expected_owner_size, expected_target_size = {
            AgentRole.COORDINATOR: (2, 3),
            AgentRole.ARTIFACT_BUILDER: (3, 4),
            AgentRole.ANALYZER: (3, 4),
            AgentRole.PLAN_SUMMARIZER: (3, 4),
            AgentRole.RUN_SUMMARIZER: (1, 4),
        }[self.role]
        scope_values = tuple(str(value) for value in scope.values())
        role_subject_matches = (
            target_values == scope_values
            if self.role in {AgentRole.ARTIFACT_BUILDER, AgentRole.ANALYZER}
            else owner_values == scope_values
        )
        if (
            len(owner_values) != expected_owner_size
            or len(target_values) != expected_target_size
            or target_values[: len(owner_values)] != owner_values
            or not role_subject_matches
        ):
            raise ValueError("Agent action target does not belong to its Session owner")
        if self.role is AgentRole.COORDINATOR:
            target_plan_id = request.action.get("target_plan_id")
            if (
                not isinstance(target_plan_id, str)
                or re.fullmatch(r"p[0-9]{3}", target_plan_id) is None
            ):
                raise ValueError(
                    "Coordinator action requires Runtime-owned target_plan_id"
                )
        if self.role is AgentRole.ARTIFACT_BUILDER:
            sources = request.action.get("source_artifact_ref_ids")
            if (
                not isinstance(sources, list)
                or not sources
                or len(sources) != len(set(sources))
                or any(not non_empty_text(item) for item in sources)
            ):
                raise ValueError(
                    "Artifact Builder action requires unique "
                    "source_artifact_ref_ids"
                )
        paths = [
            *(item.path for item in request.context_files),
            *(item.path for item in request.context_references),
        ]
        if len(paths) != len(set(paths)):
            raise ValueError("context file paths must be unique")
        if self.role is AgentRole.COORDINATOR and not {
            "subject/ranking.json",
            "subject/portfolio-target.json",
        }.issubset(paths):
            raise ValueError(
                "Coordinator context requires frozen ranking and portfolio target"
            )
        for item in request.context_files:
            relative = PurePosixPath(item.path)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or item.path in {"manifest.json", "action.json", "task.json"}
            ):
                raise ValueError(f"unsafe context path: {item.path}")
            if not isinstance(item.content, bytes) or not item.source_ref:
                raise ValueError(
                    f"context file requires materialized bytes and source_ref: {item.path}"
                )
        for item in request.context_references:
            relative = PurePosixPath(item.path)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or item.path in {"manifest.json", "action.json", "task.json"}
                or not item.source_ref.startswith("engine://trial-artifacts/")
                or not re.fullmatch(r"[0-9a-f]{64}", item.sha256)
                or item.size_bytes < 0
            ):
                raise ValueError(f"invalid durable context reference: {item.path}")
        for prefix in _REQUIRED_CONTEXT_PREFIXES[self.role]:
            if not any(path == prefix or path.startswith(prefix) for path in paths):
                raise ValueError(f"{self.role} input requires context under {prefix}")

    def decode_output(self, attempt: Path) -> T:
        return self.decoder(
            attempt / "output" / self.output_path,
            {"path": self.output_path, "kind": self.output_kind},
        )

    def validate_output(self, output: T) -> ValidationReport:
        return self.validator(output)

    def finalize_output(self, attempt: Path, output: T) -> None:
        del output
        artifacts = []
        for path, kind in (
            (self.output_path, self.output_kind),
            *self.auxiliary_outputs,
        ):
            content = (attempt / "output" / path).read_bytes()
            artifacts.append(
                {
                    "path": path,
                    "kind": kind,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size_bytes": len(content),
                }
            )
        manifest = _read_object(attempt / "input" / "manifest.json")
        target = attempt / "_meta" / "delivery.json"
        target.parent.mkdir(exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "producer": "ade-harness",
                    "run_id": manifest["run_id"],
                    "role": manifest["role"],
                    "subject_id": manifest["subject_id"],
                    "basis_revision": manifest["basis_revision"],
                    "artifacts": artifacts,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class StagedAnalyzerContract:
    task_id: str
    domain: str
    role: AgentRole
    skill_id: str
    analysis_validator: Validator[AnalysisReport]
    output_path: str = "analysis.md"
    output_kind: str = "trial_analysis"
    auxiliary_outputs: tuple[tuple[str, str], ...] = (
        ("findings.md", "trial_findings"),
    )

    @property
    def output_paths(self) -> tuple[str, ...]:
        return ("review-plan.json", "analysis.md", "findings.md")

    def output_paths_for(self, action: Mapping[str, object]) -> tuple[str, ...]:
        stage = _analysis_stage(action)
        return (
            ("review-plan.json",)
            if stage == "review_design"
            else ("analysis.md", "findings.md")
        )

    def validate_input(self, request: AgentInputRequest) -> None:
        BoundRoleContract(
            self.task_id,
            self.domain,
            self.role,
            self.skill_id,
            self.output_path,
            self.output_kind,
            decode_analysis_report,
            self.analysis_validator,
            self.auxiliary_outputs,
        ).validate_input(request)
        _analysis_stage(request.action)

    def decode_output(self, attempt: Path) -> AnalysisReviewPlan | AnalysisReport:
        action = _read_object(attempt / "input" / "action.json")
        if _analysis_stage(action) == "review_design":
            return decode_analysis_review_plan(
                attempt / "output" / "review-plan.json",
                {"path": "review-plan.json", "kind": "analysis_review_plan"},
            )
        return decode_analysis_report(
            attempt / "output" / "analysis.md",
            {"path": "analysis.md", "kind": "trial_analysis"},
        )

    def validate_output(
        self, output: AnalysisReviewPlan | AnalysisReport
    ) -> ValidationReport:
        if isinstance(output, AnalysisReviewPlan):
            return ValidationReport()
        return self.analysis_validator(output)

    def finalize_output(
        self,
        attempt: Path,
        output: AnalysisReviewPlan | AnalysisReport,
    ) -> None:
        paths = (
            (("review-plan.json", "analysis_review_plan"),)
            if isinstance(output, AnalysisReviewPlan)
            else (("analysis.md", "trial_analysis"), *self.auxiliary_outputs)
        )
        _finalize_paths(attempt, paths)


def _analysis_stage(action: Mapping[str, object]) -> str:
    stage = action.get("stage")
    if stage not in {"review_design", "synthesis"}:
        raise ValueError("Analyzer action stage must be review_design or synthesis")
    return str(stage)


def decode_analysis_review_plan(
    path: Path,
    declaration: Mapping[str, object],
) -> AnalysisReviewPlan:
    del declaration
    content = path.read_bytes()
    value = _read_object(path)
    try:
        if value.get("schema_version") != "ade.analysis_review_plan.v2":
            raise ValueError(
                "review plan schema_version must be ade.analysis_review_plan.v2"
            )
        hypotheses = _strings(value["hypotheses"], "hypotheses")
        if not hypotheses:
            raise ValueError("hypotheses must be non-empty")
        raw_batches = value["batches"]
        if not isinstance(raw_batches, list):
            raise ValueError("batches must be a list")
        batches = []
        for index, raw in enumerate(raw_batches):
            batch = _mapping(raw, f"batches[{index}]")
            raw_rubrics = batch["rubrics"]
            if not isinstance(raw_rubrics, list) or not raw_rubrics:
                raise ValueError(f"batches[{index}].rubrics must be non-empty")
            selection = _analysis_review_selection(
                _mapping(batch["selection"], f"batches[{index}].selection"),
                index=index,
            )
            rubrics = tuple(
                dict(
                    _mapping(
                        item,
                        f"batches[{index}].rubrics[{rubric_index}]",
                        path=f"review-plan.json#/batches/{index}/rubrics/{rubric_index}",
                    )
                )
                for rubric_index, item in enumerate(raw_rubrics)
            )
            batches.append(
                AnalysisReviewBatch(
                    batch_id=str(batch["batch_id"]),
                    pool=str(batch["pool"]),
                    investigation_purpose=str(batch["investigation_purpose"]),
                    selection=selection,
                    rubrics=rubrics,
                    hypothesis_ref=(
                        str(batch["hypothesis_ref"])
                        if batch.get("hypothesis_ref") is not None
                        else None
                    ),
                )
            )
        return AnalysisReviewPlan(
            schema_version=str(value["schema_version"]),
            hypotheses=hypotheses,
            batches=tuple(batches),
            content=content,
        )
    except KeyError as error:
        field = str(error.args[0])
        raise _output_error(
            "missing_field",
            f"{field} is required",
            f"review-plan.json#/{field}",
        )


def _analysis_review_selection(
    value: Mapping[str, object], *, index: int
) -> AnalysisReviewSelection:
    allowed = {"mode", "source_artifact_ids", "record_ids", "group_ids"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"batches[{index}].selection has unknown fields: {unknown}")
    mode = value.get("mode")
    if mode not in {"all", "records", "groups"}:
        raise ValueError(f"batches[{index}].selection.mode is invalid")
    source_artifact_ids = _strings(
        value.get("source_artifact_ids", []), "source_artifact_ids"
    )
    record_ids = _strings(value.get("record_ids", []), "record_ids")
    group_ids = _strings(value.get("group_ids", []), "group_ids")
    if mode == "all" and (record_ids or group_ids):
        raise ValueError("all selection must not include record_ids or group_ids")
    if mode == "records" and (not record_ids or group_ids):
        raise ValueError("records selection requires only record_ids")
    if mode == "groups" and (not group_ids or record_ids):
        raise ValueError("groups selection requires only group_ids")
    return AnalysisReviewSelection(
        mode=mode,
        source_artifact_ids=source_artifact_ids,
        record_ids=record_ids,
        group_ids=group_ids,
    )


def _finalize_paths(attempt: Path, paths: tuple[tuple[str, str], ...]) -> None:
    artifacts = []
    for path, kind in paths:
        content = (attempt / "output" / path).read_bytes()
        artifacts.append(
            {
                "path": path,
                "kind": kind,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
        )
    manifest = _read_object(attempt / "input" / "manifest.json")
    target = attempt / "_meta" / "delivery.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "producer": "ade-harness",
                "run_id": manifest["run_id"],
                "role": manifest["role"],
                "subject_id": manifest["subject_id"],
                "scope": manifest["scope"],
                "subject_ref": manifest["subject_ref"],
                "basis_revision": manifest["basis_revision"],
                "artifacts": artifacts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def decode_planning_decision(
    path: Path,
    declaration: Mapping[str, object],
) -> PlanningDecision:
    del declaration
    content = path.read_bytes()
    value = _read_object(path)
    manifest = _read_object(path.parents[1] / "input" / "manifest.json")
    action = _read_object(path.parents[1] / "input" / "action.json")
    plan_path = path.with_name("plan.md")
    try:
        plan_content = plan_path.read_bytes()
        plan_text = plan_content.decode("utf-8").strip()
    except (OSError, UnicodeDecodeError) as error:
        raise _output_error("invalid_plan", str(error), "plan.md")
    relation = _mapping(value.get("relation"), "relation")
    try:
        relation_kind = str(relation["kind"])
    except KeyError as error:
        raise _output_error(
            "missing_field",
            "kind is required",
            "decision.json#/relation/kind",
        ) from error
    try:
        related_plan_keys = _strings(
            relation["related_plans"],
            "related_plans",
        )
    except KeyError as error:
        raise _output_error(
            "missing_field",
            "related_plans is required",
            "decision.json#/relation/related_plans",
        ) from error
    try:
        return PlanningDecision(
            schema_version=str(value["schema_version"]),
            decision="open_plan",
            plan_id=str(action["target_plan_id"]),
            basis_revision=_integer(manifest["basis_revision"], "basis_revision"),
            rationale=plan_text,
            hypothesis=plan_text,
            expected_observation=plan_text,
            risk=plan_text,
            design=_mapping(value.get("design"), "design"),
            relation_kind=relation_kind,
            related_plan_keys=related_plan_keys,
            content=content,
            supporting_artifacts=(
                SupportingArtifact(
                    "plan.md", "planning_decision_report", plan_content
                ),
            ),
        )
    except KeyError as error:
        field = str(error.args[0])
        raise _output_error("missing_field", f"{field} is required", f"decision.json#/{field}")


def decode_artifact(path: Path, declaration: Mapping[str, object]) -> ArtifactDelivery:
    content = path.read_bytes()
    payload: Mapping[str, object] | None = None
    if path.suffix == ".json":
        payload = _read_object(path)
    return ArtifactDelivery(
        schema_version="1",
        path=path.name,
        kind=str(declaration["kind"]),
        content=content,
        metadata={
            str(key): value
            for key, value in declaration.items()
            if key not in {"path", "kind"}
        },
        payload=payload,
    )


def decode_executable_artifact(
    path: Path,
    declaration: Mapping[str, object],
) -> ArtifactDelivery:
    output = decode_artifact(path, declaration)
    design_path = path.with_name("design.md")
    try:
        design_content = design_path.read_bytes()
        design_content.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise _output_error("invalid_design", str(error), "design.md")
    if (path.parents[1] / "input" / "reflection").is_dir():
        decisions = [
            line.removeprefix("Reflection decision: ").strip()
            for line in design_content.decode("utf-8").splitlines()
            if line.startswith("Reflection decision: ")
        ]
        if len(decisions) != 1 or decisions[0] not in {"finalize", "revise"}:
            raise _output_error(
                "invalid_reflection_decision",
                "A Builder reflection must choose finalize or revise",
                "design.md",
            )
    return replace(
        output,
        payload=None,
        supporting_artifacts=(
            SupportingArtifact(
                path="design.md",
                kind="design_report",
                content=design_content,
            ),
        ),
    )


def decode_analysis(path: Path, declaration: Mapping[str, object]) -> AnalysisFinding:
    del declaration
    value = _read_object(path)
    try:
        return AnalysisFinding(
            schema_version=str(value["schema_version"]),
            finding=str(value["finding"]),
            evidence_unit_ids=_strings(value["evidence_unit_ids"], "evidence_unit_ids"),
            uncertainty=str(value["uncertainty"]),
            contradictions=_strings(value["contradictions"], "contradictions"),
            recommended_next_checks=_strings(
                value["recommended_next_checks"],
                "recommended_next_checks",
            ),
            content=path.read_bytes(),
        )
    except KeyError as error:
        field = str(error.args[0])
        raise _output_error("missing_field", f"{field} is required", f"analysis.json#/{field}")


def decode_analysis_report(
    path: Path,
    declaration: Mapping[str, object],
) -> AnalysisReport:
    del declaration
    try:
        content = path.read_bytes()
        content.decode("utf-8")
        findings = path.with_name("findings.md").read_bytes()
        findings.decode("utf-8")
        evidence, coverage = _analysis_sidecars(path)
    except (OSError, UnicodeDecodeError) as error:
        raise _output_error("invalid_analysis_report", str(error), path.name)
    except json.JSONDecodeError as error:
        raise _output_error("invalid_analysis_sidecar", str(error), path.name)
    return AnalysisReport(
        content=content,
        findings_content=findings,
        evidence_content=evidence,
        review_coverage_content=coverage,
    )


def decode_plan_summary(path: Path, declaration: Mapping[str, object]) -> PlanSummary:
    del declaration
    return _decode_summary(path, PlanSummary, "## Assessment")


def decode_run_summary(path: Path, declaration: Mapping[str, object]) -> RunSummary:
    del declaration
    return _decode_summary(path, RunSummary, "## Current Research State")


def _decode_summary(path: Path, output_type, assessment_section: str):
    del assessment_section
    try:
        content = path.read_bytes()
        text = content.decode("utf-8")
        manifest = _read_object(path.parents[1] / "input" / "manifest.json")
        sources = _summary_sources(path.parents[1] / "input")
        source_manifest_digests = tuple(
            (str(item["snapshot_id"]), str(item["manifest_sha256"]))
            for item in sources
        )
        value = {
            "schema_version": "1",
            "subject_id": manifest["subject_id"],
            "basis_revision": manifest["basis_revision"],
            "sources": sources,
            "citation_ids": [item["snapshot_id"] for item in sources],
        }
        evidence_content = (json.dumps(value, sort_keys=True) + "\n").encode()
        return output_type(
            schema_version=str(value["schema_version"]),
            subject_id=str(value["subject_id"]),
            basis_revision=_integer(value["basis_revision"], "basis_revision"),
            source_ids=tuple(item[0] for item in source_manifest_digests),
            source_manifest_digests=source_manifest_digests,
            citation_ids=_strings(value["citation_ids"], "citation_ids"),
            assessment=text.strip(),
            content=content,
            evidence_content=evidence_content,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise _output_error("invalid_summary", str(error), path.name)
    except KeyError as error:
        field = str(error.args[0])
        raise _output_error("missing_field", f"{field} is required", f"evidence.json#/{field}")


def _section_body(text: str, section: str) -> str:
    start = text.find(section)
    if start < 0:
        return ""
    start += len(section)
    end = text.find("\n## ", start)
    return text[start : len(text) if end < 0 else end].strip()


def _analysis_sidecars(path: Path) -> tuple[bytes, bytes]:
    from ade.agent_runtime.analysis_validation import (
        StagedReviewContractError,
        derive_analysis_sidecars,
    )

    try:
        return derive_analysis_sidecars(
            path.parents[1],
            analysis_content=path.read_bytes(),
            findings_content=path.with_name("findings.md").read_bytes(),
        )
    except StagedReviewContractError as error:
        raise RoleOutputError(
            DeliveryViolation(
                "invalid_staged_review_input",
                str(error),
                "input/review/packet.json",
                repairable=False,
            )
        ) from error


def _summary_sources(input_dir: Path) -> list[dict[str, str]]:
    manifest = _read_object(input_dir / "manifest.json")
    declarations = manifest.get("inputs")
    if not isinstance(declarations, list):
        raise ValueError("input manifest requires declarations")
    sources_by_id: dict[str, str] = {}
    role = manifest.get("role")
    source_paths = {
        AgentRole.PLAN_SUMMARIZER.value: {
            "memory/manifest.json",
            "subject/plan.md",
            "subject/trial/manifest.json",
        },
        AgentRole.RUN_SUMMARIZER.value: {
            "memory/manifest.json",
            "subject/plan-memory/manifest.json",
            "subject/PLAN_UPDATE.md",
            "run/manifest.json",
        },
    }.get(role, set())
    memory_source = next(
        (
            str(item["source_ref"])
            for item in declarations
            if isinstance(item, dict)
            and item.get("path") == "memory/manifest.json"
            and isinstance(item.get("source_ref"), str)
        ),
        None,
    )
    for item in declarations:
        if (
            isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and item["path"] in source_paths
            and isinstance(item.get("source_ref"), str)
            and isinstance(item.get("sha256"), str)
        ):
            snapshot_id = str(item["source_ref"])
            if (
                item["path"] == "subject/plan.md"
                and snapshot_id == memory_source
            ):
                continue
            digest = str(item["sha256"])
            previous = sources_by_id.setdefault(snapshot_id, digest)
            if previous != digest:
                raise ValueError("summary source identity has conflicting digests")
    if not sources_by_id:
        raise ValueError("summary input has no accepted bound source")
    return [
        {"snapshot_id": snapshot_id, "manifest_sha256": digest}
        for snapshot_id, digest in sorted(sources_by_id.items())
    ]


def _read_object(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _output_error("invalid_output_json", str(error), path.name)
    if not isinstance(value, dict):
        raise _output_error("invalid_output_type", "expected JSON object", path.name)
    return value


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise _output_error("invalid_field", f"{field} must be an integer", f"#/{field}")
    return value


def _mapping(
    value: object,
    field: str,
    *,
    path: str | None = None,
) -> Mapping[str, object]:
    if not isinstance(value, dict):
        code = "missing_design" if field == "design" else "invalid_field"
        raise _output_error(code, f"{field} must be an object", path or f"decision.json#/{field}")
    return value


def _strings(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise _output_error("invalid_field", f"{field} must be a string array", f"#/{field}")
    return tuple(value)


def violation(
    code: str,
    message: str,
    path: str,
    *,
    repairable: bool = True,
) -> ValidationReport:
    return ValidationReport((DeliveryViolation(code, message, path, repairable),))


def planning_base_violations(output: PlanningDecision) -> list[DeliveryViolation]:
    violations: list[DeliveryViolation] = []
    if output.schema_version != "1" or output.decision != "open_plan":
        violations.append(
            repairable_violation(
                "invalid_decision",
                "schema 1 open_plan is required",
                "decision.json",
            )
        )
    if output.basis_revision < 0:
        violations.append(
            repairable_violation(
                "invalid_basis_revision",
                "basis_revision must be non-negative",
                "decision.json#/basis_revision",
            )
        )
    fields = {
        "subject_id": output.plan_id,
        "rationale": output.rationale,
        "hypothesis": output.hypothesis,
        "expected_observation": output.expected_observation,
        "risk": output.risk,
    }
    for key, value in fields.items():
        if not non_empty_text(value):
            violations.append(
                repairable_violation(
                    f"missing_{key}",
                    f"{key} must be non-empty",
                    f"decision.json#/{key}",
                )
            )
    comparisons = output.design.get("comparisons")
    if not isinstance(comparisons, Mapping) or set(comparisons) != {"hypothesis"}:
        violations.append(
            repairable_violation(
                "missing_comparisons",
                "design.comparisons must declare only the hypothesis expectation",
                "decision.json#/design/comparisons",
            )
        )
    else:
        hypothesis = comparisons.get("hypothesis")
        if not isinstance(hypothesis, Mapping):
            violations.append(
                repairable_violation(
                    "invalid_hypothesis_comparator",
                    "comparisons.hypothesis must be an object",
                    "decision.json#/design/comparisons/hypothesis",
                )
            )
        else:
            if set(hypothesis) != {"reason", "expected_observation"}:
                violations.append(
                    repairable_violation(
                        "invalid_hypothesis_comparator",
                        "hypothesis requires exactly reason and expected_observation; Harness owns comparator identity",
                        "decision.json#/design/comparisons/hypothesis",
                    )
                )
            if not non_empty_text(hypothesis.get("reason")):
                violations.append(
                    repairable_violation(
                        "invalid_hypothesis_comparator",
                        "hypothesis comparator reason is required",
                        "decision.json#/design/comparisons/hypothesis/reason",
                    )
                )
            expected = hypothesis.get("expected_observation")
            allowed = {
                "strict_improvement",
                "no_regression",
                "diagnostic_only",
            }
            if (
                not isinstance(expected, Mapping)
                or set(expected) != {"primary", "secondary"}
                or any(value not in allowed for value in expected.values())
            ):
                violations.append(
                    repairable_violation(
                        "invalid_expected_observation",
                        "expected_observation requires canonical primary and secondary relations",
                        "decision.json#/design/comparisons/hypothesis/expected_observation",
                    )
                )
    relation_kinds = {item.value for item in PlanRelationKind}
    if output.relation_kind not in relation_kinds:
        violations.append(
            repairable_violation(
                "invalid_plan_relation",
                "relation kind must be new_direction, revisit, contradiction, or combine",
                "decision.json#/relation/kind",
            )
        )
        return violations
    if (
        len(output.related_plan_keys) != len(set(output.related_plan_keys))
        or any(
            re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]*/c[0-9]{3}/p[0-9]{3}",
                key,
            )
            is None
            for key in output.related_plan_keys
        )
    ):
        violations.append(
            repairable_violation(
                "invalid_related_plans",
                "related_plans must contain unique <run>/cNNN/pNNN SubjectRefs",
                "decision.json#/relation/related_plans",
            )
        )
        return violations
    expected_count = {
        PlanRelationKind.NEW_DIRECTION.value: 0,
        PlanRelationKind.REVISIT.value: 1,
        PlanRelationKind.CONTRADICTION.value: 1,
    }.get(output.relation_kind)
    if expected_count is not None and len(output.related_plan_keys) != expected_count:
        violations.append(
            repairable_violation(
                "invalid_plan_relation_cardinality",
                f"{output.relation_kind} requires {expected_count} related Plans",
                "decision.json#/relation/related_plans",
            )
        )
    if (
        output.relation_kind == PlanRelationKind.COMBINE.value
        and len(output.related_plan_keys) < 2
    ):
        violations.append(
            repairable_violation(
                "invalid_plan_relation_cardinality",
                "combine requires at least two related Plans",
                "decision.json#/relation/related_plans",
            )
        )
    return violations


def analysis_violations(output: AnalysisFinding) -> list[DeliveryViolation]:
    violations: list[DeliveryViolation] = []
    if output.schema_version != "1":
        violations.append(
            repairable_violation("invalid_schema", "schema 1 is required", "analysis.json")
        )
    if not non_empty_text(output.finding):
        violations.append(
            repairable_violation(
                "missing_finding",
                "finding must be non-empty",
                "analysis.json#/finding",
            )
        )
    if (
        not output.evidence_unit_ids
        or len(output.evidence_unit_ids) != len(set(output.evidence_unit_ids))
        or not all(non_empty_text(item) for item in output.evidence_unit_ids)
    ):
        violations.append(
            repairable_violation(
                "invalid_evidence_ids",
                "evidence_unit_ids must be non-empty and unique",
                "analysis.json#/evidence_unit_ids",
            )
        )
    if not non_empty_text(output.uncertainty):
        violations.append(
            repairable_violation(
                "missing_uncertainty",
                "uncertainty must be non-empty",
                "analysis.json#/uncertainty",
            )
        )
    if not all(
        non_empty_text(item)
        for item in (*output.contradictions, *output.recommended_next_checks)
    ):
        violations.append(
            repairable_violation(
                "invalid_analysis_lists",
                "analysis list entries must be non-empty strings",
                "analysis.json",
            )
        )
    return violations


_ARTIFACT_CITATION = re.compile(r"\[\[artifact:([^\]\s]+)\]\]")
_REVIEW_CITATION = re.compile(r"\[\[review:([^\]\s]+)\]\]")
_SOURCE_CITATION = re.compile(r"\[\[source:([^\]\s]+)\]\]")


def analysis_report_violations(
    output: AnalysisReport,
    sections: tuple[str, ...],
) -> list[DeliveryViolation]:
    try:
        text = output.content.decode("utf-8")
    except UnicodeDecodeError:
        return [
            repairable_violation(
                "invalid_analysis_report",
                "analysis.md must be valid UTF-8",
                "analysis.md",
            )
        ]
    if not text.strip():
        return [
            repairable_violation(
                "empty_analysis_report",
                "analysis.md must be non-empty",
                "analysis.md",
            )
        ]
    del sections
    try:
        findings = output.findings_content.decode("utf-8")
    except UnicodeDecodeError:
        return [repairable_violation("invalid_findings", "findings.md must be UTF-8", "findings.md")]
    if not findings.strip():
        return [repairable_violation("empty_findings", "findings.md must be non-empty", "findings.md")]
    try:
        evidence = json.loads(output.evidence_content)
        coverage = json.loads(output.review_coverage_content)
    except json.JSONDecodeError:
        return [repairable_violation("invalid_analysis_sidecar", "analysis sidecars must be JSON", "evidence.json")]
    if not isinstance(evidence, dict) or evidence.get("schema_version") not in {
        "1",
        "ade.analysis_evidence.v2",
    }:
        return [repairable_violation("invalid_evidence", "evidence.json schema is invalid", "evidence.json")]
    artifact_ids = evidence.get("artifact_ids")
    review_ids = evidence.get("review_ids")
    if not _sorted_unique_strings(artifact_ids, non_empty=True) or not _sorted_unique_strings(review_ids):
        return [repairable_violation("invalid_evidence_ids", "evidence IDs must be sorted and unique", "evidence.json")]
    citation_text = text + "\n" + findings
    if sorted(set(_ARTIFACT_CITATION.findall(citation_text))) != artifact_ids:
        return [repairable_violation("artifact_citation_mismatch", "Markdown artifact citations must equal evidence.json", "evidence.json#/artifact_ids")]
    if sorted(set(_REVIEW_CITATION.findall(citation_text))) != review_ids:
        return [repairable_violation("review_citation_mismatch", "Markdown review citations must equal evidence.json", "evidence.json#/review_ids")]
    violations: list[DeliveryViolation] = []
    binding_fields = (
        ("command_id",)
        if evidence.get("schema_version") == "ade.analysis_evidence.v2"
        else ("trial_id", "attempt_id", "manifest_sha256")
    )
    for field in binding_fields:
        if not non_empty_text(evidence.get(field)):
            violations.append(
                repairable_violation(
                    "invalid_evidence_binding",
                    f"evidence.json requires {field}",
                    f"evidence.json#/{field}",
                )
            )
    if (
        not isinstance(coverage, dict)
        or coverage.get("schema_version")
        not in {"1", "ade.analysis_review_coverage.v1"}
        or not isinstance(coverage.get("passed"), bool)
    ):
        violations.append(
            repairable_violation(
                "invalid_review_coverage",
                "review_coverage.json schema and passed are required",
                "review_coverage.json",
            )
        )
    return violations


def _sorted_unique_strings(value: object, *, non_empty: bool = False) -> bool:
    return (
        isinstance(value, list)
        and (bool(value) or not non_empty)
        and all(non_empty_text(item) for item in value)
        and value == sorted(set(value))
    )


def summary_violations(output: PlanSummary | RunSummary) -> list[DeliveryViolation]:
    try:
        text = output.content.decode("utf-8")
    except UnicodeDecodeError:
        return [repairable_violation("invalid_memory", "MEMORY.md must be UTF-8", "MEMORY.md")]
    if not text.strip():
        return [repairable_violation("empty_memory", "MEMORY.md must be non-empty", "MEMORY.md")]
    ids = output.source_ids
    digests = output.source_manifest_digests
    if output.schema_version != "1" or output.basis_revision < 0 or not non_empty_text(output.subject_id):
        return [repairable_violation("invalid_summary_evidence", "evidence identity and schema are invalid", "evidence.json")]
    if not ids or ids != tuple(sorted(set(ids))) or tuple(item[0] for item in digests) != ids:
        return [repairable_violation("invalid_summary_sources", "sources must be non-empty, sorted, and unique", "evidence.json#/sources")]
    if any(not re.fullmatch(r"[0-9a-f]{64}", digest) for _, digest in digests):
        return [repairable_violation("invalid_source_manifest_digest", "source manifest digests must be SHA-256", "evidence.json#/sources")]
    if output.citation_ids != tuple(sorted(set(output.citation_ids))) or output.citation_ids != ids:
        return [repairable_violation(
            "summary_source_binding_mismatch",
            "Harness source IDs must equal accepted snapshot sources",
            "_meta/delivery.json",
        )]
    return []


def repairable_violation(code: str, message: str, path: str) -> DeliveryViolation:
    return DeliveryViolation(code, message, path, repairable=True)


def non_empty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def experiment_design_violations(
    output: ArtifactDelivery,
) -> list[DeliveryViolation]:
    if (
        not output.supporting_artifacts
        or output.supporting_artifacts[0].path != "design.md"
        or output.supporting_artifacts[0].kind != "design_report"
    ):
        return [
            repairable_violation(
                "invalid_experiment_design",
                "Builder requires Harness-bound design.md",
                "design.md",
            )
        ]
    try:
        text = output.supporting_artifacts[0].content.decode("utf-8")
    except UnicodeDecodeError:
        text = ""
    if not text.strip():
        return [repairable_violation(
            "empty_experiment_design",
            "design.md must be non-empty UTF-8",
            "design.md",
        )]
    return []


def string_list(value: object, *, non_empty: bool = False) -> bool:
    return (
        isinstance(value, list)
        and (not non_empty or bool(value))
        and all(non_empty_text(item) for item in value)
    )


def _output_error(code: str, message: str, path: str) -> RoleOutputError:
    return RoleOutputError(DeliveryViolation(code, message, path, repairable=True))
