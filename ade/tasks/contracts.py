"""Contracts used by vertical Task Plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Protocol, TypeAlias, TypeVar

from ade.core.agent import AgentRole
from ade.core.validation import DeliveryViolation, ValidationReport

@dataclass(frozen=True)
class ArtifactSpec:
    kind: str
    required: bool


@dataclass(frozen=True)
class RankingSpec:
    metric_id: str
    direction: str = "maximize"


@dataclass(frozen=True)
class AnalyzerArtifactSelector:
    categories: tuple[str, ...]
    kinds: tuple[str, ...]


@dataclass(frozen=True)
class AnalyzerCoverageSpec:
    mode: Literal["all", "fraction"]
    fraction: float | None = None
    unit: Literal["records", "groups"] = "records"

    def __post_init__(self) -> None:
        if self.mode == "all" and self.fraction is not None:
            raise ValueError("all coverage must not declare a fraction")
        if self.mode == "fraction" and (
            self.fraction is None or not 0 < self.fraction <= 1
        ):
            raise ValueError("fraction coverage requires a value in (0, 1]")


@dataclass(frozen=True)
class AnalyzerReviewFields:
    question: str
    response: str


@dataclass(frozen=True)
class AnalyzerGroupingSpec:
    group_key: str | None = None
    position_key: str | None = None
    response_index_key: str | None = None


@dataclass(frozen=True)
class AnalyzerEvidenceSpec:
    pool_id: str
    semantic_role: str
    artifact_selector: AnalyzerArtifactSelector
    coverage: AnalyzerCoverageSpec
    review_fields: AnalyzerReviewFields
    grouping: AnalyzerGroupingSpec | None = None
    context_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentContextFile:
    path: str
    content: bytes
    source_ref: str


@dataclass(frozen=True)
class AgentContextReference:
    path: str
    source_ref: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class AgentInputRequest:
    role: AgentRole
    run_id: str
    subject_id: str
    basis_revision: int
    action: Mapping[str, object]
    task: Mapping[str, object]
    context_files: tuple[AgentContextFile, ...] = ()
    context_references: tuple[AgentContextReference, ...] = ()


@dataclass(frozen=True)
class AgentInputSpec:
    schema_version: str
    task_id: str
    skill_id: str
    role: AgentRole
    run_id: str
    subject_id: str
    basis_revision: int
    action: Mapping[str, object]
    task: Mapping[str, object]
    context_files: tuple[AgentContextFile, ...]
    context_references: tuple[AgentContextReference, ...] = ()


@dataclass(frozen=True)
class SupportingArtifact:
    path: str
    kind: str
    content: bytes


@dataclass(frozen=True)
class PlanningDecision:
    schema_version: str
    decision: str
    plan_id: str
    basis_revision: int
    rationale: str
    hypothesis: str
    expected_observation: str
    risk: str
    design: Mapping[str, object]
    content: bytes = b""
    supporting_artifacts: tuple[SupportingArtifact, ...] = ()
    relation_kind: str = ""
    related_plan_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArtifactDelivery:
    schema_version: str
    path: str
    kind: str
    content: bytes
    metadata: Mapping[str, object]
    payload: Mapping[str, object] | None = None
    supporting_artifacts: tuple[SupportingArtifact, ...] = ()


@dataclass(frozen=True)
class ArtifactCompilationRequest:
    delivery: ArtifactDelivery
    task_config: Mapping[str, object]
    planning_decision: Mapping[str, object]


@dataclass(frozen=True)
class CompiledArtifact:
    path: str
    kind: str
    content: bytes


@dataclass(frozen=True)
class BaselineArtifactRequest:
    task_config: Mapping[str, object]
    seed: int


@dataclass(frozen=True)
class EngineObjectPayload:
    uri: str
    content: bytes


@dataclass(frozen=True)
class EngineArtifactBindingRequest:
    binding_uri: str
    compiled_kind: str
    compiled_content: bytes
    task_config: Mapping[str, object]
    engine_config: Mapping[str, object]
    planning_decision: Mapping[str, object] = field(default_factory=dict)
    final_realization: Mapping[str, object] = field(default_factory=dict)
    is_baseline: bool = False


@dataclass(frozen=True)
class EngineArtifactBinding:
    input_payload: Mapping[str, object]
    objects: tuple[EngineObjectPayload, ...]


@dataclass(frozen=True)
class AnalysisFinding:
    schema_version: str
    finding: str
    evidence_unit_ids: tuple[str, ...]
    uncertainty: str
    contradictions: tuple[str, ...]
    recommended_next_checks: tuple[str, ...]
    content: bytes = b""


@dataclass(frozen=True)
class AnalysisReport:
    content: bytes
    findings_content: bytes
    evidence_content: bytes
    review_coverage_content: bytes


@dataclass(frozen=True)
class AnalysisReviewSelection:
    mode: Literal["all", "records", "groups"]
    source_artifact_ids: tuple[str, ...] = ()
    record_ids: tuple[str, ...] = ()
    group_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnalysisReviewBatch:
    batch_id: str
    pool: str
    investigation_purpose: str
    selection: AnalysisReviewSelection
    rubrics: tuple[Mapping[str, object], ...]
    hypothesis_ref: str | None = None


@dataclass(frozen=True)
class AnalysisReviewPlan:
    schema_version: str
    hypotheses: tuple[str, ...]
    batches: tuple[AnalysisReviewBatch, ...]
    content: bytes


@dataclass(frozen=True)
class PlanSummary:
    schema_version: str
    subject_id: str
    basis_revision: int
    source_ids: tuple[str, ...]
    source_manifest_digests: tuple[tuple[str, str], ...]
    citation_ids: tuple[str, ...]
    assessment: str
    content: bytes
    evidence_content: bytes

    @property
    def current_claim(self) -> str:
        return self.assessment


@dataclass(frozen=True)
class RunSummary:
    schema_version: str
    subject_id: str
    basis_revision: int
    source_ids: tuple[str, ...]
    source_manifest_digests: tuple[tuple[str, str], ...]
    citation_ids: tuple[str, ...]
    assessment: str
    content: bytes
    evidence_content: bytes

    @property
    def current_claim(self) -> str:
        return self.assessment


AgentRoleOutput: TypeAlias = (
    PlanningDecision
    | ArtifactDelivery
    | AnalysisFinding
    | AnalysisReviewPlan
    | AnalysisReport
    | PlanSummary
    | RunSummary
)
OutputT = TypeVar("OutputT", bound=AgentRoleOutput, covariant=True)


class RoleOutputError(ValueError):
    def __init__(self, violation: DeliveryViolation) -> None:
        super().__init__(violation.message)
        self.violation = violation


class ArtifactCompilationError(ValueError):
    def __init__(self, report: ValidationReport) -> None:
        if report.ok:
            raise ValueError("ArtifactCompilationError requires violations")
        super().__init__(report.violations[0].message)
        self.report = report


class ArtifactRealizationIntegrityError(ArtifactCompilationError):
    """Harness-owned realization inputs cannot support artifact admission."""


class SummaryAdmissionError(ValueError):
    def __init__(self, report: ValidationReport) -> None:
        if report.ok:
            raise ValueError("SummaryAdmissionError requires violations")
        super().__init__(report.violations[0].message)
        self.report = report


class AnalysisAdmissionError(ValueError):
    def __init__(self, report: ValidationReport) -> None:
        if report.ok:
            raise ValueError("AnalysisAdmissionError requires violations")
        super().__init__(report.violations[0].message)
        self.report = report


class AgentRoleContract(Protocol[OutputT]):
    task_id: str
    domain: str
    role: AgentRole
    skill_id: str
    output_path: str
    output_kind: str
    auxiliary_outputs: tuple[tuple[str, str], ...]

    def output_paths_for(self, action: Mapping[str, object]) -> tuple[str, ...]: ...

    def validate_input(self, request: AgentInputRequest) -> None: ...

    def decode_output(self, attempt: Path) -> OutputT: ...

    def validate_output(self, output: OutputT) -> ValidationReport: ...

    def finalize_output(self, attempt: Path, output: OutputT) -> None: ...


@dataclass(frozen=True)
class EngineCommandRequest:
    command_id: str
    run_id: str
    coordinator_id: str
    plan_id: str
    trial_id: str
    input_ref: str
    output_uri: str
    logical_command_id: str = ""
    attempt_id: str = "attempt-001"
    attempt_index: int = 1


@dataclass(frozen=True)
class TrialResult:
    coordinator_id: str
    plan_id: str
    trial_id: str
    succeeded: bool
    output_refs: tuple[str, ...]
    error: str | None
