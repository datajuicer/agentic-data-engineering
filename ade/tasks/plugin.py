"""Protocol implementation shared by vertical Task Plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypeAlias

from ade.core.actions import Action, ActionKind
from ade.core.agent import AgentRole
from ade.core.artifacts import ArtifactRef
from ade.core.engine import (
    EngineCommand,
    EngineReceipt,
    EngineReceiptStatus,
    EvaluateCommand,
    TrainRFTCommand,
    TrainSFTCommand,
)
from ade.core.validation import DeliveryViolation, ValidationReport
from ade.core.outcomes import PlanningDecisionOutcome
from ade.core.plan import PlanState, PlanStatus
from ade.tasks.contracts import (
    AgentInputRequest,
    AgentInputSpec,
    AgentRoleContract,
    AnalyzerEvidenceSpec,
    ArtifactCompilationRequest,
    ArtifactCompilationError,
    BaselineArtifactRequest,
    EngineArtifactBinding,
    EngineArtifactBindingRequest,
    CompiledArtifact,
    PlanningDecision,
    ArtifactSpec,
    EngineCommandRequest,
    RankingSpec,
    TrialResult,
)

CommandType: TypeAlias = type[EvaluateCommand] | type[TrainSFTCommand] | type[TrainRFTCommand]
ArtifactCompiler: TypeAlias = Callable[
    [ArtifactCompilationRequest],
    CompiledArtifact,
]
EngineArtifactBinder: TypeAlias = Callable[
    [EngineArtifactBindingRequest],
    EngineArtifactBinding,
]
EngineConfigPreparer: TypeAlias = Callable[
    [dict[str, Any], Any], dict[str, Any]
]
ArtifactAcceptanceValidator: TypeAlias = Callable[["ArtifactAcceptanceContext"], ArtifactRef | None]


@dataclass(frozen=True)
class ArtifactAcceptanceContext:
    run_id: str
    trial_key: Any
    state: Any
    trial: Any
    delivery: Any
    compiled: CompiledArtifact
    planning_decision: dict[str, object]
    lifecycle: Any
BaselineArtifactFactory: TypeAlias = Callable[
    [BaselineArtifactRequest],
    CompiledArtifact,
]
_COMMAND_KINDS = {
    EvaluateCommand: "evaluate",
    TrainSFTCommand: "train_sft",
    TrainRFTCommand: "train_rft",
}


@dataclass(frozen=True)
class TaskPlugin:
    task_id: str
    command_type: CommandType
    artifact: ArtifactSpec | None
    ranking: RankingSpec | None
    analyzer_evidence_specs: tuple[AnalyzerEvidenceSpec, ...] = ()
    contracts: tuple[AgentRoleContract, ...] = ()
    artifact_compiler: ArtifactCompiler | None = None
    engine_artifact_binder: EngineArtifactBinder | None = None
    baseline_factory: BaselineArtifactFactory | None = None
    engine_config_preparer: EngineConfigPreparer | None = None
    artifact_acceptance_validator: ArtifactAcceptanceValidator | None = None

    @property
    def engine_command_kind(self) -> str:
        return _COMMAND_KINDS[self.command_type]

    def validate_config(self, config: dict[str, Any]) -> ValidationReport:
        task = config.get("task")
        task_id = task.get("id") if isinstance(task, dict) else config.get("task_id")
        if task_id != self.task_id:
            return ValidationReport(
                (DeliveryViolation("task_config_mismatch", f"expected task {self.task_id}"),)
            )
        return ValidationReport()

    def skill_for(self, role: AgentRole) -> str:
        return self.role_contract(role).skill_id

    @property
    def skills(self) -> tuple[tuple[AgentRole, str], ...]:
        return tuple((contract.role, contract.skill_id) for contract in self.contracts)

    def role_contract(self, role: AgentRole) -> AgentRoleContract:
        matches = [contract for contract in self.contracts if contract.role is role]
        if len(matches) != 1:
            raise KeyError(f"task {self.task_id} has no unique contract for role {role}")
        return matches[0]

    def artifact_spec(self) -> ArtifactSpec | None:
        return self.artifact

    def planning_outcome(
        self,
        action: Action,
        output: PlanningDecision,
        decision_ref: ArtifactRef,
        call_id: str,
    ) -> PlanningDecisionOutcome:
        if action.kind is not ActionKind.CALL_AGENT:
            raise ValueError("planning output requires a CALL_AGENT action")
        if output.basis_revision != action.basis_revision:
            raise ValueError("planning output basis does not match action")
        report = self.role_contract(AgentRole.COORDINATOR).validate_output(output)
        if not report.ok:
            raise ValueError(report.violations[0].message)
        if decision_ref.kind != "planning_decision":
            raise ValueError("Coordinator output requires a planning_decision artifact")
        return PlanningDecisionOutcome(
            action_id=action.action_id,
            run_id=action.run_id,
            subject_id=action.subject_id,
            basis_revision=action.basis_revision,
            call_id=call_id,
            plan=PlanState(
                plan_id=output.plan_id,
                coordinator_id=action.subject_id,
                status=PlanStatus.ACTIVE,
                basis_revision=action.basis_revision,
                decision_ref_id=decision_ref.artifact_id,
            ),
            decision_ref=decision_ref,
        )

    def build_agent_input(self, request: AgentInputRequest) -> AgentInputSpec:
        contract = self.role_contract(request.role)
        contract.validate_input(request)
        return AgentInputSpec(
            schema_version="1",
            task_id=self.task_id,
            skill_id=contract.skill_id,
            role=request.role,
            run_id=request.run_id,
            subject_id=request.subject_id,
            basis_revision=request.basis_revision,
            action=request.action,
            task=request.task,
            context_files=request.context_files,
            context_references=request.context_references,
        )

    def validate_artifact(self, artifact: ArtifactRef) -> ValidationReport:
        if self.artifact is None:
            return ValidationReport(
                (DeliveryViolation("artifact_not_supported", f"task {self.task_id} has no artifact"),)
            )
        if artifact.kind != self.artifact.kind:
            return ValidationReport(
                (
                    DeliveryViolation(
                        "artifact_kind_mismatch",
                        f"expected artifact kind {self.artifact.kind}",
                    ),
                )
            )
        return ValidationReport()

    def compile_artifact(
        self,
        request: ArtifactCompilationRequest,
    ) -> CompiledArtifact:
        if self.artifact is None or self.artifact_compiler is None:
            raise ValueError(f"task {self.task_id} has no artifact compiler")
        try:
            compiled = self.artifact_compiler(request)
        except ArtifactCompilationError:
            raise
        except ValueError as error:
            feedback_paths = (request.delivery.path,) + tuple(
                artifact.path
                for artifact in request.delivery.supporting_artifacts
                if artifact.kind == "design_report"
            )
            raise ArtifactCompilationError(
                ValidationReport(
                    tuple(
                        DeliveryViolation(
                            "artifact_compilation_failed",
                            str(error),
                            path,
                            repairable=True,
                        )
                        for path in feedback_paths
                    )
                )
            ) from error
        if compiled.kind != self.artifact.kind:
            raise ValueError(
                f"artifact compiler returned {compiled.kind}, expected {self.artifact.kind}"
            )
        if not compiled.path or not compiled.content:
            raise ValueError("artifact compiler returned an empty artifact")
        return compiled

    def build_engine_command(self, request: EngineCommandRequest) -> EngineCommand:
        if not request.input_ref or not request.output_uri:
            raise ValueError("Engine command input_ref and output_uri are required")
        return self.command_type(
            command_id=request.command_id,
            run_id=request.run_id,
            coordinator_id=request.coordinator_id,
            plan_id=request.plan_id,
            trial_id=request.trial_id,
            input_ref=request.input_ref,
            output_uri=request.output_uri,
            logical_command_id=request.logical_command_id or request.command_id,
            attempt_id=request.attempt_id,
            attempt_index=request.attempt_index,
        )

    def bind_engine_artifact(
        self,
        request: EngineArtifactBindingRequest,
    ) -> EngineArtifactBinding:
        if self.engine_artifact_binder is None:
            raise ValueError(f"task {self.task_id} has no Engine artifact binder")
        if self.artifact is None or request.compiled_kind != self.artifact.kind:
            raise ValueError("Engine binding artifact kind does not match task")
        binding = self.engine_artifact_binder(request)
        if not binding.objects:
            raise ValueError("Engine artifact binder returned no objects")
        if not binding.input_payload.get("artifact_ref"):
            raise ValueError("Engine artifact binder omitted artifact_ref")
        return binding

    def prepare_engine_config(
        self, config: dict[str, Any], state: Any
    ) -> dict[str, Any]:
        """Apply task-owned runtime facts without branching shared Control code."""
        if self.engine_config_preparer is None:
            return config
        return self.engine_config_preparer(config, state)

    def validate_artifact_acceptance(
        self, context: ArtifactAcceptanceContext
    ) -> ArtifactRef | None:
        if self.artifact_acceptance_validator is None:
            return None
        return self.artifact_acceptance_validator(context)

    def build_baseline_artifact(
        self,
        request: BaselineArtifactRequest,
    ) -> CompiledArtifact:
        if self.artifact is None or self.baseline_factory is None:
            raise ValueError(f"task {self.task_id} has no baseline factory")
        baseline = self.baseline_factory(request)
        if baseline.kind != self.artifact.kind:
            raise ValueError(
                f"baseline factory returned {baseline.kind}, expected {self.artifact.kind}"
            )
        if not baseline.path or not baseline.content:
            raise ValueError("baseline factory returned an empty artifact")
        return baseline

    def normalize_engine_receipt(self, receipt: EngineReceipt) -> TrialResult:
        return TrialResult(
            coordinator_id=receipt.coordinator_id,
            plan_id=receipt.plan_id,
            trial_id=receipt.trial_id,
            succeeded=receipt.status is EngineReceiptStatus.SUCCEEDED,
            output_refs=receipt.output_refs,
            error=receipt.error,
        )

    def ranking_spec(self) -> RankingSpec | None:
        return self.ranking
