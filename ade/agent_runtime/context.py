"""Project accepted state into role-scoped Agent input packages."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path, PurePosixPath
from typing import Mapping, Protocol

from ade.agent_runtime.experiment_package import decode_experiment_package
from ade.agent_runtime.input_package import AgentInputPackage, AgentInputPackageBuilder
from ade.core.agent import AgentRole
from ade.core.artifacts import ArtifactRef
from ade.core.plan import PlanKind, PlanRelationKind, PlanStatus
from ade.core.ranking import score_pair_sort_key
from ade.core.run import RunState
from ade.core.snapshot import SnapshotKind, SnapshotRef
from ade.core.trial import TrialKind, TrialPhase
from ade.memory.records import objective_comparison_record_path
from ade.tasks.contracts import AgentContextFile, AgentContextReference, AgentInputRequest
from ade.tasks.plugin import TaskPlugin
from ade.tasks.registry import TaskRegistry
from ade.tasks.reward_design.compliance import (
    direct_reference_trial,
    encode_records,
    load_published_baseline_records,
)
from ade.tasks.curriculum_learning.fixed_pool import build_fixed_pool_from_task


class ContextRepository(Protocol):
    def load(self, run_id: str) -> RunState: ...

    def read_artifact(self, run_id: str, ref: ArtifactRef) -> bytes: ...


_ACTION_KINDS = {
    AgentRole.COORDINATOR: "coordinate",
    AgentRole.ARTIFACT_BUILDER: "build_artifact",
    AgentRole.ANALYZER: "analyze_trial",
    AgentRole.PLAN_SUMMARIZER: "summarize_plan",
    AgentRole.RUN_SUMMARIZER: "summarize_run",
}


class RoleContextPackager:
    def __init__(
        self,
        *,
        repository: ContextRepository,
        tasks: TaskRegistry,
        packages: AgentInputPackageBuilder | None = None,
        trial_artifacts_root: str | Path | None = None,
    ) -> None:
        self.repository = repository
        self.tasks = tasks
        self.packages = packages or AgentInputPackageBuilder()
        repository_layout = getattr(repository, "layout", None)
        self.trial_artifacts_root = (
            Path(trial_artifacts_root).resolve()
            if trial_artifacts_root is not None
            else repository_layout.runs_root.parent / "engine-work" / "trial_artifacts"
        )

    def build(
        self,
        *,
        run_id: str,
        role: AgentRole,
        subject_id: str,
        basis_revision: int,
        action_id: str,
        coordinator_id: str | None = None,
        plan_id: str | None = None,
        action_fields: Mapping[str, object] | None = None,
    ) -> AgentInputPackage:
        state = self.repository.load(run_id)
        if basis_revision != state.revision:
            raise ValueError(
                f"stale Agent context basis {basis_revision}, current revision {state.revision}"
            )
        if not action_id:
            raise ValueError("action_id is required")
        plugin = self.tasks.get(state.task.task_id)
        contract = plugin.role_contract(role)
        context_files = self._context_files(
            state,
            role,
            subject_id,
            plugin,
            coordinator_id=coordinator_id,
            plan_id=plan_id,
        )
        context_references = self._context_references(
            role,
            context_files,
        )
        extra_action = dict(action_fields or {})
        if {"kind", "action_id", "scope", "subject_ref"}.intersection(
            extra_action
        ):
            raise ValueError("Agent action fields cannot override Harness identity")
        scope, full_subject_ref = _agent_subject_identity(
            run_id=run_id,
            role=role,
            subject_id=subject_id,
            coordinator_id=coordinator_id,
            plan_id=plan_id,
        )
        request = AgentInputRequest(
            role=role,
            run_id=run_id,
            subject_id=subject_id,
            basis_revision=state.revision,
            action={
                "kind": _ACTION_KINDS[role],
                "action_id": action_id,
                "scope": scope,
                "subject_ref": full_subject_ref,
                **extra_action,
            },
            task={
                "task_id": state.task.task_id,
                "domain": contract.domain,
            },
            context_files=context_files,
            context_references=context_references,
        )
        spec = plugin.build_agent_input(request)
        return self.packages.build(spec, contract)

    @staticmethod
    def _context_references(
        role: AgentRole,
        context_files: tuple[AgentContextFile, ...],
    ) -> tuple[AgentContextReference, ...]:
        if role is not AgentRole.ANALYZER:
            return ()
        manifest_files = [
            item for item in context_files
            if item.path == "experiment/manifest.json"
        ]
        if len(manifest_files) != 1:
            raise ValueError("Experiment Package manifest is missing")
        manifest = json.loads(manifest_files[0].content)
        references = []
        for artifact in manifest.get("artifacts", ()):
            if not isinstance(artifact, Mapping):
                continue
            storage = artifact.get("storage")
            path = artifact.get("path")
            if (
                not isinstance(storage, Mapping)
                or storage.get("mode") != "durable_reference"
                or not isinstance(path, str)
            ):
                continue
            references.append(
                AgentContextReference(
                    path=path,
                    source_ref=str(storage.get("uri") or ""),
                    sha256=str(storage.get("sha256") or ""),
                    size_bytes=int(storage.get("size_bytes", -1)),
                )
            )
        return tuple(sorted(references, key=lambda item: item.path))

    def _context_files(
        self,
        state: RunState,
        role: AgentRole,
        subject_id: str,
        plugin: TaskPlugin,
        *,
        coordinator_id: str | None,
        plan_id: str | None,
    ) -> tuple[AgentContextFile, ...]:
        if role is AgentRole.COORDINATOR:
            if not any(
                item.coordinator_id == subject_id for item in state.coordinators
            ):
                raise ValueError("unknown Coordinator subject")
            memories = getattr(self.repository, "memory_versions", None)
            if memories is None:
                raise ValueError("Coordinator requires Memory store")
            run_memory_id = state.memory.run_head
            memory_files = memories.read_run_version(
                run_id=state.run_id,
                memory_id=run_memory_id.rsplit("/", 1)[-1],
            )
            return (
                AgentContextFile(
                    "memory/manifest.json",
                    memory_files["manifest.json"],
                    run_memory_id,
                ),
                AgentContextFile(
                    "memory/MEMORY.md",
                    memory_files["MEMORY.md"],
                    run_memory_id,
                ),
                *(
                    AgentContextFile(f"memory/{path}", content, run_memory_id)
                    for path, content in sorted(memory_files.items())
                    if path.startswith(("outcomes/", "findings/"))
                ),
                AgentContextFile(
                    "subject/PLAN_CATALOG.md",
                    _plan_catalog_markdown(state, self.repository),
                    f"run://{state.run_id}/run.json#plan_catalog",
                ),
                AgentContextFile(
                    "subject/ranking.json",
                    _encode(
                        {
                            "metric_id": state.ranking.metric_id,
                            "direction": state.ranking.direction,
                            "revision": state.ranking.revision,
                            "entries": [
                                {
                                    "level": item.level,
                                    "subject_id": item.subject_id,
                                    "score": item.score,
                                    "secondary_score": item.secondary_score,
                                    "evaluation_profile": item.evaluation_profile,
                                    "accepted_revision": item.accepted_revision,
                                    "artifact_ref_id": item.artifact_ref_id,
                                    "artifact_digest": item.artifact_digest,
                                    "source_eligible": item.source_eligible,
                                    "representative_trial_id": item.representative_trial_id,
                                }
                                for item in state.ranking.entries
                            ],
                        }
                    ),
                    f"run://{state.run_id}/run.json#ranking",
                ),
                AgentContextFile(
                    "subject/portfolio-target.json",
                    _encode(_portfolio_target(state)),
                    (
                        f"run://{state.run_id}/run.json#ranking"
                        f"@revision-{state.ranking.revision}"
                    ),
                ),
                _coordinator_budget_file(state),
                _task_config_file(state),
            )
        if role is AgentRole.ARTIFACT_BUILDER:
            trial = _trial(state, coordinator_id, plan_id, subject_id)
            plan = _plan(state, trial.coordinator_id, trial.plan_id)
            current = _snapshot(
                state,
                plan.latest_snapshot_ref_id,
                SnapshotKind.PLAN,
                f"current Plan {plan.coordinator_id}/{plan.plan_id}",
            )
            seeds = _seed_plan_snapshots(state, plan)
            task_context: tuple[AgentContextFile, ...] = ()
            if plugin.task_id == "reward_design":
                task_context = self._reward_builder_context(state, trial)
            elif plugin.task_id == "data_selection":
                task_context = self._data_selection_builder_context(state, trial)
            elif plugin.task_id == "curriculum_learning":
                task_context = self._curriculum_builder_context(state, trial)
            return (
                *_snapshot_context_files(current, "snapshots/current-plan"),
                *(
                    item
                    for seed in seeds
                    for item in _snapshot_context_files(
                        seed,
                        f"snapshots/seed-plans/{seed.coordinator_id}/{seed.plan_id}",
                    )
                ),
                *task_context,
                _task_config_file(state),
            )
        if role is AgentRole.ANALYZER:
            trial = _trial(state, coordinator_id, plan_id, subject_id)
            package_id = trial.package_ref_id
            if not package_id:
                raise ValueError(f"Trial has no accepted Experiment Package: {subject_id}")
            package_ref = _artifact(state, package_id, "experiment_package")
            package = decode_experiment_package(
                self.repository.read_artifact(state.run_id, package_ref)
            )
            package = _normalize_experiment_package_paths(package)
            if "experiment/manifest.json" not in package:
                raise ValueError("Experiment Package manifest is missing")
            review_context: tuple[AgentContextFile, ...] = ()
            if trial.analysis_review_packet_ref_id is not None:
                packet_ref = _artifact(
                    state,
                    trial.analysis_review_packet_ref_id,
                    "analysis_review_packet",
                )
                coverage_ref = _artifact(
                    state,
                    str(trial.analysis_review_coverage_ref_id),
                    "analysis_review_coverage",
                )
                review_context = (
                    AgentContextFile(
                        "review/packet.json",
                        self.repository.read_artifact(state.run_id, packet_ref),
                        packet_ref.uri,
                    ),
                    AgentContextFile(
                        "review/coverage.json",
                        self.repository.read_artifact(state.run_id, coverage_ref),
                        coverage_ref.uri,
                    ),
                )
            return (
                *tuple(
                    AgentContextFile(path, content, f"{package_ref.uri}#{path}")
                    for path, content in sorted(package.items())
                ),
                *review_context,
            )
        if role is AgentRole.PLAN_SUMMARIZER:
            plan = _plan(state, coordinator_id, subject_id)
            pending = tuple(
                trial
                for trial in state.trials
                if trial.coordinator_id == plan.coordinator_id
                and trial.plan_id == plan.plan_id
                and trial.phase in {
                    TrialPhase.ANALYSIS_READY,
                    TrialPhase.PLAN_SUMMARIZING,
                }
            )
            if len(pending) != 1:
                raise ValueError("Plan Summarizer requires one pending Trial")
            trial = pending[0]
            if not trial.plan_memory_basis or not trial.trial_record_ref:
                raise ValueError("Plan Summarizer basis is incomplete")
            memories = getattr(self.repository, "memory_versions", None)
            records = getattr(self.repository, "trial_records", None)
            if memories is None or records is None:
                raise ValueError("Plan Summarizer requires Memory and Trial Record stores")
            parent_files = memories.read_plan_version(
                run_id=state.run_id,
                coordinator_id=plan.coordinator_id,
                plan_id=plan.plan_id,
                memory_id=trial.plan_memory_basis.rsplit("/", 1)[-1],
            )
            record_files = records.read_files(
                run_id=state.run_id,
                coordinator_id=trial.coordinator_id,
                plan_id=trial.plan_id,
                trial_id=trial.trial_id,
            )
            plan_source_ref = trial.plan_memory_basis
            if plan.kind is PlanKind.SEARCH:
                plan_ref = _artifact(
                    state,
                    str(plan.decision_report_ref_id),
                    "planning_decision_report",
                )
                plan_source_ref = plan_ref.uri
            context = [
                AgentContextFile(
                    "memory/manifest.json",
                    parent_files["manifest.json"],
                    trial.plan_memory_basis,
                ),
                AgentContextFile(
                    "memory/MEMORY.md",
                    parent_files["MEMORY.md"],
                    trial.plan_memory_basis,
                ),
                *(
                    AgentContextFile(
                        f"memory/{path}", content, trial.plan_memory_basis
                    )
                    for path, content in sorted(parent_files.items())
                    if path.startswith(("outcomes/", "findings/"))
                ),
                AgentContextFile(
                    "subject/plan.md",
                    parent_files["plan.md"],
                    plan_source_ref,
                ),
                AgentContextFile(
                    "subject/trial/manifest.json",
                    record_files["manifest.json"],
                    trial.trial_record_ref,
                ),
                AgentContextFile(
                    "subject/trial/TRIAL.md",
                    _trial_markdown(state, trial),
                    trial.trial_record_ref,
                ),
                AgentContextFile(
                    "subject/trial/outcome/outcome.json",
                    _trial_outcome(state, trial),
                    trial.trial_record_ref,
                ),
            ]
            current_files = _current_trial_semantic_files(
                record_files,
                engine_attempt_index=trial.engine_attempt_index,
            )
            context.extend(
                AgentContextFile(path, content, source_ref)
                for path, content, source_ref in current_files
            )
            if trial.failure_kind or trial.analysis_failure_ref_id:
                context.append(
                    AgentContextFile(
                        "subject/trial/failure.json",
                        _encode(
                            {
                                "schema_version": "1",
                                "trial": trial.trial_record_ref,
                                "failure_kind": trial.failure_kind,
                                "analysis_failure_ref_id": trial.analysis_failure_ref_id,
                            }
                        ),
                        trial.trial_record_ref,
                    )
                )
            return tuple(context)
        if role is AgentRole.RUN_SUMMARIZER:
            if subject_id != state.run_id:
                raise ValueError("Run Summarizer subject must be the run")
            if not state.rm_merge_queue:
                raise ValueError("Run Summarizer requires an RM merge queue head")
            queue_head = state.rm_merge_queue[0]
            parts = queue_head.split("/")
            if len(parts) != 4 or parts[0] != state.run_id:
                raise ValueError("RM merge queue head is invalid")
            trial = _trial(state, parts[1], parts[2], parts[3])
            if (
                trial.phase not in {
                    TrialPhase.PLAN_SUMMARY_READY,
                    TrialPhase.RUN_SUMMARIZING,
                }
                or not trial.plan_memory_result
            ):
                raise ValueError("Run Summarizer queue-head Trial is not ready")
            memories = getattr(self.repository, "memory_versions", None)
            if memories is None:
                raise ValueError("Run Summarizer requires Memory store")
            parent_id = state.memory.run_head
            parent_files = memories.read_run_version(
                run_id=state.run_id,
                memory_id=parent_id.rsplit("/", 1)[-1],
            )
            plan_files = memories.read_plan_version(
                run_id=state.run_id,
                coordinator_id=trial.coordinator_id,
                plan_id=trial.plan_id,
                memory_id=trial.plan_memory_result.rsplit("/", 1)[-1],
            )
            update = _plan_update_markdown(state, trial, queue_head)
            catalog = _plan_catalog_markdown(state, self.repository)
            ranking = _ranking_markdown(state)
            facts_manifest = _run_facts_manifest(state, catalog, ranking)
            facts_ref = f"{state.run_id}@rev-{state.revision}"
            return (
                AgentContextFile(
                    "memory/manifest.json", parent_files["manifest.json"], parent_id
                ),
                AgentContextFile(
                    "memory/MEMORY.md", parent_files["MEMORY.md"], parent_id
                ),
                *(
                    AgentContextFile(f"memory/{path}", content, parent_id)
                    for path, content in sorted(parent_files.items())
                    if path.startswith(("outcomes/", "findings/"))
                ),
                AgentContextFile(
                    "subject/plan-memory/manifest.json",
                    plan_files["manifest.json"],
                    trial.plan_memory_result,
                ),
                AgentContextFile(
                    "subject/plan-memory/MEMORY.md",
                    plan_files["MEMORY.md"],
                    trial.plan_memory_result,
                ),
                *(
                    AgentContextFile(
                        f"subject/plan-memory/{path}",
                        content,
                        trial.plan_memory_result,
                    )
                    for path, content in sorted(plan_files.items())
                    if path.startswith(("outcomes/", "findings/"))
                ),
                AgentContextFile(
                    "subject/PLAN_UPDATE.md", update, f"{queue_head}#rm-merge"
                ),
                AgentContextFile("run/manifest.json", facts_manifest, facts_ref),
                AgentContextFile("run/PLAN_CATALOG.md", catalog, facts_ref),
                AgentContextFile("run/RANKING.md", ranking, facts_ref),
            )
        raise ValueError(f"unsupported Agent role: {role}")

    def _reward_builder_context(
        self,
        state: RunState,
        trial,
    ) -> tuple[AgentContextFile, ...]:
        sources = trial.source_artifact_ref_ids
        if not sources:
            raise ValueError("Reward Builder requires a primary reference artifact")
        refs = tuple(
            _artifact(state, artifact_id, "reward_design")
            for artifact_id in sources
        )
        reference_trial = direct_reference_trial(state, refs[0].artifact_id)
        if reference_trial is None or not reference_trial.package_ref_id:
            raise ValueError(
                "Reward Builder direct reference Trial package is unavailable"
            )
        package_ref = _artifact(
            state,
            str(reference_trial.package_ref_id),
            "experiment_package",
        )
        package = decode_experiment_package(
            self.repository.read_artifact(state.run_id, package_ref)
        )
        manifest = json.loads(package["experiment/manifest.json"])
        records, compliance_manifest = load_published_baseline_records(
            manifest,
            package,
            durable_artifacts_root=self.trial_artifacts_root,
        )
        reference_manifest = {
            "schema_version": "1",
            "primary_reference_artifact_ref_id": refs[0].artifact_id,
            "direct_reference_trial": (
                f"{state.run_id}/{reference_trial.coordinator_id}/"
                f"{reference_trial.plan_id}/{reference_trial.trial_id}"
            ),
            "donor_artifact_ref_ids": [ref.artifact_id for ref in refs[1:]],
            "source_artifact_ref_ids": list(sources),
            "artifacts": [
                {
                    "artifact_id": ref.artifact_id,
                    "kind": ref.kind,
                    "sha256": ref.digest,
                    "size_bytes": ref.size_bytes,
                    "path": (
                        "reference/primary/reward.py"
                        if index == 0
                        else f"reference/donors/{ref.artifact_id}/reward.py"
                    ),
                }
                for index, ref in enumerate(refs)
            ],
        }
        files = [
            AgentContextFile(
                "reference/manifest.json",
                _encode(reference_manifest),
                f"run://{state.run_id}/trials/{trial.trial_id}#reference",
            ),
            AgentContextFile(
                "compliance/manifest.json",
                _encode(compliance_manifest),
                package_ref.uri,
            ),
            AgentContextFile(
                "compliance/records.jsonl",
                encode_records(records),
                package_ref.uri,
            ),
        ]
        files.extend(
            AgentContextFile(
                (
                    "reference/primary/reward.py"
                    if index == 0
                    else f"reference/donors/{ref.artifact_id}/reward.py"
                ),
                self.repository.read_artifact(state.run_id, ref),
                ref.uri,
            )
            for index, ref in enumerate(refs)
        )
        return tuple(files)

    def _data_selection_builder_context(
        self,
        state: RunState,
        trial,
    ) -> tuple[AgentContextFile, ...]:
        files = list(_data_selection_training_context(state))
        sources = trial.source_artifact_ref_ids
        if not sources:
            return tuple(files)
        refs = tuple(
            _artifact(state, artifact_id, "data_selection")
            for artifact_id in sources
        )
        source_entries: list[dict[str, str]] = []
        source_manifest: dict[str, object] = {
            "schema_version": "1",
            "primary_source_artifact_ref_id": refs[0].artifact_id,
            "donor_artifact_ref_ids": [ref.artifact_id for ref in refs[1:]],
            "source_artifact_ref_ids": list(sources),
            "sources": source_entries,
        }
        for index, ref in enumerate(refs):
            reference_trial = direct_reference_trial(state, ref.artifact_id)
            if reference_trial is None or not reference_trial.package_ref_id:
                raise ValueError(
                    "Data Selection Builder source Trial package is unavailable: "
                    f"{ref.artifact_id}"
                )
            package_ref = _artifact(
                state,
                str(reference_trial.package_ref_id),
                "experiment_package",
            )
            package = decode_experiment_package(
                self.repository.read_artifact(state.run_id, package_ref)
            )
            selection_path = "experiment/realization/selection-result.json"
            selection_result = package.get(selection_path)
            if (
                selection_result is None
                and reference_trial.kind is TrialKind.BOOTSTRAP_BASELINE
            ):
                selection_path = (
                    "experiment/artifacts/sft-selection-config/selection_config.json"
                )
                selection_result = package.get(selection_path)
            if selection_result is None:
                raise ValueError(
                    "Data Selection Builder source realization is unavailable: "
                    f"{ref.artifact_id}"
                )
            try:
                selected_ids = json.loads(selection_result).get("selected_ids")
            except (AttributeError, json.JSONDecodeError) as error:
                raise ValueError(
                    "Data Selection Builder source realization is invalid: "
                    f"{ref.artifact_id}"
                ) from error
            if (
                not isinstance(selected_ids, list)
                or not selected_ids
                or any(not isinstance(item, str) or not item for item in selected_ids)
                or len(selected_ids) != len(set(selected_ids))
            ):
                raise ValueError(
                    "Data Selection Builder source realization has invalid selected_ids: "
                    f"{ref.artifact_id}"
                )
            root = (
                "sources/primary"
                if index == 0
                else f"sources/donors/{ref.artifact_id}"
            )
            files.extend((
                AgentContextFile(
                    f"{root}/selection.py",
                    self.repository.read_artifact(state.run_id, ref),
                    ref.uri,
                ),
                AgentContextFile(
                    f"{root}/selection-result.json",
                    selection_result,
                    f"{package_ref.uri}#{selection_path}",
                ),
            ))
            source_entries.append({
                "artifact_ref_id": ref.artifact_id,
                "role": "primary" if index == 0 else "donor",
                "producer_trial": (
                    f"{state.run_id}/{reference_trial.coordinator_id}/"
                    f"{reference_trial.plan_id}/{reference_trial.trial_id}"
                ),
                "selection_path": f"{root}/selection.py",
                "realization_path": f"{root}/selection-result.json",
            })
        files.append(AgentContextFile(
            "sources/manifest.json",
            _encode(source_manifest),
            f"run://{state.run_id}/trials/{trial.trial_id}#sources",
        ))
        return tuple(files)

    def _curriculum_builder_context(
        self,
        state: RunState,
        trial,
    ) -> tuple[AgentContextFile, ...]:
        inventory, stats = build_fixed_pool_from_task(state.task.config)
        rft = state.task.config.get("rft")
        judge = state.task.config.get("judge_enrichment")
        if not isinstance(rft, Mapping) or not isinstance(judge, Mapping):
            raise ValueError("Curriculum Builder task binding is incomplete")
        files: list[AgentContextFile] = [
            AgentContextFile(
                "task/candidate-inventory.json",
                _encode(inventory),
                f"run://{state.run_id}/run.json#task/candidate-inventory",
            ),
            AgentContextFile(
                "task/curriculum-contract.json",
                _encode(
                    {
                        "schema_version": "ade.curriculum_builder_contract.v1",
                        "selection_unit": "problem",
                        "total_steps": rft.get("total_training_steps"),
                        "prompts_per_step": rft.get("gen_batch_size"),
                        "rollout_n": rft.get("rollout_n"),
                        "candidate_pool": stats,
                        "judge_enrichment": dict(judge),
                    }
                ),
                f"run://{state.run_id}/run.json#task/curriculum-contract",
            ),
        ]
        sources = trial.source_artifact_ref_ids
        if not sources:
            return tuple(files)
        refs = tuple(
            _artifact(state, artifact_id, "curriculum_learning")
            for artifact_id in sources
        )
        entries: list[dict[str, object]] = []
        for index, ref in enumerate(refs):
            reference_trial = direct_reference_trial(state, ref.artifact_id)
            if reference_trial is None or not reference_trial.package_ref_id:
                raise ValueError(
                    "Curriculum Builder source Trial package is unavailable: "
                    f"{ref.artifact_id}"
                )
            package_ref = _artifact(
                state, str(reference_trial.package_ref_id), "experiment_package"
            )
            package = decode_experiment_package(
                self.repository.read_artifact(state.run_id, package_ref)
            )
            schedule_path = "experiment/realization/curriculum-schedule.json"
            schedule = package.get(schedule_path)
            if schedule is None:
                raise ValueError(
                    "Curriculum Builder source schedule is unavailable: "
                    f"{ref.artifact_id}"
                )
            root = "sources/primary" if index == 0 else f"sources/donors/{ref.artifact_id}"
            files.extend(
                (
                    AgentContextFile(
                        f"{root}/curriculum.py",
                        self.repository.read_artifact(state.run_id, ref),
                        ref.uri,
                    ),
                    AgentContextFile(
                        f"{root}/curriculum-schedule.json",
                        schedule,
                        f"{package_ref.uri}#{schedule_path}",
                    ),
                )
            )
            entries.append(
                {
                    "artifact_ref_id": ref.artifact_id,
                    "role": "primary" if index == 0 else "donor",
                    "policy_path": f"{root}/curriculum.py",
                    "schedule_path": f"{root}/curriculum-schedule.json",
                }
            )
        files.append(
            AgentContextFile(
                "sources/manifest.json",
                _encode({"schema_version": "1", "sources": entries}),
                f"run://{state.run_id}/trials/{trial.trial_id}#sources",
            )
        )
        return tuple(files)

def _trial(
    state: RunState,
    coordinator_id: str | None,
    plan_id: str | None,
    trial_id: str,
):
    matches = [
        trial
        for trial in state.trials
        if trial.coordinator_id == coordinator_id
        and trial.plan_id == plan_id
        and trial.trial_id == trial_id
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown Trial subject: {trial_id}")
    return matches[0]


def _plan(state: RunState, coordinator_id: str | None, plan_id: str):
    matches = [
        plan
        for plan in state.plans
        if plan.coordinator_id == coordinator_id and plan.plan_id == plan_id
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown Plan subject: {plan_id}")
    return matches[0]


def _artifact(state: RunState, artifact_id: str, kind: str) -> ArtifactRef:
    matches = {
        ref
        for ref in (
            *state.accepted_plan_refs,
            *state.accepted_evidence_refs,
            *state.accepted_finding_refs,
            *state.accepted_summary_refs,
        )
        if ref.artifact_id == artifact_id and ref.kind == kind
    }
    if len(matches) != 1:
        raise ValueError(f"accepted {kind} artifact is unavailable: {artifact_id}")
    return next(iter(matches))


def _snapshot(
    state: RunState,
    snapshot_id: str | None,
    kind: SnapshotKind,
    label: str,
) -> SnapshotRef:
    matches = [
        ref
        for ref in state.accepted_snapshot_refs
        if ref.snapshot_id == snapshot_id and ref.kind is kind
    ]
    if len(matches) != 1:
        raise ValueError(f"accepted {label} snapshot is unavailable")
    return matches[0]


def _normalize_experiment_package_paths(
    package: Mapping[str, bytes],
) -> dict[str, bytes]:
    """Keep legacy Analyzer packages materializable on ordinary filesystems."""
    manifest_bytes = package.get("experiment/manifest.json")
    if manifest_bytes is None:
        return dict(package)
    manifest = json.loads(manifest_bytes)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return dict(package)
    path_map: dict[str, str] = {}
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            continue
        old = artifact["path"]
        parts = PurePosixPath(old).parts
        if len(parts) < 4 or parts[:2] != ("experiment", "artifacts"):
            continue
        if len(parts[2]) <= 120:
            continue
        new = f"experiment/artifacts/artifact-{index:04d}/{parts[-1]}"
        artifact["path"] = new
        path_map[old] = new
    if not path_map:
        return dict(package)
    normalized: dict[str, bytes] = {}
    for path, content in package.items():
        normalized[path_map.get(path, path)] = content
    normalized["experiment/manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    return normalized


def _seed_plan_snapshots(state: RunState, plan) -> tuple[SnapshotRef, ...]:
    relation = plan.relation
    if relation is None:
        return ()
    if relation.kind is PlanRelationKind.NEW_DIRECTION:
        candidates = [
            item
            for item in state.plans
            if item.coordinator_id == "c000" and item.plan_id == "p000"
        ]
    else:
        # Preserve the order frozen in the relation.  ``state.plans`` is
        # append-ordered by completion, which can differ from the relation's
        # coordinator/plan ordering when plans complete asynchronously.
        candidates = []
        for key in relation.related_plan_keys:
            matches = [
                item
                for item in state.plans
                if key.run_id == state.run_id
                and key.coordinator_id == item.coordinator_id
                and key.plan_id == item.plan_id
            ]
            candidates.extend(matches)
    if not candidates or any(item.status is not PlanStatus.COMPLETED for item in candidates):
        raise ValueError("Builder seed Plans must be completed")
    expected_artifacts = tuple(item.best_artifact_ref_id for item in candidates)
    if expected_artifacts != relation.seed_artifact_ref_ids:
        raise ValueError("Builder seed artifacts do not match the frozen Plan relation")
    return tuple(
        _snapshot(
            state,
            item.latest_snapshot_ref_id,
            SnapshotKind.PLAN,
            f"seed Plan {item.coordinator_id}/{item.plan_id}",
        )
        for item in candidates
    )


def _encode(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _portfolio_target(state: RunState) -> dict[str, object]:
    candidates = [
        item
        for item in state.ranking.entries
        if item.level in {"reference", "trial_level"}
        and item.evaluation_profile == "offline"
    ]
    best = (
        min(
            candidates,
            key=lambda item: (
                *score_pair_sort_key(
                    item.score,
                    item.secondary_score,
                    direction=state.ranking.direction,
                ),
                -item.accepted_revision,
                item.subject_id,
            ),
        )
        if candidates
        else None
    )
    global_best = None
    if best is not None:
        global_best = {
            "subject_id": best.subject_id,
            "level": best.level,
            "accepted_revision": best.accepted_revision,
            "artifact_ref_id": best.artifact_ref_id,
            "artifact_digest": best.artifact_digest,
            "primary": {
                "metric_id": state.ranking.metric_id,
                "value": best.score,
            },
            "secondary": {
                "metric_id": "offline.secondary_score",
                "value": best.secondary_score,
            },
        }
    return {
        "schema_version": "1",
        "ranking_revision": state.ranking.revision,
        "evaluation_profile": "offline",
        "direction": state.ranking.direction,
        "global_best": global_best,
    }


def _trial_markdown(state: RunState, trial) -> bytes:
    score = "unavailable" if trial.offline_score is None else str(trial.offline_score)
    secondary_score = (
        "unavailable"
        if trial.offline_secondary_score is None
        else str(trial.offline_secondary_score)
    )
    return (
        "# Current Trial\n\n"
        f"- Trial: `{trial.trial_record_ref}`\n"
        f"- Phase: `{trial.phase.value}`\n"
        f"- Objective outcome: `{trial.outcome.value}`\n"
        f"- Offline metric: `{state.ranking.metric_id}`\n"
        f"- Offline score: `{score}`\n"
        f"- Offline secondary score: `{secondary_score}`\n"
        f"- Analysis status: `{trial.analysis_status or 'unavailable'}`\n"
        f"- Failure kind: `{trial.failure_kind or 'none'}`\n"
    ).encode()


def _trial_outcome(state: RunState, trial) -> bytes:
    return _encode(
        {
            "schema_version": "1",
            "trial": trial.trial_record_ref,
            "outcome": trial.outcome.value,
            "metric_id": state.ranking.metric_id,
            "offline_score": trial.offline_score,
            "offline_secondary_score": trial.offline_secondary_score,
            "analysis_status": trial.analysis_status,
            "failure_kind": trial.failure_kind,
        }
    )


def _current_trial_semantic_files(
    record_files: Mapping[str, bytes],
    *,
    engine_attempt_index: int,
) -> tuple[tuple[str, bytes, str], ...]:
    manifest = json.loads(record_files["manifest.json"])
    entries = {
        str(item["path"]): str(item["source_ref"])
        for item in manifest["files"]
        if isinstance(item, Mapping)
        and isinstance(item.get("path"), str)
        and isinstance(item.get("source_ref"), str)
    }
    targets = {
        "analysis.md": "subject/trial/analysis.md",
        "findings.md": "subject/trial/findings.md",
        "evidence.bin": "subject/trial/evidence.json",
        "review-coverage.bin": "subject/trial/review-coverage.json",
    }
    result = []
    for basename, target in targets.items():
        candidates = sorted(
            path
            for path in record_files
            if path.startswith("analysis/") and path.rsplit("/", 1)[-1] == basename
        )
        if candidates:
            source = candidates[-1]
            result.append((target, record_files[source], entries[source]))
    direct_targets = {
        "realization/final-realization.json": (
            "subject/trial/realization/final-realization.json"
        ),
        "comparisons/planning-comparators.json": (
            "subject/trial/comparisons/planning-comparators.json"
        ),
        objective_comparison_record_path(engine_attempt_index): (
            "subject/trial/comparisons/objective-comparison.json"
        ),
    }
    for source, target in direct_targets.items():
        if source in record_files:
            result.append((target, record_files[source], entries[source]))
    return tuple(result)


def _plan_catalog_markdown(
    state: RunState,
    repository: ContextRepository,
) -> bytes:
    lines = [
        "# Plan Catalog",
        "",
        f"Catalog revision: {state.plan_catalog.revision}",
        "",
        "## Accepted Plan Directions",
        "",
    ]
    accepted_plans = tuple(
        plan
        for plan in sorted(
            state.plans,
            key=lambda value: (value.coordinator_id, value.plan_id),
        )
        if plan.decision_report_ref_id is not None
        and plan.status is not PlanStatus.PROPOSED
    )
    if not accepted_plans:
        lines.append("None.")
    for plan in accepted_plans:
        report_ref = _artifact(
            state,
            str(plan.decision_report_ref_id),
            "planning_decision_report",
        )
        report = repository.read_artifact(state.run_id, report_ref).decode("utf-8")
        lines.extend(
            (
                f"### {state.run_id}/{plan.coordinator_id}/{plan.plan_id}",
                "",
                report.strip(),
                "",
            )
        )
    lines.extend(
        (
            "## Active Plan Intents",
            "",
        )
    )
    if not state.plan_catalog.active_intents:
        lines.append("None.")
    for item in state.plan_catalog.active_intents:
        related = ", ".join(item.related_plan_keys) or "none"
        lines.append(
            f"- {state.run_id}/{item.coordinator_id}/{item.plan_id}: relation={item.relation_kind}; "
            f"related={related}; accepted_revision={item.accepted_revision}; "
            f"plan_ref={item.decision_report_ref_id or item.decision_ref_id}"
        )
    lines.extend(("", "## Source-Eligible Results", ""))
    if not state.plan_catalog.source_eligible_results:
        lines.append("None.")
    for item in state.plan_catalog.source_eligible_results:
        lines.append(
            f"- {state.run_id}/{item.coordinator_id}/{item.plan_id}: "
            f"trial={state.run_id}/{item.coordinator_id}/{item.plan_id}/"
            f"{item.representative_trial_id}; score={item.score}; "
            f"secondary_score={item.secondary_score}; "
            f"profile={item.evaluation_profile}; artifact={item.artifact_ref_id}; "
            f"rm={item.rm_version}; archived_revision={item.archived_revision}"
        )
    return ("\n".join(lines) + "\n").encode()


def _ranking_markdown(state: RunState) -> bytes:
    lines = [
        "# Offline Ranking",
        "",
        f"Ranking revision: {state.ranking.revision}",
        f"Metric: `{state.ranking.metric_id}` ({state.ranking.direction})",
        "",
    ]
    if not state.ranking.entries:
        lines.append("No accepted ranking entries.")
    for item in state.ranking.entries:
        lines.append(
            f"- {item.subject_id}: score={item.score}; "
            f"secondary_score={item.secondary_score}; level={item.level}; "
            f"accepted_revision={item.accepted_revision}; "
            f"source_eligible={item.source_eligible}"
        )
    return ("\n".join(lines) + "\n").encode()


def _plan_update_markdown(state: RunState, trial, queue_head: str) -> bytes:
    return (
        "# Plan Memory Update\n\n"
        f"- Queue entry: `{queue_head}`\n"
        f"- Plan: `{state.run_id}/{trial.coordinator_id}/{trial.plan_id}`\n"
        f"- Trial: `{trial.trial_record_ref}`\n"
        f"- Plan Memory: `{trial.plan_memory_result}`\n"
        f"- Run Memory parent: `{state.memory.run_head}`\n"
        f"- Basis revision: `{state.revision}`\n"
    ).encode()


def _run_facts_manifest(state: RunState, catalog: bytes, ranking: bytes) -> bytes:
    return _encode(
        {
            "schema_version": "1",
            "identity": f"{state.run_id}@rev-{state.revision}",
            "basis_revision": state.revision,
            "files": {
                "PLAN_CATALOG.md": hashlib.sha256(catalog).hexdigest(),
                "RANKING.md": hashlib.sha256(ranking).hexdigest(),
            },
        }
    )


def _task_config_file(state: RunState) -> AgentContextFile:
    return AgentContextFile(
        "task/resolved.json",
        _encode(state.task.config),
        f"run://{state.run_id}/run.json#task/config",
    )


def _data_selection_training_context(state: RunState) -> tuple[AgentContextFile, ...]:
    data = state.task.config.get("data")
    source_path = data.get("fixed_training_data") if isinstance(data, Mapping) else None
    if not isinstance(source_path, str) or not source_path:
        return ()
    source = Path(source_path).resolve()
    if not source.is_file():
        raise ValueError(f"Data Selection fixed training data is unavailable: {source}")
    return (
        AgentContextFile(
            "task/training-data.jsonl",
            source.read_bytes(),
            f"file://{source}",
        ),
    )


def _coordinator_budget_file(state: RunState) -> AgentContextFile:
    """Expose live planning capacity even before the first run summary."""
    search_plans = sum(item.kind is PlanKind.SEARCH for item in state.plans)
    search_trials = sum(item.kind is TrialKind.SEARCH for item in state.trials)
    payload = {
        "schema_version": "1",
        "max_search_plans": state.portfolio.max_plans,
        "allocated_search_plans": search_plans,
        "remaining_search_plans": max(0, state.portfolio.max_plans - search_plans),
        "max_search_trials": state.portfolio.max_trials,
        "allocated_search_trials": search_trials,
        "remaining_search_trials": max(0, state.portfolio.max_trials - search_trials),
        "min_trials_per_plan": state.portfolio.min_trials_per_plan,
        "max_trials_per_plan": state.portfolio.max_trials_per_plan,
        "allocated_plan_trials": 0,
        "remaining_plan_trials": state.portfolio.max_trials_per_plan,
    }
    return AgentContextFile(
        "subject/budget.json",
        _encode(payload),
        f"run://{state.run_id}/run.json#budget",
    )


def _snapshot_context_files(
    ref: SnapshotRef,
    logical_root: str,
) -> tuple[AgentContextFile, ...]:
    root = Path(ref.root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declarations = manifest.get("files")
    if not isinstance(declarations, list):
        raise ValueError(f"snapshot manifest has no files: {ref.snapshot_id}")
    relative_paths = ("manifest.json", *(
        str(item["path"])
        for item in declarations
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    ))
    result = []
    for relative_path in relative_paths:
        relative = PurePosixPath(relative_path)
        source = root.joinpath(*relative.parts)
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"snapshot file is unavailable: {source}")
        result.append(
            AgentContextFile(
                f"{logical_root}/{relative.as_posix()}",
                source.read_bytes(),
                ref.snapshot_id,
            )
        )
    return tuple(result)


def _agent_subject_identity(
    *,
    run_id: str,
    role: AgentRole,
    subject_id: str,
    coordinator_id: str | None,
    plan_id: str | None,
) -> tuple[dict[str, str], str]:
    scope = {"run_id": run_id}
    if role is AgentRole.RUN_SUMMARIZER:
        if subject_id != run_id or coordinator_id is not None or plan_id is not None:
            raise ValueError("Run Summarizer identity is not Run-scoped")
    elif role is AgentRole.COORDINATOR:
        if coordinator_id is not None or plan_id is not None:
            raise ValueError("Coordinator identity cannot include Plan scope")
        scope["coordinator_id"] = subject_id
    elif role is AgentRole.PLAN_SUMMARIZER:
        if coordinator_id is None or plan_id is not None:
            raise ValueError("Plan Summarizer identity requires Coordinator scope")
        scope.update(coordinator_id=coordinator_id, plan_id=subject_id)
    else:
        if coordinator_id is None or plan_id is None:
            raise ValueError("Trial Agent identity requires Plan scope")
        scope.update(
            coordinator_id=coordinator_id,
            plan_id=plan_id,
            trial_id=subject_id,
        )
    return scope, "/".join(scope.values())
