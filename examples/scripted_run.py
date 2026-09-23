"""A local scripted Data Selection trial through ADE's real workflow.

Derived from the existing scripted workflow fixtures. Agent, training and Review
responses are deterministic; Control, artifact compilation, queues, revisions,
Trial Records and Memory publication use the production implementation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

from ade.agent_runtime.context import RoleContextPackager
from ade.agent_runtime.runtime import AgentRuntime
from ade.agent_runtime.service import AgentCallService
from ade.agent_runtime.skills import SkillResolver
from ade.agent_runtime.workspace import WorkspaceManager
from ade.controller.agent_port import CoordinatorAgentPort
from ade.controller.control import ControlLoop
from ade.controller.executor import Executor
from ade.controller.planner import Planner
from ade.controller.reducer import Reducer
from ade.controller.trial_lifecycle import TrialLifecycle
from ade.controller.workflow_driver import WorkflowDriver
from ade.core.coordinator import CoordinatorKind, CoordinatorState
from ade.core.insight import InsightGraph
from ade.core.plan import PlanKind, PlanState, PlanStatus
from ade.core.ranking import RankingEntry, RankingState
from ade.core.run import PortfolioState, ResolvedTask, RunState, RunStatus
from ade.core.snapshot import SnapshotFile, SnapshotKind
from ade.core.trial import TrialState, TrialKind, TrialPhase, TrialOutcome, TrialArchiveStatus
from ade.agent_runtime.experiment_package import encode_experiment_package
from ade.engine.command_queue import FileCommandQueue
from ade.engine.storage.object_store import FileEngineObjectStore
from ade.engine.trial_artifacts import TrialArtifactPublisher
from ade.engine.worker import EngineWorker
from ade.memory.repository import FileRunRepository
from ade.review_labor.command_queue import FileReviewCommandQueue
from ade.review_labor.worker import ReviewWorker
from ade.tasks.data_selection.handler import SFTExecutor
from ade.tasks.registry import default_task_registry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "scripted-demo"


def _test_analysis_policy():
    value = {
        "profile_id": "smoke.v1",
    }
    value["policy_digest"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return value

def _write_summary_delivery(attempt, action, role):
    is_plan = role == "plan_summarizer"
    sections = (
        (
            ("# Plan Summary", ""),
            ("## Direction and Hypothesis", "One controlled direction."),
            ("## Offline Evaluation Progress", "One offline result."),
            ("## Trial Findings", "The Trial completed analysis."),
            ("## Assessment", "The scripted lifecycle completed."),
            ("## Contradictions and Uncertainty", "The scores are synthetic."),
            (
                "## Guidance for the Next Trial",
                "Replicate the controlled result.",
            ),
        )
        if is_plan
        else (
            ("# Run Summary", ""),
            ("## Current Research State", "The scripted lifecycle completed."),
            ("## Plan Assessments", "The current Plan is assessed."),
            ("## Cross-Plan Findings", "Only one Plan exists."),
            ("## Contradictions and Uncertainty", "The scores are synthetic."),
            (
                "## Guidance for the Next Plan",
                "Review the next controlled Plan.",
            ),
        )
    )
    realization_path = (
        attempt / "input/subject/trial/realization/final-realization.json"
    )
    if is_plan and realization_path.is_file():
        realization = json.loads(
            realization_path.read_text(encoding="utf-8")
        )["realization_status"]
        objective = json.loads(
            (
                attempt
                / "input/subject/trial/comparisons/objective-comparison.json"
            ).read_text(encoding="utf-8")
        )
        relation = objective["hypothesis_comparator"]["result"]
        hypothesis = {
            "improved": "supported",
            "regressed": "rejected",
            "tied": "inconclusive",
            "not_comparable": "inconclusive",
        }[relation]
        portfolio = objective["portfolio_comparator"]["result"]
    elif not is_plan and (
        attempt / "input/subject/plan-memory/MEMORY.md"
    ).is_file():
        plan_memory = (
            attempt / "input/subject/plan-memory/MEMORY.md"
        ).read_text(encoding="utf-8")
        conclusion_lines = {
            line.split(": ", 1)[0]: line.split(": ", 1)[1]
            for line in plan_memory.splitlines()
            if line.startswith(
                ("Realization status: ", "Hypothesis result: ", "Portfolio result: ")
            )
        }
        if "Realization status" in conclusion_lines:
            realization = conclusion_lines["Realization status"]
            hypothesis = conclusion_lines["Hypothesis result"]
            portfolio = conclusion_lines["Portfolio result"]
        else:
            realization = None
    else:
        realization = None
    if realization is not None:
        sections = (
            *sections,
            (
                "## Trial Conclusions",
                f"Realization status: {realization}\n"
                f"Hypothesis result: {hypothesis}\n"
                f"Portfolio result: {portfolio}",
            ),
        )
    output = attempt / "output"
    output.joinpath("MEMORY.md").write_text(
        "\n".join(
            heading if not body else f"{heading}\n{body}"
            for heading, body in sections
        ) + "\n",
        encoding="utf-8",
    )


class ScriptedAgent:
    def __init__(self):
        self.roles = Counter()

    def invoke(self, *, skill, attempt, feedback_paths):
        action = json.loads((attempt / "input/action.json").read_text())
        role = action["role"]
        self.roles[role] += 1
        output = attempt / "output"
        if role == "coordinator":
            _write_json(output / "decision.json", {
                "schema_version": "1",
                "relation": {"kind": "new_direction", "related_plans": []},
                "design": {"comparisons": {"hypothesis": {
                    "reason": "Compare against the synthetic baseline.",
                    "expected_observation": {
                        "primary": "strict_improvement", "secondary": "no_regression"
                    },
                }}},
            })
            (output / "plan.md").write_text(
                "# Data Selection Plan\n\nExercise one scripted selection trial.\n"
            )
        elif role == "artifact_builder":
            (output / "selection.py").write_text(
                "async def select_trajectories(candidate_inventory, select_size, judge):\n"
                "    # Scripted search proposal; the candidate pool contains one row.\n"
                "    return ['openthoughts-0000']\n"
            )
            reflection = (
                "\nReflection decision: finalize\n"
                if (attempt / "input/reflection").is_dir() else ""
            )
            (output / "design.md").write_text(
                "# Selection Design\n\nSelect the single synthetic example while "
                "holding model, training configuration and seed fixed.\n" + reflection
            )
        elif role == "analyzer":
            write_analyzer_delivery(attempt, task_id="data_selection")
        else:
            _write_summary_delivery(attempt, action, role)


class _FakeReviewProcessor:
    def execute(self, command):
        return {
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
            "status": "complete",
            "batches": [
                {
                    "batch_id": batch.batch_id,
                    "pool": batch.pool,
                    "investigation_purpose": batch.investigation_purpose,
                    "review_id": f"review-{batch.batch_id}",
                    "status": "completed",
                    "requested_units": len(batch.units),
                    "completed_units": len(batch.units),
                    "failed_units": 0,
                    "results": [],
                }
                for batch in command.batches
            ],
            "usage": {},
        }

class _Workers:
    def __init__(self, engine, review):
        self.engine = engine
        self.review = review
        self.queue = engine.queue
        self.handlers = engine.handlers

    def run_once(self):
        return self.engine.run_once() or self.review.run_once()

def create_fixture(tmp_path):
    """Create a synthetic accepted baseline; no baseline training is performed."""
    run_id = RUN_ID
    task_id = "data_selection"
    coordinator_count = 1
    input_root = tmp_path / f"{run_id}-inputs"
    input_root.mkdir()
    source = input_root / "fixed_training_data.jsonl"
    source.write_text(
        json.dumps(
            {
                "conversations": [
                    {"from": "user", "value": "Compute 1 + 1."},
                    {
                        "from": "assistant",
                        "value": "<think>One plus one is two.</think> 2",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    tokenizer_root = input_root / "tokenizer"
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "[BOS]": 1,
                "[EOS]": 2,
                "Compute": 3,
                "1": 4,
                "+": 5,
                ".": 6,
                "One": 7,
                "plus": 8,
                "one": 9,
                "is": 10,
                "two": 11,
                "2": 12,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    fast.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] + ': ' + message['content'] + '\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    fast.save_pretrained(tokenizer_root)
    task_config = {
        "base_model": str(tokenizer_root),
        "select_size": 1,
        "data": {
            "fixed_training_data": str(source),
            "pool_size": 1,
            "dataset_name": "selected",
            "validation": [
                {
                    "system_prompt": {
                        "content": "Solve the problem carefully."
                    }
                }
            ],
        },
        "prompt_protocol": {
            "mode": "chat_template",
            "system_prompt": {
                "content": "Solve the problem carefully."
            },
        },
        "judge_enrichment": {"enabled": False},
    }

    baseline_files = (
        SnapshotFile.generated(
            "MEMORY.md",
            b"# Plan Summary\n\nCompleted baseline.\n",
            source_id="baseline-summary",
        ),
        SnapshotFile.generated(
            "evidence.json",
            b'{"schema_version":"1","sources":[]}\n',
            source_id="baseline-summary",
        ),
        SnapshotFile.generated(
            "state.json",
            b'{"status":"completed","best_artifact_ref_id":"artifact-baseline"}\n',
            source_id="baseline-state",
        ),
    )
    baseline_run_files = (
        SnapshotFile.generated(
            "MEMORY.md",
            b"# Run Summary\n\nCompleted baseline.\n",
            source_id="baseline-run-summary",
        ),
        SnapshotFile.generated(
            "evidence.json",
            b'{"schema_version":"1","sources":[]}\n',
            source_id="baseline-run-summary",
        ),
        SnapshotFile.generated(
            "state.json",
            b'{"status":"created","completed_plans":["c000/p000"]}\n',
            source_id="baseline-run-state",
        ),
        SnapshotFile.generated(
            "plans/c000/p000/MEMORY.md",
            b"# Plan Summary\n\nCompleted baseline.\n",
            source_id="baseline-summary",
        ),
    )
    preflight = FileRunRepository(tmp_path / f"{run_id}-snapshot-preflight")
    preflight_ref = preflight.snapshots.materialize(
        kind=SnapshotKind.PLAN,
        run_id=run_id,
        coordinator_id="c000",
        plan_id="p000",
        revision=0,
        files=baseline_files,
    )
    preflight_run_ref = preflight.snapshots.materialize(
        kind=SnapshotKind.RUN,
        run_id=run_id,
        revision=0,
        files=baseline_run_files,
    )
    repository = FileRunRepository(tmp_path / "runs")
    baseline_content = (
        b"async def select_trajectories(candidate_inventory, select_size, judge):\n"
        b"    return ['openthoughts-0000']\n"
        if task_id == "data_selection"
        else b"def compute_score(**kwargs):\n    return 0.0\n"
    )
    baseline_artifact = repository.describe_artifact(
        run_id, task_id, baseline_content
    )
    baseline_package_content = encode_experiment_package({
        "experiment/manifest.json": json.dumps({
            "schema_version": "ade.trial_artifacts.v1",
            "run_id": run_id, "coordinator_id": "c000",
            "plan_id": "p000", "trial_id": "t000",
        }).encode(),
        "experiment/realization/selection-result.json": b'{"selected_ids":["openthoughts-0000"]}\n',
    })
    baseline_package = repository.describe_artifact(
        run_id, "experiment_package", baseline_package_content
    )
    baseline_snapshot = replace(
        preflight_ref,
        root=str(
            repository.layout.plan_snapshot_dir(
                run_id,
                "c000",
                "p000",
                0,
            ).resolve()
        ),
    )
    baseline_run_snapshot = replace(
        preflight_run_ref,
        root=str(repository.layout.run_snapshot_dir(run_id, 0).resolve()),
    )
    resolved_portfolio = PortfolioState(1, 1)
    if resolved_portfolio.max_plans % coordinator_count:
        raise ValueError("Plan budget must divide across Coordinators")
    plans_per_coordinator = resolved_portfolio.max_plans // coordinator_count
    repository.create(
        RunState(
            run_id,
            0,
            RunStatus.RUNNING,
            ResolvedTask(
                task_id,
                task_id,
                config=task_config,
            ),
            resolved_portfolio,
            InsightGraph(0),
            RankingState(
                "offline.score",
                entries=(
                    RankingEntry(
                        f"{run_id}/c000/p000/t000",
                        0.0,
                        level="trial_level",
                        evaluation_profile="offline",
                        secondary_score=0.0,
                        artifact_ref_id=baseline_artifact.artifact_id,
                        artifact_digest=baseline_artifact.digest,
                        source_eligible=True,
                    ),
                ),
            ),
            coordinators=(
                CoordinatorState("c000", CoordinatorKind.BOOTSTRAP, 0, 0),
                *(
                    CoordinatorState(
                        f"c{index:03d}",
                        CoordinatorKind.SEARCH,
                        plans_per_coordinator,
                        plans_per_coordinator,
                    )
                    for index in range(1, coordinator_count + 1)
                ),
            ),
            plans=(
                PlanState(
                    "p000",
                    "c000",
                    kind=PlanKind.BOOTSTRAP,
                    status=PlanStatus.COMPLETED,
                    best_trial_id="t000",
                    best_artifact_ref_id=baseline_artifact.artifact_id,
                    best_score=0.0,
                    latest_snapshot_ref_id=baseline_snapshot.snapshot_id,
                ),
            ),
            accepted_evidence_refs=(baseline_artifact, baseline_package),
            trials=(TrialState(
                "t000", "c000", "p000",
                kind=TrialKind.BOOTSTRAP_BASELINE,
                phase=TrialPhase.ARCHIVED,
                outcome=TrialOutcome.SUCCEEDED,
                archive_status=TrialArchiveStatus.ARCHIVED,
                artifact_ref_id=baseline_artifact.artifact_id,
                package_ref_id=baseline_package.artifact_id,
                offline_score=0.0,
                offline_secondary_score=0.0,
                plan_snapshot_ref_id=baseline_snapshot.snapshot_id,
                run_snapshot_ref_id=baseline_run_snapshot.snapshot_id,
            ),),
            accepted_snapshot_refs=(baseline_snapshot, baseline_run_snapshot),
            latest_run_snapshot_ref_id=baseline_run_snapshot.snapshot_id,
            analysis_policy=_test_analysis_policy(),
        ),
        initial_artifacts=((baseline_artifact, baseline_content),
                           (baseline_package, baseline_package_content)),
    )
    assert repository.snapshots.materialize(
        kind=SnapshotKind.PLAN,
        run_id=run_id,
        coordinator_id="c000",
        plan_id="p000",
        revision=0,
        files=baseline_files,
    ) == baseline_snapshot
    assert repository.snapshots.materialize(
        kind=SnapshotKind.RUN,
        run_id=run_id,
        revision=0,
        files=baseline_run_files,
    ) == baseline_run_snapshot

def assemble(tmp_path, backend):
    repository = FileRunRepository(tmp_path / "runs")
    tasks = default_task_registry()
    queue = FileCommandQueue(tmp_path / "queue")
    review_queue = FileReviewCommandQueue(tmp_path / "review-queue")
    objects = FileEngineObjectStore(tmp_path / "objects")
    calls = AgentCallService(
        runtime=AgentRuntime(
            skills=SkillResolver(PROJECT_ROOT / ".agents/skills"),
            workspaces=WorkspaceManager.for_runs_root(
                tmp_path / "runs",
                trial_artifacts_root=tmp_path / "trial-artifacts",
            ),
            backend=backend,
        ),
        contexts=RoleContextPackager(repository=repository, tasks=tasks),
        tasks=tasks,
        max_retries=1,
        execution_mode="inline",
    )
    lifecycle = TrialLifecycle(
        repository=repository,
        tasks=tasks,
        queue=queue,
        engine_io=objects,
        review_queue=review_queue,
    )
    control = ControlLoop(
        repository=repository,
        planner=Planner(),
        executor=Executor(
            agent_port=CoordinatorAgentPort(
                calls=calls,
                repository=repository,
                tasks=tasks,
            )
        ),
        reducer=Reducer(),
    )
    driver = WorkflowDriver(
        repository=repository,
        calls=calls,
        control=control,
        trials=lifecycle,
        engine_inputs={
            "data_selection": {
                "schema_version": 1,
                "sft": {
                    "max_steps": 1,
                    "evaluation_requests": {
                        "online_validation": {},
                        "offline_validation": {},
                    },
                    "dataset": {
                        "dataset_name": "selected",
                    },
                    "request": {
                        "judge_enrichment": {"enabled": False},
                        "training_prompt_contract": {},
                    },
                },
            }
        },
    )

    worker = _Workers(
        _worker(tmp_path),
        ReviewWorker(review_queue, _FakeReviewProcessor()),
    )
    return driver, worker, repository, backend

def _worker(tmp_path):
    class FakeSFTBackend:
        def train(self, command, config):
            return {
                "model_ref": "model://scripted-sft",
                "metrics": {"loss": 0.2},
            }

        def evaluate(self, command, config, checkpoint, purpose, step):
            return {
                "status": "complete",
                "score": 0.7,
                "checkpoint": checkpoint,
                "step": step,
            }

    objects = FileEngineObjectStore(tmp_path / "objects")
    return EngineWorker(
        queue=FileCommandQueue(tmp_path / "queue"),
        handlers={
            "train_sft": SFTExecutor(
                io=objects,
                backend=FakeSFTBackend(),
                artifacts=TrialArtifactPublisher(tmp_path / "trial-artifacts"),
            ).execute
        },
    )

def write_analyzer_delivery(
    attempt: Path,
    *,
    task_id: str,
    artifact_ids: tuple[str, ...] | None = None,
) -> None:
    """Write the two semantic files required from a real Analyzer agent."""
    action = json.loads((attempt / "input" / "action.json").read_text(encoding="utf-8"))
    manifest_path = attempt / "input" / "experiment" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = {
        str(item["id"]): item
        for item in manifest.get("artifacts", ())
        if isinstance(item, dict) and item.get("id")
    }
    catalog = json.loads(
        (attempt / "input" / "experiment" / "evidence-catalog.json").read_text(
            encoding="utf-8"
        )
    )

    if action.get("stage") == "review_design":
        batches = []
        for pool in catalog["pools"]:
            pool_id = str(pool["pool_id"])
            records = list(pool["records"])
            if not records:
                continue
            coverage = pool["coverage"]
            if coverage["mode"] == "all":
                selection = {"mode": "all"}
            else:
                required = math.ceil(
                    len(records) * float(coverage["fraction"])
                )
                selected = records[:required]
                selection = {
                    "mode": "records",
                    "source_artifact_ids": sorted(
                        {str(item["source_artifact_id"]) for item in selected}
                    ),
                    "record_ids": [
                        str(item["source_record_id"]) for item in selected
                    ],
                }
            batches.append(
                {
                    "batch_id": f"{pool_id}-main",
                    "pool": pool_id,
                    "investigation_purpose": f"Inspect {pool_id} behavior.",
                    "selection": selection,
                    "rubrics": [
                        {
                            "rubric_id": "quality",
                            "instruction": "Inspect the response quality.",
                            "labels": ["ok", "bad"],
                        }
                    ],
                }
            )
        _write_json(
            attempt / "output" / "review-plan.json",
            {
                "schema_version": "ade.analysis_review_plan.v2",
                "hypotheses": ["Representative behavior may expose a quality gap."],
                "batches": batches,
            },
        )
        return

    if action.get("stage") == "synthesis":
        packet = json.loads(
            (attempt / "input" / "review" / "packet.json").read_text(
                encoding="utf-8"
            )
        )
        cited_artifacts = (
            sorted(artifact_ids)
            if artifact_ids is not None
            else sorted(artifacts)[:1]
        )
        cited_reviews = sorted(
            str(item["review_id"])
            for item in packet.get("batches", ())
            if item.get("review_id")
        )
        citations = " ".join(
            [
                *(f"[[artifact:{item}]]" for item in cited_artifacts),
                *(f"[[review:{item}]]" for item in cited_reviews),
            ]
        )
        headings = {
            "data_selection": _DATA_HEADINGS,
        }[task_id]
        report = "\n".join(
            [headings[0], *[f"{heading}\nScripted analysis (synthetic scores) {citations}" for heading in headings[1:]]]
        )
        output = attempt / "output"
        (output / "analysis.md").write_text(report + "\n", encoding="utf-8")
        (output / "findings.md").write_text(
            "# Findings Delta\n\nNew evidence: " + citations + "\n",
            encoding="utf-8",
        )
        return

    raise ValueError(f"unsupported Analyzer stage: {action.get('stage')}")

def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

_DATA_HEADINGS = (
    "# SFT Data Selection Trial Analysis",
    "## Analysis Scope and Evidence",
    "## Direct Inspection",
    "## Data Selection Diagnosis",
    "## Training and Validation Response",
    "## Findings",
    "## Contradictions and Uncertainty",
    "## Recommendations",
)

def run(output: Path) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    create_fixture(output)
    driver, worker, repository, backend = assemble(output, ScriptedAgent())
    rows = []
    for _ in range(100):
        before = repository.load(RUN_ID)
        driver.reconcile_once(RUN_ID)
        state = repository.load(RUN_ID)
        if state.revision != before.revision:
            rows.append((state.revision, state.last_transition.kind))
            print(f"{state.revision:02d} {state.last_transition.kind}")
        if state.status is RunStatus.COMPLETED:
            break
        if state.status in {RunStatus.FAILED, RunStatus.SUSPENDED, RunStatus.CANCELLED}:
            raise RuntimeError(f"Demo stopped: {state.status.value}; inspect {output}")
        worker.run_once()
    else:
        raise RuntimeError(f"Demo did not finish; inspect {output}")
    # Read the committed result back through a new repository instance.
    state = FileRunRepository(output / "runs").load(RUN_ID)
    trial = next(t for t in state.trials if t.kind.value == "search")
    if trial.archive_status.value != "archived" or trial.outcome is not TrialOutcome.SUCCEEDED:
        raise RuntimeError("The scripted Trial did not succeed and archive")
    record = repository.layout.trial_record_dir(RUN_ID, trial.coordinator_id, trial.plan_id, trial.trial_id)
    memory_files = sorted(repository.layout.run_dir(RUN_ID).glob("memory/**/MEMORY.md"))
    if not record.is_dir() or not memory_files:
        raise RuntimeError("Trial Record or Memory publication is missing")
    with (output / "transitions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("revision", "transition"))
        writer.writerows(rows)
    result = {
        "synthetic": True,
        "run_id": RUN_ID,
        "status": state.status.value,
        "revision": state.revision,
        "run_state": str(repository.layout.revision_snapshot_path(RUN_ID, state.revision)),
        "trial_outcome": trial.outcome.value,
        "trial_record": str(record),
        "memory_files": [str(p) for p in memory_files],
        "agent_calls": dict(backend.roles),
        "transitions": str(output / "transitions.csv"),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs/quickstart")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new directory; choose another path to keep prior results")
    run(args.output)


if __name__ == "__main__":
    main()
