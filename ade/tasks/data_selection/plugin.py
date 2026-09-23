"""Data Selection Task Plugin."""

from __future__ import annotations

import asyncio
import json

from ade.core.engine import TrainSFTCommand
from ade.tasks.contracts import (
    AnalyzerArtifactSelector,
    AnalyzerCoverageSpec,
    AnalyzerEvidenceSpec,
    AnalyzerGroupingSpec,
    AnalyzerReviewFields,
    ArtifactSpec,
    ArtifactCompilationError,
    RankingSpec,
)
from ade.core.validation import DeliveryViolation, ValidationReport
from ade.tasks.data_selection.baseline import build_baseline
from ade.tasks.data_selection.compiler import compile_artifact
from ade.tasks.data_selection.engine_binding import bind_engine_artifact
from ade.tasks.data_selection.fixed_pool import build_fixed_pool_from_task
from ade.tasks.data_selection.realizer import SelectionRealizer
from ade.tasks.data_selection.role_contracts import role_contracts
from ade.tasks.plugin import ArtifactAcceptanceContext, TaskPlugin
from ade.rubric_jobs.process import canonical_process_rubric


def _prior_judge_evidence(context: ArtifactAcceptanceContext) -> dict[str, dict[str, object]]:
    path = context.delivery.workspace / "input/reflection/judge-evidence.jsonl"
    if not path.is_file():
        return {}
    evidence: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("request"), dict):
            raise ValueError("prior Data Selection Judge evidence is invalid")
        key = json.dumps(row["request"], ensure_ascii=False, sort_keys=True)
        if key in evidence or not isinstance(row.get("evidence"), dict):
            raise ValueError("prior Data Selection Judge evidence is ambiguous")
        evidence[key] = row
    return evidence


def validate_artifact_acceptance(context: ArtifactAcceptanceContext):
    """Execute the admitted selector against the frozen full candidate pool."""
    try:
        inventory, training, pool_stats = build_fixed_pool_from_task(
            context.state.task.config
        )
        select_size = context.state.task.config.get("select_size")
        if type(select_size) is not int or select_size <= 0:
            raise ValueError("Data Selection task requires positive select_size")
        enrichment = context.state.task.config.get("judge_enrichment")
        judge_enabled = (
            enrichment.get("enabled") if isinstance(enrichment, dict) else None
        )
        if type(judge_enabled) is not bool:
            raise ValueError("task.judge_enrichment.enabled must be explicit boolean")

        prior = _prior_judge_evidence(context)
        used: dict[str, dict[str, object]] = {}
        cache_hits = 0
        dispatcher = None
        judge_unavailable = False
        if judge_enabled and context.lifecycle.selection_harness_judge_factory:
            dispatcher = context.lifecycle.selection_harness_judge_factory(
                context.state, context
            )
        elif judge_enabled:
            judge_unavailable = True

        async def judge_batch(
            requests: list[dict[str, object]],
        ) -> list[dict[str, object]]:
            nonlocal cache_hits
            normalized = [
                {
                    "question": str(request["question"]),
                    "response": str(request["response"]),
                    "rubric": canonical_process_rubric(request["rubric"]),
                }
                for request in requests
            ]
            results: list[dict[str, object] | None] = [None] * len(normalized)
            missing: list[dict[str, str]] = []
            missing_indexes: list[int] = []
            for index, request in enumerate(normalized):
                key = json.dumps(request, ensure_ascii=False, sort_keys=True)
                cached = prior.get(key)
                if cached is not None:
                    cache_hits += 1
                    evidence = dict(cached["evidence"])
                    results[index] = evidence
                    used[key] = {"request": request, "evidence": evidence}
                else:
                    missing.append(request)
                    missing_indexes.append(index)
            if missing:
                fresh = (
                    await dispatcher.evaluate(missing)
                    if dispatcher is not None
                    else [
                        {
                            "status": "unavailable",
                            "scores_by_dimension": {},
                            "projected_score": 0.0,
                            "fallback": True,
                            "fallback_reason": "external_evidence_unavailable",
                            "fallback_message": "Run-owned Local Judge is unavailable",
                        }
                        for _request in missing
                    ]
                )
                for index, request, evidence in zip(
                    missing_indexes, missing, fresh, strict=True
                ):
                    results[index] = dict(evidence)
                    key = json.dumps(request, ensure_ascii=False, sort_keys=True)
                    used[key] = {"request": request, "evidence": dict(evidence)}
            if any(result is None for result in results):
                raise RuntimeError("Data Selection Judge evidence join is incomplete")
            return [dict(result) for result in results if result is not None]

        async def realize():
            try:
                return await SelectionRealizer().realize(
                    context.compiled.content,
                    inventory,
                    training,
                    select_size=select_size,
                    judge_batch=judge_batch if judge_enabled else None,
                )
            finally:
                if dispatcher is not None:
                    await dispatcher.close()

        realized = asyncio.run(realize())
        dispatch_stats = (
            dict(dispatcher.stats)
            if dispatcher is not None
            else {
                "requested": len(used),
                "completed": 0,
                "fallback": len(used) if judge_enabled else 0,
                "fallback_reasons": (
                    {"external_evidence_unavailable": len(used)}
                    if judge_enabled and used
                    else {}
                ),
                "batches": 0,
            }
        )
        evidence_rows = [used[key]["evidence"] for key in sorted(used)]
        judge_stats = {
            **dispatch_stats,
            "evidence_count": len(evidence_rows),
            "cache_reused": cache_hits,
            "completed_total": sum(
                row.get("status") == "completed" and not row.get("fallback")
                for row in evidence_rows
            ),
            "fallback_total": sum(bool(row.get("fallback")) for row in evidence_rows),
        }
        all_requested_evidence_unavailable = (
            judge_enabled
            and bool(evidence_rows)
            and int(judge_stats["completed_total"]) == 0
        )
        external_unavailable = (
            judge_unavailable or all_requested_evidence_unavailable
        )
        realization_status = (
            "unverified"
            if external_unavailable
            else "verified"
        )
        realization_reason = (
            "external_evidence_unavailable"
            if external_unavailable
            else None
        )
        content = (
            json.dumps(
                {
                    "schema_version": "ade.data_selection_realization.v1",
                    "realization_status": realization_status,
                    "reason": realization_reason,
                    "selection": realized.selection,
                    "selected_rows": list(realized.selected_rows),
                    "candidate_pool": {
                        "source": context.state.task.config["data"][
                            "fixed_training_data"
                        ],
                        "stats": pool_stats,
                    },
                    "judge_binding": {
                        "enabled": judge_enabled,
                        "model_digest": (
                            context.state.run_resources["local_judge"][
                                "model_digest"
                            ]
                            if judge_enabled
                            and context.state.run_resources is not None
                            else None
                        ),
                        "stats": judge_stats,
                    },
                    "judge_evidence": [used[key] for key in sorted(used)],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        return context.lifecycle.repository.put_artifact(
            context.run_id, "data_selection_realization", content
        )
    except ArtifactCompilationError:
        raise
    except Exception as error:
        raise ArtifactCompilationError(
            ValidationReport(
                (
                    DeliveryViolation(
                        "data_selection_realization_failed",
                        (
                            "Data Selection realization failed: "
                            f"{type(error).__name__}: {error}"
                        ),
                        "selection.py",
                        repairable=True,
                    ),
                )
            )
        ) from error


def prepare_engine_config(config, state):
    """Bind the admitted Run's Judge to an isolated SFT command configuration."""
    import copy

    prepared = copy.deepcopy(config)
    sft_config = prepared.get("sft")
    if not isinstance(sft_config, dict):
        raise ValueError("Data Selection Engine config requires sft")
    if state.run_resources is not None:
        request = sft_config.get("request")
        if not isinstance(request, dict):
            raise ValueError("Data Selection Engine config requires sft.request")
        request["local_judge"] = copy.deepcopy(state.run_resources["local_judge"])
    return prepared

plugin = TaskPlugin(
    task_id="data_selection",
    command_type=TrainSFTCommand,
    artifact=ArtifactSpec("data_selection", True),
    ranking=RankingSpec("offline.ranking_score"),
    analyzer_evidence_specs=(
        AnalyzerEvidenceSpec(
            pool_id="selected_examples",
            semantic_role="complete training subset produced by selection logic",
            artifact_selector=AnalyzerArtifactSelector(("audit",), ("selected_examples",)),
            coverage=AnalyzerCoverageSpec("all"),
            review_fields=AnalyzerReviewFields("question", "response"),
            context_fields=("candidate_id",),
        ),
        AnalyzerEvidenceSpec(
            pool_id="offline_validation",
            semantic_role="complete fixed-checkpoint offline behavior",
            artifact_selector=AnalyzerArtifactSelector(("model_behavior",), ("offline_validation",)),
            coverage=AnalyzerCoverageSpec("all"),
            review_fields=AnalyzerReviewFields("prompt.text", "generation.raw_response"),
            grouping=AnalyzerGroupingSpec(
                group_key="prompt.id",
                position_key="artifact_position.value",
                response_index_key="generation.sample_index",
            ),
            context_fields=("prompt.dataset", "assessment.score", "assessment.ground_truth"),
        ),
    ),
    contracts=role_contracts(),
    artifact_compiler=compile_artifact,
    engine_artifact_binder=bind_engine_artifact,
    engine_config_preparer=prepare_engine_config,
    artifact_acceptance_validator=validate_artifact_acceptance,
    baseline_factory=build_baseline,
)
