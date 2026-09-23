"""Curriculum Learning Task Plugin."""

from __future__ import annotations

import asyncio
import json

from ade.core.engine import TrainRFTCommand
from ade.tasks.contracts import (
    AnalyzerArtifactSelector,
    AnalyzerCoverageSpec,
    AnalyzerEvidenceSpec,
    AnalyzerGroupingSpec,
    AnalyzerReviewFields,
    ArtifactCompilationError,
    ArtifactSpec,
    RankingSpec,
)
from ade.tasks.curriculum_learning.baseline import build_baseline
from ade.tasks.curriculum_learning.compiler import compile_artifact
from ade.tasks.curriculum_learning.engine_binding import (
    bind_engine_artifact,
    prepare_engine_config,
)
from ade.tasks.curriculum_learning.role_contracts import role_contracts
from ade.tasks.curriculum_learning.fixed_pool import build_fixed_pool_from_task
from ade.tasks.curriculum_learning.realizer import realize_curriculum
from ade.tasks.plugin import ArtifactAcceptanceContext, TaskPlugin
from ade.core.validation import DeliveryViolation, ValidationReport


def validate_artifact_acceptance(context: ArtifactAcceptanceContext):
    """Freeze one canonical schedule and its pre-training Judge evidence."""
    try:
        inventory, pool_stats = build_fixed_pool_from_task(context.state.task.config)
        rft = context.state.task.config.get("rft")
        enrichment = context.state.task.config.get("judge_enrichment")
        if not isinstance(rft, dict) or not isinstance(enrichment, dict):
            raise ValueError("Curriculum realization binding is incomplete")
        total_steps = rft.get("total_training_steps")
        prompts_per_step = rft.get("gen_batch_size")
        rollout_n = rft.get("rollout_n")
        if any(type(value) is not int or value <= 0 for value in (
            total_steps, prompts_per_step, rollout_n
        )):
            raise ValueError("Curriculum schedule shape must be positive integers")
        if enrichment != {"enabled": True, "owner": "curriculum_realization"}:
            raise ValueError("Curriculum realization requires its owned Judge capability")
        dispatcher = (
            context.lifecycle.selection_harness_judge_factory(context.state, context)
            if context.lifecycle.selection_harness_judge_factory
            else None
        )

        async def judge_batch(requests):
            if dispatcher is not None:
                return await dispatcher.evaluate(requests)
            return [
                {
                    "status": "unavailable",
                    "scores_by_dimension": {},
                    "projected_score": 0.0,
                    "fallback": True,
                    "fallback_reason": "external_evidence_unavailable",
                }
                for _request in requests
            ]

        async def realize():
            try:
                return await realize_curriculum(
                    context.compiled.content,
                    inventory,
                    total_steps=total_steps,
                    prompts_per_step=prompts_per_step,
                    rollout_n=rollout_n,
                    pool_stats=pool_stats,
                    judge_batch=judge_batch,
                )
            finally:
                if dispatcher is not None:
                    await dispatcher.close()

        realized = asyncio.run(realize())
        evidence = list(realized.judge_evidence)
        completed_count = sum(
            row["evidence"].get("status") == "completed"
            and not row["evidence"].get("fallback")
            for row in evidence
        )
        fallback_count = sum(
            bool(row["evidence"].get("fallback")) for row in evidence
        )
        external_unavailable = dispatcher is None or (
            bool(evidence) and completed_count == 0
        )
        status = "unverified" if external_unavailable else "verified"
        judge_stats = (
            dict(dispatcher.stats)
            if dispatcher is not None
            else {
                "requested": len(evidence),
                "completed": 0,
                "fallback": fallback_count,
            }
        )
        judge_stats.update(
            {
                "evidence_count": len(evidence),
                "completed_total": completed_count,
                "fallback_total": fallback_count,
            }
        )
        report = {
            "kind": "curriculum_realization",
            "schema_version": "ade.curriculum_realization.v1",
            "realization_status": status,
            "reason": (
                "external_evidence_unavailable" if external_unavailable else None
            ),
            "policy_source": context.compiled.content.decode("utf-8"),
            "schedule": realized.schedule,
            "judge_binding": {
                "enabled": True,
                "owner": "curriculum_realization",
                "stats": judge_stats,
            },
            "judge_evidence": evidence,
        }
        return context.lifecycle.repository.put_artifact(
            context.run_id,
            "curriculum_realization",
            (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
        )
    except ArtifactCompilationError:
        raise
    except Exception as error:
        raise ArtifactCompilationError(
            ValidationReport(
                (
                    DeliveryViolation(
                        "curriculum_realization_failed",
                        f"Curriculum realization failed: {type(error).__name__}: {error}",
                        "curriculum.py",
                        repairable=True,
                    ),
                )
            )
        ) from error


plugin = TaskPlugin(
    task_id="curriculum_learning",
    command_type=TrainRFTCommand,
    artifact=ArtifactSpec("curriculum_learning", True),
    ranking=RankingSpec("offline.ranking_score"),
    analyzer_evidence_specs=(
        AnalyzerEvidenceSpec(
            pool_id="training_rollout",
            semantic_role="training rollouts joined to the frozen problem schedule",
            artifact_selector=AnalyzerArtifactSelector(
                ("model_behavior",), ("training_rollout",)
            ),
            coverage=AnalyzerCoverageSpec("fraction", 0.25, "groups"),
            review_fields=AnalyzerReviewFields(
                "prompt.text", "generation.raw_response"
            ),
            grouping=AnalyzerGroupingSpec(
                group_key="prompt.id",
                position_key="artifact_position.value",
                response_index_key="generation.sample_index",
            ),
            context_fields=(
                "artifact_position",
                "training_context.sample_uid",
                "training_context.policy_step_at_generation",
                "training_context.outcome_score",
                "training_context.final_training_reward",
                "training_context.realized_grpo_advantage",
            ),
        ),
        AnalyzerEvidenceSpec(
            pool_id="offline_validation",
            semantic_role="complete fixed-checkpoint offline behavior",
            artifact_selector=AnalyzerArtifactSelector(
                ("model_behavior",), ("offline_validation",)
            ),
            coverage=AnalyzerCoverageSpec("all"),
            review_fields=AnalyzerReviewFields(
                "prompt.text", "generation.raw_response"
            ),
            grouping=AnalyzerGroupingSpec(
                group_key="prompt.id",
                position_key="artifact_position.value",
                response_index_key="generation.sample_index",
            ),
            context_fields=(
                "prompt.dataset",
                "assessment.score",
                "assessment.ground_truth",
            ),
        ),
    ),
    contracts=role_contracts(),
    artifact_compiler=compile_artifact,
    engine_artifact_binder=bind_engine_artifact,
    engine_config_preparer=prepare_engine_config,
    artifact_acceptance_validator=validate_artifact_acceptance,
    baseline_factory=build_baseline,
)
