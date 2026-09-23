"""Reward Design Task Plugin."""

import hashlib
import json

from ade.core.engine import TrainRFTCommand
from ade.core.validation import DeliveryViolation, ValidationReport
from ade.tasks.contracts import (
    AnalyzerArtifactSelector,
    AnalyzerCoverageSpec,
    AnalyzerEvidenceSpec,
    AnalyzerGroupingSpec,
    AnalyzerReviewFields,
    ArtifactCompilationError,
    ArtifactRealizationIntegrityError,
    ArtifactSpec,
    RankingSpec,
)
from ade.tasks.plugin import ArtifactAcceptanceContext, TaskPlugin
from ade.tasks.reward_design.baseline import build_baseline
from ade.tasks.reward_design.compiler import compile_artifact
from ade.tasks.reward_design.engine_binding import bind_engine_artifact
from ade.tasks.reward_design.engine_binding import bind_run_local_judge
from ade.tasks.reward_design.role_contracts import role_contracts
from ade.tasks.reward_design.compliance import (
    DirectReferenceRolloutUnavailable,
    direct_reference_trial as _direct_reference_trial,
    load_published_baseline_records,
    run_reward_compliance,
)
from ade.tasks.reward_design.rewards.replay import (
    RewardComplianceInputError,
    validate_reward_compliance_records,
)
from ade.engine.judge_dispatcher_impl import RewardCommandFailed


def _prior_judge_results(context, records):
    root = context.delivery.workspace / "input/reflection"
    results_path = root / "judge-results.jsonl"
    manifest_path = root / "judge-manifest.json"
    if not results_path.is_file() and not manifest_path.is_file():
        return None
    if not results_path.is_file() or not manifest_path.is_file():
        raise ValueError("prior Reward Judge evidence is incomplete")
    results = tuple(
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        len(results) != len(records)
        or any(not isinstance(item, dict) for item in results)
        or [item.get("record_id") for item in results]
        != [item.get("record_id") for item in records]
        or not isinstance(manifest, dict)
        or not isinstance(manifest.get("judge_binding"), dict)
    ):
        raise ValueError("prior Reward Judge evidence identity is invalid")
    return results, manifest


def _reward_observations(compliance, judge_results):
    rows = compliance.get("records")
    if not isinstance(rows, list):
        raise ValueError("Reward realization records are unavailable")
    statuses = [result.get("status") for result in judge_results]
    requested_statuses = [status for status in statuses if status != "not_configured"]
    external_unavailable = bool(requested_statuses) and not any(
        status == "available" for status in requested_statuses
    )
    return {
        "assigned_reward_changed": any(
            float(row["candidate_assigned_reward"])
            != float(row["reference_assigned_reward"])
            for row in rows
        ),
        "realized_advantage_changed": any(
            float(row["advantage_delta"]) != 0.0 for row in rows
        ),
        "advantage_sign_changed": any(
            bool(row["advantage_sign_change"]) for row in rows
        ),
        "judge_fallback_count": int(compliance.get("fallback_count") or 0),
        **(
            {"process_bank_sensitivity": compliance["process_bank_sensitivity"]}
            if "process_bank_sensitivity" in compliance else {}
        ),
    }, external_unavailable


def prepare_engine_config(config, state):
    """Add RFT-only step-0 and Run-owned Judge facts to the shared command path."""
    import copy

    prepared = copy.deepcopy(config)
    base_evaluation = state.bootstrap.base_evaluation
    if (
        base_evaluation is None
        or not base_evaluation.online_result_ref
        or base_evaluation.profile_status.get("online") != "succeeded"
    ):
        raise ValueError(
            "RFT Engine submission requires the shared successful step-0 online evaluation"
        )
    rft_config = prepared.get("rft")
    if not isinstance(rft_config, dict):
        raise ValueError("Reward Design Engine config requires rft")
    rft_config["step0_online_evaluation_ref"] = base_evaluation.online_result_ref
    if state.run_resources is not None:
        prepared = bind_run_local_judge(prepared, state.run_resources)
    return prepared


def validate_artifact_acceptance(context: ArtifactAcceptanceContext):
    """Replay a reward candidate against its direct reference Trial."""
    state = context.state
    if not state.bootstrap.enabled:
        return None
    lifecycle = context.lifecycle
    try:
        from ade.agent_runtime.experiment_package import decode_experiment_package

        trial = context.trial
        if not trial.source_artifact_ref_ids:
            raise ValueError("Reward Builder has no primary reference artifact")
        reference_ref = lifecycle._artifact_by_id(state, trial.source_artifact_ref_ids[0])
        if context.delivery.output.metadata.get("primary_reference_artifact_ref_id") != reference_ref.artifact_id:
            raise ValueError("reward delivery primary reference does not match Trial lineage")
        reference_trial = _direct_reference_trial(
            state, reference_ref.artifact_id
        )
        if reference_trial is None or not reference_trial.package_ref_id:
            return _publish_unverified_report(
                context,
                reference_ref.artifact_id,
            )
        package_ref = lifecycle._artifact_by_id(
            state, str(reference_trial.package_ref_id)
        )
        package = decode_experiment_package(lifecycle.repository.read_artifact(context.run_id, package_ref))
        package_manifest = json.loads(package["experiment/manifest.json"])
        records, compliance_manifest = load_published_baseline_records(
            package_manifest,
            package,
            durable_artifacts_root=(
                lifecycle.repository.layout.runs_root.parent
                / "engine-work"
                / "trial_artifacts"
            ),
        )
        validate_reward_compliance_records(records)
        process_declaration = context.delivery.output.metadata.get("process_evidence")
        if not isinstance(process_declaration, dict) or process_declaration.get(
            "status"
        ) not in {"required", "not_configured"}:
            raise ValueError("reward compliance process evidence declaration is invalid")
        judge_enabled = process_declaration["status"] == "required"
        judge_status = "not_configured"
        judge_binding = {"status": "not_configured"}
        prior_judge = _prior_judge_results(context, records) if judge_enabled else None
        if prior_judge is not None:
            judge_results, prior_manifest = prior_judge
            judge_status = str(prior_manifest.get("judge_status"))
            judge_binding = dict(prior_manifest["judge_binding"])
        elif judge_enabled and state.run_resources is not None:
            if lifecycle.reward_harness_judge_factory is None:
                raise ValueError("Run-owned Local Judge is not bound to the Harness")
            judge = lifecycle.reward_harness_judge_factory(state)
            validation = judge.validate_sync(
                run_id=context.run_id,
                coordinator_id=context.trial_key.coordinator_id,
                plan_id=context.trial_key.plan_id,
                trial_id=context.trial_key.trial_id,
                artifact_digest=hashlib.sha256(context.compiled.content).hexdigest(),
                records=tuple(
                    {"record_id": str(record["record_id"]), "question": str(record["prompt"]["text"]), "response": str(record["generation"]["raw_response"])}
                    for record in records
                ),
            )
            judge_status = str(validation["state"])
            local_judge = state.run_resources["local_judge"]
            judge_binding = {
                "status": judge_status,
                "job_id": validation["job_id"],
                "model_digest": local_judge["model_digest"],
                "rubric_digest": validation["rubric_digest"],
                "output_schema_digest": validation["output_schema_digest"],
                "projection_digest": validation["projection_digest"],
                "usage": validation["usage"],
            }
            judge_results = tuple(
                {
                    "record_id": record["record_id"],
                    "status": evidence["status"],
                    "dimensions": evidence.get("dimensions"),
                    "reason": evidence.get("reason"),
                }
                for record, evidence in zip(
                    records, validation["evidence"], strict=True
                )
            )
        elif judge_enabled:
            judge_status = "unavailable"
            judge_binding = {
                "status": "unavailable",
                "reason": "Run-owned Local Judge is not configured",
            }
            judge_results = tuple(
                {
                    "record_id": record["record_id"],
                    "status": "unavailable",
                    "dimensions": None,
                    "reason": "Run-owned Local Judge is not configured",
                }
                for record in records
            )
        else:
            judge_results = tuple(
                {
                    "record_id": record["record_id"],
                    "status": "not_configured",
                    "dimensions": None,
                    "reason": None,
                }
                for record in records
            )
        compliance = run_reward_compliance(
            reference_source=lifecycle.repository.read_artifact(context.run_id, reference_ref),
            candidate_source=context.compiled.content,
            records=records,
            judge_results=judge_results,
        )
        observations, external_unavailable = _reward_observations(
            compliance, judge_results
        )
        realization_status = "unverified" if external_unavailable else "verified"
        realization_reason = (
            "external_evidence_unavailable"
            if external_unavailable
            else None
        )
        content = (
            json.dumps(
                {
                    **compliance,
                    "realization_status": realization_status,
                    "reason": realization_reason,
                    "observations": observations,
                    "source_records": list(records),
                    "judge_results": list(judge_results),
                    "suite": compliance_manifest,
                    "source_position": compliance_manifest["selection"]["position"],
                    "selected_group_ids": compliance_manifest["selection"]["selected_group_ids"],
                    "judge_binding": judge_binding,
                    "judge_status": judge_status,
                    "judge_completed_count": sum(item["status"] == "available" for item in judge_results),
                    "judge_unavailable_count": sum(item["status"] == "unavailable" for item in judge_results),
                    "primary_reference_artifact_ref_id": reference_ref.artifact_id,
                    "direct_reference_trial": (
                        f"{context.run_id}/{reference_trial.coordinator_id}/"
                        f"{reference_trial.plan_id}/{reference_trial.trial_id}"
                    ),
                    "candidate_sha256": hashlib.sha256(context.compiled.content).hexdigest(),
                }, indent=2, sort_keys=True
            ) + "\n"
        ).encode()
        return lifecycle.repository.put_artifact(context.run_id, "reward_compliance_report", content)
    except DirectReferenceRolloutUnavailable:
        return _publish_unverified_report(
            context,
            reference_ref.artifact_id,
        )
    except RewardCommandFailed:
        raise
    except RewardComplianceInputError as error:
        raise ArtifactRealizationIntegrityError(
            ValidationReport(
                (
                    DeliveryViolation(
                        "reward_compliance_input_invalid",
                        str(error),
                        "input/compliance/records.jsonl",
                        repairable=False,
                    ),
                )
            )
        ) from error
    except Exception as error:
        raise ArtifactCompilationError(
            ValidationReport((DeliveryViolation(
                "reward_compliance_failed",
                f"reward compliance failed: {type(error).__name__}: {error}",
                "reward.py", repairable=True,
            ),))
        ) from error


def _publish_unverified_report(
    context: ArtifactAcceptanceContext,
    primary_reference_artifact_ref_id: str,
):
    content = (
        json.dumps(
            {
                "schema_version": "1",
                "realization_status": "unverified",
                "reason": "direct_reference_rollout_unavailable",
                "primary_reference_artifact_ref_id": (
                    primary_reference_artifact_ref_id
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    return context.lifecycle.repository.put_artifact(
        context.run_id,
        "reward_compliance_report",
        content,
    )

plugin = TaskPlugin(
    task_id="reward_design",
    command_type=TrainRFTCommand,
    artifact=ArtifactSpec("reward_design", True),
    ranking=RankingSpec("offline.ranking_score"),
    analyzer_evidence_specs=(
        AnalyzerEvidenceSpec(
            pool_id="training_rollout",
            semantic_role="training rollouts at task-owned artifact positions",
            artifact_selector=AnalyzerArtifactSelector(("model_behavior",), ("training_rollout",)),
            coverage=AnalyzerCoverageSpec("fraction", 0.25, "groups"),
            review_fields=AnalyzerReviewFields("prompt.text", "generation.raw_response"),
            grouping=AnalyzerGroupingSpec(
                group_key="prompt.id",
                position_key="artifact_position.value",
                response_index_key="generation.sample_index",
            ),
            context_fields=(
                "artifact_position",
                "training_context.policy_step_at_generation",
                "prompt.id",
                "generation.sample_index",
                "training_context.effective_training_reward",
                "training_context.group_credit_enabled",
                "training_context.group_uid",
                "training_context.group_type",
                "training_context.outcome_score",
                "training_context.process_evidence",
                "training_context.artifact_projection",
                "training_context.rule_evidence",
                "training_context.correctness",
                "training_context.pre_group_reward",
                "training_context.final_training_reward",
                "training_context.realized_grpo_advantage",
                "training_context.counterfactual_identity_grpo_advantage",
                "training_context.assigned_training_reward",
                "training_context.group_credit_mode",
                "training_context.group_credit_source",
                "training_context.group_credit_reason",
                "training_context.group_credit_evidence_sources",
                "training_context.group_credit_process_dimensions",
            ),
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
    baseline_factory=build_baseline,
    engine_config_preparer=prepare_engine_config,
    artifact_acceptance_validator=validate_artifact_acceptance,
)
