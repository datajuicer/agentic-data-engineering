"""Bind a frozen Curriculum schedule and repository-owned reward to RFT."""

from __future__ import annotations

import copy
import json

from ade.tasks.contracts import (
    BaselineArtifactRequest,
    EngineArtifactBinding,
    EngineArtifactBindingRequest,
    EngineObjectPayload,
)
from ade.tasks.curriculum_learning.fixed_pool import build_fixed_pool_from_task
from ade.tasks.curriculum_learning.materializer import materialize_scheduled_parquet
from ade.tasks.curriculum_learning.realizer import realize_curriculum_sync
from ade.tasks.reward_design.baseline import build_baseline as build_reward_baseline
from ade.tasks.reward_design.engine_binding import bind_engine_artifact as bind_reward


def bind_engine_artifact(request: EngineArtifactBindingRequest) -> EngineArtifactBinding:
    config = copy.deepcopy(dict(request.engine_config))
    rft = config.get("rft")
    if not isinstance(rft, dict):
        raise ValueError("Curriculum Learning Engine config requires rft")
    if request.final_realization:
        realization = dict(request.final_realization)
        if realization.get("schema_version") != "ade.curriculum_realization.v1":
            raise ValueError("Curriculum final realization schema is invalid")
        schedule = realization.get("schedule")
        if not isinstance(schedule, dict):
            raise ValueError("Curriculum final realization lacks schedule")
    elif request.is_baseline:
        inventory, pool_stats = build_fixed_pool_from_task(request.task_config)
        task_rft = request.task_config.get("rft")
        if not isinstance(task_rft, dict):
            raise ValueError("Curriculum baseline RFT shape is unavailable")
        realized = realize_curriculum_sync(
            request.compiled_content,
            inventory,
            total_steps=int(task_rft["total_training_steps"]),
            prompts_per_step=int(task_rft["gen_batch_size"]),
            rollout_n=int(task_rft["rollout_n"]),
            pool_stats=pool_stats,
            judge_batch=None,
        )
        schedule = realized.schedule
        realization = {
            "kind": "curriculum_realization",
            "schema_version": "ade.curriculum_realization.v1",
            "realization_status": "verified",
            "reason": None,
            "policy_source": request.compiled_content.decode("utf-8"),
            "schedule": schedule,
            "judge_binding": {
                "enabled": False,
                "owner": "curriculum_realization",
                "stats": {"requested": 0, "completed": 0, "fallback": 0},
            },
            "judge_evidence": [],
        }
    else:
        raise ValueError("Curriculum Search training requires a final realization")
    task_data = request.task_config.get("data")
    if not isinstance(task_data, dict) or not isinstance(task_data.get("train"), str):
        raise ValueError("Curriculum training source is unavailable")
    scheduled = materialize_scheduled_parquet(task_data["train"], schedule)
    root = request.binding_uri.rstrip("/")
    curriculum_ref = f"{root}/curriculum.py"
    schedule_ref = f"{root}/curriculum-schedule.json"
    realization_ref = f"{root}/realization-report.json"
    data_ref = f"{root}/scheduled-train.parquet"

    reward = build_reward_baseline(
        BaselineArtifactRequest(task_config=request.task_config, seed=0)
    )
    reward_binding = bind_reward(
        EngineArtifactBindingRequest(
            binding_uri=f"{root}/fixed-reward",
            compiled_kind="reward_design",
            compiled_content=reward.content,
            task_config=request.task_config,
            engine_config=config,
            is_baseline=True,
        )
    )
    payload = copy.deepcopy(dict(reward_binding.input_payload))
    bound_rft = payload["rft"]
    bound_rft["scheduled_training_data_ref"] = data_ref
    bound_rft["curriculum_artifact_ref"] = curriculum_ref
    bound_rft["curriculum_schedule_ref"] = schedule_ref
    bound_rft["curriculum_realization_ref"] = realization_ref
    bound_rft["analysis_profile_id"] = "curriculum_learning"
    return EngineArtifactBinding(
        input_payload=payload,
        objects=(
            *reward_binding.objects,
            EngineObjectPayload(curriculum_ref, request.compiled_content),
            EngineObjectPayload(schedule_ref, _json_bytes(schedule)),
            EngineObjectPayload(realization_ref, _json_bytes(realization)),
            EngineObjectPayload(data_ref, scheduled),
        ),
    )


def prepare_engine_config(config, state):
    prepared = copy.deepcopy(config)
    rft = prepared.get("rft")
    if not isinstance(rft, dict):
        raise ValueError("Curriculum Learning Engine config requires rft")
    verl_config = rft.get("verl_config")
    if not isinstance(verl_config, dict):
        raise ValueError("Curriculum Learning Engine config requires rft.verl_config")
    train = verl_config.get("train")
    sections = (
        verl_config.get("rft"),
        train.get("rft") if isinstance(train, dict) else None,
    )
    for section in sections:
        if not isinstance(section, dict):
            raise ValueError("Curriculum RFT runtime mirror is missing")
        if section.get("judge_enrichment") != {
            "enabled": False,
            "owner": "rft_reward",
        }:
            raise ValueError(
                "Curriculum reward-side Judge must be explicitly disabled"
            )
        if section.get("analysis_profile_id") != "curriculum_learning":
            raise ValueError("Curriculum RFT analysis profile is missing")
    base_evaluation = state.bootstrap.base_evaluation
    if (
        base_evaluation is None
        or not base_evaluation.online_result_ref
        or base_evaluation.profile_status.get("online") != "succeeded"
    ):
        raise ValueError(
            "Curriculum RFT submission requires successful step-0 online evaluation"
        )
    rft["step0_online_evaluation_ref"] = base_evaluation.online_result_ref
    return prepared


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
