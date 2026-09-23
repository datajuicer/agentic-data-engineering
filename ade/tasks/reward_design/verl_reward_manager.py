"""VERL reward manager with Engine-owned training outcome scoring."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import logging
import os
import sys
from pathlib import Path

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

# VERL loads this file by absolute path inside Ray workers, where the project
# root is not guaranteed to be on sys.path.
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ade.tasks.reward_design.rewards.contracts import (  # noqa: E402
    enforce_outcome_component,
    validate_reward_function,
    validate_reward_result,
)
from ade.tasks.reward_design.rewards.training_outcome import (  # noqa: E402
    TrainingOutcome,
    compute_training_outcome,
)
from ade.engine.judge_dispatcher import (  # noqa: E402
    JudgeCircuitBreaker,
    RewardCommandFailed,
    RoundRubricDispatcher,
    RubricRowUnavailable,
)
from ade.tasks.reward_design.process_evidence import (  # noqa: E402
    PROCESS_RUBRIC,
    available_process_evidence,
    not_configured_process_evidence,
    unavailable_process_evidence,
)
from ade.local_rubric_judge import HttpRubricJobGateway, LocalRubricJudgeClient  # noqa: E402
from ade.memory.usage import RunUsageLedger  # noqa: E402


logger = logging.getLogger(__name__)
RFT_JUDGE_MAX_REQUESTS_PER_JOB = 256


class ADERewardManager(RewardManagerBase):
    def __init__(self, config, tokenizer, compute_score, **_kwargs):
        validate_reward_function(compute_score)
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score
        self.max_response_length_tokens = int(config.data.max_response_length)
        reward_config = config.get("reward", {})
        group_credit_config = reward_config.get("group_credit_function") or {}
        self.group_credit_enabled = bool(group_credit_config.get("enabled", False))
        adapter_config = reward_config.get("training_outcome_adapter") or {}
        self.training_outcome_adapter_id = str(adapter_config.get("adapter_id") or "")
        if not self.training_outcome_adapter_id:
            raise ValueError("reward.training_outcome_adapter.adapter_id is required")
        self.reward_timeout_seconds = float(
            reward_config.get("compute_score_timeout_seconds", 300.0)
        )
        self.local_judge_client = None
        self.judge_config = None
        self.judge_circuit_breaker = None
        self._judge_batch_index = 0
        self._judge_worker_id = "worker-local"
        judge_config = reward_config.get("llm_judge") or {}
        if bool(judge_config.get("enabled", False)):
            import ray

            actor_id = ray.get_runtime_context().get_actor_id()
            if actor_id is not None:
                self._judge_worker_id = str(actor_id)
            gateway_url = str(judge_config.get("gateway_url") or "")
            authorization_env = str(judge_config.get("authorization_env") or "")
            if not authorization_env or authorization_env not in os.environ:
                raise ValueError("Local Judge authorization environment is unavailable")
            request_timeout = float(judge_config.get("request_timeout_seconds") or 0)
            if request_timeout <= 0:
                raise ValueError("Local Judge request_timeout_seconds is required")
            gateway = HttpRubricJobGateway(
                gateway_url,
                authorization=os.environ[authorization_env],
                timeout_seconds=request_timeout,
            )
            self.local_judge_client = LocalRubricJudgeClient(gateway)
            self.judge_config = {
                "run_id": str(judge_config.get("run_id") or ""),
                "engine_command_id": str(judge_config.get("engine_command_id") or ""),
                "logical_command_id": str(judge_config.get("logical_command_id") or ""),
                "engine_attempt_id": str(judge_config.get("engine_attempt_id") or ""),
                "engine_attempt_index": int(
                    judge_config.get("engine_attempt_index") or 0
                ),
                "coordinator_id": str(judge_config.get("coordinator_id") or ""),
                "plan_id": str(judge_config.get("plan_id") or ""),
                "trial_id": str(judge_config.get("trial_id") or ""),
                "max_failed_row_ratio": float(judge_config.get("max_failed_row_ratio")),
                "max_consecutive_unhealthy_jobs": int(
                    judge_config.get("max_consecutive_unhealthy_jobs")
                ),
            }
            self.judge_circuit_breaker = JudgeCircuitBreaker(
                max_failed_row_ratio=self.judge_config["max_failed_row_ratio"],
                max_consecutive_unhealthy_jobs=self.judge_config[
                    "max_consecutive_unhealthy_jobs"
                ],
            )
            if not self.judge_config["run_id"] or not self.judge_config["engine_command_id"]:
                raise ValueError("Local Judge Run and Engine Command identity are required")
            if (
                not self.judge_config["logical_command_id"]
                or not self.judge_config["engine_attempt_id"]
                or self.judge_config["engine_attempt_index"] < 1
            ):
                raise ValueError("Local Judge logical Command and Attempt are required")
            usage_run_dir = judge_config.get("usage_run_dir")
            self.usage_ledger = (
                RunUsageLedger(Path(str(usage_run_dir)).resolve())
                if usage_run_dir
                else None
            )

    async def run_batch(self, data: DataProto) -> list[dict]:
        if self.local_judge_client is None:
            return await asyncio.gather(
                *(self.run_single(data[index : index + 1]) for index in range(len(data)))
            )
        assert self.judge_config is not None
        self._judge_batch_index = getattr(self, "_judge_batch_index", 0) + 1
        batch_index = self._judge_batch_index
        raw_global_step = (getattr(data, "meta_info", {}) or {}).get("global_steps")
        global_step = (
            int(raw_global_step)
            if type(raw_global_step) is int and raw_global_step > 0
            else batch_index
        )
        sample_names = []
        extras = data.non_tensor_batch.get("extra_info")
        for index in range(len(data)):
            extra = extras[index] if extras is not None else {}
            if isinstance(extra, dict):
                name = str(extra.get("sample_uid") or extra.get("uid") or "")
            else:
                name = ""
            sample_names.append(name or f"sample-{index}")
        if len(set(sample_names)) != len(sample_names):
            raise ValueError("reward batch sample identities must be unique")
        sample_ids = set(sample_names)
        dispatcher = RoundRubricDispatcher(
            self.local_judge_client,
            sample_ids=sample_ids,
            submission_id=(
                f"{self.judge_config['run_id']}:"
                f"{self.judge_config['engine_command_id']}:"
                f"{getattr(self, '_judge_worker_id', 'worker-local')}:"
                f"reward-batch-{batch_index:06d}"
            ),
            job_metadata={
                "run_id": self.judge_config["run_id"],
                "engine_command_id": self.judge_config["engine_command_id"],
                "logical_command_id": self.judge_config.get("logical_command_id", ""),
                "attempt_id": self.judge_config.get("engine_attempt_id", ""),
                "attempt_index": self.judge_config.get("engine_attempt_index", 1),
                "coordinator_id": self.judge_config["coordinator_id"],
                "plan_id": self.judge_config["plan_id"],
                "trial_id": self.judge_config["trial_id"],
                "scope": {
                    "run_id": self.judge_config["run_id"],
                    "coordinator_id": self.judge_config["coordinator_id"],
                    "plan_id": self.judge_config["plan_id"],
                    "trial_id": self.judge_config["trial_id"],
                },
                "subject_ref": (
                    f"{self.judge_config['run_id']}/"
                    f"{self.judge_config['coordinator_id']}/"
                    f"{self.judge_config['plan_id']}/"
                    f"{self.judge_config['trial_id']}"
                ),
                "global_step": global_step,
                "reward_batch_index": batch_index,
            },
            circuit_breaker=self.judge_circuit_breaker,
            usage_ledger=getattr(self, "usage_ledger", None),
            max_requests_per_job=RFT_JUDGE_MAX_REQUESTS_PER_JOB,
        )
        globals_dict = self.compute_score.__globals__
        current_sample = contextvars.ContextVar("ade_judge_sample")
        process_evidence_by_sample = {}
        judge_calls_by_sample = {}

        async def batched_judge(question, response):
            sample_id = current_sample.get()
            calls = judge_calls_by_sample.get(sample_id, 0) + 1
            judge_calls_by_sample[sample_id] = calls
            if calls != 1:
                raise ValueError("reward artifact issued more than one Judge call for a rollout")
            evaluation = await dispatcher.evaluate(
                sample_id, question, response, PROCESS_RUBRIC
            )
            evidence = available_process_evidence(evaluation)
            process_evidence_by_sample[sample_id] = evidence
            return evidence
        artifact_acquires_judge = not getattr(self, "group_credit_enabled", False)
        previous = globals_dict.get("llm_judge")
        if artifact_acquires_judge:
            globals_dict["llm_judge"] = batched_judge
        try:
            tasks = [
                asyncio.create_task(self.run_single(
                    data[index : index + 1],
                    sample_id=sample_names[index],
                    sample_context=current_sample,
                    dispatcher=dispatcher,
                    process_evidence_by_sample=process_evidence_by_sample,
                    engine_judge=(
                        None if artifact_acquires_judge else batched_judge
                    ),
                ))
                for index in range(len(data))
            ]
            return await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            await dispatcher.cancel()
            raise
        finally:
            if artifact_acquires_judge:
                if previous is None:
                    globals_dict.pop("llm_judge", None)
                else:
                    globals_dict["llm_judge"] = previous
            await dispatcher.close()

    async def run_single(
        self,
        data: DataProto,
        *,
        sample_id=None,
        sample_context=None,
        dispatcher=None,
        process_evidence_by_sample=None,
        engine_judge=None,
    ) -> dict:
        token = sample_context.set(sample_id) if sample_context is not None else None
        try:
            return await self._run_single(
                data,
                sample_id=sample_id,
                process_evidence_by_sample=process_evidence_by_sample,
                engine_judge=engine_judge,
            )
        finally:
            if sample_context is not None:
                sample_context.reset(token)
            if dispatcher is not None:
                await dispatcher.mark_done(sample_id)

    async def _run_single(
        self,
        data: DataProto,
        *,
        sample_id=None,
        process_evidence_by_sample=None,
        engine_judge=None,
    ) -> dict:
        data_item = data[-1:][0]
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        prompt_mask = data_item.batch["attention_mask"][:prompt_length].bool()
        valid_prompt_ids = prompt_ids[prompt_mask]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = int(
            data_item.batch["attention_mask"][-response_length:].sum().item()
        )
        valid_response_ids = response_ids[:valid_response_length]
        question_prompt, response_str = await self.loop.run_in_executor(
            None,
            lambda: (
                self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True),
                self.tokenizer.decode(valid_response_ids, skip_special_tokens=False),
            ),
        )
        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = str(data_item.non_tensor_batch["reward_model"]["ground_truth"] or "")
        original_extra_info = data_item.non_tensor_batch.get("extra_info", {})
        extra_info = dict(original_extra_info) if isinstance(original_extra_info, dict) else {}
        sample_uid = str(
            extra_info.get("sample_uid")
            or extra_info.get("uid")
            or data_item.non_tensor_batch.get("uid")
            or "unknown"
        )
        training_outcome = await self.loop.run_in_executor(
            None,
            functools.partial(
                compute_training_outcome,
                data_source,
                response_str,
                ground_truth,
                expected_adapter_id=getattr(self, "training_outcome_adapter_id", None),
            ),
        )
        kwargs = {
            "question_prompt": question_prompt,
            "response_content": response_str,
            "extracted_answer": training_outcome.extracted_answer,
            "outcome_score": training_outcome.outcome_score,
            "response_length_tokens": valid_response_length,
            "max_response_length_tokens": self.max_response_length_tokens,
        }
        try:
            if engine_judge is not None:
                await engine_judge(question_prompt, response_str)
            if inspect.iscoroutinefunction(self.compute_score):
                score_call = self.compute_score(**kwargs)
            else:
                score_call = self.loop.run_in_executor(
                    None, lambda: self.compute_score(**kwargs)
                )
            if self.local_judge_client is None:
                result = await asyncio.wait_for(
                    score_call, timeout=self.reward_timeout_seconds
                )
            else:
                # The batch dispatcher owns the running-job deadline. FIFO queue
                # wait belongs to the deployment scheduler and cannot turn a
                # healthy job into row-local fallback before execution starts.
                result = await score_call
            if self.local_judge_client is not None and (
                process_evidence_by_sample is None
                or sample_id not in process_evidence_by_sample
            ):
                raise ValueError("reward artifact did not acquire required Judge evidence")
            validated = enforce_outcome_component(
                validate_reward_result(
                    result,
                    fallback=False,
                    group_credit_enabled=getattr(
                        self, "group_credit_enabled", False
                    ),
                ),
                training_outcome.outcome_score,
            )
        except RubricRowUnavailable as error:
            return await self._fallback_result(
                kwargs,
                training_outcome=training_outcome,
                valid_response_length=valid_response_length,
                data_source=data_source,
                sample_uid=sample_uid,
                reason=f"{type(error).__name__}: {error}",
                process_evidence=unavailable_process_evidence(str(error)),
            )

        score = validated["score"]
        process_evidence = (
            process_evidence_by_sample[sample_id]
            if process_evidence_by_sample is not None
            and sample_id in process_evidence_by_sample
            else not_configured_process_evidence()
        )
        logger.debug(
            "Training outcome extracted with adapter=%s method=%s",
            training_outcome.adapter_id,
            training_outcome.extraction_method,
        )
        return self._result(
            score,
            training_outcome=training_outcome,
            valid_response_length=valid_response_length,
            data_source=data_source,
            sample_uid=sample_uid,
            artifact_projection=validated["artifact_projection"],
            rule_evidence=validated["rule_evidence"],
            process_evidence=process_evidence,
            judge_fallback=False,
            judge_fallback_reason=None,
        )

    async def _fallback_result(
        self,
        kwargs: dict,
        *,
        training_outcome: TrainingOutcome,
        valid_response_length: int,
        data_source,
        sample_uid: str,
        reason: str,
        process_evidence: dict,
    ) -> dict:
        fallback = self.compute_score.__globals__.get("compute_fallback_score")
        validate_reward_function(fallback)
        if inspect.iscoroutinefunction(fallback):
            fallback_call = fallback(**kwargs)
        else:
            fallback_call = self.loop.run_in_executor(None, lambda: fallback(**kwargs))
        result = await asyncio.wait_for(
            fallback_call,
            timeout=self.reward_timeout_seconds,
        )
        validated = enforce_outcome_component(
            validate_reward_result(
                result,
                fallback=True,
                group_credit_enabled=getattr(self, "group_credit_enabled", False),
            ),
            training_outcome.outcome_score,
        )
        return self._result(
            validated["score"],
            training_outcome=training_outcome,
            valid_response_length=valid_response_length,
            data_source=data_source,
            sample_uid=sample_uid,
            artifact_projection=validated["artifact_projection"],
            rule_evidence=validated["rule_evidence"],
            process_evidence=process_evidence,
            judge_fallback=True,
            judge_fallback_reason=reason,
        )

    def _result(
        self,
        score: float,
        *,
        training_outcome: TrainingOutcome,
        valid_response_length: int,
        data_source,
        sample_uid: str,
        artifact_projection: float | None,
        rule_evidence: dict,
        process_evidence: dict,
        judge_fallback: bool,
        judge_fallback_reason: str | None,
    ) -> dict:
        return {
            "reward_score": score,
            "reward_extra_info": {
                "acc": score,
                "custom_reward_score": score,
                "reward_result_schema_version": "ade.reward_result.v2",
                "training_outcome_schema_version": training_outcome.schema_version,
                "training_outcome_adapter_id": training_outcome.adapter_id,
                "training_outcome_status": training_outcome.status,
                "outcome_score": training_outcome.outcome_score,
                "extracted_answer": training_outcome.extracted_answer,
                "extraction_method": training_outcome.extraction_method,
                "format_matched": training_outcome.format_matched,
                "response_length_tokens": valid_response_length,
                "max_response_length_tokens": self.max_response_length_tokens,
                "reward_timeout": False,
                "reward_error": None,
                "data_source": str(data_source),
                "sample_uid": sample_uid,
                "process_evidence": process_evidence,
                "artifact_projection": artifact_projection,
                "rule_evidence": rule_evidence,
                "pre_group_reward": score,
                "final_training_reward": score,
                "group_credit_enabled": getattr(self, "group_credit_enabled", False),
                "judge_fallback": judge_fallback,
                "judge_fallback_reason": judge_fallback_reason,
            },
        }
