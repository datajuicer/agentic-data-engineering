"""VERL RFT process backend and reproducibility records."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import re
import subprocess
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

from ade.core.dotenv import read_dotenv_value
from ade.engine.checkpoints.cleanup import cleanup_checkpoint_directories
from ade.tasks.reward_design.rewards.trace import finalize_reward_trace
from ade.engine.telemetry.training import (
    MetricProfile,
    TelemetryExportSpec,
    VERL_METRIC_PROFILE,
    export_exact_wandb_telemetry,
)
from ade.engine.telemetry.tracking import (
    audit_ref_from_dict,
    build_training_tracking,
)
from ade.engine.execution.coordinator_resources import validate_coordinator_workload_request


_BASE_MODEL = str(
    Path(__file__).resolve().parents[3]
    / "models"
    / "Qwen2.5-1.5B-Instruct-think"
)
_WANDB_INIT_TIMEOUT_SECONDS = "300"
_RAY_SECRET_ENV_NAMES = "ADE_RAY_RUNTIME_SECRET_ENV_NAMES"
_MAX_JUDGE_REWARD_WORKERS = 8
_METRIC_PATTERN = re.compile(
    r"(?P<name>(?:critic/rewards|reward|response_length|rollout)/[A-Za-z0-9_.@/-]+)"
    r"\s*[=:]\s*(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


def build_verl_training_command(config: dict[str, Any], *, run_dir: str | Path) -> list[str]:
    project_root = Path(config.get("project_root") or Path.cwd()).resolve()
    rft = _rft_config(config)
    prompt_protocol = _rft_prompt_protocol(rft)
    python = _environment_python(project_root, rft)
    source_value = Path(str(rft.get("verl_source") or project_root / "third_party/verl"))
    source_path = (
        (project_root / source_value).resolve()
        if not source_value.is_absolute()
        else source_value.resolve()
    )
    runtime_pythonpath = os.pathsep.join((str(source_path), str(project_root)))
    driver_bootstrap = (
        f"import runpy,sys;sys.path[:0]={json.dumps([str(source_path), str(project_root)])};"
        "runpy.run_module('verl.trainer.main_ppo', run_name='__main__')"
    )
    train_files, validation_files = _training_files(
        config,
        project_root=project_root,
        run_dir=Path(run_dir),
    )
    checkpoint_root = Path(run_dir).resolve() / "checkpoints" / "verl"
    rollout_segment = _safe_name(_rollout_segment(rft, config))
    rollout_data_dir = (
        Path(run_dir).resolve()
        / "engine_audit"
        / "reward_rollouts_raw"
        / rollout_segment
    )
    model = str(config.get("base_model") or config.get("model") or _BASE_MODEL)
    _validate_base_model(str(config.get("source_model_path") or model))
    wandb_environment, wandb_tracking = _wandb_environment(config, run_dir=run_dir)
    wandb_enabled = bool(wandb_tracking.get("enabled"))
    reward_path = _required_file(
        rft.get("reward_function_path"), "rft.reward_function_path"
    )
    judge_required = _judge_enrichment_requested(rft, reward_path)
    reward_manager_path = Path(__file__).with_name("verl_reward_manager.py").resolve()
    outcome_adapter = rft.get("training_outcome_adapter")
    if not isinstance(outcome_adapter, dict) or not outcome_adapter.get("adapter_id"):
        raise ValueError("rft.training_outcome_adapter.adapter_id is required")
    algorithm = str(rft.get("algorithm") or "grpo")
    rollout_backend = str(rft.get("rollout_backend") or "vllm")
    if rollout_backend != "vllm":
        raise ValueError("VERL reward-design rollout_backend must be vllm")
    gpus_per_node = int(rft.get("gpus_per_node") or rft.get("train_gpus") or 8)
    nodes = int(rft.get("nodes") or 1)
    if gpus_per_node < 1 or nodes < 1:
        raise ValueError("VERL trainer GPU and node counts must be positive")
    validate_coordinator_workload_request(
        rft, requested_gpus=gpus_per_node * nodes
    )
    train_batch_size = int(rft.get("train_batch_size", 128))
    ppo_mini_batch_size = int(rft.get("ppo_mini_batch_size", 64))
    if judge_required and train_batch_size % ppo_mini_batch_size != 0:
        raise ValueError(
            "Judge-dependent train_batch_size must be divisible by "
            "ppo_mini_batch_size"
        )
    local_judge = _local_judge_binding(config, rft) if judge_required else None
    seed = _required_seed(rft)
    rollout_n = int(rft.get("rollout_n", 5))
    log_prob_micro_batch_size_per_gpu = int(
        rft.get("log_prob_micro_batch_size_per_gpu", 1)
    )
    if not bool(rft.get("actor_use_dynamic_bsz", False)):
        data_parallel_size = (gpus_per_node * nodes) // int(
            rft.get("ulysses_sequence_parallel_size", 1)
        )
        prompt_groups_per_gpu = train_batch_size // data_parallel_size
        log_prob_micro_batch_size_per_gpu = max(
            size
            for size in range(
                1,
                min(log_prob_micro_batch_size_per_gpu, prompt_groups_per_gpu) + 1,
            )
            if prompt_groups_per_gpu % size == 0
        )
    engine_scope = _safe_name(
        str(config.get("engine_command_id") or Path(run_dir).resolve().name)
    )
    ray_namespace = f"ade-{engine_scope}"
    transfer_queue_namespace = f"transfer-queue-{engine_scope}"
    command = [
        str(python),
        "-c",
        driver_bootstrap,
        f"algorithm.adv_estimator={algorithm}",
        f"algorithm.gamma={float(rft.get('gamma', 1.0))}",
        f"algorithm.lam={float(rft.get('lambda', 1.0))}",
        f"data.train_files={json.dumps(train_files)}",
        f"data.val_files={json.dumps(validation_files)}",
        # ADE owns validation outside VERL for this RFT path.  Keep the
        # training loader in-process as well: the default worker pool can be
        # killed during Ray actor teardown after the final training step,
        # preventing the engine receipt from being written.
        "data.dataloader_num_workers=0",
        f"data.train_batch_size={train_batch_size}",
        f"data.gen_batch_size={int(rft.get('gen_batch_size', rft.get('train_batch_size', 128)))}",
        f"data.shuffle={_hydra_bool(rft.get('data_shuffle', True))}",
        f"data.seed={seed}",
        f"data.max_prompt_length={int(rft.get('max_prompt_length', 2048))}",
        f"data.max_response_length={int(rft.get('max_response_length', 2048))}",
        "data.filter_overlong_prompts=True",
        f"data.truncation={str(rft.get('truncation') or 'error')}",
        f"actor_rollout_ref.model.path={model}",
        f"actor_rollout_ref.actor.optim.lr={float(rft.get('learning_rate', 1.0e-6))}",
        f"actor_rollout_ref.actor.optim.lr_scheduler_type={str(rft.get('lr_scheduler_type') or 'constant')}",
        f"actor_rollout_ref.actor.optim.lr_warmup_steps_ratio={float(rft.get('lr_warmup_steps_ratio', 0.0))}",
        f"actor_rollout_ref.actor.optim.weight_decay={float(rft.get('weight_decay', 0.01))}",
        "actor_rollout_ref.actor.checkpoint.save_contents=['model']",
        "actor_rollout_ref.actor.checkpoint.load_contents=['model']",
        f"actor_rollout_ref.model.use_remove_padding={_hydra_bool(rft.get('use_remove_padding', True))}",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={_hydra_bool(rft.get('enable_gradient_checkpointing', True))}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={int(rft.get('ppo_micro_batch_size_per_gpu', 4))}",
        f"actor_rollout_ref.actor.ppo_epochs={int(rft.get('ppo_epochs', 1))}",
        f"actor_rollout_ref.actor.shuffle={_hydra_bool(rft.get('actor_shuffle', False))}",
        f"actor_rollout_ref.actor.data_loader_seed={seed}",
        f"actor_rollout_ref.actor.use_dynamic_bsz={_hydra_bool(rft.get('actor_use_dynamic_bsz', False))}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={int(rft.get('actor_max_token_len_per_gpu', 16384))}",
        f"actor_rollout_ref.actor.ulysses_sequence_parallel_size={int(rft.get('ulysses_sequence_parallel_size', 1))}",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={_hydra_bool(rft.get('actor_param_offload', False))}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={_hydra_bool(rft.get('actor_optimizer_offload', False))}",
        f"actor_rollout_ref.actor.fsdp_config.seed={seed}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={log_prob_micro_batch_size_per_gpu}",
        f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={int(rft.get('log_prob_max_token_len_per_gpu', 16384))}",
        f"actor_rollout_ref.ref.fsdp_config.param_offload={_hydra_bool(rft.get('reference_param_offload', True))}",
        f"actor_rollout_ref.actor.use_kl_loss={_hydra_bool(rft.get('use_kl_loss', True))}",
        f"actor_rollout_ref.actor.kl_loss_coef={float(rft.get('kl_loss_coef', 0.001))}",
        f"actor_rollout_ref.actor.kl_loss_type={str(rft.get('kl_loss_type') or 'low_var_kl')}",
        f"actor_rollout_ref.actor.clip_ratio={float(rft.get('clip_ratio', 0.2))}",
        f"actor_rollout_ref.actor.clip_ratio_low={float(rft.get('clip_ratio_low', 0.2))}",
        f"actor_rollout_ref.actor.clip_ratio_high={float(rft.get('clip_ratio_high', 0.2))}",
        f"actor_rollout_ref.actor.clip_ratio_c={float(rft.get('clip_ratio_c', 3.0))}",
        f"actor_rollout_ref.actor.loss_agg_mode={str(rft.get('loss_agg_mode') or 'token-mean')}",
        f"actor_rollout_ref.actor.entropy_coeff={float(rft.get('entropy_coeff', 0.0))}",
        f"actor_rollout_ref.actor.grad_clip={float(rft.get('grad_clip', 1.0))}",
        f"actor_rollout_ref.rollout.name={rollout_backend}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={int(rft.get('tensor_model_parallel_size', 1))}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={float(rft.get('gpu_memory_utilization', 0.6))}",
        f"actor_rollout_ref.rollout.dtype={str(rft.get('rollout_dtype') or 'bfloat16')}",
        f"actor_rollout_ref.rollout.ignore_eos={_hydra_bool(rft.get('ignore_eos', False))}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={log_prob_micro_batch_size_per_gpu}",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={int(rft.get('log_prob_max_token_len_per_gpu', 16384))}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={int(rft.get('max_num_batched_tokens', 10216))}",
        f"actor_rollout_ref.rollout.max_num_seqs={int(rft.get('max_num_seqs', 1024))}",
        f"actor_rollout_ref.rollout.enable_chunked_prefill={_hydra_bool(rft.get('enable_chunked_prefill', False))}",
        f"actor_rollout_ref.rollout.enforce_eager={_hydra_bool(rft.get('enforce_eager', True))}",
        f"actor_rollout_ref.rollout.free_cache_engine={_hydra_bool(rft.get('free_cache_engine', True))}",
        f"actor_rollout_ref.rollout.n={rollout_n}",
        f"actor_rollout_ref.rollout.seed={seed}",
        f"actor_rollout_ref.rollout.temperature={float(rft.get('temperature', 1.0))}",
        f"actor_rollout_ref.rollout.top_p={float(rft.get('top_p', 1.0))}",
        f"actor_rollout_ref.rollout.top_k={int(rft.get('top_k', -1))}",
        f"algorithm.use_kl_in_reward={_hydra_bool(rft.get('use_kl_in_reward', False))}",
        f"algorithm.kl_ctrl.type={str(rft.get('kl_controller') or 'fixed')}",
        f"algorithm.kl_ctrl.kl_coef={float(rft.get('kl_coef', 0.0))}",
        "+algorithm.filter_groups.enable=false",
        "reward.reward_manager.source=importlib",
        f"reward.reward_manager.module.path={reward_manager_path}",
        "reward.reward_manager.name=ADERewardManager",
        f"reward.custom_reward_function.path={reward_path}",
        f"reward.custom_reward_function.name={str(rft.get('reward_entrypoint') or 'compute_score')}",
        "+reward.training_outcome_adapter.adapter_id="
        f"{str(outcome_adapter['adapter_id'])}",
        f"+reward.llm_judge.enabled={_hydra_bool(judge_required)}",
        f"+reward.defer_until_batch={_hydra_bool(judge_required or _group_credit_enabled(rft))}",
        f"+ray_kwargs.ray_init.runtime_env.py_executable={python}",
        f"+ray_kwargs.ray_init.namespace={ray_namespace}",
        f"+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH={json.dumps(runtime_pythonpath)}",
        "+ray_kwargs.ray_init.runtime_env.env_vars.TRANSFER_QUEUE_NAMESPACE="
        f"{transfer_queue_namespace}",
        "trainer.logger=['console','wandb']" if wandb_enabled else "trainer.logger=['console']",
        f"trainer.project_name={wandb_tracking.get('project') or str(rft.get('project_name') or 'ade_reward_design')}",
        f"trainer.experiment_name={wandb_tracking.get('name') or _safe_name(str(config.get('task_id') or 'reward_candidate'))}",
        f"trainer.n_gpus_per_node={gpus_per_node}",
        f"trainer.nnodes={nodes}",
        f"trainer.default_local_dir={checkpoint_root}",
        f"trainer.rollout_data_dir={rollout_data_dir}",
        f"trainer.save_freq={int(rft.get('artifact_interval', rft.get('save_freq', 4)))}",
        f"trainer.test_freq={int(rft.get('test_freq', -1))}",
        "trainer.val_before_train=False",
        f"trainer.total_epochs={int(rft.get('total_epochs', 1))}",
        f"trainer.total_training_steps={_optional_hydra_value(rft.get('total_training_steps'))}",
        f"trainer.resume_mode={str(rft.get('resume_mode') or 'disable')}",
    ]
    if _group_credit_enabled(rft):
        command.extend(
            [
                "+reward.group_credit_function.enabled=true",
                f"+reward.group_credit_function.path={str(rft.get('group_credit_function_path') or reward_path)}",
                f"+reward.group_credit_function.name={str(rft.get('group_credit_entrypoint') or 'assign_group_credit')}",
                "+reward.group_credit_function.schema_version=ade.group_credit.v1",
            ]
        )
    if prompt_protocol["mode"] == "raw_completion":
        protocol_module = Path(__file__).with_name("verl_prompt_protocol.py").resolve()
        agent_loop_config = (
            Path(__file__).resolve().parents[3]
            / "configs"
            / "verl"
            / "raw-completion-agent-loop.yaml"
        ).resolve()
        if not agent_loop_config.is_file():
            raise ValueError(
                f"raw_completion agent-loop config does not exist: {agent_loop_config}"
            )
        command.extend(
            [
                f"data.custom_cls.path={protocol_module}",
                "data.custom_cls.name=RawCompletionDataset",
                "actor_rollout_ref.rollout.agent.default_agent_loop=raw_completion",
                "actor_rollout_ref.rollout.agent.agent_loop_config_path="
                f"{agent_loop_config}",
            ]
        )
    if judge_required:
        assert local_judge is not None
        # The deployment-local Judge has eight endpoints.  Keep deferred
        # reward batch work bounded to one CPU worker per endpoint instead of
        # forcing the whole rollout batch through a single worker.
        judge_reward_workers = min(_MAX_JUDGE_REWARD_WORKERS, max(1, rollout_n))
        command.extend(
            [
                f"reward.num_workers={judge_reward_workers}",
                f"+reward.llm_judge.gateway_url={local_judge['gateway_url']}",
                f"+reward.llm_judge.authorization_env={local_judge['authorization_env']}",
                f"+reward.llm_judge.request_timeout_seconds={float(local_judge['request_timeout_seconds'])}",
                f"+reward.llm_judge.max_failed_row_ratio={float(local_judge['max_failed_row_ratio'])}",
                "+reward.llm_judge.max_consecutive_unhealthy_jobs="
                f"{int(local_judge['max_consecutive_unhealthy_jobs'])}",
                f"+reward.llm_judge.run_id={str(config.get('run_id') or '')}",
                "+reward.llm_judge.usage_run_dir="
                f"{Path(str(config.get('usage_run_dir') or run_dir)).resolve()}",
                f"+reward.llm_judge.coordinator_id={str(config.get('coordinator_id') or '')}",
                f"+reward.llm_judge.plan_id={str(config.get('plan_id') or '')}",
                f"+reward.llm_judge.trial_id={str(config.get('trial_id') or '')}",
                "+reward.llm_judge.engine_command_id="
                f"{str(config.get('engine_command_id') or '')}",
                "+reward.llm_judge.logical_command_id="
                f"{str(config.get('logical_command_id') or '')}",
                "+reward.llm_judge.engine_attempt_id="
                f"{str(config.get('engine_attempt_id') or '')}",
                "+reward.llm_judge.engine_attempt_index="
                f"{int(config.get('engine_attempt_index') or 0)}",
                "+reward.compute_score_timeout_seconds=240.0",
                "+trainer.v1.sync.parameter_sync_step="
                f"{train_batch_size // ppo_mini_batch_size}",
            ]
        )
    artifact_root = config.get("trial_artifact_root")
    if artifact_root:
        command.append(f"+trial_artifact_root={Path(str(artifact_root)).resolve()}")
    for name, value in _hf_environment(project_root, rft).items():
        command.append(
            f"+ray_kwargs.ray_init.runtime_env.env_vars.{name}={json.dumps(value)}"
        )
    for name, value in sorted(wandb_environment.items()):
        command.append(
            f"+ray_kwargs.ray_init.runtime_env.env_vars.{name}={json.dumps(value)}"
        )
    scope_environment = {
        "ADE_RUN_ID": str(config.get("run_id") or ""),
        "ADE_COORDINATOR_ID": str(config.get("coordinator_id") or ""),
        "ADE_PLAN_ID": str(config.get("plan_id") or ""),
        "ADE_TRIAL_ID": str(config.get("trial_id") or ""),
        "ADE_ENGINE_COMMAND_ID": str(config.get("engine_command_id") or ""),
        "ADE_ENGINE_LOGICAL_COMMAND_ID": str(
            config.get("logical_command_id") or ""
        ),
        "ADE_ENGINE_ATTEMPT_ID": str(config.get("engine_attempt_id") or ""),
        "ADE_WORKLOAD": "training",
    }
    for name, value in scope_environment.items():
        command.append(
            f"+ray_kwargs.ray_init.runtime_env.env_vars.{name}={json.dumps(value)}"
        )
    if rft.get("placement_group_id"):
        command.append(
            "+ray_kwargs.ray_init.runtime_env.env_vars."
            "ADE_VERL_PLACEMENT_GROUP_ID="
            f"{json.dumps(str(rft['placement_group_id']))}"
        )
    return command


def build_verl_merge_command(
    config: dict[str, Any],
    *,
    actor_checkpoint: str | Path,
    target_dir: str | Path,
) -> list[str]:
    project_root = Path(config.get("project_root") or Path.cwd()).resolve()
    python = _environment_python(project_root, _rft_config(config))
    return [
        str(python),
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(Path(actor_checkpoint).resolve()),
        "--target_dir",
        str(Path(target_dir).resolve()),
    ]


def _rollout_segment(rft: dict[str, Any], config: dict[str, Any]) -> str:
    segment_step = rft.get("segment_step")
    default_segment = (
        f"segment_{int(segment_step):03d}"
        if segment_step is not None
        else str(config.get("engine_run_id") or "segment_000")
    )
    return str(rft.get("reward_trace_segment_id") or default_segment)


def run_verl_rft(
    config: dict[str, Any],
    *,
    run_dir: str | Path,
    export_final_checkpoint: bool = True,
) -> dict[str, Any]:
    rft = _rft_config(config)
    try:
        return _run_verl_rft(
            config,
            run_dir=run_dir,
            export_final_checkpoint=export_final_checkpoint,
        )
    finally:
        if rft.get("model_staging"):
            from ade.engine.execution.ray import (
                release_artifact_cache_consumer_on_gpu_nodes,
            )

            release_artifact_cache_consumer_on_gpu_nodes(
                cache_dir=str(rft["model_cache_dir"]),
                consumer_id=str(config.get("engine_command_id") or ""),
                node_ids=tuple(rft.get("model_staging_node_ids") or ()),
            )


def _run_verl_rft(
    config: dict[str, Any],
    *,
    run_dir: str | Path,
    export_final_checkpoint: bool = True,
) -> dict[str, Any]:
    started_at = time.time()
    run_path = Path(run_dir).resolve()
    run_path.mkdir(parents=True, exist_ok=True)
    logs_dir = run_path / "logs" / "verl"
    logs_dir.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(config)
    rft = _rft_config(config)
    model_staging: dict[str, Any] = {"status": "disabled", "nodes": []}
    if rft.get("model_staging"):
        from ade.engine.execution.ray import stage_model_on_gpu_nodes

        staged, model_staging = stage_model_on_gpu_nodes(
            {
                "base_model": str(config.get("base_model") or config.get("model") or _BASE_MODEL),
                "ade_run_id": str(config.get("run_id") or ""),
                "staging_consumer_id": str(config.get("engine_command_id") or ""),
                "model_staging": True,
                "model_cache_dir": rft.get("model_cache_dir"),
                "model_cache_max_gb": rft.get("model_cache_max_gb"),
                "model_cache_lock_stale_seconds": rft.get(
                    "model_cache_lock_stale_seconds"
                ),
            },
            model_field="base_model",
            node_ids=tuple(rft.get("model_staging_node_ids") or ()),
        )
        config["base_model"] = staged["base_model"]
        config["source_model_path"] = staged["source_model_path"]
        _write_json(run_path / "receipts" / "model-staging.json", model_staging)
    wandb_environment, wandb_tracking = _wandb_environment(config, run_dir=run_path)
    segment_step = int(rft.get("segment_step") or rft.get("total_training_steps") or 0)
    suffix = f"_step_{segment_step}" if rft.get("segment_step") is not None else ""
    training_log = logs_dir / f"training{suffix}.log"
    merge_log = logs_dir / f"merge{suffix}.log"
    snapshot_path = run_path / f"verl_config_snapshot{suffix}.json"
    metadata_path = run_path / f"reward_design_reproduction{suffix}.json"
    result_path = run_path / f"verl_engine_result{suffix}.json"
    source_root = _source_root(config, rft)
    train_files, validation_files = _training_files(
        config,
        project_root=Path(config.get("project_root") or Path.cwd()).resolve(),
        run_dir=run_path,
    )
    dataset_validation = {
        "train": [validate_verl_dataset(path) for path in train_files],
        "online_validation": [validate_verl_dataset(path) for path in validation_files],
    }
    command = build_verl_training_command(config, run_dir=run_path)
    snapshot = {
        "backend": "verl",
        "command": command,
        "config": config,
        "source_root": str(source_root),
    }
    _write_json(snapshot_path, snapshot)
    env = os.environ.copy()
    project_root = Path(config.get("project_root") or Path.cwd()).resolve()
    env["PYTHONPATH"] = _prepend_path(
        str(source_root), _prepend_path(str(project_root), env.get("PYTHONPATH"))
    )
    seed = _required_seed(rft)
    env["PYTHONHASHSEED"] = str(seed)
    env["RAY_ADDRESS"] = str(config.get("ray_address") or env.get("RAY_ADDRESS") or "auto")
    env.update(_hf_environment(Path(config.get("project_root") or Path.cwd()).resolve(), rft))
    env.update(wandb_environment)
    env.update(
        {
            "ADE_RUN_ID": str(config.get("run_id") or ""),
            "ADE_COORDINATOR_ID": str(config.get("coordinator_id") or ""),
            "ADE_PLAN_ID": str(config.get("plan_id") or ""),
            "ADE_TRIAL_ID": str(config.get("trial_id") or ""),
            "ADE_ENGINE_COMMAND_ID": str(config.get("engine_command_id") or ""),
            "ADE_WORKLOAD": "training",
        }
    )
    if rft.get("placement_group_id"):
        env["ADE_VERL_PLACEMENT_GROUP_ID"] = str(rft["placement_group_id"])
    if wandb_tracking.get("enabled"):
        Path(wandb_environment["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
    runtime_secret_names: list[str] = []
    if _judge_enrichment_requested(rft, rft["reward_function_path"]):
        judge_auth_env = str(_local_judge_binding(config, rft)["authorization_env"])
        if not env.get(judge_auth_env):
            dotenv_value = read_dotenv_value(judge_auth_env)
            if dotenv_value:
                env[judge_auth_env] = dotenv_value
        if not env.get(judge_auth_env):
            raise ValueError(
                f"{judge_auth_env} is required for Judge-dependent RFT"
            )
        runtime_secret_names.append(judge_auth_env)
    if wandb_tracking.get("mode") == "online":
        if not env.get("WANDB_API_KEY"):
            dotenv_key = _read_wandb_api_key_from_dotenv()
            if dotenv_key:
                env["WANDB_API_KEY"] = dotenv_key
        if not env.get("WANDB_API_KEY"):
            raise ValueError("WANDB_API_KEY is required when rft.wandb.mode is online")
        runtime_secret_names.append("WANDB_API_KEY")
    if runtime_secret_names:
        hook = "ade.tasks.reward_design.verl_runtime._inject_runtime_secrets"
        existing_hook = env.get("RAY_RUNTIME_ENV_HOOK")
        if existing_hook and existing_hook != hook:
            raise ValueError(
                "RAY_RUNTIME_ENV_HOOK is already configured; cannot propagate runtime secrets"
            )
        env["RAY_RUNTIME_ENV_HOOK"] = hook
        env[_RAY_SECRET_ENV_NAMES] = ",".join(runtime_secret_names)
    base_metadata = {
        "reward_function_digest": str(rft.get("reward_function_digest") or _sha256(Path(rft["reward_function_path"]))),
        "verl_config_path": str(snapshot_path),
        "data_files": {
            "train": [_file_identity(Path(path)) for path in train_files],
            "online_validation": [_file_identity(Path(path)) for path in validation_files],
        },
        "data_schema_validation": dataset_validation,
        "seed": seed,
        "ray_address": env["RAY_ADDRESS"],
        "logs": {"training": str(training_log), "merge": str(merge_log)},
        "wandb": wandb_tracking,
        "model_staging": model_staging,
    }
    try:
        _run_command(command, cwd=source_root, env=env, log_path=training_log)
        analysis_sources = _freeze_verl_analysis_sources(
            config=config,
            run_path=run_path,
            rft=rft,
            wandb_tracking=wandb_tracking,
            reward_function_digest=base_metadata["reward_function_digest"],
        )
        if not export_final_checkpoint:
            reward_metrics, rollout_metrics = _training_metrics(training_log)
            metadata = {
                **base_metadata,
                "commands": {"training": command},
                "reward_distribution": reward_metrics,
                "rollout_metrics": rollout_metrics,
                "checkpoint_paths": {},
                "segment_step": segment_step or None,
                "failure_category": None,
                "analysis_sources": analysis_sources,
            }
            _write_json(metadata_path, metadata)
            result = {
                "status": "completed",
                "metrics": {
                    "backend": "verl",
                    "reward_function_digest": base_metadata["reward_function_digest"],
                    "verl_config_path": str(snapshot_path),
                    "reproduction_metadata_path": str(metadata_path),
                    "reward_distribution": reward_metrics,
                    "rollout_metrics": rollout_metrics,
                    "training_log_path": str(training_log),
                    **analysis_sources,
                },
                "factual_result": "VERL RFT training completed; interval checkpoints are exported by the workflow",
                "elapsed_seconds": round(time.time() - started_at, 3),
                "segment_step": segment_step or None,
            }
            _write_json(result_path, result)
            return result
        actor_checkpoint = _latest_actor_checkpoint(run_path / "checkpoints" / "verl")
        exported_checkpoint = (
            run_path / "exported_checkpoints" / f"checkpoint-{segment_step}"
            if rft.get("segment_step") is not None
            else run_path / "exported_checkpoint"
        )
        merge_command = build_verl_merge_command(
            config,
            actor_checkpoint=actor_checkpoint,
            target_dir=exported_checkpoint,
        )
        _run_command(merge_command, cwd=source_root, env=env, log_path=merge_log)
        _validate_exported_checkpoint(
            exported_checkpoint,
            require_tokenizer=bool(rft.get("require_tokenizer_artifacts", False)),
        )
        reward_metrics, rollout_metrics = _training_metrics(training_log)
        metadata = {
            **base_metadata,
            "commands": {"training": command, "merge": merge_command},
            "reward_distribution": reward_metrics,
            "rollout_metrics": rollout_metrics,
            "checkpoint_paths": {
                "exported": str(exported_checkpoint),
            },
            "transient_actor_checkpoint_removed": str(actor_checkpoint.parent),
            "segment_step": segment_step or None,
            "failure_category": None,
            "analysis_sources": analysis_sources,
        }
        _write_json(metadata_path, metadata)
        _remove_transient_verl_checkpoint(
            actor_checkpoint.parent,
            checkpoint_root=run_path / "checkpoints" / "verl",
        )
        result = {
            "status": "completed",
            "checkpoint_path": str(exported_checkpoint),
            "best_checkpoint_path": str(exported_checkpoint),
            "metrics": {
                "backend": "verl",
                "reward_function_digest": base_metadata["reward_function_digest"],
                "verl_config_path": str(snapshot_path),
                "reproduction_metadata_path": str(metadata_path),
                "reward_distribution": reward_metrics,
                "rollout_metrics": rollout_metrics,
                "training_log_path": str(training_log),
                "merge_log_path": str(merge_log),
                **analysis_sources,
            },
            "factual_result": "VERL RFT completed and exported a Hugging Face checkpoint",
            "elapsed_seconds": round(time.time() - started_at, 3),
            "segment_step": segment_step or None,
        }
    except Exception as exc:
        failure_category = _failure_category(exc, training_log=training_log)
        metadata = {
            **base_metadata,
            "failure_category": failure_category,
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json(metadata_path, metadata)
        result = {
            "status": "error",
            "error": metadata["error"],
            "failure_category": failure_category,
            "metrics": {
                "backend": "verl",
                "reward_function_digest": base_metadata["reward_function_digest"],
                "verl_config_path": str(snapshot_path),
                "reproduction_metadata_path": str(metadata_path),
                "training_log_path": str(training_log),
                "merge_log_path": str(merge_log),
            },
            "factual_result": metadata["error"],
            "elapsed_seconds": round(time.time() - started_at, 3),
        }
        _write_json(result_path, result)
        raise RuntimeError(metadata["error"]) from exc
    _write_json(result_path, result)
    return result


def _freeze_verl_analysis_sources(
    *,
    config: dict[str, Any],
    run_path: Path,
    rft: dict[str, Any],
    wandb_tracking: dict[str, Any],
    reward_function_digest: str,
) -> dict[str, Any]:
    plan_id, trial_id = _trial_identity(config)
    trial_uid = f"{plan_id}/{trial_id}"
    trace = finalize_reward_trace(
        run_dir=run_path,
        trial_uid=trial_uid,
        reward_function_sha256=reward_function_digest,
        reward_transformations_active=bool(rft.get("use_kl_in_reward", False)),
        snapshot_steps=_analysis_snapshot_steps(rft),
        prompt_groups_per_step=int(rft.get("train_batch_size", 256)),
        responses_per_group=int(rft.get("rollout_n", 8)),
        group_credit_enabled=_group_credit_enabled(rft),
    )
    metrics: dict[str, Any] = {
        "analysis_sources_contract_version": 2 if _group_credit_enabled(rft) else 1,
        "reward_rollout_trace_manifest_path": str(
            run_path / "analysis_sources" / "reward_trace_manifest.json"
        ),
        "reward_rollout_trace_status": trace.get("status"),
    }
    if trace.get("status") == "complete":
        population_path = run_path / "analysis_sources" / "reward_population_summary.json"
        if population_path.exists():
            metrics["reward_rollout_population_summary_path"] = str(population_path)
    audit_payload = wandb_tracking.get("audit_ref")
    if not isinstance(audit_payload, dict):
        metrics["training_telemetry_available"] = False
        metrics["training_telemetry_reason"] = "training_tracking_disabled"
        return metrics
    analysis_root = run_path / "analysis_sources"
    total_training_steps = int(rft.get("total_training_steps") or 0)
    metric_profile = MetricProfile(
        profile_id=VERL_METRIC_PROFILE.profile_id,
        exact_names=VERL_METRIC_PROFILE.exact_names,
        anchored_patterns=VERL_METRIC_PROFILE.anchored_patterns,
        expected_steps=tuple(range(1, total_training_steps + 1)),
        required=total_training_steps > 0,
    )
    segment_step = rft.get("segment_step")
    training_log_name = (
        f"training_step_{int(segment_step)}.log"
        if segment_step is not None
        else "training.log"
    )
    telemetry_spec = TelemetryExportSpec(
        audit_ref=audit_ref_from_dict(audit_payload),
        metric_profile=metric_profile,
        telemetry_path=analysis_root / "training_telemetry.jsonl",
        manifest_path=analysis_root / "training_telemetry_manifest.json",
        supplemental_history_path=run_path / "logs" / "verl" / training_log_name,
    )
    telemetry = export_exact_wandb_telemetry(telemetry_spec)
    metrics["training_telemetry_manifest_path"] = str(telemetry_spec.manifest_path)
    metrics["training_telemetry_available"] = telemetry.get("status") == "complete"
    if telemetry_spec.telemetry_path.exists():
        metrics["training_telemetry_path"] = str(telemetry_spec.telemetry_path)
    telemetry_summary_path = analysis_root / "training_telemetry_summary.json"
    if telemetry_summary_path.exists():
        metrics["training_telemetry_summary_path"] = str(telemetry_summary_path)
    return metrics


def _analysis_snapshot_steps(rft: dict[str, Any]) -> tuple[int, ...]:
    final_step = rft.get("segment_step") or rft.get("total_training_steps")
    if final_step is None:
        return ()
    segment_step = int(final_step)
    interval = int(
        rft.get("artifact_interval")
        or segment_step
    )
    if segment_step < 1 or interval < 1:
        raise ValueError("RFT analysis snapshot step and interval must be positive")
    steps = list(range(interval, segment_step + 1, interval))
    if not steps or steps[-1] != segment_step:
        steps.append(segment_step)
    return tuple(steps)


def export_verl_checkpoint(
    config: dict[str, Any],
    *,
    run_dir: str | Path,
    step: int,
    poll_seconds: float = 2,
    training_failed: threading.Event | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    rft = _rft_config(config)
    actor = run_path / "checkpoints" / "verl" / f"global_step_{int(step)}" / "actor"
    artifact_root_value = config.get("trial_artifact_root")
    target = (
        Path(str(artifact_root_value)).resolve()
        / "checkpoints"
        / f"rl-step-{int(step):03d}"
        if artifact_root_value
        else run_path / "exported_checkpoints" / f"checkpoint-{int(step)}"
    )
    log_path = run_path / "logs" / "verl" / f"merge_step_{int(step)}.log"
    require_tokenizer = bool(rft.get("require_tokenizer_artifacts", False))
    if target.is_dir():
        _validate_exported_checkpoint(target, require_tokenizer=require_tokenizer)
        _remove_transient_verl_checkpoint(
            actor.parent,
            checkpoint_root=run_path / "checkpoints" / "verl",
        )
        return {
            "checkpoint_path": str(target),
            "merge_log_path": str(log_path),
            "step": int(step),
        }

    world_size = int(rft.get("nodes") or 1) * int(rft.get("gpus_per_node") or 1)
    initial_result = _result_identity(run_path / "verl_engine_result.json")
    while not _actor_checkpoint_ready(actor, world_size=world_size):
        if training_failed is not None and training_failed.is_set():
            raise RuntimeError(
                f"VERL training failed before checkpoint step {step} was ready"
            )
        _raise_if_training_failed(run_path, step=step, ignored_identity=initial_result)
        time.sleep(max(0.05, float(poll_seconds)))

    project_root = Path(config.get("project_root") or Path.cwd()).resolve()
    source_root = _source_root(config, rft)
    env = os.environ.copy()
    env["PYTHONPATH"] = _prepend_path(
        str(source_root), _prepend_path(str(project_root), env.get("PYTHONPATH"))
    )
    env.update(_hf_environment(project_root, rft))
    command = build_verl_merge_command(
        config,
        actor_checkpoint=actor,
        target_dir=target,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _run_command(command, cwd=source_root, env=env, log_path=log_path)
    _validate_exported_checkpoint(target, require_tokenizer=require_tokenizer)
    _remove_transient_verl_checkpoint(
        actor.parent,
        checkpoint_root=run_path / "checkpoints" / "verl",
    )
    return {
        "checkpoint_path": str(target),
        "merge_log_path": str(log_path),
        "step": int(step),
    }


def _rft_config(config: dict[str, Any]) -> dict[str, Any]:
    rft = config.get("rft") if isinstance(config.get("rft"), dict) else {}
    train = config.get("train") if isinstance(config.get("train"), dict) else {}
    backend = str(rft.get("backend") or train.get("backend") or "")
    if backend != "verl":
        raise ValueError("reward-design engine requires rft.backend or train.backend set to verl")
    return dict(rft)


def _required_seed(rft: dict[str, Any]) -> int:
    value = rft.get("seed")
    if type(value) is not int or value < 0:
        raise ValueError("resolved rft.seed must be a non-negative integer")
    return value


def _actor_checkpoint_ready(path: Path, *, world_size: int) -> bool:
    if not (path / "fsdp_config.json").is_file():
        return False
    if not (path / "huggingface" / "config.json").is_file():
        return False
    return len(list(path.glob(f"model_world_size_{world_size}_rank_*.pt"))) == world_size


def _raise_if_training_failed(
    run_path: Path,
    *,
    step: int,
    ignored_identity: tuple[int, int] | None,
) -> None:
    result_path = run_path / "verl_engine_result.json"
    if _result_identity(result_path) == ignored_identity:
        return
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return
    if isinstance(result, dict) and result.get("status") == "error":
        error = str(result.get("error") or "unknown training error")
        raise RuntimeError(
            f"VERL training failed before checkpoint step {step} was ready: {error}"
        )


def _result_identity(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _remove_transient_verl_checkpoint(path: Path, *, checkpoint_root: Path) -> None:
    cleanup_checkpoint_directories(
        (path,),
        allowed_root=checkpoint_root,
        name_pattern=r"global_step_\d+",
    )


def validate_verl_dataset(path: str | Path) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.is_file():
        raise ValueError(f"VERL dataset does not exist: {dataset_path}")
    parquet = pq.ParquetFile(dataset_path)
    schema = parquet.schema_arrow
    required = {"prompt", "data_source", "reward_model", "extra_info"}
    missing = sorted(required.difference(schema.names))
    if missing:
        raise ValueError(f"VERL dataset is missing required fields {missing}: {dataset_path}")
    prompt_type = schema.field("prompt").type
    if not (pa.types.is_list(prompt_type) or pa.types.is_large_list(prompt_type)):
        raise ValueError(f"VERL dataset prompt must be a list of role/content records: {dataset_path}")
    prompt_value_type = prompt_type.value_type
    if not pa.types.is_struct(prompt_value_type) or not {"role", "content"}.issubset(
        prompt_value_type.names
    ):
        raise ValueError(f"VERL dataset prompt must contain role and content fields: {dataset_path}")
    reward_type = schema.field("reward_model").type
    if not pa.types.is_struct(reward_type) or "ground_truth" not in reward_type.names:
        raise ValueError(f"VERL dataset reward_model must contain ground_truth: {dataset_path}")
    extra_info_type = schema.field("extra_info").type
    if not pa.types.is_struct(extra_info_type):
        raise ValueError(f"VERL dataset extra_info must be a struct: {dataset_path}")
    if parquet.metadata.num_rows < 1:
        raise ValueError(f"VERL dataset must contain at least one row: {dataset_path}")
    return {
        "path": str(dataset_path),
        "row_count": parquet.metadata.num_rows,
        "fields": schema.names,
    }


def _validate_base_model(value: str) -> None:
    root = Path(value)
    required = ("config.json", "tokenizer_config.json", "tokenizer.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f"local RFT base model is missing files {missing}: {root}")
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted(set((index.get("weight_map") or {}).values()))
    else:
        shards = ["model.safetensors"] if (root / "model.safetensors").is_file() else []
    missing_shards = [
        name
        for name in shards
        if not (root / str(name)).is_file() or (root / str(name)).stat().st_size == 0
    ]
    if not shards or missing_shards:
        raise ValueError(f"local RFT base model has missing weight shards {missing_shards}: {root}")


def _wandb_environment(
    config: dict[str, Any], *, run_dir: str | Path
) -> tuple[dict[str, str], dict[str, Any]]:
    rft = _rft_config(config)
    neutral = config.get("training_telemetry")
    settings = neutral if isinstance(neutral, dict) else rft.get("wandb")
    settings = settings if isinstance(settings, dict) else {}
    settings = dict(settings)
    if not str(settings.get("base_url") or "").strip():
        dotenv_base_url = read_dotenv_value("WANDB_BASE_URL")
        if dotenv_base_url:
            settings["base_url"] = dotenv_base_url
    plan_id, trial_id = _trial_identity(config)
    tracking_root = (
        Path(str(config["trial_artifact_root"])).resolve() / "audit"
        if config.get("trial_artifact_root")
        else Path(run_dir).resolve()
    )
    environment, audit = build_training_tracking(
        settings=settings,
        backend="verl",
        run_dir=tracking_root,
        run_group=str(config.get("agent_task_id") or config.get("task_id") or "ade-rft"),
        coordinator_id=str(config.get("coordinator_id") or "c000"),
        plan_id=plan_id,
        trial_id=trial_id,
        logical_command_id=str(
            config.get("logical_command_id")
            or config.get("engine_command_id")
            or "engine-command"
        ),
        attempt_id=str(config.get("engine_attempt_id") or "attempt-001"),
        fork_lineage=(
            config.get("fork_lineage")
            if isinstance(config.get("fork_lineage"), dict)
            else None
        ),
    )
    if str(audit.get("mode") or "") == "online":
        if not (os.environ.get("WANDB_API_KEY") or _read_wandb_api_key_from_dotenv()):
            raise RuntimeError("WANDB_API_KEY is missing from the VERL driver environment")
    base_url = str(settings.get("base_url") or "").strip()
    if base_url:
        from urllib.parse import urlparse

        base_url_host = urlparse(base_url).hostname or ""
        environment["NO_PROXY"] = _append_no_proxy(os.environ.get("NO_PROXY"), base_url_host)
        environment["no_proxy"] = _append_no_proxy(os.environ.get("no_proxy"), base_url_host)
    if str(audit.get("mode") or "") == "online":
        environment.update(
            {
                "WANDB_INIT_TIMEOUT": _WANDB_INIT_TIMEOUT_SECONDS,
                "HTTP_PROXY": "",
                "HTTPS_PROXY": "",
                "ALL_PROXY": "",
                "http_proxy": "",
                "https_proxy": "",
                "all_proxy": "",
            }
        )
    audit["local_authority"] = True
    return environment, audit


def _trial_identity(config: dict[str, Any]) -> tuple[str, str]:
    artifact = config.get("artifact") if isinstance(config.get("artifact"), dict) else {}
    return (
        str(config.get("plan_id") or artifact.get("plan_id") or "p000"),
        str(
            config.get("trial_id")
            or artifact.get("trial_id")
            or "t000_exact_match"
        ),
    )


def _inject_runtime_secrets(runtime_env: dict[str, Any]) -> dict[str, Any]:
    names = [
        name.strip()
        for name in os.environ.get(_RAY_SECRET_ENV_NAMES, "").split(",")
        if name.strip()
    ]
    for name in names:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"{name} is missing from the VERL driver environment")
        runtime_env.setdefault("env_vars", {})[name] = value
    return runtime_env


def _read_wandb_api_key_from_dotenv() -> str | None:
    return read_dotenv_value("WANDB_API_KEY")


def _append_no_proxy(existing: str | None, hostname: str) -> str:
    values = [value.strip() for value in str(existing or "").split(",") if value.strip()]
    if hostname not in values:
        values.append(hostname)
    return ",".join(values)


def _environment_python(project_root: Path, rft: dict[str, Any]) -> Path:
    value = rft.get("environment") or ".unified-vllm-0.19.1-verl-venv"
    environment = Path(str(value)).expanduser()
    if not environment.is_absolute():
        environment = project_root / environment
    python = environment.resolve() / "bin" / "python"
    if not python.is_file():
        raise ValueError(f"VERL environment Python does not exist: {python}")
    return python


def _hf_environment(project_root: Path, rft: dict[str, Any]) -> dict[str, str]:
    value = rft.get("hf_home")
    if value is None:
        return {}
    hf_home = Path(str(value)).expanduser()
    if not hf_home.is_absolute():
        hf_home = project_root / hf_home
    environment = {"HF_HOME": str(hf_home.resolve())}
    if bool(rft.get("hf_hub_offline")):
        environment["HF_HUB_OFFLINE"] = "1"
    return environment


def _source_root(config: dict[str, Any], rft: dict[str, Any]) -> Path:
    project_root = Path(config.get("project_root") or Path.cwd()).resolve()
    value = rft.get("verl_source") or project_root / "third_party/verl"
    source = Path(str(value)).expanduser()
    if not source.is_absolute():
        source = project_root / source
    source = source.resolve()
    if not (source / "verl" / "trainer" / "main_ppo.py").is_file():
        raise ValueError(f"VERL source tree does not contain verl.trainer.main_ppo: {source}")
    return source


def _training_files(
    config: dict[str, Any],
    *,
    project_root: Path,
    run_dir: Path | None = None,
) -> tuple[list[str], list[str]]:
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    train_values = data.get("train") or data.get("pool")
    # Engine-side validation is disabled for ADE RFT; ADE evaluation owns all
    # validation/test datasets outside the VERL training input.
    validation_values = None
    train_files = _path_list(train_values, project_root=project_root, owner="data.train")
    validation_files = _path_list(
        validation_values,
        project_root=project_root,
        owner="data.online_validation",
        allow_empty=True,
    )
    report_only = {
        str(path.resolve())
        for key in ("offline_validation", "operator_test", "test", "heldout", "held_out")
        for path in _optional_paths(data.get(key), project_root=project_root)
    }
    visible = set(train_files) | set(validation_files)
    overlap = sorted(visible.intersection(report_only))
    if overlap:
        raise ValueError(f"training-visible data overlaps report-only data: {overlap}")
    rft_value = config.get("rft")
    if not isinstance(rft_value, dict):
        raise ValueError("rft config is required")
    rft = dict(rft_value)
    prompt_protocol = _rft_prompt_protocol(rft)
    if prompt_protocol["mode"] == "chat_template":
        if run_dir is None:
            raise ValueError("chat_template RFT dataset materialization requires run_dir")
        prompt = prompt_protocol["system_prompt"]
        train_files, validation_files = _materialize_rft_prompt_overrides(
            train_files,
            validation_files,
            system_prompt=prompt["content"],
            run_dir=run_dir,
        )
    return train_files, validation_files


def _rft_prompt_protocol(rft: dict[str, Any]) -> dict[str, Any]:
    protocol = rft.get("prompt_protocol")
    if not isinstance(protocol, dict):
        raise ValueError("rft.prompt_protocol is required")
    mode = protocol.get("mode")
    if mode == "raw_completion":
        if set(protocol) != {"mode"}:
            raise ValueError("raw_completion prompt_protocol requires exactly mode")
        return protocol
    if mode != "chat_template" or set(protocol) != {
        "mode", "system_prompt", "chat_template"
    }:
        raise ValueError(
            "chat_template prompt_protocol requires mode/system_prompt/chat_template"
        )
    prompt = protocol.get("system_prompt")
    if not isinstance(prompt, dict) or set(prompt) != {"id", "content", "digest"}:
        raise ValueError("chat_template system_prompt requires id/content/digest")
    content = prompt.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("chat_template system_prompt content must be non-empty")
    if hashlib.sha256(content.encode("utf-8")).hexdigest() != prompt.get("digest"):
        raise ValueError("chat_template system_prompt digest mismatch")
    template = protocol.get("chat_template")
    if not isinstance(template, dict) or set(template) != {"id", "digest"}:
        raise ValueError("chat_template native binding requires id/digest")
    return protocol


def _materialize_rft_prompt_overrides(
    train_files: list[str],
    validation_files: list[str],
    *,
    system_prompt: str,
    run_dir: Path,
) -> tuple[list[str], list[str]]:
    import pyarrow.parquet as pq

    output_root = run_dir.resolve() / "engine_audit" / "rft_datasets"
    output_root.mkdir(parents=True, exist_ok=True)
    prompt_digest = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]
    localized: dict[str, str] = {}
    paths = [*train_files, *validation_files]
    for value in paths:
        source = Path(value).resolve()
        key = str(source)
        if key in localized:
            continue
        target = output_root / f"{source.stem}__system_{prompt_digest}{source.suffix}"
        if not target.exists():
            table = pq.read_table(source)
            rows = table.to_pylist()
            for row in rows:
                messages = row.get("prompt")
                if not isinstance(messages, list) or not messages:
                    raise ValueError(f"RFT dataset prompt must be a non-empty list: {source}")
                rewritten = []
                system_added = False
                for message in messages:
                    if not isinstance(message, dict):
                        raise ValueError(f"RFT dataset prompt messages must be mappings: {source}")
                    if message.get("role") == "system":
                        if not system_added:
                            rewritten.append({"role": "system", "content": system_prompt})
                            system_added = True
                        continue
                    rewritten.append(message)
                if not system_added:
                    rewritten.insert(0, {"role": "system", "content": system_prompt})
                row["prompt"] = rewritten
            import pyarrow as pa

            pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), target)
        localized[key] = str(target)
    return (
        [localized[str(Path(value).resolve())] for value in train_files],
        [localized[str(Path(value).resolve())] for value in validation_files],
    )


def _path_list(value: Any, *, project_root: Path, owner: str, allow_empty: bool = False) -> list[str]:
    paths = _optional_paths(value, project_root=project_root)
    if not paths:
        if allow_empty:
            return []
        raise ValueError(f"{owner} requires at least one file")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"{owner} files do not exist: {missing}")
    return [str(path.resolve()) for path in paths]


def _optional_paths(value: Any, *, project_root: Path) -> list[Path]:
    values = value if isinstance(value, list) else ([] if value is None else [value])
    paths = []
    for item in values:
        raw = item.get("path") if isinstance(item, dict) else item
        if not isinstance(raw, str) or not raw:
            continue
        path = Path(raw).expanduser()
        paths.append((project_root / path).resolve() if not path.is_absolute() else path.resolve())
    return paths


def _required_file(value: Any, owner: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} is required")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{owner} does not exist: {path}")
    return path


def _local_judge_binding(config: dict[str, Any], rft: dict[str, Any]) -> dict[str, Any]:
    value = rft.get("local_judge")
    required = {
        "gateway_url",
        "authorization_env",
        "request_timeout_seconds",
        "max_failed_row_ratio",
        "max_consecutive_unhealthy_jobs",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Judge-dependent RFT requires an exact resolved local_judge binding")
    for field in ("gateway_url", "authorization_env"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"rft.local_judge.{field} is required")
    parsed = urllib.parse.urlparse(value["gateway_url"])
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("rft.local_judge.gateway_url must be an HTTP(S) private gateway")
    if parsed.hostname in {"127.0.0.1", "localhost"}:
        raise ValueError("remote Reward workers cannot use a loopback Judge gateway")
    if parsed.port in set(range(8901, 8909)):
        raise ValueError("ADE Reward runtime cannot bind direct vLLM endpoint ports")
    request_timeout = value["request_timeout_seconds"]
    ratio = value["max_failed_row_ratio"]
    if type(request_timeout) not in {int, float} or float(request_timeout) <= 0:
        raise ValueError("rft.local_judge.request_timeout_seconds must be positive")
    consecutive = value["max_consecutive_unhealthy_jobs"]
    if type(ratio) not in {int, float} or not 0 <= float(ratio) <= 1:
        raise ValueError("rft.local_judge.max_failed_row_ratio must be within [0,1]")
    if type(consecutive) is not int or consecutive < 1:
        raise ValueError("rft.local_judge.max_consecutive_unhealthy_jobs must be positive")
    for field in ("run_id", "engine_command_id"):
        if not isinstance(config.get(field), str) or not str(config[field]).strip():
            raise ValueError(f"Judge-dependent RFT requires {field}")
    return dict(value)


def _reward_requires_llm_judge(path: str | Path) -> bool:
    reward_path = Path(path).resolve()
    try:
        tree = ast.parse(
            reward_path.read_text(encoding="utf-8"), filename=str(reward_path)
        )
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise ValueError(
            f"reward function cannot be inspected: {reward_path}"
        ) from error
    entrypoints = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "compute_score"
    ]
    if len(entrypoints) != 1:
        raise ValueError(
            "reward function requires exactly one compute_score entrypoint"
        )
    return any(
        isinstance(node, ast.Await)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "llm_judge"
        for node in ast.walk(entrypoints[0])
    )


def _judge_enrichment_requested(rft: dict[str, Any], reward_path: str | Path) -> bool:
    value = rft.get("judge_enrichment")
    if not isinstance(value, dict) or type(value.get("enabled")) is not bool:
        raise ValueError("rft.judge_enrichment.enabled must be explicit boolean")
    enabled = bool(value["enabled"])
    observed = _reward_requires_llm_judge(reward_path)
    group_credit = rft.get("group_credit")
    group_credit_enabled = (
        isinstance(group_credit, dict) and group_credit.get("enabled") is True
    )
    artifact_acquires_judge = enabled and not group_credit_enabled
    if artifact_acquires_judge != observed:
        raise ValueError(
            "reward.py Judge capability usage does not match the resolved "
            "Judge/Group Credit acquisition owner"
        )
    return enabled


def _run_command(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}; see {log_path}"
        )


def _latest_actor_checkpoint(root: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in root.glob("global_step_*/actor"):
        match = re.fullmatch(r"global_step_(\d+)", path.parent.name)
        if match and path.is_dir():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise RuntimeError(f"VERL training produced no actor checkpoint under {root}")
    return max(candidates, key=lambda item: item[0])[1]


def _validate_exported_checkpoint(path: Path, *, require_tokenizer: bool = False) -> None:
    if not (path / "config.json").is_file():
        raise RuntimeError(f"exported checkpoint is missing config.json: {path}")
    weights = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
    if not weights:
        raise RuntimeError(f"exported checkpoint has no model weights: {path}")
    if require_tokenizer and not any(
        (path / name).is_file()
        for name in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
    ):
        raise RuntimeError(f"exported checkpoint is missing tokenizer artifacts: {path}")


def _training_metrics(log_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    values: dict[str, list[float]] = {}
    if log_path.is_file():
        for match in _METRIC_PATTERN.finditer(log_path.read_text(encoding="utf-8", errors="replace")):
            values.setdefault(match.group("name"), []).append(float(match.group("value")))
    summaries = {
        name: {
            "count": len(items),
            "min": min(items),
            "max": max(items),
            "mean": sum(items) / len(items),
            "latest": items[-1],
        }
        for name, items in values.items()
    }
    reward = {name: summary for name, summary in summaries.items() if "reward" in name}
    rollout = {name: summary for name, summary in summaries.items() if "reward" not in name}
    return reward, rollout


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _failure_category(
    exc: Exception,
    *,
    training_log: Path | None = None,
) -> str:
    text = str(exc).lower()
    if training_log is not None and training_log.is_file():
        with training_log.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 1024 * 1024))
            text += "\n" + handle.read().decode("utf-8", errors="replace").lower()
    if any(
        marker in text
        for marker in (
            "rewardcommandfailed",
            "local_judge_unhealthy",
            "local rubric judge rejected job",
        )
    ):
        return "dependency_unavailable"
    if (
        "free memory on device" in text
        and "desired gpu memory utilization" in text
    ):
        return "gpu_memory_unavailable"
    if re.search(r"command failed with exit code -\d+", text):
        return "training_runtime"
    if any(
        marker in text
        for marker in ("environment", "import", "module", "brokenprocesspool")
    ):
        return "environment_compatibility"
    if "checkpoint" in text or "merge" in text:
        return "checkpoint_export"
    if "data" in text or "file" in text:
        return "data_contract"
    return "training_runtime"


def _optional_hydra_value(value: Any) -> str:
    return "null" if value is None else str(int(value))


def _group_credit_enabled(rft: dict[str, Any]) -> bool:
    group_credit = rft.get("group_credit")
    return isinstance(group_credit, dict) and group_credit.get("enabled") is True


def _hydra_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _prepend_path(value: str, existing: str | None) -> str:
    return value if not existing else f"{value}{os.pathsep}{existing}"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "reward_candidate"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
