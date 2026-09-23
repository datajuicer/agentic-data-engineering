from __future__ import annotations

import inspect
import gc
import hashlib
import json
import os
import re
import socket
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams

from ade.engine.eval.logging import get_logs_dir, setup_logging
from ade.engine.eval.contracts import EvalExample, RolloutRecord
from ade.engine.eval.extraction import split_thinking_answer
from ade.engine.eval.logs import build_eval_log_payload
from ade.engine.eval.metrics import avg_at_k_from_details, standardize_eval_metrics
from ade.engine.eval.registry import get_task, task_type_from_config
from ade.engine.eval.utils.model_protocol import (
    is_special_token,
    validate_embedding_range,
    validate_thinking_protocol,
)


def _patch_extra_special_tokens_handling() -> None:
    import transformers.tokenization_utils_base as tub
    original = tub.PreTrainedTokenizerBase._set_model_specific_special_tokens

    def safe_set(self, special_tokens):
        if isinstance(special_tokens, list):
            special_tokens = {}
        return original(self, special_tokens)

    tub.PreTrainedTokenizerBase._set_model_specific_special_tokens = safe_set


_patch_extra_special_tokens_handling()


def _safe_log_part(value: Any) -> str:
    if value is None or value == "":
        return "null"
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("_") or "null"


def _bounded_log_part(value: Any, max_length: int = 180) -> str:
    text = _safe_log_part(value)
    if len(text) <= max_length:
        return text
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    keep = max_length - len(digest) - 3
    head = max(1, keep // 2)
    tail = max(1, keep - head)
    return f"{text[:head]}__{digest}_{text[-tail:]}"


def _build_result_filename(epoch: int | None, timestamp: str, avg_k: int, suffix: str | None = None) -> str:
    if suffix:
        return f"eval_result__{_bounded_log_part(suffix)}__avgk_{avg_k}__ts_{timestamp}.json"
    return f"eval_result__epoch_{_safe_log_part(epoch)}__avgk_{avg_k}__ts_{timestamp}.json"


def _build_eval_log_name(result_suffix: str | None, epoch: int | None) -> str:
    if result_suffix:
        return f"eval__{_bounded_log_part(result_suffix)}"
    return f"eval__epoch_{_safe_log_part(epoch)}"


def _llm_accepts_arg(name: str) -> bool:
    try:
        signature = inspect.signature(LLM)
    except (TypeError, ValueError):
        return True
    return name in signature.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())


def _add_optional(kwargs: dict[str, Any], name: str, value: Any) -> None:
    if value is not None and _llm_accepts_arg(name):
        kwargs[name] = value


_TORCH_DISTRIBUTED_ENV_KEYS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_USE_AGENT_STORE",
)

_VLLM_INIT_THREAD_LOCK = threading.Lock()


def _find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _set_isolated_distributed_env(port: int) -> dict[str, str | None]:
    keys = ("MASTER_ADDR", "MASTER_PORT", *_TORCH_DISTRIBUTED_ENV_KEYS)
    old = {key: os.environ.get(key) for key in keys}
    for key in _TORCH_DISTRIBUTED_ENV_KEYS:
        os.environ.pop(key, None)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    return old


def _restore_env(values: dict[str, str | None]) -> None:
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@contextmanager
def _vllm_init_lock():
    import fcntl

    lock_path = Path(os.environ.get("ADE_VLLM_INIT_LOCK_PATH", "/tmp/ade-vllm-init.lock"))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _VLLM_INIT_THREAD_LOCK:
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _should_retry_llm_init(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(
        marker in text
        for marker in (
            "EADDRINUSE",
            "address already in use",
            "server socket has failed to listen",
            "Engine core initialization failed",
        )
    )


def _init_llm_with_port_retries(llm_kwargs: dict[str, Any], attempts: int = 5) -> LLM:
    last_exc: BaseException | None = None
    with _vllm_init_lock():
        for attempt in range(1, attempts + 1):
            port = _find_free_local_port()
            old_env = _set_isolated_distributed_env(port)
            try:
                attempt_kwargs = dict(llm_kwargs)
                _add_optional(attempt_kwargs, "master_addr", "127.0.0.1")
                _add_optional(attempt_kwargs, "master_port", port)
                print(f"[ade.engine] initializing vLLM with MASTER_PORT={port} attempt={attempt}/{attempts}", flush=True)
                return LLM(**attempt_kwargs)
            except RuntimeError as exc:
                last_exc = exc
                _cleanup_vllm_runtime()
                if attempt >= attempts or not _should_retry_llm_init(exc):
                    raise
                print(f"[ade.engine] retrying vLLM init after transient port/init failure: {exc!r}", flush=True)
                time.sleep(1.0 * attempt)
            finally:
                _restore_env(old_env)
    if last_exc:
        raise last_exc
    raise RuntimeError("vLLM initialization failed without an exception")


def _ensure_cuda_eval_env() -> None:
    os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        return
    try:
        import ray

        gpu_ids = [str(gpu_id) for gpu_id in ray.get_gpu_ids()]
    except Exception:
        gpu_ids = []
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)


def _cleanup_vllm_runtime() -> None:
    """Best-effort cleanup before a one-shot Ray eval worker returns."""
    try:
        from vllm.distributed import parallel_state

        for name in ("destroy_model_parallel", "destroy_distributed_environment", "cleanup_dist_env_and_memory"):
            cleanup = getattr(parallel_state, name, None)
            if cleanup is None:
                continue
            try:
                cleanup()
            except Exception:
                pass
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()


def _sampling_params(**kwargs: Any) -> SamplingParams:
    stop = kwargs.pop("stop", None)
    stop_values = stop if isinstance(stop, list) else ([stop] if stop else [])
    if "<|im_end|>" not in stop_values:
        stop_values.append("<|im_end|>")
    kwargs["stop"] = stop_values
    return SamplingParams(**kwargs)


def _json_sanitize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(_json_sanitize(key)): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, set):
        return [_json_sanitize(item) for item in sorted(value, key=str)]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tolist"):
        return _json_sanitize(value.tolist())
    if hasattr(value, "item"):
        try:
            return _json_sanitize(value.item())
        except Exception:
            pass
    return str(value)


def _ensure_reasoning_parser_tokenizer(
    tokenizer,
    tokenizer_path: str,
    reasoning_parser: str,
) -> tuple[Any, str]:
    if reasoning_parser != "qwen3":
        return tokenizer, tokenizer_path

    required_tokens = ("<think>", "</think>")
    encoded = {token: tokenizer.encode(token, add_special_tokens=False) for token in required_tokens}
    missing = [token for token, token_ids in encoded.items() if len(token_ids) != 1]
    if missing:
        raise ValueError(
            "vLLM qwen3 reasoning_parser requires '<think>' and '</think>' as "
            "single tokenizer tokens; evaluation never mutates a checkpoint "
            f"tokenizer. Missing={missing}; encoded_as={encoded}."
        )

    if not all(
        is_special_token(tokenizer, token, encoded[token][0])
        and token in tokenizer.all_special_tokens
        for token in required_tokens
    ):
        raise ValueError(
            "the native tokenizer must provide '<think>' and '</think>' for qwen3 reasoning_parser; "
            f"encoded_as={encoded}"
        )
    return tokenizer, tokenizer_path


def _prepare_prompts(
    model_path: str,
    dataset: dict[str, Any],
    tokenizer,
    answer_format: str,
    row_shard_index: int = 0,
    row_shard_count: int = 1,
) -> tuple[list[Any], list[EvalExample], list[int], str, list[dict[str, Any]]]:
    del model_path
    task_type = task_type_from_config(dataset, answer_format=answer_format)
    task = get_task(task_type)
    rows = task.load_examples(dataset["path"], dataset_name=dataset.get("name"), dataset_config=dataset)
    if row_shard_count > 1:
        rows_with_indices = [
            (idx, row)
            for idx, row in enumerate(rows)
            if idx % row_shard_count == row_shard_index
        ]
    else:
        rows_with_indices = list(enumerate(rows))
    row_indices = [idx for idx, _ in rows_with_indices]
    rows = [row for _, row in rows_with_indices]
    protocol = dataset.get("prompt_protocol")
    if not isinstance(protocol, dict):
        raise ValueError(
            f"dataset {dataset.get('name')!r} is missing resolved prompt_protocol"
        )
    mode = protocol.get("mode")
    if mode == "raw_completion":
        if set(protocol) != {"mode"}:
            raise ValueError("raw_completion prompt_protocol requires exactly mode")
        prompt_binding = None
        template_binding = None
        system_prompt = None
        prompt_digest = None
        template_digest = None
    elif mode == "chat_template":
        if set(protocol) != {"mode", "system_prompt", "chat_template"}:
            raise ValueError(
                "chat_template prompt_protocol requires mode/system_prompt/chat_template"
            )
        prompt_binding = protocol.get("system_prompt")
        if not isinstance(prompt_binding, dict) or set(prompt_binding) != {
            "id", "content", "digest"
        }:
            raise ValueError("resolved system_prompt requires exactly id/content/digest")
        system_prompt = prompt_binding.get("content")
        prompt_digest = prompt_binding.get("digest")
        if not isinstance(system_prompt, str) or not system_prompt:
            raise ValueError("resolved system_prompt content must be non-empty")
        if hashlib.sha256(system_prompt.encode("utf-8")).hexdigest() != prompt_digest:
            raise ValueError("resolved system_prompt digest mismatch")
        template = tokenizer.chat_template
        if not isinstance(template, str) or not template:
            raise ValueError("evaluation requires the model tokenizer native chat template")
        template_binding = protocol.get("chat_template")
        if not isinstance(template_binding, dict) or set(template_binding) != {
            "id", "digest"
        }:
            raise ValueError("dataset is missing resolved native chat-template binding")
        template_digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
        if template_binding.get("digest") != template_digest:
            raise ValueError("native chat-template digest mismatch")
    else:
        raise ValueError(
            "resolved prompt_protocol.mode must be raw_completion or chat_template"
        )
    prompts: list[Any] = []
    evidence: list[dict[str, Any]] = []
    for row in rows:
        if mode == "raw_completion":
            messages = [{"role": "user", "content": row.prompt}]
            token_ids = tokenizer(row.prompt, add_special_tokens=False)["input_ids"]
            prompt = {"prompt_token_ids": token_ids}
            rendered_prompt = row.prompt
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row.prompt},
            ]
            rendered_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt = rendered_prompt
        prompts.append(prompt)
        item = {
            "example_id": row.id,
            "prompt_protocol_mode": mode,
            "messages": messages,
            "rendered_prompt": rendered_prompt,
            "rendered_prompt_digest": hashlib.sha256(
                rendered_prompt.encode("utf-8")
            ).hexdigest(),
        }
        if mode == "raw_completion":
            item["prompt_token_ids"] = token_ids
        else:
            item.update(
                {
                    "system_prompt_id": prompt_binding["id"],
                    "system_prompt_digest": prompt_digest,
                    "chat_template_id": template_binding["id"],
                    "chat_template_digest": template_digest,
                }
            )
        evidence.append(item)
    return prompts, rows, row_indices, task_type, evidence


def _evaluated_seeds(seed: int | None, run_indices: list[int]) -> list[int] | None:
    if seed is None:
        return None
    seeds = [seed + run_index for run_index in run_indices]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"duplicate eval seeds detected: {seeds}")
    return seeds


def question_rollout_seed(seed: int | None, row_index: int, run_index: int, avg_k: int) -> int | None:
    if seed is None:
        return None
    return int(seed) + int(row_index) * int(avg_k) + int(run_index)


def _score_outputs(
    outputs,
    prompts,
    examples,
    row_indices,
    run_indices,
    avg_k,
    seed,
    answer_format: str,
    task_type: str,
    tokenizer,
    thinking_tags: tuple[str, str] | None,
    primary_metric: str | None = None,
):
    task = get_task(task_type)
    rollouts_by_repeat: list[list[RolloutRecord]] = []
    for local_index, run_index in enumerate(run_indices):
        records = []
        for i, example in enumerate(examples):
            choice = outputs[local_index * len(prompts) + i].outputs[0]
            text = choice.text
            token_ids = choice.token_ids or []
            raw_text = tokenizer.decode(token_ids, skip_special_tokens=False) if token_ids else text
            thinking_content, answer_content = split_thinking_answer(
                raw_text,
                enabled=thinking_tags is not None,
                open_tag=thinking_tags[0] if thinking_tags else "<think>",
                close_tag=thinking_tags[1] if thinking_tags else "</think>",
            )
            records.append(
                RolloutRecord(
                    example=example,
                    repeat_idx=run_index,
                    output=answer_content,
                    seed=seed + run_index if seed is not None else None,
                    thinking_content=thinking_content,
                    answer_content=answer_content,
                    num_output_tokens=len(token_ids),
                    finish_reason=str(choice.finish_reason),
                    stop_reason=choice.stop_reason,
                )
            )
        rollouts_by_repeat.append(records)
    score = task.score_rollouts(examples, rollouts_by_repeat, primary_metric=primary_metric)
    for detail, run_index in zip(score.details, run_indices):
        detail["index"] = run_index
        detail["row_indices"] = row_indices
        if seed is not None:
            detail["seed"] = seed + run_index
    task_primary_score = score.score
    avg_at_k, std, _stderr = avg_at_k_from_details(score.details)
    score.metrics = standardize_eval_metrics(
        score.metrics,
        score.details,
        primary_metric=score.primary_metric,
        configured_k=avg_k,
    )
    score.score = float(score.metrics["score"])
    score.metrics["task_primary_metric"] = score.primary_metric
    score.metrics["task_primary_score"] = task_primary_score
    score.metrics["primary_metric"] = score.primary_metric
    score.metrics["score"] = score.score
    return avg_at_k, std, score.details, score


def _score_question_n_outputs(
    outputs,
    prompts,
    examples,
    row_indices,
    run_indices,
    avg_k,
    seed,
    answer_format: str,
    task_type: str,
    tokenizer,
    thinking_tags: tuple[str, str] | None,
    primary_metric: str | None = None,
):
    task = get_task(task_type)
    rollouts_by_repeat: list[list[RolloutRecord]] = [[] for _ in run_indices]
    if len(outputs) != len(prompts):
        raise RuntimeError(f"question_n expected {len(prompts)} request outputs, got {len(outputs)}")
    for prompt_index, example in enumerate(examples):
        row_index = int(row_indices[prompt_index])
        request_output = outputs[prompt_index]
        choices = list(request_output.outputs or [])
        if len(choices) < len(run_indices):
            raise RuntimeError(
                f"question_n expected at least {len(run_indices)} choices for row {row_index}, got {len(choices)}"
            )
        for local_run_index, run_index in enumerate(run_indices):
            choice = choices[local_run_index]
            text = choice.text
            token_ids = choice.token_ids or []
            raw_text = tokenizer.decode(token_ids, skip_special_tokens=False) if token_ids else text
            thinking_content, answer_content = split_thinking_answer(
                raw_text,
                enabled=thinking_tags is not None,
                open_tag=thinking_tags[0] if thinking_tags else "<think>",
                close_tag=thinking_tags[1] if thinking_tags else "</think>",
            )
            rollouts_by_repeat[local_run_index].append(
                RolloutRecord(
                    example=example,
                    repeat_idx=run_index,
                    output=answer_content,
                    seed=question_rollout_seed(seed, row_index, run_index, avg_k),
                    thinking_content=thinking_content,
                    answer_content=answer_content,
                    num_output_tokens=len(token_ids),
                    finish_reason=str(choice.finish_reason),
                    stop_reason=choice.stop_reason,
                )
            )
    score = task.score_rollouts(examples, rollouts_by_repeat, primary_metric=primary_metric)
    for detail, run_index in zip(score.details, run_indices):
        detail["index"] = run_index
        detail["row_indices"] = row_indices
        detail["seed_strategy"] = "question_n"
        detail["seed_base"] = seed
        detail["seed_avg_k"] = avg_k
    task_primary_score = score.score
    avg_at_k, std, _stderr = avg_at_k_from_details(score.details)
    score.metrics = standardize_eval_metrics(
        score.metrics,
        score.details,
        primary_metric=score.primary_metric,
        configured_k=avg_k,
    )
    score.score = float(score.metrics["score"])
    score.metrics["task_primary_metric"] = score.primary_metric
    score.metrics["task_primary_score"] = task_primary_score
    score.metrics["primary_metric"] = score.primary_metric
    score.metrics["score"] = score.score
    score.metrics["seed_strategy"] = "question_n"
    return avg_at_k, std, score.details, score


def run_evaluation_many(
    model_path: str,
    datasets: list[dict[str, str]],
    *,
    epoch: int | None = None,
    avg_k: int = 16,
    run_indices: list[int] | None = None,
    seed: int | None = None,
    result_suffix: str | None = None,
    extra_result_metadata: dict[str, Any] | None = None,
    project_root_dir: str,
    temperature: float = 0.6,
    top_p: float = 1.0,
    top_k: int = -1,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    min_tokens: int = 0,
    dtype: str = "bfloat16",
    max_tokens: int = 16384,
    max_model_len: int | None = None,
    max_new_tokens: int | None = None,
    attention_backend: str | None = None,
    gpu_memory_utilization: float = 0.9,
    max_num_seqs: int | None = None,
    swap_space: int | None = None,
    cpu_offload_gb: int | None = None,
    enforce_eager: bool | None = None,
    max_seq_len_to_capture: int | None = None,
    kv_cache_dtype: str | None = None,
    calculate_kv_scales: bool | None = None,
    enable_prefix_caching: bool | None = None,
    capture_vllm_logs: bool = True,
    vllm_use_v1: bool | None = None,
    tokenizer_path: str | None = None,
    stop: list[str] | str | None = None,
    stop_token_ids: list[int] | None = None,
    answer_format: str = "math",
    task_type: str | None = None,
    primary_metric: str | None = None,
    row_shard_index: int = 0,
    row_shard_count: int = 1,
    data_parallel_size: int = 1,
    thinking_budget: int | None = -1,
    reasoning_parser: str | None = "",
    model_protocol: dict[str, object] | None = None,
    rollout_layout: str = "run_major",
) -> list[dict[str, Any]]:
    _ensure_cuda_eval_env()
    if vllm_use_v1 is None:
        os.environ.pop("VLLM_USE_V1", None)
    else:
        os.environ["VLLM_USE_V1"] = "1" if vllm_use_v1 else "0"
    if attention_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend.upper()
    run_indices = list(range(avg_k)) if run_indices is None else list(run_indices)
    evaluated_seeds = None if rollout_layout == "question_n" else _evaluated_seeds(seed, run_indices)
    eval_logs_dir = get_logs_dir(project_root_dir, "eval")
    log_name = _build_eval_log_name(result_suffix, epoch)
    logger = setup_logging(log_name, eval_logs_dir)
    tokenizer_path = tokenizer_path or model_path

    from transformers import AutoTokenizer

    adapter_config = Path(model_path) / "adapter_config.json"
    llm_model_path = model_path
    lora_request = None
    lora_adapter_path = None
    lora_base_model_path = None
    if adapter_config.exists():
        from vllm.lora.request import LoRARequest
        adapter_cfg = json.loads(adapter_config.read_text(encoding="utf-8"))
        lora_base_model_path = adapter_cfg.get("base_model_name_or_path")
        if not lora_base_model_path:
            raise ValueError(f"LoRA adapter config missing base_model_name_or_path: {adapter_config}")
        llm_model_path = lora_base_model_path
        lora_adapter_path = model_path
        if tokenizer_path == model_path and not (Path(model_path) / "tokenizer_config.json").exists():
            tokenizer_path = lora_base_model_path

    tokenizer_start = time.time()
    tokenizer_kwargs = {"trust_remote_code": True, "use_fast": True}
    if Path(tokenizer_path).exists():
        tokenizer_kwargs["local_files_only"] = True
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
    thinking_tags = validate_thinking_protocol(tokenizer, model_protocol)
    thinking = model_protocol.get("thinking") if isinstance(model_protocol, dict) else None
    embedding_tags = (
        thinking_tags
        if isinstance(thinking, dict)
        and thinking.get("tag_encoding") == "special_tokens"
        else None
    )
    validate_embedding_range(tokenizer, llm_model_path, embedding_tags)
    protocol_reasoning_parser = (
        str(model_protocol["thinking"]["reasoning_parser"])
        if model_protocol is not None
        else "none"
    )
    expected_reasoning_parser = (
        "" if protocol_reasoning_parser == "none" else protocol_reasoning_parser
    )
    configured_reasoning_parser = str(reasoning_parser or "")
    if configured_reasoning_parser != expected_reasoning_parser:
        raise ValueError(
            "evaluation reasoning_parser does not match model_protocol: "
            f"configured={configured_reasoning_parser!r}; "
            f"expected={expected_reasoning_parser!r}"
        )
    active_reasoning_parser = expected_reasoning_parser
    if (
        thinking_budget is not None
        and int(thinking_budget) >= 0
        and not active_reasoning_parser
    ):
        raise ValueError(
            "thinking_budget requires a model_protocol reasoning_parser"
        )
    if active_reasoning_parser:
        tokenizer, tokenizer_path = _ensure_reasoning_parser_tokenizer(
            tokenizer,
            tokenizer_path,
            active_reasoning_parser,
        )
    tokenizer_load_seconds = time.time() - tokenizer_start
    if stop_token_ids is None:
        stop_token_ids = []
    eos_ids = tokenizer.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    for eos_id in eos_ids or []:
        if eos_id not in stop_token_ids:
            stop_token_ids.append(eos_id)

    effective_max_model_len = max_model_len or max_tokens
    effective_max_new_tokens = max_new_tokens or max_tokens
    effective_data_parallel_size = max(1, int(data_parallel_size or 1))
    vllm_log_path = eval_logs_dir / f"vllm_{log_name}.log"
    vllm_log_fh = open(vllm_log_path, "w", buffering=1) if capture_vllm_logs else open(os.devnull, "w")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = vllm_log_fh, vllm_log_fh
    llm = None
    try:
        init_start = time.time()
        llm_kwargs = {
            "model": llm_model_path,
            "tokenizer": tokenizer_path,
            "dtype": dtype,
            "trust_remote_code": True,
            "tensor_parallel_size": 1,
            "data_parallel_size": effective_data_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": effective_max_model_len,
        }
        if lora_adapter_path:
            _add_optional(llm_kwargs, "enable_lora", True)
            lora_request = LoRARequest(Path(lora_adapter_path).name, 1, str(lora_adapter_path), base_model_name=str(lora_base_model_path))
        for name, value in {
            "max_num_seqs": max_num_seqs,
            "swap_space": swap_space,
            "cpu_offload_gb": cpu_offload_gb,
            "enforce_eager": enforce_eager,
            "max_seq_len_to_capture": max_seq_len_to_capture,
            "kv_cache_dtype": kv_cache_dtype,
            "calculate_kv_scales": calculate_kv_scales,
            "enable_prefix_caching": enable_prefix_caching,
        }.items():
            _add_optional(llm_kwargs, name, value)
        if active_reasoning_parser:
            from vllm.config import ReasoningConfig

            llm_kwargs["structured_outputs_config"] = {"reasoning_parser": active_reasoning_parser}
            llm_kwargs["reasoning_config"] = ReasoningConfig()
        llm = _init_llm_with_port_retries(llm_kwargs)
        llm_init_seconds = time.time() - init_start
        results = []
        for dataset in datasets:
            dataset_spec = dict(dataset)
            if task_type and "task_type" not in dataset_spec:
                dataset_spec["task_type"] = task_type
            if primary_metric and "primary_metric" not in dataset_spec:
                dataset_spec["primary_metric"] = primary_metric
            prompts, examples, row_indices, resolved_task_type, prompt_evidence = _prepare_prompts(
                model_path,
                dataset_spec,
                tokenizer,
                answer_format,
                row_shard_index=row_shard_index,
                row_shard_count=row_shard_count,
            )
            gen_start = time.time()
            outputs = []
            base_sampling_kwargs = {
                "temperature": temperature,
                "max_tokens": effective_max_new_tokens,
                "skip_special_tokens": True,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "repetition_penalty": repetition_penalty,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "min_tokens": min_tokens,
                "stop": stop,
                "stop_token_ids": stop_token_ids,
            }
            if thinking_budget is not None and int(thinking_budget) >= 0:
                base_sampling_kwargs["thinking_token_budget"] = int(thinking_budget)
            if rollout_layout == "question_n":
                sampling_list = []
                for row_index in row_indices:
                    sampling_kwargs = dict(base_sampling_kwargs)
                    sampling_kwargs["n"] = len(run_indices)
                    sampling_kwargs["seed"] = question_rollout_seed(seed, int(row_index), 0, avg_k)
                    sampling_list.append(_sampling_params(**sampling_kwargs))
                if prompts:
                    outputs.extend(llm.generate(prompts, sampling_list, lora_request=lora_request))
            else:
                for run_index in run_indices:
                    sampling_kwargs = dict(base_sampling_kwargs)
                    sampling_kwargs["seed"] = seed + run_index if seed is not None else None
                    sampling = _sampling_params(
                        **sampling_kwargs,
                    )
                    outputs.extend(llm.generate(prompts, sampling, lora_request=lora_request))
            generation_seconds = time.time() - gen_start
            selected_primary_metric = dataset_spec.get("primary_metric") or primary_metric
            if rollout_layout == "question_n":
                avg_accuracy, std_deviation, details, eval_score = _score_question_n_outputs(
                    outputs,
                    prompts,
                    examples,
                    row_indices,
                    run_indices,
                    avg_k,
                    seed,
                    answer_format,
                    resolved_task_type,
                    tokenizer,
                    thinking_tags,
                    selected_primary_metric,
                )
            else:
                avg_accuracy, std_deviation, details, eval_score = _score_outputs(
                    outputs,
                    prompts,
                    examples,
                    row_indices,
                    run_indices,
                    avg_k,
                    seed,
                    answer_format,
                    resolved_task_type,
                    tokenizer,
                    thinking_tags,
                    selected_primary_metric,
                )
            suffix = (
                f"{result_suffix}__result_dataset_{_safe_log_part(dataset['name'])}"
                if result_suffix
                else f"result_dataset_{_safe_log_part(dataset['name'])}"
            )
            result_file = eval_logs_dir / _build_result_filename(epoch, time.strftime("%Y%m%d_%H%M%S"), avg_k, suffix)
            base_payload = {
                "model_path": model_path,
                "dataset_name": dataset["name"],
                "avg_k_runs": avg_k,
                "evaluated_run_indices": run_indices,
                "evaluated_seeds": evaluated_seeds,
                "seed": seed,
                "score": eval_score.score,
                "avg_at_k_score": avg_accuracy,
                "average_accuracy": avg_accuracy,
                "score_standard_deviation": std_deviation,
                "standard_deviation": std_deviation,
                "metrics": eval_score.metrics,
                "task_type": eval_score.task_type,
                "primary_metric": eval_score.primary_metric,
                "answer_format": answer_format,
                "thinking_budget": thinking_budget,
                "reasoning_parser": active_reasoning_parser,
                "model_protocol": model_protocol,
                "dataset_binding": {
                    key: _json_sanitize(dataset_spec.get(key))
                    for key in (
                        "name",
                        "domain",
                        "task_type",
                        "protocol",
                        "artifact_digest",
                        "prompt_protocol",
                    )
                },
                "prompt_evidence": prompt_evidence,
                "rollout_layout": rollout_layout,
                "data_parallel_size": effective_data_parallel_size,
                "row_shard": {"index": row_shard_index, "count": row_shard_count, "row_count": len(row_indices)},
                "epoch": epoch,
                "timings": {
                    "tokenizer_load_seconds": tokenizer_load_seconds,
                    "llm_init_seconds": llm_init_seconds,
                    "generation_seconds": generation_seconds,
                },
            }
            if extra_result_metadata:
                base_payload.update(extra_result_metadata)
            result_payload = build_eval_log_payload(
                base_payload=_json_sanitize(base_payload),
                dataset_path=dataset["path"],
                details=_json_sanitize(details),
            )
            result_file.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info(json.dumps({
                "stage": "EVAL",
                "event": "results_saved",
                "result_path": str(result_file),
                "avg_at_k_score": avg_accuracy,
                "score_standard_deviation": std_deviation,
                "evaluated_run_indices": run_indices,
                "evaluated_seeds": evaluated_seeds,
            }))
            results.append({
                "dataset_name": dataset["name"],
                "score": eval_score.score,
                "avg_at_k_score": avg_accuracy,
                "metrics": eval_score.metrics,
                "task_type": eval_score.task_type,
                "primary_metric": eval_score.primary_metric,
                "score_standard_deviation": std_deviation,
                "result_path": str(result_file),
                "timings": result_payload["timings"],
                "result_details": _json_sanitize(details),
                "data_parallel_size": effective_data_parallel_size,
                "row_shard": {"index": row_shard_index, "count": row_shard_count, "row_count": len(row_indices)},
            })
        return results
    except BaseException:
        try:
            traceback.print_exc(file=vllm_log_fh)
            vllm_log_fh.flush()
        except Exception:
            pass
        raise
    finally:
        try:
            del llm
        except Exception:
            pass
        _cleanup_vllm_runtime()
        sys.stdout, sys.stderr = old_stdout, old_stderr
        vllm_log_fh.close()
