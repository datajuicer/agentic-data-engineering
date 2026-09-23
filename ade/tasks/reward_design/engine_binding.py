"""Bind an accepted Reward Design artifact to an Engine-owned reward module."""

from __future__ import annotations

import ast
import copy

from ade.tasks.contracts import (
    EngineArtifactBinding,
    EngineArtifactBindingRequest,
    EngineObjectPayload,
)


def bind_run_local_judge(
    config: dict[str, object], run_resources: dict[str, object]
) -> dict[str, object]:
    """Project the Run-owned Judge binding into the RFT runtime config."""
    bound = copy.deepcopy(config)
    local = run_resources["local_judge"]
    resolved_judge = {
        "gateway_url": local["gateway_url"],
        "authorization_env": local["authorization_env"],
        "request_timeout_seconds": local["timeout_policy"]["request_timeout_seconds"],
        "max_failed_row_ratio": local["circuit_policy"]["max_failed_row_ratio"],
        "max_consecutive_unhealthy_jobs": local["circuit_policy"][
            "max_consecutive_unhealthy_jobs"
        ],
    }
    rft = bound.get("rft")
    if not isinstance(rft, dict) or not isinstance(rft.get("verl_config"), dict):
        raise ValueError("Reward Design Engine config requires rft.verl_config")
    verl_config = rft["verl_config"]
    runtime_rft = dict(verl_config.get("rft") or {})
    runtime_rft["local_judge"] = resolved_judge
    train = dict(verl_config.get("train") or {})
    train_rft = dict(train.get("rft") or {})
    train_rft["local_judge"] = copy.deepcopy(resolved_judge)
    train["rft"] = train_rft
    verl_config["rft"] = runtime_rft
    verl_config["train"] = train
    return bound


def bind_engine_artifact(
    request: EngineArtifactBindingRequest,
) -> EngineArtifactBinding:
    config = copy.deepcopy(dict(request.engine_config))
    rft = config.get("rft")
    if not isinstance(rft, dict):
        raise ValueError("Reward Design Engine config requires rft")
    if "reward" in rft or _contains_reward_path(rft):
        raise ValueError("Reward Design Engine config cannot override accepted reward")
    rft.setdefault("verl_config", {"rft": {}, "train": {"rft": {}}})
    if request.is_baseline:
        verl_config = rft["verl_config"]
        for section in (
            verl_config.get("rft"),
            (verl_config.get("train") or {}).get("rft"),
        ):
            if isinstance(section, dict) and isinstance(
                section.get("judge_enrichment"), dict
            ):
                section["judge_enrichment"] = {
                    **section["judge_enrichment"],
                    "enabled": False,
                }
    reward_ref = f"{request.binding_uri.rstrip('/')}/reward.py"
    reward_binding = {
        "function_ref": reward_ref,
        "entrypoint": "compute_score",
    }
    group_credit = rft.get("group_credit")
    if isinstance(group_credit, dict) and group_credit.get("enabled") is True:
        reward_range = rft.get("verl_config", {}).get("rft", {}).get(
            "reward_range", [0.0, 1.0]
        )
        if (
            not isinstance(reward_range, list)
            or len(reward_range) != 2
            or any(type(value) not in {int, float} for value in reward_range)
        ):
            raise ValueError("enabled group credit requires a fixed reward range")
        _require_group_credit_artifact(
            request.compiled_content,
            reward_range=(float(reward_range[0]), float(reward_range[1])),
            group_size=int(
                rft.get("verl_config", {}).get("rft", {}).get("rollout_n", 8)
            ),
            observe_process_bank=(
                not request.is_baseline
                and isinstance(request.task_config.get("judge_enrichment"), dict)
                and request.task_config["judge_enrichment"].get("enabled") is True
            ),
        )
        reward_binding["score_entrypoint"] = reward_binding.pop("entrypoint")
        reward_binding.update(
            {
                "fallback_entrypoint": "compute_fallback_score",
                "group_credit_entrypoint": "assign_group_credit",
            }
        )
    rft["reward"] = reward_binding
    config["artifact_ref"] = reward_ref
    return EngineArtifactBinding(
        input_payload=config,
        objects=(EngineObjectPayload(reward_ref, request.compiled_content),),
    )


def _require_group_credit_artifact(
    content: bytes,
    *,
    reward_range: tuple[float, float],
    group_size: int,
    observe_process_bank: bool,
) -> None:
    try:
        tree = ast.parse(content.decode("utf-8"), filename="reward.py")
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ValueError("enabled group credit requires valid reward.py") from error
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    group = functions.get("assign_group_credit")
    if (
        not isinstance(
            functions.get("compute_score"), (ast.FunctionDef, ast.AsyncFunctionDef)
        )
        or not isinstance(functions.get("compute_fallback_score"), ast.FunctionDef)
        or not isinstance(group, ast.FunctionDef)
        or [argument.arg for argument in group.args.args] != ["group_input"]
        or group.args.vararg is not None
        or group.args.kwarg is not None
        or group.args.defaults
        or group.args.kw_defaults
    ):
        raise ValueError(
            "enabled group credit requires compute_score, compute_fallback_score, "
            "and synchronous assign_group_credit(group_input)"
        )
    namespace: dict[str, object] = {"__file__": "reward.py"}
    exec(compile(tree, "reward.py", "exec"), namespace, namespace)
    from ade.tasks.reward_design.group_credit import admit_group_credit_function

    admit_group_credit_function(
        namespace["assign_group_credit"],
        group_size=group_size,
        reward_range=reward_range,
        observe_process_bank=observe_process_bank,
    )


def _contains_reward_path(value: object) -> bool:
    if isinstance(value, dict):
        return "reward_function_path" in value or any(
            _contains_reward_path(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_reward_path(item) for item in value)
    return False
