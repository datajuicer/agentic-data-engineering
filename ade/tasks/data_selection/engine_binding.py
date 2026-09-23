"""Bind an accepted Data Selection script to Engine-owned inputs."""

from __future__ import annotations

import copy
import json

from ade.tasks.data_selection.fixed_pool import build_fixed_pool_from_task
from ade.tasks.contracts import (
    EngineArtifactBinding,
    EngineArtifactBindingRequest,
    EngineObjectPayload,
)


def bind_engine_artifact(
    request: EngineArtifactBindingRequest,
) -> EngineArtifactBinding:
    config = copy.deepcopy(dict(request.engine_config))
    sft = config.get("sft")
    if not isinstance(sft, dict):
        raise ValueError("Data Selection Engine config requires sft")
    task_data = request.task_config.get("data")
    if not isinstance(task_data, dict):
        raise ValueError("Data Selection task config requires data")
    dataset_name = _required_text(task_data, "dataset_name", "task.data")
    pool_size = task_data.get("pool_size")
    if type(pool_size) is not int or pool_size <= 0:
        raise ValueError("Data Selection task.data.pool_size must be positive")
    select_size = request.task_config.get("select_size")
    if type(select_size) is not int or select_size <= 0:
        raise ValueError("Data Selection task config requires positive select_size")

    if request.final_realization:
        return _bind_final_realization(
            request,
            config=config,
            sft=sft,
            dataset_name=dataset_name,
            select_size=select_size,
        )

    inventory, training, pool_stats = build_fixed_pool_from_task(request.task_config)
    root = request.binding_uri.rstrip("/")
    script_ref = f"{root}/selection.py"
    inventory_ref = f"{root}/source/candidate_inventory.jsonl"
    training_ref = f"{root}/source/training_dataset.jsonl"
    sft["dataset"] = {
        "candidate_inventory_ref": inventory_ref,
        "training_data_ref": training_ref,
        "dataset_name": dataset_name,
        "pool_size": pool_size,
        "select_size": select_size,
        "pool_stats": pool_stats,
    }
    config["artifact_ref"] = script_ref
    return EngineArtifactBinding(
        input_payload=config,
        objects=(
            EngineObjectPayload(script_ref, request.compiled_content),
            EngineObjectPayload(inventory_ref, inventory),
            EngineObjectPayload(training_ref, training),
        ),
    )


def _bind_final_realization(
    request: EngineArtifactBindingRequest,
    *,
    config: dict[str, object],
    sft: dict[str, object],
    dataset_name: str,
    select_size: int,
) -> EngineArtifactBinding:
    realization = request.final_realization
    if realization.get("schema_version") != "ade.data_selection_realization.v1":
        raise ValueError("Data Selection final realization schema is invalid")
    selection = realization.get("selection")
    selected_rows = realization.get("selected_rows")
    candidate_pool = realization.get("candidate_pool")
    if (
        not isinstance(selection, dict)
        or selection.get("selection_size") != select_size
        or not isinstance(selected_rows, list)
        or len(selected_rows) != select_size
        or any(not isinstance(row, dict) for row in selected_rows)
        or not isinstance(candidate_pool, dict)
    ):
        raise ValueError("Data Selection final realization payload is incomplete")

    root = f"{request.binding_uri.rstrip('/')}/final-realization"
    script_ref = f"{root}/selection.py"
    selection_ref = f"{root}/selection-result.json"
    data_ref = f"{root}/selected-examples.json"
    info_ref = f"{root}/dataset_info.json"
    report_ref = f"{root}/realization-report.json"
    training_binding_ref = f"{root}/training-binding.json"
    dataset_info = {
        dataset_name: {
            "file_name": "selected-examples.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    }
    training_binding = {
        "schema_version": "ade.sft_final_training_binding.v1",
        "realization_status": "final",
        "data_ref": data_ref,
        "info_ref": info_ref,
        "selection_ref": selection_ref,
        "realization_report_ref": report_ref,
        "dataset_name": dataset_name,
        "candidate_pool_identity": copy.deepcopy(candidate_pool),
        "pool_stats": copy.deepcopy(candidate_pool.get("stats") or {}),
    }
    sft["dataset"] = {
        **training_binding,
        "training_binding_ref": training_binding_ref,
    }
    config["artifact_ref"] = script_ref
    return EngineArtifactBinding(
        input_payload=config,
        objects=(
            EngineObjectPayload(script_ref, request.compiled_content),
            EngineObjectPayload(selection_ref, _json_bytes(selection)),
            EngineObjectPayload(data_ref, _json_bytes(selected_rows)),
            EngineObjectPayload(info_ref, _json_bytes(dataset_info)),
            EngineObjectPayload(report_ref, _json_bytes(realization)),
            EngineObjectPayload(
                training_binding_ref, _json_bytes(training_binding)
            ),
        ),
    )


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _required_text(value: dict[str, object], key: str, owner: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{owner} requires {key}")
    return item.strip()
