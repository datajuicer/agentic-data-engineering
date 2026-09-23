"""Resolved prompt and label contract for the supported SFT training path."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import yaml


_SUPPORTED_TEMPLATE = "qwen"
_EXPECTED_SLOTS = {
    "system": "<|im_start|>system\n{{content}}<|im_end|>\n",
    "user": (
        "<|im_start|>user\n{{content}}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "assistant": "{{content}}<|im_end|>\n",
}


def resolve_sft_prompt_contract(
    *,
    project_root: str | Path,
    recipe_path: str | Path,
    model_path: str | Path,
    prompt_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve and verify the one prompt shape supported by ADE SFT."""

    root = Path(project_root).resolve()
    recipe = Path(recipe_path).resolve()
    model = Path(model_path).resolve()
    recipe_config = _yaml_mapping(recipe, "SFT recipe")
    if recipe_config.get("stage") != "sft":
        raise ValueError("SFT recipe stage must be sft")
    template_name = _required_text(recipe_config, "template", "SFT recipe")
    if template_name != _SUPPORTED_TEMPLATE:
        raise ValueError(
            f"ADE SFT supports LlamaFactory template {_SUPPORTED_TEMPLATE!r}; "
            f"got {template_name!r}"
        )
    for field in ("train_on_prompt", "mask_history"):
        if recipe_config.get(field) is not False:
            raise ValueError(f"SFT recipe {field} must be explicitly false")

    protocol = dict(prompt_protocol)
    if protocol.get("mode") != "chat_template":
        raise ValueError("SFT prompt protocol must use chat_template")
    system_prompt = protocol.get("system_prompt")
    native_binding = protocol.get("chat_template")
    if not isinstance(system_prompt, Mapping) or not isinstance(native_binding, Mapping):
        raise ValueError(
            "SFT prompt protocol requires resolved system_prompt and chat_template"
        )
    system_content = _required_text(system_prompt, "content", "SFT system prompt")
    native_digest = _required_text(
        native_binding, "digest", "SFT native chat template"
    )

    template_source = root / "third_party/llamafactory/src/llamafactory/data/template.py"
    version_source = root / "third_party/llamafactory/src/llamafactory/extras/env.py"
    slots, stop_words, replace_eos = _template_definition(
        template_source, template_name
    )
    if slots != _EXPECTED_SLOTS or stop_words != ["<|im_end|>"] or not replace_eos:
        raise ValueError(
            "LlamaFactory qwen template no longer matches the accepted ADE SFT ChatML contract"
        )

    tokenizer_metadata = native_tokenizer_prompt_metadata(model)
    native_template = tokenizer_metadata["chat_template"]
    actual_native_digest = hashlib.sha256(native_template.encode("utf-8")).hexdigest()
    if actual_native_digest != native_digest:
        raise ValueError("SFT native chat template changed after config compilation")

    probe_user = "ADE_SFT_USER_CONTENT"
    probe_assistant = "<think>ADE_SFT_REASONING</think>ADE_SFT_FINAL"
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    native_rendered = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
    ).from_string(native_template).render(
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": probe_user},
            {"role": "assistant", "content": probe_assistant},
        ],
        tools=None,
        add_generation_prompt=False,
    )
    training_rendered = (
        slots["system"].replace("{{content}}", system_content)
        + slots["user"].replace("{{content}}", probe_user)
        + slots["assistant"].replace("{{content}}", probe_assistant)
    )
    if native_rendered != training_rendered:
        raise ValueError(
            "native tokenizer and LlamaFactory templates diverge for the accepted "
            "system/user/assistant SFT protocol"
        )

    source_eos = tokenizer_metadata["eos_token"]
    resolved_eos = stop_words[0] if replace_eos else source_eos
    if source_eos != resolved_eos:
        raise ValueError(
            "ADE SFT requires the native and LlamaFactory resolved EOS token to agree"
        )

    return {
        "schema_version": "ade.sft_prompt_contract.v1",
        "system_prompt": copy.deepcopy(dict(system_prompt)),
        "data_protocol": {
            "format": "sharegpt",
            "message_roles": ["user", "assistant"],
            "source_system_message": False,
            "tools": False,
            "assistant_targets_per_example": 1,
        },
        "serialized_prompt_format": (
            "<|im_start|>system\n{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n{assistant}<|im_end|>\n"
        ),
        "native_template": copy.deepcopy(dict(native_binding)),
        "training_template": {
            "backend": "llamafactory",
            "name": template_name,
            "implementation_version": _llamafactory_version(version_source),
            "implementation_sha256": _sha256(template_source),
            "recipe_sha256": _sha256(recipe),
            "replace_eos": replace_eos,
            "source_eos_token": source_eos,
            "resolved_eos_token": resolved_eos,
            "assistant_terminator": "<|im_end|>\n",
        },
        "label_policy": {
            "train_on_prompt": False,
            "mask_history": False,
            "prompt_labels": "ignore",
            "assistant_target_labels": "train",
            "assistant_terminator_label": "train",
        },
    }


def native_tokenizer_prompt_metadata(model_path: str | Path) -> dict[str, str]:
    """Read the native prompt fields without importing the training stack."""

    path = Path(model_path) / "tokenizer_config.json"
    if not path.is_file():
        raise ValueError(f"SFT base tokenizer config does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"SFT base tokenizer config must be a mapping: {path}")
    chat_template = value.get("chat_template")
    eos_token = value.get("eos_token")
    if not isinstance(chat_template, str) or not chat_template:
        template_path = Path(model_path) / "chat_template.jinja"
        if template_path.is_file():
            chat_template = template_path.read_text(encoding="utf-8")
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("SFT base tokenizer has no native chat template")
    if not isinstance(eos_token, str) or not eos_token:
        raise ValueError("SFT base tokenizer has no native EOS token")
    return {
        "chat_template": chat_template,
        "eos_token": eos_token,
    }


def validate_sft_prompt_contract(request: Mapping[str, Any]) -> None:
    """Fail before launch if recipe, model, or implementation drifted."""

    expected = request.get("training_prompt_contract")
    if not isinstance(expected, Mapping):
        raise ValueError("SFT request requires training_prompt_contract")
    actual = resolve_sft_prompt_contract(
        project_root=_required_text(request, "project_root", "SFT request"),
        recipe_path=_required_text(request, "recipe_path", "SFT request"),
        model_path=_required_text(request, "model", "SFT request"),
        prompt_protocol=_required_mapping(
            request, "prompt_protocol", "SFT request"
        ),
    )
    if actual != dict(expected):
        raise ValueError(
            "SFT training prompt contract changed after experiment compilation"
        )
    expected_system = actual["system_prompt"]["content"]
    if request.get("default_system") != expected_system:
        raise ValueError(
            "SFT request default_system must equal the config-resolved system prompt"
        )
    for field in ("train_on_prompt", "mask_history"):
        if request.get(field) is not False:
            raise ValueError(f"SFT request {field} must be explicitly false")


def _template_definition(
    source_path: Path,
    template_name: str,
) -> tuple[dict[str, str], list[str], bool]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    matches: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "register_template":
            continue
        keywords = {item.arg: item.value for item in node.keywords if item.arg}
        if _literal(keywords.get("name")) == template_name:
            matches.append(node)
    if len(matches) != 1:
        raise ValueError(
            f"LlamaFactory template {template_name!r} must have one definition"
        )
    keywords = {item.arg: item.value for item in matches[0].keywords if item.arg}
    slots = {
        name: _formatter_slot(keywords.get(f"format_{name}"), f"qwen {name}")
        for name in ("system", "user", "assistant")
    }
    stop_words = _literal(keywords.get("stop_words"))
    replace_eos = _literal(keywords.get("replace_eos"))
    if not isinstance(stop_words, list) or not all(
        isinstance(item, str) for item in stop_words
    ):
        raise ValueError("LlamaFactory qwen stop_words are not static strings")
    if type(replace_eos) is not bool:
        raise ValueError("LlamaFactory qwen replace_eos is not a static boolean")
    return slots, stop_words, replace_eos


def _formatter_slot(value: ast.AST | None, owner: str) -> str:
    if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
        raise ValueError(f"LlamaFactory {owner} formatter is not statically bound")
    if value.func.id != "StringFormatter":
        raise ValueError(f"LlamaFactory {owner} must use StringFormatter")
    keywords = {item.arg: item.value for item in value.keywords if item.arg}
    slots = _literal(keywords.get("slots"))
    if not isinstance(slots, list) or len(slots) != 1 or not isinstance(slots[0], str):
        raise ValueError(f"LlamaFactory {owner} must contain one static string slot")
    return slots[0]


def _llamafactory_version(path: Path) -> str:
    match = re.search(
        r'^VERSION\s*=\s*["\']([^"\']+)["\']\s*$',
        path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError("LlamaFactory version identity is unavailable")
    return match.group(1)


def _yaml_mapping(path: Path, owner: str) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{owner} must be a YAML mapping")
    return value


def _required_mapping(
    value: Mapping[str, Any], field: str, owner: str
) -> Mapping[str, Any]:
    item = value.get(field)
    if not isinstance(item, Mapping):
        raise ValueError(f"{owner}.{field} must be a mapping")
    return item


def _required_text(value: Mapping[str, Any], field: str, owner: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{owner}.{field} must be non-empty text")
    return item


def _literal(value: ast.AST | None) -> Any:
    if value is None:
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, TypeError):
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
