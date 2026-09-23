"""Canonical answer extraction shared by RFT rewards and evaluation."""

from __future__ import annotations

from typing import Any

from ade.engine.eval.extraction import split_thinking_answer
from ade.engine.eval.utils.math_exact import extract_math_answer
from ade.engine.eval.utils.python_code import extract_python_code
from ade.engine.eval.utils.science_answer import (
    extract_boxed_science_answer,
    extract_gpqa_choice_answer,
    extract_supergpqa_choice_answer,
)


def extract_reward_answer(
    data_source: str,
    response_content: str,
    extra_info: Any = None,
) -> tuple[str, str]:
    source = str(data_source or "").lower()
    _, text = split_thinking_answer(str(response_content or ""), enabled=True)
    if _is_code_source(source):
        code, method = extract_python_code(
            text, expected_entry_points=_expected_entry_points(extra_info)
        )
        return code, method
    if _is_math_source(source):
        answer = extract_math_answer(text)
        return answer, "evalchemy_last_boxed" if answer else "none"
    if _is_supergpqa_source(source):
        return extract_supergpqa_choice_answer(text)
    if _is_gpqa_source(source):
        return extract_gpqa_choice_answer(text)
    if _is_science_source(source):
        return extract_boxed_science_answer(text)
    return text, "raw"


def _is_math_source(source: str) -> bool:
    return "math" in source or source in {"jee", "jee_bench"}


def _is_code_source(source: str) -> bool:
    return any(token in source for token in ("code", "mbpp", "humaneval", "livecode", "competitive"))


def _is_science_source(source: str) -> bool:
    return any(token in source for token in ("science", "stem"))


def _is_supergpqa_source(source: str) -> bool:
    return "supergpqa" in source or "mmlu" in source


def _is_gpqa_source(source: str) -> bool:
    return "gpqa" in source


def _expected_entry_points(extra_info: Any) -> list[str]:
    if not isinstance(extra_info, dict):
        return []
    values = extra_info.get("entry_points") or extra_info.get("expected_entry_points")
    if isinstance(values, str):
        return [values]
    if isinstance(values, (list, tuple)):
        return [str(value) for value in values if value]
    for key in ("entry_point", "function_name", "function"):
        if extra_info.get(key):
            return [str(extra_info[key])]
    return []
