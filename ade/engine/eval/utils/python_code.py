from __future__ import annotations

import contextlib
import ast
import multiprocessing as mp
import re
import traceback
from typing import Any

from ..contracts import EvalExample, EvalScore, RolloutRecord, make_eval_case
from ..metrics import standardize_eval_metrics
from .base import EvalTask, first_present, require_first_present
from .execution import CappedStringIO


PYTHON_IMPORTS = "\n".join(
    [
        "import math",
        "import re",
        "import sys",
        "import copy",
        "import datetime",
        "import itertools",
        "import collections",
        "import heapq",
        "import functools",
        "import hashlib",
        "import string",
        "from typing import *",
        "from collections import *",
    ]
)


def python_fenced_code_blocks(text: str) -> list[str]:
    return re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", flags=re.DOTALL | re.IGNORECASE)


def python_code_tag_blocks(text: str) -> list[str]:
    return re.findall(r"<code(?:\s+[^>]*)?>\s*(.*?)(?:</code>|$)", text or "", flags=re.DOTALL | re.IGNORECASE)


def strip_chat_special_tokens(text: str) -> str:
    cleaned = str(text or "")
    for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
        cleaned = cleaned.replace(token, "")
    return cleaned.strip()


def _top_level_function_names(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set(re.findall(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", code or "", flags=re.MULTILINE))
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


def _select_python_block(blocks: list[str], expected_entry_points: list[str]) -> tuple[str, str]:
    expected = {name for name in expected_entry_points if name}
    if expected:
        matches = [block for block in blocks if _top_level_function_names(block) & expected]
        if matches:
            return matches[-1], "entry_point_fenced_code_block"
    matches = [block for block in blocks if _top_level_function_names(block)]
    if matches:
        return matches[-1], "function_fenced_code_block"
    return blocks[-1], "last_fenced_code_block"


def extract_python_code(text: str, *, expected_entry_points: list[str] | None = None) -> tuple[str, str]:
    cleaned = strip_chat_special_tokens(text)
    blocks = python_fenced_code_blocks(cleaned)
    if blocks:
        block, method = _select_python_block(blocks, expected_entry_points or [])
        return block.strip("\r\n"), method
    blocks = python_code_tag_blocks(cleaned)
    if blocks:
        block, _ = _select_python_block(blocks, expected_entry_points or [])
        return block.strip("\r\n"), "code_tag"
    return cleaned, "full_output"


def _run_python_test_code(test_code: str, queue: mp.Queue) -> None:
    try:
        globals_dict: dict[str, Any] = {}
        with contextlib.redirect_stdout(CappedStringIO()), contextlib.redirect_stderr(CappedStringIO()):
            exec(test_code, globals_dict)
    except AssertionError:
        queue.put({"passed": False, "reason": "AssertionError"})
    except BaseException as exc:
        queue.put({"passed": False, "reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    else:
        queue.put({"passed": True, "reason": ""})


def run_python_test_code(test_code: str, *, timeout: float = 6.0) -> dict[str, Any]:
    queue: mp.Queue = mp.Queue()
    process = mp.Process(target=_run_python_test_code, args=(test_code, queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(1)
        return {"passed": False, "reason": f"timed out after {timeout:g}s"}
    if not queue.empty():
        return dict(queue.get())
    if process.exitcode == 0:
        return {"passed": True, "reason": ""}
    return {"passed": False, "reason": f"process exited with code {process.exitcode}"}


def _detail_from_records(
    *,
    task_type: str,
    repeat_idx: int,
    rollouts: list[RolloutRecord],
    scored_cases: list[tuple[RolloutRecord, bool, str, str, str, dict[str, Any]]],
) -> dict[str, Any]:
    successes = []
    errors = []
    correct = 0
    for rollout, passed, reason, code, extraction_method, metadata in scored_cases:
        bucket = "successes" if passed else "errors"
        sample = make_eval_case(
            task_type=task_type,
            dataset_name=rollout.example.meta.get("dataset_name"),
            repeat_index=repeat_idx,
            bucket=bucket,
            index=rollout.example.row_index,
            example_id=rollout.example.id,
            input={"prompt": rollout.example.prompt},
            gold={
                "test_code": metadata.get("test_code"),
                "raw_example": rollout.example.meta.get("raw_example"),
            },
            prediction={
                "thinking_content": rollout.thinking_content,
                "answer_content": rollout.answer_content,
                "generation": code,
                "extracted_code": code,
            },
            diagnostics={
                "passed": passed,
                "correctness": passed,
                "reason": reason,
                "num_output_tokens": rollout.num_output_tokens,
                "finish_reason": rollout.finish_reason,
                "stop_reason": rollout.stop_reason,
                "code_extraction_method": extraction_method,
                "extraction_method": extraction_method,
            },
            metadata={key: value for key, value in metadata.items() if key != "test_code"},
        )
        if passed:
            correct += 1
            successes.append(sample)
        else:
            errors.append(sample)
    total = len(rollouts)
    detail = {
        "index": repeat_idx,
        "accuracy": correct / total if total else 0.0,
        "correct_count": correct,
        "total_count": total,
        "errors": errors,
        "successes": successes,
        "row_indices": [rollout.example.row_index for rollout in rollouts],
    }
    if rollouts and rollouts[0].seed is not None:
        detail["seed"] = rollouts[0].seed
    return detail


class PythonFunctionalTask(EvalTask):
    primary_metric = "accuracy_avg"
    execution_timeout = 6.0

    def build_test_code(self, example: EvalExample, code: str) -> str:
        raise NotImplementedError

    def output_metadata(self, example: EvalExample) -> dict[str, Any]:
        return {}

    def expected_entry_points(self, example: EvalExample) -> list[str]:
        return []

    def extract_code(self, example: EvalExample, text: str) -> tuple[str, str]:
        return extract_python_code(text, expected_entry_points=self.expected_entry_points(example))

    def score_rollouts(
        self,
        examples: list[EvalExample],
        rollouts_by_repeat: list[list[RolloutRecord]],
        *,
        primary_metric: str | None = None,
    ) -> EvalScore:
        details: list[dict[str, Any]] = []
        run_stats: list[dict[str, Any]] = []
        for repeat_idx, rollouts in enumerate(rollouts_by_repeat):
            scored_cases = []
            for rollout in rollouts:
                code, extraction_method = self.extract_code(rollout.example, rollout.output)
                if not code.strip():
                    result = {"passed": False, "reason": "empty code generation"}
                    test_code = ""
                else:
                    test_code = self.build_test_code(rollout.example, code)
                    result = run_python_test_code(test_code, timeout=self.execution_timeout)
                metadata = {**self.output_metadata(rollout.example), "test_code": test_code}
                scored_cases.append(
                    (
                        rollout,
                        bool(result.get("passed")),
                        str(result.get("reason") or ""),
                        code,
                        extraction_method,
                        metadata,
                    )
                )
            detail = _detail_from_records(
                task_type=self.task_type,
                repeat_idx=repeat_idx,
                rollouts=rollouts,
                scored_cases=scored_cases,
            )
            details.append(detail)
            run_stats.append(
                {
                    "repetition": repeat_idx + 1,
                    "num_total": detail["total_count"],
                    "num_solved": detail["correct_count"],
                    "accuracy": detail["accuracy"],
                }
            )

        accuracy_avg, accuracy_std, accuracy_std_err = self.average_score(run_stats)
        metrics: dict[str, Any] = {
            "num_total": len(examples),
            "solved_avg": sum(float(item["num_solved"]) for item in run_stats) / len(run_stats) if run_stats else 0.0,
            "run_stats": run_stats,
            "accuracy_avg": accuracy_avg,
            "accuracy_std": accuracy_std,
            "accuracy_std_err": accuracy_std_err,
            "num_repeat": len(run_stats),
        }
        selected_metric = primary_metric or self.primary_metric
        metrics = standardize_eval_metrics(metrics, details, primary_metric=selected_metric)
        return EvalScore(self.task_type, selected_metric, float(metrics["score"]), metrics, details)


class MBPPTask(PythonFunctionalTask):
    task_type = "mbpp"

    def row_to_example(self, row: dict[str, Any], idx: int, *, dataset_name: str | None = None) -> EvalExample:
        prompt = require_first_present(row, ("prompt", "text", "question", "problem"), task_type=self.task_type, row_index=idx, field_name="prompt")
        tests = first_present(row, ("test_list", "tests", "test"))
        test_list = tests if isinstance(tests, list) else [str(tests)] if tests is not None else []
        test_imports = row.get("test_imports") if isinstance(row.get("test_imports"), list) else []
        example_id = str(first_present(row, ("task_id", "id", "index")) or f"{dataset_name or 'dataset'}/{idx}")
        prompt_text = (
            "Please write a Python function that solves the problem. "
            "Return the completed code in a Python code block.\n"
            f">>> Problem:\n{str(prompt).strip()}\n"
            f">>> Test Cases:\n" + "\n".join(str(item) for item in test_list)
        )
        return EvalExample(
            id=example_id,
            prompt=prompt_text,
            reference="",
            row_index=idx,
            meta={
                "dataset_name": dataset_name,
                "raw_example": dict(row),
                "test_list": test_list,
                "test_imports": test_imports,
            },
        )

    def build_test_code(self, example: EvalExample, code: str) -> str:
        imports = "\n".join(str(item) for item in example.meta.get("test_imports") or [])
        tests = "\n".join(str(item) for item in example.meta.get("test_list") or [])
        return "\n".join(part for part in (PYTHON_IMPORTS, imports, code, tests) if part.strip())

    def expected_entry_points(self, example: EvalExample) -> list[str]:
        names: list[str] = []
        for test in example.meta.get("test_list") or []:
            names.extend(re.findall(r"\bassert\s+([A-Za-z_]\w*)\s*\(", str(test)))
        return list(dict.fromkeys(names))

    def output_metadata(self, example: EvalExample) -> dict[str, Any]:
        return {"benchmark": "mbpp"}


class HumanEvalTask(PythonFunctionalTask):
    task_type = "human_eval"

    def row_to_example(self, row: dict[str, Any], idx: int, *, dataset_name: str | None = None) -> EvalExample:
        prompt = require_first_present(row, ("prompt",), task_type=self.task_type, row_index=idx, field_name="prompt")
        test = require_first_present(row, ("test",), task_type=self.task_type, row_index=idx, field_name="test")
        entry_point = require_first_present(row, ("entry_point",), task_type=self.task_type, row_index=idx, field_name="entry point")
        example_id = str(first_present(row, ("task_id", "id", "index")) or f"{dataset_name or 'dataset'}/{idx}")
        prompt_text = (
            "Please continue to complete the Python function. Do not modify the given signature. "
            "Return the completed function in a Python code block.\n"
            "```python\n"
            f"{str(prompt).rstrip()}\n"
            "```"
        )
        return EvalExample(
            id=example_id,
            prompt=prompt_text,
            reference="",
            row_index=idx,
            meta={
                "dataset_name": dataset_name,
                "raw_example": dict(row),
                "canonical_prompt": str(prompt),
                "test": str(test),
                "entry_point": str(entry_point),
            },
        )

    def build_test_code(self, example: EvalExample, code: str) -> str:
        canonical_prompt = str(example.meta.get("canonical_prompt") or "")
        entry_point = str(example.meta.get("entry_point") or "")
        if f"def {entry_point}" not in code:
            code = canonical_prompt.rstrip() + "\n" + code
        return "\n".join(
            [
                PYTHON_IMPORTS,
                code,
                str(example.meta.get("test") or ""),
                f"check({entry_point})",
            ]
        )

    def output_metadata(self, example: EvalExample) -> dict[str, Any]:
        return {"benchmark": "human_eval", "entry_point": example.meta.get("entry_point")}

    def expected_entry_points(self, example: EvalExample) -> list[str]:
        entry_point = str(example.meta.get("entry_point") or "")
        return [entry_point] if entry_point else []
