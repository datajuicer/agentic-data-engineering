from __future__ import annotations

import re
import math
from pathlib import Path
from fractions import Fraction
from typing import Any

from ..contracts import EvalExample, EvalScore, RolloutRecord, make_eval_case
from ..metrics import standardize_eval_metrics
from .base import EvalTask, first_present, require_first_present
from .hmmt_matharena.parser import check_answers as _matharena_check_answers
from .hmmt_matharena.parser import extract_answer as _matharena_extract_answer
from .hmmt_matharena.parser import parse_answer as _matharena_parse_answer


BOXED_REASONING_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."


def evalchemy_last_boxed_only_string(text: str) -> str | None:
    text = text or ""
    idx = text.rfind("\\boxed")
    if "\\boxed " in text:
        return "\\boxed " + text.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    pos = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while pos < len(text):
        if text[pos] == "{":
            num_left_braces_open += 1
        if text[pos] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = pos
                break
        pos += 1
    if right_brace_idx is None:
        return None
    return text[idx : right_brace_idx + 1]


def evalchemy_remove_boxed(text: str | None) -> str:
    if text is None:
        raise ValueError("missing boxed answer")
    if "\\boxed " in text:
        left = "\\boxed "
        if not text.startswith(left):
            raise ValueError(f"invalid boxed answer: {text}")
        return text[len(left) :]
    left = "\\boxed{"
    if not text.startswith(left) or not text.endswith("}"):
        raise ValueError(f"invalid boxed answer: {text}")
    return text[len(left) : -1]


def evalchemy_boxed_answer(text: str) -> tuple[str, str]:
    try:
        return evalchemy_remove_boxed(evalchemy_last_boxed_only_string(text)), "evalchemy_last_boxed"
    except Exception:
        return "", "none"


def replace_latex_frac(text: str) -> str:
    pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    while True:
        new = pattern.sub(r"(\1)/(\2)", text)
        if new == text:
            return text
        text = new


def normalize_math_answer(value: Any) -> str:
    text = str(value or "").strip()
    boxed, method = evalchemy_boxed_answer(text)
    if method != "none":
        text = boxed
    text = text.strip().strip("$")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.replace("\\%", "%")
    text = re.sub(r"(?<!\\),(\d{3})(?!\d)", r"\1", text)
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = replace_latex_frac(text)
    text = text.replace("{", "(").replace("}", ")")
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = text.replace("−", "-")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"\.$", "", text)
    if text.endswith(".0"):
        text = text[:-2]
    return text


def numeric_value(text: str) -> float | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    divisor = 100.0 if cleaned.endswith("%") else 1.0
    cleaned = cleaned[:-1] if cleaned.endswith("%") else cleaned
    try:
        return float(Fraction(cleaned)) / divisor
    except Exception:
        pass
    try:
        return float(cleaned) / divisor
    except Exception:
        return None


def math_equivalent(expected: str, predicted: str) -> bool:
    lhs = normalize_math_answer(expected)
    rhs = normalize_math_answer(predicted)
    if lhs == rhs:
        return True
    lhs_num = numeric_value(lhs)
    rhs_num = numeric_value(rhs)
    return lhs_num is not None and rhs_num is not None and math.isclose(lhs_num, rhs_num, rel_tol=0.0, abs_tol=1e-9)


def find_boxed_content(text: str) -> str | None:
    last = None
    for match in re.finditer(r"\\+(?:boxed|fbox)(?:\s*\{|\s+)", text or ""):
        token = match.group(0)
        if token.rstrip().endswith("{"):
            pos = match.end()
            level = 1
            while pos < len(text) and level > 0:
                if text[pos] == "{":
                    level += 1
                elif text[pos] == "}":
                    level -= 1
                pos += 1
            if level == 0:
                last = text[match.end():pos - 1].strip()
        else:
            value = text[match.end():].split("$")[0].splitlines()[0].strip()
            if value:
                last = value
    return last


def normalize_answer(answer: Any) -> str:
    if answer is None:
        return ""
    text = str(answer).strip()
    if not text:
        return ""
    text = re.sub(r"\.$", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\\\s+(\w+)", r"\\\1", text)
    return re.sub(r"\s*([+\-*/=<>()[\]{},.^])\s*", r"\1", text)


def extract_math_answer(text: str, *, is_reference: bool = False) -> str:
    boxed_answer, method = evalchemy_boxed_answer(text)
    if method != "none":
        return normalize_math_answer(boxed_answer)
    if not is_reference:
        return ""
    boxed = find_boxed_content(text)
    if boxed is not None:
        return normalize_answer(boxed)
    return normalize_answer(text)


def math_answers_equivalent(expected: Any, predicted: Any, *, dataset_name: str | None = None) -> bool:
    return math_equivalent(str(expected), str(predicted))


def safe_prediction_text(value: Any) -> str:
    try:
        return str(value)
    except ValueError:
        return f"<unprintable {type(value).__name__}>"


class MathExactTask(EvalTask):
    task_type = "math_exact"
    primary_metric = "accuracy_avg"

    def load_examples(
        self,
        path: str | Path,
        *,
        dataset_name: str | None = None,
        dataset_config: dict[str, Any] | None = None,
    ) -> list[EvalExample]:
        return super().load_examples(path, dataset_name=dataset_name, dataset_config=dataset_config)

    def row_to_example(self, row: dict[str, Any], idx: int, *, dataset_name: str | None = None) -> EvalExample:
        meta = dict(row.get("meta") if isinstance(row.get("meta"), dict) else {})
        prompt = require_first_present(row, ("instruction", "question", "problem", "prompt"), task_type=self.task_type, row_index=idx, field_name="prompt")
        reference = require_first_present(
            row,
            (
                "expected_answer",
                "answer",
                "final_answer",
                "gold",
                "target",
                "output",
                "solution",
                "reference",
            ),
            task_type=self.task_type,
            row_index=idx,
            field_name="reference answer",
        )
        example_id = str(first_present(row, ("id", "index", "task_id")) or f"{dataset_name or 'dataset'}/{idx}")
        prompt_text = str(prompt or "")
        if "instruction" not in row and "\\boxed" not in prompt_text:
            prompt_text = f"Problem: {prompt_text}\nMark your solution with \\boxed\nAnswer:"
        return EvalExample(
            id=example_id,
            prompt=prompt_text,
            reference=str(reference or ""),
            row_index=idx,
            meta={**meta, "dataset_name": dataset_name or meta.get("dataset_name")},
        )

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
            successes = []
            errors = []
            correct = 0
            for rollout in rollouts:
                dataset_name = str(rollout.example.meta.get("dataset_name") or "").lower()
                expected = normalize_math_answer(rollout.example.reference)
                predicted = extract_math_answer(rollout.output)
                extraction_method = "evalchemy_last_boxed" if predicted else "none"
                if "hmmt" in dataset_name:
                    predicted, extraction_method, equivalent = self._score_hmmt(rollout.output, rollout.example.reference)
                else:
                    equivalent = bool(predicted) and math_equivalent(expected, predicted)
                bucket = "successes" if equivalent else "errors"
                sample = make_eval_case(
                    task_type=self.task_type,
                    dataset_name=rollout.example.meta.get("dataset_name"),
                    repeat_index=repeat_idx,
                    bucket=bucket,
                    index=rollout.example.row_index,
                    example_id=rollout.example.id,
                    input={"question": rollout.example.prompt},
                    gold={"expected": expected, "reference_full": rollout.example.reference},
                    prediction={
                        "predicted": predicted,
                        "thinking_content": rollout.thinking_content,
                        "answer_content": rollout.answer_content,
                    },
                    diagnostics={
                        "passed": equivalent,
                        "correctness": equivalent,
                        "equivalent": equivalent,
                        "extraction_method": extraction_method,
                        "num_output_tokens": rollout.num_output_tokens,
                        "finish_reason": rollout.finish_reason,
                        "stop_reason": rollout.stop_reason,
                    },
                )
                if equivalent:
                    correct += 1
                    successes.append(sample)
                else:
                    errors.append(sample)
            total = len(rollouts)
            accuracy = correct / total if total else 0.0
            detail = {
                "index": repeat_idx,
                "accuracy": accuracy,
                "correct_count": correct,
                "total_count": total,
                "errors": errors,
                "successes": successes,
                "row_indices": [rollout.example.row_index for rollout in rollouts],
            }
            if rollouts and rollouts[0].seed is not None:
                detail["seed"] = rollouts[0].seed
            details.append(detail)
            run_stats.append(
                {
                    "repetition": repeat_idx + 1,
                    "num_total": total,
                    "num_solved": correct,
                    "accuracy": accuracy,
                }
            )

        accuracy_avg, accuracy_std, accuracy_std_err = self.average_score(run_stats)
        solved_avg = sum(float(item["num_solved"]) for item in run_stats) / len(run_stats) if run_stats else 0.0
        metrics = {
            "num_total": len(examples),
            "solved_avg": solved_avg,
            "run_stats": run_stats,
            "accuracy_avg": accuracy_avg,
            "accuracy_std": accuracy_std,
            "accuracy_std_err": accuracy_std_err,
            "num_repeat": len(run_stats),
        }
        if len(run_stats) == 1:
            metrics.update(
                {
                    "num_solved": run_stats[0]["num_solved"],
                    "accuracy": run_stats[0]["accuracy"],
                }
            )
        selected_metric = primary_metric or self.primary_metric
        metrics = standardize_eval_metrics(metrics, details, primary_metric=selected_metric)
        score = float(metrics["score"])
        return EvalScore(self.task_type, selected_metric, score, metrics, details)

    @staticmethod
    def _score_hmmt(output: str, reference: Any) -> tuple[str, str, bool]:
        list_answer = "," in str(reference)
        pred, _ = _matharena_extract_answer(output, False, True, list_answer)
        gold, _ = _matharena_parse_answer(str(reference))
        equivalent = bool(_matharena_check_answers(pred, gold))
        return safe_prediction_text(pred), "matharena_extract_answer", equivalent
