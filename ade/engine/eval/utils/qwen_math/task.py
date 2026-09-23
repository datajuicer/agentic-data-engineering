from __future__ import annotations

import multiprocessing
import queue
from typing import Any

from ...contracts import EvalExample, EvalScore, RolloutRecord, make_eval_case
from ...metrics import standardize_eval_metrics
from ..base import EvalTask
from .grader import math_equal
from .parser import extract_answer, strip_string


QWEN_PROMPT = (
    "{problem}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
)
QWEN_MATH_GRADER_TIMEOUT_SECONDS = 3.0
QWEN_MATH_GRADER_SHUTDOWN_GRACE_SECONDS = 0.25
QWEN_MATH_GRADER_RESULT_GRACE_SECONDS = 0.1


def _math_equal_worker(prediction: str, reference: str, result_queue) -> None:
    try:
        result_queue.put(bool(math_equal(prediction, reference)))
    except Exception:
        result_queue.put(False)


def qwen_math_equal_with_timeout(
    prediction: str,
    reference: str,
    *,
    timeout_seconds: float = QWEN_MATH_GRADER_TIMEOUT_SECONDS,
) -> tuple[bool, bool]:
    context = multiprocessing.get_context("fork")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_math_equal_worker,
        args=(prediction, reference, result_queue),
    )
    process.start()
    process.join(timeout_seconds)
    timed_out = process.is_alive()
    if timed_out:
        process.terminate()
        process.join(QWEN_MATH_GRADER_SHUTDOWN_GRACE_SECONDS)
        if process.is_alive():
            process.kill()
            process.join(QWEN_MATH_GRADER_SHUTDOWN_GRACE_SECONDS)
        if process.is_alive():
            _close_result_queue(result_queue)
            raise RuntimeError("Qwen math grader process did not stop")
        _close_result_queue(result_queue)
        return False, True
    try:
        return bool(result_queue.get(timeout=QWEN_MATH_GRADER_RESULT_GRACE_SECONDS)), False
    except queue.Empty:
        return False, False
    finally:
        _close_result_queue(result_queue)


def _close_result_queue(result_queue) -> None:
    result_queue.cancel_join_thread()
    result_queue.close()


def qwen_extraction_method(text: str) -> str:
    if "boxed" in text:
        return "qwen_math.last_boxed"
    if "final answer is" in text or "he answer is" in text:
        return "qwen_math.answer_phrase"
    if "答案是" in text:
        return "qwen_math.chinese_answer_phrase"
    return "qwen_math.last_number"


class QwenMathEvaluationTask(EvalTask):
    """Shared SimpleRL Qwen Math extraction/grading mechanics."""

    primary_metric = "accuracy_avg"
    parser_data_name = "math"

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
            successes: list[dict[str, Any]] = []
            errors: list[dict[str, Any]] = []
            correct = 0
            for rollout in rollouts:
                reference = strip_string(rollout.example.reference)
                prediction = strip_string(
                    extract_answer(rollout.output, self.parser_data_name)
                )
                equivalent, _ = qwen_math_equal_with_timeout(
                    prediction,
                    reference,
                )
                sample = make_eval_case(
                    task_type=self.task_type,
                    dataset_name=rollout.example.meta.get("dataset_name"),
                    repeat_index=repeat_idx,
                    bucket="successes" if equivalent else "errors",
                    index=rollout.example.row_index,
                    example_id=rollout.example.id,
                    input={"question": rollout.example.prompt},
                    gold={
                        "expected": reference,
                        "reference_full": rollout.example.meta.get(
                            "reference_full",
                            rollout.example.reference,
                        ),
                    },
                    prediction={
                        "predicted": prediction,
                        "thinking_content": rollout.thinking_content,
                        "answer_content": rollout.answer_content,
                    },
                    diagnostics={
                        "passed": equivalent,
                        "correctness": equivalent,
                        "equivalent": equivalent,
                        "extraction_method": qwen_extraction_method(rollout.output),
                        "grader": "simplelr_qwen_math.math_equal",
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

        solved_avg = (
            sum(float(item["num_solved"]) for item in run_stats) / len(run_stats)
            if run_stats
            else 0.0
        )
        metrics = standardize_eval_metrics(
            {
                "num_total": len(examples),
                "solved_avg": solved_avg,
                "run_stats": run_stats,
                "num_repeat": len(run_stats),
            },
            details,
            primary_metric=primary_metric or self.primary_metric,
        )
        if len(run_stats) == 1:
            metrics.update(
                {
                    "num_solved": run_stats[0]["num_solved"],
                    "accuracy": run_stats[0]["accuracy"],
                }
            )
        selected_metric = primary_metric or self.primary_metric
        return EvalScore(
            self.task_type,
            selected_metric,
            float(metrics["score"]),
            metrics,
            details,
        )
