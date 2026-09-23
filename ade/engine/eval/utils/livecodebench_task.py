from __future__ import annotations

import copy
import contextlib
import json
import re
import signal
import threading
from collections import defaultdict
from typing import Any

from ..contracts import EvalExample, EvalScore, RolloutRecord, make_eval_case
from ..metrics import standardize_eval_metrics
from .base import EvalTask, first_present, read_json_or_jsonl


def fenced_code_blocks(text: str) -> list[str]:
    return re.findall(r"```(?:[a-zA-Z]*)\n(.*?)```", text or "", flags=re.DOTALL)


def _ensure_lcb_imports():
    from .livecodebench import lcb_run, post_process_code, translate_private_test_cases

    return lcb_run, post_process_code, translate_private_test_cases


def _has_stdin_test(public_test_cases: Any) -> bool:
    if isinstance(public_test_cases, str):
        tests = json.loads(public_test_cases)
    else:
        tests = public_test_cases or []
    return any(isinstance(item, dict) and item.get("testtype") == "stdin" for item in tests)


class _HardTimeout(Exception):
    pass


@contextlib.contextmanager
def _hard_timeout(seconds: int):
    if seconds <= 0 or threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise_timeout(_signum, _frame):
        raise _HardTimeout(f"timed out after {seconds}s")

    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class LiveCodeBenchTask(EvalTask):
    task_type = "livecodebench"
    primary_metric = "accuracy_avg"

    def row_to_example(self, row: dict[str, Any], idx: int, *, dataset_name: str | None = None) -> EvalExample:
        raw = dict(row)
        raw["entry_point"] = raw["starter_code"]
        raw["task_id"] = raw["question_id"]
        raw["is_stdin"] = _has_stdin_test(raw["public_test_cases"])
        if raw["is_stdin"]:
            prefix = "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."
        else:
            prefix = "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."
        example_id = str(first_present(raw, ("task_id", "question_id", "id", "index")) or f"{dataset_name or 'dataset'}/{idx}")
        return EvalExample(
            id=example_id,
            prompt=prefix + str(raw["question_content"]),
            reference="",
            row_index=idx,
            meta={
                "dataset_name": dataset_name,
                "raw_example": raw,
                "difficulty": raw.get("difficulty"),
            },
        )

    def load_examples(
        self,
        path,
        *,
        dataset_name: str | None = None,
        dataset_config: dict[str, Any] | None = None,
    ) -> list[EvalExample]:
        split = (dataset_config or {}).get("split") or (dataset_config or {}).get("hf_split")
        return [self.row_to_example(row, idx, dataset_name=dataset_name) for idx, row in enumerate(read_json_or_jsonl(path, split=split))]

    def score_rollouts(
        self,
        examples: list[EvalExample],
        rollouts_by_repeat: list[list[RolloutRecord]],
        *,
        primary_metric: str | None = None,
    ) -> EvalScore:
        details: list[dict[str, Any]] = []
        run_stats: list[dict[str, Any]] = []
        raw_metrics: list[dict[str, Any]] = []
        for repeat_idx, rollouts in enumerate(rollouts_by_repeat):
            successes = []
            errors = []
            correct = 0
            per_difficulty_correct: dict[str, int] = defaultdict(int)
            per_difficulty_total: dict[str, int] = defaultdict(int)
            for rollout in rollouts:
                blocks = fenced_code_blocks(rollout.output)
                difficulty = str(rollout.example.meta.get("difficulty") or "")
                per_difficulty_total[difficulty] += 1
                if not blocks:
                    passed = False
                    reason = "Does not contain code component."
                    code = ""
                    execution_details: list[list[Any]] = []
                    extraction_method = "none"
                else:
                    lcb_run, post_process_code, translate_private_test_cases = _ensure_lcb_imports()
                    code = post_process_code(blocks[-1])
                    extraction_method = "last_fenced_code_block"
                    problem = copy.deepcopy(rollout.example.meta["raw_example"])
                    if "test" not in problem:
                        problem["test"] = translate_private_test_cases(problem["private_test_cases"])
                    try:
                        with _hard_timeout(30):
                            results = list(lcb_run(problem, code, timeout=6, is_extracted=not problem["is_stdin"]))
                    except _HardTimeout as exc:
                        results = [(False, f"LCB hard timeout: {exc}")]
                    except Exception as exc:
                        results = [(False, f"LCB execution error: {type(exc).__name__}: {exc}")]
                    execution_details = [list(item) for item in results]
                    passed = all(item[0] for item in results)
                    reason = "" if passed else next((item[1] for item in results if not item[0]), "Code is incorrect.")
                bucket = "successes" if passed else "errors"
                sample = make_eval_case(
                    task_type=self.task_type,
                    dataset_name=rollout.example.meta.get("dataset_name"),
                    repeat_index=repeat_idx,
                    bucket=bucket,
                    index=rollout.example.row_index,
                    example_id=rollout.example.id,
                    input={"prompt": rollout.example.prompt},
                    gold={
                        "raw_example": rollout.example.meta.get("raw_example"),
                        "execution_details": execution_details,
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
                    metadata={"difficulty": difficulty},
                )
                if passed:
                    correct += 1
                    per_difficulty_correct[difficulty] += 1
                    successes.append(sample)
                else:
                    errors.append(sample)
            total = len(rollouts)
            accuracy = correct / total if total else 0.0
            per_diff_acc = {
                f"accuracy_{difficulty}": per_difficulty_correct.get(difficulty, 0) / count
                for difficulty, count in per_difficulty_total.items()
                if count
            }
            raw_metrics.append({
                "total_correct": correct,
                "total_finish": total,
                "accuracy": accuracy,
                "per_difficulty_correct": dict(per_difficulty_correct),
                "per_difficulty_total": dict(per_difficulty_total),
                **per_diff_acc,
            })
            detail = {
                "index": repeat_idx,
                "accuracy": accuracy,
                "correct_count": correct,
                "total_count": total,
                "errors": errors,
                "successes": successes,
                "row_indices": [rollout.example.row_index for rollout in rollouts],
                **per_diff_acc,
            }
            if rollouts and rollouts[0].seed is not None:
                detail["seed"] = rollouts[0].seed
            details.append(detail)
            run_stats.append({"repetition": repeat_idx + 1, "num_total": total, "num_solved": correct, "accuracy": accuracy})

        accuracy_avg, accuracy_std, accuracy_std_err = self.average_score(run_stats)
        metrics: dict[str, Any] = {
            "accuracy_avg": accuracy_avg,
            "accuracy_std": accuracy_std,
            "accuracy_std_err": accuracy_std_err,
            "raw_metrics": raw_metrics,
            "num_total": len(examples),
            "solved_avg": sum(float(item["num_solved"]) for item in run_stats) / len(run_stats) if run_stats else 0.0,
            "run_stats": run_stats,
            "num_repeat": len(run_stats),
        }
        selected_metric = primary_metric or self.primary_metric
        metrics = standardize_eval_metrics(metrics, details, primary_metric=selected_metric)
        return EvalScore(self.task_type, selected_metric, float(metrics["score"]), metrics, details)
