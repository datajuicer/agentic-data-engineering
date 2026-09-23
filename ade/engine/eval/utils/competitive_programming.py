from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from ..contracts import EvalExample, EvalScore, RolloutRecord, make_eval_case
from ..metrics import standardize_eval_metrics
from .base import EvalTask, first_present, require_first_present
from .execution import bounded_python_stdio_source
from .python_code import extract_python_code


def _normalized_output(value: str) -> list[str]:
    return str(value or "").split()


def _run_stdio(code: str, case: dict[str, Any], timeout: float) -> tuple[bool, str]:
    try:
        with tempfile.TemporaryDirectory(prefix="ade-competitive-exec-") as working_dir:
            program = Path(working_dir) / "program.py"
            program.write_text(bounded_python_stdio_source(code), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-I", str(program)],
                input=str(case.get("input") or ""),
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
                cwd=working_dir,
            )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout:g}s"
    if result.returncode != 0:
        return False, f"process exited with {result.returncode}: {result.stderr[-500:]}"
    if _normalized_output(result.stdout) != _normalized_output(str(case.get("output") or "")):
        return False, "wrong answer"
    return True, ""


class CompetitiveProgrammingTask(EvalTask):
    task_type = "competitive_programming"
    primary_metric = "accuracy_avg"
    execution_timeout = 6.0

    def row_to_example(
        self,
        row: dict[str, Any],
        idx: int,
        *,
        dataset_name: str | None = None,
    ) -> EvalExample:
        prompt = require_first_present(
            row,
            ("description", "question", "problem", "prompt"),
            task_type=self.task_type,
            row_index=idx,
            field_name="problem statement",
        )
        raw_tests = row.get("official_tests") or row.get("tests") or row.get("examples") or []
        tests: list[dict[str, str]] = []
        for item in raw_tests:
            if isinstance(item, dict):
                tests.append({"input": str(item.get("input") or ""), "output": str(item.get("output") or "")})
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                tests.append({"input": str(item[0]), "output": str(item[1])})
        if not tests:
            raise ValueError(f"{self.task_type} dataset row {idx} has no executable stdio tests")
        sections = [
            str(row.get("title") or "Programming problem"),
            str(prompt),
            "Input:\n" + str(row.get("input_format") or row.get("input") or ""),
            "Output:\n" + str(row.get("output_format") or row.get("output") or ""),
            "Return a complete Python 3 program in a Python code block.",
        ]
        return EvalExample(
            id=str(first_present(row, ("id", "problem_id", "task_id", "index")) or f"{dataset_name or 'dataset'}/{idx}"),
            prompt="\n\n".join(sections),
            reference="",
            row_index=idx,
            meta={"dataset_name": dataset_name, "raw_example": dict(row), "stdio_tests": tests},
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
            successes, errors = [], []
            correct = 0
            for rollout in rollouts:
                code, extraction_method = extract_python_code(rollout.output)
                passed, reason = (False, "empty code generation")
                if code.strip():
                    results = [_run_stdio(code, case, self.execution_timeout) for case in rollout.example.meta["stdio_tests"]]
                    passed = all(result[0] for result in results)
                    reason = next((result[1] for result in results if not result[0]), "")
                case = make_eval_case(
                    task_type=self.task_type,
                    dataset_name=rollout.example.meta.get("dataset_name"),
                    repeat_index=repeat_idx,
                    bucket="successes" if passed else "errors",
                    index=rollout.example.row_index,
                    example_id=rollout.example.id,
                    input={"prompt": rollout.example.prompt},
                    gold={"test_count": len(rollout.example.meta["stdio_tests"])},
                    prediction={"generation": code},
                    diagnostics={"passed": passed, "reason": reason, "extraction_method": extraction_method},
                )
                (successes if passed else errors).append(case)
                correct += int(passed)
            total = len(rollouts)
            accuracy = correct / total if total else 0.0
            details.append({"index": repeat_idx, "accuracy": accuracy, "correct_count": correct, "total_count": total, "errors": errors, "successes": successes})
            run_stats.append({"repetition": repeat_idx + 1, "num_total": total, "num_solved": correct, "accuracy": accuracy})
        accuracy_avg, accuracy_std, accuracy_std_err = self.average_score(run_stats)
        metrics = standardize_eval_metrics(
            {"accuracy_avg": accuracy_avg, "accuracy_std": accuracy_std, "accuracy_std_err": accuracy_std_err, "num_total": len(examples), "run_stats": run_stats, "num_repeat": len(run_stats)},
            details,
            primary_metric=primary_metric or self.primary_metric,
        )
        return EvalScore(self.task_type, primary_metric or self.primary_metric, float(metrics["score"]), metrics, details)


class CodeELOTask(CompetitiveProgrammingTask):
    task_type = "codeelo"


class CodeforcesTask(CompetitiveProgrammingTask):
    task_type = "codeforces"
