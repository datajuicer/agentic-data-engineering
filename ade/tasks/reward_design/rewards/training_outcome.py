"""Fixed, versioned training-answer extraction and outcome comparison.

This module is Engine-owned.  Generated reward functions receive only the
resulting ``outcome_score`` and cannot reinterpret dataset ground truth.
"""

from __future__ import annotations

import math
import multiprocessing
import re
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from typing import Final

from ade.engine.eval.utils.qwen_math.grader import math_equal as qwen_math_equal
from ade.engine.eval.utils.qwen_math.parser import extract_answer
from ade.engine.eval.utils.qwen_math.task import qwen_extraction_method
from ade.tasks.reward_design.rewards.guru_math_reference import grade_answer, match_answer
from ade.tasks.reward_design.rewards.guru_math_grader import math_equal


SCHEMA_VERSION: Final = "ade.training_outcome.v1"
MATH_ADAPTER_ID: Final = "math_hf_math_verify.v1"
GURU_MATH_ADAPTER_ID: Final = "guru_naive_dapo.v1"
QWEN_MATH_GRADER_TIMEOUT_SECONDS: Final = 30.0

SOURCE_ADAPTER_BINDINGS: Final[dict[str, str]] = {
    "math": MATH_ADAPTER_ID,
    "math_qwen__merged_deduped_dapo_or1_dataset": MATH_ADAPTER_ID,
    "math_qwen__deepscaler_preview": MATH_ADAPTER_ID,
    "math__merged_deduped_dapo_or1_dataset": GURU_MATH_ADAPTER_ID,
    "math__deepscaler_preview": GURU_MATH_ADAPTER_ID,
}

_QWEN_MATH_POOL: ProcessPoolExecutor | None = None
_QWEN_MATH_POOL_LOCK = threading.Lock()


@dataclass(frozen=True)
class TrainingOutcome:
    schema_version: str
    adapter_id: str
    data_source: str
    extracted_answer: str
    extraction_method: str
    status: str
    outcome_score: float
    format_matched: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def adapter_id_for_source(data_source: object) -> str:
    source = str(data_source or "")
    try:
        return SOURCE_ADAPTER_BINDINGS[source]
    except KeyError as error:
        raise ValueError(
            f"RFT training data_source {source!r} has no TrainingOutcomeAdapter binding"
        ) from error


def validate_adapter_binding(
    binding: object,
    *,
    data_sources: set[str] | frozenset[str],
) -> dict[str, object]:
    if not isinstance(binding, dict):
        raise ValueError("task.rft.training_outcome_adapter must be an object")
    if not set(binding).issubset({"adapter_id", "data_sources", "extraction_protocol"}):
        raise ValueError(
            "task.rft.training_outcome_adapter has unknown fields"
        )
    if set(binding) < {"adapter_id", "data_sources"}:
        raise ValueError(
            "task.rft.training_outcome_adapter requires adapter_id and data_sources"
        )
    declared_sources = binding["data_sources"]
    if (
        not isinstance(declared_sources, list)
        or not declared_sources
        or any(not isinstance(source, str) or not source for source in declared_sources)
    ):
        raise ValueError("training_outcome_adapter.data_sources must be non-empty strings")
    declared = set(declared_sources)
    if declared != set(data_sources):
        raise ValueError(
            "training_outcome_adapter.data_sources do not match the training dataset"
        )
    expected_ids = {adapter_id_for_source(source) for source in declared}
    if len(expected_ids) != 1 or binding["adapter_id"] not in expected_ids:
        raise ValueError(
            "training_outcome_adapter.adapter_id does not match the source binding"
        )
    result = {"adapter_id": str(binding["adapter_id"]), "data_sources": sorted(declared)}
    if "extraction_protocol" in binding:
        protocol = binding["extraction_protocol"]
        if protocol != "math_qwen_boxed.v1":
            raise ValueError(
                "training_outcome_adapter.extraction_protocol must be math_qwen_boxed.v1"
            )
        result["extraction_protocol"] = protocol
    return result


def parquet_data_sources(path: str) -> frozenset[str]:
    """Read only the source column used for config admission."""
    import pyarrow.parquet as parquet

    table = parquet.read_table(path, columns=["data_source"])
    values = table.column("data_source").unique().to_pylist()
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError("RFT training dataset has invalid data_source values")
    return frozenset(values)


def compute_training_outcome(
    data_source: object,
    response_content: object,
    ground_truth: object,
    *,
    expected_adapter_id: str | None = None,
) -> TrainingOutcome:
    source = str(data_source or "")
    adapter_id = adapter_id_for_source(source)
    if expected_adapter_id is not None and adapter_id != expected_adapter_id:
        raise ValueError(
            f"TrainingOutcomeAdapter mismatch: source {source!r} requires {adapter_id!r}, "
            f"runtime declared {expected_adapter_id!r}"
        )
    response = str(response_content or "")
    expected = str(ground_truth or "")
    if adapter_id == MATH_ADAPTER_ID:
        return _simplelr_outcome(source, response, expected)
    return _guru_math_outcome(source, response, expected)


def _simplelr_outcome(source: str, response: str, expected: str) -> TrainingOutcome:
    model_output = re.sub(
        r"^.*?<\|im_start\|>assistant",
        "<|im_start|>assistant",
        response,
        flags=re.DOTALL,
        count=1,
    )
    for stop_word in ("</s>", "<|im_end|>", "<|endoftext|>"):
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    extracted = extract_answer(model_output, data_name="math")
    method = qwen_extraction_method(model_output)
    format_matched = method == "qwen_math.last_boxed"
    return _record(
        adapter_id=MATH_ADAPTER_ID,
        source=source,
        extracted=extracted,
        method=method,
        expected=expected,
        format_matched=format_matched,
        comparator=_qwen_math_equal_with_timeout,
    )


def _guru_math_outcome(source: str, response: str, expected: str) -> TrainingOutcome:
    format_matched, extracted = match_answer(response)
    method = (
        "guru.naive_dapo.post_think_last_boxed"
        if format_matched
        else "guru.naive_dapo.post_think_raw"
    )
    return _record(
        adapter_id=GURU_MATH_ADAPTER_ID,
        source=source,
        extracted=extracted,
        method=method,
        expected=expected,
        format_matched=format_matched,
        comparator=_guru_equal,
    )


def _record(
    *,
    adapter_id: str,
    source: str,
    extracted: str,
    method: str,
    expected: str,
    format_matched: bool,
    comparator,
) -> TrainingOutcome:
    if not expected:
        status, score = "invalid_ground_truth", 0.0
    elif not extracted:
        status, score = "empty_extraction", 0.0
    else:
        try:
            score = 1.0 if comparator(extracted, expected) else 0.0
            status = "scored"
        except (BrokenProcessPool, ImportError):
            # A failed grading environment must fail the Attempt, not score the answer.
            raise
        except Exception:
            status, score = "comparison_error", 0.0
    if not math.isfinite(score) or score not in (0.0, 1.0):
        raise ValueError("TrainingOutcomeAdapter must emit a finite binary outcome")
    return TrainingOutcome(
        schema_version=SCHEMA_VERSION,
        adapter_id=adapter_id,
        data_source=source,
        extracted_answer=str(extracted or ""),
        extraction_method=method,
        status=status,
        outcome_score=score,
        format_matched=bool(format_matched),
    )


def _qwen_math_equal_with_timeout(extracted: str, expected: str) -> bool:
    global _QWEN_MATH_POOL
    if _QWEN_MATH_POOL is None:
        with _QWEN_MATH_POOL_LOCK:
            if _QWEN_MATH_POOL is None:
                _QWEN_MATH_POOL = ProcessPoolExecutor(
                    max_workers=4,
                    mp_context=multiprocessing.get_context("spawn"),
                )
    future = _QWEN_MATH_POOL.submit(
        _qwen_math_equal_in_worker, extracted, expected
    )
    try:
        return bool(future.result(timeout=QWEN_MATH_GRADER_TIMEOUT_SECONDS))
    except FutureTimeoutError:
        future.cancel()
        return False


def _qwen_math_equal_in_worker(extracted: str, expected: str) -> bool:
    return bool(qwen_math_equal(extracted, expected, timeout=True))


def _guru_equal(extracted: str, expected: str) -> bool:
    correct, _normalized = grade_answer(extracted, expected)
    if correct:
        return True
    try:
        if "\\pi" in extracted or "\\pi" in expected:
            return any(
                math_equal(extracted, expected, timeout=True, pi=pi)
                for pi in (math.pi, 3.14)
            )
        return bool(math_equal(extracted, expected, timeout=True))
    except (BrokenProcessPool, ImportError):
        raise
    except Exception:
        return False
