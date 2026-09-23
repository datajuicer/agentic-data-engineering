"""Engine-owned binding from reward calls to one Rubric Job per RL step."""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any

from ade.local_rubric_judge.client import JobRejected, LocalRubricJudgeClient
from ade.memory.usage import RunUsageLedger
from ade.rubric_jobs import Rubric, RubricInputRow, encode_input_jsonl, join_results
from ade.rubric_jobs.process import parse_process_rubric


class RubricJudgeUnavailable(RuntimeError):
    pass


class RubricRowUnavailable(RubricJudgeUnavailable):
    """One terminal Judge row is unavailable and may use row-local fallback."""


class RewardCommandFailed(RuntimeError):
    pass


class JudgeCircuitBreaker:
    def __init__(self, *, max_failed_row_ratio: float, max_consecutive_unhealthy_jobs: int) -> None:
        if not 0.0 <= max_failed_row_ratio <= 1.0:
            raise ValueError("max_failed_row_ratio must be within [0,1]")
        if max_consecutive_unhealthy_jobs < 1:
            raise ValueError("max_consecutive_unhealthy_jobs must be positive")
        self.max_failed_row_ratio = max_failed_row_ratio
        self.max_consecutive_unhealthy_jobs = max_consecutive_unhealthy_jobs
        self.consecutive_unhealthy_jobs = 0

    def admit(self, result) -> None:
        if result.status.state in {"failed", "cancelled"}:
            self._unhealthy(f"Judge job ended {result.status.state}")
        ratio = result.status.error_rows / max(1, result.status.total_rows)
        if ratio > self.max_failed_row_ratio:
            raise RewardCommandFailed(
                f"Judge failed-row ratio {ratio:.6f} exceeds {self.max_failed_row_ratio:.6f}"
            )
        else:
            self.consecutive_unhealthy_jobs = 0

    def unavailable(self, reason: str) -> None:
        self._unhealthy(reason)

    def _unhealthy(self, reason: str) -> None:
        self.consecutive_unhealthy_jobs += 1
        if self.consecutive_unhealthy_jobs >= self.max_consecutive_unhealthy_jobs:
            raise RewardCommandFailed(reason)
        raise RubricJudgeUnavailable(reason)


class StepRubricDispatcher:
    """Collect the known Judge-dependent rows for one full reward batch."""

    def __init__(
        self,
        client: LocalRubricJudgeClient,
        *,
        submission_id: str,
        expected_requests: int | None,
        job_metadata: dict[str, Any],
        circuit_breaker: JudgeCircuitBreaker | None = None,
        usage_ledger: RunUsageLedger | None = None,
    ) -> None:
        if expected_requests is not None and expected_requests < 1:
            raise ValueError("expected_requests must be positive")
        self.client = client
        self.submission_id = submission_id
        self.expected_requests = expected_requests
        self.job_metadata = dict(job_metadata)
        self.circuit_breaker = circuit_breaker
        self.usage_ledger = usage_ledger
        self._requests: list[tuple[RubricInputRow, asyncio.Future[float]]] = []
        self._dispatch_task: asyncio.Task[None] | None = None
        self._cancelled = False

    @property
    def request_count(self) -> int:
        return len(self._requests)

    def seal(self) -> None:
        if self._dispatch_task is not None:
            return
        if not self._requests:
            raise RuntimeError("cannot submit an empty RL step Judge job")
        self.expected_requests = len(self._requests)
        self._dispatch_task = asyncio.create_task(self._dispatch())

    async def score(
        self, question_prompt: str, response_content: str, process_rubric: str
    ) -> float:
        if self._cancelled:
            raise asyncio.CancelledError
        if self._dispatch_task is not None:
            raise RuntimeError("RL step Judge job was already submitted")
        row = _rubric_row(
            len(self._requests), question_prompt, response_content, process_rubric
        )
        future: asyncio.Future[float] = asyncio.get_running_loop().create_future()
        self._requests.append((row, future))
        if self.expected_requests is not None and len(self._requests) == self.expected_requests:
            self._dispatch_task = asyncio.create_task(self._dispatch())
        elif self.expected_requests is not None and len(self._requests) > self.expected_requests:
            raise RuntimeError("reward artifact issued more Judge calls than expected")
        return await future

    async def _dispatch(self) -> None:
        rows = tuple(row for row, _ in self._requests)
        try:
            submitted = await self.client.submit(
                submission_id=self.submission_id,
                input_jsonl=encode_input_jsonl(rows),
                job_metadata=self.job_metadata,
            )
            result = await self.client.wait(submitted.job_id)
            if self.circuit_breaker is not None:
                self.circuit_breaker.admit(result)
            if self.usage_ledger is not None:
                self.usage_ledger.append(
                    {
                        "event_id": f"local-rubric-job:{result.status.job_id}",
                        "category": "artifact_model",
                        "component": "local_judge",
                        "provider": "local_rubric_judge",
                        "transition_intent": "engine_training_evidence",
                        **self.job_metadata,
                        **result.usage.to_dict(),
                    }
                )
            joined = {
                output.record_id: output
                for output in join_results(rows, result.rows)
            }
            for row, future in self._requests:
                output = joined[row.record_id]
                if output.status == "error":
                    assert output.error is not None
                    future.set_exception(
                        RubricRowUnavailable(
                            f"{output.error.code}: {output.error.message}"
                        )
                    )
                else:
                    assert output.result is not None
                    try:
                        future.set_result(
                            project_process_score(
                                projection=row.metadata["projection"],
                                result=output.result,
                            )
                        )
                    except RubricJudgeUnavailable as error:
                        future.set_exception(RubricRowUnavailable(str(error)))
        except Exception as error:
            if isinstance(error, JobRejected):
                error = RewardCommandFailed(f"Local Rubric Judge rejected job: {error}")
            if (
                self.circuit_breaker is not None
                and not isinstance(error, (RubricJudgeUnavailable, RewardCommandFailed))
            ):
                try:
                    self.circuit_breaker.unavailable(str(error))
                except Exception as policy_error:
                    error = policy_error
            wrapped = (
                error
                if isinstance(error, RewardCommandFailed)
                else RewardCommandFailed(
                    f"Local Rubric Judge job failed: {type(error).__name__}: {error}"
                )
            )
            for _, future in self._requests:
                if not future.done():
                    future.set_exception(wrapped)

    async def close(self) -> None:
        if self._cancelled:
            if self._dispatch_task is not None:
                await self._dispatch_task
            return
        if self.expected_requests is None:
            if not self._requests:
                return
            self.seal()
        if len(self._requests) != self.expected_requests:
            error = RewardCommandFailed(
                "reward artifact did not issue the expected number of Judge calls"
            )
            for _, future in self._requests:
                if not future.done():
                    future.set_exception(error)
        if self._dispatch_task is not None:
            await self._dispatch_task

    async def cancel(self) -> None:
        """Cancel every non-terminal job handle owned by this Engine client."""
        self._cancelled = True
        await self.client.cancel_all()
        for _, future in self._requests:
            if not future.done():
                future.cancel()


class RoundRubricDispatcher:
    """Batch Judge requests across multi-round, branched artifact execution.

    Each active sample contributes at most one request (or a terminal marker) to
    a round.  The dispatcher submits bounded heterogeneous JSONL jobs, waits for each,
    then opens the next barrier.  This keeps RFT rollouts and pre-training SFT
    selection on the same capability/control path without making a dispatcher
    per trajectory.
    """

    def __init__(self, client: LocalRubricJudgeClient, *, sample_ids: set[str],
                 submission_id: str,
                 job_metadata: dict[str, Any],
                 circuit_breaker: JudgeCircuitBreaker | None = None,
                 usage_ledger: RunUsageLedger | None = None,
                 row_failure_fallback: float | None = None,
                 max_requests_per_job: int | None = None) -> None:
        if not sample_ids:
            raise ValueError("sample_ids must not be empty")
        if max_requests_per_job is not None and max_requests_per_job < 1:
            raise ValueError("max_requests_per_job must be positive")
        self.client = client
        self.sample_ids = set(sample_ids)
        self.submission_id = submission_id
        self.job_metadata = dict(job_metadata)
        self.circuit_breaker = circuit_breaker
        self.usage_ledger = usage_ledger
        self.row_failure_fallback = row_failure_fallback
        self.max_requests_per_job = max_requests_per_job
        self._round = 0
        self._requests: dict[str, tuple[RubricInputRow, asyncio.Future[dict[str, Any]]]] = {}
        self._done: set[str] = set()
        self._terminal: set[str] = set()
        self._dispatch: asyncio.Task[None] | None = None
        self._condition = asyncio.Condition()
        self._cancelled = False

    async def score(self, sample_id: str, question_prompt: str,
                    response_content: str, process_rubric: str) -> float:
        evaluation = await self.evaluate(
            sample_id, question_prompt, response_content, process_rubric
        )
        return float(evaluation["projected_score"])

    async def evaluate(self, sample_id: str, question_prompt: str,
                       response_content: str, process_rubric: str) -> dict[str, Any]:
        if sample_id not in self.sample_ids:
            raise ValueError(f"unknown Judge sample_id: {sample_id}")
        async with self._condition:
            while self._dispatch is not None:
                await self._condition.wait()
            if sample_id in self._requests or sample_id in self._done:
                raise RuntimeError("sample issued more than one Judge request in a round")
            future: asyncio.Future[dict[str, Any]] = (
                asyncio.get_running_loop().create_future()
            )
            row = _rubric_row(len(self._requests), question_prompt, response_content, process_rubric)
            self._requests[sample_id] = (row, future)
            self._maybe_start_locked()
        return await future

    async def mark_done(self, sample_id: str) -> None:
        if sample_id not in self.sample_ids:
            raise ValueError(f"unknown Judge sample_id: {sample_id}")
        async with self._condition:
            while self._dispatch is not None:
                await self._condition.wait()
            self._done.add(sample_id)
            self._terminal.add(sample_id)
            self._maybe_start_locked()

    def _maybe_start_locked(self) -> None:
        ready = len(self._requests) + len(self._done) + len(self._terminal - self._done)
        chunk_ready = (
            self.max_requests_per_job is not None
            and len(self._requests) >= self.max_requests_per_job
        )
        if self._dispatch is None and (ready == len(self.sample_ids) or chunk_ready):
            if self._requests:
                self._dispatch = asyncio.create_task(self._dispatch_round())
            else:
                self._round += 1
                self._done.clear()

    async def _dispatch_round(self) -> None:
        async with self._condition:
            requests = self._requests
            self._requests = {}
            done = set(self._done)
            self._done.clear()
            round_no = self._round
        rows = tuple(item[0] for item in requests.values())
        try:
            submitted = await self.client.submit(
                submission_id=f"{self.submission_id}:round-{round_no}",
                input_jsonl=encode_input_jsonl(rows), job_metadata=self.job_metadata)
            result = await self.client.wait(submitted.job_id)
            if self.circuit_breaker is not None:
                self.circuit_breaker.admit(result)
            joined = {output.record_id: output for output in join_results(rows, result.rows)}
            for row, future in requests.values():
                output = joined[row.record_id]
                if output.status == "error":
                    if self.row_failure_fallback is not None:
                        future.set_result({
                            "status": "unavailable",
                            "scores_by_dimension": {},
                            "projected_score": self.row_failure_fallback,
                        })
                    else:
                        future.set_exception(RubricRowUnavailable(str(output.error)))
                else:
                    assert output.result is not None
                    future.set_result(project_process_evaluation(
                        projection=row.metadata["projection"], result=output.result))
        except Exception as error:
            wrapped = error if isinstance(error, RewardCommandFailed) else RewardCommandFailed(str(error))
            for _, future in requests.values():
                if not future.done():
                    future.set_exception(wrapped)
        finally:
            async with self._condition:
                self._round += 1
                self._dispatch = None
                self._condition.notify_all()

    async def close(self) -> None:
        async with self._condition:
            if self._requests or self._done:
                self._maybe_start_locked()
            task = self._dispatch
        if task is not None:
            await task

    async def cancel(self) -> None:
        self._cancelled = True
        await self.client.cancel_all()


class SelectionJudgeBatchDispatcher:
    """Submit bounded batches for SFT selection enrichment.

    Selection artifacts may choose different rubrics per candidate, so the
    batch contains heterogeneous canonical rubric rows. Row errors are
    converted to explicit unavailable evidence; a rejected job remains a
    selection contract failure.
    """

    def __init__(
        self,
        client: LocalRubricJudgeClient,
        *,
        max_batch_size: int,
        submission_id: str,
        job_metadata: dict[str, Any],
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("selection Judge max_batch_size must be positive")
        self.client = client
        self.max_batch_size = max_batch_size
        self.submission_id = submission_id
        self.job_metadata = dict(job_metadata)
        self._batch_number = 0
        self._requested = 0
        self._completed = 0
        self._fallbacks: dict[str, int] = {}

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "requested": self._requested,
            "completed": self._completed,
            "fallback": sum(self._fallbacks.values()),
            "fallback_reasons": dict(sorted(self._fallbacks.items())),
            "batches": self._batch_number,
            "max_batch_size": self.max_batch_size,
        }

    async def evaluate(
        self, requests: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        if not requests:
            raise ValueError("selection Judge batch must not be empty")
        self._requested += len(requests)
        results: list[dict[str, Any]] = []
        for offset in range(0, len(requests), self.max_batch_size):
            batch = requests[offset : offset + self.max_batch_size]
            batch_results = await self._evaluate_batch(batch)
            results.extend(batch_results)
            for result in batch_results:
                if result.get("fallback"):
                    reason = str(result.get("fallback_reason") or "unknown")
                    self._fallbacks[reason] = self._fallbacks.get(reason, 0) + 1
                else:
                    self._completed += 1
        return results

    async def _evaluate_batch(
        self, requests: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        rows = tuple(
            _rubric_row(
                index,
                request["question"],
                request["response"],
                request["rubric"],
            )
            for index, request in enumerate(requests)
        )
        submission_id = f"{self.submission_id}:batch-{self._batch_number}"
        self._batch_number += 1
        try:
            submitted = await self.client.submit(
                submission_id=submission_id,
                input_jsonl=encode_input_jsonl(rows),
                job_metadata=self.job_metadata,
            )
            result = await self.client.wait(submitted.job_id)
            joined = join_results(rows, result.rows)
        except JobRejected:
            raise
        except Exception as error:
            return [
                _selection_fallback("batch_failed", str(error))
                for _ in rows
            ]

        projected: list[dict[str, Any]] = []
        for row, output in zip(rows, joined, strict=True):
            if output.status == "error":
                assert output.error is not None
                projected.append(
                    _selection_fallback(output.error.code, output.error.message)
                )
                continue
            try:
                assert output.result is not None
                value = project_process_evaluation(
                    projection=row.metadata["projection"],
                    result=output.result,
                )
                projected.append({**value, "fallback": False})
            except (KeyError, TypeError, ValueError, RubricJudgeUnavailable) as error:
                projected.append(_selection_fallback("invalid_score", str(error)))
        return projected

    async def close(self) -> None:
        """Close the dispatcher after all submitted batches have completed."""
        return None


def _selection_fallback(reason: str, message: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "scores_by_dimension": {},
        "projected_score": 0.0,
        "fallback": True,
        "fallback_reason": reason,
        "fallback_message": message,
    }


def _rubric_row(
    index: int, question: str, response: str, process_rubric: str
) -> RubricInputRow:
    declaration = parse_process_rubric(process_rubric, required=True)
    assert declaration is not None
    rubric = Rubric.from_dict(
        {
            "template": declaration["template"],
            "required_variables": declaration["required_variables"],
            "output_schema": declaration["output_schema"],
        }
    )
    return RubricInputRow(
        f"reward-row-{index:08d}",
        rubric,
        {"question": question, "response": response},
        {"row_index": index, "projection": declaration["projection"]},
    )


def project_process_score(*, projection: object, result: dict[str, Any]) -> float:
    return float(project_process_evaluation(projection=projection, result=result)["projected_score"])


def project_process_evaluation(
    *, projection: object, result: dict[str, Any]
) -> dict[str, Any]:
    scores = result.get("scores_by_dimension")
    if not isinstance(scores, dict):
        raise RubricJudgeUnavailable("Judge result lacks scores_by_dimension")
    if not isinstance(projection, dict) or set(projection) != {"dimensions"}:
        raise RubricJudgeUnavailable("rubric projection is missing")
    dimensions = projection.get("dimensions")
    if not isinstance(dimensions, list):
        raise RubricJudgeUnavailable("rubric projection dimensions are invalid")
    by_id = {
        str(item.get("id")): item
        for item in dimensions
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if set(scores) != set(by_id):
        raise RubricJudgeUnavailable("Judge dimension set does not match rubric")
    for name, value in scores.items():
        levels = by_id[name].get("score_levels")
        allowed = {
            float(level["value"])
            for level in levels
            if isinstance(level, dict) and type(level.get("value")) in {int, float}
        } if isinstance(levels, list) else set()
        if type(value) not in {int, float} or float(value) not in allowed:
            raise RubricJudgeUnavailable("Judge returned an undeclared score level")
    score = sum(float(by_id[name]["weight"]) * float(scores[name]) for name in scores)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RubricJudgeUnavailable("projected process score is out of bounds")
    return {
        "status": "completed",
        "scores_by_dimension": dict(scores),
        "projected_score": score,
    }
