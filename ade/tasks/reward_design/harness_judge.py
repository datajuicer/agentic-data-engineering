"""Harness-owned bounded validation submission for reward artifacts."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Sequence

from ade.engine.judge_dispatcher_impl import RewardCommandFailed
from ade.local_rubric_judge.client import LocalRubricJudgeClient
from ade.rubric_jobs import (
    Rubric,
    RubricInputRow,
    canonical_digest,
    encode_input_jsonl,
    join_results,
)
from ade.tasks.reward_design.process_evidence import (
    PROCESS_RUBRIC,
    available_process_evidence,
)


class RewardHarnessJudge:
    def __init__(self, client: LocalRubricJudgeClient) -> None:
        self.client = client

    def validate_sync(self, **kwargs: Any) -> dict[str, Any]:
        """Controller-facing boundary; TrialLifecycle is intentionally synchronous."""
        return asyncio.run(self.validate(**kwargs))

    async def validate(
        self,
        *,
        run_id: str,
        coordinator_id: str,
        plan_id: str,
        trial_id: str,
        artifact_digest: str,
        records: Sequence[dict[str, str]],
    ) -> dict[str, Any]:
        scope = {
            "run_id": run_id,
            "coordinator_id": coordinator_id,
            "plan_id": plan_id,
            "trial_id": trial_id,
        }
        subject_ref = f"{run_id}/{coordinator_id}/{plan_id}/{trial_id}"
        declaration = _declaration()
        rubric = Rubric.from_dict(
            {
                "template": declaration["template"],
                "required_variables": declaration["required_variables"],
                "output_schema": declaration["output_schema"],
            }
        )
        rows = tuple(
            RubricInputRow(
                str(record["record_id"]),
                rubric,
                {
                    "question": str(record["question"]),
                    "response": str(record["response"]),
                },
                {
                    "scope": scope,
                    "subject_ref": subject_ref,
                    "row_index": index,
                },
            )
            for index, record in enumerate(records)
        )
        submitted = await self.client.submit(
            submission_id=f"{subject_ref}:harness:{artifact_digest}",
            input_jsonl=encode_input_jsonl(rows),
            job_metadata={
                "scope": scope,
                "subject_ref": subject_ref,
                "purpose": "reward_artifact_validation",
                "artifact_digest": artifact_digest,
            },
        )
        result = await self.client.wait(submitted.job_id)
        if result.status.state in {"failed", "cancelled"}:
            raise RewardCommandFailed(
                f"Harness Local Rubric Judge job ended {result.status.state}"
            )
        joined = join_results(rows, result.rows)
        evidence = []
        for output in joined:
            if output.status == "error":
                assert output.error is not None
                evidence.append(
                    {
                        "status": "unavailable",
                        "reason": f"{output.error.code}: {output.error.message}",
                    }
                )
            else:
                assert output.result is not None
                evidence.append(
                    available_process_evidence(
                        {"status": "completed", **output.result}
                    )
                )
        return {
            "job_id": result.status.job_id,
            "state": result.status.state,
            "rubric_digest": canonical_digest(rubric.template),
            "output_schema_digest": canonical_digest(rubric.output_schema),
            "projection_digest": canonical_digest(declaration["projection"]),
            "evidence": evidence,
            "usage": result.usage.to_dict(),
        }


def _declaration() -> dict[str, Any]:
    return json.loads(PROCESS_RUBRIC)
