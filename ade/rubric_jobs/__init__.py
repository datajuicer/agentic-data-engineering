"""Provider-neutral ``ade.rubric_jobs.v1`` data contract."""

from ade.rubric_jobs.models import (
    JOB_STATES,
    TERMINAL_JOB_STATES,
    JobError,
    JobResult,
    JobStatus,
    Rubric,
    RubricInputRow,
    RubricOutputRow,
    TokenUsage,
)
from ade.rubric_jobs.validation import (
    PROTOCOL_VERSION,
    canonical_digest,
    encode_input_jsonl,
    encode_output_jsonl,
    join_results,
    parse_input_jsonl,
    parse_output_jsonl,
    render_template_once,
    submission_content_digest,
)

__all__ = [
    "JOB_STATES",
    "PROTOCOL_VERSION",
    "TERMINAL_JOB_STATES",
    "JobError",
    "JobResult",
    "JobStatus",
    "Rubric",
    "RubricInputRow",
    "RubricOutputRow",
    "TokenUsage",
    "canonical_digest",
    "encode_input_jsonl",
    "encode_output_jsonl",
    "join_results",
    "parse_input_jsonl",
    "parse_output_jsonl",
    "render_template_once",
    "submission_content_digest",
]
