"""Actions selected by the deterministic Planner."""

from dataclasses import dataclass
from enum import StrEnum


class ActionKind(StrEnum):
    CALL_AGENT = "call_agent"
    SUBMIT_ENGINE = "submit_engine"
    COLLECT_ENGINE = "collect_engine"
    REVIEW_BATCH = "review_batch"
    COMPLETE_RUN = "complete_run"


@dataclass(frozen=True)
class Action:
    action_id: str
    run_id: str
    kind: ActionKind
    subject_id: str
    basis_revision: int
