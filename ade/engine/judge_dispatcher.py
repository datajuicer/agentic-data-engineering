"""Shared Engine Judge capability dispatcher for RFT and SFT adapters.

The implementation is kept behind this task-neutral Engine boundary.  Task
artifacts choose when and why to request a score; this module owns batching,
round barriers, identity, and terminal row handling.
"""

from ade.engine.judge_dispatcher_impl import (
    JudgeCircuitBreaker,
    RoundRubricDispatcher,
    SelectionJudgeBatchDispatcher,
    project_process_evaluation,
    RubricRowUnavailable,
    RubricJudgeUnavailable,
    RewardCommandFailed,
)

__all__ = [
    "JudgeCircuitBreaker",
    "RoundRubricDispatcher",
    "SelectionJudgeBatchDispatcher",
    "project_process_evaluation",
    "RubricRowUnavailable",
    "RubricJudgeUnavailable",
    "RewardCommandFailed",
]
