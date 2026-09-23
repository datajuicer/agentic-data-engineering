"""Evaluation Task Plugin."""

from ade.core.engine import EvaluateCommand
from ade.tasks.contracts import RankingSpec
from ade.tasks.plugin import TaskPlugin

plugin = TaskPlugin(
    "evaluation",
    EvaluateCommand,
    None,
    RankingSpec("offline.score"),
)
