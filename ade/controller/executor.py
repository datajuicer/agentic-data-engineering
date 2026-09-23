"""Action execution without state mutation."""

from __future__ import annotations

from typing import Protocol

from ade.core.actions import Action
from ade.core.outcomes import (
    AgentRetryHeldOutcome,
    CoordinatorCallSubmittedOutcome,
    PlanningDecisionOutcome,
    AgentRetrySubmittedOutcome,
    CoordinatorFailedOutcome,
)
from ade.core.run import ActiveAgentCallRef


class AgentPort(Protocol):
    def prepare(self, action: Action) -> CoordinatorCallSubmittedOutcome: ...
    def collect(
        self, active: ActiveAgentCallRef
    ) -> PlanningDecisionOutcome | AgentRetryHeldOutcome | AgentRetrySubmittedOutcome | CoordinatorFailedOutcome | None: ...


class Executor:
    def __init__(self, *, agent_port: AgentPort) -> None:
        self.agent_port = agent_port

    def prepare(self, action: Action) -> CoordinatorCallSubmittedOutcome:
        return self.agent_port.prepare(action)

    def collect(
        self, active: ActiveAgentCallRef
    ) -> PlanningDecisionOutcome | AgentRetryHeldOutcome | AgentRetrySubmittedOutcome | CoordinatorFailedOutcome | None:
        return self.agent_port.collect(active)
