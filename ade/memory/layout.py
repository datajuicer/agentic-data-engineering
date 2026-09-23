"""Canonical filesystem paths for one ADE Run."""

from __future__ import annotations

from pathlib import Path
import re

from ade.core.agent import AgentCall, AgentCallScope, AgentRole, AgentSession

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class RunLayout:
    def __init__(self, runs_root: str | Path) -> None:
        self.runs_root = Path(runs_root).resolve()

    def run_dir(self, run_id: str) -> Path:
        self.validate_id(run_id, "run_id")
        return self.runs_root / run_id

    def coordinator_dir(self, run_id: str, coordinator_id: str) -> Path:
        self.validate_id(coordinator_id, "coordinator_id")
        return self.run_dir(run_id) / "coordinators" / coordinator_id

    def plan_dir(self, run_id: str, coordinator_id: str, plan_id: str) -> Path:
        self.validate_id(plan_id, "plan_id")
        return self.coordinator_dir(run_id, coordinator_id) / "plans" / plan_id

    def trial_dir(
        self,
        run_id: str,
        coordinator_id: str,
        plan_id: str,
        trial_id: str,
    ) -> Path:
        self.validate_id(trial_id, "trial_id")
        return self.plan_dir(run_id, coordinator_id, plan_id) / "trials" / trial_id

    def trial_record_dir(
        self,
        run_id: str,
        coordinator_id: str,
        plan_id: str,
        trial_id: str,
    ) -> Path:
        return self.trial_dir(run_id, coordinator_id, plan_id, trial_id) / "record"

    def run_memory_version_dir(self, run_id: str, memory_id: str) -> Path:
        self.validate_id(memory_id, "Run Memory ID")
        return self.run_dir(run_id) / "memory" / "run" / "versions" / memory_id

    def plan_memory_version_dir(
        self,
        run_id: str,
        coordinator_id: str,
        plan_id: str,
        memory_id: str,
    ) -> Path:
        self.validate_id(memory_id, "Plan Memory ID")
        return (
            self.run_dir(run_id)
            / "memory"
            / "plans"
            / coordinator_id
            / plan_id
            / "versions"
            / memory_id
        )

    def revision_snapshot_path(self, run_id: str, revision: int) -> Path:
        if revision < 0:
            raise ValueError("revision must be non-negative")
        return (
            self.run_dir(run_id)
            / "state"
            / "revisions"
            / f"rev-{revision}"
            / "run.json"
        )

    def agent_call_dir(self, call: AgentCall) -> Path:
        self.validate_id(call.call_id, "call_id")
        owner_scope = call.scope
        owner_trial_id = call.trial_id
        if call.role in {
            AgentRole.ARTIFACT_BUILDER,
            AgentRole.ANALYZER,
            AgentRole.PLAN_SUMMARIZER,
        }:
            owner_scope = AgentCallScope.PLAN
            owner_trial_id = None
        elif call.role is AgentRole.RUN_SUMMARIZER:
            owner_scope = AgentCallScope.RUN
        owner = self._agent_owner(
            run_id=call.run_id,
            scope=owner_scope,
            coordinator_id=call.coordinator_id,
            plan_id=call.plan_id,
            trial_id=owner_trial_id,
        )
        role_dir = call.role.value.replace("_", "-")
        return owner / "agents" / role_dir / "calls" / call.call_id

    def agent_session_dir(self, session: AgentSession) -> Path:
        self.validate_id(session.session_id, "session_id")
        owner = self._agent_owner(
            run_id=session.run_id,
            scope=session.scope,
            coordinator_id=session.coordinator_id,
            plan_id=session.plan_id,
            trial_id=session.trial_id,
        )
        return owner / "agents" / session.role.value.replace("_", "-")

    def plan_snapshot_dir(
        self,
        run_id: str,
        coordinator_id: str,
        plan_id: str,
        revision: int,
    ) -> Path:
        return self.plan_dir(run_id, coordinator_id, plan_id) / "snapshots" / self._revision_label(revision)

    def run_snapshot_dir(self, run_id: str, revision: int) -> Path:
        return self.run_dir(run_id) / "snapshots" / "run" / self._revision_label(revision)

    def trial_snapshot_dir(
        self,
        run_id: str,
        coordinator_id: str,
        plan_id: str,
        trial_id: str,
        revision: int,
    ) -> Path:
        return (
            self.trial_dir(run_id, coordinator_id, plan_id, trial_id)
            / "snapshots"
            / self._revision_label(revision)
        )

    def command_dir(
        self,
        run_id: str,
        coordinator_id: str,
        plan_id: str,
        trial_id: str,
        command_id: str,
    ) -> Path:
        self.validate_id(command_id, "command_id")
        return (
            self.trial_dir(run_id, coordinator_id, plan_id, trial_id)
            / "engine"
            / "commands"
            / command_id
        )

    def review_command_dir(
        self,
        run_id: str,
        coordinator_id: str,
        plan_id: str,
        trial_id: str,
        command_id: str,
    ) -> Path:
        self.validate_id(command_id, "Review command_id")
        return (
            self.trial_dir(run_id, coordinator_id, plan_id, trial_id)
            / "review"
            / "commands"
            / command_id
        )

    def agent_call_uri(self, call: AgentCall) -> str:
        relative = self.agent_call_dir(call).relative_to(self.run_dir(call.run_id))
        return f"run://{call.run_id}/{relative.as_posix()}"

    def _agent_owner(
        self,
        *,
        run_id: str,
        scope: AgentCallScope,
        coordinator_id: str | None,
        plan_id: str | None,
        trial_id: str | None,
    ) -> Path:
        if scope is AgentCallScope.RUN:
            return self.run_dir(run_id)
        if scope is AgentCallScope.COORDINATOR:
            assert coordinator_id is not None
            return self.coordinator_dir(run_id, coordinator_id)
        if scope is AgentCallScope.PLAN:
            assert coordinator_id is not None and plan_id is not None
            return self.plan_dir(run_id, coordinator_id, plan_id)
        assert coordinator_id is not None and plan_id is not None and trial_id is not None
        return self.trial_dir(run_id, coordinator_id, plan_id, trial_id)

    @staticmethod
    def _revision_label(revision: int) -> str:
        if revision < 0:
            raise ValueError("snapshot revision must be non-negative")
        return f"{revision:03d}"

    @staticmethod
    def validate_id(value: str, label: str) -> None:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError(f"{label} contains unsafe characters")
