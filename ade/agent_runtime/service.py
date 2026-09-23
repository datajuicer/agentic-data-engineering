"""Single application entry for typed Agent Calls."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from ade.agent_runtime.context import RoleContextPackager
from ade.agent_runtime.delivery import DeliveryGate
from ade.agent_runtime.input_package import AgentInputPackage
from ade.agent_runtime.runtime import AcceptedCall, AgentRuntime, RejectedCall
from ade.core.agent import (
    AgentCall,
    AgentCallScope,
    AgentRole,
    AgentSession,
    AgentSessionStatus,
    AttemptKind,
    validate_agent_target,
)
from ade.core.run import ActiveAgentCallRef
from ade.core.scope import agent_action_id
from ade.core.validation import DeliveryViolation, ValidationReport
from ade.tasks.registry import TaskRegistry
from ade.tasks.contracts import AgentContextReference
from ade.memory.usage import RunUsageLedger


class AgentCallService:
    def __init__(
        self,
        *,
        runtime: AgentRuntime,
        contexts: RoleContextPackager,
        tasks: TaskRegistry,
        max_retries: int,
        max_reflections: int = 2,
        max_delivery_bytes: int = 1_000_000,
        heartbeat_timeout_seconds: float = 900.0,
        execution_mode: str = "inline",
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if max_delivery_bytes < 1:
            raise ValueError("max_delivery_bytes must be positive")
        if heartbeat_timeout_seconds <= 0:
            raise ValueError("heartbeat_timeout_seconds must be positive")
        if execution_mode not in {"inline", "external"}:
            raise ValueError("Agent execution_mode must be inline or external")
        self.runtime = runtime
        self.contexts = contexts
        self.tasks = tasks
        self.max_retries = max_retries
        if max_reflections < 0:
            raise ValueError("max_reflections must be non-negative")
        self.max_reflections = max_reflections
        self.max_delivery_bytes = max_delivery_bytes
        self.heartbeat_timeout_seconds = float(heartbeat_timeout_seconds)
        self.execution_mode = execution_mode

    def is_expired(
        self,
        active: ActiveAgentCallRef,
        *,
        now: float | None = None,
    ) -> bool:
        observed = time.time() if now is None else float(now)
        state = self.contexts.repository.load(active.run_id)
        role = AgentRole(active.role)
        contract = self.tasks.get(state.task.task_id).role_contract(role)
        call = self._active_call(active, contract.skill_id)
        kind = AttemptKind.INITIAL if active.retry_index == 0 else AttemptKind.RETRY
        attempt = self.runtime.workspaces.attempt_dir(
            call, active.retry_index, kind
        )
        persisted = self.runtime.workspaces.load_attempt_heartbeat(attempt)
        last_heartbeat = max(
            active.last_heartbeat_at,
            persisted if persisted is not None else active.last_heartbeat_at,
        )
        return observed - last_heartbeat >= self.heartbeat_timeout_seconds

    @staticmethod
    def _active_call(active: ActiveAgentCallRef, skill_id: str) -> AgentCall:
        role = AgentRole(active.role)
        scope = {
            AgentRole.COORDINATOR: AgentCallScope.COORDINATOR,
            AgentRole.ARTIFACT_BUILDER: AgentCallScope.TRIAL,
            AgentRole.ANALYZER: AgentCallScope.TRIAL,
            AgentRole.PLAN_SUMMARIZER: AgentCallScope.PLAN,
            AgentRole.RUN_SUMMARIZER: AgentCallScope.RUN,
        }[role]
        return AgentCall(
            call_id=active.call_id,
            run_id=active.run_id,
            role=role,
            skill_id=skill_id,
            subject_id=active.subject_id,
            basis_revision=active.basis_revision,
            max_retries=active.max_retries,
            target_subject_ref=active.target_subject_ref,
            session_id=active.session_id,
            scope=scope,
            coordinator_id=active.coordinator_id,
            plan_id=active.plan_id,
            trial_id=active.trial_id,
            reflection_index=active.reflection_index,
        )

    def expire_active(self, active: ActiveAgentCallRef) -> RejectedCall:
        if not self.is_expired(active):
            raise ValueError("active Agent Attempt has not reached heartbeat timeout")
        state = self.contexts.repository.load(active.run_id)
        role = AgentRole(active.role)
        contract = self.tasks.get(state.task.task_id).role_contract(role)
        scope = {
            AgentRole.COORDINATOR: AgentCallScope.COORDINATOR,
            AgentRole.ARTIFACT_BUILDER: AgentCallScope.TRIAL,
            AgentRole.ANALYZER: AgentCallScope.TRIAL,
            AgentRole.PLAN_SUMMARIZER: AgentCallScope.PLAN,
            AgentRole.RUN_SUMMARIZER: AgentCallScope.RUN,
        }[role]
        call = AgentCall(
            call_id=active.call_id,
            run_id=active.run_id,
            role=role,
            skill_id=contract.skill_id,
            subject_id=active.subject_id,
            basis_revision=active.basis_revision,
            max_retries=active.max_retries,
            target_subject_ref=active.target_subject_ref,
            session_id=active.session_id,
            scope=scope,
            coordinator_id=active.coordinator_id,
            plan_id=active.plan_id,
            trial_id=active.trial_id,
            reflection_index=active.reflection_index,
        )
        kind = AttemptKind.INITIAL if active.retry_index == 0 else AttemptKind.RETRY
        attempt = self.runtime.workspaces.attempt_contract(
            call, active.retry_index, kind
        )
        workspace = self.runtime.workspaces.attempt_dir(
            call, active.retry_index, kind
        )
        report = ValidationReport(
            (
                DeliveryViolation(
                    "heartbeat_timeout",
                    "Agent Attempt produced no heartbeat within "
                    f"{self.heartbeat_timeout_seconds:g} seconds",
                    None,
                    repairable=True,
                ),
            )
        )
        self.runtime.workspaces.record_attempt_result(workspace, report)
        self.runtime.workspaces.record_receipt(call, attempt, report)
        session = _resolve_session(
            state,
            role=role,
            subject_id=active.subject_id,
            coordinator_id=active.coordinator_id,
            plan_id=active.plan_id,
            trial_id=active.trial_id,
        )
        return RejectedCall(
            call=call,
            attempt=attempt,
            workspace=workspace,
            validation=report,
            output=None,
            session=self.runtime.workspaces.load_session(session),
        )

    def prepare(
        self,
        *,
        run_id: str,
        role: AgentRole,
        subject_id: str,
        basis_revision: int,
        action_id: str,
        target_subject_ref: str,
        coordinator_id: str | None = None,
        plan_id: str | None = None,
        action_fields: Mapping[str, object] | None = None,
    ) -> ActiveAgentCallRef:
        package, _contract, session, call = self._binding(
            run_id=run_id,
            role=role,
            subject_id=subject_id,
            basis_revision=basis_revision,
            action_id=action_id,
            target_subject_ref=target_subject_ref,
            coordinator_id=coordinator_id,
            plan_id=plan_id,
            action_fields=action_fields,
        )
        session = self.runtime.workspaces.ensure_session(session, call)
        call_record = self.runtime.workspaces.call_dir(call) / "call.json"
        if not call_record.is_file():
            self.runtime.workspaces.create_attempt(
                call,
                0,
                AttemptKind.INITIAL,
                package.workspace_inputs(),
                package.references,
            )
        else:
            attempt_path = self.runtime.workspaces.attempt_dir(
                call, 0, AttemptKind.INITIAL
            )
            if not attempt_path.is_dir():
                raise ValueError("prepared Agent Call has no initial attempt")
        attempt = self.runtime.workspaces.attempt_contract(
            call,
            0,
            AttemptKind.INITIAL,
        )
        state = self.contexts.repository.load(run_id)
        memory_manifest = dict(package.files)["manifest.json"]
        now = time.time()
        plan_memory_basis = None
        if call.trial_id is not None:
            trial = next(
                item for item in state.trials
                if item.coordinator_id == call.coordinator_id
                and item.plan_id == call.plan_id
                and item.trial_id == call.trial_id
            )
            plan_memory_basis = trial.plan_memory_basis
        elif call.plan_id is not None:
            plan = next(
                item for item in state.plans
                if item.coordinator_id == call.coordinator_id
                and item.plan_id == call.plan_id
            )
            plan_memory_basis = plan.plan_memory_head
        return ActiveAgentCallRef(
            run_id=run_id,
            session_id=session.session_id,
            call_id=call.call_id,
            attempt_id=attempt.attempt_id,
            role=role.value,
            subject_id=subject_id,
            basis_revision=basis_revision,
            action_id=action_id,
            target_subject_ref=call.target_subject_ref,
            action_fields={
                **dict(action_fields or {}),
                "owner_scope": _target_scope(run_id, call.owner_subject_ref),
                "owner_subject_ref": call.owner_subject_ref,
                "target_scope": _target_scope(run_id, call.target_subject_ref),
                "target_subject_ref": call.target_subject_ref,
            },
            coordinator_id=call.coordinator_id,
            plan_id=call.plan_id,
            trial_id=call.trial_id,
            retry_index=0,
            max_retries=call.max_retries,
            status="submitted",
            submitted_at=now,
            last_heartbeat_at=now,
            resume_handle=session.resume_handle,
            lease_generation=0,
            memory_view_id=(
                "memory-view-" + hashlib.sha256(memory_manifest).hexdigest()[:24]
            ),
            run_memory_basis=state.memory.run_head,
            plan_memory_basis=plan_memory_basis,
            max_reflections=(
                self.max_reflections
                if role is AgentRole.ARTIFACT_BUILDER
                else 0
            ),
        )

    def execute_active(
        self,
        active: ActiveAgentCallRef,
    ) -> AcceptedCall | RejectedCall | None:
        return self._execute_active(
            active,
            dispatch=self.execution_mode == "inline",
        )

    def dispatch_active(
        self,
        active: ActiveAgentCallRef,
    ) -> AcceptedCall | RejectedCall:
        """Execute one committed Attempt from the independent Agent worker."""
        result = self._execute_active(active, dispatch=True)
        assert result is not None
        self._record_usage(active, result.workspace)
        return result

    def revalidate_active(
        self,
        active: ActiveAgentCallRef,
    ) -> AcceptedCall | RejectedCall:
        """Revalidate persisted output without dispatching the Agent backend."""
        result = self._execute_active(active, dispatch=False, revalidate=True)
        assert result is not None
        return result

    def has_terminal(self, active: ActiveAgentCallRef) -> bool:
        state = self.contexts.repository.load(active.run_id)
        role = AgentRole(active.role)
        skill_id = self.tasks.get(state.task.task_id).role_contract(role).skill_id
        call = self._active_call(active, skill_id)
        path = self.runtime.workspaces.call_dir(call) / "receipt.json"
        if not path.is_file():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("persisted Agent receipt is invalid") from error
        return (
            isinstance(payload, dict)
            and payload.get("call_id") == active.call_id
            and payload.get("attempt_id") == active.attempt_id
        )

    def interrupt(
        self,
        active: ActiveAgentCallRef,
        *,
        grace_seconds: float = 5.0,
    ) -> dict[str, object]:
        """Stop one exact Agent Attempt and retain an operational audit."""

        if grace_seconds <= 0:
            raise ValueError("Agent interrupt grace_seconds must be positive")

        state = self.contexts.repository.load(active.run_id)
        role = AgentRole(active.role)
        skill_id = self.tasks.get(state.task.task_id).role_contract(role).skill_id
        call = self._active_call(active, skill_id)
        kind = AttemptKind.INITIAL if active.retry_index == 0 else AttemptKind.RETRY
        attempt = self.runtime.workspaces.attempt_dir(
            call, active.retry_index, kind
        )
        process_path = attempt / "scratch" / "backend-process.json"
        pid = None
        terminated = False
        unresolved = None
        if process_path.is_file():
            try:
                process = json.loads(process_path.read_text(encoding="utf-8"))
                if process.get("status") == "running":
                    pid = int(process["pid"])
                    try:
                        pgid = os.getpgid(pid)
                        os.killpg(pgid, signal.SIGTERM)
                        deadline = time.monotonic() + grace_seconds
                        while time.monotonic() < deadline:
                            try:
                                os.kill(pid, 0)
                            except ProcessLookupError:
                                terminated = True
                                break
                            time.sleep(0.05)
                        if not terminated:
                            os.killpg(pgid, signal.SIGKILL)
                            try:
                                os.kill(pid, 0)
                            except ProcessLookupError:
                                terminated = True
                            else:
                                unresolved = "Agent backend process remained live"
                    except ProcessLookupError:
                        terminated = True
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                unresolved = f"{type(error).__name__}: {error}"
        audit = {
            "schema_version": "ade.agent_attempt_cancellation.v1",
            "run_id": active.run_id,
            "coordinator_id": active.coordinator_id,
            "plan_id": active.plan_id,
            "trial_id": active.trial_id,
            "call_id": active.call_id,
            "attempt_id": active.attempt_id,
            "target_subject_ref": active.target_subject_ref,
            "pid": pid,
            "terminated": terminated,
            "recorded_at": time.time(),
            **({"unresolved": unresolved} if unresolved is not None else {}),
        }
        cancellation_path = attempt / "scratch" / "operator-cancellation.json"
        cancellation_path.parent.mkdir(parents=True, exist_ok=True)
        cancellation_path.write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return audit

    def record_dispatch_failure(
        self,
        active: ActiveAgentCallRef,
        error: Exception,
    ) -> RejectedCall:
        state = self.contexts.repository.load(active.run_id)
        role = AgentRole(active.role)
        contract = self.tasks.get(state.task.task_id).role_contract(role)
        call = self._active_call(active, contract.skill_id)
        kind = AttemptKind.INITIAL if active.retry_index == 0 else AttemptKind.RETRY
        attempt = self.runtime.workspaces.attempt_contract(
            call, active.retry_index, kind
        )
        workspace = self.runtime.workspaces.attempt_dir(
            call, active.retry_index, kind
        )
        report = ValidationReport(
            (
                DeliveryViolation(
                    "agent_backend_failed",
                    f"{type(error).__name__}: {error}",
                    None,
                    repairable=True,
                ),
            )
        )
        self.runtime.workspaces.record_worker_error(workspace, error)
        self.runtime.workspaces.record_attempt_result(workspace, report)
        self.runtime.workspaces.record_receipt(call, attempt, report)
        self._record_usage(active, workspace, status="failed")
        session = _resolve_session(
            state,
            role=role,
            subject_id=active.subject_id,
            coordinator_id=active.coordinator_id,
            plan_id=active.plan_id,
            trial_id=active.trial_id,
        )
        return RejectedCall(
            call=call,
            attempt=attempt,
            workspace=workspace,
            validation=report,
            output=None,
            session=self.runtime.workspaces.load_session(session),
        )

    def _record_usage(
        self,
        active: ActiveAgentCallRef,
        workspace,
        *,
        status: str = "complete",
    ) -> None:
        layout = getattr(self.contexts.repository, "layout", None)
        if layout is None:
            return
        prompt = completion = cached = reasoning = requests = 0
        events = workspace / "scratch" / "codex-events.jsonl"
        if events.is_file():
            for line in events.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(value, dict):
                    continue
                usage = value.get("usage")
                if value.get("type") != "turn.completed" or not isinstance(usage, dict):
                    continue
                prompt += int(usage.get("input_tokens") or 0)
                completion += int(usage.get("output_tokens") or 0)
                cached += int(usage.get("cached_input_tokens") or 0)
                reasoning += int(usage.get("reasoning_output_tokens") or 0)
                requests += 1
        session_record = workspace / "scratch" / "codex-session.json"
        try:
            session_value = json.loads(session_record.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            session_value = {}
        if not isinstance(session_value, dict):
            session_value = {}
        if requests == 0 or session_value.get("mode") == "resume":
            usage = session_value.get("usage")
            if isinstance(usage, dict):
                input_tokens = usage.get("input_tokens")
                output_tokens = usage.get("output_tokens")
                cached_tokens = usage.get("cached_input_tokens")
                reasoning_tokens = usage.get("reasoning_output_tokens")
                if all(
                    type(value) is int and value >= 0
                    for value in (
                        input_tokens,
                        output_tokens,
                        cached_tokens,
                        reasoning_tokens,
                    )
                ):
                    prompt = input_tokens
                    completion = output_tokens
                    cached = cached_tokens
                    reasoning = reasoning_tokens
                    requests = 1
        ledger = RunUsageLedger(layout.run_dir(active.run_id))
        usage_available = bool(prompt or completion)
        transition_intent = {
            AgentRole.COORDINATOR.value: "planning",
            AgentRole.ARTIFACT_BUILDER.value: "artifact_construction",
            AgentRole.ANALYZER.value: {
                "review_design": "analysis_review_design",
                "synthesis": "analysis_synthesis",
            }.get(str(active.action_fields.get("stage")), "trial_analysis"),
            AgentRole.PLAN_SUMMARIZER.value: "plan_summary",
            AgentRole.RUN_SUMMARIZER.value: "run_summary",
        }[active.role]
        scope = {
            "run_id": active.run_id,
            "coordinator_id": active.coordinator_id,
            "plan_id": active.plan_id,
            "trial_id": active.trial_id,
            "session_id": active.session_id,
            "call_id": active.call_id,
            "attempt_id": active.attempt_id,
            "basis_revision": active.basis_revision,
            "transition_intent": transition_intent,
        }
        ledger.append_or_enrich(
            {
                "event_id": active.attempt_id,
                "category": "agent_execution",
                "component": "agent",
                "provider": "codex",
                "role": active.role,
                "status": status,
                **scope,
                "usage_status": "complete" if usage_available else "unavailable",
                "prompt_tokens": prompt if usage_available else None,
                "completion_tokens": completion if usage_available else None,
                "total_tokens": prompt + completion if usage_available else None,
                "cached_tokens": cached if usage_available else None,
                "reasoning_tokens": reasoning if usage_available else None,
                "attempts": requests or 1,
                "retries": 0,
                "requests": requests or 1,
            }
        )
        for manifest_path in sorted(
            (workspace / "scratch" / "reviews").glob("*/*/manifest.json")
        ):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            usage = manifest.get("usage") if isinstance(manifest, dict) else None
            review_id = manifest.get("review_id") if isinstance(manifest, dict) else None
            if not isinstance(usage, dict) or not isinstance(review_id, str):
                continue
            ledger.append(
                {
                    "event_id": f"review-labor-job:{review_id}",
                    "category": "analyzer_review",
                    "component": "review_labor",
                    "provider": "review_labor",
                    "role": active.role,
                    **scope,
                    "usage_status": str(usage.get("usage_status") or "unavailable"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "cached_tokens": usage.get("cached_tokens"),
                    "reasoning_tokens": usage.get("reasoning_tokens"),
                    "attempts": int(usage.get("attempts") or usage.get("requests") or 0),
                    "retries": int(usage.get("retries") or 0),
                    "requests": int(usage.get("requests") or 0),
                }
            )

    def _execute_active(
        self,
        active: ActiveAgentCallRef,
        *,
        dispatch: bool,
        revalidate: bool = False,
    ) -> AcceptedCall | RejectedCall | None:
        state = self.contexts.repository.load(active.run_id)
        role = AgentRole(active.role)
        contract = self.tasks.get(state.task.task_id).role_contract(role)
        scope = {
            AgentRole.COORDINATOR: AgentCallScope.COORDINATOR,
            AgentRole.ARTIFACT_BUILDER: AgentCallScope.TRIAL,
            AgentRole.ANALYZER: AgentCallScope.TRIAL,
            AgentRole.PLAN_SUMMARIZER: AgentCallScope.PLAN,
            AgentRole.RUN_SUMMARIZER: AgentCallScope.RUN,
        }[role]
        coordinator_id = active.coordinator_id
        plan_id = active.plan_id
        trial_id = active.trial_id
        call = AgentCall(
            call_id=active.call_id,
            run_id=active.run_id,
            role=role,
            skill_id=contract.skill_id,
            subject_id=active.subject_id,
            basis_revision=active.basis_revision,
            max_retries=active.max_retries,
            target_subject_ref=active.target_subject_ref,
            session_id=active.session_id,
            scope=scope,
            coordinator_id=coordinator_id,
            plan_id=plan_id,
            trial_id=trial_id,
            reflection_index=active.reflection_index,
        )
        attempt = self.runtime.workspaces.attempt_dir(
            call, 0, AttemptKind.INITIAL
        ) / "input"
        files, references = _persisted_input_bindings(attempt)
        package = AgentInputPackage(
            schema_version="1",
            task_id=state.task.task_id,
            skill_id=contract.skill_id,
            role=role,
            run_id=active.run_id,
            subject_id=active.subject_id,
            basis_revision=active.basis_revision,
            files=files,
            references=references,
        )
        action = json.loads(dict(files)["action.json"])
        output_paths = contract.output_paths_for(action)
        gate = DeliveryGate(
            allowed_paths=set(output_paths),
            required_paths=set(output_paths),
            max_delivery_bytes=self.max_delivery_bytes,
        )
        if revalidate:
            result = self.runtime.revalidate(call, package, gate, contract)
        elif dispatch:
            result = self.runtime.recover(call, package, gate, contract)
        else:
            result = self.runtime.inspect(call, package, gate, contract)
        if result is None:
            return None
        session = _resolve_session(
            state,
            role=role,
            subject_id=active.subject_id,
            coordinator_id=coordinator_id,
            plan_id=plan_id,
            trial_id=trial_id,
        )
        return replace(
            result,
            session=self.runtime.workspaces.load_session(session),
        )

    def prepare_retry(
        self,
        active: ActiveAgentCallRef,
        previous: AcceptedCall | RejectedCall,
        report: ValidationReport,
    ) -> ActiveAgentCallRef:
        if previous.call.call_id != active.call_id:
            raise ValueError("Agent retry belongs to another active Call")
        if previous.attempt.attempt_id != active.attempt_id:
            raise ValueError("Agent retry does not target the active Attempt")
        state = self.contexts.repository.load(active.run_id)
        role = AgentRole(active.role)
        contract = self.tasks.get(state.task.task_id).role_contract(role)
        scope = {
            AgentRole.COORDINATOR: AgentCallScope.COORDINATOR,
            AgentRole.ARTIFACT_BUILDER: AgentCallScope.TRIAL,
            AgentRole.ANALYZER: AgentCallScope.TRIAL,
            AgentRole.PLAN_SUMMARIZER: AgentCallScope.PLAN,
            AgentRole.RUN_SUMMARIZER: AgentCallScope.RUN,
        }[role]
        call = AgentCall(
            call_id=active.call_id,
            run_id=active.run_id,
            role=role,
            skill_id=contract.skill_id,
            subject_id=active.subject_id,
            basis_revision=active.basis_revision,
            max_retries=active.max_retries,
            target_subject_ref=active.target_subject_ref,
            session_id=active.session_id,
            scope=scope,
            coordinator_id=active.coordinator_id,
            plan_id=active.plan_id,
            trial_id=active.trial_id,
            reflection_index=active.reflection_index,
        )
        initial = self.runtime.workspaces.attempt_dir(
            call, 0, AttemptKind.INITIAL
        ) / "input"
        files, references = _persisted_input_bindings(initial)
        package = AgentInputPackage(
            schema_version="1",
            task_id=state.task.task_id,
            skill_id=contract.skill_id,
            role=role,
            run_id=active.run_id,
            subject_id=active.subject_id,
            basis_revision=active.basis_revision,
            files=files,
            references=references,
        )
        attempt, _path = self.runtime.prepare_retry_attempt(
            call,
            package,
            contract,
            previous,
            report,
        )
        now = time.time()
        return replace(
            active,
            attempt_id=attempt.attempt_id,
            retry_index=attempt.number,
            status="submitted",
            submitted_at=now,
            last_heartbeat_at=now,
            lease_generation=active.lease_generation + 1,
            **(
                {
                    "round_status": "agent_running",
                    "current_delivery_ref": None,
                    "current_realization_ref": None,
                }
                if role is AgentRole.ARTIFACT_BUILDER
                else {}
            ),
        )

    def prepare_reflection(
        self,
        active: ActiveAgentCallRef,
        reflection_files: Mapping[str, tuple[bytes, str]],
    ) -> ActiveAgentCallRef:
        if AgentRole(active.role) is not AgentRole.ARTIFACT_BUILDER:
            raise ValueError("reflection requires an Artifact Builder Call")
        if active.round_status != "realization_ready":
            raise ValueError("Builder reflection requires a completed realization")
        if active.reflection_index >= active.max_reflections:
            raise ValueError("Builder reflection budget is exhausted")
        required = {
            "reflection/round.json",
            "reflection/realization-report.json",
        }
        if not required.issubset(reflection_files):
            raise ValueError("Builder reflection input is incomplete")
        state = self.contexts.repository.load(active.run_id)
        contract = self.tasks.get(state.task.task_id).role_contract(
            AgentRole.ARTIFACT_BUILDER
        )
        current_call = self._active_call(active, contract.skill_id)
        proposal_input = self.runtime.workspaces.attempt_dir(
            replace(current_call, reflection_index=0),
            0,
            AttemptKind.INITIAL,
        ) / "input"
        files, references = _persisted_input_bindings(proposal_input)
        package = AgentInputPackage(
            schema_version="1",
            task_id=state.task.task_id,
            skill_id=contract.skill_id,
            role=AgentRole.ARTIFACT_BUILDER,
            run_id=active.run_id,
            subject_id=active.subject_id,
            basis_revision=active.basis_revision,
            files=files,
            references=references,
        ).with_materialized_overlay(reflection_files)
        reflected_call = replace(
            current_call,
            reflection_index=active.reflection_index + 1,
        )
        attempt_path = self.runtime.workspaces.create_attempt(
            reflected_call,
            0,
            AttemptKind.INITIAL,
            package.workspace_inputs(),
            package.references,
        )
        prior = self.runtime.workspaces.attempt_dir(
            current_call,
            active.retry_index,
            AttemptKind.INITIAL
            if active.retry_index == 0
            else AttemptKind.RETRY,
        )
        self.runtime._carry_attempt_audits(prior, attempt_path)
        attempt = self.runtime.workspaces.attempt_contract(
            reflected_call, 0, AttemptKind.INITIAL
        )
        now = time.time()
        return replace(
            active,
            attempt_id=attempt.attempt_id,
            reflection_index=reflected_call.reflection_index,
            retry_index=0,
            round_status="agent_running",
            current_delivery_ref=None,
            current_realization_ref=None,
            status="submitted",
            submitted_at=now,
            last_heartbeat_at=now,
            lease_generation=active.lease_generation + 1,
        )

    def _binding(
        self,
        *,
        run_id: str,
        role: AgentRole,
        subject_id: str,
        basis_revision: int,
        action_id: str,
        target_subject_ref: str,
        coordinator_id: str | None,
        plan_id: str | None,
        action_fields: Mapping[str, object] | None,
    ):
        state = self.contexts.repository.load(run_id)
        scope, resolved_coordinator_id, resolved_plan_id, trial_id = _resolve_scope(
            state,
            role,
            subject_id,
            coordinator_id=coordinator_id,
            plan_id=plan_id,
        )
        target_scope = _target_scope(run_id, target_subject_ref)
        owner_subject_ref = validate_agent_target(
            run_id=run_id,
            role=role,
            coordinator_id=resolved_coordinator_id,
            plan_id=resolved_plan_id,
            target_subject_ref=target_subject_ref,
        )
        if (
            role in {AgentRole.ARTIFACT_BUILDER, AgentRole.ANALYZER}
            and target_scope.get("trial_id") != subject_id
        ):
            raise ValueError(f"{role.value} Call target must be its Trial subject")
        if (
            role is AgentRole.COORDINATOR
            and dict(action_fields or {}).get("target_plan_id")
            != target_scope.get("plan_id")
        ):
            raise ValueError("Coordinator Call target must match target_plan_id")
        expected_action_id = agent_action_id(
            owner_subject_ref=owner_subject_ref,
            target_subject_ref=target_subject_ref,
            role_segment=_role_segment(role, action_fields),
            basis_revision=basis_revision,
        )
        if action_id != expected_action_id:
            raise ValueError(
                "Agent action_id does not encode its canonical Session owner and Call target"
            )
        owner_scope = _target_scope(run_id, owner_subject_ref)
        package = self.contexts.build(
            run_id=run_id,
            role=role,
            subject_id=subject_id,
            basis_revision=basis_revision,
            action_id=action_id,
            coordinator_id=coordinator_id,
            plan_id=plan_id,
            action_fields={
                **dict(action_fields or {}),
                "owner_scope": owner_scope,
                "owner_subject_ref": owner_subject_ref,
                "target_scope": target_scope,
                "target_subject_ref": target_subject_ref,
            },
        )
        contract = self.tasks.get(package.task_id).role_contract(role)
        session = _resolve_session(
            state,
            role=role,
            subject_id=subject_id,
            coordinator_id=resolved_coordinator_id,
            plan_id=resolved_plan_id,
            trial_id=trial_id,
        )
        call = AgentCall(
            call_id=_call_id(
                run_id=run_id,
                role=role,
                subject_id=subject_id,
                basis_revision=basis_revision,
                action_id=action_id,
                target_subject_ref=target_subject_ref,
                coordinator_id=resolved_coordinator_id,
                plan_id=resolved_plan_id,
            ),
            run_id=run_id,
            role=role,
            skill_id=contract.skill_id,
            subject_id=subject_id,
            basis_revision=basis_revision,
            max_retries=self.max_retries,
            target_subject_ref=target_subject_ref,
            session_id=session.session_id,
            scope=scope,
            coordinator_id=resolved_coordinator_id,
            plan_id=resolved_plan_id,
            trial_id=trial_id,
        )
        return package, contract, session, call


def _persisted_input_bindings(
    input_dir: Path,
) -> tuple[tuple[tuple[str, bytes], ...], tuple[AgentContextReference, ...]]:
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    declarations = {
        str(item["path"]): item
        for item in manifest.get("inputs", ())
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    references = tuple(
        AgentContextReference(
            path=path,
            source_ref=str(item.get("source_ref") or ""),
            sha256=str(item.get("sha256") or ""),
            size_bytes=int(item.get("size_bytes", -1)),
        )
        for path, item in sorted(declarations.items())
        if item.get("mode") == "durable_hardlink"
    )
    reference_paths = {item.path for item in references}
    files = tuple(
        sorted(
            (path.relative_to(input_dir).as_posix(), path.read_bytes())
            for path in input_dir.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.relative_to(input_dir).as_posix() not in reference_paths
        )
    )
    return files, references


def _call_id(
    *,
    run_id: str,
    role: AgentRole,
    subject_id: str,
    basis_revision: int,
    action_id: str,
    target_subject_ref: str,
    coordinator_id: str | None,
    plan_id: str | None,
) -> str:
    identity = "\0".join(
        (
            run_id,
            role.value,
            coordinator_id or "",
            plan_id or "",
            subject_id,
            str(basis_revision),
            action_id,
            target_subject_ref,
        )
    )
    return f"agent-{hashlib.sha256(identity.encode()).hexdigest()[:24]}"


def _role_segment(
    role: AgentRole,
    action_fields: Mapping[str, object] | None,
) -> str:
    if role is AgentRole.ARTIFACT_BUILDER:
        return "artifact-builder"
    if role is AgentRole.ANALYZER:
        stage = str(dict(action_fields or {}).get("stage", "")).replace("_", "-")
        if stage not in {"review-design", "synthesis"}:
            raise ValueError("Analyzer action requires a canonical stage")
        return f"analyzer-{stage}"
    return role.value.replace("_", "-")


def _target_scope(run_id: str, target_subject_ref: str) -> dict[str, str]:
    parts = target_subject_ref.split("/")
    if parts[0] != run_id or not 1 <= len(parts) <= 4 or any(not item for item in parts):
        raise ValueError("Agent Call target must be a full SubjectRef")
    return dict(
        zip(
            ("run_id", "coordinator_id", "plan_id", "trial_id"),
            parts,
            strict=False,
        )
    )


def _resolve_scope(
    state,
    role,
    subject_id,
    *,
    coordinator_id: str | None,
    plan_id: str | None,
):
    if role is AgentRole.COORDINATOR:
        if coordinator_id is not None or plan_id is not None:
            raise ValueError("Coordinator call cannot identify a Plan")
        if not any(item.coordinator_id == subject_id for item in state.coordinators):
            raise ValueError("unknown Coordinator subject")
        return AgentCallScope.COORDINATOR, subject_id, None, None
    if role is AgentRole.RUN_SUMMARIZER:
        if subject_id != state.run_id:
            raise ValueError(f"{role.value} must belong to the run")
        return AgentCallScope.RUN, None, None, None
    if role is AgentRole.PLAN_SUMMARIZER:
        if plan_id is not None:
            raise ValueError("Plan Summarizer plan_id comes from subject_id")
        if not any(
            plan.coordinator_id == coordinator_id and plan.plan_id == subject_id
            for plan in state.plans
        ):
            raise ValueError(f"unknown Plan subject: {subject_id}")
        return AgentCallScope.PLAN, coordinator_id, subject_id, None
    if role in {AgentRole.ARTIFACT_BUILDER, AgentRole.ANALYZER}:
        matches = [
            trial
            for trial in state.trials
            if trial.coordinator_id == coordinator_id
            and trial.plan_id == plan_id
            and trial.trial_id == subject_id
        ]
        if len(matches) != 1:
            raise ValueError(f"unknown Trial subject: {subject_id}")
        trial = matches[0]
        return (
            AgentCallScope.TRIAL,
            trial.coordinator_id,
            trial.plan_id,
            trial.trial_id,
        )
    raise ValueError(f"unsupported Agent role: {role}")


def _resolve_session(
    state,
    *,
    role: AgentRole,
    subject_id: str,
    coordinator_id: str | None,
    plan_id: str | None,
    trial_id: str | None,
) -> AgentSession:
    session_subject_id = subject_id
    session_trial_id = trial_id
    if role in {AgentRole.ARTIFACT_BUILDER, AgentRole.ANALYZER}:
        if plan_id is None:
            raise ValueError(f"{role.value} Session requires a Plan owner")
        session_subject_id = plan_id
        session_trial_id = None
    matches = [
        session
        for session in state.agent_sessions
        if session.role is role
        and session.subject_id == session_subject_id
        and session.coordinator_id == coordinator_id
        and session.plan_id == plan_id
        and session.trial_id == session_trial_id
    ]
    if len(matches) > 1:
        raise ValueError("logical Agent Session is ambiguous")
    if matches:
        if matches[0].status is not AgentSessionStatus.ACTIVE:
            raise ValueError("logical Agent Session is closed")
        return matches[0]
    role_id = role.value.replace("_", "-")
    scope_ids = tuple(
        value
        for value in (coordinator_id, plan_id, session_trial_id)
        if value is not None
    )
    if role is AgentRole.RUN_SUMMARIZER:
        scope_ids = (state.run_id,)
    return AgentSession(
        session_id="-".join((role_id, *scope_ids)),
        run_id=state.run_id,
        role=role,
        subject_id=session_subject_id,
        coordinator_id=coordinator_id,
        plan_id=plan_id,
        trial_id=session_trial_id,
    )
