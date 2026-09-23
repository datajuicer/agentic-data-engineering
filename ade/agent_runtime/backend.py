"""Codex process backend."""

from __future__ import annotations

import os
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any

from ade.agent_runtime.skills import ResolvedSkill


class CodexBackend:
    def __init__(
        self,
        command: tuple[str, ...] = ("codex", "exec"),
        model: str = "gpt-5.6-sol",
        reasoning_effort: str = "xhigh",
        sandbox_mode: str = "danger-full-access",
        approval_policy: str = "never",
    ) -> None:
        if not command:
            raise ValueError("Codex command is required")
        if not model:
            raise ValueError("Codex model is required")
        if not reasoning_effort:
            raise ValueError("Codex reasoning effort is required")
        if sandbox_mode != "danger-full-access":
            raise ValueError("ADE Codex sandbox_mode must be danger-full-access")
        if approval_policy != "never":
            raise ValueError("ADE Codex approval_policy must be never")
        self.command = command
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.sandbox_mode = sandbox_mode
        self.approval_policy = approval_policy

    def invoke(
        self,
        *,
        skill: ResolvedSkill,
        attempt: Path,
        feedback_paths: tuple[str, ...],
    ) -> None:
        mounted = attempt / ".agents" / "skills" / skill.skill_id
        if not mounted.exists():
            shutil.copytree(skill.path, mounted)
            for path in mounted.rglob("*"):
                path.chmod(0o555 if path.is_dir() else 0o444)
            mounted.parent.chmod(0o555)
            mounted.parent.parent.chmod(0o555)
        elif not mounted.is_dir():
            raise ValueError("mounted Agent skill path is not a directory")
        project_config = skill.path.parents[2] / ".codex" / "config.toml"
        if project_config.is_file():
            config_dir = attempt / ".codex"
            config_dir.mkdir(exist_ok=True)
            config = config_dir / "config.toml"
            if not config.exists():
                config.write_bytes(project_config.read_bytes())
                config.chmod(0o444)
                config_dir.chmod(0o555)
            elif not config.is_file():
                raise ValueError("mounted Codex config path is not a file")
        prompt = (
            f"Read .agents/skills/{skill.skill_id}/SKILL.md completely and follow "
            "that mounted skill package. Read input/manifest.json first; it declares "
            "every Agent-readable input for this Attempt. For inputs "
            "with mode direct, read only the exact canonical_path declared there and "
            "verify its sha256; those payloads are intentionally not copied under input/. "
            "Read materialized source evidence from input/ and, when "
            "input/experiment/manifest.json exists, from only the exact artifact_root "
            "plus artifacts[].path entries it authorizes; do not scan artifact_root. "
            "You may read tool artifacts created for this attempt under scratch/. "
            "Write the declared delivery under output/."
        )
        if feedback_paths:
            prompt += (
                " This is a retry attempt: read input/retry.json and "
                "input/prior-delivery/ before acting. Revise only these paths: "
                f"{', '.join(feedback_paths)}."
            )
        environment = os.environ.copy()
        user_bin = Path.home() / ".local" / "bin"
        if user_bin.is_dir():
            path_entries = environment.get("PATH", "").split(os.pathsep)
            if str(user_bin) not in path_entries:
                environment["PATH"] = os.pathsep.join(
                    (str(user_bin), *path_entries)
                )
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["ADE_PROJECT_ROOT"] = str(skill.path.parents[2].resolve())
        environment["ADE_AGENT_WORKSPACE"] = str(attempt.resolve())
        environment["ADE_REVIEW_OUTPUT_ROOT"] = str(
            (attempt / "scratch" / "reviews").resolve()
        )
        session_path = self._session_path(attempt)
        session = self._read_session(session_path)
        call_dir = attempt.parents[1]
        call = self._read_session(call_dir / "call.json")
        prior_session_id = session.get("current_session_id")
        resume = bool(isinstance(prior_session_id, str) and prior_session_id)
        completed = self._run(
            prompt=prompt,
            attempt=attempt,
            environment=environment,
            resume_session_id=prior_session_id if resume else None,
        )
        fallback_reason = None
        resume_stdout = ""
        resume_stderr = ""
        if resume and completed.returncode:
            fallback_reason = f"resume exited with status {completed.returncode}"
            resume_stdout = completed.stdout
            resume_stderr = completed.stderr
            completed = self._run(
                prompt=prompt,
                attempt=attempt,
                environment=environment,
                resume_session_id=None,
            )
        (attempt / "scratch" / "codex-events.jsonl").write_text(
            completed.stdout,
            encoding="utf-8",
        )
        (attempt / "scratch" / "codex.stdout").write_text(completed.stdout, encoding="utf-8")
        (attempt / "scratch" / "codex.stderr").write_text(completed.stderr, encoding="utf-8")
        if fallback_reason is not None:
            (attempt / "scratch" / "codex-resume.stdout").write_text(
                resume_stdout, encoding="utf-8"
            )
            (attempt / "scratch" / "codex-resume.stderr").write_text(
                resume_stderr, encoding="utf-8"
            )
        session_id = self._session_id(completed.stdout)
        usage_after = self._session_token_usage(session_id, environment)
        # Codex starts a new CLI invocation for every ADE Agent Call, including
        # `codex exec resume`. Its total_token_usage is cumulative within that
        # invocation, not across the resumed Session. It is therefore already
        # the exact increment owned by this Call.
        session_usage = usage_after
        mode = "resume" if resume and fallback_reason is None else (
            "fallback_new" if fallback_reason is not None else "new"
        )
        attempt_record = {
            "schema_version": "1",
            "attempt": attempt.name,
            "mode": mode,
            "requested_session_id": prior_session_id if resume else None,
            "session_id": session_id or (prior_session_id if mode == "resume" else None),
            "fallback_reason": fallback_reason,
            "returncode": completed.returncode,
            "usage": session_usage,
            "usage_source": (
                "codex_session_token_count" if session_usage is not None else None
            ),
        }
        (attempt / "scratch" / "codex-session.json").write_text(
            json.dumps(attempt_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        history = session.get("attempts")
        if not isinstance(history, list):
            history = []
        persisted = {
            **session,
            "schema_version": "1",
            "current_session_id": attempt_record["session_id"],
            "attempts": [*history, attempt_record],
        }
        session_path.write_text(
            json.dumps(persisted, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if completed.returncode:
            raise RuntimeError(f"Codex exited with status {completed.returncode}")

    @classmethod
    def _session_path(cls, attempt: Path) -> Path:
        call_dir = attempt.parents[1]
        call = cls._read_session(call_dir / "call.json")
        session_id = call.get("session_id")
        session_record = call.get("session_record_path")
        if isinstance(session_id, str) and session_id and isinstance(
            session_record, str
        ) and session_record:
            path = Path(session_record).resolve()
            session = cls._read_session(path)
            if session.get("session_id") != session_id:
                raise ValueError("persisted Agent Session identity mismatch")
            return path
        return call_dir / "codex-session.json"

    def _run(
        self,
        *,
        prompt: str,
        attempt: Path,
        environment: dict[str, str],
        resume_session_id: str | None,
    ) -> subprocess.CompletedProcess[str]:
        command = list(self.command)
        if "--model" not in command:
            command.extend(("--model", self.model))
        if not any(
            value.startswith("model_reasoning_effort=") for value in command
        ):
            command.extend(
                ("--config", f'model_reasoning_effort="{self.reasoning_effort}"')
            )
        if "--sandbox" not in command and "-s" not in command:
            command.extend(("--sandbox", self.sandbox_mode))
        if not any(value.startswith("approval_policy=") for value in command):
            command.extend(
                ("--config", f'approval_policy="{self.approval_policy}"')
            )
        if resume_session_id is not None:
            command.extend(("resume", resume_session_id))
        if "--json" not in command:
            command.append("--json")
        process = subprocess.Popen(
            [*command, prompt],
            cwd=attempt,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        process_record = attempt / "scratch" / "backend-process.json"
        process_record.write_text(
            json.dumps(
                {
                    "schema_version": "ade.agent_backend_process.v1",
                    "pid": process.pid,
                    "status": "running",
                    "started_at": time.time(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        stdout, stderr = process.communicate()
        process_record.write_text(
            json.dumps(
                {
                    "schema_version": "ade.agent_backend_process.v1",
                    "pid": process.pid,
                    "status": "exited",
                    "returncode": process.returncode,
                    "started_at": json.loads(
                        process_record.read_text(encoding="utf-8")
                    )["started_at"],
                    "completed_at": time.time(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            args=[*command, prompt],
            returncode=int(process.returncode),
            stdout=stdout,
            stderr=stderr,
        )

    @staticmethod
    def _read_session(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _session_id(events: str) -> str | None:
        for line in events.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict):
                continue
            for key in ("thread_id", "session_id"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate:
                    return candidate
            thread = value.get("thread")
            if isinstance(thread, dict):
                candidate = thread.get("id")
                if isinstance(candidate, str) and candidate:
                    return candidate
        return None

    @staticmethod
    def _session_token_usage(
        session_id: object,
        environment: dict[str, str],
    ) -> dict[str, int] | None:
        if not isinstance(session_id, str) or not re.fullmatch(
            r"[A-Za-z0-9-]+", session_id
        ):
            return None
        codex_root = Path(
            environment.get("CODEX_HOME") or (Path.home() / ".codex")
        )
        sessions_root = codex_root / "sessions"
        candidates = tuple(sessions_root.rglob(f"*-{session_id}.jsonl"))
        if len(candidates) != 1:
            return None
        latest: dict[str, int] | None = None
        try:
            lines = candidates[0].open(encoding="utf-8", errors="replace")
        except OSError:
            return None
        with lines:
            for line in lines:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = value.get("payload") if isinstance(value, dict) else None
                if not isinstance(payload, dict) or payload.get("type") != "token_count":
                    continue
                info = payload.get("info")
                total = info.get("total_token_usage") if isinstance(info, dict) else None
                if not isinstance(total, dict):
                    continue
                fields = (
                    "input_tokens",
                    "output_tokens",
                    "cached_input_tokens",
                    "reasoning_output_tokens",
                )
                if all(type(total.get(field)) is int for field in fields):
                    latest = {field: int(total[field]) for field in fields}
        return latest
