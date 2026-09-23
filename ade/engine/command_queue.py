"""Filesystem transport for typed Engine commands and receipts."""

from __future__ import annotations

import json
import math
import os
from dataclasses import replace
import fcntl
from pathlib import Path
import re
import time

from ade.core.engine import EngineCommand, EngineReceipt, EngineReceiptStatus
from ade.engine.protocol import decode_command, decode_receipt, encode_command, encode_receipt
from ade.engine.storage.atomic import write_json_atomic

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class FileCommandQueue:
    def __init__(
        self,
        root: str | Path,
        *,
        claim_timeout_seconds: float = 1800.0,
        heartbeat_timeout_seconds: float = 1800.0,
    ) -> None:
        if not math.isfinite(claim_timeout_seconds) or claim_timeout_seconds <= 0:
            raise ValueError("claim_timeout_seconds must be positive")
        if (
            not math.isfinite(heartbeat_timeout_seconds)
            or heartbeat_timeout_seconds <= 0
        ):
            raise ValueError("heartbeat_timeout_seconds must be positive")
        self.root = Path(root).resolve()
        self.inbox = self.root / "inbox"
        self.processing = self.root / "processing"
        self.receipts = self.root / "receipts"
        self.liveness = self.root / "liveness"
        self.claim_timeout_seconds = float(claim_timeout_seconds)
        self.heartbeat_timeout_seconds = float(heartbeat_timeout_seconds)
        for path in (self.inbox, self.processing, self.receipts, self.liveness):
            path.mkdir(parents=True, exist_ok=True)

    def submit(self, command: EngineCommand) -> Path:
        self._validate_id(command.command_id)
        receipt = self.receipts / f"{command.command_id}.json"
        if receipt.exists():
            return receipt
        processing = self.processing / f"{command.command_id}.json"
        payload = encode_command(command)
        if processing.exists():
            if json.loads(processing.read_text(encoding="utf-8")) != payload:
                raise ValueError(f"command {command.command_id} is immutable")
            return processing
        path = self.inbox / f"{command.command_id}.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != payload:
                raise ValueError(f"command {command.command_id} is immutable")
            return path
        self._write_liveness(
            command.command_id,
            {
                "schema_version": "1",
                "command_id": command.command_id,
                "status": "submitted",
                "submitted_at": time.time(),
                "claimed_at": None,
                "last_heartbeat_at": None,
            },
        )
        # Publish liveness before the command becomes visible to claimers and
        # stale-command monitors.  Otherwise they can observe a command for
        # which no liveness record exists yet.
        write_json_atomic(path, payload)
        return path

    def claim_next(self, *, coordinator_id: str | None = None) -> EngineCommand | None:
        for source in sorted(self.inbox.glob("*.json")):
            if coordinator_id is not None:
                try:
                    candidate = decode_command(
                        json.loads(source.read_text(encoding="utf-8"))
                    )
                except FileNotFoundError:
                    continue
                if candidate.coordinator_id != coordinator_id:
                    continue
            command = self.claim(source.stem)
            if command is not None:
                return command
        return None

    def claim(self, command_id: str) -> EngineCommand | None:
        self._validate_id(command_id)
        source = self.inbox / f"{command_id}.json"
        target = self.processing / source.name
        try:
            os.replace(source, target)
        except FileNotFoundError:
            return None
        command = decode_command(json.loads(target.read_text(encoding="utf-8")))
        now = time.time()
        record = self._load_liveness(command_id, required=False)
        if record is None:
            record = {
                "schema_version": "1",
                "command_id": command_id,
                "status": "submitted",
                "submitted_at": now,
                "claimed_at": None,
                "last_heartbeat_at": None,
            }
        self._write_liveness(
            command_id,
            {
                **record,
                "status": "running",
                "claimed_at": now,
                "worker_pid": os.getpid(),
                "worker_id": os.environ.get("ADE_GENERALIZATION_WORKER_ID"),
                "last_heartbeat_at": now,
            },
        )
        return command

    def heartbeat(
        self,
        command_id: str,
        *,
        at: float | None = None,
        progress_required: bool = False,
        progress_marker: str | None = None,
    ) -> None:
        """Persist worker liveness without changing ADE Run revision."""

        self._validate_id(command_id)
        if not (self.processing / f"{command_id}.json").is_file():
            if self.has_receipt(command_id):
                return
            raise ValueError(f"Engine command {command_id} is not running")
        record = self._load_liveness(command_id)
        if record.get("status") != "running":
            raise ValueError(f"Engine command {command_id} has no running lease")
        observed = time.time() if at is None else float(at)
        updated = {**record, "last_heartbeat_at": observed}
        if progress_required:
            updated["progress_required"] = True
            if (
                progress_marker is not None
                and progress_marker != record.get("progress_marker")
            ):
                updated["progress_marker"] = str(progress_marker)
                updated["last_progress_at"] = observed
        self._write_liveness(
            command_id,
            updated,
        )

    def publish_receipt(self, receipt: EngineReceipt) -> Path:
        self._validate_id(receipt.command_id)
        if not receipt.logical_command_id:
            command_path = self.processing / f"{receipt.command_id}.json"
            if not command_path.is_file():
                command_path = self.inbox / f"{receipt.command_id}.json"
            if command_path.is_file():
                command = decode_command(
                    json.loads(command_path.read_text(encoding="utf-8"))
                )
                receipt = replace(
                    receipt,
                    logical_command_id=(
                        command.logical_command_id or command.command_id
                    ),
                    attempt_id=command.attempt_id,
                    attempt_index=command.attempt_index,
                )
        path = self.receipts / f"{receipt.command_id}.json"
        payload = encode_receipt(receipt)
        lock_path = self.receipts / f".{receipt.command_id}.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if path.exists():
                if json.loads(path.read_text(encoding="utf-8")) != payload:
                    raise ValueError(
                        f"receipt for {receipt.command_id} is immutable"
                    )
                return path
            write_json_atomic(path, payload)
            processing = self.processing / f"{receipt.command_id}.json"
            processing.unlink(missing_ok=True)
            inbox = self.inbox / f"{receipt.command_id}.json"
            inbox.unlink(missing_ok=True)
            liveness = self._load_liveness(receipt.command_id, required=False)
            if liveness is not None:
                self._write_liveness(
                    receipt.command_id,
                    {
                        **liveness,
                        "status": "terminal",
                        "receipt_id": receipt.receipt_id,
                        "terminal_at": time.time(),
                    },
                )
        return path

    def interrupt(
        self,
        command_id: str,
        *,
        failure_kind: str = "operator_interrupted",
        reason: str = "Run pause requested",
        retryable: bool = True,
    ) -> EngineReceipt:
        """Terminalize one physical Attempt before stopping its worker."""

        self._validate_id(command_id)
        if self.has_receipt(command_id):
            return self.load_receipt(command_id)
        command_path = self.processing / f"{command_id}.json"
        if not command_path.is_file():
            command_path = self.inbox / f"{command_id}.json"
        if not command_path.is_file():
            raise ValueError(f"Engine command {command_id} is not interruptible")
        command = decode_command(
            json.loads(command_path.read_text(encoding="utf-8"))
        )
        receipt = EngineReceipt(
            receipt_id=f"receipt-{command_id}-{failure_kind.replace('_', '-')}",
            command_id=command_id,
            run_id=command.run_id,
            coordinator_id=command.coordinator_id,
            plan_id=command.plan_id,
            trial_id=command.trial_id,
            status=EngineReceiptStatus.FAILED,
            error=f"{failure_kind}: {reason}",
            logical_command_id=(
                command.logical_command_id or command.command_id
            ),
            attempt_id=command.attempt_id,
            attempt_index=command.attempt_index,
            failure_kind=failure_kind,
            retryable=retryable,
        )
        try:
            self.publish_receipt(receipt)
        except ValueError:
            if not self.has_receipt(command_id):
                raise
        return self.load_receipt(command_id)

    def command_ids(
        self,
        *,
        run_id: str,
        coordinator_id: str | None = None,
    ) -> tuple[str, ...]:
        """List nonterminal Commands for a Run or one of its Coordinators."""

        matches: set[str] = set()
        for root in (self.inbox, self.processing):
            for source in sorted(root.glob("*.json")):
                try:
                    command = decode_command(
                        json.loads(source.read_text(encoding="utf-8"))
                    )
                except FileNotFoundError:
                    continue
                if (
                    command.run_id == run_id
                    and (coordinator_id is None or command.coordinator_id == coordinator_id)
                    and not self.has_receipt(command.command_id)
                ):
                    matches.add(command.command_id)
        return tuple(sorted(matches))

    def processing_command_ids(
        self, *, coordinator_id: str | None = None
    ) -> tuple[str, ...]:
        """Return command IDs with a durable processing lease."""
        matches: list[str] = []
        for source in sorted(self.processing.glob("*.json")):
            try:
                command = decode_command(
                    json.loads(source.read_text(encoding="utf-8"))
                )
            except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
                continue
            if coordinator_id is None or command.coordinator_id == coordinator_id:
                matches.append(command.command_id)
        return tuple(matches)

    def load_receipt(self, command_id: str) -> EngineReceipt:
        self._validate_id(command_id)
        path = self.receipts / f"{command_id}.json"
        return decode_receipt(json.loads(path.read_text(encoding="utf-8")))

    def has_receipt(self, command_id: str) -> bool:
        self._validate_id(command_id)
        return (self.receipts / f"{command_id}.json").is_file()

    def expire_stale(
        self,
        *,
        now: float | None = None,
        coordinator_id: str | None = None,
    ) -> tuple[str, ...]:
        """Terminalize expired work without creating a second execution."""

        observed = time.time() if now is None else float(now)
        terminal: list[str] = []
        for source in sorted(self.inbox.glob("*.json")):
            command_id = source.stem
            try:
                command = decode_command(
                    json.loads(source.read_text(encoding="utf-8"))
                )
            except FileNotFoundError:
                continue
            if (
                coordinator_id is not None
                and command.coordinator_id != coordinator_id
            ):
                continue
            if (self.receipts / source.name).exists():
                source.unlink()
                continue
            record = self._load_liveness(command_id, required=False)
            if record is None:
                # A command may be observed between transport publication and
                # liveness publication by an older producer.  Leave it for a
                # later scan; it is not safe to infer a timeout without a
                # timestamp.
                continue
            submitted_at = float(record["submitted_at"])
            if observed - submitted_at < self.claim_timeout_seconds:
                continue
            # Claim the transport file solely to win the race against a worker.
            target = self.processing / source.name
            try:
                os.replace(source, target)
            except FileNotFoundError:
                continue
            self.publish_receipt(
                EngineReceipt(
                    receipt_id=f"receipt-{command_id}-claim-timeout",
                    command_id=command_id,
                    run_id=command.run_id,
                    coordinator_id=command.coordinator_id,
                    plan_id=command.plan_id,
                    trial_id=command.trial_id,
                    status=EngineReceiptStatus.FAILED,
                    error="claim_timeout: Engine Command was not claimed before its liveness deadline",
                    logical_command_id=(
                        command.logical_command_id or command.command_id
                    ),
                    attempt_id=command.attempt_id,
                    attempt_index=command.attempt_index,
                    failure_kind="claim_timeout",
                    retryable=True,
                )
            )
            terminal.append(command_id)
        for source in sorted(self.processing.glob("*.json")):
            command_id = source.stem
            try:
                command = decode_command(
                    json.loads(source.read_text(encoding="utf-8"))
                )
            except FileNotFoundError:
                continue
            if (
                coordinator_id is not None
                and command.coordinator_id != coordinator_id
            ):
                continue
            if (self.receipts / source.name).exists():
                source.unlink()
                continue
            record = self._load_liveness(command_id, required=False)
            if record is None:
                # Do not let one incomplete command crash the Engine process
                # and take unrelated active work down with it.
                continue
            last_heartbeat = (
                record.get("last_heartbeat_at")
                or record.get("claimed_at")
                or record.get("submitted_at")
            )
            # Evaluation grading can be silent while its process heartbeat advances.
            progress_required = (
                command.kind in {"train_sft", "train_rft"}
                and record.get("progress_required") is True
            )
            last_progress = (
                record.get("last_progress_at")
                or record.get("claimed_at")
                or record.get("submitted_at")
            )
            liveness_reference = last_progress if progress_required else last_heartbeat
            if observed - float(liveness_reference) < self.heartbeat_timeout_seconds:
                continue
            failure_kind = (
                "training_progress_stalled"
                if progress_required
                else "worker_lost"
            )
            error = (
                "training_progress_stalled: claimed training Command has "
                "process heartbeats but no advancing durable step marker"
                if progress_required
                else "worker_lost: claimed Engine Command has no terminal Receipt"
            )
            self.publish_receipt(
                EngineReceipt(
                    receipt_id=f"receipt-{command_id}-{failure_kind.replace('_', '-')}",
                    command_id=command_id,
                    run_id=command.run_id,
                    coordinator_id=command.coordinator_id,
                    plan_id=command.plan_id,
                    trial_id=command.trial_id,
                    status=EngineReceiptStatus.FAILED,
                    error=error,
                    logical_command_id=(
                        command.logical_command_id or command.command_id
                    ),
                    attempt_id=command.attempt_id,
                    attempt_index=command.attempt_index,
                    failure_kind=failure_kind,
                    retryable=True,
                )
            )
            terminal.append(command_id)
        return tuple(terminal)

    def terminalize_claimed(
        self,
        *,
        coordinator_id: str | None = None,
        run_id: str | None = None,
        worker_pid: int | None = None,
    ) -> tuple[str, ...]:
        """Record worker loss after its supervised process has exited."""

        terminal: list[str] = []
        for source in sorted(self.processing.glob("*.json")):
            try:
                command = decode_command(
                    json.loads(source.read_text(encoding="utf-8"))
                )
            except FileNotFoundError:
                continue
            if (
                coordinator_id is not None
                and command.coordinator_id != coordinator_id
            ):
                continue
            if run_id is not None and command.run_id != run_id:
                continue
            if worker_pid is not None and self.load_liveness(command.command_id).get("worker_pid") != worker_pid:
                continue
            if self.has_receipt(command.command_id):
                source.unlink(missing_ok=True)
                continue
            self.publish_receipt(
                EngineReceipt(
                    receipt_id=f"receipt-{command.command_id}-worker-lost",
                    command_id=command.command_id,
                    run_id=command.run_id,
                    coordinator_id=command.coordinator_id,
                    plan_id=command.plan_id,
                    trial_id=command.trial_id,
                    status=EngineReceiptStatus.FAILED,
                    error=(
                        "worker_lost: supervised Engine Worker exited "
                        "without a terminal Receipt"
                    ),
                    logical_command_id=(
                        command.logical_command_id or command.command_id
                    ),
                    attempt_id=command.attempt_id,
                    attempt_index=command.attempt_index,
                    failure_kind="worker_lost",
                    retryable=True,
                )
            )
            terminal.append(command.command_id)
        return tuple(terminal)

    def load_liveness(self, command_id: str) -> dict[str, object]:
        self._validate_id(command_id)
        return self._load_liveness(command_id)

    def _load_liveness(
        self,
        command_id: str,
        *,
        required: bool = True,
    ) -> dict[str, object] | None:
        path = self.liveness / f"{command_id}.json"
        if not path.is_file():
            if required:
                raise ValueError(f"Engine command {command_id} has no liveness record")
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("command_id") != command_id:
            raise ValueError(f"Engine command {command_id} has invalid liveness")
        return payload

    def _write_liveness(self, command_id: str, payload: dict[str, object]) -> None:
        write_json_atomic(self.liveness / f"{command_id}.json", payload)

    @staticmethod
    def _validate_id(value: str) -> None:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError("command_id contains unsafe characters")
