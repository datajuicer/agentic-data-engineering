"""Filesystem queue for durable Analyzer Review commands."""

from __future__ import annotations

import json
import fcntl
import os
from pathlib import Path
import re

from ade.engine.storage.atomic import write_json_atomic
from ade.review_labor.protocol import (
    ReviewCommand,
    ReviewReceipt,
    ReviewReceiptStatus,
    decode_command,
    decode_receipt,
    encode_command,
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class FileReviewCommandQueue:
    def __init__(
        self,
        root: str | Path,
        *,
        coordinator_id: str | None = None,
        recover_claimed: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        self.inbox = self.root / "inbox"
        self.processing = self.root / "processing"
        self.receipts = self.root / "receipts"
        self.progress = self.root / "progress"
        self.coordinator_id = coordinator_id
        for path in (self.inbox, self.processing, self.receipts, self.progress):
            path.mkdir(parents=True, exist_ok=True)
        if recover_claimed:
            self._recover_claimed_commands()

    def _recover_claimed_commands(self) -> None:
        """Return commands left claimed by a stopped Review Worker to the inbox."""
        for source in sorted(self.processing.glob("*.json")):
            try:
                value = json.loads(source.read_text(encoding="utf-8"))
            except FileNotFoundError:
                continue
            if (
                self.coordinator_id is not None
                and value.get("coordinator_id") != self.coordinator_id
            ):
                continue
            if (self.receipts / source.name).is_file():
                source.unlink(missing_ok=True)
                continue
            target = self.inbox / source.name
            if target.exists():
                if json.loads(target.read_text(encoding="utf-8")) != json.loads(
                    source.read_text(encoding="utf-8")
                ):
                    raise ValueError(
                        f"Conflicting recovered Review command {source.stem}"
                    )
                source.unlink()
                continue
            os.replace(source, target)

    def submit(self, command: ReviewCommand) -> Path:
        self._validate(command.command_id)
        payload = json.loads(json.dumps(encode_command(command)))
        for path in (
            self.receipts / f"{command.command_id}.json",
            self.processing / f"{command.command_id}.json",
            self.inbox / f"{command.command_id}.json",
        ):
            if not path.exists():
                continue
            existing = json.loads(path.read_text(encoding="utf-8"))
            if path.parent == self.receipts:
                return path
            if existing != payload:
                raise ValueError(f"Review command {command.command_id} is immutable")
            return path
        path = self.inbox / f"{command.command_id}.json"
        write_json_atomic(path, payload)
        self.update_progress(
            command.command_id,
            {
                "schema_version": 1,
                "command_id": command.command_id,
                "attempt_id": command.attempt_id,
                "attempt_index": command.attempt_index,
                "state": "submitted",
                "batch_index": 0,
                "batch_count": len(command.batches),
                "completed_units": 0,
                "error_units": 0,
                "total_units": sum(len(batch.units) for batch in command.batches),
                "last_heartbeat_at": None,
            },
        )
        return path

    def update_progress(self, command_id: str, payload: dict[str, object]) -> Path:
        self._validate(command_id)
        path = self.progress / f"{command_id}.json"
        lock_path = self.progress / f".{command_id}.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            write_json_atomic(path, dict(payload))
        return path

    def load_progress(self, command_id: str) -> dict[str, object]:
        self._validate(command_id)
        return json.loads(
            (self.progress / f"{command_id}.json").read_text(encoding="utf-8")
        )

    def claim_next(self) -> ReviewCommand | None:
        for source in sorted(self.inbox.glob("*.json")):
            if self.coordinator_id is not None:
                try:
                    value = json.loads(source.read_text(encoding="utf-8"))
                except FileNotFoundError:
                    continue
                if value.get("coordinator_id") != self.coordinator_id:
                    continue
            target = self.processing / source.name
            try:
                os.replace(source, target)
            except FileNotFoundError:
                continue
            return decode_command(json.loads(target.read_text(encoding="utf-8")))
        return None

    def publish_receipt(self, receipt: ReviewReceipt) -> Path:
        self._validate(receipt.command_id)
        path = self.receipts / f"{receipt.command_id}.json"
        payload = json.loads(json.dumps(receipt.to_dict()))
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != payload:
                raise ValueError(f"Review receipt for {receipt.command_id} is immutable")
            return path
        write_json_atomic(path, payload)
        (self.processing / f"{receipt.command_id}.json").unlink(missing_ok=True)
        (self.inbox / f"{receipt.command_id}.json").unlink(missing_ok=True)
        progress = self.progress / f"{receipt.command_id}.json"
        if progress.is_file():
            self.update_progress(
                receipt.command_id,
                {
                    **self.load_progress(receipt.command_id),
                    "state": receipt.status.value,
                    "last_heartbeat_at": None,
                },
            )
        return path

    def has_receipt(self, command_id: str) -> bool:
        self._validate(command_id)
        return (self.receipts / f"{command_id}.json").is_file()

    def load_receipt(self, command_id: str) -> ReviewReceipt:
        self._validate(command_id)
        return decode_receipt(
            json.loads(
                (self.receipts / f"{command_id}.json").read_text(encoding="utf-8")
            )
        )

    def interrupt(
        self,
        command_id: str,
        *,
        reason: str = "Coordinator cancellation requested",
    ) -> ReviewReceipt:
        """Terminalize one exact inbox/processing Review Attempt."""

        self._validate(command_id)
        if self.has_receipt(command_id):
            return self.load_receipt(command_id)
        command_path = self.processing / f"{command_id}.json"
        if not command_path.is_file():
            command_path = self.inbox / f"{command_id}.json"
        if not command_path.is_file():
            raise ValueError(f"Review command {command_id} is not interruptible")
        command = decode_command(
            json.loads(command_path.read_text(encoding="utf-8"))
        )
        receipt = ReviewReceipt(
            receipt_id=f"receipt-{command_id}-coordinator-cancelled",
            command_id=command.command_id,
            logical_command_id=command.logical_command_id,
            attempt_id=command.attempt_id,
            attempt_index=command.attempt_index,
            run_id=command.run_id,
            coordinator_id=command.coordinator_id,
            plan_id=command.plan_id,
            trial_id=command.trial_id,
            status=ReviewReceiptStatus.FAILED,
            packet={
                "schema_version": "ade.analysis_review_packet.v1",
                "command_id": command.command_id,
                "logical_command_id": command.logical_command_id,
                "attempt_id": command.attempt_id,
                "attempt_index": command.attempt_index,
                "run_id": command.run_id,
                "coordinator_id": command.coordinator_id,
                "plan_id": command.plan_id,
                "trial_id": command.trial_id,
                "scope": {
                    "run_id": command.run_id,
                    "coordinator_id": command.coordinator_id,
                    "plan_id": command.plan_id,
                    "trial_id": command.trial_id,
                },
                "subject_ref": (
                    f"{command.run_id}/{command.coordinator_id}/"
                    f"{command.plan_id}/{command.trial_id}"
                ),
                "status": "cancelled",
                "batches": [],
                "usage": {},
            },
            error=f"coordinator_cancelled: {reason}",
        )
        self.publish_receipt(receipt)
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

    def terminalize_claimed(self, *, run_id: str | None = None) -> tuple[str, ...]:
        """Publish retryable failures for commands left by a dead worker."""
        terminal: list[str] = []
        for source in sorted(self.processing.glob("*.json")):
            try:
                command = decode_command(
                    json.loads(source.read_text(encoding="utf-8"))
                )
            except FileNotFoundError:
                continue
            if run_id is not None and command.run_id != run_id:
                continue
            if self.has_receipt(command.command_id):
                source.unlink(missing_ok=True)
                continue
            packet = {
                "schema_version": "ade.analysis_review_packet.v1",
                "command_id": command.command_id,
                "logical_command_id": command.logical_command_id,
                "attempt_id": command.attempt_id,
                "attempt_index": command.attempt_index,
                "run_id": command.run_id,
                "coordinator_id": command.coordinator_id,
                "plan_id": command.plan_id,
                "trial_id": command.trial_id,
                "scope": {
                    "run_id": command.run_id,
                    "coordinator_id": command.coordinator_id,
                    "plan_id": command.plan_id,
                    "trial_id": command.trial_id,
                },
                "subject_ref": (
                    f"{command.run_id}/{command.coordinator_id}/"
                    f"{command.plan_id}/{command.trial_id}"
                ),
                "status": "unavailable",
                "batches": [],
                "usage": {},
            }
            self.publish_receipt(
                ReviewReceipt(
                    receipt_id=f"receipt-{command.command_id}-worker-lost",
                    command_id=command.command_id,
                    logical_command_id=command.logical_command_id,
                    attempt_id=command.attempt_id,
                    attempt_index=command.attempt_index,
                    run_id=command.run_id,
                    coordinator_id=command.coordinator_id,
                    plan_id=command.plan_id,
                    trial_id=command.trial_id,
                    status=ReviewReceiptStatus.FAILED,
                    packet=packet,
                    error=(
                        "worker_lost: supervised Review Worker exited "
                        "without a terminal Receipt"
                    ),
                )
            )
            terminal.append(command.command_id)
        return tuple(terminal)

    @staticmethod
    def _validate(value: str) -> None:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError("Review command identity contains unsafe characters")
