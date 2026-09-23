"""Independent worker for Harness-owned Analyzer Review batches."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from typing import Protocol

from ade.review_labor.usage import aggregate_usage

from ade.review_labor.command_queue import FileReviewCommandQueue
from ade.review_labor.protocol import (
    ReviewCommand,
    ReviewReceipt,
    ReviewReceiptStatus,
)


class ReviewCommandProcessor(Protocol):
    def execute(self, command: ReviewCommand) -> dict[str, object]: ...


class ReviewWorker:
    def __init__(self, queue: FileReviewCommandQueue, processor: ReviewCommandProcessor) -> None:
        self.queue = queue
        self.processor = processor

    def run_once(self) -> ReviewReceipt | None:
        command = self.queue.claim_next()
        if command is None:
            return None
        stopped = threading.Event()

        def maintain_progress_heartbeat() -> None:
            while not stopped.wait(5.0):
                try:
                    self.queue.update_progress(
                        command.command_id,
                        {
                            **self.queue.load_progress(command.command_id),
                            "last_heartbeat_at": time.time(),
                        },
                    )
                except (FileNotFoundError, ValueError):
                    return

        heartbeat = threading.Thread(
            target=maintain_progress_heartbeat,
            name=f"ade-review-heartbeat-{command.command_id}",
            daemon=True,
        )
        heartbeat.start()
        try:
            execute_with_progress = getattr(
                self.processor, "execute_with_progress", None
            )
            if execute_with_progress is None:
                packet = self.processor.execute(command)
            else:
                packet = execute_with_progress(
                    command,
                    lambda progress: self.queue.update_progress(
                        command.command_id,
                        {
                            **self.queue.load_progress(command.command_id),
                            **progress,
                            "last_heartbeat_at": time.time(),
                        },
                    ),
                )
            _validate_packet(command, packet)
            status = (
                ReviewReceiptStatus.COMPLETED_WITH_ERRORS
                if packet.get("status") in {"partial", "unavailable"}
                else ReviewReceiptStatus.COMPLETED
            )
            receipt = ReviewReceipt(
                receipt_id=f"receipt-{command.command_id}",
                command_id=command.command_id,
                logical_command_id=command.logical_command_id,
                attempt_id=command.attempt_id,
                attempt_index=command.attempt_index,
                run_id=command.run_id,
                coordinator_id=command.coordinator_id,
                plan_id=command.plan_id,
                trial_id=command.trial_id,
                status=status,
                packet=dict(packet),
            )
        except Exception as error:
            receipt = ReviewReceipt(
                receipt_id=f"receipt-{command.command_id}",
                command_id=command.command_id,
                logical_command_id=command.logical_command_id,
                attempt_id=command.attempt_id,
                attempt_index=command.attempt_index,
                run_id=command.run_id,
                coordinator_id=command.coordinator_id,
                plan_id=command.plan_id,
                trial_id=command.trial_id,
                status=ReviewReceiptStatus.FAILED,
                packet=_unavailable_packet(command),
                error=f"{type(error).__name__}: {error}",
            )
        finally:
            stopped.set()
            heartbeat.join()
        self.queue.publish_receipt(receipt)
        return receipt


class ReviewLaborBatchProcessor:
    """Direct Python adapter; no Agent/MCP call participates in execution."""

    def __init__(self, service, *, workspace_root: str | Path, output_root: str | Path) -> None:
        self.service = service
        self.workspace_root = Path(workspace_root).resolve()
        self.output_root = Path(output_root).resolve()

    def execute(self, command: ReviewCommand) -> dict[str, object]:
        return self.execute_with_progress(command, None)

    def execute_with_progress(self, command, progress_callback):
        batches = []
        usage_rows = []
        failed = False
        completed_before = 0
        error_before = 0
        total_units = sum(len(batch.units) for batch in command.batches)
        for batch_index, batch in enumerate(command.batches, start=1):
            def report(status):
                if progress_callback is None:
                    return
                progress_callback(
                    {
                        "state": "running",
                        "batch_index": batch_index,
                        "batch_count": len(command.batches),
                        "judge_job_id": status.job_id,
                        "completed_units": completed_before + status.completed_rows,
                        "error_units": error_before + status.error_rows,
                        "total_units": total_units,
                    }
                )

            result = self.service.review_batch_to_file(
                batch.units,
                batch.rubrics,
                self.output_root / command.command_id,
                submission_id=f"review:{command.command_id}:{batch.batch_id}",
                phase="initial_coverage",
                round_id=batch.batch_id,
                investigation_purpose=batch.investigation_purpose,
                identity_metadata={
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
                    "logical_command_id": command.logical_command_id,
                    "attempt_id": command.attempt_id,
                    "attempt_index": command.attempt_index,
                },
                progress_callback=report,
            )
            manifest_path = self.workspace_root / str(result["manifest_path"])
            results_path = self.workspace_root / str(result["results_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            rows = [
                json.loads(line)
                for line in results_path.read_text(encoding="utf-8").split("\n")
                if line.strip()
            ]
            failed = failed or result["status"] == "partial"
            completed_before += int(result["completed_units"])
            error_before += int(result["failed_units"])
            usage_rows.extend(manifest.get("unit_usage", ()))
            batches.append(
                {
                    "batch_id": batch.batch_id,
                    "pool": batch.pool,
                    "investigation_purpose": batch.investigation_purpose,
                    "review_id": result["review_id"],
                    "status": result["status"],
                    "requested_units": len(batch.units),
                    "completed_units": int(result["completed_units"]),
                    "failed_units": int(result["failed_units"]),
                    "fallback_units": int(result.get("fallback_units", 0)),
                    "manifest": manifest,
                    "results": rows,
                }
            )
        return {
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
            "status": "partial" if failed else "complete",
            "batches": batches,
            "usage": aggregate_usage(usage_rows),
        }


def _unavailable_packet(command: ReviewCommand) -> dict[str, object]:
    return {
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
        "usage": aggregate_usage(()),
    }


def _validate_packet(command: ReviewCommand, packet: object) -> None:
    if not isinstance(packet, dict):
        raise ValueError("Review processor packet must be an object")
    expected = {
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
    }
    if any(packet.get(key) != value for key, value in expected.items()):
        raise ValueError("Review processor packet identity is invalid")
    if packet.get("status") not in {"complete", "partial", "unavailable"}:
        raise ValueError("Review processor packet status is invalid")
    if not isinstance(packet.get("batches"), list):
        raise ValueError("Review processor packet batches are invalid")
    if not isinstance(packet.get("usage"), dict):
        raise ValueError("Review processor packet usage is invalid")
