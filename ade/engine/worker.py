"""Engine Worker that knows commands but not ADE RunState."""

from __future__ import annotations

from collections.abc import Callable
import threading

from ade.core.engine import (
    EngineAttemptError,
    EngineCommand,
    EngineReceipt,
    EngineReceiptStatus,
)
from ade.engine.command_queue import FileCommandQueue

EngineHandler = Callable[[EngineCommand], tuple[str, ...]]
EngineProgressProbe = Callable[[EngineCommand], str | None]


class EngineWorker:
    def __init__(
        self,
        *,
        queue: FileCommandQueue,
        handlers: dict[str, EngineHandler],
        heartbeat_interval_seconds: float = 30.0,
        coordinator_id: str | None = None,
        progress_probes: dict[str, EngineProgressProbe] | None = None,
    ) -> None:
        if heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")
        self.queue = queue
        self.handlers = dict(handlers)
        self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)
        self.coordinator_id = coordinator_id
        self.progress_probes = dict(progress_probes or {})
        self._active_command_id: str | None = None

    @property
    def active_command_id(self) -> str | None:
        return self._active_command_id

    def run_once(self) -> EngineReceipt | None:
        command = self.queue.claim_next(coordinator_id=self.coordinator_id)
        if command is None:
            return None
        self._active_command_id = command.command_id
        handler = self.handlers.get(command.kind)
        stopped = threading.Event()
        heartbeat_errors: list[Exception] = []
        progress_probe = self.progress_probes.get(command.kind)

        def maintain_heartbeat() -> None:
            while not stopped.wait(self.heartbeat_interval_seconds):
                try:
                    self.queue.heartbeat(
                        command.command_id,
                        progress_required=progress_probe is not None,
                        progress_marker=(
                            progress_probe(command)
                            if progress_probe is not None
                            else None
                        ),
                    )
                except Exception as error:
                    heartbeat_errors.append(error)
                    stopped.set()

        heartbeat = threading.Thread(
            target=maintain_heartbeat,
            name=f"ade-engine-heartbeat-{command.command_id}",
            daemon=True,
        )
        heartbeat.start()
        try:
            if handler is None:
                raise ValueError(f"no handler for Engine command kind {command.kind}")
            output_refs = tuple(handler(command))
            if heartbeat_errors:
                raise RuntimeError("engine heartbeat persistence failed") from heartbeat_errors[0]
            receipt = EngineReceipt(
                receipt_id=f"receipt-{command.command_id}",
                command_id=command.command_id,
                run_id=command.run_id,
                coordinator_id=command.coordinator_id,
                plan_id=command.plan_id,
                trial_id=command.trial_id,
                status=EngineReceiptStatus.SUCCEEDED,
                output_refs=output_refs,
                logical_command_id=(
                    command.logical_command_id or command.command_id
                ),
                attempt_id=command.attempt_id,
                attempt_index=command.attempt_index,
            )
        except Exception as error:
            attempt_error = (
                error if isinstance(error, EngineAttemptError) else None
            )
            receipt = EngineReceipt(
                receipt_id=f"receipt-{command.command_id}",
                command_id=command.command_id,
                run_id=command.run_id,
                coordinator_id=command.coordinator_id,
                plan_id=command.plan_id,
                trial_id=command.trial_id,
                status=EngineReceiptStatus.FAILED,
                output_refs=(
                    attempt_error.output_refs if attempt_error is not None else ()
                ),
                error=str(error),
                logical_command_id=(
                    command.logical_command_id or command.command_id
                ),
                attempt_id=command.attempt_id,
                attempt_index=command.attempt_index,
                failure_kind=(
                    attempt_error.failure_kind
                    if attempt_error is not None
                    else "engine_failed"
                ),
                retryable=(
                    attempt_error.retryable
                    if attempt_error is not None
                    else False
                ),
            )
        finally:
            stopped.set()
            heartbeat.join()
        # A liveness monitor may have terminalized this same immutable command
        # while the handler was returning. Its first terminal Receipt wins.
        if self.queue.has_receipt(command.command_id):
            return self.queue.load_receipt(command.command_id)
        self.queue.publish_receipt(receipt)
        return receipt
