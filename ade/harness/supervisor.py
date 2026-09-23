"""Run the five ADE component processes as one observable process group."""

from __future__ import annotations

from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time
from typing import Callable

from ade.controller.state import StateCoordinator
from ade.core.failures import FailureState
from ade.core.outcomes import RunSuspendedOutcome
from ade.core.run import RunStatus
from ade.harness.experiment_config import ExperimentConfigCompiler
from ade.harness.processes import initialize_control_run
from ade.harness.wiring import build_agent_process
from ade.harness.run_cleanup import cleanup_cancelled_run
from ade.harness.runtime_roots import deployment_runtime_roots
from ade.engine.command_queue import FileCommandQueue
from ade.engine.storage.atomic import write_json_atomic
from ade.review_labor.command_queue import FileReviewCommandQueue
from ade.memory.repository import FileRunRepository, StaleRevisionError
from ade.tasks.registry import default_task_registry


class RunSupervisor:
    def __init__(
        self,
        *,
        project_root: str | Path,
        runs_root: str | Path | None = None,
        queue_root: str | Path | None = None,
        object_root: str | Path | None = None,
        work_root: str | Path | None = None,
        process_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.runs_root = Path(runs_root).resolve() if runs_root is not None else None
        self.queue_root = Path(queue_root).resolve() if queue_root is not None else None
        self.object_root = Path(object_root).resolve() if object_root is not None else None
        self.work_root = Path(work_root).resolve() if work_root is not None else None
        self.process_factory = process_factory
        self.process_environment = os.environ.copy()

    def run(
        self,
        *,
        experiment_config: str | Path,
        deployment_config: str | Path,
        run_id: str,
        resume: bool,
        initial_state_reference: str | None = None,
        initial_state_frontier: tuple[str, ...] = (),
        poll_interval: float = 10.0,
    ) -> int:
        config = Path(experiment_config)
        if not config.is_absolute():
            config = self.project_root / config
        deployment = Path(deployment_config)
        if not deployment.is_absolute():
            deployment = self.project_root / deployment
        deployment = deployment.resolve()
        experiment = ExperimentConfigCompiler(
            default_task_registry(), project_root=self.project_root
        ).compile_file(
            config,
            deployment_config=deployment,
            run_id=run_id,
        )
        self._resolve_runtime_roots(experiment.resolved)
        self.process_environment["RAY_ADDRESS"] = str(
            experiment.run.run_resources["ray_cluster"]["address"]
        )

        # Engine/Agent/Review workers use deployment-scoped queues and roots.
        # A worker left behind by a previous Run can therefore claim this Run's
        # command before the freshly-started worker does.  The deployment is
        # exclusive, so reject a live sibling Run and fence stale workers before
        # creating the new process group.  The deployment-owned Judge is not an
        # ADE harness worker and is intentionally excluded here.
        fenced_worker_pids = self._prepare_deployment_workers(run_id=run_id)

        assert self.runs_root is not None
        assert self.queue_root is not None
        assert self.object_root is not None
        initialize_control_run(
            experiment=experiment,
            project_root=self.project_root,
            runs_root=self.runs_root,
            queue_root=self.queue_root,
            object_root=self.object_root,
            resume=resume,
            initial_state_reference=initial_state_reference,
            initial_state_frontier=initial_state_frontier,
        )
        log_root = self.runs_root / run_id / "services" / "supervisor"
        log_root.mkdir(parents=True, exist_ok=True)
        processes: list[tuple[str, subprocess.Popen]] = []
        logs = []
        service_commands: dict[str, tuple[str, ...]] = {}
        service_logs = {}
        shutdown_requested = False
        pause_submitted = False
        previous_handlers = {
            signum: signal.getsignal(signum)
            for signum in (signal.SIGTERM, signal.SIGINT)
        }

        def request_shutdown(_signum, _frame) -> None:
            nonlocal shutdown_requested
            if shutdown_requested:
                raise KeyboardInterrupt
            shutdown_requested = True

        def submit_pause_if_requested() -> None:
            nonlocal pause_submitted
            if not shutdown_requested or pause_submitted:
                return
            repository = FileRunRepository(self.runs_root)
            state = repository.load(run_id)
            if state.status.value in {
                "paused",
                "suspended",
                "completed",
                "failed",
                "cancelled",
            }:
                pause_submitted = True
                return
            if state.pause_requested:
                pause_submitted = True
                return
            try:
                StateCoordinator(repository).request_pause(
                    run_id,
                    reason="supervisor received shutdown signal",
                )
                pause_submitted = True
            except StaleRevisionError:
                return

        for signum in previous_handlers:
            signal.signal(signum, request_shutdown)
        try:
            for name, command in self.commands(
                experiment_config=config,
                deployment_config=deployment,
                run_id=run_id,
                resume=True,
                poll_interval=poll_interval,
                coordinator_count=int(
                    dict(experiment.control["search"])["coordinator_count"]
                ),
            ):
                log = (log_root / f"{name}.log").open("ab")
                logs.append(log)
                service_commands[name] = command
                service_logs[name] = log
                processes.append(
                    (
                        name,
                        self.process_factory(
                            command,
                            cwd=self.project_root,
                            env=self.process_environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        ),
                    )
                )
            self._write_worker_provenance(
                run_id=run_id,
                fenced_worker_pids=fenced_worker_pids,
                processes=processes,
                service_commands=service_commands,
                resume=resume,
            )
            exit_code = self._wait_for_control(
                processes,
                run_dir=self.runs_root / run_id,
                max_restarts=(
                    int(
                        dict(
                            experiment.control["engine"]["automatic_recovery"]
                        )["max_attempts"]
                    )
                    - 1
                ),
                restart_process=lambda name: self._restart_process(
                    name,
                    command=service_commands[name],
                    log=service_logs[name],
                ),
                on_tick=submit_pause_if_requested,
                suspend_run=lambda name, code: self._suspend_for_component(
                    run_id,
                    component=name,
                    exit_code=code,
                ),
                expected_worker_exit=lambda name: self._coordinator_exit_expected(
                    self.runs_root / run_id,
                    name,
                ),
            )
            self._drain_suspended_work(
                run_id=run_id,
                processes=processes,
                engine_config=experiment.control["engine"],
                agent_config=experiment.control.get("agent", {}),
                artifact_builder_config=experiment.control.get("artifact_builder", {}),
                poll_interval=poll_interval,
            )
            return exit_code
        finally:
            for _name, process in reversed(processes):
                if process.poll() is None:
                    try:
                        process.send_signal(signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for _name, process in reversed(processes):
                if process.poll() is None:
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            try:
                state = FileRunRepository(self.runs_root).load(run_id)
                if state.status in {
                    RunStatus.COMPLETED,
                    RunStatus.FAILED,
                    RunStatus.CANCELLED,
                }:
                    run_resources = state.run_resources or {}
                    ray_cluster = (
                        run_resources.get("ray_cluster", {})
                        if isinstance(run_resources, dict)
                        else {}
                    )
                    artifact_staging = (
                        run_resources.get("artifact_staging", {})
                        if isinstance(run_resources, dict)
                        else {}
                    )
                    cleanup_cancelled_run(
                        run_id=run_id,
                        run_dir=self.runs_root / run_id,
                        queue_root=self.queue_root,
                        review_root=self.queue_root.parent / "review",
                        ray_address=(
                            str(ray_cluster.get("address"))
                            if isinstance(ray_cluster, dict) and ray_cluster.get("address")
                            else None
                        ),
                        artifact_cache_dirs=(
                            (str(artifact_staging["cache_dir"]),)
                            if isinstance(artifact_staging, dict)
                            and artifact_staging.get("enabled")
                            and artifact_staging.get("cache_dir")
                            else ()
                        ),
                        process_pids=(
                            int(process.pid)
                            for _name, process in processes
                            if getattr(process, "pid", None) is not None
                        ),
                    )
            except (OSError, ValueError, RuntimeError):
                # The Run state and worker shutdown remain authoritative; the
                # cleanup receipt records resource failures when possible.
                pass
            for log in logs:
                log.close()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)

    def _drain_suspended_work(
        self, *, run_id: str, processes: list, engine_config: dict,
        agent_config: dict, artifact_builder_config: dict, poll_interval: float,
    ) -> None:
        """Finish published work without restarting Control or admitting outputs."""
        repository = FileRunRepository(self.runs_root)
        if repository.load(run_id).status is not RunStatus.SUSPENDED:
            return
        engine = FileCommandQueue(
            self.queue_root,
            claim_timeout_seconds=float(engine_config["claim_timeout_seconds"]),
            heartbeat_timeout_seconds=float(engine_config["heartbeat_timeout_seconds"]),
        )
        review = FileReviewCommandQueue(
            self.queue_root.parent / "review", recover_claimed=False,
        )
        agent = None
        while True:
            state = repository.load(run_id)
            if state.status is not RunStatus.SUSPENDED:
                return
            pending = False
            dead_engines = set()
            dead_reviews = set()
            for name, process in processes:
                if process.poll() is None:
                    continue
                if name == "engine" or name.startswith("engine-c"):
                    dead_engines.add(name)
                if name == "review" or name.startswith("review-c"):
                    dead_reviews.add(name)
            live_review_pending = False
            for coordinator in state.coordinators:
                coordinator_id = coordinator.coordinator_id
                if "engine" not in dead_engines and f"engine-{coordinator_id}" not in dead_engines:
                    engine.expire_stale(coordinator_id=coordinator_id)
                    pending |= bool(engine.command_ids(
                        run_id=run_id, coordinator_id=coordinator_id,
                    ))
                # Dead Engine owners may leave live Ray jobs. Preserve their
                # claims for exact-job repair; resume refuses those claims.
                if "review" not in dead_reviews and f"review-{coordinator_id}" not in dead_reviews:
                    live_review_pending |= bool(review.command_ids(
                        run_id=run_id, coordinator_id=coordinator_id,
                    ))
            pending |= live_review_pending
            if dead_reviews and not live_review_pending:
                # The queue's terminalize API is Run-scoped. Wait until live
                # Review shards finish before fencing dead shards' claims.
                # Unclaimed work on dead shards stays in inbox for resume.
                review.terminalize_claimed(run_id=run_id)
            for active in state.active_agent_calls:
                if active.status == "retry_pending":
                    continue
                if agent is None:
                    agent = build_agent_process(
                        project_root=self.project_root, runs_root=self.runs_root,
                        agent_config=agent_config,
                        artifact_builder_config=artifact_builder_config,
                    )
                if agent.calls.has_terminal(active):
                    continue
                if agent.calls.is_expired(active):
                    agent.calls.expire_active(active)
                else:
                    pending = True
            if not pending:
                return
            time.sleep(poll_interval)

    @staticmethod
    def _wait_for_control(
        processes: list[tuple[str, subprocess.Popen]],
        *,
        run_dir: Path,
        max_restarts: int,
        restart_process: Callable[[str], subprocess.Popen | None] | None = None,
        on_tick: Callable[[], None] | None = None,
        suspend_run: Callable[[str, int], bool] | None = None,
        expected_worker_exit: Callable[[str], bool] | None = None,
        restart_stability_seconds: float = 30.0,
    ) -> int:
        if max_restarts < 0:
            raise ValueError("max_restarts cannot be negative")
        control_name, control = processes[-1]
        if control_name != "control":
            raise ValueError("Control must be the final supervised process")
        ignored_exits: set[str] = set()
        restart_counts: dict[str, int] = {}
        restart_started_at: dict[str, float] = {}
        while True:
            if on_tick is not None:
                on_tick()
            control_code = control.poll()
            if control_code is None and control_name in restart_started_at:
                if (
                    time.monotonic() - restart_started_at[control_name]
                    >= restart_stability_seconds
                ):
                    restart_counts[control_name] = 0
                    restart_started_at.pop(control_name, None)
            if control_code is not None:
                if RunSupervisor._run_is_terminal(run_dir):
                    return int(control_code)
                if suspend_run is None and int(control_code) == 0:
                    return 0
                count = restart_counts.get(control_name, 0)
                replacement = None
                if restart_process is not None and count < max_restarts:
                    replacement = restart_process(control_name)
                    restart_counts[control_name] = count + 1
                if replacement is not None:
                    processes[-1] = (control_name, replacement)
                    control = replacement
                    restart_started_at[control_name] = time.monotonic()
                    continue
                if suspend_run is not None and suspend_run(
                    control_name, int(control_code)
                ):
                    return 0
                raise RuntimeError(
                    f"{control_name} process exited before Run terminal "
                    f"with code {control_code}"
                )
            terminal = RunSupervisor._run_is_terminal(run_dir)
            for index, (name, process) in enumerate(processes[:-1]):
                code = process.poll()
                if code is None and name in restart_started_at:
                    if (
                        time.monotonic() - restart_started_at[name]
                        >= restart_stability_seconds
                    ):
                        restart_counts[name] = 0
                        restart_started_at.pop(name, None)
                    continue
                if code is not None and not terminal:
                    if name in ignored_exits:
                        continue
                    if (
                        expected_worker_exit is not None
                        and expected_worker_exit(name)
                    ):
                        ignored_exits.add(name)
                        continue
                    count = restart_counts.get(name, 0)
                    replacement = None
                    if restart_process is not None and count < max_restarts:
                        replacement = restart_process(name)
                        restart_counts[name] = count + 1
                    if replacement is not None:
                        processes[index] = (name, replacement)
                        restart_started_at[name] = time.monotonic()
                        continue
                    if name == "monitor":
                        ignored_exits.add(name)
                        continue
                    if suspend_run is not None and suspend_run(name, int(code)):
                        return 0
                    raise RuntimeError(
                        f"{name} process exited before Run terminal with code {code}"
                    )
            time.sleep(0.25)

    @staticmethod
    def _coordinator_exit_expected(run_dir: Path, name: str) -> bool:
        if not any(
            name.startswith(prefix)
            for prefix in ("engine-c", "agent-c", "review-c")
        ):
            return False
        coordinator_id = name.rsplit("-", 1)[-1]
        if coordinator_id == "c000":
            return False
        try:
            state = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            receipt = json.loads(
                (
                    run_dir
                    / "reports"
                    / "cleanup"
                    / "coordinators"
                    / f"{coordinator_id}.json"
                ).read_text(encoding="utf-8")
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
        coordinator = next(
            (
                item
                for item in state.get("coordinators", ())
                if isinstance(item, dict)
                and item.get("coordinator_id") == coordinator_id
            ),
            None,
        )
        return bool(
            isinstance(coordinator, dict)
            and coordinator.get("control_status")
            in {"cancel_requested", "cancelled"}
            and receipt.get("status") in {"in_progress", "complete"}
            and name in receipt.get("expected_processes", ())
        )

    def _suspend_for_component(
        self,
        run_id: str,
        *,
        component: str,
        exit_code: int,
    ) -> bool:
        assert self.runs_root is not None
        repository = FileRunRepository(self.runs_root)
        for _ in range(2):
            state = repository.load(run_id)
            if state.status in {
                RunStatus.PAUSED,
                RunStatus.SUSPENDED,
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }:
                return True
            try:
                StateCoordinator(repository).apply(
                    run_id,
                    RunSuspendedOutcome(
                        run_id=run_id,
                        subject_ref=run_id,
                        basis_revision=state.revision,
                        failure=FailureState(
                            code=f"{component}_restart_exhausted",
                            message=(
                                f"{component} process exited repeatedly; "
                                f"last exit code {exit_code}"
                            ),
                            retryable=True,
                        ),
                    ),
                    event_type="run_suspended_for_component",
                )
                return True
            except StaleRevisionError:
                continue
        return False

    def _restart_process(
        self,
        name: str,
        *,
        command: tuple[str, ...],
        log,
    ) -> subprocess.Popen | None:
        if name == "engine" or name.startswith("engine-c"):
            assert self.queue_root is not None
            coordinator_id = (
                name.removeprefix("engine-")
                if name.startswith("engine-")
                else None
            )
            FileCommandQueue(self.queue_root).terminalize_claimed(
                coordinator_id=coordinator_id
            )
        elif name == "review" or name.startswith("review-"):
            assert self.queue_root is not None
            FileReviewCommandQueue(
                self.queue_root.parent / "review",
                coordinator_id=(
                    name.removeprefix("review-")
                    if name.startswith("review-")
                    else None
                ),
            ).terminalize_claimed()
        return self.process_factory(
            command,
            cwd=self.project_root,
            env=self.process_environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    @staticmethod
    def _run_is_terminal(run_dir: Path) -> bool:
        path = run_dir / "run.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        return isinstance(value, dict) and value.get("status") in {
            "completed",
            "failed",
            "cancelled",
            "paused",
            "suspended",
        }

    def commands(
        self,
        *,
        experiment_config: Path,
        deployment_config: Path,
        run_id: str,
        resume: bool,
        poll_interval: float,
        coordinator_count: int = 1,
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        if any(
            root is None
            for root in (
                self.runs_root,
                self.queue_root,
                self.object_root,
                self.work_root,
            )
        ):
            raise ValueError("supervisor runtime roots are not resolved")
        base = (sys.executable, "-m", "ade.harness.cli")
        monitor = base + (
            "monitor", "run", str(experiment_config), "--run-id", run_id,
            "--deployment", str(deployment_config),
            "--project-root", str(self.project_root), "--runs-root", str(self.runs_root),
        )
        engine = base + (
            "engine", "worker", "--queue-root", str(self.queue_root),
            "--object-root", str(self.object_root), "--work-root", str(self.work_root),
            "--poll-interval", str(poll_interval),
        )
        agent = base + (
            "agent", "worker", str(experiment_config), "--run-id", run_id,
            "--deployment", str(deployment_config),
            "--project-root", str(self.project_root), "--runs-root", str(self.runs_root),
            "--poll-interval", str(poll_interval),
        )
        review = base + (
            "review", "worker", str(experiment_config), "--run-id", run_id,
            "--deployment", str(deployment_config),
            "--project-root", str(self.project_root),
            "--runs-root", str(self.runs_root),
            "--queue-root", str(self.queue_root.parent / "review"),
            "--work-root", str(self.work_root.parent / "review-work" / run_id),
            "--poll-interval", str(poll_interval),
        )
        control = base + (
            "control", "run", str(experiment_config), "--run-id", run_id,
            "--deployment", str(deployment_config),
            "--project-root", str(self.project_root), "--runs-root", str(self.runs_root),
            "--queue-root", str(self.queue_root), "--object-root", str(self.object_root),
            "--poll-interval", str(poll_interval),
        )
        if resume:
            control += ("--resume",)
        if coordinator_count <= 1:
            services = (
                ("engine", engine),
                ("agent", agent),
                ("review", review),
            )
        else:
            coordinator_ids = tuple(
                f"c{index:03d}" for index in range(coordinator_count + 1)
            )
            engines = tuple(
                (
                    f"engine-{coordinator_id}",
                    engine + ("--coordinator-id", coordinator_id),
                )
                for coordinator_id in coordinator_ids
            )
            agents = tuple(
                (
                    f"agent-{coordinator_id}",
                    agent + ("--coordinator-id", coordinator_id),
                )
                for coordinator_id in coordinator_ids
            )
            global_agent = (
                "agent-global",
                agent + ("--global-only",),
            )
            reviews = tuple(
                (
                    f"review-{coordinator_id}",
                    base
                    + (
                        "review",
                        "worker",
                        str(experiment_config),
                        "--deployment",
                        str(deployment_config),
                        "--run-id",
                        run_id,
                        "--project-root",
                        str(self.project_root),
                        "--queue-root",
                        str(self.queue_root.parent / "review"),
                        "--work-root",
                        str(self.work_root.parent / "review-work" / run_id / coordinator_id),
                        "--poll-interval",
                        str(poll_interval),
                        "--coordinator-id",
                        coordinator_id,
                    ),
                )
                for coordinator_id in coordinator_ids
            )
            services = (*engines, *agents, global_agent, *reviews)
        return (("monitor", monitor), *services, ("control", control))

    def _resolve_runtime_roots(self, resolved: dict[str, object]) -> None:
        roots = deployment_runtime_roots(self.project_root, resolved)
        defaults = (
            ("runs_root", roots["control"]),
            ("queue_root", roots["queue"]),
            ("object_root", roots["objects"]),
            ("work_root", roots["engine_work"]),
        )
        for attribute, default in defaults:
            if getattr(self, attribute) is None:
                setattr(self, attribute, default.resolve())

    def _prepare_deployment_workers(self, *, run_id: str) -> tuple[int, ...]:
        assert self.runs_root is not None
        assert self.queue_root is not None
        assert self.object_root is not None
        assert self.work_root is not None

        active_siblings = []
        for state_path in sorted(self.runs_root.glob("*/run.json")):
            if state_path.parent.name == run_id:
                continue
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if state.get("status") not in {
                "completed", "failed", "cancelled", "paused", "suspended",
            }:
                active_siblings.append(state_path.parent.name)
        if active_siblings:
            raise RuntimeError(
                "deployment has another nonterminal ADE Run; refusing to "
                f"fence its workers: {', '.join(active_siblings)}"
            )

        roots = tuple(
            str(path.resolve())
            for path in (
                self.runs_root,
                self.queue_root,
                self.object_root,
                self.work_root,
                self.work_root.parent / "review",
                self.work_root.parent / "review-work",
            )
        )
        stale_pids = self._deployment_worker_pids(roots)
        for state_path in sorted(self.runs_root.glob("*/run.json")):
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("status") != "suspended":
                continue
            if stale_pids or self._has_processing_engine_work(
                self.queue_root, state_path.parent.name,
            ):
                raise RuntimeError(
                    "suspended Run still has live deployment processes or unfinished "
                    "Engine claims; wait for drain or explicitly fence orphaned work "
                    "before resume"
                )
        for pid in stale_pids:
            self._terminate_deployment_worker(pid)
        return stale_pids

    @staticmethod
    def _has_processing_engine_work(queue_root: Path, run_id: str) -> bool:
        # A trainer can outlive its heartbeat owner. Missing local worker PIDs
        # therefore do not authorize expiry/reclamation of an orphaned claim.
        for path in (Path(queue_root) / "processing").glob("*.json"):
            if (Path(queue_root) / "receipts" / path.name).is_file():
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("run_id") == run_id:
                return True
        return False

    def _write_worker_provenance(
        self,
        *,
        run_id: str,
        fenced_worker_pids: tuple[int, ...],
        processes: list[tuple[str, subprocess.Popen]],
        service_commands: dict[str, tuple[str, ...]],
        resume: bool,
    ) -> None:
        assert self.runs_root is not None
        try:
            result = subprocess.run(
                ("git", "-C", str(self.project_root), "rev-parse", "HEAD"),
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            git_commit = result.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            git_commit = None
        write_json_atomic(
            self.runs_root / run_id / "reports" / "worker-provenance.json",
            {
                "schema_version": "ade.worker_provenance.v1",
                "run_id": run_id,
                "event": "resume" if resume else "start",
                "git_commit": git_commit,
                "fenced_worker_pids": list(fenced_worker_pids),
                "workers": [
                    {
                        "name": name,
                        "pid": getattr(process, "pid", None),
                        "command": list(service_commands[name]),
                    }
                    for name, process in processes
                ],
            },
        )

    @staticmethod
    def _deployment_worker_pids(roots: tuple[str, ...]) -> tuple[int, ...]:
        found: list[int] = []
        for entry in Path("/proc").glob("[0-9]*"):
            try:
                pid = int(entry.name)
                raw = (entry / "cmdline").read_bytes()
            except (OSError, ValueError):
                continue
            argv = tuple(item.decode("utf-8", "replace") for item in raw.split(b"\0") if item)
            if RunSupervisor._is_deployment_process(argv, roots):
                found.append(pid)
        return tuple(sorted(found))

    @staticmethod
    def _is_deployment_process(argv: tuple[str, ...], roots: tuple[str, ...]) -> bool:
        if not argv or "ade.harness.cli" not in argv:
            return False
        module_index = argv.index("ade.harness.cli")
        if module_index + 2 >= len(argv):
            return False
        component, action = argv[module_index + 1:module_index + 3]
        return (
            (
                action == "worker"
                and component in {"engine", "agent", "review"}
            )
            or (action == "run" and component in {"monitor", "control"})
        ) and any(root in argv for root in roots)

    @staticmethod
    def _terminate_deployment_worker(pid: int) -> None:
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            return
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
