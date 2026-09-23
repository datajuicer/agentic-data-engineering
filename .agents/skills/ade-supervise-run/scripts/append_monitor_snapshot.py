#!/usr/bin/env python3
"""Append one revision-bounded ADE observation to a Run's monitor.md."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
from typing import Any


TERMINAL_RUN_STATUSES = {"cancelled", "completed", "failed"}
TERMINAL_PLAN_STATUSES = {"cancelled", "completed", "failed", "rejected"}
TERMINAL_CONTROL_STATUSES = {"cancelled", "finished_early"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture one ADE run observe result and append monitor.md."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--master-ip", required=True)
    parser.add_argument(
        "--mode",
        choices=("normal_wait", "diagnostic", "terminal"),
        required=True,
    )
    parser.add_argument(
        "--health",
        help="Optional Agent judgment: normal or a concise anomaly plus evidence path.",
    )
    parser.add_argument("--ade-executable", default="ade")
    parser.add_argument("--snapshot-time")
    parser.add_argument("--print-summary", action="store_true")
    return parser


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _parse_time(value: str | None) -> datetime:
    if value is None:
        return _utc_now()
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("--snapshot-time must include a timezone")
    return parsed.astimezone(timezone.utc).replace(microsecond=0)


def _format_time(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _capture_observe(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        args.ade_executable,
        "--project-root",
        str(args.project_root),
        "--runs-root",
        str(args.control_root),
        "run",
        "observe",
        args.run_id,
        "--queue-root",
        str(args.queue_root),
    ]
    result = subprocess.run(
        command,
        cwd=args.project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _cell(value: object | None) -> str:
    if value is None or value == "":
        return "none"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _scope(item: dict[str, Any]) -> tuple[object, object, object]:
    scope = item.get("scope") or {}
    return (
        item.get("coordinator_id", scope.get("coordinator_id")),
        item.get("plan_id", scope.get("plan_id")),
        item.get("trial_id", scope.get("trial_id")),
    )


def _active_work(
    state: dict[str, Any],
    coordinator_id: str,
    plan_id: str | None,
    trial_id: str | None,
) -> str:
    active: list[str] = []
    collections = (
        ("Agent", state.get("active_agent_calls", ()), "role", "attempt_id"),
        ("Engine", state.get("active_engine_commands", ()), "kind", "command_id"),
        ("Review", state.get("active_review_commands", ()), None, "command_id"),
    )
    for label, items, kind_key, id_key in collections:
        for item in items:
            item_coordinator, item_plan, item_trial = _scope(item)
            if item_coordinator != coordinator_id:
                continue
            if item_plan != plan_id or item_trial != trial_id:
                continue
            kind = item.get(kind_key) if kind_key else None
            name = f"{label}:{kind}" if kind else label
            active.append(f"{name}:{item.get(id_key) or 'unknown'}")
    return ", ".join(active) or "none"


def _timeline_rows(path: Path, revision: int) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        row
        for row in rows
        if row.get("revision") and int(row["revision"]) <= revision
    ]


def _progress(
    rows: list[dict[str, str]],
    coordinator_id: str,
    plan_id: str | None,
    trial_id: str | None,
) -> str:
    matching = [
        row
        for row in rows
        if row.get("coordinator_id") == coordinator_id
        and (row.get("plan_id") or None) == plan_id
        and (row.get("trial_id") or None) == trial_id
    ]
    if not matching:
        return "none"
    latest = max(matching, key=lambda row: int(row["revision"]))
    return f"rev-{latest['revision']}:{latest.get('transition') or 'state'}"


def _coordinator_state(
    state: dict[str, Any], coordinator: dict[str, Any], plans: list[dict[str, Any]]
) -> str:
    if state.get("status") in TERMINAL_RUN_STATUSES:
        return "terminal"
    if coordinator.get("control_status") in TERMINAL_CONTROL_STATUSES:
        return "terminal"
    coordinator_id = coordinator["coordinator_id"]
    active_items = (
        *state.get("active_agent_calls", ()),
        *state.get("active_engine_commands", ()),
        *state.get("active_review_commands", ()),
    )
    if any(_scope(item)[0] == coordinator_id for item in active_items):
        return "running"
    if any(
        slot.get("coordinator_id") == coordinator_id
        for slot in state.get("planning_queue", ())
    ):
        return "planning"
    if any(plan.get("status") not in TERMINAL_PLAN_STATUSES for plan in plans):
        return "running"
    if coordinator.get("kind") == "bootstrap":
        return "drained" if plans else "waiting"
    if len(plans) >= int(coordinator.get("effective_plan_limit") or 0):
        return "drained"
    return "waiting"


def _recovery(state: dict[str, Any]) -> str:
    recovery = state.get("recovery")
    if not recovery:
        return "none"
    fields = (
        recovery.get("status") or recovery.get("phase") or "active",
        recovery.get("logical_work_ref"),
        recovery.get("old_attempt_id"),
        recovery.get("new_attempt_id"),
    )
    return "; ".join(str(value) for value in fields if value)


def _results(run_dir: Path, revision: int) -> list[str]:
    path = run_dir / "reports" / "results.csv"
    if not path.is_file():
        return ["- none"]
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lines = []
    for row in rows:
        row_revision = row.get("revision")
        if row_revision and int(row_revision) > revision:
            continue
        statuses = ", ".join(
            f"{name}={row.get(name) or 'none'}"
            for name in ("online_status", "offline_status", "operator_status")
        )
        lines.append(
            f"- `{row.get('subject_ref') or 'unknown'}`: "
            f"phase={row.get('trial_phase') or 'none'}, "
            f"outcome={row.get('trial_outcome') or 'none'}, {statuses}"
        )
    return lines or ["- none"]


def _tracking_health(run_dir: Path) -> tuple[str, str, bool]:
    health_path = run_dir / "tracking" / "run-monitor-health.json"
    if not health_path.is_file():
        health = f"unavailable ({health_path})"
        healthy = False
    else:
        payload = _read_json(health_path)
        tracking = payload.get("tracking") or {}
        monitor_healthy = (
            payload.get("process_status") == "running"
            and tracking.get("status") == "healthy"
        )
        health = (
            f"process={payload.get('process_status') or 'unknown'}, "
            f"tracking={tracking.get('status') or 'unknown'}, "
            f"local_step={tracking.get('last_local_step')}, "
            f"remote_step={tracking.get('last_remote_step')}, "
            f"remote_checked={tracking.get('last_remote_checked_at') or 'none'}, "
            f"remote_error={tracking.get('remote_error') or 'none'} "
            f"({health_path})"
        )
        healthy = monitor_healthy
    reports = sorted(run_dir.glob("tracking/**/reconciliation-latest.json"))
    reconciliation_items = []
    for path in reports:
        payload = _read_json(path)
        pending = int(payload.get("pending_retry") or 0)
        missing = int(payload.get("remote_missing") or 0)
        healthy = healthy and pending == 0 and missing == 0
        reconciliation_items.append(
            f"evaluation(local={int(payload.get('local_requests') or 0)}, "
            f"attempted={int(payload.get('attempted') or 0)}, "
            f"published={int(payload.get('published') or 0)}, "
            f"pending={pending}, remote_missing={missing}, "
            f"error={payload.get('remote_check_error') or 'none'}) ({path})"
        )
    seed_status_path = (
        run_dir / "tracking" / "seed-import" / "history-status.json"
    )
    if seed_status_path.is_file():
        seed_status = _read_json(seed_status_path)
        status = seed_status.get("status") or "unknown"
        healthy = healthy and status in {
            "published",
            "disabled",
            "not_applicable",
        }
        reconciliation_items.append(
            f"seed-history(status={status}, "
            f"error={seed_status.get('error_type') or 'none'}) "
            f"({seed_status_path})"
        )
    reconciliation = ", ".join(reconciliation_items) or "none"
    return health, reconciliation, healthy


def _render(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    revision: int,
    snapshot_time: datetime,
    run_dir: Path,
) -> str:
    last_transition = state.get("last_transition") or {}
    lines = [
        f"## Snapshot — {_format_time(snapshot_time)}",
        "",
        f"- Run: `{args.run_id}`",
        (
            f"- Status: `{state.get('status')}`; revision: `{revision}`; "
            f"bootstrap: `{(state.get('bootstrap') or {}).get('status') or 'none'}`"
        ),
        (
            f"- Last transition: `{last_transition.get('kind') or 'none'}`; "
            f"subject: `{last_transition.get('subject_ref') or 'none'}`"
        ),
        f"- Recovery: `{_recovery(state)}`",
    ]
    timeline = _timeline_rows(run_dir / "reports" / "timeline.csv", revision)
    plans = state.get("plans", [])
    trials = state.get("trials", [])
    for coordinator in sorted(
        state.get("coordinators", []), key=lambda value: value["coordinator_id"]
    ):
        coordinator_id = coordinator["coordinator_id"]
        coordinator_plans = sorted(
            (plan for plan in plans if plan["coordinator_id"] == coordinator_id),
            key=lambda value: value["plan_id"],
        )
        budget = (
            "bootstrap"
            if coordinator.get("kind") == "bootstrap"
            else f"{len(coordinator_plans)}/{coordinator.get('effective_plan_limit')}"
        )
        lines.extend(
            [
                "",
                f"### Coordinator `{coordinator_id}`",
                "",
                f"- Control: `{coordinator.get('control_status') or 'none'}`",
                f"- Plan budget: `{budget}`",
                (
                    "- Current state: `"
                    f"{_coordinator_state(state, coordinator, coordinator_plans)}`"
                ),
                "",
                (
                    "| Plan | Plan status | Trial | Trial phase | Outcome | "
                    "Active work | Latest durable progress |"
                ),
                "|---|---|---|---|---|---|---|",
            ]
        )
        if not coordinator_plans:
            lines.append("| none | none | none | none | none | none | none |")
            continue
        for plan in coordinator_plans:
            plan_trials = sorted(
                (
                    trial
                    for trial in trials
                    if trial["coordinator_id"] == coordinator_id
                    and trial["plan_id"] == plan["plan_id"]
                ),
                key=lambda value: value["trial_id"],
            )
            if not plan_trials:
                lines.append(
                    "| {plan} | {status} | none | none | none | {active} | {progress} |".format(
                        plan=_cell(plan["plan_id"]),
                        status=_cell(plan.get("status")),
                        active=_cell(
                            _active_work(
                                state, coordinator_id, plan["plan_id"], None
                            )
                        ),
                        progress=_cell(
                            _progress(
                                timeline, coordinator_id, plan["plan_id"], None
                            )
                        ),
                    )
                )
                continue
            for trial in plan_trials:
                lines.append(
                    "| {plan} | {status} | {trial} | {phase} | {outcome} | {active} | {progress} |".format(
                        plan=_cell(plan["plan_id"]),
                        status=_cell(plan.get("status")),
                        trial=_cell(trial["trial_id"]),
                        phase=_cell(trial.get("phase")),
                        outcome=_cell(trial.get("outcome")),
                        active=_cell(
                            _active_work(
                                state,
                                coordinator_id,
                                plan["plan_id"],
                                trial["trial_id"],
                            )
                        ),
                        progress=_cell(
                            _progress(
                                timeline,
                                coordinator_id,
                                plan["plan_id"],
                                trial["trial_id"],
                            )
                        ),
                    )
                )
    tracking_health, reconciliation, monitor_healthy = _tracking_health(run_dir)
    health = args.health
    if health is None:
        status = state.get("status")
        if status in {"failed", "recovering", "suspended"}:
            health = f"anomaly: Run status={status}"
        elif not monitor_healthy and args.mode != "terminal":
            health = "anomaly: internal Run Monitor is not healthy"
        else:
            health = "normal"
    next_snapshot = (
        "none"
        if args.mode == "terminal"
        else _format_time(snapshot_time + timedelta(seconds=600))
    )
    lines.extend(
        [
            "",
            f"### Accepted results at revision {revision}",
            "",
            *_results(run_dir, revision),
            "",
            "### Health and schedule",
            "",
            f"- Health: `{health}`",
            f"- Run Monitor: `{tracking_health}`",
            f"- W&B reconciliation: `{reconciliation}`",
            f"- Mode: `{args.mode}`",
            f"- Next snapshot: `{next_snapshot}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = _parser().parse_args()
    args.project_root = args.project_root.resolve()
    args.control_root = args.control_root.resolve()
    args.queue_root = args.queue_root.resolve()
    snapshot_time = _parse_time(args.snapshot_time)
    observe = _capture_observe(args)
    if observe.get("run_id") != args.run_id:
        raise ValueError("run observe returned a different Run ID")
    revision = int(observe["revision"])
    run_dir = args.control_root / args.run_id
    revision_path = run_dir / "state" / "revisions" / f"rev-{revision}" / "run.json"
    state = _read_json(revision_path)
    if state.get("run_id") != args.run_id or int(state.get("revision", -1)) != revision:
        raise ValueError("immutable RunState does not match run observe")

    monitor_path = run_dir / "monitor.md"
    if not monitor_path.exists() or monitor_path.stat().st_size == 0:
        header = "\n".join(
            [
                f"# ADE Run Monitor — {args.run_id}",
                "",
                f"- Experiment: `{args.experiment}`",
                f"- Deployment: `{args.deployment}`; Master IP: `{args.master_ip}`",
                f"- Started: `{_format_time(snapshot_time)}`",
                "- Cadence: 10 minutes",
                "",
            ]
        )
    else:
        header = ""
    snapshot = _render(
        args=args,
        state=state,
        revision=revision,
        snapshot_time=snapshot_time,
        run_dir=run_dir,
    )
    with monitor_path.open("a", encoding="utf-8") as handle:
        handle.write(header)
        handle.write(snapshot)
    if args.print_summary:
        print(
            f"{_format_time(snapshot_time)} revision={revision} "
            f"status={state.get('status')} monitor={monitor_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
