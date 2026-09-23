"""Backend-neutral construction of exact training-run tracking identities."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from ade.engine.telemetry.training import TrainingRunAuditRef
from ade.engine.storage.atomic import write_json_atomic


WANDB_API_TIMEOUT_SECONDS = 30
WANDB_INIT_TIMEOUT_SECONDS = 60.0
WANDB_FINISH_TIMEOUT_SECONDS = 120.0


def bounded_wandb_settings(wandb_module: Any) -> Any | None:
    """Bound SDK startup and final upload without changing publication identity."""
    settings_type = getattr(wandb_module, "Settings", None)
    if not callable(settings_type):
        return None
    return settings_type(
        init_timeout=WANDB_INIT_TIMEOUT_SECONDS,
        finish_timeout=WANDB_FINISH_TIMEOUT_SECONDS,
        finish_timeout_raises=True,
    )


def build_training_tracking(
    *,
    settings: Mapping[str, Any],
    backend: str,
    run_dir: str | Path,
    run_group: str,
    coordinator_id: str,
    plan_id: str,
    trial_id: str,
    logical_command_id: str,
    attempt_id: str,
    fork_lineage: Mapping[str, Any] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    if not bool(settings.get("enabled", False)):
        return {}, {"enabled": False, "backend": backend}
    project = _project_for_run(settings, current_run_id=run_group, fork_lineage=fork_lineage)
    mode = str(settings.get("mode") or "offline").strip().lower()
    if mode not in {"offline", "online"}:
        raise ValueError("training telemetry mode must be offline or online")
    entity_env = str(settings.get("entity_env") or "WANDB_ENTITY")
    base_url_env = str(settings.get("base_url_env") or "WANDB_BASE_URL")
    entity = str(settings.get("entity") or os.environ.get(entity_env) or "").strip()
    base_url = str(settings.get("base_url") or os.environ.get(base_url_env) or "").strip()
    if base_url and not urlparse(base_url).hostname:
        raise ValueError("training telemetry base_url must include a hostname")
    group = run_group.strip()
    if not group:
        raise ValueError("training tracking Run group is required")
    safe_coordinator = _safe(coordinator_id)
    safe_plan = _safe(plan_id)
    safe_trial = _safe(trial_id)
    resolved_run_dir = Path(run_dir).resolve()
    safe_attempt = _safe(attempt_id)
    subject_ref = f"{group}/{safe_coordinator}/{safe_plan}/{safe_trial}"
    # The group and audit carry the full ADE Run identity.  Keep W&B's
    # human-facing name and external ID within the service's 128-character
    # limit; the ID retains the unique Run suffix and qualified workload tail.
    display_name = (
        f"{safe_coordinator}/{safe_plan}/{safe_trial}/training/{safe_attempt}"
    )
    run_id = _wandb_id(f"{subject_ref}/training/{safe_attempt}")[-128:]
    audit_root = (
        resolved_run_dir
        if resolved_run_dir.name == "audit"
        else resolved_run_dir / "engine_audit"
    )
    local_run_dir = audit_root / "wandb" / run_id
    environment = {
        "WANDB_PROJECT": project,
        "WANDB_RUN_GROUP": group,
        "WANDB_RUN_ID": run_id,
        "WANDB_NAME": display_name,
        "WANDB_RESUME": "allow",
        "WANDB_MODE": mode,
        "WANDB_DIR": str(local_run_dir),
        "WANDB_DISABLE_STATS": "true",
        "WANDB_INIT_TIMEOUT": str(int(WANDB_INIT_TIMEOUT_SECONDS)),
        "WANDB_FINISH_TIMEOUT": str(int(WANDB_FINISH_TIMEOUT_SECONDS)),
        # Training success is scientific authority; a bounded tracking flush
        # must not turn a completed training process into an Engine failure.
        "WANDB_FINISH_TIMEOUT_RAISES": "false",
    }
    if entity:
        environment["WANDB_ENTITY"] = entity
    if base_url:
        environment["WANDB_BASE_URL"] = base_url
    if mode == "online":
        environment.update(_wandb_no_proxy_environment(base_url))
    lineage_tags = _lineage_tags(fork_lineage)
    if lineage_tags:
        environment["WANDB_TAGS"] = ",".join(lineage_tags)
    audit_ref = TrainingRunAuditRef(
        backend=backend,
        base_url=base_url,
        entity=entity,
        project=project,
        run_id=run_id,
        trial_uid=subject_ref,
        local_run_dir=str(local_run_dir),
    )
    return environment, {
        "enabled": True,
        "mode": mode,
        "project": project,
        "entity": entity or None,
        "base_url": base_url or None,
        "run_id": run_id,
        "coordinator_id": safe_coordinator,
        "plan_id": safe_plan,
        "trial_id": safe_trial,
        "local_run_dir": str(local_run_dir),
        "name": environment["WANDB_NAME"],
        "group": group,
        "subject_ref": subject_ref,
        "logical_command_id": logical_command_id,
        "attempt_id": attempt_id,
        "fork_lineage": dict(fork_lineage or {}),
        "resume": "allow",
        "audit_ref": audit_ref.to_dict(),
    }


def audit_ref_from_dict(value: Mapping[str, Any]) -> TrainingRunAuditRef:
    return TrainingRunAuditRef(
        backend=_required(value, "backend"),
        base_url=str(value.get("base_url") or ""),
        entity=str(value.get("entity") or ""),
        project=_required(value, "project"),
        run_id=_required(value, "run_id"),
        trial_uid=_required(value, "trial_uid"),
        local_run_dir=_required(value, "local_run_dir"),
    )


def publish_evaluation_tracking(
    *,
    settings: Mapping[str, Any],
    command_id: str,
    run_id: str,
    coordinator_id: str,
    plan_id: str,
    trial_id: str,
    subject_kind: str,
    purpose: str,
    result: Mapping[str, Any],
    local_root: str | Path,
    artifact_position: Mapping[str, Any] | None = None,
    position_order: Sequence[int] | None = None,
    evaluation_ordinal: int | None = None,
    fork_lineage: Mapping[str, Any] | None = None,
    wandb_module: Any = None,
    defer_online: bool = False,
) -> dict[str, Any]:
    """Append one completed evaluation to its subject's stable W&B run.

    The local evaluation result is already a scientific fact when this function
    runs.  W&B is therefore a projection: a provider failure leaves a durable
    same-identity retry request instead of failing the Engine command.
    """
    if not bool(settings.get("enabled", False)):
        return {"enabled": False}
    if settings.get("provider") != "wandb":
        raise ValueError("evaluation tracking provider must be wandb")
    mode = str(settings.get("mode") or "offline").strip().lower()
    if mode not in {"offline", "online"}:
        raise ValueError("evaluation tracking mode must be offline or online")
    project = _project_for_run(settings, current_run_id=run_id, fork_lineage=fork_lineage)
    entity_env = str(settings.get("entity_env") or "WANDB_ENTITY")
    base_url_env = str(settings.get("base_url_env") or "WANDB_BASE_URL")
    api_key_env = str(settings.get("api_key_env") or "WANDB_API_KEY")
    entity = str(os.environ.get(entity_env) or "").strip()
    base_url = str(os.environ.get(base_url_env) or "").strip()
    api_key = str(os.environ.get(api_key_env) or "").strip()
    group = run_id.strip()
    if not group:
        raise ValueError("evaluation tracking Run group is required")
    safe_coordinator = _safe(coordinator_id)
    safe_plan = _safe(plan_id)
    safe_trial = _safe(trial_id)
    safe_purpose = _safe(purpose)
    safe_command = _safe(command_id)
    job_type, identity_scope = _evaluation_tracking_role(
        subject_kind=subject_kind,
        coordinator_id=safe_coordinator,
        plan_id=safe_plan,
        trial_id=safe_trial,
        purpose=safe_purpose,
    )
    full_identity_scope = f"{group}/{identity_scope}"
    tracking_id = _wandb_id(full_identity_scope)
    external_tracking_id = _wandb_id(
        f"{full_identity_scope}/evaluation-v10"
    )[-128:]
    display_name = identity_scope[-128:]
    local_dir = Path(local_root).resolve() / tracking_id
    # Imported source command IDs can be longer than the filesystem's single
    # component limit even though the remote W&B identity is already bounded.
    # Keep the stable command suffix for the local projection as well.
    record_dir = local_dir / "records" / safe_command[-128:]
    record_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir = record_dir / "wandb"
    result_path = record_dir / "evaluation-result.json"
    position_value = _artifact_position_value(artifact_position)
    training_step = (artifact_position or {}).get("train_step")
    if training_step is None and (artifact_position or {}).get("unit") != "epoch":
        training_step = position_value
    if evaluation_ordinal is not None and (
        type(evaluation_ordinal) is not int or evaluation_ordinal < 1
    ):
        raise ValueError("evaluation tracking ordinal must be a positive integer")
    position_marker = _tracking_position_marker(
        local_dir,
        position_value if position_value is not None else -1,
        purpose=safe_purpose,
    )
    status_path = record_dir / "tracking-status.json"
    if _tracking_marker_status(position_marker) == "published":
        write_json_atomic(
            status_path,
            {"schema_version": EVALUATION_TRACKING_SCHEMA_VERSION, "status": "published", "error_type": None},
        )
        return {
            "enabled": True,
            "provider": "wandb",
            "mode": mode,
            "project": project,
            "entity": entity or None,
            "group": group,
            "run_id": external_tracking_id,
            "name": display_name,
            "job_type": job_type,
            "result_artifact": _evaluation_artifact_name(
                tracking_id, safe_purpose, position_value
            ),
            "artifact_position": dict(artifact_position or {}),
            "fork_lineage": dict(fork_lineage or {}),
            "local_run_dir": str(wandb_dir),
            "status": "published",
        }
    existing_result = _read_json_mapping(result_path)
    effective_result = _prefer_evaluation_result(existing_result, result)
    # The evaluation result is the durable local source of truth.  Publish it
    # atomically before attempting the remote projection so a monitor restart
    # can always replay a complete payload.
    write_json_atomic(result_path, dict(effective_result))
    request_path = record_dir / "tracking-request.json"
    write_json_atomic(
        request_path,
        {
            "schema_version": EVALUATION_TRACKING_SCHEMA_VERSION,
            "settings": {
                key: settings[key]
                for key in (
                    "enabled",
                    "provider",
                    "mode",
                    "project",
                    "entity_env",
                    "base_url_env",
                    "api_key_env",
                )
                if key in settings
            },
            "command_id": command_id,
            "run_id": run_id,
            "coordinator_id": coordinator_id,
            "plan_id": plan_id,
            "trial_id": trial_id,
            "subject_kind": subject_kind,
            "purpose": purpose,
            "artifact_position": dict(artifact_position or {}),
            "checkpoint_step": training_step,
            "position_order": list(position_order) if position_order is not None else None,
            "evaluation_ordinal": evaluation_ordinal,
            "fork_lineage": dict(fork_lineage or {}),
        },
    )
    environment = {
        "WANDB_MODE": mode,
        "WANDB_PROJECT": project,
        "WANDB_RUN_GROUP": group,
        "WANDB_RUN_ID": external_tracking_id,
        "WANDB_NAME": display_name,
        "WANDB_RESUME": "allow",
        "WANDB_DIR": str(wandb_dir),
        "WANDB_DISABLE_STATS": "true",
    }
    if entity:
        environment["WANDB_ENTITY"] = entity
    if base_url:
        environment["WANDB_BASE_URL"] = base_url
    if api_key:
        environment["WANDB_API_KEY"] = api_key
    if mode == "online":
        environment.update(_wandb_no_proxy_environment(base_url))
    lineage_tags = _lineage_tags(fork_lineage)
    if lineage_tags:
        environment["WANDB_TAGS"] = ",".join(lineage_tags)
    # W&B rejects artifact names longer than 128 characters.  The Run,
    # command, trial, purpose, and full fork lineage remain authoritative in
    # the artifact metadata and durable tracking request; keep the unique
    # qualified tail for the external artifact identity.
    artifact_name = _evaluation_artifact_name(
        tracking_id, safe_purpose, position_value
    )
    ordered = _tracking_position_order(position_order, position_value)
    result_payload = {
        "enabled": True,
        "provider": "wandb",
        "mode": mode,
        "project": project,
        "entity": entity or None,
        "group": group,
        "run_id": external_tracking_id,
        "name": environment["WANDB_NAME"],
        "job_type": job_type,
        "result_artifact": artifact_name,
        "artifact_position": dict(artifact_position or {}),
        "fork_lineage": dict(fork_lineage or {}),
        "local_run_dir": str(wandb_dir),
    }
    if status_path.is_file():
        try:
            existing_status = json.loads(
                status_path.read_text(encoding="utf-8")
            ).get("status")
        except (OSError, ValueError, json.JSONDecodeError):
            existing_status = None
        if existing_status == "published":
            return {**result_payload, "status": "published"}
    if mode == "online" and defer_online:
        write_json_atomic(
            status_path,
            {"schema_version": EVALUATION_TRACKING_SCHEMA_VERSION, "status": "pending_retry", "error_type": None},
        )
        _write_tracking_position_marker(
            local_dir,
            position_value,
            status="pending_retry",
            purpose=safe_purpose,
        )
        return {**result_payload, "status": "pending_retry"}
    if ordered is not None and not _prior_tracking_positions_terminal(
        local_dir, ordered, position_value, purpose=safe_purpose
    ):
        write_json_atomic(
            status_path,
            {"schema_version": EVALUATION_TRACKING_SCHEMA_VERSION, "status": "pending_retry", "error_type": None},
        )
        _write_tracking_position_marker(
            local_dir,
            position_value,
            status="pending_retry",
            purpose=safe_purpose,
        )
        return {**result_payload, "status": "pending_retry"}
    publication_status = "pending_retry"
    publication_error_type: str | None = None
    try:
        if mode == "online" and not api_key:
            raise RuntimeError(
                f"{api_key_env} is required for online evaluation tracking"
            )
        with _EVALUATION_WANDB_LOCK:
            if _tracking_marker_status(position_marker) == "published":
                return {**result_payload, "status": "published"}
            previous = {name: os.environ.get(name) for name in environment}
            os.environ.update(environment)
            run = None
            exit_code = 1
            try:
                if wandb_module is None:
                    import wandb as wandb_module

                bounded_settings = bounded_wandb_settings(wandb_module)
                run = wandb_module.init(
                    project=project,
                    entity=entity or None,
                    group=group,
                    job_type=job_type,
                    name=environment["WANDB_NAME"],
                    id=external_tracking_id,
                    reinit="create_new",
                    resume="allow",
                    dir=str(wandb_dir),
                    tags=lineage_tags or None,
                    config={
                        "ade_run_id": run_id,
                        "engine_command_id": command_id,
                        "coordinator_id": coordinator_id,
                        "plan_id": plan_id,
                        "trial_id": trial_id,
                        "subject_kind": subject_kind,
                        "evaluation_purpose": purpose,
                        "tracking_role": job_type,
                        "schema_version": EVALUATION_TRACKING_SCHEMA_VERSION,
                        "checkpoint_step": training_step,
                        "checkpoint_unit": (
                            "train_step" if training_step is not None
                            else (artifact_position or {}).get("unit")
                        ),
                        "subject_ref": (
                            f"{run_id}/{coordinator_id}/{plan_id}/"
                            f"{'base' if subject_kind == 'base_model' else trial_id}"
                        ),
                        "fork_lineage": dict(fork_lineage or {}),
                        **{
                            key: value
                            for key, value in dict(fork_lineage or {}).items()
                            if key.startswith("imported_from_")
                            or key in {"source_command_id", "source_attempt_id"}
                        },
                    },
                    **(
                        {"settings": bounded_settings}
                        if bounded_settings is not None
                        else {}
                    ),
                )
                prefix = f"evaluation/{safe_purpose}"
                define_metric = getattr(run, "define_metric", None)
                if callable(define_metric):
                    define_metric(
                        f"{prefix}/*",
                        step_metric="evaluation/checkpoint_position",
                    )
                metrics: dict[str, int | float] = {f"{prefix}/completed": 1}
                effective_ordinal = evaluation_ordinal
                if (
                    effective_ordinal is None
                    and ordered is not None
                    and position_value is not None
                ):
                    effective_ordinal = ordered.index(position_value) + 1
                if training_step is not None:
                    metrics["evaluation/checkpoint_step"] = training_step
                    metrics["evaluation/checkpoint_position"] = training_step
                elif position_value is not None:
                    metrics["evaluation/checkpoint_position"] = position_value
                if effective_ordinal is not None:
                    metrics["evaluation/evaluation_ordinal"] = effective_ordinal
                for name, value in _evaluation_metrics(effective_result).items():
                    metrics[f"{prefix}/{_safe(name)}"] = value
                # Use a dense, monotonic history axis for W&B.  The actual
                # checkpoint position remains available in the metrics and
                # metadata; the history step is only the publication ordinal.
                evaluation_step = effective_ordinal
                if evaluation_step is None:
                    run.log(metrics)
                else:
                    run.log(metrics, step=evaluation_step)
                artifact = wandb_module.Artifact(
                    name=artifact_name,
                    type="evaluation-result",
                    metadata={
                        "ade_run_id": run_id,
                        "engine_command_id": command_id,
                        "coordinator_id": coordinator_id,
                        "plan_id": plan_id,
                        "trial_id": trial_id,
                        "subject_kind": subject_kind,
                        "purpose": purpose,
                        "artifact_position": dict(artifact_position or {}),
                        "fork_lineage": dict(fork_lineage or {}),
                    },
                )
                artifact.add_file(str(result_path), name="evaluation-result.json")
                run.log_artifact(artifact)
                exit_code = 0
                publication_status = "published"
            finally:
                try:
                    if run is not None:
                        run.finish(exit_code=exit_code)
                finally:
                    for name, value in previous.items():
                        if value is None:
                            os.environ.pop(name, None)
                        else:
                            os.environ[name] = value
    except Exception as error:
        publication_error_type = type(error).__name__
    write_json_atomic(
        status_path,
        {
            "schema_version": EVALUATION_TRACKING_SCHEMA_VERSION,
            "status": publication_status,
            "error_type": publication_error_type,
        },
    )
    if ordered is not None:
        _write_tracking_position_marker(
            local_dir,
            position_value,
            status=publication_status,
            purpose=safe_purpose,
        )
    elif publication_status == "published":
        _write_tracking_position_marker(
            local_dir, -1, status="published", purpose=safe_purpose
        )
    return {
        **result_payload,
        "status": publication_status,
        **(
            {"error_type": publication_error_type}
            if publication_error_type is not None
            else {}
        ),
    }


def retry_pending_evaluation_tracking(
    local_root: str | Path,
    *,
    wandb_module: Any = None,
    max_requests: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Retry durable W&B projections without recomputing an evaluation."""
    root = Path(local_root).resolve()
    pending: list[tuple[str, int, Path, Mapping[str, Any]]] = []
    for request_path in root.glob("*/records/*/tracking-request.json"):
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            status_path = request_path.with_name("tracking-status.json")
            status = (
                json.loads(status_path.read_text(encoding="utf-8"))
                if status_path.is_file()
                else {"status": "pending_retry"}
            )
            if status.get("status") == "published":
                continue
            position = _artifact_position_value(request.get("artifact_position"))
            pending.append(
                (
                    str(request_path.parents[2]),
                    (
                        request.get("evaluation_ordinal")
                        if type(request.get("evaluation_ordinal")) is int
                        else position or 0
                    ),
                    request_path,
                    request,
                )
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    selected = sorted(
        pending, key=lambda item: (item[0], item[1], str(item[2]))
    )
    if max_requests is not None:
        if max_requests < 1:
            raise ValueError("max_requests must be positive")
        selected = selected[:max_requests]
    results = []
    for _identity, _position, request_path, request in selected:
        try:
            result = json.loads(
                request_path.with_name("evaluation-result.json").read_text(
                    encoding="utf-8"
                )
            )
            results.append(
                publish_evaluation_tracking(
                    settings=request["settings"],
                    command_id=str(request["command_id"]),
                    run_id=str(request["run_id"]),
                    coordinator_id=str(request["coordinator_id"]),
                    plan_id=str(request["plan_id"]),
                    trial_id=str(request["trial_id"]),
                    subject_kind=str(request["subject_kind"]),
                    purpose=str(request["purpose"]),
                    result=result,
                    local_root=root,
                    artifact_position=request.get("artifact_position"),
                    position_order=request.get("position_order"),
                    evaluation_ordinal=request.get("evaluation_ordinal"),
                    fork_lineage=request.get("fork_lineage"),
                    wandb_module=wandb_module,
                )
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return tuple(results)


def reconcile_evaluation_tracking(
    local_root: str | Path,
    *,
    wandb_module: Any = None,
    max_requests: int | None = None,
    verify_remote: bool = True,
) -> dict[str, Any]:
    """Reconcile durable local projections without rerunning evaluation.

    The publisher remains the single writer.  Reconciliation only replays
    durable requests and records an audit receipt, so it cannot create a
    second upload protocol or interfere with an in-flight evaluation.
    """
    root = Path(local_root).resolve()
    before = tuple(root.glob("*/records/*/tracking-request.json"))
    remote_missing = 0
    remote_checked = 0
    remote_check_error: str | None = None
    api = None
    if wandb_module is None:
        try:
            import wandb as wandb_module
        except ImportError:
            wandb_module = None
    for request_path in before if verify_remote else ():
        status_path = request_path.with_name("tracking-status.json")
        status = _read_json_mapping(status_path) or {}
        if (
            status.get("status") != "published"
            or wandb_module is None
            or not callable(getattr(wandb_module, "Api", None))
        ):
            continue
        request = _read_json_mapping(request_path)
        if request is None:
            continue
        try:
            settings = request["settings"]
            if api is None:
                base_url = str(
                    os.environ.get(
                        str(settings.get("base_url_env") or "WANDB_BASE_URL")
                    )
                    or ""
                ).strip()
                api_kwargs = {"overrides": {"base_url": base_url}} if base_url else {}
                api = wandb_module.Api(
                    timeout=WANDB_API_TIMEOUT_SECONDS,
                    **api_kwargs,
                )
            project = _project_for_run(
                settings,
                current_run_id=str(request["run_id"]),
                fork_lineage=request.get("fork_lineage"),
            )
            _job_type, identity_scope = _evaluation_tracking_role(
                subject_kind=str(request["subject_kind"]),
                coordinator_id=_safe(str(request["coordinator_id"])),
                plan_id=_safe(str(request["plan_id"])),
                trial_id=_safe(str(request["trial_id"])),
                purpose=_safe(str(request["purpose"])),
            )
            external_id = _wandb_id(
                f"{request['run_id']}/{identity_scope}/evaluation-v10"
            )[-128:]
            entity_env = str(settings.get("entity_env") or "WANDB_ENTITY")
            entity = str(os.environ.get(entity_env) or "").strip()
            path = (
                f"{entity}/{project}/{external_id}"
                if entity
                else f"{project}/{external_id}"
            )
            api.run(path)
            remote_checked += 1
        except Exception as error:
            remote_check_error = type(error).__name__
            message = str(error).lower()
            confirmed_missing = any(
                token in message
                for token in ("404", "not found", "does not exist", "could not find")
            )
            if not confirmed_missing:
                continue
            remote_missing += 1
            write_json_atomic(
                status_path,
                {
                    "schema_version": EVALUATION_TRACKING_SCHEMA_VERSION,
                    "status": "pending_retry",
                    "error_type": remote_check_error,
                },
            )
            request_position = _artifact_position_value(
                request.get("artifact_position")
            )
            marker = _tracking_position_marker(
                request_path.parents[2],
                request_position if request_position is not None else -1,
                purpose=_safe(str(request["purpose"])),
            )
            marker.unlink(missing_ok=True)
    results = retry_pending_evaluation_tracking(
        root,
        wandb_module=wandb_module,
        max_requests=max_requests,
    )
    published = sum(item.get("status") == "published" for item in results)
    pending = sum(
        (_read_json_mapping(path.with_name("tracking-status.json")) or {}).get(
            "status"
        )
        != "published"
        for path in before
    )
    report = {
        "schema_version": EVALUATION_TRACKING_SCHEMA_VERSION,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "local_requests": len(before),
        "attempted": len(results),
        "published": published,
        "pending_retry": pending,
        "remote_checked": remote_checked,
        "remote_missing": remote_missing,
        "remote_check_error": remote_check_error,
    }
    write_json_atomic(root / "reconciliation-latest.json", report)
    return report


def republish_seed_wandb_history(
    *,
    source_run_id: str,
    source_project_id: str | None = None,
    target_run_id: str,
    settings: Mapping[str, Any],
    selected_subject_refs: Sequence[str],
    wandb_module: Any = None,
    max_new_runs: int | None = None,
    known_published_source_run_ids: Sequence[str] = (),
) -> tuple[dict[str, Any], ...]:
    """Project non-evaluation source W&B history into the target Run group.

    A Run Seed is a visual starting point as well as a collection of
    accepted ADE facts.  Evaluation replay covers the latter, but it cannot
    reproduce the charts from the source monitor and training runs.  Read the
    source project through the W&B API and copy scalar history into stable
    target-group runs. Source W&B artifacts are deliberately excluded: the Run
    Seed materializer already owns accepted artifacts and checkpoint state is
    not inherited. Durable evaluation replay owns evaluation projections, so
    evaluation-class source runs are excluded here. The source is never
    modified.
    """
    if not bool(settings.get("enabled", False)):
        return ()
    if settings.get("provider") != "wandb":
        raise ValueError("Run Seed W&B history requires the wandb provider")
    mode = str(settings.get("mode") or "offline").strip().lower()
    if mode != "online":
        raise ValueError("Run Seed W&B history requires online tracking")
    if not source_run_id.strip() or not target_run_id.strip():
        raise ValueError("source and target Run IDs are required")
    if max_new_runs is not None and max_new_runs < 1:
        raise ValueError("max_new_runs must be positive")
    project = _project_for_run(
        settings, current_run_id=target_run_id, fork_lineage=None
    )
    entity_env = str(settings.get("entity_env") or "WANDB_ENTITY")
    base_url_env = str(settings.get("base_url_env") or "WANDB_BASE_URL")
    api_key_env = str(settings.get("api_key_env") or "WANDB_API_KEY")
    entity = str(os.environ.get(entity_env) or "").strip()
    base_url = str(os.environ.get(base_url_env) or "").strip()
    api_key = str(os.environ.get(api_key_env) or "").strip()
    if not api_key:
        raise RuntimeError(f"{api_key_env} is required for online W&B history")

    if wandb_module is None:
        import wandb as wandb_module

    previous = {
        name: os.environ.get(name)
        for name in (
            "WANDB_PROJECT",
            "WANDB_ENTITY",
            "WANDB_BASE_URL",
            "WANDB_API_KEY",
            "WANDB_MODE",
            "WANDB_RUN_GROUP",
            "WANDB_DIR",
        )
    }
    environment = {
        "WANDB_PROJECT": project,
        "WANDB_MODE": "online",
        "WANDB_RUN_GROUP": target_run_id,
        "WANDB_API_KEY": api_key,
        "WANDB_DISABLE_STATS": "true",
    }
    if entity:
        environment["WANDB_ENTITY"] = entity
    if base_url:
        environment["WANDB_BASE_URL"] = base_url
        environment.update(_wandb_no_proxy_environment(base_url))
    os.environ.update(environment)
    try:
        api_kwargs = {"overrides": {"base_url": base_url}} if base_url else {}
        api = wandb_module.Api(
            timeout=WANDB_API_TIMEOUT_SECONDS,
            **api_kwargs,
        )
        source_project = source_project_id or source_run_id
        source_path = f"{entity}/{source_project}" if entity else source_project
        source_runs_by_id = {
            str(source.id): source
            for source in api.runs(
                source_path, filters={"group": source_run_id}, per_page=100
            )
        }
        source_runs = tuple(source_runs_by_id.values())
        if not source_runs:
            raise ValueError(
                f"Run Seed W&B project has no runs in group {source_run_id}"
            )
        results: list[dict[str, Any]] = []
        known_published = set(known_published_source_run_ids)
        processed_unknown = 0
        for source in sorted(source_runs, key=lambda item: str(item.id)):
            source_id = str(source.id)
            source_job_type = str(source.job_type or "")
            if source_job_type in {
                "base-evaluation",
                "bootstrap-evaluation",
                "trial-evaluation",
                "operator-evaluation",
            }:
                continue
            source_config = dict(source.config)
            bound_subject = next(
                (
                    str(source_config[key])
                    for key in ("subject_ref", "trial_uid")
                    if isinstance(source_config.get(key), str)
                    and str(source_config[key]).startswith(f"{source_run_id}/")
                ),
                None,
            )
            if bound_subject is not None and not any(
                bound_subject == selected
                or bound_subject.startswith(f"{selected}/")
                for selected in selected_subject_refs
            ):
                continue
            if source_job_type == "training" and bound_subject is None:
                continue
            target_id = _wandb_id(
                f"{target_run_id}/seed-import/{source_id}"
            )[-128:]
            target_name = f"imported-history/{source.name}"[-128:]
            if source_id in known_published:
                results.append(
                    {
                        "source_run_id": source_id,
                        "target_run_id": target_id,
                        "status": "already_published",
                    }
                )
                continue
            if (
                max_new_runs is not None
                and processed_unknown >= max_new_runs
            ):
                results.append(
                    {
                        "source_run_id": source_id,
                        "target_run_id": target_id,
                        "status": "pending_retry",
                    }
                )
                continue
            # Bound both remote existence checks and publications. This also
            # recovers a crash after remote finish but before the local status
            # write without scanning every source run in one monitor cycle.
            processed_unknown += 1
            # A retry/resume must not append the same source history twice.
            try:
                api.run(f"{entity}/{project}/{target_id}" if entity else f"{project}/{target_id}")
                results.append(
                    {
                        "source_run_id": source_id,
                        "target_run_id": target_id,
                        "status": "already_published",
                    }
                )
                continue
            except Exception:
                pass
            bounded_settings = bounded_wandb_settings(wandb_module)
            run = wandb_module.init(
                project=project,
                entity=entity or None,
                group=target_run_id,
                job_type="seed-import",
                name=target_name,
                id=target_id,
                reinit="create_new",
                resume="allow",
                config={
                    "ade_run_id": target_run_id,
                    "seed_import": True,
                    "source_wandb_project": source_run_id,
                    "source_wandb_project_id": source_project,
                    "source_wandb_run_id": source_id,
                    "source_wandb_name": str(source.name or ""),
                    "source_wandb_job_type": str(source.job_type or ""),
                    "source_wandb_config": source_config,
                },
                tags=(
                    "seed-import",
                    f"source-run:{source_id[-53:]}",
                ),
                **(
                    {"settings": bounded_settings}
                    if bounded_settings is not None
                    else {}
                ),
            )
            rows = 0
            finish_attempted = False
            try:
                for row in source.scan_history(page_size=1000):
                    if not isinstance(row, Mapping):
                        continue
                    payload = {
                        str(key): value
                        for key, value in row.items()
                        if key != "_step" and not str(key).startswith("_")
                    }
                    step = row.get("evaluation/checkpoint_step", row.get("_step"))
                    if payload:
                        if isinstance(step, int) and not isinstance(step, bool):
                            run.log(payload, step=step)
                        else:
                            run.log(payload)
                        rows += 1
                finish_attempted = True
                run.finish(exit_code=0)
            except Exception:
                if not finish_attempted:
                    try:
                        run.finish(exit_code=1)
                    except Exception:
                        pass
                raise
            results.append(
                {
                    "source_run_id": source_id,
                    "target_run_id": target_id,
                    "source_name": str(source.name or ""),
                    "rows": rows,
                    "status": "published",
                }
            )
        missing: list[str] = []
        for result in results:
            if result.get("status") != "published":
                continue
            target_id = str(result["target_run_id"])
            try:
                api.run(
                    f"{entity}/{project}/{target_id}"
                    if entity
                    else f"{project}/{target_id}"
                )
            except Exception:
                missing.append(str(result["source_run_id"]))
        if missing:
            raise RuntimeError(
                "Run Seed W&B history import is incomplete; missing target runs for "
                + ", ".join(sorted(missing))
            )
        return tuple(results)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _evaluation_tracking_role(
    *,
    subject_kind: str,
    coordinator_id: str,
    plan_id: str,
    trial_id: str,
    purpose: str,
) -> tuple[str, str]:
    if subject_kind not in {"base_model", "p000_baseline", "search_trial"}:
        raise ValueError(f"unsupported evaluation subject kind: {subject_kind}")
    if purpose == "operator_test":
        return (
            "operator-evaluation",
            (
                f"{coordinator_id}/{plan_id}/base-model/operator-evaluation"
                if subject_kind == "base_model"
                else f"{coordinator_id}/{plan_id}/{trial_id}/operator-evaluation"
            ),
        )
    if subject_kind == "base_model":
        return (
            "base-evaluation",
            f"{coordinator_id}/{plan_id}/base-model/evaluation",
        )
    if subject_kind == "p000_baseline":
        return (
            "bootstrap-evaluation",
            f"{coordinator_id}/{plan_id}/{trial_id}/evaluation",
        )
    return "trial-evaluation", f"{coordinator_id}/{plan_id}/{trial_id}/evaluation"


def _evaluation_metrics(result: Mapping[str, Any]) -> dict[str, float]:
    """Project metrics without changing the agent-visible evaluation result."""
    metrics: dict[str, float] = {}
    for name, value in result.items():
        if name in {"status", "results", "aggregate_metrics"}:
            continue
        if _is_number(value):
            metrics[str(name)] = float(value)
    aggregate = result.get("aggregate_metrics")
    if isinstance(aggregate, Mapping):
        for name, value in aggregate.items():
            if _is_number(value):
                metrics[str(name)] = float(value)
    rows = result.get("results")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            row_metrics = row.get("metrics")
            if not isinstance(row_metrics, Mapping):
                continue
            for name, value in row_metrics.items():
                if _is_number(value):
                    metrics[name] = float(value)
    return metrics


def _evaluation_artifact_name(
    tracking_id: str,
    purpose: str,
    position: int | None,
) -> str:
    position_name = "unpositioned" if position is None else f"position-{position:08d}"
    return f"{tracking_id}-{purpose}-{position_name}-evaluation-result"[-128:]


def _artifact_position_value(position: Mapping[str, Any] | None) -> int | None:
    if not isinstance(position, Mapping):
        return None
    value = position.get("value")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def mark_evaluation_tracking_position_terminal(
    *,
    settings: object,
    run_id: str,
    coordinator_id: str,
    plan_id: str,
    trial_id: str,
    subject_kind: str,
    purpose: str,
    local_root: str | Path,
    artifact_position: Mapping[str, Any] | None,
    position_order: object,
    status: str,
) -> None:
    if not isinstance(settings, Mapping) or not bool(settings.get("enabled", False)):
        return
    position_value = _artifact_position_value(artifact_position)
    ordered = _tracking_position_order(position_order, position_value)
    if ordered is None:
        return
    group = run_id.strip()
    _, identity_scope = _evaluation_tracking_role(
        subject_kind=subject_kind,
        coordinator_id=_safe(coordinator_id),
        plan_id=_safe(plan_id),
        trial_id=_safe(trial_id),
        purpose=_safe(purpose),
    )
    tracking_id = _wandb_id(f"{group}/{identity_scope}")
    local_dir = Path(local_root).resolve() / tracking_id
    _write_tracking_position_marker(
        local_dir, position_value, status=status, purpose=_safe(purpose)
    )


def _tracking_position_order(
    value: object,
    position: int | None,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("evaluation tracking position order must be a sequence")
    order = tuple(value)
    if (
        not order
        or any(type(item) is not int or item < 0 for item in order)
        or tuple(sorted(set(order))) != order
        or position not in order
    ):
        raise ValueError(
            "evaluation tracking position order must contain the current non-negative position in ascending unique order"
        )
    return order


def _tracking_position_marker(
    local_dir: Path,
    position: int,
    *,
    purpose: str = "default",
) -> Path:
    return local_dir / "publication-order" / f"{_safe(purpose)}-position-{position:08d}.json"


def _tracking_marker_status(marker: Path) -> str | None:
    if not marker.is_file():
        return None
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    status = value.get("status") if isinstance(value, Mapping) else None
    return str(status) if isinstance(status, str) else None


def _read_json_mapping(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _prefer_evaluation_result(
    existing: Mapping[str, Any] | None,
    incoming: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Prevent partial/empty retries from replacing a complete result."""
    if not existing:
        return incoming
    existing_status = str(existing.get("status") or "complete")
    incoming_status = str(incoming.get("status") or "complete")
    existing_score = existing.get("score")
    incoming_score = incoming.get("score")
    existing_complete = existing_status == "complete" and _is_number(existing_score)
    incoming_complete = incoming_status == "complete" and _is_number(incoming_score)
    if existing_complete and not incoming_complete:
        return existing
    if existing_complete and incoming_complete:
        return existing
    if existing_status == "complete" and incoming_status != "complete":
        return existing
    return incoming


def _write_tracking_position_marker(
    local_dir: Path,
    position: int | None,
    *,
    status: str,
    purpose: str = "default",
) -> None:
    if position is None:
        return
    marker = _tracking_position_marker(local_dir, position, purpose=purpose)
    write_json_atomic(
        marker,
        {"position": position, "status": status},
    )


def _prior_tracking_positions_terminal(
    local_dir: Path,
    order: tuple[int, ...],
    position: int | None,
    *,
    purpose: str = "default",
) -> bool:
    assert position is not None
    prior = order[: order.index(position)]
    for item in prior:
        marker = _tracking_position_marker(local_dir, item, purpose=purpose)
        if not marker.is_file():
            return False
        try:
            status = json.loads(marker.read_text(encoding="utf-8")).get("status")
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        if status not in {"published", "evaluation_failed"}:
            return False
    return True

def _required(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"training tracking {key} must be a non-empty string")
    return item.strip()


def _project_for_run(
    settings: Mapping[str, Any],
    *,
    current_run_id: str,
    fork_lineage: Mapping[str, Any] | None,
) -> str:
    """Resolve the project from the ADE Run lineage, not the task name."""
    if not current_run_id.strip():
        raise ValueError("tracking Run identity is required")
    root = (
        fork_lineage.get("lineage_root_run_id")
        if isinstance(fork_lineage, Mapping)
        else None
    )
    project = str(root or current_run_id).strip()
    if not project:
        raise ValueError("tracking project identity is required")
    return project


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value) or "item"


def _wandb_id(display_name: str) -> str:
    return "--".join(_safe(part) for part in display_name.split("/"))


def _lineage_tags(lineage: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not lineage:
        return ()
    names = (
        "lineage_root_run_id",
        "source_run_id",
        "source_revision",
        "generation",
        "imported_from_run_id",
        "imported_from_revision",
        "imported_from_deployment_id",
        "source_command_id",
        "source_attempt_id",
    )
    tags = []
    for name in names:
        if lineage.get(name) is None:
            continue
        prefix = f"{name}:"
        value = str(lineage[name])
        available = 64 - len(prefix)
        if len(value) > available:
            value = "..." + value[-(available - 3) :]
        tags.append(prefix + value)
    return tuple(tags)


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _wandb_no_proxy_environment(base_url: str) -> dict[str, str]:
    environment = {
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
        "ALL_PROXY": "",
        "http_proxy": "",
        "https_proxy": "",
        "all_proxy": "",
    }
    hostname = urlparse(base_url).hostname if base_url else None
    if hostname:
        for name in ("NO_PROXY", "no_proxy"):
            entries = [
                item.strip()
                for item in str(os.environ.get(name) or "").split(",")
                if item.strip()
            ]
            if hostname not in entries:
                entries.append(hostname)
            environment[name] = ",".join(entries)
    return environment
EVALUATION_TRACKING_SCHEMA_VERSION = "ade.evaluation_tracking.v4"
_EVALUATION_WANDB_LOCK = threading.Lock()
