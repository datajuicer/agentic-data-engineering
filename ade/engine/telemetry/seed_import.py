"""Target-local tracking projection for an admitted Run Seed."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from ade.controller.ports import RunRepository
from ade.core.run import RunState
from ade.core.trial import TrialKind
from ade.engine.telemetry.tracking import (
    publish_evaluation_tracking,
    republish_seed_wandb_history,
)
from ade.engine.storage.atomic import write_json_atomic
from ade.engine.storage.object_store import FileEngineObjectStore


class SeedTrackingImporter:
    def __init__(
        self,
        *,
        source_repository: RunRepository,
        target_repository: RunRepository,
        source_deployment_id: str | None = None,
        source_objects: FileEngineObjectStore | None = None,
    ) -> None:
        self.source_repository = source_repository
        self.target_repository = target_repository
        self.source_deployment_id = source_deployment_id
        self.source_objects = source_objects

    def records(self, source: RunState) -> tuple[dict[str, Any], ...]:
        """Read durable evaluation records selected by the source projection."""
        selected_trials = {
            (trial.coordinator_id, trial.plan_id, trial.trial_id)
            for trial in source.trials
        }
        source_root = self.source_repository.layout.run_dir(source.run_id)
        records: list[dict[str, Any]] = []
        for request_path in sorted(source_root.rglob("tracking-request.json")):
            result_path = request_path.parent / "evaluation-result.json"
            if not result_path.is_file():
                continue
            try:
                request = json.loads(request_path.read_text(encoding="utf-8"))
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(request, dict) or not isinstance(result, dict):
                continue
            required = (
                "command_id",
                "coordinator_id",
                "plan_id",
                "trial_id",
                "purpose",
            )
            if any(not str(request.get(key) or "").strip() for key in required):
                continue
            identity = (
                str(request["coordinator_id"]),
                str(request["plan_id"]),
                str(request["trial_id"]),
            )
            is_base = (
                identity[0] == "c000"
                and identity[1] == "p000"
                and (
                    identity[2] == "base"
                    or identity[2].startswith("base-")
                )
            )
            if not is_base and identity not in selected_trials:
                continue
            records.append({"request": request, "result": result})
        return tuple(records) or self._engine_records(source)

    def _engine_records(self, source: RunState) -> tuple[dict[str, Any], ...]:
        """Recover publication inputs when only accepted Engine evidence remains.

        The shared standard-RFT baseline retains complete evaluation manifests,
        but its old local W&B request directories are no longer present.
        """
        if self.source_objects is None:
            return ()
        base = source.bootstrap.base_evaluation
        manifests = [
            ref
            for ref in (
                *(base.result_refs if base is not None else ()),
                *(ref for trial in source.trials for ref in trial.result_refs),
            )
            if ref.endswith("manifest.json")
        ]
        manifests.extend(
            record.result_ref.removesuffix("result.json") + "raw/manifest.json"
            for record in source.operator_evaluations
            if record.result_ref is not None
            and record.result_ref.endswith("/result.json")
        )
        records = []
        for ref in dict.fromkeys(manifests):
            manifest = self.source_objects.read_json(ref)
            units = [unit for unit in manifest["units"] if unit["kind"] in {
                "online_validation", "offline_validation", "operator_test",
            }]
            for unit in units:
                purpose = unit["kind"]
                result = self.source_objects.read_json(unit["uri"])
                request = {
                    "command_id": result.get("evaluation_command_id") or (
                        f"{manifest['command_id']}:{unit['unit_id']}"
                    ),
                    "coordinator_id": manifest["coordinator_id"],
                    "plan_id": manifest["plan_id"],
                    "trial_id": manifest["trial_id"],
                    "purpose": purpose,
                    "source_engine_manifest_ref": ref,
                    "source_engine_result_ref": unit["uri"],
                }
                if "step" in unit:
                    request["artifact_position"] = {"unit": "rl_step", "value": unit["step"]}
                    request["position_order"] = sorted({
                        item["step"] for item in units
                        if item["kind"] == purpose and "step" in item
                    })
                elif purpose == "online_validation" and str(manifest["trial_id"]).startswith("base"):
                    request["artifact_position"] = {"unit": "rl_step", "value": 0}
                    request["position_order"] = [0]
                records.append({"request": request, "result": result})
        return tuple(records)

    def staged_files(self, source: RunState) -> tuple[tuple[str, bytes], ...]:
        files: list[tuple[str, bytes]] = [
            (
                "tracking/seed-import/history-source.json",
                (
                    json.dumps(
                        {
                            "source_run_id": source.run_id,
                            "source_project_id": self._source_project_id(source),
                            "selected_subject_refs": self._selected_subject_refs(source),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                ).encode(),
            )
        ]
        for index, record in enumerate(self.records(source)):
            root = f"tracking/seed-import/source-records/{index:04d}"
            files.extend(
                (
                    (
                        f"{root}/source-request.json",
                        (json.dumps(record["request"], sort_keys=True) + "\n").encode(),
                    ),
                    (
                        f"{root}/source-result.json",
                        (json.dumps(record["result"], sort_keys=True) + "\n").encode(),
                    ),
                )
            )
        return tuple(files)

    def target_records(self, target: RunState) -> tuple[dict[str, Any], ...]:
        root = (
            self.target_repository.layout.run_dir(target.run_id)
            / "tracking"
            / "seed-import"
            / "source-records"
        )
        records: list[dict[str, Any]] = []
        for request_path in sorted(root.glob("*/source-request.json")):
            result_path = request_path.with_name("source-result.json")
            if not result_path.is_file():
                raise ValueError(
                    f"Run Seed staged tracking result is missing: {result_path}"
                )
            records.append(
                {
                    "request": json.loads(request_path.read_text(encoding="utf-8")),
                    "result": json.loads(result_path.read_text(encoding="utf-8")),
                }
            )
        return tuple(records)

    def publish(
        self,
        source: RunState,
        target: RunState,
        *,
        target_tracking: dict[str, object],
    ) -> tuple[dict[str, object], ...]:
        records = self.target_records(target) or self.records(source)
        ordinals: dict[str, int] = {}
        grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
        for record in records:
            request = record["request"]
            if request.get("purpose") not in {
                "online_validation",
                "offline_validation",
            }:
                continue
            subject_kind = self._subject_kind(source, request)
            key = (
                str(request.get("coordinator_id") or ""),
                str(request.get("plan_id") or ""),
                (
                    "base"
                    if subject_kind == "base_model"
                    else str(request.get("trial_id") or "")
                ),
                subject_kind,
            )
            grouped.setdefault(key, []).append(record)
        for group in grouped.values():
            def order_key(record: dict[str, Any]) -> tuple[int, int, str]:
                request = record["request"]
                position = request.get("artifact_position")
                value = position.get("value") if isinstance(position, dict) else None
                return (
                    value if type(value) is int else 2**63 - 1,
                    0 if request.get("purpose") == "online_validation" else 1,
                    str(request.get("command_id") or ""),
                )

            for ordinal, record in enumerate(
                sorted(group, key=order_key), start=1
            ):
                ordinals[str(record["request"]["command_id"])] = ordinal

        results: list[dict[str, object]] = []
        for record in records:
            request = record["request"]
            results.append(
                publish_evaluation_tracking(
                    settings=target_tracking,
                    run_id=target.run_id,
                    command_id=(
                        f"seed-import:{source.run_id}:{request['command_id']}"
                    ),
                    coordinator_id=str(request["coordinator_id"]),
                    plan_id=str(request["plan_id"]),
                    trial_id=str(request["trial_id"]),
                    subject_kind=self._subject_kind(source, request),
                    purpose=str(request["purpose"]),
                    result=record["result"],
                    local_root=(
                        self.target_repository.layout.run_dir(target.run_id)
                        / "tracking"
                        / "seed-import"
                    ),
                    artifact_position=request.get("artifact_position"),
                    position_order=request.get("position_order"),
                    evaluation_ordinal=ordinals.get(
                        str(request["command_id"])
                    ),
                    fork_lineage={
                        "seeded_from_run_id": source.run_id,
                        "seeded_from_revision": source.revision,
                        **(
                            {
                                "seeded_from_deployment_id": (
                                    self.source_deployment_id
                                )
                            }
                            if self.source_deployment_id is not None
                            else {}
                        ),
                        "source_command_id": request["command_id"],
                        **{
                            key: request[key]
                            for key in ("source_engine_manifest_ref", "source_engine_result_ref")
                            if key in request
                        },
                        **(
                            {"source_attempt_id": request["attempt_id"]}
                            if request.get("attempt_id") is not None
                            else {}
                        ),
                    },
                    defer_online=True,
                )
            )
        return tuple(results)

    @staticmethod
    def _source_project_id(source: RunState) -> str:
        return (
            source.forked_from.lineage_root_run_id
            if source.forked_from is not None
            else source.run_id
        )

    @staticmethod
    def _selected_subject_refs(source: RunState) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    *(
                        f"{source.run_id}/{plan.coordinator_id}/{plan.plan_id}"
                        for plan in source.plans
                    ),
                    *(
                        f"{source.run_id}/{trial.coordinator_id}/{trial.plan_id}/{trial.trial_id}"
                        for trial in source.trials
                    ),
                }
            )
        )

    @staticmethod
    def _subject_kind(
        source: RunState,
        request: dict[str, Any],
    ) -> str:
        explicit = request.get("subject_kind")
        if explicit in {"base_model", "p000_baseline", "search_trial"}:
            return str(explicit)
        trial_id = str(request.get("trial_id") or "")
        if trial_id == "base" or trial_id.startswith("base-"):
            return "base_model"
        trial = next(
            (
                item
                for item in source.trials
                if item.coordinator_id == str(request.get("coordinator_id") or "")
                and item.plan_id == str(request.get("plan_id") or "")
                and item.trial_id == trial_id
            ),
            None,
        )
        if trial is None:
            raise ValueError("Run Seed evaluation subject is outside the projection")
        return (
            "p000_baseline"
            if trial.kind is TrialKind.BOOTSTRAP_BASELINE
            else "search_trial"
        )


def retry_seed_wandb_history(
    *,
    target_repository: RunRepository,
    target: RunState,
    target_tracking: dict[str, object],
    wandb_module: Any = None,
    max_new_runs: int | None = None,
    known_published_source_run_ids: tuple[str, ...] = (),
) -> tuple[dict[str, object], ...]:
    provenance = target.seeded_from
    if provenance is None:
        return ()
    metadata_path = (
        target_repository.layout.run_dir(target.run_id)
        / "tracking"
        / "seed-import"
        / "history-source.json"
    )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Run Seed W&B history source metadata is unavailable: {metadata_path}"
        ) from error
    if (
        not isinstance(metadata, dict)
        or metadata.get("source_run_id") != provenance.source_run_id
        or not isinstance(metadata.get("source_project_id"), str)
        or not isinstance(metadata.get("selected_subject_refs"), list)
        or not all(
            isinstance(item, str)
            for item in metadata["selected_subject_refs"]
        )
    ):
        raise ValueError("Run Seed W&B history source metadata is invalid")
    return republish_seed_wandb_history(
        source_run_id=provenance.source_run_id,
        source_project_id=metadata["source_project_id"],
        target_run_id=target.run_id,
        settings=target_tracking,
        selected_subject_refs=tuple(metadata["selected_subject_refs"]),
        wandb_module=wandb_module,
        max_new_runs=max_new_runs,
        known_published_source_run_ids=known_published_source_run_ids,
    )


def reconcile_seed_wandb_history(
    *,
    target_repository: RunRepository,
    target: RunState,
    target_tracking: dict[str, object],
    wandb_module: Any = None,
) -> dict[str, object]:
    """Retry Seed-only W&B history without blocking Run admission or resume."""
    status_path = (
        target_repository.layout.run_dir(target.run_id)
        / "tracking"
        / "seed-import"
        / "history-status.json"
    )
    existing = None
    if status_path.is_file():
        try:
            existing = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict) and existing.get("status") == "published":
            return existing
    known_published_source_run_ids = tuple(
        str(item["source_run_id"])
        for item in (
            existing.get("runs", []) if isinstance(existing, dict) else []
        )
        if isinstance(item, dict)
        and item.get("status") in {"published", "already_published"}
        and isinstance(item.get("source_run_id"), str)
    )
    if target.seeded_from is None:
        return {"status": "not_applicable"}
    if (
        not bool(target_tracking.get("enabled", False))
        or str(target_tracking.get("mode") or "offline").lower() != "online"
    ):
        result: dict[str, object] = {
            "status": "disabled",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "runs": [],
        }
        write_json_atomic(status_path, result)
        return result
    try:
        runs = retry_seed_wandb_history(
            target_repository=target_repository,
            target=target,
            target_tracking=target_tracking,
            wandb_module=wandb_module,
            max_new_runs=1,
            known_published_source_run_ids=known_published_source_run_ids,
        )
    except Exception as error:
        result = {
            "status": "pending_retry",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            # Keep prior confirmations so a transient listing/API failure does
            # not make later cycles re-read every already-published source.
            "runs": (
                list(existing.get("runs", []))
                if isinstance(existing, dict)
                else []
            ),
        }
    else:
        pending = any(item.get("status") == "pending_retry" for item in runs)
        result = {
            "status": "pending_retry" if pending else "published",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "error_type": None,
            "runs": list(runs),
        }
    write_json_atomic(status_path, result)
    return result
