"""Local accepted-result projection for one ADE Run."""

from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Protocol

from ade.core.operator import OperatorEvaluationRecord
from ade.core.run import RunState
from ade.core.scope import subject_ref
from ade.core.trial import TrialKind, TrialOutcome


RESULTS_FIELDNAMES = (
    "run_id",
    "revision",
    "subject_ref",
    "subject_kind",
    "result_origin",
    "imported_from_run_id",
    "imported_from_revision",
    "imported_from_deployment_id",
    "imported_boundary_kind",
    "imported_frontier",
    "coordinator_id",
    "plan_id",
    "trial_id",
    "trial_kind",
    "trial_phase",
    "trial_outcome",
    "realization_status",
    "hypothesis_comparator_subject",
    "hypothesis_result",
    "portfolio_comparator_subject",
    "portfolio_result",
    "planning_ranking_revision",
    "model_ref",
    "selected_checkpoint_ref",
    "selected_checkpoint_id",
    "selected_position_unit",
    "selected_position_value",
    "online_status",
    "online_k",
    "online_avg_at_k",
    "online_pass_at_k",
    "online_ranking_score",
    "online_secondary_score",
    "offline_status",
    "offline_k",
    "offline_avg_at_k",
    "offline_pass_at_k",
    "offline_ranking_score",
    "offline_secondary_score",
    "operator_status",
    "operator_k",
    "operator_avg_at_k",
    "operator_pass_at_k",
    "operator_ranking_score",
    "operator_secondary_score",
    "engine_result_ref",
    "operator_result_ref",
)


class JsonObjectReader(Protocol):
    def read_json(self, uri: str) -> dict[str, Any]: ...


class ResultsReportProjector:
    def __init__(self, objects: JsonObjectReader) -> None:
        self.objects = objects

    def materialize(self, state: RunState, run_dir: Path) -> None:
        rows: list[dict[str, object]] = []
        if state.bootstrap.base_evaluation is not None:
            rows.append(self._base_row(state))
        for trial in sorted(
            (
                item
                for item in state.trials
                if item.result_refs or item.outcome is not TrialOutcome.PENDING
            ),
            key=lambda item: (
                0 if item.kind is TrialKind.BOOTSTRAP_BASELINE else 1,
                item.coordinator_id,
                item.plan_id,
                item.trial_id,
            ),
        ):
            rows.append(self._trial_row(state, trial, run_dir))
        output = io.StringIO(newline="")
        writer = csv.DictWriter(
            output,
            fieldnames=RESULTS_FIELDNAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        self._write_atomic(run_dir / "reports" / "results.csv", output.getvalue())

    def _base_row(self, state: RunState) -> dict[str, object]:
        base = state.bootstrap.base_evaluation
        assert base is not None
        payloads = self._base_payloads(base.result_refs)
        online_payload = self._purpose_payload(payloads, "online_validation")
        if online_payload is None:
            online_payload = self._read_evaluation_ref(
                base.online_result_ref, "online_validation"
            )
        offline_payload = self._purpose_payload(payloads, "offline_validation")
        operator = self._operator(state, subject_ref(state.run_id, "c000", "p000", "base"))
        model_ref = str(state.task.config.get("base_model") or "")
        if not model_ref:
            candidate = (offline_payload or online_payload or {}).get("checkpoint_path")
            model_ref = candidate if isinstance(candidate, str) else ""
        row = self._empty_row(
            state,
            subject=subject_ref(state.run_id, "c000", "p000", "base"),
            subject_kind="base_model",
            coordinator_id="c000",
            plan_id="p000",
            trial_id="",
        )
        row.update(
            {
                "model_ref": model_ref,
                "online_status": (
                    str(base.profile_status.get("online") or "not_applicable")
                ),
                "offline_status": str(
                    base.profile_status.get("offline") or "unavailable"
                ),
                "engine_result_ref": ";".join(base.result_refs),
            }
        )
        self._put_metrics(row, "online", online_payload)
        self._put_metrics(row, "offline", offline_payload)
        self._put_operator(row, operator)
        return row

    def _trial_row(
        self,
        state: RunState,
        trial,
        run_dir: Path,
    ) -> dict[str, object]:
        result_ref = next(
            (ref for ref in trial.result_refs if ref.endswith("/result.json")),
            trial.result_refs[0] if trial.result_refs else None,
        )
        result = self._read_json(result_ref) or {}
        kind = (
            "p000_baseline"
            if trial.kind is TrialKind.BOOTSTRAP_BASELINE
            else "search_trial"
        )
        target = subject_ref(
            state.run_id,
            trial.coordinator_id,
            trial.plan_id,
            trial.trial_id,
        )
        row = self._empty_row(
            state,
            subject=target,
            subject_kind=kind,
            coordinator_id=trial.coordinator_id,
            plan_id=trial.plan_id,
            trial_id=trial.trial_id,
        )
        position = result.get("selected_position")
        if not isinstance(position, Mapping) and result.get("selected_step") is not None:
            position = {"unit": "rl_step", "value": result["selected_step"]}
        checkpoint_ref = result.get("checkpoint_ref") or result.get("model_ref")
        manifest = self._manifest(result.get("trial_artifact_manifest_path"))
        plan = next(
            (
                item
                for item in state.plans
                if item.coordinator_id == trial.coordinator_id
                and item.plan_id == trial.plan_id
            ),
            None,
        )
        realization_ref_id = next(
            (
                artifact_id
                for artifact_id in trial.artifact_supporting_ref_ids
                if any(
                    ref.artifact_id == artifact_id
                    and ref.kind
                    in {"data_selection_realization", "reward_compliance_report"}
                    for ref in state.accepted_evidence_refs
                )
            ),
            None,
        )
        realization = self._accepted_artifact_json(
            state, run_dir, realization_ref_id
        )
        objective = self._accepted_artifact_json(
            state, run_dir, trial.objective_comparison_ref_id
        )
        hypothesis_binding = (
            plan.hypothesis_comparator
            if plan is not None and isinstance(plan.hypothesis_comparator, Mapping)
            else {}
        )
        portfolio_binding = (
            plan.portfolio_comparator
            if plan is not None and isinstance(plan.portfolio_comparator, Mapping)
            else {}
        )
        hypothesis_result = self._hypothesis_result(state, trial)
        row.update(
            {
                "trial_kind": trial.kind.value,
                "trial_phase": trial.phase.value,
                "trial_outcome": trial.outcome.value,
                "realization_status": realization.get("realization_status", ""),
                "hypothesis_comparator_subject": hypothesis_binding.get(
                    "subject_id", ""
                ),
                "hypothesis_result": hypothesis_result,
                "portfolio_comparator_subject": portfolio_binding.get(
                    "subject_id", ""
                ),
                "portfolio_result": (
                    objective.get("portfolio_comparator", {}).get("result", "")
                    if isinstance(objective.get("portfolio_comparator"), Mapping)
                    else ""
                ),
                "planning_ranking_revision": portfolio_binding.get(
                    "ranking_revision", ""
                ),
                "model_ref": checkpoint_ref if isinstance(checkpoint_ref, str) else "",
                "selected_checkpoint_ref": (
                    checkpoint_ref if isinstance(checkpoint_ref, str) else ""
                ),
                "selected_checkpoint_id": str(
                    manifest.get("selected_checkpoint_id") or ""
                ),
                "selected_position_unit": (
                    str(position.get("unit") or "")
                    if isinstance(position, Mapping)
                    else ""
                ),
                "selected_position_value": (
                    position.get("value", "")
                    if isinstance(position, Mapping)
                    else ""
                ),
                "online_status": self._evaluation_status(
                    result.get("online_validation"), default="unavailable"
                ),
                "offline_status": self._evaluation_status(
                    result.get("offline_validation"), default="unavailable"
                ),
                "engine_result_ref": result_ref or "",
            }
        )
        self._put_metrics(row, "online", result.get("online_validation"))
        self._put_metrics(row, "offline", result.get("offline_validation"))
        self._put_operator(row, self._operator(state, target))
        return row

    @staticmethod
    def _accepted_artifact_json(
        state: RunState,
        run_dir: Path,
        artifact_id: str | None,
    ) -> dict[str, Any]:
        ref = next(
            (
                item
                for item in state.accepted_evidence_refs
                if item.artifact_id == artifact_id
            ),
            None,
        )
        if ref is None:
            return {}
        prefix = f"run://{state.run_id}/"
        if not ref.uri.startswith(prefix):
            return {}
        relative = Path(ref.uri.removeprefix(prefix))
        if relative.is_absolute() or ".." in relative.parts:
            return {}
        try:
            value = json.loads((run_dir / relative).read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _hypothesis_result(state: RunState, trial) -> str:
        snapshot = next(
            (
                ref
                for ref in state.accepted_snapshot_refs
                if ref.snapshot_id == trial.plan_snapshot_ref_id
            ),
            None,
        )
        if snapshot is None:
            return ""
        try:
            lines = (Path(snapshot.root) / "MEMORY.md").read_text(
                encoding="utf-8"
            ).splitlines()
        except (OSError, UnicodeDecodeError):
            return ""
        values = [
            line.removeprefix("Hypothesis result: ")
            for line in lines
            if line.startswith("Hypothesis result: ")
        ]
        return values[-1] if values else ""

    def _empty_row(
        self,
        state: RunState,
        *,
        subject: str,
        subject_kind: str,
        coordinator_id: str,
        plan_id: str,
        trial_id: str,
    ) -> dict[str, object]:
        seed = state.seeded_from
        imported = seed is not None and subject_kind in {
            "base_model",
            "p000_baseline",
        }
        if seed is not None and subject_kind == "search_trial":
            frontier = seed.resolved_frontier.get(coordinator_id)
            imported = (
                frontier is not None
                and plan_id.startswith("p")
                and int(plan_id[1:]) <= int(frontier[1:])
            )
        row: dict[str, object] = {name: "" for name in RESULTS_FIELDNAMES}
        row.update(
            {
                "run_id": state.run_id,
                "revision": state.revision,
                "subject_ref": subject,
                "subject_kind": subject_kind,
                "result_origin": "imported" if imported else "native",
                "imported_from_run_id": seed.source_run_id if imported else "",
                "imported_from_revision": seed.source_revision if imported else "",
                "imported_from_deployment_id": (
                    seed.source_deployment_id or "" if imported else ""
                ),
                "imported_boundary_kind": (
                    seed.boundary_kind.value if imported else ""
                ),
                "imported_frontier": (
                    ",".join(
                        f"{key}={value}"
                        for key, value in sorted(seed.resolved_frontier.items())
                    )
                    if imported
                    else ""
                ),
                "coordinator_id": coordinator_id,
                "plan_id": plan_id,
                "trial_id": trial_id,
                "online_status": "unavailable",
                "offline_status": "unavailable",
                "operator_status": "not_scheduled",
            }
        )
        return row

    def _put_operator(
        self,
        row: dict[str, object],
        record: OperatorEvaluationRecord | None,
    ) -> None:
        if record is None:
            return
        row["operator_status"] = record.status.value
        row["operator_result_ref"] = record.result_ref or ""
        self._put_metrics(
            row,
            "operator",
            self._read_evaluation_ref(record.result_ref, "operator_test"),
        )

    @staticmethod
    def _operator(
        state: RunState,
        target: str,
    ) -> OperatorEvaluationRecord | None:
        return next(
            (item for item in state.operator_evaluations if item.target_id == target),
            None,
        )

    def _put_metrics(
        self,
        row: dict[str, object],
        prefix: str,
        value: object,
    ) -> None:
        payload = self._evaluation_payload(value)
        if payload is None:
            return
        aggregate = self._aggregate_metrics(payload)
        row[f"{prefix}_k"] = self._configured_k(payload, aggregate)
        row[f"{prefix}_avg_at_k"] = self._number(aggregate.get("avg@k"))
        row[f"{prefix}_pass_at_k"] = self._number(aggregate.get("pass@k"))
        row[f"{prefix}_ranking_score"] = self._number(
            aggregate.get(
                "ranking_score",
                aggregate.get(
                    "score", payload.get("ranking_score", payload.get("score"))
                ),
            )
        )
        row[f"{prefix}_secondary_score"] = self._number(
            aggregate.get(
                "secondary_score",
                payload.get("secondary_score"),
            )
        )

    @staticmethod
    def _aggregate_metrics(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        aggregate = payload.get("aggregate_metrics")
        if isinstance(aggregate, Mapping):
            return aggregate
        results = payload.get("results")
        if (
            isinstance(results, list)
            and len(results) == 1
            and isinstance(results[0], Mapping)
            and isinstance(results[0].get("metrics"), Mapping)
        ):
            return results[0]["metrics"]
        return {}

    @staticmethod
    def _evaluation_payload(value: object) -> dict[str, Any] | None:
        if not isinstance(value, Mapping):
            return None
        for key in ("payload", "result"):
            nested = value.get(key)
            if isinstance(nested, Mapping):
                return dict(nested)
        return dict(value)

    @classmethod
    def _evaluation_status(cls, value: object, *, default: str) -> str:
        if not isinstance(value, Mapping):
            return default
        return str(value.get("status") or default)

    @classmethod
    def _configured_k(
        cls,
        payload: Mapping[str, Any],
        aggregate: Mapping[str, Any],
    ) -> object:
        direct = aggregate.get("configured_k")
        if cls._integer(direct) is not None:
            return int(direct)
        datasets = aggregate.get("dataset_metrics")
        if isinstance(datasets, Mapping):
            values = {
                int(candidate)
                for metrics in datasets.values()
                if isinstance(metrics, Mapping)
                and (candidate := cls._integer(metrics.get("configured_k"))) is not None
            }
            if len(values) == 1:
                return values.pop()
        for candidate in (
            payload.get("avg_k_runs"),
            (
                payload.get("parallel_eval", {}).get("avg_k")
                if isinstance(payload.get("parallel_eval"), Mapping)
                else None
            ),
        ):
            if cls._integer(candidate) is not None:
                return int(candidate)
        return ""

    @staticmethod
    def _number(value: object) -> object:
        if type(value) not in {int, float}:
            return ""
        numeric = float(value)
        return numeric if math.isfinite(numeric) else ""

    @staticmethod
    def _integer(value: object) -> int | None:
        if type(value) is not int or value < 1:
            return None
        return value

    def _read_json(self, ref: object) -> dict[str, Any] | None:
        if not isinstance(ref, str) or not ref:
            return None
        try:
            return self.objects.read_json(ref)
        except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _read_evaluation_ref(
        self,
        ref: object,
        purpose: str,
    ) -> dict[str, Any] | None:
        summary = self._read_json(ref)
        if not isinstance(ref, str):
            return summary
        if ref.endswith("/result.json"):
            result_root = ref[: -len("/result.json")]
        elif ref.endswith("-result.json"):
            # Referenced Bootstrap imports flatten result.json to
            # accepted-results/<index>-result.json while preserving its raw unit
            # under accepted-results/<index>/raw/units/.
            result_root = ref[: -len("-result.json")]
        else:
            return summary
        raw = self._read_json(f"{result_root}/raw/units/{purpose}.json")
        if raw is None:
            return summary
        return {**(summary or {}), **raw, "purpose": purpose}

    def _base_payloads(
        self,
        refs: tuple[str, ...],
    ) -> tuple[dict[str, Any], ...]:
        payloads: list[dict[str, Any]] = []
        for ref in refs:
            payload = self._read_json(ref)
            if payload is None:
                continue
            payloads.append(payload)
            units = payload.get("units")
            if not isinstance(units, list):
                continue
            for unit in units:
                if not isinstance(unit, Mapping):
                    continue
                raw = self._read_json(unit.get("uri"))
                if raw is not None:
                    payloads.append(
                        {
                            **raw,
                            "purpose": str(unit.get("kind") or unit.get("unit_id") or ""),
                        }
                    )
        return tuple(payloads)

    @staticmethod
    def _purpose_payload(
        payloads: tuple[dict[str, Any], ...],
        purpose: str,
    ) -> dict[str, Any] | None:
        return next(
            (
                payload
                for payload in reversed(payloads)
                if payload.get("phase") == purpose or payload.get("purpose") == purpose
            ),
            None,
        )

    def _manifest(self, value: object) -> dict[str, Any]:
        if not isinstance(value, str) or not value:
            return {}
        if value.startswith("engine://"):
            return self._read_json(value) or {}
        path = Path(value)
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _write_atomic(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
