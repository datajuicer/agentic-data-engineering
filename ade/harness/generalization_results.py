"""Strict aggregation for standalone generalization evaluation results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from ade.engine.storage.object_store import FileEngineObjectStore


RESULT_FIELDS = (
    "evaluation_id",
    "task",
    "checkpoint_id",
    "variant",
    "subject_ref",
    "checkpoint_ref",
    "dataset",
    "dataset_family",
    "rows",
    "configured_k",
    "expected_generations",
    "completed_generations",
    "valid_generations",
    "invalid_generations",
    "grader_failure_count",
    "status",
    "avg_at_k",
    "pass_at_k",
    "primary_metric",
    "primary_score",
    "system_prompt_id",
    "chat_template_id",
    "model_protocol_mode",
    "reasoning_parser",
    "logical_unit_id",
    "successful_attempt_id",
    "result_ref",
    "error",
)
COMPARISON_FIELDS = (
    "task",
    "dataset",
    "dataset_family",
    "rows",
    "configured_k",
    "primary_metric",
    "base",
    "baseline",
    "n1",
    "n3",
    "n1_runs",
    "n3_runs",
    "baseline_minus_base",
    "n1_minus_base",
    "n3_minus_base",
    "n1_minus_baseline",
    "n3_minus_baseline",
    "n3_minus_n1",
    "status",
)
_CORE_COMPARISON_VARIANTS = ("base", "baseline", "n1", "n3")
FAMILY_FIELDS = (
    "task",
    "variant",
    "dataset_family",
    "run_count",
    "complete_units",
    "total_units",
    "primary_macro",
    "status",
)


class GeneralizationResultAggregator:
    def __init__(
        self,
        *,
        evaluation_root: Path,
        objects: FileEngineObjectStore,
    ) -> None:
        self.evaluation_root = evaluation_root.resolve()
        self.objects = objects

    def aggregate(
        self,
        evaluation_id: str,
        *,
        output_dir: Path,
    ) -> dict[str, object]:
        directory = self.evaluation_root / evaluation_id
        resolved = self._json(directory / "config/resolved.json")
        state = self._json(directory / "state.json")
        if (
            resolved.get("evaluation_id") != evaluation_id
            or state.get("evaluation_id") != evaluation_id
        ):
            raise ValueError("evaluation identity mismatch")
        resolved_units = self._mapping(resolved.get("units"), "resolved.units")
        state_units = self._mapping(state.get("units"), "state.units")
        if set(resolved_units) != set(state_units):
            missing = sorted(set(resolved_units) - set(state_units))
            extra = sorted(set(state_units) - set(resolved_units))
            raise ValueError(
                f"resolved/state unit mismatch: missing={missing}, extra={extra}"
            )
        rows = [
            self._row(
                evaluation_id, unit_id, resolved_units[unit_id], state_units[unit_id]
            )
            for unit_id in resolved_units
        ]
        target = output_dir.resolve()
        target.mkdir(parents=True, exist_ok=True)
        self._write_csv(target / "results.csv", rows)
        comparisons = self._comparisons(rows)
        families = self._families(rows)
        failed = [row for row in rows if row["status"] != "complete"]
        self._write_rows(
            target / "comparison.csv",
            self._comparison_fields(rows),
            comparisons,
        )
        self._write_rows(target / "family_summary.csv", FAMILY_FIELDS, families)
        self._write_csv(target / "failed_units.csv", failed)
        (target / "summary.md").write_text(
            self._summary(evaluation_id, rows, comparisons), encoding="utf-8"
        )
        return {
            "schema_version": 1,
            "evaluation_id": evaluation_id,
            "row_count": len(rows),
            "complete_rows": sum(row["status"] == "complete" for row in rows),
            "comparison_rows": len(comparisons),
            "family_rows": len(families),
            "failed_rows": len(failed),
            "results_csv": str(target / "results.csv"),
            "comparison_csv": str(target / "comparison.csv"),
            "family_summary_csv": str(target / "family_summary.csv"),
            "failed_units_csv": str(target / "failed_units.csv"),
            "summary_md": str(target / "summary.md"),
        }

    def _row(
        self,
        evaluation_id: str,
        unit_id: str,
        resolved: object,
        state: object,
    ) -> dict[str, object]:
        unit = self._mapping(resolved, f"resolved unit {unit_id}")
        unit_state = self._mapping(state, f"state unit {unit_id}")
        attempts = unit_state.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise ValueError(f"unit {unit_id} has no attempts")
        successful = [
            attempt
            for attempt in attempts
            if isinstance(attempt, dict)
            and isinstance(attempt.get("receipt"), dict)
            and attempt["receipt"].get("status") == "succeeded"
        ]
        if len(successful) > 1:
            raise ValueError(f"unit {unit_id} has duplicate successful attempts")
        request = self._mapping(unit.get("request"), f"unit {unit_id}.request")
        datasets = request.get("datasets")
        if (
            not isinstance(datasets, list)
            or len(datasets) != 1
            or not isinstance(datasets[0], dict)
        ):
            raise ValueError(f"unit {unit_id} must resolve exactly one dataset")
        dataset = datasets[0]
        prompt = self._mapping(
            dataset.get("prompt_protocol"), "dataset.prompt_protocol"
        )
        system = self._mapping(prompt.get("system_prompt"), "prompt.system_prompt")
        template = self._mapping(prompt.get("chat_template"), "prompt.chat_template")
        model_protocol = self._mapping(
            request.get("model_protocol"), "request.model_protocol"
        )
        thinking = self._mapping(
            model_protocol.get("thinking"), "model_protocol.thinking"
        )
        metrics_policy = self._mapping(unit.get("metrics"), f"unit {unit_id}.metrics")
        primary_metric = str(metrics_policy.get("primary") or "")
        base: dict[str, object] = {
            "evaluation_id": evaluation_id,
            "task": unit.get("suite_id"),
            "checkpoint_id": unit.get("checkpoint_id", unit.get("variant")),
            "variant": unit.get("variant"),
            "subject_ref": unit.get("subject_ref"),
            "checkpoint_ref": unit.get("checkpoint_ref"),
            "dataset": unit.get("dataset_id"),
            "dataset_family": unit.get("family"),
            "rows": int(unit["expected_rows"]),
            "configured_k": int(unit["samples_per_input"]),
            "expected_generations": int(unit["expected_generations"]),
            "completed_generations": 0,
            "valid_generations": 0,
            "invalid_generations": 0,
            "grader_failure_count": 0,
            "status": str(unit_state.get("status") or "missing"),
            "avg_at_k": "",
            "pass_at_k": "",
            "primary_metric": primary_metric,
            "primary_score": "",
            "system_prompt_id": system.get("id"),
            "chat_template_id": template.get("id"),
            "model_protocol_mode": thinking.get("mode"),
            "reasoning_parser": request.get("reasoning_parser"),
            "logical_unit_id": unit_id,
            "successful_attempt_id": "",
            "result_ref": "",
            "error": self._last_error(attempts),
        }
        if not successful:
            return base

        attempt = successful[0]
        receipt = attempt["receipt"]
        raw_ref = self._raw_unit_ref(unit_id, receipt)
        payload = self.objects.read_json(raw_ref)
        results = payload.get("results")
        if (
            not isinstance(results, list)
            or len(results) != 1
            or not isinstance(results[0], dict)
        ):
            raise ValueError(
                f"unit {unit_id} raw result must contain exactly one dataset"
            )
        result = results[0]
        if result.get("dataset_name") != unit.get("dataset_id"):
            raise ValueError(f"unit {unit_id} raw dataset identity mismatch")
        metrics = self._mapping(result.get("metrics"), f"unit {unit_id}.metrics")
        observed_rows = self._count(metrics.get("num_total"), "num_total")
        observed_k = self._count(
            metrics.get("configured_k", metrics.get("num_repeat")), "configured_k"
        )
        completed = observed_rows * observed_k
        completion_rate = self._rate(
            metrics.get("completion_rate", 1.0), "completion_rate"
        )
        valid = round(completed * completion_rate)
        grader_failures = self._count(
            metrics.get("grader_failure_count", metrics.get("grader_failures", 0)),
            "grader_failure_count",
            allow_zero=True,
        )
        # Retained raw evaluations may count case timeouts as grader failures.
        # Their scores already include these cases as wrong in the full
        # denominator; publish them without a special status or error label.
        grader_timeouts = self._count(
            metrics.get("grader_timeout_count", 0),
            "grader_timeout_count",
            allow_zero=True,
        )
        grader_failures = max(0, grader_failures - grader_timeouts)
        expected = int(unit["expected_generations"])
        raw_status = str(payload.get("status") or "")
        status = "complete"
        errors: list[str] = []
        if raw_status not in {"complete", "completed"} or completed != expected:
            status = "partial"
            errors.append(
                f"completed_generations={completed}, expected_generations={expected}, raw_status={raw_status}"
            )
        if valid < completed:
            errors.append(f"invalid_generations={completed - valid}")
        if grader_failures:
            status = (
                "partial" if status == "partial" else "complete_with_grader_failures"
            )
            errors.append(f"grader_failure_count={grader_failures}")
        avg_at_k = self._score(
            metrics.get("avg@k", metrics.get("avg_at_k_score")), "avg_at_k"
        )
        pass_at_k = self._score(metrics.get("pass@k"), "pass_at_k")
        primary_score = pass_at_k if primary_metric == "pass_at_k" else avg_at_k
        base.update(
            {
                "completed_generations": completed,
                "valid_generations": valid,
                "invalid_generations": completed - valid,
                "grader_failure_count": grader_failures,
                "status": status,
                "avg_at_k": avg_at_k,
                "pass_at_k": pass_at_k,
                "primary_score": primary_score,
                "successful_attempt_id": attempt.get("attempt_id"),
                "result_ref": raw_ref,
                "error": "; ".join(errors),
            }
        )
        return base

    def _raw_unit_ref(self, unit_id: str, receipt: dict[str, Any]) -> str:
        refs = receipt.get("output_refs")
        if not isinstance(refs, (list, tuple)):
            raise ValueError(f"unit {unit_id} successful receipt has no output refs")
        manifests = [
            str(ref) for ref in refs if str(ref).endswith("/raw/manifest.json")
        ]
        if len(manifests) != 1:
            raise ValueError(f"unit {unit_id} requires exactly one raw manifest ref")
        manifest = self.objects.read_json(manifests[0])
        units = manifest.get("units")
        if not isinstance(units, list):
            raise ValueError(f"unit {unit_id} raw manifest has no units")
        matches = [
            str(item.get("uri"))
            for item in units
            if isinstance(item, dict) and item.get("kind") == "operator_test"
        ]
        if len(matches) != 1:
            raise ValueError(
                f"unit {unit_id} requires exactly one operator_test raw unit"
            )
        return matches[0]

    @staticmethod
    def _last_error(attempts: list[object]) -> str:
        for attempt in reversed(attempts):
            if not isinstance(attempt, dict) or not isinstance(
                attempt.get("receipt"), dict
            ):
                continue
            error = attempt["receipt"].get("error")
            if error:
                return str(error)
        return ""

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        GeneralizationResultAggregator._write_rows(path, RESULT_FIELDS, rows)

    @staticmethod
    def _write_rows(
        path: Path,
        fields: tuple[str, ...],
        rows: list[dict[str, object]],
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _comparisons(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
        variants_by_task: dict[str, set[str]] = {}
        for row in rows:
            task = str(row["task"])
            grouped.setdefault((task, str(row["dataset"])), []).append(row)
            variants_by_task.setdefault(task, set()).add(str(row["variant"]))
        additional_variants = GeneralizationResultAggregator._additional_variants(rows)
        result: list[dict[str, object]] = []
        for (task, dataset), selected in sorted(grouped.items()):
            variants = _CORE_COMPARISON_VARIANTS + tuple(
                variant
                for variant in additional_variants
                if variant in variants_by_task[task]
            )
            by_variant: dict[str, list[dict[str, object]]] = {}
            for row in selected:
                by_variant.setdefault(str(row["variant"]), []).append(row)
            first = selected[0]
            scores = {
                variant: (
                    fmean(
                        float(row["primary_score"])
                        for row in by_variant[variant]
                    )
                    if variant in by_variant
                    and all(
                        row["status"] == "complete"
                        and row["primary_score"] != ""
                        for row in by_variant[variant]
                    )
                    else ""
                )
                for variant in variants
            }

            def delta(left: str, right: str) -> float | str:
                if scores[left] == "" or scores[right] == "":
                    return ""
                return float(scores[left]) - float(scores[right])

            missing = [variant for variant in variants if variant not in by_variant]
            incomplete = [
                variant
                for variant in variants
                if variant in by_variant
                and any(row["status"] != "complete" for row in by_variant[variant])
            ]
            status = "complete"
            if missing:
                status = "missing:" + ",".join(missing)
            elif incomplete:
                status = "incomplete:" + ",".join(incomplete)
            comparison: dict[str, object] = {
                "task": task,
                "dataset": dataset,
                "dataset_family": first["dataset_family"],
                "rows": first["rows"],
                "configured_k": first["configured_k"],
                "primary_metric": first["primary_metric"],
                **{
                    variant: scores[variant]
                    for variant in _CORE_COMPARISON_VARIANTS
                },
                "n1_runs": len(
                    {
                        str(row["checkpoint_id"])
                        for row in by_variant.get("n1", [])
                    }
                ),
                "n3_runs": len(
                    {
                        str(row["checkpoint_id"])
                        for row in by_variant.get("n3", [])
                    }
                ),
                "baseline_minus_base": delta("baseline", "base"),
                "n1_minus_base": delta("n1", "base"),
                "n3_minus_base": delta("n3", "base"),
                "n1_minus_baseline": delta("n1", "baseline"),
                "n3_minus_baseline": delta("n3", "baseline"),
                "n3_minus_n1": delta("n3", "n1"),
                "status": status,
            }
            for variant in additional_variants:
                if variant not in variants_by_task[task]:
                    continue
                comparison[
                    GeneralizationResultAggregator._extra_score_field(variant)
                ] = scores[variant]
                comparison[
                    GeneralizationResultAggregator._extra_delta_field(variant, "base")
                ] = delta(variant, "base")
                comparison[
                    GeneralizationResultAggregator._extra_delta_field(
                        variant, "baseline"
                    )
                ] = delta(variant, "baseline")
                comparison[
                    GeneralizationResultAggregator._extra_delta_field(variant, "n3")
                ] = delta(variant, "n3")
            result.append(comparison)
        return result

    @staticmethod
    def _additional_variants(
        rows: list[dict[str, object]],
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                {str(row["variant"]) for row in rows}
                - set(_CORE_COMPARISON_VARIANTS)
            )
        )

    @staticmethod
    def _extra_score_field(variant: str) -> str:
        return f"score_{variant}"

    @staticmethod
    def _extra_delta_field(variant: str, comparator: str) -> str:
        return f"delta_{variant}_minus_{comparator}"

    @staticmethod
    def _comparison_fields(
        rows: list[dict[str, object]],
    ) -> tuple[str, ...]:
        additional_fields: list[str] = []
        for variant in GeneralizationResultAggregator._additional_variants(rows):
            additional_fields.extend(
                [
                    GeneralizationResultAggregator._extra_score_field(variant),
                    GeneralizationResultAggregator._extra_delta_field(variant, "base"),
                    GeneralizationResultAggregator._extra_delta_field(
                        variant, "baseline"
                    ),
                    GeneralizationResultAggregator._extra_delta_field(variant, "n3"),
                ]
            )
        return COMPARISON_FIELDS[:-1] + tuple(additional_fields) + ("status",)

    @staticmethod
    def _families(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
        for row in rows:
            key = (
                str(row["task"]),
                str(row["variant"]),
                str(row["dataset_family"]),
            )
            grouped.setdefault(key, []).append(row)
        result = []
        for (task, variant, family), selected in sorted(grouped.items()):
            complete = [row for row in selected if row["status"] == "complete"]
            ready = len(complete) == len(selected)
            result.append(
                {
                    "task": task,
                    "variant": variant,
                    "dataset_family": family,
                    "run_count": len(
                        {str(row["checkpoint_id"]) for row in selected}
                    ),
                    "complete_units": len(complete),
                    "total_units": len(selected),
                    "primary_macro": (
                        fmean(float(row["primary_score"]) for row in complete)
                        if ready
                        else ""
                    ),
                    "status": "complete" if ready else "incomplete",
                }
            )
        return result

    @staticmethod
    def _summary(
        evaluation_id: str,
        rows: list[dict[str, object]],
        comparisons: list[dict[str, object]],
    ) -> str:
        lines = [
            f"# Generalization evaluation: {evaluation_id}",
            "",
            "Scores are percentages rendered from the decimal values in `results.csv`.",
            "N1 and N3 values are equal-weight means across audit Run-best checkpoints.",
            "No cross-task overall score is computed.",
            "",
            "| Task | Variant | Runs | Complete units | Primary macro |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        keys = sorted({(str(row["task"]), str(row["variant"])) for row in rows})
        for task, variant in keys:
            selected = [
                row for row in rows if row["task"] == task and row["variant"] == variant
            ]
            complete = [row for row in selected if row["status"] == "complete"]
            if len(complete) != len(selected):
                macro = f"n/a ({len(complete)}/{len(selected)} complete)"
            elif task == "sft-math":
                family_scores = [
                    fmean(
                        float(row["primary_score"])
                        for row in complete
                        if row["dataset_family"] == family
                    )
                    for family in sorted(
                        {str(row["dataset_family"]) for row in complete}
                    )
                ]
                macro = GeneralizationResultAggregator._percent(fmean(family_scores))
            else:
                macro = GeneralizationResultAggregator._percent(
                    fmean(float(row["primary_score"]) for row in complete)
                )
            lines.append(
                f"| {task} | {variant} | "
                f"{len({str(row['checkpoint_id']) for row in selected})} | "
                f"{len(complete)}/{len(selected)} | {macro} |"
            )
        lines.extend(
            [
                "",
                "## Unit results",
                "",
                "| Task | Checkpoint | Variant | Dataset | Status | avg@K | pass@K | Primary |",
                "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            avg = GeneralizationResultAggregator._optional_percent(row["avg_at_k"])
            passed = GeneralizationResultAggregator._optional_percent(row["pass_at_k"])
            primary = GeneralizationResultAggregator._optional_percent(
                row["primary_score"]
            )
            lines.append(
                f"| {row['task']} | {row['checkpoint_id']} | {row['variant']} | "
                f"{row['dataset']} | {row['status']} | "
                f"{avg} | {passed} | {primary} |"
            )
        lines.extend(
            [
                "",
                "## Variant-average comparisons",
                "",
                "Only complete, contract-matched unit pairs produce deltas.",
                "",
                "| Task | Dataset | Base | Baseline | N1 avg (runs) | N3 avg (runs) | N1−Baseline | N3−Baseline | N3−N1 | Status |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in comparisons:
            values = [
                GeneralizationResultAggregator._optional_percent(row[name])
                for name in (
                    "base",
                    "baseline",
                    "n1",
                    "n3",
                    "n1_minus_baseline",
                    "n3_minus_baseline",
                    "n3_minus_n1",
                )
            ]
            lines.append(
                f"| {row['task']} | {row['dataset']} | "
                + " | ".join(
                    (
                        values[0],
                        values[1],
                        f"{values[2]} ({row['n1_runs']})",
                        f"{values[3]} ({row['n3_runs']})",
                        *values[4:],
                    )
                )
                + f" | {row['status']} |"
            )
        additional_variants = GeneralizationResultAggregator._additional_variants(rows)
        variants_by_task: dict[str, set[str]] = {}
        for row in rows:
            variants_by_task.setdefault(str(row["task"]), set()).add(
                str(row["variant"])
            )
        if additional_variants:
            lines.extend(
                [
                    "",
                    "## Additional run comparisons",
                    "",
                    "Each added run is compared within the same task and dataset contract.",
                    "",
                    "| Task | Dataset | Variant | Score | ΔBase | ΔBaseline | ΔN3 | Status |",
                    "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
                ]
            )
            for row in comparisons:
                for variant in additional_variants:
                    if variant not in variants_by_task[str(row["task"])]:
                        continue
                    values = [
                        GeneralizationResultAggregator._optional_percent(row.get(field, ""))
                        for field in (
                            GeneralizationResultAggregator._extra_score_field(variant),
                            GeneralizationResultAggregator._extra_delta_field(
                                variant, "base"
                            ),
                            GeneralizationResultAggregator._extra_delta_field(
                                variant, "baseline"
                            ),
                            GeneralizationResultAggregator._extra_delta_field(
                                variant, "n3"
                            ),
                        )
                    ]
                    lines.append(
                        f"| {row['task']} | {row['dataset']} | {variant} | "
                        + " | ".join(values)
                        + f" | {row['status']} |"
                    )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _percent(value: float) -> str:
        return f"{value * 100:.2f}%"

    @staticmethod
    def _optional_percent(value: object) -> str:
        return (
            "n/a"
            if value == ""
            else GeneralizationResultAggregator._percent(float(value))
        )

    @staticmethod
    def _mapping(value: object, owner: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"{owner} must be a mapping")
        return value

    @staticmethod
    def _json(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"JSON file must contain an object: {path}")
        return value

    @staticmethod
    def _count(value: object, owner: str, *, allow_zero: bool = False) -> int:
        if type(value) is not int or value < (0 if allow_zero else 1):
            raise ValueError(
                f"{owner} must be {'non-negative' if allow_zero else 'positive'}"
            )
        return value

    @staticmethod
    def _rate(value: object, owner: str) -> float:
        if type(value) not in {int, float} or not 0 <= float(value) <= 1:
            raise ValueError(f"{owner} must be between zero and one")
        return float(value)

    @staticmethod
    def _score(value: object, owner: str) -> float:
        if type(value) not in {int, float}:
            raise ValueError(f"{owner} must be numeric")
        return float(value)
