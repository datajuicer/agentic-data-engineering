"""Shared exact-run training telemetry freezing for all training backends."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Iterator, Mapping

from ade.engine.storage.atomic import write_json_atomic, write_text_atomic


TELEMETRY_SCHEMA_VERSION = "ade.training_telemetry.v2"
_FORBIDDEN_SEGMENT = re.compile(
    r"(?:^|[/_.-])(online|validation|val|test|heldout|held_out|operator|checkpoint|selection)(?:$|[/_.-])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TrainingRunAuditRef:
    backend: str
    entity: str
    project: str
    run_id: str
    local_run_dir: str
    base_url: str = ""
    trial_uid: str = ""

    @property
    def exact_identity(self) -> str:
        return f"{self.entity}/{self.project}/{self.run_id}"

    def to_dict(self) -> dict[str, str]:
        return {
            "backend": self.backend,
            "base_url": self.base_url,
            "entity": self.entity,
            "project": self.project,
            "run_id": self.run_id,
            "trial_uid": self.trial_uid,
            "local_run_dir": self.local_run_dir,
        }


@dataclass(frozen=True)
class MetricProfile:
    profile_id: str
    exact_names: tuple[str, ...] = ()
    anchored_patterns: tuple[str, ...] = ()
    expected_steps: tuple[int, ...] = ()
    required: bool = False


@dataclass(frozen=True)
class TelemetryExportSpec:
    audit_ref: TrainingRunAuditRef
    metric_profile: MetricProfile
    telemetry_path: Path
    manifest_path: Path
    supplemental_history_path: Path | None = None


VERL_METRIC_PROFILE = MetricProfile(
    profile_id="verl_training.v1",
    anchored_patterns=(
        r"^(?:train|actor|critic|optimizer|reward|rollout|performance|perf|response_length|prompt_length|global_seqlen|timing_s)/[A-Za-z0-9_.@/-]+$",
    ),
)
LLAMAFACTORY_METRIC_PROFILE = MetricProfile(
    profile_id="llamafactory_training.v1",
    exact_names=("loss", "learning_rate", "epoch", "grad_norm"),
    anchored_patterns=(r"^train/[A-Za-z0-9_.@/-]+$",),
)


def export_exact_wandb_telemetry(
    spec: TelemetryExportSpec,
    *,
    api_factory: Callable[..., Any] | None = None,
    page_size: int = 10_000,
) -> dict[str, Any]:
    """Fetch only the audited exact W&B run and freeze its full history."""
    local_candidates = _local_wandb_candidates(spec.audit_ref)
    if local_candidates:
        try:
            parsed_candidates = []
            for candidate in local_candidates:
                try:
                    candidate_rows = _local_wandb_history(
                        candidate, audit_ref=spec.audit_ref
                    )
                except ValueError:
                    continue
                candidate_steps = _profile_history_steps(
                    candidate_rows, spec.metric_profile
                )
                parsed_candidates.append(
                    (len(candidate_steps), len(candidate_rows), candidate, candidate_rows)
                )
            if not parsed_candidates:
                raise ValueError("local W&B binaries lack usable history records")
            _, _, local_path, history_rows = max(
                parsed_candidates,
                key=lambda item: (item[0], item[1], item[2].stat().st_mtime_ns, str(item[2])),
            )
            binary_steps = _profile_history_steps(history_rows, spec.metric_profile)
            step_sources = {"local_wandb_binary": sorted(binary_steps)}
            source_paths = {"local_wandb_binary": str(local_path)}
            source = "local_wandb_binary"
            supplemental_path = spec.supplemental_history_path
            if supplemental_path is not None and supplemental_path.is_file():
                expected_steps = set(spec.metric_profile.expected_steps)
                missing_steps = expected_steps - binary_steps
                supplemental_rows = [
                    row
                    for row in _verl_training_log_history(supplemental_path)
                    if _history_step(row) in missing_steps
                ]
                if supplemental_rows:
                    supplemental_steps = sorted({_history_step(row) for row in supplemental_rows})
                    history_rows.extend(supplemental_rows)
                    step_sources["training_log"] = supplemental_steps
                    source_paths["training_log"] = str(supplemental_path)
                    source = "local_wandb_binary+training_log"
            return freeze_training_telemetry(
                spec,
                returned_identity=spec.audit_ref.exact_identity,
                history_pages=_history_pages(history_rows, page_size=page_size),
                source=source,
                source_path=str(local_path),
                source_paths=source_paths,
                step_sources=step_sources,
            )
        except Exception as exc:
            return write_unavailable_telemetry_manifest(
                spec, reason=f"local_wandb_export_incomplete:{type(exc).__name__}"
            )
    try:
        if api_factory is None:
            import wandb

            api_factory = wandb.Api
        api_kwargs = {"overrides": {"base_url": spec.audit_ref.base_url}} if spec.audit_ref.base_url else {}
        api = api_factory(**api_kwargs)
        run = api.run(spec.audit_ref.exact_identity)
        returned_identity = _returned_identity(run)
        pages = _history_pages(run.scan_history(page_size=page_size), page_size=page_size)
        return freeze_training_telemetry(
            spec,
            returned_identity=returned_identity,
            history_pages=pages,
        )
    except Exception as exc:
        return write_unavailable_telemetry_manifest(
            spec, reason=f"wandb_export_incomplete:{type(exc).__name__}"
        )


def freeze_training_telemetry(
    spec: TelemetryExportSpec,
    *,
    returned_identity: str,
    history_pages: Iterable[Iterable[Mapping[str, Any]]],
    source: str = "wandb_history",
    source_path: str | None = None,
    source_paths: Mapping[str, str] | None = None,
    step_sources: Mapping[str, list[int]] | None = None,
) -> dict[str, Any]:
    if returned_identity != spec.audit_ref.exact_identity:
        raise ValueError("W&B exporter returned the wrong exact run identity")
    patterns = tuple(_compile_anchored(pattern) for pattern in spec.metric_profile.anchored_patterns)
    rows: list[dict[str, Any]] = []
    rejected: set[str] = set()
    observed_steps: set[int] = set()
    for page_index, page in enumerate(history_pages):
        for history_index, history in enumerate(page):
            if not isinstance(history, Mapping):
                raise ValueError("W&B history rows must be mappings")
            step = _history_step(history)
            accepted_metric = False
            for name, value in history.items():
                if name in {"_step", "step", "wandb_history_step"}:
                    continue
                metric_name = str(name)
                if _FORBIDDEN_SEGMENT.search(metric_name):
                    rejected.add(metric_name)
                    continue
                if not _metric_allowed(metric_name, spec.metric_profile, patterns):
                    rejected.add(metric_name)
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                accepted_metric = True
                rows.append(
                    {
                        "schema_version": TELEMETRY_SCHEMA_VERSION,
                        "backend": spec.audit_ref.backend,
                        "trial_uid": spec.audit_ref.trial_uid,
                        "source_run_id": spec.audit_ref.run_id,
                        "wandb_history_step": step,
                        "metric_name": metric_name,
                        "metric_value": float(value),
                        "source_page_index": page_index,
                        "source_row_index": history_index,
                    }
                )
            if accepted_metric:
                observed_steps.add(step)
    rows.sort(key=lambda row: (row["wandb_history_step"], row["metric_name"]))
    missing_steps = sorted(set(spec.metric_profile.expected_steps) - observed_steps)
    status = "complete" if not missing_steps else "incomplete"
    if spec.metric_profile.required and status != "complete":
        raise ValueError(f"required training telemetry is incomplete; missing steps: {missing_steps}")
    content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        for row in rows
    )
    if any(_FORBIDDEN_SEGMENT.search(str(row["metric_name"])) for row in rows):
        raise ValueError("forbidden metric value survived telemetry projection")
    spec.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(spec.telemetry_path, content)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    summary_path = spec.telemetry_path.with_name("training_telemetry_summary.json")
    write_json_atomic(summary_path, _telemetry_summary(rows))
    summary_digest = _sha256(summary_path)
    manifest = {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "status": status,
        "source": source,
        "source_identity": {
            "base_url": spec.audit_ref.base_url,
            "entity": spec.audit_ref.entity,
            "project": spec.audit_ref.project,
            "run_id": spec.audit_ref.run_id,
        },
        "exact_run_identity": spec.audit_ref.exact_identity,
        "trial_uid": spec.audit_ref.trial_uid,
        "backend": spec.audit_ref.backend,
        "metric_profile_id": spec.metric_profile.profile_id,
        "telemetry_path": str(spec.telemetry_path),
        "telemetry_row_count": len(rows),
        "telemetry_sha256": digest,
        "telemetry_summary_path": str(summary_path),
        "telemetry_summary_sha256": summary_digest,
        "observed_steps": sorted(observed_steps),
        "missing_expected_steps": missing_steps,
        "rejected_metric_names": sorted(rejected),
        "excluded_remote_surfaces": ["summary", "config", "files", "media", "artifacts"],
    }
    if source_path is not None:
        manifest["source_path"] = source_path
    if source_paths is not None:
        manifest["source_paths"] = dict(source_paths)
    if step_sources is not None:
        manifest["step_sources"] = {
            name: sorted(int(step) for step in steps)
            for name, steps in step_sources.items()
        }
    write_json_atomic(spec.manifest_path, manifest)
    return manifest


def _local_wandb_candidates(audit_ref: TrainingRunAuditRef) -> list[Path]:
    local_root = Path(audit_ref.local_run_dir)
    if not local_root.is_dir():
        return []
    return sorted(local_root.rglob(f"run-{audit_ref.run_id}.wandb"), key=str)


def _local_wandb_history(
    path: Path,
    *,
    audit_ref: TrainingRunAuditRef,
) -> list[dict[str, Any]]:
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    store = DataStore()
    store.open_for_scan(str(path))
    rows: list[dict[str, Any]] = []
    saw_run = False
    try:
        while True:
            data = store.scan_data()
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            record_type = record.WhichOneof("record_type")
            if record_type == "run":
                saw_run = True
                if record.run.run_id != audit_ref.run_id:
                    raise ValueError("local W&B binary has the wrong run id")
                if audit_ref.project and record.run.project != audit_ref.project:
                    raise ValueError("local W&B binary has the wrong project")
            if record_type != "history":
                continue
            row: dict[str, Any] = {}
            for item in record.history.item:
                key = item.key or "/".join(item.nested_key)
                if not key:
                    continue
                row[key] = json.loads(item.value_json)
            if row:
                rows.append(row)
    finally:
        store.close()
    if not saw_run:
        raise ValueError("local W&B binary lacks an exact-run record")
    if not rows:
        raise ValueError("local W&B binary lacks history records")
    return rows


def _profile_history_steps(
    rows: Iterable[Mapping[str, Any]], profile: MetricProfile
) -> set[int]:
    patterns = tuple(_compile_anchored(pattern) for pattern in profile.anchored_patterns)
    steps: set[int] = set()
    for row in rows:
        if any(
            name not in {"_step", "step", "wandb_history_step", "training/global_step"}
            and not _FORBIDDEN_SEGMENT.search(str(name))
            and _metric_allowed(str(name), profile, patterns)
            and not isinstance(value, bool)
            and isinstance(value, (int, float))
            for name, value in row.items()
        ):
            steps.add(_history_step(row))
    return steps


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_VERL_STEP = re.compile(r"(?:^|\s)step:(\d+)(?:\s|$)")
_VERL_METRIC = re.compile(
    r"^([A-Za-z0-9_.@/-]+):(?:np\.(?:float64|int64)\()?"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def _verl_training_log_history(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = _ANSI_ESCAPE.sub("", raw_line)
        step_match = _VERL_STEP.search(line)
        if step_match is None:
            continue
        row: dict[str, Any] = {"_step": int(step_match.group(1))}
        for segment in line.split(" - "):
            metric_match = _VERL_METRIC.match(segment.strip())
            if metric_match is not None:
                row[metric_match.group(1)] = float(metric_match.group(2))
        if len(row) > 1:
            rows.append(row)
    return rows


def _telemetry_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        metrics.setdefault(str(row["metric_name"]), []).append(
            (int(row["wandb_history_step"]), float(row["metric_value"]))
        )
    summaries: dict[str, dict[str, Any]] = {}
    for name, values in sorted(metrics.items()):
        ordered = sorted(values)
        metric_values = [value for _, value in ordered]
        summaries[name] = {
            "count": len(ordered),
            "step_min": ordered[0][0],
            "step_max": ordered[-1][0],
            "first": ordered[0][1],
            "last": ordered[-1][1],
            "min": min(metric_values),
            "max": max(metric_values),
            "mean": sum(metric_values) / len(metric_values),
        }
    return {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "summary_role": "complete_normalized_training_telemetry",
        "telemetry_row_count": len(rows),
        "metric_count": len(summaries),
        "metric_summaries": summaries,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_unavailable_telemetry_manifest(
    spec: TelemetryExportSpec, *, reason: str
) -> dict[str, Any]:
    manifest = {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "status": "unavailable",
        "training_telemetry_available": False,
        "reason": reason,
        "exact_run_identity": spec.audit_ref.exact_identity,
        "backend": spec.audit_ref.backend,
        "ranking_impact": "none",
    }
    spec.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(spec.manifest_path, manifest)
    return manifest


def _compile_anchored(pattern: str) -> re.Pattern[str]:
    if not pattern.startswith("^") or not pattern.endswith("$"):
        raise ValueError("telemetry metric patterns must be anchored with ^ and $")
    return re.compile(pattern)


def _metric_allowed(
    name: str, profile: MetricProfile, patterns: tuple[re.Pattern[str], ...]
) -> bool:
    return name in profile.exact_names or any(pattern.fullmatch(name) for pattern in patterns)


def _history_step(row: Mapping[str, Any]) -> int:
    value = row.get(
        "training/global_step",
        row.get("_step", row.get("wandb_history_step", row.get("step"))),
    )
    if isinstance(value, bool):
        raise ValueError("W&B history step must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("W&B history row lacks an integer step") from exc


def _returned_identity(run: Any) -> str:
    entity = str(getattr(run, "entity", "") or "")
    project_value = getattr(run, "project", "")
    project = str(getattr(project_value, "name", project_value) or "")
    run_id = str(getattr(run, "id", "") or "")
    if not entity or not project or not run_id:
        raise ValueError("W&B run response lacks exact identity fields")
    return f"{entity}/{project}/{run_id}"


def _history_pages(rows: Iterable[Mapping[str, Any]], *, page_size: int) -> Iterator[list[Mapping[str, Any]]]:
    page: list[Mapping[str, Any]] = []
    for row in rows:
        page.append(row)
        if len(page) >= page_size:
            yield page
            page = []
    if page:
        yield page
