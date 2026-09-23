"""Build Analyzer inputs from immutable Engine trial artifact manifests."""

from __future__ import annotations

import hashlib
import json
import io
import gzip
from collections import Counter
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

from ade.core.experiment import validate_analysis_profile
from ade.tasks.contracts import AnalyzerEvidenceSpec


_SCHEMA_VERSION = "2"
_SCOPE_FIELDS = ("run_id", "coordinator_id", "plan_id", "trial_id")


class EngineObjectReader(Protocol):
    def read_json(self, uri: str) -> dict[str, Any]: ...

    def read_bytes(self, uri: str) -> bytes: ...


class EngineExperimentPackageBuilder:
    def __init__(self, reader: EngineObjectReader) -> None:
        self.reader = reader

    def build(
        self,
        manifest_ref: str,
        *,
        result_ref: str | None = None,
        task_id: str | None = None,
        evidence_specs: tuple[AnalyzerEvidenceSpec, ...] = (),
    ) -> dict[str, bytes]:
        engine_manifest = self.reader.read_json(manifest_ref)
        units = engine_manifest.get("units")
        schema_version = engine_manifest.get("schema_version")
        if schema_version != _SCHEMA_VERSION:
            raise ValueError("Engine raw manifest schema_version must be 2")
        if not isinstance(units, list):
            raise ValueError("Engine raw manifest units must be a list")
        if result_ref is not None:
            result = self.reader.read_json(result_ref)
            trial_manifest = result.get("trial_artifact_manifest_path")
            if trial_manifest:
                return self._build_trial_artifact_reference(
                    Path(str(trial_manifest)),
                    task_id=task_id,
                    evidence_specs=evidence_specs,
                    analysis=engine_manifest.get("analysis"),
                )
        return self._build_package(engine_manifest, units)

    @staticmethod
    def _build_trial_artifact_reference(
        manifest_path: Path,
        *,
        task_id: str | None,
        evidence_specs: tuple[AnalyzerEvidenceSpec, ...],
        analysis: object,
    ) -> dict[str, bytes]:
        path = manifest_path.resolve()
        if path.name != "manifest.json" or not path.is_file():
            raise ValueError("published trial artifact manifest is unavailable")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("published trial artifact manifest is invalid") from error
        root = path.parent.resolve()
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != "ade.trial_artifacts.v1"
            or Path(str(manifest.get("artifact_root") or "")).resolve() != root
        ):
            raise ValueError("published trial artifact manifest contract is invalid")
        _scope_identity(manifest)
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("published trial artifact manifest lacks artifacts")
        if not task_id or not evidence_specs:
            raise ValueError("task-owned Analyzer evidence specs are required")
        seen: set[str] = set()
        published_artifacts: list[dict[str, Any]] = []
        package: dict[str, bytes] = {}
        visible_kinds = {
            kind
            for spec in evidence_specs
            for kind in spec.artifact_selector.kinds
        }
        for artifact_index, item in enumerate(artifacts):
            if not isinstance(item, dict):
                raise ValueError("published trial artifacts must be objects")
            artifact_id = str(item.get("id") or "")
            if not artifact_id or artifact_id in seen:
                raise ValueError("published trial artifact IDs must be unique")
            seen.add(artifact_id)
            relative_value = item.get("path")
            visible = (
                item.get("category") == "audit"
                or item.get("kind") in visible_kinds
            ) and item.get("kind") != "online_validation"
            if relative_value is None:
                if visible:
                    published_artifacts.append(dict(item))
                continue
            relative = PurePosixPath(str(relative_value))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("published trial artifact path is unsafe")
            if not visible:
                continue
            package_dir = (
                artifact_id
                if len(artifact_id) <= 120
                else f"artifact-{artifact_index:04d}"
            )
            package_path = f"experiment/artifacts/{package_dir}/{relative.name}"
            if _is_durable_behavior(item):
                index_binding = item.get("index")
                if not isinstance(index_binding, Mapping):
                    raise ValueError(
                        f"durable behavior index is missing: {artifact_id}"
                    )
                index_relative = PurePosixPath(str(index_binding.get("path") or ""))
                if (
                    not index_relative.parts
                    or index_relative.is_absolute()
                    or ".." in index_relative.parts
                ):
                    raise ValueError(
                        f"durable behavior index path is unsafe: {artifact_id}"
                    )
                index_target = root.joinpath(*index_relative.parts)
                index_content = _read_verified_binding(
                    index_target,
                    index_binding,
                    f"{artifact_id} index",
                )
                index_package_path = f"experiment/indexes/{package_dir}.json"
                package[index_package_path] = index_content
                published = dict(item)
                published["path"] = package_path
                published["storage"] = {
                    "mode": "durable_reference",
                    "uri": _trial_artifact_uri(manifest, relative),
                    "sha256": str(item["sha256"]),
                    "size_bytes": int(item["size_bytes"]),
                }
                published["index"] = {
                    **dict(index_binding),
                    "path": index_package_path,
                }
                published_artifacts.append(published)
                continue
            target = root.joinpath(*relative.parts)
            if item.get("kind") == "candidate_pool":
                if (
                    not target.is_file()
                    or target.is_symlink()
                    or not target.resolve().is_relative_to(root)
                    or type(item.get("size_bytes")) is not int
                    or target.stat().st_size != item["size_bytes"]
                    or not isinstance(item.get("sha256"), str)
                ):
                    raise ValueError(
                        f"published candidate pool binding is invalid: {artifact_id}"
                    )
                published = dict(item)
                published["path"] = package_path
                published["storage"] = {
                    "mode": "durable_reference",
                    "uri": _trial_artifact_uri(manifest, relative),
                    "sha256": str(item["sha256"]),
                    "size_bytes": int(item["size_bytes"]),
                }
                published_artifacts.append(published)
                continue
            if (
                not target.exists()
                or target.is_symlink()
                or not target.resolve().is_relative_to(root)
            ):
                raise ValueError(
                    f"published trial artifact path is unavailable: {relative}"
                )
            if not target.is_file():
                raise ValueError(f"Analyzer-visible artifact is not a file: {artifact_id}")
            package[package_path] = _verify_artifact_file(target, item, artifact_id)
            published = dict(item)
            published["path"] = package_path
            published_artifacts.append(published)
        package_manifest = dict(manifest)
        if analysis is not None:
            validate_analysis_profile(analysis)
            package_manifest["analysis"] = analysis
        package_manifest["artifact_root"] = "experiment/artifacts"
        package_manifest["source_manifest_binding"] = {
            "schema_version": str(manifest["schema_version"]),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        package_manifest["artifacts"] = published_artifacts
        catalog = _evidence_catalog(
            task_id=task_id,
            manifest=package_manifest,
            package=package,
            evidence_specs=evidence_specs,
        )
        package["experiment/manifest.json"] = (
            json.dumps(package_manifest, indent=2, sort_keys=True) + "\n"
        ).encode()
        package["experiment/evidence-catalog.json"] = (
            json.dumps(catalog, indent=2, sort_keys=True) + "\n"
        ).encode()
        package["experiment/EVIDENCE_GUIDE.md"] = _evidence_guide(catalog).encode()
        return package


    def _build_package(
        self,
        manifest: dict[str, Any],
        units: list[Any],
    ) -> dict[str, bytes]:
        scope = _scope_identity(manifest)
        analysis = manifest.get("analysis")
        validate_analysis_profile(analysis)
        package: dict[str, bytes] = {}
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in units:
            if not isinstance(item, dict):
                raise ValueError("Engine analysis manifest units must be objects")
            if item.get("visibility") != "agent":
                continue
            unit_id = str(item.get("unit_id") or "")
            uri = str(item.get("uri") or "")
            filename = str(item.get("filename") or "")
            media_type = str(item.get("media_type") or "")
            if (
                not unit_id
                or not uri
                or not filename
                or not media_type
                or "/" in unit_id
                or "\\" in unit_id
                or unit_id in {".", ".."}
            ):
                raise ValueError(
                    "Engine analysis units require safe unit_id, uri, filename, and media_type"
                )
            relative_filename = PurePosixPath(filename)
            if (
                relative_filename.is_absolute()
                or len(relative_filename.parts) != 1
                or ".." in relative_filename.parts
                or "\\" in filename
            ):
                raise ValueError(f"unsafe Engine analysis filename: {filename}")
            if unit_id in seen:
                raise ValueError(f"duplicate Engine analysis unit ID: {unit_id}")
            seen.add(unit_id)
            content = self.reader.read_bytes(uri)
            digest = hashlib.sha256(content).hexdigest()
            expected_digest = item.get("sha256")
            if expected_digest is not None and str(expected_digest) != digest:
                raise ValueError(f"Engine analysis unit digest mismatch: {unit_id}")
            expected_size = item.get("size_bytes")
            if expected_size is not None and int(expected_size) != len(content):
                raise ValueError(f"Engine analysis unit size mismatch: {unit_id}")
            path = f"experiment/units/{unit_id}/{filename}"
            package[path] = content
            unit = {
                "unit_id": unit_id,
                "kind": str(item.get("kind") or ""),
                "path": path,
                "media_type": media_type,
                "sha256": digest,
                "size_bytes": len(content),
            }
            if item.get("step") is not None:
                unit["step"] = int(item["step"])
            selected.append(unit)
        referenced = _analysis_unit_ids(analysis["topology"])
        undeclared = sorted(referenced - seen)
        if undeclared:
            raise ValueError(
                f"analysis topology references undeclared unit IDs: {undeclared}"
            )
        package_manifest = {
            "schema_version": _SCHEMA_VERSION,
            **scope,
            "unit_ids": [item["unit_id"] for item in selected],
            "units": selected,
            "analysis": analysis,
        }
        package["experiment/manifest.json"] = json.dumps(
            package_manifest,
            indent=2,
            sort_keys=True,
        ).encode()
        return package


def attach_curriculum_realization(
    package: Mapping[str, bytes], report_content: bytes
) -> dict[str, bytes]:
    """Attach the accepted pre-training Curriculum controls to a package."""
    try:
        report = json.loads(report_content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Curriculum realization report is invalid") from error
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != "ade.curriculum_realization.v1"
        or report.get("realization_status")
        not in {"verified", "deviated", "unverified"}
    ):
        raise ValueError("Curriculum realization report is not accepted")
    schedule = report.get("schedule")
    policy_source = report.get("policy_source")
    if (
        not isinstance(schedule, dict)
        or schedule.get("schema_version") != "ade.curriculum_schedule.v1"
        or not isinstance(schedule.get("summary"), dict)
        or not isinstance(policy_source, str)
        or not policy_source.strip()
    ):
        raise ValueError("Curriculum realization package is incomplete")
    files = dict(package)
    additions = {
        "experiment/realization/final-realization.json": report_content,
        "experiment/realization/curriculum.py": policy_source.encode(),
        "experiment/realization/curriculum-schedule.json": (
            json.dumps(schedule, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode(),
        "experiment/realization/schedule-summary.json": (
            json.dumps(
                schedule["summary"],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    }
    collisions = sorted(set(files).intersection(additions))
    if collisions:
        raise ValueError(
            f"Curriculum realization package paths already exist: {collisions}"
        )
    files.update(additions)
    try:
        manifest = json.loads(files["experiment/manifest.json"])
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Curriculum Experiment Package manifest is invalid") from error
    if not isinstance(manifest, dict):
        raise ValueError("Curriculum Experiment Package manifest must be an object")
    manifest["realization"] = {
        "schema_version": "1",
        "status": report["realization_status"],
        "reason": report.get("reason"),
        "files": sorted(additions),
    }
    files["experiment/manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    return files


def _verify_artifact_file(
    path: Path,
    artifact: dict[str, Any],
    artifact_id: str,
) -> bytes:
    if not path.is_file():
        raise ValueError(f"published artifact is not a file: {artifact_id}")
    content = path.read_bytes()
    if artifact.get("size_bytes") != len(content):
        raise ValueError(f"published artifact size mismatch: {artifact_id}")
    if artifact.get("sha256") != hashlib.sha256(content).hexdigest():
        raise ValueError(f"published artifact digest mismatch: {artifact_id}")
    return content


def _read_verified_binding(
    path: Path,
    binding: Mapping[str, Any],
    artifact_id: str,
) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"published artifact is not a file: {artifact_id}")
    content = path.read_bytes()
    if binding.get("size_bytes") != len(content):
        raise ValueError(f"published artifact size mismatch: {artifact_id}")
    if binding.get("sha256") != hashlib.sha256(content).hexdigest():
        raise ValueError(f"published artifact digest mismatch: {artifact_id}")
    return content


def _is_durable_behavior(artifact: Mapping[str, Any]) -> bool:
    return (
        artifact.get("category") == "model_behavior"
        and artifact.get("kind") in {"offline_validation", "training_rollout"}
        and isinstance(artifact.get("sha256"), str)
        and isinstance(artifact.get("size_bytes"), int)
        and isinstance(artifact.get("index"), Mapping)
    )


def _trial_artifact_uri(
    manifest: Mapping[str, Any],
    relative: PurePosixPath,
) -> str:
    scope = _scope_identity(manifest)
    attempt_id = str(manifest.get("attempt_id") or "")
    if not attempt_id:
        raise ValueError("published trial artifact attempt_id is missing")
    return "/".join(
        (
            "engine://trial-artifacts",
            *(str(scope[field]) for field in _SCOPE_FIELDS),
            attempt_id,
            relative.as_posix(),
        )
    )


def _evidence_catalog(
    *,
    task_id: str,
    manifest: Mapping[str, Any],
    package: Mapping[str, bytes],
    evidence_specs: tuple[AnalyzerEvidenceSpec, ...],
) -> dict[str, Any]:
    analysis = manifest.get("analysis")
    topology = analysis.get("topology") if isinstance(analysis, Mapping) else {}
    group_evidence = (
        isinstance(topology, Mapping)
        and topology.get("semantic_evidence_unit") == "groups"
        and any(spec.coverage.unit == "groups" for spec in evidence_specs)
    )
    semantic_steps = {
        int(step)
        for step in (
            topology.get("semantic_evidence_steps", ())
            if isinstance(topology, Mapping)
            else ()
        )
    }
    artifacts = tuple(
        item for item in manifest.get("artifacts", ()) if isinstance(item, Mapping)
    )
    pools = []
    pool_artifact_ids: set[str] = set()
    for spec in evidence_specs:
        group_coverage = (
            group_evidence
            and spec.pool_id == "training_rollout"
            and spec.coverage.unit == "groups"
        )
        selected = [
            item
            for item in artifacts
            if item.get("category") in spec.artifact_selector.categories
            and item.get("kind") in spec.artifact_selector.kinds
            and (
                item.get("path") in package
                or _is_package_durable_reference(item)
            )
        ]
        if group_coverage:
            selected = [
                item
                for item in selected
                if int(
                    (
                        item.get("artifact_position")
                        or (item.get("metadata") or {}).get("artifact_position")
                        or {}
                    ).get("value", -1)
                )
                in semantic_steps
            ]
        records = []
        artifact_entries = []
        group_counts: Counter[str] = Counter()
        position_counts: Counter[str] = Counter()
        group_members: dict[str, list[dict[str, str]]] = {}
        candidate_relations: list[dict[str, str]] = []
        for artifact in selected:
            artifact_id = str(artifact["id"])
            artifact_record_start = len(records)
            artifact_group_ids: set[str] = set()
            pool_artifact_ids.add(artifact_id)
            path = str(artifact["path"])
            durable_index = _durable_index(package, artifact, artifact_id)
            decoded = (
                None
                if durable_index is not None
                else _jsonl_records(package[path], path)
            )
            eligible_count = 0
            indexed_records = (
                durable_index["records"]
                if durable_index is not None
                else tuple(
                    {
                        "record_id": record.get("record_id"),
                        "line_index": line_index,
                        "record": record,
                    }
                    for line_index, record in enumerate(decoded or ())
                )
            )
            for indexed in indexed_records:
                if not isinstance(indexed, Mapping):
                    raise ValueError(f"Analyzer evidence index is invalid: {artifact_id}")
                record = indexed.get("record")
                projection = indexed.get("projection")
                view = (
                    record
                    if isinstance(record, Mapping)
                    else projection
                    if isinstance(projection, Mapping)
                    else {}
                )
                record_id = str(indexed.get("record_id") or "")
                present = indexed.get("review_fields_present")
                fields_present = (
                    bool(present.get(spec.review_fields.question))
                    and bool(present.get(spec.review_fields.response))
                    if isinstance(present, Mapping)
                    else _nonempty_text(_field_value(view, spec.review_fields.question))
                    and _nonempty_text(_field_value(view, spec.review_fields.response))
                )
                if not record_id or not fields_present:
                    continue
                eligible_count += 1
                grouping = spec.grouping
                group_id = (
                    _field_value(view, grouping.group_key)
                    if grouping is not None and grouping.group_key
                    else None
                )
                if (
                    group_coverage
                    and group_id is not None
                ):
                    group_id = f"{artifact_id}::{group_id}"
                position = (
                    _field_value(view, grouping.position_key)
                    if grouping is not None and grouping.position_key
                    else None
                )
                response_index = (
                    _field_value(view, grouping.response_index_key)
                    if grouping is not None and grouping.response_index_key
                    else None
                )
                identity = {
                    "source_artifact_id": artifact_id,
                    "source_record_id": record_id,
                }
                entry: dict[str, Any] = {
                    **identity,
                    "line_index": int(indexed.get("line_index", -1)),
                    "group_id": None if group_id is None else str(group_id),
                    "position": position,
                    "response_index": response_index,
                    "context": {
                        field.rsplit(".", 1)[-1]: _field_value(view, field)
                        for field in (
                            spec.context_fields
                            if group_coverage or spec.pool_id != "training_rollout"
                            else spec.context_fields[:8]
                        )
                    },
                }
                records.append(entry)
                if group_id is not None:
                    group_counts[f"{artifact_id}::{group_id}"] += 1
                    group_members.setdefault(str(group_id), []).append(identity)
                    artifact_group_ids.add(str(group_id))
                if position is not None:
                    position_counts[str(position)] += 1
                candidate_record_id = view.get("candidate_record_id")
                if candidate_record_id is not None:
                    candidate_relations.append(
                        {
                            **identity,
                            "candidate_record_id": str(candidate_record_id),
                        }
                    )
            if group_coverage:
                rollout_n = artifact.get("rollout_n")
                if type(rollout_n) is not int or rollout_n < 1:
                    raise ValueError(
                        f"complete-group artifact lacks rollout_n: {artifact_id}"
                    )
                complete_group_ids = {
                    group_id
                    for group_id in artifact_group_ids
                    if group_counts[f"{artifact_id}::{group_id}"] == rollout_n
                }
                records[artifact_record_start:] = [
                    entry
                    for entry in records[artifact_record_start:]
                    if entry["group_id"] in complete_group_ids
                ]
                eligible_count = len(records) - artifact_record_start
                for group_id in artifact_group_ids - complete_group_ids:
                    del group_counts[f"{artifact_id}::{group_id}"]
                    group_members.pop(group_id, None)
            artifact_entries.append(
                {
                    "artifact_id": artifact_id,
                    "path": path,
                    "status": artifact.get("status"),
                    "record_count": (
                        int(durable_index["behavior"]["record_count"])
                        if durable_index is not None
                        else len(decoded or ())
                    ),
                    "eligible_record_count": eligible_count,
                    "artifact_position": artifact.get("metadata", {}).get(
                        "artifact_position"
                    ) or artifact.get("artifact_position"),
                }
            )
        if group_coverage:
            position_counts = Counter(
                str(entry["position"])
                for entry in records
                if entry["position"] is not None
            )
        for entry in records:
            if entry["group_id"] is not None:
                entry["group_size"] = group_counts[
                    f"{entry['source_artifact_id']}::{entry['group_id']}"
                ]
        coverage = asdict(spec.coverage)
        if not group_coverage:
            coverage.pop("unit", None)
        eligible_group_count = len(group_counts)
        pools.append(
            {
                "pool_id": spec.pool_id,
                "semantic_role": spec.semantic_role,
                "artifact_selector": asdict(spec.artifact_selector),
                "coverage": coverage,
                "review_fields": asdict(spec.review_fields),
                "grouping": asdict(spec.grouping) if spec.grouping else None,
                "context_fields": list(spec.context_fields),
                "artifacts": artifact_entries,
                "eligible_record_count": len(records),
                **(
                    {"eligible_group_count": eligible_group_count}
                    if group_coverage
                    else {}
                ),
                "records": records,
                "group_members": group_members,
                "group_size_distribution": _count_distribution(group_counts.values()),
                "position_counts": dict(sorted(position_counts.items())),
                "candidate_relations": candidate_relations,
            }
        )
    controls = [
        {
            "artifact_id": str(item["id"]),
            "category": item.get("category"),
            "kind": item.get("kind"),
            "path": item.get("path"),
        }
        for item in artifacts
        if str(item.get("id")) not in pool_artifact_ids
    ]
    return {
        "schema_version": (
            "ade.analyzer_evidence_catalog.v2"
            if group_evidence
            else "ade.analyzer_evidence_catalog.v1"
        ),
        "task_id": task_id,
        "scope": {field: manifest[field] for field in _SCOPE_FIELDS},
        "source_manifest_binding": manifest["source_manifest_binding"],
        "selected_checkpoint_id": manifest.get("selected_checkpoint_id"),
        "pools": pools,
        "controls": controls,
    }


def _is_package_durable_reference(artifact: Mapping[str, Any]) -> bool:
    storage = artifact.get("storage")
    index = artifact.get("index")
    return (
        isinstance(storage, Mapping)
        and storage.get("mode") == "durable_reference"
        and isinstance(storage.get("uri"), str)
        and isinstance(index, Mapping)
        and isinstance(index.get("path"), str)
    )


def _durable_index(
    package: Mapping[str, bytes],
    artifact: Mapping[str, Any],
    artifact_id: str,
) -> dict[str, Any] | None:
    if not _is_package_durable_reference(artifact):
        return None
    binding = artifact["index"]
    assert isinstance(binding, Mapping)
    path = str(binding["path"])
    try:
        payload = json.loads(package[path])
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Analyzer evidence index is invalid: {artifact_id}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "ade.behavior_index.v1"
        or not isinstance(payload.get("behavior"), dict)
        or not isinstance(payload.get("records"), list)
        or payload["behavior"].get("sha256")
        != artifact["storage"].get("sha256")
        or payload["behavior"].get("size_bytes")
        != artifact["storage"].get("size_bytes")
        or payload["behavior"].get("record_count") != len(payload["records"])
    ):
        raise ValueError(f"Analyzer evidence index binding is invalid: {artifact_id}")
    return payload


def _jsonl_records(content: bytes, path: str) -> list[dict[str, Any]]:
    decoded = gzip.decompress(content).decode("utf-8") if path.endswith(".gz") else content.decode("utf-8")
    # JSONL records are delimited by LF.  str.splitlines() also treats
    # Unicode line/paragraph separators (U+2028/U+2029) as delimiters; those
    # characters are valid inside a generated JSON string and are reachable
    # in model responses.
    rows = [json.loads(line) for line in decoded.split("\n") if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Analyzer evidence artifact must contain JSON objects: {path}")
    return rows


def _field_value(record: Mapping[str, Any], path: str | None) -> Any:
    value: Any = record
    for part in (path or "").split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _count_distribution(values: Any) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _evidence_guide(catalog: Mapping[str, Any]) -> str:
    lines = [
        "# Analyzer Evidence Guide",
        "",
        f"Task: `{catalog['task_id']}`",
        "",
        "This guide is generated from the frozen evidence catalog and current Trial artifacts.",
        "Read `evidence-catalog.json` before opening record files.",
        "",
        "## Review pools",
        "",
    ]
    for pool in catalog["pools"]:
        coverage = pool["coverage"]
        requirement = (
            f"all eligible {coverage.get('unit', 'records')}"
            if coverage["mode"] == "all"
            else f"at least {coverage['fraction']:.0%} of eligible {coverage.get('unit', 'records')} globally"
        )
        eligible_label = (
            f"Eligible groups: {pool['eligible_group_count']}; eligible rows: {pool['eligible_record_count']}"
            if coverage.get("unit") == "groups"
            else f"Eligible records: {pool['eligible_record_count']}"
        )
        lines.extend(
            [
                f"### `{pool['pool_id']}`",
                "",
                str(pool["semantic_role"]),
                "",
                f"{eligible_label}; coverage: {requirement}.",
                f"Question field: `{pool['review_fields']['question']}`; response field: `{pool['review_fields']['response']}`.",
                f"Artifacts: {', '.join(item['artifact_id'] for item in pool['artifacts']) or 'none'}.",
                f"Position counts: {json.dumps(pool['position_counts'], sort_keys=True)}.",
                f"Group-size distribution: {json.dumps(pool['group_size_distribution'], sort_keys=True)}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Bound controls",
            "",
            "Non-pool audit and offline control artifacts remain readable through the Analyzer manifest.",
            "Online evaluation result payloads are intentionally absent from this package.",
            "",
        ]
    )
    for control in catalog["controls"]:
        if control.get("kind") == "candidate_pool":
            lines.extend(
                [
                    f"`{control['artifact_id']}` (`candidate_pool`) is supporting candidate input; it is not a Review pool.",
                    "",
                ]
            )
    return "\n".join(lines)


def _verify_artifact_directory(
    path: Path,
    artifact: dict[str, Any],
    artifact_id: str,
) -> None:
    if not path.is_dir():
        raise ValueError(f"published artifact is not a directory: {artifact_id}")
    declared = artifact.get("files")
    if not isinstance(declared, list):
        raise ValueError(f"published artifact file inventory is invalid: {artifact_id}")
    actual: list[dict[str, Any]] = []
    tree = hashlib.sha256()
    total = 0
    for child in sorted(path.rglob("*"), key=lambda value: value.as_posix()):
        if child.is_symlink():
            raise ValueError(f"published artifact contains a symlink: {artifact_id}")
        if not child.is_file():
            continue
        relative = child.relative_to(path).as_posix()
        content = child.read_bytes()
        size = len(content)
        digest = hashlib.sha256(content).hexdigest()
        actual.append({"path": relative, "sha256": digest, "size_bytes": size})
        total += size
        tree.update(f"{relative}\0{size}\0{digest}\n".encode())
    if declared != actual:
        raise ValueError(f"published artifact file inventory mismatch: {artifact_id}")
    if artifact.get("file_count") != len(actual) or artifact.get("size_bytes") != total:
        raise ValueError(f"published artifact tree size mismatch: {artifact_id}")
    if artifact.get("tree_sha256") != tree.hexdigest():
        raise ValueError(f"published artifact tree digest mismatch: {artifact_id}")


def _analysis_unit_ids(value: Any, *, key: str = "") -> set[str]:
    if isinstance(value, dict):
        result: set[str] = set()
        for child_key, child in value.items():
            result.update(_analysis_unit_ids(child, key=str(child_key)))
        return result
    if isinstance(value, list) and not key.endswith("_unit_ids"):
        result: set[str] = set()
        for item in value:
            result.update(_analysis_unit_ids(item))
        return result
    if key.endswith("_unit_id"):
        return {str(value)} if isinstance(value, str) and value else set()
    if key.endswith("_unit_ids"):
        if not isinstance(value, list):
            raise ValueError(f"analysis topology {key} must be a list")
        return {str(item) for item in value if str(item)}
    return set()


def encode_experiment_package(package: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for name, content in sorted(package.items()):
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe Experiment Package path: {name}")
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_STORED if name.endswith(".gz") else ZIP_DEFLATED
            info.external_attr = 0o444 << 16
            archive.writestr(info, content)
    return buffer.getvalue()


def decode_experiment_package(content: bytes) -> dict[str, bytes]:
    with ZipFile(io.BytesIO(content), "r") as archive:
        result = {}
        for info in archive.infolist():
            relative = PurePosixPath(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe Experiment Package path: {info.filename}")
            result[info.filename] = archive.read(info)
        manifest_content = result.get("experiment/manifest.json")
        if manifest_content is None:
            raise ValueError("Experiment Package manifest is missing")
        try:
            manifest = json.loads(manifest_content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Experiment Package manifest is invalid") from error
        if not isinstance(manifest, dict):
            raise ValueError("Experiment Package manifest must be an object")
        schema_version = manifest.get("schema_version")
        if schema_version == _SCHEMA_VERSION:
            validate_analysis_profile(manifest.get("analysis"))
        elif schema_version != "ade.trial_artifacts.v1":
            raise ValueError(
                "Experiment Package schema_version must be 2 or "
                "ade.trial_artifacts.v1"
            )
        _scope_identity(manifest)
        return result


def _scope_identity(manifest: dict[str, Any]) -> dict[str, str]:
    identity: dict[str, str] = {}
    for field in _SCOPE_FIELDS:
        value = manifest.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Experiment Package {field} is required")
        identity[field] = value
    return identity
