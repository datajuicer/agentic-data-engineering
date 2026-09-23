"""Immutable trial artifact folders shared by SFT and RFT."""

from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any, Iterable, Mapping

from ade.engine.storage.atomic import write_json_atomic, write_text_atomic

_CATEGORIES = frozenset(
    {"checkpoint", "model_behavior", "eval_result", "audit", "package_metadata"}
)
_STATUSES = frozenset(
    {"complete", "partial", "failed", "unavailable", "skipped", "pruned"}
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PAYLOAD_DIRECTORIES = (
    ("checkpoints", "checkpoint"),
    ("model_behavior", "model_behavior"),
    ("eval_results", "eval_result"),
    ("audit", "audit"),
)


def artifact_positions(*, total: int, interval: int) -> tuple[int, ...]:
    if total < 1 or interval < 1:
        raise ValueError("artifact total and interval must be positive")
    positions = list(range(interval, total + 1, interval))
    if not positions or positions[-1] != total:
        positions.append(total)
    return tuple(positions)


def remove_generated_wandb_symlinks(audit_root: str | Path) -> None:
    wandb_root = Path(audit_root) / "wandb"
    if not wandb_root.is_dir():
        return
    for current, directories, files in os.walk(wandb_root, followlinks=False):
        for name in (*directories, *files):
            path = Path(current) / name
            if path.is_symlink():
                path.unlink()


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    category: str
    kind: str
    status: str = "complete"
    path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BehaviorWriteResult:
    sha256: str
    size_bytes: int
    record_count: int
    index_path: Path
    index_sha256: str
    index_size_bytes: int


@dataclass(frozen=True)
class TrialArtifactWorkspace:
    root: Path
    run_id: str
    coordinator_id: str
    plan_id: str
    trial_id: str
    attempt_id: str

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def model_behavior(self) -> Path:
        return self.root / "model_behavior"

    @property
    def eval_results(self) -> Path:
        return self.root / "eval_results"

    @property
    def audit(self) -> Path:
        return self.root / "audit"


class TrialArtifactPublisher:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def workspace(
        self,
        run_id: str,
        coordinator_id: str,
        plan_id: str,
        trial_id: str,
        *,
        attempt_id: str | None = None,
    ) -> TrialArtifactWorkspace:
        _validate_id(run_id, "run_id")
        _validate_id(coordinator_id, "coordinator_id")
        _validate_id(plan_id, "plan_id")
        _validate_id(trial_id, "trial_id")
        trial_root = self.root / run_id / coordinator_id / plan_id / trial_id
        trial_root.mkdir(parents=True, exist_ok=True)
        if attempt_id is not None:
            if re.fullmatch(r"attempt-\d{3}", attempt_id) is None:
                raise ValueError(f"invalid attempt_id: {attempt_id!r}")
            attempt_root = trial_root / attempt_id
            if attempt_root.exists():
                raise FileExistsError(
                    f"Engine Attempt workspace already exists: {attempt_root}"
                )
            attempt_root.mkdir()
            for name, _ in _PAYLOAD_DIRECTORIES:
                (attempt_root / name).mkdir()
            return TrialArtifactWorkspace(
                root=attempt_root.resolve(),
                run_id=run_id,
                coordinator_id=coordinator_id,
                plan_id=plan_id,
                trial_id=trial_id,
                attempt_id=attempt_id,
            )
        attempts = sorted(
            (
                path
                for path in trial_root.glob("attempt-*")
                if path.is_dir() and re.fullmatch(r"attempt-\d{3}", path.name)
            ),
            key=lambda path: path.name,
        )
        if attempts and not (attempts[-1] / "manifest.json").exists():
            attempt_root = attempts[-1]
        else:
            attempt_root = trial_root / f"attempt-{len(attempts) + 1:03d}"
            attempt_root.mkdir()
        for name, _ in _PAYLOAD_DIRECTORIES:
            (attempt_root / name).mkdir(exist_ok=True)
        return TrialArtifactWorkspace(
            root=attempt_root.resolve(),
            run_id=run_id,
            coordinator_id=coordinator_id,
            plan_id=plan_id,
            trial_id=trial_id,
            attempt_id=attempt_root.name,
        )

    def publish(
        self,
        workspace: TrialArtifactWorkspace,
        *,
        trial_status: str,
        selected_checkpoint_id: str | None = None,
        records: tuple[ArtifactRecord, ...] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        root = workspace.root.resolve()
        if not root.is_relative_to(self.root):
            raise ValueError("trial artifact workspace escapes publisher root")
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            raise ValueError("trial artifact attempt is already published")
        self._validate_tree(root)
        artifacts = self._inventory(root, records)
        assets_path = root / "ASSETS.md"
        write_text_atomic(
            assets_path,
            _assets_markdown(
                workspace,
                trial_status=trial_status,
                selected_checkpoint_id=selected_checkpoint_id,
                artifacts=artifacts,
            ),
        )
        artifacts.append(_file_artifact(root, assets_path, "package_metadata"))
        artifacts.sort(key=_artifact_sort_key)
        manifest = {
            "schema_version": "ade.trial_artifacts.v1",
            "run_id": workspace.run_id,
            "coordinator_id": workspace.coordinator_id,
            "plan_id": workspace.plan_id,
            "trial_id": workspace.trial_id,
            "attempt_id": workspace.attempt_id,
            "artifact_root": str(root),
            "trial_status": trial_status,
            "selected_checkpoint_id": selected_checkpoint_id,
            "metadata": dict(metadata or {}),
            "artifacts": artifacts,
        }
        write_json_atomic(manifest_path, manifest)
        self._make_read_only(root)
        return manifest_path

    @staticmethod
    def _validate_tree(root: Path) -> None:
        allowed = {name for name, _ in _PAYLOAD_DIRECTORIES}
        for child in root.iterdir():
            if child.name not in allowed:
                raise ValueError(f"unexpected trial artifact root entry: {child.name}")
        for current, directories, files in os.walk(root, followlinks=False):
            current_path = Path(current)
            for name in (*directories, *files):
                path = current_path / name
                if path.is_symlink():
                    raise ValueError(f"trial artifact cannot contain symlink: {path}")
                if name.startswith(".") and name.endswith(".tmp"):
                    raise ValueError(f"trial artifact contains temporary file: {path}")
                mode = path.lstat().st_mode
                if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                    raise ValueError(f"trial artifact contains unsupported file type: {path}")

    @staticmethod
    def _inventory(
        root: Path,
        declared: tuple[ArtifactRecord, ...],
    ) -> list[dict[str, Any]]:
        by_path: dict[str, ArtifactRecord] = {}
        pathless: list[ArtifactRecord] = []
        seen_ids: set[str] = set()
        for record in declared:
            _validate_record(record)
            if record.artifact_id in seen_ids:
                raise ValueError(f"duplicate trial artifact id: {record.artifact_id}")
            seen_ids.add(record.artifact_id)
            if record.path is None:
                pathless.append(record)
                continue
            relative = _safe_relative(record.path)
            if relative in by_path:
                raise ValueError(f"duplicate trial artifact path: {relative}")
            by_path[relative] = record

        artifacts: list[dict[str, Any]] = []
        inventoried_paths: set[str] = set()
        for directory_name, category in _PAYLOAD_DIRECTORIES:
            directory = root / directory_name
            if category == "checkpoint":
                for checkpoint in sorted(directory.iterdir(), key=lambda path: path.name):
                    relative = checkpoint.relative_to(root).as_posix()
                    if not checkpoint.is_dir():
                        artifacts.append(
                            _merge_record(
                                _lightweight_file_artifact(root, checkpoint, category),
                                by_path.get(relative),
                            )
                        )
                        inventoried_paths.add(relative)
                        continue
                    artifacts.append(
                        _merge_record(
                            _lightweight_directory_artifact(root, checkpoint, category),
                            by_path.get(relative),
                        )
                    )
                    inventoried_paths.add(relative)
                continue
            for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
                if not path.is_file():
                    continue
                relative = path.relative_to(root).as_posix()
                record = by_path.get(relative)
                if (
                    record is not None
                    and record.category == "model_behavior"
                    and record.kind == "online_validation"
                ):
                    discovered = _lightweight_file_artifact(root, path, category)
                elif (
                    record is not None
                    and isinstance(record.metadata.get("sha256"), str)
                    and isinstance(record.metadata.get("size_bytes"), int)
                ):
                    discovered = _bound_file_artifact(root, path, category, record)
                else:
                    discovered = _file_artifact(root, path, category)
                artifacts.append(
                    _merge_record(
                        discovered,
                        record,
                    )
                )
                inventoried_paths.add(relative)
        undeclared_paths = sorted(set(by_path) - inventoried_paths)
        if undeclared_paths:
            raise ValueError(
                f"declared trial artifact paths do not exist: {undeclared_paths}"
            )
        artifacts.extend(_record_payload(record) for record in pathless)
        artifacts.sort(key=_artifact_sort_key)
        return artifacts

    @staticmethod
    def _make_read_only(root: Path) -> None:
        paths = sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True)
        for path in paths:
            path.chmod(0o555 if path.is_dir() else 0o444)
        root.chmod(0o555)


def write_evaluation_artifacts(
    workspace: TrialArtifactWorkspace,
    *,
    purpose: str,
    position_unit: str,
    position_value: int,
    checkpoint_artifact_id: str,
    ranking_score: float | None,
    payload: Mapping[str, Any],
    status: str = "complete",
) -> tuple[ArtifactRecord, ...]:
    if purpose not in {"online_validation", "offline_validation"}:
        raise ValueError("trial evaluation artifact purpose is unsupported")
    if position_unit not in {"epoch", "rl_step"}:
        raise ValueError("trial evaluation position unit is unsupported")
    if status not in {"complete", "partial"}:
        raise ValueError("trial evaluation artifact status is unsupported")
    if status == "complete" and ranking_score is None:
        raise ValueError("complete trial evaluation requires ranking_score")
    label = f"{position_unit.replace('_', '-')}-{int(position_value):03d}"
    behavior_id = f"{purpose.replace('_', '-')}-behavior-{label}"
    details = _evaluation_details(payload)
    secondary_score = payload.get("secondary_score")
    if secondary_score is not None:
        secondary_score = float(secondary_score)
    records: list[ArtifactRecord] = []
    behavior_ids: list[str] = []
    if details:
        behavior_path = workspace.model_behavior / purpose / f"{label}.jsonl.gz"
        behavior_path.parent.mkdir(parents=True, exist_ok=True)
        normalized = (
            _behavior_record(
                workspace,
                purpose=purpose,
                position_unit=position_unit,
                position_value=position_value,
                checkpoint_artifact_id=checkpoint_artifact_id,
                detail=detail,
                index=index,
            )
            for index, detail in enumerate(details)
        )
        index_path = (
            workspace.audit
            / "analysis_sources"
            / "behavior_indexes"
            / purpose
            / f"{label}.json"
        )
        written = _write_behavior_gzip_jsonl_atomic(
            behavior_path,
            normalized,
            index_path=index_path,
        )
        relative = behavior_path.relative_to(workspace.root).as_posix()
        metadata: dict[str, Any] = {"record_count": written.record_count}
        if purpose != "online_validation":
            metadata.update(
                {
                    "sha256": written.sha256,
                    "size_bytes": written.size_bytes,
                    "index": {
                        "path": written.index_path.relative_to(workspace.root).as_posix(),
                        "sha256": written.index_sha256,
                        "size_bytes": written.index_size_bytes,
                        "record_count": written.record_count,
                    },
                }
            )
        records.append(
            ArtifactRecord(
                artifact_id=behavior_id,
                category="model_behavior",
                kind=purpose,
                status=status,
                path=relative,
                metadata=metadata,
            )
        )
        behavior_ids.append(behavior_id)

    eval_id = f"{purpose.replace('_', '-')}-{label}"
    result_path = (
        workspace.eval_results
        / f"{purpose.replace('_', '-')}-{label}.json"
    )
    metrics = payload.get("aggregate_metrics")
    if not isinstance(metrics, Mapping):
        metrics = {
            str(item.get("dataset_name") or f"dataset-{index}"): dict(
                item.get("metrics") or {}
            )
            for index, item in enumerate(payload.get("results") or ())
            if isinstance(item, Mapping)
        }
    write_json_atomic(
        result_path,
        {
            "schema_version": "ade.eval_result.v1",
            "evaluation_id": eval_id,
            "purpose": purpose,
            "status": status,
            "artifact_position": {
                "unit": position_unit,
                "value": int(position_value),
            },
            "checkpoint_artifact_id": checkpoint_artifact_id,
            "metrics": dict(metrics),
            "ranking_score": (
                float(ranking_score)
                if status == "complete" and ranking_score is not None
                else None
            ),
            "secondary_score": (
                secondary_score if status == "complete" else None
            ),
            "behavior_artifact_ids": behavior_ids,
            "completeness": {
                "observed_records": len(details),
            },
        },
    )
    records.append(
        ArtifactRecord(
            artifact_id=eval_id,
            category="eval_result",
            kind=purpose,
            status=status,
            path=result_path.relative_to(workspace.root).as_posix(),
            metadata={
                "ranking_score": (
                    float(ranking_score)
                    if status == "complete" and ranking_score is not None
                    else None
                ),
                "secondary_score": (
                    secondary_score if status == "complete" else None
                ),
                "behavior_artifact_ids": behavior_ids,
            },
        )
    )
    return tuple(records)


def write_training_rollout_artifact(
    workspace: TrialArtifactWorkspace,
    *,
    source_path: str | Path,
    step: int,
    prompt_groups: int,
    rollout_n: int,
) -> ArtifactRecord:
    source = Path(source_path)
    if not source.is_file():
        raise ValueError(f"RFT training rollout source does not exist: {source}")
    if step < 1 or prompt_groups < 1 or rollout_n < 1:
        raise ValueError("RFT rollout artifact dimensions must be positive")
    target = (
        workspace.model_behavior
        / "training_rollouts"
        / f"rl-step-{step:03d}.jsonl.gz"
    )
    record_count = 0
    group_counts: dict[str, int] = {}

    def records() -> Iterable[dict[str, Any]]:
        nonlocal record_count
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                detail = json.loads(line)
                if not isinstance(detail, Mapping):
                    raise ValueError("RFT rollout source must contain JSON objects")
                required_evidence = {
                    "group_credit_enabled",
                    "group_uid",
                    "group_type",
                    "outcome_score",
                    "process_evidence",
                    "artifact_projection",
                    "rule_evidence",
                    "pre_group_reward",
                    "final_training_reward",
                    "realized_grpo_advantage",
                    "counterfactual_identity_grpo_advantage",
                }
                if detail.get("group_credit_enabled") is True:
                    required_evidence.update(
                        {
                            "assigned_training_reward",
                            "group_credit_mode",
                            "group_credit_source",
                            "group_credit_reason",
                            "group_credit_evidence_sources",
                            "group_credit_process_dimensions",
                        }
                    )
                missing_evidence = sorted(required_evidence.difference(detail))
                if missing_evidence:
                    raise ValueError(
                        "RFT rollout source lacks authoritative reward siblings: "
                        f"{missing_evidence}"
                    )
                group_id = detail.get("prompt_group_id")
                if not isinstance(group_id, str) or not group_id:
                    raise ValueError("RFT rollout source requires prompt_group_id")
                group_counts[group_id] = group_counts.get(group_id, 0) + 1
                record_count += 1
                yield _behavior_record(
                    workspace,
                    purpose="training_rollout",
                    position_unit="rl_step",
                    position_value=step,
                    checkpoint_artifact_id=f"checkpoint-rl-step-{step:03d}",
                    detail=detail,
                    index=index,
                )

    index_path = (
        workspace.audit
        / "analysis_sources"
        / "behavior_indexes"
        / "training_rollouts"
        / f"rl-step-{step:03d}.json"
    )
    written = _write_behavior_gzip_jsonl_atomic(
        target,
        records(),
        index_path=index_path,
    )
    expected = prompt_groups * rollout_n
    complete = (
        record_count == expected
        and len(group_counts) == prompt_groups
        and all(count == rollout_n for count in group_counts.values())
    )
    return ArtifactRecord(
        artifact_id=f"training-rollouts-rl-step-{step:03d}",
        category="model_behavior",
        kind="training_rollout",
        status="complete" if complete else "partial",
        path=target.relative_to(workspace.root).as_posix(),
        metadata={
            "artifact_position": {"unit": "rl_step", "value": step},
            "prompt_group_count": prompt_groups,
            "rollout_n": rollout_n,
            "expected_record_count": expected,
            "record_count": record_count,
            "sha256": written.sha256,
            "size_bytes": written.size_bytes,
            "index": {
                "path": written.index_path.relative_to(workspace.root).as_posix(),
                "sha256": written.index_sha256,
                "size_bytes": written.index_size_bytes,
                "record_count": written.record_count,
            },
        },
    )


def stage_training_telemetry_artifacts(
    workspace: TrialArtifactWorkspace,
    manifest_path: str | Path,
    *,
    position_unit: str,
    target_namespace: str,
    move_sources: bool = False,
) -> tuple[ArtifactRecord, ...]:
    """Stage normalized backend telemetry through the shared trial boundary."""
    if position_unit not in {"epoch", "rl_step"}:
        raise ValueError("training telemetry position unit is unsupported")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        manifest_file = Path(manifest_path)
        sources = [
            {
                "source_id": "training-telemetry",
                "kind": "training_telemetry",
                "path": manifest.get("telemetry_path"),
            },
            {
                "source_id": "training-telemetry-summary",
                "kind": "training_telemetry_summary",
                "path": manifest.get("telemetry_summary_path"),
            },
            {
                "source_id": "training-telemetry-manifest",
                "kind": "training_telemetry_manifest",
                "path": str(manifest_file),
            },
        ]
    telemetry_kinds = {
        "training_telemetry",
        "training_telemetry_manifest",
        "training_telemetry_summary",
    }
    records: list[ArtifactRecord] = []
    for source in sources:
        if not isinstance(source, dict) or source.get("kind") not in telemetry_kinds:
            continue
        source_id = str(source.get("source_id") or "")
        source_path = Path(str(source.get("path") or ""))
        if not source_id or not source_path.is_file():
            continue
        suffix = "".join(source_path.suffixes)
        target = workspace.audit / "metrics" / target_namespace / f"{source_id}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"training telemetry artifact already exists: {target}")
        if move_sources:
            source_path.replace(target)
        else:
            shutil.copy2(source_path, target)
        metadata = {}
        if source.get("step") is not None:
            metadata["artifact_position"] = {
                "unit": position_unit,
                "value": int(source["step"]),
            }
        records.append(
            ArtifactRecord(
                artifact_id=f"audit-{source_id}",
                category="audit",
                kind=str(source["kind"]),
                path=target.relative_to(workspace.root).as_posix(),
                metadata=metadata,
            )
        )
    return tuple(records)


def write_sft_selection_audit(
    workspace: TrialArtifactWorkspace,
    *,
    selection_path: str | Path,
    selected_data_path: str | Path,
    candidate_pool_path: str | Path | None = None,
) -> tuple[ArtifactRecord, ...]:
    """Persist the exact SFT selection inputs in Analyzer-readable form."""
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    selected = json.loads(Path(selected_data_path).read_text(encoding="utf-8"))
    if not isinstance(selection, Mapping) or not isinstance(selected, list):
        raise ValueError("SFT selection audit inputs are invalid")
    selected_ids = selection.get("selected_ids")
    if not isinstance(selected_ids, list) or len(selected_ids) != len(selected):
        raise ValueError("SFT selected IDs do not match selected dataset rows")
    if any(not isinstance(item, Mapping) for item in selected):
        raise ValueError("SFT selected dataset rows must be objects")
    root = workspace.audit / "data_selection"
    config_path = root / "selection_config.json"
    summary_path = root / "selection_summary.json"
    examples_path = root / "selected_examples.jsonl.gz"
    write_json_atomic(config_path, selection)
    write_json_atomic(
        summary_path,
        {
            "schema_version": "ade.selection_summary.v1",
            "selection_size": selection.get("selection_size", len(selected_ids)),
            "total_tokens": selection.get("total_tokens"),
            "statistics": selection.get("statistics", {}),
            "validation": selection.get("validation", {}),
        },
    )
    normalized = []
    trajectory_to_candidate = {
        str(trajectory_id): str(group["candidate_id"])
        for group in selection.get("selected_groups") or ()
        if isinstance(group, Mapping) and group.get("candidate_id") is not None
        for trajectory_id in group.get("trajectory_ids") or ()
    }
    for record_id, row in zip(selected_ids, selected):
        question, response = _selected_example_question_response(row)
        normalized.append(
            {
                "schema_version": "ade.selected_example.v2",
                "record_id": str(record_id),
                "candidate_record_id": str(record_id),
                "candidate_id": trajectory_to_candidate.get(str(record_id)),
                "question": question,
                "response": response,
            }
        )
    _write_gzip_jsonl_atomic(examples_path, normalized)
    records = [
        ArtifactRecord(
            "sft-selection-config",
            "audit",
            "selection_config",
            path=config_path.relative_to(workspace.root).as_posix(),
        ),
        ArtifactRecord(
            "sft-selection-summary",
            "audit",
            "selection_summary",
            path=summary_path.relative_to(workspace.root).as_posix(),
        ),
        ArtifactRecord(
            "sft-selected-examples",
            "audit",
            "selected_examples",
            path=examples_path.relative_to(workspace.root).as_posix(),
            metadata={"record_count": len(normalized)},
        ),
    ]
    if candidate_pool_path is not None:
        pool_path = Path(candidate_pool_path)
        if not pool_path.is_file():
            raise FileNotFoundError(f"SFT candidate pool does not exist: {pool_path}")
        pool_bytes = pool_path.read_bytes()
        pool_rows = [
            json.loads(line)
            for line in pool_bytes.decode("utf-8").split("\n")
            if line.strip()
        ]
        if not pool_rows or not all(isinstance(row, Mapping) for row in pool_rows):
            raise ValueError("SFT candidate pool must be non-empty JSONL objects")
        normalized_pool = []
        seen_candidate_records: set[str] = set()
        for row in pool_rows:
            record_id = str(row.get("trajectory_id") or row.get("record_id") or "")
            if not record_id or record_id in seen_candidate_records:
                raise ValueError("SFT candidate pool requires unique trajectory identities")
            question, response = _selected_example_question_response(row)
            seen_candidate_records.add(record_id)
            normalized_pool.append(
                {
                    **dict(row),
                    "schema_version": "ade.candidate_example.v1",
                    "record_id": record_id,
                    "candidate_id": row.get("candidate_id"),
                    "question": question,
                    "response": response,
                }
            )
        missing_candidates = sorted(set(map(str, selected_ids)) - seen_candidate_records)
        if missing_candidates:
            raise ValueError(
                f"SFT selected examples lack candidate records: {missing_candidates}"
            )
        candidate_path = root / "candidate_pool.jsonl"
        write_text_atomic(
            candidate_path,
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in normalized_pool),
        )
        records.append(
            ArtifactRecord(
                "sft-candidate-pool",
                "audit",
                "candidate_pool",
                path=candidate_path.relative_to(workspace.root).as_posix(),
                metadata={"record_count": len(normalized_pool)},
            )
        )
    return tuple(records)


def _selected_example_question_response(
    row: Mapping[str, Any],
) -> tuple[str, str]:
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        questions = [
            str(message.get("value") or "").strip()
            for message in conversations
            if isinstance(message, Mapping)
            and message.get("from") in {"user", "human"}
            and str(message.get("value") or "").strip()
        ]
        responses = [
            str(message.get("value") or "").strip()
            for message in conversations
            if isinstance(message, Mapping)
            and message.get("from") in {"assistant", "gpt"}
            and str(message.get("value") or "").strip()
        ]
        if questions and len(responses) == 1:
            return "\n\n".join(questions), responses[0]
    instruction = str(row.get("instruction") or "").strip()
    input_text = str(row.get("input") or "").strip()
    response = str(row.get("output") or "").strip()
    question = (
        f"{instruction}\n\nInput:\n{input_text}"
        if instruction and input_text
        else instruction
    )
    if not question or not response:
        raise ValueError("SFT selected example requires question and response text")
    return question, response


def write_rft_reward_audit(
    workspace: TrialArtifactWorkspace,
    *,
    reward_path: str | Path,
    runtime_config: Mapping[str, Any],
) -> tuple[ArtifactRecord, ...]:
    """Persist the accepted reward implementation and effective VERL config."""
    root = workspace.audit / "reward"
    function_path = root / "reward.py"
    config_path = root / "runtime_config.json"
    write_text_atomic(function_path, Path(reward_path).read_text(encoding="utf-8"))
    write_json_atomic(config_path, runtime_config)
    return (
        ArtifactRecord(
            "rft-reward-function",
            "audit",
            "reward_function",
            path=function_path.relative_to(workspace.root).as_posix(),
        ),
        ArtifactRecord(
            "rft-reward-runtime-config",
            "audit",
            "reward_runtime_config",
            path=config_path.relative_to(workspace.root).as_posix(),
        ),
    )


def _file_artifact(root: Path, path: Path, category: str) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    return {
        "id": _default_artifact_id(category, relative),
        "category": category,
        "kind": _default_kind(category),
        "status": "complete",
        "path": relative,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _lightweight_file_artifact(
    root: Path,
    path: Path,
    category: str,
) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    return {
        "id": _default_artifact_id(category, relative),
        "category": category,
        "kind": _default_kind(category),
        "status": "complete",
        "path": relative,
        "size_bytes": path.stat().st_size,
    }


def _lightweight_directory_artifact(
    root: Path,
    path: Path,
    category: str,
) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    return {
        "id": _default_artifact_id(category, relative),
        "category": category,
        "kind": _default_kind(category),
        "status": "complete",
        "path": relative,
    }


def _bound_file_artifact(
    root: Path,
    path: Path,
    category: str,
    record: ArtifactRecord,
) -> dict[str, Any]:
    expected_size = int(record.metadata["size_bytes"])
    if path.stat().st_size != expected_size:
        raise ValueError(f"declared artifact size mismatch: {record.artifact_id}")
    relative = path.relative_to(root).as_posix()
    return {
        "id": _default_artifact_id(category, relative),
        "category": category,
        "kind": _default_kind(category),
        "status": "complete",
        "path": relative,
        "sha256": str(record.metadata["sha256"]),
        "size_bytes": expected_size,
    }


def _merge_record(
    discovered: dict[str, Any],
    declared: ArtifactRecord | None,
) -> dict[str, Any]:
    if declared is None:
        return discovered
    result = dict(discovered)
    result.update(
        {
            "id": declared.artifact_id,
            "category": declared.category,
            "kind": declared.kind,
            "status": declared.status,
        }
    )
    result.update(dict(declared.metadata))
    return result


def _record_payload(record: ArtifactRecord) -> dict[str, Any]:
    payload = {
        "id": record.artifact_id,
        "category": record.category,
        "kind": record.kind,
        "status": record.status,
    }
    payload.update(dict(record.metadata))
    return payload


def _assets_markdown(
    workspace: TrialArtifactWorkspace,
    *,
    trial_status: str,
    selected_checkpoint_id: str | None,
    artifacts: list[dict[str, Any]],
) -> str:
    lines = [
        "# Trial Assets",
        "",
        f"- Run: `{workspace.run_id}`",
        f"- Coordinator: `{workspace.coordinator_id}`",
        f"- Plan: `{workspace.plan_id}`",
        f"- Trial: `{workspace.trial_id}`",
        f"- Attempt: `{workspace.attempt_id}`",
        f"- Status: `{trial_status}`",
        f"- Selected checkpoint: `{selected_checkpoint_id}`",
        "",
        "| Asset | Category | Kind | Status | Path | Bytes |",
        "|---|---|---|---|---|---:|",
    ]
    for item in artifacts:
        lines.append(
            f"| `{item['id']}` | `{item['category']}` | `{item['kind']}` | "
            f"`{item['status']}` | `{item.get('path', '')}` | "
            f"{item.get('size_bytes', '')} |"
        )
    return "\n".join(lines) + "\n"


def _validate_record(record: ArtifactRecord) -> None:
    _validate_id(record.artifact_id, "artifact_id")
    if record.category not in _CATEGORIES:
        raise ValueError(f"unsupported trial artifact category: {record.category}")
    if not record.kind:
        raise ValueError("trial artifact kind is required")
    if record.status not in _STATUSES:
        raise ValueError(f"unsupported trial artifact status: {record.status}")
    if record.path is None and record.status == "complete":
        raise ValueError("complete trial artifact requires a path")


def _safe_relative(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe trial artifact path: {value}")
    return path.as_posix()


def _validate_id(value: str, field: str) -> None:
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{field} contains unsafe characters")


def _default_artifact_id(category: str, relative: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", relative).strip("-")
    return f"{category}-{slug}"


def _default_kind(category: str) -> str:
    return {
        "checkpoint": "hf_checkpoint",
        "model_behavior": "model_behavior",
        "eval_result": "evaluation_result",
        "audit": "training_audit",
        "package_metadata": "asset_description",
    }[category]


def _artifact_sort_key(item: Mapping[str, Any]) -> tuple[str, str]:
    return (str(item.get("path") or "~"), str(item["id"]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_details(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    direct = payload.get("records")
    if isinstance(direct, list):
        return [dict(item) for item in direct if isinstance(item, Mapping)]
    details: list[dict[str, Any]] = []
    for result in payload.get("results") or ():
        if not isinstance(result, Mapping):
            continue
        result_path = Path(str(result.get("result_path") or ""))
        if not result_path.is_file():
            continue
        loaded = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            continue
        rows = loaded.get("result_details")
        if not isinstance(rows, list):
            rows = _logged_evaluation_details(loaded.get("case_details"))
        dataset_name = str(result.get("dataset_name") or "")
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            detail = dict(row)
            detail.setdefault("dataset_name", dataset_name)
            details.append(detail)
    return details


def _logged_evaluation_details(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    details: list[dict[str, Any]] = []
    for run in value:
        if not isinstance(run, Mapping):
            continue
        sample_index = int(run.get("index") or 0)
        for case in run.get("cases") or ():
            if not isinstance(case, Mapping):
                continue
            prompt = case.get("input")
            gold = case.get("gold")
            prediction = case.get("prediction")
            diagnostics = case.get("diagnostics")
            prompt = prompt if isinstance(prompt, Mapping) else {}
            gold = gold if isinstance(gold, Mapping) else {}
            prediction = prediction if isinstance(prediction, Mapping) else {}
            diagnostics = (
                diagnostics if isinstance(diagnostics, Mapping) else {}
            )
            passed = diagnostics.get("passed")
            details.append(
                {
                    "case_id": case.get("case_id") or case.get("example_id"),
                    "prompt": prompt.get("question") or prompt.get("prompt"),
                    "messages": prompt.get("messages"),
                    "response_content": prediction.get("answer_content"),
                    "extracted_answer": prediction.get("predicted"),
                    "ground_truth": gold.get("expected") or gold.get("reference_full"),
                    "score": float(bool(passed)) if passed is not None else None,
                    "run_index": sample_index,
                    "seed": run.get("seed"),
                    "finish_reason": diagnostics.get("finish_reason"),
                    "response_length_tokens": diagnostics.get("num_output_tokens"),
                    "evaluation_case": dict(case),
                }
            )
    return details


def _behavior_record(
    workspace: TrialArtifactWorkspace,
    *,
    purpose: str,
    position_unit: str,
    position_value: int,
    checkpoint_artifact_id: str,
    detail: Mapping[str, Any],
    index: int,
) -> dict[str, Any]:
    case_id = str(
        detail.get("case_id")
        or detail.get("prompt_group_id")
        or detail.get("prompt_id")
        or detail.get("id")
        or detail.get("index")
        or index
    )
    sample_index = int(
        detail.get("sample_index")
        or detail.get("response_index")
        or detail.get("run_index")
        or detail.get("sample")
        or 0
    )
    identity = hashlib.sha256(
        (
            f"{workspace.run_id}/{workspace.coordinator_id}/{workspace.plan_id}/"
            f"{workspace.trial_id}/{workspace.attempt_id}/{purpose}/"
            f"{position_unit}/{position_value}/{case_id}/{sample_index}"
        ).encode()
    ).hexdigest()[:24]
    score = detail.get("score")
    if score is None:
        score = detail.get("reward")
    if score is None:
        score = detail.get("effective_training_reward")
    if score is None:
        score = detail.get("custom_reward_score")
    if score is None and detail.get("passed") is not None:
        score = float(bool(detail["passed"]))
    training_context = None
    if purpose == "training_rollout":
        training_context = {
            "sample_uid": detail.get("sample_uid"),
            "policy_step_at_generation": detail.get("policy_step_at_generation"),
            "prompt_group_id": detail.get("prompt_group_id") or case_id,
            "response_index": sample_index,
            "effective_training_reward": score,
            "group_credit_enabled": detail.get("group_credit_enabled"),
            "group_uid": detail.get("group_uid"),
            "group_type": detail.get("group_type"),
            "outcome_score": detail.get("outcome_score"),
            "process_evidence": detail.get("process_evidence"),
            "artifact_projection": detail.get("artifact_projection"),
            "rule_evidence": detail.get("rule_evidence"),
            "pre_group_reward": detail.get("pre_group_reward"),
            "final_training_reward": detail.get("final_training_reward"),
            "realized_grpo_advantage": detail.get("realized_grpo_advantage"),
            "counterfactual_identity_grpo_advantage": detail.get(
                "counterfactual_identity_grpo_advantage"
            ),
            "correctness": detail.get("correctness"),
        }
        if detail.get("group_credit_enabled") is True:
            training_context.update(
                {
                    "assigned_training_reward": detail.get(
                        "assigned_training_reward"
                    ),
                    "group_credit_mode": detail.get("group_credit_mode"),
                    "group_credit_source": detail.get("group_credit_source"),
                    "group_credit_reason": detail.get("group_credit_reason"),
                    "group_credit_evidence_sources": detail.get(
                        "group_credit_evidence_sources"
                    ),
                    "group_credit_process_dimensions": detail.get(
                        "group_credit_process_dimensions"
                    ),
                }
            )
    return {
        "schema_version": "ade.model_behavior.v1",
        "record_id": identity,
        "run_id": workspace.run_id,
        "coordinator_id": workspace.coordinator_id,
        "plan_id": workspace.plan_id,
        "trial_id": workspace.trial_id,
        "attempt_id": workspace.attempt_id,
        "source": purpose,
        "artifact_position": {
            "unit": position_unit,
            "value": int(position_value),
        },
        "prompt": {
            "id": case_id,
            "dataset": detail.get("dataset_name"),
            "messages": detail.get("messages"),
            "text": detail.get("prompt") or detail.get("question_prompt"),
        },
        "generation": {
            "sample_index": sample_index,
            "raw_response": (
                detail.get("response_content")
                or detail.get("response")
                or detail.get("prediction")
            ),
            "extracted_answer": detail.get("extracted_answer"),
            "finish_reason": detail.get("finish_reason"),
            "response_tokens": (
                detail.get("response_length_tokens")
                or detail.get("response_tokens")
            ),
        },
        "assessment": {
            "ground_truth": detail.get("ground_truth"),
            "score": score,
            "outcome_score": detail.get("outcome_score"),
            "process_evidence": detail.get("process_evidence"),
            "artifact_projection": detail.get("artifact_projection"),
            "rule_evidence": detail.get("rule_evidence"),
        },
        "provenance": {
            "checkpoint_artifact_id": checkpoint_artifact_id,
            "seed": detail.get("seed"),
        },
        "training_context": training_context,
        "details": dict(detail),
    }


def _write_behavior_gzip_jsonl_atomic(
    path: Path,
    records: Iterable[dict[str, Any]],
    *,
    index_path: Path,
) -> BehaviorWriteResult:
    path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    digest = hashlib.sha256()
    size_bytes = 0
    index_records: list[dict[str, Any]] = []

    class DigestingWriter:
        def __init__(self, raw: Any) -> None:
            self.raw = raw

        def write(self, content: bytes) -> int:
            nonlocal size_bytes
            written = self.raw.write(content)
            digest.update(content[:written])
            size_bytes += written
            return written

        def flush(self) -> None:
            self.raw.flush()

        def __getattr__(self, name: str) -> Any:
            return getattr(self.raw, name)

    try:
        with os.fdopen(descriptor, "wb") as raw:
            output = DigestingWriter(raw)
            with gzip.GzipFile(
                fileobj=output,
                mode="wb",
                compresslevel=1,
                mtime=0,
            ) as compressed:
                for line_index, record in enumerate(records):
                    index_records.append(_behavior_index_entry(record, line_index))
                    compressed.write(
                        (
                            json.dumps(
                                record,
                                sort_keys=True,
                                separators=(",", ":"),
                                ensure_ascii=False,
                            )
                            + "\n"
                        ).encode()
                    )
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
        index_payload = {
            "schema_version": "ade.behavior_index.v1",
            "behavior": {
                "sha256": digest.hexdigest(),
                "size_bytes": size_bytes,
                "record_count": len(index_records),
            },
            "records": index_records,
        }
        write_json_atomic(index_path, index_payload)
        return BehaviorWriteResult(
            sha256=digest.hexdigest(),
            size_bytes=size_bytes,
            record_count=len(index_records),
            index_path=index_path,
            index_sha256=_sha256(index_path),
            index_size_bytes=index_path.stat().st_size,
        )
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _behavior_index_entry(record: Mapping[str, Any], line_index: int) -> dict[str, Any]:
    prompt = record.get("prompt") if isinstance(record.get("prompt"), Mapping) else {}
    generation = (
        record.get("generation")
        if isinstance(record.get("generation"), Mapping)
        else {}
    )
    assessment = (
        record.get("assessment")
        if isinstance(record.get("assessment"), Mapping)
        else {}
    )
    training_context = (
        record.get("training_context")
        if isinstance(record.get("training_context"), Mapping)
        else None
    )
    return {
        "record_id": str(record.get("record_id") or ""),
        "line_index": line_index,
        "review_fields_present": {
            "prompt.text": isinstance(prompt.get("text"), str)
            and bool(str(prompt.get("text")).strip()),
            "generation.raw_response": isinstance(
                generation.get("raw_response"), str
            )
            and bool(str(generation.get("raw_response")).strip()),
        },
        "projection": {
            "artifact_position": record.get("artifact_position"),
            "prompt": {"id": prompt.get("id"), "dataset": prompt.get("dataset")},
            "generation": {"sample_index": generation.get("sample_index")},
            "assessment": {
                "score": assessment.get("score"),
                "ground_truth": assessment.get("ground_truth"),
            },
            "training_context": training_context,
        },
    }


def _write_gzip_jsonl_atomic(
    path: Path,
    records: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(
                fileobj=raw,
                mode="wb",
                compresslevel=1,
                mtime=0,
            ) as compressed:
                for record in records:
                    compressed.write(
                        (
                            json.dumps(
                                record,
                                sort_keys=True,
                                separators=(",", ":"),
                                ensure_ascii=False,
                            )
                            + "\n"
                        ).encode()
                    )
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
