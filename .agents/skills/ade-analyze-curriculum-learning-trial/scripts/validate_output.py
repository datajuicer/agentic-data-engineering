#!/usr/bin/env python3
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "output")
files = {path.name for path in root.iterdir()} if root.is_dir() else set()
allowed = ({"review-plan.json"}, {"analysis.md", "findings.md"})
if files not in allowed or any(not path.read_text().strip() for path in root.iterdir()):
    raise SystemExit("expected review-plan.json or non-empty analysis.md/findings.md")
if files == {"review-plan.json"}:
    value = json.loads((root / "review-plan.json").read_text())
    if not isinstance(value, dict) or set(value) != {"schema_version", "hypotheses", "batches"}:
        raise SystemExit("review-plan.json has unsupported top-level fields")
    if value["schema_version"] != "ade.analysis_review_plan.v2":
        raise SystemExit("review-plan.json must use ade.analysis_review_plan.v2")
    if not isinstance(value["hypotheses"], list) or not value["hypotheses"] or any(
        not isinstance(item, str) or not item.strip() for item in value["hypotheses"]
    ):
        raise SystemExit("review-plan.json hypotheses must be non-empty strings")
    if not isinstance(value["batches"], list) or not value["batches"]:
        raise SystemExit("review-plan.json batches must be non-empty")
    batch_ids = set()
    for batch in value["batches"]:
        required = {"batch_id", "pool", "investigation_purpose", "selection", "rubrics"}
        if not isinstance(batch, dict) or not required.issubset(batch) or set(batch) - required - {"hypothesis_ref"}:
            raise SystemExit("review-plan.json batch shape is invalid")
        if not all(isinstance(batch[key], str) and batch[key].strip() for key in ("batch_id", "pool", "investigation_purpose")):
            raise SystemExit("review-plan.json batch identity is invalid")
        if batch["batch_id"] in batch_ids:
            raise SystemExit("review-plan.json batch IDs must be unique")
        batch_ids.add(batch["batch_id"])
        selection = batch["selection"]
        allowed = {"mode", "source_artifact_ids", "record_ids", "group_ids"}
        if not isinstance(selection, dict) or set(selection) - allowed:
            raise SystemExit("review-plan.json selection shape is invalid")
        mode = selection.get("mode")
        records = selection.get("record_ids", [])
        groups = selection.get("group_ids", [])
        if mode == "all" and (records or groups):
            raise SystemExit("all selection cannot name records or groups")
        if mode == "records" and (not records or groups):
            raise SystemExit("records selection requires only record_ids")
        if mode == "groups" and (not groups or records):
            raise SystemExit("groups selection requires only group_ids")
        if mode not in {"all", "records", "groups"}:
            raise SystemExit("review-plan.json selection mode is invalid")
        rubrics = batch["rubrics"]
        if not isinstance(rubrics, list) or not rubrics:
            raise SystemExit("review-plan.json rubrics must be non-empty")
        for rubric in rubrics:
            if not isinstance(rubric, dict) or set(rubric) != {"rubric_id", "instruction", "labels"}:
                raise SystemExit("review-plan.json rubric shape is invalid")
            labels = rubric["labels"]
            if (
                not isinstance(rubric["rubric_id"], str)
                or not rubric["rubric_id"].strip()
                or not isinstance(rubric["instruction"], str)
                or not rubric["instruction"].strip()
                or not isinstance(labels, list)
                or len(labels) < 2
                or any(not isinstance(label, str) or not label.strip() for label in labels)
                or len(labels) != len(set(labels))
            ):
                raise SystemExit("review-plan.json rubric values are invalid")
else:
    analysis = (root / "analysis.md").read_text()
    sections = (
        "## Analysis Scope and Evidence",
        "## Direct Inspection",
        "## Curriculum Schedule Diagnosis",
        "## Training Dynamics and Validation Response",
        "## Findings",
        "## Contradictions and Uncertainty",
        "## Recommendations",
    )
    if any(section not in analysis for section in sections):
        raise SystemExit("analysis.md is missing required sections")
