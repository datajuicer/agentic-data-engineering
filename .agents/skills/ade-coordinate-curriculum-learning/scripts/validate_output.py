#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "output")
files = {path.name for path in root.iterdir()} if root.is_dir() else set()
if files != {"plan.md", "decision.json"} or not (root / "plan.md").read_text().strip():
    raise SystemExit("expected only non-empty plan.md and decision.json")
value = json.loads((root / "decision.json").read_text())
design = value.get("design") if isinstance(value, dict) else None
if value.get("schema_version") != "1" or not isinstance(design, dict) or type(design.get("judge_enrichment")) is not bool:
    raise SystemExit("decision design requires boolean judge_enrichment")
relation = value.get("relation")
if not isinstance(relation, dict) or set(relation) != {"kind", "related_plans"}:
    raise SystemExit("decision relation is invalid")
kind = relation["kind"]
related = relation["related_plans"]
if kind not in {"new_direction", "revisit", "contradiction", "combine"} or not isinstance(related, list):
    raise SystemExit("decision relation kind or related_plans is invalid")
if len(related) != len(set(related)) or any(
    not isinstance(item, str)
    or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/c[0-9]{3}/p[0-9]{3}", item) is None
    for item in related
):
    raise SystemExit("related_plans must be unique full Plan SubjectRefs")
expected_count = {"new_direction": 0, "revisit": 1, "contradiction": 1}.get(kind)
if (expected_count is not None and len(related) != expected_count) or (kind == "combine" and len(related) < 2):
    raise SystemExit("decision relation cardinality is invalid")
comparisons = design.get("comparisons")
if not isinstance(comparisons, dict) or set(comparisons) != {"hypothesis"}:
    raise SystemExit("design.comparisons requires exactly hypothesis")
hypothesis = comparisons["hypothesis"]
allowed = {"strict_improvement", "no_regression", "diagnostic_only"}
if (
    not isinstance(hypothesis, dict)
    or set(hypothesis) != {"reason", "expected_observation"}
    or not isinstance(hypothesis.get("reason"), str)
    or not hypothesis["reason"].strip()
    or not isinstance(hypothesis.get("expected_observation"), dict)
    or set(hypothesis["expected_observation"]) != {"primary", "secondary"}
    or any(item not in allowed for item in hypothesis["expected_observation"].values())
):
    raise SystemExit("hypothesis expectation is invalid")
