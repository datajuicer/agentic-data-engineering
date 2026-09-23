#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import re
import sys


PLAN_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/c[0-9]{3}/p[0-9]{3}")
RELATIONS = {"new_direction", "revisit", "contradiction", "combine"}
EXPECTED_RELATIONS = {"strict_improvement", "no_regression", "diagnostic_only"}


def fail(message: str) -> None:
    raise SystemExit(message)


def load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"{path}: invalid JSON: {error}")
    if not isinstance(value, dict):
        fail(f"{path}: expected object")
    return value


def non_empty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_relation(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"kind", "related_plans"}:
        fail("decision.json: relation requires exactly kind and related_plans")
    kind = value["kind"]
    related = value["related_plans"]
    if kind not in RELATIONS:
        fail("decision.json: unsupported relation kind")
    if (
        not isinstance(related, list)
        or any(
            not isinstance(item, str) or PLAN_REF.fullmatch(item) is None
            for item in related
        )
        or len(related) != len(set(related))
    ):
        fail(
            "decision.json: related_plans require unique "
            "<run>/cNNN/pNNN SubjectRefs"
        )
    expected = (
        0
        if kind == "new_direction"
        else 1
        if kind in {"revisit", "contradiction"}
        else None
    )
    if expected is not None and len(related) != expected:
        fail(f"decision.json: {kind} requires {expected} related Plans")
    if kind == "combine" and len(related) < 2:
        fail("decision.json: combine requires at least two related Plans")


def validate_comparisons(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"hypothesis"}:
        fail("decision.json: comparisons require exactly hypothesis")
    hypothesis = value["hypothesis"]
    if not isinstance(hypothesis, dict) or set(hypothesis) != {
        "reason",
        "expected_observation",
    }:
        fail("decision.json: hypothesis expectation fields are invalid")
    if not non_empty(hypothesis["reason"]):
        fail("decision.json: hypothesis reason is required")
    expected = hypothesis["expected_observation"]
    if (
        not isinstance(expected, dict)
        or set(expected) != {"primary", "secondary"}
        or any(item not in EXPECTED_RELATIONS for item in expected.values())
    ):
        fail("decision.json: hypothesis expected_observation is invalid")


def main() -> None:
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "output")
    try:
        files = {path.name for path in output.iterdir() if path.is_file()}
    except OSError as error:
        fail(f"{output}: cannot read output directory: {error}")
    if files != {"decision.json", "plan.md"}:
        fail("output: require exactly decision.json and plan.md")
    try:
        if not (output / "plan.md").read_text(encoding="utf-8").strip():
            fail("plan.md: require non-empty UTF-8 content")
    except (OSError, UnicodeDecodeError) as error:
        fail(f"plan.md: invalid UTF-8: {error}")
    decision = load(output / "decision.json")
    if (
        set(decision) != {"schema_version", "relation", "design"}
        or decision["schema_version"] != "1"
    ):
        fail("decision.json: require schema_version 1, relation, and design")
    validate_relation(decision["relation"])
    design = decision["design"]
    if not isinstance(design, dict) or set(design) != {"comparisons"}:
        fail("decision.json: design requires exactly comparisons")
    validate_comparisons(design["comparisons"])


if __name__ == "__main__":
    main()
