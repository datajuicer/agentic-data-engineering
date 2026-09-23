"""JSONL, rendering, schema, and identity validation for Rubric Jobs."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from ade.rubric_jobs.models import RubricInputRow, RubricOutputRow


PROTOCOL_VERSION = "ade.rubric_jobs.v1"
_PLACEHOLDER = re.compile(r"{{([A-Za-z_][A-Za-z0-9_]*)}}")
_ANY_BRACES = re.compile(r"{{.*?}}")


def canonical_digest(value: object) -> str:
    content = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def parse_input_jsonl(content: bytes | str) -> tuple[RubricInputRow, ...]:
    values = _parse_jsonl(content, "input")
    rows = tuple(RubricInputRow.from_dict(value) for value in values)
    if not rows:
        raise ValueError("Rubric Job input JSONL must not be empty")
    _reject_duplicate_ids((row.record_id for row in rows), "input")
    for row in rows:
        _validate_input_row(row)
    return rows


def parse_output_jsonl(content: bytes | str) -> tuple[RubricOutputRow, ...]:
    values = _parse_jsonl(content, "output")
    rows = tuple(RubricOutputRow.from_dict(value) for value in values)
    _reject_duplicate_ids((row.record_id for row in rows), "output")
    return rows


def encode_input_jsonl(rows: Iterable[RubricInputRow]) -> bytes:
    return _encode_jsonl(row.to_dict() for row in rows)


def encode_output_jsonl(rows: Iterable[RubricOutputRow]) -> bytes:
    return _encode_jsonl(row.to_dict() for row in rows)


def render_template_once(row: RubricInputRow) -> str:
    """Render only original placeholders and delimit substituted values."""

    _validate_input_row(row)

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        return f'<rubric_input name="{name}">\n{row.inputs[name]}\n</rubric_input>'

    return _PLACEHOLDER.sub(replace, row.rubric.template)


def submission_content_digest(
    rows: Iterable[RubricInputRow], job_metadata: dict[str, Any]
) -> str:
    if not isinstance(job_metadata, dict):
        raise ValueError("job metadata must be an object")
    return canonical_digest(
        {
            "protocol": PROTOCOL_VERSION,
            "rows": [row.to_dict() for row in rows],
        }
    )


def join_results(
    inputs: Iterable[RubricInputRow], outputs: Iterable[RubricOutputRow]
) -> tuple[RubricOutputRow, ...]:
    input_rows = tuple(inputs)
    output_rows = tuple(outputs)
    _reject_duplicate_ids((row.record_id for row in input_rows), "input")
    _reject_duplicate_ids((row.record_id for row in output_rows), "output")
    expected = {row.record_id: row for row in input_rows}
    actual = {row.record_id: row for row in output_rows}
    missing = sorted(set(expected) - set(actual))
    unknown = sorted(set(actual) - set(expected))
    if missing:
        raise ValueError(f"Rubric Job result is missing record IDs: {missing}")
    if unknown:
        raise ValueError(f"Rubric Job result has unknown record IDs: {unknown}")
    joined = []
    for input_row in input_rows:
        output_row = actual[input_row.record_id]
        if output_row.metadata != input_row.metadata:
            raise ValueError(f"Rubric Job metadata mismatch for {input_row.record_id}")
        if output_row.rubric_digest != canonical_digest(input_row.rubric.template):
            raise ValueError(f"Rubric digest mismatch for {input_row.record_id}")
        if output_row.output_schema_digest != canonical_digest(
            input_row.rubric.output_schema
        ):
            raise ValueError(f"output schema digest mismatch for {input_row.record_id}")
        if output_row.status == "ok":
            assert output_row.result is not None
            try:
                Draft202012Validator(input_row.rubric.output_schema).validate(
                    output_row.result
                )
            except ValidationError as error:
                raise ValueError(
                    f"dynamic output schema rejected {input_row.record_id}: {error.message}"
                ) from error
        joined.append(output_row)
    return tuple(joined)


def _validate_input_row(row: RubricInputRow) -> None:
    placeholders = tuple(_PLACEHOLDER.findall(row.rubric.template))
    if _ANY_BRACES.sub("", row.rubric.template) != _PLACEHOLDER.sub(
        "", row.rubric.template
    ):
        raise ValueError(f"rubric template contains an invalid placeholder: {row.record_id}")
    if len(set(placeholders)) != len(placeholders):
        raise ValueError(f"rubric template repeats a placeholder: {row.record_id}")
    required = set(row.rubric.required_variables)
    if required != set(placeholders) or required != set(row.inputs):
        raise ValueError(
            f"rubric variables, placeholders, and inputs must match: {row.record_id}"
        )
    try:
        Draft202012Validator.check_schema(row.rubric.output_schema)
    except SchemaError as error:
        raise ValueError(
            f"rubric output_schema is invalid for {row.record_id}: {error.message}"
        ) from error
    if row.rubric.output_schema.get("type") != "object":
        raise ValueError(f"rubric output_schema must describe an object: {row.record_id}")


def _parse_jsonl(content: bytes | str, label: str) -> tuple[object, ...]:
    try:
        text = content.decode("utf-8") if isinstance(content, bytes) else content
        # JSONL records are separated by LF.  ``str.splitlines()`` also treats
        # valid JSON string characters such as U+2028 and U+2029 as record
        # boundaries, corrupting otherwise valid model responses.
        lines = [line for line in text.split("\n") if line.strip()]
        return tuple(json.loads(line) for line in lines)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Rubric Job {label} is invalid JSONL") from error


def _encode_jsonl(values: Iterable[object]) -> bytes:
    return b"".join(
        (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for value in values
    )


def _reject_duplicate_ids(record_ids: Iterable[str], label: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for record_id in record_ids:
        if record_id in seen:
            duplicates.add(record_id)
        seen.add(record_id)
    if duplicates:
        raise ValueError(f"Rubric Job {label} has duplicate record IDs: {sorted(duplicates)}")
