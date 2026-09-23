from __future__ import annotations

import json
from typing import Any

from ..contracts import EvalExample
from .base import first_present, require_first_present
from .python_code import PYTHON_IMPORTS, PythonFunctionalTask


class EvalPlusFunctionalTask(PythonFunctionalTask):
    benchmark = "evalplus"

    def row_to_example(
        self,
        row: dict[str, Any],
        idx: int,
        *,
        dataset_name: str | None = None,
    ) -> EvalExample:
        prompt = require_first_present(
            row,
            ("prompt", "text", "question"),
            task_type=self.task_type,
            row_index=idx,
            field_name="prompt",
        )
        entry_point = require_first_present(
            row,
            ("entry_point",),
            task_type=self.task_type,
            row_index=idx,
            field_name="entry point",
        )
        canonical_solution = require_first_present(
            row,
            ("canonical_solution", "code"),
            task_type=self.task_type,
            row_index=idx,
            field_name="canonical solution",
        )
        base_input = _evalplus_inputs(row, "base_input")
        plus_input = _evalplus_inputs(row, "plus_input")
        example_id = str(
            first_present(row, ("task_id", "id", "index"))
            or f"{dataset_name or 'dataset'}/{idx}"
        )
        if self.task_type == "mbpp_plus":
            base_input = _mbpp_deserialize_inputs(example_id, base_input)
            plus_input = _mbpp_deserialize_inputs(example_id, plus_input)
        return EvalExample(
            id=example_id,
            prompt=(
                "Complete the Python function. Return executable Python code in a Python code block.\n"
                f"```python\n{str(prompt).rstrip()}\n```"
            ),
            reference="",
            row_index=idx,
            meta={
                "dataset_name": dataset_name,
                "raw_example": dict(row),
                "canonical_prompt": str(prompt),
                "canonical_solution": str(canonical_solution),
                "entry_point": str(entry_point),
                "base_input": base_input,
                "plus_input": plus_input,
                "atol": float(row.get("atol") or 0.0),
            },
        )

    def build_test_code(self, example: EvalExample, code: str) -> str:
        prompt = str(example.meta["canonical_prompt"])
        entry_point = str(example.meta["entry_point"])
        canonical = str(example.meta["canonical_solution"])
        if f"def {entry_point}" not in code:
            code = prompt.rstrip() + "\n" + code
        if f"def {entry_point}" not in canonical:
            canonical = prompt.rstrip() + "\n" + canonical
        inputs = list(example.meta.get("base_input") or []) + list(
            example.meta.get("plus_input") or []
        )
        atol = float(example.meta.get("atol") or 0.0)
        return "\n".join(
            [
                PYTHON_IMPORTS,
                "import copy",
                f"_candidate_source = {code!r}",
                f"_reference_source = {canonical!r}",
                "_candidate_ns = dict(globals())",
                "_reference_ns = dict(globals())",
                "exec(_candidate_source, _candidate_ns)",
                "exec(_reference_source, _reference_ns)",
                f"_candidate = _candidate_ns[{entry_point!r}]",
                f"_reference = _reference_ns[{entry_point!r}]",
                f"_cases = {inputs!r}",
                f"_atol = {atol!r}",
                "def _ade_equal(_actual, _expected):",
                "    if isinstance(_actual, (int, float, complex)) and isinstance(_expected, (int, float, complex)):",
                "        return abs(_actual - _expected) <= _atol if _atol else _actual == _expected",
                "    if isinstance(_actual, dict) and isinstance(_expected, dict):",
                "        return _actual.keys() == _expected.keys() and all(_ade_equal(_actual[k], _expected[k]) for k in _actual)",
                "    if isinstance(_actual, (list, tuple)) and isinstance(_expected, (list, tuple)):",
                "        return len(_actual) == len(_expected) and all(_ade_equal(a, e) for a, e in zip(_actual, _expected))",
                "    _result = _actual == _expected",
                "    return bool(_result.all()) if hasattr(_result, 'all') else bool(_result)",
                "for _case in _cases:",
                "    _args = tuple(_case) if isinstance(_case, (list, tuple)) else (_case,)",
                "    _expected = _reference(*copy.deepcopy(_args))",
                "    _actual = _candidate(*copy.deepcopy(_args))",
                "    assert _ade_equal(_actual, _expected), (_case, _actual, _expected)",
            ]
        )

    def expected_entry_points(self, example: EvalExample) -> list[str]:
        return [str(example.meta["entry_point"])]

    def output_metadata(self, example: EvalExample) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "entry_point": example.meta.get("entry_point"),
            "base_case_count": len(example.meta.get("base_input") or []),
            "plus_case_count": len(example.meta.get("plus_input") or []),
        }


class HumanEvalPlusTask(EvalPlusFunctionalTask):
    task_type = "humaneval_plus"
    benchmark = "humaneval_plus"


class MBPPPlusTask(EvalPlusFunctionalTask):
    task_type = "mbpp_plus"
    benchmark = "mbpp_plus"


def _evalplus_inputs(row: dict[str, Any], key: str) -> list[Any]:
    value = row.get(key)
    if value is None and isinstance(row.get(f"{key}_json"), str):
        value = json.loads(row[f"{key}_json"])
    return list(value or [])


def _mbpp_deserialize_inputs(task_id: str, inputs: list[Any]) -> list[Any]:
    number = int(task_id.split("/")[-1])
    tuple_rows = {2, 116, 132, 143, 222, 261, 273, 394, 399, 421, 424, 429, 470, 560, 579, 596, 616, 630, 726, 740, 744, 809}
    nested_tuple_rows = {63, 64, 70, 94, 120, 237, 272, 299, 400, 409, 417, 438, 473, 614, 780}
    first_arg_tuple_rows = {250, 405, 446, 617, 720, 763, 808}
    if number in tuple_rows:
        return [[tuple(item) for item in case] for case in inputs]
    if number in nested_tuple_rows:
        return [[[tuple(item) for item in group] for group in case] for case in inputs]
    if number in {75, 413, 444, 753}:
        return [[[tuple(item) for item in case[0]], case[1]] for case in inputs]
    if number in {106, 750}:
        return [[case[0], tuple(case[1])] for case in inputs]
    if number == 115:
        return [[[(set(item) if item else {}) for item in case[0]]] for case in inputs]
    if number == 124:
        return [(float(case[0]), complex(case[1])) for case in inputs]
    if number in first_arg_tuple_rows:
        return [[tuple(case[0]), case[1]] for case in inputs]
    if number in {259, 401, 445}:
        return [[tuple(tuple(item) for item in group) for group in case] for case in inputs]
    if number == 278:
        return [[tuple(tuple(item) if isinstance(item, list) else item for item in case[0])] for case in inputs]
    if number == 307:
        return [[tuple(case[0]), case[1], case[2]] for case in inputs]
    if number == 722:
        return [[{key: tuple(value) for key, value in case[0].items()}, *case[1:]] for case in inputs]
    if number == 252:
        return [[complex(case[0])] for case in inputs]
    if number in {580, 615, 791}:
        return [_lists_to_tuples(case) for case in inputs]
    return inputs


def _lists_to_tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_lists_to_tuples(item) for item in value)
    return value
