"""Restricted source and realized-shape contract for Curriculum Learning."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence

from ade.tasks.safety_policy import validate_python_safety


ALLOWED_IMPORTS = frozenset({"json"})
ENTRYPOINT_ARGUMENTS = (
    "candidate_inventory",
    "total_steps",
    "prompts_per_step",
    "judge_batch",
)


def validate_curriculum_source(source: str) -> ast.Module:
    try:
        tree = ast.parse(source, filename="curriculum.py")
    except SyntaxError as error:
        raise ValueError("curriculum proposal is not valid Python") from error
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "build_curriculum"
    ]
    non_import_nodes = [node for node in tree.body if not isinstance(node, ast.Import)]
    valid = (
        len(non_import_nodes) == 1
        and len(functions) == 1
        and tuple(argument.arg for argument in functions[0].args.args)
        == ENTRYPOINT_ARGUMENTS
        and not functions[0].decorator_list
        and functions[0].args.vararg is None
        and functions[0].args.kwarg is None
        and not functions[0].args.defaults
        and not functions[0].args.kw_defaults
    )
    if not valid:
        raise ValueError("curriculum proposal is outside restricted Python entrypoint")
    validate_python_safety(tree, allowed_imports=ALLOWED_IMPORTS)
    return tree


def canonical_step_lists(
    schedule: object,
    *,
    inventory: Sequence[Mapping[str, object]],
    total_steps: int,
    prompts_per_step: int,
) -> list[list[str]]:
    if not isinstance(schedule, list) or len(schedule) != total_steps:
        raise ValueError(f"curriculum must return exactly {total_steps} steps")
    source_order: dict[str, int] = {}
    for index, row in enumerate(inventory):
        problem_id = row.get("problem_id")
        order = row.get("source_order")
        if (
            not isinstance(problem_id, str)
            or not problem_id
            or type(order) is not int
            or order < 0
            or problem_id in source_order
        ):
            raise ValueError(f"candidate inventory row {index} is invalid")
        source_order[problem_id] = order
    canonical: list[list[str]] = []
    for step_index, step in enumerate(schedule, 1):
        if (
            not isinstance(step, list)
            or len(step) != prompts_per_step
            or any(not isinstance(item, str) or not item for item in step)
            or len(step) != len(set(step))
        ):
            raise ValueError(
                f"curriculum step {step_index} must contain exactly "
                f"{prompts_per_step} unique problem IDs"
            )
        unknown = set(step) - source_order.keys()
        if unknown:
            raise ValueError(
                f"curriculum step {step_index} contains unauthorized problem IDs"
            )
        canonical.append(sorted(step, key=source_order.__getitem__))
    return canonical
