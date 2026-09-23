"""Restricted pure-function contract for Data Selection scripts."""

from __future__ import annotations

import ast

from ade.tasks.safety_policy import validate_python_safety

ALLOWED_IMPORTS = frozenset({"json"})


def validate_selection_source(source: str) -> ast.Module:
    try:
        tree = ast.parse(source, filename="selection.py")
    except SyntaxError as error:
        raise ValueError("selection proposal is not valid Python") from error
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "select_trajectories"
    ]
    non_import_nodes = [node for node in tree.body if not isinstance(node, ast.Import)]
    valid_entrypoint = (
        len(non_import_nodes) == 1
        and len(functions) == 1
        and tuple(argument.arg for argument in functions[0].args.args)
        == ("candidate_inventory", "select_size", "judge")
        and not functions[0].decorator_list
        and functions[0].args.vararg is None
        and functions[0].args.kwarg is None
        and not functions[0].args.defaults
        and not functions[0].args.kw_defaults
    )
    if not valid_entrypoint:
        raise ValueError("selection proposal is outside restricted Python entrypoint")
    validate_python_safety(tree, allowed_imports=ALLOWED_IMPORTS)
    return tree
