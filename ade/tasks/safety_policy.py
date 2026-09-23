"""Shared safety checks for Agent-produced Python task artifacts."""

from __future__ import annotations

import ast


SAFE_IMPORTS = frozenset(
    {
        "__future__",
        "math",
        "re",
        "ade.engine.eval.utils.math_exact",
    }
)

FORBIDDEN_CALLS = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "eval",
        "exec",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "open",
        "setattr",
        "vars",
    }
)


def python_safety_violations(
    tree: ast.AST,
    *,
    allowed_imports: frozenset[str] = SAFE_IMPORTS,
) -> tuple[tuple[str, str], ...]:
    """Return common safety violations as ``(kind, message)`` pairs."""
    violations: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Try, ast.TryStar)):
            violations.append(("try", "task artifact cannot catch execution failures"))
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = node.module if isinstance(node, ast.ImportFrom) else None
            names = [item.name for item in node.names]
            imported = [module] if module is not None else names
            if any(name not in allowed_imports for name in imported):
                violations.append(("import", "task artifact imports an untrusted module"))
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            violations.append(("dunder_name", "task artifact uses a restricted Python name"))
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            violations.append(
                ("dunder_attribute", "task artifact uses a restricted Python attribute")
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in FORBIDDEN_CALLS
        ):
            violations.append(("call", "task artifact uses a side-effecting Python call"))
    return tuple(violations)


def validate_python_safety(
    tree: ast.AST,
    *,
    allowed_imports: frozenset[str] = SAFE_IMPORTS,
) -> None:
    """Reject common side-effect and introspection capabilities."""
    violations = python_safety_violations(tree, allowed_imports=allowed_imports)
    if violations:
        raise ValueError(violations[0][1])
