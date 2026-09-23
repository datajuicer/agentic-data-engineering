from __future__ import annotations

import builtins
import contextlib
import io
import json
import os
import resource
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_EXECUTION_TIMEOUT_SECONDS = 6.0


def compute_score(
    question_prompt: str,
    response_content: str,
    extracted_answer: str,
    ground_truth: str,
    response_length_tokens: int,
    max_response_length_tokens: int,
) -> float:
    del question_prompt, response_content, response_length_tokens, max_response_length_tokens
    code = str(extracted_answer or "")
    if not code:
        return 0.0
    try:
        tests = json.loads(str(ground_truth))
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0.0
    if not isinstance(tests, dict):
        return 0.0
    try:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--sandbox"],
            input=json.dumps({"code": code, "tests": tests}),
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_EXECUTION_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0.0
    return 1.0 if completed.returncode == 0 else 0.0


def _execute(code: str, tests: dict[str, Any]) -> bool:
    try:
        _set_resource_limits()
        _install_audit_hook()
        if isinstance(tests.get("functional"), str):
            namespace = _execution_namespace()
            exec(compile(code, "<candidate>", "exec"), namespace)
            exec(compile(tests["functional"], "<tests>", "exec"), namespace)
        else:
            _run_stdio_tests(code, tests)
    except BaseException:
        return False
    return True


def _run_stdio_tests(code: str, tests: dict[str, Any]) -> None:
    inputs = tests.get("inputs")
    outputs = tests.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or len(inputs) != len(outputs):
        raise ValueError("stdin ground truth requires equally sized inputs and outputs")
    compiled = compile(code, "<candidate>", "exec")
    for stdin_value, expected in zip(inputs, outputs, strict=True):
        stdin = io.StringIO(str(stdin_value))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            previous = sys.stdin
            sys.stdin = stdin
            try:
                exec(compiled, _execution_namespace())
            finally:
                sys.stdin = previous
        if stdout.getvalue().strip() != str(expected).strip():
            raise AssertionError("stdout mismatch")


def _execution_namespace() -> dict[str, Any]:
    import collections
    import functools
    import itertools
    import math
    import typing

    return {
        "__name__": "__main__",
        "__builtins__": builtins,
        "collections": collections,
        "functools": functools,
        "itertools": itertools,
        "math": math,
        **vars(typing),
    }


def _set_resource_limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    memory = 2 * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))


def _install_audit_hook() -> None:
    stdlib_roots = tuple(
        Path(value).resolve() for value in {sys.base_prefix, sys.prefix} if value
    )

    def inside(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event.startswith("socket.") or event in {
            "subprocess.Popen",
            "os.system",
            "os.posix_spawn",
            "os.posix_spawnp",
            "os.exec",
            "os.spawn",
        }:
            raise PermissionError(f"blocked operation: {event}")
        if event == "open" and args:
            try:
                path = Path(os.fspath(args[0])).expanduser().resolve()
            except (TypeError, ValueError, OSError):
                return
            mode = str(args[1]) if len(args) > 1 else "r"
            if any(flag in mode for flag in ("w", "a", "x", "+")):
                raise PermissionError(f"blocked operation: write {path}")
            if not any(inside(path, root) for root in stdlib_roots):
                raise PermissionError(f"blocked operation: read {path}")

    sys.addaudithook(audit)


def _main() -> int:
    if sys.argv[1:] != ["--sandbox"]:
        return 2
    try:
        payload = json.loads(sys.stdin.read())
        code = payload["code"]
        tests = payload["tests"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 2
    return 0 if isinstance(code, str) and isinstance(tests, dict) and _execute(code, tests) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
