"""Shared bounded execution helpers for dataset tasks."""

from __future__ import annotations

import io


MAX_CAPTURED_OUTPUT_CHARS = 1_048_576


class CappedStringIO(io.StringIO):
    def __init__(self, limit: int = MAX_CAPTURED_OUTPUT_CHARS):
        super().__init__()
        self._limit = limit

    def write(self, value: str) -> int:
        remaining = max(0, self._limit - self.tell())
        if remaining:
            super().write(value[:remaining])
        return len(value)


def bounded_python_stdio_source(code: str) -> str:
    return f"""
import io
import sys
import traceback

class _CappedStringIO(io.StringIO):
    def __init__(self, limit={MAX_CAPTURED_OUTPUT_CHARS}):
        super().__init__()
        self._limit = limit

    def write(self, value):
        remaining = max(0, self._limit - self.tell())
        if remaining:
            super().write(value[:remaining])
        return len(value)

_real_stdout = sys.stdout
_real_stderr = sys.stderr
_stdout = _CappedStringIO()
_stderr = _CappedStringIO()
sys.stdout = sys.__stdout__ = _stdout
sys.stderr = sys.__stderr__ = _stderr
_exit_code = 0
try:
    exec(compile({code!r}, '<generated>', 'exec'), {{'__name__': '__main__'}})
except SystemExit as exc:
    _exit_code = exc.code if isinstance(exc.code, int) else 1
except BaseException:
    traceback.print_exc()
    _exit_code = 1
finally:
    sys.stdout = sys.__stdout__ = _real_stdout
    sys.stderr = sys.__stderr__ = _real_stderr
    _real_stdout.write(_stdout.getvalue())
    _real_stderr.write(_stderr.getvalue())
raise SystemExit(_exit_code)
"""
