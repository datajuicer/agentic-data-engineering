"""Global gate for role-declared semantic Agent outputs."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from ade.core.validation import DeliveryViolation, ValidationReport


class DeliveryGate:
    def __init__(
        self,
        *,
        allowed_paths: set[str],
        required_paths: set[str] | None = None,
        max_delivery_bytes: int = 1_000_000,
    ) -> None:
        self.allowed_paths = frozenset(allowed_paths)
        self.required_paths = frozenset(
            allowed_paths if required_paths is None else required_paths
        )
        if not self.required_paths.issubset(self.allowed_paths):
            raise ValueError("required semantic outputs must be allowed")
        self.max_delivery_bytes = max_delivery_bytes

    def validate(self, attempt: str | Path) -> ValidationReport:
        output = Path(attempt) / "output"
        violations: list[DeliveryViolation] = []
        for relative_name in sorted(self.required_paths):
            relative = PurePosixPath(relative_name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe configured semantic output: {relative_name}")
            if not output.joinpath(*relative.parts).is_file():
                violations.append(
                    DeliveryViolation(
                        "missing_semantic_output",
                        f"output/{relative_name} is required",
                        path=relative_name,
                        repairable=True,
                    )
                )
        actual = {
            path.relative_to(output).as_posix()
            for path in output.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        total_bytes = 0
        for path in output.rglob("*"):
            if path.is_symlink():
                violations.append(
                    DeliveryViolation("output_symlink", path.relative_to(output).as_posix())
                )
            elif path.is_file() and "__pycache__" not in path.parts:
                total_bytes += path.stat().st_size
        if total_bytes > self.max_delivery_bytes:
            violations.append(
                DeliveryViolation("delivery_too_large", "delivery artifacts exceed size limit")
            )
        for relative_name in sorted(actual - self.allowed_paths):
            violations.append(
                DeliveryViolation(
                    "undeclared_output",
                    relative_name,
                    path=relative_name,
                    repairable=True,
                )
            )
        return ValidationReport(tuple(violations))
