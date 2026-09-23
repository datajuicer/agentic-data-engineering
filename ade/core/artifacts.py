"""Immutable artifact references."""

from __future__ import annotations

from dataclasses import dataclass
import re

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    kind: str
    uri: str
    digest: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.artifact_id or not self.kind or not self.uri:
            raise ValueError("artifact_id, kind, and uri are required")
        if not _SHA256.fullmatch(self.digest):
            raise ValueError("artifact digest must be a lowercase sha256")
        if self.size_bytes < 0:
            raise ValueError("artifact size_bytes must be non-negative")

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ArtifactRef":
        return cls(
            artifact_id=str(value["artifact_id"]),
            kind=str(value["kind"]),
            uri=str(value["uri"]),
            digest=str(value["digest"]),
            size_bytes=int(value["size_bytes"]),
        )
