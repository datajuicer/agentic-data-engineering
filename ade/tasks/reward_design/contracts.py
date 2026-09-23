"""Typed results returned by Engine training backends."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RFTCheckpointResult:
    step: int
    checkpoint_path: str
    online_validation: dict[str, Any]


@dataclass(frozen=True)
class RFTTrainingResult:
    checkpoints: tuple[RFTCheckpointResult, ...]
    analysis_manifest_path: str | None = None
