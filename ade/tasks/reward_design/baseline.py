"""Task-owned fixed math reward baseline for Reward Design."""

from __future__ import annotations

from pathlib import Path

from ade.tasks.reward_design.rewards.baselines import baseline_math
from ade.tasks.contracts import BaselineArtifactRequest, CompiledArtifact


def build_baseline(request: BaselineArtifactRequest) -> CompiledArtifact:
    del request
    source = Path(baseline_math.__file__).read_bytes()
    return CompiledArtifact(
        path="reward.py",
        kind="reward_design",
        content=source,
    )
