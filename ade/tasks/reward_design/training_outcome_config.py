"""Reward Design config admission for Engine-owned training outcomes."""

from __future__ import annotations

from ade.tasks.reward_design.rewards.training_outcome import (
    parquet_data_sources,
    validate_adapter_binding,
)

__all__ = ["parquet_data_sources", "validate_adapter_binding"]
