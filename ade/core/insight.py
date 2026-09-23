"""Canonical accepted insight state."""

from dataclasses import dataclass


@dataclass(frozen=True)
class InsightNode:
    insight_id: str
    statement: str
    evidence_ref_ids: tuple[str, ...]


@dataclass(frozen=True)
class InsightGraph:
    revision: int
    nodes: tuple[InsightNode, ...] = ()


@dataclass(frozen=True)
class InsightView:
    view_id: str
    scope: str
    basis_revision: int
    source_ids: tuple[str, ...]
    artifact_ref_id: str
