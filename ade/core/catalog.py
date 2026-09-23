"""Run-global Plan intent and source-eligible result catalog."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ActivePlanIntent:
    coordinator_id: str
    plan_id: str
    accepted_revision: int
    decision_ref_id: str
    decision_report_ref_id: str | None
    relation_kind: str
    related_plan_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceEligibleResult:
    coordinator_id: str
    plan_id: str
    representative_trial_id: str
    score: float
    evaluation_profile: str
    artifact_ref_id: str
    artifact_digest: str
    rm_version: str
    archived_revision: int
    secondary_score: float | None = None


@dataclass(frozen=True)
class PlanCatalogState:
    revision: int = 0
    active_intents: tuple[ActivePlanIntent, ...] = ()
    source_eligible_results: tuple[SourceEligibleResult, ...] = ()

    def __post_init__(self) -> None:
        if self.revision < 0:
            raise ValueError("Plan Catalog revision must be non-negative")
        active = tuple(
            (item.coordinator_id, item.plan_id) for item in self.active_intents
        )
        sources = tuple(
            (item.coordinator_id, item.plan_id)
            for item in self.source_eligible_results
        )
        if len(active) != len(set(active)) or len(sources) != len(set(sources)):
            raise ValueError("Plan Catalog scopes must be unique")
