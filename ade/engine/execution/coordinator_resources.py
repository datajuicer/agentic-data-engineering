"""Deterministic Coordinator-scoped GPU resource-group contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class WorkloadAllocation:
    coordinator_id: str
    workload: str
    gpu_slots: tuple[int, ...]


class CoordinatorResourceBusyError(ValueError):
    pass


class CoordinatorResourceBook:
    def __init__(
        self,
        *,
        coordinator_count: int,
        total_gpus: int,
        capacity_gpus: int,
        allocations: Mapping[str, int],
        training_with_online_validation: bool,
        offline_with_operator_evaluation: bool,
    ) -> None:
        if coordinator_count < 1 or capacity_gpus < 1:
            raise ValueError("Coordinator resource dimensions must be positive")
        if total_gpus < coordinator_count * capacity_gpus:
            raise ValueError("cluster cannot admit all Coordinator resource groups")
        required = {
            "training",
            "online_validation",
            "offline_validation",
            "operator_evaluation",
        }
        if set(allocations) != required or any(
            type(value) is not int or value < 1 or value > capacity_gpus
            for value in allocations.values()
        ):
            raise ValueError("Coordinator workload allocations are invalid")
        if (
            training_with_online_validation
            and allocations["training"] + allocations["online_validation"]
            > capacity_gpus
        ):
            raise ValueError("training + online allocation exceeds group capacity")
        if (
            offline_with_operator_evaluation
            and allocations["offline_validation"]
            + allocations["operator_evaluation"]
            > capacity_gpus
        ):
            raise ValueError("offline + operator allocation exceeds group capacity")
        self.coordinator_count = coordinator_count
        self.capacity_gpus = capacity_gpus
        self.allocations = dict(allocations)
        self.training_with_online_validation = training_with_online_validation
        self.offline_with_operator_evaluation = offline_with_operator_evaluation
        self._active: dict[str, list[WorkloadAllocation]] = {}

    def acquire(self, coordinator_id: str, workload: str) -> WorkloadAllocation:
        index = self._coordinator_index(coordinator_id)
        if workload not in self.allocations:
            raise ValueError(f"unknown Coordinator workload: {workload}")
        active = self._active.setdefault(str(index), [])
        if any(
            not self._compatible(workload, other.workload)
            for other in active
        ):
            raise CoordinatorResourceBusyError("Coordinator workloads cannot overlap")
        base = index * self.capacity_gpus
        count = self.allocations[workload]
        occupied = {slot for item in active for slot in item.gpu_slots}
        slots = next(
            (
                candidate
                for offset in range(0, self.capacity_gpus - count + 1, count)
                if not occupied.intersection(
                    candidate := tuple(range(base + offset, base + offset + count))
                )
            ),
            None,
        )
        if slots is None:
            raise CoordinatorResourceBusyError("Coordinator GPU capacity is occupied")
        allocation = WorkloadAllocation(coordinator_id, workload, slots)
        active.append(allocation)
        return allocation

    def release(self, allocation: WorkloadAllocation) -> None:
        active = self._active.get(str(self._coordinator_index(allocation.coordinator_id)))
        if active is None or allocation not in active:
            raise ValueError("Coordinator workload allocation is not active")
        active.remove(allocation)

    def _compatible(self, left: str, right: str) -> bool:
        pair = {left, right}
        if left == right == "online_validation":
            return True
        return (
            pair == {"training", "online_validation"}
            and self.training_with_online_validation
        ) or (
            pair == {"offline_validation", "operator_evaluation"}
            and self.offline_with_operator_evaluation
        )

    def _coordinator_index(self, coordinator_id: str) -> int:
        if not coordinator_id.startswith("c") or not coordinator_id[1:].isdigit():
            raise ValueError("Coordinator ID must be cNNN")
        number = int(coordinator_id[1:])
        # Bootstrap c000 runs before Search and reuses the first physical
        # Coordinator group; c001 is the first Search owner of that group.
        index = 0 if number == 0 else number - 1
        if index < 0 or index >= self.coordinator_count:
            raise ValueError("Coordinator is outside the configured resource groups")
        return index


def resolved_resource_policy(runtime: Mapping[str, object]) -> dict[str, object]:
    cluster = runtime["cluster"]
    allocations = runtime["allocations"]
    concurrency = runtime["concurrency"]
    assert isinstance(cluster, Mapping)
    assert isinstance(allocations, Mapping)
    assert isinstance(concurrency, Mapping)
    return {
        "scope": "coordinator",
        "total_gpus": int(cluster["total_gpus"]),
        "capacity_gpus": int(runtime["coordinator_capacity_gpus"]),
        "allocations": {
            name: int(value["gpus"])
            for name, value in allocations.items()
            if isinstance(value, Mapping)
        },
        "concurrency": dict(concurrency),
    }


def validate_coordinator_workload_request(
    request: Mapping[str, object],
    *,
    requested_gpus: int,
) -> tuple[str, str]:
    policy = request.get("coordinator_resource_policy")
    workload = request.get("workload_allocation")
    owner = request.get("coordinator_resource_owner")
    if policy is None and workload is None and owner is None:
        return "standalone", "standalone"
    if not isinstance(policy, Mapping):
        raise ValueError("Coordinator resource policy is required")
    if not isinstance(workload, str) or not workload:
        raise ValueError("Coordinator workload allocation is required")
    if not isinstance(owner, str) or not owner:
        raise ValueError("Coordinator resource owner is required")
    allocations = policy.get("allocations")
    if not isinstance(allocations, Mapping) or workload not in allocations:
        raise ValueError("Coordinator workload is absent from the resource policy")
    expected = int(allocations[workload])
    if int(requested_gpus) != expected:
        raise ValueError(
            f"{workload} requests {requested_gpus} GPUs; configured allocation is {expected}"
        )
    return owner, workload
