from __future__ import annotations

import re
import time
import uuid
from typing import Any

import ray
from ray.util import placement_group, remove_placement_group

from ade.engine.execution.coordinator_resources import (
    CoordinatorResourceBusyError,
    CoordinatorResourceBook,
    WorkloadAllocation,
)


ALLOCATOR_NAME = "ade_gpu_lease_allocator"
ALLOCATOR_NAMESPACE = "ade"


class LeaseBook:
    def __init__(self) -> None:
        self._leases: dict[str, dict[str, Any]] = {}

    @property
    def pending_gpus(self) -> int:
        return sum(
            int(lease["allocated_gpus"])
            for lease in self._leases.values()
            if lease["status"] == "pending"
        )

    def reserve(
        self,
        *,
        owner: str,
        kind: str,
        requested_gpus: int,
        minimum_gpus: int,
        ray_available_gpus: int,
    ) -> dict[str, Any] | None:
        requested_gpus = max(1, int(requested_gpus))
        minimum_gpus = max(1, int(minimum_gpus))
        allocatable = max(0, int(ray_available_gpus) - self.pending_gpus)
        allocated_gpus = min(requested_gpus, allocatable)
        if allocated_gpus < minimum_gpus:
            return None
        lease_id = uuid.uuid4().hex
        lease = {
            "lease_id": lease_id,
            "owner": str(owner),
            "kind": str(kind),
            "requested_gpus": requested_gpus,
            "minimum_gpus": minimum_gpus,
            "allocated_gpus": allocated_gpus,
            "status": "pending",
            "created_at": time.time(),
        }
        self._leases[lease_id] = lease
        return dict(lease)

    def update(self, lease_id: str, **values: Any) -> dict[str, Any]:
        lease = self._leases[lease_id]
        lease.update(values)
        return dict(lease)

    def mark_active(self, lease_id: str) -> dict[str, Any]:
        return self.update(lease_id, status="active", activated_at=time.time())

    def get(self, lease_id: str) -> dict[str, Any]:
        return dict(self._leases[lease_id])

    def release(self, lease_id: str) -> dict[str, Any] | None:
        lease = self._leases.pop(lease_id, None)
        return dict(lease) if lease is not None else None

    def release_owner(self, owner: str) -> list[dict[str, Any]]:
        lease_ids = [
            lease_id
            for lease_id, lease in self._leases.items()
            if lease["owner"] == owner
        ]
        return [
            lease
            for lease_id in lease_ids
            for lease in [self.release(lease_id)]
            if lease is not None
        ]

    def release_inactive_command_leases(
        self, active_command_ids: set[str]
    ) -> list[dict[str, Any]]:
        """Release command-owned leases whose command is no longer processing.

        Engine workers can disappear after acquiring a lease.  The detached
        allocator outlives that worker, so the lease otherwise remains active
        forever and blocks the next attempt.  Command-owned lease suffixes are
        exact Engine command IDs; leave non-command/standalone leases alone.
        """
        stale_ids = []
        for lease_id, lease in self._leases.items():
            owner = str(lease.get("owner") or "")
            command_id = owner.rsplit("/", 1)[-1]
            if owner.count("/") >= 3 and command_id not in active_command_ids:
                stale_ids.append(lease_id)
        return [
            lease
            for lease_id in stale_ids
            for lease in [self.release(lease_id)]
            if lease is not None
        ]


def placement_group_bundles(
    *,
    kind: str,
    allocated_gpus: int,
    distributed_train: bool,
    target_node_resource: str | None = None,
) -> list[dict[str, int | float]]:
    allocated_gpus = max(1, int(allocated_gpus))
    node_constraint = (
        {str(target_node_resource): 0.001} if target_node_resource else {}
    )
    if kind == "train" and not distributed_train:
        return [{"CPU": 1, "GPU": allocated_gpus, **node_constraint}]
    cpu_per_gpu = 10 if kind == "train" else 1
    return [
        {"CPU": cpu_per_gpu, "GPU": 1, **node_constraint}
        for _ in range(allocated_gpus)
    ]


def _placement_group_name(owner: str, lease_id: str) -> str:
    safe_owner = re.sub(r"[^A-Za-z0-9_.-]+", "_", owner).strip("_") or "task"
    return f"ade_gpu_lease_{safe_owner[:80]}_{lease_id[:12]}"


@ray.remote(num_cpus=0, max_restarts=-1)
class GpuLeaseAllocator:
    def __init__(self) -> None:
        self._book = LeaseBook()
        self._coordinator_books: dict[str, CoordinatorResourceBook] = {}
        self._coordinator_allocations: dict[
            str, tuple[str, WorkloadAllocation]
        ] = {}
        self._external_allocations: dict[str, tuple[str, WorkloadAllocation]] = {}
        self._allocation_events: list[dict[str, Any]] = []

    def acquire(
        self,
        *,
        owner: str,
        kind: str,
        requested_gpus: int,
        minimum_gpus: int,
        distributed_train: bool = False,
        coordinator_owner: str | None = None,
        workload: str | None = None,
        coordinator_policy: dict[str, Any] | None = None,
        target_node_resource: str | None = None,
    ) -> dict[str, Any] | None:
        try:
            owned_allocation = self._admit_coordinator(
                coordinator_owner, workload, coordinator_policy
            )
        except CoordinatorResourceBusyError:
            return None
        lease = self._book.reserve(
            owner=owner,
            kind=kind,
            requested_gpus=requested_gpus,
            minimum_gpus=minimum_gpus,
            ray_available_gpus=int(ray.available_resources().get("GPU", 0) or 0),
        )
        if lease is None:
            self._release_coordinator_allocation(owned_allocation)
            return None
        pg_name = _placement_group_name(owner, lease["lease_id"])
        try:
            pg = placement_group(
                placement_group_bundles(
                    kind=kind,
                    allocated_gpus=lease["allocated_gpus"],
                    distributed_train=distributed_train,
                    target_node_resource=target_node_resource,
                ),
                strategy="PACK",
                name=pg_name,
                lifetime="detached",
            )
        except Exception:
            self._book.release(lease["lease_id"])
            self._release_coordinator_allocation(owned_allocation)
            raise
        if owned_allocation is not None:
            self._coordinator_allocations[lease["lease_id"]] = owned_allocation
        allocation = owned_allocation[1] if owned_allocation is not None else None
        return self._book.update(
            lease["lease_id"],
            placement_group=pg,
            placement_group_name=pg_name,
            distributed_train=bool(distributed_train),
            coordinator_owner=coordinator_owner,
            workload=workload,
            target_node_resource=target_node_resource,
            coordinator_gpu_slots=(
                list(allocation.gpu_slots) if allocation is not None else None
            ),
        )

    def mark_active(self, lease_id: str) -> dict[str, Any]:
        lease = self._book.mark_active(lease_id)
        placement_group_handle = self._book._leases[lease_id].get(
            "placement_group"
        )
        if placement_group_handle is not None:
            table = ray.util.placement_group_table(placement_group_handle)
            node_ids = sorted(
                {
                    str(node_id)
                    for node_id in (
                        table.get("bundles_to_node_id") or {}
                    ).values()
                }
            )
            lease = self._book.update(
                lease_id, placement_node_ids=node_ids
            )
        self._allocation_events.append(
            self._lease_event(lease, event="allocation_started")
        )
        return lease

    def release(self, lease_id: str) -> dict[str, Any] | None:
        lease = self._book.release(lease_id)
        if lease is not None:
            self._allocation_events.append(
                self._lease_event(lease, event="allocation_ended")
            )
        self._release_coordinator_allocation(
            self._coordinator_allocations.pop(lease_id, None)
        )
        if lease is not None and lease.get("placement_group") is not None:
            remove_placement_group(lease["placement_group"])
        return lease

    def release_owner(self, owner: str) -> list[dict[str, Any]]:
        lease_ids = [
            lease_id
            for lease_id, lease in self._book._leases.items()
            if lease["owner"] == owner
        ]
        leases = self._book.release_owner(owner)
        for lease in leases:
            self._allocation_events.append(
                self._lease_event(lease, event="allocation_ended")
            )
        for lease_id in lease_ids:
            self._release_coordinator_allocation(
                self._coordinator_allocations.pop(lease_id, None)
            )
        for lease in leases:
            if lease.get("placement_group") is not None:
                remove_placement_group(lease["placement_group"])
        return leases

    def release_inactive_command_leases(
        self, active_command_ids: list[str]
    ) -> list[dict[str, Any]]:
        active = {str(command_id) for command_id in active_command_ids}
        leases = self._book.release_inactive_command_leases(active)
        for lease in leases:
            lease_id = str(lease["lease_id"])
            self._allocation_events.append(
                self._lease_event(lease, event="allocation_ended")
            )
            self._release_coordinator_allocation(
                self._coordinator_allocations.pop(lease_id, None)
            )
            if lease.get("placement_group") is not None:
                remove_placement_group(lease["placement_group"])
        return leases

    def _admit_coordinator(
        self,
        owner: str | None,
        workload: str | None,
        policy: dict[str, Any] | None,
    ) -> tuple[str, WorkloadAllocation] | None:
        if owner is None and workload is None and policy is None:
            return None
        if not owner or not workload or not isinstance(policy, dict):
            raise ValueError("Coordinator lease identity is incomplete")
        run_id, separator, coordinator_id = owner.rpartition("/")
        if not separator or not run_id:
            raise ValueError("Coordinator lease owner must be run_id/coordinator_id")
        book = self._coordinator_books.get(run_id)
        if book is None:
            concurrency = policy["concurrency"]
            book = CoordinatorResourceBook(
                coordinator_count=int(policy["total_gpus"]) // int(policy["capacity_gpus"]),
                total_gpus=int(policy["total_gpus"]),
                capacity_gpus=int(policy["capacity_gpus"]),
                allocations=policy["allocations"],
                training_with_online_validation=bool(
                    concurrency["training_with_online_validation"]
                ),
                offline_with_operator_evaluation=bool(
                    concurrency["offline_with_operator_evaluation"]
                ),
            )
            self._coordinator_books[run_id] = book
        return run_id, book.acquire(coordinator_id, workload)

    def _release_coordinator_allocation(
        self,
        owned: tuple[str, WorkloadAllocation] | None,
    ) -> None:
        if owned is None:
            return
        run_id, allocation = owned
        self._coordinator_books[run_id].release(allocation)

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                key: value
                for key, value in lease.items()
                if key != "placement_group"
            }
            for lease in self._book._leases.values()
        ]

    def admit_external(
        self,
        *,
        coordinator_owner: str,
        workload: str,
        coordinator_policy: dict[str, Any],
    ) -> str:
        owned = self._admit_coordinator(
            coordinator_owner, workload, coordinator_policy
        )
        assert owned is not None
        token = uuid.uuid4().hex
        self._external_allocations[token] = owned
        run_id, allocation = owned
        self._allocation_events.append(
            self._external_event(
                token,
                run_id,
                allocation,
                event="allocation_started",
            )
        )
        return token

    def release_external(self, token: str) -> None:
        owned = self._external_allocations.pop(token, None)
        if owned is None:
            raise ValueError("external Coordinator workload token is unknown")
        run_id, allocation = owned
        self._allocation_events.append(
            self._external_event(
                token,
                run_id,
                allocation,
                event="allocation_ended",
            )
        )
        self._release_coordinator_allocation(owned)

    def monitor_snapshot(self, run_id: str) -> dict[str, Any]:
        leases = []
        for lease in self._book._leases.values():
            owner = str(lease.get("coordinator_owner") or "")
            if owner == run_id or owner.startswith(f"{run_id}/"):
                leases.append(
                    {key: value for key, value in lease.items() if key != "placement_group"}
                )
        external = []
        for token, (owner_run_id, allocation) in self._external_allocations.items():
            if owner_run_id == run_id:
                external.append(
                    {
                        "token": token,
                        "run_id": owner_run_id,
                        "coordinator_id": allocation.coordinator_id,
                        "workload": allocation.workload,
                        "gpu_slots": list(allocation.gpu_slots),
                    }
                )
        return {
            "leases": leases,
            "external_allocations": external,
            "events": [
                dict(event)
                for event in self._allocation_events
                if event.get("run_id") == run_id
            ],
        }

    @staticmethod
    def _lease_event(lease: dict[str, Any], *, event: str) -> dict[str, Any]:
        coordinator_owner = str(lease.get("coordinator_owner") or "")
        run_id, _, coordinator_id = coordinator_owner.rpartition("/")
        return {
            "event_id": f"{lease['lease_id']}:{event}",
            "event": event,
            "at": time.time(),
            "run_id": run_id,
            "coordinator_id": coordinator_id,
            "workload": lease.get("workload"),
            "lease_id": lease["lease_id"],
            "allocated_gpus": lease["allocated_gpus"],
            "gpu_slots": lease.get("coordinator_gpu_slots"),
            "placement_group_name": lease.get("placement_group_name"),
        }

    @staticmethod
    def _external_event(
        token: str,
        run_id: str,
        allocation: WorkloadAllocation,
        *,
        event: str,
    ) -> dict[str, Any]:
        return {
            "event_id": f"{token}:{event}",
            "event": event,
            "at": time.time(),
            "run_id": run_id,
            "coordinator_id": allocation.coordinator_id,
            "workload": allocation.workload,
            "external_token": token,
            "allocated_gpus": len(allocation.gpu_slots),
            "gpu_slots": list(allocation.gpu_slots),
        }


def get_gpu_lease_allocator() -> Any:
    return GpuLeaseAllocator.options(
        name=ALLOCATOR_NAME,
        namespace=ALLOCATOR_NAMESPACE,
        lifetime="detached",
        get_if_exists=True,
    ).remote()


def acquire_gpu_lease(
    *,
    owner: str,
    kind: str,
    requested_gpus: int,
    minimum_gpus: int,
    distributed_train: bool = False,
    poll_seconds: float = 2.0,
    coordinator_owner: str | None = None,
    workload: str | None = None,
    coordinator_policy: dict[str, Any] | None = None,
    target_node_resource: str | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    allocator = get_gpu_lease_allocator()
    started_at = time.monotonic()
    while True:
        lease = ray.get(
            allocator.acquire.remote(
                owner=owner,
                kind=kind,
                requested_gpus=requested_gpus,
                minimum_gpus=minimum_gpus,
                distributed_train=distributed_train,
                coordinator_owner=coordinator_owner,
                workload=workload,
                coordinator_policy=coordinator_policy,
                target_node_resource=target_node_resource,
            )
        )
        if lease is not None:
            break
        if timeout_seconds is not None and time.monotonic() - started_at >= float(
            timeout_seconds
        ):
            raise TimeoutError(
                "timed out waiting for GPU lease"
            )
        time.sleep(max(0.1, float(poll_seconds)))
    try:
        remaining = (
            None if timeout_seconds is None
            else max(0.0, float(timeout_seconds) - (time.monotonic() - started_at))
        )
        try:
            ray.get(lease["placement_group"].ready(), timeout=remaining)
        except ray.exceptions.GetTimeoutError as error:
            raise TimeoutError("timed out waiting for GPU placement group") from error
        ray.get(allocator.mark_active.remote(lease_id=lease["lease_id"]))
        return lease
    except Exception:
        ray.get(allocator.release.remote(lease_id=lease["lease_id"]))
        raise


def release_gpu_lease(lease: dict[str, Any] | None) -> None:
    if not lease:
        return
    allocator = get_gpu_lease_allocator()
    ray.get(allocator.release.remote(lease_id=str(lease["lease_id"])))


def release_gpu_leases_for_owner(owner: str) -> None:
    allocator = get_gpu_lease_allocator()
    ray.get(allocator.release_owner.remote(owner=str(owner)))


def release_inactive_command_leases(active_command_ids: set[str]) -> None:
    allocator = get_gpu_lease_allocator()
    active = {str(command_id) for command_id in active_command_ids}
    leases = ray.get(allocator.snapshot.remote())
    for lease in leases:
        owner = str(lease.get("owner") or "")
        command_id = owner.rsplit("/", 1)[-1]
        if owner.count("/") >= 3 and command_id not in active:
            # Use the stable pre-existing actor API so a long-lived detached
            # allocator does not need to be restarted for this repair.
            ray.get(allocator.release_owner.remote(owner=owner))


def admit_external_coordinator_workload(
    *,
    coordinator_owner: str,
    workload: str,
    coordinator_policy: dict[str, Any],
) -> str:
    allocator = get_gpu_lease_allocator()
    return str(
        ray.get(
            allocator.admit_external.remote(
                coordinator_owner=coordinator_owner,
                workload=workload,
                coordinator_policy=coordinator_policy,
            )
        )
    )


def release_external_coordinator_workload(token: str) -> None:
    allocator = get_gpu_lease_allocator()
    ray.get(allocator.release_external.remote(token=token))


def lease_metadata(lease: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in lease.items()
        if key != "placement_group"
    }
