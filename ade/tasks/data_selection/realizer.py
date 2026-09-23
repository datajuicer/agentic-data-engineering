"""Harness/Engine-owned realization of an admitted Data Selection artifact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Mapping, Protocol

from ade.tasks.data_selection.selection_runtime import execute_selection_async


JudgeBatch = Callable[[list[dict[str, object]]], Awaitable[list[dict[str, object]]]]


class SelectionObjectStore(Protocol):
    def put_json(self, uri: str, value: object) -> None: ...


@dataclass(frozen=True)
class SelectionRealization:
    selection: dict[str, object]
    selected_rows: tuple[dict[str, object], ...]


class SelectionRealizer:
    """Run the one canonical selector implementation against frozen inputs."""

    async def realize(
        self,
        script: bytes,
        candidate_inventory: bytes,
        training_dataset: bytes,
        *,
        select_size: int,
        judge_batch: JudgeBatch | None = None,
    ) -> SelectionRealization:
        selection, selected_rows = await execute_selection_async(
            script,
            candidate_inventory,
            training_dataset,
            select_size=select_size,
            judge_batch=judge_batch,
        )
        return SelectionRealization(selection, tuple(selected_rows))

    @staticmethod
    def publish_final_binding(
        io: SelectionObjectStore,
        realization: SelectionRealization,
        *,
        root_uri: str,
        dataset_name: str,
        candidate_pool_ref: str,
        pool_stats: Mapping[str, object],
    ) -> dict[str, object]:
        if not dataset_name:
            raise ValueError("SFT selection requires dataset_name")
        root = root_uri.rstrip("/")
        selection_ref = f"{root}/selection-result.json"
        data_ref = f"{root}/selected-examples.json"
        info_ref = f"{root}/dataset_info.json"
        io.put_json(selection_ref, realization.selection)
        io.put_json(data_ref, list(realization.selected_rows))
        io.put_json(
            info_ref,
            {
                dataset_name: {
                    "file_name": "selected-examples.json",
                    "formatting": "sharegpt",
                    "columns": {"messages": "conversations"},
                    "tags": {
                        "role_tag": "from",
                        "content_tag": "value",
                        "user_tag": "user",
                        "assistant_tag": "assistant",
                    },
                }
            },
        )
        return {
            "realization_status": "final",
            "data_ref": data_ref,
            "info_ref": info_ref,
            "selection_ref": selection_ref,
            "candidate_pool_ref": candidate_pool_ref,
            "dataset_name": dataset_name,
            "pool_stats": dict(pool_stats),
        }
