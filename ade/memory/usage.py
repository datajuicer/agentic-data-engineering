"""Stable-ID Run usage ledger with resumable deduplication."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

from ade.engine.storage.atomic import write_json_atomic
from ade.rubric_jobs import TokenUsage


class RunUsageLedger:
    TOKEN_FIELDS = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "attempts",
        "retries",
        "requests",
    )

    def __init__(self, run_dir: str | Path) -> None:
        self.root = Path(run_dir) / "usage"
        self.events = self.root / "events.jsonl"
        self.summary = self.root / "usage-summary.json"
        self.token_summary = self.root / "token-summary.json"
        self.by_transition = self.root / "by-transition.json"
        self.by_component = self.root / "by-component.json"
        self.by_plan_trial = self.root / "by-plan-trial.json"
        self.lock = self.root / ".usage.lock"
        self.root.mkdir(parents=True, exist_ok=True)
        self.events.touch(exist_ok=True)

    def append(self, event: dict[str, object]) -> bool:
        return self._append(event, enrich_unavailable=False)

    def append_or_enrich(self, event: dict[str, object]) -> bool:
        """Append an event, or fill provider usage that was initially unavailable."""
        return self._append(event, enrich_unavailable=True)

    def _append(
        self,
        event: dict[str, object],
        *,
        enrich_unavailable: bool,
    ) -> bool:
        event_id = event.get("event_id")
        category = event.get("category")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("usage event_id is required")
        if not isinstance(category, str) or not category:
            raise ValueError("usage category is required")
        usage = TokenUsage.from_dict(
            {
                "usage_status": event.get("usage_status"),
                **{field: event.get(field) for field in self.TOKEN_FIELDS},
            }
        )
        normalized = {"schema_version": "2", **event, **usage.to_dict()}
        with self._locked():
            rows = self._rows()
            existing = next(
                (item for item in rows if item.get("event_id") == event_id),
                None,
            )
            if existing is not None:
                if existing != normalized:
                    if enrich_unavailable and self._is_usage_enrichment(
                        existing, normalized
                    ):
                        replaced = tuple(
                            normalized if item.get("event_id") == event_id else item
                            for item in rows
                        )
                        self._write_events(replaced)
                        self._write_summary(replaced)
                        return True
                    raise ValueError("usage event identity is immutable")
                return False
            with self.events.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(normalized, sort_keys=True) + "\n")
                handle.flush()
            self._write_summary((*rows, normalized))
        return True

    def _write_events(self, rows: tuple[dict[str, object], ...]) -> None:
        temporary = self.events.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
        temporary.replace(self.events)

    @classmethod
    def _is_usage_enrichment(
        cls,
        existing: dict[str, object],
        replacement: dict[str, object],
    ) -> bool:
        if existing.get("usage_status") != "unavailable":
            return False
        if replacement.get("usage_status") != "complete":
            return False
        ignored = {"usage_status", *cls.TOKEN_FIELDS}
        return {
            key: value for key, value in existing.items() if key not in ignored
        } == {
            key: value for key, value in replacement.items() if key not in ignored
        }

    def _rows(self) -> tuple[dict[str, object], ...]:
        rows = []
        for line in self.events.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("usage ledger row is invalid")
                rows.append(value)
        return tuple(rows)

    def _write_summary(self, rows: tuple[dict[str, object], ...]) -> None:
        existing = (
            json.loads(self.summary.read_text(encoding="utf-8"))
            if self.summary.is_file()
            else {}
        )
        categories: dict[str, dict[str, object]] = {}
        transitions: dict[str, dict[str, object]] = {}
        components: dict[str, dict[str, object]] = {}
        plan_trials: dict[str, dict[str, object]] = {}
        for row in rows:
            self._accumulate(categories, str(row["category"]), row)
            self._accumulate(
                transitions,
                str(row.get("transition_intent") or "unattributed"),
                row,
            )
            self._accumulate(
                components,
                str(row.get("component") or row["category"]),
                row,
            )
            plan_key = "/".join(
                str(row.get(field) or "_")
                for field in ("coordinator_id", "plan_id", "trial_id")
            )
            self._accumulate(plan_trials, plan_key, row)
        total: dict[str, object] = {
            **{field: 0 for field in self.TOKEN_FIELDS},
            "usage_status": "complete",
            "unavailable_requests": 0,
        }
        for counters in categories.values():
            for field in self.TOKEN_FIELDS:
                value = counters[field]
                if value is not None:
                    total[field] = int(total[field]) + int(value)
            total["unavailable_requests"] = int(total["unavailable_requests"]) + int(
                counters["unavailable_requests"]
            )
            total["usage_status"] = self._combined_status(
                str(total["usage_status"]), str(counters["usage_status"])
            )
        write_json_atomic(
            self.summary,
            {
                "schema_version": "2",
                "incremental_usage": total,
                "by_category": categories,
                "lineage_usage": existing.get("lineage_usage"),
                "event_count": len(rows),
            },
        )
        write_json_atomic(
            self.token_summary,
            {
                "schema_version": "2",
                "usage_status": total["usage_status"],
                "unavailable_requests": total["unavailable_requests"],
                **{
                    field: total[field]
                    for field in (
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "cached_tokens",
                        "reasoning_tokens",
                        "requests",
                    )
                },
            },
        )
        write_json_atomic(
            self.by_transition,
            {"schema_version": "1", "by_transition": transitions},
        )
        write_json_atomic(
            self.by_component,
            {"schema_version": "1", "by_component": components},
        )
        write_json_atomic(
            self.by_plan_trial,
            {"schema_version": "1", "by_plan_trial": plan_trials},
        )

    @classmethod
    def _accumulate(
        cls,
        groups: dict[str, dict[str, object]],
        key: str,
        row: dict[str, object],
    ) -> None:
        counters = groups.setdefault(
            key,
            {
                **{field: 0 for field in cls.TOKEN_FIELDS},
                "usage_status": "complete",
                "unavailable_requests": 0,
            },
        )
        for field in cls.TOKEN_FIELDS:
            value = row[field]
            if value is not None:
                counters[field] = int(counters[field]) + int(value)
        row_status = str(row.get("usage_status") or "unavailable")
        counters["usage_status"] = cls._combined_status(
            str(counters["usage_status"]), row_status
        )
        if row_status == "unavailable":
            counters["unavailable_requests"] = int(
                counters["unavailable_requests"]
            ) + int(row.get("requests") or 0)

    @staticmethod
    def _combined_status(left: str, right: str) -> str:
        if "unavailable" in {left, right} or "partial" in {left, right}:
            return "partial"
        return "complete"

    @contextmanager
    def _locked(self):
        with self.lock.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
