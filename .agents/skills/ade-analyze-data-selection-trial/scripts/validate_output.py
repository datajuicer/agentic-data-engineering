from __future__ import annotations

import json
from pathlib import Path
import sys


POOLS = {"selected_examples", "offline_validation"}


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "output")
    files = {path.name for path in root.iterdir() if path.is_file()}
    if "review-plan.json" in files:
        assert files == {"review-plan.json"}
        value = json.loads((root / "review-plan.json").read_text(encoding="utf-8"))
        assert value.get("schema_version") == "ade.analysis_review_plan.v2"
        assert value.get("hypotheses") and all(isinstance(item, str) and item.strip() for item in value["hypotheses"])
        batches = value.get("batches")
        assert isinstance(batches, list) and batches
        assert {batch.get("pool") for batch in batches} == POOLS
        for batch in batches:
            assert batch.get("batch_id") and batch.get("investigation_purpose")
            assert batch.get("selection") == {"mode": "all"}
            assert isinstance(batch.get("rubrics"), list) and batch["rubrics"]
            rubric_ids = []
            for rubric in batch["rubrics"]:
                assert isinstance(rubric, dict)
                assert isinstance(rubric.get("rubric_id"), str) and rubric["rubric_id"].strip()
                assert isinstance(rubric.get("instruction"), str) and rubric["instruction"].strip()
                labels = rubric.get("labels")
                assert isinstance(labels, list) and len(labels) >= 2
                assert all(isinstance(label, str) and label.strip() for label in labels)
                assert len(labels) == len(set(labels))
                rubric_ids.append(rubric["rubric_id"])
            assert len(rubric_ids) == len(set(rubric_ids))
        return
    assert files == {"analysis.md", "findings.md"}
    assert (root / "analysis.md").read_text(encoding="utf-8").strip()
    assert (root / "findings.md").read_text(encoding="utf-8").strip()


if __name__ == "__main__":
    main()
