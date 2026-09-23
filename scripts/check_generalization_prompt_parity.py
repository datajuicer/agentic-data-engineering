#!/usr/bin/env python3
"""Compare generalization inference contracts with operator-supplied saved requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ade.harness.evaluation import EvaluationConfigCompiler  # noqa: E402
from ade.tasks.data_selection.llamafactory_prompt_protocol import (  # noqa: E402
    native_tokenizer_prompt_metadata,
)


# Example request paths: replace these with requests from your own Runs.
_CASES = {
    "sft_math_n1_00000d": (
        "sft-math.n1-00000d.aime26",
        "runs/deployments/deployment-example-22/objects/inputs/"
        "openthoughts-math-sft-data-selection-ADE-formal-20000101T000000Z-00000d-"
        "c001-p005-t001-operator-test-attempt-001.json",
    ),
    "sft_math_n1_000006": (
        "sft-math.n1-000006.aime26",
        "runs/deployments/deployment-example-20/objects/inputs/"
        "openthoughts-math-sft-data-selection-ADE-formal-n1-20000101T000000Z-000006-"
        "c001-p001-t001-operator-test-attempt-001.json",
    ),
    "sft_math_n1_00000c": (
        "sft-math.n1-00000c.aime26",
        "runs/deployments/deployment-example-20/objects/inputs/"
        "openthoughts-math-sft-data-selection-ADE-formal-n1-20000101T000000Z-00000c-"
        "c001-p002-t001-operator-test-attempt-001.json",
    ),
    "sft_math_n3_000001": (
        "sft-math.n3-000001.aime26",
        "runs/deployments/deployment-example-99/objects/inputs/"
        "openthoughts-math-sft-data-selection-ADE-formal-n3-20000101T000000Z-000001-"
        "c003-p001-t001-operator-test-attempt-001.json",
    ),
    "sft_math_n3_000015": (
        "sft-math.n3-000015.aime26",
        "runs/deployments/deployment-example-99/objects/inputs/"
        "openthoughts-math-sft-data-selection-ADE-formal-n3-20000101T000000Z-000015-"
        "c001-p004-t001-operator-test-attempt-001.json",
    ),
    "sft_math_n3_000003": (
        "sft-math.n3-000003.aime26",
        "runs/deployments/deployment-example-99/objects/inputs/"
        "openthoughts-math-sft-data-selection-ADE-formal-n3-20000101T000000Z-000003-"
        "c002-p001-t001-operator-test-attempt-001.json",
    ),
    "sft_code_n1_000012": (
        "sft-code.n1-000012.humaneval_plus",
        "runs/deployments/deployment-example-21/objects/inputs/"
        "openthoughts-code-sft-data-selection-ADE-formal-20000101T000000Z-000012-"
        "c001-p003-t001-operator-test-attempt-001.json",
    ),
    "sft_code_n1_000004": (
        "sft-code.n1-000004.humaneval_plus",
        "runs/deployments/deployment-example-21/objects/inputs/"
        "openthoughts-code-sft-data-selection-ADE-formal-n1-20000101T000000Z-000004-"
        "c001-p005-t001-operator-test-attempt-001.json",
    ),
    "sft_code_n1_000007": (
        "sft-code.n1-000007.humaneval_plus",
        "runs/deployments/deployment-example-21/objects/inputs/"
        "openthoughts-code-sft-data-selection-ADE-formal-n1-20000101T000000Z-000007-"
        "c001-p002-t001-operator-test-attempt-001.json",
    ),
    "sft_code_n3_000016": (
        "sft-code.n3-000016.humaneval_plus",
        "runs/deployments/deployment-example-99/objects/inputs/"
        "openthoughts-code-sft-data-selection-ADE-formal-n3-20000101T000000Z-000016-"
        "c001-p004-t001-operator-test-attempt-001.json",
    ),
    "sft_code_n3_00000a": (
        "sft-code.n3-00000a.humaneval_plus",
        "runs/deployments/deployment-example-99/objects/inputs/"
        "openthoughts-code-sft-data-selection-ADE-formal-20000101T000000Z-00000a-"
        "c002-p004-t001-operator-test-attempt-001.json",
    ),
    "sft_code_n3_000010": (
        "sft-code.n3-000010.humaneval_plus",
        "runs/deployments/deployment-example-99/objects/inputs/"
        "openthoughts-code-sft-data-selection-ADE-formal-20000101T000000Z-000010-"
        "c001-p003-t001-operator-test-attempt-001.json",
    ),
    "rft_math_n1_000002": (
        "rft-math.n1-000002.aime24",
        "runs/deployments/deployment-example-20/objects/inputs/"
        "math-math-rft-reward-design-ADE-formal-100-step-n1-20000101T000000Z-000002-"
        "c001-p001-t001-operator-test-attempt-001.json",
    ),
    "rft_math_n1_00000e": (
        "rft-math.n1-00000e.aime24",
        "runs/deployments/deployment-example-21/objects/inputs/"
        "math-math-rft-reward-design-ADE-formal-20000101T000000Z-00000e-"
        "c001-p004-t001-operator-test-attempt-001.json",
    ),
    "rft_math_n1_00000b": (
        "rft-math.n1-00000b.aime24",
        "runs/deployments/deployment-example-20/objects/inputs/"
        "math-math-rft-reward-design-ADE-formal-n1-20000101T000000Z-00000b-"
        "c001-p005-t001-operator-test-attempt-001.json",
    ),
    "rft_math_n3_000014": (
        "rft-math.n3-000014.aime24",
        "runs/deployments/deployment-example-99/objects/inputs/"
        "math-math-rft-reward-design-ADE-formal-100-step-n3-20000101T000000Z-000014-"
        "c001-p002-t001-operator-test-attempt-001.json",
    ),
}
_INFERENCE_FIELDS = (
    "model_protocol",
    "thinking_budget",
    "decoding",
    "temperature",
    "top_p",
    "gpu_memory_utilization",
    "max_model_len",
    "max_new_tokens",
    "max_num_seqs",
)


def _render(request: dict[str, object], user: str) -> str:
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    dataset = request["datasets"][0]
    protocol = dataset["prompt_protocol"]
    template = native_tokenizer_prompt_metadata(request["checkpoint"])["chat_template"]
    return ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
    ).from_string(template).render(
        messages=[
            {"role": "system", "content": protocol["system_prompt"]["content"]},
            {"role": "user", "content": user},
        ],
        tools=None,
        add_generation_prompt=True,
    )


def check(project_root: Path, deployment: Path) -> dict[str, object]:
    root = project_root.resolve()
    compiled = EvaluationConfigCompiler(project_root=root).compile_file(
        root / "configs/evaluations/audit-best-generalization.yaml",
        deployment_config=deployment.resolve(),
    )
    results: dict[str, object] = {}
    for name, (unit_id, historical_relative) in _CASES.items():
        current = compiled.units[unit_id]["request"]
        historical_payload = json.loads(
            (root / historical_relative).read_text(encoding="utf-8")
        )
        historical = historical_payload["evaluation"]["request"]
        if current["datasets"][0]["prompt_protocol"] != historical["datasets"][0]["prompt_protocol"]:
            raise ValueError(f"{name}: prompt protocol differs from historical request")
        for field in _INFERENCE_FIELDS:
            if current.get(field) != historical.get(field):
                raise ValueError(f"{name}: request field changed: {field}")
        current_parser = current.get("reasoning_parser") or ""
        historical_parser = historical.get("reasoning_parser") or ""
        if current_parser != historical_parser:
            raise ValueError(f"{name}: request field changed: reasoning_parser")
        probe = f"ADE_PROMPT_PARITY_PROBE::{name}"
        current_render = _render(current, probe)
        historical_render = _render(historical, probe)
        if current_render != historical_render:
            raise ValueError(f"{name}: rendered chat prompt changed")
        results[name] = {
            "unit_id": unit_id,
            "historical_request": historical_relative,
            "prompt_protocol_equal": True,
            "inference_fields_equal": True,
            "rendered_prompt_equal": True,
        }
    for unit_id, unit in compiled.units.items():
        dataset = unit["request"]["datasets"][0]
        task_type = dataset["task_type"]
        protocol = dataset["protocol"]
        for field in (
            "user_prompt_builder", "reference_extractor", "prediction_extractor", "grader"
        ):
            if not str(protocol[field]).startswith(f"{task_type}."):
                raise ValueError(f"{unit_id}: non-canonical dataset {field}")
    return {
        "schema_version": 1,
        "status": "passed",
        "cases": results,
        "canonical_dataset_protocol_units": len(compiled.units),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--deployment",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    print(json.dumps(check(args.project_root, args.deployment), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
