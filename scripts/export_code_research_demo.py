#!/usr/bin/env python3
"""Export numeric, anonymous curve data for the code research demo.

The input case and Run artifacts remain local. Only the explicit fields below
are exported; raw logs, paths, identities, dates and model responses are omitted.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def short(key: str) -> str:
    lane, plan = key.split("/")
    return f"C{int(lane[1:])}/P{int(plan[1:])}"


def number(value) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite metric")
    return round(value, 8)


def operator_score(run, key: str, objects: Path):
    record = next((r for r in run["operator_evaluations"] if r["target_id"].endswith(key)), None)
    if not record or not record.get("result_ref"):
        return None
    uri = record["result_ref"].replace("/result.json", "/raw/units/operator_test.json")
    path = (objects / uri.removeprefix("engine://")).resolve()
    if not path.is_relative_to(objects.resolve()) or not path.is_file():
        return None
    results = read(path).get("results", [])
    if not results:
        return None
    result = results[0]
    metrics = result["metrics"]
    return {"pass": number(metrics["pass@3"]), "average": number(metrics["avg@3"])}


def export(case_path: Path) -> dict:
    case = read(case_path)
    control = Path(case["case"]["source_control_path"])
    run = read(control / "run.json")
    deployment = control.parent.parent
    objects = deployment / "objects"
    artifact_root = deployment / "engine-work/trial_artifacts" / run["run_id"]
    completion = {p["plan_key"]: i + 1 for i, p in enumerate(sorted(case["plans"], key=lambda p: p["completion_revision"]))}
    proposal = {p["plan_key"]: i + 1 for i, p in enumerate(sorted(case["plans"], key=lambda p: p["decision_revision"]))}
    baseline = next(t for t in run["trials"] if t["coordinator_id"] == "c000")
    baseline_key = f"/c000/p000/{baseline['trial_id']}"
    baseline_run = read(control.parent / run["bootstrap"]["reference_run_id"] / "run.json")
    result = {
        "schema": 1,
        "selected": short(case["case"]["selected_plan_key"]),
        "baseline": {"pass": number(baseline["offline_score"]), "average": number(baseline["offline_secondary_score"]), "operator": operator_score(baseline_run, baseline_key, objects)},
        "plans": [],
        "relations": [],
    }
    for p in case["plans"]:
        key = p["plan_key"]
        source = artifact_root / key / "t001/attempt-001"
        actual = read(control / p["source_plan_record"])
        if abs(actual["best_score"] - p["validation_score"]) > 1e-10:
            raise ValueError("case score differs from accepted Plan")
        rows = [json.loads(line) for line in (source / "checkpoints/trainer_log.jsonl").read_text().splitlines() if line.strip()]
        training = [{"step": int(r["current_steps"]), "loss": number(r["loss"]), "epoch": number(r["epoch"])} for r in rows if "loss" in r]
        online = []
        for f in sorted((source / "eval_results").glob("online-validation-epoch-*.json")):
            payload = read(f)
            epoch = int(payload["artifact_position"]["value"])
            # Use an observed training-log position, never an inferred epoch/step conversion.
            matching = [r for r in rows if abs(float(r["epoch"]) - epoch) < 1e-6]
            if not matching:
                raise ValueError("no recorded step for evaluated epoch")
            online.append({"epoch": epoch, "step": int(matching[-1]["current_steps"]), "score": number(payload["ranking_score"])})
        manifest = read(source / "manifest.json")
        selected_epoch = int(manifest["selected_checkpoint_id"].rsplit("-", 1)[1])
        result["plans"].append({
            "id": short(key), "lane": int(key.split('/')[0][1:]), "plan": int(key.split('/')[1][1:]),
            "proposalOrder": proposal[key], "completionOrder": completion[key],
            "pass": number(p["validation_score"]), "average": number(p["validation_secondary"]),
            "hypothesis": {"Supported": "supported", "Challenged": "rejected", "Unresolved": "inconclusive"}[p["adjudication"]],
            "operator": operator_score(run, f"/{key}/t001", objects),
            "selectedEpoch": selected_epoch, "training": training, "online": online,
        })
    for r in case["relations"]:
        if r["source_plan_key"] and r["explicitness"] == "explicit" and r["available_before_decision"] and r["main_text_eligible"]:
            result["relations"].append({"source": short(r["source_plan_key"]), "target": short(r["target_plan_key"]), "kind": r["relation_type"]})
    result["knowledge"] = [{"id": k["id"], "sources": [short(s) for s in k["sources"]], "target": short(k["target"])} for k in case["case"]["figure"]["knowledge_transfers"]]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-json", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("examples/code-research-demo/case-data.js"))
    args = parser.parse_args()
    data = export(args.case_json)
    args.output.write_text("// Public numeric snapshot. No runtime connection is required.\nwindow.ADE_CASE = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"Exported {len(data['plans'])} Plans, {sum(len(p['training']) for p in data['plans'])} training points and {sum(len(p['online']) for p in data['plans'])} validation points.")


if __name__ == "__main__":
    main()
