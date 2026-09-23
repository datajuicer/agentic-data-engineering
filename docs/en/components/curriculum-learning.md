# Curriculum Learning task

[Components](README.md)

## Modifiable surface

The Builder changes `curriculum.py` and describes the schedule in `design.md`. The fixed pool supplies problem IDs and source order; the task fixes total steps and prompts per step. The required interface is:

```text
async def build_curriculum(candidate_inventory, total_steps, prompts_per_step, judge_batch)
```

It returns a list of steps, each a list of problem IDs. `canonical_step_lists` requires exactly the requested number of steps, exactly the requested number of unique IDs within each step, and IDs present in the supplied inventory. It canonicalizes within-step order by source order. Thus the proposal controls cohort membership over time, not arbitrary within-step row permutations; the same problem can recur across steps.

## Implementation flow

The role contract and compiler validate the source. The realizer executes it against frozen inventory with the configured Judge capability and records evidence. `materialize_scheduled_parquet` turns the accepted schedule into sequential VERL rows. The task's Engine binder connects that materialization to the RFT backend; frozen scheduling relies on the configured no-shuffle training contract. Reward/outcome controls remain task-owned rather than becoming a second generated reward proposal.

Formal analysis uses the configured semantic training intervals and rollout/evaluation evidence. Final realization, accepted code, schedule materialization and Engine artifacts are linked through Trial state and its Record. Read the accepted schedule alongside observed training evidence when diagnosing whether the intended curriculum actually ran.

## Baseline and usage

The formal Curriculum baseline trains independently from the configured base model without an initial-state reference. Its task-owned P000 generates a deterministic random schedule from the Run seed, then uses the fixed outcome reward for the configured 100 training steps. The schedule is materialized before training and backend data shuffle remains disabled. Evaluation, Curriculum-specific analysis and Memory publication use the normal Run lifecycle. See [operations](../operations.md).

Inspect the selected task in the full environment after [input preparation](../data-preparation.md):

```bash
ade experiment inspect configs/experiments/math-math-rft-curriculum-learning-baseline-formal.yaml --deployment configs/deployments/local.yaml --run-id curriculum-inspect
```

Complete the independent Curriculum baseline first, then inspect/import it as the Seed for Curriculum ADE. The baseline Operation Prompt uses `NONE` for both Initial State Reference and Initial State Frontier; ADE uses the completed Curriculum Run ID and frontier `bootstrap`.

Operation Prompt templates: [baseline](../../../examples/math-rft-curriculum-learning/baseline-operation.md) and [N=1 ADE](../../../examples/math-rft-curriculum-learning/ade-operation.md).

## Diagnose

Wrong step length, duplicate IDs in a step and unknown IDs are proposal errors. For ADE, missing or incompatible Seed evidence requires a completed compatible Curriculum baseline. If materialized rows disagree with observed training, inspect schedule, shuffle setting and backend binding before interpreting score changes. Pipeline completion can coexist with a rejected curriculum hypothesis.

## Implementation entry points

| Source | Responsibility |
| --- | --- |
| [ade/tasks/curriculum_learning/plugin.py](../../../ade/tasks/curriculum_learning/plugin.py) | Task hooks |
| [ade/tasks/curriculum_learning/schedule_contract.py](../../../ade/tasks/curriculum_learning/schedule_contract.py) | Source and step shape |
| [ade/tasks/curriculum_learning/realizer.py](../../../ade/tasks/curriculum_learning/realizer.py) | Schedule execution and evidence |
| [ade/tasks/curriculum_learning/materializer.py](../../../ade/tasks/curriculum_learning/materializer.py) | Sequential Parquet materialization |
| [ade/tasks/curriculum_learning/engine_binding.py](../../../ade/tasks/curriculum_learning/engine_binding.py) | RFT binding |
| [ade/tasks/curriculum_learning/baseline.py](../../../ade/tasks/curriculum_learning/baseline.py) | Deterministic random schedule baseline |
| [ade/tasks/curriculum_learning/role_contracts.py](../../../ade/tasks/curriculum_learning/role_contracts.py) | Role deliveries |
