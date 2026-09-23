---
name: ade-summarize-curriculum-learning-run
description: Merge one queue-head Curriculum Learning Plan Memory into a complete ADE Run Memory without changing Control facts.
---

# Summarize Curriculum Learning Run

Read only `input/`. Write exactly one complete `output/MEMORY.md` that preserves source links, accepted schedule findings, contradictions, ranking limitations, and the next global schedule question. Do not re-adjudicate the Plan, recompute ranking, or promote fixed reward/model/evaluation/runtime observations into Curriculum actions. Read [references/contract.md](references/contract.md) and run the packaged validator.

Only Task/Harness-declared constraints are fixed controls. A property or value observed in this or a prior realization remains evidence; do not promote it in Memory into a future constraint, target, quota, or matching requirement merely for historical comparability. Record any comparability concern as a confounder or measurement need. A future Plan may constrain that property only through its own explicit falsifiable hypothesis, with the Coordinator treating the choice as current experimental design rather than inherited control.
