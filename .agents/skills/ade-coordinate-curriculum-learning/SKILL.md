---
name: ade-coordinate-curriculum-learning
description: Propose one falsifiable static problem-schedule intervention for an ADE Curriculum Learning RFT Plan. Use only for the isolated Curriculum Coordinator input package.
---

# Coordinate Curriculum Learning

Read only `input/`. Propose exactly one schedule hypothesis; the only mutable axis is which authorized problem IDs occur at each frozen RL step. Model, source rows, reward, Group Credit, optimizer, rollout, evaluation, checkpoint policy, runtime, and root seed are fixed controls.

Control authority belongs to the Task and Harness: only constraints they explicitly declare are inherited fixed controls. Any property or value observed in a prior realization or recommended by prior Memory remains evidence. Do not preserve, match, bound, or optimize it merely because a previous Run had it. Introduce an additional constraint only when the current Plan's explicit falsifiable hypothesis requires it, and identify that choice as part of the current schedule intervention or experimental design, not as an inherited control.

The hypothesis may use static inventory metadata and may declare whether schedule realization should use the injected Judge capability. Judge evidence can describe difficulty, prerequisites, solution quality, or clusters, but it does not choose the schedule and cannot use training progress or rollout results. Do not prescribe a reward or training-parameter change.

Read [references/contract.md](references/contract.md), then write exactly `output/plan.md` and `output/decision.json`. Include one directional hypothesis, expected and rejecting observations, fixed controls, and `design.judge_enrichment` as a boolean use decision. Run `python .agents/skills/ade-coordinate-curriculum-learning/scripts/validate_output.py output` before delivery.
