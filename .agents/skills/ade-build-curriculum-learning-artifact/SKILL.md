---
name: ade-build-curriculum-learning-artifact
description: Build one restricted curriculum.py schedule policy and design.md from an accepted ADE Curriculum Learning Plan.
---

# Build Curriculum Learning Artifact

Read only declared `input/` files. Use the frozen eligible candidate inventory and resolved `total_steps/prompts_per_step`; inherit the primary parent schedule when declared. Deliver only `curriculum.py` and `design.md`.

`curriculum.py` must expose one async `build_curriculum(candidate_inventory, total_steps, prompts_per_step, judge_batch)`. Return exactly one list of authorized problem IDs per step, with exactly `prompts_per_step` unique IDs in each step; cross-step reuse is allowed. Do not read files, environment, network, time, randomness, Engine/Review state, or modify problem content. The only external capability is an awaited bounded `judge_batch([{question,response,rubric}, ...])`; use it only when the accepted Plan needs static enrichment. `rubric` is the complete canonical process-rubric JSON string documented in the contract, not free-form instructions or a job envelope. Judge returns structured score evidence, not a schedule choice. Harness freezes evidence and the canonical schedule.

Read [references/contract.md](references/contract.md). Explain inheritance, schedule logic, Judge use, and fixed controls in `design.md`. Run `python .agents/skills/ade-build-curriculum-learning-artifact/scripts/validate_output.py output` before delivery.

When `input/reflection/` is declared, inspect the realized schedule and evidence
as implementation feedback. Add exactly one `Reflection decision: finalize` or
`Reflection decision: revise` line to `design.md`; revise only to make the
artifact faithfully implement the accepted Plan, not to satisfy an observed
distribution or effect target.
