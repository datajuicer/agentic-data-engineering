---
name: ade-analyze-curriculum-learning-trial
description: Design and synthesize evidence-bound analysis for one ADE Curriculum Learning RFT Trial in two Harness-controlled stages.
---

# Analyze Curriculum Learning Trial

Read only `input/`. Treat `experiment/realization/curriculum-schedule.json` as per-step data authority and use `training_context.sample_uid` to join rollout evidence. Reward, Group Credit, evaluator, model, optimizer, runtime, and checkpoint policy are fixed controls; recommendations may change only schedule logic.

For `review_design`, read the evidence catalog and write only `output/review-plan.json` using schema `ade.analysis_review_plan.v2`. Review `training_rollout` by complete groups at the catalog quota and select `offline_validation` with mode `all`; never enumerate or include online validation. The Harness expands these selectors, so do not copy record payloads into the plan. For `synthesis`, use the frozen packet/coverage and write only `output/analysis.md` and `output/findings.md`. Diagnose subject/level/coverage/revisit progression, scheduled-versus-consumed IDs, cohort outcome/reward/advantage, telemetry, and matched offline behavior. Do not claim curriculum improvement from distribution or reward mean alone.

Read [references/contract.md](references/contract.md) and run `python .agents/skills/ade-analyze-curriculum-learning-trial/scripts/validate_output.py output`.
