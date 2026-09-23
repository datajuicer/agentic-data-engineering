# GRPO RFT run summarization workflow

1. Preserve the complete parent RM and merge only the declared queue-head PM;
   do not reconstruct the Run from all Plans.
2. Compare plans using evaluation agreement, exploit resistance, rollout-group
   discrimination, reward variance, and stability rather than mean reward.
3. Normalize differences in score bounds, evidence availability and projection,
   rollout-group size, sample count, training step, and evaluation coverage.
4. Preserve zero-variance groups, saturation, non-finite values, length or
   format bias, high-reward false positives, low-reward false negatives, and
   replay contradictions.
5. Identify reward components or verifier edge cases that dominate the ranking.
   Treat extraction/equivalence, `TrainingOutcomeAdapter`, evaluation
   parsers/graders, ground truth, and checkpoint selection as given
   Task/Engine fixed controls. Preserve defects as measurement caveats; never
   turn their repair into reward-search guidance.
6. Separate evidence about the reward signal from evidence about final model
   quality.
   Separately preserve Engine-owned raw evidence, Artifact projection/pre-group
   reward, final or assigned reward, realized advantage versus the outcome-only
   counterfactual, assignment direction when enabled, and matched downstream
   validation. Do not use reward density, process score, non-flat group count,
   or non-zero advantage count as a substitute for model quality.
7. Attribute the PM's claims to its Plan and record them faithfully; Run
   Summary does not re-adjudicate Plan-local hypotheses. Preserve its current
   realization/hypothesis/portfolio conclusion values exactly.
8. State the strongest global GRPO claim that survives independent validation
   and exploit audits.
9. Select one next question whose answer would most affect reward-plan
   allocation.

Keep every statement traceable to `source_ids`. Ranking is decision support,
not proof.
