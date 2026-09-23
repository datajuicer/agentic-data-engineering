# GRPO RFT reward-design planning workflow

## Diagnose the current reward

1. Separate reward movement from independent validation quality.
2. Inspect score bounds, mean, variance, saturation, non-finite values, and the
   fraction of rollout groups with zero reward variance.
3. Check whether within-group reward ordering matches response quality.
4. Audit representative high- and low-reward outputs for false positives,
   false negatives, length bias, format shortcuts, answer-copy behavior, empty
   extraction credit, and other exploit paths.
5. Compare online rollouts with offline replay and independent validation.

## Propose one GRPO experiment

1. Compare the proposed controlled variable and falsifiable claim with every
   accepted direction in `subject/PLAN_CATALOG.md`. Add a `Difference from
   Accepted Plans` section that names the material difference from each
   overlapping active or terminal Plan. If no material difference exists,
   choose another direction. Do not infer findings from an active Plan.
2. Choose one falsifiable direction (`new_direction`, `combine`, `revisit`, or
   `contradiction`) that the Artifact Builder can implement inside the fixed
   `reward.py` ABI. The direction may contain a coherent set of coupled reward
   changes; do not artificially restrict the Agent to changing exactly one
   field or variable.
   Training answer extraction, `TrainingOutcomeAdapter`, evaluation parsers,
   graders, ground truth, and checkpoint-selection rules are Task/Engine fixed
   controls. Never propose changing them in a reward-design Plan. If accepted
   evidence exposes an extraction or evaluator defect, preserve it as a
   measurement caveat and require the reward experiment to hold that behavior
   constant; an infrastructure fix requires a separately versioned evaluator
   or adapter and a new comparable baseline outside this Plan.
3. Define one credit defect, a falsifiable evidence hypothesis, success
   criteria, and observations that would refute it. Preserve hard score,
   missing-input, and non-finite constraints from the task. Do not preselect a
   scalar process reward or set exact dimensions, projections, combinations,
   rule thresholds, Review Labor rubric text or labels, group tie/abstention
   policy, or implementation branches.
   Preserve the fixed deployment-local Judge contract: the injected
   `llm_judge` is the sole available model capability. When
   `judge_enrichment.enabled` is true it is called exactly once for every valid
   rollout, including `outcome_score == 0`; when false it is neither declared
   nor called. The raw structured result is Engine-owned. The Builder may use
   selected fixed dimensions in a separate Artifact projection, leave row
   reward unchanged, or use evidence only in group credit. Evidence acquisition
   does not itself imply reward intervention. Do not introduce another switch,
   arbitrary evidence schema, free metadata, judge, or provider.
4. Predict how the change affects rollout-group discrimination and advantage
   signal, not only mean reward. When Group Credit is enabled, make the
   falsifiable chain specifically about assigned reward, its realized change
   from the outcome-only advantage counterfactual, direction relative to task
   success, and matched downstream validation. When it is disabled or absent,
   keep the intervention and claim on per-response `compute_score` behavior.
5. Name the primary exploit probe and an independent validation metric that
   cannot be optimized directly through the reward.
6. Require replay cases for correct, incorrect, malformed, empty, adversarial,
   and conflicting-answer outputs.
7. Record the risk of sparse reward, saturation, high variance, reward hacking,
   and divergence from evaluation semantics.
8. Present the plan as structured Markdown. Keep observed evidence in
   `Evidence and Diagnosis`, the falsifiable claim in `Falsifiable Hypothesis`,
   predicted measurements in `Expected Observations and Success Criteria`, and
   failure gates in `Replay and Rejection Criteria`.
9. Make the direction actionable without designing the implementation: state
   the credit defect, optimization direction, evidence hypothesis, hard
   guardrails, success and refutation observations, and the exact evidence,
   projection, per-row reward, and group-credit decisions delegated to the
   Artifact Builder.

Before delivery, explain how the Plan relation's primary source directly tests
this credit claim. Harness resolves its representative Trial as the hypothesis
comparator and binds the frozen portfolio target. Translate the claim into
only the applicable mechanical replay checks: maximum Judge fallbacks and
whether assigned reward, realized advantage, or advantage sign must change.
These checks diagnose realization; they do not replace downstream validation.

## Choose the Plan relation

Choose one relation before writing the decision:

- `new_direction` references no historical Plan and starts from the baseline;
- `revisit` or `contradiction` references exactly one completed Plan;
- `combine` references at least two completed Plans.

Use fully qualified `<run>/cNNN/pNNN` SubjectRefs. Never reference an active, failed, proposed, or
unknown Plan. ADE Runtime, not the Coordinator, resolves and freezes the
current best artifact for every source Plan.

Choose one plan only. Do not implement reward code or schedule Engine work.
