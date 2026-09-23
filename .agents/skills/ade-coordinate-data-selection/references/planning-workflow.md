# Long-CoT SFT data-selection planning workflow

## Establish the selection unit

Treat a complete problem and its associated reasoning trajectories as a
trajectory group. Do not plan around unstable row positions. Decide whether
the experiment keeps one trajectory per problem or an intact multi-trajectory
group, and make that controlled variable explicit.

## Diagnose the current portfolio

Treat the candidate-pool domain composition and the evaluation domain as
distinct facts. The evaluation distribution defines the downstream objective;
it does not by itself make same-domain records eligible or other-domain records
ineligible. A mixed-domain training selection is valid because records from
other domains may contribute transferable reasoning, knowledge, representation,
formatting, or instruction-following capabilities. Treat same-domain
specialization, cross-domain transfer, and mixed allocation as competing
hypotheses whose value is established by downstream evidence, not by domain
labels alone.

1. Compare accepted trials using the configured downstream validation metrics.
   Treat realization statistics as explanatory evidence.
2. Inspect correctness, reasoning validity, tokenizer-derived length,
   difficulty, domain, language, source, and trajectory multiplicity.
3. Check exact duplicates, near-duplicate problems or trajectories,
   benchmark contamination, train/evaluation overlap, and truncation.
4. Identify coverage holes and confounders such as selecting only the longest
   correct trajectories or changing length and difficulty together.
5. Read the declared `budget` view and state how much Plan and Trial budget
   remains. Treat this as a search-policy constraint: when substantial Plan
   budget remains, preserve room for exploration; when Plan budget is scarce or
   evidence is strong, targeted exploitation may be appropriate. No relation
   kind is a preset default; justify the choice from the evidence and budget.

## Propose one experiment

1. Compare the controlled variable and hypothesis with every accepted direction
   in `subject/PLAN_CATALOG.md`. Add a `Difference from Accepted Plans` section
   naming the material difference from each overlapping active or terminal
   Plan. If no material difference exists, choose another direction. Do not
   infer experimental findings from an active Plan.
2. Choose one falsifiable direction (`new_direction`, `combine`, `revisit`, or
   `contradiction`) that is meaningfully different from accepted Plans. The
   direction may require a coherent set of coupled selection changes; do not
   artificially restrict the Agent to changing exactly one field or variable.
3. Define the trajectory-group policy, inclusion/exclusion rule, deduplication
   scope, and deterministic tie-break for that direction.
4. Describe composition preferences only when they are part of the declared
   intervention. Do not derive them from a prior realization.
5. For every selection signal or objective in the proposal, including quality,
   relevance, difficulty, length, domain composition, or diversity, define the
   observable evidence, the selection decision it is intended to control, the
   expected portfolio effect, and the downstream observation that would reject
   its use. The Plan may combine structured metadata, deterministic features
   extracted from authorized candidate content, and Judge evidence when their
   joint role forms one coherent falsifiable intervention; do not prescribe a
   single aggregate score or require signals that the hypothesis does not need.
6. State the controlled variable and variables held fixed.
7. Predict an independent downstream validation observation.
8. Record contamination, teacher-quality, diversity, and truncation risks.

Before delivery, explain the hypothesis against the relation's primary source
from the declared profile-matched baseline/source-eligible results and state
why it tests this specific intervention. Harness resolves its representative
Trial and freezes the portfolio target. Do not encode realization statistics
as structured requirements. Partial Judge
fallback remains realization telemetry governed by the selector's
ranking/backfill policy.

A hard domain gate is a selection intervention, not a default consequence of
the evaluation configuration. Justify it against the available mixed-pool
evidence and state what downstream result would reject it. A same-domain-only
result beating one mixed baseline supports that observed allocation comparison;
it does not establish that every other-domain record has negative utility or
that the same-domain-only allocation is optimal.

## Choose the Plan relation

Choose the relation after comparing the Plan Catalog, evidence, and remaining
budget. Use the relation to express the intended search posture, not to encode
an allocation:

- favor `new_direction` when the remaining Plan budget can support a genuinely
  different hypothesis and the catalog does not cover it;
- use `combine` only when at least two completed Plans provide complementary
  evidence whose joint test is worth the extra complexity;
- use `revisit` for targeted exploitation of one completed Plan when its
  evidence justifies refinement under the remaining Trial budget;
- use `contradiction` when one completed Plan makes a falsifiable claim that
  the new evidence or budget-aware objective should directly challenge.

This is a judgment rule, not an alternating schedule: do not force relation
diversity when the evidence points elsewhere, but explicitly explain why the
chosen relation is appropriate for the current remaining budget. The Harness
still enforces relation cardinality and completed-Plan eligibility.

Choose one relation before writing the decision:

- `new_direction` references no historical Plan and starts from the baseline;
- `revisit` or `contradiction` references exactly one completed Plan;
- `combine` references at least two completed Plans.

Use fully qualified `<run>/cNNN/pNNN` SubjectRefs. Do not reference an active,
failed, proposed, or unknown Plan. Do not select or copy artifact IDs; ADE
Runtime resolves and freezes each source Plan's current best artifact.

Choose one plan only. Do not schedule trials or Engine work. Never use
operator-test outcomes to choose the plan.
