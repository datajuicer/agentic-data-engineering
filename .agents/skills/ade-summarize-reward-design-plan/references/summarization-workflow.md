# GRPO RFT plan summarization workflow

Preserve the complete parent Memory, then read the immutable Plan and current
Trial inputs in full. Treat the blind Analyzer's findings as an independent
interpretation of evidence, not as Plan state authority.

1. Reconstruct each precommitted claim and its expected and rejecting
   observations, plus the reward components, score bounds, fixed extractor,
   missing-input policy, and primary exploit hypothesis. Keep training and
   evaluation extraction/parser/grader behavior as Task/Engine fixed controls,
   not mutable reward design.
2. Order accepted findings by source ID and distinguish baseline, ablation,
   replay, replication, and failed trials.
3. Summarize Engine-owned raw evidence, Artifact projection/pre-group reward,
   final or assigned reward, actual realized advantage, outcome-only
   counterfactual, reward variance, saturation, and within-group discrimination.
4. Preserve high-reward false positives, low-reward false negatives, length or
   format bias, exploit evidence, and non-finite failures.
5. Compare online behavior with offline replay and independent validation.
6. Do not treat higher mean reward as model improvement when evaluation
   agreement is absent.
   Do not treat denser reward, process score, more non-flat groups, or more
   non-zero advantages as support. For enabled Group Credit record whether the assigned reward changed
   realized advantage relative to the outcome-only counterfactual, whether the
   direction matched task quality, and whether matched downstream validation
   improved.
7. State whether support generalizes across tasks and rollout groups or depends
   on one verifier edge case.
8. Keep realization, hypothesis, and portfolio conclusions separate. Copy
   realization status and portfolio result from Harness artifacts. Use frozen
   hypothesis comparator, objective primary/secondary metric relations,
   realized replay, and blind findings to adjudicate
   `supported|rejected|inconclusive`; do not retrofit the hypothesis, recompute
   metric direction, or let aggregate `improved` hide a regressed axis.
9. State the narrowest supported claim and the next reward-side probe that best
   separates competing reward hypotheses and is implementable inside
   `reward.py`. Preserve extraction/evaluator defects as measurement caveats;
   do not turn their repair into the next Plan intervention. A fix requires a
   separately versioned infrastructure run and a new comparable baseline.

Every assertion must be traceable to `source_ids`.
