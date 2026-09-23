# Long-CoT SFT plan summarization workflow

1. Preserve the complete parent Memory, then read the immutable Plan to recover
   each precommitted claim, expected observation, and rejecting observation.
2. Read only the current Trial's objective outcome and blind Analyzer findings;
   keep their independent interpretation intact.
3. Summarize realization statistics when available. Treat them as explanatory
   evidence, not inherited constraints or Ranking objectives.
4. Preserve evidence about correctness, reasoning validity, duplicate rate,
   contamination, truncation, and group integrity.
5. Separate independent validation gains from training loss changes or teacher
   style imitation.
6. State whether support is broad or isolated to one trajectory length,
   difficulty, domain, or near-duplicate family.
7. Preserve confounders when effective epochs or multiple selection variables
   changed together.
8. Keep candidate-pool domains, selected-domain composition, and evaluation
   domains distinct. A mixture-level result does not establish the causal
   utility of every included or excluded domain; preserve materially different
   untested allocations as uncertainty.
9. Keep realization, hypothesis, and portfolio conclusions separate. Copy
   realization status and portfolio result from Harness artifacts. Use the
   frozen hypothesis comparator, objective primary/secondary metric relations,
   realized behavior, and blind findings to adjudicate
   `supported|rejected|inconclusive`; do not retrofit the hypothesis, recompute
   metric direction, or let aggregate `improved` hide a regressed axis.
10. State the narrowest supported claim and one next question that would most
   change the plan decision.

Every assertion must be traceable to `source_ids`. Do not create a new
experimental finding.
