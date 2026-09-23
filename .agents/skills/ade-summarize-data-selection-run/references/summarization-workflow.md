# Long-CoT SFT run summarization workflow

1. Preserve the complete parent RM and merge only the declared queue-head PM;
   do not reconstruct the Run from all Plans.
2. Compare plans using the configured independent validation metrics.
3. Treat realization statistics as explanatory evidence, not Ranking
   objectives or inherited constraints.
4. Preserve differences in trajectory-group policy, correctness filtering,
   reasoning validity, deduplication, contamination controls, truncation,
   difficulty, domain, language, and source mix.
5. Keep candidate-pool domains, selected-domain composition, and evaluation
   domains distinct. Treat same-domain specialization, cross-domain transfer,
   and mixed allocation as empirical alternatives; do not infer that
   other-domain data is harmful or that a same-domain-only allocation is
   optimal from one mixture comparison.
6. Identify whether the ranking rewards one narrow length bin, template,
   source, or near-duplicate family.
7. Preserve failed trials, competing selection mechanisms, contradictions, and
   incomparable metrics.
8. Attribute the PM's claims to its Plan and record them faithfully; Run
   Summary does not re-adjudicate Plan-local hypotheses. Preserve its current
   realization/hypothesis/portfolio conclusion values exactly.
9. State the strongest global claim that survives those limitations.
10. Select one next question whose answer would most affect the search.

Keep every statement traceable to `source_ids`. Ranking is decision support,
not proof.
