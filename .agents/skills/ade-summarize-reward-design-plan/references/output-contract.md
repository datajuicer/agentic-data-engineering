# Output contract

Write exactly `output/MEMORY.md`: a complete Plan memory preserving valid
parent knowledge and integrating accepted reward, rollout, validation, replay,
exploit, and failure findings. Preserve contradictions and uncertainty and
state the next unresolved question. Cite consumed snapshots as
`[[source:<source-id>]]`. Record the Plan hypothesis as `supported`, `rejected`,
or `inconclusive` without rewriting it. Include exactly one current block:

```text
## Trial Conclusions
Realization status: verified | deviated | unverified
Hypothesis result: supported | rejected | inconclusive
Portfolio result: new_best | not_new_best | not_comparable
```

Copy realization and portfolio from Harness artifacts; judge only hypothesis,
using the Harness-owned primary/secondary metric relations for its declared
expectations. Harness owns source binding and sidecars.
