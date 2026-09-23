# Output contract

Write exactly `output/MEMORY.md`: a complete Plan memory that preserves valid
parent knowledge and incorporates accepted Trial outcomes and findings. Cover
direction and hypothesis, progress, findings, assessment, contradictions,
uncertainty, and the next unresolved question. For each precommitted claim,
record `supported`, `rejected`, or `inconclusive` without rewriting the
commitment. Include exactly one current `## Trial Conclusions` block:

```text
Realization status: verified | deviated | unverified
Hypothesis result: supported | rejected | inconclusive
Portfolio result: new_best | not_new_best | not_comparable
```

Copy realization and portfolio from Harness artifacts; judge only hypothesis,
using the Harness-owned primary/secondary metric relations for its declared
expectations. Cite declared sources as `[[source:<source-id>]]`. Harness owns
sidecars.
