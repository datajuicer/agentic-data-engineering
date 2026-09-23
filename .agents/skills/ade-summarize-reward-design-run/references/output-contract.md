# Output contract

Write exactly `output/MEMORY.md`: a complete Run memory preserving valid
parent knowledge and integrating the queue-head Plan result. Cover global
reward-design state, Plan assessments, rollout and replay findings,
contradictions, uncertainty, ranking limitations, remaining risks, and the next
global question. Cite consumed inputs as `[[source:<source-id>]]`.
Harness owns source binding and all sidecars.

Keep fixed extraction/evaluation defects as measurement caveats. Do not
recommend modifying an extractor, adapter, parser, grader, ground truth, or
checkpoint-selection rule within the reward-design Run.

Include exactly one current `## Trial Conclusions` block. Its `Realization
status`, `Hypothesis result`, and `Portfolio result` must match the queue-head
PM byte-for-value.
