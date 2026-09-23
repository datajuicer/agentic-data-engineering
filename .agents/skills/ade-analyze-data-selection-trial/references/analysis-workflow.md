# SFT data-selection analysis workflow

Stage A should distinguish three questions: what quality and diversity exist
in the candidate population; what the selection logic retained or omitted;
and what behavior remains after training. Candidate records are supporting
input for the first two questions, not a Review pool. Use the catalog's
explicit candidate relation to compare selected and candidate records rather
than text matching.

Use final realization—not Builder design prose—to establish selected IDs,
rows, composition, and Judge availability. Treat realization statistics as
explanatory evidence rather than constraints. Explain how these facts qualify
mechanism evidence, but leave the Plan's
`supported|rejected|inconclusive` verdict to the non-blind Plan Summarizer.

Describe candidate-pool composition, selected composition, and evaluation
domain separately. When designing Review rubrics for a mixed selection, assess
each record under its own task contract rather than applying the evaluation
domain's answer requirements to every record.

Form preliminary hypotheses from catalog facts, full-pool/selected statistics,
complete normalized training telemetry, and representative complete records.
Give each Review pool a concrete investigation purpose and observable rubrics,
then select `selected_examples` and `offline_validation` with `mode=all`.
Do not enumerate record IDs or create a candidate-data batch.

In Stage B, interpret every terminal Review row with its source evidence.
Connect candidate distribution and selection bias to supervision quality,
loss/learning-rate/epoch or step behavior, and complete offline validation.
Separate supported selection effects from fixed-control limitations and
alternative explanations. Provider reasoning is advisory, not an independent
fact. Distinguish an observed mixture-level validation result from a causal
claim about the utility of every included or excluded domain, and preserve
untested domain allocations as uncertainty.
