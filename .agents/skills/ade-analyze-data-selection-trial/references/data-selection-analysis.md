# Data-selection analysis background

Candidate-pool domains and evaluation domains describe different parts of the
experiment. The evaluation distribution defines the downstream objective, but
does not determine which training domains are useful. Same-domain records may
provide direct task alignment, while other domains may contribute transferable
reasoning, knowledge, representations, formatting, or instruction-following
capabilities. A mixed-domain selection is therefore a normal experiment, not a
contamination defect merely because its composition differs from evaluation.

Separate intrinsic supervision quality from downstream utility. Assess whether
a record is correct, complete, and learnable under its own task contract; do
not judge an otherwise valid record as defective because it lacks the target
evaluation domain's output form. Downstream validation measures the utility of
the selected mixture as a whole. Domain similarity may be a useful feature or
controlled variable, but is not a default eligibility rule.

Candidate quality sets the available learning signal: incorrect targets,
ambiguous instructions, formatting defects, duplicated patterns, difficulty
imbalance, and narrow content coverage can all limit optimization. The selected
subset changes both which gradients are presented and how often different
failure modes are reinforced.

Compare selected records with the candidate population using declared identity
relations and population statistics. Ask whether useful difficulty and content
were retained, whether noisy or misleading supervision was concentrated, and
whether omissions plausibly explain offline failures. Then check that account
against complete training telemetry: a smooth loss curve does not establish
good supervision, while unstable or saturated optimization can weaken a
selection explanation.

Offline behavior is the outcome surface. Prefer hypotheses that jointly explain
candidate composition, selected supervision, optimization response, and
offline successes or residual errors. Recommendations must change selection
logic over the fixed candidate pool, not the fixed trainer or evaluator.

Keep domain claims as narrow as the observed comparisons. A same-domain-only
selection outperforming one mixed baseline supports that allocation comparison,
not the conclusions that other-domain data is generally harmful or that a
same-domain-only allocation is optimal. Those stronger claims require
controlled evidence across relevant mixtures or selection mechanisms.
