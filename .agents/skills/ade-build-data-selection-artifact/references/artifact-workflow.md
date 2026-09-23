# Long-CoT SFT selection artifact workflow

## Choose one controlled intervention

Read the current Plan snapshot first. For the first Trial, compare its frozen
seed Plan snapshots and follow the Harness-provided source artifact IDs. For a
later Trial, use the single preceding artifact ID in `input/action.json`.
State one falsifiable hypothesis and
change only the selection logic named by that hypothesis.

Read the real primary `selection.py` before implementing the change. When the
intervention starts from the parent's selected population, use the canonical
`selected_ids` in the primary `selection-result.json` as the realized parent
set; do not infer it by rerunning the parent selector. Source files exist only
in the Builder input workspace. Copy the necessary stable IDs or express
equivalent deterministic inheritance logic inside the delivered entrypoint so
that the committed `selection.py` has no runtime dependency on those files.
The Plan guides the intervention rather than acting as a hidden mechanical
acceptance rule; document any intentional departure from its inheritance
proposal in `design.md`.

The evaluation domain is a fixed outcome target, not an automatic candidate
eligibility rule. Mixed-domain training is a valid design. Implement a domain
quota, preference, or exclusion only when the accepted Plan declares it as part
of the intervention; do not derive a same-domain hard gate from the evaluation
configuration or task name.

Use the candidate inventory's tokenizer token counts. Preserve trajectory
group identity. Treat near-duplicate and truncation signals as evidence only
when declared. Do not invent source quality, contamination, or hidden labels.
Historical realization statistics are not execution constraints.

## Write the executable artifact

Define exactly:

```python
async def select_trajectories(candidate_inventory, select_size, judge):
    return ["stable-trajectory-id"]
```

`candidate_inventory` is a list of declared group objects and `select_size` is
the positive number of trajectory IDs the function must return. `judge` accepts
a list of request objects containing question text, response text, and a
complete canonical RFT-style `process_rubric` JSON string. It returns one
structured score evidence object per request with `scores_by_dimension`,
`projected_score`, and terminal `status`. The Engine may split the list into
bounded provider batches. Use these results as features in the selection
strategy; they are not select/reject commands.

Judge intrinsic supervision quality against each candidate's own task and
response contract. In a mixed pool, route or phrase rubrics so that an
otherwise valid record is not penalized merely for lacking the evaluation
domain's answer format. If the Plan uses estimated target-domain relevance,
keep that feature conceptually separate from intrinsic correctness and let
downstream validation determine the selected mixture's utility.

Every Judge rubric must be the complete canonical process-rubric declaration;
there is no selection-only binary rubric format. Combine dimension scores with
declared heuristics and deterministic ranking or bucketing logic as required by
the Plan. Do not mutate `candidate_inventory`.

When justified by the accepted Plan, combine structured metadata,
deterministic rule-extracted signals from authorized candidate content, and
Judge evidence through buckets, routing, lexicographic ranking, quotas,
coverage constraints, or other deterministic portfolio logic. Do not force
the strategy into a single scalar score. For every signal or objective used,
including quality, relevance, difficulty, length, domain composition, or
diversity, state its observable definition and the selection decision it
controls. Prefer the simplest strategy that faithfully implements the accepted
hypothesis, and do not introduce objectives or signals outside that Plan.

Use only keys actually present in the authorized candidate inventory. The
canonical group schema provides `candidate_id`, `problem_id`,
`trajectory_ids`, `token_count`, `domain`, `difficulty`, and `length_bin`;
the Engine additionally provides read-only `training_examples` for the
group when a Judge request needs the original question/response. Do not
mutate these examples. When enough authorized trajectories satisfy the Plan, return exactly
`select_size` complete-group IDs. Harness passes the same value resolved from
`task.select_size` in admission and Engine execution, and runs a bounded probe
against the declared inventory, so an empty selection, unknown ID, partial
group, duplicate, or wrong selection size consumes retry before Engine.

Engine resolves the returned IDs, requires unique authorized IDs and complete
trajectory groups, computes the realized statistics, and materializes the
training dataset. The Builder must not claim those checks have passed.

## Record the design

In `design.md`, explain the hypothesis, the executable intervention, the
expected offline observation, the failure signal, the controls held constant,
and implementation details. Keep the design consistent with `selection.py`.

Write only `selection.py` and `design.md` to `output/`.
Do not execute the selection function locally or create validation artifacts;
Harness performs the static and bounded inventory checks.

On a reflection round, inspect the materialized selection and Judge evidence.
Make the smallest artifact change justified by an actual artifact issue. Do
not change the accepted Plan, frozen pool/runtime binding, training controls,
or persisted evidence.
