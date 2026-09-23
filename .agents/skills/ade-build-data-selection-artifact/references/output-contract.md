# Semantic output contract

Write exactly `selection.py` and `design.md` under `output/`.

`selection.py` defines deterministic async
`select_trajectories(candidate_inventory, select_size, judge)` and returns
exactly `select_size` stable trajectory IDs. `judge` accepts a list of
`question`/`response`/`rubric` request objects and returns one structured
evaluation evidence object per request. The Engine may split the request
list into bounded provider batches. Judge evidence uses the canonical RFT
`process_rubric` contract and never returns a direct select/reject decision.

The Engine-provided candidate groups may include read-only
`training_examples` containing the original sample conversations for Judge
requests. The artifact may use them but must not mutate them or assume that
the accepted selection artifact stores this enriched view.

The accepted artifact does not include the Builder's `input/sources/` tree.
Any parent identities needed by the strategy must be encoded inside the
entrypoint, and `selection.py` must not read or import parent files at runtime.

An unavailable or fallback evidence object is not a genuine zero score. The
selection design must declare how it is ranked, bucketed, or backfilled, and
must still return exactly `select_size` IDs after row failures.

Use this expected `design.md` outline as writing guidance, not a mechanically
enforced schema:

1. Objective and Source Inheritance
2. Implementation Design
3. Fixed Controls and Mutation Boundary
4. Development Probes and Observations
5. Fallbacks, Risks, and Known Limitations

Do not write identity, manifest, provenance, delivery, digest, selection-result,
or self-check sidecars. Harness creates those facts and independently executes
the selection against the fixed pool.
