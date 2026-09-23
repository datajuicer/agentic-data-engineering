# Analyzer input guide

Read inputs in this order:

1. Scope and stage from `input/manifest.json`, `input/action.json`, and
   `input/task.json`. The root manifest is the exhaustive inventory for this
   Attempt: every other Agent-readable input has a digest and source binding.
2. The generated `input/experiment/EVIDENCE_GUIDE.md` material map.
3. `input/experiment/evidence-catalog.json` for actual artifact positions,
   prompt groups, response multiplicity, context bindings, identities, and
   coverage.
4. Executed reward/config, population statistics, complete telemetry, and the
   records needed for current hypotheses.
5. In synthesis only, the frozen Review packet and coverage.

`input/experiment/manifest.json` is the narrower Trial evidence allowlist; it
does not replace the root Attempt inventory. `review_design` has no Review
packet. `synthesis` additionally requires exactly `input/review/packet.json`
and `input/review/coverage.json`.

On retry, also require `input/retry.json` and the current stage's prior outputs
under `input/prior-delivery/`. Read every violation, treat prior delivery only
as a repair basis rather than new evidence, and modify only
`retry.json#feedback_paths`. Harness-internal Review history is not an Agent
input and must not appear under `input/`.

All counts, positions, group sizes, and field bindings are per-Trial facts from
the guide/catalog, never Skill constants.

For a Search Trial, declared `input/experiment/realization/` files contain the
final report, 32 direct source groups, candidate/direct group results, and
Judge manifest. The canonical P000 bootstrap baseline has no Builder
realization directory; use its declared reward, training, and evaluation
artifacts as the executed baseline facts. No accepted Plan, planning
comparator, expected observation, or objective comparison is an Analyzer
input.
