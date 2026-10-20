# ADE research homepage

Open [index.html](index.html) in a browser. The complete directory is a buildless,
offline-capable research homepage and can be served by any static host. The
original directory name is retained for existing links.

The homepage presents the paper's method, Table 3 results across four tasks,
research insights, three interactive cases, and a copyable citation. Figures come
from the supplied paper sources. The compact layout uses local assets and system
fonts; it makes no automatic network requests and starts no ADE Runs.

## Interactive cases

| Case | Selected Plan | Final in-loop ranking | Online curve |
| --- | --- | --- | --- |
| Math data selection | C3/P2 | AIME24 Pass@32, then Average@32 | Pass@8 |
| Code data selection | C1/P3 | LiveCodeBench Pass@3, then Average@3 | Pass@1 |
| Math reward design | C1/P3 | MATH-S Average@3, then Pass@3 | Average@1 |

Each case contains one recorded C=3 Run with 15 terminal Plans. The explorer
supports case switching, result-completion replay, Plan inspection, source-curve
comparisons, direct follow-up highlighting and JSON downloads. Use
`?case=math#research`, `?case=code#research`, or `?case=reward#research` for direct
links. The browser back/forward buttons restore the selected case.

- Columns indicate local Plan order; `#` indicates global proposal order.
- Relations retain their explicit types: revisit, combine or contradiction.
  Separate curated diagnostic knowledge links are available for the Code case.
- A selected strategy is not necessarily a supported hypothesis.
- Reward Design C2/P4 failed before evaluation. Its score is null, not zero.
  C2/P5 has a score but no accepted agent-authored conclusion.
- SFT loss and VERL actor loss are labeled separately. Checkpoint markers use
  recorded epochs or RL steps, independently of the last training point.
- Operator test is separate from strategy selection. It is one benchmark, not
  the four-benchmark held-out Macro Avg.
- Missing secondary baseline metrics or unavailable traces remain absent.

The homepage table reports the paper's **Best Run per capacity**, selected by
in-loop scores. The progress figure reports **means across three Runs**. Each
interactive case is **one Run**; these statistics are not interchangeable.
Curriculum Learning appears in the paper results, but has no interactive case.

## Regenerate snapshots

The public snapshots contain only display labels, scores, curve points and
relationships. Original Run identities, deployment paths, credentials and raw
model responses stay outside the site. Exporters read an already restored archive;
they do not download data, execute archived code or run training.

From the repository root, with Node.js 22+ and `unzip` installed:

```bash
node scripts/export_research_demo.mjs <RESTORED_ARCHIVE_ROOT>
node scripts/export_paper_results.mjs <EXTRACTED_PAPER_SOURCE>
```

The restored root contains `repositories/agentic-data-engineering/`. The paper
source contains `section/experiments.tex`. The first command validates accepted
Plan scores and relation chronology, then exports the Math and Reward snapshots
to `additional-cases.js`. The second extracts Table 3's four tasks, four methods
and six benchmark columns into `results-data.js`, retaining the printed precision.
`case-copy.js` contains the existing curated Code narrative.

The Code snapshot remains reproducible with the original exporter and original
local deployment:

```bash
.unified-vllm-0.19.1-verl-venv/bin/python scripts/export_code_research_demo.py \
  --case-json <LOCAL_CASE_JSON> \
  --output examples/code-research-demo/case-data.js
```

For browser verification, cover all three cases, both curve modes, failed and
missing-data states, replay/reset, source navigation, downloads, direct links,
the four results tabs, and narrow screens. The files also work over `file://`;
when clipboard access is unavailable, the citation button selects the text for
manual copying.
