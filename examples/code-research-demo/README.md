# Code research demo

<!-- [Open live demo](https://ade-code-research.ruomengd.chatgpt.site) -->

Open [index.html](index.html) in a browser. The page works directly from the checkout, offline, with no installation or build step. The complete `code-research-demo/` directory can also be served by any static web server.

- Follow one code data-selection run with 15 Plans across three Coordinators. Replay follows result completion order; the card number shows proposal order.
- Switch between recorded strategy ancestry and knowledge transfer. Select a Plan to inspect its question, intervention, findings and hypothesis assessment.
- Inspect recorded training loss and online validation against actual training steps; compare with the Plan's parents. Curves end at early stopping. The selected checkpoint is marked independently of the training endpoint.
- Baseline is a final-score reference. Online validation uses K=1; final in-loop validation and operator test use K=3. Operator test is kept separate from strategy selection and is not the four-benchmark generalization average.
- Download the displayed numeric data.

The public snapshot contains only short Plan labels, scores, curve points and relationships. It includes no source paths, dates, original Run IDs, deployment identities, credentials or raw model responses. The page makes no network requests and does not start an ADE Run.

Maintainers with the original local case and artifacts can regenerate the numeric snapshot from the repository root:

```bash
.unified-vllm-0.19.1-verl-venv/bin/python scripts/export_code_research_demo.py \
  --case-json <LOCAL_CASE_JSON> \
  --output examples/code-research-demo/case-data.js
```
