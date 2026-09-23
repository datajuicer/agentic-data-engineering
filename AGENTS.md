# Working on ADE

ADE (Agentic Data Engineering) supports agent-driven experiments on data selection,
reward design and curricula. Keep changes focused on a complete, usable workflow.

## Repository map

- `ade/`: Harness/Control, Agent runtime, Engine, tasks, evaluation and Memory.
- `configs/`, `prompts/`: experiment contracts, deployment examples and prompts.
- `.agents/skills/`: task-role Skills and operator-side supervision Skills.
- `examples/`: Operation Prompts and the optional scripted CPU demo.
- `docs/en/`: public usage and component guides.
- `scripts/`, `dataset/`, `requirements/`: installation and input preparation.
- `third_party/llamafactory/`, `third_party/verl/`: vendored training backends with their own licenses.

## Before editing

Read the README and the relevant component guide, source and call sites. Inspect
`git status --short` and preserve existing user changes. Resolve concrete conflicts
between documentation and implementation explicitly. Historical Runs and old tests
do not define the public contract.

Follow existing patterns and prefer standard-library solutions. Keep each change
within the requested scope; avoid speculative abstractions and compatibility layers.
Do not change benchmark splits, prompts, scoring, training budgets or checkpoint
selection semantics without explicit authorization. Modify a vendored backend only
when the active integration requires a verified change.

## Local verification

Run commands from the repository root. Use the single repository environment
`.unified-vllm-0.19.1-verl-venv` for development, tests, the CPU demo and all Runs.
Install it with `bash scripts/recreate_unified_vllm_env.sh`; see
`docs/en/installation.md`. Do not create a separate `.venv` or use `uv sync`.
Compatible nodes sharing this checkout at the same absolute path use one installation.

Before a check, identify the failure it would detect and what would change if it
failed. Run only the focused check or workflow needed for the modified path.
Use locally available tests when appropriate; the public checkout does not include
the ADE test suite. Do not run the full suite unless requested. For pipeline behavior,
inspect actual state, receipts and artifacts; test counts do not establish readiness.

## Experiments and resources

Real experiments use an Operation Prompt plus an exact experiment/deployment config,
handed to an operator-side Agent using `.agents/skills/ade-supervise-run/SKILL.md`.
Standalone post-hoc matrices use `ade-supervise-generalization`. Code/documentation
work does not authorize starting either workflow.

Do not guess model/data paths, service identities or resource capacity. GPU work,
paid Agent calls, service startup and repairs require an explicit operation scope.
Never stop unrelated jobs, restart shared services or broaden resource use on your
own. Keep scientific decisions in ADE's task roles and accepted state in Control.

## Documentation and delivery

Use English for public documentation, examples and interfaces.
Keep the README, Quickstart and examples usable without chat history or private paths.
Do not add development-phase trackers, Gate histories or routine progress diaries.
Report what changed, the verification performed and any material remaining limits.

Keep credentials, `.env`, downloaded models/data and generated Runs/logs out of Git.
Preserve third-party notices. Use Git for tracked-file rollback; do not create routine
backup archives. Resolve exact deletion targets and ask before removing expensive
untracked data unless the user has already approved that scope.
