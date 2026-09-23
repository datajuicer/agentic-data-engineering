# Quickstart: give an experiment to a supervising Agent

[Walkthrough](walkthrough.md) | [Installation](installation.md) | [Optional CPU demo](cpu-demo.md)

ADE's public execution entry is **an Operation Prompt + an exact experiment config,
handed to one supervising Agent**. That Agent uses
[`ade-supervise-run`](../../.agents/skills/ade-supervise-run/SKILL.md) to perform
admission, start ADE, monitor durable state, recover operational failures and verify
terminal cleanup. ADE itself runs the Coordinator, Builder, Analyzer and Summarizers.

Choose the task family first; each provides its own baseline, N=1 ADE and N=3 ADE prompts:

| Task family | Operation Prompts |
| --- | --- |
| Math SFT | [Baseline](../../examples/math-sft/baseline-operation.md) · [ADE N=1](../../examples/math-sft/ade-operation.md) · [ADE N=3](../../examples/math-sft/ade-n3-operation.md) |
| Code SFT | [Baseline](../../examples/code-sft/baseline-operation.md) · [ADE N=1](../../examples/code-sft/ade-operation.md) · [ADE N=3](../../examples/code-sft/ade-n3-operation.md) |
| Reward Design | [Baseline](../../examples/math-rft-reward-design/baseline-operation.md) · [ADE N=1](../../examples/math-rft-reward-design/ade-operation.md) · [ADE N=3](../../examples/math-rft-reward-design/ade-n3-operation.md) |
| Curriculum Learning | [Baseline](../../examples/math-rft-curriculum-learning/baseline-operation.md) · [ADE N=1](../../examples/math-rft-curriculum-learning/ade-operation.md) · [ADE N=3](../../examples/math-rft-curriculum-learning/ade-n3-operation.md) |

The steps below use math SFT with N=1. For another family, copy its own prompts and prepare the
inputs and resource allocation selected by its exact config. Every baseline starts
independently; ADE imports the completed baseline from the same family.

## 1. Prepare the runtime and inputs

From the checkout, follow [installation](installation.md) and install the full runtime:

```bash
bash scripts/recreate_unified_vllm_env.sh
```

Prepare the [models and datasets](data-preparation.md) referenced by your selected task, configure the [deployment](deployment.md), and fill the four fields in [`.env`](deployment.md#environment-variables). Model choices, evaluation datasets, search budgets and GPU requirements come from the selected configuration.

The operator installs the environment on participating nodes and starts the assigned
[Ray head/workers](ray-cluster.md). The supervising Agent checks the selected cluster;
it does not infer node assignments or start an unrelated cluster.

## 2. Fill the baseline Operation Prompt

Copy [baseline-operation.md](../../examples/math-sft/baseline-operation.md) to a local
location such as `runs/operation-prompts/baseline.md`. Fill the absolute checkout and
full-environment paths, Ray head IP and explicit resource/service authorization.
Its exact experiment is already selected:

```text
configs/experiments/openthoughts-math-sft-data-selection-baseline-formal.yaml
```

Keep `Initial State Reference: NONE`, `Initial State Frontier: NONE` and `Run mode: NEW`.
The [walkthrough](walkthrough.md) explains every field, authorization scope and expected
outputs. An unfilled template does not authorize a run.

## 3. Start the supervising Agent manually (Codex CLI)

Open a terminal on the configured **Ray head host**. Install and authenticate Codex CLI,
install `tmux`, and make the full environment, inputs and `.env` accessible there.
Complete the Operation Prompt above first. The supervising Agent is the Codex session
started below; `ade run start` is the ADE process it will manage.

Enter your checkout's absolute path and create a persistent terminal:

```bash
cd /absolute/path/to/ade-opensource
tmux new-session -s ade-baseline -c "$PWD"
```

Run the following **inside tmux**:

```bash
source .unified-vllm-0.19.1-verl-venv/bin/activate
codex login status
# If not authenticated, run codex login and complete its interactive instructions.
codex -m gpt-5.6-luna -c 'model_reasoning_effort="medium"' --sandbox danger-full-access --ask-for-approval never \
  'Use $ade-supervise-run. Read .agents/skills/ade-supervise-run/SKILL.md and its complete workflow. Execute the completed, authorized Operation Prompt in runs/operation-prompts/baseline.md. After admission, start the baseline exactly once, retain the Supervisor process handle, and supervise through terminal acceptance and resource cleanup. Do not stop after launching the process.'
```

This starts the session and submits its initial instruction immediately. Single quotes
preserve the literal `$ade-supervise-run`. The permission options allow deployment actions;
the Operation Prompt bounds GPU, service and recovery authorization. The outer supervising Codex uses
`gpt-5.6-luna`; the experiment YAML still selects internal research-role models.

Expect the Agent to inspect configuration and run preflight, then report the actual Run ID
and `<run-root>/monitor.md`. It owns the foreground ADE Supervisor and follows the Skill
through monitoring and cleanup. A quiet terminal does not imply stopped work. Do not run
`ade run start` manually or open a second Agent for the same Run.

Press `Ctrl-b`, then `d`, to detach. After an SSH disconnect, log back into the head and run:

```bash
tmux attach-session -t ade-baseline
```

If that session already exists, attach first instead of creating another. If Codex exited
but ADE may still be running, retain the actual Run ID and use an `EXISTING` handoff as
explained in the [walkthrough](walkthrough.md); do not resubmit a `NEW` prompt. tmux keeps
terminal processes alive; the Agent remains responsible for continuous supervision.

## 4. Submit the ADE operation

After baseline completion and cleanup, copy
[ade-operation.md](../../examples/math-sft/ade-operation.md). Fill the same
local facts and authorization, and set `Initial State Reference` to the **actual
completed baseline Run ID**. Keep `Initial State Frontier: bootstrap`.
Hand this prompt to a supervising Agent for the separate ADE Run. It checks Seed
compatibility and uses the fixed
[`ADE-formal-n1.yaml`](../../configs/experiments/openthoughts-math-sft-data-selection-ADE-formal-n1.yaml):
the Coordinator, Plan and Trial counts come from that configuration. N counts Coordinators.
For N=3, use [ade-n3-operation.md](../../examples/math-sft/ade-n3-operation.md)
with its N=3 configuration and resource allocation. The baseline handoff is the same.

To launch ADE manually, create a separate `ade-run` tmux session and use the same
Codex command above with `runs/operation-prompts/ade.md`, asking it to start ADE
from the supplied baseline. Do this only after baseline acceptance and cleanup. Use
`tmux attach-session -t ade-run` to return to that session.

From the checkout on the head, create the ADE terminal:

```bash
tmux new-session -s ade-run -c "$PWD"
```

Then, inside that tmux session:

```bash
source .unified-vllm-0.19.1-verl-venv/bin/activate
codex -m gpt-5.6-luna -c 'model_reasoning_effort="medium"' --sandbox danger-full-access --ask-for-approval never \
  'Use $ade-supervise-run. Read .agents/skills/ade-supervise-run/SKILL.md and its complete workflow. START the authorized ADE in runs/operation-prompts/ade.md using its completed baseline and bootstrap frontier. Check admission, start exactly once and supervise through terminal acceptance and cleanup. Do not retrain the baseline.'
```

## Read the result

Start with the Agent's final report and `<run-root>/monitor.md`, then follow
[results](results.md) to accepted Trial Records, Memory, rankings and cleanup.
A failed hypothesis can be a valid completed experiment; Run completion is not a claim
of improved model quality.
