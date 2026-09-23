# ADE Run Workflow for a Supervising Code Agent

This workflow is the complete operator procedure. Operation Prompts select an
experiment and supply only Run-specific facts; they do not duplicate lifecycle,
preflight, recovery, or terminal-acceptance rules. Set `PROJECT_ROOT` and `ADE_ENV`
to the exact absolute Repository and Execution environment values in the selected
Operation Prompt. The environment must be the complete runtime installed for this
checkout; it may be node-local. Use it for every Supervisor, ADE, Ray, probe and
skill-helper command:

```bash
cd "$PROJECT_ROOT"
source "$ADE_ENV/bin/activate"
ADE_PYTHON="$ADE_ENV/bin/python"
test -x "$ADE_PYTHON"
"$ADE_PYTHON" -c 'import ade; print(ade.__file__)'
```

Require the imported ADE source to belong to the selected checkout. Missing or
conflicting paths block preflight. Do not substitute another checkout, shell
Python or a worker's environment. The public prompt templates are under
`examples/math-sft/`; unresolved placeholders and `NOT_AUTHORIZED` do not authorize
live preflight, services or GPU use.

The repository root is the project root. The selected experiment, selected
deployment config, and read-only `ade experiment inspect` result are
authoritative for task, resource, deployment, evaluation, Judge, and tracking
values.

## Overview

This is the operational contract for a fresh code agent that prepares, starts,
monitors, repairs, and closes one ADE Run. The agent is an operator-side glue
layer. It does not participate in Coordinator, Builder, Analyzer, Summarizer,
selection, reward, evaluation, or ranking decisions.

The workflow is deliberately independent of task scale. The selected experiment
YAML owns the task type, model, data, seed, number of Coordinators, Plan and Trial
budgets, training schedule, evaluation policy, analysis policy, and tracking
settings. The Operation Prompt selects one exact deployment YAML, which owns
Ray, GPU allocation, Judge, staging, and deployment authorization bindings.
Never copy an experimental or deployment value from this workflow into a Run.

One formal Run has one independent supervising agent, one long-running goal, one
experiment config, one deployment config, one Run-owned checkout/deployment attachment, and one W&B
group. The agent does not coordinate with, wait for, confirm the revision of, or
infer decisions from another N=1/N=3 Run. A separate Run has its own supervising
agent and evidence.

The initial prompt's explicit authorization covers the GPU and external-service
use resolved from the experiment and the recovery actions in this workflow. Once
admitted, continue until terminal acceptance or a concrete blocker. Routine
preflight, automatic recovery, an authorized service repair, same-Run resume, and
a typed fork with sufficient evidence do not require another human confirmation.
Stop for the Operator only when a new scientific decision, an action outside the
granted resource/repair scope, or an unresolved safe-boundary decision is needed.

## Required inputs

Obtain these before preflight:

- the repository root and exact experiment YAML;
- the exact deployment YAML selected by the Operation Prompt;
- the exact absolute full execution environment path supplied by the Operator;
- confirmation that the repository path is the checkout assigned to this Run;
- the current master IP supplied by the Operator;
- the Operation Prompt's Initial State Reference and Initial State Frontier;
  both are `NONE` for native Bootstrap, while a seeded Run supplies one
  completed source Run ID plus either `bootstrap` or exactly one
  `cNNN=pNNN` entry per Search Coordinator;
- whether this is a new Run, a same-Run resume, or a typed evidence-backed fork;
- for a new Run, the experiment ID; the Run ID is generated mechanically by the
  Harness using the rule below. The operator and every Agent must not invent a
  semantic Run ID;
- explicit authorization to consume the resources declared by the selected
  deployment for this experiment.

Keep an exact list named `SUPERVISOR_PREFLIGHT_PATHS` of every disposable path
created by this Supervisor before the target Run starts. In particular, record
the `runs/local-judge-preflight-*` Run root printed by
`scripts/probe_local_judge.py`. These paths remain available for startup
diagnosis, but final terminal cleanup must remove them before the Supervisor
exits.

If the experiment YAML, deployment YAML, or master IP is missing, stop and ask
the Operator before contacting external services or allocating GPUs. Do not
guess a deployment from the machine on which the shared filesystem happens to
be mounted.

The Operation Prompt must provide the exact deployment YAML, expected Master IP,
and explicit authorization to use its resources for the selected experiment.
Once those facts are present, do not ask for them again. The
compiled deployment is authoritative for the Judge model, gateway port, GPU
count, Ray address and workload allocations. Ray chooses the Judge node and
physical GPU IDs; admission persists the actual host and gateway URL. Ask again only when the prompt fact is missing, the
prompt and compiled deployment disagree, or the requested action exceeds the
declared resource or repair scope.

Secrets are not user-facing Run inputs. Their environment-variable names come
from the compiled experiment and deployment; their values come from the ignored
project `.env` or the process environment. Confirm required variables are
non-empty without printing them. Never put credentials, authorization headers,
or `.env` contents in logs, resolved config, repair records, or Git.

## Run modes and interrupted Agent sessions

`NEW` selects a new Run. `EXISTING` selects an actual Run ID whose current process
and state must be reconciled before action; observe a live supervisor rather than
starting a duplicate. `RESUME` selects an actual paused/suspended Run after diagnosis;
use `ade run resume` with the same config and ID, without Seed arguments. A typed
fork requires its exact cause/evidence selector and follows the fork section below.
These are Operation Prompt instructions for the Agent, not CLI `--mode` flags.
A replacement Agent must first establish whether the original supervisor is still
live. A disconnected chat or quiet log alone does not authorize a restart.

## Configuration and naming authority

After activating the repository-owned environment above, compile without
creating a Run:

```bash
ade --project-root "$PROJECT_ROOT" experiment inspect "$EXPERIMENT_CONFIG" \
  --deployment "$DEPLOYMENT_CONFIG" \
  --run-id "$RUN_ID"
```

This command is read-only. Its `resolved` object and `config_digest` are the
preflight authority. Its `runtime_roots` are derived from the selected deployment
and must be used as printed. Do not reconstruct settings from component YAMLs,
override roots to legacy global directories, or edit the experiment after Run
creation.

For a new Run, generate exactly:

```text
<experiment-id>-<UTC timestamp>-<6 lowercase hexadecimal characters>
```

The timestamp is UTC in `YYYYMMDDTHHMMSSZ` form. For example:

```text
math-math-rft-reward-design-ADE-formal-20000101T000000Z-00000f
```

`ade create <experiment.yaml> --deployment <deployment.yaml>` generates this ID
when `--run-id` is omitted.
The standard supervising `experiment inspect` and `run start` path requires an
explicit `--run-id`, so the supervising Agent must construct it mechanically,
for example:

```bash
RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_SUFFIX="$(od -An -N3 -tx1 /dev/urandom | tr -d ' \n')"
RUN_ID="${EXPERIMENT_ID}-${RUN_TIMESTAMP}-${RUN_SUFFIX}"
```

Use only path-safe letters, digits, `.`, `_`, and `-`. The suffix prevents two
runners sharing a filesystem and W&B project from choosing the same identity.
Do not use names such as `formal01`, `smoke-a`, a deployment name, or a manually
chosen semantic suffix. Do not encode `N`, model size, dataset, or GPU count
separately when those facts are already in the experiment; the experiment ID
should carry the useful family name. An Agent must never choose, rename, or
propose a Run ID; Run identity is Harness/Operator mechanical metadata.

The experiment config owns `bootstrap.reference.enabled`. When it is `false`,
the Operation Prompt must say `Initial State Reference: NONE` and `Initial
State Frontier: NONE`; do not pass initial-state arguments. When it is `true`,
both values are required. The reference is a completed source Run ID. The
frontier is either `bootstrap` or one `cNNN=pNNN` selection for every source
Search Coordinator.

Before allocating resources or creating the target Run, execute the read-only
seed preflight with the same experiment and frontier:

```bash
ade --project-root "$PROJECT_ROOT" run seed inspect "$EXPERIMENT_CONFIG" \
  --deployment "$DEPLOYMENT_CONFIG" \
  --source-run "$INITIAL_STATE_REFERENCE" \
  --frontier "$FRONTIER_ENTRY_1" \
  --frontier "$FRONTIER_ENTRY_2"
```

Pass one `--frontier` for Bootstrap (`--frontier bootstrap`) or repeat it for
the Coordinator map; do not mix the two forms. Require source status
`completed`, the expected source deployment, exact terminal revisions, the
computed anchor revision, `requested_frontier == resolved_frontier`, no
discarded nonterminal scope with archived/RM-merged facts, compatible scientific
config, readable durable refs, and at least one remaining target Plan slot.
Bootstrap Seed may use a normal completed Search Run: ADE loads its exact
`bootstrap.baseline_revision` and imports only Base/P000. It does not require a
baseline-only source.

A Coordinator frontier imports the contiguous closed Plan-slot prefix for every
Coordinator at the single anchor revision equal to the latest requested terminal
revision. Other Coordinators are projected at that same revision; unmerged
nonterminal work is discarded and recorded in `seeded_from`. ADE creates target
revision 0 with transition `run_seeded`, keeps `forked_from` empty, counts
imported Plan slots against the target's total budget, and starts only the
remaining slots. Seed is neither resume nor fork and never warm-starts an
optimizer/checkpoint.

Run Seed copies accepted local artifacts, snapshots, Memory, Trial Records and
Engine/operator result objects into the target deployment before normal work.
It replays selected durable evaluation records once under the target group with
Seed provenance. Eligible selected non-evaluation history is republished under
stable `job_type=seed-import` / `imported-history/...` identities; historical
`baseline-import` identities are not migrated or deleted. Resume never
re-resolves the source tip or depends on the source workspace.
Seed history reconciliation streams scalar history only. It must not download
or re-upload source W&B artifacts: accepted artifacts are already materialized
target-locally, and checkpoint state is outside the Run Seed contract.
Scientific incompatibility or a missing imported fact is an admission failure:
do not create a paused/suspended target or replace imported evaluation by
rerunning Base, P000, or Search Trials.

W&B naming has three levels:

1. The W&B project is derived from the root ADE Run ID. A fork reuses
   `forked_from.lineage_root_run_id`; it does not create a second project.
2. One ADE Run is one W&B `group`, exactly equal to the current ADE Run ID.
3. Training, evaluation, operator evaluation, and the Run monitor are separate
   W&B runs inside that group. W&B limits both its human-facing `name` and
   external run ID to 128 characters. The canonical identity is therefore the
   project + exact group + external ID tuple; RunState and the tracking audit
   retain the unabbreviated SubjectRef. Training display names use
   `<coordinator>/<plan>/<trial>/training/<attempt>`. Identity checks must use
   the tuple, not the shortened display name. When the path-safe
   `<run-id>--<qualified-subject>--<role>[--attempt]` exceeds the service limit,
   retain its rightmost 128 characters; this preserves the globally unique Run
   suffix and qualified workload tail while the exact group supplies the full
   Run ID. Bare `pNNN` or `tNNN` is never an external identity.
   W&B artifact names have the same 128-character service limit. Evaluation
   publishers retain the rightmost 128 characters of the qualified artifact
   name; the durable tracking request and artifact metadata retain the complete
   Run, command, trial, purpose, and fork-lineage identity.

Evaluation identities are explicit and must not be inferred from a bare
`p000`, `base-online`, or `base-offline` execution label:

| Scientific subject | W&B job type | Canonical display role |
| --- | --- | --- |
| Base Model online/offline | `base-evaluation` | `c000/p000/base-model/evaluation` |
| P000 Baseline online/offline | `bootstrap-evaluation` | `c000/p000/p000-t000-baseline/evaluation` |
| Search Trial online/offline | `trial-evaluation` | `<coordinator>/<plan>/<trial>/evaluation` |
| Base Model/P000 Baseline/Search Trial operator result | `operator-evaluation` | `<target>/operator-evaluation` |
| selected source non-evaluation history | `seed-import` | `imported-history/<source-role>` |

RFT Base position 0 and Base offline evaluation append to the same Base Model
stream. P000/Search streams contain only their own checkpoint positions; do
not copy the Base position-0 result into a Trial stream. Imported evaluation
facts are replayed once from durable local records into these canonical target
streams. Non-evaluation history is allowlisted by the selected source subjects
and republished only through the separate `seed-import` identity.

`cNNN/pNNN/tNNN` are ordinals scoped by their parents. Across state, queues,
Plan Catalog, PM/RM merge, Engine/Review/Judge, Ray, filesystem manifests,
reports, CLI and W&B, use the full SubjectRef:

```text
<run-id>/cNNN/pNNN
<run-id>/cNNN/pNNN/tNNN
```

`tracking.entity_env` is optional at runtime: when its variable is unset, W&B
uses the default entity owned by the authenticated API key. Do not invent an
entity. `tracking.base_url_env` may likewise be unset when the default W&B
service is intended. The API key named by `tracking.api_key_env` is mandatory
for online mode.

W&B transmission protocol:

- Local evaluation and Seed-import records are the authority. ADE writes the
  durable result/request before attempting the remote projection.
- A newly executed evaluation makes its first projection attempt immediately
  after the local write. Imported Seed evaluations are the deliberate exception:
  target creation only stages their complete local requests, and the Run
  Monitor starts projection in its first reconciliation cycle so a large
  frontier cannot serialize W&B timeouts in admission. A provider failure
  leaves the same stable identity in `pending_retry`; it must not fail training
  or evaluation or create a second logical run.
- The Engine/evaluation publisher owns the immediate attempt, the Run Monitor
  owns background retries, and reconciliation runs once every ten minutes.
  Reconciliation only replays durable requests and records a receipt; it does
  not rerun evaluation or compete with the normal publisher.
- Online Run Monitor health is not inferred from a successful local `init` or
  `log` call. The monitor reads back its exact remote project/group/run ID and
  history step with a bounded Public API request. Before the first read-back,
  health is `pending_remote`; zero remote history after ten minutes is
  `unhealthy`. After the first success, a remote history step that does not
  advance for twenty minutes is also `unhealthy`.
- SDK initialization is bounded to 60 seconds, terminal upload/finish to 120
  seconds, and Public API reads to 30 seconds. A timeout leaves durable local
  inputs for retry; it must not hold Run admission, resume, or Control progress.
- Training subprocess finish timeout is non-raising so a tracking flush cannot
  turn completed training into an Engine failure. ADE-owned publishers record
  timeout as pending/unhealthy tracking and retain their local input.
- Seed non-evaluation history is never copied synchronously in Run creation or
  resume. Target creation writes the Seed history request, and the Run Monitor
  reconciles it in the background under the stable `seed-import` identities.
  One reconciliation cycle attempts at most one pending evaluation across the
  whole Run (rotating fairly across local roots) and checks or publishes at most
  one previously unknown Seed-history run,
  so multiple records cannot multiply a provider timeout into a blocked monitor
  cycle. Locally confirmed Seed runs are not re-read on later cycles.
- `evaluation/checkpoint_step` and `evaluation/checkpoint_position` are the
  real training step of the evaluated checkpoint. Evaluation charts use
  `evaluation/checkpoint_position` as their axis. W&B `_step` is the monotonic
  publication ordinal, also recorded as `evaluation/evaluation_ordinal`;
  online and offline results at the same checkpoint occupy separate rows.
- Every run with the same `job_type` uses the same metric, matrix, artifact,
  and schema-version layout.

The Run Monitor performs this reconciliation in the background. The supervising
Agent must not create a second polling loop for it. During each normal
ten-minute status snapshot below, read the latest
`tracking/*/reconciliation-latest.json` once; `run-monitor-health.json` reports
the exact local/remote Run Monitor steps but does not aggregate
evaluation-publication pending counts. The snapshot helper reads each latest
reconciliation receipt once and writes its local/attempted/published/pending/
remote-missing counts plus Seed-history status into `monitor.md`. A pending
request is a transmission issue, not an evaluation failure; repair it through
the publisher/reconciler without rerunning ADE work.

## Preflight

Perform the checks in order. A failure stops admission to the next layer.
Record only safe facts and receipt paths under a deployment-scoped preflight
directory in `runs/`; do not use `/tmp` for durable evidence.

### 1. Compile and inspect the complete experiment

Run `ade experiment inspect` and verify all of the following from its output:

- the config compiles with the current schema and all source files resolve;
- `task.type`, `training`, `evaluation`, `analysis`, and `search` describe the
  requested experiment;
- the root seed is present and propagated by the compiler;
- every model, dataset, backend source, and runtime environment path exists;
- checkpoint production and online/offline/operator evaluation schedules match;
- `resolved.operator_evaluation.enabled` and the operator purpose, test dataset,
  evaluator, sampling, decoding, seed, model, and prompt semantics match the
  referenced source contract; deployment paths, tracking settings, resource
  allocations, command IDs, and timeouts are not scientific compatibility
  fields;
- `search.coordinator_count`, Plan and Trial budgets are internally consistent;
- every allocation fits `runtime.coordinator_capacity_gpus`; derive the Run's
  required live capacity as
  `search.coordinator_count * runtime.coordinator_capacity_gpus`, and require
  that value to fit `runtime.cluster.total_gpus`;
- Agent backend, model, execution timeout, retry budget, sandbox mode, and
  approval policy are explicit;
- automatic-recovery `max_attempts` (total physical Attempts including the
  initial Attempt) and dependency readiness timeout are explicit;
- tracking is online, preserves the root-derived project resolved from the ADE
  Run identity, and names only environment variables for entity, base URL,
  and API key;
- the deployment ID and every reported runtime root are deployment-scoped.

The compiler validates structure and local paths. It does not prove remote
capacity, credentials, service health, or free disk; the later checks do that.

For a new Run, require that the resolved Run directory does not exist and that
the W&B project contains no group with the chosen Run ID. For resume, require the
opposite: the Run exists, and recompiling the same experiment produces the same
resolved-config artifact expected by ADE. Never reuse an old Run ID for a new
experiment.

### 2. Record code provenance and shared-filesystem context

The Run Monitor automatically appends `git rev-parse HEAD` to
`<run-root>/reports/source-provenance.jsonl` at every start/resume, together
with the resolved config digest and UTC timestamp. It mirrors the current
commit and provenance index in the W&B Monitor Run's `ade` config. The local
JSONL is the complete provenance authority; W&B is only a projection. This is
provenance for the code loaded by the initial workers; it is not a cleanliness
gate. Do not pause or stop a healthy
Run merely because another developer or Agent created a commit or left an
uncommitted change elsewhere in the checkout.

The supplied checkout may be shared by several Runs. Existing workers continue
to use the code loaded at their start. Do not intentionally reload changed source
into a live Run's workers; if a repair is needed, use the durable pause/resume
boundary below. Record the exact source revision used by every replacement
worker after resume. Deployment-scoped runtime roots remain independent from
source provenance.

Before starting a New Run or resuming after a source repair, the Supervisor
must fence stale Run-owned workers under the resolved deployment roots and start
a fresh Monitor, Agent, Engine, Review, and Control process group from the
current checkout. Engine and Review queues are deployment-scoped, so an old
worker can otherwise claim a new Run's command with old code. First require
that no other nonterminal ADE Run owns the exclusive deployment, then record
the replacement workers' PIDs, startup time, checkout `HEAD`, and resolved Run
ID. Do not fence Ray or a Judge launch attached to this same nonterminal Run in
this worker-fencing step.

The Local Judge is a deployment-scoped service composed of one ADE Rubric Job
gateway plus eight vLLM endpoint processes. It remains attached across same-Run
recovery and may be handed directly from a paused/suspended source to its direct
fork, but it is not retained after an independent Run reaches completed, failed,
or cancelled. At that final boundary, lifecycle cleanup cancels nonterminal
jobs and stops the exact recorded gateway and all eight endpoints. If Judge
gateway code, launcher/job-store code, dependencies, service environment, model
files, or vLLM generation/runtime settings change during a nonterminal Run,
cancel only its nonterminal jobs and restart the exact recorded launch before
continuation.

### 3. Bind the supplied master IP

The supplied master IP must match the head host in the compiled Ray address.
The deployment ID must uniquely identify its runtime roots, and its named Judge
authorization variable must be set. A mismatch blocks admission: correct the
Operation Prompt's deployment selection, not the frozen experiment.

The head's GPU advertisement is an operator-supplied node startup fact, configured
with `ade cluster head --gpus`; it can be zero. The Judge is not pinned to the head.
Ray places its eight-GPU actor on an eligible node. Verify that node's actual host,
node ID, assigned device IDs and gateway URL from the admitted service handle;
do not require its host to equal the head. All eligible nodes need accessible
model, source, runtime and shared output paths.

### 4. Verify Ray and shared paths without allocating GPUs

Before any Run creation or service launch, execute and retain the result of:

```bash
"$ADE_ENV/bin/ray" status --address "$RAY_ADDRESS"
```

Connect to the compiled Ray address and inspect live node membership and
resources. `runtime.cluster.total_gpus` is the configured full-pool capacity; it
is not the live admission minimum for every Run size. Derive the admission
minimum from the resolved experiment as
`search.coordinator_count * runtime.coordinator_capacity_gpus`, plus the eight-GPU Judge reservation when starting a new Judge actor. For same-Run resume with a live matching actor, its reservation is already occupied and must not be counted again as a free-GPU requirement. Require both the
observed alive-node GPU total and currently available GPU total to be at least
that value, node advertisements to match the Operator's declared capacity, and the
alive per-node shape to satisfy every resolved allocation. Do not infer a
fixed GPU profile from this workflow: read `coordinator_capacity_gpus`, each
workload allocation, and the full-pool capacity from the resolved runtime
configuration for the selected experiment. Record missing and extra nodes;
missing nodes block only when they make the Run's derived capacity or placement
shape unavailable. Confirm the cluster's exclusive ADE Run contract: no other
nonterminal ADE Run, active placement group, or unreleased lease may own the same
cluster.

#### Ray placement-group reconciliation

The allocator lease table and Ray's placement-group table are separate durable
views. A cancelled or terminal Run can leave a Ray placement group in
`CREATED` state after its allocator lease has disappeared. That group still
reserves GPU resources and can make a new evaluation wait indefinitely; an
advancing Engine heartbeat is not evidence that evaluation is making progress.

Before admitting a New Run, and after cancelling or terminalizing any Run,
reconcile the two views:

Read both views directly from GCS and the existing detached ADE allocator with
the skill helper; it does not require a Ray Dashboard or State API server and
does not create the allocator when none exists:

```bash
"$ADE_PYTHON" \
  .agents/skills/ade-supervise-run/scripts/inspect_ray_admission.py \
  --address "$RAY_ADDRESS" \
  --output "$CONTROL_ROOT/preflight/$RUN_ID/ray-admission.json"
```

For a New Run on an exclusive deployment, require the receipt to report
`status=complete`, no non-removed placement groups, no allocator leases and no Judge actors.
For same-Run resume, reconcile listed resources against that Run's live ownership;
do not require its legitimate reservations to be absent.
The helper deliberately filters historical `REMOVED` groups from its detailed
output while retaining their count. If it reports `blocked`, use the listed
group IDs, names, states, bundle counts, node assignments, and leases for the
ownership reconciliation below. Do not substitute `ray list placement-groups`:
that command depends on the optional Dashboard API and is unavailable on valid
head nodes started with `--webui=`.

1. List every non-removed placement group and record its name, state, bundle
   count, and node assignment. Inspect the allocator snapshot for the same
   deployment/run scope.
2. For every `CREATED` ADE GPU-lease group, require a matching live allocator
   lease and a nonterminal owner. A group with no matching lease, or one owned
   by a terminal/cancelled Run, is orphaned for admission purposes.
3. Resolve the exact old Run and exact placement-group IDs. Confirm that no
   active worker, command, or lease still belongs to that old Run. Remove only
   those exact orphan groups, then re-read both tables and require them to be
   `REMOVED`/absent and the GPUs to be available.
4. Never remove a group merely because it is old-looking, and never remove a
   group belonging to the current nonterminal Run. If ownership cannot be
   resolved, stop admission and report the concrete conflict.

Cleanup is incomplete until `ray status` shows the derived free-GPU capacity
and the allocator snapshot contains no lease for the terminal owner. During a
long evaluation, if the command heartbeat advances but no lease is present and
no shard/artifact progress exists, perform this reconciliation before waiting
for the command timeout.

#### Physical GPU admission check

Ray's available-GPU count is an allocator fact, not proof that the devices are
usable. A cancelled Run can release its Ray lease while its worker process or
CUDA context continues to hold device memory. Before admitting a New Run,
inspect every worker that can receive a resolved allocation with `nvidia-smi`
(device UUID, memory used/free, and compute applications), and record the
output in the deployment-scoped preflight receipt.

Admission requires all of the following:

1. Every material GPU process or CUDA context is attributable to a currently
   authorized service or nonterminal Run. A process left by a cancelled or
   terminal Run is stale, even when Ray reports the GPU as available.
2. Physical free memory on each target device is sufficient for the first
   resolved workload. If the deployment has no separate memory budget, require
   no material unexplained usage; do not infer usability from Ray's free-GPU
   count alone.
3. If any stale process or context from an old Run remains, stop admission
   immediately and report the exact worker, device UUID, PID, memory usage,
   owning/last-known Run, and the `nvidia-smi` receipt. For a New Run, leave it
   unstarted; for an existing Run, request a durable pause. The Agent must not
   terminate that process, reset a GPU, reset or restart the VCluster/Ray
   resource, or broaden cleanup scope. Wait for explicit Operator direction.
   Do not start a Run and let its first Engine command reveal the conflict.

The Agent may perform only read-only inspection at this boundary. In
particular, `nvidia-smi --gpu-reset`, VCluster/vGPU reset, Ray restart, and
process termination are not admission actions. A later Operator-directed
repair must establish its exact target and ownership before it is attempted,
then repeat this physical GPU check from the beginning.

This is a hard gate in addition to placement-group and lease reconciliation. A
successful Ray capacity check, Judge preflight, or W&B preflight cannot
override a failed physical GPU check.

On every participating worker, perform read-only existence checks for all
resolved model, dataset, backend, and shared output parent paths. A path existing
on the master alone is insufficient. Do not load a model or create a placement
group during this check.

Estimate retained storage from the compiled checkpoint schedule and Trial
budget, using actual model/checkpoint sizes when available. Include training
artifacts, evaluation outputs, rollout evidence, W&B local files, and transient
staging. Compare the estimate with filesystem bytes and inodes. The configured
retention policy is authoritative; do not silently delete or reduce checkpoints
to make admission pass.

### 5. Verify Codex and Review Labor

Run `codex login status` and confirm the installed CLI supports the model and
execution policy in the resolved Agent config. ADE supplies sandbox and approval
values on every new and resumed invocation; local Codex defaults are not the Run
contract.

Inspect `resolved.analysis.provider`. The public formal configs use
`local_analyzer`, backed by the deployment Judge. It has no separate provider API
key or externally supplied model endpoint: verify the configured Judge authorization
variable and complete Judge admission in step 7. Check actual Analyzer Review
packets and usage during the Run; do not call an unstarted service in this step.
For a configured remote Review provider, confirm its named environment variables;
a changed binding requires one authorized minimal structured-response/usage probe.
Do not invoke Review through MCP or start an Analyzer session merely to test credentials.

### 6. Verify W&B login, write/read, grouping, and listing

Temporarily unset upper- and lowercase `HTTP_PROXY`, `HTTPS_PROXY`, and
`ALL_PROXY` for every W&B operation. With the base URL, project, API-key variable,
and optional entity from the resolved config:

Load the ignored project `.env` before unsetting those proxy variables so the
resolved API-key and optional base-URL/entity variables remain exported. In the
currently pinned W&B client, the public SDK `viewer` property may incorrectly
raise `relogin required` even when that same key can create and read runs. When
that specific mismatch occurs, verify the viewer with an authenticated GraphQL
`Viewer` request to `<base-url>/graphql` using the same API key, without logging
the key or authorization header. This is accepted only if the direct viewer
request succeeds and the complete SDK write/read/list/artifact/delete transaction
below also succeeds; a write-only probe or a failed direct viewer still blocks.

Run the probe with the same bounds as production: construct the disposable run
with `wandb.Settings(init_timeout=60, finish_timeout=120,
finish_timeout_raises=True)` and construct every Public API client with
`wandb.Api(timeout=30, ...)`. A timeout is a failed preflight receipt, not a
reason to leave the supervising terminal waiting indefinitely.

1. authenticate and read the viewer/default entity;
2. create one disposable `job_type=preflight` run in a temporary group derived
   from the intended ADE Run ID;
3. log one scalar and one small artifact, finish the run, and read both back;
4. list the project with a group filter and require that exact probe to appear;
5. delete only the exact disposable probe after successful read-back.

`relogin required`, a wrong project/entity, failed artifact read-back, or broken
group filtering blocks the ADE Run. Do not switch tracking to offline. During
the actual Run, ADE handles the same proxy isolation and stable IDs itself.

Immediately after the Supervisor starts, verify the durable local tracking
record and the remote W&B listing agree on the exact tuple
`project + group + external run ID` for the Run Monitor. The expected values are
`project=<resolved root-derived project>`, `group=<run-id>`, and
`external_id=<run-id>--run-monitor`. Repeat the same identity check for the
first Base Model and P000 Baseline evaluation streams before treating Bootstrap
as healthy. Require the Base Model stream to use `base-evaluation` and
`c000/p000/base-model/evaluation`, and the P000 stream to use
`bootstrap-evaluation` and `c000/p000/p000-t000-baseline/evaluation`. A link
that is readable but belongs to another project, group, Run ID, subject kind,
or source revision is a runtime tracking-health failure. Repair that projection
before treating Bootstrap as operationally healthy; it does not roll back an
already accepted local scientific boundary.

### 7. Inspect or admit the Local Judge when required

Only experiments whose resolved `judge_enrichment.enabled` and Run resources
enable the Local Judge require this step. Use the deployment source selected by
the Operation Prompt and `scripts/probe_local_judge.py`; for RFT pass `resolved.seed`
and the actual full-step population derived from compiled
`train_batch_size * rollout_n`. For SFT selection, derive the selection batch
population from the resolved candidate inventory contract instead. Do not copy
either value from this workflow.

```bash
"$ADE_PYTHON" scripts/probe_local_judge.py \
  --project-root "$PROJECT_ROOT" \
  --deployment "$DEPLOYMENT_CONFIG" \
  --seed "$RESOLVED_SEED" \
  --full-step-rows "$RESOLVED_FULL_STEP_ROWS"
```

`DEPLOYMENT_CONFIG` is the exact Operation Prompt selection listed by experiment inspection;
the remaining variables are derived from the same resolved output.

The probe performs the supported lifecycle:

- inspect deployment-owned `service.json`, recorded PIDs, endpoint health,
  gateway protocol, model identity, and launch identity; gateway `/v1` must
  report the actual `launch_id/protocol/model_digest` matching that handle;
- reuse only a launch attached to the same nonterminal lifecycle when every
  binding and health check matches;
- if it is unhealthy or its binding changed, cancel only its pending jobs, stop
  that exact recorded launch, and start a replacement;
- run the configured single-schema, mixed-schema, and bounded batch checks;
- detach the probe, stop its exact gateway and eight endpoints, and remove that
  stopped launch's transient jobs/log root.

After the probe exits, append its printed `runs/local-judge-preflight-*` root to
`SUPERVISOR_PREFLIGHT_PATHS`. Retain that receipt through target startup so a
startup failure is diagnosable. Do not delete any launch directory before its
recorded PIDs are released.

Require the printed receipt, complete usage, expected row status, stable launch
identity during the probe, then require `released=true`, no ready gateway after
detach, and `launch_root_removed=true`. Target Run startup creates and records
its own exact launch. Never kill Judge processes by name. A process set is not
ready merely because all target ports answer: if `/v1` reports another launch
ID, treat it as a foreign old gateway and repair the recorded lifecycle
ownership before admitting a Run.

### 8. Final admission decision

Before `start`, all of these must be true at the same time:

- the recorded `ade experiment inspect` result was produced from the exact
  admitted checkout HEAD, its top-level and resolved `config_digest` values
  agree, and `resolved.engine.automatic_recovery` contains explicit positive
  `max_attempts` and `dependency_readiness_timeout_seconds` values;
- config and master binding accepted;
- code revision known and shared checkout safe;
- stale deployment workers fenced and fresh worker provenance recorded;
- remote paths and disk accepted;
- Ray topology can place every resolved allocation and the Run's derived live
  capacity is free;
- physical GPU memory and compute-process ownership are clean on every target
  worker, with no unresolved stale context and a fresh `nvidia-smi` receipt;
- Codex, Review, W&B, and any required Judge checks passed;
- no existing local Run directory or remote W&B group uses the new Run ID;
- the Operator's GPU authorization covers the resolved request.

If any fact is missing, stop. Do not create a partial Run and hope a later worker
discovers the problem.

## Start and normal monitoring

Create or continue one long-running supervising goal whose objective is the
terminal acceptance checklist in this workflow. The goal remains active across
component waits and recoveries; a slow healthy training, Judge FIFO wait, or
temporary W&B upload failure is progress/waiting state, not a reason to end the
goal. Do not create a second supervisor goal for each Plan or Attempt.

Before admission, the supervising Agent works continuously through the ordered
preflight checks and resolves any in-scope operational blocker. It does not
sleep on the normal monitoring cadence while an admission check remains to be
performed. Once every final-admission fact above is simultaneously true, start
the Run exactly once. After startup, the Supervisor, Control, Run Monitor, and
workers own continuous execution, heartbeats, queue progress, background W&B
reconciliation, and automatic recovery. The supervising Agent observes this
system; it must not drive normal progress by repeatedly invoking ADE commands.

Start a new Run with the foreground supervisor:

```bash
ade --project-root "$PROJECT_ROOT" run start "$EXPERIMENT_CONFIG" \
  --deployment "$DEPLOYMENT_CONFIG" \
  --run-id "$RUN_ID"
```

For a seeded Run, append the same values that passed seed inspect. Bootstrap
uses one frontier argument:

```bash
ade --project-root "$PROJECT_ROOT" run start "$EXPERIMENT_CONFIG" \
  --deployment "$DEPLOYMENT_CONFIG" \
  --run-id "$RUN_ID" \
  --initial-state-reference "$INITIAL_STATE_REFERENCE" \
  --initial-state-frontier bootstrap
```

A Coordinator frontier repeats `--initial-state-frontier cNNN=pNNN` once for
every Coordinator. Never pass initial-state arguments to `run resume`.

Do not pass manual runtime roots during normal operation. The supervisor derives
them from the deployment, creates the Run before its logs, starts the Run monitor
first, starts the required Coordinator-scoped Agent/Engine/Review workers plus
the global Agent worker when needed, and starts Control last. The number of
workers follows `search.coordinator_count`; the command is identical for every
supported N.

Keep the foreground supervisor under a durable operator session. After the Run
has started successfully, the supervising Agent owns one append-only observation
journal at:

```text
<CONTROL_ROOT>/<RUN_ID>/monitor.md
```

This file is the durable output of the supervising Agent's observation loop. It
is distinct from ADE's internal Run Monitor process and
`tracking/run-monitor-health.json`. Create `monitor.md` only after the Harness
has created the Run directory. Give it one heading containing the Run ID,
experiment path, deployment, UTC start time, and `Cadence: 10 minutes`; then
append every snapshot below without rewriting earlier entries. The journal is
operator evidence, not a Control input, and must never be used to infer or edit
Run state.

```markdown
# ADE Run Monitor — <run-id>

- Experiment: `<experiment path>`
- Deployment: `<deployment name and master IP>`
- Started: `<YYYY-MM-DDTHH:MM:SSZ>`
- Cadence: 10 minutes
```

Take the first read-only snapshot immediately after successful startup. Take
subsequent normal snapshots every ten minutes using the `control`, `queue`, and
`run` paths returned by experiment inspection:

```bash
"$ADE_PYTHON" .agents/skills/ade-supervise-run/scripts/append_monitor_snapshot.py \
  --project-root "$PROJECT_ROOT" \
  --control-root "$CONTROL_ROOT" \
  --queue-root "$QUEUE_ROOT" \
  --run-id "$RUN_ID" \
  --experiment "$EXPERIMENT_CONFIG" \
  --deployment "$DEPLOYMENT_CONFIG" \
  --master-ip "$MASTER_IP" \
  --mode "$MONITOR_MODE"
```

The helper captures exactly one `run observe`, binds the immutable RunState
revision, reads the bounded report and health inputs once, and appends the fixed
Markdown snapshot without streaming raw JSON. Pass `--health '<specific anomaly
and evidence path>'` when focused evidence changes the helper's basic health
classification. Do not call `run observe` separately for the same snapshot.
Every scheduled or event-driven observation must therefore produce exactly one
appended `monitor.md` entry; do not observe and discard the result. Do not run
`run history`, repeat the helper, invoke an Agent worker, or perform a full
evidence scan between ten-minute snapshots merely to see whether a healthy Run
changed. Use `run history` only for focused diagnosis, recovery review, or
terminal acceptance.

Use the revision returned by `run observe` as the snapshot boundary. Read that
immutable revision's
`state/revisions/rev-<revision>/run.json` once for Coordinator, Plan, Trial,
outcome, and active-work fields. Read `reports/results.csv`,
`reports/timeline.csv`, the per-Coordinator timeline files, the latest W&B
reconciliation report, and `tracking/run-monitor-health.json` once, retaining
only revision-bearing rows or facts at or before the observed revision. Treat
the non-revisioned health and reconciliation files as latest external-health
facts read at the snapshot time. Do not reread the live `run.json` until the
next snapshot and do not run a consistency retry loop if a newer revision
commits while the report is being rendered.

The monitoring modes and transitions are fixed:

- **startup**: work continuously through admission and Run startup. After the
  Run directory exists, take and append the first snapshot immediately. If it
  is healthy, enter `normal_wait`.
- **normal_wait**: after appending a healthy snapshot, compute
  `next_snapshot_at` as the snapshot start time plus 600 seconds and invoke one
  real blocking wait for the remaining interval. Use the environment's sleep,
  timer, scheduled wait, or process-wait primitive. Do not split the interval
  into short waits, issue commands, reread files, call `run observe`, or emit
  narration such as “still waiting” or “reading the prompt.” Silence is the
  intended state. On timer expiry, take one snapshot and append it.
- **diagnostic**: enter only when a snapshot shows a concrete anomaly: a
  suspended, failed, or recovering Run; an overdue configured heartbeat or
  boundary; unhealthy Run Monitor state; inconsistent committed state and
  accepted artifacts; or an explicit service health alarm. Perform the smallest
  focused diagnosis or authorized recovery. A follow-up `run observe` is
  allowed only after an action or event that can change state, or at most once
  per minute while waiting for a known short recovery transition. Append every
  such observation. Return to `normal_wait` immediately after one healthy
  snapshot. A long healthy Engine, validation, Judge FIFO or Analyzer wait is
  not diagnostic mode. A W&B retry remains a normal wait only inside the
  explicit ten-minute first-upload or twenty-minute post-success stall window;
  crossing either bound is a concrete diagnostic anomaly.
- **terminal**: when the foreground Supervisor exits or an explicit terminal
  event arrives, wake immediately, append one final snapshot, and run the
  terminal checklist. Do not wait for the next scheduled snapshot. If the wait
  primitive cannot receive process or alarm events, let it finish its single
  ten-minute interval; do not simulate event detection with polling.

Routine terminal/chat output is not the monitoring record. By default, write
the full snapshot only to `monitor.md`. If the host interface requires a visible
heartbeat, emit at most one concise line after the append with the UTC timestamp,
revision, status, and monitor path; never print waiting narration or mirror raw
`run observe` JSON.

At each snapshot, append the current state of every Coordinator and every Plan
and Trial materialized by the observed revision. Sort by Coordinator, Plan, and
Trial ID. Do not create placeholder rows for future budgeted Plans that do not
yet exist. The Markdown format is fixed:

```markdown
## Snapshot — <YYYY-MM-DDTHH:MM:SSZ>

- Run: `<run-id>`
- Status: `<status>`; revision: `<revision>`; bootstrap: `<status>`
- Last transition: `<kind>`; subject: `<subject ref|none>`
- Recovery: `<status and Attempt boundary|none>`

### Coordinator `<coordinator-id>`

- Control: `<control status>`
- Plan budget: `<materialized>/<effective limit>|bootstrap`
- Current state: `<planning|running|waiting|drained|terminal>`

| Plan | Plan status | Trial | Trial phase | Outcome | Active work | Latest durable progress |
|---|---|---|---|---|---|---|
| `<plan-id|none>` | `<status|none>` | `<trial-id|none>` | `<TrialState.phase|none>` | `<outcome|none>` | `<Agent/Engine/Review IDs or none>` | `<transition/checkpoint/evaluation/analysis/archive marker or none>` |

<!-- Repeat the Coordinator section and include every materialized Plan and Trial. -->

### Accepted results at revision `<revision>`

- `<each accepted Base Model/P000 Baseline/Search Trial result row, or none>`

### Health and schedule

- Health: `<normal|specific anomaly and evidence path>`
- Run Monitor: `<process/tracking status, local step, remote step, last remote check/error, health path>`
- W&B reconciliation: `<evaluation counts and Seed-history status with receipt paths>`
- Mode: `<normal_wait|diagnostic|terminal>`
- Next snapshot: `<UTC timestamp exactly 600 seconds after this snapshot started, or none>`
```

Include bootstrap Coordinator `c000` when it is present. For a Coordinator with
no materialized Plan, use one `none` row and state whether it is planning,
waiting, drained, or terminal. Report `Plan budget: bootstrap` for `c000`; its
P000 is not a Search Plan budget entry. For a Plan with no materialized Trial,
likewise use one `none` Trial row. Take Plan status, Trial `phase`, and Trial
`outcome` directly from the immutable RunState revision; do not infer them from
a process, W&B, or an incomplete artifact. List active work from committed
Agent Calls, Engine Commands, and Review Commands whose owner or target matches
the subject. Use the latest accepted transition, checkpoint, evaluation,
operator-evaluation, analysis, summary, or archive boundary at or before the
revision as the concise progress marker.

The `Accepted results at revision` section summarizes all accepted rows in
`reports/results.csv`, including imported Base Model, P000 Baseline, and selected
Search Trial rows. Imported rows carry their source Run, anchor revision,
frontier, and deployment provenance and do not wait for W&B synchronization.
The table is rebuilt at Run creation, after committed state updates, and on
resume. It reads only committed RunState and target-owned accepted Engine
objects, never W&B, and never triggers an evaluation. Keep online, offline, and
operator status and their configured K, `avg@k`, `pass@k`, and ranking score
separate; an unavailable or not-applicable metric remains empty rather than
being inferred from another score.

Do not routinely inspect every receipt, log, Ray lease, GPU sample, Judge job,
token summary, or W&B object. Escalate to the relevant detailed evidence only
when the snapshot identifies a concrete anomaly, such as a suspended, failed,
or recovering Run; an overdue active-command heartbeat or configured boundary;
an unhealthy `tracking/run-monitor-health.json`; or inconsistent committed
state and accepted artifacts. Use the per-Coordinator timeline, command receipt,
checkpoint/evaluation output, resource telemetry, Judge status, or W&B
reconciliation evidence needed to diagnose that anomaly.

Do not wait for the next ten-minute snapshot when the foreground supervisor
exits, the Run enters a terminal/suspended/failed state, or an explicit system
health alarm arrives. Handle that event immediately according to the recovery
or terminal procedure below. These event-driven actions are not an additional
polling loop. A healthy Run with no new transition at a snapshot is not by
itself stalled. Do not repeatedly re-read unchanged logs or restart a healthy
process to obtain a fresh snapshot.

Treat Base Model/P000 Baseline as the first runtime staging boundary. Before
calling a long Run healthy, inspect its complete evaluation Receipt, shard
results, behavior artifacts and logs for vLLM memory pressure,
scheduler/concurrency errors,
timeouts, missing W&B evidence, or mismatched scope. Do not infer P000 health
from a terminal process alone. For a seeded Run, require every selected imported
row and its terminal operator record in local state; do not schedule replacement
Base Model, P000 Baseline, or Search Trial evaluations merely because W&B is
delayed.

Training and online evaluation may overlap as declared by the runtime. Multiple
online evaluations may execute concurrently when resources are free, while W&B
evaluation publication remains ordered by checkpoint position. Operator
evaluation is a per-Trial side branch and may overlap Analyzer work. Analyzer
must visibly progress through Stage A, Review Worker, and Stage B; the Agent must
not poll or retry Review rows.

When training is configured with online validation, a checkpoint boundary can
temporarily stop training-step logs while validation materializes a compressed
behavior file. The writer uses a deployment-scoped
`model_behavior/online_validation/.epoch-<N>...tmp` path and renames it to
`epoch-<N>.jsonl.gz` only after the file is complete. During this boundary,
`trainer_log.jsonl` and the training command heartbeat may remain unchanged,
the training placement group may still be `CREATED`, and the training GPUs may
be idle; none of those observations alone is a stall or a retry trigger.

Before declaring such a command stalled, locate the expected online-validation
temporary file and take two size/mtime samples separated by a meaningful
interval. Continued growth is provider/artifact progress: keep the Attempt
running and wait for the final epoch file and the next training-step record.
If the temporary file is closed and the final file appears, inspect the
validation receipt before judging the next boundary. Only when the temporary
file is absent or unchanged and no other command, checkpoint, receipt, or
provider progress exists should the normal liveness/recovery rules be applied.

A temporary lack of transitions is not itself a failure. Use command heartbeat,
GPU activity, service logs, checkpoint production, and provider progress to
distinguish active work from a stall. Do not restart a healthy long-running
training or Analyzer call merely because its duration differs from a smoke Run.
During `recovering`, verify the old Attempt terminal/fence, exact lease reclaim,
dependency readiness, new Attempt identity and uninterrupted unrelated
Coordinator timelines. A temporary W&B failure retries in the background from
canonical local evidence; it never triggers retraining or fork.

## Repair authority and required record

The supervising agent has normal code and filesystem authority, but may repair
only a concrete implementation or environment defect supported by current Run
evidence. Reasonable repairs include ADE orchestration, backend integration,
service lifecycle, telemetry, and deterministic artifact plumbing. Run the
smallest focused check that would have detected the observed failure.

Do not edit canonical Run state, revisions, queues, receipts, accepted artifacts,
Agent outputs, or frozen config. Do not change dataset, model, seed, training
schedule, resource budget, evaluation/extraction/grading semantics, reward
meaning, or search budget as a repair. Such a change is a new experiment and
requires Operator direction.

Routine component restart, Attempt replay, Judge replacement, and W&B retry are
already recorded by transitions, Receipts and usage; do not write a manual
repair report for them. Every source, environment, or deployment-service
intervention, and every fork, must be logged under that source Run before
continuation:

```text
<run-root>/reports/repairs/YYYYMMDDTHHMMSSZ-short-slug.md
```

The record must contain:

- symptom and UTC time;
- exact state revision and relevant log/receipt/artifact paths;
- root cause and why the change is within operational semantics;
- changed files, environment action, and scoped Git commit;
- focused verification and its result;
- active Calls, Commands, Review jobs, and Ray leases at the repair boundary;
- the resume/fork decision and exact revision used;
- source and new Run IDs when a fork is required.

Do not bundle unrelated worktree changes into the repair commit. A source repair
uses this Run-owned checkout at a durable pause/suspension boundary, produces a
new exact revision, runs the focused validation, and follows the recovery/fork
decision below without asking for routine approval already granted by the prompt.

Assume that other supervising Agents may be working concurrently and may commit
independent fixes. Immediately before any pause-boundary source repair, re-read
the current Git `HEAD` and worktree instead of relying on the revision observed
earlier in the session. Apply the repair on top of that current HEAD, including
unrelated commits that landed since the Run started; do not reset to or build a
private branch from the old Run commit. Create a scoped repair commit and record
both the pre-repair and post-repair revisions. Before same-Run resume, re-read
`HEAD` again and start replacement workers from the exact newest revision that
contains the repair. Record that revision in the resume log. If concurrent
changes overlap and cannot be combined without a semantic decision, keep the
Run paused and resolve the overlap explicitly.

When a source repair changes code imported by an already-started Engine or Review
worker, treat that worker as stale: source edits are not hot-loaded. Do not
dispatch new work to it or rely on an unverified process restart. At the durable
pause boundary, fence the old Run-owned worker, verify that no stale worker can
claim the Run's deployment-scoped queue, and start a replacement worker from the
repaired Git `HEAD`. Record the replacement PID, startup time, Run ID, and loaded
Git revision before resuming. Use same-Run pause/resume when the old execution is
fenceable; use a typed fork only when the fork contract below is satisfied.

The real-Agent Pre-GPU runner supports both canonical training tasks. It selects
the scripted typed Engine handler from the resolved task: SFT/Data Selection uses
`train_sft`, and RFT/Reward Design uses `train_rft`. Both paths use the same
Control, Agent, queue, recovery, and terminal loop. These are CPU-only fake
backends and do not validate real SFT/RFT training or model throughput.

## Automatic recovery, suspension, fork, or new experiment

The normal path is same-Run automatic recovery. A retryable infrastructure or
dependency failure with no related scientific acceptance causes:

```text
running | bootstrapping
  -> automatic_recovery_started
  -> recovering
  -> automatic_recovery_attempt_submitted
  -> previous running | bootstrapping
```

The foreground Supervisor remains alive and unrelated Coordinators continue.
Verify that the frozen data/artifact/model/seed/reward/config are unchanged, the
old physical Attempt is terminal or explicitly fenced, only its exact logical
workload lease is reclaimed, resource admission is healthy, and recovery budget
remains. The replacement uses a new physical Attempt ID; never execute the same
physical Attempt twice and never resume a training-intermediate optimizer state.

Base, Trial, and Operator Evaluate Commands follow the same rule. A retryable
evaluation Receipt is audit-only and cannot become a degraded baseline,
scientific trial failure, or terminal operator verdict. Harness-owned malformed
Review packet/coverage similarly triggers restaging and a new Review Attempt
before scientific acceptance; `repairable=false` only means the Analyzer Agent
cannot repair it.

`suspended` is the escalation path, not the first response. It is valid only
after the resolved automatic-recovery budget/readiness window is exhausted, a
dependency remains unhealthy, or Control cannot prove safe continuation. The
supervising agent diagnoses and repairs within its initial authorization, checks
the exact terminal/fence and resource facts, and resumes the same Run without a
new human confirmation:

```bash
ade --project-root "$PROJECT_ROOT" run resume "$EXPERIMENT_CONFIG" \
  --deployment "$DEPLOYMENT_CONFIG" \
  --run-id "$RUN_ID"
```

Resume preserves the Run ID, W&B group, resolved config, accepted revisions,
logical sessions and durable pending work. Monitor restart exhaustion only
degrades telemetry; required component exhaustion must leave a typed durable
suspension rather than a falsely running Run.

For a planned source/environment/service intervention, prefer a boundary with
no active external execution or Ray lease and request a durable pause:

```bash
ade --runs-root "$CONTROL_ROOT" pause "$RUN_ID" \
  --reason "inspect and repair runtime dependency"
```

Do not kill an individual training subprocess. Pause terminalizes/fences current
physical work and preserves logical work as `retry_pending`; resume creates the
new Attempt. The stable Run monitor continues the same W&B history.

### Operator-issued pause, early-finish, and cancel instructions

Translate an Operator instruction by its scope and intent. Do not treat these
controls as synonyms:

- "pause" or "stop temporarily" means the existing Run `pause`; it preserves
  the same Run for a later resume;
- "finish Coordinator cNNN after K Plans" means graceful Coordinator finish;
- "finish Coordinator cNNN after its current allocated work" means graceful
  Coordinator finish without `--after-plans`;
- "limit every Coordinator in this Run to K Plans" means Run-level graceful
  finish;
- "terminate/cancel this Run" means irreversible Run-level `cancel`;
- "immediately cancel only Coordinator cNNN" requests Coordinator hard cancel.
  That command is not operationally available yet. Do not substitute Run
  `cancel`, do not kill its worker, and do not reinterpret it as graceful
  finish; report the unsupported control boundary to the Operator.

Use the `runtime_roots.control` value printed by `ade experiment inspect` as
`CONTROL_ROOT`. First inspect the current state and exact Coordinator IDs:

```bash
ade --runs-root "$CONTROL_ROOT" status "$RUN_ID"
```

The status response exposes each Coordinator's `control_status`, original,
requested and effective Plan limits, current allocated slots, request reason
and request revision. Only Search Coordinators such as `c001` are controllable;
never target bootstrap Coordinator `c000`.

For a graceful single-Coordinator finish at a total of `K` allocated Plan
slots, execute:

```bash
ade --runs-root "$CONTROL_ROOT" coordinator finish "$RUN_ID" \
  --coordinator-id "$COORDINATOR_ID" \
  --after-plans "$K" \
  --reason "$OPERATOR_REASON"
```

`K` is the total target for that Coordinator, not "K more Plans", and may be
zero. It cannot exceed the original per-Coordinator limit. If the Coordinator
already has more than `K` allocated slots, ADE does not roll them back: it
records requested `K` and uses the already allocated count as the reachable
effective floor. To stop creating slots immediately while allowing all current
work to drain, omit `--after-plans`:

```bash
ade --runs-root "$CONTROL_ROOT" coordinator finish "$RUN_ID" \
  --coordinator-id "$COORDINATOR_ID" \
  --reason "$OPERATOR_REASON"
```

For a Run-wide reduction to `K` Plans per Search Coordinator, execute:

```bash
ade --runs-root "$CONTROL_ROOT" run finish "$RUN_ID" \
  --plans-per-coordinator "$K" \
  --reason "$OPERATOR_REASON"
```

This atomically records the same requested target for all nonterminal Search
Coordinators. A Coordinator that already allocated more work retains that
allocated count as its effective floor. Neither finish command edits the frozen
experiment, original budget, accepted Plans, Trials, artifacts, PM/RM, ranking,
or W&B evidence. They only lower durable effective limits.

Coordinator and Run finish require Bootstrap to be complete, Run status
`running`, and no pending pause request. Resume a paused Run first; resolve a
suspended Run before issuing finish. A repeated command with the same target
and reason is idempotent. A different target/reason after intent is recorded is
rejected rather than silently rewriting Operator intent.

Both finish commands return after committing durable `finish_requested`
intent; they do not wait for terminal completion. Keep the foreground
Supervisor alive. Continue with:

```bash
ade --runs-root "$CONTROL_ROOT" status "$RUN_ID"
ade --runs-root "$CONTROL_ROOT" run history "$RUN_ID"
```

For one Coordinator, require `coordinator_finished_early` after all of its
already allocated work, summaries, evaluations and RM entries drain. For a
Run-level request, require every affected Coordinator to reach
`finished_early`. Final Run acceptance requires `status=completed`,
`completion_kind=operator_early_finish`, transition `run_finished_early`, all
Trials archived, and empty planning/RM/Agent/Engine/Review queues. Do not stop
the Supervisor merely because the request command returned successfully.

If finalization reports `snapshot revision is immutable`, treat it as a
finalize retry blocker: preserve the existing snapshot and inspect the
Supervisor `control.log`, `events.jsonl`, Run status, and the conflicting
snapshot manifest. Do not delete or overwrite the snapshot. If the Operator
intends to abandon the Run, use the cancellation path below.

### User-requested Run termination and cleanup

When the Operator explicitly requests termination of the current Run, use the
Run-scoped cancellation path. It first writes the durable `cancelled` fence,
then stops new dispatch and cleans only resources owned by that Run. It must
not stop or restart the deployment's shared Ray/recluster services or touch any
sibling Run. It must stop the exact Local Judge launch attached to the cancelled
Run.

Execute the irreversible Run-level cancellation with the Operator's stated
reason:

```bash
ade --runs-root "$CONTROL_ROOT" cancel "$RUN_ID" \
  --reason "$OPERATOR_REASON"
```

`cancel` is a top-level command; it is not under `ade run`. A repeated cancel
for an already-cancelled Run is an idempotent cleanup reconciliation and may be
used when the first receipt was incomplete.

Use this command only when the Operator intends to abandon the entire Run. It
is not a single-Coordinator command and it is not a temporary pause. After it
returns, inspect the terminal state and the cleanup receipt described below.

Cleanup proceeds in this order:

1. stop new Engine, Agent, Review, evaluation, and Judge dispatch;
2. send `SIGTERM` to the Run-owned supervisor worker process groups and wait
   for the grace period;
3. cancel Run-owned external jobs and terminalize only claimed queue entries
   whose command `run_id` matches the cancelled Run;
4. send `SIGKILL` only to still-live process groups proven to belong to the
   Run;
5. release the Run's GPU leases and Ray placement groups by owner/lease ID;
6. remove only exact temporary artifacts whose names begin with
   `<run-id>--` from the resolved system temp root (including the user-scoped
   temp directory used by Ray). This includes the cancelled Run's Ray temp
   files, but never the shared Ray session root or unrelated `/tmp` entries;
7. cancel remaining Judge jobs, stop the exact persisted gateway/eight-endpoint
   launch, require all recorded PIDs released, and remove that stopped launch's
   jobs/log root;
8. finish the Run Monitor and preserve local W&B/evaluation evidence;
9. write `reports/cleanup/run-cleanup.json`, including removed temporary paths
   and any unresolved cleanup errors.

Never use `pkill`, `killall`, a cluster-wide Ray shutdown, or GPU-only process
matching. A process that merely uses a GPU is not sufficient ownership proof.
If a process, lease, or placement group cannot be attributed to the Run, leave
it running and record it as unresolved cleanup evidence.

Cleanup is complete only when the receipt shows no Run-owned processes, queue
claims, GPU leases, or placement groups remaining, and
`shared_cluster_touched` is `false`. An `incomplete` receipt is an operator
repair condition, not permission to broaden the kill scope. After a process
exits during the grace period, repeat the same top-level `cancel` command to
reconcile the receipt; stale PIDs are discarded only when they no longer prove
ownership of a live Run process.

Fork is exceptional and requires positive typed evidence. The only causes are:

- `invalid_accepted_fact`: history contains an immutable AcceptedBoundary whose
  scientific fact is wrong;
- `unfenceable_execution`: the exact suspension revision retains the active
  logical work and records
  `attempt:<logical-work-ref>/<attempt-id>@writer:<backend-writer-ref>` in both
  `failure.evidence_ref` and transition `origin_refs`; that physical Attempt may
  still write and cannot be terminalized/fenced;
- `continue_cancelled`: the Operator explicitly continues a terminal cancelled
  Run.

An AcceptedBoundary is an exact `<run-id>@rev-NNNNNN` whose history entry exposes
`FactClass=scientific`, transition ID/kind, full ScopeKey/SubjectRef, logical work,
accepted fact refs, and origin Receipt/Delivery. Logs, process exit, failed
Attempt Receipt, `recovering`, or `suspended` are not accepted boundaries.

For an invalid accepted fact, cite the bad acceptance, not a hand-selected replay
revision:

```bash
ade --runs-root "$CONTROL_ROOT" run history "$RUN_ID"
ade --runs-root "$CONTROL_ROOT" run fork \
  --invalidate "$RUN_ID@rev-NNNNNN"
```

For `--unfenceable`, pass the exact suspension revision; for
`--continue-cancelled`, pass the exact cancelled terminal revision. ADE validates
the embedded typed Attempt/writer or cancelled transition, traces logical work to the pre-dispatch replay boundary,
and generates the child ID:

```text
<lineage-root-run-id>-f<generation>-r<replay-revision>-<short-unique-suffix>
```

There is no free `--new-run`. Fork revision 0 records lineage root, generation,
direct source, cause, invalid acceptance/unfenceable Attempt and replay boundary.
The child has its own W&B group equal to its child ID and lineage tags; source
W&B history stays with the source. It inherits only immutable accepted facts
before the replay boundary, never active execution, lease or backend session.

Start a completely new experiment when scientific config, input, intent, model,
data, seed, reward meaning, training/evaluation semantics or authorized resource
contract changes. Do not misuse fork as a configuration override.

## Final terminal cleanup

This section applies whenever the Run reaches the final status `completed`,
`failed`, or `cancelled`, whether naturally or after an Operator cancel. A
`paused` or `suspended` Run is resumable and must not delete its Supervisor
preflight evidence or stop resources needed for an authorized resume.

After the terminal snapshot is appended and the foreground Supervisor has
stopped normal dispatch, run one idempotent reconciliation from the required
environment:

```bash
"$ADE_PYTHON" \
  .agents/skills/ade-supervise-run/scripts/reconcile_terminal_run.py \
  --project-root "$PROJECT_ROOT" \
  --control-root "$CONTROL_ROOT" \
  --queue-root "$QUEUE_ROOT" \
  --run-id "$RUN_ID"
```

Require `<run-root>/reports/cleanup/run-cleanup.json` to report `status=complete`,
no unresolved processes, no queue claims, no GPU leases or external allocations,
no temporary-artifact errors, and `artifact_cache_cleanup.status` equal to
`complete` or `not_required`. This is the acceptance proof that checkpoint
staging consumers and Run-owned cache entries were released. A failed first
attempt may be repeated with the same command after the identified process or
service exits; a successful retry clears resolved errors while preserving the
cumulative list of removed resources.

Then remove the exact disposable preflight Run roots recorded in
`SUPERVISOR_PREFLIGHT_PATHS`. Repeat `--path` once for every recorded path; omit
all `--path` arguments only when the list is empty:

```bash
"$ADE_PYTHON" \
  .agents/skills/ade-supervise-run/scripts/cleanup_terminal_transients.py \
  --project-root "$PROJECT_ROOT" \
  --control-root "$CONTROL_ROOT" \
  --run-id "$RUN_ID" \
  --path "$PREFLIGHT_PATH_1" \
  --path "$PREFLIGHT_PATH_2"
```

The helper accepts only explicitly listed `runs/local-judge-preflight-*` paths
directly under this repository and writes
`reports/cleanup/supervisor-transients.json`. Require `status=complete` and each
requested path either removed or already absent. Never discover paths with a
broad glob at deletion time.

Finally execute `"$ADE_ENV/bin/ray" status --address "$RAY_ADDRESS"` again and
require the derived free-GPU capacity, no allocator lease or placement group for
this Run, and no Run-owned worker. Remove exact `<run-id>--*` files reported by
the cleanup receipt from the system/Ray temp area. Do not delete `/tmp/ray`, its
live `session_latest`, or another Run's logs. The normal Supervisor attaches to
an existing Ray cluster and therefore owns no Ray session directory; if an
exceptional repair explicitly launched a Ray runtime, its exact session path
and PIDs must have been recorded when created, and that one stopped session may
be removed only after all of its recorded processes are dead.

For a Judge-enabled Run, require the Run attachment state to be `detached`, the
matching deployment `service.json` to be `stopped`, every PID from that exact
launch handle to be dead, its gateway/eight endpoint ports no longer to report
that launch, and its transient jobs/log root to be absent. Do not use process
name matching. Append one concise terminal-cleanup entry with both cleanup
receipt paths and the post-cleanup Ray/Judge result to `monitor.md`. The
Supervisor must not exit or claim terminal acceptance while either receipt is
incomplete.

## Terminal acceptance

Do not declare success from `status=completed` alone. Verify:

- configured Coordinator, Plan, and Trial budgets closed as resolved;
- every Coordinator's full-SubjectRef timeline closed independently, every
  accepted Plan is present once in the Plan Catalog, and every RM merge occurred
  once in serial order;
- every Trial has its resolved task-specific checkpoints, online evaluations,
  offline evaluation, terminal operator record and result when completed,
  artifact manifest, Analyzer outputs, PM/RM updates, Ranking entry, and archive
  transition; SFT additionally requires selection/pool artifacts, while RFT
  additionally requires rollout and reward evidence;
- the Base Model has its accepted offline result, RFT position-0 online result
  when applicable, and terminal operator record; the P000 Baseline has its
  selected checkpoint, online/offline results, and terminal operator record;
  referenced Base Model/P000 Baseline rows retain imported provenance and
  target-owned result refs, with no replacement evaluation commands;
- `reports/results.csv` is at the terminal revision and agrees with the accepted
  Base Model, P000 Baseline, Search Trial, and operator records;
- no Agent, Engine, or Review command remains active;
- no Run remains in `recovering`; every automatic recovery has terminal old/new
  Attempt evidence and did not duplicate a scientific acceptance;
- all Run-owned Ray leases, placement groups, child processes, exact temp paths,
  and checkpoint staging/cache consumers are released, with a complete
  `reports/cleanup/run-cleanup.json`;
- the internal Run Monitor is stopped cleanly with healthy final tracking and
  retained local GPU/token/timeline evidence, and the supervising Agent has
  appended the final terminal snapshot to `<run-root>/monitor.md`;
- the W&B project lists the complete exact group: one canonical Base Model
  evaluation stream, one canonical P000 Baseline evaluation stream, every
  imported or newly executed Search Trial evaluation stream, target-native
  training streams, operator evaluations, the Run monitor, and only selected
  source non-evaluation history under `seed-import`/`imported-history`;
- the Run Monitor's local and remote W&B identities match the resolved
  project/group/external-ID tuple before Bootstrap is accepted;
- W&B training histories contain backend metrics, evaluation histories preserve
  checkpoint order, and artifacts can be read back;
- when resolved `judge_enrichment.enabled=true`, no queued/running Judge job
  remains, the exact gateway/eight-endpoint launch is stopped, its recorded PIDs
  are dead, and its transient jobs/log root is absent;
- every Supervisor-created `runs/local-judge-preflight-*` directory is removed
  with a complete `reports/cleanup/supervisor-transients.json` receipt;
- every source/environment/service intervention and fork has its Run repair
  record and scoped commit; routine replay/restart records come from structured
  history rather than handwritten reports.

Report the Run ID, terminal revision/outcome, exact Git commit, experiment config
digest, W&B project/group, artifact and Receipt locations, resource cleanup,
stopped Judge status, per-Coordinator closure, token/usage completeness, any
partial-but-terminal provider rows, and all repair/fork lineage. Mark the
long-running goal complete only after this acceptance is actually satisfied.
