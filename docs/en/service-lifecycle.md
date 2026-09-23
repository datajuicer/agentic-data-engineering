# ADE, Agent and Judge configuration and lifecycle

[Deployment](deployment.md) | [Operations](operations.md)

Use the full environment on the deployment head and run commands from the repository root. The public entry points are `ade experiment inspect`, `ade run start` and `ade run resume`; complete commands, baseline/Seed prerequisites and monitoring commands are in [operations](operations.md). Ray node setup is a separate operator action described in [Ray cluster](ray-cluster.md).

The operator hands an [Operation Prompt](quickstart.md) to a [supervising Agent](components/supervision.md), which owns preflight, monitoring and acceptance. The Supervisor below is ADE's Python process manager; it is launched by that Agent.

## Configuration ownership

| Component | Configuration | Execution |
| --- | --- | --- |
| ADE | `configs/experiments/` selects task, inputs, budgets, Agent and Engine settings; `--deployment` binds machine resources | Supervisor and its component workers use the Python interpreter running ADE |
| Agent | Experiment `agent`: backend, model, reasoning effort, sandbox, approval policy, retries and heartbeat; `artifact_builder.max_reflections` controls repair | The supported backend executes `codex exec`; the executable is found in the inherited PATH, with `~/.local/bin` prepended when present |
| Judge | Deployment `run_resources.local_judge`: model path/identity, GPU count, gateway port, authorization environment name, timeouts, vLLM and generation settings | vLLM uses the explicit `vllm.executable`; the gateway uses ADE's Python interpreter |
| Credentials | Exported environment or ignored repository `.env`, using `.env.example` as a template | Exported values take precedence; `.env` fills missing variables and does not expand shell expressions |

The experiment compiler combines the experiment and deployment, validates their contract and derives runtime roots. Inspection can read input files but does not start services. Resume checks the persisted configuration binding: changing YAML is not a way to change an active Run's scientific settings.

Agent authentication belongs to the operating account's backend setup. ADE does not provision it. Control and independent Agent workers share one runtime construction function, so model, permissions and workspace settings are identical. Control schedules external calls; the Agent worker performs execution. Role Skills come from `.agents/skills`; delivery files and receipts remain subject to ADE validation.

Use the installed `ade-judge-vllm` executable in deployment configuration, as shown in [the public example](../../configs/deployments/example.yaml). Its dependency overlay belongs to the full installation. The current Judge topology reserves eight GPUs on one automatically selected node through a named detached actor in the same Ray cluster, and eight local vLLM endpoints on ports 8901–8908 plus the configured gateway port. Supply the value of the named authorization variable before starting a Run. The launcher reports the missing variable's name, never its value. Analyzer Review is a separate worker and transport; it is not the Agent backend or the reward gateway.

## Startup and ownership

1. The operator installs dependencies, prepares inputs and credentials, configures the deployment and starts Ray nodes.
2. `ade run start` compiles the configuration, resolves deployment roots and checks for conflicting deployment workers. It initializes Run state and admits resources, starting or reusing the deployment Judge and waiting for readiness.
3. The supervisor launches Monitor, Engine, Agent, Review and Control processes, records worker provenance and captures their logs. N=1 uses one Engine/Agent/Review set. Multiple Coordinators receive separate worker sets and an additional global Agent worker.
4. Control advances durable state. Workers execute queued work and publish results; the supervisor monitors exits and applies bounded recovery. It does not start or stop Ray nodes.

Do not start component workers manually alongside `ade run start`. Low-level `control run`, `agent worker`, `engine worker` and `review worker` are composition/debugging entry points; they are not additional setup steps.

Supervisor passes the compiled deployment address as `RAY_ADDRESS` to every component, including restarted workers. Engine entry points connect to that address before processing commands; when invoking a debug `engine worker` or `engine run-one` directly, export the same explicit `RAY_ADDRESS`.

Debug Review entry points also require `--runs-root` for the accepted Run state; Supervisor supplies it automatically. Judge host and gateway addresses come from admission, not the deployment input.

## Pause, recovery and shutdown

| Boundary | ADE / Agent | Judge |
| --- | --- | --- |
| Pause request or first supervisor SIGINT/SIGTERM | Stop admitting new work, drain to a durable pause; supervisor then closes component processes | Retain the deployment service and attachment; pause does not free Judge GPUs |
| Resume | Continue the same Run's persisted work after configuration and admission checks | Reuse a healthy matching launch; replace an unhealthy or mismatched launch |
| Automatic recovery | Bounded worker/attempt recovery; exhausted recovery suspends the Run | Admission can recover the same Run's service |
| Completion, failure or cancellation | Terminal Run cleanup targets recorded work/processes; inspect the cleanup receipt | Detach and stop the exact persisted gateway/vLLM launch, then remove its transient jobs/logs |

A second interrupt forces supervisor shutdown and is not evidence of completed cleanup. A healthy Judge can also be transferred through the explicit safe-boundary handoff contract; an unrelated new Run cannot take an attached deployment. Use the documented pause/resume/cancel commands, not manual state edits or process-name killing.

## State and diagnostics

With default roots, let `<deployment>` be `runs/deployments/<cluster-id>`; confirm it using configuration inspection.

| Location | Contents |
| --- | --- |
| `<deployment>/control/<run-id>/` | Run state, Agent workspaces, Memory and reports |
| `<deployment>/control/<run-id>/services/supervisor/` | Component logs |
| `<deployment>/control/<run-id>/reports/worker-provenance.json` | Worker process identities and launch commands |
| `<deployment>/control/<run-id>/reports/cleanup/run-cleanup.json` | Terminal cleanup outcome |
| `<deployment>/run-services/` | Deployment Judge service handle and per-Run attachments |
| `<deployment>/local-judge/deployment-<cluster-id>/<launch-id>/` | Live Judge logs and transient jobs; exact paths are in the service handle |

The service handle records the Ray actor name, node ID and launch identity. A detached actor outlives its Supervisor. Enable Ray child-process-tree cleanup as described in the cluster guide. Stop gateway/vLLM children before releasing the actor at the terminal boundary.

Judge launch logs are transient and may be removed after stop/replacement; retain needed diagnostics before terminal cleanup. Durable Run reports and receipts are the result record. If startup fails, distinguish configuration/input errors, Judge admission/readiness errors and worker failures using the error and these paths. Do not infer successful cleanup solely from a terminal Run status.

Implementation: [composition](../../ade/harness/wiring.py), [supervisor](../../ade/harness/supervisor.py), [admission wiring](../../ade/harness/processes.py), [Judge lifecycle](../../ade/local_rubric_judge/lifecycle.py), [launcher](../../ade/local_rubric_judge/launcher.py), [environment loading](../../ade/harness/environment.py).
