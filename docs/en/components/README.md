# Component implementation and usage

[Project README](../../../README.md) | [Operations](../operations.md)

These guides explain the current source boundaries, inputs/outputs, configuration, usage and diagnosis. Start with architecture, then follow the component that owns the behavior you need. The three task pages define the generated-code surfaces separately. Internal Python classes are implementation entry points, not a promise of an independently versioned public SDK.

Commands run from the repository root. All examples and formal commands use `.unified-vllm-0.19.1-verl-venv`; formal Runs also require prepared inputs and deployment. `--help` only prints options; commands that advance work or enqueue evaluation are identified in their pages. The CPU example validates its scripted path, not real backends or services.

| Component | Coverage |
| --- | --- |
| [Operator-side supervision](supervision.md) | Operation Prompts, Skills, monitoring and terminal acceptance |
| [Architecture and concepts](architecture.md) | Object ownership and one Trial flow |
| [Control and Reducer](control.md) | Scheduling, typed Outcomes and accepted state |
| [Agent runtime and roles](agent-runtime.md) | Context, Skill, delivery and attempts |
| [Engine and training backends](engine.md) | Queues, SFT/RFT, checkpoints and receipts |
| [Data Selection](data-selection.md) | Selector interface and realization |
| [Reward Design](reward-design.md) | Reward contract, Judge and Group Credit |
| [Curriculum Learning](curriculum-learning.md) | Schedule interface and independent baseline |
| [Review, Rubric and Judge](review-judge.md) | Review evidence and model service boundaries |
| [Memory and Workspace](memory-workspace.md) | Immutable publications and working deliveries |
| [Configuration and CLI](configuration-cli.md) | Layer ownership and actual entry points |
| [Evaluation and analysis](evaluation-analysis.md) | Evaluation purposes, matrices and reports |
| [Monitoring and recovery](recovery.md) | Retry, pause/resume, Seed, fork and cleanup |

## Design authority and validation

These guides describe the supported component contracts and their implementation entry points. Start with [Control](control.md) and [recovery](recovery.md), then inspect the code relevant to a change.
