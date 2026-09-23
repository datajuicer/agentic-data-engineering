# Reward Design task

[Components](README.md)

## Inputs and output contract

The task fixes the model, training data, prompt, trusted outcome scorer and RFT recipe. The Builder delivers `reward.py` and `design.md`. Generated `compute_score` is asynchronous and takes exactly:

```text
question_prompt, response_content, extracted_answer, outcome_score,
response_length_tokens, max_response_length_tokens
```

`compute_fallback_score` uses the same parameter list. The returned `ade.reward_result.v2` object has `schema_version`, `score`, `outcome_score`, `artifact_projection` and `rule_evidence`. Score bounds, preservation of trusted outcome and fallback semantics are enforced by the result contract. Fallback preserves outcome with a null artifact projection; it is not arbitrary zero reward.

When Judge enrichment is enabled, code uses the injected awaited `llm_judge` capability under the compiler's call contract. Credentials and gateway construction remain Harness-owned. When Group Credit is enabled, the additional `assign_group_credit(group_input)` interface operates on complete rollout groups; pre-group score must preserve outcome. Input/output schemas and evidence-source rules live in `group_credit.py`, not in an informal Agent convention.

## Implementation flow

Role decoding and static compilation validate source, supporting design and reward/Judge behavior. Engine binding supplies the accepted artifact to the VERL runtime and reward manager. Trusted extraction/outcome scoring feeds the generated function; group credit and reward traces record how final training reward was assigned. The RFT handler manages checkpoint evaluation and publishes evidence for analysis. The Agent cannot redefine the benchmark scorer through its reward proposal.

## Configure and use

Start from `configs/tasks/formal-math-reward-design.yaml` and the corresponding formal experiment family. Prepare the MATH data/model and inspect with the full environment:

```bash
ade experiment inspect configs/experiments/math-math-rft-reward-design-baseline-formal.yaml --deployment configs/deployments/local.yaml --run-id reward-inspect
```

Then use the baseline → Seed → search commands in [operations](../operations.md), substituting this family's configs. A sample reward copied outside its task binding is not a valid formal run. Read the exact role input and compiler contract before editing a proposal; Group Credit changes the required result semantics.

## Inspect and diagnose

Follow accepted reward code, design and Engine artifact references from the Trial Record. For unexpected learning behavior, inspect trusted outcome, process evidence, pre-group reward and final training reward in the published traces before interpreting aggregate loss. Signature/schema violations belong to the proposal; missing Judge evidence or timeout belongs to the configured execution capability; invalid group outputs are checked at the group boundary. Scientific improvement is determined by accepted evaluation, not reward magnitude alone.

## Implementation entry points

| Source | Responsibility |
| --- | --- |
| [ade/tasks/reward_design/plugin.py](../../../ade/tasks/reward_design/plugin.py) | Task registration |
| [ade/tasks/reward_design/role_contracts.py](../../../ade/tasks/reward_design/role_contracts.py) | Reward and supporting delivery |
| [ade/tasks/reward_design/compiler.py](../../../ade/tasks/reward_design/compiler.py) | Source and capability admission |
| [ade/tasks/reward_design/rewards/contracts.py](../../../ade/tasks/reward_design/rewards/contracts.py) | Function and reward result schemas |
| [ade/tasks/reward_design/group_credit.py](../../../ade/tasks/reward_design/group_credit.py) | Complete-group credit assignment |
| [ade/tasks/reward_design/verl_reward_manager.py](../../../ade/tasks/reward_design/verl_reward_manager.py) | Training reward integration |
| [ade/tasks/reward_design/process_evidence.py](../../../ade/tasks/reward_design/process_evidence.py) | Process evidence adapter |
| [ade/tasks/reward_design/engine_binding.py](../../../ade/tasks/reward_design/engine_binding.py) | VERL input binding |
