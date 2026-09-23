# Engine backend sources

| Variant | Capability | ADE adapter | Source |
| --- | --- | --- | --- |
| LlamaFactory | SFT | [llamafactory.py](../ade/engine/backends/llamafactory.py) | [llamafactory/](llamafactory/) |
| VERL | RFT | [verl.py](../ade/engine/backends/verl.py) | [verl/](verl/) |

Configure each variant through `runtime.backends.sft` or `runtime.backends.rft`.
Both use the repository-owned full environment, installed by
[`scripts/recreate_unified_vllm_env.sh`](../scripts/recreate_unified_vllm_env.sh).
SFT and RFT have distinct training and checkpoint contracts; changing the backend
name does not convert one training mode into the other.

See [Engine documentation](../docs/en/components/engine.md) and
[third-party notices](../THIRD_PARTY_NOTICES.md).
