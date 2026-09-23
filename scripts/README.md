# Script entry points

| Entry | Purpose |
| --- | --- |
| `recreate_unified_vllm_env.sh` | Full ADE/LlamaFactory/VERL installation, including Judge packages. Use `--help` for settings. |
| `patch_transfer_queue_namespace.py` | Installed-package patch called by the installer; normally no manual step. |
| `build_benchmark_catalog.py` | Prepare selected benchmark artifacts; see the data guide. |
| `prepare_rft_guru_3k.py` | Dataset-specific preparation utility, not an environment installer. |
| `check_generalization_data.py`, `check_generalization_prompt_parity.py` | Focused existing generalization checks. |
| `probe_*.py`, `run_ade_workflow_smoke.py` | Existing development/validation tools; some invoke real services or GPU work. Not setup steps. |
| `cleanup_ckpt_staging.sh` | Historical date-bound cache maintenance; ignores usage markers. Not a general deployment cleanup command. |

Ray operations use `ade cluster`, as described in the [Ray guide](../docs/en/ray-cluster.md). The command executes only on the current node. Run/Agent/Judge supervision uses `ade run start`; stopping Ray is a separate node operation.

Data preparation: [input preparation guide](../docs/en/data-preparation.md). No probe, cleanup script or benchmark download is run automatically by the cluster CLI.

Backend probes resolve Judge placement through Ray admission. Curriculum probes only read an already running deployment service from `runs/deployments/<deployment>/run-services/service.json`; they do not launch it. Pass an explicit `--deployment` to generalization checks. The generalization audit config and prompt-parity case list contain synthetic example identities. Replace their checkpoint and request paths with artifacts from your own Runs before use.
