# Models, training data and benchmarks

[Quickstart](quickstart.md) | [Deployment](deployment.md) | [Operations](operations.md) | [Results](results.md)

These instructions prepare inputs for the formal configurations. They require the [full runtime](installation.md), network access for downloads, and storage for the selected model/data. Run from the repository root with that environment activated. The CPU demo creates its own synthetic inputs and does not need these downloads.

## Choose the preparation steps

Prepare inputs before starting a Run; training reads the paths in the selected configuration. The commands below use the supplied configurations' defaults. If you change input paths, update the corresponding configuration as well.

| Task | Training data preparation | Benchmark preparation |
| --- | --- | --- |
| Math SFT | Download OpenThoughts, then run `dataset/openthoughts/create_openthoughts_mixed_sharegpt.py` | `scripts/build_benchmark_catalog.py aime24 aime25` |
| Code SFT | Use the same OpenThoughts preparation | `scripts/build_benchmark_catalog.py livecodebench_le_2024_03_30 livecodebench_gt_2024_03_30` |
| Reward Design / Curriculum Learning | Build `math_500`, then run `dataset/math/build_math_train_validation.py` | The split script creates `math_val`; validate both with `scripts/build_benchmark_catalog.py math_val math_500 --check` |

Run these scripts with `python` in the installed environment. The sections below include download commands, output paths and the required execution order. Existing prepared inputs can be reused; `--check` validates benchmark artifacts without downloading them.

## Match the task configuration

Paths below are relative to the repository root. Keep the paths or edit the corresponding task configuration before starting a Run.

| Task configuration in `configs/tasks/` | Base model | Training input | Validation → operator test |
| --- | --- | --- | --- |
| `formal-long-cot-data-selection.yaml` | `models/Qwen2.5-7B-Instruct` | OpenThoughts mixture below | `aime24` → `aime25` |
| `formal-long-cot-data-selection-code.yaml` | Same | Same | `livecodebench_le_2024_03_30` → `livecodebench_gt_2024_03_30` |
| `formal-math-reward-design.yaml` | `models/Qwen2.5-0.5B` | MATH training Parquet below | `math_val` → `math_500` |
| `formal-math-curriculum-learning.yaml` | Same | Same | `math_val` → `math_500` |

The SFT tasks share a fixed 3,840-row math/code/science pool and select 384 rows. They do not use a different training pool for each evaluation domain. Do not change catalog IDs or splits merely to satisfy a missing-file error.

## Download the selected model

Sources: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) for SFT, [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) for RFT. Choose an explicit source revision and set `ADE_MODEL_REVISION` before executing this example. Use a local directory that matches your task:

```bash
export ADE_MODEL_REPO=Qwen/Qwen2.5-7B-Instruct
export ADE_MODEL_DIR=models/Qwen2.5-7B-Instruct
# Set ADE_MODEL_REVISION to your selected Hugging Face commit revision.
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ['ADE_MODEL_REPO'],
    revision=os.environ['ADE_MODEL_REVISION'],
    local_dir=os.environ['ADE_MODEL_DIR'],
)
PY
```

For RFT set the repository to `Qwen/Qwen2.5-0.5B` and the directory to `models/Qwen2.5-0.5B`. Download the complete model snapshot, including tokenizer/configuration files. For the Judge, the public deployment example points to [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B); prepare its snapshot separately on the serving machine and set that machine's model path in your deployment YAML. The Judge download is not required for the scripted CPU demo.

Record the source revision for each model and dataset you download. Keep local paths consistent with the selected task configuration and observe each source’s license and access conditions.

## OpenThoughts mixture for SFT

Download [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) with both its `data/` and `metadata/` Parquet files. Set `ADE_OPENTHOUGHTS_REVISION` to your chosen source commit first:

```bash
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='open-thoughts/OpenThoughts-114k',
    repo_type='dataset',
    revision=os.environ['ADE_OPENTHOUGHTS_REVISION'],
    local_dir='data/raw_data/OpenThoughts-114k',
)
PY
python dataset/openthoughts/create_openthoughts_mixed_sharegpt.py
```

The converter joins the conversation and metadata inputs and samples math/code/science in the matched-domain proportions, using seed 42 and 3,840 rows. Its default output is:

```text
data/sft/openthoughts/openthoughts_mcs_3840_proportional.jsonl
```

Use `--help` for source/output overrides. The formal task points at this exact JSONL path.

## Build the evaluation artifacts

The authoritative source revisions, local formats, filtering rules and derived subsets are in [the benchmark catalog](../../configs/benchmarks/catalog.yaml). Build only the datasets needed by your selected experiment. The builder uses those contracts; `--check` checks existing artifacts without downloading or writing.

SFT math:

```bash
python scripts/build_benchmark_catalog.py aime24 aime25
python scripts/build_benchmark_catalog.py aime24 aime25 --check
```

SFT code:

```bash
python scripts/build_benchmark_catalog.py livecodebench_le_2024_03_30 livecodebench_gt_2024_03_30
python scripts/build_benchmark_catalog.py livecodebench_le_2024_03_30 livecodebench_gt_2024_03_30 --check
```

Additional benchmarks selected by an experiment or evaluation configuration must also be built. For example, build `math_500` before its derived `math_100` subset. Avoid invoking the builder without IDs when you only need one task; that requests the entire catalog.

## MATH training and validation for RFT

Build MATH-500 first, then use the dedicated split script:

```bash
python scripts/build_benchmark_catalog.py math_500
python dataset/math/build_math_train_validation.py
python scripts/build_benchmark_catalog.py math_val math_500 --check
```

The split script uses the pinned `EleutherAI/hendrycks_math` training source at revision `21a5633873b6a120296cce3e2df9d5550074f4a3`. It draws a 500-row validation set matching the MATH-500 subject/level distribution, then selects 3,072 training rows from the remaining source examples with seed 42. It produces:

- `data/rft/math/math_train3072_val500_seed42/train.parquet`
- `data/benchmarks/math-val/validation.parquet`
- `data/rft/math/math_train3072_val500_seed42/split.manifest.json`

`math_val` is a locally derived artifact. The benchmark builder alone does not create this split: run the dedicated script before `--check`.

## Final generalization inputs

The task table above lists in-loop validation and Run-owned operator test. Final generalization uses four held-out datasets per task in a separate operation; follow the [suite table, preparation commands and request template](generalization.md). `math_val` is MATH-S and is also needed for math SFT generalization.

## Continue to deployment

Prepare the inputs selected by your task configuration and run the benchmark catalog checks. Then follow [deployment configuration](deployment.md) to set credentials, paths and Ray resources.
