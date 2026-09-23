# Canonical MATH train/validation split

Build the canonical RFT training artifact and the MATH validation benchmark
from the original Hugging Face MATH train split:

```bash
.unified-vllm-0.19.1-verl-venv/bin/python \
  dataset/math/build_math_train_validation.py
```

The defaults are the production contract: the pinned
`EleutherAI/hendrycks_math` train split (7,500 rows), a seed-42 500-row
validation sample stratified to the exact MATH500 `(subject, level)` counts,
and a 3,072-row training sample drawn from the remaining 7,000 candidates. MATH500 is read only for its
target distribution and no-overlap check.

The training artifact is written to
`data/rft/math/math_train3072_val500_seed42/train.parquet`; the validation
benchmark is written to `data/benchmarks/math-val` as a Hugging Face disk
dataset with the `test` split. The resulting `split.manifest.json` records the
source revision, target/actual distributions, output digest, and overlap
counts.

The validation benchmark is registered as `math_val`. It uses the same
MATH/Qwen boxed prompt and Qwen math reference/grader contract as `math_500`, while
remaining disjoint from the MATH500 test set.
