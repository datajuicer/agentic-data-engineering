# OpenThoughts

Utilities for `data/raw_data/OpenThoughts-114k`.

Scripts:

- `create_openthoughts_domain_sharegpt.py`: sample math/code/science ShareGPT JSONL files.
- `create_openthoughts_mixed_sharegpt.py`: sample one 3840-row math/code/science
  mix using the original matched-domain ratio.
- `stat_openthoughts_chat_template_token_lengths.py`: compute token-length statistics for generated JSONL files.

Create the proportional 3840-row mix:

```bash
PYTHONPATH="$PWD" ./.unified-vllm-0.19.1-verl-venv/bin/python \
  dataset/openthoughts/create_openthoughts_mixed_sharegpt.py
```
