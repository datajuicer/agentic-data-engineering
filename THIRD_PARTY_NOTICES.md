# Third-party source

ADE includes backend source and evaluation helpers. Keep their license files, copyright headers and source references when redistributing these components. This index does not select or replace the ADE project license, which is still pending in [LICENSE](LICENSE).

| Included source | Source record | Included license |
| --- | --- | --- |
| `third_party/llamafactory/` | [Upstream and component documentation](third_party/llamafactory/README.md) | [Apache-2.0](third_party/llamafactory/LICENSE) |
| `third_party/verl/` | [Component documentation](third_party/verl/README.md); [recorded source pins](requirements/verl/source-pins.json) | [Apache-2.0](third_party/verl/LICENSE) |
| `ade/engine/eval/utils/hmmt_matharena/` | Copyright notice names SRI Lab, ETH Zurich | [MIT](ade/engine/eval/utils/hmmt_matharena/LICENSE) |
| `ade/engine/eval/utils/qwen_math/` | [Grader source references](ade/engine/eval/utils/qwen_math/grader.py) | No separate license file is currently included in this directory; source/license attribution remains a publication follow-up. |

The bundled backends are the ADE-imported source snapshots, not an automatic download of current upstream releases. The source import was based on ADE commit `07753c245c57469312ed6d796e06b075ace6ce98`, without its Git history or runtime artifacts. The VERL pin file is an inherited source record, not proof that every bundled file is an unmodified upstream file. The installation-time TransferQueue change is explicit in [the patch script](scripts/patch_transfer_queue_namespace.py).

Downloaded models, datasets and Python distributions are not covered by the ADE license placeholder. Their source pages and package metadata provide their respective terms; [input preparation](docs/en/data-preparation.md) links to the configured data and model sources.
