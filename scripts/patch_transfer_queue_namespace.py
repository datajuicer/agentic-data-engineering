from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


_DECLARATION = '_TQ_NAMESPACE = os.environ.get("TRANSFER_QUEUE_NAMESPACE", "transfer_queue")'
_REPLACEMENTS = (
    (
        '_TQ_CONTROLLER: Any = None\n',
        f'_TQ_CONTROLLER: Any = None\n{_DECLARATION}\n',
    ),
    (
        'ray.get_actor("TransferQueueController", namespace="transfer_queue")',
        'ray.get_actor("TransferQueueController", namespace=_TQ_NAMESPACE)',
    ),
    (
        'name="TransferQueueController", namespace="transfer_queue"',
        'name="TransferQueueController", namespace=_TQ_NAMESPACE',
    ),
)


def patch_interface(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    if _DECLARATION in source:
        for _, replacement in _REPLACEMENTS[1:]:
            if replacement not in source:
                raise RuntimeError(f"partial TransferQueue namespace patch: {path}")
        return False

    updated = source
    for original, replacement in _REPLACEMENTS:
        if updated.count(original) != 1:
            raise RuntimeError(
                f"TransferQueue 0.1.8 source does not match expected namespace site: {original}"
            )
        updated = updated.replace(original, replacement)
    path.write_text(updated, encoding="utf-8")
    return True


def installed_interface() -> Path:
    spec = importlib.util.find_spec("transfer_queue.interface")
    if spec is None or spec.origin is None:
        raise RuntimeError("TransferQueue interface module is unavailable")
    return Path(spec.origin).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", type=Path)
    args = parser.parse_args()
    path = args.interface.resolve() if args.interface else installed_interface()
    changed = patch_interface(path)
    print(f"transfer_queue_namespace_patch={'applied' if changed else 'present'} path={path}")


if __name__ == "__main__":
    main()
