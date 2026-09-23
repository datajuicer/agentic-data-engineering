"""Launch the pinned Judge stack, isolated from training's Transformers/API.

Ports the existing production vllm_compat launcher into ADE. The overlay is
installed from requirements/judge.txt; no sibling repository is required.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


def _patch_prometheus_route_matching() -> None:
    from prometheus_fastapi_instrumentator import routing
    from starlette.routing import Match, Mount

    def compatible_get_route_name(scope, routes, route_name=None):
        for route in routes:
            matches = getattr(route, "matches", None)
            path = getattr(route, "path", None)
            if not callable(matches) or path is None:
                continue
            match, child_scope = matches(scope)
            if match == Match.FULL:
                current_name = path
                child_scope = {**scope, **child_scope}
                if isinstance(route, Mount) and route.routes:
                    child_name = compatible_get_route_name(
                        child_scope, route.routes, current_name
                    )
                    current_name = None if child_name is None else current_name + child_name
                return current_name
            if match == Match.PARTIAL and route_name is None:
                route_name = path
        return route_name

    routing._get_route_name = compatible_get_route_name


def main() -> None:
    packages = Path(sys.prefix) / "judge-packages"
    if not (packages / "transformers").is_dir():
        raise SystemExit("Judge dependencies are missing; run scripts/recreate_unified_vllm_env.sh")
    sys.path.insert(0, str(packages))
    # vLLM inspects models in a fresh Python process; it needs the same overlay.
    os.environ["PYTHONPATH"] = str(packages) + (
        os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
    )

    # vLLM 0.19.1 supplies two arguments to this Transformers v4 helper.
    from transformers import configuration_utils

    original = configuration_utils.layer_type_validation

    def compatible_layer_type_validation(layer_types, _num_hidden_layers=None):
        return original(layer_types)

    configuration_utils.layer_type_validation = compatible_layer_type_validation
    _patch_prometheus_route_matching()

    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
