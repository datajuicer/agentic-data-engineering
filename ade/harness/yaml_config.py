"""Strict YAML loading helpers for versioned ADE configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects silently shadowed mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML mapping without permitting duplicate mapping keys."""

    value = yaml.load(
        path.read_text(encoding="utf-8"),
        Loader=_UniqueKeySafeLoader,
    )
    if not isinstance(value, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return value
