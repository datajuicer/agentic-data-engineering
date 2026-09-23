"""Task-neutral Experiment Package analysis profiles."""

from __future__ import annotations

import re
from typing import Any


_PROFILE = re.compile(r"^[a-z][a-z0-9_]*$")


def analysis_profile(
    profile: str,
    *,
    version: int,
    topology: dict[str, Any],
) -> dict[str, Any]:
    value = {
        "profile": profile,
        "schema_version": f"ade.{profile}_experiment.v{int(version)}",
        "topology": topology,
    }
    validate_analysis_profile(value)
    return value


def validate_analysis_profile(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Experiment analysis profile must be an object")
    profile = str(value.get("profile") or "")
    schema_version = str(value.get("schema_version") or "")
    topology = value.get("topology")
    if not _PROFILE.fullmatch(profile):
        raise ValueError("Experiment analysis profile ID is invalid")
    expected_prefix = f"ade.{profile}_experiment.v"
    if not schema_version.startswith(expected_prefix):
        raise ValueError(
            "Experiment analysis schema_version does not match profile"
        )
    version = schema_version.removeprefix(expected_prefix)
    if not version.isdigit() or int(version) < 1:
        raise ValueError("Experiment analysis schema version is invalid")
    if not isinstance(topology, dict):
        raise ValueError("Experiment analysis topology must be an object")
    return value
