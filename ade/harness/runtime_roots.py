"""Deployment-scoped runtime-root derivation."""

from pathlib import Path


def deployment_runtime_roots(
    project_root: str | Path,
    resolved: dict[str, object],
) -> dict[str, Path]:
    deployment = resolved.get("deployment")
    if not isinstance(deployment, dict) or not isinstance(
        deployment.get("id"), str
    ):
        raise ValueError("supervised Run requires a resolved deployment ID")
    deployment_id = str(deployment["id"])
    if not deployment_id or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for char in deployment_id
    ):
        raise ValueError("resolved deployment ID is not a safe path segment")
    root = Path(project_root).resolve() / "runs" / "deployments" / deployment_id
    return {
        "deployment": root.resolve(),
        "control": (root / "control").resolve(),
        "queue": (root / "queue").resolve(),
        "objects": (root / "objects").resolve(),
        "engine_work": (root / "engine-work").resolve(),
        "review": (root / "review").resolve(),
        "review_work": (root / "review-work").resolve(),
    }
