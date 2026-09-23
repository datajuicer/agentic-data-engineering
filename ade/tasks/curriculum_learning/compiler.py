"""Static admission for Curriculum Learning artifacts."""

from __future__ import annotations

from ade.tasks.contracts import ArtifactCompilationRequest, CompiledArtifact
from ade.tasks.curriculum_learning.schedule_contract import validate_curriculum_source


def compile_artifact(request: ArtifactCompilationRequest) -> CompiledArtifact:
    if request.delivery.kind != "curriculum_learning_proposal":
        raise ValueError(
            "Curriculum Learning compiler requires curriculum_learning_proposal"
        )
    try:
        source = request.delivery.content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("curriculum proposal is not valid Python") from error
    validate_curriculum_source(source)
    content = source.encode()
    if not content.endswith(b"\n"):
        content += b"\n"
    return CompiledArtifact("curriculum.py", "curriculum_learning", content)
