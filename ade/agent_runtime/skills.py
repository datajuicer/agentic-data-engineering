"""Explicit repo Skill resolution."""

from dataclasses import dataclass
from pathlib import Path
import re

_SKILL_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class ResolvedSkill:
    skill_id: str
    path: Path
    allow_implicit_invocation: bool


class SkillResolver:
    def __init__(self, skills_root: str | Path) -> None:
        self.skills_root = Path(skills_root).resolve()

    def resolve(self, skill_id: str) -> ResolvedSkill:
        if not _SKILL_ID.fullmatch(skill_id):
            raise ValueError("unsafe skill_id")
        path = self.skills_root / skill_id
        skill_file = path / "SKILL.md"
        metadata_file = path / "agents" / "openai.yaml"
        if not skill_file.is_file() or not metadata_file.is_file():
            raise FileNotFoundError(f"incomplete Skill package: {skill_id}")
        skill_text = skill_file.read_text(encoding="utf-8")
        if not re.search(rf"^name:\s*{re.escape(skill_id)}\s*$", skill_text, re.MULTILINE):
            raise ValueError(f"Skill frontmatter name mismatch: {skill_id}")
        metadata = metadata_file.read_text(encoding="utf-8")
        implicit = not bool(
            re.search(r"^\s*allow_implicit_invocation:\s*false\s*$", metadata, re.MULTILINE)
        )
        return ResolvedSkill(skill_id, path, implicit)
