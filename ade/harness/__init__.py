"""ADE process assembly and operator service."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ade.harness.service import Harness

__all__ = ["Harness"]


def __getattr__(name: str):
    if name == "Harness":
        from ade.harness.service import Harness

        return Harness
    raise AttributeError(name)
