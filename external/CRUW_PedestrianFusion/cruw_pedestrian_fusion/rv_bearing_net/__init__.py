"""Task-specific causal visual bearing evidence for Radar-centric RVFusion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .contracts import RVBearingOutput
    from .model import RVBearingNet

__all__ = ("RVBearingNet", "RVBearingOutput")


def __getattr__(name: str) -> Any:
    if name == "RVBearingNet":
        from .model import RVBearingNet

        return RVBearingNet
    if name == "RVBearingOutput":
        from .contracts import RVBearingOutput

        return RVBearingOutput
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")



