from __future__ import annotations

from torch import nn

from .backbones import DREAMBEV, ERODNetCoreBEV, TRODNetCoreBEV
from .recovery_experts import DREAMRecoveryRouteExpertBEV
from .clutter_expert import DREAMClutterExpertBEV
from .dry_complex_expert import DREAMDryComplexExpertBEV
from .heterogeneous_experts import DREAMHeterogeneousExpertBEV
from .route_experts import DREAMRouteExpertBEV
from .weak_return_expert import DREAMWeakReturnExpertBEV


def build_model(name: str, **kwargs: object) -> nn.Module:
    registry = {
        "dream_bev": DREAMBEV,
        "erodnet_core": ERODNetCoreBEV,
        "trodnet_core": TRODNetCoreBEV,
        "dream_route_expert": DREAMRouteExpertBEV,
        "dream_heterogeneous_expert": DREAMHeterogeneousExpertBEV,
        "dream_recovery_expert": DREAMRecoveryRouteExpertBEV,
        "dream_clutter_expert": DREAMClutterExpertBEV,
        "dream_weak_return_expert": DREAMWeakReturnExpertBEV,
        "dream_dry_complex_expert": DREAMDryComplexExpertBEV,
    }
    try:
        return registry[name.lower()](**kwargs)
    except KeyError as error:
        choices = ", ".join(sorted(registry))
        raise ValueError(f"Unknown radar model {name!r}; choose one of: {choices}") from error


__all__ = ["build_model"]



