from __future__ import annotations

from torch import Tensor, nn

from .contracts import RadarEvidence, VisualEvidence


class _FrozenAdapter(nn.Module):
    wrapped_name: str

    def _freeze(self, module: nn.Module) -> None:
        module.requires_grad_(False)
        module.eval()

    def train(self, mode: bool = True) -> "_FrozenAdapter":
        super().train(mode)
        getattr(self, self.wrapped_name).eval()
        return self


class FrozenRadarAdapter(_FrozenAdapter):
    wrapped_name = "backbone"

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self._freeze(self.backbone)

    def forward(self, radar_window: Tensor) -> RadarEvidence:
        self.backbone.eval()
        polar = self.backbone.forward_polar_feature(radar_window)
        shared = self.backbone.shared_from_polar_feature(polar)
        logits = self.backbone.decoder.decode_shared(shared)
        return RadarEvidence(polar_feature=polar, bev_feature=shared, logits=logits)


class FrozenVisualAdapter(_FrozenAdapter):
    wrapped_name = "model"

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self._freeze(self.model)

    def forward(self, rgb_clip: Tensor) -> VisualEvidence:
        self.model.eval()
        output = self.model.forward_clip(rgb_clip)
        return VisualEvidence(
            boxes=output.boxes,
            raw_scores=output.raw_scores,
            raw_tokens=output.raw_tokens,
            temporal_tokens=output.temporal_tokens,
            temporal_support=output.temporal_support,
            valid_mask=output.valid_mask,
        )




