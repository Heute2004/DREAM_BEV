from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor, nn
from torch.nn import functional

from .common import BEVDecoder, ConvNormAct, RadarBEVBase, ResidualBlock
from .ragt_lite import RAGTLiteAdapters, RAGTLiteVariant

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


def extract_dream_bev_state(payload: object) -> Mapping[str, Tensor]:
    """Resolve a supported checkpoint envelope before strict DREAM-BEV loading."""

    if not isinstance(payload, Mapping):
        raise ValueError("DREAM-BEV checkpoint payload must be a mapping")
    marker = payload.get("model")
    state = payload.get("model_state")
    if state is None and isinstance(marker, Mapping):
        state = marker
    if state is None and isinstance(marker, str) and marker != "dream_bev":
        raise ValueError("DREAM-BEV checkpoint does not contain a state dictionary")
    if state is None:
        state = payload
    if not isinstance(state, Mapping) or not state:
        raise ValueError("DREAM-BEV checkpoint does not contain a state dictionary")
    return state


class ERODNetCoreBEV(RadarBEVBase):
    """E-RODNet core adaptation reused across the private and CARRADA tasks."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(ConvNormAct(2, 64, dimensions=3), ConvNormAct(64, 64, dimensions=3))
        self.blocks = nn.ModuleList([_MetaFormer3DBlock(64) for _ in range(3)])
        self.reduce = nn.Sequential(ConvNormAct(64, 48), ResidualBlock(48))
        self.decoder = BEVDecoder(48, self.geometry)

    def forward(self, radar: Tensor) -> dict[str, Tensor]:
        self._validate_input(radar)
        feature = self.stem(radar.permute(0, 2, 1, 3, 4))
        for block in self.blocks:
            feature = block(feature)
        polar_feature = self.reduce(feature[:, :, -1])
        shared_feature = self.decoder.encode_shared(self.polar_to_bev(polar_feature))
        return {"logits": self.decoder.decode_shared(shared_feature), "shared_feature": shared_feature}


class TRODNetCoreBEV(RadarBEVBase):
    """Compact T-RODNet core adaptation without its full Swin U-Net decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(2, 64, stride=(1, 2, 2), dimensions=3),
            ConvNormAct(64, 64, dimensions=3),
        )
        self.temporal_attention = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
        self.temporal_norm = nn.LayerNorm(64)
        self.reduce = nn.Sequential(ConvNormAct(64, 48), ResidualBlock(48))
        self.decoder = BEVDecoder(48, self.geometry)

    def forward(self, radar: Tensor) -> dict[str, Tensor]:
        self._validate_input(radar)
        feature = self.stem(radar.permute(0, 2, 1, 3, 4))
        batch, channels, frames, height, width = feature.shape
        tokens = feature.permute(0, 3, 4, 2, 1).reshape(batch * height * width, frames, channels)
        attended, _ = self.temporal_attention(tokens, tokens, tokens, need_weights=False)
        latest = self.temporal_norm(tokens[:, -1] + attended[:, -1])
        latest = latest.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        polar_feature = self.reduce(
            functional.interpolate(latest, size=(128, 128), mode="bilinear", align_corners=False)
        )
        shared_feature = self.decoder.encode_shared(self.polar_to_bev(polar_feature))
        return {"logits": self.decoder.decode_shared(shared_feature), "shared_feature": shared_feature}


class _MetaFormer3DBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.token_mixer = nn.Conv3d(channels, channels, kernel_size=(3, 5, 5), padding=(1, 2, 2), groups=channels)
        self.norm1 = nn.GroupNorm(8, channels)
        self.mlp = nn.Sequential(nn.Conv3d(channels, channels * 2, kernel_size=1), nn.GELU(), nn.Conv3d(channels * 2, channels, kernel_size=1))
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, value: Tensor) -> Tensor:
        value = value + self.token_mixer(self.norm1(value))
        return value + self.mlp(self.norm2(value))


class _CausalMambaTokenMixer(nn.Module):
    """Selective state-space mixing over a past-only temporal token sequence."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "DREAM-BEV requires mamba_ssm. Install the optional Mamba "
                "dependency before constructing the radar anchor."
            )
        self.mixer = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(channels)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must be [batch,time,channels]")
        return self.norm(tokens + self.mixer(tokens))


class DREAMBEV(RadarBEVBase):
    """Causal DREAM-BEV radar anchor used by the complete fusion hierarchy."""

    def __init__(self, lite_variant: RAGTLiteVariant = "base") -> None:
        super().__init__()
        self.stem = nn.Sequential(ConvNormAct(2, 64, stride=2), ConvNormAct(64, 64))
        self.temporal = _CausalMambaTokenMixer(64)
        self.reduce = nn.Sequential(ConvNormAct(64, 48), ResidualBlock(48))
        self.decoder = BEVDecoder(48, self.geometry)
        self.ragt_lite = RAGTLiteAdapters(64, lite_variant)

    def forward_polar_feature(self, radar: Tensor) -> Tensor:
        """Return the final native-polar feature before polar-to-BEV projection."""

        self._validate_input(radar)
        conditioned_radar, evidence = self.ragt_lite.condition_radar(radar)
        batch, frames = conditioned_radar.shape[:2]
        feature = self.stem(conditioned_radar.reshape(batch * frames, 2, 128, 128))
        _, channels, height, width = feature.shape
        sequence = feature.reshape(batch, frames, channels, height, width)
        sequence = self.ragt_lite.condition_temporal(sequence, evidence)
        tokens = sequence.permute(0, 3, 4, 1, 2).reshape(
            batch * height * width, frames, channels
        )
        latest = self.temporal(tokens)[:, -1].reshape(
            batch, height, width, channels
        ).permute(0, 3, 1, 2)
        latest = functional.interpolate(
            latest, size=(128, 128), mode="bilinear", align_corners=False
        )
        return self.reduce(latest)

    def shared_from_polar_feature(self, polar_feature: Tensor) -> Tensor:
        """Project a 48-channel polar feature through the frozen BEV interface."""

        if polar_feature.ndim != 4 or polar_feature.shape[1:] != (48, 128, 128):
            raise ValueError("polar_feature must be [batch,48,128,128]")
        bev_feature = self.ragt_lite.condition_bev(self.polar_to_bev(polar_feature))
        return self.decoder.encode_shared(bev_feature)

    def forward_shared(self, radar: Tensor) -> Tensor:
        """Return the stable BEV interface used by the route specialists."""

        return self.shared_from_polar_feature(self.forward_polar_feature(radar))

    def forward(self, radar: Tensor) -> dict[str, Tensor]:
        shared_feature = self.forward_shared(radar)
        return {
            "logits": self.decoder.decode_shared(shared_feature),
            "shared_feature": shared_feature,
        }
