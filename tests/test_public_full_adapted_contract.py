from __future__ import annotations

from pathlib import Path

import pytest
import torch

from radar_bev.public_full_adapted import (
    PUBLIC_FULL_SPECS,
    PublicFullPolarBEVAdapter,
    RODNET_STACKED_NUM,
    T_RODNET_COMPACT_CONFIGURATION,
    prepare_e_rodnet_input,
    prepare_fcn8s_input,
    prepare_public_full_input,
    prune_t_rodnet_inactive_modules,
    public_source_metadata,
)


class _TemporalPolarHead(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value[:, :1]


class _ERODNetTemporalHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.received_shape: tuple[int, ...] | None = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.received_shape = tuple(value.shape)
        return value[:, :1, :, 0]


class _ERODNetProbabilityHead(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (value.shape[0], 1, value.shape[2], 128, 128),
            0.75,
            device=value.device,
        )


class _CompactTRODNetLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.downsample = torch.nn.Conv3d(1, 1, kernel_size=1, bias=False)
        self.downsamplelast = torch.nn.Conv3d(1, 1, kernel_size=1, bias=False)


class _CompactTRODNetBody(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_CompactTRODNetLayer() for _ in range(3)])


class _CompactTRODNet(torch.nn.Module):
    def __init__(self, embed_dim: int = 32) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.swin_unet = _CompactTRODNetBody()


def test_public_full_screen_uses_only_explicit_adapted_names_and_source_pinned_models() -> None:
    expected = {
        "t_rodnet_compact",
        "rodnet_cdc",
        "rodnet_hg",
        "fcn8s",
        "e_rodnet",
    }
    assert set(PUBLIC_FULL_SPECS) == expected
    for spec in PUBLIC_FULL_SPECS.values():
        assert spec.paper_name.endswith("-adapted")
        assert "inspired" not in spec.paper_name.lower()
        assert spec.source_modified is False
        assert spec.source_sha256
    assert RODNET_STACKED_NUM == {"rodnet_hg": 2}


def test_trodnet_compact_uses_the_same_pinned_source_with_a_documented_execution_graph() -> None:
    spec = PUBLIC_FULL_SPECS["t_rodnet_compact"]

    assert spec.paper_name == "T-RODNet-core-adapted"
    assert T_RODNET_COMPACT_CONFIGURATION == {
        "embed_dim": 64,
        "win_size": 4,
        "inactive_modules": [
            "swin_unet.layers.0.downsamplelast",
            "swin_unet.layers.1.downsample",
            "swin_unet.layers.2.downsamplelast",
        ],
    }


def test_trodnet_compact_prunes_only_inactive_last_downsamplers_for_the_canonical_contract() -> None:
    model = _CompactTRODNet(embed_dim=64)
    parameter_count_before = sum(parameter.numel() for parameter in model.parameters())

    compact = prune_t_rodnet_inactive_modules(model)

    assert compact is model
    assert isinstance(model.swin_unet.layers[0].downsamplelast, torch.nn.Identity)
    assert isinstance(model.swin_unet.layers[1].downsample, torch.nn.Identity)
    assert isinstance(model.swin_unet.layers[2].downsamplelast, torch.nn.Identity)
    assert sum(parameter.numel() for parameter in model.parameters()) < parameter_count_before


def test_trodnet_compact_rejects_an_embed_dimension_with_a_different_execution_graph() -> None:
    with pytest.raises(ValueError, match="embed_dim=64"):
        prune_t_rodnet_inactive_modules(_CompactTRODNet(embed_dim=32))


def test_erodnet_provenance_is_source_pinned() -> None:
    spec = PUBLIC_FULL_SPECS["e_rodnet"]

    assert spec.paper_name == "E-RODNet-adapted"
    assert spec.repository == "https://github.com/lupeng-xm/E-RODNet.git"
    assert spec.commit
    assert spec.source_modified is False
    assert spec.source_sha256


def test_public_full_input_preserves_current_frame_of_shared_causal_16_frame_contract() -> None:
    radar = torch.arange(16 * 2 * 128 * 128, dtype=torch.float32).reshape(1, 16, 2, 128, 128)
    prepared = prepare_public_full_input(radar)
    assert prepared.shape == (1, 2, 16, 128, 128)
    torch.testing.assert_close(prepared[:, :, -1], radar[:, -1])


def test_erodnet_adapter_repeats_only_existing_evidence_into_its_public_chirp_slots() -> None:
    radar = torch.arange(16 * 2 * 128 * 128, dtype=torch.float32).reshape(1, 16, 2, 128, 128)
    prepared = prepare_e_rodnet_input(radar)

    assert prepared.shape == (1, 2, 16, 4, 128, 128)
    for chirp_index in range(4):
        torch.testing.assert_close(prepared[:, :, :, chirp_index], radar.permute(0, 2, 1, 3, 4))

    backbone = _ERODNetTemporalHead()
    model = PublicFullPolarBEVAdapter("e_rodnet", source_root=None, backbone=backbone)
    output = model(radar)

    assert backbone.received_shape == (1, 2, 16, 4, 128, 128)
    assert output["polar_logits"].shape == (1, 1, 128, 128)
    assert output["logits"].shape == (1, 1, 200, 234)


def test_erodnet_adapter_converts_the_public_sigmoid_response_back_to_logits() -> None:
    model = PublicFullPolarBEVAdapter("e_rodnet", source_root=None, backbone=_ERODNetProbabilityHead())

    output = model(torch.randn(1, 16, 2, 128, 128))

    expected = torch.full((1, 1, 128, 128), torch.logit(torch.tensor(0.75)))
    torch.testing.assert_close(output["polar_logits"], expected)


def test_fcn8s_input_stacks_the_same_shared_sixteen_frame_contract_without_reordering() -> None:
    radar = torch.arange(16 * 2 * 128 * 128, dtype=torch.float32).reshape(1, 16, 2, 128, 128)

    prepared = prepare_fcn8s_input(radar)

    assert prepared.shape == (1, 32, 128, 128)
    torch.testing.assert_close(prepared[:, 0], radar[:, 0, 0])
    torch.testing.assert_close(prepared[:, 1], radar[:, 0, 1])
    torch.testing.assert_close(prepared[:, 30], radar[:, 15, 0])
    torch.testing.assert_close(prepared[:, 31], radar[:, 15, 1])


def test_public_full_adapter_emits_the_shared_metric_bev_contract() -> None:
    model = PublicFullPolarBEVAdapter(
        variant="t_rodnet_compact",
        source_root=None,
        backbone=_TemporalPolarHead(),
    )
    output = model(torch.randn(2, 16, 2, 128, 128))
    assert set(output) >= {"logits", "polar_logits"}
    assert output["polar_logits"].shape == (2, 1, 128, 128)
    assert output["logits"].shape == (2, 1, 200, 234)


def test_public_full_adapter_rejects_noncausal_or_nonshared_input() -> None:
    with pytest.raises(ValueError, match=r"\[batch,16,2,128,128\]"):
        prepare_public_full_input(torch.zeros(1, 8, 2, 128, 128))


def test_public_full_training_runner_is_directly_invocable() -> None:
    root = Path(__file__).parents[1]
    runner = root / "scripts" / "train_public_full_adapted.py"
    assert runner.is_file()
