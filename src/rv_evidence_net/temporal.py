from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from .contracts import TemporalRecoveryOutput


class CausalSelectiveStateSpaceBlock(nn.Module):
    """Small input-conditioned causal state-space block.

    This is an explicit short-sequence selective scan. It deliberately avoids an
    undeclared `mamba_ssm` binary dependency while preserving the important Mamba
    property for this project: a causal, input-conditioned recurrent state.
    """

    def __init__(
        self,
        model_dim: int,
        state_dim: int,
        expansion: int = 2,
        convolution_kernel: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if convolution_kernel < 1:
            raise ValueError("convolution_kernel must be positive")
        self.model_dim = model_dim
        self.state_dim = state_dim
        self.inner_dim = model_dim * expansion
        self.norm = nn.LayerNorm(model_dim)
        self.input_projection = nn.Linear(model_dim, self.inner_dim * 2)
        self.causal_convolution = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=convolution_kernel,
            padding=convolution_kernel - 1,
            groups=self.inner_dim,
        )
        self.delta_projection = nn.Linear(self.inner_dim, self.inner_dim)
        self.b_projection = nn.Linear(self.inner_dim, state_dim)
        self.c_projection = nn.Linear(self.inner_dim, state_dim)
        self.a_log = nn.Parameter(torch.zeros(self.inner_dim, state_dim))
        self.skip = nn.Parameter(torch.ones(self.inner_dim))
        self.output_projection = nn.Linear(self.inner_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: Tensor, valid_mask: Tensor) -> Tensor:
        """Apply a causal scan.

        Args:
            sequence: `[batch, time, model_dim]`.
            valid_mask: `[batch, time]`; invalid time steps hold the latent state.
        """

        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [batch, time, model_dim]")
        if valid_mask.shape != sequence.shape[:2]:
            raise ValueError("valid_mask must match sequence batch and time dimensions")

        residual = sequence
        normalized = self.norm(sequence)
        content, gate = self.input_projection(normalized).chunk(2, dim=-1)
        content = self.causal_convolution(content.transpose(1, 2))[..., : sequence.shape[1]].transpose(1, 2)
        content = functional.silu(content)

        delta = functional.softplus(self.delta_projection(content)) + 1e-4
        b_value = torch.tanh(self.b_projection(content))
        c_value = torch.tanh(self.c_projection(content))
        a_value = -torch.exp(self.a_log).to(dtype=sequence.dtype)
        skip = self.skip.to(dtype=sequence.dtype)
        state = sequence.new_zeros(sequence.shape[0], self.inner_dim, self.state_dim)
        outputs: list[Tensor] = []

        for time_index in range(sequence.shape[1]):
            delta_t = delta[:, time_index]
            content_t = content[:, time_index]
            transition = torch.exp(delta_t.unsqueeze(-1) * a_value.unsqueeze(0))
            injected = (
                delta_t.unsqueeze(-1)
                * b_value[:, time_index].unsqueeze(1)
                * content_t.unsqueeze(-1)
            )
            candidate_state = transition * state + injected
            state_is_valid = valid_mask[:, time_index].view(-1, 1, 1)
            state = torch.where(state_is_valid, candidate_state, state)
            output_t = (state * c_value[:, time_index].unsqueeze(1)).sum(dim=-1) + skip * content_t
            output_t = functional.silu(gate[:, time_index]) * output_t
            output_t = output_t * valid_mask[:, time_index].unsqueeze(-1).to(dtype=output_t.dtype)
            outputs.append(output_t)

        output = torch.stack(outputs, dim=1)
        return residual + self.dropout(self.output_projection(output))


class CausalMambaSSMBlock(nn.Module):
    """Causal `mamba_ssm` block with explicit invalid-observation handling.

    Each proposal trajectory is processed as its ordered valid observations.
    This lets an invalid frame hold temporal state instead of injecting a
    synthetic zero observation into the official Mamba recurrence.
    """

    def __init__(
        self,
        model_dim: int,
        state_dim: int,
        expansion: int = 2,
        convolution_kernel: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm.modules.mamba_simple import Mamba
        except ImportError as error:
            raise ImportError(
                "temporal_backend='mamba_ssm' requires the dedicated WSL Mamba environment"
            ) from error
        self.norm = nn.LayerNorm(model_dim)
        self.mamba = Mamba(
            d_model=model_dim,
            d_state=state_dim,
            d_conv=convolution_kernel,
            expand=expansion,
            use_fast_path=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: Tensor, valid_mask: Tensor) -> Tensor:
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [batch, time, model_dim]")
        if valid_mask.shape != sequence.shape[:2]:
            raise ValueError("valid_mask must match sequence batch and time dimensions")

        residual = sequence
        normalized = self.norm(sequence)
        output = torch.zeros_like(normalized)
        for batch_index in range(sequence.shape[0]):
            valid_indices = valid_mask[batch_index].nonzero(as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                continue
            valid_sequence = normalized[batch_index : batch_index + 1, valid_indices]
            output[batch_index, valid_indices] = self.mamba(valid_sequence).squeeze(0).to(dtype=output.dtype)
        return residual + self.dropout(output)


class CausalMambaEvidenceRecovery(nn.Module):
    """Causal proposal-level Mamba-style visual evidence recovery.

    Invalid current proposals remain invalid. The module may preserve state across
    a missing observation but it is never allowed to create a new visual box.
    """

    def __init__(
        self,
        token_dim: int = 128,
        geometry_dim: int = 6,
        model_dim: int = 128,
        state_dim: int = 16,
        layers: int = 2,
        dropout: float = 0.1,
        temporal_backend: str = "selective_scan",
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be at least one")
        if temporal_backend not in {"selective_scan", "mamba_ssm"}:
            raise ValueError("temporal_backend must be 'selective_scan' or 'mamba_ssm'")
        self.token_dim = token_dim
        self.geometry_dim = geometry_dim
        self.temporal_backend = temporal_backend
        feature_dim = token_dim + geometry_dim + 2
        self.input_adapter = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        block_type = CausalSelectiveStateSpaceBlock if temporal_backend == "selective_scan" else CausalMambaSSMBlock
        self.blocks = nn.ModuleList(block_type(model_dim, state_dim, dropout=dropout) for _ in range(layers))
        self.history_norm = nn.LayerNorm(model_dim)
        self.residual_projection = nn.Linear(model_dim, token_dim)
        self.mix_gate = nn.Sequential(
            nn.Linear(token_dim + model_dim + 1, token_dim),
            nn.Sigmoid(),
        )
        self.token_norm = nn.LayerNorm(token_dim)
        self.score_delta = nn.Linear(model_dim, 1)
        self.support_head = nn.Linear(model_dim, 1)

    def forward(
        self,
        raw_tokens: Tensor,
        box_geometry: Tensor,
        raw_scores: Tensor,
        valid_mask: Tensor,
    ) -> TemporalRecoveryOutput:
        """Recover short-term evidence without changing proposal validity.

        Shapes are `[batch, time, proposals, ...]`. Geometry must contain
        normalized center, size, and consecutive center displacement.
        """

        if raw_tokens.ndim != 4:
            raise ValueError("raw_tokens must have shape [batch, time, proposals, token_dim]")
        expected_prefix = raw_tokens.shape[:3]
        if box_geometry.shape != (*expected_prefix, self.geometry_dim):
            raise ValueError("box_geometry has an incompatible shape")
        if raw_scores.shape != expected_prefix or valid_mask.shape != expected_prefix:
            raise ValueError("raw_scores and valid_mask must match raw token prefix dimensions")

        batch_size, time_steps, proposal_count, _ = raw_tokens.shape
        sequence_tokens = raw_tokens.permute(0, 2, 1, 3).reshape(batch_size * proposal_count, time_steps, -1)
        sequence_geometry = box_geometry.permute(0, 2, 1, 3).reshape(batch_size * proposal_count, time_steps, -1)
        sequence_scores = raw_scores.permute(0, 2, 1).reshape(batch_size * proposal_count, time_steps, 1)
        sequence_valid = valid_mask.permute(0, 2, 1).reshape(batch_size * proposal_count, time_steps)

        features = torch.cat((sequence_tokens, sequence_geometry, sequence_scores, sequence_valid.unsqueeze(-1).float()), dim=-1)
        hidden = self.input_adapter(features)
        for block in self.blocks:
            hidden = block(hidden, sequence_valid)
        hidden = self.history_norm(hidden)

        residual = self.residual_projection(hidden)
        mixture = self.mix_gate(torch.cat((sequence_tokens, hidden, sequence_scores), dim=-1))
        refined = self.token_norm(sequence_tokens + mixture * residual)
        raw_score_logits = torch.logit(sequence_scores.clamp(1e-4, 1.0 - 1e-4))
        calibrated = torch.sigmoid(raw_score_logits + self.score_delta(hidden))
        support = torch.sigmoid(self.support_head(hidden))

        valid_float = sequence_valid.unsqueeze(-1).to(dtype=refined.dtype)
        refined = refined * valid_float
        calibrated = calibrated * valid_float
        support = support * valid_float

        def restore_token(tensor: Tensor) -> Tensor:
            return tensor.reshape(batch_size, proposal_count, time_steps, -1).permute(0, 2, 1, 3).contiguous()

        def restore_scalar(tensor: Tensor) -> Tensor:
            return tensor.reshape(batch_size, proposal_count, time_steps).permute(0, 2, 1).contiguous()

        return TemporalRecoveryOutput(
            refined_tokens=restore_token(refined),
            temporal_support=restore_scalar(support),
            calibrated_scores=restore_scalar(calibrated),
            valid_mask=valid_mask,
        )



