"""Deployment decoders for bounded CRUW RVFusion candidate posteriors."""

from __future__ import annotations

import numpy as np


def decode_admitted_posterior_mean(
    candidate_xy_m: np.ndarray,
    candidate_logits: np.ndarray,
    *,
    nonidentity_bias: float | np.ndarray,
    temperature: float = 1.0,
    expectation_noop_logit_offset: float = 0.0,
    candidate_logit_adjustment: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep a MAP no-op exact; otherwise return the bounded posterior mean.

    Candidate zero is the immutable radar fallback. The posterior expectation is
    used only when a non-zero candidate wins after Development-calibrated bias.
    Its expectation-only no-op offset never changes the admission decision.
    """

    candidates = np.asarray(candidate_xy_m, dtype=np.float32)
    logits = np.asarray(candidate_logits, dtype=np.float32).copy()
    if candidates.ndim != 3 or candidates.shape[-1] != 2:
        raise ValueError("candidate_xy_m must have shape [frame,candidate,2]")
    if logits.shape != candidates.shape[:2]:
        raise ValueError("candidate_logits must align with candidate_xy_m")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    bias = np.asarray(nonidentity_bias, dtype=np.float32)
    if bias.ndim == 0:
        logits[:, 1:] += float(bias)
    elif bias.shape == (logits.shape[0],):
        logits[:, 1:] += bias[:, None]
    else:
        raise ValueError("nonidentity_bias must be scalar or have one value per frame")
    if candidate_logit_adjustment is not None:
        adjustment = np.asarray(candidate_logit_adjustment, dtype=np.float32)
        if adjustment.shape != logits.shape:
            raise ValueError("candidate_logit_adjustment must align with candidate logits")
        logits += adjustment
    selected = logits.argmax(axis=1)
    accepted = selected != 0
    expectation_logits = logits.copy()
    expectation_logits[:, 0] += float(expectation_noop_logit_offset)
    shifted = expectation_logits - expectation_logits.max(axis=1, keepdims=True)
    posterior = np.exp(shifted / float(temperature))
    posterior /= posterior.sum(axis=1, keepdims=True)
    prediction = candidates[:, 0].copy()
    prediction[accepted] = (posterior[accepted, :, None] * candidates[accepted]).sum(axis=1)
    # Enforce the immutable radar-range contract exactly, not just numerically.
    prediction[:, 1] = candidates[:, 0, 1]
    return prediction, accepted
