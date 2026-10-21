# Cross-Modal Guard Calibration Contract

## Scope

This contract calibrates the fast cross-modal consistency guard for the
frozen DREAM-BEV recovery and RVFusion stack. It does not retrain a radar
expert, the visual bearing model, the candidate-local posterior, or the
policy router.

The guard compares the calibrated visual lateral ray with the frozen
expert-radar coordinate using the sum of visual and local radar uncertainty.
It maintains a causal exponential moving average of the signed standardized
innovation independently for every sequence. A transient disagreement is not
sufficient to reject visual evidence. When persistent same-direction
disagreement exceeds the selected threshold, the output is the exact
expert-radar identity prediction.

For the frozen radar coordinate $(\hat{x}^R_t,\hat{y}^R_t)$, the visual
cross-range uncertainty is

```text
sigma_V = clip(y_R * sigma_theta / cos(theta)^2 + 0.05, 0.025, 0.50) m.
```

The radar uncertainty is the standard deviation of the normalized expert-radar
BEV x-profile within +/-0.50 m of the frozen radar peak at `y_R`, clipped to
`[0.05, 0.50] m`. The signed innovation is

```text
z = (x_V - x_R) / sqrt(sigma_V^2 + sigma_R^2 + 1e-6 m^2).
```

The `1e-6 m^2` term is a numerical variance floor only; it is not a tuned
confidence parameter.

## Data Isolation

- Calibration: `Train` only, using the fixed 87-sequence IF-only manifest.
- Diagnostic stress curves: `Development Val` only, using the fixed
  34-sequence manifest split.
- Locked Test: forbidden for threshold choice, stress evaluation, and this
  calibration procedure.

## Fixed Candidate Set And Selection Rule

The causal moving-average coefficient and candidate persistent-innovation
thresholds are fixed before reading any Development stress result:

```text
ema_alpha = 0.10
ema_threshold in {0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00}
```

Each candidate is replayed with the same frozen checkpoint and causal
segment-policy trace on two Train-only conditions: clean bearing evidence and
a fixed `+5` degree bearing-bias calibration perturbation. Let `A_clean(e)`
be clean Train AP@0.30, `A_clean(ref)` the no-guard clean Train AP@0.30,
`A_bias(e)` the biased Train AP@0.30, and `A_radar(bias)` the corresponding
expert-radar AP@0.30. Select the smallest `e` satisfying both:

```text
A_clean(e) >= A_radar(clean) + 0.010
A_bias(e)  >= A_radar(bias)
```

Both constraints are training-side safety criteria: the clean condition must
retain a positive visual contribution over expert-radar, while the severe
bearing perturbation must not fall below expert-radar. Selecting the smallest
feasible threshold gives the strongest protective setting under this contract.
If no candidate satisfies both conditions, do not deploy the guard. No
Development or Locked-Test metric may influence the selected threshold.

## Frozen Stress Evaluation

After selection, evaluate exactly these Development-only perturbations:

- bearing bias: 0, 1, 2, 3, 4, 5, 6 degrees;
- visual lag: 0, 2, 4, 8 frames;
- visual dropout: 0, 25, 50, 75 percent.

For each condition, report expert-radar, unguarded RVFusion, guarded
RVFusion, AP@0.25/0.30/0.50, mean/P90 localization error, guard pass rate,
policy identity rate, and negative-transfer rate. Early Development replays
under the clean-only rule are implementation diagnostics, not final
model-selection evidence.

## Canonical Replay Requirement

The semantic policy/radar trace used by the main Development hierarchy is the
canonical source for every mechanism control. A zero-bias replay must reproduce
the main Development prediction frame-for-frame before any perturbation row is
reported. Perturbed visual evidence is joined to that fixed trace by
`(sequence, frame_id)`; target coordinates, radar coordinates, and confidence
scores must match exactly. This prevents a stress script from silently changing
policy admission or guard semantics.

Guard-off is a Development-only mechanism ablation. It keeps the learned policy
admission but disables the causal consistency decision. It is never reported as
the deployed system or used in Locked Test, public transfer, or headline tables.

The reproducible entry point is:

```text
python scripts/sweep_causal_guard_calibration.py \
  --policy-causal-replay <train-policy-radar-trace.csv> \
  --clean-evidence-audit <train-clean-evidence.csv> \
  --bias-evidence-audit <train-bias5-evidence.csv> \
  --manifest <train-development-manifest.csv> \
  --output-dir <new-train-calibration-directory>
```
