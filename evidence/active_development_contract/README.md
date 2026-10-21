# Active Development Evidence Contract

This directory contains the paper-facing Development-only evidence for the
deployed RVFusion configuration. Every result uses the same frozen
candidate-stage checkpoint, direct-utility selection, lattice correction, and
Train-calibrated causal consistency guard. The result files contain 9,969
scored Development frames drawn only from the IF-only Train/Development
manifest; no Locked Test record, feature, checkpoint choice, or threshold
selection is used here.

- `operator/`: candidate-local posterior, compact quantized-residual control,
  and guard mechanism results.
- `route_matrix/`: guarded forced-route counterfactual matrix over the named
  Development ledger groups.
- `stress/`: paired guard-off and guarded bearing-bias, temporal-lag, and
  RGB-dropout replays.
- `policy_frontier/`: interchangeable CNN/VLM policy-backend replay with the
  downstream hierarchy held fixed.

Each subdirectory includes the machine-readable summary and SHA256 manifest
written on the laboratory server. These files are the complete Development
evidence set shipped by the minimal reproducibility workspace.
