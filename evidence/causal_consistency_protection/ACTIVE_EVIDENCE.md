# Active Guard Evidence

The canonical deployed guard threshold is selected on Train only:

- EMA alpha: 0.10
- EMA threshold: 0.50
- Selection rule: smallest threshold preserving a one percentage-point clean
  AP@0.30 gain and not falling below expert-radar under the Train +5 degree
  bearing-bias condition.

Use these artifacts for the manuscript and reproducibility checks:

1. `train_guard_selection.json`
2. `development_guard_summary.json`
3. `final_tau05/locked_metrics_tau05.json` in the paper evidence mirror
4. `final_tau05/locked_sequence_bootstrap_10000.json`

These current-contract artifacts define the complete guard evidence shipped in
the reproducibility package.
