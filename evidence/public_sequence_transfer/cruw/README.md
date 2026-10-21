# CRUW Public-Sequence Transfer Evidence

This directory contains only the paper-facing 12-sequence fit / 4-sequence
Development protocol. Vehicles and cyclists remain in the input as distractors, and
the official CRUW test input is never read.

The current protected results are:

| Expert anchor | Expert AP@0.30 | Protected RVFusion AP@0.30 | Gain |
| --- | ---: | ---: | ---: |
| PB-MHBR | 0.7168 | 0.7319 | +1.51 pp |
| GG-SRRE | 0.7172 | 0.7316 | +1.43 pp |

`protected/` contains the current causal-guard replays. `public_baselines/` contains
the independently reloaded RC-ROSNet, CRN, and RCBEVDet task adaptations used in the
paired tables. All rows share the same public-sequence split, radar anchor, expert,
RGB evidence, metric decoder, and evaluator within each expert-specific comparison.

The coordinate field called `y` in the implementation is the Radar-H forward
coordinate. The public evidence supports transfer under this contract; it is not an
official CRUW leaderboard result or a multi-target localization claim.
