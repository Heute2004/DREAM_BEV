# CARRADA Radar-Representation Diagnostic

This directory contains the paper-facing 24-source-sequence fit / 6-source-sequence
Development protocol and the pre-defined 496-frame single-pedestrian localization
subset. Every row uses causal eight-frame RA/RA-MTI input and the same metric-BEV
evaluator. Cars and cyclists remain distractors.

`frozen_comparison.json` is the current machine-readable Development comparison, and
`frozen_dream_bev.json` records the selected DREAM-BEV checkpoint and independent
reload metrics. `protocol.json` freezes the source-sequence partition.

This is a custom public-sequence radar-representation diagnostic. It is not the
official CARRADA benchmark, does not use a dedicated public test split, and is not a
complete radar-vision fusion evaluation.
