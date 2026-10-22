# Paper alignment

The source release is bound to the manuscript and supplement identified in
`contracts/paper_binding.json`. Packaging does not select a different checkpoint
or change predictions. Source-copy receipts remain in the local release audit.

| Component | Implementation |
| --- | --- |
| Radar anchor | `src/radar_bev/models/backbones.py`, `scripts/train_radar.py` |
| Five experts | `src/radar_bev/models/`, `scripts/train_experts.py` |
| Frozen person encoder | `src/rv_evidence_net/`, `scripts/train_person_encoder.py` |
| Monotonic residual bearing | `src/rv_bearing_net/monotonic_residual.py`, `scripts/train_bearing.py` |
| Bearing supervision cache | `scripts/build_training_teacher_cache.py` |
| CNN/VLM policy | `src/condition_policy/`, `train_rgb_control.py`, `train_condition_policy.sh` |
| Fusion and losses | `src/rvfusion/lattice_fusion.py`, `lattice_losses.py` |
| Delayed admission | `policy_contract.py`, `causal_guard_replay.py` |
| Consistency protection | `src/rvfusion/cross_modal_consistency.py` |
| Locked result | `evidence/causal_consistency_protection/final_tau05/` |
| Operator and routing studies | `evidence/active_development_contract/`, `evidence/route_closure_20260819/canonical/` |
| Figure 6 | `scripts/plot_visual_conflict_curves.py` and the full-sweep CSV |
| Public transfer | `external/CRUW_PedestrianFusion/`, `external/CARRADA_DREAM_BEV/` |

Locked AP@0.30 is 0.7992337413 for expert radar and 0.8359054290 for protected
fusion. The private guard uses alpha 0.10 and threshold 0.50. CRUW uses its
separately frozen 0.80 threshold. Do not substitute guard/decoder configurations
between datasets.

The common-input operator study and full perturbation sweep are distinct frozen
evaluations. Their clean values must not be interchanged. The full-sweep 50%
dropout point is used in Figure 6, rather than an earlier single-point diagnostic.
The two source chains, replay entries, state initialization, and unresolved
upstream bindings are listed in [DEVELOPMENT_REPLAYS.md](DEVELOPMENT_REPLAYS.md).
Training selection histories and observed stopping counts are summarized in
[TRAINING_STOPPING.md](TRAINING_STOPPING.md).

The encoder model differs from its audit copy only in an RGB input-error message
and trailing whitespace. The release restores the archived encoder loader's
explicit sequence partitions and adds a validator for the packaged manifest.
This fixes a packaging omission; no trained architecture, loss, or weight changes.

DREAM denotes Deep Radar Experts for Adaptive Multimodal BEV Localization.
BEV means bird's-eye view. The manuscript identity and page counts are recorded
in `contracts/paper_binding.json`.
