# Reproducibility Guide

## Preconditions

1. Create a Python 3.11 environment, then install the project with
   `python -m pip install -e .[development]`. Add the `mamba` extra only on a
   CUDA runtime supported by `mamba-ssm`.
2. Copy `configs/data_paths.example.yaml` to `configs/data_paths.yaml` and set the
   local data, work, checkpoint, and InternVL paths.
3. Verify the source tree before creating any cache or output:

```powershell
python -m pytest tests -q
```

## Data and policy preparation

The project consumes `contracts/frame_manifest.csv`. Do not split individual frames;
all commands must respect the sequence split recorded in `contracts/sequence_split.csv`.

```powershell
python scripts/build_condition_dataset.py --help
python scripts/prepare_condition_meta.py --help
python scripts/materialize_policy.py --help
```

Materialize policy predictions before fusion evaluation. The materialized file must
contain an initial identity interval and only complete, fixed hold intervals afterwards.
The canonical schedule is 30 radar frames per RGB capture, a 32-frame measured
availability delay, and a 30-frame hold interval. `scripts/train_fusion.py` defaults
to the same schedule and to segment-level policy admission. Every reportable
deployment replay additionally applies the Train-calibrated frame-rate causal
consistency guard.

```powershell
python scripts/validate_policy_contract.py --policy-jsonl <materialized_policy.jsonl> --audit-csv <fusion_audit.csv> --trust-policy soft
```

## Training order

1. Train the radar anchor with `scripts/train_radar.py`.
2. Freeze the radar anchor and train mechanism-specific experts with
   `scripts/train_experts.py`. The only supported expert families are:

   ```powershell
   # Local specialists: clean indoor, cluttered indoor, weak outdoor return, wet outdoor return.
   python scripts/train_experts.py --expert-family local --adapter-route <route> ...

   # Recovery specialists: clutter, weak-return, or wet-return recovery.
   python scripts/train_experts.py --expert-family recovery --adapter-route <route> ...

   # Candidate re-ranking specialist for dry, complex outdoor observations.
   python scripts/train_experts.py --expert-family dry_complex --adapter-route outdoor_dry_complex ...
   ```

   Every invocation derives its Train and Development sequence lists from
   `contracts/routing_manifest.jsonl`; no command accepts a manually typed sequence list.
3. Train the RGB person encoder with `scripts/train_person_encoder.py` using
   the frozen protocol and the externally supplied teacher manifest/profile.
   It uses only Train and Development sequences. Export proposals with
   `scripts/cache_visual_evidence.py` and build the training teacher cache with
   `scripts/build_training_teacher_cache.py`. Then train the RGB bearing model
   with `scripts/train_bearing.py` and export causal bearing caches with
   `scripts/export_bearing_cache.py`. The encoder checkpoint must match the
   hash declared by the selected bearing checkpoint; do not mix caches from
   different encoders. See `evidence/implementation/bearing_training_metadata.json`.
4. Train the RGB condition policy using the upstream InternVL implementation and the
   local metadata from `scripts/prepare_condition_meta.py`.
5. Materialize causal policy intervals with `scripts/materialize_policy.py`.
6. Train forward-coordinate-preserving fusion with `scripts/train_fusion.py`. Use the same
   entry point with `--evaluation-only` for frozen evaluation.

## Final evaluation

The locked test is evaluated once after checkpoints, policy parsing, hold schedule,
guard calibration, and evaluation code have been frozen. Use
`scripts/bootstrap_sequences.py` for sequence-level confidence intervals, then replay
the selected coordinate stream with `scripts/replay_causal_cross_modal_guard.py`.
The selectivity report must be built from the final deployed `visual_accepted` gate,
never from a radar-health diagnostic or segment admission alone.

## Source-backed public fusion controls

The three public fusion controls use the same frozen expert-radar anchor, causal
RGB evidence cache, sequence manifest, and evaluator as the proposed fusion
module. They are task adaptations of the pinned public fusion cores, not original
multi-camera or 3-D benchmark reproductions. Clone each official repository at the
commit recorded by `rvfusion.public_official_fusion`. Train on Train and select
the frozen checkpoint by Development AP@0.30. The archived heads selected epoch
1 and ended after three consecutive non-improving epochs, at epoch 4. The
module-specific histories and stopping rules are in
[TRAINING_STOPPING.md](TRAINING_STOPPING.md):

```powershell
python scripts/train_external_rvfusion_baselines.py --baseline rcrosnet --public-source-root <official-source> ...
python scripts/train_external_rvfusion_baselines.py --baseline crn --public-source-root <official-source> ...
python scripts/train_external_rvfusion_baselines.py --baseline rcbevdet --public-source-root <official-source> ...
```

Use `scripts/evaluate_external_rvfusion_baselines.py` only for frozen
Development replays. The evaluator records the official-source commit, checkpoint
hash, input cache, and split and rejects existing output directories. These controls
are not used to choose the proposed model or to reopen the locked test. The
study is contextual adaptation evidence. Reaching the recorded patience does
not establish equal optimizer-step counts or optimization parity across models.

## Source-backed public radar controls

The paper-facing radar controls are task adaptations of pinned public model
cores: RODNet-CDC, RODNet-HG, FCN-8s, E-RODNet core, and T-RODNet core. They use
the same causal radar input, sequence split, evaluator, and checkpoint-selection
rule as DREAM-BEV. Their source metadata is recorded by the adapter at runtime.

```powershell
python scripts/train_public_full_adapted.py --help
python scripts/evaluate_backbone_strict_dev.py --help
```

These commands train on Train and select on Development. They do not reopen the
locked test or introduce a second protocol.

For the two compact core adaptations, use `scripts/train_radar.py --model
erodnet_core` or `--model trodnet_core`, with the same private manifest and
causal 16-frame configuration. The full-model command above is not an alias for
these compact cores. CARRADA's reported E-RODNet core directly reuses the same
class through `private_erodnet_core --private-model-root <checkout>/src`.
See `external/README.md` for the separate eight-frame public-data contract.

The release contains source and frozen numerical evidence, not the human-image
dataset or trained model binaries. Full retraining requires the assets in
`docs/DATA_AND_WEIGHTS.md`; CLI defaults are not a substitute for a selected
run's configuration. Source and result-table verification can run without them.

## Policy backend and conflict audits

The policy interface is backend-configurable. An efficient CNN and a semantic
vision-language policy must both emit the same structured route, admission, and
condition schema before they are replayed through the frozen recovery and
fusion stack. Do not retrain any radar, expert, bearing, or fusion component
when changing only the policy backend.

Use `scripts/force_expert_route_trace.py` and
`scripts/run_route_counterfactual_matrix_dev.sh` for the Development-only 5x5
expert-route audit. The runner changes only the held active specialist, keeps
segment admission unchanged, and rejects locked-test paths.

For cross-modal conflict checks, use
`scripts/build_bearing_stress_cache.py` to create a new, deterministic
Development cache, then invoke
`scripts/run_bearing_stress_evaluation_dev.sh`. The runner needs explicit
environment variables for every data/cache/checkpoint path; this is deliberate
so that no private machine layout is embedded in the released code. It accepts
only `bearing_bias`, `temporal_lag`, or `visual_dropout`, uses evaluation-only
mode, rejects locked-test paths, and refuses to overwrite an artifact. These
audits use a frozen fusion checkpoint and must not be used to retune it.

For the paper's behavior curves, `scripts/run_visual_conflict_sweep_dev.sh`
wraps the same single-point evaluator and the frozen causal compositor. It
evaluates bearing bias at 1--6 degrees, temporal lag at 2, 4, and 8 frames,
and seeded visual-evidence dropout at 25, 50, and 75 percent. All three axes
reuse the canonical clean Development replay as their zero-severity point.
This zero-severity cache is not the separate common-input operator cache;
see [DEVELOPMENT_REPLAYS.md](DEVELOPMENT_REPLAYS.md) for both records.
The batch runner requires explicit frozen paths, rejects locked-test output
paths, writes no checkpoint, and emits paired guard-off/protected metrics.
`scripts/summarize_visual_conflict_sweep.py` verifies the 9,969-frame scope,
constant radar reference, clean-point identity, and source hashes before
creating `visual_conflict_sweep.csv`.

## Evidence synchronization

`contracts/paper_result_contract.json` is the machine-readable bridge between the
manuscript and this workspace. Before packaging, run the complete test suite; it
checks the split counts, headline locked metrics, bootstrap interval, source hashes,
and the forward-coordinate terminology. The evidence locations and their permitted
claims are listed in `docs/EVIDENCE_INDEX.md`.
