# Project Layout

This directory packages the local source of the DREAM-BEV Information Fusion
system. Private data, frozen weights and some upstream dependencies remain
external prerequisites; see `DATA_AND_WEIGHTS.md` and `THIRD_PARTY.md`.

| Path | Purpose |
| --- | --- |
| `contracts/` | Frozen sequence split, frame manifest, route labels, and causal policy schedule. |
| `configs/` | Portable configuration examples; machine-specific paths are ignored by Git. |
| `src/radar_bev/` | Radar anchor, route specialists, losses, decoding, metrics, and protocol validation. |
| `src/rv_evidence_net/` | RGB detection evidence used to prepare visual training inputs. |
| `src/rv_bearing_net/` | Causal RGB bearing evidence for lateral correction. |
| `src/condition_policy/` | Structured RGB condition-policy parsing and validation. |
| `src/rvfusion/` | Radar-H-forward-coordinate-preserving candidate posterior, held segment policy, causal consistency guard, and exact identity fallback. |
| `scripts/` | Training, cache preparation, policy materialization, evaluation, and contract validation entry points. |
| `tests/` | Contract and regression tests required before an experiment is accepted. |
| `third_party/` | Pinned public source trees used by the reported radar and policy adaptations. |
| `external/` | CRUW and CARRADA training/evaluation adapters and their separate protocols. |
| `paper/tables/` | Unmodified numerical tables from the bound manuscript. |
| `artifacts/` | Empty-by-default destination for validated regenerated artifacts only. |

The `radar_bev` and `rvfusion` packages contain source validation and checkpoint-loading helpers
for the three task-adapted public fusion controls. Their official repositories remain
source-pinned by URL, commit, and required-file hashes.

## Single source of truth

`contracts/frame_manifest.csv`, `contracts/routing_manifest.jsonl`, and
`contracts/policy_schedule.json` define the data split, route supervision, and causal
policy schedule. Scripts must consume these contracts rather than embedded sequence
lists, experiment-specific overrides, or generated traces.
The full frame manifest is an on-request data asset, not part of the public
Git tree. A missing manifest must be obtained and hash-verified, not regenerated
from arbitrary data or silently replaced with a new split.

## Generated content

Bulk checkpoints, caches, predictions, evaluation reports, and figures are generated
outside the source tree or under ignored output paths. The release contains only the
current manuscript mainline plus the compact, hash-audited evidence snapshots listed
in `docs/EVIDENCE_INDEX.md`. Before treating a generated fusion result as canonical, run
`scripts/validate_policy_contract.py` against its held policy and
`scripts/replay_causal_cross_modal_guard.py` against the deployed replay.
