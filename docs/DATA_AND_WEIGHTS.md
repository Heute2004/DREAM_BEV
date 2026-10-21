# Data and weights

## Included

Complete-sequence split definitions, route contracts, policy timing, source,
tests, and compact frozen metric reports are included. The complete frame
manifest contains private position labels and is supplied on request, alongside
frame-level replays. Its original SHA256 is retained in `contracts/protocol.json`.
All 15 withheld frame-level files are listed in `contracts/on_request_assets.json`.
No raw images, ADC arrays or trained weights are included.

## Required for model execution

Request private data and paper checkpoints from the contacts in the README.
Configure a local copy of `configs/data_paths.example.yaml`, ignored by Git.

| Asset | Role |
| --- | --- |
| RA/RA-MTI and BEV labels | Radar/expert fitting and evaluation |
| RGB, person labels, teacher manifest/profile | Person-encoder fitting |
| Frozen person-encoder checkpoint | Top-three proposal cache |
| Geometry teacher cache | Bearing training only |
| Radar, five experts, bearing and fusion checkpoints | Frozen model inference |
| InternVL3-1B base plus adapter, or CNN policy | Policy inference |
| Policy trace, visual cache and radar trace | Output replay without retraining |

The person encoder is bound to SHA256
`7373a7f5e943a96414a0ce11be2593e4117c7a51f70a7cec1cfe73e847053204`.
Its metadata is in `evidence/implementation/`. Other frozen reports retain their
checkpoint hashes, but this packaging run did not fetch or revalidate all remote
weights. Never substitute a similarly named checkpoint.

The private runtime `--data-root` contains `seq_022`, etc. Manifest paths include
`seq_022/Radar/RA/frame_0000.npy` and `seq_022/RGB/frame_0000.jpg`. The on-disk
spelling `RA-MIT` is intentional; the paper calls the representation RA-MTI.

Person-encoder training requires explicit `--teacher-manifest` and
`--teacher-profile` files. The published sequence contract controls fitting and
validation; Locked Test split metadata does not authorize loading test imagery.

## Verification boundary

`verify_release.py` and unit tests check integrity and contracts. Re-exporting
archived metrics is not an independent numerical rerun. Exact retraining also
needs the archived per-stage arguments, CUDA environment, external assets and
upstream dependencies. CLI defaults are not asserted to be the selected paper
configuration. No training or raw Locked Test evaluation is launched by package
verification.

In a public checkout, three tests explicitly skip because the private frame
manifest is unavailable. This is reported rather than replaced by fabricated
test data. After obtaining the original manifest, rerun the full suite and
verify its original hash before any model execution.
