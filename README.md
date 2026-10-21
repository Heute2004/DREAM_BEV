# DREAM-BEV

Code for **DREAM-BEV: Condition-routed radar-vision fusion for human localization**.

DREAM-BEV selects a radar recovery expert from scene conditions, proposes a
bounded visual correction, and withdraws that correction when persistent
cross-modal conflict is detected. RGB does not change the radar-derived forward
coordinate. CNN and VLM policies share the same delayed policy interface.

This repository contains the paper's radar, expert, person-encoder, bearing,
policy, fusion, and evaluation code, plus CRUW and CARRADA adaptation entry
points. Raw participant data, checkpoints, cached tensors, development-search
launchers, and previous manuscript revisions are excluded.
Private frame-level annotations and prediction traces are also supplied on
request. Their hashes remain in the contracts; the public repository does not
publish the full human-position labels.

## Quick start

```bash
python -m pip install -e '.[development]'
python scripts/verify_release.py
python -m pytest tests -q
python scripts/export_paper_results.py --output-dir outputs/paper_results
```

The first check uses only the Python standard library. Exporting archived metrics
does not run a model or read raw Locked Test data. These checks do not reproduce
training from scratch. Three data-dependent tests skip when the on-request
frame manifest is absent; verification reports that missing asset explicitly.

For Figure 6:

```bash
python -m pip install -e '.[figures]'
python scripts/plot_visual_conflict_curves.py
```

Use Linux/CUDA with the `mamba` extra for the archived Mamba-based training
configuration. The CPU test suite checks contracts and components, not published
GPU speed. Public-model dependencies are in the `public` extra. InternVL has a
separate upstream environment; see [third_party/InternVL.md](third_party/InternVL.md).

## Reproduction paths

| Purpose | Entry |
| --- | --- |
| Private-data pipeline | [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) |
| Paper-to-code mapping | [docs/PAPER_ALIGNMENT.md](docs/PAPER_ALIGNMENT.md) |
| Data and pretrained assets | [docs/DATA_AND_WEIGHTS.md](docs/DATA_AND_WEIGHTS.md) |
| Public-data adaptations | [external/README.md](external/README.md) |
| Public source identities | [THIRD_PARTY.md](THIRD_PARTY.md) |
| Evidence index | [docs/EVIDENCE_INDEX.md](docs/EVIDENCE_INDEX.md) |

The private dataset uses 87 Train, 34 Development, and eight Locked Test
sequences. Radar input is causal 16-frame RA/RA-MTI; visual evidence uses an
eight-frame causal history. Policy captures are 30 radar frames apart, with a
32-frame availability delay and identity initialization. Public adaptation
protocols are documented separately.

## Availability

Data will be made available on request. Contact **scutlab407@163.com** or
corresponding author **fengke@scut.edu.cn** for raw radar ADC captures,
synchronized RGB and depth images, processed radar representations, and
annotation layers. Transfer must respect participant permissions and applicable
data terms. Obtain public CRUW/CARRADA data from their original distributors.

Repository: <https://github.com/Heute2004/DREAM_BEV>. Code and aggregate results
are public; private data and frame-level annotations remain request-based.
See [RIGHTS.md](RIGHTS.md) before redistribution.
