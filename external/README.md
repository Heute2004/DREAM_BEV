# Public-data adaptations

Existing project scripts are copied with their transitive local imports. No new
network or comparator is added. Search launchers and switched-split protocols
are not shipped. Use the launcher to set import paths:

```bash
python scripts/run_public.py cruw train_fullscene_dream_bev --help
python scripts/run_public.py cruw train_cruw_expert --help
python scripts/run_public.py cruw train_cruw_paper_expert --help
python scripts/run_public.py cruw train_paper_rv_bearing --help
python scripts/run_public.py cruw train_cruw_rvfusion --help
python scripts/run_public.py cruw train_cruw_public_fusion_baseline --help
python scripts/run_public.py cruw replay_cruw_rvfusion_consistency_guard --help
python scripts/run_public.py carrada train_radar_anchor --help
python scripts/run_public.py carrada train_public_rodnet_baseline --help
python scripts/run_public.py carrada train_public_temporal_baseline --help
```

## CRUW

The protocol uses 32 full-scene fitting sequences and four Development sequences,
with a 12-sequence single-person fitting subset. Development: 001, 007, 028, 039.
Official CRUW test data are excluded. Each fusion block shares its PB-MHBR or
GG-SRRE radar anchor. Use the decoder recorded by that block's frozen report;
the 0.70 decoder files are not a replacement for every archived decoder.

The public RGB encoder is an adapted model, not the private frozen-proposal
checkpoint. Grids, fitting and evaluation remain separate from the private
full-system protocol. Cars and cyclists remain in the input; no distractor-
erasure script is included.

## CARRADA

There are 24 fitting and six Development sequences. Fit combines official Train
and Test; Development uses official Validation. This is not an official
leaderboard result. The manuscript uses 496 unique-pedestrian Development frames.
A separate 425-frame fixed-identity episode replay is included but must not
silently replace the paper's population.

The radar controls include CDC, HG, FCN-8s and E/T core adaptations.
`private_erodnet_core` expects the private `radar_bev` source root, exposed by
the launcher. Check the frozen variant and episode option before evaluation.

Obtain raw datasets and devkits separately. Install the `public` and CUDA
`mamba` dependencies as needed. CLI/source validation is not a GPU experiment.
