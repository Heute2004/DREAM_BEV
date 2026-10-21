# Release checks

The packaging check on 21 October 2026 used the manuscript identity in
`contracts/paper_binding.json`. It did not train a model, access raw Locked Test
data, change model selection, or replace the manuscript or editable figures.

| Check | Observed result |
| --- | --- |
| Original source-package tests | 144 passed |
| Complete local package, before the public-exclusion check | 149 passed |
| Clean public package, including the public-exclusion check | 147 passed, 3 data-dependent tests skipped |
| Training/cache/evaluation command `--help` checks | 34 passed |
| E/T compact-core implementation comparison | Three class bodies match original AST, excluding docstrings |
| Frozen numerical-record export | Passed; raw-precision and rounded records distinguished |
| Fig. 6 plot-data alignment | 30 source rows, 60 drawn points, 9,969 frames per point; passed |
| Fig. 6 visual inspection | No text/data collisions observed |

Run `python scripts/verify_release.py` for source/evidence hashes and Python
syntax, and `python -m pytest tests -q` for the executable tests. The release
manifest excludes itself, Git metadata and generated outputs.

These are packaging and component checks, not an end-to-end GPU reproduction.
The external data, weights and per-stage selected run configurations listed in
`DATA_AND_WEIGHTS.md` must be available before full training or inference can be
claimed reproduced. GPU timings were not remeasured in this release check.

## Restored dependencies

The previous local package omitted the standalone person-encoder fitting and
bearing teacher-cache entry points, and omitted compact E/T core classes used
by paper controls. This release restores those dependencies from their original
source, exposes local import paths, and preserves the selected network bodies.
No new experimental network is introduced.

The CARRADA E-RODNet-core result is available locally only as a four-decimal
audit transcription. `current_paper_rows.json` labels this precision explicitly;
it does not fabricate full-precision predictions. The older broad comparison
record remains an immutable numerical source, while current table inclusion
comes from `paper/tables/table_14_carrada_external_transfer.tex`.

## Public-data boundary

The remote repository was found to be public during the final upload preflight.
The public release therefore excludes 15 private frame-level annotation/replay
files, recorded by hash in `contracts/on_request_assets.json`. They are retained
in the local complete bundle and supplied on request. Three tests that need the
full frame manifest explicitly skip in a clean public checkout; their success in
the complete local package must not be misreported as a public-data validation.

The packaging test environment was Windows, Python 3.11.15, PyTorch
2.11.0+cu130, torchvision 0.26.0+cu130, NumPy 2.4.4 and pytest 9.1.1.
Plotting used Matplotlib 3.11.0 and pandas 2.3.3. This environment is recorded
for these checks only; it is not the archived GPU training environment. The
original dependency pins remain in `pyproject.toml`.
