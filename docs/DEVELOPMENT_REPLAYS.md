# Two clean Development replays

The operator comparison and the conflict sweep use different archived candidate
outputs. They cover 9,969 scored Development frames and share the radar
reference, but their clean fusion values are not interchangeable.

| Clean replay | Guard off AP@0.30 | Protected AP@0.30 | Off / protected harmed fraction |
| --- | ---: | ---: | ---: |
| Common-input operator comparison | 0.79175025 | 0.77420405 | 0.26833183 / 0.13391514 |
| Zero-severity conflict control | 0.79254687 | 0.77517955 | 0.20854649 / 0.10472465 |

Machine-readable records are in `evidence/development_replays/operator_clean.json`
and `stress_clean.json`. These are provenance reconstructions from archived
reports and released code, not newly recovered execution logs or new evaluations.
They bind the available input and result hashes and identify unresolved fields.

Both replays use `scripts/replay_causal_cross_modal_guard.py`, alpha 0.10,
threshold 0.50, learned policy admission, and the 32-frame initial policy delay.
The operator replay joins a frozen policy/radar trace with a separately
re-evaluated candidate audit. The clean conflict replay reads policy and
candidate evidence from the same archived audit.

The released replay implementation sorts records by sequence and frame, creates
a fresh guard object, and starts each sequence's EMA at zero. Ineligible frames
do not update the EMA. Learned admission is read from the supplied trace; the
replay does not create or repair its warm-up interval. Guard-off keeps the same
policy trace and candidate coordinates while disabling the guard.

The replay itself loads no model checkpoint. The associated training archive
identifies a fusion checkpoint, but neither clean record alone establishes the
complete upstream cache-generation command, checkpoint binding, or temporal
state initialization. Those missing fields are null rather than inferred from
an adjacent run. The numerical cause of the different candidate outputs has
not been isolated. Guard effects must therefore be compared within each replay,
not by subtracting values across the two caches.

Private frame-level traces, labels, and cached tensors are not published here.
No training, model inference, or Locked Test access was performed to assemble
these records.
