# Public Fusion-Core Fixed-Budget Evidence

The three source-backed task adaptations use the same frozen expert-radar
anchor, RGB evidence, 87/34 Train/Development split, seed `20260808`, fixed
four-epoch budget, checkpoint-selection metric, and evaluator. This evidence
does not claim full convergence or optimization parity with the separately
trained protected hierarchy.

| Adapted core | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Selected epoch | Checkpoint SHA256 |
|---|---:|---:|---:|---:|---:|---|
| RC-ROSNet | 0.75493666 | 0.75302701 | 0.75109369 | 0.75173687 | 1 | `74982DC8E5B4512A1A21408639C6FE505595916CA1FCFF7044EA722AA380DDC1` |
| CRN | 0.74207154 | 0.73490337 | 0.70266739 | 0.71335597 | 1 | `899793B4949DFF9C2F05C46C30E2B0E3FEDD3D90486D488D77A5092EA708C493` |
| RCBEVDet | 0.73209832 | 0.61067789 | 0.63087031 | 0.72592978 | 1 | `A86F792A304B7C0AA2BA1CCFE04DB2FD06B3EFF857D4BAD5CF7036959C537DB8` |
