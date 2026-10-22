# Training selection and stopping

The supplementary curves are archived Development selection histories, not
new training runs. `evidence/training_stopping/epoch_metrics.csv` contains 82
epoch-level observations from 12 runs. `stopping_summary.csv` records the
selected epoch, terminal epoch, and terminal non-improvement count.

| Component | Selection criterion | Patience | Selected / terminal epoch |
| --- | --- | ---: | ---: |
| DREAM-BEV anchor | AP@0.50, maximize | 3 | 3 / 6 |
| CLOR | AP@0.30, maximize | 4 | 6 / 10 |
| TECS | AP@0.30, maximize | 4 | 3 / 7 |
| LTGE-CEN | AP@0.30, maximize | 4 | 1 / 5 |
| GG-SRRE | AP@0.30, maximize | 4 | 3 / 7 |
| RV-BearingNet | Archived selection score, minimize | 4 | 7 / 11 |
| RVFusion candidate stage | Archived phase score, maximize | 3 | 1 / 4 |
| RC-ROSNet | AP@0.30, maximize | 3 | 1 / 4 |
| CRN | AP@0.30, maximize | 3 | 1 / 4 |
| RCBEVDet | AP@0.30, maximize | 3 | 1 / 4 |
| Universal active | AP@0.30, maximize | 4 | 5 / 9 |
| Universal bank | AP@0.30, maximize | 4 | 7 / 11 |

Each trace first reaches its recorded patience at the terminal epoch. This is
an operational validation stopping criterion, not proof of a global optimum.
It does not imply identical optimization steps, learning rates, or selection
metrics across components. In particular, the public fusion heads' four
recorded epochs include a selected first epoch followed by three non-improving
epochs; they are not evidence of universal architecture superiority.

The RVFusion phase score is a candidate-stage selection score using the
trainer's Development-label oracle gate for checkpoint selection, not the
deployed protected AP. PB-MHBR is not gradient-trained in this private-data
configuration. Frozen pretrained components and policy adaptation do not share
the stopping histories listed here.

The universal controls use learning rate 0.002 and batch size 8. The 0.001 and
batch-size-16 configuration reported for the trained route-specific experts
does not apply to these controls.

Only aggregate curves and source hashes are public. Exact original history and
training-source snapshots are retained with the local manuscript evidence.
