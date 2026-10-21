# InternVL Dependency

The condition policy uses the upstream InternVL training implementation. Clone the
upstream repository into `third_party/InternVL`, checkout commit
`2410d1dbf208f0e799459aff9376e5747dbf41a2`, and place the pristine
`OpenGVLab/InternVL3-1B` base model at the configured model path.

```bash
git clone https://github.com/OpenGVLab/InternVL.git third_party/InternVL
git -C third_party/InternVL checkout 2410d1dbf208f0e799459aff9376e5747dbf41a2
```

The local project owns dataset construction, the output contract, causal policy
materialization, evaluation, and all fusion code. The upstream repository is used only
for multimodal fine-tuning and inference. Use
`scripts/train_condition_policy.sh` for the pinned LoRA contract.
