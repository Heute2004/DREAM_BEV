# Third-party sources

These adapters are task adaptations, not reproductions of the original 3-D
benchmarks. Source notices are retained; website assets and example media are
omitted. Inspect upstream licences before use or redistribution.

| Component | Source identity |
| --- | --- |
| RODNet CDC/HG and T-RODNet core | `Zhuanglong2/T-RODNet`, `39a2444c2bdf83f56645fb9ea74da1ca2f2eeab9` |
| E-RODNet | `lupeng-xm/E-RODNet`, `da4be8011090ad565a85cc39f849ad9f0688ef14` |
| FCN-8s | `wkentaro/pytorch-fcn`, `4b988509bfd1a613d5e8482595a1c6654047d69b` |
| InternVL | `OpenGVLab/InternVL`, `2410d1dbf208f0e799459aff9376e5747dbf41a2` |
| RC-ROSNet | `Zhuanglong2/RC-ROSNet`, `8ff57d49efa90934dca2032adf7f6625b5ad9248` |
| CRN | `youngskkim/CRN`, `5e9d2fa2f91c714b297e75ca666d27fc4ad0d13d` |
| RCBEVDet | `VDIGPKU/RCBEVDet`, `15c83ccdd5a8cfd3b7c0390eacd8664cf842d513` |

Radar source hashes are enforced by `radar_bev.public_full_adapted`. Fetch fusion
or policy sources separately using `scripts/fetch_public_sources.py`; no model
weights are downloaded. Fusion runtime checks commits and required files.
The RCBEVDet pin includes a nested official-source tree that must not be replaced
by a guessed import from another release.
