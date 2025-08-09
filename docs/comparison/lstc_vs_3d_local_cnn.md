# LSTC/LSTP vs 3D-Local-CNN (for Gait)

This note compares our LSTC/LSTP design to a 3D-Local-CNN style approach (e.g., Aliyun "3D-Local-CNN-for-Gait-Recognition").

## Design paradigm
- 3D-Local-CNN: introduce adaptive selection of local 3D volumes on top of regular 3D CNN blocks; plug-and-play for existing backbones.
- LSTC/LSTP: impose a height-stripe prior; within each stripe do depthwise 3D + 1x1x1 (LSTC), use asymmetric branches (temporal/spatial/joint) and stripe-wise top-k temporal pooling (LSTP).

## Locality and prior
- 3D-Local-CNN: learns flexible, irregular local regions (cross-joint/slanted patterns), better for misalignment/occlusion.
- LSTC/LSTP: structured prior (head/torso/legs) aligns well with normalized silhouettes; stronger bias, simpler training.

## Capacity, bias and robustness
- 3D-Local-CNN: higher expressiveness; needs careful regularization and data to avoid instability.
- LSTC/LSTP: explicit factorization (T/S/TS), stable optimization; hard top-k can be replaced by soft/attentive variants.

## Efficiency and complexity
- 3D-Local-CNN: extra selection/gating brings overhead during train/infer.
- LSTC/LSTP: depthwise 3D + 1x1x1 is lean; easier to deploy.

## Engineering and ecosystem
- 3D-Local-CNN: earlier Python/CUDA stacks may require migration.
- LSTC/LSTP: PyTorch 2.x, uv env, CI, DDP, datasets (CASIA-B/OU-MVLP), pipelines, ablations, EMA/GradClip/AMP.

## Complementary fusion ideas
- Stripe prior + adaptive gating: keep stripes, add learned spatio-temporal gate within stripes over LSTC outputs.
- Soft top-k / attention for LSTP: replace hard selection with differentiable weighting.
- Multi-scale temporal: dilated kT or multi-branch temporal scales.

## Summary
- 3D-Local-CNN excels in flexible local modeling; LSTC/LSTP excels in efficiency/robustness and pipeline readiness. Combining stripe priors with adaptive mechanisms can yield the best of both worlds.
