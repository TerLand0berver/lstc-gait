# LSTC/LSTP Theory and Complexity

This note formalizes the Local Spatio-Temporal Convolution (LSTC), the asymmetric spatio-temporal block, and Local Spatio-Temporal Pooling (LSTP), and estimates their parameter/computation complexity.

## Notation
- Input tensor: X ∈ R^{N×C_in×T×H×W}
- Height stripes: partition H into S disjoint intervals {H_s}
- Depthwise 3D kernel (per-channel): K_d ∈ R^{C_in×1×kT×kH×kW}
- Pointwise mixer: W_pw ∈ R^{C_out×C_in×1×1×1}
- 1×1×1 fusion (after branch concat): W_fuse ∈ R^{C_out×(C_t+C_s+C_j)×1×1×1}

## LSTC (Local Spatio-Temporal Convolution)
For each stripe s:
1) Crop: X_s = X[:, :, :, H_s, :]
2) Depthwise 3D conv:
   Y_s = Conv3D_depthwise(X_s, K_d)  where groups = C_in
3) Concatenate along height: Y = Concat_H({Y_s})
4) Pointwise 1×1×1 mix across channels: Z = Conv3D(Y, W_pw)

Remarks
- Locality: depthwise conv is restricted to each stripe, preventing cross-stripe leakage.
- Adaptivity: learnable K_d across time/space jointly captures local gait motion; W_pw mixes channels globally.

Parameters
- Depthwise: C_in · kT · kH · kW
- Pointwise: C_in · C_out
Total: C_in · (kT · kH · kW + C_out)

FLOPs (per output element)
- Depthwise: ~ C_in · kT · kH · kW
- Pointwise: ~ C_in · C_out
(Scaled by output size T′·H′·W′)

## Asymmetric Spatio-Temporal Block
Three parallel branches on the same input X:
- Temporal-only: Conv3D(kT,1,1) → BN → ReLU (output C_t)
- Spatial-only: Conv3D(1,kH,kW) → BN → ReLU (output C_s)
- Joint (LSTC): LSTC(kT,kH,kW,S) (output C_j)
Fuse by channel concat and 1×1×1 projection to C_out.

Rationale
- Temporal-only: captures motion cues without spatial mixing.
- Spatial-only: captures shape/texture patterns per frame.
- Joint: captures coupled local spatio-temporal patterns guided by stripes.

## LSTP (Local Spatio-Temporal Pooling)
Given feature map F ∈ R^{N×C×T×H×W}:
1) For each stripe s: F_s = F[:, :, :, H_s, :]
2) Frame-wise spatial average: f_s(t) = Mean_{H_s,W}(F_s[:, :, t, :, :]) ∈ R^{N×C}
3) Scoring per frame: score_s(t) = || f_s(t) ||_2
4) Select top-k frames T_k(s) by score_s(t)
5) Stripe vector: v_s = (1/k) · Σ_{t∈T_k(s)} f_s(t)
6) Final sequence descriptor: z = Concat_{s=1..S}(v_s) ∈ R^{N×(C·S)} (optionally projected to D)

Why LSTP
- Emphasizes the most discriminative temporal instances per local region; robust to noisy/irrelevant frames.

## Design Guidelines
- Stripes S: 6–10 (default 8)
- Kernels: start with kT=3, kH=7, kW=3; decrease kH in deeper layers
- LSTP top-k: 2–4
- Downsampling: prefer spatial-only pooling; keep T to support LSTP

## Ablation Ideas
- Remove LSTC (only temporal/spatial branches)
- Replace LSTP with simple temporal average/max
- Vary S, kT, kH, kW, top-k
- Share vs unshare local conv weights across stripes

## Notes
- LSTC is equivalent to depthwise-separable 3D conv with height-localized support per stripe, followed by a global 1×1×1 projection.
- The asymmetric block can be re-parameterized (train with branches, deploy as fused 3D) in future work.
