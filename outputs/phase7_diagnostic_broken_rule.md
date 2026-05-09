# Phase 7 diagnostic — DLC keypoint-anchored crop

- Phase 5 primary AUC: **0.7985**
- Phase 6(b) face-bbox AUC: **0.4689**
- Phase 7 DLC AUC: **0.5788**
- Δ vs Phase 5: **-0.2198**
- Δ vs Phase 6(b): **+0.1099**
- Phase 7 bootstrap CI: [0.4179, 0.7500]

## Locked gates

- **G1** (AUC ≥ 0.70): FAIL (0.5788)
- **G2 load-bearing** (AUC ≥ 0.7485): FAIL
- **G3 reportable** (median IoU ≥ 0.6): FAIL (0.0000)

**Verdict: CONFIDENT_MISPLACEMENT_FAIL**

**Next action**: Geometry/side-assignment review first; if that doesn't recover, orientation-aware Phase 8a.

## Per-clip categories vs Phase 5 primary

| Category | n | Share |
|---|---:|---:|
| BOTH_RIGHT | 16 | 47.1% |
| BOTH_WRONG | 5 | 14.7% |
| DLC_NEWLY_RECOVERED | 2 | 5.9% |
| DLC_NEWLY_LOST | 11 | 32.4% |

Net shift DLC_NEWLY_RECOVERED − DLC_NEWLY_LOST = **-9** clips

## Per-clip categories vs Phase 6(b) face-bbox

| Category | n | Share |
|---|---:|---:|
| BOTH_RIGHT | 9 | 26.5% |
| BOTH_WRONG | 10 | 29.4% |
| DLC_BEATS_FBB | 9 | 26.5% |
| FBB_BEATS_DLC | 6 | 17.6% |

## Routing diagnostic (Stage 1 §9 dual-axis)

- DLC_NEWLY_LOST count: 11
- Mean target-eye-keypoint confidence on lost clips: **0.8944** (thresholds: <0.5=bottleneck, ≥0.7=high)
- Median IoU vs Phase 5 manual on lost clips: **0.0000** (thresholds: ≤0.3=off-eye, ≥0.5=on-eye)

### Secondary diagnostic (informational, NOT routing)

- OE-concentration (n_OE_in_lost / n_lost): 1/11 = 9.1%
  (Phase 6(b) was 6.2%; this is informational for whether DLC handles OE clips face-bbox failed on)

## IoU vs Phase 5 manual eye boxes (all 34 clips)

- Median IoU: **0.0000** (G3 gate ≥ 0.6: FAIL)
- Mean IoU: 0.1203
- Clips with IoU ≤ 0.30 (off-eye): 27/34
- Clips with IoU ≥ 0.50 (on-eye): 1/34

## Orientation-extreme clips — per-clip breakdown

| Clip | vs P5 | vs P6b | P5 score | P6b score | P7 score | IoU vs manual | mid kp conf |
|---|---|---|---:|---:|---:|---:|---:|
| `action_S9.mp4_4_.mp4` | DLC_NEWLY_RECOVERED | BOTH_RIGHT | -0.117 | +0.511 | +1.438 | 0.392 | 0.893 |
| `action_S9.mp4_7_.mp4` | BOTH_RIGHT | BOTH_RIGHT | +0.165 | +1.235 | +0.281 | 0.358 | 0.894 |
| `background_S1.mp4_12_.mp4` | DLC_NEWLY_LOST | FBB_BEATS_DLC | +1.315 | +2.382 | -0.563 | 0.000 | 0.899 |
| `background_S5.mp4_10_.mp4` | BOTH_RIGHT | DLC_BEATS_FBB | +0.699 | -0.406 | +1.371 | 0.275 | 0.934 |

## Swap pair (action_S5_5 ↔ bg_S10_3, ambiguous-side fallback)

| Clip | vs P5 | P5 score | P7 score | IoU vs manual |
|---|---|---:|---:|---:|
| `action_S5.mp4_5_.mp4` | BOTH_WRONG | -0.413 | -0.114 | 0.239 |
| `background_S10.mp4_3_.mp4` | DLC_NEWLY_LOST | +0.297 | -0.851 | 0.000 |

## action_S9.mp4_4_ (most informative single clip per Stage 1 §10)

- IoU vs Phase 5 manual: 0.392
- Mid-frame keypoint conf: 0.893
- Phase 5 score: -0.117
- Phase 6(b) score: +0.511
- Phase 7 score: +1.438
- Phase 7 correct: ✓

**Verdict (per locked routing in Stage 1 §10)**: AMBIGUOUS (correct=True, IoU=0.392)

## All clips — per-clip table

| Clip | Source | Truth | P5 | P6b | P7 | Δ vs P5 | vs P5 | vs P6b | IoU | mid kp conf |
|---|---|---|---:|---:|---:|---:|---|---|---:|---:|
| `background_S1.mp4_12_.mp4` | S1 | ACTION | +1.315 | +2.382 | -0.563 | -1.878 | DLC_NEWLY_LOST | FBB_BEATS_DLC | 0.000 | 0.899 |
| `action_S3.mp4_2_.mp4` | S3 | ACTION | +1.100 | +0.281 | -0.572 | -1.672 | DLC_NEWLY_LOST | FBB_BEATS_DLC | 0.234 | 0.989 |
| `action_S4.mp4_15_.mp4` | S4 | ACTION | +0.060 | -0.652 | -1.294 | -1.354 | DLC_NEWLY_LOST | BOTH_WRONG | 0.063 | 0.758 |
| `background_S10.mp4_3_.mp4` | S10 | ACTION | +0.297 | -1.975 | -0.851 | -1.148 | DLC_NEWLY_LOST | BOTH_WRONG | 0.000 | 0.868 |
| `action_S10.mp4_0_.mp4` | S10 | ACTION | +0.042 | -1.703 | -0.891 | -0.933 | DLC_NEWLY_LOST | BOTH_WRONG | 0.000 | 0.855 |
| `background_S1.mp4_7_.mp4` | S1 | BACKGROUND | -0.116 | +0.154 | -1.046 | -0.929 | BOTH_RIGHT | DLC_BEATS_FBB | 0.000 | 0.894 |
| `background_S8.mp4_7_.mp4` | S8 | ACTION | +1.005 | +1.028 | +0.112 | -0.893 | BOTH_RIGHT | BOTH_RIGHT | 0.000 | 0.922 |
| `background_S12.mp4_9_.mp4` | S12 | ACTION | +0.499 | +0.447 | -0.302 | -0.802 | DLC_NEWLY_LOST | FBB_BEATS_DLC | 0.000 | 0.921 |
| `background_S2.mp4_10_.mp4` | S2 | ACTION | +0.952 | -0.710 | +0.199 | -0.753 | BOTH_RIGHT | DLC_BEATS_FBB | 0.000 | 0.888 |
| `action_S3.mp4_8_.mp4` | S3 | ACTION | +2.131 | +1.984 | +1.390 | -0.741 | BOTH_RIGHT | BOTH_RIGHT | 0.000 | 0.860 |
| `background_S10.mp4_11_.mp4` | S10 | ACTION | +0.318 | -0.148 | -0.176 | -0.493 | DLC_NEWLY_LOST | BOTH_WRONG | 0.000 | 0.883 |
| `background_S11.mp4_0_.mp4` | S11 | ACTION | +1.334 | +0.845 | +0.920 | -0.413 | BOTH_RIGHT | BOTH_RIGHT | 0.000 | 0.918 |
| `action_S6.mp4_2_.mp4` | S6 | ACTION | +0.517 | -0.764 | +0.293 | -0.224 | BOTH_RIGHT | DLC_BEATS_FBB | 0.000 | 0.924 |
| `background_S12.mp4_2_.mp4` | S12 | BACKGROUND | +0.516 | +0.679 | +0.328 | -0.189 | BOTH_WRONG | BOTH_WRONG | 0.362 | 0.954 |
| `action_S11.mp4_6_.mp4` | S11 | BACKGROUND | +0.421 | +0.309 | +0.330 | -0.092 | BOTH_WRONG | BOTH_WRONG | 0.000 | 0.888 |
| `action_S5.mp4_2_.mp4` | S5 | BACKGROUND | -1.057 | +0.445 | -1.133 | -0.076 | BOTH_RIGHT | DLC_BEATS_FBB | 0.474 | 0.955 |
| `action_S11.mp4_0_.mp4` | S11 | BACKGROUND | +1.446 | +0.269 | +1.385 | -0.061 | BOTH_WRONG | BOTH_WRONG | 0.000 | 0.899 |
| `background_S7.mp4_12_.mp4` | S7 | BACKGROUND | -0.490 | +0.460 | -0.489 | +0.001 | BOTH_RIGHT | DLC_BEATS_FBB | 0.000 | 0.951 |
| `action_S9.mp4_7_.mp4` | S9 | ACTION | +0.165 | +1.235 | +0.281 | +0.116 | BOTH_RIGHT | BOTH_RIGHT | 0.358 | 0.894 |
| `background_S12.mp4_7_.mp4` | S12 | ACTION | +1.066 | +1.015 | +1.191 | +0.125 | BOTH_RIGHT | BOTH_RIGHT | 0.000 | 0.915 |
| `background_S8.mp4_3_.mp4` | S8 | ACTION | -0.940 | +0.164 | -0.749 | +0.191 | BOTH_WRONG | FBB_BEATS_DLC | 0.000 | 0.955 |
| `background_S2.mp4_11_.mp4` | S2 | ACTION | +0.346 | +0.483 | +0.573 | +0.227 | BOTH_RIGHT | BOTH_RIGHT | 0.000 | 0.677 |
| `action_S5.mp4_5_.mp4` | S5 | ACTION | -0.413 | +0.089 | -0.114 | +0.300 | BOTH_WRONG | FBB_BEATS_DLC | 0.239 | 0.959 |
| `action_S8.mp4_12_.mp4` | S8 | BACKGROUND | -0.905 | +0.986 | -0.531 | +0.375 | BOTH_RIGHT | DLC_BEATS_FBB | 0.000 | 0.893 |
| `background_S6.mp4_2_.mp4` | S6 | BACKGROUND | -0.474 | +0.002 | +0.104 | +0.578 | DLC_NEWLY_LOST | BOTH_WRONG | 0.303 | 0.922 |
| `background_S6.mp4_3_.mp4` | S6 | BACKGROUND | -0.580 | -0.359 | +0.006 | +0.586 | DLC_NEWLY_LOST | FBB_BEATS_DLC | 0.283 | 0.920 |
| `action_S2.mp4_7_.mp4` | S2 | BACKGROUND | -0.977 | -0.104 | -0.346 | +0.631 | BOTH_RIGHT | BOTH_RIGHT | 0.618 | 0.949 |
| `background_S5.mp4_10_.mp4` | S5 | ACTION | +0.699 | -0.406 | +1.371 | +0.672 | BOTH_RIGHT | DLC_BEATS_FBB | 0.275 | 0.934 |
| `action_S4.mp4_0_.mp4` | S4 | ACTION | +0.289 | -1.193 | +1.134 | +0.845 | BOTH_RIGHT | DLC_BEATS_FBB | 0.000 | 0.897 |
| `background_S3.mp4_3_.mp4` | S3 | ACTION | -0.325 | +0.243 | +0.572 | +0.896 | DLC_NEWLY_RECOVERED | BOTH_RIGHT | 0.490 | 0.965 |
| `background_S7.mp4_17_.mp4` | S7 | BACKGROUND | -0.740 | +0.711 | +0.192 | +0.932 | DLC_NEWLY_LOST | BOTH_WRONG | 0.000 | 0.936 |
| `background_S9.mp4_8_.mp4` | S9 | BACKGROUND | -0.266 | +0.263 | +0.688 | +0.955 | DLC_NEWLY_LOST | BOTH_WRONG | 0.000 | 0.868 |
| `background_S7.mp4_9_.mp4` | S7 | BACKGROUND | -1.266 | +0.049 | -0.229 | +1.037 | BOTH_RIGHT | DLC_BEATS_FBB | 0.000 | 0.937 |
| `action_S9.mp4_4_.mp4` | S9 | ACTION | -0.117 | +0.511 | +1.438 | +1.555 | DLC_NEWLY_RECOVERED | BOTH_RIGHT | 0.392 | 0.893 |
