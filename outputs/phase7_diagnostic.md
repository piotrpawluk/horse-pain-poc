# Phase 7 diagnostic — DLC keypoint-anchored crop

- Phase 5 primary AUC: **0.7985**
- Phase 6(b) face-bbox AUC: **0.4689**
- Phase 7 DLC AUC: **0.8462**
- Δ vs Phase 5: **+0.0476**
- Δ vs Phase 6(b): **+0.3773**
- Phase 7 bootstrap CI: [0.6629, 0.9615]

## Locked gates

- **G1** (AUC ≥ 0.70): PASS (0.8462)
- **G2 load-bearing** (AUC ≥ 0.7485): PASS
- **G3 reportable** (median IoU ≥ 0.6): FAIL (0.3573)

**Verdict: OUTPERFORM_PHASE_5_AUC_ONLY**

**Next action**: AUC > Phase 5 (Δ +0.0476) but paired-DeLong p=0.6194 ≥ 0.05; per G2 asymmetry (AUC load-bearing, paired p supportive), the AUC evidence governs. Phase 8 confirmation at higher N recommended.

## Per-clip categories vs Phase 5 primary

| Category | n | Share |
|---|---:|---:|
| BOTH_RIGHT | 21 | 61.8% |
| BOTH_WRONG | 1 | 2.9% |
| DLC_NEWLY_RECOVERED | 6 | 17.6% |
| DLC_NEWLY_LOST | 6 | 17.6% |

Net shift DLC_NEWLY_RECOVERED − DLC_NEWLY_LOST = **+0** clips

## Per-clip categories vs Phase 6(b) face-bbox

| Category | n | Share |
|---|---:|---:|
| BOTH_RIGHT | 15 | 44.1% |
| BOTH_WRONG | 7 | 20.6% |
| DLC_BEATS_FBB | 12 | 35.3% |
| FBB_BEATS_DLC | 0 | 0.0% |

## Routing diagnostic (Stage 1 §9 dual-axis)

- DLC_NEWLY_LOST count: 6
- Mean target-eye-keypoint confidence on lost clips: **0.8902** (thresholds: <0.5=bottleneck, ≥0.7=high)
- Median IoU vs Phase 5 manual on lost clips: **0.3882** (thresholds: ≤0.3=off-eye, ≥0.5=on-eye)

### Secondary diagnostic (informational, NOT routing)

- OE-concentration (n_OE_in_lost / n_lost): 1/6 = 16.7%
  (Phase 6(b) was 6.2%; this is informational for whether DLC handles OE clips face-bbox failed on)

## IoU vs Phase 5 manual eye boxes (all 34 clips)

- Median IoU: **0.3573** (G3 gate ≥ 0.6: FAIL)
- Mean IoU: 0.3778
- Clips with IoU ≤ 0.30 (off-eye): 10/34
- Clips with IoU ≥ 0.50 (on-eye): 5/34

## Orientation-extreme clips — per-clip breakdown

| Clip | vs P5 | vs P6b | P5 score | P6b score | P7 score | IoU vs manual | mid kp conf |
|---|---|---|---:|---:|---:|---:|---:|
| `action_S9.mp4_4_.mp4` | DLC_NEWLY_RECOVERED | BOTH_RIGHT | -0.117 | +0.511 | +1.104 | 0.330 | 0.900 |
| `action_S9.mp4_7_.mp4` | BOTH_RIGHT | BOTH_RIGHT | +0.165 | +1.235 | +1.003 | 0.358 | 0.850 |
| `background_S1.mp4_12_.mp4` | BOTH_RIGHT | BOTH_RIGHT | +1.315 | +2.382 | +1.636 | 0.390 | 0.920 |
| `background_S5.mp4_10_.mp4` | DLC_NEWLY_LOST | BOTH_WRONG | +0.699 | -0.406 | -0.582 | 0.275 | 0.804 |

## Swap pair (action_S5_5 ↔ bg_S10_3, ambiguous-side fallback)

| Clip | vs P5 | P5 score | P7 score | IoU vs manual |
|---|---|---:|---:|---:|
| `action_S5.mp4_5_.mp4` | DLC_NEWLY_RECOVERED | -0.413 | +0.734 | 0.239 |
| `background_S10.mp4_3_.mp4` | DLC_NEWLY_LOST | +0.297 | -0.048 | 0.338 |

## action_S9.mp4_4_ (most informative single clip per Stage 1 §10)

- IoU vs Phase 5 manual: 0.330
- Mid-frame keypoint conf: 0.900
- Phase 5 score: -0.117
- Phase 6(b) score: +0.511
- Phase 7 score: +1.104
- Phase 7 correct: ✓

**Verdict (per locked routing in Stage 1 §10)**: AMBIGUOUS (correct=True, IoU=0.330)

## All clips — per-clip table

| Clip | Source | Truth | P5 | P6b | P7 | Δ vs P5 | vs P5 | vs P6b | IoU | mid kp conf |
|---|---|---|---:|---:|---:|---:|---|---|---:|---:|
| `action_S11.mp4_0_.mp4` | S11 | BACKGROUND | +1.446 | +0.269 | -0.250 | -1.696 | DLC_NEWLY_RECOVERED | DLC_BEATS_FBB | 0.275 | 0.935 |
| `background_S6.mp4_2_.mp4` | S6 | BACKGROUND | -0.474 | +0.002 | -1.787 | -1.313 | BOTH_RIGHT | DLC_BEATS_FBB | 0.303 | 0.922 |
| `background_S5.mp4_10_.mp4` | S5 | ACTION | +0.699 | -0.406 | -0.582 | -1.281 | DLC_NEWLY_LOST | BOTH_WRONG | 0.275 | 0.804 |
| `action_S3.mp4_8_.mp4` | S3 | ACTION | +2.131 | +1.984 | +0.921 | -1.210 | BOTH_RIGHT | BOTH_RIGHT | 0.370 | 0.970 |
| `background_S6.mp4_3_.mp4` | S6 | BACKGROUND | -0.580 | -0.359 | -1.560 | -0.980 | BOTH_RIGHT | BOTH_RIGHT | 0.283 | 0.920 |
| `action_S6.mp4_2_.mp4` | S6 | ACTION | +0.517 | -0.764 | -0.293 | -0.811 | DLC_NEWLY_LOST | BOTH_WRONG | 0.455 | 0.966 |
| `action_S4.mp4_0_.mp4` | S4 | ACTION | +0.289 | -1.193 | -0.316 | -0.605 | DLC_NEWLY_LOST | BOTH_WRONG | 0.235 | 0.978 |
| `background_S2.mp4_10_.mp4` | S2 | ACTION | +0.952 | -0.710 | +0.352 | -0.600 | BOTH_RIGHT | DLC_BEATS_FBB | 0.524 | 0.942 |
| `action_S11.mp4_6_.mp4` | S11 | BACKGROUND | +0.421 | +0.309 | -0.076 | -0.497 | DLC_NEWLY_RECOVERED | DLC_BEATS_FBB | 0.245 | 0.960 |
| `action_S2.mp4_7_.mp4` | S2 | BACKGROUND | -0.977 | -0.104 | -1.348 | -0.371 | BOTH_RIGHT | BOTH_RIGHT | 0.618 | 0.949 |
| `background_S10.mp4_3_.mp4` | S10 | ACTION | +0.297 | -1.975 | -0.048 | -0.345 | DLC_NEWLY_LOST | BOTH_WRONG | 0.338 | 0.967 |
| `background_S11.mp4_0_.mp4` | S11 | ACTION | +1.334 | +0.845 | +1.036 | -0.298 | BOTH_RIGHT | BOTH_RIGHT | 0.312 | 0.925 |
| `action_S3.mp4_2_.mp4` | S3 | ACTION | +1.100 | +0.281 | +0.912 | -0.189 | BOTH_RIGHT | BOTH_RIGHT | 0.240 | 0.943 |
| `background_S8.mp4_7_.mp4` | S8 | ACTION | +1.005 | +1.028 | +0.934 | -0.071 | BOTH_RIGHT | BOTH_RIGHT | 0.266 | 0.932 |
| `action_S4.mp4_15_.mp4` | S4 | ACTION | +0.060 | -0.652 | +0.059 | -0.000 | BOTH_RIGHT | DLC_BEATS_FBB | 0.707 | 0.908 |
| `action_S5.mp4_2_.mp4` | S5 | BACKGROUND | -1.057 | +0.445 | -0.929 | +0.128 | BOTH_RIGHT | DLC_BEATS_FBB | 0.474 | 0.955 |
| `background_S1.mp4_7_.mp4` | S1 | BACKGROUND | -0.116 | +0.154 | +0.054 | +0.171 | DLC_NEWLY_LOST | BOTH_WRONG | 0.439 | 0.968 |
| `background_S10.mp4_11_.mp4` | S10 | ACTION | +0.318 | -0.148 | +0.516 | +0.198 | BOTH_RIGHT | DLC_BEATS_FBB | 0.446 | 0.958 |
| `action_S8.mp4_12_.mp4` | S8 | BACKGROUND | -0.905 | +0.986 | -0.687 | +0.218 | BOTH_RIGHT | DLC_BEATS_FBB | 0.494 | 0.925 |
| `background_S7.mp4_12_.mp4` | S7 | BACKGROUND | -0.490 | +0.460 | -0.198 | +0.293 | BOTH_RIGHT | DLC_BEATS_FBB | 0.373 | 0.918 |
| `background_S1.mp4_12_.mp4` | S1 | ACTION | +1.315 | +2.382 | +1.636 | +0.321 | BOTH_RIGHT | BOTH_RIGHT | 0.390 | 0.920 |
| `background_S12.mp4_7_.mp4` | S12 | ACTION | +1.066 | +1.015 | +1.502 | +0.436 | BOTH_RIGHT | BOTH_RIGHT | 0.357 | 0.942 |
| `background_S12.mp4_9_.mp4` | S12 | ACTION | +0.499 | +0.447 | +0.948 | +0.448 | BOTH_RIGHT | BOTH_RIGHT | 0.341 | 0.949 |
| `background_S2.mp4_11_.mp4` | S2 | ACTION | +0.346 | +0.483 | +0.878 | +0.532 | BOTH_RIGHT | BOTH_RIGHT | 0.556 | 0.765 |
| `background_S7.mp4_17_.mp4` | S7 | BACKGROUND | -0.740 | +0.711 | -0.172 | +0.568 | BOTH_RIGHT | DLC_BEATS_FBB | 0.303 | 0.974 |
| `background_S3.mp4_3_.mp4` | S3 | ACTION | -0.325 | +0.243 | +0.259 | +0.583 | DLC_NEWLY_RECOVERED | BOTH_RIGHT | 0.490 | 0.965 |
| `background_S7.mp4_9_.mp4` | S7 | BACKGROUND | -1.266 | +0.049 | -0.683 | +0.584 | BOTH_RIGHT | DLC_BEATS_FBB | 0.282 | 0.886 |
| `background_S9.mp4_8_.mp4` | S9 | BACKGROUND | -0.266 | +0.263 | +0.330 | +0.597 | DLC_NEWLY_LOST | BOTH_WRONG | 0.669 | 0.970 |
| `action_S9.mp4_7_.mp4` | S9 | ACTION | +0.165 | +1.235 | +1.003 | +0.838 | BOTH_RIGHT | BOTH_RIGHT | 0.358 | 0.850 |
| `background_S12.mp4_2_.mp4` | S12 | BACKGROUND | +0.516 | +0.679 | +1.588 | +1.072 | BOTH_WRONG | BOTH_WRONG | 0.000 | 0.686 |
| `action_S5.mp4_5_.mp4` | S5 | ACTION | -0.413 | +0.089 | +0.734 | +1.147 | DLC_NEWLY_RECOVERED | BOTH_RIGHT | 0.239 | 0.959 |
| `action_S9.mp4_4_.mp4` | S9 | ACTION | -0.117 | +0.511 | +1.104 | +1.221 | DLC_NEWLY_RECOVERED | BOTH_RIGHT | 0.330 | 0.900 |
| `action_S10.mp4_0_.mp4` | S10 | ACTION | +0.042 | -1.703 | +1.404 | +1.362 | BOTH_RIGHT | DLC_BEATS_FBB | 0.400 | 0.919 |
| `background_S8.mp4_3_.mp4` | S8 | ACTION | -0.940 | +0.164 | +1.168 | +2.108 | DLC_NEWLY_RECOVERED | BOTH_RIGHT | 0.458 | 0.905 |
