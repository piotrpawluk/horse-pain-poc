# Phase 6 (b) diagnostic — face-bbox-positioned crop

- Phase 5 primary AUC: **0.7985**
- Phase 6 (b) AUC: **0.4689**
- Δ vs Phase 5: **-0.3297**
- Phase 6 (b) bootstrap CI: [0.2600, 0.6856]

## Locked gates

- G1 (AUC ≥ 0.70): **FAIL** (0.4689 < 0.70)
- G2 (AUC ≥ 0.7485): **FAIL**

**Verdict: DISTRIBUTED_FAIL**

**Next action**: DLC SuperAnimal-Quadruped is next; orientation-aware face-bbox would not be sufficient.

## Per-clip categories vs Phase 5 primary

| Category | n | Share |
|---|---:|---:|
| BOTH_RIGHT | 11 | 32.4% |
| BOTH_WRONG | 3 | 8.8% |
| FBB_NEWLY_RECOVERED | 4 | 11.8% |
| FBB_NEWLY_LOST | 16 | 47.1% |

Net shift FBB_NEWLY_RECOVERED − FBB_NEWLY_LOST = **-12** clips

## Failure-mode attribution (locked routing)

- FBB_NEWLY_LOST count: 16
- Of those, in 4 orientation-extreme clips: 1/16
- **loss_concentration_pct = 6.2%**

- < 50% → distributed failure; DLC SuperAnimal-Quadruped is next (half-day-tier).

## IoU vs Phase 5 manual eye boxes (middle keyframe, n=34)

- Median IoU: **0.1651** (locked gate ≥ 0.6 → FAIL)
- Mean IoU: 0.2294
- Clips with IoU ≤ 0.30 (off-eye): 23/34
- Clips with IoU ≥ 0.50 (on-eye): 7/34

## Orientation-extreme clips — per-clip breakdown

| Clip | Category vs P5 | P5 score | P6b score | IoU vs manual | P6b correct? |
|---|---|---:|---:|---:|---|
| `action_S9.mp4_4_.mp4` | FBB_NEWLY_RECOVERED | -0.117 | +0.511 | 0.000 | ✓ |
| `action_S9.mp4_7_.mp4` | BOTH_RIGHT | +0.165 | +1.235 | 0.000 | ✓ |
| `background_S1.mp4_12_.mp4` | BOTH_RIGHT | +1.315 | +2.382 | 0.000 | ✓ |
| `background_S5.mp4_10_.mp4` | FBB_NEWLY_LOST | +0.699 | -0.406 | 0.000 | ✗ |

## action_S9.mp4_4_ — locked interpretation routing

- IoU vs Phase 5 manual: 0.000
- Phase 6 (b) correct: ✓
- Phase 5 score: -0.117
- Phase 6 (b) score: +0.511

**Verdict (per locked routing in pre-reg)**: CORRECT_OFF_EYE → face-bbox got it right but wider scope captures whole-body cues even off-eye; suggestive but not conclusive.

## All clips (sorted by score_delta_vs_phase5)

| Clip | Source | Truth | P5 score | P6b score | Δ | Cat | IoU | OE? |
|---|---|---|---:|---:|---:|---|---:|---|
| `background_S10.mp4_3_.mp4` | S10 | ACTION | +0.297 | -1.975 | -2.272 | FBB_NEWLY_LOST | 0.408 |  |
| `action_S10.mp4_0_.mp4` | S10 | ACTION | +0.042 | -1.703 | -1.744 | FBB_NEWLY_LOST | 0.437 |  |
| `background_S2.mp4_10_.mp4` | S2 | ACTION | +0.952 | -0.710 | -1.662 | FBB_NEWLY_LOST | 0.042 |  |
| `action_S4.mp4_0_.mp4` | S4 | ACTION | +0.289 | -1.193 | -1.482 | FBB_NEWLY_LOST | 0.338 |  |
| `action_S6.mp4_2_.mp4` | S6 | ACTION | +0.517 | -0.764 | -1.281 | FBB_NEWLY_LOST | 0.000 |  |
| `action_S11.mp4_0_.mp4` | S11 | BACKGROUND | +1.446 | +0.269 | -1.176 | BOTH_WRONG | 0.618 |  |
| `background_S5.mp4_10_.mp4` | S5 | ACTION | +0.699 | -0.406 | -1.105 | FBB_NEWLY_LOST | 0.000 | OE |
| `action_S3.mp4_2_.mp4` | S3 | ACTION | +1.100 | +0.281 | -0.820 | BOTH_RIGHT | 0.000 |  |
| `action_S4.mp4_15_.mp4` | S4 | ACTION | +0.060 | -0.652 | -0.712 | FBB_NEWLY_LOST | 0.000 |  |
| `background_S11.mp4_0_.mp4` | S11 | ACTION | +1.334 | +0.845 | -0.488 | BOTH_RIGHT | 0.624 |  |
| `background_S10.mp4_11_.mp4` | S10 | ACTION | +0.318 | -0.148 | -0.466 | FBB_NEWLY_LOST | 0.476 |  |
| `action_S3.mp4_8_.mp4` | S3 | ACTION | +2.131 | +1.984 | -0.147 | BOTH_RIGHT | 0.543 |  |
| `action_S11.mp4_6_.mp4` | S11 | BACKGROUND | +0.421 | +0.309 | -0.113 | BOTH_WRONG | 0.507 |  |
| `background_S12.mp4_9_.mp4` | S12 | ACTION | +0.499 | +0.447 | -0.052 | BOTH_RIGHT | 0.070 |  |
| `background_S12.mp4_7_.mp4` | S12 | ACTION | +1.066 | +1.015 | -0.052 | BOTH_RIGHT | 0.000 |  |
| `background_S8.mp4_7_.mp4` | S8 | ACTION | +1.005 | +1.028 | +0.023 | BOTH_RIGHT | 0.254 |  |
| `background_S2.mp4_11_.mp4` | S2 | ACTION | +0.346 | +0.483 | +0.137 | BOTH_RIGHT | 0.182 |  |
| `background_S12.mp4_2_.mp4` | S12 | BACKGROUND | +0.516 | +0.679 | +0.163 | BOTH_WRONG | 0.000 |  |
| `background_S6.mp4_3_.mp4` | S6 | BACKGROUND | -0.580 | -0.359 | +0.221 | BOTH_RIGHT | 0.159 |  |
| `background_S1.mp4_7_.mp4` | S1 | BACKGROUND | -0.116 | +0.154 | +0.270 | FBB_NEWLY_LOST | 0.207 |  |
| `background_S6.mp4_2_.mp4` | S6 | BACKGROUND | -0.474 | +0.002 | +0.476 | FBB_NEWLY_LOST | 0.206 |  |
| `action_S5.mp4_5_.mp4` | S5 | ACTION | -0.413 | +0.089 | +0.503 | FBB_NEWLY_RECOVERED | 0.145 |  |
| `background_S9.mp4_8_.mp4` | S9 | BACKGROUND | -0.266 | +0.263 | +0.529 | FBB_NEWLY_LOST | 0.171 |  |
| `background_S3.mp4_3_.mp4` | S3 | ACTION | -0.325 | +0.243 | +0.568 | FBB_NEWLY_RECOVERED | 0.037 |  |
| `action_S9.mp4_4_.mp4` | S9 | ACTION | -0.117 | +0.511 | +0.628 | FBB_NEWLY_RECOVERED | 0.000 | OE |
| `action_S2.mp4_7_.mp4` | S2 | BACKGROUND | -0.977 | -0.104 | +0.873 | BOTH_RIGHT | 0.122 |  |
| `background_S7.mp4_12_.mp4` | S7 | BACKGROUND | -0.490 | +0.460 | +0.951 | FBB_NEWLY_LOST | 0.705 |  |
| `background_S1.mp4_12_.mp4` | S1 | ACTION | +1.315 | +2.382 | +1.067 | BOTH_RIGHT | 0.000 | OE |
| `action_S9.mp4_7_.mp4` | S9 | ACTION | +0.165 | +1.235 | +1.070 | BOTH_RIGHT | 0.000 | OE |
| `background_S8.mp4_3_.mp4` | S8 | ACTION | -0.940 | +0.164 | +1.105 | FBB_NEWLY_RECOVERED | 0.271 |  |
| `background_S7.mp4_9_.mp4` | S7 | BACKGROUND | -1.266 | +0.049 | +1.316 | FBB_NEWLY_LOST | 0.579 |  |
| `background_S7.mp4_17_.mp4` | S7 | BACKGROUND | -0.740 | +0.711 | +1.451 | FBB_NEWLY_LOST | 0.544 |  |
| `action_S5.mp4_2_.mp4` | S5 | BACKGROUND | -1.057 | +0.445 | +1.502 | FBB_NEWLY_LOST | 0.000 |  |
| `action_S8.mp4_12_.mp4` | S8 | BACKGROUND | -0.905 | +0.986 | +1.891 | FBB_NEWLY_LOST | 0.155 |  |
