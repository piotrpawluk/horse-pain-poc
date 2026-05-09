# Phase 6 (a) — per-clip Phase 3 vs Phase 5 prediction diff

Pre-registered diagnostic from `docs/phase5_audit.md`. Threshold=0 on RidgeClassifier decision_function, both pipelines share the same 34 clips and the same in-LOSO label vector.

## Label-source clarification (locked)

Three reference label sets exist in the repo. The same 34 clips are labeled differently across them:

- **Set A — RME filename taxonomy**: `action_*` / `background_*` from filenames. The dataset publisher's labels for the original ear-motion behavior. Not what Phase 5 primary trained on.
- **Set B — Piotr morning verification** (`outputs/eye_verification_clips.txt`): the labels Phase 3 + Phase 5 primary were both trained AND evaluated against. Phase 5 primary's reported AUC of 0.7985 is the AUC against this set. **This is the directly-decomposable label set and the primary diagnostic uses it.**
- **Set C — Piotr tightened-rubric relabel** (`outputs/eye_relabel_unmasked.txt`): Phase 5 sens_rubric's training labels. Differs from Set B on several clips (e.g., `bg_S1_12` is ACTION in Set B, BACKGROUND in Set C).

**Primary diagnostic** uses Set B (matches the AUCs it decomposes); **supplementary** Sets A and C are computed for sensitivity. Categories under each set are *not* the same — the prediction-shift structure depends on the reference label.

## Primary (Set B — verification labels)

- Phase 3 pooled AUC: **0.6813** (per-clip correct@thr=0: 23/34)
- Phase 5 primary pooled AUC: **0.7985** (per-clip correct@thr=0: 27/34)
- Δ AUC: +0.1172

| Category | Count | Share |
|---|---:|---:|
| BOTH_RIGHT | 18 | 52.9% |
| BOTH_WRONG | 2 | 5.9% |
| V3_NEWLY_RECOVERED | 9 | 26.5% |
| V3_NEWLY_LOST | 5 | 14.7% |

**Net shift (recovered − lost): +4 clips** (ratio recovered/lost = 1.8)

## Mechanical interpretation rule (user-locked decision)

- If `recovered ≫ lost` → cropping is a **uniform lever**. (b) gate: geometric face-bbox crop captures the same rescue mechanism.
- If `recovered ≈ lost` → cropping **shifts** which clips classify well; doesn't uniformly improve. (b) gate: preserve recoveries without losing new losses (harder target, different criteria).

## Supplementary — sensitivity to label choice

| Variant | n | P3 right@thr=0 | P5 right@thr=0 | Both right | Both wrong | Recovered | Lost | Net | Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| set_b_verification | 34 | 23 | 27 | 18 | 2 | 9 | 5 | +4 | 1.8 |
| set_a_rme_filename | 34 | 18 | 18 | 11 | 9 | 7 | 7 | +0 | 1.0 |
| set_c_tightened | 34 | 18 | 22 | 13 | 7 | 9 | 5 | +4 | 1.8 |

Reading: if the recovered/lost ratio holds direction across all three reference sets, the prediction-shift finding is robust to label-reference choice. If it inverts under any set, the finding is label-set-specific.

## V3_NEWLY_RECOVERED (n=9)

| Clip | Source | Truth (Set B) | P3 score | P5 score | Δ score |
|---|---|---|---:|---:|---:|
| `background_S2.mp4_10_.mp4` | S2 | ACTION | -0.667 | +0.952 | +1.619 |
| `action_S3.mp4_2_.mp4` | S3 | ACTION | -0.287 | +1.100 | +1.387 |
| `background_S10.mp4_3_.mp4` | S10 | ACTION | -0.536 | +0.297 | +0.833 |
| `action_S4.mp4_15_.mp4` | S4 | ACTION | -0.079 | +0.060 | +0.138 |
| `background_S9.mp4_8_.mp4` | S9 | BACKGROUND | +0.252 | -0.266 | -0.518 |
| `background_S6.mp4_2_.mp4` | S6 | BACKGROUND | +0.515 | -0.474 | -0.989 |
| `background_S6.mp4_3_.mp4` | S6 | BACKGROUND | +0.637 | -0.580 | -1.216 |
| `action_S8.mp4_12_.mp4` | S8 | BACKGROUND | +0.362 | -0.905 | -1.268 |
| `action_S5.mp4_2_.mp4` | S5 | BACKGROUND | +0.592 | -1.057 | -1.649 |

## V3_NEWLY_LOST (n=5)

| Clip | Source | Truth (Set B) | P3 score | P5 score | Δ score |
|---|---|---|---:|---:|---:|
| `background_S12.mp4_2_.mp4` | S12 | BACKGROUND | -0.204 | +0.516 | +0.720 |
| `background_S3.mp4_3_.mp4` | S3 | ACTION | +0.028 | -0.325 | -0.353 |
| `action_S5.mp4_5_.mp4` | S5 | ACTION | +0.524 | -0.413 | -0.937 |
| `action_S9.mp4_4_.mp4` | S9 | ACTION | +0.919 | -0.117 | -1.036 |
| `background_S8.mp4_3_.mp4` | S8 | ACTION | +0.300 | -0.940 | -1.240 |

## BOTH_RIGHT (n=18)

| Clip | Source | Truth (Set B) | P3 score | P5 score | Δ score |
|---|---|---|---:|---:|---:|
| `action_S3.mp4_8_.mp4` | S3 | ACTION | +0.761 | +2.131 | +1.370 |
| `background_S1.mp4_7_.mp4` | S1 | BACKGROUND | -1.413 | -0.116 | +1.297 |
| `background_S1.mp4_12_.mp4` | S1 | ACTION | +0.454 | +1.315 | +0.861 |
| `background_S7.mp4_12_.mp4` | S7 | BACKGROUND | -1.108 | -0.490 | +0.618 |
| `background_S8.mp4_7_.mp4` | S8 | ACTION | +0.447 | +1.005 | +0.558 |
| `background_S5.mp4_10_.mp4` | S5 | ACTION | +0.192 | +0.699 | +0.507 |
| `background_S7.mp4_17_.mp4` | S7 | BACKGROUND | -1.227 | -0.740 | +0.487 |
| `action_S6.mp4_2_.mp4` | S6 | ACTION | +0.170 | +0.517 | +0.348 |
| `action_S2.mp4_7_.mp4` | S2 | BACKGROUND | -1.284 | -0.977 | +0.307 |
| `background_S11.mp4_0_.mp4` | S11 | ACTION | +1.148 | +1.334 | +0.186 |
| `background_S10.mp4_11_.mp4` | S10 | ACTION | +0.418 | +0.318 | -0.100 |
| `action_S10.mp4_0_.mp4` | S10 | ACTION | +0.165 | +0.042 | -0.123 |
| `background_S7.mp4_9_.mp4` | S7 | BACKGROUND | -0.924 | -1.266 | -0.343 |
| `background_S12.mp4_9_.mp4` | S12 | ACTION | +0.996 | +0.499 | -0.497 |
| `action_S4.mp4_0_.mp4` | S4 | ACTION | +1.015 | +0.289 | -0.725 |
| `background_S12.mp4_7_.mp4` | S12 | ACTION | +2.020 | +1.066 | -0.954 |
| `background_S2.mp4_11_.mp4` | S2 | ACTION | +1.615 | +0.346 | -1.270 |
| `action_S9.mp4_7_.mp4` | S9 | ACTION | +1.891 | +0.165 | -1.726 |

## BOTH_WRONG (n=2)

| Clip | Source | Truth (Set B) | P3 score | P5 score | Δ score |
|---|---|---|---:|---:|---:|
| `action_S11.mp4_0_.mp4` | S11 | BACKGROUND | +1.007 | +1.446 | +0.439 |
| `action_S11.mp4_6_.mp4` | S11 | BACKGROUND | +0.541 | +0.421 | -0.120 |
