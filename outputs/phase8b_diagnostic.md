# Phase 8b diagnostic — DLC ear-keypoint cropping vs whole-frame

- Phase 8b pooled AUC: **0.9008**
- Whole-frame baseline AUC (full RME 283 clips): **0.8746**
- Whole-frame baseline AUC (aligned to Phase 8b clips, 283 clips): 0.8746
- Δ observed (aligned): **+0.0262**

- Phase 8b bootstrap CI: [0.8608, 0.9406]
- Subject-bootstrap CI on Δ (B=10000): [-0.0285, +0.0732], median +0.0262
- Paired DeLong vs whole-frame: Δ=+0.0262, z=1.011, p=0.3122

## Locked gates (Phase 8b §8)

- **G1** (AUC ≥ 0.80, strong): PASS (0.9008)
- **G2** (0.65 ≤ AUC < 0.80, modest): FAIL
- **G3** (AUC < 0.65, fails): FAIL
- **G4 supportive** (paired DeLong p < 0.05): FAIL (p=0.3122)

## Joint verdict (per locked verdict-reporting protocol)

**COMPETITIVE** — Strong absolute AUC; paired test inconclusive. DLC competitive with whole-frame but indistinguishable.

## Per-source AUC

| Source | Phase 8b AUC | Whole-frame AUC | Δ | Sign |
|---|---:|---:|---:|---|
| S1 | 0.7959 | 0.8163 | -0.0204 | − |
| S2 | 1.0000 | 0.9267 | +0.0733 | + |
| S3 | 0.9688 | 0.9948 | -0.0260 | − |
| S4 | 0.8958 | 0.9115 | -0.0156 | − |
| S5 | 0.9740 | 0.9026 | +0.0714 | + |
| S6 | 0.8222 | 0.9556 | -0.1333 | − |
| S7 | 0.7500 | 1.0000 | -0.2500 | − |
| S8 | 0.9844 | 0.6328 | +0.3516 | + |
| S9 | 0.8322 | 0.7832 | +0.0490 | + |
| S10 | 1.0000 | 1.0000 | +0.0000 |  |
| S11 | 0.7955 | 0.9205 | -0.1250 | − |
| S12 | 0.8929 | 1.0000 | -0.1071 | − |

Sources with Δ > 0: 4/12

## Crop pipeline statistics (Decision 1 fallback handling)

- Per-frame-only clips (no fallback): 257/283
- Clips with single-middle-frame fallback applied to some frames: 26/283
- Failed clips (status=fail): 0

- Fallback frame % distribution: median=0.0%, max=100.0%, n_clips_with_any_fallback=26
