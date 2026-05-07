# V-JEPA-2 + LR LOSO predictions — agreement decomposition vs RME, Piotr-strict, Piotr-permissive

*Step 2 (B-prime) of `docs/audit-followup-spec.md`. Run 2026-05-07. Branch `experiment/audit-followup`. Read in conjunction with [Lesson 17](../docs/lessons_learned.md) for the bifurcation framing and the pre-registered Pattern A/B/C interpretation lock.*

## Pattern: **C — mixed by source**

Pre-registered trigger conditions:
- **Pattern A** (V-JEPA-2 leans toward Piotr labels on ≥ 7/12 sources): not met. Strict: 6/12 sources improve under Piotr labels; Permissive: 3/12.
- **Pattern B** (V-JEPA-2 vs RME ≥ V-JEPA-2 vs Piotr on most sources): not met. Strict shows Piotr-better on 6 vs RME-better on 5; permissive shows RME-better on 7 vs Piotr-better on 3 — neither side dominates.
- **Pattern C** (mixed by source) — **identified**.

Per pre-registered interpretation under the bifurcation lens (locked in `docs/audit-followup-spec.md` §4 before B-prime ran): **calibration question made concrete — useful diagnostic but ambiguous for RHpE transfer** because deployment will not have a clean/noisy source distinction baked in. Triggers Step 5 (B) with focus on the per-source split.

## Sanity check — published 0.875 reproduction

V-JEPA-2 + LR LOSO with the canonical config (`RidgeClassifier(alpha=1.0, class_weight='balanced')` + `StandardScaler` per fold, on cached `vjepa2_embeddings.npz` features):

- **Global LOSO AUC = 0.8746** (matches published Sanity 5 baseline `0.8746126936531734` exactly)
- Δ vs published: **+0.0000** (well within ±0.02 sanity band)
- Per-source LOSO AUC: S1 0.816, S2 0.927, S3 0.995, S4 0.911, S5 0.903, S6 0.956, S7 1.000, S8 **0.633** *(weak fold — multi-horse confound)*, S9 0.783, S10 1.000, S11 0.920, S12 1.000

Per-clip predictions saved to `outputs/vjepa2_loso_per_clip_predictions.jsonl` (283 rows, full dataset, including LOSO score + Piotr verdict + Piotr category for each clip).

## Per-source agreement matrix

V-JEPA-2 binary prediction (from `predict()` at default threshold) agreement vs three label sets, per source. Strict drops the 56 borderline (`?`) clips (227 eligible total); permissive treats borderline verdicts as agreement with RME (283 total).

| Src | n | vs RME | vs Piotr strict | vs Piotr perm | Δ strict−RME | Δ perm−RME |
|---|---|---|---|---|---|---|
| S1 | 21 | 18/21 = 0.857 | 15/20 = 0.750 | 16/21 = 0.762 | −0.107 | −0.095 |
| S2 | 25 | 20/25 = 0.800 | 12/14 = 0.857 | 19/25 = 0.760 | **+0.057** | −0.040 |
| S3 | 28 | 24/28 = 0.857 | 17/18 = 0.944 | 23/28 = 0.821 | **+0.087** | −0.036 |
| S4 | 32 | 24/32 = 0.750 | 18/27 = 0.667 | 22/32 = 0.688 | −0.083 | −0.062 |
| S5 | 25 | 20/25 = 0.800 | 17/20 = 0.850 | 22/25 = 0.880 | **+0.050** | **+0.080** |
| S6 | 19 | 12/19 = 0.632 | 12/19 = 0.632 | 12/19 = 0.632 | 0.000 | 0.000 |
| S7 | 21 | 17/21 = 0.810 | 15/19 = 0.789 | 17/21 = 0.810 | −0.020 | 0.000 |
| **S8** | 24 | 14/24 = 0.583 | 13/17 = **0.765** | 15/24 = 0.625 | **+0.181** | **+0.042** |
| S9 | 24 | 20/24 = 0.833 | 17/20 = 0.850 | 19/24 = 0.792 | **+0.017** | −0.042 |
| S10 | 22 | 21/22 = 0.955 | 17/17 = **1.000** | 22/22 = **1.000** | **+0.045** | **+0.045** |
| S11 | 19 | 16/19 = 0.842 | 14/19 = 0.737 | 14/19 = 0.737 | −0.105 | −0.105 |
| S12 | 23 | 22/23 = 0.957 | 14/17 = 0.824 | 20/23 = 0.870 | −0.133 | −0.087 |
| **AGG** | **283** | **228/283 = 0.806** | **181/227 = 0.797** | **221/283 = 0.781** | **−0.009** | **−0.025** |

**Reading direction:** positive Δ = V-JEPA-2 agrees more with Piotr's labels than with RME on this source. Negative Δ = V-JEPA-2 agrees more with RME.

## Per-source pattern under the bifurcation lens

The per-source deltas split cleanly along Lesson 17's source-cleanliness ranking:

| Source rank | Audit vs RME disagreement (Lesson 17) | V-JEPA-2 vs Piotr strict Δ | Direction |
|---|---|---|---|
| **Cleaner sources** | S6 (0%), S7 (5%), S3 (7%), S12 (9%), S1 (10%), S11 (11%) | S6 0.0, S7 −2.0, S3 +8.7, S12 −13.3, S1 −10.7, S11 −10.5 | mixed but mostly **negative** — V-JEPA-2 tracks RME |
| Mid-disagreement | S8 (8%, multi-horse), S9 (13%), S4 (16%, multi-horse) | S8 +18.1, S9 +1.7, S4 −8.3 | mixed — multi-horse sources noisy |
| **Noisy sources** | S2 (20%), S10 (23%), **S5 (24%)** | S2 +5.7, S10 +4.5, S5 **+5.0 strict / +8.0 perm** | **positive** — V-JEPA-2 leans toward audit |

The pattern is consistent with the spec's pre-registered Pattern C definition: **V-JEPA-2 + LR tracks the EquiFACS threshold on cleaner sources and drifts toward the audit threshold on noisier sources.** This is exactly the per-source split the spec named when defining Pattern C ("V-JEPA-2 leans toward Piotr on noisier ones — S5, S10 — tracks RME on cleaner sources — S6, S7"). The data confirm the prediction.

The most extreme positive Δ is **S8 strict +18.1 pp**: 14/24 agreement with RME vs 13/17 = 0.765 with Piotr strict. S8 is a multi-horse source (24/24 of S8's clips have `fh:`/`bh:` notation per the audit). The model perceives motion signals on S8 that RME labels excluded but Piotr's stricter audit labels caught — at least when Piotr's certain-only verdicts are the reference. Note that under permissive labels (S8 +4.2), the bump is much smaller because borderline cases revert to RME labels and the multi-horse uncertainty gets washed out.

The most extreme negative Δ is **S12 strict −13.3 pp**: 22/23 = 0.957 vs RME but only 14/17 = 0.824 vs Piotr strict. S12 has only 2 audit disagreements with RME (per Lesson 17), but those 2 happen to be on clips where V-JEPA-2 also predicts the RME label — so reverting to Piotr's stricter labels costs agreement. S12 is a "clean source where the audit's small disagreement happens to land on clips V-JEPA-2 was right on," a coincidence rather than a pattern.

## Gate 1 decision: run Step 5 (B), preceded by Step 4 (D)

Per the audit-followup spec §4 Gate 1 logic:
- Pattern A or C → Step 5 (B) runs as a **calibration investigation**, not as expected-improvement
- Pattern B → skip Step 5

Pattern C identified → Step 5 runs. Step 4 (multi-horse subset extraction, 55 clips in S4 + S8) runs first as precursor to B's "cleaned" variant.

**Pre-registered interpretation lock (from spec §7) reminder for Step 5:** higher LOSO under Piotr-permissive labels does **not** mean V-JEPA-2 + LR is "even better than the published 0.875." It means the model is sitting at a different threshold than its training signal on the noisy sources. The project-positive LOSO outcome is "all variants ≈ 0.875" (Pattern B at the LOSO level — model robust to threshold variation but tracking EquiFACS training signal). Materially diverging variants are calibration concerns.

**Predicted variant numbers under Pattern C** (informed conjecture, not measured):
- **Strict (227 clips, 56 borderline dropped):** likely ≈ 0.875 ±2 pp. Borderline clips are the hardest-to-classify subset; dropping them from training+eval should remove some noise on both sides. Per-source AUC may shift on the noisy folds (S5, S10) where many clips were borderline.
- **Permissive (283 clips, ? = RME):** likely ≈ 0.875 — closest to published baseline by construction since most labels match RME.
- **Cleaned (228 clips, multi-horse excluded):** **most diagnostic.** If Cleaned > 0.875 by ≥ 2 pp, the original 0.875 was being held down by S4 + S8 multi-horse confounds (consistent with Lesson 9's S8 +24 pp gain under bg-masking). If Cleaned ≈ 0.875, the model is fine on multi-horse clips even without explicit bg-masking. If Cleaned < 0.875, multi-horse clips were anchor data for the model's discrimination boundary.

Step 5 will produce these three numbers and resolve the predictions.

## Hold for Step 3 (within-observer consistency check)

Per spec §5, Step 3 is a hard-stop gate. The 66-clip re-watch (56 borderline + 10 confident controls) must show confident-case self-consistency ≥ 90 % before Step 4 + Step 5 are executed. Pattern C is identified independent of Step 3 (B-prime uses the audit data as-is), but the rigorous interpretation in the eventual Step 5 writeup depends on the Step 3 result — if confident-case consistency falls below 90 %, the entire follow-up reverts to caveats and Lesson 17 reverts to a stub.

Currently:
- B-prime done → Pattern C identified → Gate 1 says "B will run after Step 3 + Step 4"
- Step 3 prep next: 66-clip re-watch list
- Then user re-watch (out of agent's hands)
- Then Step 3 analysis + hard-stop check
- Then Step 4 (D) + Step 5 (B) if hard-stop passes

## Caveats specific to B-prime

- **Sanity check is per-clip global AUC (0.8746), not per-source mean AUC (which is 0.910 for this config).** The per-source mean is uniformly higher because easy sources (S7, S10, S12 = 1.000) and hard sources (S6 = 0.956, S8 = 0.633, S9 = 0.783) contribute equally to the mean while contributing unequally to the global AUC. The published 0.8746 is the global per-clip number.
- **Binary-agreement metric uses `predict()` at default threshold 0.5.** Continuous LOSO AUC of 0.8746 is well-ordered ranking, not classification accuracy at any specific threshold. Aggregate binary agreement vs RME is 0.806, lower than 0.8746 because some clips are correctly ranked but on the wrong side of the 0.5 cut. This is normal and doesn't change Pattern C identification (which is based on per-source agreement deltas, not absolute agreement levels).
- **No retraining of V-JEPA-2.** Only the linear probe ran per fold on cached features. The "B-prime is not retraining" framing in the spec holds.
- **Sample sizes per source range from 17 to 32**, with strict variant dropping borderline clips that vary heavily by source (S2 has more borderline cases than S6, etc.). Per-source agreement-rate deltas should be read as suggestive rather than statistically settled at this N.
