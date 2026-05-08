# Track B Phase 3 — AUC method + result

**Frozen:** 2026-05-08, accompanies `outputs/eye_loso_results.json` from
`tools/eye_loso_lr.py` run on user-provided eye labels
(`outputs/eye_verification_clips.txt`).

**Pre-registrations honored** (`outputs/track_b_phase1_preregistration.md`,
hash recorded in `docs/preregistration_hashes.md`):

- AUC ≥ 0.65 → adopt v1 crop, write up Track A
- 0.55–0.65 → v2 profile-aware crop, ONCE
- < 0.55 → eye-track-failed conclusion

## Method (locked before run)

- **Primary metric:** pooled-prediction AUC over all 34 aligned (truth, score)
  pairs concatenated across LOSO folds. Computed once on the pool, not as
  the mean of per-fold AUCs.
- **Per-fold AUC distribution** (min / median / max / n_defined / n_skipped)
  reported as the secondary diagnostic. n_skipped counts folds where the
  held-out source had a single-class test set (AUC undefined).
- **Confidence interval:** DeLong (1988) analytical 95 % CI on the pooled
  AUC. Wald-style `AUC ± 1.96 · √Var(AUC)`, clipped to [0, 1]. Reported
  before the p-value because effect size + precision is the honest framing
  for n = 34.
- **Permutation test:** n = 1000, seed = 42. Each permutation shuffles
  labels GLOBALLY once (group structure not preserved under the null —
  matches the protocol used for the 0.8746 ear baseline in
  `outputs/iter65_sanity5_loso_rme_results.json`), then runs the FULL
  LOSO loop and recomputes pooled AUC. The known property is that
  strong source-level structure can slightly inflate type-I error rate;
  pre-registered thresholds are far enough from the null mean that this
  isn't a deciding factor.
- **P-value form:** `(sum(null ≥ observed) + 1) / (n + 1)` so the floor is
  ~0.001 at n = 1000 (no p = 0).
- **Pipeline per fold:**
  - `StandardScaler()` instantiated fresh, `fit_transform` on the training
    fold only, `transform` on the test fold. NO precomputed scaling on
    the full dataset (would leak test statistics into training).
  - `RidgeClassifier(alpha=1.0, class_weight="balanced")` instantiated
    fresh per fold (no state carryover even though Ridge overwrites on
    `fit`; audit-friendly).
  - `decision_function` for the score (Ridge has no `predict_proba`).
- **Score comparability across fold-trained models:** decision-function
  scores from different fold-trained Ridge classifiers are not strictly
  bit-exact-comparable; standard CV practice treats them as comparable
  on a fixed-α-fixed-class-weight setup, which is what we do here. This
  is the same assumption built into the 0.8746 ear baseline. Documented
  here for reviewer transparency.
- **Defensive asserts at function entry:** `len(X) == len(y) == len(groups)`,
  binary labels, no NaN in features.

## Alignment

- Embeddings npz (`outputs/vjepa2_embeddings_eye.npz`, 34 rows after
  Phase 2 `--exclude` fix) is canonical.
- Labels file has 36 entries; **2 dropped** as expected — labels exist
  but no embedding:
  - `background_S1.mp4_11_.mp4` — label=ACTION, reason=manual_excluded_eye_under_20pct
  - `background_S4.mp4_7_.mp4` — label=BACKGROUND, reason=YOLO_no_detection_static
- Aligned: **34 rows, 12 sources, class balance 13 bg / 21 action**.
- Verification: every embedding has a label (no inverse mismatch).

## Result

- **Pooled AUC = 0.6813** (95 % DeLong CI **[0.4866, 0.8760]**)
- **Permutation p-value = 0.0579** (n = 1000, +1-form, null mean ≈ 0.487, std ≈ 0.121)
- **Per-fold AUC distribution:** min 0.000, median 1.000, max 1.000, on 8 defined folds, 4 skipped (single-class test sets: S3 all-action, S4 all-action, S7 all-background, S10 all-action).

Per-fold detail:

| Source | n_test | n_pos | n_neg | defined | fold AUC |
|---|---|---|---|---|---|
| S1  | 2 | 1 | 1 | ✓ | 1.000 |
| S2  | 3 | 2 | 1 | ✓ | 1.000 |
| S3  | 3 | 3 | 0 | — (single-class) | undef |
| S4  | 2 | 2 | 0 | — (single-class) | undef |
| S5  | 3 | 2 | 1 | ✓ | **0.000** |
| S6  | 3 | 1 | 2 | ✓ | **0.000** |
| S7  | 3 | 0 | 3 | — (single-class) | undef |
| S8  | 3 | 2 | 1 | ✓ | 0.500 |
| S9  | 3 | 2 | 1 | ✓ | 1.000 |
| S10 | 3 | 3 | 0 | — (single-class) | undef |
| S11 | 3 | 1 | 2 | ✓ | 1.000 |
| S12 | 3 | 2 | 1 | ✓ | 1.000 |

## Decision per pre-registration

Pooled AUC 0.6813 ≥ 0.65 → **decision branch `>=0.65`** ("adopt v1 crop, write up Track A").

The mechanical decision is `>=0.65`. **The honest framing is more nuanced:**

1. **The 95 % CI [0.49, 0.88] overlaps chance.** The lower bound is below 0.5. With n = 34, the pooled AUC is too imprecise to confidently distinguish the model from chance even though the point estimate clears the threshold.
2. **The p-value 0.058 is above conventional α = 0.05.** Under the global-shuffle null, results this extreme or better occur ~5.8 % of the time by chance. Borderline.
3. **The per-fold AUC distribution is bimodal.** 5 of 8 defined folds are perfect (1.000); 2 are perfectly inverted (0.000 — S5 and S6); 1 is at chance (S8 = 0.500). The aggregate 0.68 averages over this; the underlying behavior is "the model classifies some sources perfectly and others perfectly wrong." This is not a stable signal across the population — it's a small-N average with extremely high per-source variance.
4. **4 of 12 sources are class-degenerate** at this label/N regime (S3, S4, S7, S10 each had all-action or all-background test clips). The pooled AUC includes their predictions, but per-fold AUC cannot be computed for them — they contribute to the spread silently.

The pre-registration acted on the point estimate by design; it did not condition on CI width or per-fold spread. The decision branch is the locked outcome, and that is what is reported. The interpretation above is the honest writeup framing.

## What this result actually supports

- **A meaningful effect-size point estimate** (0.68) above the pre-registered threshold (0.65), with simultaneous statistical features that argue against confident generalization at n = 34.
- **The two source-inversions (S5, S6)** are the most informative subjects for a follow-up: either the eye-region crops on those sources contain visual confounds the model latched onto inversely, or those sources have label noise relative to the V-JEPA-2 representation. Worth visual inspection of the S5/S6 crops as a follow-up if Track A writeup proceeds.
- **The four class-degenerate sources** (S3 all-action, S4 all-action, S7 all-background, S10 all-action) are a structural feature of the 36-clip RHpE stratified subset under user's eye labels — not a bug. They highlight that this label/N regime is at the floor of where LOSO can reliably operate.

## What this result does not support

- A confident "V-JEPA-2 features generalize to eye behavior at the same fidelity as ear (0.875)" claim. The CI is 39 pp wide; that's effectively "between chance and excellent."
- A clinical-pre-screening or production-classifier claim. n = 34 with bimodal per-fold AUC is at best a methodology pilot.
- A negative claim either ("V-JEPA-2 fails on eye"). The point estimate clears the pre-registered threshold; the data does not falsify the eye-features hypothesis.

The honest reading: this is a **proof-of-concept-scale pilot whose pre-registered threshold cleared on the point estimate, but whose precision is too low to publish a definitive number**. Multi-rater eye labels at higher N is the obvious next step (commission second observer, expand to the next 36 clips, re-run identical pipeline). The methodology is now load-bearing.
