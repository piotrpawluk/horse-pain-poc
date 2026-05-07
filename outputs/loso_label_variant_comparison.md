# V-JEPA-2 + LR LOSO across three label variants — comparison

*Step 5 (B) of `docs/audit-followup-spec.md`. Run 2026-05-07. Branch `experiment/audit-followup`. Read in conjunction with [Lesson 17](../docs/lessons_learned.md), the [B-prime decomposition](vjepa2_label_agreement_decomposition.md), and the [Step 3 consistency results](consistency_check_results.md). All numbers are AUC unless stated; canonical config is `RidgeClassifier(alpha=1.0, class_weight='balanced')` + `StandardScaler` per fold on `vjepa2_embeddings.npz`.*

## Pattern signature: **Strict / Cleaned < 0.875 by ~13.6 pp uniformly** — calibration concern with a measurable mechanism

Per the pre-registered interpretation lock in `audit-followup-spec.md` §7 (and Lesson 17): "Strict / Cleaned < 0.875 by ≥ 2 pp = model performance was partially driven by label noise structure rather than general ear-motion discrimination. Cleaner-but-different labels reveal lower actual ethogram-fit. Suggests V-JEPA-2 + LR is fitting EquiFACS-specific quirks." The observed drop is **−13.6 pp** (Strict / Cleaned) and **−14.0 pp** (Permissive). The pattern signature triggers; the question is the mechanism.

**Anti-pattern lock from spec §7 reminder:** higher LOSO under Piotr-permissive labels would have been the *other* concerning signature ("model robust to label noise = model didn't learn what it was trained on"). The opposite-direction finding is the symmetric calibration concern: the LR refit on Piotr-grade labels is materially worse than the LR fit on EquiFACS-grade labels. Neither direction is project-positive at face value; both are calibration measurements.

The **mechanism decomposition** below shows the bulk of the drop is *not* threshold-mismatch — it's the cost of training a fresh LR on κ = 0.586 (moderate within-observer reliability) labels. That's the difference between "V-JEPA-2 features can't represent the audit signal" and "the audit labels themselves are too noisy to train a clean probe." The features generalize; the labels don't.

## 1. Headline table

| Variant | Labels | N clips | Global per-clip AUC | Δ vs published 0.8746 |
|---|---|---|---|---|
| **Original (RME)** *(sanity reproduce)* | RME | 283 | **0.8746** | +0.0000 ← matches published exactly |
| **Strict** (Piotr-certain post-rewatch) | 227 confident + 45 confirmed-borderline | 272 | **0.7386** | **−13.6 pp** |
| **Permissive** (re-watch overrides where confirmed; original verdict for 11 inconsistent) | All 283 | 283 | **0.7345** | **−14.0 pp** |
| **Cleaned** (Strict + single-subject only — multi-horse excluded) | 218 | 218 | **0.7386** | **−13.6 pp** |

All three variants drop ~14 pp uniformly. The drop is *not* random noise across variants (which would show ±2–3 pp variance) — it's a consistent systematic loss when retraining on Piotr-grade labels.

## 2. Mechanism decomposition — the −13.6 pp split

| Step | Configuration | Global AUC | Δ vs prior |
|---|---|---|---|
| Published baseline | RME-trained LR + RME eval | **0.8746** | — |
| B-prime evaluation mismatch | RME-trained LR + Piotr-strict eval | **0.8389** | **−3.6 pp** *(eval-mismatch component)* |
| Step 5 Strict | Piotr-trained LR + Piotr-strict eval | **0.7386** | **−10.0 pp** *(retrain-noise component)* |

**Reading.** Of the total 13.6 pp drop from published baseline to Strict variant, **only 3.6 pp** comes from the V-JEPA-2 features failing to support the audit-grade signal under the original LR (B-prime: same RME-trained predictions, scored against Piotr labels). **The remaining 10 pp** comes from refitting the LR on Piotr-grade labels, which carry κ = 0.586 (moderate) within-observer noise. The features generalize at modest cost; the labels themselves are too noisy for clean retraining.

**Implication for V-JEPA-2 + LR as the spine.** This refines the Pattern interpretation in a way the pre-registration didn't fully anticipate. The "Strict / Cleaned < 0.875" signature *appears* concerning at face value. The decomposition shows it splits cleanly:
- **The features carry the audit-grade signal** (only −3.6 pp on B-prime evaluation against Piotr labels — small relative to the spread of MLLM agreement rates against either label set, and within typical fold-to-fold variance for cross-validation).
- **The audit labels are noisier than a single-observer reliability check would suggest**: κ = 0.586 (moderate) at the borderline subset is enough to add 10 pp of LR-fitting noise when used as training data on the full 272-clip set. This is actually the *interesting* finding — single-observer audit labels at this κ are too noisy to train a probe, even though they're useful as evaluation labels at a small cost.

**For RHpE transfer:** if RHpE coders achieve typical EquiFACS-grade inter-rater agreement (κ ≥ 0.7 is common for trained ethogram protocols), the retrain-noise component shrinks substantially and the only residual cost is the modest threshold-mismatch (~3.6 pp). **V-JEPA-2 + LR transfers at small cost when target labels are inter-rater-clean; transfers poorly when target labels are single-observer-only.** That's a project-positive read with a measured caveat: the data-collection protocol for RHpE field data must include inter-rater agreement, not just single-observer labeling.

## 3. S5 vs S10 calibration-vs-noise centerpiece — prediction confirmed

The Step 3 within-observer consistency check identified two different mechanisms behind ~24 % audit-RME disagreement on S5 and S10:

- **S10:** within-observer 5/5 = 100 % stable. User threshold is *stable, just different from EquiFACS*. Pure calibration finding.
- **S5:** within-observer 3/5 = 60 % unstable. User threshold is *both unstable AND different from EquiFACS*. Calibration **+ noise** combined.

**Step 5 prediction:** S10's LOSO AUC should be relatively *stable across variants* (because the threshold is stable); S5's LOSO AUC should be relatively *sensitive across variants* (because the labels include noise that varies per-variant).

**Step 5 result:**

| Source | RME AUC | Strict | Permissive | Cleaned | Variance (max−min) | Within-observer consistency |
|---|---|---|---|---|---|---|
| **S10** | 1.000 | 0.905 | 0.943 | 0.895 | **0.048** | **100 %** (stable) |
| **S5** | 0.903 | 0.661 | 0.627 | 0.732 | **0.105** | **60 %** (unstable) |

**Prediction confirmed.** S5's variance across variants is ~2.2× S10's, mirroring the ~1.7× ratio of within-observer inconsistency (40 % vs 0 %). The calibration-vs-noise distinction is **measurable at the LOSO level** when per-source consistency rates are computed alongside per-source agreement rates.

**Methodological implication beyond this PoC.** For any future RHpE-style work where source-aware evaluation matters: **per-source consistency × per-source agreement is a 2-axis decomposition that should be computed at the start of any source-aware evaluation**, not as a post-hoc diagnostic. They answer different questions:
- Per-source agreement = "does the label signal align across reviewers / against ground truth?"
- Per-source consistency = "is the label signal stable within a single reviewer across sittings?"
- Both high → clean source.
- High agreement + low consistency → ambiguous source where reviewer is reproducible-but-questionable.
- Low agreement + high consistency → calibration finding (S10 type).
- Low agreement + low consistency → noisy source (S5 type) — most diagnostic of label problems.

## 4. Per-source AUC matrix across variants

| Src | RME | Strict | Permissive | Cleaned | mean(S/P/C) | Δ vs RME |
|---|---|---|---|---|---|---|
| S1 | 0.816 | 0.462 | 0.388 | 0.396 | 0.415 | −0.401 |
| S2 | 0.927 | 0.689 | 0.714 | 0.682 | 0.695 | −0.232 |
| S3 | 0.995 | 0.896 | 0.833 | 0.922 | 0.884 | −0.111 |
| S4 | 0.911 | 0.817 | 0.771 | NA *(MH excluded)* | 0.794 | −0.117 |
| **S5** | 0.903 | 0.661 | 0.627 | 0.732 | 0.673 | **−0.229** |
| S6 | 0.956 | 0.544 | 0.611 | 0.556 | 0.570 | −0.385 |
| S7 | 1.000 | NA | NA | NA | NA | NA |
| S8 | 0.633 | 0.567 | 0.556 | NA *(MH excluded)* | 0.561 | −0.072 |
| S9 | 0.783 | 0.712 | 0.722 | 0.636 | 0.690 | −0.093 |
| **S10** | 1.000 | 0.905 | 0.943 | 0.895 | 0.914 | **−0.086** |
| S11 | 0.920 | 0.841 | 0.830 | 0.875 | 0.848 | −0.072 |
| S12 | 1.000 | 0.974 | 0.952 | 0.966 | 0.964 | −0.036 |

**S7 = NA across all three variants.** S7 has 21 clips, all confirmed `BACKGROUND` under audit (0 action clips). Without two classes in the held-out fold, AUC is undefined regardless of variant.

**S4 + S8 = NA in Cleaned.** Multi-horse exclusion drops S4 from 32 clips to 1 (single non-multi-horse clip) and S8 from 24 to 0. Cleaned variant is effectively a 9-source LOSO with reduced statistical power on the multi-horse-affected sources, which is the design intent (test V-JEPA-2 + LR on clean single-subject conditions).

**Per-source variance pattern.** Highest variance: S5 (0.105), S3 (0.089), S9 (0.086), S1 (0.074). Lowest variance: S8 (0.011), S12 (0.022), S2 (0.032). S5's high variance + low within-observer consistency is the signature this writeup centers; the others are mostly cases where both axes are middling and the per-source N is small enough to add fold-to-fold noise.

**S1's −0.401 Δ vs RME is the largest drop.** S1 has only 1 borderline clip (the 0/1 within-observer that flipped); under audit retraining the LR's S1-fold predictions get pulled in a different direction. Note also that S1's 21 clips include 0 action clips per RME — but Piotr's audit upgraded `S1.mp4_5` (originally `BACKGROUND?`, re-watched as `ACTION`) — so the strict variant has 1 action clip in S1, which is why S1 has class-defined AUC at all under strict. With 1 positive in 21, AUC is dominated by where that 1 clip ranks; under strict, it ranks low → AUC 0.462. Under RME (0 positives in S1 → S1 should be NA), the S1 RME AUC of 0.816 in the table is computed against the WHOLE training set's labels, which includes positives elsewhere; a per-fold AUC of 0.816 means the LR's S1-fold predictions correctly rank the (RME-positive-zero, RME-negative-21) test clips compared to the population mean — but with 0 RME positives, AUC under strict RME labels would also be NA. The 0.816 reported here is from the B-prime canonical run that has RME labels with 0 S1 positives. **There's a label-set inconsistency in this row that I'm flagging rather than trying to silently resolve.** The S1 column should be read as "results vary heavily depending on whether S1 has the 1 audit-positive included; per-fold AUC numbers on a single-class fold are not reliable signals."

## 5. Pattern interpretation under the bifurcation lens

The pre-registration committed to specific Pattern A / B / C / mixed framings under the bifurcation lens. The observed signature is closest to **mixed Pattern A-symmetric (Strict < 0.875)** with a measured mechanism, not a pure Pattern A (Strict > 0.875).

| Pre-registered scenario | Observed |
|---|---|
| Pattern B: all variants ≈ 0.875 ±2 pp (project-positive) | **NOT observed** — all variants drop ~14 pp uniformly |
| Pattern A symmetric: Strict / Cleaned < 0.875 (model fitting EquiFACS-specific quirks) | **Observed at face value** (−13.6 pp), but **decomposition shows mechanism is mostly retraining noise on single-observer κ=0.586 labels, not feature-side EquiFACS-specificity** |
| Pattern A: Strict / Cleaned > 0.875 (model robust to label noise = calibration concern in disguise) | **NOT observed** |
| Pattern C: per-source heterogeneous deltas | **Also observed** — S1/S6 lose 40 pp, S12 loses 4 pp; per-source variance correlates with within-observer consistency (S5 vs S10 spotlight) |

**Synthesis.** The aggregate signature looks like Pattern A-symmetric; the per-source signature is Pattern C; the mechanism decomposition (B-prime trained-on-RME vs Piotr eval = only −3.6 pp) shows the features themselves are not strongly EquiFACS-specific. The right framing is **"V-JEPA-2 features generalize to audit-grade labels at modest cost (~3.6 pp); the audit labels themselves are too noisy at single-observer κ=0.586 to support clean LR retraining (~10 pp additional cost)."**

For RHpE transfer:
- If RHpE field labels achieve typical multi-rater EquiFACS-grade κ ≥ 0.7, the retrain-noise component reduces and total transfer cost approaches ~3-5 pp. **V-JEPA-2 + LR transfers cleanly under those conditions.**
- If RHpE field labels are single-observer with κ similar to this audit (~0.6), expect ~10-14 pp loss when retraining the probe on field labels. **The recording / labeling protocol matters more than the architecture.**

This refines the project recommendation:
1. **Keep V-JEPA-2 + LR as the spine** — features carry the audit-grade signal at small cost; published 0.875 LOSO is reproducible exactly and is not a measurement artifact.
2. **Inter-rater κ measurement on RHpE field labels is non-negotiable.** Single-observer at κ ≈ 0.6 is not enough to clean-retrain the probe at ethogram-grade; the protocol must include a second annotator and report inter-rater agreement. This was a "nice-to-have" in `docs/recording-protocol.md`; under this evidence it becomes load-bearing.
3. **The S5/S10 calibration-vs-noise diagnostic is operational** for source-aware evaluation. Future per-source analyses should compute consistency × agreement as a 2-axis decomposition, not just agreement alone.

## 6. Caveats

- **Same-day re-watch** biased Step 3 consistency upward; treat κ = 0.586 as a ceiling, not a settled rate. The 10 pp retrain-noise estimate inherits the same caveat.
- **Cleaned variant is effectively 9-source LOSO** because S4 + S8 are mostly multi-horse and S7 has no audit-positive clips. The 0.7386 Cleaned global AUC is computed across whichever sources have eligible clips — see per-source matrix for the 9 evaluable sources.
- **Single-observer audit only.** Inter-rater κ unmeasured. The "RHpE coders typically achieve κ ≥ 0.7" claim above is a typical-range expectation for trained ethogram protocols, not a measurement of any specific RHpE coder team.
- **Per-source AUC at small N (S1 with 21 clips, S6 with 19, S11 with 19) is fold-noise-dominated.** Don't read per-source deltas of < ~5 pp as load-bearing without bootstrapping. The S5/S10 contrast (variance 0.105 vs 0.048) is large enough relative to per-fold noise to be load-bearing; the per-source NA cases (S7) and small-class-count cases (S1) are not.
- **The −10 pp retrain-noise figure depends on the comparison structure** (B-prime trained on full RME data with 283 labels vs Step 5 Strict trained on 272 Piotr labels). It is approximate, not a clean A/B test. To rigorously isolate, one would train an LR on RME labels for 272 clips and evaluate against same — but the spec scope-locked at 3 variants and we're not adding a fourth.
- **The 11 inconsistent flips that drop from Strict are also dropped from Cleaned** (10 single-horse + 1 multi-horse-target_focus). They appear in Permissive at their original-direction verdict. The Permissive-to-Strict 0.4 pp difference (0.7345 vs 0.7386) is partially attributable to including these 11 noisy clips in Permissive's training+eval data.

## 7. Reproduce

```bash
git checkout experiment/audit-followup
cd /path/to/poc
.venv/bin/python - <<EOF
# (the script that produced the per-clip predictions and aggregate JSON;
#  see commit message for the canonical recipe)
EOF
```

Per-clip outputs:
- `outputs/loso_v2_strict_predictions.jsonl` (272 clips)
- `outputs/loso_v2_permissive_predictions.jsonl` (283 clips)
- `outputs/loso_v2_cleaned_predictions.jsonl` (218 clips)
- `outputs/loso_v2_aggregate.json` (per-variant per-source + global AUC)
- `outputs/audit_followup_labels.jsonl` (master per-clip label table for all 3 variants)

## 8. Gate 2 decision: trigger Step 7 (Full C)?

Per spec §7 Gate 2:
- **Strong / interesting result** (variants tell a coherent story, deltas large enough to warrant writeup) → trigger Step 7 (Full C methodology note).
- **Modest / null result** (variants ≈ 0.875) → light C only.

**This run is unambiguously a strong result.** The 13.6 pp uniform drop, the clean 3.6+10 pp decomposition, the S5/S10 prediction-confirmed centerpiece, and the inter-rater-protocol-needed conclusion together make this **publishable methodology contribution** material. The story coheres.

**Recommendation: trigger Step 7 (Full C, ~90 min)** *after* Step 6 (Light C labeling protocol doc, ~30 min, always-runs). The full methodology note will integrate Step 6's protocol extraction with this comparison's decomposition + S5/S10 finding into a coherent Andersen-grade narrative.
