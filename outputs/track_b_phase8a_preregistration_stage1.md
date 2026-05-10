# Track B Phase 8a — Stress-test Phase 7 finding

## Stage 1 pre-registration

**Frozen 2026-05-10 BEFORE any 8a compute.** Stage 1 locks all 8a
parameters, gates, and routing decisions before observation. Phase 8a
operates entirely on existing artifacts (Phase 5 manual outputs, Phase
7 corrected outputs, V-JEPA-2 v3 + v4 embeddings, Phase 5 manual eye
boxes); no new data collection, no new model inference required.

Phase 8a is a methodological self-audit: stress-test whether Phase 7's
+0.048 AUC advantage over Phase 5 manual is robust or noise, and what
the G3/AUC divergence actually represents.

---

## What 8a tests

Phase 7 closed with verdict **OUTPERFORM_PHASE_5_AUC_ONLY** — AUC
0.8462 vs Phase 5 manual at 0.7985 (Δ +0.048), with paired-DeLong
p=0.619 inconclusive at n=34 due to the SE-inflation pattern Phase 5
audit identified. The locked verdict's qualifier ("AUC_ONLY") was a
deliberate acknowledgment that the level test passes but the
significance test does not.

Phase 8a directly investigates four open questions Phase 7 left
ungated:

1. **Is the +0.048 advantage robust to subject-bootstrap resampling?**
   (Replaces the underpowered paired DeLong with a more direct
   precision estimator.)
2. **Is the advantage uniform across sources, or driven by 1-2 high-Δ
   sources?** (Robustness via per-source ablation.)
3. **Is the advantage outlier-recovery on hard cases or uniform
   improvement?** (Distribution shape via per-clip score divergence.)
4. **Does the G3/AUC divergence reflect "feature equivalence under
   different geometry" or "different features that happen to predict
   equally well"?** (Mechanism via V-JEPA-2 feature similarity.)

Plus a fifth test that operationalizes the audit-doc reframe:

5. **Does the AUC come from high-IoU clips (DLC matches manual when
   on-eye) or from low-IoU clips also predicting well (DLC genuinely
   robust to placement)?** (Reframe verification via G3-IoU-conditional
   AUC.)

---

## What 8a is NOT

- Not a re-run of Phase 7 LOSO. Existing `eye_loso_results_phase7.json`
  (corrected-rule) and `eye_loso_results_phase5_primary.json` outputs
  drive all 8a analyses.
- Not new data collection or new annotation. All five tests operate on
  existing artifacts hash-locked in the audit chain.
- Not an opportunity to re-tune Phase 7's locked parameters. The
  side-assignment correction (§4 v2), confidence threshold (§7 v1),
  and all Stage 1 + Stage 2 amendments remain locked.
- Not a substitute for Phase 9 N-expansion. 8a quantifies the precision
  of the +0.048 claim at n=34 but cannot resolve the question of
  whether the advantage holds at higher N.

---

## Locked tests

### Test 1 — Subject-bootstrap CI on Δ AUC

**Method**: Parent-clip resampling, B=10000, seed=42. For each
bootstrap iteration, resample sources with replacement (matching the
LOSO fold structure); for each resampled source, take all its clips;
compute pooled AUC for both Phase 5 (v3) and Phase 7 (v4-corrected)
predictions on the resampled clip set; take the per-bootstrap
difference Δ_b = AUC_v4_b − AUC_v3_b. Report bootstrap distribution of
Δ_b: mean, median, 95% CI bounds.

**Rationale**: Phase 7's paired DeLong (p=0.619) was uninformative.
Subject-bootstrap CI on Δ directly characterizes the precision of the
delta estimator, sidestepping DeLong's SE-inflation issue at n=34 with
weak prediction-pairing between distinct cropping pipelines.

**Pre-registered three-band gate**:

| Bootstrap CI lower bound (LB) | Verdict | Phase 9 implication |
|---|---|---|
| **LB ≥ +0.02** | Phase 7 verdict robust | DLC strictly better at this scale; Phase 9 N-expansion's marginal value drops |
| **−0.02 < LB < +0.02** | Underpowered to determine direction | Confirms paired-DeLong's inconclusive verdict; Phase 9 N-expansion is the resolution |
| **LB ≤ −0.02** | Bootstrap suggests Phase 7 verdict was favorable to DLC by chance | **Phase 7 retraction-style amendment** documenting the revised verdict; halt 8b until retraction is committed |

The retraction-trigger band is pre-registered explicitly. Phase 7's
verdict shouldn't survive 8a's test if 8a says the direction was wrong.

**Failure-mode pre-registration**: If LB ≤ −0.02 fires, 8a halts;
all subsequent tests in 8a become informational-only (still computed
for the audit doc, but routing decisions defer to the retraction
amendment). Phase 8b is blocked until Phase 7 retraction is committed.

**Retraction procedure (LOCKED)**: If Phase 8a Test 1 lower bound
≤ −0.02 fires the retraction trigger, the Phase 7 audit doc receives
an appended section **"Phase 7 Retraction (per Phase 8a Stage 1 Test 1)"**
which:

(a) **Explicitly retracts** the OUTPERFORM_PHASE_5_AUC_ONLY verdict
   from Phase 7's locked routing in `docs/phase7_audit.md`. The
   retraction is a structural amendment, not a verdict-softening
   reframe.

(b) **Replaces the verdict** with the Phase 8a-evidenced verdict.
   Two pre-registered candidate replacement verdicts depending on
   bootstrap distribution shape:
   - `UNDERPOWERED_INDISTINGUISHABLE` if the bootstrap distribution
     is roughly symmetric around 0 with the LB sitting in the
     negative band by chance (i.e., median Δ near 0, std large).
   - `PHASE_7_FAVORABLE_BY_CHANCE` if the bootstrap distribution
     skews negative (median Δ < 0; the +0.048 point estimate was a
     positive outlier of the bootstrap distribution).

   Choice between the two is mechanical: median Δ_bootstrap > −0.01 →
   UNDERPOWERED_INDISTINGUISHABLE; median Δ_bootstrap ≤ −0.01 →
   PHASE_7_FAVORABLE_BY_CHANCE.

(c) **Preserves the original verdict text inline** as a struck-through
   block in the Phase 7 audit doc. Both findings appear in the audit
   chain — the original verdict (with its reasoning) is not erased,
   it is *retracted with reason*. Future readers see the empirical
   falsification trail, not just the corrected end-state.

(d) **Original Phase 7 hash remains valid** as a historical artifact
   pinning the pre-retraction state. The retraction adds a new
   audit-doc hash to the chain (Phase 7 audit doc supersession with
   explicit retraction notation in the chain entry). This mirrors
   how Stage 2 amendments superseded Stage 1 hashes in earlier phases.

(e) **Phase 8b is blocked** until the retraction commit lands in the
   audit chain. The user-approval checkpoint between 8a result and
   8b entry has an additional gate: explicit acknowledgment of the
   retraction.

The retraction-procedure pre-registration here ensures the
operational moves under cognitive load of a falsified favorable
verdict are mechanical, not negotiable. The §7 and §4 amendments to
Phase 7 itself preserved empirical-falsification evidence in the
audit chain; this clause does the same for Phase 7's verdict if Phase
8a falsifies it.

### Test 2 — Per-source LOSO ablation

**Method**: For each of the 12 sources S1-S12, drop that source from
the dataset; recompute pooled AUC for both Phase 5 (v3) and Phase 7
(v4-corrected) on the remaining 11-source dataset; report ΔAUC per
dropped source.

**Rationale**: If the +0.048 advantage is concentrated in 1-2 sources
(dropping which collapses Δ to 0 or negative), the result is fragile.
If 9-12 sources contribute positive Δ (dropping each only modestly
shifts Δ), the result is robust.

**Pre-registered three-band gate**:

| Sources contributing positive Δ | Verdict |
|---|---|
| **≥9/12** | Robust — distributed across most sources |
| **6-8/12** | Modest — present but heterogeneous |
| **≤5/12** | Fragile — concentrated in a minority of sources |

Method note: "contributing positive Δ" means dropping that source
yields a non-negative Δ on the remaining 11. The drop-source recomputed
Δ measures the source's contribution to the original advantage.

**Cost**: ~24 LOSO runs at ~5 sec each = ~2 min total. Cheap.

### Test 3 — Per-clip score divergence

**Method**: For all 34 clips, compute (v4_corrected_score −
v3_score). Report distribution: mean, median, std, p25, p75, min, max,
histogram bins. Categorize per Phase 6(a) framework using the same
threshold logic (RidgeClassifier decision_function threshold=0):
`BOTH_RIGHT`, `BOTH_WRONG`, `DLC_NEWLY_RECOVERED`, `DLC_NEWLY_LOST`
(already computed in Phase 7 diagnostic).

**Rationale**: Distinguishes outlier-recovery (most clips near zero,
few large-positive) from uniform improvement (tight distribution
around +0.048). Different mechanisms, different Phase 9 implications.

**Reportable, not gated**: feeds the mechanism narrative in audit doc
without binary verdict. Already partially computed in Phase 7
diagnostic (DLC_NEWLY_RECOVERED=6, DLC_NEWLY_LOST=6); 8a expands
with full distribution + per-clip identity.

### Test 4 — V-JEPA-2 feature similarity

**Method**: For each of the 34 clips, compute cosine similarity
between Phase 5 v3 embedding (1024-d) and Phase 7 v4-corrected
embedding (1024-d). Report distribution: mean, median, std, p25, p75.

Inputs:
- `outputs/vjepa2_embeddings_eye_v3_m15.npz` (Phase 5 v3 embeddings)
- `outputs/vjepa2_embeddings_eye_v4.npz` (Phase 7 corrected embeddings)

**Rationale**: Direct empirical test of the G3/AUC divergence. If
features are similar despite low IoU (mean cos > 0.7), V-JEPA-2
abstracts crop-placement differences — both crops yield similar
features. If features differ despite similar AUC (mean cos < 0.7), the
two pipelines extract genuinely different features that happen to
predict equally well, which is a stronger Phase 9 hypothesis worth
testing at higher N.

**Pre-registered two-band gate**:

| Median cosine similarity | Verdict |
|---|---|
| **≥ 0.7** | Same-features-different-geometry (V-JEPA-2 robust to crop placement; less interesting; the +0.048 is mostly noise within feature equivalence) |
| **< 0.7** | Different-features-similar-prediction (Phase 9 hypothesis: keypoint anchoring extracts signal that human-annotated tight crops miss; could be reproducible at higher N) |

The 0.7 threshold is borrowed from the project's intra-rater IoU
threshold (Phase 5b median 0.765). Anchored on the project's own
calibration of "high agreement."

### Test 5 — G3-IoU-conditional AUC

**Method**: Bucket the 34 clips into three groups by IoU between
Phase 7 corrected eye box (middle frame) and Phase 5 manual middle
keyframe box:
- `IoU < 0.30` (off-eye)
- `0.30 ≤ IoU < 0.50` (mid)
- `IoU ≥ 0.50` (on-eye)

Phase 7 diagnostic already reports per-clip IoUs; reuse those.
Compute per-bucket pooled AUC for Phase 7 corrected predictions.
Report bucket sizes + per-bucket AUC + per-bucket bootstrap CI
(B=10000, seed=42, parent-clip resampling within bucket).

**Rationale**: Operationalizes the audit-doc reframe — "manual is
*a* near-optimum, alternatives may be equivalent or better." Two
distinct mechanisms produce the +0.048:

1. **High-IoU drives AUC**: Phase 7's AUC is concentrated on clips
   where DLC happens to land on the eye (matching manual). Off-eye
   clips contribute little. Reframe is qualified to "DLC matches
   manual when on-eye; off-eye performance is noisy."
2. **Low-IoU clips also predict well**: Phase 7's AUC on off-eye
   clips is comparable to on-eye AUC. DLC is genuinely robust to
   crop placement; the +0.048 reflects a different anatomical
   reference that's at least as discriminative as Phase 5's manual
   eye region. Reframe is confirmed.

**Reportable, not gated as binary verdict**. The per-bucket AUCs feed
the mechanism narrative; if low-IoU AUC ≥ high-IoU AUC, that's the
strongest evidence for the placement-robustness reading.

**Sample-size caveat**: at n=34, bucket sizes will be small
(approximately 10-15 per bucket given Phase 7's IoU distribution:
10/34 ≤ 0.30, 5/34 ≥ 0.50, 19/34 in 0.30-0.50). Per-bucket AUCs are
informative-but-noisy; bootstrap CI captures the uncertainty.

---

## Diagnostic instrumentation (alongside the 5 tests)

1. **Test 1 + 2 cross-check**: if test 2 routes "fragile" but test 1
   routes "robust" (LB ≥ +0.02), report the apparent inconsistency.
   This can happen when one source has a very large positive Δ
   (driving the overall +0.048) but the bootstrap CI is wide enough
   to still exclude 0 due to the source's contribution. Worth
   surfacing.
2. **Test 3 outlier identification**: Top-3 positive score deltas
   (clips where DLC most strongly improved over manual) and Top-3
   negative deltas (clips where DLC most strongly lost). Names + values.
   Already computed in Phase 7 diagnostic; surface in 8a audit.
3. **Test 4 + 5 cross-check**: per-clip table with `score_delta`,
   `cos_sim`, `IoU` columns. Look for clips where cos_sim is high but
   score_delta is large — those are clips where features are similar
   but predictions diverge, which is a different mechanism than the
   ones tests 4 and 5 categorize. Surface as "outlier per-clip pattern."
4. **Per-source per-bucket cross-tab**: 12 sources × 3 IoU buckets.
   Identifies sources that are systematically off-eye vs on-eye
   relative to Phase 5 manual annotations. Could indicate per-source
   anatomical-reference differences.

---

## Anti-patterns (LOCKED)

1. **No re-tuning of Phase 7 parameters.** All Stage 1 + Stage 2
   amendments locked; 8a operates on existing artifacts only.
2. **No mid-test re-interpretation of gates.** The three-band test 1
   gate (LB ≥ +0.02 / [-0.02, +0.02] / LB ≤ -0.02) is locked;
   routing decisions follow it mechanically. Same for test 2 and
   test 4 gates.
3. **No new "robustness" tests added mid-run** if 8a's results land
   in unexpected regions. The five tests are locked here. Phase 9
   pre-registers further tests if needed.
4. **No celebration of test 1 LB ≥ +0.02 as resolving Phase 9.**
   Bootstrap CI at n=34 is precision-bounded; +0.02 LB doesn't mean
   N-expansion is unnecessary, only that its marginal value drops.
5. **No retraction-band silencing.** If test 1 LB ≤ −0.02, the
   amendment must explicitly retract Phase 7's verdict; subsequent
   reframes (e.g., "let's call it INCONCLUSIVE_PHASE_5_AUC_ONLY")
   are not allowed. Retraction means retraction.

---

## Sequencing

| Step | Action | Output |
|---|---|---|
| 0 | User approval of this Stage 1 pre-reg | Hash-locked artifact |
| 1 | Build `tools/phase8a_stress_test.py` (single tool, all 5 tests) | tool source |
| 2 | Run all 5 tests on existing artifacts | `outputs/phase8a_results.json` |
| 3 | Diagnostic instrumentation + verdict per locked gates | (in same JSON) |
| 4 | Audit doc draft `docs/phase8a_audit.md` | doc |
| 5 | User-approval checkpoint #2 (results review, audit doc lock) | — |
| 6 | Hash chain extension + commit + subtree-push | mirror sync |

**User-approval checkpoints (binding pauses)**:

1. After this Stage 1 doc approval, before any 8a compute.
2. After 8a results computed + audit doc drafted, before final hash-lock
   and Phase 8b entry decision.

If test 1 fires the retraction-band, an additional checkpoint is
inserted: user must explicitly approve the Phase 7 retraction
amendment before 8b proceeds.

---

## Phase 9 entry conditions (informed by 8a routing)

Per refinement 6 of the second-judge review (locked here for
mechanical Phase 9 entry decision when 8a + 8b both close):

| 8a outcome (test 1 LB) | 8b outcome | Phase 9 priority |
|---|---|---|
| ≥ +0.02 (robust) | strong (≥0.80) | Multi-rater κ when 2nd rater available; N expansion lower marginal value |
| ≥ +0.02 (robust) | modest (0.65-0.80) | Cross-behavior detector cost-benefit study; multi-rater κ |
| Underpowered ([−0.02, +0.02]) | strong | N expansion to resolve paired-DeLong on eye; methodology cross-validation already done by 8b |
| Underpowered | modest | N expansion + multi-rater κ both; broader methodology validation needed |
| Underpowered | <0.65 | Phase 7 result is eye-specific; narrow project narrative; N expansion only if eye-specific work is worth scaling |
| ≤ −0.02 (retraction) | (any) | Halt forward work; Phase 7 retraction amendment + project-narrative revision |

---

## Cost / time estimate

| Step | Estimate |
|---|---:|
| Pre-reg approval cycle | ~30 min (this doc + user review) |
| Tool implementation | ~60 min |
| 5-test execution + diagnostics | ~10 min compute |
| Audit doc draft + user review | ~60 min |
| Hash chain + commit + push | ~20 min |
| **Total wall-clock** | **~3 hours** |

---

## Two nuances deferred to audit doc (per second-judge review)

1. **8b "modest" band framing nuance** — if 8b lands in 0.83-0.85,
   that's an exceptionally strong outcome despite landing in the
   "modest" band per the recalibrated gates. Audit doc framing should
   acknowledge: DLC at 0.83 with generic training competing against
   custom-trained YOLO at 0.875 is a stronger generalization claim
   than the band's neutral language conveys.
2. **Multi-rater κ "no rater within 6 months" fallback wording** —
   pre-reg specifies action when rater is found; audit doc adds:
   "If no qualified second rater is identified within 6 months,
   multi-rater κ remains an open methodological gap to be addressed
   when rater availability changes; no time-based expiration of the
   pre-registered scope."

These are framing/wording, not parameter changes; they belong in the
audit doc, not the pre-reg.

---

## RME pipeline framing verification (closing CC's flagged inaccuracy)

**Confirmed 2026-05-10 via inspection of
`notebooks/01_read_my_ears_replicate.ipynb`**: Read My Ears protocol
uses `vendor/horse-face-ear-detection/horse_ear_detection/yolov8l_horse_ear_detection.pt`
— a custom-trained YOLOv8l ear detector trained on horse-specific ear
data. The 0.875 LOSO baseline reproduces the paper's protocol with
this custom-trained detector.

For Phase 8b, the comparison is:
- **RME baseline** = custom-trained YOLOv8l ear detector + face mask + ear bbox crop + linear probe
- **DLC keypoint-anchored ear (Phase 8b)** = generic SuperAnimal-Quadruped → ear keypoints → keypoint-anchored bbox

Phase 8b's recalibrated gates (≥0.80 strong / 0.65–0.80 modest /
<0.65 fails) are anchored on the asymmetry: generic multi-species
pose model competing against horse-ear-specific custom detector.
Phase 8b's verification is locked in its own Stage 1 pre-reg
(separate doc, drafted after 8a closes).

---

## User approval signature

User has reviewed and approves Phase 8a Stage 1 lock as drafted, including:

- All 5 tests with locked methodology
- Test 1 three-band gate (LB ≥ +0.02 robust / [−0.02, +0.02] underpowered / LB ≤ −0.02 retraction trigger)
- Test 2 three-band gate (≥9 / 6-8 / ≤5 sources)
- Test 4 two-band gate (median cos ≥0.7 / <0.7)
- Test 3 + 5 reportable-not-gated
- 4 diagnostic instrumentation rules
- 5 anti-patterns
- 6 Phase 9 entry conditions (mechanical routing)
- 2 user-approval checkpoints
- 2 audit-doc framing nuances deferred (not pre-reg)

User signs off → CC builds `tools/phase8a_stress_test.py` → runs → drafts audit doc → user-approval checkpoint #2 → final commit + Phase 8b entry decision.

---

## Erratum (added 2026-05-10 per Phase 8b Stage 1 verification)

**Forward-pointing additive correction; original locked content above
is unchanged.**

The "RME custom-YOLO ear detector" framing in §"RME pipeline framing
verification (closing CC's flagged inaccuracy)" of this document is
**factually correct about the existence of the custom YOLO ear
detector** in `vendor/horse-face-ear-detection/horse_ear_detection/`,
and the project's `notebooks/01_read_my_ears_replicate.ipynb`
genuinely uses that detector. **However, the implication carried in
this pre-reg's gate-calibration rationale — that the project's LOSO
0.875 reference baseline uses the custom YOLO ear-cropped pipeline —
is incorrect.**

Phase 8b Stage 1 pre-registration verification traced the actual
LOSO 0.875 reference to its source: `notebooks/02_vjepa2_zeroshot.ipynb`
extracts whole-frame V-JEPA-2 ViT-L mean-pool features on raw RME
mp4s (no ear cropping, no face masking). Cached features at
`outputs/vjepa2_embeddings.npz` (283 × 1024) produce Sanity 5's
`ssv2_motion` LOSO 0.8746126936531734.

The custom YOLO ear detector exists and is used in notebook 01 to
reproduce the **paper's** original optical-flow pipeline. It is NOT
used in the project's V-JEPA-2 LOSO 0.875 reproduction (which is the
reference baseline cited throughout this document).

**What this means for Phase 8a's gate values**: nothing. Phase 8a's
five tests (subject-bootstrap CI on Δ AUC; per-source LOSO ablation;
per-clip score divergence; V-JEPA-2 feature similarity; G3-IoU-
conditional AUC) are all computed on Phase 5 manual eye-cropped
features vs Phase 7 corrected DLC eye-cropped features. The framing
error about RME baseline is in this pre-reg's *secondary* discussion
(closing the Phase 8b gate-recalibration justification); it does NOT
affect Phase 8a's actual test methodology or results.

The Phase 8a result (RETRACTION verdict, UNDERPOWERED_INDISTINGUISHABLE
replacement) stands as recorded.

**Phase 8b's full corrected framing** is documented in
`outputs/track_b_phase8b_preregistration_stage1.md` ⚠ Framing
correction surfaced pre-lock section. That document's gate-
interpretation framing is the authoritative reference; this Phase 8a
pre-reg's secondary discussion of "RME custom-YOLO baseline" should
be read as historical-error-preserved-for-audit-trail-transparency,
not as a locked-rule-claim.

**Why this erratum is additive, not retroactive**:
- Phase 8a Stage 1 pre-reg is locked + committed + merged + pushed
  (hash `c0bf2bdca8d534ec50683525078f4ced7f5f3fab86c7bd5696a6ed96f95d392c`)
- Modifying the original locked content would violate the discipline
  pattern's audit-chain-witness role
- This erratum is appended-only; the original above is untouched
- Future readers see both: original framing-as-locked + correction
- Same additive pattern as Phase 7's §7/§4 amendments and Phase 7
  retraction (originals struck-through-not-erased)

The discipline pattern is built around additive corrections. This is
one of them.
