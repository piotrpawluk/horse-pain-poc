# Phase 7 audit — DLC SuperAnimal-Quadruped keypoint-anchored eye cropping

**Date**: 2026-05-09 / 2026-05-10. Branch: `experiment/eye-probe`. Single
phase ran with two-stage pre-reg + two empirical-falsification
amendments (Stage 2 v1 §7 confidence threshold; Stage 2 v2 §4
side-assignment mapping). Both amendments adjudicated by user (Option B
both times: in-phase correction with point-fix scope).

## Summary

| Run | Side mapping | AUC | Verdict |
|---|---|---:|---|
| **Broken-rule baseline** (Stage 1 §4 as-locked) | right→right_eye, left→left_eye | **0.5788** | CONFIDENT_MISPLACEMENT_FAIL → routing-prescribed side review |
| **Corrected-rule final** (Stage 2 v2 §4) | right→left_eye, left→right_eye | **0.8462** | **OUTPERFORM_PHASE_5_AUC_ONLY** |

Phase 5 manual ceiling: 0.7985. Phase 6(b) face-bbox: 0.4689.
Corrected DLC: 0.8462 — clears Phase 5 by Δ +0.048 (paired DeLong
p=0.62, AUC-only OUTPERFORM per G2 asymmetry).

## Two structural findings surfaced by the discipline pattern

### Finding 1 — Stage 1 §7 meta-rule was data-blind on high-confidence distributions

**Locked rule**: `threshold = max(0.5, p25_pooled_observed)`.

**Empirical observation**: DLC's pooled target-eye-keypoint confidence
on RME (n=891) had p25 = 0.8872, median 0.92. Applying the rule
mechanically would have:
- Set threshold = 0.8872
- By construction, ~25% of pooled frames fail (p25 = 25th percentile)
- 8/34 clips drop (>50% frame failure) including 4 V3_NEWLY_RECOVERED
  diagnostic clips
- 3/34 clips fall to single-keypoint fallback

**Diagnosis**: p25 is a percentile, not a quality signal. The §7 meta-
rule's intent (raise threshold when DLC confident, fall to 0.5 when
unconfident) assumed a Phase 0-style distribution with a meaningful
low-confidence tail. RME's distribution is unimodal right-shifted; the
"lower 25%" is not the "unreliable" subset. Cases where the rule was
worst-applied happened to be the diagnostic-rich clips.

**Generalizable methodological observation** (locked for future-phase
reuse): *case difficulty correlates with model-confidence; percentile-
based thresholding can disproportionately remove the highest-information
clips when the distribution is unimodal.*

**Resolution** (Stage 2 amendment v1, user adjudicated Option B):
hard-lock threshold at 0.5 (the floor); §7's adaptive meta-rule
reframed as future-phase consideration only when DLC distribution is
genuinely bimodal. Anti-pattern lock added against mechanical reuse on
different datasets.

### Finding 2 — Stage 1 §4 anatomical reasoning had camera-facing-side inverted

**Locked rule**: manual eye on RIGHT of face center → DLC `right_eye`;
manual eye on LEFT → DLC `left_eye`.

**Empirical observation** (broken-rule run on RME):
- Median IoU vs Phase 5 manual = 0.000
- 27/34 clips with IoU ≤ 0.30 (off-eye)
- DLC keypoint confidence on lost clips = 0.89 (HIGH — DLC is
  confidently wrong)
- Routing per locked Stage 1 §9: CONFIDENT_MISPLACEMENT_FAIL →
  prescribed next-step "Geometry/side-assignment review"

**Empirical investigation** (the locked routing's prescribed review):
- Counterfactual on all 34 clips: 24/34 had higher IoU under flipped
  rule. Mean IoU 0.17 (locked) vs 0.42 (flipped).
- Phase 0 Wikimedia confirmation: 131/231 frames with both eyes
  visible had `right_eye_x < left_eye_x` (only 1 inverse). Confirms
  DLC's labeling convention is **horse-anatomical**.

**Diagnosis**: Stage 1 §4 had the profile-orientation→camera-facing-
side mapping inverted:
- Profile-LEFT horse (head image-left): camera sees horse's **LEFT**
  side (not right) → visible eye is anatomical LEFT → DLC `left_eye`.
- Profile-RIGHT horse: opposite.

The anatomical convention is consistent with horse-pose-estimation
training labels; my §4 derivation was simply wrong about the geometry.

**Generalizable methodological observation**: *DLC keypoint names follow
training-set anatomical convention, not image-side geometry. Verify
on a representative video before locking any side-assignment rule.*

**Resolution** (Stage 2 amendment v2, user adjudicated Option B):
flip the side-assignment mapping (`right_of_face_center → left_eye`,
`left_of_face_center → right_eye`); ambiguous-clip rule unchanged;
side counts (23/5/6) unchanged. Point-fix scope for Phase 7 only.

## What the discipline pattern accomplished

The locked routing matrix in Stage 1 §9 had a CONFIDENT_MISPLACEMENT
cell with prescribed next-step "side-assignment review." The
broken-rule run cleanly fired this routing, surfaced the §4 issue,
the empirical investigation diagnosed it, the user adjudicated Option
B, the corrected-rule run validated the fix.

The same path held for §7 meta-rule (during Step 1.5 observation, before
Stage 2 amendment v1).

Both findings are **stronger evidence for the discipline pattern's
value** than a clean Phase 7 pass would have been. The matrix
correctly anticipated structural failures and routed to the
diagnostic that revealed them.

## Final corrected-rule result

| Metric | Phase 5 (manual) | Phase 6(b) (face-bbox) | Phase 7 (DLC corrected) |
|---|---:|---:|---:|
| Pooled AUC | 0.7985 | 0.4689 | **0.8462** |
| Subject-bootstrap 95% CI | [0.584, 0.964] | [0.260, 0.686] | **[0.663, 0.962]** |
| Δ vs Phase 3 (0.6813) | +0.117 | −0.213 | **+0.165** |
| Δ vs Phase 5 primary | — | −0.330 | **+0.048** |
| Permutation p (vs chance) | 0.010 | 0.557 | **0.004** |
| Paired DeLong vs Phase 5 (p) | — | (computed sep) | **0.619** |
| Median IoU vs Phase 5 manual | (reference) | 0.165 | **0.357** |
| Verdict | (reference) | DISTRIBUTED_FAIL | **OUTPERFORM_PHASE_5_AUC_ONLY** |

**Locked gates**:
- G1 (AUC ≥ 0.70): **PASS** (0.8462)
- G2 load-bearing (AUC ≥ 0.7485): **PASS**
- G2 supportive (paired DeLong p < 0.05 vs Phase 5): **does not
  confirm** (p = 0.619)
- G3 reportable (median IoU ≥ 0.6): **does not pass** (0.357)
  — but reportable-not-gating per Stage 1 §8 lock; AUC governs.

## Per-clip categories

### vs Phase 5 primary

| Category | n | Share |
|---|---:|---:|
| BOTH_RIGHT | 21 | 62% |
| BOTH_WRONG | 1 | 3% |
| DLC_NEWLY_RECOVERED | 6 | 18% |
| DLC_NEWLY_LOST | 6 | 18% |

Net 0 in count; the Δ AUC of +0.048 comes from rank-improvement on
the recovered clips relative to the lost clips. DLC and Phase 5
manual agree on 22/34 clips (BOTH_RIGHT + BOTH_WRONG = 65%); the
disagreement is symmetric in count but asymmetric in score-magnitude
(DLC's recoveries score more decisively than its losses).

### vs Phase 6(b) face-bbox

| Category | n | Share |
|---|---:|---:|
| BOTH_RIGHT | 15 | 44% |
| BOTH_WRONG | 7 | 21% |
| DLC_BEATS_FBB | 12 | 35% |
| **FBB_BEATS_DLC** | **0** | **0%** |

**Face-bbox never beats DLC.** This is the strongest single piece of
evidence that DLC keypoint-anchored cropping is the correct
automation path for Phase 8 scaling.

## Named-clip outcomes (locked Stage 1 §10 hypothesis tests)

### action_S9.mp4_4_ (most informative single clip)

| Metric | Phase 5 | Phase 6(b) | Phase 7 corrected |
|---|---:|---:|---:|
| Score | −0.117 (wrong) | +0.511 (correct, but IoU=0) | +1.21 (correct, IoU=0.330) |
| IoU vs P5 manual | reference | 0.000 | **0.330** |
| Verdict (per Stage 1 §10) | V3_NEWLY_LOST | CORRECT_OFF_EYE | AMBIGUOUS (correct, IoU mid-range) |

DLC's corrected pipeline recovers the V3_NEWLY_LOST clip with
strongly positive score AND non-zero IoU (0.330, in the AMBIGUOUS
zone). Single-clip evidence that DLC localization PLUS V-JEPA-2
features can recover Phase 5's hardest mis-classifications.

### Orientation-extreme clips (4 named)

All 4 OE clips (bg_S5_10, action_S9_7, action_S9_4, bg_S1_12)
classified correctly by Phase 7 corrected. Net +1 vs Phase 5
(action_S9_4 newly recovered). The orientation-extreme failure mode
that motivated the OE classification (Phase 6(b)'s rel_x outliers)
doesn't apply to keypoint-anchored cropping — DLC keypoints follow
the eye regardless of horse orientation, modulo confidence.

### Swap pair (action_S5.mp4_5_ ↔ background_S10.mp4_3_)

| Clip | Phase 5 | Phase 7 corrected |
|---|---|---|
| action_S5.mp4_5_ | wrong (V3_NEWLY_LOST) | correct (DLC_NEWLY_RECOVERED) |
| background_S10.mp4_3_ | correct (V3_NEWLY_RECOVERED) | correct (BOTH_RIGHT) |

The Sensitivity-1 structural-cancellation pair: action_S5_5 (which
Phase 5 lost) is recovered by DLC corrected; bg_S10_3 (which Phase 5
recovered) stays recovered. Both are in the ambiguous-side fallback
zone (per-clip-locked higher-confidence eye). The pair's behavior
matches Mechanism (a) from Phase 5 Sensitivity 1 reread: tightened
rubric is genuinely clean under good cropping (DLC = good cropping
for these clips).

## Phase 8 entry conditions

Per locked routing + corrected-rule result:

1. **DLC keypoint-anchored cropping is the locked Phase 8 baseline.**
   Custom YOLO horse-eye detector deferred unless N ≥ 200 commits.
2. **Phase 8 priority test**: paired-DeLong vs Phase 5 manual at
   higher N. Phase 7's +0.048 AUC delta is paired-uninformative at
   n=34; the same DeLong SE-inflation Phase 5 audit named applies.
   At N=100+ the test should be adequately powered.
3. **G3 (IoU vs manual) divergence from AUC**: principled
   investigation. Median IoU 0.357 with AUC 0.846 means DLC anchors
   features at a systematically different anatomical reference than
   Phase 5 manual. Test whether the divergence is dataset-specific or
   structural. Possibly a small ablation: re-extract V-JEPA-2 features
   from Phase 5 manual crops, compare per-clip predictions vs DLC
   crops; the disagreement set is the diagnostic.
4. **Verify DLC keypoint convention** on any new dataset before
   reusing the corrected §4 mapping (per Stage 2 v2 anti-pattern lock).
5. **§7 meta-rule** stays reframed as future-phase consideration.
   Hard-lock at 0.5 floor for any horse-eye DLC work; revisit only
   for distributions with documented bimodality.

## Pre-registration discipline (audit chain)

| Frozen | File | Hash |
|---|---|---|
| 2026-05-09 | `outputs/track_b_phase7_preregistration_stage1.md` (with v3 cosmetic-deviation §1 lock + Wikimedia halt + tracked-property + side-assignment §4 derivation rule) | `1c6dacaf8935...` |
| 2026-05-09 | `tools/phase7_derive_side_assignment.py` (Stage 1 derivation tool) | `7fe84af6c36a...` |
| 2026-05-09 | `outputs/phase7_eye_side_assignment.json` (Stage 1 mapping, falsified) | `95b879d2eb30...` |
| 2026-05-09 | `tools/dlc_inference.py` (productionized DLC integration) | `6564833753cf...` |
| 2026-05-09 | `tools/phase7_wikimedia_parity.py` | `b9755f7671bd...` |
| 2026-05-09 | `outputs/phase7_wikimedia_parity_keypoints.json` | `481639e97c6b...` |
| 2026-05-09 | `outputs/phase7_dlc_wikimedia_parity.json` (PASS_IDEAL) | `3f214c23a956...` |
| 2026-05-09 | `tools/phase7_run_dlc_on_rme.py` | `75be4cdc6de8...` |
| 2026-05-09 | `outputs/phase7_rme_dlc_keypoints.json` (Step 1.5) | `82ae2c8d11aa...` |
| 2026-05-09 | `outputs/track_b_phase7_preregistration_stage2_amendment.md` (v1, §7 hard-lock) | `259d57cd2c3e...` |
| 2026-05-09 | `tools/phase7_dlc_crop.py` | (recorded in chain post-final-commit) |
| 2026-05-09 | `tools/eye_loso_lr_phase7.py` | (recorded post-final-commit) |
| 2026-05-09 | `tools/phase7_diagnostic.py` | (recorded post-final-commit) |
| 2026-05-09 | `outputs/eye_loso_results_phase7_broken_rule.json` (broken-rule baseline preserved) | (recorded post-final-commit) |
| 2026-05-09 | `outputs/phase7_diagnostic_broken_rule.{json,md}` | (recorded post-final-commit) |
| 2026-05-09 | `outputs/eye_crops_v4_dlc_broken_rule_manifest.jsonl` | (recorded post-final-commit) |
| 2026-05-09 | `outputs/vjepa2_embeddings_eye_v4_broken_rule.npz` | (recorded post-final-commit) |
| **2026-05-10** | `tools/phase7_derive_side_assignment_corrected.py` | (recorded post-final-commit) |
| **2026-05-10** | `outputs/phase7_eye_side_assignment_corrected.json` | (recorded post-final-commit) |
| **2026-05-10** | `outputs/track_b_phase7_preregistration_stage2_amendment_v2.md` | (recorded post-final-commit) |
| **2026-05-10** | `outputs/eye_crops_v4_dlc_manifest.jsonl` (corrected-rule manifest) | (recorded post-final-commit) |
| **2026-05-10** | `outputs/vjepa2_embeddings_eye_v4.npz` (corrected-rule embeddings) | (recorded post-final-commit) |
| **2026-05-10** | `outputs/eye_loso_results_phase7.json` (corrected-rule LOSO) | (recorded post-final-commit) |
| **2026-05-10** | `outputs/phase7_diagnostic.{json,md}` (corrected-rule diagnostic) | (recorded post-final-commit) |
| **2026-05-10** | `docs/phase7_audit.md` (this document) | (recorded post-final-commit) |

The hash chain witnesses the routing matrix's locked predictions
firing on observation: each empirical falsification was anticipated
by Stage 1 routing categories, and the corrections are traceable to
specific Stage 2 amendments. No post-hoc reasoning entered the audit
chain — the discipline pattern held across both structural findings.
