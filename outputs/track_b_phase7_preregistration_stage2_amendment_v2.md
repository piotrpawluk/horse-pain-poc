# Track B Phase 7 — Stage 2 amendment v2 (§4 side-assignment correction)

**Frozen 2026-05-10 AFTER initial Phase 7 broken-rule run BUT BEFORE
the corrected-rule re-run was committed.** This amendment supersedes
Stage 1 §4's mapping for Phase 7 only — empirical falsification of
the locked anatomical reasoning. Stage 2 amendment v1 (confidence
threshold §7) remains in force.

---

## What Stage 1 §4 said

Eye-side assignment per clip:
- manual eye on RIGHT of face center → DLC `right_eye` keypoint
- manual eye on LEFT of face center → DLC `left_eye` keypoint
- ambiguous (|offset| < 0.05) → per-clip lock (rule (a))

Anatomical reasoning behind the mapping (Stage 1 §4 prose):
> "Anatomical mapping: visible eye in image-right of face = horse's
> right_eye, etc."

## What the broken-rule Phase 7 run observed

Phase 7 ran on the 34 RME clips with Stage 1 §4 mapping. LOSO result:
- Pooled AUC 0.5788 (G1 FAIL)
- Δ vs Phase 5 = −0.220
- Median IoU vs Phase 5 manual = 0.000
- 27/34 clips with IoU ≤ 0.30 (off-eye)
- 11 DLC_NEWLY_LOST clips
- Mean keypoint conf on lost clips = 0.89 (HIGH)
- Routing verdict (per locked Stage 1 §9): CONFIDENT_MISPLACEMENT_FAIL
- Locked next-step: "Geometry/side-assignment review first"

The locked routing predicted exactly this failure mode and prescribed
the side-assignment review as the next experiment.

## The empirical falsification

The locked routing's prescribed review (executed before any re-run):

**Test 1 — counterfactual on RME (n=34):**

For each clip, compute IoU vs Phase 5 manual eye box under both:
- Locked rule's target eye keypoint
- Flipped-rule's target eye keypoint

| Comparison | Count |
|---|---:|
| Locked rule better | 8/34 |
| **Flipped rule better** | **24/34** |
| Equal | 2/34 |
| Mean IoU at locked rule | 0.17 |
| **Mean IoU at flipped rule** | **0.42 (2.5×)** |

**Test 2 — Phase 0 Wikimedia confirmation (n=287):**

For each frame, look at right_eye/left_eye keypoint x-coordinates in
frames where BOTH eyes are confidently localized (frontal/3-quarter
view, n=231):

| Pattern | Count |
|---|---:|
| `right_eye_x < left_eye_x` (right_eye on image-LEFT) | 131/231 |
| `left_eye_x < right_eye_x` | 1/231 |
| Equal x | 99/231 |

**Conclusion**: DLC labeling convention is **horse-anatomical**
(right_eye = horse's anatomical right eye, which appears image-LEFT
of left_eye when the horse faces toward camera — mirror).

## Why Stage 1 §4 was wrong

The §4 anatomical reasoning conflated "horse profile orientation
relative to image" with "which side of the horse is camera-facing"
in the wrong direction. The empirically-correct chain:

- **Profile-LEFT horse** (head pointing image-LEFT, body extending
  image-RIGHT): camera sees horse's **LEFT** side → visible eye =
  anatomical LEFT eye → DLC labels as `left_eye`. The eye appears
  in the right-half of the face bbox (between nose at image-LEFT
  and back-of-head at image-RIGHT).
- **Profile-RIGHT horse** (head pointing image-RIGHT): camera sees
  horse's **RIGHT** side → visible eye = anatomical RIGHT eye →
  DLC `right_eye`. Eye in left-half of face bbox.

Stage 1 §4 had the camera-facing-side reasoning inverted. The
mapping flips:

| Manual eye position | Profile orientation | Visible eye (anatomical) | Corrected DLC keypoint |
|---|---|---|---|
| RIGHT of face center | profile-LEFT (head image-left) | LEFT | **`left_eye`** |
| LEFT of face center | profile-RIGHT (head image-right) | RIGHT | **`right_eye`** |
| \|offset\| < 0.05 | near-frontal | both visible / ambiguous | per-clip lock (unchanged) |

## Lock — corrected mapping

Phase 7 §4 mapping (CORRECTED, point-fix scope for Phase 7 only):

```
right_of_face_center → DLC `left_eye`     (was `right_eye` in Stage 1)
left_of_face_center  → DLC `right_eye`    (was `left_eye` in Stage 1)
ambiguous            → per-clip-lock rule (a) (unchanged)
```

The ambiguous-clip rule and margin threshold (0.05) are unchanged.
Side counts (23 right / 5 left / 6 ambiguous) are unchanged — only
the target keypoint name flips for the 28 side-assigned clips.

Hash-locked artifact: `outputs/phase7_eye_side_assignment_corrected.json`
(SHA-256 to follow in hash chain).

## Scope of the amendment (point-fix)

**§4 corrected mapping applies to Phase 7's RME data only;
future-phase reuse requires explicit verification of DLC labeling
convention on the new dataset.**

A future agent must NOT assume the corrected mapping holds
universally. SuperAnimal-Quadruped's training labels are
horse-anatomical, so the convention should be consistent across
horse datasets — but verification is cheap (§3-style parity check
on a dataset-representative video) and should be locked in before
relying on the rule.

## Result of corrected-rule re-run

Phase 7 with corrected mapping (`outputs/eye_loso_results_phase7.json`):

| Metric | Value |
|---|---:|
| **Pooled AUC** | **0.8462** |
| Subject-bootstrap 95% CI | [0.6629, 0.9615] |
| Δ vs Phase 3 | +0.1648 |
| **Δ vs Phase 5 primary** | **+0.0476** |
| Δ vs Phase 6(b) face-bbox | +0.3773 |
| Permutation p (vs chance) | 0.0040 |
| Paired DeLong vs Phase 5 (Δ, z, p) | (+0.0476, 0.50, **0.6194**) |
| Median IoU vs Phase 5 manual | 0.357 |
| 10/34 clips off-eye, 5/34 on-eye | (G3 < 0.6 → reportable, not gated) |

**G1**: AUC ≥ 0.70 → **PASS** (0.8462)
**G2 load-bearing**: AUC ≥ 0.7485 → **PASS**
**G2 supportive**: paired DeLong p ≥ 0.05 → **does not confirm**
(p = 0.62; same DeLong-paired SE-inflation issue Phase 5 audit
identified)

**Locked verdict (per Stage 1 §9 + §8 G2 asymmetry)**:
**OUTPERFORM_PHASE_5_AUC_ONLY**

The AUC evidence (0.8462 > 0.7985, +0.048) governs per the locked
G2 asymmetry rule (AUC load-bearing, paired p supportive but cannot
override). Paired-DeLong is uninformative at n=34 here for the same
mechanism Phase 5 audit named: prediction-pair correlation between
two pipelines is insufficient to drive p < 0.05 at small Δ.

**Next action** (locked routing + diagnostic-derived):
Phase 8 confirmation at higher N. Specifically: re-run DLC pipeline
on a larger dataset (~100+ clips) to test whether the AUC
improvement holds, with paired-DeLong adequately powered.

## What Phase 7 establishes (with corrected rule)

1. **DLC keypoint-anchored cropping is a viable Phase 8 scaling
   baseline.** Both gates pass; Phase 7 corrected AUC clears the
   project's pre-locked realistic LOSO band by a margin. Custom YOLO
   horse-eye detector deferred unless N ≥ 200 commits.

2. **Mid-range IoU (0.357) with strong AUC** confirms the locked
   G3-reportable-not-gating decision was correct. DLC keypoints
   anchor crops at slightly different geometric reference than Phase 5
   manual annotations, but V-JEPA-2 features extracted from those
   crops ARE more discriminative for the eye-region behavior task.
   IoU is a localization proxy, not a quality signal for downstream
   classification.

3. **Two structural findings surfaced by the discipline pattern**:
   - §7 meta-rule (percentile-as-quality-signal) — empirically falsified;
     Stage 2 amendment v1 hard-locked threshold at 0.5
   - §4 anatomical-mapping (camera-facing-side reasoning inverted) —
     empirically falsified; Stage 2 amendment v2 (this doc) flips
     the keypoint mapping

   Both surfaced through the locked routing matrix correctly
   anticipating and routing to the diagnostic that revealed them.
   The discipline pattern is doing exactly what it was designed to
   do.

## What Phase 7 does NOT establish

- **Statistically significant superiority over Phase 5 manual**.
  Paired DeLong p = 0.62; the AUC delta of +0.048 isn't paired-
  significant at n=34. Phase 8 needs to test this at higher N.
- **DLC's localization-vs-prediction divergence is principled, not a
  bug**. With median IoU 0.357 (well below Phase 5 ≥ 0.6 IoU gate),
  the question of whether DLC anchors the V-JEPA-2 features at a
  systematically different anatomical reference than Phase 5 is
  unanswered. Phase 8+ could test this with explicit IoU-vs-AUC
  ablations.
- **Generalization beyond the 34 RME clips**. The two structural
  amendments (§7 v1 + §4 v2) are point-fixes for this dataset.

## Approval signature

User adjudicated Option B: in-phase empirical correction (parallel
to §7 meta-rule). Lock §4 corrected mapping, re-run pipeline, lock
both broken-rule and corrected-rule results in the audit chain.

User signs off → CC commits Stage 2 amendment v2 + corrected
artifacts + audit doc → user reviews at checkpoint #5 (after audit
doc, before final hash-lock).
