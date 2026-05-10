# Track B Phase 8b — Cross-behavior generalization on RME ear data

## Stage 1 pre-registration

**Frozen 2026-05-10 BEFORE any 8b compute.** Stage 1 locks all 8b
parameters, gates, and routing decisions before observation. Phase 8b
operates on existing RME data (283 clips, 12 sources) plus new DLC
inference and V-JEPA-2 extraction; no new data collection required.

**This pre-reg is drafted post-Phase-8a's RETRACTION of Phase 7's
verdict.** Phase 7 verdict was retracted to UNDERPOWERED_INDISTINGUISHABLE
per Phase 8a Test 1 bootstrap CI on Δ AUC. Phase 8b cross-behavior
generalization remains a meaningful test independent of the Phase 7
retraction; the question "does DLC keypoint-anchored cropping
generalize from eye to ear?" is structurally separate from the
question "does DLC eye cropping outperform manual eye annotation?"
User adjudicated Path B (proceed with 8b under unchanged locked
gates).

---

## ⚠ Framing correction surfaced pre-lock (substantive)

While verifying the RME pipeline before drafting, the project's LOSO
0.875 reproduction was traced to its source. The result:

**The project's LOSO 0.875 result on RME is from whole-frame V-JEPA-2
mean-pool features, NOT from custom-trained YOLO ear-cropped features.**

Source: `notebooks/02_vjepa2_zeroshot.ipynb` extracts V-JEPA-2 ViT-L
features on raw RME mp4 clips (16 evenly-spaced frames, mean-pool over
patch tokens). Cached at `outputs/vjepa2_embeddings.npz` (283 × 1024).
Sanity 5's `ssv2_motion` config uses these features → LOSO 0.8746.

The custom-trained YOLOv8l ear detector
(`vendor/horse-face-ear-detection/horse_ear_detection/yolov8l_horse_ear_detection.pt`)
exists and is used in `notebooks/01_read_my_ears_replicate.ipynb` to
reproduce the **paper's** original optical-flow pipeline. It is NOT
used in the project's V-JEPA-2 LOSO 0.875 reproduction.

**Implication for Phase 8b**: the actual comparator for the
"cross-behavior generalization" test is **whole-frame V-JEPA-2 LOSO
0.875**, not "custom-YOLO ear-cropped V-JEPA-2." Earlier discussion
(my own Phase 7 audit doc framing + Phase 8a pre-reg hash chain entry)
carried this error. The Phase 7 audit doc is already locked-and-merged;
the framing-correction is recorded here in Phase 8b pre-reg so the
audit chain has the corrected reference for any future reader.

**The numeric gates remain UNCHANGED per locked goal-shifting-warning**
(≥0.80 strong / 0.65-0.80 modest / <0.65 fails), but their
interpretation shifts from "compete with custom-trained YOLO" to
"preserve whole-frame V-JEPA-2 LOSO 0.875 under DLC ear-keypoint
cropping." This is **harder** than competing with custom-YOLO would
have been, because whole-frame V-JEPA-2 has access to the entire scene
structure (not just the ROI).

User decision per Phase 8a sequence: gates stay locked. The framing
correction does not retroactively change Phase 8a's verdict (which
fired on bootstrap CI on Δ AUC for Phase 5 vs Phase 7, both of which
ARE eye-cropped pipelines — that comparison remains valid).

---

## What 8b tests

**Primary question**: does V-JEPA-2 + DLC ear-keypoint-anchored crop
preserve the discriminative signal that V-JEPA-2 + whole-frame extracts
from RME on the action-vs-background ear-movement task?

If yes (≥0.80 AUC), the DLC cropping methodology generalizes from eye
to ear without breaking the signal. The cross-behavior generalization
claim is: "DLC keypoint-anchored cropping is a viable preprocessing
strategy that preserves discriminative information across behaviors."

If no (<0.65), DLC ear cropping breaks the RME signal — either the ear
keypoints aren't precise enough, the bbox geometry doesn't capture the
right region, or whole-frame context is necessary. Different
implications depending on test diagnostics.

If middle (0.65-0.80), DLC ear cropping LOSES information vs whole-frame
but isn't catastrophic — methodology partially generalizes, with
behavior-specific tradeoffs.

**Secondary questions** (reportable, not gated):
- Bilateral-info preservation: does both-ears bbox capture asymmetric-
  ear behaviors? RHpE Action 1 is fundamentally bilateral.
- DLC ear keypoint reliability: are the 4 ear keypoints (right_earbase,
  right_earend, left_earbase, left_earend) confident and stable across
  RME frames?
- Per-source generalization: does the cross-behavior result hold across
  all 12 sources, or concentrated?

---

## What 8b is NOT

- Not a re-run of the paper's optical-flow ear-detection pipeline. The
  comparator is the project's whole-frame V-JEPA-2 reproduction, not
  the paper's custom-YOLO + optical-flow.
- Not a custom-YOLO-vs-DLC test. The custom YOLO ear detector exists
  in vendor/ but isn't used by the project's LOSO 0.875.
- Not a Phase 7 retraction reversal. Phase 7's eye-region verdict
  remains UNDERPOWERED_INDISTINGUISHABLE.
- Not opportunistic re-tuning of Phase 7-era parameters. All locked
  Phase 7 elements (DLC version 3.0.0rc13, conf threshold 0.5, frame-
  failure X=25%/Y=50%, etc.) carry forward unchanged.

---

## Locked design decisions

### Decision 1 — Ear crop geometry: BOTH-EARS bbox encompassing all 4 keypoints

**Decision**: bbox encompasses all 4 ear keypoints (`right_earbase`,
`right_earend`, `left_earbase`, `left_earend`) that meet confidence
threshold 0.5, with margin 15% (matches Phase 5/7 convention) +
square-pad to 224×224.

**Rationale**:
- RHpE Action 1 ("asymmetrical or both ears back") is bilateral —
  signal from one ear alone is structurally insufficient for asymmetry
  detection.
- Both-ears bbox eliminates the side-assignment complexity from Phase 7
  §4 (no need to choose left vs right; we use both).
- Both-ears bbox is the natural anatomical scope of "ear region" and
  matches what a human annotator would draw if asked to crop "the ear
  area."
- Phase 7 single-eye precedent doesn't carry forward because eyes are
  typically singly-visible in profile (only one anatomical eye per
  shot); ears stick out of head structure on both sides and are
  bilaterally visible in most poses.

**Locked geometry**:
```
For each frame:
  confident_kps = {kp_idx in {right_earbase, right_earend,
                              left_earbase, left_earend}
                   : keypoints[kp_idx][confidence] >= 0.5}
  if len(confident_kps) >= 3:        # require ≥3 of 4 confident
    bbox = enclosing_rect(confident_kps)
    eye_w, eye_h = bbox dimensions
  else:                              # fallback
    bbox = single_middle_frame's bbox (across the clip)

apply margin 15%, square-pad, resize to 224×224
```

**Anti-pattern lock**: do NOT switch to single-ear bbox mid-phase
based on observed AUC. The both-ears choice is locked here for
empirical reasons (bilateral signal preservation); if 8b fails, the
failure mode reveals whether single-ear would work, but switching
mid-run violates pre-reg.

### Decision 2 — RME parity check (mandatory before any new compute)

**Locked rule**: Before running any DLC inference on RME, reproduce
the existing whole-frame V-JEPA-2 LOSO 0.8746 result bit-exact using:
- Cached features: `outputs/vjepa2_embeddings.npz` (283 × 1024)
- Canonical Ridge LOSO config: `RidgeClassifier(alpha=1.0,
  class_weight='balanced')` + `StandardScaler` per fold
- 12 LOSO folds (one per source)

**Gate**: Pooled AUC must equal 0.8746126936531734 (the value in
`outputs/iter65_sanity5_loso_rme_results.json`'s `best_config.loso_auc`)
to within numerical noise (≤1e-10 deviation).

**Failure handling**: If parity fails, halt Phase 8b. The 8b Δ-vs-
baseline comparison only makes sense if the baseline number is
reproducible. Same pattern as Phase 7's Wikimedia parity halt.

This is a **5-minute sanity check** with cached features; trivial cost,
non-trivial protection against silent infrastructure drift.

### Decision 3 — Stage 1.5 ear keypoint quality inspection

**Adapted from Phase 7's side-assignment Stage 1.5**: with both-ears
bbox geometry (Decision 1), side-assignment doesn't apply. Instead,
Stage 1.5 verifies **ear keypoint reliability on RME clips** before
locking the full Phase 8b crop pipeline.

**Locked procedure**:
1. Run DLC SuperAnimal-Quadruped on a stratified sample of ~10 RME
   clips (5 action + 5 background; sources spread across S1, S5, S8,
   S10, S12).
2. For each clip, inspect mean confidence of the 4 ear keypoints
   (right_earbase, right_earend, left_earbase, left_earend) across
   frames.
3. Compute fraction of frames per clip with **≥3 of 4 ear keypoints
   confident** (the geometry-fallback threshold from Decision 1).

**Gate**: ≥80% of frames in ≥80% of sample clips meet the ≥3-of-4
threshold. Phase 0 baseline on Wikimedia showed median eye keypoint
confidence ~0.86 with ~9% of frames below 0.5; ear keypoints should
behave similarly given they're trained from the same SuperAnimal
labels. If the ear-keypoint failure rate is substantially higher
than Phase 0's eye baseline (>2× ~9% = >18%), revisit Decision 1
geometry before locking.

**Time cost**: ~30 min DLC inference on 10 clips + 5 min analysis.
Cheap insurance.

**Reference-floor verification (computed pre-lock, locked here)**: To
ensure the ≥80%×80% threshold isn't too strict for video-realistic
data, the threshold is verified against Phase 5's existing manual
gold-standard subset (34 RME clips with V-JEPA-2 + linear probe LOSO
0.7985 manually validated). Phase 5 clips share the source population
of full RME and are a known-good reference.

**Result** (computed from `outputs/phase7_rme_dlc_keypoints.json`):
- 33/34 Phase 5 clips (**97.1%**) meet the ≥80% per-clip frame
  threshold with ≥3 of 4 ear keypoints confident
- Mean ear-keypoint confidences: right_earbase 0.86, right_earend 0.91,
  left_earbase 0.82, left_earend 0.92 — all comparable to Phase 0's
  eye-keypoint baseline (~0.85)
- One clip (`action_S3.mp4_2_`) sits just below at 77.8% (7/9 frames);
  all others at 100%

**Verdict**: threshold PASSES on Phase 5 reference data with margin.
The ≥80%×80% gate is locked as appropriately calibrated; not too
strict for video-realistic horse footage at RME's controlled-lab
quality.

### Decision 4 — Bg-masking: NO masking (matches Sanity 5 best config)

**Decision**: NO bg-masking applied to v4-ear crops; whole-frame V-JEPA-2
ingestion of the cropped mp4 directly.

**Rationale**: Sanity 5 results showed:
- `ssv2_motion` (no bg-masking): LOSO 0.8746 ← best
- `ssv2_bgmasked`: LOSO 0.7642 (-0.11)

Bg-masking HURTS RME LOSO. The conditional-bg-masking Lesson 9 result
(secondary motion in frame degrades cross-source robustness) applies to
DIFFERENT data — the project's iter-6.5 anchor data with multi-horse
scenes. RME's controlled lab clips are mostly single-horse with clean
backgrounds; bg-masking just removes useful contextual signal.

For Phase 8b: NO bg-masking. Match the best Sanity 5 config exactly.

If Phase 8b fails (<0.65), bg-masking sensitivity test could be a
Phase 9 escalation — but pre-registered as **deferred**, not part of
8b's locked design.

---

## Test hierarchy (different from Phase 8a)

Phase 8b has n=283 with 12 sources — paired DeLong is adequately
powered (unlike Phase 7's n=34). Test hierarchy:

| Test | Role | Threshold | Phase 8a comparison |
|---|---|---|---|
| **Test 1: AUC-vs-gate** | **Load-bearing** | ≥0.80 strong / 0.65-0.80 modest / <0.65 fails | Same gate structure, but binding |
| **Test 2: Paired DeLong vs whole-frame** | **Supportive significance** | p < 0.05 → significant non-equivalence | Phase 8a's was inconclusive at n=34; here it's expected to be informative |
| **Test 3: Subject-bootstrap CI on Δ** | Precision characterization | Reportable | Phase 8a's was load-bearing because n=34; here supplementary |
| **Test 4: Per-source AUC distribution** | Robustness | Reportable | Same as Phase 8a Test 2 |
| **Test 5: Stage 1.5 verification (Decision 3)** | Pre-compute gate | ≥80% × 80% threshold | Adapted from Phase 7 Stage 1.5 |

The load-bearing test is the AUC-vs-gate threshold. Paired DeLong and
bootstrap provide statistical interpretation; with adequate power at
n=283, both should be informative.

---

## Locked gates

| Gate | Threshold | Action |
|---|---|---|
| **G1 — Band membership** | Pooled AUC ≥ 0.80 | strong: methodology generalizes despite generic DLC training |
| **G2 — Mid band** | 0.65 ≤ AUC < 0.80 | modest: methodology generalizes but with information loss vs whole-frame |
| **G3 — Failure** | AUC < 0.65 | DLC ear cropping breaks RME signal; eye-specific narrative |
| **G4 — Significance (supportive)** | Paired DeLong p < 0.05 vs whole-frame | informational; supports interpretation regardless of band |

**Goal-shifting anti-pattern lock**: gates ≥0.80/0.65-0.80/<0.65 are
locked here. Do NOT recalibrate retroactively based on observed AUC.

### Verdict-reporting protocol (LOCKED)

The locked gates work in absolute terms (G1-G3 use absolute AUC). The
RME baseline (whole-frame V-JEPA-2 LOSO 0.875) creates a verdict-
clarity issue under the corrected framing: the "strong" label (G1,
AUC ≥ 0.80) could fire on an outcome where Δ vs whole-frame is
meaningfully negative. To prevent the absolute-AUC verdict from
silently obscuring the comparison-to-baseline finding, all 8b verdict
reporting in the audit doc and downstream artifacts MUST surface both
G1-G3 (absolute) AND G4 (paired DeLong vs whole-frame) jointly.

**Locked outcome-reporting matrix** (all rows must appear in audit
doc verdict section, regardless of which fires):

| G1-G3 (absolute) | G4 (paired DeLong vs whole-frame) | Joint reading |
|---|---|---|
| Strong (≥0.80) | Positive significant (DLC > whole-frame, p < 0.05) | **DLC ear cropping outperforms whole-frame baseline** (best case) |
| Strong (≥0.80) | Inconclusive (p ≥ 0.05) | DLC cropping competitive but indistinguishable from whole-frame at this n |
| Strong (≥0.80) | Negative significant (DLC < whole-frame, p < 0.05) | **DLC cropping is competitive in absolute terms but degrades whole-frame baseline** — locked-gate-passes does NOT mean methodology-improves-baseline. Worth flagging. |
| Modest (0.65-0.80) | Positive significant | (unlikely combination — flag if observed) |
| Modest (0.65-0.80) | Inconclusive | DLC cropping loses information vs whole-frame, possibly noise |
| Modest (0.65-0.80) | Negative significant | DLC cropping clearly degrades whole-frame; methodology partial |
| Fails (<0.65) | (any G4) | Cross-behavior generalization fails regardless of comparison |

This matrix is **the verdict-reporting protocol**, not a re-locked gate
matrix. G1-G3 fires the methodology-band verdict (strong / modest /
fails); G4 fires the comparison-to-baseline verdict (positive sig /
inconclusive / negative sig). Both verdicts are reported jointly per
the matrix; the joint reading is the audit doc's substantive finding.

The locked anti-pattern: do NOT report G1-G3 verdict alone if G4
contradicts it. The "strong absolute AUC + significantly worse than
whole-frame" cell is a real possibility under the corrected framing
and must not be silently celebrated as a methodology-success.

---

## Pre-registered failure-mode routing

Per Phase 8a Stage 1's locked Phase 9 routing matrix (informed by
both 8a and 8b outcomes):

| 8b outcome | Phase 9 priority |
|---|---|
| AUC ≥ 0.80 (strong) | Multi-rater κ on Phase 5 eye labels (when 2nd rater available); cross-behavior generalization established despite Phase 7 retraction |
| 0.65 ≤ AUC < 0.80 (modest) | N expansion + multi-rater κ; methodology generalizes with caveats |
| AUC < 0.65 (fails) | Eye-specific narrow narrative; methodology doesn't generalize without behavior-specific validation |

If 8b lands strong, the project narrative is: "Cross-behavior
generalization established at n=283 with adequate power, despite
Phase 7's eye-specific verdict being retracted as underpowered at
n=34. Methodology generalizes; eye-specific superiority claim
specifically does not hold at small N."

---

## Diagnostic instrumentation (reportable, not gated)

1. **Per-source AUC** (12 sources × LOSO holdout AUC). Tell us whether
   8b's verdict is concentrated or distributed.
2. **DLC ear keypoint confidence distribution** across all 283 RME clips
   (per-clip means + pooled). Tells us whether the Stage 1.5 sample
   generalizes to full data.
3. **Per-clip score table**: clip × source × phase8b_score × paper_label.
   Identifies outlier clips for Phase 9 inspection.
4. **Frame-level statistics per clip** (matches Phase 7 manifest format):
   detection rate, fallback-mode incidence (per-frame v3-ear-bbox /
   single-middle-frame fallback / drop), mean confidence.
5. **Geometry-fallback rate**: fraction of frames using single-middle-
   frame fallback per Decision 1's gate (<3 of 4 ear keypoints
   confident). High rate → ear keypoints unreliable on RME, would
   warrant Decision 1 revision in a future phase.

---

## Anti-patterns (LOCKED)

1. **No retroactive gate adjustment.** Gates ≥0.80/0.65-0.80/<0.65
   are locked. Don't tighten or loosen based on observed AUC.
2. **No mid-phase geometry switch** (single-ear vs both-ears). If
   both-ears bbox fails, that's the finding — don't switch to single-ear
   to "rescue" the result.
3. **No bg-masking opt-in** based on result. Bg-masking off is the
   locked choice; turning it on would violate Decision 4 lock.
4. **No paired-DeLong override of AUC-vs-gate.** If paired DeLong p ≥
   0.05 (no significant difference vs whole-frame) but AUC ≥ 0.80, the
   AUC gate wins (strong verdict). G4 is supportive, not load-bearing.
5. **No retraction of Phase 8a routing.** Phase 8b's outcome doesn't
   reverse Phase 7's UNDERPOWERED_INDISTINGUISHABLE retraction. Phase
   7 stays retracted; 8b is independent.
6. **No 4-keypoint-confident threshold adjustment** mid-phase. The
   ≥3-of-4 threshold from Decision 1 is locked.

---

## Sequencing

| Step | Action | Output |
|---|---|---|
| 0 | User approval of this Stage 1 pre-reg | hash-locked artifact |
| 1 | RME parity check (Decision 2) — reproduce LOSO 0.8746 with cached features | parity result; halt if fails |
| 2 | Stage 1.5 ear keypoint inspection (Decision 3) on ~10 RME clips | inspection report; halt if <80%×80% |
| 3 | Build `tools/phase8b_dlc_ear_crop.py` (per-clip ear bbox from 4 keypoints + locked geometry) | tool source |
| 4 | DLC inference on 283 RME clips | `outputs/phase8b_rme_dlc_keypoints.json` |
| 5 | Crop pipeline with locked geometry (Decision 1) | `outputs/eye_crops_v4_ear_dlc/<clip>.mp4` + manifest |
| 6 | V-JEPA-2 extraction with parity check on at least one v4-ear clip | `outputs/vjepa2_embeddings_ear_v4.npz` |
| 7 | LOSO via canonical run_loso(...) | `outputs/eye_loso_results_phase8b.json` |
| 8 | Diagnostic + paired DeLong vs whole-frame baseline | `outputs/phase8b_diagnostic.{json,md}` |
| 9 | Audit doc draft `docs/phase8b_audit.md` | doc |
| 10 | User-approval checkpoint #5 (audit doc lock + Phase 9 entry decision) | — |
| 11 | Hash chain + commit + subtree-push + merge to main | mirror sync |

**User-approval checkpoints (5, matching Phase 7 density)**:

1. After this Stage 1 doc approval, before any 8b compute.
2. After RME parity check (Step 1), before Stage 1.5 inspection.
3. After Stage 1.5 ear keypoint inspection (Step 2), before locking
   the full crop pipeline.
4. After LOSO + diagnostic (Step 8), before audit doc lock.
5. After audit doc draft (Step 9), before final hash-lock and Phase 9
   entry decision.

---

## Phase 9 entry conditions (from Phase 8b outcome, per Phase 8a routing)

Per Phase 8a Stage 1 Phase 9 routing matrix (8a × 8b → priority):

| 8a outcome | 8b outcome | Phase 9 priority (locked) |
|---|---|---|
| RETRACTION fired (Phase 8a outcome) | strong (≥0.80) | Multi-rater κ when 2nd rater available; methodology generalization established |
| RETRACTION fired | modest (0.65-0.80) | N expansion + multi-rater κ; methodology generalizes with information-loss caveat |
| RETRACTION fired | fails (<0.65) | Eye-specific narrow narrative; methodology doesn't generalize without behavior-specific validation; halt forward DLC-cropping work |

Phase 8a fired RETRACTION; therefore the priority depends on Phase 8b
outcome alone among these three rows.

---

## Cost / time estimate

| Step | Estimate |
|---|---:|
| Pre-reg approval cycle | ~30 min |
| RME parity check (Step 1) | ~5 min compute + ~10 min review |
| Stage 1.5 ear inspection (Step 2) | ~30 min DLC + ~15 min review |
| Crop tool implementation (Step 3) | ~60 min |
| DLC inference on 283 RME clips (Step 4) | ~2-3 h compute |
| Crop pipeline (Step 5) | ~15 min |
| V-JEPA-2 extraction (Step 6) | ~15 min |
| LOSO + paired DeLong (Step 7-8) | ~10 min compute + ~30 min analysis |
| Audit doc draft (Step 9) | ~60 min |
| Hash chain + commit + push (Step 11) | ~30 min |
| **Total wall-clock** | **~5-6 hours over 1-2 days** |

---

## User approval signature

User has reviewed and approves Phase 8b Stage 1 lock as drafted, including:

- Framing-correction note on RME baseline (whole-frame V-JEPA-2, not
  custom-YOLO) surfaced prominently
- Decision 1: BOTH-EARS bbox encompassing 4 keypoints with ≥3-of-4
  confidence gate, m=15% margin, 224×224 square-pad
- Decision 2: RME parity check before any new compute (halt if fails)
- Decision 3: Stage 1.5 ear keypoint quality inspection (≥80% × 80%
  threshold)
- Decision 4: NO bg-masking (matches Sanity 5 best config)
- Test hierarchy: AUC-vs-gate (load-bearing), paired DeLong vs
  whole-frame (supportive significance), bootstrap (precision)
- Gates UNCHANGED at ≥0.80/0.65-0.80/<0.65 per goal-shifting warning
- 6 anti-patterns
- 5 user-approval checkpoints
- Phase 9 entry routing locked per Phase 8a Stage 1's matrix

User signs off → CC executes Step 1 (RME parity check) → user
approves at checkpoint #2 → Stage 1.5 inspection → user approves at
checkpoint #3 → Steps 3-9 proceed → user approves at checkpoints #4
and #5 → final commit + Phase 9 entry decision.
