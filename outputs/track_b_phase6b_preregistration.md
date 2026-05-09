# Track B Phase 6 (b) — face-bbox-positioned crop pipeline pre-registration

**Frozen 2026-05-09 BEFORE the (b) crop run.** Locks all parameters,
gates, failure-mode interpretation, and diagnostic reporting before any
LOSO observation. Hash chain extension follows the Phase 5 pre-reg
precedent (`outputs/track_b_phase5_preregistration.md`).

## What (b) tests

Given Phase 6 (a)'s mixed-regime finding (recovered/lost ratio 1.80, +0.117
AUC delta = 9 recoveries − 5 losses, two non-exclusive mechanisms — off-axis
motion stripping + catchlight motion confound), (b) tests whether
**anatomically-positioned crop derived from the existing YOLOv8l horse-face
detector preserves Phase 5 primary's AUC at scale.**

This is the cheapest-first lane in the Phase 6 cropping-automation track.
If (b) fails specifically on orientation-extreme clips while preserving
median-position clips, the next iteration is orientation-aware face-bbox.
If (b) fails uniformly, the next tool is DLC SuperAnimal-Quadruped eye
keypoints (acknowledged half-day setup, not "already in stack").

## Position parameter (locked, data-derived)

Computed by `tools/phase6b_derive_position.py` (frozen at hash
`2fd724a592921c79d8cec7b9da469ef71c8183f5d6deb8559403094d94977a21`) from
Phase 5 manual eye boxes (middle keyframe per clip) within face bboxes
detected on the corresponding middle native frames:

| Param | Median (LOCKED) | IQR | Min–Max |
|---|---:|---:|---:|
| rel_x | **0.5060** | [0.40, 0.55] | [0.05, 0.70] |
| rel_y | **0.3926** | [0.36, 0.41] | tight |
| rel_w | **0.1419** | [0.13, 0.15] | tight |
| rel_h | **0.0596** | [0.05, 0.06] | tight |

`rel_y/w/h` are anatomically tight (consistent across clips). `rel_x` is
bimodal due to horse profile orientation; the median is near-center but
the spread is wide. Four orientation-extreme clips with `rel_x` outside
the IQR are pre-named for diagnostic reporting (see §Orientation-extreme
clips).

Frozen artifact: `outputs/phase6b_position_param.json`
(hash `d1521a35c2de6ee4d12de6ddd0ce7f5fbae425cad16964490373b9810667b7de`).

## Pipeline parameters (locked)

| Parameter | Value | Justification |
|---|---|---|
| Face detector | `vendor/horse-face-ear-detection/horse_face_detection/yolov8l_horse_face_detection.pt` | Already in vendor; same model used in `infer_face_then_ear.py` reference script. No alternative considered to avoid model-shopping degree of freedom. |
| `conf` threshold | **0.5** | Matches vendor reference `infer_face_then_ear.py:50` (`face_results = model(frame, conf=0.5, imgsz=640)`). NOT YOLOv8 default (which is 0.25); this is the vendor-recommended value for this specific model. Locked to vendor's documented setting; not a free parameter. |
| `imgsz` | **640** | Same vendor reference value (`infer_face_then_ear.py:50`). |
| Eye-box position | (rel_x, rel_y, rel_w, rel_h) = **(0.5060, 0.3926, 0.1419, 0.0596)** | Median across 34 Phase 5 manual annotations (data-derived, hash-locked). |
| Margin | **m=15%** | Matches Phase 5 primary's locked margin. Not a free parameter; controlled to isolate the face-bbox-vs-manual-eye-box contrast. |
| Square-pad + resize | 224×224 with `cv2.INTER_AREA` | Matches `tools/eye_crop_pipeline_v3.py`. |
| FPS for V-JEPA-2 input | native fps (no resampling) | Matches v3 pipeline. |
| V-JEPA-2 model | `facebook/vjepa2-vitl-fpc16-256-ssv2` | Same as v3 / Phase 5 primary. |
| LOSO classifier | `RidgeClassifier(alpha=1.0, class_weight='balanced')` + `StandardScaler` per fold | Project canonical config. |
| Reference label set | **Set B** (`outputs/eye_verification_clips.txt`) | Matches Phase 5 primary's training labels; the AUCs being compared are computed against this set. |

## Frame-level YOLO failure handling (LOCKED RULE)

Per-frame face detection is more dynamic than Phase 5's 3-keyframe manual
+ interpolation. The locked policy for missing detections:

1. **Per-frame attempt**: run YOLOv8l on every native frame with
   `conf=0.5, imgsz=640`. If `len(results.boxes) > 0`, take the
   highest-confidence box.
2. **Single-frame failure (no detection at conf ≥ 0.5)**: interpolate
   the face bbox linearly between the nearest temporally-preceding
   detected frame and the nearest temporally-following detected frame.
   Edge cases:
   - Failure on the first frame: use the next detected frame's bbox.
   - Failure on the last frame: use the previous detected frame's bbox.
3. **Clip-level threshold**: if **> 25% of frames** in a clip fail
   detection at `conf ≥ 0.5`, fall back to **single-middle-frame face
   bbox applied across the entire clip** (matches the v3 STATIC-mode
   geometry but with face-bbox-derived eye position instead of manual
   middle box). The clip is logged in the manifest with
   `fallback_mode: "single_middle_frame"` and the per-frame failure
   count.
4. **Total clip failure** (no face detected on any frame at
   `conf ≥ 0.5`): clip is dropped from the LOSO run. The manifest
   records `status: "fail"` with reason `"no_face_detected_any_frame"`.
   Pre-committed: if any clip drops from this rule, the (b) verdict is
   computed on the surviving subset and the dropped clip is reported
   prominently. **Expected to be rare** given the position-parameter
   derivation succeeded on 34/34 clips at the same conf threshold.

The 25% threshold is locked at this round number rather than tuned
post-hoc. Rationale: at 25% the interpolation chain bridges no more than
~3-4 native frames (typical clip 13-18 frames; 25% = 3-4 frames). Beyond
that the temporal coherence of interpolated boxes degrades and a
single-bbox fallback is more honest than multi-frame interpolation
through a wide gap.

Each clip's manifest entry will record:
- `n_frames_native`, `n_frames_detected`, `n_frames_interpolated`,
  `n_frames_failed`
- `frame_failure_pct`
- `fallback_mode` ∈ {"per_frame", "interpolated", "single_middle_frame"}
- `mean_face_conf`

## Gates (BOTH required for PASS)

| Gate | Threshold | Rationale |
|---|---|---|
| **G1 — Band membership** | Pooled AUC ≥ **0.70** | Lower bound of project's pre-locked realistic LOSO band (per Phase 5 audit Lesson 11 anchor: 0.70–0.80 realistic; ≥0.85 unrealistic). Point-matching 0.7985 at n=34 is a noise game (Phase 5 CI [0.584, 0.964]). |
| **G2 — Non-inferiority vs v3** | Paired-DeLong p ≥ **0.05** vs Phase 5 primary, **AND** AUC ≥ Phase 5 AUC − 0.05 = **0.7485** | Same shape as Phase 5's gate against Phase 3. The 5pp protection floor catches silent slow erosion the paired test could miss at n=34 (DeLong-paired's SE inflates when the prediction-pairing is weak — see Phase 5 audit's DeLong-gap finding). |

## Failure-mode attribution routing (LOCKED)

If (b) does not clear both gates, the next-tool decision is **routed by
the structure of the failure**, not by aggregate AUC alone. This is
locked here so the next-experiment choice cannot be back-derived from the
data.

**Define orientation-extreme contribution**: among the 4 pre-named
orientation-extreme clips (§below), count the number whose `(b)`-vs-Phase 5
predicted-class flips from correct to wrong under Set B (i.e.,
FBB_NEWLY_LOST relative to Phase 5 primary). Call this `n_OE_loss`.

Define `loss_concentration_pct = n_OE_loss / total_FBB_NEWLY_LOST_count`
(if total losses are 0, undefined and the routing collapses to "PASS").

| (b) outcome | Verdict | Next tool |
|---|---|---|
| Both gates clear | PASS — viable Phase 6 scaling baseline | Lock for scaling track; defer DLC/YOLO unless N≥200 |
| AUC > 0.7985 AND paired p < 0.05 | OUTPERFORM v3 | Strong finding: tight eye crop was over-tightening; face-bbox is the new baseline |
| G1 passes, G2 fails | NON-INFERIOR FAIL | Per-clip diagnostic table determines next tool: if `loss_concentration_pct > 50%` → orientation-aware face-bbox (lighter iteration); else → DLC SuperAnimal-Quadruped |
| G1 fails AND `loss_concentration_pct > 50%` | ORIENTATION-DOMINATED FAIL | Orientation-aware face-bbox is next; DLC deferred. Light-tier escalation. |
| G1 fails AND `loss_concentration_pct < 50%` (non-orientation-extreme clips also failing) | DISTRIBUTED FAIL | DLC SuperAnimal-Quadruped is next; orientation-aware face-bbox would not be sufficient. Half-day-tier escalation. |
| G1 fails AND `loss_concentration_pct` is ambiguous (the 4 OE clips contribute exactly 50% within ±1 clip) | AMBIGUOUS | Run BOTH orientation-aware face-bbox AND DLC; let the empirical comparison disambiguate. |

This routes the next experiment based on the structure of the failure,
not the headline number. Locked here so the next-tool decision cannot
become post-hoc.

## Orientation-extreme clips (pre-named for diagnostic reporting)

The 4 clips with `|rel_x − 0.5| > 0.15` from the position-parameter
derivation, ordered by displacement:

| Clip | rel_x | Phase 6 (a) category (Set B) | Notes |
|---|---:|---|---|
| `background_S5.mp4_10_.mp4` | 0.052 | BOTH_RIGHT | Both pipelines correct in (a); useful as orientation-extreme control |
| `action_S9.mp4_7_.mp4` | 0.180 | BOTH_RIGHT | Both pipelines correct (margin shrunk under v3 from +1.891 to +0.165 — already at narrow margin) |
| `action_S9.mp4_4_.mp4` | 0.257 | **V3_NEWLY_LOST** | The most informative single clip in (b). Phase 3 score +0.919 (correct), v3 score −0.117 (wrong). If face-bbox crop also lands off-eye, the diagnostic must distinguish that from "crop on eye but tight crop loses off-axis motion signal." |
| `background_S1.mp4_12_.mp4` | 0.696 | BOTH_RIGHT | v3 strengthened margin from +0.454 to +1.315 |

For each of these 4 clips, the (b) audit will report:
- The actual face-bbox-derived eye-region crop coordinates per frame
  (the (x, y, w, h) at frame indices 0, mid, last).
- IoU between (b)'s tight eye box (face-bbox-derived) and Phase 5's
  manual middle keyframe box. If IoU < 0.30, the crop landed off the
  eye — orientation displacement is the failure mode. If IoU ≥ 0.50,
  the crop landed on the eye — failure (if any) is from the tight-crop
  signal-stripping mechanism, not localization.
- Did Phase 5 primary's prediction (correct or wrong) change under (b)?
  Specifically for `action_S9.mp4_4_`: Phase 3 was correct, Phase 5
  primary was wrong. If (b) is wrong → face-bbox doesn't help. If (b)
  is correct → face-bbox unintentionally addressed off-axis motion
  stripping (a positive finding for the cropping-loosening hypothesis).

**Locked interpretation routing for `action_S9.mp4_4_` specifically**:
- (b) wrong AND face-bbox IoU < 0.30 vs manual eye box → crop missed
  the eye; orientation displacement is the failure. Next tool:
  orientation-aware face-bbox.
- (b) wrong AND face-bbox IoU ≥ 0.50 vs manual eye box → crop on eye
  but signal still missing. Off-axis motion or catchlight is the
  mechanism; next test could be larger margin (m=80% from the
  pre-registered bimodal hypothesis) or DLC.
- (b) correct AND IoU ≥ 0.50 → face-bbox preserves v1's recovery
  on this clip. Strong positive finding for the off-axis-motion
  hypothesis (looser-than-v3 crop captures the same signal v1 had).
- (b) correct AND IoU < 0.30 → face-bbox got it right but for a
  different reason (e.g., wider scope captures whole-body cues even
  off-eye). Suggestive but not conclusive; flag and report.

## Diagnostic instrumentation (reported alongside, not gates)

- **Pooled AUC** + DeLong 95% CI + permutation p (vs chance) + paired
  DeLong vs Phase 5 primary (matches Phase 5 primary's instrumentation).
- **Subject-bootstrap CI** (B=10000) for distribution-free precision.
- **Per-clip category re-computation** with Phase 5 primary as reference
  (BOTH_RIGHT / FBB_NEWLY_RECOVERED / FBB_NEWLY_LOST / BOTH_WRONG
  against v3, not against Phase 3). Recovered count out of the 9
  V3_NEWLY_RECOVERED clips is the most informative single statistic.
- **Cross-stage diff** (a)→(b): of the 5 V3_NEWLY_LOST clips, how many
  does (b) recover? Each recovery is suggestive of which mechanism (off-axis
  motion → looser crop helps; or catchlight → tighter ROI doesn't help).
- **Per-clip IoU** between (b)'s face-bbox-derived eye box (middle
  frame) and Phase 5's manual middle keyframe box. Median (gate ≥ 0.6
  per project's locked threshold). Per-clip values for the 4
  orientation-extreme clips named explicitly.
- **Frame-level YOLO statistics** per clip: detection rate,
  fallback-mode incidence, mean face confidence.
- **Failure-mode attribution**: `loss_concentration_pct` reported
  numerically, routing decision indicated.

## Anti-patterns

1. **No mid-run parameter tuning.** All parameters locked above. Re-run
   only if the locked rule literally fails to execute (e.g., file
   format incompatibility), not if results are uninteresting.
2. **No alternative position parameters.** The (rel_x, rel_y, rel_w, rel_h)
   values are the median from data; no "let's try wider rel_w because the
   IQR is wide" mid-run adjustment. If face-bbox at locked position
   fails, the failure is the finding.
3. **No retroactive failure-mode definition.** The
   `loss_concentration_pct > 50%` rule is locked here; routing
   decisions follow it mechanically.
4. **No removing orientation-extreme clips from the LOSO.** They stay
   in; their behavior IS the diagnostic.
5. **No alternative `conf` thresholds.** 0.5 matches vendor reference
   and was used in position-parameter derivation. If many clips fail
   at conf=0.5 mid-run, the manifest records that and the verdict is
   computed on the surviving subset; we do NOT lower conf to 0.25 mid-run.

## Reproduction

```bash
cd poc
.venv/bin/python tools/phase6b_face_bbox_crop.py --margin-pct 15
.venv/bin/python tools/extract_vjepa2.py \
    --crops-dir outputs/eye_crops_phase6b_m15 \
    --out outputs/vjepa2_embeddings_phase6b_m15.npz
.venv/bin/python tools/eye_loso_lr_phase5.py \
    --mode primary \
    --embeddings outputs/vjepa2_embeddings_phase6b_m15.npz \
    --labels outputs/eye_verification_clips.txt \
    --out outputs/eye_loso_results_phase6b.json
.venv/bin/python tools/phase6b_diagnostic.py \
    --result outputs/eye_loso_results_phase6b.json \
    --reference outputs/eye_loso_results_phase5_primary.json \
    --crop-manifest outputs/eye_crops_phase6b_m15_manifest.jsonl \
    --position-param outputs/phase6b_position_param.json \
    --out outputs/phase6b_diagnostic.json
```

Final manifest binds: hash of this pre-registration → hash of crop
manifest → hash of embeddings → hash of LOSO result → hash of diagnostic.
Audit chain extension as Phase 5.
