# Phase 6 audit — prediction-shift quantification + face-bbox cropping result

**Date**: 2026-05-09. Branch: `experiment/eye-probe`. Two pre-registered
Phase 6 lanes ran in this audit cycle. Both verdicts derived mechanically
from locked decision rules; no post-hoc interpretation.

## Summary

| Lane | Pre-reg | Result | Verdict |
|---|---|---|---|
| (a) per-clip Phase 3 vs Phase 5 prediction diff | `phase5_audit.md` §"Pre-registered Phase 6 instrumentation" | 9 V3_NEWLY_RECOVERED, 5 V3_NEWLY_LOST, ratio 1.80, mixed regime; robust to label-tightening (Set B/C identical aggregates, 13/14 directional clips identical with single Sensitivity-1-mechanism swap) | Mixed regime confirmed; quantitative basis for (b) gate construction |
| (b) face-bbox-positioned crop with locked anatomical position | `outputs/track_b_phase6b_preregistration.md` | AUC **0.4689** (Δ vs Phase 5 = **−0.330**, below chance), median IoU vs Phase 5 manual boxes **0.165** | **DISTRIBUTED_FAIL** → DLC SuperAnimal-Quadruped is next |

## Phase 6 (a) — recap

Per-clip prediction shift between Phase 3 (v1 crops, AUC 0.6813) and Phase
5 primary (v3 manual gold-standard crops, AUC 0.7985) on the same 34
clips:

| Category | n | Share |
|---|---:|---:|
| BOTH_RIGHT | 18 | 53% |
| BOTH_WRONG | 2 | 6% |
| V3_NEWLY_RECOVERED | 9 | 26% |
| V3_NEWLY_LOST | 5 | 15% |

Net +4, ratio 1.80. Mixed regime confirmed the DeLong-paired gap finding
from Phase 5 audit: v3's +0.117 AUC delta is decomposable into 9
recoveries minus 5 losses, not uniform improvement. Two non-exclusive
mechanisms identified for V3_NEWLY_LOST: (1) off-axis motion stripping
(4/5 newly-lost are ACTION clips that v1's wider scope captured but v3's
tight eye-region crop excluded); (2) catchlight motion confound at tight
crops (bg_S8.mp4_3_ specifically). Both mechanisms predicted similar (b)
outcomes but diverge on Phase 7+ direction. Full discussion and locked
mechanism framing in `phase5_audit.md` §"Phase 6 (a) result —
prediction-shift quantified".

## Phase 6 (b) — face-bbox-positioned crop

### Pipeline

Per locked pre-reg (hash `c8c00fb81e98...`):

- Per-frame YOLOv8l face detection (`yolov8l_horse_face_detection.pt`,
  conf=0.5, imgsz=640).
- Tight eye box derived from face bbox at locked anatomical position
  (rel_x=0.5060, rel_y=0.3926, rel_w=0.1419, rel_h=0.0596 — median from
  34-clip Phase 5 manual annotation).
- Margin 15%, square-pad, 224×224, V-JEPA-2 forward, LOSO via canonical
  `run_loso(...)` from `tools/eye_loso_lr_phase5.py`.
- Reference labels: Set B (`eye_verification_clips.txt`).

### Result

| Metric | Value |
|---|---|
| Pooled AUC | **0.4689** |
| Subject-bootstrap 95 % CI | [0.2600, 0.6856] |
| Δ vs Phase 5 primary | **−0.3297** |
| Δ vs Phase 3 (v1) | −0.2125 (paired DeLong p = 0.0742) |
| Permutation p vs chance | 0.5574 |
| Median IoU vs Phase 5 manual boxes | **0.1651** |
| Clips with IoU ≤ 0.30 (off-eye) | 23/34 |
| Clips with IoU ≥ 0.50 (on-eye) | 7/34 |

Pipeline executed cleanly: 34/34 clips processed, all in `per_frame`
detection mode, zero interpolation, zero single-middle-frame fallback,
zero clip drops. Mean face confidence 0.85–0.94 across clips. Locked
failure handling never triggered — the failure mechanism is
**positional, not detection-based**.

### Locked gate evaluation

- **G1 (AUC ≥ 0.70)**: FAIL (0.4689)
- **G2 (AUC ≥ 0.7485, paired non-inferiority)**: FAIL
- IoU proxy gate (median ≥ 0.6): FAIL (0.165)

### Per-clip categories vs Phase 5 primary

| Category | n | Share |
|---|---:|---:|
| BOTH_RIGHT | 11 | 32% |
| BOTH_WRONG | 3 | 9% |
| FBB_NEWLY_RECOVERED | 4 | 12% |
| **FBB_NEWLY_LOST** | **16** | **47%** |

Net **−12 clips** vs Phase 5 primary — face-bbox loses on nearly half
the dataset. Among the 16 FBB_NEWLY_LOST clips, 1 is in the
4-clip orientation-extreme set (bg_S5.mp4_10_); the other 15 are
non-OE clips.

### Failure-mode attribution (mechanically routed)

- `loss_concentration_pct = 1/16 = 6.2%`
- Below the locked 50% threshold → **DISTRIBUTED_FAIL**
- Per locked routing: **DLC SuperAnimal-Quadruped is the next tool**;
  orientation-aware face-bbox would NOT be sufficient because most of
  the failure is non-OE.

### Mechanism: positional miscalibration, not orientation

The locked anatomical position (rel_x=0.506, rel_w=0.142) was derived
as a median across 34 clips. The pre-registration flagged the wide
rel_x spread (IQR [0.40, 0.55], range [0.05, 0.70]) as a known
limitation. The empirical magnitude of the effect was understated:

- Median IoU 0.165 across all 34 clips means the face-bbox-derived
  crop is **mostly outside the actual eye region** even on
  non-orientation-extreme clips.
- 23/34 clips have IoU ≤ 0.30 — the "off-eye" threshold from the
  pre-reg.
- 0/4 orientation-extreme clips have IoU > 0.0 — locked anatomical
  position completely misses the eye on every OE clip.

**Why the median fails for non-OE clips too**: the IQR [0.40, 0.55]
seems tight in normalized units, but at typical face-bbox widths of
500-700 pixels, an IQR of 0.15 = 75-105 pixels of horizontal
variation. The eye box width itself (rel_w × face_w) is only ~85
pixels. So the eye position varies in pixel space by approximately
the same amount as the eye box width — meaning a single-position
median has near-zero IoU with each individual clip's eye location
unless that clip's rel_x happens to be very close to 0.506.

The "anatomically positioned crop" hypothesis assumes horse profile
orientation is bimodal (left-facing / right-facing) so a tight
distribution clusters around the median. The data show orientation is
effectively a continuous variable across clips — there's no median
that consistently locates the eye.

### action_S9.mp4_4_ — locked interpretation routing

Pre-named as the most informative single clip (V3_NEWLY_LOST + orientation-extreme + rel_x=0.257).

| Metric | Value |
|---|---|
| Phase 5 score | −0.117 (wrong) |
| Phase 6 (b) score | +0.511 (correct) |
| IoU vs Phase 5 manual | **0.000** |
| Locked routing label | **CORRECT_OFF_EYE** |

The crop didn't land on the eye but the prediction is correct anyway.
Per locked routing: "face-bbox got it right but wider scope captures
whole-body cues even off-eye; suggestive but not conclusive." This
single-clip recovery is consistent with V-JEPA-2 + LR picking up
whole-body / face-region motion features that survive off-eye crops —
useful for some clips but unreliable as a general signal (the global
AUC 0.47 confirms the unreliability).

### Other suggestive sub-findings (not load-bearing)

- **bg_S8.mp4_3_** (the perceptual-floor / catchlight clip): IoU 0.271,
  Phase 5 score −0.940 (wrong), Phase 6 (b) score +0.164 (correct).
  Wider crop scope diluted the catchlight motion confound that hurt
  v3 — consistent with the audit's pre-registered catchlight-as-motion-
  confound hypothesis. Single-clip evidence; not generalizable from
  one observation.
- **action_S5.mp4_5_** (V3_NEWLY_LOST in (a)): also recovered by face-bbox
  (Phase 5 −0.413 → P6b +0.089). One of the off-axis-motion-hypothesis
  candidates. Suggestive, not conclusive.
- **Some clips with IoU ≥ 0.5 still failed**: action_S11.mp4_0_ (IoU
  0.618, BOTH_WRONG), bg_S7.mp4_9_ (IoU 0.579, FBB_NEWLY_LOST), bg_S7.mp4_17_
  (IoU 0.544, FBB_NEWLY_LOST). On-eye crops can still produce wrong
  predictions — V-JEPA-2 + LR isn't robust to crop tightness alone.

These are observations, not findings. The headline result is the
distributed positional failure.

## What Phase 6 establishes

1. **The (a) prediction-shift finding is robust to label-tightening.**
   Set B/C aggregate counts identical (9/5/+4/1.80), with 13/14
   directional clips agreeing. The single-clip swap is exactly the
   Sensitivity 1 structural-cancellation mechanism. Set A's collapse to
   7/7/+0 is a clean negative control confirming +0.117 comes from
   labeled-target alignment, not filename-taxonomy artifact.

2. **Median anatomical position from 34-clip data is insufficient for
   automated cropping.** Face-bbox at locked position fails because
   horse orientation creates rel_x variation comparable to the eye-box
   width itself. No median position preserves eye location across the
   continuum of profile/frontal poses.

3. **Pipeline robustness is not the bottleneck.** YOLOv8l face detection
   at conf=0.5 succeeded on 34/34 clips at every native frame, mean
   confidence 0.85–0.94, zero interpolation needed. The face detector
   works; the position-from-face-bbox derivation doesn't.

4. **DISTRIBUTED_FAIL routing fires cleanly.** The pre-registration's
   `loss_concentration_pct < 50%` rule mechanically chose DLC over
   orientation-aware face-bbox without any post-hoc interpretation. The
   discipline pattern held.

## What Phase 6 does NOT establish

- DLC's actual performance. (b)'s failure routes to DLC; whether DLC
  succeeds is a separate experiment.
- Whether orientation-aware face-bbox would have succeeded if loss
  concentration had been > 50%. The DISTRIBUTED_FAIL routing skips
  this lane.
- Whether V-JEPA-2 + LR carries eye-region signal at all on
  automated crops, or whether ALL automated cropping at this dataset
  scale degrades the signal. (b) tested one specific automated
  approach; n=34 is too small to draw architectural conclusions.

## Implications for collaborator pitch

1. **Phase 5's manual annotation effort was paid for once and now
   has a third life as a validation set.** First life: cropping
   intervention for the architecture-ceiling test. Second life: IoU
   validation set for any automated tool. Third life (post-Phase 6):
   the 16 FBB_NEWLY_LOST clips are a per-clip test bed showing exactly
   which clips need ROI accuracy (most of them) vs which are robust to
   off-eye crops (4-7 clips).

2. **The "anatomically positioned crop" hypothesis was a defensible but
   ultimately empirically failed first-tier test.** It was the cheapest
   experiment to run (existing vendor face detector, ~30 min code, ~30
   min run), and its failure routes to DLC with no ambiguity. The
   discipline of running the cheap test first paid off in time spent
   on DLC setup not being wasted on speculation.

3. **DLC is the next tool.** Honest half-day setup — not "already in
   stack from Phase 0" as initially proposed; the project has a single
   sample output file from an experimental DLC run, not integrated
   pipeline infrastructure. Phase 7's first task is bringing up DLC
   SuperAnimal-Quadruped properly: model load, inference on the 34
   Phase 5 clips, eye-keypoint-to-bbox geometry, evaluation against
   the same Phase 5 manual boxes used to derive Phase 6 (b)'s position
   parameter.

4. **Custom YOLO eye detector is deferred until Phase 7 reports.** The
   pre-registered failure-mode routing only escalates to custom YOLO
   if DLC also fails or if N≥200 scope is committed.

## Pre-registration discipline (audit chain extension)

Phase 6 (a) artifacts (frozen 2026-05-09):

| File | SHA-256 |
|---|---|
| `tools/phase6_per_clip_diff.py` | `b735e53d9579...` |
| `outputs/phase6a_per_clip_prediction_diff.json` | `303a09eedf0d...` |
| `outputs/phase6a_per_clip_prediction_diff.md` | `e8c36cfa5b3e...` |

Phase 6 (b) pre-registration + position parameter (frozen BEFORE the
crop run):

| File | SHA-256 |
|---|---|
| `tools/phase6b_derive_position.py` | `2fd724a59292...` |
| `outputs/phase6b_position_param.json` | `d1521a35c2de...` |
| `outputs/track_b_phase6b_preregistration.md` | `c8c00fb81e98...` |

Phase 6 (b) result + diagnostic (frozen at run completion):

- `tools/phase6b_face_bbox_crop.py` — per-frame YOLO face → anatomical
  eye box → margin → V-JEPA-2 ingestion
- `tools/eye_loso_lr_phase6b.py` — focused runner calling canonical
  `run_loso(...)` from Phase 5 module
- `tools/phase6b_diagnostic.py` — per-clip diff + IoU + locked routing
- `outputs/eye_crops_phase6b_m15_manifest.jsonl` — frame-level YOLO
  statistics per clip
- `outputs/eye_loso_results_phase6b.json` — LOSO output (AUC 0.4689,
  paired DeLong vs Phase 3, factor-d, permutation)
- `outputs/phase6b_diagnostic.json` + `phase6b_diagnostic.md` —
  per-clip diff vs Phase 5 + IoU + verdict + routing

The hash chain is the witness that the routing decision could not have
been retroactively constructed: the failure-mode attribution rule was
locked in `track_b_phase6b_preregistration.md` (hash `c8c00fb81e98...`)
**before** the (b) crop run, and the diagnostic tool implements the
locked rule mechanically.
