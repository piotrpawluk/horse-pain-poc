# Track B Phase 1 — Pre-registration (eye-crop pipeline + v2 fallback)

**Frozen:** 2026-05-08, BEFORE running the eye-crop pipeline on the 36-clip RME subset.

**Authority:** User review of Track B Phase 1 proposal, 2026-05-08. User explicitly required pre-committing the fix to apply if Phase 3 LOSO AUC lands in the ambiguous 0.55–0.65 zone — to prevent goal-shifting after results land.

## What's locked in v1

- **Crop heuristic v1:** upper 40 % vertical (slightly more than naive 1/3 for safety margin) × full bbox width; square-padded around the bbox center; resized to 224×224 (V-JEPA-2 ViT-L native input).
- **Detection cadence:** single YOLO face detection on the middle frame of each clip; drift check on first/middle/last frame; per-frame detection only for clips where center drift > 10 % of bbox width.
- **Failure handling:** YOLO returns 0 detections → skip clip + log to `outputs/eye_crops_failed.txt`; never fall back to whole-frame crop.
- **Contact sheet inspection:** all 36 clips reviewed visually before Phase 2; per-clip eye-visible Y/N annotated; clips where the eye is < 20 % of the crop area are excluded from downstream V-JEPA-2 extraction.
- **V-JEPA-2 parity test (Phase 2 gate):** before running V-JEPA-2 on eye crops, parity test re-extracts one ear clip via the lifted CLI script and asserts cosine similarity ≥ 0.999 against the cached `outputs/vjepa2_embeddings.npz`. Phase 2 cannot proceed if parity fails.

## Pre-committed fallback v2 (apply iff Phase 3 LOSO AUC ∈ [0.55, 0.65))

If the v1 pilot lands in the ambiguous zone:

- **v2 crop heuristic:** profile-aware. The upper 40 % strip is split into left half and right half. For each half, a 32-pixel-radius ROI sweep computes mean spatial-frequency content (eye region has more high-frequency texture than mostly-flat forehead/cheek). The half with higher mean spatial-frequency wins. Tie-break (within 5 %): output BOTH halves as separate samples (doubles per-clip data).
- **Crop output for v2:** square-padded around the chosen half's center, resized to 224×224.
- **Single re-evaluation:** one LOSO run on v2 crops. **No further crop iterations.**

### v2 decision rule (locked)

| v2 LOSO AUC | Conclusion | Action |
|---|---|---|
| **≥ 0.65** | Eye signal present; v1 crop quality was the blocker | Adopt v2 as the production heuristic; continue Track B writeup |
| **0.55–0.65** | Heuristic isn't the bottleneck. Either labels are too noisy at single-observer N=36, or eye signal genuinely weaker than ear at this scale | Stop crop iteration. Either commission multi-rater eye labels OR expand scope to 100+ clips before any further claims |
| **< 0.55** | Eye signal is below the V-JEPA-2 + LR threshold at this label/N regardless of crop quality | Document as Lesson 19; conclude V-JEPA-2 + LR is ear-track-validated but eye-track-failed at the audit-followup label scale |

## Anti-patterns this pre-registration prevents

1. **"Let me try one more crop heuristic"** after v2 fails — explicitly forbidden. Only v1 → v2 → stop. Heuristic engineering is bounded at two iterations to prevent the same prompt-engineering trap that closed the MLLM track.
2. **"V2 looks better, let me try a v3"** in the success path — also forbidden. V2 becomes the production heuristic if it clears 0.65; no further crop work.
3. **Per-frame detection retrofitted on all clips** if static-detection seems weaker — only applies on the drift-flagged subset (drift > 10 %).
4. **Whole-frame fallback** when YOLO fails — explicitly forbidden. "Skip + log" only.
5. **Reading sub-0.55 LOSO AUC as "V-JEPA-2 doesn't carry eye signal"** without checking class balance and N. With 36 clips and possibly few eye-action positives, AUC can be low for sample-composition reasons. The v2 decision rule explicitly accounts for this.

## What this pre-registration does NOT cover

- **Phase 1 → Phase 2 → Phase 3 sequencing.** Phase 1 = crop. Phase 2 = V-JEPA-2 extraction. Phase 3 = LOSO LR (waits for user's morning eye labels). Each phase has its own gates documented inline in the relevant code/docs.
- **Eye-blink-only vs general eye-region.** The user's morning labels will define the operational target. v1 crop is general eye-region; if labels are blink-only, crop scope unchanged but downstream interpretation differs.
- **Comparison with the prior MLLM runs.** Lesson 18 closed that question; comparing V-JEPA-2 + LR LOSO AUC against MiniCPM 1/6 sanity is the wrong frame and not part of this pre-registration.

## Audit trail

- This file committed to git BEFORE the v1 pipeline runs.
- v2 spec is locked at the same moment as v1; not added later.
- v3 does not exist as a path. If v2 fails to clear 0.65, the conclusion is one of the two scope-related branches above, not "try harder on cropping."
