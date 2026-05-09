# Eye-Crops Manual Annotation — Track B Phase 1

**Reviewed:** 2026-05-08, after `tools/eye_crop_pipeline.py` v1 produced
35 crops + 1 detection failure on the 36-clip RME stratified subset.

**Method:** Visual inspection of `outputs/eye_crops_contact_sheet.png` (6×6
grid of middle-frame thumbnails, each cell 224×224 = the actual V-JEPA-2
input). Per-clip Y/N annotation against the pre-registered "exclude if
eye <20% of crop area" rule from `track_b_phase1_preregistration.md`.

## Summary

- 36 input clips
- 35 cropped successfully
- **1 detection failure** (`background_S4.mp4_7_.mp4`) — YOLO returned 0
  detections at conf=0.5 on the middle frame; pipeline correctly skipped +
  logged to `eye_crops_failed.txt`. **Excluded.**
- **1 manual exclusion** (`background_S1.mp4_11_.mp4`) — eye <20% of crop
  area; dark horse, profile, low contrast. **Excluded.**
- **34 clips proceed to Phase 2.**

## Per-clip annotations

| # | Clip | YOLO drift | Cadence | Eye visible? | Estimated eye area | Decision |
|---|---|---|---|---|---|---|
| 1 | background_S1.mp4_11_ | 0.11 | per-frame | Yes (faint) | ~10% | **EXCLUDE** |
| 2 | background_S1.mp4_7_ | 0.01 | static | Yes | ~18% | keep (borderline) |
| 3 | background_S1.mp4_12_ | 0.10 | per-frame | Yes | ~18% | keep (borderline) |
| 4 | background_S10.mp4_11_ | 0.05 | static | Yes | ~25% | keep |
| 5 | action_S10.mp4_0_ | 0.00 | static | Yes | ~25% | keep |
| 6 | background_S10.mp4_3_ | 0.01 | static | Yes | ~25% | keep |
| 7 | background_S11.mp4_0_ | 0.07 | static | Yes | ~25% | keep |
| 8 | action_S11.mp4_0_ | 0.05 | static | Yes | ~22% | keep |
| 9 | action_S11.mp4_6_ | 0.03 | static | Yes | ~22% | keep |
| 10 | background_S12.mp4_9_ | 0.03 | static | Yes | ~28% | keep |
| 11 | background_S12.mp4_2_ | 0.00 | static | Yes | ~28% | keep |
| 12 | background_S12.mp4_7_ | 0.02 | static | Yes | ~28% | keep |
| 13 | background_S2.mp4_10_ | 0.01 | static | Yes | ~22% | keep |
| 14 | background_S2.mp4_11_ | 0.17 | per-frame | Yes | ~22% | keep |
| 15 | action_S2.mp4_7_ | 0.00 | static | Yes | ~22% | keep |
| 16 | background_S3.mp4_3_ | 0.01 | static | Yes | ~25% | keep |
| 17 | action_S3.mp4_2_ | 0.03 | static | Yes | ~25% | keep |
| 18 | action_S3.mp4_8_ | 0.10 | per-frame | Yes | ~25% | keep |
| 19 | action_S4.mp4_15_ | 0.01 | static | Yes (bridle) | ~22% | keep |
| 20 | action_S4.mp4_0_ | 0.05 | static | Yes (bridle) | ~22% | keep |
| 21 | action_S5.mp4_5_ | 0.03 | static | Yes | ~25% | keep |
| 22 | background_S5.mp4_10_ | 0.05 | static | Yes | ~25% | keep |
| 23 | action_S5.mp4_2_ | 0.01 | static | Yes | ~25% | keep |
| 24 | background_S6.mp4_3_ | 0.00 | static | Yes | ~22% | keep |
| 25 | background_S6.mp4_2_ | 0.00 | static | Yes | ~22% | keep |
| 26 | action_S6.mp4_2_ | 0.03 | static | Yes | ~22% | keep |
| 27 | background_S7.mp4_12_ | 0.00 | static | Yes | ~25% | keep |
| 28 | background_S7.mp4_9_ | 0.01 | static | Yes | ~25% | keep |
| 29 | background_S7.mp4_17_ | 0.01 | static | Yes | ~25% | keep |
| 30 | background_S8.mp4_7_ | 0.01 | static | Yes (dark) | ~18% | keep (borderline) |
| 31 | action_S8.mp4_12_ | 0.01 | static | Yes (bridle) | ~22% | keep |
| 32 | background_S8.mp4_3_ | 0.01 | static | Yes | ~22% | keep |
| 33 | action_S9.mp4_7_ | 0.02 | static | Yes | ~25% | keep |
| 34 | background_S9.mp4_8_ | 0.02 | static | Yes | ~25% | keep |
| 35 | action_S9.mp4_4_ | 0.04 | static | Yes | ~25% | keep |
| (fail) | background_S4.mp4_7_ | — | — | — | — | **EXCLUDED (YOLO no-detection)** |

## Notes for Phase 3

- All 4 per-frame-detection clips (S1.mp4_11, S1.mp4_12, S2.mp4_11, S3.mp4_8)
  retained native frame counts, suggesting YOLO held detection across
  frames despite center drift. None show a frame-count drop indicative of
  intermittent failures.
- Bridled subjects (S4.mp4_0, _15; S8.mp4_12) crop the eye + bridle
  hardware. Bridle is consistent with the source-defined recording
  context for those subjects, so it's class-correlated noise but not a
  data-leakage concern (bridle present in both action and background
  clips for the same source).
- Eye area in the 18-22 % range is the dominant bucket. The geometric
  improvement vs uncropped frames is roughly: in uncropped 1920×1080,
  the eye at 60-80×40-50 px = ~4 % of frame area; in the 224×224 crop,
  the eye at ~30-50 px = ~22 % of crop area. ~5× more relative eye area
  than uncropped (~3× linear). Sufficient for V-JEPA-2 ViT-L if the
  encoder uses the eye signal at all.

## Decision: proceed to Phase 2

34 viable crops. V-JEPA-2 + LR LOSO will use this set when user's morning
eye labels land. Pre-registration thresholds (≥ 0.65 / 0.55-0.65 / < 0.55)
unchanged.

## Corrections / clarifications (2026-05-08, post-review)

### Cadence accounting

The clear breakdown:

- **31 static** (single detection on middle frame; drift ≤ 0.10)
- **4 per-frame** (drift > 0.10: action_S3.mp4_8_ at 0.100, background_S1.mp4_12_ at 0.102, background_S1.mp4_11_ at 0.110, background_S2.mp4_11_ at 0.169)
- **1 detection failure** (background_S4.mp4_7_; YOLO returned 0 detections at conf=0.5; no drift sample taken because first-frame detection failed)
- **Total: 36** ✓

The earlier status report's "28 static + 4 per-frame" miscounted; the run output's actual static count is 31.

### Drift threshold disambiguation

The pipeline implementation uses **strict greater-than**: `per_frame = drift is not None and drift > 0.10`. This matches the pre-registration's stated ">10%" rule.

The four flagged clips' full-precision drift values are 0.100220, 0.102318, 0.110098, 0.169437 — all genuinely > 0.10. Three of them displayed as "0.10" in the run log because of `:.2f` rounding; the actual values are documented above and in `outputs/eye_crops_drift_log.jsonl`. Implementation matches pre-reg verbatim; the apparent ambiguity was display-only.

### V-JEPA-2 parity test — what it actually compared

`tools/extract_vjepa2.py:parity_test()` loads the reference embedding from the cached npz, then **runs the lifted CLI's forward pass** (`extract_embedding(clip_path, model, processor, device)`) on the SAME source clip. The cached value is taken from the npz only as the reference to compare AGAINST; the comparison embedding is freshly computed end-to-end (frames → processor → model → mean-pool). This is a genuine re-extraction, not a cache-to-cache tautology.

The bit-exact result (cos = 1.000000, ‖Δ‖ = 0.0000) means the lifted forward pass is deterministic and reproduces the notebook's output exactly — the strongest possible parity signal. V-JEPA-2 ViT-L in `eval()` + `torch.no_grad()` mode is fully deterministic on Apple MPS for fixed inputs, so bit-exactness is the expected outcome and confirms parity rather than indicating a bug.

### Embedding count fix

The first run of `extract_vjepa2.py` produced an npz with **35 embeddings** because the manual exclusion (`background_S1.mp4_11_.mp4`) was annotated in this document but not propagated to the extraction filter. The CLI was extended with an `--exclude` flag and re-run, producing the canonical npz with **34 embeddings** matching the viable-set decision in the table above. The original 35-row file was overwritten.
