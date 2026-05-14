# Phase 10 — Data quality notes (Prudnik unridden championship)

*Generated 2026-05-12T08:59:04.929386+00:00. n = 122 clips, total 8.73 GB. Reference: `docs/labeling-protocol-2026-05.md` (canonical labeling protocol); Lesson 21 (RME ↔ RHpE-#7 framing).*

## Pre-export filter

Source pool was reduced to **122 clips** post-filter; some clips were excluded by the user during export for distortion / messiness. This is a quality-first filter, not a random sample. Acceptable for transfer-test scope (Phase 10a tests whether the pipeline generalizes; it does not need a perfectly representative championship sample).

## 1. Orientation — load-bearing distribution shift vs RME (CRITICAL)

**All 122/122 clips are PORTRAIT orientation** (W < H, iPhone vertical recording). RME paper training data is almost certainly LANDSCAPE (horizontal horse profile, the dominant convention for clinical equine video).

**Why this matters:**
- V-JEPA-2 ViT-L resizes inputs to 224×224 — aspect ratio is squashed by resize but   the feature distribution may shift relative to RME training
- DLC SuperAnimal-Quadruped keypoint detector was trained on a mix of orientations but   may underperform on portrait horse profiles (long axis of horse perpendicular to   short axis of frame)
- The Phase 7-corrected anatomical-side mapping (Lesson 20) was derived on landscape   RME clips; portrait orientation may interact non-trivially with the `left_eye` ↔   `right_eye` convention

**Recommendation:** if Phase 10a transfer fails, **portrait orientation is the leading hypothesis** before reaching for backbone / classifier explanations. Phase 10a Stage 1 pre-reg should pre-register a 5-clip pre-flight (V-JEPA-2 + RME-trained probe on randomly chosen portrait Prudnik clips) before the full 122-clip transfer test.

## 2. Duration distribution — biased long vs RME

| Statistic | Value |
|---|---:|
| min | 0.15s |
| median | 17.57s |
| mean | 20.87s |
| max | 70.70s |
| std | 15.42s |

Bucketed against RME training reference (5–15s):

| Bucket | n | % | Note |
|---|--:|--:|---|
| < 5s | 12 | 9.8% | drop candidate (V-JEPA-2 frame floor); see §3 |
| 5–15s (RME match) | 42 | 34.4% | single-window; no split needed |
| 15–30s | 41 | 33.6% | splitting recommended; see splitting_recommendations.md |
| 30–60s | 23 | 18.9% | multi-window required |
| 60–120s | 4 | 3.3% | multi-window required; longest clip = 70.70s |

**56% of clips (68/122) > 15s** — the simplified-B1 long-form pipeline (Phase 9 scaffold) will be exercised on real multi-window data for the first time. Phase 9 Limitation 2 had deferred this; Phase 10 unblocks it.

## 3. Clips below V-JEPA-2 frame minimum

V-JEPA-2 ViT-L ingestion requires ≥16 evenly-sampled frames. At 60 fps that is 0.267s; at 30 fps that is 0.533s.

**1 clip(s)** below V-JEPA-2 frame minimum:

| clip_id | duration | fps | min_duration_required |
|---|---:|---:|---:|
| `IMG_1165` | 0.150s | 60.0 | 0.267s |

**Sub-5s clips (below RME training distribution): 12 total.** Listed for labeling-pass decision (drop, label as best-effort single window, or use as one window of a multi-clip aggregate if RHpE-style session reconstruction is added later).

| clip_id | duration | fps | resolution | size_mb | V-JEPA-2 viable? |
|---|---:|---:|---|---:|---|
| `IMG_1165` | 0.150s | 60.0 | 2160x3840 | 0.8 | **no** |
| `IMG_1103` | 0.483s | 60.0 | 2160x3840 | 2.0 | yes |
| `IMG_1128` | 0.917s | 60.0 | 2160x3840 | 4.7 | yes |
| `IMG_1050` | 0.950s | 60.0 | 1080x1920 | 2.1 | yes |
| `IMG_1024` | 1.833s | 30.0 | 2160x3840 | 2.9 | yes |
| `IMG_1088` | 2.118s | 60.0 | 2160x3840 | 10.0 | yes |
| `IMG_1025` | 2.333s | 30.0 | 2160x3840 | 4.3 | yes |
| `IMG_1106` | 2.752s | 60.0 | 2160x3840 | 13.1 | yes |
| `IMG_1022` | 3.252s | 60.0 | 1080x1920 | 6.6 | yes |
| `IMG_1077` | 3.468s | 60.0 | 1080x1920 | 7.4 | yes |
| `IMG_1149` | 3.735s | 60.0 | 2160x3840 | 17.6 | yes |
| `IMG_1129` | 4.985s | 60.0 | 2160x3840 | 23.5 | yes |

## 4. fps mix

| fps | n | % |
|---:|--:|--:|
| 30 | 4 | 3.3% |
| 60 | 118 | 96.7% |

**fps mix is a long-form pipeline concern.** Sliding-window aggregation (Phase 11+ if needed) must be fps-aware or normalize to a common rate. Per-clip ingestion via V-JEPA-2's 16-frame even-sampling is fps-invariant — single-clip operating-point decisions are unaffected.

## 5. Resolution mix

| Resolution | n | % | Note |
|---|--:|--:|---|
| 2160x3840 | 68 | 55.7% | iPhone 4K vertical |
| 1080x1920 | 54 | 44.3% | iPhone HD vertical |

V-JEPA-2 resizes to 224×224 internally; resolution above ~1000 doesn't change inference quality but does change file size, decode time, and intermediate buffer footprint. Mixed-resolution input is benign for V-JEPA-2 but worth flagging for any future fixed-aspect-ratio cropping pipeline.

## 6. Codec + pixel format

| codec | pix_fmt | n |
|---|---|--:|
| h264 | yuvj420p | 122 |

**Universal compatibility** — `h264` + `yuvj420p` reads natively under OpenCV / torchvision / ffmpeg / V-JEPA-2 preprocessing. No HEVC quirks, no codec-specific decode failures expected.

## 7. File size

min: 0.8 MB | median: 50.2 MB | max: 284.8 MB | total: 8.73 GB

## Concerns ranked for Phase 10a pre-reg consideration

1. **Portrait orientation** (§1) — load-bearing distribution shift. Lead hypothesis if    transfer test underperforms.
2. **Duration distribution longer than RME** (§2) — actually favorable for exercising    the simplified-B1 multi-window pipeline; recast as feature, not bug.
3. **Clips below V-JEPA-2 frame minimum** (§3, currently 0) — n=0 in current data, but    sub-1s clips (4 of them) are operationally fragile and worth pre-flight verification.
4. **fps mix** (§4) — minor for single-clip operating-point work; meaningful when    sliding-window aggregation comes into scope.
5. **Resolution mix** (§5) — benign for V-JEPA-2; flagged for forward-look.
6. **Codec** (§6) — no concern.
