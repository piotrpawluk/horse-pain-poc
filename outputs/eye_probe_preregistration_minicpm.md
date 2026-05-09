# Pre-registration — MiniCPM-V 4.5 eye-probe

**Frozen:** 2026-05-08, BEFORE any MiniCPM-V 4.5 inference run.
**Author:** Claude (Opus 4.7), under user direction (Plans/przeanalizuj-oba-dokumenty-w-serialized-crab.md V3 + 7-tightening review).
**Reason:** Pre-committed thresholds eliminate gray-zone goal-shifting after results land. Pre-committed predictions in `expected_diagnostic_minicpm_blink.json` eliminate post-hoc rationalization on the sanity test.

## Hypothesis being tested

**H1 (capability):** MiniCPM-V 4.5 — architecturally engineered for fine-grained motion (MotionBench, FavorBench, native 10 FPS support, 96x video-token compression) — can distinguish eye-region motion from stillness on uncropped horse video at the 36-clip pilot scale.

**H0 (structural insufficiency):** Even MiniCPM-V 4.5 fails on this geometry. The eye is approx 3-4% of a 1920x1080 frame and approx 16 px wide after the model's 448x448 downsample. If the architecturally-right tool also collapses (matching the Qwen v1/v2/v3 trilogy), zero-shot VLM on uncropped frames is structurally insufficient regardless of model fitness, and Track B (ROI crop + V-JEPA-2 + LR) is the architectural answer.

## Pre-flight gate (n=6, user-labeled clips)

3 verified blink (action_S2.mp4_2, action_S2.mp4_8, action_S4.mp4_4) + 3 verified no-blink (action_S4.mp4_2, action_S2.mp4_7, action_S4.mp4_10).

**Frame policy for sanity:** all native frames sent (NOT fps=10). Reason: removes frame-count as a confound for the architecturally-right model on the pre-flight test. The 36-clip run uses fps=10 for parity with prior Qwen/Gemini runs.

**Pass criterion:** **5 of 6 correct** (binomial p ~ 0.11 vs random). Below 5/6 -> stop, no 36-clip run.

A clip is "correct" iff:
- **Blink clip:** classification=action AND observed_change contains eyelid/blink/lid/closure language AND frame pair has i != j
- **No-blink clip:** classification=background OR (classification=action with a non-blink feature). NOT correct: action with eyelid/blink/lid-descent language (false-positive template inheritance).

A clip is **structurally failed** (counted as incorrect even if classification matches) when:
- observed_change is identical across 2+ clips (template hallucination — the v3 Qwen failure mode)
- most_changed_frame_pair has i == j
- observed_change cites no named eye feature

Per-clip predictions are committed in `outputs/expected_diagnostic_minicpm_blink.json`.

## 36-clip success criteria (binary OvR AUC, per GATE.md)

Run only if pre-flight passes. Use user's morning labels as ground truth.

| AUC | Interpretation | Action |
|-----|---|---|
| **>= 0.65** | Meaningful signal | Continue to GLM-4.6V-Flash as confirmation. Begin Track A writeup. |
| **0.55-0.65** | Ambiguous | Run on extended set (next 36 clips from RME) before any conclusion. |
| **< 0.55** | Structural insufficiency confirmed | Stop. Pivot to Track B (YOLO eye-ROI crop + V-JEPA-2 + LR per Read My Ears recipe). |

The 0.65 threshold matches GATE.md's per-behavior signal-strength metric. Below 0.55 is below chance with margin (binomial-confidence-interval-aware on n=36).

## Anti-patterns this pre-registration prevents

- **"MiniCPM hit 56% accuracy, what now?"** -> Already committed: 0.55-0.65 AUC = ambiguous = extended set, NOT a pass.
- **"The model said X and that's plausible"** -> Predictions committed in `expected_diagnostic_minicpm_blink.json` BEFORE the run.
- **Mid-experiment prompt iteration** -> Prompt frozen in `tools/prompts/eye_v3_minicpm.txt`. Bump version (eye_v4_*) for any change.
- **Confounded model/prompt comparison** -> Same v3 schema across MiniCPM/GLM with vendor-specific adaptations committed up-front in their own prompt files.

## Structural caveats acknowledged before run

1. **Geometry:** eye region is approx 3-4% of frame area at 1920x1080. After MiniCPM's 448x downsample, approx 16 px wide. Dominant prior risk, independent of prompt or model.
2. **Frame floor:** shortest sanity clip (action_S2.mp4_7) is 0.24 s = 6 native frames, 2 at fps=10. Sanity uses native FPS to remove this confound. 36-clip run uses fps=10 for parity.
3. **Single-rater labels:** user's morning verification is single-observer. Same kappa caveat as ear track.

## Confounds NOT controlled

- Model + prompt + input-policy bundle: cannot isolate "model" vs "prompt" vs "fps policy". Acceptable for screening pilot; ablations only if pilot produces signal worth investigating.
- v3 schema: identical to Qwen2.5-VL v3 run for direct comparability, but MiniCPM may benefit from vendor-specific deviations. Trade-off accepted for cross-model comparability.

## Exit-on-failure commitments

- **Pre-flight fails (<5/6):** Lesson 18 in `docs/lessons_learned.md` — "MiniCPM-V 4.5 (architecturally-right tool, MotionBench-tuned) failed pre-flight on n=6 hand-verified eye blinks. Combined with Qwen v1/v2/v3 collapse trilogy, structural insufficiency of zero-shot VLM on uncropped horse video confirmed across model families. Track B is the architectural answer."
- **36-clip AUC < 0.55:** same Lesson 18, with stronger evidence (n=36).
- **MiniCPM passes; GLM corroborates at AUC >= 0.65:** Lesson 18 different — "Two independent open-weight VLMs detect eye-blink signal on uncropped horse video. Cross-behavior bifurcation (collapse on ear, signal on eye) worth investigating."
- **MiniCPM passes; GLM disagrees:** holds — write up the per-model finding without a unified claim. Don't average disagreement.

## Time budget

Per user's tightening #7: realistic 2-3 h total, not the 1.5 h originally proposed.
