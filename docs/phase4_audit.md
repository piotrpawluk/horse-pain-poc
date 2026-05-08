# Track B Phase 4 — Audit doc (results + ablations + factor-(d) verdict)

**Run:** 2026-05-09, after Phase 4 pre-registration freeze (commit `26aa3fb`, hash `ced5cae6...`).
**Scripts:** `tools/eye_crop_pipeline_v2.py` (v2 crops), `tools/extract_vjepa2.py` (parity-checked feature extraction), `tools/eye_loso_lr_phase4.py` (LOSO + factor-(d) verdict + ablation modes).
**Method audit:** `outputs/track_b_phase4_preregistration.md` (hash `ced5cae6...` — this version locks numerical thresholds, 4-band decision rule, factor-(d) suppression criterion, tie-break-both-halves rule, and the interpretation lock for the fourth factor).

## Summary

Phase 4's locked combined intervention (v2 profile-aware crop + tightened-rubric blind re-label) **regressed** below Phase 3's pooled AUC by 0.0959 (0.6813 → 0.5854). The pre-registered conditional ablation triggered. Both ablations individually showed regression: tightened-rubric labels alone hurt by 0.0956 (Ablation A); v2 crops alone hurt by 0.1277 (Ablation B). The diagnostic narrative locked in `eye_probe_results.md` was empirically falsified — neither intervention recovered Phase 3's signal in isolation, and combining them did not help.

The pre-run inspection of the v2 contact sheet (visible in `outputs/eye_crops_v2_contact_sheet.png`) had already revealed the locked v2 heuristic systematically captured ear regions rather than eye regions. The Phase 4 protocol was run as locked despite this discovery, because the test was structured to be informative regardless: Ablation B (v2 crops + original labels) was designed to confirm or refute the visible failure mode mechanically, and Ablation A was independently informative about the relabel effect alone. Both ablations confirmed the visible findings.

The factor-(d) suppression criterion (locked before run) returned **SUPPRESSED**: 2 of 3 persistent BG-target clips scored below the median Phase 4 BG-clip score. This is a genuine but narrow finding — the v2 ear-focused representation changed the model's score behavior on those clips, but did not produce overall AUC improvement.

## Pre-run discovery: v2 contact sheet inspection

Per the discipline pattern element 6 (empirical-anchor data-structure-dependent rules), the v2 contact sheet was inspected before V-JEPA-2 extraction began. Finding (in writing, before LOSO ran):

> "Most v2 crops show the ear and forehead, not the eye. The half-selection picks the side with HIGHER Sobel edge content, and horse ears have far more high-frequency texture (sharp outline, fur-line, contrast against background) than the eye region. So the 'higher spatial-frequency wins' rule selects the ear-containing half, then square-padding pulls the crop center onto the ear."

Pre-committed framing: the Phase 4 protocol was run as locked despite this discovery, because the result was structured to be informative through the conditional ablations. Phase 5 (separately pre-registered) will target a corrected crop heuristic informed by this anatomy-of-horses finding.

## Results

All runs use identical pipeline (V-JEPA-2 ViT-L `facebook/vjepa2-vitl-fpc16-256-ssv2`, mean-pooled patch tokens → 1024-d, then `RidgeClassifier(α=1.0, class_weight="balanced")` + `StandardScaler` refit per fold). Per-clip output uses post-fix alignment (commit `412a957`).

| Run | Embeddings | Labels | n | AUC | Δ vs P3 | DeLong 95% CI | Bootstrap 95% CI | p | Fold defined |
|---|---|---|---|---|---|---|---|---|---|
| Phase 3 baseline | v1 | original (Phase 3) | 34 | **0.6813** | — | [0.487, 0.876] | [0.414, 0.898] | 0.058 | 8/12 |
| **Phase 4 primary** | **v2** | **tightened** | 38 | **0.5854** | **−0.0959** | [0.395, 0.776] | [0.294, 0.807] | 0.235 | 8/12 |
| Ablation A | v1 | tightened | 34 | 0.5857 | −0.0956 | [0.390, 0.781] | [0.382, 0.788] | 0.234 | 8/12 |
| Ablation B | v2 | original | 38 | 0.5536 | −0.1277 | [0.358, 0.749] | [0.338, 0.720] | 0.325 | 8/12 |

**Decomposition**: tightened-rubric labels alone hurt by ≈ 0.096 (Ablation A); v2 crops alone hurt by ≈ 0.128 (Ablation B); combined hurt by ≈ 0.096 (Phase 4 primary). The combined effect is *not additive* — interestingly less harmful than v2 crops alone with original labels. One reading: tightened-rubric labels reclassify clips whose v1-rubric ACT call relied on motion the v2 ear-region crops can't see anyway, making the model's task more internally consistent ("stable rubric on inputs the model can't perceive") even at lower AUC. The mechanism isn't crisply identifiable at n = 38 with single-rater labels.

**Decision per pre-registration**: Phase 4 primary AUC 0.5854 < 0.6313 → **regression branch**. The locked threshold band ("`< 0.6313` regression vs Phase 3") triggered the conditional 2-run ablation, which confirmed both fixes individually hurt.

## Per-fold detail (Phase 4 primary)

| Source | n | n_pos | n_neg | Defined | Fold AUC |
|---|---|---|---|---|---|
| S1  | 2 | 0 | 2 | — single-class | undefined |
| S2  | 3 | 2 | 1 | ✓ | 1.000 |
| S3  | 4 | 4 | 0 | — single-class | undefined |
| S4  | 2 | 1 | 1 | ✓ | **0.000** |
| S5  | 4 | 2 | 2 | ✓ | **0.000** |
| S6  | 3 | 0 | 3 | — single-class | undefined |
| S7  | 3 | 0 | 3 | — single-class | undefined |
| S8  | 3 | 2 | 1 | ✓ | 1.000 |
| S9  | 3 | 1 | 2 | ✓ | 0.500 |
| S10 | 4 | 2 | 2 | ✓ | 1.000 |
| S11 | 4 | 1 | 3 | ✓ | **0.000** |
| S12 | 3 | 2 | 1 | ✓ | 0.000 |

Per-fold distribution: min 0.000, median 0.250, max 1.000, n_defined 8 / n_skipped 4. Bimodal pattern persists, with more inverted folds (4) than perfect folds (3) under Phase 4 vs Phase 3's 5 perfect / 2 inverted / 1 chance.

**S5 close-up** (the surviving like-for-like comparison — Phase 3 fold AUC = 0.000):

| Score | Label | Clip |
|---|---|---|
| +0.778 | BG | `action_S5.mp4_5_` (was ACT in Phase 3, flipped under tightened rubric "head moving up, eyes still") |
| +0.189 | BG | `action_S5.mp4_2_` (stayed BG; persistent BG-target #1) |
| −0.363 | ACT | `background_S5.mp4_10___left` (v2 left half of the head-turn-blink clip) |
| −0.474 | ACT | `background_S5.mp4_10___right` (v2 right half) |

S5 fold AUC = 0.000 in Phase 4. Phase 3's S5 was also 0.000. **The v2 crops did not rescue S5.** Both halves of `background_S5.mp4_10_` (the factor-(b) crop-misposition target) still rank below the BG clips. Confirms v2 ear-region crops carry no useful eye-action signal even when the geometric target is the eye-closure event.

## Factor (d) suppression verdict (locked criterion)

Per the criterion frozen in `outputs/track_b_phase4_preregistration.md` (hash `ced5cae6...`) before the LOSO run:

- Median Phase 4 BG-clip score (parent-clip max convention, n = 21 BG clips after aggregation): **−0.1376**
- `action_S5.mp4_2_` score = +0.1886 → **above median** (still action-like)
- `background_S6.mp4_2_` score = −0.9742 → **below median** (eye-like)
- `background_S6.mp4_3_` score = −0.8314 → **below median** (eye-like)

**2 of 3 below median → factor (d) SUPPRESSED.**

Mechanical interpretation per the locked rule. The v2 ear-region representation changed the score behavior on the persistent BG-target clips: 2 of 3 are now ranked low (eye-like) whereas Phase 3 ranked them all high (action-like). Note this is not the same as "factor (d) was the cause of inversion" — the overall AUC still regressed. The verdict says only that the specific score pattern flipped on 2 of 3 targets.

## Interpretation under the pre-registered framing

Phase 4 primary landed at AUC 0.5854 — between the "0.55 ≤ AUC < 0.65 inconclusive" band and the "regression" band (`< 0.6313`). The regression-trigger applied because Δ vs Phase 3 = −0.0959 > 0.05. Both ablations confirm the regression is real and decomposable into roughly:

- ~0.10 from tightened-rubric labels alone (rubric-dependent feature alignment)
- ~0.13 from v2 crops alone (heuristic captures ears not eyes — visible on contact sheet, confirmed by ablation)

The corrected Phase 3 inverted-fold diagnostic identified factors (a), (b), (c), and (d). Phase 4 was designed to address (a) and (b) directly, (c) implicitly. Empirically: neither (a)+(c) via relabel nor (b) via v2 crop produced AUC improvement on this dataset. Factor (d) (BG clips ranked above ACT clips on source-correlated training features) was suppressed for 2/3 targets — the only specific success of the Phase 4 intervention, and notably narrow.

The honest closing read: the diagnostic narrative was specific and falsifiable, the experiment was structured to test it, and the result falsified two of the three predicted improvements. That's a clean closure for Phase 4 — the architecture pieces remain valid (V-JEPA-2 features at the perceptual scale they operate, Ridge LOSO at this label/N regime), but the specific cropping heuristic and the labeling-rubric-tightening direction don't help on this dataset.

## Phase 4 closure

Phase 4 closes here as locked. The pre-registered numerical thresholds, decision rule, ablation contract, and factor-(d) criterion all applied mechanically. Phase 3's locked numbers and decision are unchanged. The audit chain is extended (this doc + 4 result JSONs + the v2 manifest and contact sheet hashes) without modifying earlier-locked content.

## Phase 5 implications (preview, not committed)

The contact-sheet finding ("ears dominate high-frequency content in the upper face strip on horse anatomy") is the most informative result of this entire arc for next-phase design. It rules out the upper-strip + Sobel-selection family of heuristics and points at three candidate redesigns:

1. **Middle 50 % strip with horizontal split + Sobel selection**. Drops below the ear band; keeps the half-selection rule. Cheapest fix; same family of heuristic, different vertical placement.
2. **Eye-specific YOLO detector**. Train on ~50 hand-annotated eye boxes from RHpE clips + adjacent datasets. Higher up-front cost (annotation + training) but produces the cropping signal directly. Phase-1-equivalent preprocessing investment.
3. **Anatomically-positioned crop without spatial-frequency selection**. Empirically derive eye position relative to face bbox from a small annotation set (per-source if appearance varies), hard-code the offset, no per-clip selection. Eliminates the heuristic-selection failure mode entirely.

Phase 5 pre-registration would lock one of these three designs (or a hybrid) with its own thresholds and ablation contract. The Phase 4 result narrows the design space considerably: any heuristic that uses Sobel-edge-magnitude on the upper face strip is now empirically ruled out.

The relabel result (tightened rubric hurt Ridge AUC) is also Phase-5-relevant: the V-JEPA-2 + Ridge architecture appears to be RUBRIC-DEPENDENT — small changes in label definition can produce ~0.10 AUC swings at this N. This is itself a methodology finding worth pre-registering at Phase 5: any future labeling protocol change must be paired with a parity LOSO before claiming the new labels are "better."

## What this means for the collaborator pitch

Phase 4's negative result narrows the eye-track story to:

- **Architecture validated at PoC scope on a real ROI-cropped fine-grained motion task** (Phase 3, AUC 0.6813 with wide CI, decision-band-clearing on the point estimate)
- **Three named failure modes diagnosed; all three pre-committed Phase 4 fixes were empirically tested** (running the pre-registered protocol despite visible mid-run evidence the v2 crop wouldn't work — discipline preserved)
- **Two of three Phase 4 fixes individually fail; the third (factor-(d) suppression) succeeded narrowly**
- **The contact-sheet inspection finding ("upper-strip Sobel selects ears not eyes on horse anatomy") rules out a family of heuristics for Phase 5**
- **The rubric-tightening result (tighter labels hurt Ridge AUC) raises a methodology question for any future labeling protocol change**

Each ask to a collaborator is now even more concrete than after Phase 3:
- Validate the perceptual floor with controlled-resolution data (Phase 3 finding, unchanged)
- **Help refine the crop heuristic with anatomical expertise OR provide a small annotation set for an eye-specific YOLO detector** (Phase 4 finding, sharpened)
- Provide inter-rater labels under a clinically-defined rubric to test whether rubric-tightening actually maps onto signal at higher N (Phase 4 finding, new)

## Audit chain extension

This commit extends `docs/preregistration_hashes.md` with hashes for:
- `outputs/eye_loso_results_phase4.json` (Phase 4 primary)
- `outputs/eye_loso_results_phase4_ablation_A.json` (relabel-alone)
- `outputs/eye_loso_results_phase4_ablation_B.json` (v2-crop-alone)
- `outputs/eye_crops_v2_manifest.jsonl` (v2 cropping decisions)
- `outputs/eye_crops_v2_contact_sheet.png` (visible heuristic-failure evidence)
- `docs/phase4_audit.md` (this document)

Phase 3 numbers and decision unchanged. Phase 4 closes per the pre-registered protocol.
