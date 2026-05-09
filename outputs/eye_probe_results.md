# Track A — Eye-region behavior classification on Read My Ears (Phase 3 pilot)

**Branch:** `experiment/eye-probe`
**Atomic commit chain (Phase 3 build):** `09f286e` → `4de3f2c` → `45c84de` → `e12e3fa` → `6f4352e`
**Method audit:** `docs/phase3_auc_method.md`
**Hash chain:** `docs/preregistration_hashes.md` (6 frozen documents)
**Underlying artifact:** `outputs/eye_loso_results.json`

## Summary

We pre-registered a single-rater leave-one-source-out (LOSO) Ridge classifier pilot on V-JEPA-2 ViT-L features extracted from YOLO-derived eye-region crops of 34 short clips drawn from 12 sources of the Read My Ears stratified subset. The pre-registered primary metric (pooled-prediction AUC) is **0.6813** with a 95 % DeLong analytical confidence interval of **[0.4866, 0.8760]**; permutation p-value (n = 1000, group-naive shuffle, +1-form) is **0.058**. The mechanical pre-registered decision (AUC ≥ 0.65 → adopt v1 crop, write up Track A) is honored. The result establishes a proof-of-concept-scale signal for the V-JEPA-2-on-eye-crops architecture; it does not establish stable cross-subject precision.

## Headline numbers

- **Pooled AUC:** 0.6813
- **95 % DeLong CI:** [0.4866, 0.8760] (analytical, locked decision metric)
- **95 % subject-bootstrap CI:** [0.4138, 0.8980] (n = 2000 source-resamples, retrospective complement)
- **Permutation p:** 0.058 (n = 1000, +1-form)
- **Decision per pre-registration:** ≥ 0.65 (mechanical pass on the point estimate)

The DeLong lower bound 0.49 is below chance (0.50); the subject-bootstrap lower bound 0.41 is 9 pp wider still. The bootstrap is the more conservative precision estimator on this data structure: DeLong assumes independence of held-out predictions, which LOSO violates because predictions within a held-out source are correlated. Subject-bootstrap accounts for that source-level dependence by resampling sources (not clips) with replacement. Both intervals are reported for methodological completeness; the pre-registered decision operates on the DeLong point estimate per Phase 3 protocol parity. With n = 34 the pooled estimate clears the threshold under either CI method, but precision is too wide under either to confidently distinguish from chance on this dataset alone.

## Pipeline (locked)

1. **Cropping.** YOLO face detection (`vendor/horse-face-ear-detection/yolov8l_horse_face_detection.pt`) on the middle frame of each clip; eye-region heuristic v1 = upper 40 % of face bbox × full bbox width, square-padded around the strip's center, resized to 224 × 224. Drift check on first / middle / last frame; 4 clips with bbox-center drift > 10 % switched to per-frame detection. Failure handling: YOLO returns 0 detections → skip + log, never fall back to whole-frame.
2. **Embedding.** V-JEPA-2 ViT-L (`facebook/vjepa2-vitl-fpc16-256-ssv2`) on 16 evenly-spaced frames per clip, mean-pooled `last_hidden_state` patch tokens → 1024-d. Parity test before extraction: re-extract one cached ear-baseline clip via the lifted CLI forward pass, assert cosine similarity ≥ 0.999 vs cached embedding (`outputs/vjepa2_embeddings.npz`). **Observed: cos = 1.000000, ‖Δ‖ = 0.0000** — bit-exact match to the cached baseline; deterministic forward pass confirmed.
3. **Classification.** `RidgeClassifier(alpha=1.0, class_weight="balanced")` instantiated fresh per fold; `StandardScaler` instantiated fresh per fold, `fit_transform` on training fold only, `transform` on test fold (no precomputed scaling — leakage-free). Score via `decision_function`.
4. **Aggregation.** Pooled-prediction AUC over all 34 (truth, score) pairs concatenated across folds. Per-fold AUC distribution (min / median / max + n_defined + n_skipped) reported as the spread diagnostic. DeLong (1988) analytical 95 % CI on the pooled AUC. Permutation test n = 1000, seed = 42, global label shuffle, full LOSO loop per permutation, +1-form p-value.

## Sample composition

- **n = 34 viable clips** after Phase 1 cropping (35 successful crops − 1 manually excluded for eye occupying < 20 % of crop area on a dark profile clip; the 36th clip was a YOLO no-detection skip).
- **12 sources (S1–S12).** Class balance under user's blind-pass eye labels: 21 action, 13 background.
- **Two excluded clips were labeled but had no embedding** — dropped from LOSO with explicit log: `background_S1.mp4_11_` (label = ACTION, reason = manual_excluded_eye_under_20pct) and `background_S4.mp4_7_` (label = BACKGROUND, reason = YOLO_no_detection_static).
- **Four sources were class-degenerate** at this label/N regime: S3 (3 action, 0 bg), S4 (2 action, 0 bg), S7 (0 action, 3 bg), S10 (3 action, 0 bg). Their predictions contributed to the pooled pool; their per-fold AUCs are undefined.

## Per-fold diagnostic — the load-bearing pattern

Of the 8 defined LOSO folds, **5 achieved perfect classification** (S1, S2, S9, S11, S12 — fold AUC = 1.000), **2 inverted completely** (S5, S6 — fold AUC = 0.000 — the model produced wrong-direction predictions), and **1 was at chance** (S8 — fold AUC = 0.500). The remaining 4 sources (S3, S4, S7, S10) had single-class held-out sets and were excluded from the per-fold metric but contributed predictions to the pooled pool.

| Source | n_test | n_pos | n_neg | Defined | Fold AUC |
|---|---|---|---|---|---|
| S1  | 2 | 1 | 1 | ✓ | 1.000 |
| S2  | 3 | 2 | 1 | ✓ | 1.000 |
| S3  | 3 | 3 | 0 | — (single-class) | undefined |
| S4  | 2 | 2 | 0 | — (single-class) | undefined |
| S5  | 3 | 2 | 1 | ✓ | **0.000** |
| S6  | 3 | 1 | 2 | ✓ | **0.000** |
| S7  | 3 | 0 | 3 | — (single-class) | undefined |
| S8  | 3 | 2 | 1 | ✓ | 0.500 |
| S9  | 3 | 2 | 1 | ✓ | 1.000 |
| S10 | 3 | 3 | 0 | — (single-class) | undefined |
| S11 | 3 | 1 | 2 | ✓ | 1.000 |
| S12 | 3 | 2 | 1 | ✓ | 1.000 |

Mean of defined folds = 0.6875; pooled AUC = 0.6813. The closeness confirms that decision-function scores from different fold-trained Ridge classifiers are reasonably comparable on this fixed-α-fixed-class-weight setup — the pooling assumption is not violated. The bimodal distribution (5 × 1.0, 2 × 0.0, 1 × 0.5) is the signature finding: at this label/N regime, the model classifies some sources perfectly and others perfectly wrong. The aggregate averages over this; the underlying behavior is high source-level variance with no stable middle.

### Inverted-fold diagnostic (S5/S6) — corrected 2026-05-09

The first version of this diagnostic (committed earlier in `eac021a`) was built partly on a corrupted `per_clip` JSON: a bug in `eye_loso_lr.py` paired clip names from one ordering with labels and scores from another (LOSO traversal vs alphabetical sort). The pooled AUC, DeLong CI, bootstrap CI, permutation p, and per-fold AUCs were always correct — they operate on parallel arrays internally consistent with each other — but the post-loop name↔value join was scrambled. The bug was fixed; pooled metrics reproduced bit-for-bit; this section is rewritten from the corrected per-clip data. See `docs/phase3_per_clip_correction.md` for the full bug history + regression-test pin.

The corrected per-clip ranking inside S5 and S6 (predicted score, high → low):

**S5 fold, AUC = 0.000:**

| Score | Label | Clip | Observation |
|---|---|---|---|
| +0.592 | **BG** | `action_S5.mp4_2_` | "eyes still" |
| +0.524 | ACT | `action_S5.mp4_5_` | "gaze change, muscle tension change above the orbit" |
| +0.192 | ACT | `background_S5.mp4_10_` | "blinking" (head turned, eye in bottom-corner of v1 crop) |

**S6 fold, AUC = 0.000:**

| Score | Label | Clip | Observation |
|---|---|---|---|
| +0.637 | **BG** | `background_S6.mp4_3_` | "eyes still" |
| +0.515 | **BG** | `background_S6.mp4_2_` | "eyes still" |
| +0.170 | ACT | `action_S6.mp4_2_` | "sclera becoming less visible" |

The original three-factor diagnostic correctly explained why the ACT-labeled clips were pulled DOWN in score:

- **(a) Sub-perceptual-floor ACTION labels.** `action_S5.mp4_5_` (slight muscle-tension shift above the orbit) and `action_S6.mp4_2_` (few-pixel sclera reduction) are at or below the perceptual floor of V-JEPA-2 ViT-L on 224 × 224 crops. The labels reflect what a clinician sees with zoom + frame-stepping; the model architecturally cannot see them at this input resolution. Phase 4 rubric tightening (no zoom, no frame-stepping → BACKGROUND) flips these to BG, removing the inconsistency.
- **(b) Crop-positioning failure.** `background_S5.mp4_10_` has the horse's head turned with the eye in the bottom-corner of the v1 crop; V-JEPA-2 sees mostly head-rotation features rather than the labeled blink. Phase 4 v2 profile-aware crop (locked in `outputs/track_b_phase1_preregistration.md`) addresses this.
- **(c) Sub-second clip duration.** Three of the six inverted-fold clips are under 1 s. V-JEPA-2's 16-frame sampling against a sub-second native window compresses temporal-feature activation. Phase 4 does **not** apply a duration filter (empirically dropped — the ffprobe sweep showed sub-second clips are ~5 × biased toward ACTION class, so a filter would create class-balance confound; see `outputs/track_b_phase4_preregistration.md`). The Phase 4 rubric tightening absorbs most of factor (c) implicitly: most sub-second ACTION clips will be reclassified as BACKGROUND, flattening the duration/class correlation.

**The corrected per-clip data also reveals a fourth pattern the original diagnostic missed:** three BG-labeled clips ranked above their fold's ACT clips:

- `action_S5.mp4_2_` (BG, "eyes still") — top of S5 fold at +0.592
- `background_S6.mp4_3_` (BG, "eyes still") — top of S6 fold at +0.637
- `background_S6.mp4_2_` (BG, "eyes still") — second in S6 at +0.515

For an inverted fold (AUC = 0.000), it's not enough that the ACT clips rank low — the BG clips must also rank high relative to them. These three BG clips do exactly that. The Ridge classifier trained on 11 source folds learned features that correlate with ACTION on the training set; when applied to held-out S5/S6, those features fire on these BG clips despite the user-label saying "eyes still." Plausible drivers: head pose, framing, ear position in the crop margin, or other source-correlated visual cues that happen to overlap with ACTION-class features on the training distribution. **This factor is structural to small-N LOSO with source-distinct visual contexts and is not directly addressable by the Phase 4 v2-crop + relabel intervention.**

The honest reading of the corrected diagnostic: factors (a) and (b) are addressable by Phase 4's pre-registered fixes; factor (c) is implicitly handled by the rubric tightening; **factor (d) — BG clips ranked above ACT clips on source-correlated training features — is not addressable by Phase 4** and remains a structural caveat on the eye-track at n = 34 with single-rater labels. Phase 4 may recover some inversion signal but is not expected to fully resolve S5/S6 unless factor (d) is incidentally suppressed.

No fix is retrofitted into the Phase 3 pipeline or numbers; the diagnostic improves the interpretation of the Phase 3 result without modifying it. Phase 4, when launched, carries fixes (a) and (b) under its own pre-registration.

## Claim scope

**This result establishes** that V-JEPA-2 features over YOLO-derived eye crops, classified by Ridge LOSO with the locked pipeline above, can clear a pre-registered AUC threshold of 0.65 at proof-of-concept scale on the 34-clip RME stratified subset under single-observer eye-region-change labels.

**This result does not establish** that the signal generalizes across subjects with stable precision. The per-fold inversion on S5 and S6, the CI lower bound below chance, and the bimodal per-fold distribution all indicate that subject-level variance is the dominant uncertainty at n = 34. The result supports a methodology-pilot claim and explicitly does not support clinical-pre-screening or production-classifier claims.

**The closest comparable result on the same protocol shape** is the Read My Ears ear-track baseline (LOSO AUC 0.875 on 283 clips, `outputs/iter65_sanity5_loso_rme_results.json`). The two are comparable in pipeline shape but distinct in task and in scale. The precision difference reflects label/N constraints at the eye scope, not necessarily an intrinsic difference in task difficulty between ear and eye behaviors.

## Pre-registration discipline (audit chain)

Six frozen documents are listed with SHA-256 hashes in `docs/preregistration_hashes.md`:

1. `outputs/eye_probe_preregistration_minicpm.md` — sanity-test thresholds for the closed MLLM track
2. `outputs/expected_diagnostic_minicpm_blink.json` — pre-committed per-clip MLLM predictions
3. `outputs/track_b_phase1_preregistration.md` — Track B Phase 1 + locked v2 fallback rule
4. `outputs/eye_crops_annotations.md` — per-clip eye-visible Y/N + post-review corrections
5. `outputs/eye_verification_clips.txt` — user's blind labels (frozen post-collection)
6. `docs/phase3_auc_method.md` — Phase 3 method + result audit doc

The hashes prove content lock at freeze time; the commit timestamps witness existence. The mechanical pre-registered rule (AUC ≥ 0.65 → adopt v1, write up Track A) was applied to the point estimate without conditioning on CI width or per-fold spread — those features inform interpretation but not the locked decision. The pre-committed v2 profile-aware-crop fallback (Track B Phase 1 pre-reg) was not invoked because the point estimate cleared the threshold; v2 is reserved for the 0.55–0.65 ambiguous zone if ever encountered, and the pre-registration explicitly forbids further crop iteration beyond v2.

## Reproducibility

- **Random seed:** 42 (permutation test); 43 (verification list ordering, separate)
- **Python:** 3.11.13
- **Frameworks:** torch 2.11.0, transformers 5.8.0, scikit-learn 1.8.0, numpy 2.4.4, opencv 4.13.0, ultralytics 8.4.46
- **Hardware:** Apple MacBook Pro 14" 2023, M2 Max, 96 GB unified memory; macOS Tahoe 26.3.1; V-JEPA-2 forward pass on Apple MPS
- **V-JEPA-2 model variant:** `facebook/vjepa2-vitl-fpc16-256-ssv2` (ViT-L, 16 frames per clip, hidden_size = 1024)
- **YOLO face weights:** `yolov8l_horse_face_detection.pt` from `vendor/horse-face-ear-detection/horse_face_detection/`
- **Determinism:** V-JEPA-2 in `eval()` + `torch.no_grad()` mode reproduces the cached baseline embedding bit-exactly (cos = 1.000000); the parity test in `tools/extract_vjepa2.py` enforces this on every new extraction.
- **Re-running the result:** `python tools/eye_loso_lr.py` against the canonical embeddings (`outputs/vjepa2_embeddings_eye.npz`, n = 34) and labels (`outputs/eye_verification_clips.txt`, n = 36 with 2 dropped) reproduces `outputs/eye_loso_results.json` exactly.

## Future work (out of scope for this writeup)

The closed-track MLLM zero-shot result on this same task — Lesson 18 (`docs/lessons_learned.md`) — established that VLM zero-shot on uncropped horse video is structurally insufficient for fine-grained eye perception across four prompt × model variants (Qwen2.5-VL v1/v2/v3 + MiniCPM-V 4.5 v3). Track A is the architectural answer Lesson 18 named; this writeup is the empirical clearance of its pre-registered threshold at POC scale.

## Phase 5 result (added 2026-05-09, evening)

**Phase 5 primary result: pooled AUC 0.7985 (subject-bootstrap CI [0.584, 0.964]; permutation p = 0.010; Δ vs Phase 3 = +0.1172, DeLong-paired z = 1.067, p = 0.286).** Decision per pre-registered 3-band rule: **MIDDLE**. The Top band required both Δ ≥ 0.10 AND DeLong-paired p < 0.05; the Δ size cleared, the paired test did not at this n.

Pre-registered locked Middle-band sentence applies exactly: *"Cropping helped within statistical noise — effect size large but paired test couldn't reject at this n. Realistic POC interpretation; same collaborator asks as Phase 3 plus eye-detector annotation."*

**Headline (Lesson 11 anchor):** the 0.7985 AUC lands at the top of the project's pre-locked realistic LOSO band (Lesson 11: 0.70–0.80 target; ≥ 0.85 explicitly unrealistic on diverse data). The architecture clears the project's pre-locked realistic band on the second behavior, even at MIDDLE-band statistical confidence. Honest framing: performance at top of realistic POC band; precision insufficient for clinical use; level test wide CI requires N expansion.

**The DeLong-vs-permutation gap is itself a finding.** Permutation p = 0.010 (vs chance null) and DeLong-paired p = 0.286 (vs Phase 3 null) are both correctly computed; they answer different questions. The pre-registered top-band gate uses paired-DeLong because the question Phase 5 was designed to answer is whether v3 improves over Phase 3, not whether v3 differs from chance. The fact that paired-DeLong fails despite Δ being at the simulated MDE-90% threshold says **v3 didn't just amplify Phase 3's discrimination — it shifted which clips are well-classified.** The two pipelines get partially-disjoint subsets right; v3 trades old failure modes for new ones. Phase 6 instrumentation: per-clip diff between Phase 3 and Phase 5 primary predictions. Mechanism behind the prediction-shift is the next-tier diagnostic.

Sensitivity 1 (rubric-tax under good crops): v3+tightened AUC = 0.8179 vs primary 0.7985, Δ = +0.0194 → within ±0.085 MDE band. Locked verdict: Phase 4's −0.10 rubric-tax does NOT reproduce under v3 cropping. Two consistent mechanisms (rubric clean under good crops vs neutral on low-confidence clips) are observationally indistinguishable at n=34; the distinction matters mainly if v3 is hard to scale. Audit doc carries the full caveat.

Sensitivity 2 (margin curve at 10/15/40/80%): pair-differences +0.044, −0.051, +0.048; none exceeds bootstrap half-width (~0.19). Locked categorical: **FLAT**. Margin choice doesn't matter at this scale; Phase 6 picks on other criteria. Suspicious sub-noise pattern: m=15 and m=80 both at ~0.79, m=10 and m=40 lower — pre-registered hypothesis for Phase 6 N expansion: two distinct informative regimes (tight eye-resolution + face context), not a single sweet spot.

Factor-(d) verdict: SUPPRESSED (2/3 BG-targets below median, same suppression direction as Phase 4 v2). The match across two very different crop pipelines suggests factor (d) reflects **clip-level properties orthogonal to crop mechanism**, not crop-quality dependence — earlier framing "fixable by cropping" overstated the causal chain. Phase 6 should instrument clip-level features (lighting, framing, source-camera) and multi-rater κ on the 3 persistent BG-targets.

Intra-rater consistency on 5 random clips (≥4-6h gap, distinct mask seed): median IoU = 0.765, well above locked 0.6 gate. Gold-standard framing supported. Documented limitation: ≥48h gap is the methodologically ideal; ≥4-6h is the locked compromise, deferred to Phase 6.

Lowest IoU clip (`background_S8.mp4_3_`, mean 0.587) attributed by annotator to subtle sclera/gaze change combined with catchlight reflection rendering the visible eye boundary ambiguous between passes. Pre-registered before any per-fold S8 inspection: if S8 fold inverts under any Phase 5 config, the principled reading is *perceptual-floor* (signal below V-JEPA-2's access at 224×224) not crop-quality — same diagnostic class as the S5/S6 sub-pixel ACTION clips that drove Phase 3 inversion. Catchlight-as-confounder also pre-registered as a candidate mechanism for the margin-curve shape if a future Phase 6 N expansion produces monotone-improving with looser margins. See `docs/phase5_audit.md` for full locked framing.

Full audit at `docs/phase5_audit.md` (hash registered in `docs/preregistration_hashes.md`).

## Phase 4 result (added 2026-05-09)

A solo combined intervention applying v2 profile-aware crop + tightened-rubric blind re-label was pre-registered (`outputs/track_b_phase4_preregistration.md`, hash `ced5cae6...`) and run on 2026-05-09 against the same 34-clip set (38 v2 outputs after the locked tie-break-both-halves rule). **Phase 4 primary result: pooled AUC 0.5854 (DeLong 95 % CI [0.395, 0.776]; subject-bootstrap CI [0.294, 0.807]; permutation p = 0.235). Δ vs Phase 3 = −0.0959, triggering the pre-registered regression branch.**

The conditional 2-run ablation locked before unmask:
- **Ablation A (v1 crops + tightened labels)** isolates the relabel effect: AUC 0.5857 (Δ = −0.0956). Tightened-rubric labels alone hurt by ≈ 0.10.
- **Ablation B (v2 crops + original labels)** isolates the v2 crop effect: AUC 0.5536 (Δ = −0.128). v2 crops alone hurt by ≈ 0.13. The contact-sheet inspection that flagged "ears dominate high-frequency content in the upper-40 % strip" was confirmed mechanically.

The factor-(d) suppression criterion (locked before run): 2 of 3 persistent BG-target clips scored strictly below the median Phase 4 BG-clip score → **factor (d) SUPPRESSED** (`background_S6.mp4_2_` and `background_S6.mp4_3_` below; `action_S5.mp4_2_` still above). Mechanical verdict, narrow finding — the v2 representation changed score behavior on those specific clips without producing overall AUC improvement.

**The locked diagnostic narrative was empirically falsified for factors (a) and (b)** — neither tightened labels nor v2 crops recovered Phase 3's signal in isolation. Combining them did not help (combined Δ ≈ relabel-alone Δ; less harmful than v2-alone, suggesting tightened labels somehow rescue some of the v2 damage by aligning the rubric with what the model can perceive). Factor (d) was suppressed for 2 of 3 targets — the only specific Phase 4 success.

**Phase 5 implications:** the contact-sheet finding rules out the upper-strip-Sobel-selection family of heuristics for horse-anatomy crop design. Three candidate redesigns named in `docs/phase4_audit.md`: middle-50 % strip with horizontal split, eye-specific YOLO detector, or anatomically-positioned crop without selection. The rubric-tightening result also raises a methodology question for any future labeling-protocol change at this N. Phase 5 would lock one redesign with its own pre-registration; this writeup does not commit to that work.

Full audit at `docs/phase4_audit.md` (hash registered in `docs/preregistration_hashes.md`).
