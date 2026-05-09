# Track B Phase 4 — Pre-registration (solo replication with named-fix application)

**Frozen:** 2026-05-08, BEFORE the Phase 4 re-label or v2 crop / extraction runs. Hash recorded in `docs/preregistration_hashes.md` at this commit.

**Phase 3 close-out** (audit reference, unchanged): pooled AUC 0.6813 (DeLong CI [0.4866, 0.8760]; subject-bootstrap CI [0.4138, 0.8980]; permutation p = 0.058). Per-fold spread bimodal: 5 perfect, 2 inverted (S5, S6), 1 chance, 4 single-class skipped. Decision branch `>= 0.65` honored mechanically. Post-Phase-3 manual crop inspection on S5/S6 produced a diagnostic naming three contributing factors: (a) sub-perceptual-floor labels on 3 clips, (b) crop-positioning failure on 1 clip (`background_S5.mp4_10_`), (c) sub-second clip duration on several clips.

**Phase 4 constraints (per user, 2026-05-08):** no second observer available. PoC is meant to ATTRACT collaborators, not require them. Execution time budget ~hours, not weeks. Phase 4 design therefore stays solo and applies the diagnostic-named fixes that a single observer can execute under documented memory-contamination risk.

## What Phase 4 does

A single combined intervention applying TWO of the three diagnostic-named fixes, on the same 34-clip viable set, under fresh pre-registration. Third fix dropped per empirical sweep (see "Sub-second filter — empirically dropped" below).

### Fix 1 — v2 profile-aware crop

Per the locked v2 spec in `outputs/track_b_phase1_preregistration.md`:

- Upper 40 % strip of YOLO face bbox is split into left half and right half.
- For each half: mean spatial-frequency content (Sobel edge magnitude or Laplacian variance) computed in a 32 px ROI sweep.
- Half with higher spatial-frequency content wins (eye region has more high-frequency texture than mostly-flat forehead/cheek).
- Tie-break (within 5 % of higher half): both halves output as separate samples; the resulting sample inherits the parent clip's source for LOSO grouping.
- Crop output: square-padded around the chosen half's center, resized to 224 × 224, applied to all native frames of the clip.

Implementation: `tools/eye_crop_pipeline_v2.py`. The Phase 1 v1 pipeline (`tools/eye_crop_pipeline.py`) is left untouched.

### Fix 2 — Tightened-rubric blind re-label

Same observer; one-week interval since Phase 3; filename-masked re-label under explicitly tightened rubric:

> **ACTION** = visible eye-region change at native 1920×1080 resolution at normal-speed playback. No zoom, no frame-stepping.
>
> **BACKGROUND** = no perceptible change at native resolution at normal-speed playback. Includes "change visible only via zoom or frame-stepping" — these are NOT ACTION.
>
> **ACTION? / BACKGROUND?** allowed for borderline; excluded from LOSO.

Operative tightening vs Phase 3: the explicit "no zoom, no frame-stepping" clause. Phase 3 rubric did not constrain inspection method.

Filenames masked `clip_M001` ... `clip_M034` with `mask_seed = 44` (different from Phase 3's seed 43, to avoid order-cue carryover). Mapping in `outputs/eye_relabel_keymap.json` (hash recorded in `preregistration_hashes.md`); user does not consult the keymap until after re-label submission.

**Memory-contamination risk acknowledged.** Same observer one week after Phase 3 close, with knowledge of which sources were inverted (S5, S6). Filename masking protects against filename-recognition cues but does not protect against horse-appearance recognition or rubric-debate memory. The Phase 4 audit doc and Track A writeup explicitly frame the result as **"rubric-conditioned same-observer re-label, single-week interval, memory-contamination risk acknowledged"** — NOT inter-rater κ measurement. Inter-rater work is Phase 5+ scope under separate pre-registration.

### Fix 3 NOT applied — sub-second clip filter (empirically dropped)

The 5-minute ffprobe sweep on the 34 viable clips before this pre-reg was frozen revealed structural class-by-duration correlation:

| Cutoff | N kept | Dropped (act / bg) | Class shift |
|---|---|---|---|
| 0.4 s | 29 | 5 / 0 | drops only ACTION |
| 0.6 s | 23 | 9 / 2 | drops 4.5× more ACTION |
| 0.8 s | 21 | 11 / 2 | flips majority class |
| 1.0 s | 19 | 13 / 2 | dropped set 87 % ACTION |

ACTION clips are systematically shorter than BACKGROUND clips in this dataset. Any duration cutoff above ~0.4 s creates a class-balance shift that confounds with the architecture comparison. The filter was dropped from Phase 4 before lock-in. The sweep itself is in the audit chain (`outputs/phase3_subject_bootstrap_ci.json` shares the freeze commit; the sweep result is documented in this pre-reg).

Issue (c) is mostly absorbed by Fix 2: most sub-second ACTION-labeled clips were on sub-pixel evidence and will be re-labeled BACKGROUND under the tightened rubric. The duration/class correlation flattens implicitly.

## Pre-registered thresholds (locked, 4 bands)

Phase 4 evaluation uses the SAME pipeline as Phase 3 (`RidgeClassifier(alpha=1.0, class_weight="balanced")` + `StandardScaler` refit per fold + pooled-prediction AUC primary + DeLong analytical 95 % CI + permutation test n = 1000 with global label shuffle, +1-form p), substituting (a) v2 crops for v1 crops and (b) tightened-rubric labels for original labels.

| Phase 4 outcome | Conclusion | Action |
|---|---|---|
| **Pooled AUC ≥ 0.72 AND DeLong CI lower bound ≥ 0.55** | Diagnostic correct; combined fix recovers signal | Track A writeup updated with Phase 4 result; collaborator pitch advances |
| **0.65 ≤ AUC < 0.72** | Improvement consistent with diagnostic, precision still wide | Bottleneck shifts from "named failures" to "N too small for tight CI"; explicit collaborator ask is multi-rater κ + N expansion (Phase 5 scope) |
| **0.55 ≤ AUC < 0.65** | Inconclusive; named fixes neither helped meaningfully nor regressed | Phase 3 result stands as the headline; Phase 4 documented as null result |
| **AUC < 0.6313** (Phase 3 0.6813 − 0.05) | Regression; diagnostic narrative wrong; one of the fixes broke things | Conditional ablation runs (below) to isolate the breaking fix |

Subject-bootstrap CI is reported alongside DeLong but does not modify the threshold gates. DeLong is the locked decision metric, matching Phase 3 protocol parity; the bootstrap is the conservative companion estimator.

## Conditional ablation (locked; runs ONLY if regression triggers)

If Phase 4 pooled AUC < 0.6313:

| Run | Configuration | Isolates |
|---|---|---|
| **Ablation A** | v1 crops (existing Phase 3 set) + tightened-rubric Phase 4 labels | Effect of relabel alone |
| **Ablation B** | v2 crops + Phase 3 original labels | Effect of v2 crop alone |

Same `eye_loso_lr.py` pipeline. Each ablation's pooled AUC is compared to Phase 3's 0.6813. Whichever ablation regresses identifies the breaking fix. Documented in `docs/phase4_audit.md`.

Pre-committed: only TWO ablations, no further iteration after them. If both ablations regress (combined-only succeeded by chance, or measurement noise), the conclusion is "Phase 4 narrative is wrong; revisit hypothesis"; Phase 5 with new pre-registration would carry that.

## What Phase 4 explicitly does NOT do

1. **Per-clip cherry-picking** based on Phase 3 prediction outcomes. Re-label rubric applies blindly to all 34 clips; filename masking enforces this.
2. **Hyperparameter tuning** on Phase 3 results. Same RidgeClassifier(α=1.0, balanced), same StandardScaler-per-fold, same `num_frames=16`, same V-JEPA-2 ViT-L variant.
3. **Iterate on the v2 heuristic** if v2 doesn't move AUC. v2 is the locked alternative; if it doesn't help, the result stands.
4. **Re-do Phase 3.** Phase 3 result stands at AUC 0.6813. Phase 4 produces a parallel result with its own audit, not a replacement.
5. **Apply the sub-second filter.** Empirically dropped per the duration sweep; would have created a class-balance confound.
6. **Frame the re-label as inter-rater κ.** It is intra-rater consistency under tightened rubric in the same week. Memory-contamination risk is documented and not denied.
7. **Run more than the two pre-committed ablations.** Open-ended ablation iteration is element-5 violation (sequence phases).

## Atomic outputs (when Phase 4 lands)

- `outputs/eye_relabel_filled.txt` — user's submitted re-labels (masked file with VERDICTs filled)
- `outputs/eye_relabel_unmasked.txt` — agent-unmasked version with real filenames + new labels
- `outputs/eye_crops_v2/<clip>.mp4` — v2 profile-aware crops (one per source clip, ≥ 1 per tied-clip)
- `outputs/vjepa2_embeddings_eye_v2.npz` — V-JEPA-2 features on v2 crops, parity test re-confirmed cos ≥ 0.999 vs ear baseline
- `outputs/eye_loso_results_phase4.json` — same schema as Phase 3, plus subject-bootstrap CI
- `outputs/eye_loso_results_phase4_ablation_A.json` (conditional, only on regression)
- `outputs/eye_loso_results_phase4_ablation_B.json` (conditional, only on regression)
- `docs/phase4_audit.md` — method + result + decision branch + (if regression) ablation interpretation
- Updates to `outputs/eye_probe_results.md` (Track A writeup) with Phase 4 result and locked sentence framing
- Updated `docs/preregistration_hashes.md`

## Interpretation lock for the fourth factor (added 2026-05-09, BEFORE unmask)

The Phase 3 per_clip alignment bug fix (commit `412a957`) regenerated `outputs/eye_loso_results.json` with corrected per-clip data. The Track A writeup's inverted-fold diagnostic was rewritten from the corrected ranking and identified a **fourth contributing factor** to S5/S6 inversion that was not in the original three-factor narrative:

> Three BG-labeled clips ("eyes still") ranked higher than their fold's ACT clips: `action_S5.mp4_2_` (top of S5 at +0.592), `background_S6.mp4_3_` (top of S6 at +0.637), `background_S6.mp4_2_` (second in S6 at +0.515). Plausible driver: source-correlated visual cues (head pose, framing, lighting) overlapping with ACTION-class features in the Ridge classifier's training distribution. Structural to small-N LOSO with source-distinct visual contexts.

Phase 4's locked intervention (v2 profile-aware crop + tightened-rubric blind re-label) addresses **factors (a) sub-perceptual-floor labels** and **(b) crop-positioning failure**. **Factor (c) sub-second clips** is implicitly addressed by the rubric tightening (most sub-second ACTION-labeled clips reclassify as BACKGROUND under "no zoom, no frame-stepping"). **Factor (d) — BG clips ranked above ACT clips on source-correlated training features — is NOT addressable by the Phase 4 interventions.** This is the residual structural confound for the eye-track at n = 34 with single-rater labels.

### Refined interpretation per threshold band — locked before unmask, no threshold change

The numerical thresholds locked at this document's original freeze (commit `6a89314`, hash `0b970228...`) **do not change**. What is locked here is the *interpretation* of each band given the now-known fourth factor. This is interpretation-locking, not goal-shifting; a reviewer can verify via this hash that the interpretation was committed before the unmasked re-label was processed and before the Phase 4 result was computed.

| Phase 4 outcome | Refined interpretation |
|---|---|
| **Pooled AUC ≥ 0.72** AND DeLong CI lower bound ≥ 0.55 | Factors (a)+(b)+(c) correctly diagnosed AND factor (d) was either incidentally suppressed by v2 crop's better representation OR was a smaller share of S5/S6 inversion than the corrected diagnostic feared. Track A writeup advances; collaborator pitch is "validated PoC + named residual structural confound (d) for Phase 5+." |
| **0.65 ≤ AUC < 0.72** | Factors (a)+(b)+(c) were correctly diagnosed; **factor (d) is the residual structural gap**, plausibly ~5–8 pp at this n / source-correlation regime. Phase 4 succeeded on what it could address; (d) is the bottleneck for any further AUC improvement. The explicit collaborator ask shifts from "validate the architecture" to "multi-rater κ + N expansion to dilute the source-correlation effect that drives factor (d)." **This band is NOT 'partial failure'** — it is the expected outcome under a fully-correct (a)+(b)+(c) diagnostic when (d) is non-trivial. |
| **0.55 ≤ AUC < 0.65** | Either factors (a)+(b)+(c) didn't drive the inversion as much as the diagnostic claimed, or factor (d) is larger than expected — both are regressions of the diagnostic narrative. Inconclusive at this n. |
| **AUC < 0.6313** (regression vs Phase 3) | Diagnostic narrative is wrong; the conditional 2-run ablation (locked above) runs to isolate the breaking fix. |

The original 4-band threshold table earlier in this document remains the canonical decision rule. This subsection is supplementary interpretation pre-registered under the same audit chain. No threshold values changed.

### Why this matters for the writeup framing

If Phase 4 lands in 0.65–0.72, the headline sentence is:

> "Phase 4 (v2 profile-aware crop + tightened-rubric re-label) advanced pooled AUC from 0.6813 to [X.XX] (95% DeLong CI [...]). The corrected Phase 3 inverted-fold diagnostic identified four contributing factors; Phase 4 addressed three of them (a–c) and not the fourth (d, source-correlated training-feature carryover). A result in the 0.65–0.72 band is consistent with the (a)+(b)+(c) interventions succeeding while (d) persists as the residual structural confound at single-rater n = 34. Phase 5+ work would target (d) via multi-rater κ + N expansion."

That sentence is locked here as the framing for the 0.65–0.72 outcome, before the result is known. If the result lands in another band, this subsection's framing still applies to its band — the sentence above is the canonical 0.65–0.72 framing.

## Factor-(d) suppression criterion (locked 2026-05-09, BEFORE Phase 4 LOSO run)

The corrected Phase 3 diagnostic identified factor (d) as the residual structural confound: three BG-labeled clips ranked higher than their fold's ACT clips in Phase 3 LOSO, plausibly driven by source-correlated visual cues overlapping with ACTION-class features in training. Factor (d) is structural to small-N LOSO and not addressable by Phase 4's (a)+(b)+(c) interventions. Whether v2 crop's better representation incidentally suppresses (d) on these clips, or (d) persists, is a binary question that needs a pre-committed answer rule so the interpretation cannot drift post-hoc.

### The three persistent BG-target clips

These three clips were labeled BG in Phase 3 v1 verification AND remain labeled BG in Phase 4 tightened-rubric re-label. Their Phase 4 decision-function scores are the (d) test:

- `action_S5.mp4_2_.mp4` — Phase 3 score +0.592, Phase 4 v1-rubric verdict "eyes still", Phase 4 tightened-rubric verdict "still eyes"
- `background_S6.mp4_2_.mp4` — Phase 3 score +0.515, Phase 4 v1-rubric verdict "eyes still", Phase 4 tightened-rubric verdict "still eyes"
- `background_S6.mp4_3_.mp4` — Phase 3 score +0.637, Phase 4 v1-rubric verdict "eyes still", Phase 4 tightened-rubric verdict "still eyes"

### Decision rule (locked)

After Phase 4 LOSO completes, compute the median decision-function score across all Phase 4 BG-labeled clips (n = 20 under the unmasked tightened-rubric labels). Then:

- **Factor (d) is SUPPRESSED** if **at least 2 of the 3** persistent BG-target clips above score **strictly below** the median of all 20 Phase 4 BG-labeled clips' decision-function scores.
- **Factor (d) is PERSISTENT** if **0 or 1** of the 3 persistent BG-target clips score strictly below the median.

Edge case: if the median falls exactly on one of the 3 targets' scores (rare with continuous scores from `decision_function`), that target counts as NOT below. Strict `<` only.

If the v2 tie-break-both-halves rule produces multiple embeddings for any of the 3 target clips, the score used for this criterion is the **maximum** of that clip's outputs (most action-like score). Same convention applies to "all BG clips" used for median computation: each clip contributes its maximum score regardless of how many tie-break duplicates it has. This keeps the criterion robust to the duplication rule.

### Why this matters

Without a pre-committed criterion, "factor (d) suppressed/persistent" is a post-hoc story argued from whatever score pattern lands. With this criterion, the interpretation is mechanical: count the 3 targets, count those below median, ≥2 → suppressed, ≤1 → persistent.

The criterion is recorded in the Phase 4 result JSON output (`outputs/eye_loso_results_phase4.json`) under field `factor_d_suppression`: `{"persistent_bg_clips": [...], "median_bg_score": X.XX, "n_below_median": N, "verdict": "suppressed" | "persistent"}`. Audit chain: this hash + that JSON.

## Hash chain commitment

This freeze commit registers in `docs/preregistration_hashes.md`:

- `outputs/track_b_phase4_preregistration.md` — this document (binding the design)
- `outputs/eye_relabel_blind.txt` — masked label file with locked rubric (binding the rubric content)
- `outputs/eye_relabel_keymap.json` — clip-mask → real-filename mapping (binding the masking; hash committed BEFORE user opens the masked file)
- `outputs/phase3_subject_bootstrap_ci.json` — bootstrap CI computed before any Phase 4 step

The audit chain: this commit timestamps and locks the four files together; the user re-labels AFTER the commit; subsequent Phase 4 work can be verified against these hashes by `shasum -a 256 outputs/<file>`.

## Time budget

- Re-label (user): ~ 45 min, this week
- v2 pipeline build (agent): ~ 60 min, parallel with re-label
- v2 crop extraction + V-JEPA-2 + LOSO + permutation (agent): ~ 10 min wall-clock
- Conditional ablation if needed (agent): ~ 15 min wall-clock for both runs
- Audit doc + writeup updates + commit (agent): ~ 45 min

Total wall-clock: ~ 3-4 h, mostly parallelizable. Doable today/tomorrow.
