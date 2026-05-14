# Phase 10a-prelim audit — Prudnik transfer test

**Status:** complete, verdict locked
**Date:** 2026-05-14
**Pre-registration:** `outputs/track_b_phase10a_prelim_preregistration.md` (locked 2026-05-13)
**Run logs:** `outputs/phase10a_prelim_run.log` (DLC, 14h), `outputs/phase10a_prelim_run_resume4.log` (V-JEPA-2 + scoring, ~80s)
**Result artifacts:**
- `outputs/phase10a_prelim_results.json` — headline metrics
- `outputs/phase10a_prelim_audit_extras.json` — per-clip + diagnostic
- `outputs/phase10a_prelim_vjepa2_features.npz` — features for re-analysis
- `outputs/phase10a_prelim_reliability_diagram.png` — calibration curve
- `outputs/phase10a_dlc_throughput.json` — DLC per-clip timing + MPS probe

---

## Scope statement (read this first)

Phase 10a-prelim establishes that the Phase 8b RME classifier does NOT transfer to Prudnik championship footage **as currently piped**, and identifies **upstream DLC keypoint failure as the proximate cause** (mean ear-keypoint confidence drops 54% from Phase 8b RME 0.860 → Prudnik 0.397; 66.7% of Prudnik clips fall below the 0.5 threshold the entire crop pipeline depends on).

Phase 10a-prelim **does NOT establish** whether the method would transfer with clean inputs. That question — conceptual non-transfer (Lesson 21) vs. fixable engineering — is **untestable at this DLC quality** and is explicitly deferred to Phase 10b Path A.

A reader anchoring on the headline AUC = 0.38 (below random) without reading this scope statement will misread the result. The pre-reg's `WEAK_OR_NO_TRANSFER` routing fired correctly; the diagnostic explains the verdict; the explanation is **a** cause, not the **only** cause.

---

## Headline (n=90)

| Metric | Value | Interpretation |
|---|---|---|
| n (action / background) | 90 (74 / 16) | 82/18 class split — heavy imbalance vs Phase 8b's ~50/50 |
| Pooled AUC | **0.3843** | Below 0.5; CI crosses 0.5 → "no signal," not "anti-signal" |
| 95% CI (clip bootstrap, B=10000) | **[0.2272, 0.5528]** | Wide CI, crosses 0.5 |
| Permutation p (vs chance) | **0.9121** | No detectable signal |
| ECE pre-calibration | 0.3923 | High — calibration broken before T-scaling |
| ECE post-calibration (T_median=0.494) | **0.4664** | **WORSE post-calibration** — see §Calibration pathology |
| Pre==post AUC (G6 invariance gate) | PASS (\|Δ\|=0.00e+00) | T-scaling correctly preserves ranking |
| **Verdict band (D6)** | **WEAK_OR_NO_TRANSFER** | AUC < 0.60 threshold |
| **Routing decision (D6)** | **PAUSE labeling. Diagnose first.** | Pre-reg matrix routed correctly |

The 3-band verdict structure (STRONG ≥0.75 / AMBIGUOUS [0.60, 0.75) / WEAK <0.60) and locked routing per D6 worked exactly as designed. Preliminary read on n=90 saved an investment of ~3 days of labeling on the remaining ~180 derived rows that would have shipped a null-AUC dataset.

---

## Sanity checks (pre-reg-locked)

| # | Check | Status | Notes |
|---|---|---|---|
| 1 | Classifier bit-exact reproducibility (manual decision_function vs sklearn) | PASS | max \|Δ\| = 0.00e+00 across 3 sample clips |
| 2 | DLC walltime ≤ 2× expected (warn if exceeded) | **FIRED** | 50,622s / 2,790s expected = **18.1× ratio**. Cause: portrait orientation + Faster R-CNN CPU fallback (see MPS probe in `outputs/phase10a_dlc_throughput.json` and Lesson 22 candidate notes) |
| 3 | Operating-point verification (τ_ear sensitivity) | DEFERRED | Per pre-reg G3: n_neg=16 makes τ_ear verification meaningless (expected FP ≈ 0.8). Locked decision. |
| 4 | First-clip end-to-end trace | PASS (mechanically) | IMG_1103 (background, medium conf): DLC frames=29 (7 fallback fired), V-JEPA-2 dim=1024 mean=0.0401 std=1.7098, decision_score=-1.3098, prob_post=0.0659 → predicted background → matches label. Mechanism worked on this clip; AUC failure is cohort-wide systematic, not single-clip bug. |

---

## DLC diagnostic — the upstream failure

The headline AUC is explained by an upstream DLC keypoint quality collapse. This was the most decisive ~5 min diagnostic available; it ran post-verdict before subgroup analysis to separate upstream-vs-downstream failure modes.

### Per-keypoint confidence comparison (Prudnik vs Phase 8b RME)

| Ear keypoint | Cohort | n samples | mean | median | p25 | ≥0.5 | ≥0.9 |
|---|---|---|---|---|---|---|---|
| right_earbase | Prudnik | 42,505 | 0.387 | 0.540 | 0.223 | 54.4% | 0.9% |
| right_earbase | Phase 8b RME | 7,125 | 0.846 | 0.871 | 0.813 | **97.6%** | 33.2% |
| right_earend | Prudnik | 42,505 | 0.383 | 0.505 | 0.143 | 50.4% | 8.1% |
| right_earend | Phase 8b RME | 7,125 | 0.884 | 0.919 | 0.874 | **96.9%** | 62.7% |
| left_earbase | Prudnik | 42,505 | 0.404 | 0.583 | 0.244 | 60.0% | 0.5% |
| left_earbase | Phase 8b RME | 7,125 | 0.809 | 0.819 | 0.774 | **98.8%** | 10.6% |
| left_earend | Prudnik | 42,505 | 0.415 | 0.569 | 0.151 | 55.5% | 10.6% |
| left_earend | Phase 8b RME | 7,125 | 0.902 | 0.932 | 0.891 | **98.3%** | 71.8% |

**Aggregate (170k Prudnik samples, 28k RME samples):**

- Prudnik mean = 0.397, ≥0.5 = 55.1%, ≥0.9 = 5.0%
- Phase 8b RME mean = 0.860, ≥0.5 = 97.9%, ≥0.9 = 44.6%
- **Δ_mean = −0.463 (−54% relative)**

**Per-clip mean ear confidence:**

- Phase 8b RME (n=283): mean = 0.853, median = 0.877, only **1.4% of clips below 0.5**
- Prudnik (n=90): mean = 0.369, median = 0.425, **66.7% of clips below 0.5** (28.9% below 0.3)

The pipeline's `EAR_CONF_THRESHOLD = 0.5` is hardcoded at the boundary where most Prudnik clips fall. When `confident_indices` returns empty for a frame, `compute_ear_bboxes` falls back to the clip-level fallback frame's bbox — meaning many frames are cropped using a generic position rather than per-frame ear localization. V-JEPA-2 then encodes those generic crops, producing features that bear no relationship to actual ear motion. The classifier scores noise.

**This is the upstream failure.** The classifier is not being tested on ear motion; it is being scored on garbage crops on a majority of Prudnik clips.

### Subgroup analysis (positive findings, not nulls)

#### By frame_type

| Subgroup | n | mean per-clip conf | <0.5 |
|---|---|---|---|
| head-zoom | 16 | 0.406 | 11/16 |
| full-body | 74 | 0.361 | 49/74 |

**Finding: making the horse bigger in frame barely helped.** Head-zoom shows only +4.5pp over full-body, and both subgroups remain mostly below the 0.5 threshold. If pure scale/distance were the dominant degradation factor, head-zoom should have substantially rescued DLC. It didn't.

This is **modest evidence against "just zoom in more"** as the fix and **modest evidence for compounding factors** (handheld camera shake, motion blur from championship-speed motion, codec artifacts) acting independent of apparent horse size. The mechanism is more than scale.

n=16 caveats the head-zoom subgroup statistically; the finding is "head-zoom did not produce a recovery commensurate with the scale change," not "head-zoom is equivalent to full-body."

#### By multi_horse (loose end, not finding)

| Subgroup | n | mean per-clip conf | <0.5 |
|---|---|---|---|
| single horse (multi_horse=0) | 59 | 0.344 | 40/59 |
| multi horse (multi_horse=1) | 31 | 0.416 | 20/31 |

Multi-horse clips show **higher** mean DLC confidence than single-horse — the opposite of what naive intuition predicts. The most plausible explanation is a framing-distance confound (single-horse Prudnik clips happen to be wider arena shots when one horse runs solo, while multi-horse clips happen to be closer paddock/warm-up framings where competitors are visible together). This was not controlled for in the original Prudnik clip selection.

**Logged as a loose end, not a finding.** Investigation deferred; relevant for Phase 10b if a Path B/C detector swap is considered (because detector behavior on small-in-frame vs close-in-frame horses likely differs).

#### By label

| Subgroup | n | mean per-clip conf | <0.5 |
|---|---|---|---|
| action | 74 | 0.378 | 49/74 |
| background | 16 | 0.327 | 11/16 |

Background clips show marginally lower confidence than action clips. Plausibly because background (no ear motion → potentially head-down, ear-occluded) framings happen to be slightly worse for ear detection. Not statistically significant at n=16; recorded for completeness.

---

## Calibration pathology — ECE worsening explanation

**Observation:** ECE pre-calibration = 0.3923; ECE post-calibration (T_median = 0.494) = 0.4664. Calibration got **worse** after applying the temperature scaling fitted on Phase 8b LOSO scores.

**Two entangled mechanisms, both supported:**

**(a) Input distribution drift.** The temperature scaling transform was fitted on Phase 8b LOSO scores produced by feeding clean ear crops through V-JEPA-2 and the classifier. When the Prudnik pipeline feeds garbage crops through the same forward path, the score distribution shape is different — likely more compressed (closer to 0 because all features look similar to V-JEPA-2 when ears aren't actually in the crop). T<1 sharpens distributions; sharpening a compressed distribution produces predicted probabilities concentrated near 0/1 that don't match an empirical class rate of 82/18.

**(b) Base-rate calibration pathology.** Even with clean inputs, applying a T-scaling transform fitted on a ~50/50 distribution (Phase 8b deliberate balance: 138 action / 145 background) to an 82/18 distribution (Prudnik selection: 74 action / 16 background) produces systematic ECE inflation. On 82/18 data, the optimal calibrated output for an uninformative classifier is "predict majority class with probability ≈ 0.82." Temperature sharpening pushes scores toward 0/1, away from this base-rate-aware optimum. ECE penalizes this displacement.

**Key consequence (load-bearing for Phase 10b):** mechanism (b) would **persist even with perfectly clean DLC inputs**. If Path A (manual ear bboxes on n≈20) runs and the resulting AUC is reported alongside calibration metrics, the ECE comparison alone cannot distinguish "DLC drift residual" from "base-rate pathology." Phase 10b Path A must therefore report **uncalibrated AUC + uncalibrated decision-score distribution** alongside calibrated metrics. The AUC is invariant to monotonic transforms (per G6 PASS), so uncalibrated AUC reflects only the discriminative-signal question and isolates the methodology preservation claim from the calibration-transform question.

This is a **measurement specification** for Phase 10b, not just a caveat.

---

## Limitations status (per pre-reg)

| ID | Pre-reg limitation | Status after Phase 10a-prelim |
|---|---|---|
| L1 | Selection bias on labeled subset (easy/representative clips labeled first) | **REVISITED.** L1 remains load-bearing. n=90 is an upper bound on full Phase 10a-full AUC if Phase 10b doesn't fix DLC. With DLC failing on 66.7% of clips even at this selection, the full-cohort transfer is plausibly worse, not better. |
| L2 | Single-source transfer test (one venue, no LOSO) | Unchanged. |
| L3 | RME-style labeling protocol verbatim (no Prudnik-specific adaptation) | Unchanged. |
| L4 | Calibration imported from Phase 8c (not re-fit on Prudnik) | **EMPIRICALLY SUPPORTED.** ECE worsening (0.39 → 0.47) is consistent with the pre-reg's L4 hypothesis. See §Calibration pathology for full mechanism. |
| L5 | Clip-bootstrap CI only (no LOSO CI possible at n=1 source) | Unchanged. |
| L6 | Portrait orientation distribution shift hypothesis | **PARTIALLY SUPPORTED, MORE NUANCED THAN ORIGINALLY HYPOTHESIZED.** The pre-reg's framing was "portrait orientation may distribution-shift V-JEPA-2 features." The diagnostic instead shows: portrait + scale + handheld shake + codec compound to break **DLC** specifically. V-JEPA-2 feature drift cannot be assessed until DLC produces real ear crops to feed in. |
| L7 | Sub-V-JEPA-2 floor handling (only 1 clip filtered) | Unchanged. n=89 → 90 actually (one clip became viable post-filtering). |
| L8 | Sample size adequacy (n=90 vs Phase 8b n=283) | **REVISITED.** Adequate to deliver the WEAK band verdict + diagnostic. Inadequate for hypothesis testing within the band. The pre-reg's smoke-test framing was correct: this n is for routing, not for fine-grained inference. |

---

## Compute / device finding (cross-reference)

Per `outputs/phase10a_dlc_throughput.json`:

- DLC per-frame rate: 1.18 s/frame mean (median 1.16, range [1.13, 1.39]) on 90 clips
- 50,622s total DLC walltime (~14h) — Sanity #2 fired at 18.1× the original 2,790s estimate
- MPS probe (May 13) established that the bottleneck is **Faster R-CNN detector silently falling back to CPU** because of MPS-incompatible ops in DLC 3.0.0rc14
- The 10× compute gap vs Phase 8b RME (31.1 s/clip landscape) is **detector-architecture-specific, NOT orientation-fundamental** — pose-stage hrnet_w32 runs ~6× faster on MPS when alone

**This is recorded as a separate methodological finding from the Phase 10a-prelim transfer-test result.** The DLC compute cost is not what made the classifier fail to transfer; it's just what made the diagnostic slow.

---

## Phase 10b proposal — Path A first as diagnostic gate

The pre-reg's WEAK_OR_NO_TRANSFER routing said "Diagnose first." This audit doc's diagnosis says: **upstream DLC failure is the proximate cause; conceptual non-transfer remains untestable until DLC is fixed**.

Phase 10b therefore has a sequencing principle: **diagnostic test first, production fix conditional on its result**.

### Path A — manual ear-bbox diagnostic (n≈20, diagnostic-first)

**Purpose:** bypass DLC entirely on a small Prudnik subset with hand-labeled ear bboxes. Test the question: *with clean ear crops, does the Phase 8b classifier transfer?*

**Outcomes that distinguish hypotheses:**

| Path A result | Interpretation | Phase 10b next step |
|---|---|---|
| AUC ≥ 0.70 (uncalibrated) | Method is sound; DLC reliability is the engineering problem | Build Path B (better detector) for production |
| AUC ∈ [0.55, 0.70) | Partial transfer; method works on a real signal but performance gap remains | Investigate residual drift; Path B + scope refinement |
| AUC ~0.50 | **Conceptual non-transfer confirmed.** Even with perfect crops, method does not transfer. | Method-level rethink; potential Lesson 21 escalation to thesis framing decision |
| AUC < 0.50 anti-signal | Genuine label inversion (extremely unlikely) | Investigate label protocol or feature inversion |

**Locked measurement spec:**

1. **Uncalibrated AUC reported as primary metric** alongside calibrated AUC (G6 invariance gate preserved; AUC reflects discriminative signal independent of calibration transform pathology).
2. **Uncalibrated decision-score distribution histogram** alongside reliability diagram. The distribution shape diagnoses whether garbage crops produced score compression (mechanism (a) from §Calibration pathology) vs the base-rate pathology (mechanism (b)).
3. Per-clip score logging on both calibrated and uncalibrated paths for downstream investigation.

**Locked clip selection criterion (stratified, anti-selection-bias):**

The temptation is to pick the n≈20 highest-DLC-confidence Prudnik clips. **Do not do this.** It biases toward easy footage and produces an optimistic AUC that doesn't generalize. Instead, stratified selection across:

- **frame_type** — both head-zoom and full-body subgroups represented (target ≥6 of each)
- **DLC confidence range** — span low (<0.3), mid (0.3-0.5), high (>0.5) confidence clips (target ≥6 across the range; do NOT cluster at the high end)
- **label balance** — both action and background represented in roughly the Prudnik cohort ratio
- **multi_horse status** — both included (test whether framing-distance confound from §Subgroup affects manual-crop transfer)

This is the same selection-bias trap as L1 on the original n=90 selection. The Path A scoping document (forthcoming) will lock the specific 20 clips and rationale before any labeling begins, so it's a deliberate decision rather than an accident.

**Cost estimate:** ~1-2 hours of manual bbox-drawing attention. ~30 minutes pipeline run on cached features (V-JEPA-2 reuse) + ad-hoc scoring.

### Conditional Paths B/C/D (production fix, gated on Path A)

| Path | Mechanism | Status |
|---|---|---|
| **B. Stronger horse-detector pre-crop** | YOLO/horse-detector for tight bbox, then DLC on the crop region | Conditional on Path A showing AUC ≥ 0.55. Pure engineering work, ~1-2 days. |
| **C. DLC `pcutoff` tuning** | Lower confidence threshold to admit more keypoints | **NOT a real fix** per critic — moves garbage past the gate rather than fixing it. May run as a quick data point for completeness, but not a candidate production fix. |
| **D. Different pose model variant** | DLC SuperAnimal variants trained on diverse-context data, or alternative models entirely | Conditional on Path A + B showing residual issues. Research time. |

**Path A is not a slower alternative to Path B.** It is the gating test that determines whether Path B is worth building.

---

## Decisions

**D1.** Phase 10a-prelim verdict band is WEAK_OR_NO_TRANSFER per pre-reg D6. Locked.

**D2.** Phase 10a-full labeling investment is PAUSED per pre-reg routing matrix. Locked.

**D3.** Phase 10b proceeds with Path A (n≈20 manual-bbox diagnostic) as the next preliminary read. Selection stratified per §Phase 10b proposal. Locked measurement spec: uncalibrated AUC + score distribution alongside calibrated metrics.

**D4.** Cloud-DLC migration (Modal CUDA Phase 8b retrain + tolerance-equivalence chain) is **DEFERRED** to Phase 11+. Rationale: CUDA does not address a model that cannot find ears on its native compute substrate. The cloud-DLC scaffolding (`tools/cloud_dlc/`) remains valuable as parked infrastructure; see Lesson 23 candidate below.

**D5.** ECE worsening interpretation is locked to two mechanisms (input distribution drift + base-rate calibration pathology). Phase 10b Path A measurement spec includes uncalibrated metrics specifically to separate the two.

---

## Lesson 23 candidate (compute-substrate detour producing a diagnostic instrument)

The cloud-DLC migration work (Modal CUDA scaffolding, smoke tests, FAILURE_PLAYBOOK, audit footnotes) was originally scoped as a throughput fix for Phase 10a-full. During its execution, the May 13 MPS probe established that DLC's xy keypoints are device-stable across CPU vs MPS at 0.0 px max delta — which, when juxtaposed with Phase 8b's RME DLC outputs available in `outputs/phase8b_rme_dlc_keypoints.json`, **enabled the Prudnik-vs-RME confidence comparison that diagnosed the Phase 10a-prelim transfer failure**.

The diagnostic instrument that determined Phase 10b's direction (Path A first) was produced as a side effect of compute-substrate due diligence, not as a planned diagnostic deliverable.

**Generalizable rule (Lesson 23, draft):** discipline-pattern investments forced by methodology rigor can produce instruments that pay forward beyond their original scope. The cloud-DLC smoke-test required a quantitative DLC keypoint comparison; that comparison framework was then immediately reusable for diagnosing a different question (transfer failure root-cause) that had not been anticipated at the time of the smoke-test design. The lesson is not "always build extra diagnostic tooling speculatively"; the lesson is "methodology-rigor investments often have higher information value than their stated purpose."

To be promoted to a full Lesson 23 entry in `docs/lessons_learned.md` at the Phase 10b transition commit (when the cloud-DLC compute-substrate work formally re-enters scope or formally exits to Phase 11+).

---

## Audit chain

- Pre-registration: `outputs/track_b_phase10a_prelim_preregistration.md` (locked 2026-05-13, hash recorded in `docs/preregistration_hashes.md`)
- Deployed classifier: `outputs/phase10a_prelim_deployed_classifier.json` (Phase 10a-prelim Step 1.5; in-sample RME AUC = 1.0000 per Sanity #1)
- DLC throughput finding: `outputs/phase10a_dlc_throughput.json` (sanity #2 firing + MPS probe + detector-architecture attribution)
- Cloud-DLC audit footnotes (Phase 11+ deferred material): `outputs/cloud_dlc_audit_footnotes.md`
- Lesson 21 (RME ≠ RHpE Behavior #7): `docs/lessons_learned.md`
- Lesson 22 (DLC `__version__` not load-bearing): `docs/lessons_learned.md`
- Lesson 23 candidate (compute-substrate detour producing diagnostic instrument): drafted in this audit doc §Lesson 23 candidate; pending promotion at Phase 10b transition commit

## Bugs surfaced and fixed during Phase 10a-prelim execution

Documented for audit transparency — these were latent bugs in `tools/phase10a_prelim_run.py` discovered by end-to-end execution. None affected Phase 8b RME results because the buggy code paths had never been executed end-to-end before today:

1. `confident_indices(kps, threshold=...)` — invalid kwarg in Phase 10a's call site. Phase 8b's `confident_indices(kps)` uses module-level `CONF_THRESHOLD=0.5` matching Phase 10a's `EAR_CONF_THRESHOLD=0.5`. Surgical fix: drop the kwarg. No methodology change.
2. `keypoints_to_phase8b_format` — produced 4-element dict-list; `confident_indices`/`enclosing_rect` expect 13-element tuple-list indexed at positions 6,7,11,12. Surgical fix: rewrite to return 13-element tuple-list with placeholder `(0,0,0)` tuples at non-ear positions. Unit-tested end-to-end.
3. `select_clip_fallback_frame` return value treated as scalar — function returns `(frame_idx, confident_indices)` tuple. Surgical fix: unpack at the call site to match Phase 8b's own usage.
4. `parse_dlc_h5_for_ears` — `df.xs(individual, level="individuals")` did not drop the scorer level; subsequent `row[(bp, "x")]` lookup hit KeyError silently swallowed by `try/except`, producing empty per-frame dicts. Surgical fix: explicit `df_p.droplevel("scorer", axis=1)` before bodypart access.

The bugs were caught by surgical iteration over the run; each fix preserved Phase 8b's locked semantics (same thresholds, same indices, same fallback selection logic). The pattern of latent bugs in `phase10a_prelim_run.py` indicates the file was never executed end-to-end before today — every prior attempt died at the DLC stage. **The DLC failures were doing the project a favor by hiding the chain of latent issues; they surfaced only when DLC was finally cached and the downstream pipeline ran.**
