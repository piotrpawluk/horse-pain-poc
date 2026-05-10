# Phase 8b audit — Cross-behavior generalization on RME ear data

**Date**: 2026-05-10. Branch: `experiment/phase8b`. Cross-behavior
generalization test of DLC keypoint-anchored cropping (eye → ear).
Tests whether the DLC + V-JEPA-2 + linear probe methodology preserves
the discriminative ear-movement signal from RME relative to the
project's whole-frame V-JEPA-2 baseline (LOSO 0.875 from
`outputs/vjepa2_embeddings.npz`).

> **⚠ Framing note (carried from Phase 8b Stage 1 pre-reg)**
>
> Phase 8b's RME baseline reference is **whole-frame V-JEPA-2** (notebook
> 02 → cached features → Sanity 5 ssv2_motion LOSO 0.875), NOT
> custom-trained YOLO ear-cropped V-JEPA-2. The custom-trained YOLO ear
> detector exists in vendor/ and is used in notebook 01 to reproduce
> the paper's optical-flow pipeline, but is not part of the project's
> V-JEPA-2 reference baseline.
>
> This means Phase 8b tests "does DLC ear-keypoint cropping preserve
> whole-frame V-JEPA-2's signal?" — a HARDER test than competing
> against custom-trained YOLO would have been. The numeric gates
> (≥0.80 / 0.65-0.80 / <0.65) remain UNCHANGED per goal-shifting
> warning; interpretation shifts under the corrected framing.

## Headline result — COMPETITIVE

**Joint verdict (per locked 7-row matrix): COMPETITIVE.**
G1 strong + G4 inconclusive at n=283 with adequate paired-test power =
**methodology preserves baseline signal; does not significantly improve
baseline.**

This is not a "DLC ear cropping is strong" result. The locked verdict-
reporting protocol exists specifically to prevent that anti-pattern.
At n=283 the paired DeLong test has reasonable statistical power; p=0.31
is "the data does not support a significant improvement claim, with
adequate power to detect one if it existed" — not "underpowered."
Structurally different from Phase 7/8a's n=34 SE-inflation problem.

| Metric | Value |
|---|---|
| Phase 8b pooled AUC | **0.9008** |
| Whole-frame baseline AUC (Sanity 5 ssv2_motion) | 0.8746 |
| Δ vs whole-frame (aligned, 283 clips) | **+0.0262** |
| Subject-bootstrap CI on Δ (B=10000) | **[−0.0285, +0.0732]** (straddles 0) |
| Paired DeLong p vs whole-frame | **0.3122** (inconclusive) |
| Per-source +Δ count | **4/12** (well below ≥9/12 robustness floor) |
| Permutation p vs chance | 0.0010 (signal real) |
| **G1−G3 verdict** | **STRONG (PASS)** |
| **G4 paired-DeLong verdict** | **INCONCLUSIVE (FAIL)** |
| **Joint verdict** | **COMPETITIVE** |

## Pre-Step-4 gates cleared

- ✓ **Checkpoint #1**: Stage 1 pre-reg approved
- ✓ **Checkpoint #2**: Step 1 RME parity check PASS_BIT_EXACT (deviation
  0.00e+00)
- ✓ **Checkpoint #3**: Stage 1.5 ear keypoint inspection PASS (9/10 =
  90%, gate ≥80%×80% cleared by margin)
- ✓ **Checkpoint #4**: LOSO + diagnostic verdict approved with
  audit-doc-framing requirements (this document responds)

## Stage 1.5 inspection — pre-Step-4 ear-keypoint reliability

(Full results in `outputs/phase8b_stage15_inspection.json`.)

DLC SuperAnimal-Quadruped's ear keypoints on the locked stratified
sample (10 RME clips, alphabetical-first action+bg per source from
S1, S5, S8, S10, S12):

```
9/10 clips meet ≥80% per-clip frame threshold (≥3-of-4 ear keypoints
confident at 0.5).

Per-clip pass rates:
  action_S1.mp4_0_      100.0%  ✓
  background_S1.mp4_0_   71.0%  ✗  ← single sub-gate clip
  action_S5.mp4_0_      100.0%  ✓
  background_S5.mp4_0_  100.0%  ✓
  action_S8.mp4_0_      100.0%  ✓
  background_S8.mp4_0_  100.0%  ✓
  action_S10.mp4_0_     100.0%  ✓
  background_S10.mp4_0_ 100.0%  ✓
  action_S12.mp4_0_     100.0%  ✓
  background_S12.mp4_0_ 100.0%  ✓

Pooled ear keypoint confidence (n=244 frames):
  right_earbase   median 0.842  mean 0.828  3.7% below 0.5
  right_earend    median 0.908  mean 0.879  4.1% below 0.5
  left_earbase    median 0.799  mean 0.801  0.0% below 0.5
  left_earend     median 0.931  mean 0.909  0.4% below 0.5
```

**Reference-floor cross-check**: 33/34 (97.1%) of Phase 5 RME-population
clips also meet the gate; mean ear keypoint confidences 0.82–0.93 match
Stage 1.5's 0.80–0.91 within sampling variation.

**Bimodal pass-rate pattern**: 9 clips at 100%, 1 at 71%, no gradual
middle. Stage 1.5 prediction: "DLC's per-clip behavior on RME is
'reliably good or reliably struggling,' not 'uniformly noisy.'"
Population-scale Step 5 result confirms this pattern (see below).

## Step 4: full RME DLC inference

(Full results in `outputs/phase8b_rme_dlc_keypoints.json`.)

- **283/283 clips processed, 0 failed.**
- Wall time: 8813.3s (2h 27m), 31.1 sec/clip avg.
- DLC version 3.0.0rc13. Locked params (carry forward from Phase 7):
  superanimal_name=`superanimal_quadruped`, model_name=`hrnet_w32`,
  detector_name=`fasterrcnn_resnet50_fpn_v2`, video_adapt=False,
  pseudo_threshold=0.1, create_labeled_video=False.

## Step 5–6: crop pipeline + V-JEPA-2 extraction

(Manifest: `outputs/eye_crops_v4_ear_dlc_manifest.jsonl`; embeddings:
`outputs/vjepa2_embeddings_ear_v4.npz`.)

**Population-level crop pipeline statistics:**

- 283/283 ok, 0 fail
- 257/283 (90.8%) per-frame-only (no fallback fired)
- 26/283 (9.2%) used single-middle-frame fallback per locked Decision 1
- Stage 1.5 sample showed 9/10 (90%) pass; population matches sample
  within 1pp — bimodal pattern from Stage 1.5 confirmed

**V-JEPA-2 extraction:**

- 283 × 1024-d embeddings, parity-check passed
- Wall time: 256s

### Named fallback-clip table (locked Decision 1 verification)

26 clips fired the single-middle-frame fallback per locked rule. Per-
clip table confirms the rule was applied mechanically; no manual
override.

**By source distribution:**

| Source | # fallback clips | Heavy (≥50% fallback) |
|---|---:|---:|
| S4 | 7 | 4 |
| S2 | 5 | 1 |
| S9 | 5 | 1 |
| S1 | 2 | 0 |
| S3 | 2 | 0 |
| S8 | 2 | 0 |
| S5 | 1 | 0 |
| S7 | 1 | 0 |
| S12 | 1 | 0 |

S4 dominates the heavy-fallback cluster (4 of 6 clips at ≥50% fallback
are S4). Possible source-specific cause (lighting, camera angle,
horse-face occlusion) — flagged as observation, not actioned in this
audit.

**Heavy fallback table (≥50% fallback):**

| Clip | Source | Label | Fallback % | Frames written |
|---|---|---|---:|---:|
| `action_S4.mp4_9_.mp4` | S4 | action | 100.0% | 6 |
| `background_S4.mp4_7_.mp4` | S4 | background | 95.2% | 21 |
| `action_S9.mp4_2_.mp4` | S9 | action | 90.9% | 22 |
| `action_S4.mp4_10_.mp4` | S4 | action | 62.5% | 8 |
| `background_S2.mp4_6_.mp4` | S2 | background | 52.6% | 38 |
| `action_S4.mp4_7_.mp4` | S4 | action | 50.0% | 6 |

**Locked Decision 1 verification:** all 26 fallback clips applied the
rule "if <3 of 4 ear keypoints confident in a frame, fall back to the
clip-level frame with most confident ear keypoints (≥3 of 4 confident
preferred; else all 4 regardless of confidence)." No clip-level overrides
were issued. Full per-clip table in `outputs/phase8b_audit_extras.json`.

## Step 7–8: LOSO + diagnostic

(LOSO result: `outputs/eye_loso_results_phase8b.json`; diagnostic:
`outputs/phase8b_diagnostic.{json,md}`; extras:
`outputs/phase8b_audit_extras.json`.)

### Test hierarchy results (per Phase 8b §"Test hierarchy")

**Load-bearing — AUC-vs-gate (G1–G3):**
- Phase 8b pooled AUC: **0.9008**
- Subject-bootstrap 95% CI: [0.8608, 0.9406]
- Permutation p (vs chance): 0.0010
- **Locked gate verdict: G1 STRONG (PASS)**

**Supportive — paired DeLong vs whole-frame baseline (G4):**
- n=283 (no clips dropped from alignment)
- Paired DeLong: Δ=+0.0262, z=1.011, **p=0.3122**
- **G4 verdict: INCONCLUSIVE** (does not reject H₀ at α=0.05)

**Precision — subject-bootstrap CI on Δ vs whole-frame:**
- 95% CI: [−0.0285, +0.0732] (straddles 0)
- Median Δ: +0.0262
- P(Δ ≤ 0) = 15.8%

### Bootstrap distribution shape on Δ

(Per audit-doc requirement #2.) Subject-bootstrap distribution of
Δ AUC = AUC(Phase 8b) − AUC(whole-frame), B=10000, source-resampled
with seed=42.

| Statistic | Value |
|---|---:|
| n_kept | 10000 |
| mean | +0.0253 |
| median | +0.0262 |
| std | 0.0259 |
| min | −0.075 (approx) |
| max | +0.119 (approx) |
| Q25 | +0.0086 |
| Q75 | +0.0431 |
| CI 95% | [−0.0285, +0.0732] |
| Skewness (Fisher–Pearson g₁) | **−0.196** |
| P(Δ ≤ 0) | 0.1582 |

Distribution is approximately symmetric (|skew| < 0.5; standard rule of
thumb for "near-symmetric"). Mean ≈ median ≈ +0.026 reinforces
symmetry. The IQR [+0.009, +0.043] is shifted slightly positive but
the lower tail crosses zero substantially. 84% of bootstrap samples
yield positive Δ; 16% yield negative Δ. Symmetry rules out the
"asymmetric one-sided improvement" interpretation that would otherwise
have softened the COMPETITIVE verdict.

### Per-source AUC breakdown (named, with Δ direction)

(Per audit-doc requirement #1.)

| Source | Phase 8b AUC | Whole-frame AUC | Δ | Direction |
|---|---:|---:|---:|---|
| S1 | 0.7959 | 0.8163 | −0.0204 | − |
| S2 | 1.0000 | 0.9267 | +0.0733 | **+** |
| S3 | 0.9688 | 0.9948 | −0.0260 | − |
| S4 | 0.8958 | 0.9115 | −0.0156 | − |
| S5 | 0.9740 | 0.9026 | +0.0714 | **+** |
| S6 | 0.8222 | 0.9556 | −0.1333 | − |
| S7 | 0.7500 | 1.0000 | −0.2500 | − |
| S8 | 0.9844 | 0.6328 | +0.3516 | **+** |
| S9 | 0.8322 | 0.7832 | +0.0490 | **+** |
| S10 | 1.0000 | 1.0000 | +0.0000 | tie |
| S11 | 0.7955 | 0.9205 | −0.1250 | − |
| S12 | 0.8929 | 1.0000 | −0.1071 | − |

**4/12 sources show positive Δ** (S2, S5, S8, S9). One tie (S10, both
1.0). Seven negative-Δ sources, one of which (S7) is a hard regression
(−0.25). This is well below the ≥9/12 robustness floor convention
(carried from Phase 8a Test 2 framework).

The +0.026 numerical advantage is concentrated, not distributed:
S8 alone contributes Δ +0.3516, doing most of the heavy lifting in the
pooled aggregate. Without S8, the pooled Δ would likely flip negative
(not formally computed — flagged for Phase 9 if desired). This is the
distributional pattern of "fragile aggregate advantage" rather than
"methodology consistently helps."

**Source-level interpretation:** if a methodology genuinely improved
over whole-frame, the expected per-source Δ distribution would
concentrate positive (≥9/12, Phase 8a Test 2 convention). 4/12 with
heavy negative tails is the signature of "competitive but
indistinguishable across the source population" — which is exactly
what the COMPETITIVE joint verdict means.

### Per-clip 4-way joint categorization

(Per audit-doc requirement #3.) At threshold 0 (RidgeClassifier
decision_function default boundary):

| Category | Count | % |
|---|---:|---:|
| BOTH_RIGHT | 189 | 66.8% |
| BOTH_WRONG | 15 | 5.3% |
| DLC_BEATS_WHOLE_FRAME | 40 | 14.1% |
| WHOLE_FRAME_BEATS_DLC | 39 | 13.8% |

**40 vs 39 — methods win and lose on each other's clips at almost
identical rates.** This is the cleanest direct evidence of
"competitive but indistinguishable." If DLC ear cropping genuinely
improved over whole-frame, DLC_BEATS_WHOLE_FRAME would substantially
exceed WHOLE_FRAME_BEATS_DLC. Instead, the two categories are
within 1 clip of each other (40 ≈ 39).

The +1 numerical edge for DLC at the per-clip level corresponds to
the +0.0262 Δ AUC at the pooled level — both measure the same
small, statistically-not-significant advantage.

BOTH_WRONG = 15/283 (5.3%) is the irreducible-error floor: clips
neither methodology can call. BOTH_RIGHT = 189/283 (66.8%) is the
shared-success base — both methodologies agree on the easier clips,
then split nearly evenly on the harder ones.

## Joint verdict (per locked 7-row matrix)

| G1–G3 | G4 | Joint reading |
|---|---|---|
| Strong (≥0.80) | Positive sig | OUTPERFORM (best case) |
| Strong | **Inconclusive** | **COMPETITIVE** ← **THIS RESULT** |
| Strong | Negative sig | STRONG_BUT_DEGRADES |
| Modest (0.65–0.80) | Positive sig | UNLIKELY_COMBINATION (flag) |
| Modest | Inconclusive | MODEST_INCONCLUSIVE |
| Modest | Negative sig | MODEST_DEGRADES |
| Fails (<0.65) | (any) | FAILS |

**Joint verdict: COMPETITIVE.** Strong absolute AUC (0.9008 ≥ 0.80);
paired DeLong test inconclusive (p=0.3122) at n=283 with adequate
power. DLC ear-keypoint-anchored cropping is statistically
indistinguishable from whole-frame V-JEPA-2 on RME ear data at the
n=283 scale.

## Phase 9 entry conditions (locked, from Phase 8a routing matrix)

Phase 8a fired RETRACTION; therefore the Phase 9 priority depends on
Phase 8b outcome alone among these three rows (per Phase 8a Stage 1
routing matrix):

| 8b outcome | Phase 9 priority |
|---|---|
| **Strong (≥0.80)** ← THIS | **Multi-rater κ when 2nd rater available; methodology generalization established despite Phase 7 retraction** |
| Modest (0.65–0.80) | N expansion + multi-rater κ; methodology generalizes with caveats |
| Fails (<0.65) | Eye-specific narrow narrative; methodology doesn't generalize without behavior-specific validation |

**Phase 9 priority (post-8b): multi-rater κ when 2nd rater available;
methodology generalization established despite Phase 7 retraction.**

The locked routing matrix does not adjust for the COMPETITIVE-vs-
OUTPERFORM nuance — both fall into the "Strong" row and route to the
same Phase 9 priority. Respect the lock.

### Reframed Phase 9 motivation under COMPETITIVE outcome

The original Phase 9 question was "does cropping methodology
generalize?" — answered as "yes, in absolute terms (G1 STRONG)." The
COMPETITIVE outcome reframes the *next* Phase 9 question:

> Given that explicit cropping does not significantly improve over
> whole-frame on RME ear data at adequate sample size, what is the
> methodology's value proposition?

Possible value propositions worth investigating in Phase 9 (flagged
for forward-look, not requiring resolution here):

- **Computational efficiency.** 224×224 crops process faster than
  full-frame V-JEPA-2 forward passes. If real-world deployment needs
  sub-second per-clip latency on ROI-relevant behaviors, cropping has
  practical value independent of LOSO AUC.
- **Interpretability.** Cropped regions tied to anatomical keypoints
  provide explanation surface ("the model attended to the ear region")
  that whole-frame does not. Useful for clinical deployment scrutiny.
- **Behavior-specific localization.** When behaviors require focal
  attention (subtle ear position differences, eye microexpressions),
  cropping may matter even when average AUC does not reflect it.
  Per-source Δ heterogeneity (S8 +0.35 vs S7 −0.25) hints at this —
  cropping helps on some sources, hurts on others.
- **Robustness to non-target subjects.** Whole-frame is vulnerable to
  second-horse contamination (Lesson 9). Cropping is conditionally
  robust to this contamination.

These benefits exist independently of LOSO AUC improvement and are
candidate Phase 9 questions, not defensive rationalization for the
COMPETITIVE outcome.

## What Phase 8b establishes

- **DLC keypoint-anchored cropping preserves whole-frame V-JEPA-2
  baseline performance** on RME ear data at n=283. The methodology
  travels across behaviors (eye → ear) without significant degradation.
- **Cross-behavior generalization survives Phase 7's retraction** in
  the modest sense: the DLC + V-JEPA-2 + linear probe pipeline does
  not collapse when applied to a different RHpE behavior at adequate
  sample size and with adequate labels.
- **Per-clip 4-way analysis** quantifies how "competitive" works in
  practice: BOTH_RIGHT 189 / BOTH_WRONG 15 / DLC_BEATS_WHOLE_FRAME 40 /
  WHOLE_FRAME_BEATS_DLC 39 — methods agree on 72% of clips, split
  near-evenly on the other 28%.
- **Bimodal DLC ear-keypoint reliability** predicted by Stage 1.5
  generalizes to population: 90.8% per-frame-confident, 9.2% need
  fallback (vs Stage 1.5's 90% / 10%). Locked Decision 1 fallback
  rule fired mechanically on all 26 affected clips.

## What Phase 8b does NOT establish

- **DLC ear cropping does NOT outperform whole-frame V-JEPA-2** at
  n=283 with adequate paired-test power. The methodology is
  statistically indistinguishable from baseline.
- **Per-source robustness is NOT shown** — only 4/12 sources show
  positive Δ; the +0.026 pooled advantage is concentrated in S8
  (Δ +0.35) and would likely flip without it.
- **Not a reversal of Phase 7's retraction.** Phase 7 stays
  UNDERPOWERED_INDISTINGUISHABLE.
- **Not a multi-rater κ resolution** — single-observer (RME paper
  labels) caveat applies.
- **Not a Phase 9 N-expansion result.**
- **Not evidence that DLC methodology adds value beyond preservation.**
  Open question routed to Phase 9 motivation reframe.

## Methodology trail

The honest project narrative shifts from pre-8b to post-8b:

- **Pre-8b expected:** "DLC keypoint-anchored cropping is a viable
  cropping methodology for RHpE behaviors, demonstrated on two
  behaviors."
- **Post-8b honest:** "DLC keypoint-anchored cropping is a viable
  cropping methodology that **preserves whole-frame baseline
  performance** on RME ear data at adequate sample size, but
  **does not significantly improve over whole-frame V-JEPA-2.**
  Cross-behavior generalization is established in the sense that
  the methodology travels (does not degrade signal); whether it
  adds value beyond whole-frame remains an open question routed
  to Phase 9."

Combined with Phase 7's retraction, the project's central
methodological claim becomes: **"explicit cropping preserves signal
across behaviors at moderate sample sizes"** — narrower than "explicit
cropping helps," but defensible on the evidence.

### Combined Phase 8a + 8b narrative (across two RHpE behaviors)

Phase 8a stress-tested Phase 7's eye-region comparison and fired the
locked RETRACTION trigger; Phase 7's verdict was retracted to
UNDERPOWERED_INDISTINGUISHABLE at n=34. Phase 8b extended the same
DLC + V-JEPA-2 + linear probe methodology to a second RHpE behavior
(ear) at n=283 with adequate paired-test power, and landed COMPETITIVE
under the locked verdict-reporting protocol.

Reading the two phases together, the project's combined post-8b claim is:

> **DLC keypoint-anchored cropping is a viable methodology that
> preserves baseline signal across two RHpE behaviors (eye, ear).**
> Whether it improves over simpler baselines (manual annotation on
> eye, whole-frame V-JEPA-2 on ear) remains an open question across
> both behaviors:
>
> - **Eye comparison underpowered at n=34** (Phase 7 retracted via
>   Phase 8a's stress-test).
> - **Ear comparison statistically indistinguishable at n=283** with
>   adequate power (Phase 8b COMPETITIVE).
>
> The methodology demonstration is **"this approach travels and
> doesn't degrade signal,"** not "this approach improves performance."
> Phase 9 priorities shift accordingly: **multi-rater κ** (label
> quality, addresses the single-observer caveat shared by both
> behaviors) and **reframed-value-proposition investigation**
> (computational efficiency, interpretability, behavior-specific
> localization, robustness to non-target subjects per Lesson 9) —
> none of which depend on LOSO AUC improvement claims.

This is narrower than the pre-8a/8b expected story and more honest.
The discipline pattern (pre-registration → locked thresholds →
mechanical retraction → independent extension at adequate n) made
both findings defensible on their own terms and jointly defensible as
a combined narrative.

The discipline pattern protected against multiple kinds of failure
modes through this audit cycle:

- Stage 1.5 reference-floor verification ensured the gate wasn't too
  strict for video-realistic data
- Locked Decision 1 fallback handling caught the 9.2% bimodal-pattern
  clips mechanically without halt
- Pre-locked verdict-reporting protocol (7-row matrix) prevented
  silent celebration of "G1 STRONG" without the matched G4
  INCONCLUSIVE caveat
- Per-clip 4-way categorization gave direct evidence of "wins and
  losses balance out" (40 vs 39) that no single aggregate statistic
  exposes
- Phase 8a's RETRACTION trigger was preserved as recorded; Phase 8b's
  outcome is independent and does not retroactively bias
  interpretation of the retraction

## Pre-registration audit chain

Phase 8b artifacts:

- `outputs/track_b_phase8b_preregistration_stage1.md` (Stage 1 pre-reg)
- `tools/phase8b_rme_parity.py` + `outputs/phase8b_rme_parity.json`
  (Step 1)
- `tools/phase8b_stage15_ear_inspection.py` +
  `outputs/phase8b_stage15_inspection.json` (Step 2)
- `tools/phase8b_run_dlc_on_full_rme.py` +
  `outputs/phase8b_rme_dlc_keypoints.json` (Step 4)
- `tools/phase8b_dlc_ear_crop.py` +
  `outputs/eye_crops_v4_ear_dlc_manifest.jsonl` (Step 5)
- `outputs/vjepa2_embeddings_ear_v4.npz` (Step 6)
- `tools/eye_loso_lr_phase8b.py` +
  `outputs/eye_loso_results_phase8b.json` (Step 7)
- `tools/phase8b_diagnostic.py` +
  `outputs/phase8b_diagnostic.{json,md}` (Step 8)
- `tools/phase8b_audit_extras.py` +
  `outputs/phase8b_audit_extras.json` (audit-doc requirements 1, 3, 5)
- `docs/phase8b_audit.md` (this document)

A future reader can reconstruct: what was claimed, when, on what
evidence, with all locked rules and gates traceable to pre-registration
hashes.
