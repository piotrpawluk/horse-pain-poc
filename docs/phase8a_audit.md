# Phase 8a audit — stress-test of Phase 7's +0.048 AUC advantage

**Date**: 2026-05-10. Branch: `experiment/phase8a`. Five locked tests
ran on existing artifacts per `outputs/track_b_phase8a_preregistration_stage1.md`
(hash `c0bf2bdca8d5...`). Test 1 fired the pre-registered RETRACTION
trigger; per locked procedure, Phase 7's verdict is retracted to
**UNDERPOWERED_INDISTINGUISHABLE**.

## Headline result

| Metric | Value |
|---|---|
| Phase 5 manual AUC (reference) | 0.7985 |
| Phase 7 DLC corrected AUC (observed) | 0.8462 |
| Observed Δ | +0.0476 |
| Bootstrap CI 95% on Δ (B=10000, parent-clip) | [−0.1304, +0.2754] |
| Bootstrap median Δ | +0.0386 |
| **Test 1 verdict (locked gate)** | **RETRACTION** |
| **Replacement verdict (mechanical, median > −0.01)** | **UNDERPOWERED_INDISTINGUISHABLE** |

The bootstrap LB at −0.13 is well below the retraction trigger of
−0.02. The replacement-verdict rule (median > −0.01 →
UNDERPOWERED_INDISTINGUISHABLE; median ≤ −0.01 → PHASE_7_FAVORABLE_BY_CHANCE)
fires on median = +0.0386, choosing UNDERPOWERED_INDISTINGUISHABLE
mechanically.

**The retraction is not "Phase 7 was wrong on this data." Phase 7's
empirical result (AUC 0.846 on the 34 RME clips) stands as recorded.
The retraction is "Phase 7's claim of superiority cannot be supported
at n=34, n=12 sources scale."**

**Critical framing**: the retraction trigger fires on the bootstrap
**CI width**, not on the bootstrap **direction**. Specifically:

- P(Δ > 0) under the bootstrap = **66%** (the bulk of probability
  mass remains on Δ > 0)
- P(Δ ≥ +0.02) under the bootstrap = **63%**
- P(Δ ≤ 0) = 34%
- P(Δ ≤ −0.02) = 27%

The bootstrap distribution is **directionally positive but with
substantial probability mass extending below zero**. The 95% CI lower
bound of −0.13 fires the retraction trigger because it is below −0.02,
but this reflects CI width (uncertainty), not central tendency
(direction). The honest framing is:

- The data we have suggests DLC keypoint-anchored cropping ≈ Phase 5
  manual at n=34, with directional advantage probably positive
- The claim "DLC outperforms Phase 5" is unsupported at n=34 because
  the bootstrap CI's lower bound includes negative values
- The claim "DLC underperforms Phase 5" is also unsupported (median
  +0.039, P(Δ ≤ 0) only 34%)

If the audit doc were to frame this as "Phase 7 was wrong," that
overstates falsification. The data don't say Phase 7 was wrong; they
say the comparative claim is **underpowered to confirm**.

## All 5 test results

### Test 1 — Subject-bootstrap CI on Δ AUC

```
B=10000, seed=42, parent-clip resampling at source level
n_bootstrap_kept: ~10000 (all valid)
delta_mean:    +0.0408
delta_median:  +0.0386
delta_std:      0.10
95% CI:        [-0.1304, +0.2754]
P(Δ ≤ 0):      ~0.34
P(Δ ≤ -0.02):  ~0.27
P(Δ ≥ +0.02):  ~0.63
```

**Verdict**: **RETRACTION** (LB −0.1304 ≤ −0.02 trigger)
**Next action**: Phase 7 retraction amendment per locked procedure;
Phase 8b BLOCKED until retraction commit lands.

### Test 2 — Per-source LOSO ablation

10/12 sources contribute positive Δ when dropped (i.e., dropping that
source from the data leaves Δ remaining positive on the other 11).

**Verdict per locked gate**: **ROBUST** (≥9/12)

### Test 3 — Per-clip score divergence

```
n: 34
mean Δ score:    +0.0753
median Δ score:  +0.1846
std:              0.8367
min:             -1.6962
max:             +2.1080
```

**Reportable**: wide per-clip variance dominates the +0.048 mean. The
distribution is heavy-tailed: max +2.11, min −1.70, std 0.84 around a
median of +0.18. Outlier-driven, not uniform shift.

**Top-3 outliers per pre-reg cross-check #2** (Δ = phase7_score −
phase5_score):

| Direction | Clip | Truth | P5 score | P7 score | Δ |
|---|---|---|---:|---:|---:|
| **DLC most strongly improved** | `background_S8.mp4_3_` | ACTION (relabeled) | −0.940 | +1.168 | **+2.108** |
| | `action_S10.mp4_0_` | ACTION | +0.042 | +1.404 | +1.362 |
| | `action_S9.mp4_4_` | ACTION | −0.117 | +1.104 | +1.221 |
| **DLC most strongly lost** | `action_S11.mp4_0_` | BACKGROUND (relabeled) | +1.446 | −0.250 | −1.696 |
| | `background_S6.mp4_2_` | BACKGROUND | −0.474 | −1.787 | −1.313 |
| | `background_S5.mp4_10_` | ACTION (relabeled) | +0.699 | −0.582 | −1.281 |

**Pattern observations** (informational, not gated):
- All 3 strongly-improved clips are **ACTION clips Phase 5 was wrong on
  or barely correct** — DLC keypoint-anchored cropping recovered them
  decisively. `background_S8.mp4_3_` is the perceptual-floor /
  catchlight clip Phase 5 audit named; its recovery by DLC is the
  single most striking score-shift in the dataset.
- 2 of 3 strongly-lost clips are clips DLC pushed CONFIDENTLY wrong
  (action_S11.mp4_0_ goes from +1.446 to −0.250; bg_S6.mp4_2_ goes
  from −0.474 to −1.787). DLC's failures are decisive when they happen.
- The pattern (asymmetric: 3 strong recoveries with peak Δ = +2.11; 3
  strong losses with peak Δ = −1.70) is consistent with the bootstrap
  CI's wide tails. Different clips drive Δ in opposite directions; the
  median +0.18 is a relatively small net positive amid large per-clip
  movements.

### Test 4 — V-JEPA-2 feature similarity

```
v3↔v4 same-clip cosine sim:       median 0.8849, mean 0.8841, p25 0.8526, p75 0.9132
v3↔v3 different-clips baseline:   median 0.9013, mean 0.9015, p25 0.8838, p75 0.9197
Margin (same-clip − cross-clip): -0.0164
```

**Verdict per locked gate**: **SAME_FEATURES_DIFFERENT_GEOMETRY**
(median ≥ 0.7).

### Threshold heuristic note (locked from pre-reg)

The 0.7 threshold for SAME_FEATURES_DIFFERENT_GEOMETRY vs
DIFFERENT_FEATURES_SIMILAR_PREDICTION is a heuristic anchored on the
project's intra-rater IoU threshold (0.765 from Phase 5b). It is
**NOT empirically validated** for V-JEPA-2 feature similarity. The
cross-clip baseline (0.9013) provides post-hoc calibration: the same-
clip v3↔v4 median (0.8849) is **below** the cross-clip baseline by
0.016, meaning v3 and v4 features of the SAME clip are *less similar*
than v3 features of DIFFERENT clips. This is a striking finding —
V-JEPA-2 cropping-induced feature change is comparable in magnitude to
clip-to-clip variation. The locked verdict (SAME_FEATURES) is
mechanically applied per the 0.7 threshold but the post-hoc baseline
calibration suggests a more nuanced reading: **the two pipelines
extract features that differ at roughly the same magnitude as the
between-clip differences**, which weakens the "interchangeable
features" narrative the verdict label suggests.

This is exactly the kind of nuance a Phase 9 calibration study would
investigate: build the v3↔v3 baseline rigorously across sources/labels,
not just the random-clip-pairs heuristic used here.

### Test 5 — G3-IoU-conditional AUC

```
Bucket           n    AUC      Bootstrap 95% CI
─────────────────────────────────────────────────
off_eye (<0.30)  10   0.6400   [0.2083, 1.0000]
mid     (≥0.30)  19   0.9487   [0.8286, 1.0000]
on_eye  (≥0.50)   5   0.8333   [0.2500, 1.0000]
```

**Reportable**: the *mid* IoU bucket (0.30-0.50) has the highest AUC
(0.95), not the on-eye bucket (0.83) as the "DLC matches manual when
on-eye" narrative would predict. Off-eye clips have AUC 0.64 with very
wide CI. Pattern doesn't support either of the two pre-registered
narratives cleanly; sample sizes are too small (n=5-19 per bucket) to
draw firm conclusions. Phase 9 with larger samples could resolve.

The mid-bucket AUC dominance is itself worth flagging as a Phase 9
pre-registered hypothesis: maybe DLC keypoint anchoring works best at
intermediate IoU because the keypoint center is reliably AT the eye
center but the geometric box drifts from human-annotated tight crops
in non-pathological ways. The wide CI on the on-eye bucket (n=5) means
this could just be sample noise.

## Cross-check observations

### Test 1 (RETRACTION) vs Test 2 (ROBUST) — apparent inconsistency

The pre-reg flagged this exact pattern: "If test 2 routes 'fragile'
but test 1 routes 'robust' (LB ≥ +0.02), report the apparent
inconsistency." Here it's the inverse direction: Test 1 RETRACTION
(LB ≤ −0.02) but Test 2 ROBUST (10/12 sources +Δ). Same pre-reg
logic applies in reverse.

**Interpretation**: the two tests measure different uncertainties.
- **Test 1 (bootstrap with replacement)**: "what's the typical Δ if we'd
  drawn a different set of 12 sources?" — wide CI because at n=12
  sources, source-bootstrap has high variance. Some bootstrap iterations
  draw multiple copies of the 2 negative-Δ sources and few copies of
  the 10 positive-Δ sources, generating wide tails.
- **Test 2 (LOSO drop-one)**: "is Δ positive on most subsets of 11/12
  sources?" — robust because dropping any single source leaves the
  remaining 11 mostly positive-Δ.

**They're not contradictory; they answer different questions about
generalizability.** The data we have (Phase 5 + Phase 7 outputs on 34
clips × 12 sources) supports both findings simultaneously: the
+0.048 is empirically real and stable to single-source deletion,
but extrapolating to the population of horse-eye-region datasets is
uncertain at this scale.

The locked retraction procedure fires on Test 1 because that's what
the gate-ladder says. The audit chain preserves both findings —
neither is erased.

### Test 4 + 5 cross-tab

Per-clip table with `cos_sim`, `IoU`, and `score_delta` is in
`outputs/phase8a_results.json` → `cross_checks.test4_test5_per_clip_table`.
Notable patterns (informational, not gated):
- High-cos / high-IoU / small-score-delta: the modal case
- High-cos / low-IoU / small-score-delta: clips where DLC and Phase 5
  produce similar features despite different geometry
- Low-cos / large-score-delta (either direction): clips where the
  features genuinely diverged AND so did the predictions — the most
  interesting Phase 9 hypothesis cases

### Per-source per-bucket cross-tab

Available in `outputs/phase8a_results.json` →
`cross_checks.per_source_per_bucket`. Some sources (e.g., S6, S7) have
multiple clips concentrated in the off-eye bucket; others (e.g., S2,
S5) have most clips in the mid bucket. Phase 9 N expansion should
balance source × bucket coverage.

## Phase 7 retraction (per locked procedure)

Per the Phase 8a pre-reg locked retraction-procedure clause, an
appended section is added to `docs/phase7_audit.md`:

- Original verdict (OUTPERFORM_PHASE_5_AUC_ONLY) preserved as
  struck-through inline text
- Replacement verdict: **UNDERPOWERED_INDISTINGUISHABLE**
- Reason: Phase 8a Test 1 bootstrap LB = −0.1304 ≤ −0.02 (retraction
  trigger); replacement verdict chosen mechanically by median
  Δ_bootstrap = +0.0386 > −0.01

The retraction is a structural amendment, not a softening reframe.
Phase 7's empirical result (AUC 0.8462, the per-clip categorizations,
the locked gate verdicts G1/G2 PASS) all stand as historical record.
What is retracted is the comparative *claim* about superiority over
Phase 5 manual.

The original Phase 7 audit doc hash (`d838ddd3dc07...`) remains valid
as the pre-retraction historical artifact. The retraction adds a new
hash to the chain (Phase 7 audit doc supersession with explicit
retraction notation).

## Phase 8b status

**BLOCKED** per locked pre-reg clause: Phase 8b cannot proceed until
the Phase 7 retraction amendment is committed and pushed to the audit
chain.

## Phase 9 entry conditions (from Phase 8a outcome)

Per the locked routing matrix in pre-reg:

| 8a outcome | 8b outcome | Phase 9 priority |
|---|---|---|
| Underpowered ([-0.02, +0.02]) | strong | N expansion to resolve paired-DeLong; methodology cross-validation already done by 8b |
| Underpowered | modest | N expansion + multi-rater κ both |
| Underpowered | <0.65 | Eye-specific narrow narrative |
| ≤ −0.02 (retraction) | (any) | **Halt forward work; Phase 7 retraction amendment + project-narrative revision** |

Phase 8a fired the retraction. **Phase 9 priority is now: complete the
retraction first, then assess whether to proceed with 8b at all.**

Per the original pre-reg logic, retraction halts forward work pending
project-narrative revision. The user-approval checkpoint here decides
whether to:
- (A) Commit retraction + draft revised project-narrative + halt Phase 8b indefinitely
- (B) Commit retraction + proceed with 8b under modified verdict-band recalibration (since the cross-behavior generalization question is still informative even with a retracted Phase 7 verdict on eye-region)
- (C) Other path informed by user judgment

Per the locked retraction-procedure clause: "Phase 8b is blocked until
the retraction commit lands in the audit chain. The user-approval
checkpoint between 8a result and 8b entry has an additional gate:
explicit acknowledgment of the retraction."

The first action is the retraction commit. Subsequent path is user-decided
post-retraction.

## What Phase 8a establishes

1. **Phase 7's +0.048 advantage at n=34 is empirically real on this
   dataset** (Test 2: 10/12 sources contribute positive Δ when
   individually dropped) but **does not establish superiority over
   Phase 5 manual at this scale** (Test 1: bootstrap LB −0.13 fires
   the retraction trigger).
2. **The per-clip score divergence is dominated by within-clip
   variance** (Test 3: std 0.84 around mean +0.075). The +0.048 is a
   small signal in a large noise distribution.
3. **V-JEPA-2 features of v3 and v4 cropped clips differ at
   approximately the same magnitude as features differ between
   different clips** (Test 4: same-clip 0.88 vs cross-clip 0.90). The
   cropping change is non-trivial in feature space.
4. **The mid-IoU bucket carries the AUC** (Test 5: 0.95 at IoU
   0.30-0.50). Neither the "high-IoU drives AUC" nor "low-IoU also
   predicts well" narrative cleanly holds at this n.
5. **The discipline pattern's retraction-procedure clause works
   mechanically.** Test 1 fired; the pre-locked replacement verdict
   chose UNDERPOWERED_INDISTINGUISHABLE without negotiation. The
   user-approval checkpoint review is still required, but the
   procedural moves under cognitive load were pre-determined.

## What Phase 8a does NOT establish

- **Phase 7's verdict was wrong-on-this-data**. It wasn't. The
  retraction is about generalizability claim, not data correctness.
- **DLC keypoint-anchored cropping is strictly worse than Phase 5
  manual**. Bootstrap median is +0.0386 (positive). Direction is
  ambiguous, not negative.
- **Phase 9 N expansion will resolve the question**. Probably will,
  but Phase 8a quantifies precision at n=34, not predicts outcome at
  higher N.
- **Phase 8b cross-behavior generalization is uninformative**. It
  could still test whether the methodology generalizes — but with
  Phase 7's eye-region verdict retracted, the Phase 8b framing needs
  revision before commitment.

## Methodology trail (the discipline-pattern claim)

Phase 8a is the strongest demonstration of the discipline pattern in
the project so far. Pre-reg specified the retraction trigger and
procedure exactly because this kind of finding was anticipated as
possible. When it fired, the procedural moves were mechanical.

The audit chain will preserve:
- Phase 7's original audit doc (`d838ddd3dc07...`) as historical record
- Phase 7's retraction amendment (new hash in chain) as the corrected verdict
- Phase 8a audit doc (this document) as the empirical falsification evidence
- Phase 8a results JSON as the raw test outputs

A future reader can reconstruct exactly what was claimed, when, and on what evidence — including the empirical falsification of the original claim. That is what the discipline pattern is for.
