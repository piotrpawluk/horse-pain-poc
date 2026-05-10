# Phase 8c audit — calibration package on Phase 8b RME ear output

**Date**: 2026-05-10. Branch: `experiment/phase8c`. **Calibration methodology
infrastructure** (B3 in user-facing nomenclature) demonstrated on **one
behavior** (RHpE ear movement, Phase 8b output, n=283 / 12 sources). Strictly
downstream of Phase 8b — no new training, no new compute on the spine.

> **Scope note.** This is calibration infrastructure on a single behavior, not
> a calibration package for RHpE as a whole. Each future behavior added to the
> project will need its own per-behavior temperature scaling, τ_k, and
> reliability diagram. The infrastructure built here (calibration LOSO,
> ECE/Brier/NLL/reliability, source-bootstrap CI on TPR, exact Poisson-
> binomial session-level OP) is reusable; the per-behavior work is not
> amortizable across behaviors.

## Headline result — WELL_CALIBRATED + G1 STRUCTURAL FINDING

**Joint verdict: WELL_CALIBRATED on the load-bearing calibration metrics
(G2 PASS, G3 well_calibrated, G4 PASS) with one substantive structural
finding on the AUC-invariance sanity check (G1 FAIL).**

The calibration package delivers cleanly: ECE drops from 0.1118 to 0.0397
(–65% relative), Brier from 0.1420 to 0.1289, all 12 per-fold T values cluster
in [0.46, 0.52] well within the locked plausibility band. The G1 FAIL is
**not a calibration defect** — it surfaces an internal inconsistency between
pre-reg D1's claim of AUC invariance (true for global T) and D2's per-source
calibration LOSO design (which makes T monotonic *within source* but not
*across the pooled set*). Reported as substantive methodological finding;
Phase 7 §4 anatomical-mapping precedent applies.

## Pre-Step gates cleared

- ✓ **Checkpoint #1**: pre-reg approved (`outputs/track_b_phase8c_preregistration.md`)
- ✓ **Checkpoint #2**: tool implementation verified (syntax + import + 5 unit-style probes incl. session OP both methods)
- ✓ **Checkpoint #3**: run completed; results, extras, and reliability diagram landed; this audit doc responds

## Headline metrics

| Metric | Pre-cal | Post-cal | Δ |
|---|---:|---:|---:|
| Pooled AUC | 0.900850 | 0.898651 | **−0.002199** ⚠ (Finding 1) |
| ECE (10 equal-freq bins) | 0.1118 | **0.0397** | −0.0721 (~65% reduction) |
| Brier | 0.1420 | 0.1289 | −0.0131 |
| NLL | — | 0.4066 | — |
| Murphy REL (binned) | 0.0156 | **0.0021** | −0.0135 (~87% reduction) |
| Murphy RES (binned) | 0.1227 | 0.1209 | −0.0018 (~1.5% reduction) |
| Murphy UNC (binned) | 0.2498 | 0.2498 | 0 (fixed by labels) |

| Operating point quantity | Value |
|---|---|
| τ_ear at FPR=0.05 | **0.8138** |
| TPR at τ_ear | 0.5435 |
| TPR 95% CI (B=1000 source-bootstrap, seed=42) | [0.3077, 0.7500] |
| n_negatives_used / n_positives_used | 145 / 138 |
| Session OP P(≥8 \| H0) **exact** Poisson-binomial | **1.393e−05** |
| Session OP P(≥8 \| H0) Poisson(λ=1.2) approximation | 3.698e−05 |
| Approximation relative error | **1.66× (166%)** — Finding 3 |

## Gate verdicts

| Gate | Threshold | Observed | Verdict | Note |
|---|---|---:|---|---|
| **G1 — sanity AUC invariance** | post = pre to within 1e−10 | Δ = −2.20e−3 | **FAIL** | Structural; see Finding 1 |
| **G2 — calibration improvement** | post ECE < pre ECE | Δ = −0.0721 | **PASS** | ~65% relative reduction |
| **G3 — calibration quality band** | <0.05 well / 0.05–0.10 acceptable / ≥0.10 poor | 0.0397 | **well_calibrated** | Strict-< boundary convention |
| **G4 — T plausibility** | All T_S ∈ [0.3, 5.0] | All in [0.4567, 0.5244] | **PASS** | Tight cluster — see Finding 2 |

Per the locked anti-pattern lock (#7 in pre-reg), G1's FAIL is reported without
softening or retroactive threshold relaxation. The structural reason is
analyzed in Finding 1; future per-source-calibration designs should adopt a
calibration-LOSO-aware sanity check (a candidate formulation is sketched in
Finding 1's "forward-looking observation" subsection but **not** locked here).

---

## Finding 1 — G1 structural failure (D1↔D2 internal inconsistency)

**Substantive finding.** Pre-reg Decision 1 (temperature scaling) claimed AUC
invariance under post-hoc rescaling: "Preserves the rank ordering of
`decision_function` outputs (T > 0 is monotonic), so per-clip ranking and AUC
are mathematically unchanged." That claim is true for **global** T (single
scalar applied to all data). Pre-reg Decision 2 (calibration LOSO) requires
**per-source** T (12 distinct T values, one per LOSO fold). Per-source T is
monotonic *within each source* but **NOT globally across the pooled set**.

### Mechanism (worked example using observed T values)

Consider two clips from different sources with similar raw scores:

- A clip in S9 with score = 1.0000 → calibrated p = σ(1.0000 / 0.4567) = σ(2.190) = **0.8993**
- A clip in S2 with score = 0.9000 → calibrated p = σ(0.9000 / 0.5244) = σ(1.716) = **0.8479**

Pre-cal (T=1) sigmoid: σ(1.0000) = 0.7311 vs σ(0.9000) = 0.7109. Pre-cal: A > B.
Post-cal: A = 0.8993 > B = 0.8479. Same order in this case.

But for tighter pairs the rank can swap:

- S9 clip with score = 0.5000 → σ(0.5/0.4567) = **0.7491**
- S2 clip with score = 0.5500 → σ(0.55/0.5244) = **0.7384**

Pre-cal: σ(0.5500) = 0.6342 > σ(0.5000) = 0.6225. **Pre-cal: B > A.**
Post-cal: A = 0.7491 > B = 0.7384. **Post-cal: A > B.**

A cross-source rank swap. AUC counts these.

### Empirical confirmation — per-source AUC is bit-exact invariant

Per-source AUC computed pre vs post calibration on the actual Phase 8c output:

| Source | n | n_pos | n_neg | AUC_pre | AUC_post | |Δ| |
|---|--:|--:|--:|---:|---:|---:|
| S1 | 21 | 7 | 14 | 0.7959 | 0.7959 | ≤1e−10 ✓ |
| S2 | 25 | 10 | 15 | 1.0000 | 1.0000 | ≤1e−10 ✓ |
| S3 | 28 | 16 | 12 | 0.9688 | 0.9688 | ≤1e−10 ✓ |
| S4 | 32 | 24 | 8 | 0.8958 | 0.8958 | ≤1e−10 ✓ |
| S5 | 25 | 14 | 11 | 0.9740 | 0.9740 | ≤1e−10 ✓ |
| S6 | 19 | 9 | 10 | 0.8222 | 0.8222 | ≤1e−10 ✓ |
| S7 | 21 | 1 | 20 | 0.7500 | 0.7500 | ≤1e−10 ✓ |
| S8 | 24 | 16 | 8 | 0.9844 | 0.9844 | ≤1e−10 ✓ |
| S9 | 24 | 13 | 11 | 0.8322 | 0.8322 | ≤1e−10 ✓ |
| S10 | 22 | 10 | 12 | 1.0000 | 1.0000 | ≤1e−10 ✓ |
| S11 | 19 | 11 | 8 | 0.7955 | 0.7955 | ≤1e−10 ✓ |
| S12 | 23 | 7 | 16 | 0.8929 | 0.8929 | ≤1e−10 ✓ |
| **Pooled** | **283** | **138** | **145** | **0.900850** | **0.898651** | **0.002199** |

**12/12 sources are bit-exact invariant; pooled drops by 0.0022.** This
confirms the mechanism: within-source ranking is preserved (per-source AUC
unchanged), the pooled AUC drop is purely cross-source rank shuffle. The Δ is
small because per-fold T values are tight (Finding 2).

The per-source AUC numbers also bit-exact-match Phase 8b's per-source AUC
table (`phase8b_audit.md` "Per-source AUC breakdown"), confirming the
pre-cal probabilities (sigmoid of raw scores) preserve raw-score ordering
exactly — as expected.

### Disposition

**No retroactive softening.** Per anti-pattern #7 in the pre-reg, G1's
threshold is not relaxed mid-phase. G1 is reported as FAIL with structural
explanation. The reported Δ (-0.0022) is recorded as the property-of-record.

**Phase 7 §4 precedent.** This is structurally analogous to Phase 7's
anatomical-mapping falsification (where Stage 1 §4's reasoning was inverted
relative to empirical observation). In both cases, the discipline pattern's
job is to surface the inconsistency honestly and route Stage 2 amendments
to the next phase rather than retro-edit the locked artifact. Phase 7's §4
finding was reported as Lesson 20; this G1 finding is recorded here in the
audit doc and propagates to Phase 9 prereg drafting as a forward-looking
sanity-check redesign question.

### Forward-looking observation (NOT a Phase 8c lock; informs Phase 9 scoping)

A calibration-LOSO-aware G1 sanity check should test:

1. **Per-source AUC invariance** (per-fold within-source check) — bit-exact
   under per-source T, as confirmed above. This is the correct sanity check
   for D2's per-source design.
2. **Bounded pooled AUC drift conditional on T variance** — e.g., |Δ pooled
   AUC| ≤ k × σ(T) for some empirically calibrated k. With T values clustered
   in [0.46, 0.52] (range 0.066), Δ_AUC = 0.0022 ≈ 0.033 × range, suggesting
   k ≈ 0.04 as an order-of-magnitude rule-of-thumb. **Not locked here**;
   surfaced as Phase 9 drafting input.

---

## Finding 2 — Per-source T values cluster tightly (RidgeClassifier under-confidence pattern)

All 12 per-fold T values fall in [0.4567, 0.5244], median 0.494, IQR
[0.477, 0.510]:

| Source | T_S | Source | T_S |
|---|---:|---|---:|
| S1 | 0.4763 | S7 | 0.5102 |
| S2 | 0.5244 | S8 | 0.4698 |
| S3 | 0.5033 | S9 | **0.4567** (min) |
| S4 | 0.4819 | S10 | 0.5102 |
| S5 | 0.5230 | S11 | 0.4774 |
| S6 | 0.4841 | S12 | 0.5073 |
|  |  | **S2** | **0.5244** (max) |

**All T_S < 1**, meaning the raw `RidgeClassifier.decision_function` outputs
are systematically *underconfident* across all 12 sources by a similar amount
— calibration is *sharpening* the sigmoid (T < 1 → score / T amplifies).

This is a coherent property of `RidgeClassifier(alpha=1.0,
class_weight='balanced')` paired with a sigmoid link (which is not the
classifier's native probability output): the decision_function is a signed
distance, not a log-odds. The empirical clustering of T around 0.5 reflects
the consistent under-confidence pattern, not source-specific drift.

**Diagnostic implication.** The cross-source distribution shift in *raw
scores* is small. Calibration is correcting a *shared* systematic
under-confidence, not adapting to per-source idiosyncrasies. This is a
mild signal that classifier output distributions are reasonably stable
across sources — Phase 8b's per-source AUC heterogeneity (S7=0.75 vs
S10=1.00) is **not** primarily driven by per-source calibration drift,
but by per-source signal strength differences.

**Phase 8c stands as documented; observation informs Phase 9+ scoping.**
A simpler calibration design (e.g., a *single* global T fit via nested CV)
might be defensible at this clustering level — but evaluating that
alternative is Phase 9 work, not Phase 8c retro-design.

---

## Finding 3 — Poisson approximation overestimates Poisson-binomial tail at RHpE-relevant scale

The session-level operating point under the independence assumption (D5)
was computed via two methods:

| Method | P(session flagged \| H0) |
|---|---:|
| **Exact Poisson-binomial** (24-Bernoulli convolution; reduces to Binomial(24, 0.05) under uniform p) | **1.393e−05** |
| Poisson(λ=24×0.05=1.2) approximation | 3.698e−05 |
| Absolute error | 2.305e−05 |
| **Relative error** | **1.66× (166%)** |

The Poisson approximation **overestimates** the binomial tail by 1.66× in the
regime that matters for RHpE clinical-utility framing (n=24 behaviors, target
per-behavior FPR ≈ 0.05, λ ≈ 1–2). At n=24 / p=0.05, Poisson is in its
mediocre-approximation regime — the rule of thumb "Poisson ≈ Binomial when
n large and p small with np moderate" hits diminishing returns when n is
fixed at modest values.

### Generalizable methodological observation

> **At RHpE-relevant scale (n=24 behaviors, target FPR ~0.05, session
> threshold ≥8/24), the standard Poisson approximation to Poisson-binomial
> tail probabilities overestimates by 1.66× compared to exact computation.
> The exact Poisson-binomial calculation (24-Bernoulli convolution; trivial
> in numpy) is the load-bearing reference; future RHpE session-level OP
> analyses should compute exactly rather than approximate. The qualitative
> conclusion (~1e−5 either way → essentially zero session-level FPR under
> independence) is unchanged, but the quantitative claim is.**

This applies beyond Phase 8c — to any clinical-utility claim about the
RHpE ≥8/24 operating point computed under per-behavior independence.

The qualitative story is unchanged either way: under the independence
assumption (which is empirically unverified and probably violated — see
Limitation 1 below), the session-level FPR at the ≥8/24 threshold is
vanishingly small. The 166% quantitative gap is the methodological
sharpening, not the load-bearing claim.

---

## Finding 4 — Per-source ECE heterogeneity (concentrated improvements + small-n flat sources)

Per-source ECE pre vs post calibration (sorted by post-cal ECE ascending):

| Source | n | bins used | ECE pre | ECE post | Δ |
|---|--:|--:|---:|---:|---:|
| S7 | 21 | 4 | 0.1911 | **0.0607** | −0.1304 |
| S12 | 23 | 4 | 0.1730 | 0.0829 | −0.0901 |
| S9 | 24 | 4 | 0.1150 | 0.0885 | −0.0265 |
| S5 | 25 | 5 | 0.2001 | 0.0955 | −0.1046 |
| S1 | 21 | 4 | 0.1496 | 0.1284 | −0.0213 |
| S4 | 32 | 6 | 0.1833 | 0.1301 | −0.0532 |
| S2 | 25 | 5 | 0.2676 | 0.1566 | −0.1111 |
| S6 | 19 | 3 | 0.1564 | 0.1605 | **+0.0041** |
| S11 | 19 | 3 | 0.1848 | 0.1850 | **+0.0001** |
| S10 | 22 | 4 | 0.2758 | 0.2026 | −0.0732 |
| S8 | 24 | 4 | 0.2078 | 0.2127 | **+0.0050** |
| S3 | 28 | 5 | 0.2749 | 0.2177 | −0.0572 |

**9/12 sources improve substantially with calibration; 3/12 (S6, S8, S11) are
essentially flat (|Δ| ≤ 0.005).** All three flat sources are at the smaller-n
end (19–24 clips), where per-source ECE is computed with reduced bin counts
(3–4 bins) to avoid degeneracy. At this n and bin density, the metric noise
floor itself is a meaningful fraction of the change — the flat result is
characterized as "metric-noise-dominated," not as calibration failure.

This per-source heterogeneity mirrors Phase 8b's per-source AUC
heterogeneity (`phase8b_audit.md` table: 4/12 sources show positive ΔAUC
vs whole-frame, S8 alone contributes Δ +0.35) — both phases find that
methodology effects are concentrated rather than uniformly distributed
across sources. The per-source distribution patterns are an irreducible
property of the 12-source LOSO setup, not a methodology defect.

**Anti-pattern preservation.** Per pre-reg lock #1 (no method shopping)
and the spirit of #7 (no goal-shifting on calibration verdicts), the audit
**does not** propose source-specific calibration parameters or alternative
calibration methods to "fix" the flat sources. Per-source ECE
heterogeneity is reported as honest characterization; the calibration
package is locked at one global temperature scaling per LOSO fold.

---

## Finding 5 — Brier improvement decomposed (Murphy 1973): reliability dominates, resolution barely shifts

Murphy decomposition (binned, 10 equal-frequency bins) of the Brier score:

```
Brier ≈ Reliability − Resolution + Uncertainty   (+ within-bin variance)

           REL (↓ better)   RES (↑ better)   UNC (fixed)   BS_binned   BS_actual
pre-cal:        0.0156          0.1227          0.2498       0.1428      0.1420
post-cal:       0.0021          0.1209          0.2498       0.1310      0.1289

ΔREL = −0.0135  (~87% relative reduction — calibration win)
ΔRES = −0.0018  (~1.5% relative reduction — cross-source rank shuffle artifact)
ΔUNC =  0      (UNC depends only on labels, fixed at ȳ(1−ȳ) = 0.488 × 0.512)
```

The Brier improvement (Δ_BS_actual = −0.0131) decomposes cleanly:

- **~87% comes from reliability improvement** — calibration is doing what it
  was designed to do: bringing per-bin predicted probabilities close to per-
  bin observed accuracies.
- **~1.5% comes from resolution loss** — and this loss is exactly the G1
  finding's mechanism. **Within-fold rank invariance preserves resolution;
  the across-fold cross-source rank shuffle is the only artifact** that
  changes resolution at all. The 0.0018 RES drop quantifies the cost of
  the cross-source rank changes that drive G1's pooled AUC drop. Tiny in
  absolute terms; qualitatively zero relative to the reliability gain.
- **0% from uncertainty** — labels are unchanged, so UNC = ȳ(1−ȳ) is
  invariant by construction.

This decomposition vindicates the calibration-LOSO design choice: the
across-fold rank shuffle is the *only* per-source-T side-effect at the
classifier-quality layer, and it's a **0.0018 cost on resolution that buys
0.0135 gain on reliability** — a >7× exchange rate in favor of calibration.

The BS_binned vs BS_actual gap (~0.001-0.002) is the within-bin variance
correction, expected at this binning density. Both reliability and resolution
are computed with the same 10-bin equal-frequency partition; the ΔREL and
ΔRES values are directly comparable.

---

## Reliability diagram interpretation

(See `outputs/phase8c_reliability_diagram.png` — 2-panel side-by-side.)

**Pre-cal (left panel):** raw `sigmoid(decision_function)` produces probabilities
concentrated in [0.15, 0.92] — sigmoid of bounded scores cannot reach the tails.
The empirical accuracy curve is S-shaped, deviating from the diagonal:

- Lowest bin (predicted ~0.16) shows observed accuracy ~0.00 — model is
  overconfident on its low-confidence predictions
- Mid bins (0.45–0.55) cluster near 0.43–0.48 — slight overconfidence
- Top bins (predicted ~0.91) show observed accuracy ~0.97 — model is
  *underconfident* on its high-confidence predictions

The S-shape is the visual signature of a poorly calibrated classifier with
intact discrimination — exactly what raw RidgeClassifier sigmoid produces.

**Post-cal (right panel):** probabilities now span [0.05, 0.99] — full range.
The accuracy curve hugs the diagonal, with all 10 bins' Wilson 95% CIs
overlapping the perfect-calibration line. The lowest bins (predicted ~0.05,
~0.09) show observed accuracy ~0.00 — calibrated correctly. The highest bins
(predicted ~0.95, ~0.99) show observed accuracy ~0.95–0.97 — also calibrated
correctly.

**Visual verdict consistent with ECE = 0.04 well_calibrated.**

---

## What Phase 8c establishes

- **Calibration methodology infrastructure works on RME ear data.** ECE
  drops from 0.1118 to 0.0397 (well_calibrated band); Brier from 0.1420 to
  0.1289; Murphy decomposition shows the improvement is overwhelmingly
  reliability-driven (87% vs 1.5% resolution change).
- **Reusable infrastructure components.** Temperature scaling fitter,
  source-aware calibration LOSO loop, ECE/Brier/NLL/Wilson-CI machinery,
  reliability diagram generator, source-bootstrap CI on operating-point
  TPR, exact Poisson-binomial CDF for session-level OP — all packaged in
  `tools/phase8c_calibration.py`, all reusable on any future
  `(score, label, source)` tuple stream.
- **Per-behavior operating point τ_ear = 0.8138 at FPR = 0.05.** TPR at
  this threshold is 0.5435 with 95% CI [0.3077, 0.7500] from B=1000
  source-resampled bootstrap. Wide CI is honest characterization at
  n=283 / 12 sources — bootstrapping at the source level inflates CIs
  relative to clip-level bootstrapping.
- **Session-level operating point analytical chain stands.** Under the
  independence assumption, P(session flagged | H0) ≈ 1.39e−5 (exact
  Poisson-binomial). The independence assumption is the load-bearing
  caveat (Limitation 1) — under correlated firings (likely for RHpE)
  the actual session FPR is higher than this analytical floor. **Direction
  of violation unknown without empirical co-occurrence verification.**
- **G1's structural finding** — per-source calibration LOSO does NOT
  preserve pooled AUC. Per-source AUC IS bit-exact invariant (12/12
  sources). The pooled drop (Δ = -0.0022) is purely cross-source rank
  shuffle. Future per-source-calibration designs should adopt
  calibration-LOSO-aware sanity checks; the G1 sanity check as locked
  in pre-reg was based on a global-T monotonicity assumption that D2
  (per-source T) violates by construction.
- **Methodological observation on Poisson approximation** — generalizable
  beyond Phase 8c: at RHpE-relevant scale (n=24, λ ≈ 1.2), Poisson
  approximation overestimates Poisson-binomial tail by 1.66×. Use exact
  computation in any future RHpE session-level OP analysis.

## What Phase 8c does NOT establish

- **Not a calibration package for RHpE as a whole.** Single-behavior
  scope. Each future behavior added needs its own per-behavior T, τ_k,
  reliability diagram, ECE/Brier — not amortizable across behaviors.
- **Not a session-level RHpE classifier.** The session-level OP analytical
  derivation (1.39e−5 under independence) is an extrapolation under
  unverified assumptions, not a measured property of the model.
- **Not evidence that per-behavior independence holds.** Limitation 1
  (the load-bearing assumption violation risk) is unaddressed by Phase 8c.
  Empirical co-occurrence verification is Phase 9+ work, gated on N
  expansion beyond ear movement.
- **Not a multi-rater κ resolution.** Single-observer (RME paper labels)
  caveat from Phase 8b carries forward. Calibration cannot fix label
  noise; it can only reflect it. Calibrated probabilities are honest
  about the (noisy) label distribution, not about the underlying
  ground-truth pain state.
- **Not a comparison of calibration methods.** Temperature scaling was
  locked single-method per pre-reg D1 anti-pattern lock; no isotonic /
  Platt-2-param / Bayesian alternative was tested. This is by design,
  not omission.
- **Not a re-tuning of the operating point on TPR or any other metric.**
  τ_ear was selected at FPR = 0.05 per pre-reg D4 anti-pattern lock,
  with Dyson 2018 baseline rate (~2/24 in non-lame horses ≈ 0.083 per-
  behavior FPR) as the empirical anchor. The resulting TPR (0.5435 with
  CI [0.31, 0.75]) is the finding, not a number to optimize.
- **Not a Phase 9 prereg.** Phase 9 priorities (multi-rater κ, simplified-
  B1 long-form aggregation, reframed cropping value-proposition
  investigation) draft separately, gated on SLU/Palichleb response and
  on this audit doc landing.

## Limitations (carry forward from pre-reg, all locked)

1. **L1 — Independence assumption (load-bearing).** RHpE behavior
   co-occurrence is plausibly nonzero (tail swish + ear pinning,
   eye-cluster behaviors, head/body co-occurrence). Published
   correlation matrices not located in `v2/research/q3_architecture.md`
   open-question survey. Empirical verification deferred to Phase 9+
   when N expands beyond single-behavior.
2. **L2 — Single-behavior scope.** Ear movement only. Each future behavior
   needs its own per-behavior calibration. Infrastructure reusable; per-
   behavior work not amortizable.
3. **L3 — Single-observer label noise carries forward.** RME paper labels
   are single-observer. Calibration reflects but cannot fix label-noise
   ceiling. Multi-rater κ on ≥20% audit subset remains the load-bearing
   missing methodology step (synthesis Q6 + SLU outreach).
4. **L4 — RidgeClassifier vs LogisticRegression deferral.** Phase 8b
   pipeline uses RidgeClassifier; sigmoid + temperature on its
   `decision_function` output is the canonical Platt-style restriction
   (Guo 2017). A future LogisticRegression re-implementation would
   produce probabilities natively. Switching classifier is a Phase 9
   decision, not a Phase 8c question.

## Phase 9 entry condition signals (forward-look, not Phase 9 scope)

Phase 8c's landing **unblocks** the following Phase 9 candidates (entry
condition signals, not scope locks; Phase 9 prereg drafts in a separate
cycle per the discipline pattern):

- **Simplified-B1 long-form aggregation pipeline** is now ready to draft —
  consumes Phase 8c's calibrated probabilities (`outputs/phase8c_audit_extras.json`'s
  `prob_post_cal`) and τ_ear (0.8138 from `outputs/phase8c_calibration_results.json`).
  Per Dyson 2018 presence/absence scoring rule (`v2/research/dyson_scoring_check.md`),
  the simplified pipeline is sub-day work: max sliding-window calibrated
  probability per behavior → threshold at τ → presence/absence → sum 24
  binaries → ≥8 flag.
- **Phase 9 calibration-LOSO-aware sanity check redesign** (Finding 1
  forward-look) — relax G1 to per-source AUC invariance + bounded pooled
  AUC drift conditional on T variance. Specific formulation deferred to
  Phase 9 drafting.
- **Empirical RHpE behavior co-occurrence verification** (Limitation 1)
  — once N expands beyond ear, compute co-occurrence matrix on training
  set and test the independence assumption. This is the one experiment
  that could materially change the session-level OP claim.
- **Multi-rater κ on ≥20% audit subset** — gated on SLU
  (`v2/outreach/slu_collaboration_email.md`) or Palichleb
  (`v2/outreach/palichleb_outreach_email.md`) response. Independent of
  Phase 8c's calibration but synergistic with the simplified-B1 deliverable
  for clinical-utility framing.

## Methodology trail — discipline pattern preserved

Phase 8c continued the discipline pattern that protected Phases 5/6/7/8a/8b
through their own audit cycles:

- **Pre-registration locked before compute.** Five locked decisions, eight
  anti-patterns, four limitations all surfaced pre-lock. Stage 1 pre-reg
  hash-lockable; numerical results computed against the locked spec.
- **Anti-pattern lock #7 respected on G1.** No retroactive softening of the
  1e−10 invariance threshold despite the structural finding that the
  threshold was based on a flawed premise. The Δ_AUC = -0.0022 is
  reported as the property-of-record; the redesign is forward-routed
  to Phase 9.
- **G1 finding handled per Phase 7 §4 precedent.** Phase 7's anatomical-
  mapping inversion was reported as Lesson 20 with the locked rule kept
  intact and a Stage 2 amendment routed forward; Phase 8c's G1 D1↔D2
  inconsistency follows the same pattern — report the finding, route
  the redesign forward, do not retro-edit the pre-reg.
- **Pre-locked verdict-reporting protocol surfaces both PASS and FAIL gates
  jointly.** WELL_CALIBRATED + G1 STRUCTURAL FINDING is the joint reading;
  reporting only "WELL_CALIBRATED" without the G1 caveat would be an
  anti-pattern violation.
- **Empirical sub-finding strengthening.** Per-source AUC invariance check
  (added at user-checkpoint #2's suggestion, computed before audit doc
  draft) provides bit-exact mechanistic confirmation of the G1 finding's
  interpretation — this is the strongest single empirical hit of the audit
  cycle, and it came from the user-prompted "easily computable" addition
  rather than the locked pre-reg instrumentation. **Worth flagging the
  pattern** for future audit cycles: pre-flight numeric checks during
  checkpoint #1 or #2 that surface mechanistic interpretations of expected
  vs observed gates often produce strong audit-doc material.

### What this audit cycle protected against

- **Silent celebration of "well_calibrated" verdict** without the G1 caveat
  — the pre-locked verdict-reporting protocol surfaced both gate verdicts
  jointly.
- **Retroactive softening of G1 threshold** to manufacture a PASS — anti-
  pattern lock #7 prevented this.
- **Method shopping for a calibration that "preserves AUC"** — D1 anti-
  pattern lock prevented this.
- **Source-specific calibration "fix" for the 3 flat sources** — Finding 4
  anti-pattern preservation prevented this.
- **Treating the Poisson approximation as load-bearing** — Finding 3's
  generalizable methodological observation rooted the analytical chain
  on the exact Poisson-binomial calculation; the approximation is
  reported as supporting documentation only.

## Pre-registration audit chain

Phase 8c artifacts:

- `outputs/track_b_phase8c_preregistration.md` (Stage 1 pre-reg, ~280 lines, 5 locked decisions + 8 anti-patterns + 4 known limitations)
- `tools/phase8c_calibration.py` (~620 lines after format pass; 14 top-level functions; matches all 5 locked decisions per import-time + AST verification)
- `outputs/phase8c_calibration_results.json` (~7 KB; pooled metrics + per-fold T + per-source ECE/Brier + operating point + session OP both methods + gates + config + limitations)
- `outputs/phase8c_audit_extras.json` (~102 KB; per-clip pre-cal, post-cal, T_applied, bin assignments, above-τ flags + bin records)
- `outputs/phase8c_reliability_diagram.png` (~123 KB; 2-panel pre/post with Wilson CI + bin-count underlay + diagonal reference)
- `docs/phase8c_audit.md` (this document)

A future reader can reconstruct: what was locked, what was observed,
what gates fired, what findings were surfaced, what disposition each
finding received, what was deferred to Phase 9, and what was
explicitly NOT changed despite empirical pressure.
