# Track B — Phase 8c (B3 calibration package on Phase 8b RME ear output)

## Stage 1 pre-registration

**Drafted 2026-05-10. Frozen on user approval BEFORE any 8c compute.**
This pre-reg locks the methodology, metrics, gates, and anti-patterns
for Phase 8c — a calibration-package spike that operates on the
existing Phase 8b LOSO output (`outputs/eye_loso_results_phase8b.json`,
283 clips × 12 sources, RidgeClassifier `decision_function` scores) and
produces (i) calibrated per-clip probabilities, (ii) reliability
diagram + ECE + Brier + NLL, (iii) per-fold operating thresholds at
fixed FPR, and (iv) an analytical session-level operating-point
derivation under the independence assumption.

**Naming bridge:** the user-facing label is "B3 calibration package"
(per `v2/research/synthesis.md` Q6 + post-Phase-8b discussion). The
project-internal label is **Phase 8c** to match the established
phase-numbered audit/preregistration nomenclature. Both refer to the
same artifact set.

**No new data, no new compute on the spine.** Phase 8c does not
re-run the classifier, re-extract embeddings, or modify the Phase 8b
audit chain. It consumes Phase 8b's existing `per_clip` array and
produces strictly downstream calibration artifacts.

---

## ⚠ Framing — what Phase 8c IS and IS NOT

### What Phase 8c IS

- **Calibration methodology infrastructure** demonstrated on one
  behavior (RHpE ear movement, Phase 8b output).
- A reusable temperature-scaling + reliability-diagram + ECE/Brier +
  τ-selection pipeline that takes any `(per_clip score, label, source)`
  tuple stream and produces calibrated probabilities + audit-grade
  reporting.
- A small but real analytical step toward TRIPOD+AI / STARD-AI / MI-
  CLAIM compliance: ECE, Brier, reliability diagram are explicitly
  table-stakes per Pacholec Part II (`v2/research/synthesis.md` Q5).
- An operating-point selection mechanism (per-behavior τ_k at fixed
  FPR=0.05 on negative-source clips) that gives downstream session
  scoring a defensible per-behavior threshold under Dyson 2018's
  presence/absence rule (`v2/research/dyson_scoring_check.md`).

### What Phase 8c is NOT

- **NOT "the calibration package for RHpE."** Phase 8c calibrates
  ear-movement only (one of 24 behaviors). Generalization to other
  behaviors is asserted on architectural grounds (per-behavior
  independent probes, `v2/research/q3_architecture.md`) but is empirically
  untested — each future behavior will need its own per-behavior T.
- **NOT a session-level RHpE classifier.** The session-level operating-
  point analysis (Decision 5) is an analytical extrapolation under the
  **independence assumption across the 24 RHpE behaviors**, which is
  empirically unverified. See "Known limitations" below.
- **NOT a comparison of calibration methods.** Temperature scaling is
  locked here on the basis that isotonic overfits at N≈300 (Ojeda 2023)
  and Platt-with-2-parameters is unnecessary at single-behavior scope.
  Phase 8c does not "shop methods."
- **NOT a re-run of Phase 8b's LOSO.** The classifier outputs from
  Phase 8b are taken as fixed. Phase 8c operates strictly downstream.
- **NOT a multi-rater κ resolution.** Single-observer (RME paper labels)
  caveat from Phase 8b carries forward unchanged.
- **NOT a Phase 9 prereg.** Phase 9 (multi-rater κ + N expansion)
  is gated on SLU collaboration response (`v2/outreach/`) and is
  drafted separately after 8c lands.

---

## Locked design decisions

### Decision 1 — Calibration method: temperature scaling (one parameter)

**Decision:** post-hoc temperature scaling applied to RidgeClassifier
`decision_function` outputs. For each clip *i*:

```
p_i = σ(score_i / T)        where σ(x) = 1 / (1 + exp(-x))
```

T fit by minimizing negative log-likelihood (NLL) on a held-out
calibration set (per Decision 2, calibration LOSO).

**Rationale:**
- Single parameter (T > 0) — appropriate at N=283 / 12 sources where
  isotonic regression overfits (Ojeda 2023, *Stat Med* 10.1002/sim.9921;
  cited in `v2/research/synthesis.md` Q6).
- RidgeClassifier produces signed distance from hyperplane, not
  probability. Sigmoid + temperature is the canonical "Platt-style"
  conversion (Platt 1999 introduced sigmoid + 2 parameters; temperature
  scaling is the 1-parameter restriction, Guo et al. 2017).
- Preserves the rank ordering of `decision_function` outputs (T > 0
  is monotonic), so per-clip ranking and AUC are mathematically
  unchanged. **AUC is invariant to temperature scaling.**
- One-parameter fit is robust and reproducible across runs (no
  isotonic step-function noise).

**Anti-pattern lock (Decision 1):** do not fit a second parameter
("learnable bias") under a "Platt is more general" rationale. The
1-param restriction is locked on the basis of N-appropriate
methodology, not on the basis of observed fit quality.

### Decision 2 — Calibration LOSO (source-aware, no double-dipping)

**Decision:** for each held-out source S ∈ {S1..S12}, fit T on the
283 − n_S OOF scores from the other 11 sources, then apply T to source
S's OOF scores.

```
For S in sources:
    train_S = {(score_i, label_i) : source_i ≠ S}
    T_S = argmin NLL(σ(score / T), label) over train_S    # T > 0
    For i in source S:
        p_i = σ(score_i / T_S)
```

**Rationale:**
- The Phase 8b `per_clip.score` values are **already** out-of-fold
  predictions (each clip's score was produced by a model trained on
  the OTHER 11 sources). Fitting T on all 283 scores then evaluating
  on the same 283 scores would be circular.
- Calibration LOSO is the natural extension of the project's existing
  source-aware LOSO discipline. It costs nothing (T fit is seconds per
  source) and is the only honest evaluation of calibration generalization.
- Produces 12 fold-specific T values — informative diagnostic for
  whether calibration generalizes uniformly across sources, or whether
  some sources need source-specific T. (Diagnostic, not gated.)

**Anti-pattern lock (Decision 2):** do NOT fit a single global T on
all 283 OOF scores and report ECE on those same 283 scores. That is
in-sample calibration and produces optimistic ECE. The reported ECE
must be from the source-LOSO calibration loop above.

### Decision 3 — Metrics

**Locked metric set (all reported, all in audit doc):**

| Metric | Definition | Reporting |
|---|---|---|
| **ECE** (Expected Calibration Error) | 10 equal-frequency bins; mean over bins of \|acc(b) − conf(b)\| weighted by bin count / N | Pre-cal + post-cal + delta |
| **Brier score** | (1/N) Σ (p_i − y_i)² | Pre-cal + post-cal + delta |
| **NLL** (negative log-likelihood) | (1/N) Σ −[y_i log p_i + (1−y_i) log(1−p_i)] | Post-cal only (pre-cal NLL on raw scores requires a calibration to define probabilities anyway) |
| **Reliability diagram** | 10 equal-frequency bins; per-bin (mean confidence, accuracy) plot with diagonal reference | Pre-cal + post-cal side-by-side, single PNG |
| **Per-fold T values** | T_S for each S ∈ {S1..S12} | Table; flag if T_S < 0.5 or > 5 (extreme) |
| **AUC** | Pre-cal pooled AUC (= Phase 8b AUC = 0.9008) and post-cal pooled AUC | Required to be equal — temperature scaling is monotonic; mismatch is a bug indicator |

**Bin choice rationale:** equal-frequency (quantile) binning over
equal-width because (a) score distribution under RidgeClassifier is
not uniform — equal-width creates near-empty bins at the extremes
and over-loaded bins near 0; (b) per-bin sample-count weighting in ECE
becomes meaningless when bins are near-empty. 10 bins gives ~28 clips
per bin at N=283, sufficient for stable per-bin estimates.

**Anti-pattern lock (Decision 3):** do not switch bin count post-hoc
based on observed reliability-diagram smoothness. 10-equal-frequency is
locked here.

### Decision 4 — Per-behavior threshold τ at fixed FPR=0.05

**Decision:** select τ_ear such that FPR on negative-source clips
(label=0) is exactly 0.05, where the FPR is computed across the pooled
calibration-LOSO output (all 283 calibrated probabilities).

```
neg_probs = [p_i for i in {0..282} : label_i = 0]
τ_ear = quantile(neg_probs, 0.95)
TPR_ear = mean(p_i ≥ τ_ear for i in {0..282} : label_i = 1)
```

**Rationale:**
- Dyson 2018 reports baseline behavior count of ~2/24 in non-lame
  horses (synthesis Q4 + `dyson_scoring_check.md`). This is consistent
  with a per-behavior FPR of ~2/24 ≈ 0.083 in the field. Setting τ at
  FPR=0.05 is conservative-but-anchored; reports both the chosen
  threshold and TPR at the threshold.
- Bootstrap CI on TPR (B=1000 source-resampled bootstrap, seed=42, same
  protocol as Phase 8b's subject bootstrap) reported alongside the
  point estimate.

**Anti-pattern lock (Decision 4):** do not "tune" τ to maximize TPR
or any other metric. FPR=0.05 is the locked target. If the resulting
TPR is uninformative, that is the finding, not a reason to re-tune.

### Decision 5 — Session-level operating-point derivation (analytical, under independence)

**Decision:** under the assumption that the 24 RHpE behaviors fire
independently given the underlying state, the session-level binary
classifier "≥8/24 behaviors flagged" has an operating point computable
analytically as the tail of a sum of independent Bernoulli random
variables (Poisson-binomial distribution).

```
Under H0 (no pain):
    For each behavior k, P(flagged | H0) = FPR_k = 0.05 (locked, Decision 4)
    Session FPR_8 = P(Σ_k flagged_k ≥ 8 | H0)
                  = 1 - F_PoissonBinomial(7; n=24, p=0.05)
                  ≈ 1 - F_Poisson(7; λ=24*0.05=1.2)
                  ≈ 1.7e-5  (rough, exact depends on per-behavior FPR variance)

Under H1 (pain present):
    For each behavior k, P(flagged | H1) = TPR_k (per behavior)
    Session TPR_8 = P(Σ_k flagged_k ≥ 8 | H1)
```

For Phase 8c (single behavior calibrated), the analytical computation
is reported with **explicit notation** that it assumes:
1. All 24 RHpE behaviors achieve calibration comparable to ear (each
   per-behavior FPR = 0.05). Empirically untested.
2. Per-behavior firings are statistically independent given the
   subject. **Empirically unverified — flagged as the load-bearing
   limitation.**

**Rationale:**
- Without this analytical step, Phase 8c produces only a per-behavior
  metric, not a session-level interpretation. The session-level
  operating point is what TRIPOD+AI / STARD-AI requires for clinical-
  utility framing.
- The analytical computation is small (Poisson-binomial CDF, scipy or
  hand-coded). Cost is negligible.
- Reporting both the session-level computation AND its load-bearing
  assumptions is the correct epistemic posture: the result is
  conditional on assumptions, and the assumptions are explicit.

**Anti-pattern lock (Decision 5):** do NOT report the session-level
operating point as a measured property of the model. It is an
analytical extrapolation under stated assumptions. The audit doc must
present the assumptions inline with the computation, not in a footnote.

---

## Test hierarchy

Phase 8c is a calibration-quality assessment, not a comparison test.
There is no "model A vs model B" question. Test hierarchy:

| Test | Role | Threshold | What it tells us |
|---|---|---|---|
| **Test 1 — Pre-cal vs post-cal ECE** | **Load-bearing** | Post-cal ECE < pre-cal ECE | Temperature scaling actually improves calibration (sanity check; expected) |
| **Test 2 — Post-cal ECE absolute level** | Reportable | < 0.05 ≈ "well calibrated"; 0.05–0.10 acceptable; > 0.10 poorly calibrated | Calibration quality ranking (informational, not gated) |
| **Test 3 — Brier reduction** | Reportable | Δ > 0 | Probabilistic-prediction quality improvement |
| **Test 4 — Per-source ECE consistency** | Diagnostic | All 12 sources' ECE in interval | Whether calibration generalizes uniformly across sources |
| **Test 5 — Pre-cal vs post-cal AUC** | Sanity check | Equal to within numerical noise (≤1e-10) | Temperature scaling is monotonic; mismatch indicates bug |
| **Test 6 — Per-fold T plausibility** | Diagnostic | All T_S in [0.3, 5.0] | Extreme T values would indicate per-source distribution shift in scores |
| **Test 7 — Bootstrap CI on TPR @ FPR=0.05** | Reportable | B=1000 source-resampled | Precision of the per-behavior operating-point claim |

---

## Locked gates

| Gate | Threshold | Action |
|---|---|---|
| **G1 — Sanity** | post-cal AUC = pre-cal AUC (≤1e-10 deviation) | Halt and debug if violated |
| **G2 — Calibration improvement** | post-cal ECE < pre-cal ECE | If violated: report as "calibration neutral on already-well-calibrated raw scores" — possible at AUC 0.90 — and report both ECEs without escalation. Surfaces honestly. |
| **G3 — Calibration quality (informational)** | post-cal ECE thresholds: <0.05 (well) / 0.05–0.10 (acceptable) / >0.10 (poor) | Report verdict band; do not re-tune |
| **G4 — Per-fold T plausibility** | All T_S ∈ [0.3, 5.0] | Out-of-band T values flagged in audit doc; investigate source-specific score distribution |

**G2 framing:** unlike Phase 8b's gates which fired pass/fail verdicts
on a comparison, Phase 8c's gates are calibration-quality reports.
Failing G2 is not a Phase 8c failure — it is a finding. Anti-pattern
lock: do not adjust T fitting procedure if G2 fails; report and move on.

---

## Diagnostic instrumentation (reportable, not gated)

1. **Per-fold T values** (12 numbers + median + IQR)
2. **Per-source ECE** (pre-cal and post-cal, 12 sources × 2)
3. **Per-source Brier** (12 sources × 2)
4. **Reliability diagram** PNG: 2 panels (pre-cal, post-cal), 10 equal-
   frequency bins, diagonal reference, error bars (Wilson 95% CI on
   per-bin accuracy), bin-count overlay as bar chart underlay.
5. **Per-clip extras**: clip × source × pre-cal score × post-cal prob ×
   label × calibration-LOSO held-out flag. Stored in
   `outputs/phase8c_audit_extras.json` for downstream Phase 9 use.
6. **Operating-point report**: τ_ear at FPR=0.05, TPR at τ_ear,
   bootstrap 95% CI on TPR.
7. **Session-level analytical computation**: under the locked Decision 5
   independence assumption, P(session flagged | H0) and the per-
   behavior TPR needed for session-level TPR thresholds (50%, 80%, 95%).

---

## Anti-patterns (LOCKED)

1. **No method shopping.** Temperature scaling is locked. Do not fit
   isotonic, Platt-2-param, or any alternative calibrator and report
   "best wins." Decision 1 framing.
2. **No global-T calibration on in-sample data.** Calibration LOSO
   (Decision 2) is the locked evaluation; reporting ECE on a globally
   fit T over the same scores is the methodology bug to avoid.
3. **No τ tuning on TPR or any non-FPR metric.** FPR=0.05 is the
   locked operating-point target. Decision 4.
4. **No claim of session-level operating point as measured.** It is
   an analytical extrapolation under the stated independence
   assumption. Decision 5.
5. **No bin-count adjustment based on observed reliability diagram
   smoothness.** 10 equal-frequency bins is locked. Decision 3.
6. **No re-running of Phase 8b classifier, embeddings, or LOSO.** Phase
   8c is strictly downstream. The Phase 8b `per_clip` array is the
   fixed input.
7. **No goal-shifting on calibration-quality verdicts.** G3's bands
   (well / acceptable / poor) are locked here at 0.05 / 0.10. Don't
   relabel "0.07 is well-calibrated" post-hoc.
8. **No suppression of out-of-band T values.** If any T_S < 0.3 or
   > 5.0, report it explicitly in the audit doc and discuss — don't
   silently clip.

---

## Known limitations (surfaced pre-lock)

These are the load-bearing caveats on Phase 8c's interpretation. Each
must appear explicitly in `docs/phase8c_audit.md`'s "What 8c does NOT
establish" section.

### Limitation 1 — Independence assumption for session-level operating point

The session-level operating-point computation (Decision 5) assumes the
24 RHpE behaviors fire independently given the underlying pain state.
**This assumption is empirically unverified.** RHpE behaviors are
plausibly correlated:

- **Tail swish + ear pinning** may co-occur as a generalized stress
  response.
- **Eye expression behaviors** may cluster (a horse with anxious
  expression may display multiple eye-region behaviors together).
- **Head-position behaviors** may cluster with body-position behaviors
  (musculoskeletal compensation).

Published RHpE behavior co-occurrence matrices were not located in
the q3_architecture survey (`v2/research/q3_architecture.md` Open Question:
"Published RHpE behavior-correlation matrices not located; compute co-
occurrence on the PoC's own training set early"). Phase 8c flags this
as the most load-bearing assumption violation risk for the session-level
extrapolation. Empirical co-occurrence verification is **explicitly
deferred to Phase 9 or later** — feasible only when N expands beyond
ear-movement to multiple behaviors with paired labels.

If behaviors are positively correlated, the session-level FPR is
**higher** than the independence model predicts (more co-occurrence →
more sessions trip ≥8 by chance). If negatively correlated, FPR is
lower. Direction unknown without empirical verification.

### Limitation 2 — Single-behavior scope

Phase 8c calibrates ear movement only. Each future behavior added to
the project will need:
- Its own per-behavior temperature scaling (per-behavior probe → per-
  behavior T)
- Its own per-behavior τ_k at FPR=0.05
- Per-behavior reliability diagram + ECE + Brier
- Its own calibration-LOSO loop (separate per behavior)

The infrastructure built in Phase 8c is reusable but the per-behavior
work is not amortizable across behaviors.

### Limitation 3 — Single-observer label noise carries forward

Phase 8b labels are RME paper labels (single-observer convention).
Calibration cannot fix label noise; it can only reflect it. Per-behavior
calibration on noisy labels produces calibrated probabilities that are
honest about the (noisy) label distribution, not about the underlying
ground-truth pain state. **Multi-rater κ on at least a 20% audit subset
remains the load-bearing missing methodology step** (synthesis Q6 +
SLU outreach).

### Limitation 4 — RidgeClassifier vs LogisticRegression choice carries forward

The Phase 8b classifier is `RidgeClassifier(alpha=1.0,
class_weight='balanced')`, whose `decision_function` is a
hyperplane-distance, not a log-odds. Temperature scaling on this
output is the canonical Platt-style restriction (Guo 2017), which is
appropriate, but a future re-implementation with `LogisticRegression`
would produce probabilities natively and potentially require less
post-hoc calibration. **Whether to switch classifier is a Phase 9
decision, not a Phase 8c question.** Phase 8c locks RidgeClassifier
in for compatibility with the Phase 8b spine.

---

## Sequencing

| Step | Action | Output |
|---|---|---|
| 0 | User approval of this Stage 1 pre-reg | hash-locked artifact |
| 1 | Build `tools/phase8c_calibration.py` (load Phase 8b output, calibration LOSO, metrics, plots) | tool source |
| 2 | Run `phase8c_calibration.py` end-to-end (seconds, not minutes) | `outputs/phase8c_calibration_results.json`, `outputs/phase8c_audit_extras.json`, `outputs/phase8c_reliability_diagram.png` |
| 3 | Audit doc draft `docs/phase8c_audit.md` | doc |
| 4 | User-approval checkpoint #2 (audit doc lock) | — |
| 5 | Hash chain + commit | mirror sync |

**User-approval checkpoints (2, lighter than Phase 8b's 5 because
this is downstream-only):**

1. After this Stage 1 doc approval, before any 8c compute or tool build.
2. After audit doc draft (Step 3), before commit + push.

---

## Cost / time estimate

| Step | Estimate |
|---|---:|
| Pre-reg approval cycle | ~15–30 min |
| Tool implementation (Step 1) | ~120 min |
| Run + verify (Step 2) | ~10 min compute + ~30 min review |
| Audit doc draft (Step 3) | ~60 min |
| Commit + push (Step 5) | ~15 min |
| **Total wall-clock** | **~4–5 hours over ~1 day** |

This matches the "B3 = 1-day calibration package, solo-shippable"
estimate from `v2/research/synthesis.md` Q6 and the post-Phase-8b
discussion thread.

---

## Phase 9 entry conditions (forward-look, not part of 8c lock)

Phase 8c does not gate Phase 9 entry by calibration-quality verdict —
calibration is downstream methodology, not a comparator. Phase 9 entry
remains gated on:

- **SLU collaboration response** (`v2/outreach/slu_collaboration_email.md`)
  for multi-rater κ track
- **Palichleb response** (`v2/outreach/palichleb_outreach_email.md`)
  for Polish vet network bridge / Dyson channel
- **Phase 8c calibration package landing** as the methodology infrastructure
  prerequisite

Phase 9 prereg drafts after Phase 8c lands. Phase 9 priorities (per
`v2/research/synthesis.md` ranking and `phase8b_audit.md` routing):

1. Multi-rater κ on 20% audit subset (gated on SLU/Palichleb response)
2. Simplified-B1 long-form aggregation pipeline (per `dyson_scoring_check.md`
   presence/absence scope) — sub-day work; uses Phase 8c calibrated
   probabilities + τ_ear from Decision 4
3. Reframed cropping value-proposition investigation (per `phase8b_audit.md`
   reframed motivation) — computational efficiency, interpretability,
   per-source robustness

---

## User approval signature

User has reviewed and approves Phase 8c Stage 1 lock as drafted, including:

- Decision 1: temperature scaling (one parameter), σ(score / T)
- Decision 2: calibration LOSO (source-aware, no double-dipping)
- Decision 3: ECE (10 equal-frequency bins) + Brier + NLL + reliability diagram
- Decision 4: τ_ear at FPR=0.05 on negative-source clips, with bootstrap CI on TPR
- Decision 5: session-level OP via Poisson-binomial under independence (analytical, with explicit assumption surfacing)
- 4 locked gates (G1 sanity / G2 calibration improvement / G3 quality bands / G4 T plausibility)
- 8 anti-patterns
- 4 known limitations surfaced pre-lock (independence assumption load-bearing)
- 2 user-approval checkpoints
- Naming: project-internal "Phase 8c", user-facing "B3 calibration package"

User signs off → CC executes Step 1 (tool implementation) → CC runs
Step 2 (compute) → CC drafts audit doc Step 3 → user approves at
checkpoint #2 → final commit. No SLU/Palichleb response required —
Phase 8c is solo-shippable in parallel with email lane.
