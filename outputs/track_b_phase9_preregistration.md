# Track B — Phase 9 (Simplified-B1 long-form aggregation + G1 sanity-check redesign)

## Stage 1 pre-registration

**Drafted 2026-05-11. Frozen on user approval BEFORE any 9 compute.**
This pre-reg locks the methodology, metrics, gates, and anti-patterns
for Phase 9 — the **simplified-B1 long-form aggregation pipeline** (per
Dyson 2018 presence/absence scoring rule, locked in
`v2/research/dyson_scoring_check.md`) plus a small **calibration-LOSO-aware
sanity check redesign** (the G1 forward-look from Phase 8c Finding 1).
Both items are solo-shippable; multi-rater κ track and behavior
co-occurrence verification are out of Phase 9 scope and routed to
later phases.

**No new training, no new compute on the classifier or calibration.**
Phase 9 operates strictly downstream of Phase 8c's outputs
(`outputs/phase8c_calibration_results.json` and
`outputs/phase8c_audit_extras.json`). It does not modify the Phase 8b/8c
audit chain.

**Naming bridge:** user-facing label is "simplified-B1 + sanity-check
redesign" (per the post-Phase-8c discussion thread and the v2 synthesis
Q4/Q6 cross-cuts). Project-internal label is **Phase 9** to match the
established phase-numbered audit/preregistration nomenclature.

---

## ⚠ Framing — what Phase 9 IS and IS NOT

### What Phase 9 IS

- **Methodology lock for the simplified-B1 long-form aggregation pipeline.**
  Per behavior k: `present_k = (max_window_prob_k ≥ τ_k)`. Session score
  = Σ_k present_k. Flag if ≥ 8/24 per Dyson 2018. Parameterised to
  extend trivially to k = 24 once additional per-behavior probes exist.
- **Reusable implementation** packaged in a single self-contained Python
  tool that consumes Phase 8c's `per_clip` calibrated probabilities and
  `τ_ear` to produce per-clip presence flags + per-source confusion
  matrices + a forward-look multi-behavior session-level scaffold.
- **Calibration-LOSO-aware sanity check redesign** (Phase 8c G1 forward-
  look operationalised). Relaxes the global-T-monotonicity G1 to
  per-source AUC invariance (bit-exact for D2-compliant designs) with
  a bounded pooled-AUC-drift secondary gate. Locked as the standard
  G1 for any future per-source calibration phase.
- **Honest characterisation of the single-behavior demonstration.**
  At k = 1 the "session score" is binary {0, 1}; the ≥ 8/24 threshold
  is not exercised. The pipeline scaffold demonstrates the *mechanism*,
  not the *clinical claim*. Future multi-behavior expansion (Phase 10+)
  exercises the threshold.

### What Phase 9 is NOT

- **NOT session-level pain detection.** Single behavior (ear movement);
  the Dyson ≥ 8/24 rule requires at least 8 distinct behaviors and
  cannot fire at k = 1.
- **NOT a learned temporal aggregation.** The simplified pipeline is
  max-window-prob + threshold, by design (per Dyson presence/absence
  scope-collapse from `dyson_scoring_check.md`). Learned TAL is
  permanently out of scope for the RHpE clinical claim.
- **NOT a sliding-window study.** RME clips are 5–15 s; each clip is
  treated as one window. Multi-window aggregation within longer
  sessions is deferred to a later phase (needs longer source videos).
- **NOT a re-tuning of τ_ear.** Phase 8c locked τ_ear = 0.8138 at
  FPR = 0.05. Phase 9 consumes this verbatim; does not re-fit.
- **NOT a calibration revisit.** Phase 8c's calibration outputs are
  taken as fixed. Phase 9 does not re-fit temperature scaling or
  re-run calibration LOSO.
- **NOT a multi-rater κ resolution.** Gated on SLU
  (`v2/outreach/slu_collaboration_email.md`) and Palichleb
  (`v2/outreach/palichleb_outreach_email.md`) response. Independent
  Phase 10+ work.
- **NOT a behavior co-occurrence verification.** Requires N expansion
  beyond ear movement (Phase 10+); Phase 9 inherits the independence
  assumption from Phase 8c's session OP as a documented limitation.
- **NOT a re-run of Phase 8b/8c.** Phase 9 is strictly downstream.

---

## Locked design decisions

### Decision 1 — Aggregation rule: per-behavior max-window-prob + threshold (Dyson presence/absence)

**Decision:** for each behavior k ∈ {1..K} (current K = 1, ear), and
for each session (current "session" = one RME clip):

```
For each window w in session_w:
    score_w   = V-JEPA-2 features → linear probe → decision_function (Phase 8b)
    p_w       = sigmoid(score_w / T_source_aware)         (Phase 8c calibration LOSO)
max_window_prob_k = max over w of p_w
present_k         = 1 if max_window_prob_k ≥ τ_k else 0
```

Session score = Σ_k present_k. Flag for vet review if session score ≥ 8.

**Rationale:**
- Dyson 2018 + Dyson & Pollard 2023 score per-behavior as presence /
  absence (each behavior is either observed at least once during the
  session or not). Threshold ≥ 8/24 counts distinct behaviors observed,
  not frequencies. Locked in `v2/research/dyson_scoring_check.md`.
- Max-window-prob is the natural aggregator for "did this behavior
  occur at any point during the session?" — robust to short bursts and
  long quiescent periods alike.
- No temporal NMS, no event counting, no learned TAL — the bridge
  collapses to thresholding under the presence/absence rule
  (synthesis Q4 cross-cut).
- For the K = 1 demonstration each RME clip is treated as a session
  with one window (RME clips are short, ear movement is a discrete
  event present-or-absent over the clip). Multi-window aggregation
  becomes meaningful only when source videos are longer than the
  V-JEPA-2 window (16 frames at 224 × 224).

**Anti-pattern lock (D1):** do NOT switch to per-window event counting,
sliding-window NMS, or learned aggregation under the rationale that
"it might generalise better." The simplified rule is locked on
Dyson 2018 + 2023 evidence, not on observed performance.

### Decision 2 — Window definition for the K = 1 demonstration

**Decision:** for Phase 9's RME ear demonstration, **each RME clip =
one window**. The 283 clips become 283 one-window "sessions" for the
purpose of mechanically exercising the present_k logic.

**Rationale:**
- RME clips are 5–15 seconds; each clip is one V-JEPA-2 forward pass
  (16 frames evenly sampled). There is no multi-window structure
  within a clip to aggregate over.
- The pipeline scaffold must still parameterise the window→session
  mapping correctly so multi-window aggregation lights up trivially
  when applied to longer videos in a future phase.
- Treating each clip as a session yields per-source FPR/TPR that
  directly inherit Phase 8c's operating-point metrics — no new
  uncertainty added, just the mechanism wired correctly.

**Anti-pattern lock (D2):** do NOT introduce synthetic windowing
within RME clips to manufacture multi-window data. The single-window
demonstration is honest; synthetic chunking would add complexity
without clarifying the methodology.

### Decision 3 — Inputs and outputs (no recomputation)

**Decision:** Phase 9 tool consumes:
- `outputs/phase8c_audit_extras.json`'s `per_clip` array (283 records,
  each with `prob_post_cal`, `source`, `label`, `clip`, `above_tau_ear`)
- `outputs/phase8c_calibration_results.json`'s
  `operating_point.tau_ear` (= 0.8138)

Phase 9 tool produces:
- `outputs/phase9_simplified_b1_results.json` — per-clip presence_k,
  per-source confusion matrices, pooled FPR/TPR sanity, multi-behavior
  session-level scaffold parameters
- `outputs/phase9_audit_extras.json` — per-clip extras (clip, source,
  label, prob_post_cal, present_ear, would_session_flag_at_k_eq_1)
- `outputs/phase9_per_source_confusion.png` — per-source confusion
  matrix grid (12 panels, action_pred × action_true)

**Anti-pattern lock (D3):** do NOT call the V-JEPA-2 encoder, the
RidgeClassifier, or the temperature scaling fitter from Phase 9 code.
Phase 9 is strictly downstream of Phase 8c output JSONs.

### Decision 4 — Session-level scaffold for K = 24 (parameterised but not exercised)

**Decision:** Phase 9 tool exposes a `compute_session_score(per_clip,
behaviors, taus)` function that takes:
- `per_clip`: list of `{clip, source, label, prob_post_cal}` records
  (or extended to include `behavior` field for K > 1 inputs)
- `behaviors`: list of behavior keys (current default ["ear"])
- `taus`: dict mapping behavior → τ_k (current default {"ear": 0.8138})

and returns per-clip `present_k` for each behavior plus per-clip session
score (Σ_k present_k) and per-clip flag (≥ 8).

For Phase 9's K = 1 demonstration, this function is exercised but
session_score ∈ {0, 1} and the ≥ 8 flag never fires. The function is
documented + unit-tested with a synthetic K = 24 case covering four
boundary cases (all-zero → 0, seven → 0 [just-below-threshold,
off-by-one check], exactly-eight → 1 [at threshold], all-one → 1
[super-threshold]) to verify the threshold logic. **Synthetic test
data is not real RHpE data and does not constitute clinical
validation.**

**Rationale:**
- Parameterising for K = 24 now means future multi-behavior work
  (Phase 10+) consumes the same tool without redesign.
- The synthetic test cases verify the threshold mechanics (off-by-one,
  threshold direction) without claiming clinical applicability.
- The audit doc explicitly states that the multi-behavior scaffold is
  *mechanism-only*, not validated; clinical claims are blocked on
  multi-behavior probe development.

**Anti-pattern lock (D4):** do NOT report a "session-level RHpE
classifier" claim from Phase 9 outputs. Session score at K = 1 is
binary and the ≥ 8/24 rule is not exercised. The audit doc must
state this explicitly in the headline.

### Decision 5 — Calibration-LOSO-aware sanity check redesign (Phase 8c G1 forward-look)

**Decision:** redesign the calibration-quality sanity check (Phase 8c's
G1) to be appropriate for the D2-style per-source calibration LOSO
design. The locked replacement:

```
G1a — Per-source AUC invariance (LOAD-BEARING, bit-exact):
      For each source S, AUC(pre_cal[S], labels[S]) =
      AUC(post_cal[S], labels[S]) within 1e-10.
      PASS if all sources invariant; FAIL otherwise.

G1b — Bounded pooled AUC drift conditional on T variance (DIAGNOSTIC):
      Let:
        ΔAUC_pooled        = |AUC(pre_cal_pooled) − AUC(post_cal_pooled)|
                             where pre_cal_pooled and post_cal_pooled are
                             concatenations of per-clip scores across
                             all 12 sources from Phase 8c's per_clip output.
        range(T_per_source) = max(T_S) − min(T_S) over the 12 LOSO folds
                              (Phase 8c calibration_results.json).

      Locked invariant:
        ΔAUC_pooled ≤ k × range(T_per_source) with k = 0.04
        (anchored on Phase 8c: Δ_AUC = 0.0022, T_range = 0.0677, ratio 0.0325).

      Reportable: ratio = ΔAUC_pooled / range(T_per_source).
      FLAG if ratio > 0.04 (cross-source rank-shuffle effect exceeds
      Phase 8c's empirical baseline). FLAG is reportable, not gate-halting.
```

**Rationale:**
- Phase 8c demonstrated that the original G1 ("|ΔAUC_pooled| ≤ 1e-10")
  was based on the global-T-monotonicity claim from D1, which D2's
  per-source design violates by construction. Reporting G1 FAIL
  without softening was correct per anti-pattern lock #7, but the
  underlying check needs replacement for future phases.
- Per-source AUC invariance is the *correct* invariance for D2
  designs: each source's calibration is monotonic within source, so
  per-source rank ordering is preserved bit-exactly.
- Bounded pooled drift quantifies the across-source rank shuffle
  effect as a function of T variance, providing a calibrated
  reportable for future per-source calibration designs.
- k = 0.04 from Phase 8c's empirical anchor is the simplest defensible
  constant. Wider T variance ranges in future phases would test
  whether this scales linearly.

**Anti-pattern lock (D5):** do NOT retroactively re-grade Phase 8c's
G1 verdict using the new G1a/G1b rules. Phase 8c's G1 FAIL stands as
recorded (locked artifact); Phase 9's G1a/G1b is the redesign for
future phases.

### Decision 6 — Per-source FPR consistency check

**Decision:** verify that the per-source FPR at τ_ear = 0.8138 is
approximately FPR_TARGET = 0.05 (the operating-point design target),
or surface deviations honestly.

```
For each source S:
    FPR_S = (1/n_neg_S) × Σ_{i in S, label_i=0} [prob_post_cal_i ≥ τ_ear]
Report per-source FPR_S distribution; expected ~0.05 ± noise floor.
```

**Rationale:**
- Phase 8c selected τ_ear from the *pooled* negative-source clip
  distribution (Decision 4 of Phase 8c). Per-source FPR is implicit
  and may differ from 0.05 source-by-source due to calibration
  heterogeneity.
- Reporting per-source FPR provides a diagnostic for whether the
  pooled operating point is honest at the per-source level, or
  whether some sources have systematically higher/lower per-source
  FPR than the global target.
- This is a *reportable* not a *gate* — Phase 9 does not adjust τ_ear
  if per-source FPR varies. Source-specific operating points are
  Phase 10+ work if needed.

**Anti-pattern lock (D6):** do NOT introduce per-source τ_S in Phase 9.
The pooled τ_ear is the locked operating point; per-source variation
is documented, not corrected.

---

## Test hierarchy

Phase 9 is a methodology lock + demonstration on existing Phase 8c
output. There is no comparison test. Test hierarchy:

| Test | Role | Threshold | What it tells us |
|---|---|---|---|
| **Test 1 — Tool runs end-to-end** | Sanity | exit code 0 | Pipeline mechanics work on Phase 8c input format |
| **Test 2 — Per-clip present_k matches `above_tau_ear`** | Sanity | bit-exact across 283 clips | Tool consumes Phase 8c output correctly |
| **Test 3 — Pooled FPR ≈ FPR_TARGET on negatives** | Sanity | \|FPR_pooled − 0.05\| ≤ 0.005 | τ_ear inheritance from Phase 8c is honest |
| **Test 4 — Per-source FPR distribution** | Reportable | dispersion at min/max source | Whether per-source FPR is concentrated near 0.05 or wide |
| **Test 5 — Per-source TPR distribution** | Reportable | min/max source TPR | Per-source operating-point heterogeneity |
| **Test 6 — Synthetic K = 24 unit tests** | Sanity | all-zero → 0; **seven → 0 (just-below-threshold, off-by-one check)**; exactly-eight → 1 (at threshold); all-one → 1 (super-threshold) | Multi-behavior threshold mechanics correct + off-by-one guarded |
| **Test 7 — G1a per-source AUC invariance** | Methodology | bit-exact 12/12 | New calibration sanity check works as designed (re-verifies Phase 8c data) |
| **Test 8 — G1b pooled drift ratio** | Methodology | report ratio | Quantify Phase 8c-style cross-source rank shuffle |

---

## Locked gates

| Gate | Threshold | Action |
|---|---|---|
| **G1 — Tool execution** | exit code 0; no exceptions | Halt if violated |
| **G2 — Phase 8c consistency** | per-clip present_k bit-exact match to `above_tau_ear` field | Halt and debug if violated; indicates tool consumes wrong field or applies different τ |
| **G3 — Pooled FPR sanity** | \|FPR_pooled − FPR_TARGET\| ≤ 0.005 | FAIL surfaces τ_ear's effective FPR drift; report but do not adjust |
| **G4 — G1a per-source AUC invariance** | all 12 sources bit-exact (≤ 1e−10) | **HALT on FAIL** (downstream outputs not written; non-zero exit code). PASS confirms the new sanity check works; FAIL indicates either tool bug or per-source non-monotonicity that invalidates the gate machinery. Matches G1/G2/G5 halt-on-failure pattern, consistent with G1a's LOAD-BEARING bit-exact label in D5. |
| **G5 — Synthetic K = 24 unit tests** | all four boundary cases pass (all-zero, seven, exactly-eight, all-one) | Halt and debug if violated; threshold logic bug (incl. off-by-one) |

---

## Diagnostic instrumentation (reportable, not gated)

1. **Per-source FPR_S, TPR_S, n_pos_S, n_neg_S** (12 sources × 4 columns)
2. **Per-source confusion matrix grid** (`outputs/phase9_per_source_confusion.png` — 12 panels, action_pred × action_true at τ_ear)
3. **Per-fold T variance and pooled drift ratio** (`|ΔAUC_pooled| / range(T_per_source)`) — re-verified from Phase 8c data, expected ratio ≈ 0.033
4. **K = 24 scaffold smoke test results** (synthetic data, three boundary cases)
5. **`compute_session_score` API documentation** — function signature, parameter types, expected behaviour at K = 1 vs K = 24
6. **Forward-look notes on multi-window aggregation** — what the pipeline needs to do differently when source videos exceed one V-JEPA-2 window (16 frames at sampling rate)

---

## Anti-patterns (LOCKED)

1. **No event counting, temporal NMS, learned TAL, or counting-aware
   losses.** The Dyson presence/absence rule collapses the bridge to
   thresholding. Decision 1 lock.
2. **No synthetic windowing of RME clips** to manufacture multi-window
   structure. Decision 2 lock.
3. **No re-fit of τ_ear or temperature scaling.** Phase 8c outputs are
   inputs to Phase 9. Decision 3 lock.
4. **No claim of session-level RHpE classification from K = 1 data.**
   Decision 4 lock; the audit doc headline must state this explicitly.
5. **No retroactive re-grading of Phase 8c's G1 verdict** under the
   new G1a/G1b rules. Decision 5 lock.
6. **No per-source τ_S introduction.** Per-source FPR variation is
   reported, not corrected. Decision 6 lock.
7. **No alternative aggregation rule shopping** ("we tried max, mean,
   median, and quantile aggregation; best wins"). Max-window-prob is
   the locked rule on Dyson 2018 + 2023 evidence.
8. **No re-running of Phase 8b classifier or Phase 8c calibration.**
   Phase 9 is strictly downstream.
9. **No goal-shifting on G1a/G1b's k = 0.04 constant** based on
   observed pooled drift. The constant is locked from Phase 8c's
   empirical anchor here; future phases may revise with documentation.

---

## Known limitations (surfaced pre-lock)

### Limitation 1 — Single-behavior scope (K = 1)

The simplified-B1 pipeline is exercised with K = 1 (ear movement
only). The ≥ 8/24 session threshold cannot fire at K = 1. Phase 9
demonstrates the mechanism (max-window-prob + threshold + per-behavior
binary + sum + threshold) without exercising the clinical claim. The
mechanism is parameterised to extend to K = 24 trivially; the clinical
claim requires multi-behavior probes (Phase 10+).

### Limitation 2 — Single-window-per-session (RME clip structure)

RME clips are 5–15 s each. There is no multi-window structure within
a clip to aggregate over with max-window-prob. Phase 9 treats each
clip as a single-window "session" — a degenerate case of the locked
aggregation rule. Real-world sessions are 10–15 minutes and require
sliding-window inference + max aggregation over the resulting
probability stream. This is not exercised in Phase 9. Future work
needs longer source videos than RME provides.

### Limitation 3 — Independence assumption (carries forward from Phase 8c)

The session-level operating point under independence remains the only
analytical tool for predicting session-level FPR/TPR. Phase 9 does not
verify independence (gated on N expansion). The analytical claim
P(session flagged | H0) ≈ 1.39e−5 (Phase 8c exact Poisson-binomial)
remains "UNVERIFIED — see Phase 8c Limitation 1."

### Limitation 4 — Single-observer label noise

RME paper labels are single-observer. Phase 9's per-source FPR/TPR
metrics reflect (noisy) label distribution, not ground-truth pain
state. Multi-rater κ on ≥ 20% audit subset remains the load-bearing
missing methodology step. Gated on SLU/Palichleb response.

### Limitation 5 — k = 0.04 pooled-drift constant from N = 1 phase

G1b's k = 0.04 is anchored on Phase 8c's empirical Δ_AUC / T_range
ratio (0.033, rounded up modestly). This is a single-phase empirical
anchor. Future phases will test whether this scales linearly with T
variance or whether non-linear correction is needed. Phase 9 reports
the ratio explicitly so the constant can be revised with documented
evidence in a later phase.

---

## Sequencing

| Step | Action | Output |
|---|---|---|
| 0 | User approval of this Stage 1 pre-reg | hash-locked artifact |
| 1 | Build `tools/phase9_simplified_b1.py` (consumes Phase 8c JSON outputs; implements Decisions 1-6; includes synthetic K=24 unit tests) | tool source |
| 2 | Run `phase9_simplified_b1.py` end-to-end (seconds, not minutes) | `outputs/phase9_simplified_b1_results.json`, `outputs/phase9_audit_extras.json`, `outputs/phase9_per_source_confusion.png` |
| 3 | Audit doc draft `docs/phase9_audit.md` | doc |
| 4 | User-approval checkpoint #2 (audit doc lock) | — |
| 5 | Hash chain + commit on `experiment/phase9` + subtree push to mirror + merge to main + subtree push main | mirror sync |

**User-approval checkpoints (2, matching Phase 8c's downstream-only cadence):**

1. After this Stage 1 doc approval, before any 9 compute or tool build.
2. After audit doc draft (Step 3), before commit + push.

---

## Cost / time estimate

| Step | Estimate |
|---|---:|
| Pre-reg approval cycle | ~15–30 min |
| Tool implementation (Step 1) — 6 decisions + scaffold + unit tests | ~90 min |
| Run + verify (Step 2) | ~5 min compute + ~20 min review |
| Audit doc draft (Step 3) | ~45 min |
| Commit + push (Step 5) | ~15 min |
| **Total wall-clock** | **~3 hours over ~½ day** |

Phase 9 is the smallest single-cycle scope in the post-Phase-7 chain,
matching the "downstream-of-downstream" position: Phase 9 = aggregation
rule on top of Phase 8c calibration, which sits on top of Phase 8b
classifier output.

---

## Phase 10+ entry conditions (forward-look, not Phase 9 lock)

Phase 9's landing **unblocks**:

- **Multi-behavior expansion** — once a second per-behavior probe lands
  (e.g., eye-region Phase 7-corrected output extended to full RME labels),
  the simplified-B1 pipeline lights up at K = 2; the ≥ 8/24 threshold
  still does not fire meaningfully until K → larger
- **Multi-window sliding inference** — needs longer source videos
  (not RME's 5–15 s clips); could use audit-followup or future
  field-collected data
- **Per-source τ_S investigation** — if Phase 9's per-source FPR
  diagnostic reveals systematic source-specific deviations, Phase 10+
  may revisit per-source operating points

**Phase 10+ remains gated on:**

- **Multi-rater κ track** (SLU/Palichleb response) — Phase 9 inherits
  Phase 8c's single-observer caveat; κ work is independent of the
  aggregation pipeline
- **Behavior co-occurrence verification** — needs N expansion beyond
  ear; tests Phase 8c session-OP independence assumption (Limitation 3)

---

## User approval signature

User has reviewed and approves Phase 9 Stage 1 lock as drafted, including:

- Decision 1: max-window-prob + threshold aggregation rule (per Dyson 2018 + 2023 presence/absence)
- Decision 2: single-window-per-session for K = 1 RME demonstration
- Decision 3: strict downstream consumption of Phase 8c JSON outputs (no recomputation)
- Decision 4: parameterised K = 24 scaffold with synthetic unit tests; no clinical claim at K = 1
- Decision 5: G1 redesign — per-source AUC invariance (G1a, bit-exact) + bounded pooled drift (G1b, k = 0.04 from Phase 8c anchor)
- Decision 6: per-source FPR reportable diagnostic; no per-source τ_S
- 5 locked gates (G1 execution / G2 Phase 8c consistency / G3 pooled FPR sanity / G4 G1a per-source AUC invariance / G5 synthetic unit tests)
- 9 anti-patterns
- 5 known limitations surfaced pre-lock
- 2 user-approval checkpoints
- Naming: project-internal "Phase 9", user-facing "simplified-B1 long-form aggregation + G1 sanity-check redesign"

User signs off → CC executes Step 1 (tool implementation) → CC runs
Step 2 (compute) → CC drafts audit doc Step 3 → user approves at
checkpoint #2 → final commit + push. No SLU/Palichleb response
required; Phase 9 is solo-shippable in parallel with email lane.
