# Track B — Phase 10a-prelim (Prudnik transfer test, preliminary subset)

## Stage 1 pre-registration

**Drafted 2026-05-13. Frozen on user approval BEFORE any 10a-prelim compute.**
This pre-reg locks the methodology, metrics, gates, and anti-patterns for
**Phase 10a-prelim** — a preliminary transfer-test of the Phase 8b DLC ear-
keypoint-anchored pipeline + Phase 8c calibration on the labeled subset of
Prudnik unridden championship clips. Phase 10a-prelim is a **fail-fast
gate**: its outcome routes whether to invest 3 more labeling days toward
the full Phase 10a or pause and diagnose first.

**Naming bridge:** project-internal "Phase 10a-prelim"; user-facing
"preliminary subset transfer test before full Phase 10a labeling
commitment."

**No new training, no new compute on the calibration spine.** Phase 10a-
prelim applies the Phase 8b-trained classifier + Phase 8c-calibrated
temperature to **new external data** (Prudnik 91-clip filtered subset →
~79 after sub-V-JEPA-2-min exclusion). It does NOT modify Phase 8b/8c/9
artifacts.

---

## ⚠ Framing — what Phase 10a-prelim IS and IS NOT

### What Phase 10a-prelim IS

- **A fail-fast cost-asymmetric gate** for the larger Phase 10a-full
  commitment. ~3 hours pre-reg + run + audit informs a 3-day labeling
  decision — 6× leverage ratio per user's analysis.
- **A single-source transfer test** of the Phase 8b DLC ear-keypoint
  pipeline + Phase 8c calibration on out-of-distribution Prudnik clips
  (portrait orientation, Polish championship, unridden context — see
  `data_quality_notes.md` §1).
- **A preliminary point estimate** of pooled AUC + bootstrap CI + ECE
  drift relative to Phase 8c's RME baseline (post-cal ECE 0.04). Per-clip
  residual table for failure-mode investigation if signal is weak.
- **Mirrors Phase 8a's stress-test posture** — small fast check that gates
  a larger downstream commitment. Same shape, different scope.

### What Phase 10a-prelim is NOT

- **NOT the full Phase 10a verdict.** n=~79 is below Phase 8b's adequate-
  power baseline (n=283) and the labeled subset is non-random (see L1).
  Preliminary AUC is likely an **upper bound** on full Phase 10a-full AUC.
- **NOT a session-level RHpE detector test.** K=1 (ear motion only); ≥8/24
  threshold doesn't apply. Lesson 21 framing carries forward.
- **NOT an operating-point validation.** With n_neg=16, expected FP at
  τ_ear=0.8138 is ~0.8 — integer-noise dominates. τ_ear / FPR / TPR
  verification deferred to Phase 10a-full when n_neg expands meaningfully.
- **NOT a multi-rater κ resolution.** Single-observer (user) labels;
  multi-rater track unchanged from Phase 8c/9 gating.
- **NOT a re-training of any classifier.** The Phase 8b-trained pipeline
  is applied as-is. No re-calibration on Prudnik data.
- **NOT a comparison to whole-frame V-JEPA-2 alternative.** The DLC
  ear-keypoint-cropped pipeline (Phase 8b's primary, COMPETITIVE verdict)
  is locked as the tested artifact. Whole-frame comparison would be a
  separate phase if needed.

---

## Locked design decisions

### Decision 1 — Subset definition (~79 clips)

**Decision:** preliminary subset = currently-labeled rows in
`poc/data/prudnik/labels_pending.csv` matching ALL of:

```
label != ""                                    # actually labeled
AND label NOT IN ("action?", "background?")    # exclude borderlines
AND confidence != "low"                        # exclude low-confidence
AND archived_full_clip_label == ""             # exclude split-roots
                                               #   (those were replaced
                                               #    by derived rows)
AND vjepa2_viable(duration_s, fps)             # NEW filter — see below
```

where `vjepa2_viable(d, fps) := d * fps >= 16` (V-JEPA-2 needs 16 evenly-
sampled frames).

**Expected n: ~79** (91 from user's filters minus ~12 sub-V-JEPA-2-min
clips). Exact count computed at runtime + reported in audit doc.

**Rationale for the V-JEPA-2-min exclusion (user-locked):** padding /
interpolating frames to reach the 16-frame minimum introduces methodology
that's not Phase 10a-prelim's job to validate. Cleaner pre-reg if those
~12 clips are excluded; Phase 10a-full can revisit with explicit
padding methodology if needed.

**Long-clip handling:** the 7 unflagged clips in 15–35s range are
**INCLUDED as single-window** with caveat in L5. Off RME training
distribution but not catastrophically so; excluding them loses signal
without much methodology gain.

**Anti-pattern lock (D1):** do NOT extend the subset post-hoc by relaxing
filters (e.g., including `?` borderlines or low-conf) if the AUC outcome
disappoints. Subset is locked here at this filter combination.

### Decision 2 — Trained pipeline (Phase 8b + 8c carry-forward)

**Decision:** apply the following pipeline to each Prudnik subset clip,
**identical to Phase 8b's primary pipeline** with one new step (calibration
from Phase 8c):

```
Per clip:
  1. DLC SuperAnimal-Quadruped inference → 4 ear keypoints
     (right_earbase, right_earend, left_earbase, left_earend)
     - Same params as Phase 8b: superanimal_quadruped, hrnet_w32,
       fasterrcnn_resnet50_fpn_v2, video_adapt=False,
       pseudo_threshold=0.1
  2. Apply locked Phase 8b geometry: both-ears bbox encompassing all
     4 keypoints (≥3-of-4 confidence at 0.5), 15% margin, square-pad
     to 224x224. Fallback rule: if <3 of 4 confident in a frame, use
     single-middle-frame's bbox.
  3. V-JEPA-2 ViT-L (pretrain-only checkpoint, fpc16-256-ssv2 via
     VJEPA2Model; SSv2 head dropped) → 1024-d embedding
  4. Apply trained Phase 8b RidgeClassifier (see D3 for which one)
     → decision_function score s
  5. Apply Phase 8c temperature scaling: p = sigmoid(s / T_median)
     where T_median = 0.494 (median of Phase 8c's 12 per-fold T values
     across [0.4567, 0.5244])
```

**Anti-pattern lock (D2):** do NOT introduce a whole-frame V-JEPA-2
alternative comparison within Phase 10a-prelim. The DLC-cropped pipeline
is Phase 8b's primary (COMPETITIVE verdict) and is the locked artifact
being tested for transfer. Whole-frame comparison is Phase 10a-full or
Phase 11+ scope, not preliminary's.

### Decision 3 — Classifier deployment proxy

**Decision:** train a **NEW RidgeClassifier(alpha=1.0, class_weight='balanced')
+ StandardScaler** on ALL 283 RME clips' DLC-cropped V-JEPA-2 features
(the same features used by Phase 8b's 12-fold LOSO). This is the
"deployed" classifier — one model trained on full RME, applied to
Prudnik.

**Rationale:** Phase 8b ran 12-fold LOSO; for inference on NEW external
data (Prudnik), the natural choice is a single model trained on all
available training data. This is standard transfer-test practice and
matches what a deployed classifier would do. The all-data classifier is
expected to have decision_function scores in a similar distribution to
the per-fold classifiers (calibration's T_median = 0.494 should remain a
reasonable approximation).

**Reproducibility:** save the trained classifier's parameters
(`Ridge.coef_`, `Ridge.intercept_`, `StandardScaler.mean_`,
`StandardScaler.scale_`) to `outputs/phase10a_prelim_deployed_classifier.json`
so the Prudnik predictions are exactly reproducible.

**Anti-pattern lock (D3):** do NOT use the 12 LOSO classifiers in an
ensemble or weighted-average for Prudnik. The single all-data classifier
is locked here; deviations would introduce methodology questions not
relevant to the preliminary signal.

### Decision 4 — Frozen calibration parameters

**Decision:** apply temperature scaling with **T = 0.494** (Phase 8c's
median per-fold T). No re-fitting on Prudnik (Phase 10a-prelim has no
held-out labeled Prudnik subset large enough to fit T defensibly).

**Source-shift caveat (to surface in audit doc):** T was fit on
RME-source-aware calibration LOSO; applying it to a new source
(Prudnik) is an extrapolation. If Phase 10a-full materializes with
enough labeled Prudnik data (≥100 clips with non-trivial class
balance), re-fit T on Prudnik OOF predictions.

**Classifier-shift caveat (additional, to surface in audit doc):**
T_median was fit on per-fold OOF predictions from Phase 8c's 12 LOSO
classifiers (each trained on 11-of-12 RME sources). The deployed
classifier in D3 is trained on ALL 12 RME sources, so its
`decision_function` distribution differs from any single per-fold
classifier's OOF distribution. Applying T_median to the all-data
classifier introduces a **classifier-distribution-shift
approximation** on top of the source-shift extrapolation. Both
classifiers are `Ridge(alpha=1.0, class_weight='balanced')` on the
same DLC-cropped V-JEPA-2 features, so the distributional shift is
bounded — but it is not zero. Test 4's ECE drift partially diagnoses
the combined effect; if Prudnik post-cal ECE substantially exceeds
Phase 8c's RME post-cal ECE (0.04), **classifier-shift is a candidate
explanation alongside source-shift**, and the audit doc must
acknowledge both rather than attributing the drift entirely to
domain transfer. Phase 10a-full could optionally compare the
all-data classifier's decision_function distribution on RME against
the ensemble of 12 LOSO classifiers' OOF predictions to quantify the
classifier-shift component empirically; deferred from preliminary.

**Anti-pattern lock (D4):** do NOT re-fit T on Prudnik's preliminary
labels. n=79 with 16 negatives is insufficient for honest temperature
fitting; doing so would conflate transfer-test result with calibration-
shopping.

### Decision 5 — Metrics

**Locked metric set:**

| Metric | Definition | Reporting |
|---|---|---|
| **Pooled AUC** | sklearn `roc_auc_score(labels, calibrated_probs)` | Point estimate + clip-bootstrap 95% CI |
| **Clip-bootstrap CI on AUC** | B=10000, with-replacement at clip level (single source — no source-bootstrap possible). Seed=42. | 2.5 / 97.5 percentile band |
| **Permutation p vs chance** | B=1000 label shuffles. Reports P(AUC_shuffled ≥ AUC_observed) under H0 | Permutation p-value |
| **ECE** | 10 equal-frequency bins (or fewer if n<50 per Phase 8c convention max(2, n//5)). Pre-cal sigmoid(s) and post-cal sigmoid(s/T=0.494). | Pre-cal + post-cal + delta |
| **Brier score** | (1/n) Σ (p_i − y_i)² | Post-cal value |
| **Per-clip residuals** | clip × label × pre-cal prob × post-cal prob × correct?(at decision boundary 0) | Full table in audit_extras.json |
| **Frame-extraction failure rate** | clips where DLC < 3-of-4 confidence in ≥X% of frames (locked X=75% from Phase 8b Stage 1.5) | Reportable |
| **Sub-RME-min derived flag** | (none in preliminary — sub-V-JEPA-2-min excluded; sub-RME-min not in subset) | n/a |

**Anti-pattern lock (D5):** do NOT compute or report FPR / TPR /
operating-point-related metrics in preliminary. n_neg=16 is too small
(expected FP at FPR=0.05 target = 0.8). The pre-reg explicitly
**defers operating-point evaluation to Phase 10a-full** when n_neg
expands meaningfully (likely ≥50 negatives).

### Decision 6 — Three-band verdict structure (user-locked)

**Decision:** verdict-reporting protocol is a **3-band assessment** on
pooled AUC, with locked routing per band:

| AUC band | Verdict label | Routing |
|---|---|---|
| **AUC ≥ 0.75** | `STRONG_TRANSFER` | **CONTINUE** full Phase 10a labeling (3 days). Expect Phase 10a-full to confirm with tighter CI. Phase 10a-prelim audit doc serves as the upper-bound estimate. |
| **AUC ∈ [0.60, 0.75)** | `AMBIGUOUS_TRANSFER` | **CONTINUE** labeling AND investigate per-clip residuals. Where is the noise? Domain shift (orientation, lighting), label bias, or n=79 selection bias (L1)? Audit doc identifies hypothesis to test in Phase 10a-full. |
| **AUC < 0.60** | `WEAK_OR_NO_TRANSFER` | **PAUSE** labeling. Diagnose first. Leading hypothesis: portrait orientation (`data_quality_notes.md` §1). Re-scope Phase 10 before any more human time goes in. |

**Anti-pattern lock (D6):** do NOT collapse the AMBIGUOUS band into
"continue regardless" or "binary success/fail." The band's existence
prevents post-hoc method-shopping when the number lands at 0.65 — it
forces explicit investigation rather than ad hoc reinterpretation.

---

## Test hierarchy

| Test | Role | Threshold | Notes |
|---|---|---|---|
| **Test 1: AUC verdict band** | **LOAD-BEARING** | Per D6: ≥0.75 / [0.60, 0.75) / <0.60 | Determines next-step routing |
| **Test 2: Clip-bootstrap CI on AUC** | Precision | B=10000 | Width informs confidence; expected wider than Phase 8b due to n=79 vs 283 |
| **Test 3: Permutation p vs chance** | Sanity | B=1000 | Should be <0.05 if any real signal exists |
| **Test 4: ECE drift vs Phase 8c reference** | Calibration | post-cal ECE; compare to Phase 8c's 0.04 | Reportable; large drift may indicate domain shift |
| **Test 5: DLC ear-keypoint reliability on portrait** | Pre-flight diagnostic | Per-clip ≥3-of-4 confidence rate | Compare to Phase 8b Stage 1.5's 90% on RME. Lower rate is expected on portrait orientation; quantify. |
| **Test 6: Per-clip residual table** | Diagnostic | All 79 clips reported | Used for AMBIGUOUS-band investigation |
| **Test 7: Per-clip frame_type subgroup AUC** | Reportable | head-zoom vs full-body subgroups; **AUC reported only if n_subgroup ≥ 20 after filtering**, otherwise descriptive statistics only (per-clip predictions table without aggregate AUC) | Reveals whether the methodology transfers differently across framing types; n=20 floor prevents over-interpreting subgroup noise (head-zoom subset likely drops to ~9–14 after sub-V-JEPA-2-min exclusion, where wide CI makes any frame_type-difference claim unfalsifiable) |

---

## Locked gates

| Gate | Threshold | Action |
|---|---|---|
| **G1 — Tool execution** | exit code 0 | HALT on FAIL |
| **G2 — Pipeline output completeness** | All ~79 clips have V-JEPA-2 features + RidgeClassifier predictions; 0 failures | HALT on FAIL |
| **G3 — Pooled FPR sanity** | **DEFERRED to Phase 10a-full** (per D5 anti-pattern; n_neg=16 too small) | Not evaluated |
| **G4 — AUC verdict band** | Per D6 three-band routing | Reportable; routes Phase 10a continuation decision |
| **G5 — DLC reliability sanity** | per-clip ≥3-of-4 confidence rate ≥ 50% on Prudnik | If <50%: surface as concurrent finding alongside G4 verdict; doesn't override G4 |
| **G6 — Calibration AUC invariance (G1a from Phase 9)** | Per-clip pre-cal AUC = post-cal AUC bit-exact (single-source applied with single T_median, so monotonic) | HALT on FAIL (would indicate tool bug) |

**G3 deferral:** the pre-reg explicitly skips operating-point evaluation
in preliminary. Phase 10a-full pre-reg will revisit when n_neg supports
it.

---

## Diagnostic instrumentation (reportable, not gated)

1. **Per-clip predictions table** — clip_id, frame_type, label,
   multi_horse, raw decision_function score, sigmoid(s) pre-cal prob,
   sigmoid(s/0.494) post-cal prob, correct-at-threshold-0 flag
2. **Per-clip DLC ear keypoint confidence summary** — mean confidence per
   keypoint, fraction of frames passing ≥3-of-4 gate, fallback fired y/n
3. **Per-clip ear-bbox quality** — frame coverage of bbox, bbox stability
   over clip
4. **Pre-cal vs post-cal reliability diagram** — 2-panel, equal-frequency
   bins (max(2, n//5) bins given small n)
5. **Frame_type subgroup AUC** — head-zoom subset AUC + bootstrap CI;
   full-body subset AUC + bootstrap CI
6. **Multi-horse subgroup analysis** — AUC on `multi_horse=0` vs
   `multi_horse=1` subsets
7. **Sub-5s clips in subset** — included clips that are sub-RME-min but
   above V-JEPA-2 floor; flag and report AUC including / excluding them
8. **Label class balance per metric** — report n_action, n_background per
   subset metric

---

## Anti-patterns (LOCKED)

1. **No subset relaxation** post-hoc if AUC disappoints (D1)
2. **No whole-frame V-JEPA-2 alternative** comparison in preliminary (D2)
3. **No classifier re-training** on Prudnik labels (D3)
4. **No T re-fitting** on Prudnik labels (D4)
5. **No FPR / TPR / operating-point claims** at this n_neg (D5)
6. **No collapsing the AMBIGUOUS verdict band** into binary
   success/failure (D6)
7. **No declaring Phase 10a complete** from the preliminary subset alone.
   STRONG_TRANSFER still routes to "continue full labeling"; the
   preliminary is NOT a substitute for the full test.
8. **No re-interpretation of L1 selection bias** post-hoc as "the easy
   clips are representative." The hard clips are deferred by construction
   of which clips the user labeled first.
9. **No multi-rater κ claim** from this work; gated on outreach
10. **No Phase 8c calibration re-evaluation.** Phase 8c's calibration LOSO
    on RME stands; Phase 10a-prelim only applies its T_median to Prudnik.

---

## Known limitations (surfaced pre-lock — load-bearing)

### Limitation 1 — Selection bias on the n=79 subset (LOAD-BEARING)

The 91→79 subset is **NOT a random sample** of Prudnik. It is the
user-labeled-first, high/medium-confidence, clearly-classifiable subset.
**The remaining 180 derived rows are likely the harder cases** —
deferred multi-horse complexity, ambiguous behaviors, edge cases the
user wanted more time to consider. **Preliminary AUC on the 79 is
likely an UPPER BOUND** on what the full Phase 10a-full will produce,
not a representative point estimate.

**Required surfacing:** the audit doc's verdict section MUST state this
inline, not just in a footnote. The preliminary number must always be
reported with the caveat "upper bound estimate; full-set value will
likely be lower."

This is the highest-priority caveat in Phase 10a-prelim and the easiest
finding to misread. Phase 7-style discipline applies: do not soften L1
during interpretation under any band's outcome.

### Limitation 2 — Single source

Prudnik is **one source** (one championship). No LOSO is possible.
Bootstrap is clip-level (with replacement, B=10000, seed=42). The CI
characterization differs from Phase 8b/9's source-bootstrap and is not
directly comparable to those phases' precision claims.

### Limitation 3 — n_neg = 16 disables operating-point evaluation

With 16 negative-source-clip labels in the preliminary subset, expected
FP count at FPR=0.05 is 0.8 — integer-noise dominates. τ_ear=0.8138
(Phase 8c) cannot be meaningfully verified against the 16 negatives.
**Operating-point evaluation deferred to Phase 10a-full** when n_neg
expands (target ≥50 negatives for sane FPR estimation).

### Limitation 4 — Sub-V-JEPA-2-min clips excluded

~12 clips with `duration × fps < 16 frames` are excluded from the
preliminary by D1. Phase 10a-full revisits with padding / frame-
interpolation methodology if those clips matter to the full claim.

### Limitation 5 — Long clips (>15s, ≤35s) included as single-window

The 7 unflagged 15–35s clips are kept as single-window labels (no
derived rows exist for them; user judged them as single-verdict during
the first pass). They're off RME training distribution but not
catastrophically so. Subgroup AUC by clip-duration bucket reported in
audit for transparency.

### Limitation 6 — Portrait orientation distribution shift

All Prudnik clips are portrait orientation (`data_quality_notes.md` §1).
This is the **leading hypothesis** for any WEAK_OR_NO_TRANSFER outcome.
If AUC < 0.60 (G4 = WEAK), the audit doc routes the failure analysis to
portrait-orientation diagnosis first (e.g., DLC keypoint quality
breakdown on portrait clips per Test 5, frame_type subgroup AUC per
Test 7).

### Limitation 7 — Single-observer (user) labels

Multi-rater κ track unchanged from Phase 8c/9. Single-observer label
noise carries forward as a known limitation; not a fixable defect of
Phase 10a-prelim.

### Limitation 8 — Lesson 21 framing carries forward

This phase tests RME ear-MOTION transfer, NOT RHpE Behavior #7 detection.
Lesson 21's framing decision (interpretation A adopted) governs claims.
The pipeline-demonstration framing is what's being tested for transfer;
the RHpE-Behavior-#7 narrative is out of scope.

---

## Sequencing

| Step | Action | Output |
|---|---|---|
| 0 | User approval of this Stage 1 pre-reg | hash-locked artifact |
| 1 | Build `tools/phase10a_prelim_run.py`: subset selection, DLC inference loop, ear-crop pipeline, V-JEPA-2 feature extraction, trained classifier inference + calibration, metrics computation, bootstrap CI, ECE / reliability diagram, per-clip residual table | tool source |
| 1.5 | Train + save "deployed" RidgeClassifier on full 283-clip RME features (D3) | `outputs/phase10a_prelim_deployed_classifier.json` |
| 2 | Run end-to-end on Prudnik subset (~10–20 min wall: DLC inference is the slow step, ~3 s/clip × 79 clips ≈ 4 min; V-JEPA-2 extraction ~1.3 s/clip × 79 clips ≈ 2 min) | `outputs/phase10a_prelim_results.json`, `outputs/phase10a_prelim_audit_extras.json`, `outputs/phase10a_prelim_reliability_diagram.png` |
| 3 | Audit doc draft `docs/phase10a_prelim_audit.md` per Phase 8c/9 audit template + 3-band verdict + L1 inline + DLC reliability finding + frame_type subgroup analysis | doc |
| 4 | User-approval checkpoint #2 (audit doc lock + Phase 10a-full routing decision) | — |
| 5 | Hash chain + commit on `experiment/phase10a-prelim` + subtree-push to mirror + merge `--no-ff` to main + subtree-push main | mirror sync |

**User-approval checkpoints (2, matching Phase 8c/9 cadence):**

1. After this Stage 1 doc approval, before any 10a-prelim compute.
2. After audit doc draft (Step 3), before commit + push, **AND** before
   any commitment to continue (or pause) Phase 10a-full labeling. This
   is the load-bearing decision point.

---

## Cost / time estimate

| Step | Estimate |
|---|---:|
| Pre-reg approval cycle | ~30 min |
| Tool implementation (Step 1) — subset query + DLC + crop + V-JEPA-2 + classifier + metrics + bootstrap | ~90 min |
| Train + save deployed classifier (Step 1.5) | ~10 min |
| Run end-to-end (Step 2) — DLC + V-JEPA-2 are the slow steps | ~20 min compute + ~30 min review |
| Audit doc draft (Step 3) | ~60 min |
| Commit chain (Step 5) | ~15 min |
| **Total wall-clock** | **~4 hours over ~½ day** |

User's estimate: "~3 hours total wall-clock." Within ±30% — matches the
Phase 9 cadence for downstream methodology phases.

---

## Phase 10a-full entry conditions (gated on Phase 10a-prelim verdict)

| 10a-prelim verdict | Phase 10a-full routing |
|---|---|
| `STRONG_TRANSFER` (AUC ≥ 0.75) | Continue labeling 180 derived rows over ~3 days. Phase 10a-full applies same methodology to full set (~308 labels after derived rows complete). Expect tighter CI confirming preliminary. Likely route to Phase 11+ (K=2 multi-behavior expansion, per Lesson 21's K=2 option discussion) |
| `AMBIGUOUS_TRANSFER` (AUC ∈ [0.60, 0.75)) | Continue labeling AND investigate per-clip residuals. Hypotheses to test in Phase 10a-full: (a) selection bias L1 — does the full set show similar AUC?, (b) domain shift in specific frame_type subgroup — head-zoom vs full-body bifurcation?, (c) DLC keypoint reliability degradation on portrait orientation — per-clip diagnostic? |
| `WEAK_OR_NO_TRANSFER` (AUC < 0.60) | **PAUSE labeling.** Diagnose first per Limitation 6 (portrait-orientation hypothesis). Possible re-scopes: (a) whole-frame V-JEPA-2 baseline on Prudnik (Phase 8b's other pipeline); (b) re-train classifier on landscape-augmented RME; (c) accept that Phase 8b/8c pipeline does not transfer to portrait orientation as a finding and write up. Do not invest 3 more labeling days until diagnosis informs scope |

---

## User approval signature

User has reviewed and approves Phase 10a-prelim Stage 1 lock as drafted,
including:

- D1: subset = labeled rows ∩ {`label not in ?`, `confidence != low`,
  `archived_full_clip_label == ""`, `duration × fps ≥ 16`}; expected n≈79
- D2: Phase 8b DLC ear-keypoint pipeline + Phase 8c calibration carry-
  forward (no whole-frame alternative)
- D3: train new RidgeClassifier on full 283 RME features for deployment
  on Prudnik
- D4: T = 0.494 (Phase 8c median per-fold T); no Prudnik re-fitting
- D5: Pooled AUC + bootstrap CI + permutation p + ECE + per-clip
  residuals + frame_type subgroups; **NO operating-point analysis**
- D6: 3-band verdict (STRONG ≥0.75 / AMBIGUOUS [0.60, 0.75) / WEAK <0.60)
  with locked routing per band; AMBIGUOUS as a named verdict prevents
  collapsing
- 6 locked gates (G3 deferred per D5)
- 10 anti-patterns (no method shopping, no re-training, no operating-
  point claims, no AMBIGUOUS collapse, etc.)
- 8 known limitations surfaced pre-lock — **L1 selection bias is
  load-bearing and required-inline-in-audit-doc**
- 2 user-approval checkpoints
- Naming: project-internal "Phase 10a-prelim"; user-facing "preliminary
  subset transfer test before full Phase 10a labeling commitment"

User signs off → CC executes Step 1 (tool implementation) + Step 1.5
(train deployed classifier) → CC runs Step 2 → CC drafts audit doc Step
3 → user approves at checkpoint #2 with **explicit decision on Phase
10a-full continuation** → commit chain.
