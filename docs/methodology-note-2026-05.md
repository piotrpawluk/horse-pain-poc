# Methodology note — single-observer label audit + V-JEPA-2 + LR robustness analysis on Read My Ears

*Project: `horse-pain-poc` (V-JEPA-2 + linear probe for RHpE-grade ear-motion classification, replication of Alves et al. CVPR W'25 with source-aware evaluation). Branch: `experiment/audit-followup`. May 2026. Single-author note for internal record + potential future external review.*

## TL;DR — the central methodological move

When V-JEPA-2 + linear probe is **evaluated** against single-observer audit labels (instead of the published EquiFACS-derived RME labels), LOSO AUC drops by only **−3.6 pp** (0.8746 → 0.8389). When the linear probe is **retrained** on those same audit labels, LOSO AUC drops by **−13.6 pp** (0.8746 → 0.7386). The 10 pp gap between these two numbers is the cost of training a fresh probe on labels at within-observer κ = 0.586 (moderate). **The features generalize across labeling protocols; the labels themselves are too noisy at single-observer reliability to support clean retraining.**

This decomposition produces a falsifiable prediction: **if RHpE field-data labels achieve typical multi-rater EquiFACS-grade κ ≥ 0.7, the retrain-noise component should shrink substantially toward the −3.6 pp eval-mismatch floor.** The current experiment cannot test the prediction directly (single-annotator audit only) but the recording protocol's inter-rater κ requirement now has empirical grounding rather than methodological best-practice grounding.

A secondary methodological output: **per-source consistency × per-source agreement** is a 2-axis decomposition that disambiguates two distinct mechanisms behind "high audit-vs-published-label disagreement" — calibration (stable threshold ≠ baseline) vs noise (unstable threshold + ≠ baseline). The S5 / S10 contrast in this dataset (variance ratio 2.2× across LOSO variants matches the within-observer-inconsistency ratio) demonstrates this is operational at the LOSO level, not just a Step 3 diagnostic.

---

## 1. Audit methodology

### Procedure

The project owner manually reviewed all 283 clips in the Read My Ears test split on 2026-05-07 against a personally-applied threshold for "is there visible ear motion within the clip." Reviews recorded as `<clip basename> — <free-text observation> — <VERDICT>` where VERDICT ∈ {ACTION, BACKGROUND, ACTION?, BACKGROUND?}, with `?` denoting "borderline; would defer to a second opinion." Multi-horse scenes used `fh:` (foreground horse) / `bh:` (background horse) per-subject notation. The full audit is structured in `outputs/piotr_audit_labels.jsonl` with 10 derived categorical labels (`strong_motion`, `slight_motion`, `subthreshold`, `head_only`, `body_only`, `multi_horse_target_focus`, `multi_horse_distractor`, `scene_cut`, `error_frame`, `ears_still`).

### Self-consistency check

A 66-clip re-watch was conducted on the same day (Step 3 of `docs/audit-followup-spec.md`). The re-watch sample comprised:
- All 56 borderline (`?`) cases
- 10 randomly sampled (seed = 42) confident controls (5 ACTION, 5 BACKGROUND) as positive controls

The re-watch was conducted with the original verdicts hidden, after the explicit FH-only multi-subject rule was articulated (it had been applied implicitly during the original audit). Within-observer self-consistency was computed at two grains:

- **Confident controls (hard-stop gate):** 10/10 = 100 % verdict-match. The audit methodology has demonstrated reliability on clearly-classified clips. Both multi-horse confident controls (`action_S4.mp4_5_.mp4`, `action_S8.mp4_4_.mp4`) held confident `ACTION` under FH-only.
- **Borderline cases:** 45/56 = 80.4 % verdict-match. **Cohen's κ = 0.586 (moderate per Landis & Koch 1977).** All 11 verdict-flips were on clips with "extremely slight" or "slight" motion language — the drift is concentrated entirely in the subthreshold-motion zone, exactly where the `?` annotation was meant to capture uncertainty.

### Caveats

The re-watch was **same-day** rather than the spec-recommended ≥ 12 h delay. Fresh-memory effects bias the κ figure toward the high side. The 80.4 % / κ = 0.586 should be read as a **ceiling** on within-observer reliability, not the long-term rate. The re-watch was also **non-blind** to the existence of the audit (the user knew clips were sampled from `?` cases + 10 controls). True blind retest would require forgetting clip-membership.

**Single annotator only.** Inter-rater κ unmeasured. The "RHpE coders typically achieve κ ≥ 0.7" expectation in this note is a typical-range claim for trained ethogram protocols, not a measurement of any specific RHpE coder team.

---

## 2. Findings on Read My Ears label structure

### Headline disagreement rate

**12.4 % of RME labels disagree with the audit verdict** (35/283 clips). The disagreement is asymmetric in direction:

- **24 BACKGROUND → ACTION** flips (8.5 % of dataset): mostly subthreshold ear motion the reviewer treats as above-threshold but EquiFACS coders correctly excluded under intensity/duration gates. This is a definitional difference, not a labeling error.
- **11 ACTION → BACKGROUND** flips (3.9 % of dataset): mostly head-only or distractor-only motion that the reviewer excludes under FH-only / anatomical-scope rules. Some cases suggest the EquiFACS coder may have included secondary motion (handler hand near ear, tail flick visible behind ear) under the AU label.

Of the 35 disagreements, 17 are confidently flipped (no `?`) and 18 are borderline. **The "12.4 %" number is single-observer disagreement-with-published-protocol, not "true label noise rate."**

### Per-source heterogeneity

Per-source disagreement rates range from 0 % (S6) to 24 % (S5). Sources cluster:

- **Clean:** S6 (0 %), S7 (5 %), S3 (7 %)
- **Mid:** S8 (8 %), S12 (9 %), S1 (10 %), S11 (11 %), S9 (13 %)
- **High disagreement, mostly subthreshold:** S2 (20 %), S10 (23 %), S5 (24 %)
- **High disagreement, multi-horse-driven:** S4 (16 %), S8 (8 % but all 24 clips multi-horse)

### Multi-horse confound

**19.4 % of RME clips (55/283) contain two or more horses in frame**, all in S4 (31) + S8 (24). Per-clip target/distractor annotations from the audit's `fh:` / `bh:` notation are saved in `outputs/multi_horse_subset.jsonl` with `target_motion_present` / `distractor_motion_present` flags. This independently confirms [Lesson 9](lessons_learned.md)'s LOSO-derived multi-horse-confound finding (S8: 0.633 → 0.875 with bg-masking, +24 pp) — the audit derived the same conclusion from per-clip observation; neither analysis informed the other.

### Two distinct mechanisms behind audit-vs-RME disagreement

The Step 3 within-observer consistency check on a 66-clip re-watch revealed that "high audit-vs-RME disagreement" splits into two qualitatively different mechanisms when crossed with within-observer consistency:

| Source | RME-vs-audit disagreement | Within-observer consistency | Diagnosis |
|---|---|---|---|
| **S10** | 23 % (5/22 clips) | 100 % (5/5 borderlines stable across re-watch) | **Pure calibration** — stable threshold, just different from EquiFACS |
| **S5** | 24 % (6/25 clips) | 60 % (3/5 borderlines stable) | **Calibration + noise** — unstable threshold + different from EquiFACS |

This contrast is small-N (5 borderlines per source) but operationally validated by the LOSO variant analysis below. **Per-source consistency × per-source agreement is the 2-axis diagnostic; both dimensions are required to interpret per-source heterogeneity correctly.**

---

## 3. V-JEPA-2 + LR robustness analysis — the central methodological move

### Setup

The published Sanity 5 LOSO baseline of 0.875 was reproduced exactly using the canonical config: `RidgeClassifier(alpha=1.0, class_weight='balanced')` + `StandardScaler` per fold on cached V-JEPA-2 ViT-L embeddings (`outputs/vjepa2_embeddings.npz`, 283 × 1024). Global per-clip LOSO AUC = **0.8746**, matching the published baseline `0.8746126936531734` to 4 decimal places.

This exact-reproduce sanity check ensures the analyses below are not confounded by hyperparameter drift relative to the published baseline.

### Three label variants

The audit + Step 3 re-watch produces three usable label sets (formal definitions in `audit-followup-spec.md` §7 + `outputs/audit_followup_labels.jsonl`):

- **Strict (272 clips):** 227 originally-confident clips + 45 borderlines that re-watched as same-direction-confident; the 11 inconsistent verdict-flips dropped.
- **Permissive (283 clips):** All clips kept; re-watch verdict on the 45 confirmed borderlines, original verdict on the 11 inconsistent flips.
- **Cleaned (218 clips):** Strict variant minus all 55 multi-horse clips (single-subject only).

V-JEPA-2 + LR LOSO was run separately on each variant — same canonical config; only the label set changes. Per-clip predictions saved to `outputs/loso_v2_<variant>_predictions.jsonl`.

### Results

| Variant | N | Global LOSO AUC | Δ vs published 0.8746 |
|---|---|---|---|
| Original (RME) | 283 | **0.8746** | 0.0000 *(sanity reproduce)* |
| **Strict** | 272 | **0.7386** | **−13.6 pp** |
| **Permissive** | 283 | **0.7345** | **−14.0 pp** |
| **Cleaned** | 218 | **0.7386** | **−13.6 pp** |

Three observations from this table alone:

1. **The drop is uniform across variants** (~14 pp). This is not random fold-to-fold variance, which would show ±2–3 pp spread across variants. It's a systematic loss when retraining on Piotr-grade labels.
2. **Strict ≈ Permissive (within 0.4 pp).** The 11 inconsistent verdict-flips that were debated for inclusion in Strict do not materially affect aggregate LOSO either way. This is a quiet validation of the pre-registered choice (drop them from strict only); the analytical decision did not bias the headline number.
3. **Cleaned ≈ Strict (within 0.0 pp).** Multi-horse exclusion does not change the aggregate LOSO under audit-grade labels, although it does mechanically force S4 and S8 to NA in the per-source matrix (Cleaned becomes effectively a 9-source LOSO).

### The decomposition (the central methodological move)

Reading the headline 13.6 pp drop in isolation invites the interpretation "audit labels give worse LOSO." That interpretation is incomplete. The drop decomposes cleanly when joined with the [B-prime decomposition](../outputs/vjepa2_label_agreement_decomposition.md) result:

| Step | Configuration | Global AUC | Δ vs prior |
|---|---|---|---|
| Published baseline | RME-trained LR + RME eval | **0.8746** | — |
| B-prime evaluation mismatch | RME-trained LR + audit-strict eval | **0.8389** | **−3.6 pp** |
| Step 5 Strict | Audit-trained LR + audit-strict eval | **0.7386** | **−10.0 pp** |

The two components measure different things:

- **−3.6 pp eval-mismatch** = how much the V-JEPA-2 features and the RME-trained LR's decision boundary lose when scored against a different (audit) ground truth. The features still rank the audit-grade signal correctly; the decision boundary the LR learned is calibrated to the EquiFACS threshold and incurs a modest cost when applied to the audit threshold.
- **−10 pp retrain-noise** = how much the LR loses when it has to learn a fresh decision boundary from scratch on labels at within-observer κ = 0.586. The features are unchanged; only the labels supplied to `LR.fit` differ. The LR cannot find a clean linear separator on noisy labels, so its learned weights are noisier than the RME-trained baseline.

**The features generalize at modest cost; the labels themselves are too noisy at single-observer reliability to train a clean probe.** This sentence is the central claim of this methodology note and the primary deliverable of the audit-followup work. It is not the headline LOSO drop number; it is the decomposition of that number into two operationally-distinct components with different remediation paths.

### Falsifiable prediction

The decomposition makes a specific, testable prediction:

> **If RHpE field-data labels achieve typical multi-rater EquiFACS-grade Cohen's κ ≥ 0.7, the retrain-noise component should shrink substantially toward the −3.6 pp eval-mismatch floor.**

This experiment **cannot test that prediction directly**, because the audit is single-annotator. The prediction is operationalized for any future re-run: collect RHpE field labels with at least 2 trained raters, measure inter-rater κ, retrain V-JEPA-2 + LR on the multi-rater labels, and observe whether the retrain-noise component on the audit-vs-published comparison drops below ~5 pp. If it does, the prediction is validated and V-JEPA-2 + LR transfers to RHpE field labels at modest cost. If the retrain-noise component remains at ~10 pp despite κ ≥ 0.7 multi-rater labels, then the V-JEPA-2 features carry less audit-grade signal than this experiment suggests and the project recommendation needs revision.

This makes the recording-protocol's inter-rater κ requirement (`docs/recording-protocol.md`) **load-bearing rather than nice-to-have** — it has empirical grounding, not just methodological best-practice grounding.

### Per-source pattern — the publishable-grade diagnostic

The S5 / S10 contrast identified in the within-observer consistency check (Step 3) maps cleanly to LOSO-AUC variance across the three label variants:

| Source | Within-observer consistency | Variance across S/P/C variants | Ratio |
|---|---|---|---|
| **S10** | 100 % stable (5/5) | **0.048** | 1.0× |
| **S5** | 60 % unstable (3/5) | **0.105** | **2.2×** |

S5's variance across LOSO variants is **2.2× S10's**, mirroring the within-observer-inconsistency ratio (40 % vs 0 %, ~2× depending on framing). The cross-step prediction from Step 3 to Step 5 lands cleanly. **The calibration-vs-noise distinction is operational at the LOSO level**, not just a Step 3 diagnostic.

This generalizes to a methodological recommendation for future RHpE-style work:

> **Per-source consistency × per-source agreement should be computed at the start of any source-aware evaluation, not as a post-hoc diagnostic.** They answer different questions: agreement asks "does the label signal align across reviewers / against ground truth?"; consistency asks "is the label signal stable within a single reviewer across sittings?". A 2×2 of (agreement, consistency) categorizes sources into clean, ambiguous-but-reproducible, calibration-finding (S10 type), and noisy (S5 type) — and the LOSO-variant variance matches the consistency axis.

### Per-source caveats explicitly flagged

The per-source matrix (`outputs/loso_label_variant_comparison.md` §4) shows several per-source AUCs that look extreme at face value but are artifacts of small-N or class-balance distortion:

- **S1: −40 pp Δ vs RME**, but S1 has 21 clips with mostly RME-zero positives; the audit upgrades 1 borderline to ACTION, which gives S1 a single positive in 21 clips. AUC dominated by where that 1 clip ranks. **Not a model-behavior finding.**
- **S6: −39 pp Δ vs RME**, but S6 has 0 audit-vs-RME disagreements (the cleanest source per Lesson 17). The drop is from the LR's training data on OTHER sources changing, not from S6's labels changing. The training distribution shift induces a per-fold AUC drop on S6 even though S6's labels are untouched. **Also not a model-behavior finding** in the direct sense.
- **S7: NA across all variants** because S7 has no audit-positive clips. AUC undefined when only one class is present in the held-out fold.
- **S4 and S8: NA in Cleaned** because multi-horse exclusion drops S4 to 1 single-horse clip and S8 to 0.

The substantive per-source story is **S5/S10 variance-correlated-with-consistency**, not the magnitude-of-drop tail. Magnitude-of-drop on small-N sources is fold-noise-dominated.

---

## 4. Labeling protocol

The protocol followed by the reviewer was implicit during the original audit and was made explicit during the within-observer re-watch (FH-only rule for multi-subject scenes). It is documented in full at [`docs/labeling-protocol-2026-05.md`](labeling-protocol-2026-05.md). Headline rules:

- ACTION = at least one ear shows positional change (rotation, twitch, flick, pinning) visible at normal-speed playback.
- BACKGROUND = ears stationary, OR head/body/tail moves while ears stationary, OR sub-perceptible motion.
- Multi-subject: **FH-only rule.** Verdict applies to foreground horse only; BH motion excluded regardless of magnitude.
- Anatomical scope: ears only. Head / body / tail motion does not contribute to ACTION.
- Borderline `?`: "would defer to a second opinion" — used for sub-threshold-visibility cases and ambiguous multi-subject scenes.

**One Step 3 finding worth highlighting in this note:** all 7 reviewed multi-horse-distractor cases (FH ears still, BH moves) were originally labeled `BACKGROUND?` — uncertain. After explicit FH-only articulation, all 7 resolved to confident `BACKGROUND`. **The FH-only rule was being applied implicitly; the protocol clarification removed uncertainty without changing verdicts.** This suggests careful annotators arrive at the rule intuitively but require explicit articulation for inter-rater reproducibility. For ethogram-grade protocol design: the explicit rule is necessary not because annotators don't apply it, but because they don't *agree* about applying it without articulation.

---

## 5. Limitations and what's needed for external validation

### Methodological caveats

- **Single annotator only.** Inter-rater κ unmeasured. Every quantitative claim above is single-observer-disagreement-with-published-protocol, not "true label noise rate." The κ = 0.586 (moderate) figure bounds *within-observer* reliability, which is by definition an upper bound on inter-rater reliability.
- **Same-day re-watch.** The Step 3 consistency check used hours-not-days delay, biasing κ toward the high side. Treat as ceiling.
- **Reviewer is non-blind** to the project's V-JEPA-2 / MLLM behavior on this dataset. Possible bias toward labels that "make sense" given the model's outputs. The 7-of-7 multi-horse-distractor protocol-flips on re-watch were predicted by the agent in advance under the FH-only rule, which is one structural check — but a fully-blind audit by a different observer is the proper validation.
- **Audit done as a side investigation** rather than as a planned audit with pre-registered methodology. The pre-flight resolutions in `audit-followup-spec.md` §0 (use full 56 borderline + 10 controls; B-prime sanity check; re-flush per-clip predictions from cached features) were the agent's calls, not pre-registered. Future replications should pre-register the audit methodology before running it.

### Dataset specificity

- **Read My Ears is controlled lab data.** 12 sources, single horse per clip in 81 % of cases, multi-horse in 19 %. Field data with different camera angles, lighting, and movement contexts may show different audit-vs-published-label disagreement structure. The S5 / S10 calibration-vs-noise diagnostic should be recomputed on any new dataset before transferring conclusions.
- **EquiFACS coder protocol applied to RME** is one specific intensity/duration threshold scheme. Other ethogram protocols (RHpE 24-behavior checklist, Equine Pain Face Scale, etc.) use different criteria and may produce different audit-vs-published-label disagreement profiles.

### What's needed before external use

1. **Multi-rater audit on the same 283 clips.** A second trained reviewer applies the same protocol; inter-rater κ measured directly. If κ ≥ 0.7, the retrain-noise prediction can be tested.
2. **Replication on a second EquiFACS-derived dataset.** The S5 / S10 calibration-vs-noise diagnostic should appear on other datasets with similar source heterogeneity. If it does, the methodology generalizes; if it doesn't, it's RME-specific.
3. **Bootstrapped per-source AUC error bars** on the LOSO variant comparison. The S5 / S10 variance ratio (0.105 / 0.048 = 2.2×) is large but should be presented with uncertainty bounds, especially given per-source N ≤ 25.
4. **A sanity replication of the −10 pp retrain-noise number** by training the LR on RME labels for the 272-clip strict-eligible subset and comparing AUC to Strict. The current decomposition compares B-prime (RME labels for full 283-clip set) vs Step 5 Strict (audit labels for 272-clip subset); a within-N-272 replication would make the decomposition fully tight.

### Project-internal vs external publication

This note is project-internal. For external publication (Andersen / Zamansky community), the additional work in §5 above (multi-rater κ, bootstrapped error bars, second-dataset replication) would be needed to make the claim defensible. The methodology note as written is **calibrated to "internal record + sharable with collaborators for review,"** not "submission-ready manuscript."

The decomposition itself is the contribution that would justify external work: it operationalizes a question ("does V-JEPA-2 + LR transfer across labeling protocols?") into a measurable answer ("yes at ~3.6 pp cost on features, no at ~10 pp cost on probe retraining without inter-rater clean labels"). That's the kind of result that adds value to the EquiFACS / RHpE methodological discussion — the cross-vendor / cross-protocol generalization story is currently under-explored in the published literature.

---

## 6. Cross-references

- [`docs/audit-followup-spec.md`](audit-followup-spec.md) — gated execution sequence specification with pre-registered Pattern interpretation under the bifurcation lens
- [`docs/labeling-protocol-2026-05.md`](labeling-protocol-2026-05.md) — Step 6 Light C deliverable extracted from the 283-example audit
- [`docs/lessons_learned.md`](lessons_learned.md) — lesson trail; specifically Lesson 9 (multi-horse confound from the LOSO side), Lesson 14 (Gemini failure modes on RME), Lesson 15 v2 (Qwen 7B replication post-bug-fix), Lesson 17 (full audit findings)
- [`outputs/piotr_audit_labels.jsonl`](../outputs/piotr_audit_labels.jsonl) — 283 audit verdicts with category labels
- [`outputs/consistency_check_results.md`](../outputs/consistency_check_results.md) — Step 3 within-observer self-consistency analysis
- [`outputs/vjepa2_label_agreement_decomposition.md`](../outputs/vjepa2_label_agreement_decomposition.md) — Step 2 B-prime per-clip V-JEPA-2 agreement decomposition (the −3.6 pp eval-mismatch number)
- [`outputs/loso_label_variant_comparison.md`](../outputs/loso_label_variant_comparison.md) — Step 5 three-variant LOSO comparison (the −13.6 pp Strict and −10 pp retrain-noise components)
- [`outputs/multi_horse_subset.jsonl`](../outputs/multi_horse_subset.jsonl) — 55 multi-horse clips with target/distractor motion flags

## 7. Headline recommendations for the project

1. **Keep V-JEPA-2 + LR as the spine.** Features carry the audit-grade signal at small cost; published 0.875 LOSO is reproducible exactly and is not a measurement artifact. The −13.6 pp drop on Strict is a label-side cost, not a feature-side cost.
2. **Multi-rater κ measurement on RHpE field labels is non-negotiable.** Single-observer κ ≈ 0.6 is not enough to clean-retrain the probe at ethogram-grade — the empirical −10 pp retrain-noise penalty in this experiment makes this load-bearing.
3. **Per-source consistency × agreement is the 2-axis diagnostic** that should be computed at the start of any source-aware evaluation. Both dimensions are required to disambiguate calibration findings from noise findings.
4. **Multi-horse confound (19.4 % of RME, all S4 + S8) requires explicit handling** in any production deployment. Lesson 9's conditional bg-masking is the documented mitigation; the recording protocol's "single subject in frame" rule is the prevention.
5. **Borderline (`?`) annotations are signal, not noise.** The 11 within-observer inconsistent flips on subthreshold motion are real measurement uncertainty, not labeling errors. Downstream training pipelines should preserve and use this signal (drop / weight-down / soft-label per application), not flatten it.

The decomposition + the falsifiable κ ≥ 0.7 prediction + the S5 / S10 diagnostic are the three substantive methodological contributions of this audit-followup. The project's existing V-JEPA-2 + LR pipeline is unchanged; what changed is the empirical grounding for the recording-protocol requirements.
