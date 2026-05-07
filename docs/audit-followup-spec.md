# Audit Follow-up — Gated Execution Sequence

*Saved as `docs/audit-followup-spec.md` on branch `experiment/audit-followup`, branched from `fix/qwen-mlx-video-input`.*

*Context for this spec: the user manually audited all 283 RME clips against their own threshold for "is there visible ear motion." The audit found 12.4% disagreement with published EquiFACS-derived labels (24 bg→action, 11 action→bg, 17 confident, 18 borderline) and 19.4% multi-subject confound rate concentrated in S4 + S8. This spec sequences the follow-up work into gated steps so we don't commit to large experiments without first verifying they're worth running.*

---

## 0. Pre-flight resolutions (2026-05-07, before branching)

Two pre-flight conflicts were surfaced and resolved before this spec was committed:

- **Conflict 1 — per-clip V-JEPA-2 LOSO predictions don't exist as a saved file.** The published 0.875 LOSO output stored only aggregate per-source AUC. Cached `vjepa2_embeddings.npz` (283 × 1024 features + labels + splits + filenames) lets us regenerate per-clip predictions in ~5 min by running sklearn LogReg per LOSO fold. **Resolution:** B-prime (Step 2) regenerates per-clip predictions from cached features. **Added sanity check:** before treating predictions as comparable to the published baseline, aggregate per-clip → per-source AUC and confirm reproduction of 0.875 ±2pp. If hyperparameter mismatch shows (regularization / class_weight / solver / max_iter — typical suspects), surface and stop. Cheap insurance; same code path produces both outputs.
- **Conflict 2 — borderline (`?`) count: 56 actual vs spec's expected 30.** All 56 borderline cases will be included in the consistency check (Step 3), plus 10 confident controls (5 ACTION, 5 BACKGROUND, random seed 42), totaling **66 clips**. The 30-sample would have been enough for the gate but suboptimal for the Step 6 protocol doc, which characterizes the user's borderline-handling rules. Marginal cost: ~25 min user re-watch. Marginal value: complete per-clip consistency data on every borderline case feeds directly into the protocol doc.

---

## 1. What this spec replaces

The agent previously proposed an A+B push (~2.5h) covering Lesson 17 + Qwen re-evaluation against user labels (A) and a full V-JEPA-2 + LR LOSO re-run under user labels (B). That proposal is sound but commits to ~90 min of model retraining (B) before checking whether retraining is high-value. This spec adds two gates:

- **B-prime**: a 15-min agreement decomposition using *existing* V-JEPA-2 predictions, before deciding whether to retrain.
- **Self-consistency check**: a 30-min within-observer agreement measurement on borderline cases, necessary for any writeup claim that the new labels are "cleaner."

It also adds:

- **Light C** as a default deliverable (one-page protocol doc) rather than conditional on B.
- **D** (multi-horse subset extraction) as a precursor to B's "cleaned" variant, run only if B is triggered.

---

## 2. Sequence overview

```
Step 1: A           (~45 min)  →  always run, ships immediately
Step 2: B-prime     (~15 min)  →  always run, gates B (now with 0.875 ±2pp sanity check)
Step 3: Consistency (~30-55 min) →  always run, gates writeup quality
Gate 1: decide on B based on B-prime result
Step 4: D           (~30 min)  →  only if B will run
Step 5: B           (~90 min)  →  conditional on Gate 1
Gate 2: decide on full C based on B result
Step 6: Light C     (~30 min)  →  always run, regardless of Gate 2
Step 7: Full C      (~90 min)  →  only if Gate 2 says yes
```

Minimum total wall-clock: ~2.5h (A + B-prime + consistency + light C, no retraining; user re-watch budget +25 min for full 56-borderline coverage).
Maximum total wall-clock: ~5.5h (everything triggered, full methodology writeup).

---

## 3. Step 1 — A: Lesson 17 + Qwen re-evaluation against user labels

**Effort:** ~45 min. **Always runs.** **No gates.**

### Inputs
- The 283-clip user audit data: `temp/rmy_assessment.txt`
- Qwen v2 output JSONLs (`outputs/qwen25vl_7b_promptA_v2.jsonl`, `outputs/qwen25vl_7b_promptC_v2.jsonl`)
- Gemini 2.5 Pro and 3.1 Pre output JSONLs (already in `outputs/`)
- The 36-clip subset manifest

### Tasks
1. Convert the user audit notes into a structured JSONL: `outputs/piotr_audit_labels.jsonl` with fields `{clip, observation, verdict, certain (bool, false if "?" present), category}` where category ∈ `{strong_motion, slight_motion, subthreshold, head_only, body_only, multi_horse_target_focus, multi_horse_distractor, scene_cut, error_frame, ears_still}`. Single-pass parse from the audit notes; no interpretation beyond what's in the user's text.
2. Add **Lesson 17** to `docs/lessons_learned.md`:
   - Headline: 12.4% disagreement rate with EquiFACS-derived RME labels on full 283-clip audit by single domain-aware reviewer.
   - Per-source disagreement table (S6 0% to S5/S10 ~24%) — copy from agent's previous analysis.
   - Asymmetric direction: 24 bg→action (mostly subthreshold motion), 11 action→bg (mostly head-only or distractor-only motion).
   - Confidence stratification: 17 certain, 18 borderline.
   - Multi-horse confound: 19.4% (55 clips, all in S4 + S8).
   - Important caveat in the lesson body: this is single-observer audit; numbers should be read as "observer disagreement with published protocol" not "true label noise rate." The within-observer consistency check from Step 3 will tighten or weaken this caveat.
3. Add an **S4_7 reframe paragraph to Lesson 15 v2**:
   - Qwen v2 prompt A reasoning: "The horse's ears are moving slightly, indicating active movement…"
   - Qwen v2 prompt C reasoning: "The horse's ears are slightly rotated and twitching, indicating movement."
   - User audit confirmed the background horse twitched its ears.
   - Recategorize as multi-subject confound (Lesson 9 territory), not perceptual error.
   - V2 false-positive accounting changes from "1/22 false positive" to "0/22 perceptual error + 1/22 multi-subject confound," which are qualitatively different numbers.
4. Recompute Qwen v2 + Gemini agreement on the 36-clip subset under user labels:
   - **Strict variant**: only `certain=True` user verdicts; drop borderline cases.
   - **Permissive variant**: count `?` cases as agreement with original RME label.
   - Report both variants in `outputs/qwen_vs_gemini_comparison.md` alongside the original-label numbers.
5. Update PR #2 description with the Lesson 15 v2 reframing and the new agreement numbers.

### Deliverables
- `outputs/piotr_audit_labels.jsonl`
- Updated `docs/lessons_learned.md` with Lesson 17 + Lesson 15 v2 S4_7 paragraph
- Updated `outputs/qwen_vs_gemini_comparison.md` with strict + permissive variants
- PR #2 comment with the reframed numbers

### Exit criteria
- Lesson 17 reviewable
- Both label-set agreement numbers documented
- PR #2 unblocked

**No gate after this step. Continue to Step 2.**

---

## 4. Step 2 — B-prime: V-JEPA-2 prediction agreement decomposition

**Effort:** ~15 min. **Always runs.** **Gates Step 5 (B).**

### Goal
Determine whether V-JEPA-2 + LR's predictions track RME labels (expected — it was trained on them) or user labels (would indicate the model is finding the visual signal despite label noise). This decides whether full retraining (Step 5) is high-EV or expected-modest.

### Inputs
- Cached V-JEPA-2 features: `outputs/vjepa2_embeddings.npz` (283 × 1024)
- `outputs/piotr_audit_labels.jsonl` from Step 1
- The original RME labels from `vendor/ReadMyEars_Dataset/data/{train,val,test}.csv`

### Tasks
1. Run LogReg-per-fold LOSO on cached V-JEPA-2 features (NOT retraining V-JEPA-2 — only the linear probe per fold). Match published baseline config; record hyperparameters.
2. **Sanity check (added per pre-flight):** aggregate per-clip predictions back to per-source LOSO AUC and confirm reproduction of published 0.875 ±2pp. If outside this band, surface the hyperparameter mismatch (regularization, class_weight, solver, max_iter — usual suspects) and stop. Predictions are not comparable to the published baseline until reproduction is clean. Same pass produces both per-source AUC and per-clip predictions.
3. Save per-clip predictions to `outputs/vjepa2_loso_per_clip_predictions.jsonl`.
4. For each clip in the 283-clip set, record: `{clip, source, vjepa2_pred, vjepa2_score, rme_label, piotr_verdict, piotr_certain}`.
5. Compute three agreement matrices, broken down by source:
   - **V-JEPA-2 vs RME** (this is what 0.875 LOSO measures, replicated as a sanity check)
   - **V-JEPA-2 vs Piotr (strict, certain only)**
   - **V-JEPA-2 vs Piotr (permissive, ? = agree with RME)**
6. Plot or tabulate per-source disagreement rates side-by-side.
7. Identify the disagreement pattern:
   - **Pattern A — V-JEPA-2 leans toward Piotr labels:** V-JEPA-2 vs Piotr agreement > V-JEPA-2 vs RME agreement on at least 7/12 sources. Means model is finding visual signal despite training on RME's stricter labels. **Triggers Step 5 (B) as high-EV.**
   - **Pattern B — V-JEPA-2 tracks RME labels:** V-JEPA-2 vs RME ≥ V-JEPA-2 vs Piotr on most sources. Expected outcome — model fits its training labels. **Step 5 (B) becomes lower-EV; can skip or downgrade.**
   - **Pattern C — Mixed by source:** V-JEPA-2 leans toward Piotr on some sources (likely the noisier ones — S5, S10), tracks RME on cleaner sources (S6, S7). **Triggers Step 5 (B) with focus on noisy-source sub-analysis.**

### Deliverables
- `outputs/vjepa2_loso_per_clip_predictions.jsonl`
- `outputs/vjepa2_label_agreement_decomposition.md` with per-source tables and the identified pattern
- Pattern letter (A / B / C) and rationale logged at the top
- Sanity-check result (per-source AUC reproduction confirmed or surfaced) at the top

### Gate 1 decision
After this step, the agent decides:
- **Pattern A or C → run Step 5 (B). Run Step 4 (D) first as precursor.**
- **Pattern B → skip Step 5. Document in `outputs/vjepa2_label_agreement_decomposition.md` why retraining was not pursued.**

If Gate 1 says skip B, the agent proceeds to Step 3 (consistency check still runs) and then Step 6 (light C). The 90 min that would have gone to B is recovered.

---

## 5. Step 3 — Within-observer consistency check

**Effort:** ~55 min user re-watch (66 clips × ~50 sec each), ~15 min agent analysis. **Always runs.** **Gates writeup quality, not execution path.**

### Goal
Establish how internally consistent the user's audit is. The 12.4% disagreement rate from Step 1 is "single observer disagreement with published labels"; without a consistency measurement, the framing in any writeup must be defensive ("preliminary single-observer audit"). With a consistency measurement, the framing can be quantitative ("observer self-consistency X%, cross-observer agreement unmeasured but bounded by self-consistency").

### Tasks (per pre-flight Conflict 2 resolution: full 56 borderline + 10 confident = 66 clips)
1. **The user does this step**, not the agent. The agent prepares the materials and analyzes the result.
2. Agent prepares: a randomized list of **all 56 borderline (`?`)** clips plus 10 clips drawn from confident verdicts (5 ACTION, 5 BACKGROUND, random seed 42) as positive controls. Save the clip list to `outputs/consistency_check_clips.txt` *without* the user's prior verdicts.
3. User re-watches each clip after a delay of at least several hours (ideally >12 h, definitely with a break since the original audit). Records new verdict and observation.
4. Agent compares new verdicts to original verdicts. Computes:
   - Self-consistency overall: % of 66 clips with same verdict.
   - Self-consistency on borderline cases (the 56): % matching original.
   - Self-consistency on confident cases (the 10): % matching original (should be ~100% if methodology is sound; if not, broader audit reliability is in question).
5. If self-consistency on confident cases is <90% (≥1 of 10 wrong), **stop**. Audit reliability is not strong enough to support a writeup claim about label noise. Re-evaluate methodology.
6. Otherwise, document the self-consistency rate in Lesson 17 and use it as the headline caveat figure: "audit self-consistency X% on borderline cases, Y% on confident cases."

### Deliverables
- `outputs/consistency_check_clips.txt` (agent prep, before user re-watch)
- `outputs/consistency_check_results.md` (agent post-analysis, after user re-watch)
- Updated Lesson 17 with the self-consistency figures inserted

### Gate
- Confident-case self-consistency <90% → **stop the entire follow-up sequence.** Lesson 17 reverts to a stub. The audit data is preserved but doesn't support quantitative claims. This is a hard stop, not a discretionary one.
- Confident-case self-consistency ≥90% → continue per Gate 1's decision on Step 5.

---

## 6. Step 4 — D: Multi-horse confound subset extraction

**Effort:** ~30 min. **Conditional on Gate 1 saying B will run.**

### Goal
Define the 55 multi-horse confound clips as a labeled subset, both for B's "cleaned" LOSO variant and for future Lesson 9 follow-up.

### Tasks
1. From `outputs/piotr_audit_labels.jsonl`, extract clips with category `multi_horse_target_focus` or `multi_horse_distractor`.
2. Save to `outputs/multi_horse_subset.jsonl` with fields `{clip, source, original_label, piotr_label, fh_observation, bh_observation, target_motion_present (bool), distractor_motion_present (bool)}`.
3. Verify the count is 55, all in S4 (31 expected) + S8 (24 expected). If counts mismatch, surface and stop — there's a parsing bug.
4. Add a one-line note to Lesson 9 in `docs/lessons_learned.md`: "Multi-horse confound rate quantified at 19.4% (55/283), all S4 + S8, in `outputs/multi_horse_subset.jsonl`."

### Deliverables
- `outputs/multi_horse_subset.jsonl`
- Updated Lesson 9 cross-reference

---

## 7. Step 5 — B: V-JEPA-2 + LR LOSO re-run under user labels

**Effort:** ~90 min. **Conditional on Gate 1.**

### Goal
Compute V-JEPA-2 + LR LOSO AUC against user labels under three label variants. Determine whether the published 0.875 number under- or over-states model performance under different ground-truth assumptions.

### Important framing constraints
- The new LOSO numbers are **not** directly comparable to the published 0.875 in a "this is the better number" sense. Different label sets measure different ground truths. Both numbers are valid; they measure different things. Writeup must be precise about this.
- The point of B is to characterize V-JEPA-2 + LR's behavior under cleaner / different / cleaner-and-multi-horse-removed labels, not to claim a "better" LOSO.

### Tasks
1. Run three LOSO variants using the existing notebook 02 V-JEPA-2 + LR pipeline, swapping label sources:
   - **Strict (Piotr-certain only):** drop the 18 borderline `?` clips. Train + LOSO eval on the remaining 265 clips with user labels. *Note: actual borderline count is 56 (per pre-flight); strict variant drops all 56, leaving 227 clips. Spec retained "265" wording but actual implementation uses 56-drop.*
   - **Permissive (? = agree with RME):** keep all 283. Borderline cases use RME label as ground truth.
   - **Cleaned (multi-horse excluded):** 283 − 55 = 228 clips. Single-subject only. Use Piotr-strict labels with `?` dropped (or permissive — agent's call, but document which).
2. Each variant produces: per-source LOSO AUC, mean LOSO AUC, per-clip predictions saved to `outputs/loso_v2_<variant>_predictions.jsonl`.
3. Side-by-side comparison table in `outputs/loso_label_variant_comparison.md`:

| Variant | Labels used | N clips | Mean LOSO AUC | Per-source range |
|---|---|---|---|---|
| Original (RME) | RME | 283 | 0.875 (published) | … |
| Strict (Piotr-certain) | User, certain only | 227 | ? | … |
| Permissive | Mixed | 283 | ? | … |
| Cleaned | User, single-subject | 228 | ? | … |

4. Interpret the deltas. Possible patterns:
   - **All variants ≈ 0.875**: model is robust to labeling protocol differences within the visible-motion → AU-coded space. Strong robustness story.
   - **Strict / Cleaned > 0.875**: model performance was being held down by label noise; cleaner labels reveal higher actual ceiling. Strongest validation of V-JEPA-2 + LR as the spine.
   - **Strict / Cleaned < 0.875**: model was fitting label noise. Performance under cleaner labels is genuinely lower. Suggests V-JEPA-2 + LR's discrimination is partially tracking EquiFACS-specific quirks rather than general motion-AU patterns.
   - **Per-source heterogeneous**: model behavior varies by source. Likely correlates with per-source disagreement rate from Step 1.

### Deliverables
- Three `outputs/loso_v2_<variant>_predictions.jsonl` files
- `outputs/loso_label_variant_comparison.md` with table + interpretation
- New paragraphs in Lesson 17 incorporating the LOSO variant results

### Gate 2 decision
After this step:
- **Strong / interesting result** (all three variants tell a coherent story, deltas are large enough to warrant writeup): trigger Step 7 (full C).
- **Modest / null result** (variants ≈ 0.875): document and ship; no full C writeup. Light C still runs.

---

## 8. Step 6 — Light C: Labeling protocol extraction

**Effort:** ~30 min. **Always runs, regardless of Gate 2.**

### Goal
Extract the implicit labeling protocol the user defined through the 283-example audit into a one-page document. This is the most directly RHpE-relevant artifact — for field data collection, the protocol that needs documenting is the user's, not RME's.

### Tasks
1. Read through `outputs/piotr_audit_labels.jsonl` and the original audit notes.
2. Extract decision rules from observed patterns:
   - **Threshold for "ear motion":** describe the level of motion the user labeled as ACTION (likely "visible rotation, twitching, or pinning of either ear within the clip"). Document the gray zone ("very slight twitch" → ACTION, "extremely slight twitch" → BACKGROUND? — these are the user's calibration points).
   - **Multi-subject handling:** target horse only (foreground); background horse motion is not action even if visible.
   - **Anatomical scope:** ear region only; head movement, body movement, or tail movement do not count as action even if dramatic.
   - **Edge cases:** scene cuts and frame errors → BACKGROUND? (uncertain). Occlusion → not labelable. Off-camera ears → not labelable.
   - **Borderline policy:** clips marked `?` are "I would defer to a second opinion."
3. Produce `docs/labeling-protocol-2026-05.md` (~1 page) with:
   - Purpose: define the labeling protocol for ear-motion classification used in the May 2026 audit, intended as the basis for future RHpE field data labeling.
   - Decision rules (per the extraction above).
   - Worked examples: 3-5 clips that exemplify each major decision rule (cite by clip name; the agent doesn't need to display them).
   - Known divergences from EquiFACS / RME: more permissive on subthreshold motion; explicit on multi-subject; explicit on anatomical scope.
   - Limitations: single-annotator origin; self-consistency rate per Step 3; no second-observer validation.

### Deliverables
- `docs/labeling-protocol-2026-05.md`

### Why this runs always
The protocol document is the artifact most directly useful for the user's RHpE goal. RHpE field data will need a labeling protocol; the user has now implicitly defined one through 283 examples. Capturing it before the working memory of those decisions fades is the load-bearing step. Even if B is skipped and the writeup is minimal, the protocol doc is worth ~30 min on its own.

---

## 9. Step 7 — Full C: Methodology contribution writeup

**Effort:** ~90 min. **Conditional on Gate 2.**

### Goal
A `docs/methodology-note-2026-05.md` documenting the audit methodology, headline numbers, LOSO variant comparison, and labeling protocol as a coherent narrative — at the level of detail that would be defensible if shown to Andersen / Zamansky.

### Trigger condition
Gate 2 says "interesting result" — i.e., the LOSO variants showed deltas large enough that the comparison itself is informative. If all variants are within ~2pp of each other, this step doesn't run; the labeling-protocol doc from Step 6 captures everything that matters and the variants don't add narrative weight.

### Tasks
1. Compose a 3-5 page methodology note covering:
   - **Section 1 — Audit methodology**: single-reviewer protocol, observation-and-verdict format, multi-horse fh:/bh: notation, certain/borderline distinction, self-consistency check result.
   - **Section 2 — Findings on RME label structure**: 12.4% disagreement rate, asymmetric direction, per-source heterogeneity, multi-horse 19.4%, contextualization within EquiFACS coding criteria.
   - **Section 3 — V-JEPA-2 + LR robustness analysis**: B-prime decomposition + LOSO variant table from Step 5, interpretation of what the deltas mean for the model's discrimination behavior.
   - **Section 4 — Labeling protocol (reference to Step 6 doc)**.
   - **Section 5 — Limitations and what's needed for external validation**: single-annotator, no inter-annotator agreement, audit done as part of MLLM-experiment side investigation rather than as a planned audit, RME-specific (does not generalize to other behavior datasets without re-audit).
2. Cross-link to Lessons 9, 14, 15 v2, 17 for context.
3. Add a short "potential follow-ups" section: second-observer audit, replication on another EquiFACS-derived dataset, etc.

### Deliverables
- `docs/methodology-note-2026-05.md`
- Optional: PR #2 description gets a final paragraph linking the methodology note as the reason the §4 row 1 framing was reframed.

---

## 10. Scope discipline

- **Do NOT** retrain V-JEPA-2 + LR before B-prime confirms the retraining is high-EV.
- **Do NOT** present new LOSO numbers as "the better V-JEPA-2 result" or as superseding the published 0.875. They are additional measurements under different label sets, not better measurements.
- **Do NOT** make claims about "label noise" or "labels are wrong" based solely on user audit. Without inter-annotator agreement, the strongest defensible framing is "single-observer disagreement with published protocol, X% within-observer self-consistency."
- **Do NOT** modify the original RME labels in `vendor/ReadMyEars_Dataset/`. User labels go in a separate JSONL; comparisons happen at evaluation time, never by overwriting.
- **Do NOT** spend more than the budgeted time per step. If something blocks past budget, surface and stop — don't improvise extensions.
- **Do NOT** skip Step 3 (consistency check) even if it feels like a delay. The writeup quality of every downstream step depends on it.
- **Do NOT** skip Step 6 (light C) even if everything else is skipped. The protocol doc is the most RHpE-relevant artifact.

---

## 11. Quick start for the agent

```
1. Read this spec end-to-end. Read the agent's prior A+B+C+D proposal. Note what changed (B-prime added, consistency check added, light C made unconditional, pre-flight refinements §0).
2. Branch experiment/audit-followup from fix/qwen-mlx-video-input.
3. Step 1 (A): always run. Ship Lesson 17 + S4_7 reframe + dual-label agreement.
4. Step 2 (B-prime): always run. 0.875 ±2pp sanity check first; if clean, decide Pattern A/B/C.
5. Step 3 (consistency check): prepare materials (66 clips: 56 borderline + 10 confident); user does the re-watch; agent computes rates. HARD STOP if confident-case consistency <90%.
6. Gate 1: B runs only if Pattern A or C.
7. If B runs: Step 4 (D) first as precursor, then Step 5 (B).
8. Step 6 (light C): always run.
9. Gate 2: full C only if B's variants showed meaningful deltas.
10. Final: PR #2 update, all branches merged with their respective evidence.
```

If anything in the repo conflicts with what this spec assumes (file names, prior outputs, lesson numbering), surface and stop rather than guessing. The goal is decision quality at each gate, not raw throughput.
