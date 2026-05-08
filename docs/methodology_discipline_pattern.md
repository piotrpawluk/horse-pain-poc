# Methodology discipline pattern — pre-register, honor exits, sequence phases

**Distilled:** 2026-05-08, after the eye-probe arc (MiniCPM closure → Track B Phase 1 → Phase 2 → post-review corrections → Phase 3 pilot).

This document captures the methodology pattern that ran through the experiment as a reusable playbook for future projects. It is **not** project-specific — the discipline itself is the contribution. Worth reading before starting any preregistration-discipline-required experiment.

## The six-element pattern

1. **Pre-register thresholds, decision rules, AND exit conditions before any unblinded run.** Lock by SHA-256 hash committed to a versioned audit file. The hash binds content + commit timestamp = tamper-evident proof of "this document existed in this exact state at this exact moment, before the run."
2. **Pre-commit the failure path AND the ambiguous-zone path, not just the success path.** "If pooled AUC ≥ 0.65 → continue" is half a pre-registration; the full commitment names what happens at 0.55–0.65 and at < 0.55, and explicitly forbids invented-after-the-fact branches.
3. **Catch implementation bugs in writing, not after the run.** Paste reference code snippets (especially metric-computation logic) in the design review. Let reviewers spot leakage / scaler issues / score-comparability assumptions / parity tautologies before any compute is spent.
4. **Honor mechanical decisions even on borderline results.** When the point estimate clears the threshold but the CI overlaps chance, the locked rule still applies. Conditioning on CI width or per-fold spread *after seeing the data* is goal-shifting — exactly the failure mode the pre-registration was designed to prevent. Document the borderline framing in the writeup; do not retroactively change the threshold.
5. **Sequence phases — never amend a closed phase with a new question.** If Phase 3's result raises a follow-up question (precision, replication, second-rater κ), the answer is Phase 4 with its own pre-registration, not a Phase 3 modification. Phases close cleanly; new questions open new phases.
6. **Empirical-anchor any data-structure-dependent rule before committing to it in pre-registration.** Discussion-stage reasoning generates useful hypotheses but cannot reliably anticipate dataset-specific structure. Run a 5-minute empirical check (distribution sweep, cross-tab, class-by-feature balance) BEFORE locking any rule that depends on those properties. Element 3 protects against process bugs in writing; element 6 protects against reasoning bugs about data that only the data can reveal.

## Why each element matters

### Pre-register with hashes

Anchoring a pre-registration in git via SHA-256 hash makes "this was the rule at the time" provable rather than asserted. The audit hook is one versioned file (`docs/preregistration_hashes.md` style) that lists each frozen document by hash + size + freeze timestamp. The cost is one `shasum -a 256` per document; the value is that any future challenge — "did you change the threshold after seeing the result?" — has a one-line answer.

This works even when the pre-registration documents themselves are gitignored (e.g., methodology docs in an `outputs/` directory). The hash + the commit witness existence; nothing more is needed.

### Pre-commit the failure path

Pre-registrations that only describe "the success path" are not actually pre-registrations — they're plans dressed up as commitments. The discipline is to write down, in advance, what you will do if the result is ambiguous (e.g., AUC 0.55–0.65) and what you will do if it falsifies the hypothesis (e.g., AUC < 0.55). Without those branches in writing, the ambiguous-zone result becomes a goal-shift opportunity ("let me try one more crop heuristic"), and the failure becomes a conclusion-flipping retreat ("the pipeline still works, the labels were wrong"). With them, every outcome maps to a pre-committed action.

A useful test: **could a reviewer who hadn't seen the result predict your next move from your pre-registration?** If yes, you've written a real pre-registration. If they need to wait for the result before they can guess, you haven't.

### Catch bugs in writing

A two-line code snippet in a design review takes seconds to paste and seconds to scan. The same bug caught after a run takes anywhere from one re-run (minutes) to a complete result invalidation (the run results contaminate downstream claims). For metric-computation logic — scaler placement, fold-metric vs pooled-metric, score comparability across fold models, permutation null distribution shape, p-value form — paste the actual code. Reviewers spot data leakage and tautological tests in seconds when the code is in front of them; in prose-only design reviews, the bugs are invisible until they fire.

The eye-probe arc caught two such bugs in writing: a precomputed-scaler leakage that would have inflated AUC by ~0.05–0.15, and a parity-test ambiguity (cache-to-cache vs genuine re-extraction) that would have rendered the cosine = 1.0 result methodologically unreviewable. Both took ~15 minutes to fix in the snippet stage; both would have taken hours and a re-run to fix afterward, with the additional cost of explaining why a published number changed.

### Honor mechanical decisions on borderline results

This is the hardest element. When a pre-registered threshold clears on a borderline result — say, pooled AUC 0.6813 with 95 % CI [0.49, 0.88] — the temptation is to introduce a *new* condition the pre-registration didn't include ("CI lower bound must be above 0.5"). That is goal-shifting. The pre-registration's job is to bind your behavior in the future; if it didn't condition on CI width when it was written, it doesn't get to condition on CI width now.

The honest move is: report the mechanical decision faithfully, frame the result honestly in the writeup (CI before p-value; effect-size estimation before significance testing), and let the precision concern surface as a *separate* pre-registered question (Phase 4) with its own commitments.

Concretely: if your pre-registration said "AUC ≥ 0.65 → continue, write up Track A," and you find AUC = 0.68 with CI overlapping chance, the writeup framing is "Pooled AUC 0.68 (95 % CI [0.49, 0.88], p = 0.058). The pre-registered threshold was cleared on the point estimate. The wide CI indicates that precision at this label/N is too low to support clinical or production claims; this is a proof-of-concept-scale pilot finding. Refining the precision is the question Phase 4 addresses."

That sentence is defensible at peer review, audit, or external replication. "We saw AUC 0.68 but decided not to proceed because the CI was wide" is not — it would require explaining why CI was suddenly load-bearing when it wasn't in the pre-registration.

### Sequence phases

When Phase 3 closes with a borderline pass, the questions that come up — "is this signal real?" "does it replicate with a second observer?" "would N = 100 produce a CI that excludes chance?" — are not Phase 3 questions. They are new questions that warrant their own pre-registration, their own thresholds, their own exit conditions, and their own audit hash.

Sequencing prevents a specific pathology: results of one phase silently contaminating the design of the next. If Phase 4 thresholds were chosen *after* Phase 3's pooled AUC = 0.68 was known, then Phase 4 isn't really a fresh test — it's confirmation seeking. If Phase 4 is pre-registered with its design locked **before** Phase 3 results are known (or with the design locked publicly such that the Phase 3 result doesn't influence it), then Phase 4 is a real test and its outcome can falsify or refine the Phase 3 finding.

Operationally: Phase 3 closes, Phase 4 design begins. Phase 4 pre-registration freezes its own thresholds, hashes its own audit document, and proceeds independently. The Phase 3 audit chain is not edited; the Phase 4 audit chain extends it.

### Empirical-anchor data rules before committing

Reviewers and authors alike can reason confidently about per-instance properties ("a sub-second clip yields compressed temporal sampling under fixed-frame video models") while completely missing dataset-level structure that emerges from the joint distribution of those properties with the labels and the source variable. The fix is not to reason harder; it is to **look at the data before locking the rule**.

A 5-minute empirical check costs nothing and reveals what reasoning cannot:

- Distribution sweep — does the candidate cutoff (or threshold, or filter, or stratification) split the data along the axis you intended, or along an axis correlated with the label/source/feature you didn't want to split on?
- Cross-tab — at the chosen cutoff, what is the class balance of what you keep vs what you drop? What is the source distribution?
- Source-by-class balance — does the rule preserve the source-label combinatorics or shift them?

If the empirical check shows the rule has a confound, you have three honest moves: drop the rule (cleanest), redefine the cutoff to a value the data supports, or pre-register the confound explicitly (rare; only when the rule is unavoidable). What you do NOT do is lock the rule first and discover the confound after the run, when removing it costs an invalidated experiment.

*Case study — the sub-second filter in eye-probe Phase 4.* The Phase 4 design initially included a sub-second clip filter as one of three diagnostic-named fixes. Per-clip reasoning was sound: V-JEPA-2's 16-frame sampling against a sub-second native window compresses temporal-feature activation. A 5-minute ffprobe sweep on the 34 clips before pre-registration revealed that ACTION clips were systematically shorter than BACKGROUND clips in this dataset — at a 1.0 s cutoff the dropped set was 87 % ACTION; even at 0.4 s it was 100 % ACTION. The filter would have created a class-balance shift confounding the architecture comparison. The filter was dropped from Phase 4 design before lock-in; the empirical sweep itself became part of the audit. The Phase 4 design improved *because of* the data check, not despite it.

The general principle: prior reasoning is for hypothesis generation; data structure for commitment. Element 3 protection ("catch bugs in writing") covers process bugs that show up in code review; element 6 covers reasoning bugs about data that only the data can reveal.

## When this pattern applies

- Multi-step ML/methodology experiments where the final claim must defend at peer-review or external-audit scale.
- Cross-vendor / cross-architecture comparisons where confounding variables are easy to introduce post-hoc.
- Small-N pilots where the precision of any single metric is necessarily wide and the temptation to goal-shift is high.
- Any project where "we should publish this" or "we should productionize this" is a plausible downstream branch.

## When this pattern is overkill

- One-shot exploratory analyses with no claim attached.
- Engineering tasks (build a feature, ship a fix) where the deliverable is the artifact, not a defensible empirical claim.
- Tightly scoped CI/CD-style experiments that are already structurally pre-registered by their automation.

The cost of the pattern (pre-registration documents, hashing audit hooks, snippet-stage code review, separate phase pre-registrations) is real. Pay it on the experiments where the claim has to defend itself; skip it on the experiments where it doesn't.

## Generalization to AI-collaborator workflows

This pattern was developed in a workflow where an AI agent did the implementation and the human did the review. Two specific elements transfer especially well to that mode:

- **Snippet-stage code review** (element 3) leverages the AI's ability to paste reference implementations cheaply and the human's ability to spot bugs in code more efficiently than in prose. Use it.
- **Locking thresholds in writing before runs** (element 1) prevents the AI from goal-shifting toward whatever framing makes the run look successful — a known failure mode in agentic workflows. The hash-as-audit-hook makes the discipline cheap to enforce and tamper-evident.

The remaining three elements (failure-path commits, mechanical-decision honoring, phase sequencing) apply identically whether the work is human-only or human-AI collaborative. The discipline is universal; the implementation is what changes.

## Anti-patterns this prevents

1. **Result-conditioned threshold revision.** Setting AUC ≥ 0.65 in the pre-reg, then revising to "≥ 0.65 with CI lower bound > 0.55" after seeing AUC = 0.68 with CI [0.49, 0.88]. Caught by element 4.

   *Subtle variant: result-conditioned heuristic revision.* After Phase 3 closed, ~30 minutes of manual crop-inspection on the two inverted folds (S5, S6) produced direct evidence that the pre-committed v2 profile-aware crop heuristic — held in reserve for the 0.55–0.65 ambiguous zone — would have fixed at least one specific clip (`background_S5.mp4_10_`, where the eye was clipped to the corner of the v1 crop). With evidence in hand that "applying v2 would improve the Phase 3 number," the temptation is to retrofit it. **The discipline does not permit this.** The Phase 3 result stands; v2 evidence becomes Phase 4 input. Element 5 (sequence phases) is the rule that catches this, but its load-bearing application is exactly here — when you have a known-good fix and you still don't apply it to the closed phase. The general principle: 30 minutes of post-hoc inspection can transform a mysterious result into an interpretable one *without changing any numbers*; that is the value, and it must remain bounded to interpretation rather than retroactive modification.
2. **Mid-run prompt / heuristic / config iteration.** Running 3 MLLMs on 36 clips, getting templated outputs, and iterating prompts until one produces a defensible-looking distribution. Caught by element 1 (locked single-prompt-per-vendor) and element 2 (pre-committed exit on collapse).
3. **Open-ended phase amendments.** Phase 3 produces ambiguous result; Phase 3 then expands to "Phase 3 with second observer + extended set." Caught by element 5 — that's Phase 4, with its own pre-registration.
4. **Silent leakage in production code.** A "scaler refit per fold" comment in the design review masks a `X_scaled = StandardScaler().fit_transform(X)` precomputation in the actual implementation. Caught by element 3 (paste the actual code, not the description of it).
5. **Cache-tautological parity tests.** A "parity test passed" line that compares cached embedding to itself rather than re-extracting. Caught by element 3 (paste the actual code; reviewer asks "what does the parity_test function actually do?").
6. **Confidently-wrong dataset assumptions.** Pre-registering a filter, cutoff, or stratification rule based on per-instance reasoning that turns out to interact with dataset-level structure (class-by-feature correlation, source skew, etc.). Caught by element 6 — the 5-minute empirical sweep before lock-in.

## One-paragraph executive summary

Pre-register thresholds, decision rules, and exit conditions before any unblinded run; lock them by SHA-256 hash committed to a versioned audit file. Pre-commit the failure path and the ambiguous-zone path, not just the success path. Catch implementation bugs in writing by reviewing reference code snippets at design time, not after the run. Honor the mechanical decision even when the result is borderline; the writeup framing carries the nuance, the threshold-cross is the locked outcome. When new questions arise from a closed phase, open a new phase with its own pre-registration — never amend a closed phase. Empirical-anchor any data-structure-dependent rule with a 5-minute distribution check before locking it; reasoning generates hypotheses but data commits decisions. The cost is small at design time; the value is a result that defends itself at peer review, external replication, or audit.
