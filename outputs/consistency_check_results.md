# Step 3 — Within-observer consistency check results

*Step 3 of `docs/audit-followup-spec.md`. Re-watch executed by the project owner; analysis by the agent. Run 2026-05-07 (re-watch session same day as original audit; longer delay was not feasible — see caveats below). Branch `experiment/audit-followup`.*

## Headline

**Hard-stop gate: PASS.** Confident-case self-consistency = **10/10 = 100 %** (raw verdict-match). Both multi-horse confident controls (`action_S4.mp4_5_.mp4`, `action_S8.mp4_4_.mp4`) held confident `ACTION`. Step 4 (D) + Step 5 (B) cleared to proceed.

| Subset | n | Raw verdict-match | Protocol-adjusted | Cohen's κ |
|---|---|---|---|---|
| **Confident controls (HARD STOP GATE)** | 10 | **10/10 = 100 %** | 10/10 = 100 % | n/a (no chance baseline; 100 % is the ceiling) |
| Borderline (`?`) cases | 56 | 45/56 = 80.4 % | 45/56 = 80.4 % | **0.586 (moderate)** |
| Multi-horse total | 14 | 13/14 = 92.9 % | 13/14 = 92.9 % | high |
|   ↳ multi-horse target_focus | 7 | 6/7 = 85.7 % | 6/7 = 85.7 % | — |
|   ↳ multi-horse distractor | 7 | **7/7 = 100 %** | 7/7 = 100 % | — |
| Single-horse | 52 | 42/52 = 80.8 % | 42/52 = 80.8 % | — |
| **All 66** | 66 | **55/66 = 83.3 %** | 55/66 = 83.3 % | — |

**Raw and protocol-adjusted rates are identical here.** Reason: the 7 predicted FH-only-driven flips (`multi_horse_distractor` originally `BACKGROUND?` → re-watch `BACKGROUND`) are *certainty* flips, not *verdict* flips — the verdict on both passes is `background`. Verdict-match counts them as agreement under either rule. The protocol-adjustment mechanism would have mattered only if any distractor borderlines had originally been labeled `ACTION?` (then re-watch would resolve to `BACKGROUND` under FH-only, a verdict change explained by the protocol). The user's pre-protocol intuition on multi-horse distractor clips was **already aligned with FH-only** (just uncertain) — the rule resolves the uncertainty without changing the verdict.

## Hard-stop gate detail

The 10 confident controls (5 ACTION + 5 BACKGROUND, randomly drawn at seed=42 from the 227 confident-verdict pool). All 10 re-watch verdicts matched the original verdicts exactly:

- 5/5 confident `ACTION` controls held confident `ACTION` on re-watch. Includes the 2 multi-horse `target_focus` cases (`action_S4.mp4_5_.mp4`, `action_S8.mp4_4_.mp4`) — both have strong FH motion; under FH-only they correctly stayed `ACTION`. No FH-only-driven flips in the controls.
- 5/5 confident `BACKGROUND` controls held confident `BACKGROUND` on re-watch.

This satisfies the spec's "≥ 9/10" hard-stop threshold. The audit methodology has demonstrated reliability on clearly-classified clips. Borderline rate below is therefore the rate that actually measures judgment-call drift, not methodological reliability.

## Borderline disagreement direction (11 cases)

Of 56 borderline cases, 45 matched verdict on re-watch and 11 disagreed. The 11 disagreements split asymmetrically:

| Direction | n | Pattern |
|---|---|---|
| `BACKGROUND?` → `ACTION` (more permissive on re-watch) | **7** | Original "extremely slight twitch" calls upgraded to confident action |
| `ACTION?` → `BACKGROUND` (stricter on re-watch) | **4** | Original "extremely slight twitch" calls downgraded to confident still |

Net direction: slightly more permissive (7 toward action vs 4 toward bg), but both tails come from the same "extremely slight motion" category — the user's threshold *for the same descriptive language* genuinely varies across sittings. This is consistent with the original audit's `?` annotation precisely capturing "I would defer to a second opinion" — those clips were meant to be the unstable ones.

### Disagreement detail

**`BACKGROUND?` → `ACTION` (7 cases):**

| Clip | Source | Original observation | Re-watch observation |
|---|---|---|---|
| `background_S2.mp4_5_.mp4` | S2 | slight head movement, extremely light left ear rotation | extremely slight left ear rotation |
| `background_S2.mp4_6_.mp4` | S2 | slight head movement, extremely light left ear rotation | extremely slight left ear rotation |
| `background_S12.mp4_14_.mp4` | S12 | extremely slight left ear twitch | slight both ears rotation |
| `background_S9.mp4_4_.mp4` | S9 | extremely slight left ear twitch | head tilt, extremely slight left ear rotation |
| `background_S1.mp4_5_.mp4` | S1 | slight body movement, extremely slight left ear rotation | extremely slight both ears rotation |
| `action_S3.mp4_9_.mp4` | S3 | strong head movement, ears still | strong head move, extremely slight right ear rotation |
| `background_S8.mp4_1_.mp4` | S8 (MH target) | fh: extremely slight left ear twitch, bh: head movement, slight ears twitch | both ears twitch |

**`ACTION?` → `BACKGROUND` (4 cases):**

| Clip | Source | Original observation | Re-watch observation |
|---|---|---|---|
| `action_S3.mp4_13_.mp4` | S3 | extremely slight twitch in left ear | ears still |
| `action_S3.mp4_16_.mp4` | S3 | extremely slight twitch in right ear | ears still |
| `background_S5.mp4_3_.mp4` | S5 | slight left ear rotation | ears still |
| `background_S5.mp4_8_.mp4` | S5 | extremely slight left ear rotation | ears still |

Every single borderline disagreement involves "extremely slight" or "slight" motion language at least once across the two sittings. **The drift is concentrated entirely in the subthreshold-motion zone**, exactly where the original `?` was meant to capture uncertainty.

## Per-source borderline self-consistency

| Source | n borderline | match | rate | Notes |
|---|---|---|---|---|
| S1 | 1 | 0 | 0.000 | only 1 borderline; the bg→act flip on `S1_5` |
| **S5** | 5 | 3 | **0.600** | **lowest** — matches Lesson 17's S5 = 24 % audit disagreement (subthreshold-heavy) |
| S3 | 10 | 7 | 0.700 | mid — clean source by audit, but contains the 4 "extremely slight" S3 borderlines |
| S9 | 4 | 3 | 0.750 | — |
| S2 | 11 | 9 | 0.818 | — |
| S12 | 6 | 5 | 0.833 | — |
| S8 | 7 | 6 | 0.857 | high — driven by the 7/7 multi-horse distractor matches |
| S4 | 5 | 5 | **1.000** | perfect — all 5 are multi-horse cases (4 distractor + 1 target_focus) |
| S7 | 2 | 2 | 1.000 | — |
| S10 | 5 | 5 | **1.000** | perfect — surprising given S10 = 23 % audit disagreement; user has stable threshold that just differs from EquiFACS |

**S5 vs S10 contrast is informative.** Both sources are high-disagreement-with-RME on the audit (24 % and 23 % respectively, mostly bg→action subthreshold). But within-observer:
- **S10: 5/5 = 100 %.** User's threshold is *stable* — it just differs from EquiFACS. This is calibration, not noise.
- **S5: 3/5 = 60 %.** User's threshold is *unstable* — even within-observer, the subthreshold calls flicker.

This means the audit-RME disagreement on S5 has both a calibration component (user threshold ≠ EquiFACS) and a noise component (user threshold isn't even stable across sittings). On S10 it's pure calibration. Worth flagging in the methodology writeup.

## Multi-horse breakdown — the FH-only protocol worked

| Subcategory | n in re-watch | match | rate |
|---|---|---|---|
| `multi_horse_target_focus` | 7 | 6 | 85.7 % |
| `multi_horse_distractor` | 7 | **7/7 = 100 %** | The 7 predicted protocol-driven flips all materialized: `BACKGROUND?` (uncertain because BH motion noted) → confident `BACKGROUND` (FH ears still under FH-only rule) |

The single multi-horse target_focus disagreement is `background_S8.mp4_1_.mp4`: original `BACKGROUND?` ("fh: extremely slight left ear twitch, bh: head movement, slight ears twitch") → re-watch `ACTION` ("both ears twitch"). The change in observation language ("extremely slight left ear" vs "both ears twitch") suggests the user perceived more FH motion on the second pass — borderline subthreshold case, similar drift mechanism to the single-horse subthreshold flips.

## Cohen's κ — borderline binary self-consistency

| Statistic | Value |
|---|---|
| Observed agreement (binary verdict) | 0.804 |
| Expected agreement under chance | 0.526 |
| **Cohen's κ** | **0.586** |
| Interpretation (Landis & Koch 1977) | **moderate** (0.41–0.60) |

For test-retest within-observer on the explicitly borderline (`?`) subset, κ = 0.586 is in the "moderate" band — substantial enough for the audit to support quantitative claims when the borderline rate is reported alongside, but not strong enough to support a rigid label set on these cases. The 80 % observed agreement is consistent with this; the ≥ 90 % barrier would have been "substantial" by the same scale. **The borderline cases by definition are the unstable ones**; the same κ on confident-case re-watch would be ≈ 1.0 (10/10 with chance ≈ 0.5).

## Caveats

- **Re-watch was same-day, not >12 h delayed.** The spec called for "ideally >12 h" delay. The user re-watched on the same calendar day as the original audit. This biases self-consistency toward the high side — fresh-memory effects on observation language and verdict choice are not eliminated. The 80 % borderline rate / κ = 0.586 should be read as a **ceiling** on within-observer reliability, not the long-term rate. A second re-watch in 1–2 weeks would tighten the bound.
- **Re-watch may have been informed by knowing the original audit existed.** The user knows clips were sampled from `?` cases plus 10 controls; this could affect the priors brought to each clip ("if it's in this list, it's probably borderline"). True blind re-watch would require the user to forget which clips were in the audit and which weren't.
- **The drift is small and asymmetric.** 7 bg?→act vs 4 act?→bg. Both directions are "extremely slight" motion calls. This is the documented inter-rater problem for subthreshold motion in EquiFACS-grade annotation — even the same observer cannot reliably reproduce their own borderline calls on visually-near-zero motion.
- **N per source is small.** S1 has 1 borderline clip; S7 has 2; etc. Per-source rates are suggestive, not statistically settled.

## What this enables for downstream Steps 4 + 5

- **Hard-stop gate cleared.** Step 4 (multi-horse subset extraction, 55 clips in S4 + S8) and Step 5 (V-JEPA-2 + LR LOSO under three label variants) both proceed.
- **The 7 multi-horse distractor protocol-flips are confirmed.** When Step 5 builds the "Strict (Piotr-certain)" label set, those 7 clips contribute confident `BACKGROUND` (post-re-watch) rather than being dropped as borderline. This *shrinks* the borderline-drop count from the original 56 to 49 effective borderlines (since 7 are now resolved post-re-watch).
- **The 11 borderline disagreements should be flagged as low-confidence in any downstream label set.** They could be: (a) dropped from training/eval (most conservative), (b) given the most-recent verdict, (c) given a "soft label" of 0.5 or weighted lower. Step 5 will choose one approach and document.
- **The κ = 0.586 figure is the headline caveat for Lesson 17.** Replaces the "pending Step 3" placeholder. Any downstream claim about V-JEPA-2 LOSO or MLLM thresholds against Piotr labels carries this 0.586 within-observer-reliability tag — not a settled benchmark, but a measured-not-imagined bound.

## What gets inserted into Lesson 17

A short paragraph replacing the "Pending Step 3" section, summarizing:

> Step 3 within-observer consistency check (2026-05-07) cleared the hard-stop gate (10/10 confident-control matches = 100 %). On the 56 borderline (`?`) cases, raw verdict-match self-consistency was 45/56 = 80.4 % with Cohen's κ = 0.586 (moderate). All 7 multi-horse distractor borderlines resolved to confident `BACKGROUND` under the FH-only protocol clarification, exactly as predicted. The 11 verdict-flips (7 bg?→act, 4 act?→bg) are concentrated in the "extremely slight motion" subthreshold zone — exactly where the original `?` was meant to capture uncertainty. Re-watch was same-day rather than the spec-recommended >12 h, biasing the consistency rate toward the high side; treat 80.4 % / κ = 0.586 as a ceiling on within-observer reliability, not a settled long-term rate.

Lesson 17's "single-observer audit" caveat now reads against κ = 0.586 (moderate) as the empirical bound, not against an unmeasured `?`.
