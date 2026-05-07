# Labeling protocol — ear-motion classification (May 2026)

*Extracted from the 283-example audit conducted 2026-05-07. The protocol below was followed implicitly during the original audit and explicitly during the within-observer re-watch (Step 3 of `audit-followup-spec.md`). Within-observer Cohen's κ on borderline cases = 0.586 (moderate) — this protocol is therefore a **reproducible single-observer rule set**, not a multi-rater-validated ethogram. Intended use: the basis for RHpE field-data labeling, with inter-rater κ ≥ 0.7 measurement layered on top before any deployment claim.*

## 1. Purpose and scope

Define the labeling protocol applied to ~1.5-second video clips of horses for binary ear-motion classification (ACTION vs BACKGROUND). Designed for the Read My Ears dataset (Alves et al. CVPR W'25) re-audit and intended to scale to RHpE field-data collection (`docs/recording-protocol.md`). **This is a more permissive threshold than the EquiFACS coder protocol that produced the original RME labels** — see §8 for the divergence.

## 2. Decision rule — primary

**ACTION:** at least one ear shows a positional change during the clip — rotation, twitch, flick, or pinning — that is **visible to a careful viewer in normal-speed playback** without frame-by-frame inspection. The motion can be brief (<200 ms) or sustained.

**BACKGROUND:** ears remain in essentially the same position throughout the clip, OR head / body / tail moves while ears are stationary, OR the only visible motion is sub-perceptible at normal-speed playback.

## 3. Multi-subject handling — FH-only rule

**The verdict applies to the foreground horse only.** The foreground horse (FH) is defined as the principal subject of the framing — the horse that is largest in frame, most central, or in focus. Background-horse (BH) motion is **excluded regardless of magnitude**.

- If FH ears clearly move → ACTION (or ACTION? if subtle)
- If FH ears clearly still → BACKGROUND, even if BH ears are visibly moving
- If FH cannot be identified, or its ears are not visible (occlusion, off-frame, behind other anatomy) → BACKGROUND? or ACTION? with a note specifying the obstruction (e.g. "FH partially out of frame", "FH ear behind handler's hand")

In the May 2026 audit, multi-subject scenes occur in 19.4 % of RME clips (55/283), all in sources S4 (31 clips) + S8 (24 clips). The `fh:` / `bh:` notation used in the audit observations records per-subject motion separately for each frame; verdict applies to FH only.

**Step 3 finding worth noting:** in the original audit, all 7 reviewed `multi_horse_distractor` cases (FH ears still, BH moves) were labeled `BACKGROUND?` — uncertain. After the FH-only rule was made explicit, all 7 resolved to confident `BACKGROUND` on re-watch. **The FH-only rule was already being applied implicitly; the protocol clarification removed uncertainty without changing verdicts.** This suggests careful annotators arrive at FH-only intuitively but require explicit articulation for inter-rater consistency.

## 4. Anatomical scope — ears only

Verdict is determined by ear motion only. The following do not contribute to the ACTION label, even when present and visually striking:

- **Head movement** (tilt, rotation, lifting, lowering) without ear motion → BACKGROUND
- **Body movement** (sway, shift in stance, breathing-driven flank motion) without ear motion → BACKGROUND
- **Tail or leg motion** → not relevant; verdict determined by ears

When head or body moves and ears are simultaneously stationary, label BACKGROUND with the motion noted in the observation field (e.g. *"strong head movement, ears still — BACKGROUND"*). This frequent case happens on ~6 % of clips per the audit; the head-motion-without-ear-motion category is a real and stable BACKGROUND case.

## 5. Borderline policy — the `?` annotation

A `?` suffix on the verdict (`ACTION?` or `BACKGROUND?`) means **"borderline; I would defer to a second opinion."** Used when:

- Motion is at the threshold of visibility ("very slight" / "extremely slight" twitch, ~10 % of clips) — typically subthreshold for the ACTION rule above but not unambiguously absent
- Multi-subject scene where FH ear visibility is partially occluded
- Edit-cut artifacts (clip contains two non-contiguous frames with a transition) — see §6

Within-observer self-consistency on `?` cases: 80.4 % verdict-match (κ = 0.586) over a same-day re-watch — borderline is genuinely ambiguous, even within a single observer across sittings. **Borderlines should not be treated as label-noise to be removed; they are explicit uncertainty signals.** Downstream models should either drop them from training (the Strict variant choice in `audit-followup-spec.md`) or include them with reduced weight.

## 6. Edge cases

- **Edit-cut artifacts** (clip contains 2+ non-contiguous frames with a visible transition): label BACKGROUND? with note *"edit-cut, FH ears still"* or appropriate. Examples in audit: `background_S2.mp4_3`, `background_S12.mp4_3`. Five such clips in RME (1.8 %).
- **Occluded ears** (handler hand, fence post, motion blur on ear region): label as borderline with the obstruction specified. Re-do the clip on re-watch if possible; if persistent, the clip is not labelable and should be dropped from training data.
- **Off-camera ears** (ear leaves frame mid-clip): if ear was visibly moving before exit, ACTION; if ear was stationary before exit and re-enters stationary, BACKGROUND; if ear leaves stationary and doesn't return, BACKGROUND with note.
- **Ear twitch on the visible ear, hidden ear status unknown**: ACTION (one ear suffices for the rule).
- **Ambient motion that affects ear position passively** (head tilts, ears rotate as a consequence of head motion, no independent ear movement): BACKGROUND. Rationale: ear-motion-as-AU per EquiFACS protocol is independent ear movement, not a kinematic byproduct of head motion. The protocol matches this by anchoring ACTION on ear-only motion.

## 7. Worked examples (cite by clip name)

**Strong action examples** (unambiguous ACTION):
- `action_S1.mp4_4_.mp4` — *"strong (almost 180 deg) rotation of both ears"*
- `action_S2.mp4_0_.mp4` — *"strong rotation forward of right ear"*
- `action_S6.mp4_2_.mp4` — *"strong left ear rotation"*

**Slight action examples** (visible but small, ACTION):
- `action_S1.mp4_0_.mp4` — *"both ears slightly rotating backward"*
- `action_S5.mp4_4_.mp4` — *"slight right ear rotation"*

**Subthreshold examples** (borderline `?`):
- `action_S3.mp4_13_.mp4` — *"extremely slight twitch in left ear"* — ACTION?
- `background_S2.mp4_5_.mp4` — *"extremely slight left ear rotation"* — BACKGROUND? (drift between original audit BACKGROUND? and re-watch ACTION on this and similar clips reflects genuine borderline ambiguity, not labeling error)

**Head-only / body-only motion** (BACKGROUND despite visible activity):
- `action_S1.mp4_6_.mp4` — *"slight head tilt, ears still"* — BACKGROUND
- `action_S2.mp4_1_.mp4` — *"head moved slightly up, ears still"* — BACKGROUND
- `background_S1.mp4_1_.mp4` — *"slight body movement, ears still"* — BACKGROUND

**Multi-subject FH-only application:**
- `action_S4.mp4_2_.mp4` — *"fh: strong both ears rotation, bh: ears still"* — ACTION (FH carries motion)
- `action_S4.mp4_7_.mp4` — *"fh: strong head movement, ears still, bh: no head visible"* — BACKGROUND (FH ears still; BH not contributing)
- `background_S4.mp4_5_.mp4` — *"fh: ears still, bh: slight left ear rotation"* — BACKGROUND (FH ears still; BH motion excluded under FH-only)

**Edit-cut artifact:**
- `background_S2.mp4_3_.mp4` — *"two frames overlap, hard to distinguish"* — BACKGROUND?

## 8. Known divergences from EquiFACS / Read My Ears protocol

The EquiFACS coder protocol that produced the original Read My Ears labels applies intensity and duration thresholds: an ear Action Unit (AU) is coded only when the motion exceeds documented criteria. Subthreshold motion — "very slight" or "extremely slight" twitches — is correctly labeled BACKGROUND under EquiFACS even when visible.

The protocol above is **more permissive on subthreshold motion**. Empirically (Lesson 17 audit), this produces 12.4 % verdict disagreement with the published RME labels, structured as:

- **24 bg → action flips** (audit calls subthreshold motion ACTION, EquiFACS calls it BACKGROUND): the protocol-vs-EquiFACS gap on the visibility threshold
- **11 action → bg flips** (audit calls head-only or distractor-only motion BACKGROUND, EquiFACS labeled it ACTION): cases where the EquiFACS coder may have included secondary motion or non-ear motion under the AU label

**This protocol is not "more correct" than EquiFACS; it is differently calibrated.** Both are valid binary classifications under different threshold rules. For RHpE deployment, the choice depends on the downstream task:

- **Pre-screening for clinical workflow** ("did anything happen with the ears?") — this protocol is appropriate, since recall on subthreshold motion matters more than precision.
- **EquiFACS-grade ethogram coding** — the EquiFACS protocol is appropriate; this audit protocol over-calls action.

## 9. Limitations

- **Single-annotator origin.** This protocol was extracted from one reviewer's audit of 283 examples. Inter-rater κ unmeasured; single-observer self-consistency is κ = 0.586 (moderate) on borderline cases, ~100 % on confident verdicts (controls).
- **Subthreshold zone is genuinely ambiguous**, even for the same observer across sittings. The 11 borderline verdict-flips on same-day re-watch (Step 3) are concentrated entirely on "extremely slight motion" calls. Multi-rater audit will likely produce systematic disagreement in this zone.
- **Multi-subject FH-only rule was made explicit only after the original audit.** In the original audit, all 7 reviewed multi-horse distractor cases were labeled `BACKGROUND?` — the rule was being applied implicitly. After explicit articulation, all 7 resolved to confident `BACKGROUND` on re-watch. **Take this as evidence that careful annotators arrive at FH-only intuitively but require explicit articulation to be reproducible across raters.**
- **Same-day re-watch** (rather than the spec-recommended ≥ 12 h delay) biases the κ = 0.586 figure toward the high side. Treat as ceiling.
- **Dataset-specific.** Calibration was done against Read My Ears (controlled lab data, 12 sources, single horse per clip in 81 % of cases, multi-horse in 19 %). Field data with different camera angles, lighting, and movement contexts may require additional rule clarifications.

## 10. Recommendations for adoption

1. **Multi-rater κ measurement is non-negotiable** before any RHpE deployment uses this protocol. The empirical evidence from `outputs/loso_label_variant_comparison.md` shows a model-side cost of ~10 pp LOSO AUC when training a probe on single-annotator κ ≈ 0.6 labels. Multi-rater κ ≥ 0.7 is the load-bearing requirement, not nice-to-have.
2. **The 2-axis diagnostic** (per-source consistency × per-source agreement) should be computed at the start of any source-aware evaluation, not as a post-hoc diagnostic. See [Lesson 17](lessons_learned.md) S5 vs S10 example for the operational definition.
3. **Borderline `?` clips should be reported, not removed.** The choice of how to handle them downstream (drop / weight-down / soft-label) is a per-application decision, but the dataset must preserve the borderline annotation as a primary signal of judgment uncertainty.
4. **The protocol can be tightened to match EquiFACS by adding intensity/duration gates** (e.g., motion must persist >200 ms AND span >30° rotation to count as ACTION). The current protocol's permissive nature is a choice, not an oversight.

## 11. Cross-references

- [`docs/audit-followup-spec.md`](audit-followup-spec.md) — gated execution sequence under which this protocol was extracted
- [`docs/lessons_learned.md`](lessons_learned.md) Lesson 17 — full audit findings, dual-label MLLM agreement, S5/S10 calibration-vs-noise
- [`outputs/piotr_audit_labels.jsonl`](../outputs/piotr_audit_labels.jsonl) — 283 audit verdicts with category labels
- [`outputs/consistency_check_results.md`](../outputs/consistency_check_results.md) — Step 3 within-observer self-consistency analysis
- [`outputs/loso_label_variant_comparison.md`](../outputs/loso_label_variant_comparison.md) — Step 5 V-JEPA-2 + LR LOSO under three label variants; the −10 pp retrain-noise / −3.6 pp eval-mismatch decomposition
- [`docs/methodology-note-2026-05.md`](methodology-note-2026-05.md) — Step 7 synthesis (forthcoming)
