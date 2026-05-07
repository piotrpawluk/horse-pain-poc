# Lessons Learned — RHpE Behavior Classification PoC (Phase 0–1.5)

*Working document, written 2026-05-06 after iter 6.5 sanity checks.*

This document records methodological lessons from the iteration sequence, with emphasis on what we learned by being wrong. It is intended both as a working reference for the project and as evidence of rigor for academic outreach (Andersen, Zamansky, UPWr).

The headline: **Iter 6.5 LOSO sanity check disproved the head_position MVP** that iter 5–6 built up. Without it, Phase 2 would have started with 3000–5000 PLN and 40h of recording on top of session leakage, not behavior detection. The cost of rigor was ~3h of additional compute; the cost of skipping it would have been six weeks of misallocated work.

---

## Timeline

| Iteration | Question | Outcome |
|---|---|---|
| **0** (2026-05-04) | Does the open-source stack work on M-series? | DLC SuperAnimal-Quadruped GO 4/4 in ~45 min |
| **1** (2026-05-05) | Do we replicate Read My Ears 0.875 ear_movement? | ✓ V-JEPA-2 + linear probe = 0.854 (paper 0.875, Δ −2.1pp) |
| **1.5 iter 2** | Does zero-shot V-JEPA-2 LOO cosine detect 5 RHpE behaviors on 53 DIY clips? | overall 0.358; eye_expression 1.000 (later identified as Padma sink-effect) |
| **iter 3** | Does head-crop YOLO ROI rescue ear_position? | NO — 0/17 ear (vs 0/17 full-frame); head-crop ROI insufficient |
| **iter 4** | Does LOO cosine on 283 RME clips show signal? | LOO cosine k=1: 0.756; linear probe 0.894 (beats paper) |
| **iter 5** | Does linear probe rescue 5-class on 53 clips? | head_position AUC 0.927 (binary OvR), rest < chance |
| **iter 6** | Is V-JEPA-2 SSv2 motion-biased? Matrix 4 backbones × 4 behaviors | motion-domination hypothesis REJECTED; head_position 0.898 looks like real signal |
| **iter 6.5** | Sanity checks: weights, bg-leakage, session leakage | **head_position INVALIDATED**; the entire 53-clip dataset is session-confounded |

Files: `outputs/few_shot_validation_results.json`, `outputs/readmyears_loo_baseline_results.json`, `outputs/few_shot_validation_results_iter5_linearprobe.json`, `outputs/backbone_matrix_iter6_results.json`, `outputs/iter65_sanity*.json`.

---

## Lesson 1 — LOSO, not LOO, for behavior classifiers on small N

The discovery: **head_position LOO AUC = 0.898 vs LOSO AUC = 0.561** (Δ −34pp). On the same 12 clips, leave-one-out makes the model look ridge-strong; leave-one-session-out reveals it learned to recognize the recording session, not the behavior.

In hindsight, the structure of the data made this nearly inevitable:

| behavior | total clips | unique sessions | per-session distribution |
|---|---|---|---|
| head_position | 12 | 4 | personal_head=6, padma_eg5yUaP9APA=3, padma_PpEMHTQCpuA=2, padma_0mnugK=1 |
| tail_movement | 9 | 3 | personal_tail=6, padma_eg5yUaP9APA=2, padma_hrZgtrqbMVI=1 |

When the positive class is concentrated in 1–2 sessions, the classifier doesn't need to learn the behavior — it learns the session. LOO cross-validation cannot detect this because each session is split across folds.

**Rule for the rest of the project:** for any per-behavior classifier with ≤20 clips/class:
- Group by recording session (source video, ideally also by horse)
- Report LOSO/LOGO (leave-one-group-out) primarily; LOO only as auxiliary
- Require ≥5 unique sessions per class for LOSO to produce a meaningful confidence interval
- Permutation test on top of LOSO, not on top of LOO

This generalizes to a methodological commitment: in this domain (small-N, multi-source horse video), **LOO is not a safe baseline**. It can hide a 30+ pp leakage drop.

---

## Lesson 2 — Session leakage is the rule, not the exception

After Sanity 3 caught head_position, Sanity 3.5 (LOSO on the other 4 anchor behaviors) showed 4 of 5 behaviors had session leakage:

| behavior | LOO | LOSO | Δ | verdict |
|---|---|---|---|---|
| ear_position | 0.237 | 0.494 | +0.257 | LEAKAGE (LOO actively below chance) |
| head_position | 0.898 | 0.561 | −0.337 | LEAKAGE |
| mouth_open | 0.740 | 0.513 | −0.227 | LEAKAGE |
| **tail_movement** | 0.747 | **0.253** | **−0.495** | **CATASTROPHIC** |
| eye_expression | 0.394 | 0.773 | +0.379 | partial (but all 4 sessions are Padma, so cross-source not tested) |

`tail_movement` LOSO 0.253 — *worse than chance* — means the classifier learned an anti-correlation across sessions: whatever feature personal_tail clips share, the other sessions have its opposite. That happens when the "tail behavior" label is perfectly aligned with "filmed on iPhone in Lesznowola" vs "filmed in Padma documentary studio."

The eye_expression case is more subtle: LOSO *higher* than LOO. That means LOO was suppressing the signal because Padma sources differ visually from each other; LOSO held one Padma-style session out and the others taught the classifier "Padma vs not-Padma" as if it were the behavior. With all four positive-class sessions being Padma, even LOSO can't break this confound — we'd need cross-source negatives that look like Padma but aren't labeled positive.

**Implication:** the 53-clip anchor dataset that was supposed to be a quick few-shot validation is fundamentally **not usable as training data** for any per-behavior classifier. It can serve as held-out negative examples, or as visual sanity checks in a Gradio UI, but not as the basis for accuracy claims.

---

## Lesson 3 — Hard decision thresholds, not soft narratives

We adopted, before iter 6, a 4-level decision rule for backbone choice:
- **L1**: V-JEPA-2 PT wins everywhere within ≤2pp → unify on PT
- **L2**: Concat ≥ best single backbone on all 4 behaviors → unify on concat
- **L3**: Split >5pp both ways (DINOv2 dominates static, V-JEPA-2 dominates motion) → hybrid
- **L4**: All <0.7 AUC → backbone is not the problem; data/ROI/labels is

In practice the matrix came out AMBIGUOUS — concat lost on 2 of 4 behaviors, DINOv2 led head_position by 3pp (not 5), and SSv2/PT were identical (which itself turned out to be a Sanity 1 finding, see below). Without the pre-committed thresholds we would have written a story justifying whichever choice we had already invested in.

The point isn't that the thresholds are exactly right. The point is that **they were chosen before the experiment** and then we held to them. Half of the value of the rule is removing the option to rationalize after the fact.

This generalizes: pre-commit decision criteria, especially for backbone and architecture choices, before running the experiment that generates the comparison numbers.

---

## Lesson 4 — V-JEPA-2 architecture clarification

When iter 6 showed `vjepa2-vitl-fpc16-256-ssv2` and `vjepa2-vitl-fpc64-256` giving identical AUCs to three decimal places across 8 cells, that was a red flag, not a result.

Sanity 1 confirmed: **all 587 encoder layers are byte-identical** between the SSv2 fine-tuned and pretrain-only checkpoints. SSv2 fine-tuning adds a pooler + classifier head on top, but does not modify the encoder weights. The HuggingFace `VJEPA2Model` class loads only the encoder, dropping the head with "UNEXPECTED" warnings.

The implication that matters more than the pretrain-only debate: **whichever V-JEPA-2 ViT-L checkpoint we load via `VJEPA2Model`, we get the same pretrain-only encoder features.** There is no SSv2 motion-bias contamination to worry about, because we never see the SSv2-fine-tuned components.

This invalidates ~half of the iter 6 narrative ("are we using motion-biased features?") — but in a comforting direction. We've been using pretrain-only encoder features the whole time. The "PT vs SSv2" comparison is structurally untestable in our pipeline because they share the encoder.

The lesson generalizes: when comparing model variants from a foundation model family, verify *which weights actually load* via state_dict diff before drawing architectural conclusions. Naming conventions in HF model IDs are not contracts about pre-training/fine-tuning state.

---

## Lesson 5 — Static-frame collapse test as a feature-grounding diagnostic

For each backbone × behavior, we ran the same eval twice: once with 16 frames sampled across the clip (motion mode), once with the middle frame replicated 16× (static mode). The motion–static delta tells you whether the embedding is actually using temporal information or just snapshotting a frame.

| backbone | ear_movement Δ (motion − static) | interpretation |
|---|---|---|
| V-JEPA-2 SSv2 | +0.039 | mostly static features, ~4pp temporal contribution |
| V-JEPA-2 PT | +0.039 | (same encoder as SSv2) |
| DINOv2 | −0.013 | image-only, motion irrelevant by design |
| Concat | +0.065 | small temporal advantage |

Across 16 cells, no Δ exceeded +0.11. **The "V-JEPA-2 is motion-biased" hypothesis is rejected by direct measurement** — even on the gold-standard motion task (ear_movement), the classifier loses only ~4pp by going to a single frame. Most of the signal is static, with motion as a small additive bonus.

Operationally, this opens a use-case: if 1 frame at full resolution gives ~93% of the AUC of 16 frames, then production inference can run on photos, not video. That changes the form-factor of any future product (mobile app working on snapshots, not just clips).

But it raises a methodological flag too: if the classifier for "ear_movement" works on a single frame, it isn't really detecting movement. It's detecting something visually correlated with the label — possibly post-movement ear pose, possibly background context. Sanity 2 (background-masked re-run, AUC 0.911) ruled out the background hypothesis: the signal lives in the ear region. But the classifier's underlying competence is "recognize ears in a particular configuration," not "recognize that the ears moved."

Generalization: when a classifier trained on video gets within 5pp of itself on a single replicated frame, the classifier is not using temporal information meaningfully. Decide whether that is acceptable for your use-case (yes for many static behaviors; no if you actually need movement detection).

---

## Lesson 6 — Linear probe vs cosine similarity gap (~14pp on clean data)

In iter 4, on the 283-clip Read My Ears dataset:
- LOO cosine k=1: 0.756
- LOO cosine k=5: 0.710
- 5-fold linear probe (LogReg): **0.894**

This 14pp gap quantifies how much signal cosine similarity in raw embedding space leaves on the table compared to a learned linear projection, even when both methods see the same features. For small-N few-shot evaluation, naive cosine similarity is a *lower bound* on what the embedding contains, not an upper bound. Linear probe is the right metric for "does this embedding carry the signal."

Practical consequence: in iter 2 we read the LOO cosine 5-class accuracy of 0.358 as "embedding doesn't separate behaviors." A more honest reading was "cosine similarity doesn't separate behaviors; we don't yet know about linear probe." Iter 5's binary OvR linear probe revealed structure that iter 2 had hidden — and iter 6.5's LOSO subsequently revealed that the structure was session leakage, not behavior. But we needed both layers (probe + LOSO) to get to the truth.

---

## Lesson 7 — Sanity-check budget is insurance, not waste

Iter 6.5 cost ~3 hours of compute (Sanity 1 model verification: 10 min; Sanity 2 bg-masked: 15 min; Sanity 3 LOSO head: 30s; Sanity 3.5 LOSO all behaviors: 30s; Sanity 4 DINOv2: ~15 min). It rejected the head_position MVP that we were ready to scale to 30–50 clips, hire a certified RHpE assessor (1500–3000 PLN), and spend 4–8 weeks recording.

Cost of skipping iter 6.5 = time-discounted equivalent of ~3000 PLN + ~40 working hours + the strategic damage of presenting a 0.898 AUC to academics that LOSO would later demolish.

The rule that emerges is not "always run sanity checks" — it is "don't scale past validation thresholds without sanity checks that match the scale." A 53-clip anchor dataset doesn't need LOSO if the only claim is "first signs of signal." A scaled MVP with budgeted assessor time and a recording protocol absolutely does.

---

## Lesson 8 — Sample size requires session count, not clip count

In iter 6, we planned Track B MVP as "30–50 clips per behavior." Iter 6.5 LOSO revealed why that's the wrong unit:

> If you have 30 clips spread across 5 sessions (1 horse / 1 location / 1 lighting setup per session), LOSO becomes 5-fold with 24-train/6-test, and per-fold variance dominates the AUC estimate. You cannot tell signal from noise with 5 sessions.

The actual requirement for a meaningful LOSO confidence interval with binary classes is roughly:
- ≥10 unique sessions
- balanced labels within each session (so leave-one-session-out actually leaves out both classes)
- ≥3 clips per class per session for stable per-fold estimates

This pushes the realistic Track B MVP collection to **≥10 horses × 2–3 behavior states × 2–3 takes per state ≈ 60–100 clips** at 4–6 weeks of recording across multiple locations, not 30 clips at one venue. The corresponding budget for assessor scoring is more like 3000–5000 PLN than 1500–3000 PLN, since assessors typically charge per clip.

This is not an argument against Track B. It is a recalibration of the cost.

---

---

## Lesson 9 — Background masking is conditional preprocessing, not a global default

Sanity 5 LOSO on Read My Ears 283 clips initially looked like a defeat for background masking: average LOSO across sources dropped from 0.875 (V-JEPA-2 unmasked) to 0.764 (V-JEPA-2 with YOLO ear-bbox masking). Initial interpretation in lessons-learned drafts was "bg-masking doesn't help cross-source."

The deep-dive (`outputs/iter65_sanity5_deepdive_results.json`) reveals this was a misread of a heterogeneous result. Per-source LOSO behavior of bg-masking:

| source | unmasked AUC | bg-masked AUC | Δ |
|---|---|---|---|
| S8 | 0.633 | **0.875** | **+0.242** (rescue) |
| S9 | 0.783 | **0.888** | **+0.105** (rescue) |
| S1 | 0.816 | 0.837 | ≈ neutral |
| S5 | 0.903 | 0.916 | ≈ neutral |
| S2 | 0.927 | 0.893 | −0.034 |
| S6 | 0.956 | 0.922 | −0.034 |
| S3 | 0.995 | 0.885 | **−0.110** (hurt) |
| S12 | 1.000 | 0.661 | **−0.339** (hurt severely) |

The pattern is consistent: **sources where unmasked V-JEPA-2 is weak (presumably because of background confound) gain from masking; sources where unmasked is already strong lose from masking.** Averaging these into a single "bg-masked LOSO 0.764" hid the structure.

Visual inspection of S8 confirms the rescue mechanism. S8 frames show **two horses in the same shot** — a bay horse in the foreground plus a darker horse partially visible behind stall bars. The second horse provides an independent motion signal; under unmasked V-JEPA-2 the model picks it up and over-predicts "action" on background clips (confusion matrix: FP=6/8 background→action, FN=4/16 action→background). YOLO ear-bbox masking crops the second horse out of the field of view, eliminating the confound, AUC rises 24 points.

**Production implication.** Background masking should be applied **conditionally**, not as a global preprocessing step:

```python
if yolo_horse_detector(frame).count_instances() > 1:
    use bg_masked_features
elif yolo_human_detector(frame).count_instances() > 0:
    use bg_masked_features
else:
    use unmasked_features  # cheaper and slightly better when scene is clean
```

The condition extends beyond "two horses" — any independently moving entity in frame (handler walking, swinging objects, second horse) is a candidate confound that masking can mitigate.

**When NOT to use bg-masking.** When the scene is clean (single horse in frame, stationary handler, stable background), unmasked V-JEPA-2 features are slightly stronger AND cheaper to compute. In our Read My Ears matrix this manifested as bg-masking hurting the strong-unmasked sources S12 (1.000 → 0.661, −34 pp) and S3 (0.995 → 0.885, −11 pp). The mechanism is plausibly that masking removes peripheral context (horse body posture, background lighting cues) that V-JEPA-2 was using as weak-but-helpful auxiliary signal. So bg-masking is a fix for a specific failure mode, not a free upgrade.

**Methodological framing.** This is "context-aware preprocessing for cross-subject equine behavior recognition" — the kind of methodological note that fits between "engineering detail" and "publishable finding". Worth flagging in academic outreach.

---

## Lesson 10 — Two failure modes in cross-source ear movement detection

Sanity 5 deep-dive identified two distinct mechanisms behind the three weakest LOSO folds (S8 0.633, S9 0.783, S1 0.816). They have different signatures and call for different fixes.

**Failure mode 1: scene motion confound (S8).**
- Confusion matrix asymmetric: 6/8 background clips classified as action (model sees motion that isn't ear movement)
- Visual cause: secondary horse partially visible in frame
- Bg-masked rescue: +24 pp (cropping out the second horse fixes it)
- Implication for protocol: **single subject in frame** is a hard requirement

**Failure mode 2: subtle ear movement in instrumented context (S9).**
- Confusion matrix asymmetric the other way: 3/13 action clips classified as background (model misses real ear movement)
- Visual cause: horse fitted with heart rate monitor harness and ECG electrode patches, recorded in controlled medical setup against plain wall, restrained posture
- Bg-masked rescue: +11 pp (helps but doesn't fully fix — the signal is genuinely subtle)
- Implication for protocol: **diverse recording contexts** in training distribution. If the model has only seen casual stable footage, it will struggle on instrumented contexts even without explicit confound. Worth including a few medical-instrumented clips in training data if the deployed model is expected to see them.

These two case studies are the strongest substantive findings from Sanity 5, more useful for academic outreach than the headline LOSO=0.875 number itself.

---

## Lesson 11 — Realistic LOSO target on Polish wild data is 0.70–0.80, not 0.85+

Read My Ears LOSO 0.875 was achieved on a controlled lab study: 12 horses, single research setup, presumably consistent camera/lighting/staff/distance/breed mix. Even within that controlled context, 3 of 12 sources fell below 0.85 (S8 0.633, S9 0.783, S1 0.816) and the underlying mechanisms turned out to be specific recording-quality issues rather than fundamental model limits.

The diverse field dataset planned for Phase 2 (HKiJ peer network and beyond) will have substantially more axes of variance:
- 15–25 different stables (vs 1 research site)
- Many cameras, many phone models, no calibration
- Multiple riders, multiple trainers, multiple disciplines (recreation, sport, hippotherapy)
- Year-round seasonal variation in lighting
- Wide breed/conformation/age range

Each axis of variance is a potential session leakage vector. Sue Dyson's own publications report between-rater κ for RHpE in field conditions of 0.5–0.7 — humans don't agree as consistently in the wild as they do in controlled studies, so a model trained on field data has a lower ceiling than a model trained on controlled data.

**Realistic target: LOSO 0.70–0.80 with 0.85 as stretch goal under hard quality gates.** Not 0.85–0.90. Communicating 0.70 as a success criterion is uncomfortable but honest; setting an unmeetable 0.90 target invites cherry-picking later.

---

## Lesson 12 — V-JEPA-2 SSv2 fine-tune does not modify encoder weights (Sanity 1)

Sanity 1 verified that `facebook/vjepa2-vitl-fpc16-256-ssv2` and `facebook/vjepa2-vitl-fpc64-256` share **byte-identical encoder weights across all 587 layers**. The SSv2 fine-tune adds a pooler + classifier head; the `VJEPA2Model` class loads only the encoder, dropping the head. So whichever V-JEPA-2 ViT-L checkpoint we instantiate, we get the same pretrain-only encoder features.

Two consequences flow from this that should be stated explicitly wherever results from both checkpoints are reported:

1. **Identical AUCs across "SSv2" and "PT" rows in our backbone matrix are mechanically forced, not independent confirmation.** Sanity 5 reports SSv2 LOSO = PT LOSO = 0.875 to three decimal places across 12 sources × 2 modes; this is a consistency check that the data pipeline is not broken, not an independent validation of source-invariance.

2. **The "SSv2 fine-tune introduces motion bias" hypothesis from earlier iterations is structurally inapplicable to our pipeline** — we never see the SSv2-fine-tuned components. We have been using pretrain-only encoder features the entire time.

When future readers (or paper reviewers) see two "different" backbones giving identical results, they should be told why. Otherwise it looks like a miracle and invites suspicion that some other bug is duplicating embeddings.

---

## Lesson 13 — RHpE behavior taxonomy by detection difficulty

Sources: Dyson 2018 (canonical RHpE paper, 24 behaviors); *Train with Trust Project* mobile field guide (2023, operational definitions with quantitative thresholds).

This is a **theoretical mapping** of the 24 RHpE behaviors to the detection-pipeline patterns we have at hand or could plausibly build. It is *not* a coverage promise — "fits a pipeline pattern" does not mean "will produce LOSO ≥ 0.70 on field data". Iter-6.5 head_position (LOO 0.898 → LOSO 0.561 = session leakage) is the cautionary anchor: a Class A behavior in the taxonomy below failed catastrophically on real data.

**Four classes, ordered by tractability:**

**Class A — ROI + V-JEPA-2 + linear probe pattern (10 behaviors).** The Read My Ears pipeline shape. Each is a static visual classification or a sustained-state recognition task on a behavior-specific ROI.
- Ears Back (≥5s); Eyes Closed (2–5s); White of the Eye (repeated exposure); Intense Stare (≥5s, glazed); Mouth Open/Close (≥10s); Tongue Out; Head Tilt; Above Vertical (>30°, ≥10s); Behind Vertical (>10°, ≥10s); Tail Position (crooked/clamped).
- Implementation footnote: some "Class A" items (Above Vertical, Head Tilt, sustained-duration thresholds like Eyes Closed ≥2s) are arguably better served by DLC pose estimation + temporal aggregation than by clip-level V-JEPA-2. They land in Class A here because they share the **per-behavior ROI** structural pattern, but the right backbone for some of them may be Track C, not Track B.

**Class B — DLC keypoints + temporal/frequency analysis (8 behaviors).** Movement and gait dynamics; require pose tracking through time.
- Bit Pulled Through; Head Up/Down (repeated, off-rhythm); Head Side to Side (tossing, repeated); Tail Swishing (repeated frequency); Rushed Gait (>40 trot steps / 15s); Slowed Gait (<35 trot steps / 15s); Spontaneous Change of Pace; Stumble / Trip / Toe Drag.
- The two gait-frequency behaviors (Rushed / Slowed Gait) are operationally **deterministic** under the TWTP definition: count peaks on hoof-Y time series within a 15s window, threshold. No foundation model required, no per-clip assessor labeling required (ground truth is the count). This is a credible Track D candidate (see "Future work" below).

**Class C — Multi-modal: video + rider context + audio (4 behaviors).** Each requires information that pure horse-video doesn't contain.
- Moving on 3 Tracks (needs reference frame for hindlimb expected position); Canter Dysfunction (sophisticated gait classifier — correct/incorrect lead, disunited / cross-canter); Spooking ("against the rider's cues" — needs rider-intent tracking to distinguish requested direction change from spook); Resistant ("reluctant to go forward; needing repeated physical or verbal encouragement" — needs rider leg/hand cue tracking + audio).

**Class D — Rare events with sparse ground truth (2 behaviors).** Visually obvious when they occur, hard to collect.
- Rearing; Bucking. Detection (motion magnitude threshold) is trivial; *event detection* problem with a strong base-rate issue (1–2 episodes per 100h of naturalistic recording). Ground-truth collection is the binding constraint, not the model.

**Strategic implication.** The 8/24 pain-inference threshold is theoretically reachable using only Class A + B behaviors (18 of 24). A horse manifesting pain *primarily* through Class C behaviors (rider-dependent) would be under-scored by a video-only system — that's the structural ceiling, not an engineering deficit. Worth surfacing in any clinical-validation discussion.

**What this taxonomy is and is not.** It is a planning artifact for prioritizing Phase 2/3 scope. It is not a claim that 18 behaviors are *achievable* — that would require per-behavior empirical validation analogous to Sanity 5 for ear movement, on diverse field data, with certified RHpE assessor adjudication. The honest one-line summary: **"10 + 8 = 18 of 24 behaviors *plausibly* extend the same pipeline patterns; the remaining 6 require capabilities we don't have."**

---

## Lesson 14 — Perception/classification decoupling on Gemini 2.5 Pro and 3.1 Pro Preview (May 2026)

**Scope first.** This is a *supporting methodological observation*, not the project thesis. The thesis remains V-JEPA-2 LOSO 0.875 source-aware replication (see Lessons 1, 5, 9, 10, 12 + Sanity 5). This lesson exists because we tested whether a frontier multimodal LLM could serve as an off-the-shelf label-noise auditor on the 283-clip Read My Ears dataset (CC-BY-4.0, anonymized) and found the failure mode interesting enough to document. It does not generalize beyond what was tested.

**Tested configurations** — Gemini 2.5 Pro and Gemini 3.1 Pro Preview (Jan 2026 model card), accessed via Google AI Studio API in May 2026. Single API key, all calls between 2026-05-06 and 2026-05-07. fps=10 video sampling on 36 stratified Read My Ears clips (3 per source, S1–S12, 14 action / 22 background). Three prompts (A: generic; B: EquiFACS-coder anchored; C: Gemini-3.x best-practice with system instruction, evidence-grounded user prompt, no negative constraints). Probe = description-only prompt at temp=1.0 with no classification request. Code in `tools/gemini_audit.py`; raw JSONLs in `outputs/gemini_audit_*`.

**Three findings, in order of strength.**

### 1. Cross-rep instability is structural, not configurational

Across 5 reps per clip on the description-only probe at Google's officially recommended Gemini 3.x parameters (temperature=1.0, thinking_level=low):

- Gemini 3.1 Pro Preview, **N=20** clips (10 from initial probe + 10 N-expansion at fresh seed): **0 / 20 (0%)** clips show consistent motion/still classification across 5 reps.
- Gemini 2.5 Pro, **N=10** clips at temp=1.0 (thinking_level not supported on 2.5 Pro — it returns 400 INVALID_ARGUMENT): **2 / 10 (20%)** clips consistent.

The instability is not improved by following Google's prompt-engineering guidance. The 80% flip rate from the original temp=0.5 probe was not a temperature artifact — corrected parameters give 0% consistency on N=20.

### 2. Perception/classification decoupling — generation-general

Same model, same clips, same parameters. Compare what the description-only probe reports against what the classification-mode prompt commits to:

| Configuration | Probe motion-attribution rate (mean across reps) | Decoupled clips (probe sees motion ≥3/5 reps **and** classifier outputs "background") |
| --- | --- | --- |
| 3.1 Pro Preview + Prompt C @ corrected params | 55% | **13 / 20 (65%)** |
| 2.5 Pro + Prompt B + temp=1.0 probe | 66% | **7 / 10 (70%)** |
| 2.5 Pro + Prompt A + temp=1.0 probe | 66% | 0 / 10 — Prompt A *over*-detects, no decoupling shows |

Whenever the classification surface carries a refusal bias (Prompt B on 2.5, Prompt C on 3.1 — and the model's structural conservatism on 3.1 makes Prompts A and B also degenerate-toward-background there), the same model under the same conditions reports motion in the majority of description-only reps but commits to "background" on most of those clips when classification is requested.

**Mechanism (informed conjecture, not measured).** Post-training likely selects harder for conservative classification commitments than for conservative descriptive language. The classifier output is a different surface from the description output, and the optimization pressure is distributed differently across the two. This is consistent with how alignment training is typically structured (penalize confidently wrong classifications more than penalize confident descriptive language) but we have not verified the mechanism inside the model.

### 3. Conservatism on 3.1 Pro Preview is structural, not parameter-driven

| 3.1 Pro Preview configuration | "action" predicted / 36 |
| --- | --- |
| Prompt A, temp=0, default thinking | 0 |
| Prompt B, temp=0, default thinking | 1 |
| Prompt C (best-practice), temp=1.0, thinking=low, system-instruction grounded | 1 |

Google's recommended parameters + a 3.x-tailored prompt + system instruction → 3.1 Pro Preview classifies 35/36 clips as background regardless of true label. This is not a parameter-tuning issue. Following the documented prompt-engineering guidance does not unlock a calibrated classifier on this task on this model.

**Operational implication.** Off-the-shelf Gemini 2.5 Pro and Gemini 3.1 Pro Preview, accessed via AI Studio API at the parameters tested in May 2026, are not reliable label-noise auditors for fine-grained equine ear-movement classification on Read My Ears clips. We do not extrapolate this claim to other multimodal LLMs (Claude, GPT-5/4, open-source) or to other behavior categories without testing. Audit-style use cases on this model class require either ensemble + voting at substantially larger N (untested here) or a different approach entirely.

**What this lesson is not.** It is not evidence that frontier multimodal LLMs are universally unreliable. It is not a claim about Claude / GPT-class models. It is not the methodology paper. It is a documented, scoped, dated observation that the V-JEPA-2 + linear probe pipeline remains the right backbone for this task because the alternative we tested doesn't work in this regime.

**Forward note (added 2026-05-07).** The Lesson 14 framing of "refusal-bias collapse" describes one of two prompt-conditional MLLM modes. The audit-followup work in [Lesson 17](#lesson-17--single-reviewer-audit-of-rme-labels-may-2026-full-283-clip-dataset-consistency-check-pending) shows the same model family operates at a *non-collapsed*, audit-grade-matching threshold under Gemini 2.5 Pro + Prompt A specifically — the only configuration in the test matrix (1 of 7) that did not collapse. The unified finding is "MLLM ear-motion threshold is prompt-conditional, asymmetrically distributed across configurations, and neither mode matches EquiFACS." Lesson 14's failure-mode catalog still describes real Gemini-3.x and Qwen behavior; what's revised is the universality claim.

## Lesson 15 — Three failure modes reproduce on open-weight Qwen2.5-VL-7B *after* the video pipeline was repaired (May 2026)

**Scope.** Continuation of Lesson 14, same supporting-observation status. The original Lesson 15 (PR #2 v1) was based on Qwen results in which the model never received video — `apply_chat_template` was called on a structured-messages list, which strips multimodal content via `extract_text_from_content` before any video tokens can be inserted (see Lesson 16 for the bug). The 36/36-background-collapse claim in v1 was text-only inference disguised as video inference. This lesson is **rewritten on the post-fix v2 results**, where prompt-token counts confirm the model is genuinely processing video frames (~5500 tokens vs the v1 text-only ~110). Branch `fix/qwen-mlx-video-input`. Spec: `docs/qwen-fix-and-revalidate-spec.md`. Side-by-side numbers in `outputs/qwen_vs_gemini_comparison.md`.

**Configuration tested.** `mlx-community/Qwen2.5-VL-7B-Instruct-bf16` via mlx-vlm 0.5.0 on M2 Max (96 GB), 2026-05-07. Same 36-clip stratified subset as the Gemini runs (clip paths reconstructed from existing Gemini JSONLs — no re-stratification). Two classification probes (Prompt A at temp=0; Prompt C with system instruction at Qwen's official `temperature=1e-6, repetition_penalty=1.05`, system instruction inlined into the user-text string per the mlx-vlm 0.5.0 list+video routing bug) and one description-only probe (5 reps × 10 clips at temp=0.7). 0/122 errors. v2 wall-clock ≈ 57 min (~5× v1's 10 min — the v1 timing was a tell, in retrospect: video processing should not be 3 s/clip on 14 GB of weights).

**Result against pre-registered §4 outcome table (re-evaluated on v2).** Row 1: all three failure modes still reproduced — but the supporting evidence is now real model behavior, not a tokenization artifact.

| Failure mode | Gemini 3.1 Pro Preview | **Qwen 7B-bf16 v2 (post-fix)** |
|---|---|---|
| Refusal-bias collapse (bg-prediction rate, normalized) | 35/36 = 97 % (prompt C) | **34/36 = 94 %** (prompt A) · **35/36 = 97 %** (prompt C) |
| Cross-rep instability (5-rep stable / 10) | 1/10 (initial), 2/20 (N-expansion) | **0/10** |
| Perception/classification decoupling | 13/20 = 65 % | **2/10 = 20 %** (lower bound; conservative keyword classifier) |
| Agreement with truth (normalized) | 23/36 = 0.639 (prompt C) | **22/36 = 0.611** (prompt A) · **21/36 = 0.583** (prompt C) |

**What v2 added vs v1.** Three substantive changes:
1. **Vocabulary leakage on prompt A.** Qwen v2 sometimes outputs `still` (echoing the prompt's `STILL` wording in the user text) instead of the schema's `background`. They are semantically identical in context, but the deviation is auditable: `qwen_label` field is preserved raw in the JSONL; downstream agreement comparison applies a `still → background` normalization rule. Prompt C's strict 2-class enum suppresses this leakage.
2. **One true-action correctly classified.** `action_S3.mp4_2_.mp4` flips from v1's templated `background` to v2 prompt A's `action` — Qwen has *some* clip-conditioned perception; v1 had none. Under prompt C the same clip is still missed.
3. **One false-positive emerges.** `background_S4.mp4_7_.mp4` flips from v1's templated `background` to v2's `action` (both prompts) — v1 was structurally incapable of false positives because everything collapsed.

**Reframing the S4_7 "false positive" — target-confusion, not hallucination (added 2026-05-07 after manual review).** Independent manual review of all 22 background clips by the project owner (single reviewer; see [Lesson 17](#lesson-17--independent-manual-review-of-rme-background-labels-stub) for scope) found that `background_S4.mp4_7_.mp4` contains *real ear motion on a non-target horse in the background of the frame* — labeled "horse in background twitched ears" by the reviewer. Qwen v2 reasoning under both prompts matches: prompt A *"The horse's ears are moving slightly, indicating active movement rather than being still"*, prompt C *"The horse's ears are slightly rotated and twitching, indicating movement."* The model is detecting real motion; it cannot disambiguate the target horse from the distractor. **This is the same multi-horse confound failure mode [Lesson 9](#lesson-9--background-masking-is-conditional-preprocessing-not-a-global-default) documents on V-JEPA-2** — independently verified from a different angle. The v2 prompt-A bg-rate of 94 % therefore reads slightly differently: 1 of the 2 "false-positive action" calls is structurally a target-confusion failure, not a calibration artifact. S8 was the canonical multi-horse-confound source in Lesson 9 (V-JEPA-2 LOSO 0.633 → 0.875 with bg-masking); S4 is now confirmed to share the same problem from manual review (4 of the reviewer's 21 inspected bg clips are multi-horse confounds, all in S4 + S8). The §4 row 1 verdict is unchanged because the comparison protocol used labels-as-given, but the qualitative read of v2's behavior shifts: Qwen v2 has more clip-conditioned perception than the bg-rate alone suggests; the bottleneck is target-binding under multi-subject scenes, not refusal-bias collapse, on these specific clips.

**Per-source pattern unchanged.** v1 → v2 differs by exactly ±1 on 2 of 12 sources (S3 +1 in A, S4 −1 in both). Ten of twelve sources have identical v1 and v2 agreement counts. This is the structural-convergence point: when the dominant mode is "lean to bg-majority" — whether from no-input artifact (v1) or genuine model conservatism (v2) — per-source numbers converge because they are determined by the dataset's per-source bg/action distribution, not by perception.

**Reasoning text on prompt C remains near-templated.** Even with video reaching the model, Qwen v2 + C produces evidence sentences with only minor word-swap variation across clips (*"…remain in a consistent position throughout the frames, with no visible movement or change in their orientation"* with `frames`/`clip` and `movement`/`positional changes` shuffled). Confidence dropped slightly from v1's locked 0.95 to v2's locked 0.9. The system instruction's evidence-citation request is being satisfied syntactically rather than semantically, regardless of whether the model has the video. Compare to Gemini 3.1 + C, which produces clip-specific evidence and catches `action_S3.mp4_2_.mp4` correctly while Qwen v2 + C misses it.

**Generalization, scoped (rewritten).** The v1 generalization claim ("Lesson 14 three-failure-modes generalize from Gemini family to MLLM class") was structurally correct *by accident*: text-only inference happened to produce identical aggregate metrics to genuine-but-conservative inference, because both lean to `background` on a dataset where 22/36 are true-bg. The v2 evidence makes the same claim *correctly*: with video properly routed, Qwen 7B exhibits genuine refusal-bias collapse, genuine cross-rep instability, and genuine decoupling — matching Gemini 3.1 Pro Preview's pattern but with quantitative differences (94 % vs 100 % bg-rate on prompt A; 1/14 vs 0/14 true-action catches; 20 % vs 65 % decoupling; near-templated vs varied evidence). Within the scope tested — sub-second ear movement on RHpE, 36-clip stratified subset, May 2026 — the failure modes are not Gemini-family-specific. We do **not** extrapolate to larger Qwen sizes (32B-4bit gated and skipped per spec §4), other open-weight MLLM families, proprietary models we haven't tested (Claude, GPT-5-class), or coarser tasks where MLLMs are documented to perform well.

**Operational implication (unchanged).** The MLLM-as-classifier approach — both proprietary and open-weight, in the regimes we tested — does not produce a usable label-noise auditor on this task. **V-JEPA-2 + linear probe (LOSO 0.875) remains the spine of the pipeline.** The Qwen branch is a closed track, not a parked one.

**What this lesson is not.** Not a benchmark; N=36 is convenience-sized for tool selection. Not a publication-grade evaluation. Not a claim that all open-weight MLLMs collapse on RHpE — only Qwen2.5-VL-7B at bf16 has been tested. The bug we found in v1 is a reminder that aggregate metrics on an MLLM pipeline cannot substitute for direct verification that multimodal tokens are reaching the model.

## Lesson 16 — Verify multimodal tokens reach the model before drawing any conclusions

**Discovered via PR #2 → fix/qwen-mlx-video-input, May 2026.** This lesson exists because Lesson 15's first draft was confidently wrong. The bug took ~10 minutes to find via prompt-token inspection and ~3 hours to fix and re-validate; the bad result took ~10 minutes to produce.

**The bug.** mlx-vlm 0.5.0's `apply_chat_template` has two distinct paths for multimodal content:

| Call shape | What happens to video |
|---|---|
| `prompt=messages_list` with `content=[{"type": "video", ...}, {"type": "text", ...}]` | `extract_text_from_content` keeps only `type="text"` items. Video metadata is silently dropped before `_format_video_message` can run. **No `<\|video_pad\|>` token in the rendered prompt.** Subsequent `generate(..., video=path)` is a no-op for the prompt text — model sees text only. |
| `prompt=user_text_str` with `video=path, fps=10, num_videos=1` as kwargs | Routes through `_format_video_message`, which is gated on `kwargs.get("video")` being truthy. **`<\|vision_start\|><\|video_pad\|><\|vision_end\|>` placeholders inserted.** Generate fills them with real video tokens. |

Empirical render comparison on Qwen2.5-VL-7B (one RME clip, prompt A):

- buggy call: 159 chars, 0× `<|video_pad|>`, prompt_tokens ≈ 111
- correct call: 510 chars, 1× `<|video_pad|>`, prompt_tokens ≈ 5500

The 50× difference in `prompt_tokens` is the smoking-gun signal. Text-only inference on a 14 GB VLM at ~3 s/clip should have been suspicious from the timing alone; in retrospect, video processing on M2 Max consistently runs at ~20 s/clip.

**Related bug, same release.** Passing `prompt=[{"role": "system", "content": ...}, {"role": "user", "content": ...}]` together with `video=path` as a kwarg attaches `<|vision_start|><|video_pad|><|vision_end|>` placeholders to **both** messages, not just the user message. Verified empirically: `video_pad count: 2`. Workaround: inline the system instruction into the user-text string and use the single-string-prompt + video-kwarg path.

**Verification rule (going forward).** Before drawing any conclusion from a multimodal LLM run, verify that multimodal tokens actually reach the model:

1. **Prompt-token sanity.** Print `prompt_tokens` from the generation result. For a video clip at fps=10 of ~1 s on Qwen2.5-VL, expect ~5000+ tokens. If you see ~100, the video is missing.
2. **Rendered-prompt inspection.** Print `formatted_prompt`'s length and whether it contains `<|video_pad|>` (Qwen) or the model's vision-token marker.
3. **Latency sanity.** ~20 s/clip on M2 Max for video; ~3 s/clip means text-only.
4. **Output diversity.** Identical templated reasoning across all clips at locked confidence is a strong tell that the model is not perceiving — variance in evidence prose is the cheapest indicator of clip-conditioning.

The cost of this check is one log line. The cost of skipping it was a published-but-unmerged PR with confident generalization claims based on text-only inference. PRs from MLLM experiments should not merge until at least one of (1)–(3) is in the run log.

**Operational implication.** All Qwen runs on this repo from `fix/qwen-mlx-video-input` forward log `prompt_tokens` and assert >2000 on video probes. The v1 outputs in `outputs/qwen25vl_7b_*.jsonl` (no `_v2` suffix) are kept as diagnostic evidence of the bug, not as model-behavior evidence.

## Lesson 17 — Single-reviewer audit of RME labels (May 2026, full 283-clip dataset; consistency check pending)

> **Status: numbers below are single-reviewer disagreement-with-published-protocol rates, not "true label noise rates."** Within-observer self-consistency check is in flight per `docs/audit-followup-spec.md` Step 3. This lesson will be tightened (or weakened) once the consistency rate lands. **Read every figure here as "user's audit disagrees with EquiFACS-derived RME labels at this rate," not "RME labels are wrong at this rate."** No second observer; κ unmeasured.

**Scope.** The project owner manually reviewed all 283 clips in the Read My Ears dataset on 2026-05-07 against a personally-applied threshold for "is there visible ear motion within the clip." Reviews recorded as `<clip> — <free-text observation> — <VERDICT>` with VERDICT ∈ {ACTION, BACKGROUND, ACTION?, BACKGROUND?}, where `?` denotes "borderline; would defer to a second opinion." Multi-horse scenes were marked with `fh:` (foreground horse) / `bh:` (background horse) per-subject notation. Audit data structured into `outputs/piotr_audit_labels.jsonl` with categorical labels (`strong_motion`, `slight_motion`, `subthreshold`, `head_only`, `body_only`, `multi_horse_target_focus`, `multi_horse_distractor`, `scene_cut`, `error_frame`, `ears_still`).

**Headline.** **35/283 = 12.4 % overall disagreement** with original EquiFACS-derived RME labels. Asymmetric direction: **24 bg → action** (mostly subthreshold motion the reviewer treats as above-threshold) vs **11 action → bg** (mostly head-only / body-only motion or distractor-only motion). 17 disagreements are confidently flipped (no `?`); 18 are borderline.

**Per-source disagreement (heterogeneous):**

| Source | total | agree | disagree | rate | direction |
|---|---|---|---|---|---|
| S6 | 19 | 19 | 0 | **0.0 %** | clean |
| S7 | 21 | 20 | 1 | 4.8 % | 1 act→bg |
| S3 | 28 | 26 | 2 | 7.1 % | mixed |
| S8 | 24 | 22 | 2 | 8.3 % | both bg→act *(multi-horse load — see threshold-bifurcation section below)* |
| S12 | 23 | 21 | 2 | 8.7 % | both bg→act |
| S1 | 21 | 19 | 2 | 9.5 % | mixed |
| S11 | 19 | 17 | 2 | 10.5 % | mixed |
| S9 | 24 | 21 | 3 | 12.5 % | mostly act→bg |
| S4 | 32 | 27 | 5 | 15.6 % | mixed *(multi-horse load)* |
| **S2** | 25 | 20 | 5 | **20.0 %** | mostly bg→act (subthreshold) |
| **S10** | 22 | 17 | 5 | **22.7 %** | **all 5 bg→act (subthreshold)** |
| **S5** | 25 | 19 | 6 | **24.0 %** | mostly bg→act (subthreshold) |

**Multi-horse confound: 55/283 = 19.4 %, all in S4 (31 clips) + S8 (24 clips).** The user's `fh:`/`bh:` notation precisely identifies which subject carries the motion in each clip. This **independently confirms [Lesson 9](#lesson-9--background-masking-is-conditional-preprocessing-not-a-global-default)** from a different angle: Lesson 9 derived multi-horse confound from V-JEPA-2's LOSO collapse pattern on S4 + S8 (S8: 0.633 → 0.875 with bg-masking, +24 pp); the audit derived the same conclusion from per-clip observation. Neither analysis informed the other; convergence on S4 + S8 is independent.

### Dual-label evaluation reveals MLLM threshold bifurcation

The audit data lets us recompute MLLM agreement against a second label set on the 36-clip MLLM evaluation subset (all 36 clips audited; 31 confident verdicts, 5 borderline). Strict variant drops the 5 borderline clips (31 eligible); permissive treats borderline as agreement with RME (all 36 eligible). Computed at Step 1 of `docs/audit-followup-spec.md`.

| Configuration | bg-rate | vs RME | vs Piotr (strict, 31) | vs Piotr (permissive, 36) | Δ permissive − RME |
|---|---|---|---|---|---|
| **Gemini 2.5 Pro · A** | **33 %** *(non-collapsed)* | 22/36 = 0.611 | 20/31 = 0.645 | **25/36 = 0.694** | **+8.3 pp** |
| Gemini 2.5 Pro · B | 100 % | 22/36 = 0.611 | 14/31 = 0.452 | 19/36 = 0.528 | −8.3 pp |
| Gemini 3.1 Pre · A | 100 % | 22/36 = 0.611 | 14/31 = 0.452 | 19/36 = 0.528 | −8.3 pp |
| Gemini 3.1 Pre · B | 97 % | 20/34 = 0.588 | 14/29 = 0.483 | 19/34 = 0.559 | −2.9 pp |
| Gemini 3.1 Pre · C | 97 % | 23/36 = 0.639 | 15/31 = 0.484 | 20/36 = 0.556 | −8.3 pp |
| Qwen 7B v2 · A | 94 % | 22/36 = 0.611 | 16/31 = 0.516 | 21/36 = 0.583 | −2.8 pp |
| Qwen 7B v2 · C | 97 % | 21/36 = 0.583 | 15/31 = 0.484 | 20/36 = 0.556 | −2.7 pp |

**The data describe two distinct MLLM behavior modes, not a single failure mode.** Configurations with bg-rate ≥ 94 % (six of seven tested) collapse to "background everywhere," lose 3–8 pp under stricter audit labels, and fail both label sets simultaneously — they miss every additional action clip the audit identifies. The single configuration with bg-rate 33 % (Gemini 2.5 Pro + Prompt A) over-detects motion broadly, gains 8.3 pp moving from RME to audit labels, and matches the audit threshold approximately. **Neither mode matches EquiFACS-coder labels.** The collapsed mode operates above the EquiFACS threshold; the over-detection mode operates at or below it. The MLLM threshold for "ear motion" is therefore **prompt-conditional, not capability-bound** — the same model family produces opposite failure modes depending on prompt framing, neither of which lands on the EquiFACS protocol.

**The bifurcation is asymmetric, not 50/50.** Out of 7 configurations tested in the §4 matrix (2 model families × 3 prompt variants × 2 model generations on the Gemini side), exactly **1 sits in over-detection mode** (Gemini 2.5 Pro + Prompt A); the other 6 all sit in refusal-collapse mode. The framing is not "MLLMs split into two equal modes by prompt" — it is "the prior-release-Gemini × simplest-prompt corner is the only configuration that did not collapse; everything else, including the same model under stricter prompts and both newer models under any prompt, collapsed." This is consistent with the broader observation that newer-generation Gemini (3.1 Pro Preview) has been calibrated more conservatively on uncertain visual judgments. **Operational implication: model upgrades can silently change behavior in this dimension** — a pipeline calibrated on Gemini 2.5 Pro + Prompt A would behave qualitatively differently after a routine bump to 3.1 Pro Preview, because the threshold mode flips. Anyone building MLLM-based vision pipelines should test both threshold endpoints (over-detection and collapse) explicitly when upgrading model versions.

**Guardrail — 0.694 is calibration, not capability.** The Gemini 2.5 Pro + Prompt A number (0.694 vs Piotr-permissive) is not "Gemini 2.5 + A is the right classifier." It is a model that *disagrees with EquiFACS in the direction the audit also disagrees with EquiFACS* — calibration finding, not capability. A 33 % bg-rate means the model over-calls action on roughly 2/3 of clips. The §4 "materially cleaner" gate (≥ 0.70 agreement AND < 80 % bg-rate) is missed on both label sets. The framing is "different threshold than EquiFACS, closer to audit than EquiFACS coders" — not "useful classifier." A model that says "action" 67 % of the time on a 39 %-action dataset gets credit for matching the audit's stricter labels but is not separating signal from noise; it is biased toward the positive class.

### Three implications

**(a) The iter-6.5 inverse-LOSO Gemini 2.5 Pro fps=10 finding is retroactively explained.** The earlier per-source disagreement pattern on Gemini 2.5+A — under-agreement with RME concentrated on subthreshold-heavy sources S5, S10, and to a lesser extent S3 — was puzzling at the time. Under the audit-derived bifurcation lens, those "errors" are not hallucinations; they are correct perceptions of subthreshold motion that EquiFACS coders correctly excluded by their threshold protocol. The model perceives at one threshold; the labels reflect a stricter threshold. This is a **calibration finding**, not a perception failure, and it makes Gemini 2.5+A's earlier behavior internally coherent.

**(b) Track reframe — MLLM-as-classifier closed for EquiFACS-grade coding; MLLM-as-naive-observer-screener track opens up.** The MLLM-as-classifier track on RME-grade labels remains closed within the scope tested (no configuration hits the §4 ≥ 0.70 / < 80 % gate on either label set). But the bifurcation finding identifies a different applicable task: Gemini 2.5 Pro + Prompt A approximates the threshold a careful but untrained human reviewer would apply. That has real applications in clinical workflows — e.g. **pre-screening clinical clips for "did anything happen with the ears" before sending to expensive expert annotation.** A 67 % positive rate is acceptable for a recall-first pre-screener if downstream expert annotation does the precision work. This is out of scope for the current PoC (which targets EquiFACS-grade RHpE classification, not pre-screening) but flagged as a future direction worth dedicated tooling.

**(c) Lesson 14 needs refinement, not just generalization.** The current Lesson 14 framing of "MLLMs exhibit refusal-bias collapse" describes one of two prompt-conditional MLLM modes — and given the asymmetric distribution observed here, it is the dominant mode in newer-generation models, but it is not the universal mode. The accurate framing: **MLLM ear-motion threshold is prompt-conditional, asymmetrically distributed across configurations, and neither mode matches EquiFACS.** A forward-pointer note is added to the end of Lesson 14 reflecting this; the existing Lesson 14 failure-mode catalog still describes real Gemini-3.x and Qwen behavior accurately — what's revised is the universality claim.

### Implication for V-JEPA-2 LOSO 0.875 (the spine)

If most "subthreshold motion" cases are *correctly* labeled bg per the EquiFACS protocol but contain visible motion the reviewer would call action, then V-JEPA-2 + linear probe is doing something harder than naive labels suggest: discriminating EquiFACS-grade motion from subthreshold/distractor motion on visually similar inputs. The 0.875 result becomes **more impressive under this lens, not less** — the model is learning the duration/intensity threshold implicitly. Under the bifurcation framing, the empirical question Step 2 (B-prime) of `docs/audit-followup-spec.md` answers is **which threshold V-JEPA-2 + LR sits at**.

**Crucially, audit threshold ≠ EquiFACS threshold ≠ "right." Both protocols are valid; they apply different intensity/duration gates.** RHpE coders use ethogram-grade thresholds analogous to EquiFACS, so the project-positive outcome is **V-JEPA-2 sitting at the EquiFACS threshold** (the labels it was trained on), not the audit threshold. The Pattern A/B/C readings:

- **Pattern B (V-JEPA-2 ≈ EquiFACS) — the *good* outcome for the project.** Means V-JEPA-2 + LR successfully learned coder-grade discrimination from coder-grade labels. Direct support for RHpE transfer because RHpE uses ethogram-grade thresholds analogous to EquiFACS. Less surprising as a research finding (model fits its training labels — expected); **more useful as a project finding** (the architecture demonstrably learns ethogram-grade discrimination on this dataset). Pattern B reduces retraining EV; Step 5 can be skipped or downgraded.
- **Pattern A (V-JEPA-2 ≈ audit despite training on RME) — interesting research, *worrying* for RHpE transfer.** Two readings: (i) pretrained V-JEPA-2 features carry sub-threshold motion information that the LR couldn't filter out (capacity concern), or (ii) the LR is undertrained / under-regularized (training-discipline concern). Either way, the model did not cleanly learn what it was trained on. If V-JEPA-2 + LR cannot reliably learn the EquiFACS threshold, there is a calibration question about whether it can reliably learn RHpE behavior thresholds. Pattern A triggers Step 5 (full LOSO re-run) **as a calibration investigation, not a celebration** — the question to answer is "what changed and why."
- **Pattern C (mixed by source) — calibration question made concrete.** V-JEPA-2 likely tracks EquiFACS on clean sources (S6, S7) and audit threshold on noisy sources (S5, S10). Useful diagnostic of where the LR's training signal was clean vs noisy, but ambiguous for RHpE transfer because RHpE deployment will not have a clean / noisy source distinction baked in. Pattern C triggers Step 5 with focus on the per-source split — under what conditions does the model land at which threshold?

**Anti-pattern to avoid in Step 5 writeup.** Do not celebrate Pattern A's "model is robust to label noise" framing. Under the bifurcation lens, "model is robust to label noise" reads as "model didn't learn what it was trained on" — which is a calibration concern, not a strength. Higher LOSO under Piotr-permissive labels does not mean V-JEPA-2 + LR is "even better than the published 0.875"; it means V-JEPA-2 + LR is sitting at a different threshold than its training signal, which raises questions about reproducibility on RHpE-grade tasks. The Pattern interpretation is fixed in this pre-registration before B-prime runs to prevent post-hoc framing drift.

**B-prime result (added 2026-05-07):** Pattern C identified. Per-clip V-JEPA-2 + LR LOSO predictions reproduce the published 0.8746 **AUC** exactly (sanity check passed; canonical config is `RidgeClassifier(alpha=1.0, class_weight='balanced')` + `StandardScaler` per fold on `vjepa2_embeddings.npz`). Note: the spec's Pattern A/B/C trigger is defined in terms of binary **agreement** (V-JEPA-2 `predict()` output vs labels), which is a different metric than AUC — both are computed from the same predictions but aggregate them differently. Reconciliation: AUC vs RME = 0.875, vs Piotr-strict = 0.839, vs Piotr-perm = 0.828; binary accuracy vs RME = 0.806, vs Piotr-strict = 0.797, vs Piotr-perm = 0.781. Both metrics agree on direction (V-JEPA-2 leans RME at aggregate); per-source binary-accuracy deltas drive Pattern C identification per spec wording.

**Per-source agreement deltas (binary, per spec §4)** vs Piotr-strict: positive on noisy sources (S5 +5.0 pp, S2 +5.7, S3 +8.7, **S8 +18.1**), negative on cleaner sources (S1 −10.7, S11 −10.5, **S12 −13.3**). The split tracks Lesson 17's source-cleanliness ranking exactly: V-JEPA-2 + LR sits at the EquiFACS threshold on cleaner sources (where audit and RME agree, so RME-trained model trivially also matches audit) and drifts toward the audit threshold on noisier sources (where the multi-horse confound + subthreshold-motion signals push V-JEPA-2's discrimination boundary toward what the audit also caught). Step 5 (B) will run after Step 3 (consistency check) clears the hard-stop gate, with focus on the per-source split. Full breakdown — including the AUC ↔ binary-accuracy reconciliation, per-source AUC table with NA caveats on small strict subsets, and predicted Step 5 variant numbers — in [`outputs/vjepa2_label_agreement_decomposition.md`](../outputs/vjepa2_label_agreement_decomposition.md). **Calibration question made concrete; not validation, not invalidation, but specifically the diagnostic outcome the spec named.**

**Per-source disagreement correlates with V-JEPA-2's weak-fold pattern.** S5, S10, S2 (highest disagreement with audit) and S4, S8 (multi-horse load) overlap substantially with the sources where V-JEPA-2 LOSO struggled in iter-6.5 sanity runs. This is a strong signal but not a clean attribution: the model could be (a) failing where labels are noisy by audit standards, (b) failing where the visual task is genuinely harder, or (c) both. B-prime disambiguates by comparing V-JEPA-2 predictions against both label sets per-source — in particular, Pattern C is the answer that makes the (a) vs (b) attribution operational.

**Overlap with the 36-clip MLLM evaluation subset.** All 36 clips were audited (full coverage). 31 are confident verdicts; 5 are borderline. RME vs Piotr disagreement on the subset is **3/36 strict (9.7 %)** — slightly cleaner than the dataset-wide 12.4 %, partially explaining why the §4 row 1 verdict on Qwen v2 holds across both label sets at the aggregate level even though the bifurcation framing changes the qualitative reading.

**S4_7 reframing is locked.** Lesson 15 v2 already records: Qwen v2's "false positive" on `background_S4.mp4_7` is target-confusion on a multi-horse clip (the audit confirms `fh: strong head movement, slight left ear rotation, bh: both ears rotation`), not a hallucination. The v2 prompt-A bg-rate of 94 % therefore overstates conservatism by 1 clip on the multi-horse subcase — Qwen perceived real motion, just on the non-target horse. This is consistent with the bifurcation finding: Qwen v2 + A still sits in the collapse mode (94 %), but it is one clip closer to the over-detection mode than its sister configurations because of the multi-horse perception.

**What changed vs the option-A stub of this lesson.** The stub was based on the user's review of 21 background clips only and named the dual-label finding only as a comparative footnote ("non-collapsed configuration aligns best"). This rigorous version is based on all 283 clips (full dataset), names the underlying finding (threshold bifurcation), centers it as the load-bearing artifact, and pulls in the asymmetry plus the implications for clinical pre-screening and Lesson 14 refinement. The conclusions on disagreement rate, multi-horse load, and per-source heterogeneity all hold and are now precisely quantified.

### Hard caveats — single-observer, no κ

- **One reviewer.** No inter-rater agreement measurement. The 12.4 % disagreement could shift up or down with a second reviewer. The Step 3 within-observer consistency check bounds (but does not replace) inter-rater κ.
- **Reviewer is non-blind.** The reviewer is the project owner who has been thinking about V-JEPA-2 / MLLM behavior on this dataset for months. Possible bias toward labels that "make sense" given the model's outputs.
- **Subthreshold category is the soft boundary.** Most of the 24 bg→action flips are in the "very slight twitch" / "slight rotation" zone. EquiFACS coders would label these bg by protocol (intensity/duration thresholds); the reviewer would label them action by a more permissive "any visible motion" threshold. This is a definitional difference, not a labeling error — and it is precisely the threshold gap that opens the bifurcation finding above.
- **Multi-horse confound is the hard finding.** The 55 multi-horse clips are observation-grounded, not threshold-disagreement. S4 + S8 carry visible motion on non-target horses in the frame, and that fact is independent of any threshold.
- **Implication for the §4 Qwen verdict — unchanged at the aggregate-pass-fail level.** Row 1 holds under both label sets. The MLLM-as-classifier track stays closed. What changes is the qualitative reading: at least 1 of Qwen v2's 2 "false positive" calls is a Lesson 9 multi-horse confound rather than a hallucination, and the bifurcation framing identifies a meaningful structural difference between collapsed and over-detection modes that the §4 row-by-row comparison does not capture.

### What this lesson is not

- Not a claim that "RME labels are wrong." The 12.4 % disagreement reflects threshold differences (subthreshold motion correctly bg per EquiFACS protocol but visible to the reviewer) and known confounds (multi-horse), not annotation errors. EquiFACS coders applied a documented protocol; this audit applied a different threshold.
- Not a multi-reviewer study; not a κ measurement; not publishable in its current form without inter-rater work.
- Not a recommendation to relabel `vendor/ReadMyEars_Dataset/`. User labels live in a separate JSONL.
- Not a basis for any V-JEPA-2 LOSO retraining decision yet. Step 2 (B-prime) is gating that.
- Not a claim that Gemini 2.5 + Prompt A is the right MLLM for any classification task on RME. It is a configuration that happens to match a naive-observer threshold rather than the EquiFACS threshold; that is calibration information, not a recommendation.

### Pending Step 3 (within-observer consistency)

When 66-clip re-watch results land, the borderline self-consistency rate (likely ~70-90 %) and confident-case self-consistency rate (must be ≥ 90 % per spec hard-stop gate) will be inserted here as the headline caveat figure. Until then, every quantitative claim above is single-observer-disagreement-with-published-protocol, not an audit reliability claim.

**FH-only protocol clarification adopted 2026-05-07, before re-watch.** In multi-horse clips, the verdict applies to the foreground horse only (largest / most central / in focus); background-horse motion is excluded regardless of magnitude. The original audit's `fh:`/`bh:` notation made the multi-horse split visible per clip but did not commit to a verdict rule; FH-only is the now-explicit rule. **14 of the 66 re-watch clips are multi-horse** (all S4 + S8); 7 are `multi_horse_distractor` borderlines that have a predictable protocol-driven flip from `?` to confident `BACKGROUND` under FH-only. Step 3 will compute both raw self-consistency (conservative; includes those 7 protocol-flips as "inconsistencies") and protocol-adjusted self-consistency (excludes them as protocol clarification). The protocol-adjusted rate is the headline figure; the raw rate is the conservative bound. The 2 multi-horse confident controls (`action_S4.mp4_5_.mp4`, `action_S8.mp4_4_.mp4`) are `target_focus` cases with strong FH motion — they should remain confident `ACTION` under FH-only, so a flip on either is a real inconsistency that triggers the hard-stop gate.

---

## What worked (verified, build on)

- **V-JEPA-2 ViT-L encoder features** (1024-d, pretrain-only by construction in our pipeline — see Lesson 12)
- **Read My Ears protocol** (face mask + ear bbox crop + linear probe): LOO 0.97, bg-masked LOO 0.91, **LOSO 0.875** (Sanity 5 — source-invariant on their data)
- **Linear probe (LogReg / RidgeClassifier) + LOO observed AUC + permutation test + LOSO** as a four-layer evaluation stack
- **Hard pre-committed decision thresholds** for architecture choices
- **Static-frame collapse diagnostic** for distinguishing temporal vs static feature reliance
- **Conditional background masking** (Lesson 9): apply when YOLO detects >1 subject in frame, skip otherwise — heterogeneous gains/losses across sources averaged 0.764 LOSO but +24 pp on hardest fold

## What didn't work (verified, don't rebuild)

- **5-class softmax on 53 anchor clips** (iter 2/3/5): too small, session-confounded, eye_expression sink-effect
- **head_position 0.898 LOO** as MVP candidate (Sanity 3 LOSO 0.561 = session leakage)
- **5-fold CV with class_weight='balanced' on imbalanced small-N classes** (mouth_open N=3 collapsed to all-zero predictions in iter 5)
- **DINOv2+V-JEPA-2 concat** (lost on 2 of 4 behaviors in iter 6, LOSO 0.747 vs SSv2 0.875 in Sanity 5)
- **DINOv2 alone as universal backbone**: LOSO 0.514 on Read My Ears, but more importantly **anti-correlated for 4 of 12 sources** (S1 0.388, S6 0.233, S9 0.154, S12 0.393). Image-only foundation models can learn label inversion when label is source-specific.
- **Background masking as global default** (Lesson 9): hurts strong sources by ~10 pp while helping weak ones — must be conditional
- **The 53-clip DIY anchor dataset as a training set** for any per-behavior classifier (iter 6.5)

## Open questions

- ROI-cropped per-behavior classifiers (ear / eye / mouth / hindquarter) on properly diversified data — never reached because the anchor dataset was disqualified before we got there.
- ~~Whether the original Read My Ears paper used LOSO or LOO; if LOO, their 0.875 may be inflated~~ **RESOLVED 2026-05-06 (Sanity 5)**: split is clip-level random (S1, S2, S3... in train+val+test), but **LOSO leave-one-source-out reproduces 0.875 exactly** on V-JEPA-2 SSv2 motion. The paper claim is robust under source-aware split — it appears either lucky methodology or genuinely source-invariant signal in their controlled lab data. Earlier suspicion that their 0.875 was inflated has been falsified empirically. See `outputs/iter65_sanity5_loso_rme_results.json`.
- Track C (DLC keypoints + temporal features) — never tested; deferred to Phase 3 if Track B succeeds.
- Whether DINOv2 alone is sufficient for production (Sanity 4 result, see `outputs/iter65_sanity4_dinov2_bgmask_results.json`).

---

## Recommendations for Phase 2 (post-iter-6.5)

1. **Track A "head_position MVP" — kill.** No path forward without complete data redesign with balanced sessions.
2. **Track B "ear_position via ROI replication" — proceed, with revised sizing.** ≥10 horses × 2–3 ear states × 2–3 takes = 60–100 clips, budget 3000–5000 PLN for assessor, 4–6 weeks recording.
3. **LOSO required as primary evaluation metric**, not LOO. Permutation test on top.
4. **Pre-commit a realistic Track B success criterion before recording starts** (Lesson 11): LOSO AUC **≥0.70** across ≥8 of ≥10 sources is a passing MVP; ≥0.80 is strong; ≥0.85 is unrealistic on diverse Polish data and shouldn't be promised. RME 0.875 was on a controlled lab study.
5. **Pipeline architecture** — V-JEPA-2 SSv2 encoder + ear-bbox crop + linear probe + **conditional background masking** (Lesson 9): default unmasked, switch to bg-masked when YOLO detects secondary subject in frame.
6. **Recording-protocol gates** (Lesson 10):
   - Single subject in frame, no secondary horses or moving humans visible
   - Diverse recording contexts in training distribution if deployed model is expected to see them (medical-instrumented vs casual-stable vs under-saddle)
7. **Track C (DLC keypoints) — remains deferred** to Phase 3.
8. **Track D candidate — Rushed / Slowed Gait via deterministic step counting** (see Lesson 13, Class B). Operational definition under TWTP guide is *"more than 40 trot steps per 15 seconds"* (Rushed) / *"fewer than 35"* (Slowed). DLC SuperAnimal-Quadruped → hoof Y-coordinate time series → bandpass + peak detection → count peaks per 15s window. No assessor adjudication required (ground truth is the count). 1–2 weekend prototype, contingent on DLC keypoint quality on arena footage being good enough — currently unverified for our deployment conditions and is the binding risk. Worth running before committing to Track B MVP, as it adds 2/24 behaviors to coverage at a fraction of the cost.

The point of this document is not that the project is in trouble. It is that we now know what we don't know, which is genuinely the best position from which to plan Phase 2.
