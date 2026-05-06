# Lessons Learned — RHpE Behavior Classification PoC (Faza 0–1.5)

*Working document, written 2026-05-06 after iter 6.5 sanity checks.*

This document records methodological lessons from the iteration sequence, with emphasis on what we learned by being wrong. It is intended both as a working reference for the project and as evidence of rigor for academic outreach (Andersen, Zamansky, UPWr).

The headline: **Iter 6.5 LOSO sanity check disproved the head_position MVP** that iter 5–6 built up. Without it, Faza 2 would have started with 3000–5000 PLN and 40h of recording on top of session leakage, not behavior detection. The cost of rigor was ~3h of additional compute; the cost of skipping it would have been six weeks of misallocated work.

---

## Timeline

| Iteration | Question | Outcome |
|---|---|---|
| **0** (2026-05-04) | Czy stack open-source działa na M-series? | DLC SuperAnimal-Quadruped GO 4/4 w ~45 min |
| **1** (2026-05-05) | Czy reprodukujemy Read My Ears 0.875 ear_movement? | ✓ V-JEPA-2 + linear probe = 0.854 (paper 0.875, Δ −2.1pp) |
| **1.5 iter 2** | Czy V-JEPA-2 zerolot LOO cosine wykrywa 5 RHpE behaviors na DIY 53 klipach? | overall 0.358; eye_expression 1.000 (later identified as Padma sink-effect) |
| **iter 3** | Czy head-crop YOLO ROI ratuje ear_position? | NIE — 0/17 ear (vs 0/17 full-frame); ROI head-crop niewystarczająca |
| **iter 4** | Czy LOO cosine na 283 RME daje sygnał? | LOO cosine k=1: 0.756; linear probe 0.894 (przebija paper) |
| **iter 5** | Czy linear probe ratuje 5-class na 53 klipach? | head_position AUC 0.927 (binary OvR), reszta < chance |
| **iter 6** | Czy V-JEPA-2 SSv2 jest motion-biased? Macierz 4 backbones × 4 behaviors | Hipoteza motion-domination ODRZUCONA; head_position 0.898 wygląda na real signal |
| **iter 6.5** | Sanity checks: weights, bg-leakage, session leakage | **head_position UNIEWAŻNIONY**; cały 53-clip dataset session-confounded |

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

The Polish HKiJ peer network dataset planned for Faza 2 will have substantially more axes of variance:
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
- Track C (DLC keypoints + temporal features) — never tested; deferred to Faza 3 if Track B succeeds.
- Whether DINOv2 alone is sufficient for production (Sanity 4 result, see `outputs/iter65_sanity4_dinov2_bgmask_results.json`).

---

## Recommendations for Faza 2 (post-iter-6.5)

1. **Track A "head_position MVP" — kill.** No path forward without complete data redesign with balanced sessions.
2. **Track B "ear_position via ROI replication" — proceed, with revised sizing.** ≥10 horses × 2–3 ear states × 2–3 takes = 60–100 clips, budget 3000–5000 PLN for assessor, 4–6 weeks recording.
3. **LOSO required as primary evaluation metric**, not LOO. Permutation test on top.
4. **Pre-commit a realistic Track B success criterion before recording starts** (Lesson 11): LOSO AUC **≥0.70** across ≥8 of ≥10 sources is a passing MVP; ≥0.80 is strong; ≥0.85 is unrealistic on diverse Polish data and shouldn't be promised. RME 0.875 was on a controlled lab study.
5. **Pipeline architecture** — V-JEPA-2 SSv2 encoder + ear-bbox crop + linear probe + **conditional background masking** (Lesson 9): default unmasked, switch to bg-masked when YOLO detects secondary subject in frame.
6. **Recording-protocol gates** (Lesson 10):
   - Single subject in frame, no secondary horses or moving humans visible
   - Diverse recording contexts in training distribution if deployed model is expected to see them (medical-instrumented vs casual-stable vs under-saddle)
7. **Track C (DLC keypoints) — remains deferred** to Faza 3.

The point of this document is not that the project is in trouble. It is that we now know what we don't know, which is genuinely the best position from which to plan Faza 2.
