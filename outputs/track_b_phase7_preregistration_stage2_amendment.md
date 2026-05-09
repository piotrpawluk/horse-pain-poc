# Track B Phase 7 — Stage 2 amendment

**Frozen 2026-05-09 BEFORE Step 3 (crop pipeline run).** Stage 2
fills concrete threshold values per Stage 1's locked meta-rules,
informed by Step 1.5 observations on the 34 RME clips
(`outputs/phase7_rme_dlc_keypoints.json`, hash `82ae2c8d11aa...`).

This amendment supersedes Stage 1 §7's meta-rule for Phase 7 only —
see §1 below. Stage 1 §6 and §5 meta-rules apply as written; this
amendment fills in their concrete values.

---

## §1 — Confidence threshold (HARD-LOCK at 0.5; meta-rule point-fixed)

### What Stage 1 §7 said

> "Threshold = `max(0.5, p25_pooled_observed)`."

### What Step 1.5 observed

Pooled distribution of target-eye-keypoint confidence across 34 RME
clips × 891 frames:

| Stat | Value |
|---|---:|
| min | 0.2016 |
| **p25** | **0.8872** |
| median | 0.9190 |
| p75 | 0.9417 |
| max | 1.0000 |

Mechanical application of the meta-rule would set threshold = 0.8872.

### Why mechanical application was rejected

At threshold 0.8872, **by construction** ~25% of pooled frames are
below threshold (since p25 = 25th percentile). The Stage 1 §7
intent — "raise the threshold when DLC is confident; fall to 0.5
floor when DLC is unconfident" — assumed Phase 0-like distributions
where p25 sits near a meaningful low-confidence tail. Phase 0's p25
was around 0.7 with ~9% of frames below 0.5 (true low-confidence
detections).

RME's distribution has no meaningful low-confidence tail: median 0.92,
p75 0.94, with 0.13% of frames below 0.5. The lower 25% of a
right-shifted unimodal distribution is not the same as the
"unreliable" subset. The meta-rule conflated *percentile* with
*quality signal*.

**Per-clip operational consequence at threshold 0.8872** (computed in
Step 1.5):

| Outcome | Count |
|---|---:|
| Clips with >50% frames below threshold (DROP from LOSO) | **8/34** (24%) |
| Clips with 25-50% frames below (per-clip-locked single-keypoint fallback) | 3/34 (9%) |
| Clips with ≤25% frames below (operate per-frame normally) | 23/34 (68%) |

**Of the 8 dropped clips, 4 are V3_NEWLY_RECOVERED** — exactly the
clips Phase 5 manual annotation specifically rescued. This is not
coincidence — it's a load-bearing methodological finding (see §5
below).

### Lock

Phase 7 confidence threshold = **0.5** (the Stage 1 §7 floor, applied
directly).

At threshold 0.5, RME fail rate = 0.13% (1 frame in 891). Effectively
all 34 clips operate per-frame. The diagnostic-rich clips Phase 5
rescued are preserved in the LOSO.

### Scope of the amendment (point-fix)

**§7 hard-lock at 0.5 applies to Phase 7's RME data only; future-phase
reuse requires §7 redesign per locked discipline pattern.**

If a future phase encounters DLC on different data with **genuinely
bimodal** confidence distribution (a meaningful low-confidence cluster
distinct from the high-confidence body), the meta-rule needs proper
redesign — not mechanical reapplication of `max(0.5, p25)`. The
current rule is **structurally** percentile-based, not quality-based.
A redesign would need a quality anchor (e.g., calibration against
known-bad detections, or a second-mode-detection algorithm) rather
than a positional statistic.

**Anti-pattern locked**: a future agent must NOT apply
`max(0.5, p25_pooled)` to a different dataset and assume the rule
held. The rule failed empirically on Phase 7's RME data. Reuse
requires explicit verification that the data has a bimodal confidence
distribution — and even then, the threshold value should anchor on
the bimodality boundary, not on an arbitrary percentile.

---

## §2 — Frame-failure thresholds (defaults stand: X=25%, Y=50%)

### What Stage 1 §6 said

> "X% and Y% are locked in stage 2 (after observing DLC eye-keypoint
> confidence distribution on RME clips). Meta-rule for setting them:
> Default X = 25% and Y = 50% (Phase 6(b) values), unless DLC
> failure characteristics differ substantially. 'Substantially
> different' = if observed failure rate on RME clips is more than 2×
> Phase 0's ~9%, X and Y need adjustment to keep the fallback regime
> uncrowded."

### What Step 1.5 observed (at the §1-locked threshold of 0.5)

| Comparison | Phase 0 | Phase 7 RME | Trigger |
|---|---:|---:|---|
| Frame failure rate at threshold 0.5 | ~8.5% | **0.13%** | ≥18% |

RME fail rate (0.13%) is **65× lower** than Phase 0's baseline (~8.5%);
miles below the 18% trigger for adjustment.

### Lock

Phase 7 frame-failure thresholds (per Stage 1 §6 structure):

| Threshold | Locked value |
|---|---:|
| **X** (single-frame failure → interpolate from neighbors) | continuous case |
| **X_clip** (>X_clip% → fall back to single-middle-frame keypoint per clip) | **25%** |
| **Y_clip** (>Y_clip% → drop clip from LOSO) | **50%** |

These are Phase 6(b)'s values. Per-clip recomputation at threshold 0.5:

| Outcome | Count at threshold 0.5 |
|---|---:|
| Clips with >50% frames below 0.5 (would be DROPPED) | **0/34** |
| Clips with 25-50% frames below (per-clip-locked fallback) | **0/34** |
| Clips with ≤25% frames below (operate per-frame) | **34/34** |

All 34 clips operate per-frame; zero fallbacks, zero drops. The 6
ambiguous-side clips (per Stage 1 §4) still use the per-clip-locked
fallback rule for *eye selection* (right_eye vs left_eye), but operate
per-frame for keypoint quality.

---

## §3 — Absolute fallback eye dimensions (LOCKED from Phase 5 manual median)

### What Stage 1 §5 said

> "**Fallback**: option C — eye-keypoint position + locked absolute
> pixel dimensions (`abs_eye_w_px`, `abs_eye_h_px` derived from Phase 5
> manual boxes median absolute dimensions, computed in stage 2)."

### Lock

Computed from Phase 5 manual middle-keyframe eye boxes
(`outputs/eye_boxes_phase5a.json`, n=34 clips):

| Dimension | Locked value | Source |
|---|---:|---|
| `abs_eye_w_px` | **62 px** | median absolute eye width across 34 manual boxes |
| `abs_eye_h_px` | **47 px** | median absolute eye height across 34 manual boxes |

**Application**: at frames where the head-bbox proxy from Stage 1 §5
has **fewer than 6 of 9 confident** head keypoints (i.e., proxy is
untrustworthy), fall back to a tight eye box centered on the eye
keypoint with these locked pixel dimensions.

Note: at Phase 7's RME data, no clip is expected to trigger this
fallback frequently — at threshold 0.5, 0.13% frame fail rate means
head-keypoint proxies are nearly always available. This fallback is
defensive, not load-bearing.

---

## §4 — Wikimedia video hash binding (closes Stage 1 §3 outstanding item)

| Field | Value |
|---|---|
| Video path | `data/sample_horse.mp4` |
| SHA-256 | `049353b040c7323a6c4a77eb25a8ab2994ffad95faea36f1022e2ee2b777e2b9` |
| Frame count | 287 |
| Resolution | 640 × 480 |
| FPS | 30.0 |
| Notebook reference | `notebooks/00_smoke_dlc_sample.ipynb` |
| Phase 0 sample JSON | `outputs/sample_horse_superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2_before_adapt.json` |

---

## §5 — Generalizable methodological observation (locked into Phase 7 audit)

The Step 1.5 finding that **4 of 8 clips the §7 meta-rule would drop
at threshold 0.8872 are V3_NEWLY_RECOVERED diagnostic clips** is not
coincidence. Locked phrasing for the eventual `docs/phase7_audit.md`:

> Stage 1 §7 meta-rule (threshold = max(0.5, p25_pooled)) was
> structurally data-blind: at p25 = 0.89 on a high-confidence
> distribution, the rule excludes 25% of frames by construction
> regardless of true reliability. Empirically, 4 of 8 clips that
> would be dropped at this threshold are V3_NEWLY_RECOVERED diagnostic
> clips — the cases Phase 5 manual annotation specifically rescued.
> Case difficulty appears to correlate with DLC keypoint confidence,
> meaning percentile thresholding disproportionately removes the
> highest-information clips. Stage 2 amendment locks threshold at 0.5
> (the floor that was always operationally appropriate); §7's
> meta-rule structure is reframed as a future-phase consideration
> only when DLC confidence distribution is genuinely bimodal.

This is a Phase-7-derived methodological finding generalizable beyond
this experiment: **percentile-based confidence thresholding can
preferentially remove diagnostic-rich cases when case-difficulty
correlates with model-confidence**. Worth recording as a Lesson in
`docs/lessons_learned.md` after Phase 7 completes.

---

## §6 — What this amendment does NOT change

- All Stage 1 §1, §2, §3, §4, §5 (geometry option A + ≥6/9 minimum),
  §8 (gates G1, G2, G3), §9 (DLC-specific routing matrix), §10
  (diagnostic instrumentation), §11 (anti-patterns), §12 (sequencing)
  remain locked as written.
- The 34 RME clips are unchanged.
- The side-assignment artifact
  (`outputs/phase7_eye_side_assignment.json`, hash `95b879d2eb30...`)
  is unchanged.
- The Phase 5 manual eye boxes (`outputs/eye_boxes_phase5a.json`) are
  unchanged.

---

## §7 — Approval signature

User has reviewed and approves the Stage 2 amendment as drafted:

- **§7 hard-lock at 0.5** for Phase 7 only; meta-rule reframed as
  future-phase consideration; explicit point-fix scope to prevent
  mechanical reapplication on different datasets
- **§6 X_clip = 25%, Y_clip = 50%** (Phase 6(b) defaults stand)
- **§5 abs_eye_w_px = 62, abs_eye_h_px = 47** (Phase 5 manual median)
- **Wikimedia video hash binding** (`049353b040c732...`)
- **Generalizable methodological observation** locked for audit doc
  inclusion

User signs off → CC executes Step 3 (crop pipeline) under fully-locked
Stage 1 + Stage 2 parameters → user approves at checkpoint #5 (after
LOSO + diagnostic, before audit doc).
