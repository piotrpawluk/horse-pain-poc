# Track B Phase 5 — Pre-registration (gold-standard manual eye-region crops + 2×2 LOSO)

**Frozen:** 2026-05-09, BEFORE the user's eye-box annotation begins. Hash recorded in `docs/preregistration_hashes.md` at the same atomic commit.

**Phase 4 close-out** (audit reference, unchanged — do not amend): Phase 4 primary v2 + tightened-rubric labels regressed to AUC 0.5854 (Δ vs Phase 3 = −0.0959). Both pre-registered ablations confirmed both interventions individually hurt: tightened-rubric labels alone hurt by ~0.10 (Ablation A), v2 crops alone hurt by ~0.13 (Ablation B). Factor (d) verdict: SUPPRESSED (2/3 BG-targets below median). Pre-run inspection of v2 contact sheet revealed the heuristic systematically captures ear regions due to ears dominating high-frequency content in the upper-40 % face strip on horse anatomy. Phase 4 closed per locked protocol; full audit at `docs/phase4_audit.md`.

**Phase 5 question:** Is cropping quality the dominant bottleneck holding back the eye-track AUC, or is the architecture / labels / N the bottleneck? Phase 4 conflated two questions; Phase 5 decouples them by using user-annotated gold-standard eye-region bounding boxes — converting the cropping question from "which heuristic should we lock?" into "if cropping were perfect, what is the architecture's ceiling?"

**Constraints honored** (per user, 2026-05-09): solo, fast (~5 h wall-clock with ~4-6 h annotation gap for intra-rater), no second observer, no N expansion. PoC is meant to ATTRACT collaborators, not require them.

## Step 0 — Pre-run sanity already complete (locked here)

### Step 0a — Phase 3 reproduction sanity ✓ PASSED

Re-ran `tools/eye_loso_lr.py` against existing v1 embeddings + original labels at this commit's working tree. Pooled AUC = `0.6813186813186813` bit-exact match against the Phase 3 baseline. No code drift since Phase 3 commit (`6f4352e`). Full output at `outputs/eye_loso_results.PHASE5_REPRO_CHECK.json` (hash registered).

### Step 0b — MDE synthetic estimate ✓ COMPLETE

Subject-bootstrap on actual Phase 3 fold structure with simulated v3 predictions at controlled true Δ-AUC:

| Target Δ | Achieved Δ | Power @ α=0.05 (paired, two-sided) |
|---|---|---|
| 0.000 | +0.004 | 6.7 % (calibration ✓) |
| 0.030 | +0.022 | 16.7 % |
| 0.050 | +0.042 | 46.7 % |
| 0.070 | +0.065 | 70.0 % |
| **0.100** | **+0.096** | **90.0 %** |
| 0.150 | +0.135 | 96.7 % |
| 0.200 | +0.198 | 100.0 % |

**MDE-80 % ≈ 0.085** for paired Δ test against Phase 3.
**Single-AUC level CI half-width ≈ 0.24** (subject-bootstrap), so AUC ≥ 0.74 needed for bootstrap LB ≥ 0.50; AUC ≥ 0.84 needed for LB ≥ 0.60 (the level test is harsher than the paired test on this fold structure — that's why Phase 5 uses the paired test for the top-band gate, not the level test).

Both numbers will be cited in the Phase 5 audit doc and in any threshold rationale.

## What Phase 5 does

A 2×2 factorial (cropping × labels) with an additional margin-sensitivity curve, all on the same 34-clip viable set. Single observer; clips re-used from Phase 3/4 (no new clip selection).

### Annotation method

You hand-draw axis-aligned bounding boxes containing the visible eye + a 15 % margin (locked: see rubric below). For each clip you annotate the **first**, **middle**, and **last** frame independently; the agent computes pairwise IoU between (first, middle) and (middle, last):

- If both IoUs > 0.7 → use middle-frame box for all native frames (static)
- If either IoU < 0.7 → linearly interpolate box center + scale across frame indices (interpolated)

This addresses head-rotation cases like `background_S5.mp4_10_` that motivated Phase 5 to begin with.

### Locked annotation rubric

> **Bounding box content**: smallest axis-aligned rectangle containing the entire visible eye (sclera + lid + lash margin) plus a 15 % margin on all sides. The 15 % margin is computed from the eye's tightest box: bbox_height × 0.15 added to top and bottom; bbox_width × 0.15 added to left and right.
>
> **Partial occlusion / head turn**: capture the visible portion only. Do not attempt to estimate where the eye would be behind occlusion.
>
> **Eye not visible**: if the eye is not visible at all in the frame being annotated, mark the frame as "no_eye_visible". If all three frames (first, middle, last) of a clip have no_eye_visible, mark the clip as excluded from Phase 5 LOSO and document in the audit.
>
> **Symmetry**: do NOT bias box tightness or margin asymmetry based on memory of which clips were inverted in Phase 3 or relabeled in Phase 4. Annotate purely on geometric visibility.

This rubric is hashed alongside the keymap before annotation begins.

### Annotator blinding (UUID rename)

Filenames are masked with UUIDs (seed=45) before annotation. The keymap JSON is hash-locked at this pre-reg commit. You annotate by UUID; do not consult the keymap until after submission. PNG frames extracted at deterministic indices matching V-JEPA-2's frame-sampling convention (`np.linspace(0, n_total - 1, 16).astype(int)` then taking indices 0, 7-or-8, 15 for first/middle/last):

- `outputs/eye_box_frames/<uuid>_f<idx>.png` × (3 frames × 34 clips = 102 PNGs)

Annotation submission format: `outputs/eye_boxes_phase5a.json` mapping `uuid → {f_first: [x,y,w,h], f_middle: [...], f_last: [...]}` (or `"no_eye_visible"` per frame).

### Intra-rater consistency check (4–6 h gap)

After Phase 5a primary annotation lands, agent prepares a re-annotation request: 5 randomly-selected UUIDs (seed=46, distinct from primary mask seed 45) re-extracted with different ordering. You re-annotate ≥4 h later (same-day; the ≥48 h ideal is documented as a known limitation, deferred to Phase 6). Agent computes IoU per frame per re-annotated clip; reports median + min + max in audit doc.

**Pre-registered gate:** if median IoU < 0.6 across the re-annotated set, the audit doc and Track A writeup explicitly flag the "gold-standard" framing as overclaim — v3 boxes are then characterized as "single observer's best-effort eye-region annotation" with explicit re-annotation noise.

### V-JEPA-2 extraction with 3-assertion parity

Before any new extraction, the parity check asserts:
1. **Cosine similarity ≥ 0.999** vs cached `outputs/vjepa2_embeddings.npz` on the parity-test clip
2. **Frame indices** (`np.linspace(0, n_total - 1, 16).astype(int)`) reproduce bit-exactly for the parity clip
3. **Input resolution** = 224 × 224 (asserted from processor config; the model name says `-256-` but the processor resizes to 224)

Any failure halts before extraction. Outputs `outputs/vjepa2_embeddings_eye_v3.npz` (38 entries: 34 clips × 1 box per clip, no tie-break-both-halves rule because Phase 5 uses gold-standard boxes not heuristic selection).

### LOSO LR pipeline

Identical to Phase 3/4 (`RidgeClassifier(α=1.0, class_weight="balanced")` + `StandardScaler` refit per fold + post-fix per_clip alignment + permutation test n=1000 + DeLong analytical CI as reference + subject-bootstrap CI B=10000 for primary decision metric + DeLong paired test for Δ vs Phase 3).

Three locked configurations:

1. **Phase 5 primary**: v3 (gold) crops + **original Phase 3 labels** + 15 % margin → primary metric
2. **Phase 5 sensitivity 1**: v3 (gold) crops + **tightened Phase 4 labels** + 15 % margin → rubric-tax under good crops
3. **Phase 5 sensitivity 2**: v3 (gold) crops + original labels + **margin curve {10 %, 15 %, 40 %, 80 %}** → context-vs-tightness diagnostic. Same boxes; only crop margin varies.

Note: 15 % margin run is shared between configs 1 and 3 (one extraction, two reports).

## Pre-registered thresholds (locked, MDE-aware, disjoint)

### Primary — v3 + original labels + 15 % margin (3 disjoint bands)

| Band | Condition (binding) | Conclusion |
|---|---|---|
| **Top** | **Δ vs Phase 3 ≥ 0.10** AND **DeLong-paired p < 0.05** | Cropping was a real bottleneck. Architecture clears the realistic POC band (Lesson 11: 0.70-0.80 target, ≥ 0.85 unrealistic) given good crops. Open questions: automating good crops + robustness at higher N + multi-rater κ. |
| **Middle** | **AUC ≥ 0.6313** AND NOT (Top condition) | Cropping helped within statistical noise (or effect size large but paired test couldn't reject at this n). Realistic POC interpretation; same collaborator asks as Phase 3 plus eye-detector annotation. |
| **Regression** | **AUC < 0.6313** (Δ < −0.05) | Cropping wasn't the dominant constraint, or v3 boxes lost context. Conditional ablation runs. |

Rationale:
- **Top binding gates**: Δ ≥ 0.10 is at MDE-90 % power per Step 0b. Δ ≥ 0.10 arithmetically implies AUC ≥ 0.7813 against Phase 3's 0.6813; bootstrap LB on a 0.78 AUC is approximately 0.54 (above chance). The paired-p test adds a real second constraint — Δ may be high but paired SE may be too wide to reject.
- **Disjoint partition**: regression boundary aligned at 0.6313 (= Phase 3 − 0.05 = MDE-aware ablation trigger). Top excludes Middle by the conjunction. Middle absorbs the MDE-noise window cleanly and includes the case "Δ ≥ 0.10 but p ≥ 0.05" (effect size large but paired test couldn't reject).

### Sensitivity 1 — v3+tightened Δ vs v3+original (3 rows incl. positive tail)

| Δ regime | Conclusion |
|---|---|
| **Δ < −0.085** | Rubric-tax intrinsic to V-JEPA-2 + Ridge at this n; tightened rubric needs different architecture for similar AUC |
| **−0.085 ≤ Δ ≤ +0.085** | Tightened rubric is clean under good cropping; Phase 4's −0.10 was crop-interaction, not rubric-architecture mismatch |
| **Δ > +0.085** | Tightening HELPS under good crops — Phase 4's tightening direction was correct; the v1-crop interaction was inverted. Investigate which clips drive the gain (factor-(a) re-labeled clips may align with the encoder's signal under proper cropping). Phase 4 vindication. |

Window is the MDE-80 % from Step 0b.

### Sensitivity 2 — margin curve {10, 15, 40, 80} % (4 noise-tolerant categorical shapes)

Per-point bootstrap CI computed (B=10000 subject-resampling). Bootstrap half-width ≈ 0.24 per point. Compute neighbor-pair AUC differences across {10→15, 15→40, 40→80}:

| Curve shape | Condition |
|---|---|
| **Monotone-up** | All three pair-differences ≥ 0 AND at least one pair > bootstrap half-width AND no contradicting pair > half-width |
| **Monotone-down** | All three pair-differences ≤ 0 AND at least one pair > half-width AND no contradicting pair > half-width |
| **Inverted-U** | (40 % AUC) > (15 % AUC) AND (40 % AUC) > (80 % AUC) AND max-vs-neighbor differences both > half-width |
| **Flat** | None of the above; data doesn't support a curve-shape claim at this MDE |

Interpretation:
- Monotone-up: tighter is worse; v3 was over-tight; Phase 6 commits to looser margins
- Monotone-down: tighter is better; Phase 6 commits to tighter heuristic
- Inverted-U at 40 %: sweet spot exists; Phase 6 targets that range
- Flat: margin doesn't matter at this scale; Phase 6 chooses on other criteria

### Factor-(d) suppression criterion (locked, same as Phase 4)

The 3 persistent BG-target clips remain `action_S5.mp4_2_`, `background_S6.mp4_2_`, `background_S6.mp4_3_`. Computed against the median Phase 5-primary BG-clip score (parent-clip scores; v3 doesn't tie-break, so each clip has one score). ≥ 2 of 3 strictly below median → SUPPRESSED; ≤ 1 → PERSISTENT. Recorded in `outputs/eye_loso_results_phase5_primary.json` field `factor_d_suppression`.

### Conditional ablation (only on regression, AUC < 0.6313)

If the primary metric regresses below Phase 3 − 0.05:
- **Factor (a) — context loss**: covered upfront by sensitivity 2 (margin curve). If the margin curve shows monotone-up or inverted-U with peak > 15 %, context loss is the explanation.
- **Factor (b) — annotation noise**: covered upfront by intra-rater IoU report. If median IoU < 0.6, annotation noise is at issue.
- **Factor (c) — infrastructure drift**: covered by Step 0a reproduction sanity (already passed, bit-exact).

No additional ablation runs are required if the primary regresses, since (a)/(b)/(c) are already instrumented. The audit doc consolidates the diagnostic from those three pre-existing reports.

## What Phase 5 explicitly does NOT do

1. **No heuristic-cropping retest.** v2's heuristic family ruled out by Phase 4 + contact sheet evidence; manual annotation is the cleaner test.
2. **No N expansion.** Same 34 viable clips. N is the bottleneck for CI precision but isn't what Phase 5 measures.
3. **No multi-rater κ.** Single observer. Same caveat as Phase 3/4. Intra-rater check at 4-6 h gap (deferred from ideal ≥ 48 h) is documented as known limitation.
4. **No new architecture.** V-JEPA-2 ViT-L stays the encoder.
5. **No new labels collection.** Both label sets exist (original Phase 3 + tightened Phase 4); Phase 5 reuses them.
6. **No tie-break-both-halves duplication.** v3 is gold-standard manual; one box per clip; 34 entries in v3 npz vs 38 in v2.
7. **No threshold revision after seeing the result.** Bands above are locked at this commit's hash.

## Atomic outputs (when Phase 5 lands)

- `outputs/eye_box_keymap_phase5.json` — UUID → real-filename mapping (frozen at this commit)
- `outputs/eye_box_frames/<uuid>_f{0,N/2,N-1}.png` × 102 — annotation PNGs
- `outputs/eye_boxes_phase5a.json` — your primary annotations (34 clips × 3 frames)
- `outputs/eye_boxes_phase5b.json` — your re-annotation of 5 clips (intra-rater)
- `outputs/phase5_intra_rater_iou.json` — IoU computation result + verdict on rubric framing
- `outputs/eye_crops_v3/<clip>.mp4` × 34 — gold-standard 224×224 crop videos at 15 % margin
- `outputs/vjepa2_embeddings_eye_v3.npz` — V-JEPA-2 features at 15 % margin, parity-checked
- `outputs/eye_loso_results_phase5_primary.json` — primary v3+original+15 %
- `outputs/eye_loso_results_phase5_sens_rubric.json` — sensitivity 1 v3+tightened+15 %
- `outputs/eye_loso_results_phase5_sens_margin_{10,40,80}.json` — sensitivity 2 margin points
- `docs/phase5_audit.md` — method + result + decision branch + curve-shape verdict + intra-rater verdict + Phase 6 implications
- Updates to `outputs/eye_probe_results.md` (Track A writeup) with Phase 5 result and locked sentence framing
- Updated `docs/preregistration_hashes.md`

## Hash chain commitment

This freeze commits to git BEFORE annotation begins:

- `outputs/track_b_phase5_preregistration.md` — this document
- `outputs/eye_box_keymap_phase5.json` — UUID mapping (locked before user opens annotation PNGs)
- `outputs/eye_loso_results.PHASE5_REPRO_CHECK.json` — Step 0a sanity output

Audit chain pattern matches Phase 4: keymap hash committed before user can match UUIDs to clips.

## Time budget

| Step | Owner | Time |
|---|---|---|
| Pre-reg + keymap + PNGs lock (this commit) | agent | 30 min |
| You annotate 34 clips × 3 frames @ 15 % margin (UUID-masked) | you | 90 min |
| ≥ 4-6 h gap | — | 4-6 h |
| You re-annotate 5 random clips (different mask order) | you | 15 min |
| Agent computes IoU + reports | agent | 5 min |
| `tools/eye_crop_pipeline_v3.py` build (boxes → crops + 3-frame interpolation) | agent | 30 min |
| V-JEPA-2 extraction with 3-assertion parity | agent | 5 min |
| Primary + sensitivity 1 + sensitivity 2 (4 margin points) LOSO runs | agent | 10 min |
| Conditional regression ablation (covered upfront by sens-2 + intra-rater) | agent | 0 min if no regression |
| Phase 5 audit doc + Track A update + atomic commit + push | agent | 60 min |
| **Total wall-clock** | | **~5-7 h spread across morning + afternoon** |
