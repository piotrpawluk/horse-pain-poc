# horse-pain-poc

> Automated RHpE scoring with V-JEPA-2 — methodology-first PoC

[![phase8c](https://img.shields.io/badge/phase_8c-WELL_CALIBRATED_(ECE_0.04)-success)](docs/phase8c_audit.md)
[![phase8b](https://img.shields.io/badge/phase_8b-COMPETITIVE_(n%3D283)-success)](docs/phase8b_audit.md)
[![phase7](https://img.shields.io/badge/phase_7-DLC_corrected_outperforms_Phase_5_(AUC_only)-success)](docs/phase7_audit.md)
[![baseline](https://img.shields.io/badge/Read_My_Ears_LOSO-0.875-success)](docs/lessons_learned.md)
[![eye_phase7](https://img.shields.io/badge/eye_region_DLC_LOSO-0.8462-success)](docs/phase7_audit.md)
[![eye_phase5](https://img.shields.io/badge/eye_region_manual_LOSO-0.7985-blue)](docs/phase5_audit.md)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10--3.11-blue)](pyproject.toml)

The Ridden Horse Pain Ethogram (Dyson 2018) is a 24-behavior checklist for detecting musculoskeletal pain in ridden horses. This repo explores whether a **frozen video foundation model** (V-JEPA-2, Meta 2025) plus **session-aware evaluation** (LOSO) can replicate the published Read My Ears baseline (Alves CVPR W'25, AUC 0.875) and generalize to wilder field data.

> **Scope note.** This is *single-behavior classification*, not pain detection. RHpE requires ≥8 of 24 behaviors co-occurring before pain inference is appropriate (Dyson 2018). The current MVP focuses on one behavior (ear movement). A behavior-by-behavior tractability mapping appears in [`docs/lessons_learned.md` Lesson 13](docs/lessons_learned.md) — short summary: 10 RHpE behaviors *theoretically* fit the Read My Ears ROI + V-JEPA-2 pattern, 8 require DLC keypoints + temporal analysis, 4 need rider/audio context, 2 are rare-event detection. "Theoretical fit" is not "verified per-behavior performance" — iter-6.5 LOSO collapse on head_position (LOO 0.898 → LOSO 0.561) is the cautionary anchor. Multi-behavior pain assessment is a 2+ year horizon and requires clinical validation with certified RHpE assessors.

![DLC SuperAnimal-Quadruped keypoints overlaid on a sample horse video — 5 frames with skeleton overlay](docs/example_output.png)
*5 frames from `00_smoke_dlc_sample.ipynb` — DLC SuperAnimal-Quadruped zero-shot on [Horse_walking_in_corral_MVI_7490](https://commons.wikimedia.org/wiki/File:Horse_walking_in_corral_MVI_7490.MOV.ogv) (Wikimedia Commons CC).*

This is a research prototype, **not a diagnostic tool**.

## Status (as of 2026-05)

| Behavior | Approach | LOSO AUC | Status |
| --- | --- | --- | --- |
| ear_movement (Read My Ears replication) | V-JEPA-2 + linear probe | **0.875** | ✓ replicates paper claim under source-aware split |
| ear_movement (Phase 8b cross-behavior generalization, n=283) | DLC ear keypoint-anchored crop (both-ears bbox) + V-JEPA-2 + linear probe | **0.9008** | ✓ **COMPETITIVE**: G1 STRONG (AUC ≥ 0.80) + G4 INCONCLUSIVE (paired DeLong p=0.3122 at adequate power); Δ +0.0262 vs whole-frame baseline straddles 0 (CI [−0.029, +0.073]); per-source 4/12 +Δ (concentrated in S8 +0.35); per-clip 40 vs 39 BEATS — methodology preserves whole-frame baseline, does not significantly improve. See [`docs/phase8b_audit.md`](docs/phase8b_audit.md) |
| ear_movement (Phase 8c calibration package) | temperature scaling on Phase 8b output via source-aware calibration LOSO | post-cal **AUC 0.8987** / **ECE 0.0397** | ✓ **WELL_CALIBRATED + G1 STRUCTURAL FINDING**: ECE 0.1118 → 0.0397 (~65% reduction); Brier 0.1420 → 0.1289; Murphy decomposition REL ↓87% / RES ↓1.5% / UNC fixed; τ_ear=0.8138 at FPR=0.05 with TPR=0.5435 [0.31, 0.75]; session-level OP P(≥8\|H0) under independence ≈ **1.39e−5** (exact Poisson-binomial; Poisson approx 3.70e−5 = 166% rel err). G1 FAIL is structural (D1↔D2 internal inconsistency); per-source AUC bit-exact invariant 12/12 confirms pooled drop is purely cross-source rank shuffle. See [`docs/phase8c_audit.md`](docs/phase8c_audit.md) |
| eye_region (Phase 3 v1 — heuristic full-upper-frame crop) | V-JEPA-2 + RidgeClassifier LOSO | 0.681 | ⚪ middle band; baseline for Phase 5 cropping intervention |
| eye_region (Phase 5 v3 — manual gold-standard 3-keyframe + IoU interpolation) | V-JEPA-2 + RidgeClassifier LOSO | **0.7985** | ✓ middle band, ceiling under near-perfect crops; n=34, single-observer (κ unmeasured); bootstrap CI [0.584, 0.964] |
| eye_region (Phase 6 (b) — face-bbox-positioned automated crop at locked median anatomical position) | YOLOv8l face → fixed (rel_x, rel_y, rel_w, rel_h) | 0.469 | ✗ **DISTRIBUTED_FAIL**: median IoU 0.165 vs Phase 5 manual boxes; locked routing → DLC SuperAnimal-Quadruped is next |
| eye_region (Phase 7 — DLC SuperAnimal-Quadruped keypoint-anchored crop, **broken §4 mapping**) | DLC right_eye/left_eye → eye box at locked anatomical proportions | 0.579 | ✗ **CONFIDENT_MISPLACEMENT_FAIL**: mean keypoint conf 0.89 + median IoU 0.000 vs Phase 5 manual; locked routing → "side-assignment review" → empirical falsification of §4 anatomical mapping (24/34 clips IoU-better under flipped rule) |
| eye_region (Phase 7 — DLC keypoint-anchored crop, **corrected §4 mapping**) | DLC right_eye/left_eye with horse-anatomical convention | **0.8462** | ✓ **OUTPERFORM_PHASE_5_AUC_ONLY**: G1 + G2 load-bearing PASS; Δ vs Phase 5 manual = +0.048 (paired DeLong p=0.619 inconclusive at n=34, AUC governs); 12 DLC_BEATS_FBB / **0** FBB_BEATS_DLC vs Phase 6(b); G3-vs-AUC divergence reframes Phase 5 — manual is *a* near-optimum, not *the* ceiling |
| head_position | V-JEPA-2 full-frame + LR | 0.561 | ✗ session leakage (LOO 0.898 → LOSO 0.561, Δ −34pp) |
| eye_expression (early iter) | V-JEPA-2 + LR | n/a | ✗ all positive sessions from one source — confound, dropped |
| ear_position (anchor data) | V-JEPA-2 full-frame + LR | <0.5 | ✗ requires ear-region ROI crop, not full-frame |
| ear_movement (MLLM-as-classifier on RME 36-clip subset) | Gemini 2.5/3.1 Pro + Qwen2.5-VL-7B | n/a (refusal-bias collapse) | ✗ all 3 Lesson 14 failure modes reproduced cross-vendor — see [Lesson 15](docs/lessons_learned.md). v1 results were invalid due to mlx-vlm video-routing bug; v2 confirms the conclusion legitimately — see [Lesson 16](docs/lessons_learned.md) |
| eye_region (MLLM-as-classifier zero-shot on RME 36-clip subset, v1 + v2 + v3) | Gemini 2.5/3.1 Pro + Qwen2.5/3-VL | n/a (templated-evidence collapse on uncropped frames) | ✗ Lesson 18: VLM zero-shot structurally insufficient on tiny eye ROI in uncropped video — see [Lesson 18](docs/lessons_learned.md). Confirmed track closure; ROI crop + V-JEPA-2 + LR is the path. |

**Current focus.** Phase 9 prereg drafting — multi-rater κ track (gated on SLU/Palichleb collaboration response, see `v2/outreach/`) and simplified-B1 long-form aggregation per Dyson 2018 presence/absence scoring rule (unblocked by Phase 8c's calibrated probabilities + τ_ear). Phases 8a/8b/8c closed under the locked discipline pattern: **Phase 8a fired RETRACTION** — Phase 7's eye-region OUTPERFORM verdict retracted to UNDERPOWERED_INDISTINGUISHABLE on Phase 8a's bootstrap CI on Δ AUC at n=34. **Phase 8b — COMPETITIVE** at n=283: DLC ear keypoint-anchored cropping preserves whole-frame V-JEPA-2 baseline signal (AUC 0.9008 vs 0.8746) but does not significantly improve over it (paired DeLong p=0.3122 with adequate power; per-clip 40 BEATS vs 39 LOSES near-symmetric). **Phase 8c — WELL_CALIBRATED + G1 STRUCTURAL FINDING**: temperature scaling on Phase 8b output drops ECE from 0.1118 to 0.0397 (Murphy decomposition: 87% from reliability gain, 1.5% from cross-source rank-shuffle resolution loss); τ_ear=0.8138 at FPR=0.05 with TPR=0.5435 [0.31, 0.75]; session-level OP P(≥8|H0) under independence ≈ 1.39e−5 (exact Poisson-binomial — generalizable observation: Poisson approximation overestimates by 166% at n=24 / λ=1.2). Combined post-8c narrative across two RHpE behaviors and three phases: cropping methodology preserves baseline signal but does not significantly improve over simpler baselines; calibration infrastructure now in place for clinical-utility framing under TRIPOD+AI / STARD-AI; multi-rater κ on ≥20% audit subset remains the load-bearing missing methodology step. See [`docs/phase7_audit.md`](docs/phase7_audit.md), [`docs/phase8a_audit.md`](docs/phase8a_audit.md), [`docs/phase8b_audit.md`](docs/phase8b_audit.md), [`docs/phase8c_audit.md`](docs/phase8c_audit.md), [`docs/methodology_discipline_pattern.md`](docs/methodology_discipline_pattern.md).

## Key methodological findings

- **Read My Ears 0.875 holds under source-aware LOSO** (Sanity 5). Replication confirms the paper claim; the random clip-level split happened not to inflate due to visual heterogeneity of their 12 sources. Earlier suspicion that 0.875 was inflated has been falsified empirically.
- **LOO/LOSO gap ~10pp on this dataset.** Any future ear-related LOO result should be mentally adjusted by ~10pp to estimate the LOSO baseline. On the 53-clip DIY data the gap is up to 34pp — small-N + multi-source makes LOO fundamentally unsafe.
- **Conditional bg-masking** ([Lesson 9](docs/lessons_learned.md)). Secondary motion in frame (a second horse, a walking handler) degrades V-JEPA-2 cross-source robustness. YOLO-detected scene motion → switch to bg-masked features (S8 fold: 0.633 → 0.875, **+24pp**); clean scene → unmasked (S12: 1.000 → 0.661 if forced through masking, **−34pp**). Conditional preprocessing, not a global default.
- **DINOv2 alone fails cross-source.** Image-only mean-pooled DINOv2 LOO 0.780 → LOSO 0.514 (chance), with anti-correlation on 4 of 12 sources. Temporal context (V-JEPA-2) is necessary, not nice-to-have.
- **V-JEPA-2 SSv2 fine-tune ≡ pretrain-only checkpoint at the encoder** ([Lesson 12](docs/lessons_learned.md)). All 587 encoder layers are byte-identical between `vjepa2-vitl-fpc16-256-ssv2` and `vjepa2-vitl-fpc64-256` — the SSv2 head is dropped when loading via `VJEPA2Model`. Comparisons of "SSv2 vs PT" in our pipeline measure the same encoder.
- **Per-source temperature scaling sharpens RidgeClassifier under-confidence + surfaces a calibration-LOSO subtlety** (Phase 8c). All 12 fold-specific T values cluster in [0.46, 0.52] (median 0.494) — RidgeClassifier `decision_function` outputs are systematically *underconfident* across all sources by a similar amount; calibration *sharpens* the sigmoid (T < 1). ECE drops 0.1118 → 0.0397; Murphy decomposition shows the gain is overwhelmingly reliability-driven (87% reduction in REL, 1.5% reduction in RES — the small RES drop is the cross-source rank-shuffle artifact). Per-source calibration LOSO does NOT preserve pooled AUC (Δ=−0.0022) — but per-source AUC IS bit-exact invariant 12/12 — confirming the pre-reg's D1 "AUC invariance" claim was internally inconsistent with D2's per-source design (true for global T, broken under per-source T). G1 reported FAIL without softening per discipline pattern; Phase 7 §4 anatomical-mapping precedent applies. Generalizable methodological observation surfaced: at RHpE-relevant scale (n=24 behaviors, λ ≈ 1.2), the standard Poisson approximation to Poisson-binomial tail probabilities overestimates by 1.66× — future RHpE session-level OP analyses should compute exactly, not approximate. See [`docs/phase8c_audit.md`](docs/phase8c_audit.md).

## What works

- **V-JEPA-2 ViT-L encoder features** (1024-d, pretrain-only by construction in our pipeline)
- **Read My Ears protocol** (face mask + ear bbox crop + linear probe) — LOO 0.97, bg-masked LOO 0.91, **LOSO 0.875** (source-invariant on their data)
- **Manual gold-standard eye-region cropping with 3-keyframe annotation + IoU-based interpolation** (Phase 5 v3) — LOSO **0.7985** at n=34 (middle band, single-observer caveat); intra-rater median IoU 0.765; serves both as Phase 5's cropping intervention AND as the validation set for any automated cropping tool evaluated in Phase 6+
- **DLC SuperAnimal-Quadruped keypoint-anchored automated cropping** (Phase 7 corrected) — LOSO **0.8462** at n=34 with horse-anatomical side-assignment mapping; G1 + G2 load-bearing both PASS; 12 DLC_BEATS_FBB / 0 FBB_BEATS_DLC vs Phase 6(b) face-bbox; paired-DeLong vs Phase 5 inconclusive at n=34 (AUC governs per locked G2 asymmetry). The cleanest automation strategy in the project so far. See [`docs/phase7_audit.md`](docs/phase7_audit.md).
- **Cross-behavior generalization of DLC keypoint-anchored cropping** (Phase 8b, ear) — both-ears bbox over 4 DLC ear keypoints with ≥3-of-4 confidence gate + locked single-middle-frame fallback; AUC **0.9008** at n=283 with 12 sources, paired DeLong p=0.3122 vs whole-frame baseline (adequate power → INCONCLUSIVE means *competitive*, not underpowered). 9.2% of clips fired the locked fallback rule mechanically; bimodal pass-rate pattern from Stage 1.5 verified at population scale. Methodology preserves whole-frame baseline signal across two RHpE behaviors (eye + ear); does not significantly improve over simpler baselines. See [`docs/phase8b_audit.md`](docs/phase8b_audit.md).
- **Calibration methodology infrastructure on Phase 8b ear output** (Phase 8c) — temperature scaling per source-aware calibration LOSO (median T 0.494, all 12 in [0.46, 0.52]) drops ECE from 0.1118 to **0.0397** (well_calibrated band, ~65% relative reduction); Brier 0.1420 → 0.1289; τ_ear=0.8138 at FPR=0.05 with B=1000 source-bootstrap CI on TPR [0.3077, 0.7500]; session-level OP P(≥8|H0) under independence ≈ **1.39e−5** (exact Poisson-binomial). Reusable infrastructure (calibration LOSO loop, ECE/Brier/NLL/Wilson-CI machinery, reliability diagram with bin-count underlay, source-bootstrap on TPR, exact Poisson-binomial CDF) packaged in [`tools/phase8c_calibration.py`](tools/phase8c_calibration.py). See [`docs/phase8c_audit.md`](docs/phase8c_audit.md).
- **Linear probe + LOO observed AUC + permutation test + LOSO** as a four-layer evaluation stack
- **Hard pre-committed decision thresholds** for architecture choices (4-level rule before running comparison) — extended in Phase 5/6/7 to a [6-element discipline pattern](docs/methodology_discipline_pattern.md): pre-register, pre-commit failure interpretation, catch bugs in writing, honor mechanical decisions, sequence phases, empirical-anchor. Both Phase 6(b) DISTRIBUTED_FAIL → DLC routing AND Phase 7's two structural findings (§7 + §4) fired mechanically through locked routing matrices, with Stage 2 amendments adjudicated transparently. Strongest single-cycle demonstration: Phase 7's broken-rule run produced AUC 0.5788 + CONFIDENT_MISPLACEMENT_FAIL → prescribed "side-assignment review" → empirical falsification of §4 → Stage 2 v2 correction → AUC 0.8462. See [`docs/phase7_audit.md`](docs/phase7_audit.md).
- **Static-frame collapse diagnostic** for distinguishing temporal vs static feature reliance
- **Conditional background masking** — apply when YOLO detects > 1 subject in frame, skip otherwise

## MLLM-as-classifier — tested cross-vendor, closed track

Two MLLM-as-classifier branches were run as supporting methodological observations to V-JEPA-2 + linear probe — both closed as of May 2026:

- `gemini-augmentation` (merged): Gemini 2.5 Pro + 3.1 Pro Preview tested on a 36-clip stratified Read My Ears subset under three prompts including Google's official Gemini-3.x best-practice config. Three failure modes documented in [Lesson 14](docs/lessons_learned.md) — refusal-bias collapse (35/36 background on 3.1 Pro Preview at best-practice params), cross-rep instability, perception/classification decoupling. Tool: [`tools/gemini_audit.py`](tools/gemini_audit.py). Writeup: [`docs/gemini-integration.md`](docs/gemini-integration.md).
- `experiment/qwen-mlx` (v1, **buggy** — held in PR #2, do not merge as-is) and `fix/qwen-mlx-video-input` (v2, fixed): Qwen2.5-VL-7B-bf16 self-hosted via mlx-vlm 0.5.0 on M2 Max, same 36-clip subset, same prompts (verbatim from `gemini_audit.py`). v1 had a load-bearing bug — `apply_chat_template(prompt=messages_list, ...)` strips video content via `extract_text_from_content`, so the model never received video tokens (prompt_tokens ≈ 110, text-only baseline). v1's "all 3 failure modes reproduced, 36/36 background" was text-only inference disguised as video inference. v2 ([Lesson 16](docs/lessons_learned.md) details the fix) routes video correctly (prompt_tokens ≈ 5500) — the **same §4 row 1 outcome** still holds with one true-action correctly caught (`action_S3.mp4_2_.mp4` on prompt A) and one apparent false-positive (`background_S4.mp4_7_.mp4`) which independent manual review then reframed as a Lesson 9 multi-horse confound — the model perceived real motion on a non-target horse in the frame, not a hallucination ([Lesson 17](docs/lessons_learned.md) stub). Per-source vector v1↔v2 differs by ±1 on only 2 of 12 sources. [Lesson 15](docs/lessons_learned.md) is rewritten on v2 evidence. Specs: [`docs/qwen-experiment-spec.md`](docs/qwen-experiment-spec.md), [`docs/qwen-fix-and-revalidate-spec.md`](docs/qwen-fix-and-revalidate-spec.md). Side-by-side numbers in [`outputs/qwen_vs_gemini_comparison.md`](outputs/qwen_vs_gemini_comparison.md). Tool: [`tools/qwen_audit.py`](tools/qwen_audit.py).

The MLLM-as-classifier track is closed within the scope tested. **V-JEPA-2 + linear probe (LOSO 0.875) remains the spine.** Frontier multimodal LLMs do not replace it on fine-grained motion in the regimes we tested — and they don't reliably augment it as label-noise auditors either.

## What doesn't work

- **5-class softmax on 53 anchor clips** — too small, session-confounded; eye_expression sink-effect
- **head_position 0.898 LOO** as MVP candidate — Sanity 3 LOSO 0.561 = session leakage
- **DINOv2 + V-JEPA-2 concat** — lost on 2 of 4 behaviors in the iter-6 matrix; LOSO 0.747 vs SSv2 0.875 in Sanity 5
- **DINOv2 alone as universal backbone** — LOSO 0.514 on Read My Ears, anti-correlated on 4 of 12 sources
- **Background masking as a global default** — hurts strong sources by ~10pp while helping weak ones; must be conditional
- **The 53-clip DIY anchor dataset as a training set** for any per-behavior classifier (iter 6.5)
- **VLM zero-shot on uncropped horse video for eye-region behavior** (v1+v2+v3 prompts across Gemini 2.5 Pro / 3.1 Pro Preview / Qwen 2.5/3-VL) — three flavors of collapse on the same underlying failure mode; eye occupies ~1–2% of pixel area in uncropped frame, no prompt rescues that geometry. Lesson 18.
- **Face-bbox-positioned crop at single-position median anatomical placement** (Phase 6 (b)) — AUC **0.4689** (below chance, Δ vs Phase 5 = −0.330); median IoU vs Phase 5 manual boxes = **0.165** (23/34 clips off-eye); horse profile orientation gives effectively continuous rel_x distribution that no median preserves. Pre-registered failure-mode attribution routes to DLC (`loss_concentration_pct = 6.2%`, well below 50% orientation-aware threshold). Pipeline robustness wasn't the bottleneck — YOLOv8l face detection succeeded 34/34 clips per-frame at conf=0.5, mean confidence 0.85–0.94, zero interpolation needed; the failure is positional, not detection-based.
- **Percentile-based confidence thresholding (`max(0.5, p25_pooled)`) on high-confidence right-shifted distributions** (Phase 7 §7 meta-rule, falsified) — at p25=0.89 on RME, mechanically applying the rule would drop 8/34 clips by construction (4 of which are V3_NEWLY_RECOVERED diagnostic clips). Case difficulty correlates with model confidence; percentile thresholding disproportionately removes the highest-information clips. Stage 2 amendment v1 hard-locks 0.5; meta-rule reframed as future-phase consideration. Lesson 19.
- **Anatomical-mapping reasoning that conflates image-side with horse-side geometry** (Phase 7 §4 mapping, falsified) — Stage 1 §4 had camera-facing-side reasoning inverted (24/34 RME clips show flipped rule has 2.5× higher mean IoU; Phase 0 Wikimedia confirms DLC labeling is horse-anatomical). Stage 2 amendment v2 flips the keypoint mapping. Lesson 20.

Full methodology trail in [`docs/lessons_learned.md`](docs/lessons_learned.md) — 20 lessons across iter 1–6.5 + Phase 1–7 (eye-region track), including why LOO is not a safe baseline, why sample size has to be counted in sessions not clips, why VLM zero-shot can't substitute for ROI-cropped V-JEPA-2 + linear probe, and the two structural Phase 7 findings about percentile thresholding and keypoint convention verification.

## How to engage

**For ML researchers / academics.** The substantive contributions live in [`docs/lessons_learned.md`](docs/lessons_learned.md): conditional bg-masking with quantified per-source costs (Lesson 9), the two failure modes in cross-source ear movement detection (Lesson 10, including the S8 two-horses confound case study), and the LOSO replication of Read My Ears 0.875 (Lesson 1; raw `outputs/iter65_sanity5_*.json` files are produced locally by `setup.sh` + the notebooks, not committed). Methodology critique welcome via Issues.

**For data contributors.** Field dataset collection is in progress, targeting ≥ 10 horses × 2–3 ear states × 2–3 takes = 60–100 clips across ≥ 10 unique sessions. Read [`docs/recording-protocol.md`](docs/recording-protocol.md) before recording (one page, ~5 min). Welfare > PoC — no provocations, no induced stress, naturalistic training-session footage only. GDPR-compliant consent template included (English + Polish).

**For replication.** [`GATE.md`](GATE.md) documents the original Phase 0 GO/NO-GO criteria (all passed). Quickstart below — `setup.sh` is idempotent, runs on macOS Apple Silicon or Colab T4 fallback, 6 notebooks staged 00 → 99.

## Quickstart (macOS Apple Silicon, local)

```bash
git clone https://github.com/piotrpawluk/horse-pain-poc
cd horse-pain-poc
bash setup.sh
source .venv/bin/activate
jupyter lab notebooks/00_smoke_dlc_sample.ipynb
```

Notebook order: `00` (DLC sanity) → `01` (Read My Ears replication) → `02` (V-JEPA-2 zero-shot) → `04` (few-shot 5-behavior validation; the iter-6.5 caveats above apply). Full results in [`GATE.md`](GATE.md), full methodology in [`docs/lessons_learned.md`](docs/lessons_learned.md).

## Quickstart (Google Colab, fallback)

Open `notebooks/99_colab_fallback.ipynb` in [Google Colab](https://colab.research.google.com/) (File → Upload notebook). Free T4 is sufficient. No local setup required.

## Repo structure

```
.
├── setup.sh                  idempotent installer (uv-based; macOS / Linux)
├── pyproject.toml            pinned deps (DLC 3.0.0rc14, torch 2.11, transformers 5.7, gradio, webvtt-py)
├── GATE.md                   Phase 0 GO/NO-GO criteria + lessons (historical)
├── docs/
│   ├── lessons_learned.md    12 methodological lessons from iter 1-6.5 (must-read)
│   └── recording-protocol.md field data collection protocol (welfare-first, GDPR template)
├── notebooks/
│   ├── 00_smoke_dlc_sample.ipynb            DLC SuperAnimal-Quadruped zero-shot
│   ├── 01_read_my_ears_replicate.ipynb      Read My Ears (CVPR W'25) replication
│   ├── 02_vjepa2_zeroshot.ipynb             V-JEPA-2 + linear probe baseline
│   ├── 03_xclip_zeroshot.ipynb              X-CLIP text-conditioned (negative result)
│   ├── 04_few_shot_rhpe_validation.ipynb    5-behavior few-shot validation (iter 6.5 caveats apply)
│   └── 99_colab_fallback.ipynb              Google Colab T4 backup
├── tools/
│   └── subtitle_search.py                   VTT keyword parser (anchor clipping helper)
└── .gitignore
```

`data/`, `checkpoints/`, `outputs/`, `vendor/` are gitignored — `setup.sh` fetches them (sample horse video from Wikimedia Commons CC, weights from HuggingFace).

## Setup gotchas (for replication)

1. **DLC 3.0** stable not yet released (May 2026); pin `>=3.0.0rc14` with `--prerelease=allow`
2. **matplotlib pin `<3.9`** (DLC requirement)
3. **HF Hub 1.x** removed `huggingface_hub.commands.huggingface_cli` — use the Python API `snapshot_download`
4. **HEVC in iPhone .MOV clips** — OpenCV reads them natively; macOS TCC quarantine xattrs block some files imported from Photos library (`xattr -d com.apple.quarantine <file>` after import)
5. **Inference cost**: V-JEPA-2 ViT-L 16 frames @ 256×256 ≈ 1.3 s/clip on MPS; 283 clips ≈ 6 min

## Methodology stack rationale

- **V-JEPA-2 ViT-L** ([Meta, June 2025](https://arxiv.org/abs/2506.09985)) — foundation video model, 1024-d encoder features. Used as pretrain-only backbone (the SSv2 fine-tune does not modify encoder weights — see [Lesson 12](docs/lessons_learned.md)).
- **DINOv2 large** (image-only, 1024-d) — alternative image-only baseline; in our pipeline ~10pp behind V-JEPA-2 on ear movement and anti-correlated cross-source on 4 of 12 sources.
- **DeepLabCut SuperAnimal-Quadruped** ([Nature Comm 2024](https://www.nature.com/articles/s41467-024-48792-2)) — zero-shot pose estimation for 45+ species; staged for Track C (temporal behaviors, deferred to Phase 3).
- **Read My Ears** (Alves et al., [CVPR W'25](https://arxiv.org/abs/2505.03554)) — baseline ROI pipeline: face mask + ear bbox + classifier; the per-behavior ROI pattern generalizes to other RHpE behaviors.
- **scikit-learn RidgeClassifier / LogisticRegression** — linear probe on cached embeddings, seconds to train on CPU.
- **uv** instead of conda — ~10× faster installer, deterministic resolution.

## Ethics / disclaimer

This is a **research prototype, not a diagnostic tool**. Any clinical application requires validation by a certified RHpE assessor and veterinary consultation. Animal welfare is **not negotiable** — if a horse shows pain signals during data collection, the session stops and the horse goes to a vet. No induced stress, no provocations: data must come from naturalistic training sessions only. The recording protocol enforces this categorically.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- **Mathis Lab** — DeepLabCut + SuperAnimal-Quadruped
- **Alves, Andersen, Zamansky et al.** — Read My Ears (CVPR W'25)
- **Sue Dyson** — RHpE as a clinical framework
- **Wikimedia Commons** — sample horse video under CC license

## Citation

If you cite this repo informally:

> Pawluk, P. (2026). *horse-pain-poc: Automated RHpE scoring with V-JEPA-2 — methodology-first PoC*. GitHub. https://github.com/piotrpawluk/horse-pain-poc

Contact: piotr.pawluk@gmail.com or open an Issue.
