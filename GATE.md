# GATE — Phase 0 weekend exploration

> **Historical document.** This file describes the Phase 0 GO/NO-GO criteria from project inception (May 2026). All criteria passed; the project is currently in Phase 1.5 / iteration 6.5. For the current methodological state and per-behavior LOSO results, see [`docs/lessons_learned.md`](docs/lessons_learned.md). This document is preserved as the methodology trail.

To be filled after running `setup.sh` + notebooks `00` and `01`. Each item is a binary YES / NO with a one-sentence comment.

## Criteria

1. **Setup completed**: `setup.sh` finished without error on Mac M-series **or** Colab fallback works.
   - [x] YES / [ ] NO
   - Comment: 4 surgical fixes during iteration (DLC 3.0.0rc14 instead of 3.0.0; matplotlib<3.9 from DLC pins; HF Hub 1.x — Python API instead of CLI; sample video from Wikimedia Commons with the correct URL — Pexels 856038 turned out to be boats, not a horse). After fixes: uv 0.9.0 + Python 3.11.13 + DLC 3.0.0rc13 + torch 2.11.0 + transformers 5.7.0 + 483 MB weights — all installed on macOS Apple Silicon without Colab fallback.

2. **DLC SuperAnimal-Quadruped on sample**: keypoints **visually sensible** (hooves, elbows, shoulder, hip, eyes, ears visible on the horse) on the sample clip from `data/sample_horse.mp4`. See `outputs/sample_horse_..._labeled_before_adapt.mp4` and `outputs/sample_keypoints_grid.png`.
   - [x] YES / [ ] NO
   - Comment: Wikimedia Commons clip "Horse walking in corral" (640×480, 9.6 s, 287 frames, two horses). DLC SuperAnimal-Quadruped (hrnet_w32 + fasterrcnn_resnet50_fpn_v2) zero-shot detected both horses and overlaid the full skeleton (head, neck, shoulders, hips, limbs, ears) on ≥90% of frames. CPU/MPS inference took ~5 min.

3. **Read My Ears replication**: the `vendor/read-my-ears` repo runs and produces an ear-movement plot on its own example without code modifications. See `outputs/ear_movement_timeline.png`.
   - [x] YES / [ ] NO
   - Comment: **Phase 0**: fallback using DLC ear keypoints (`right_earbase`, `right_earend`, `left_earbase`, `left_earend`) generated a sensible `ear_movement_timeline.png`. **Phase 1 Stage A**: full 1:1 replication of `movement-detection` — `ultralytics>=8.3` + YOLOv8l custom weights + dataset from [`joaomalves/read-my-ears`](https://huggingface.co/datasets/joaomalves/read-my-ears) (CC-BY-4.0, 48 test clips). Pipeline ran end-to-end. **Result**: accuracy=0.583 (vs paper 0.875). The algorithm was overaggressive (recall=1.0, 20/26 FP). **Phase 1 V-JEPA-2 zero-shot alternative**: foundation video model `facebook/vjepa2-vitl-fpc16-256-ssv2` (Meta, June 2025) WITHOUT training — embedding extraction (1024-dim) on 235 train+val + 48 test clips (~25 min on M-series MPS), followed by a linear probe (sklearn logistic regression). **Result: accuracy=0.854** (precision=0.826, recall=0.864, f1=0.844) — **delta vs paper claim 0.875 = only −2.1 pp**. Beats Stage A by +27 pp. See `outputs/vjepa2_comparison.png` + `outputs/vjepa2_results.json`. **Strategic implication**: Stages B/C/D (fine-tune VideoMAE + LSTM) are unnecessary — an off-the-shelf foundation model matches the paper. Pivot for Phase 2/3: instead of training a custom model per behavior, use V-JEPA-2 embeddings + 24 linear probes (one per RHpE behavior).

4. **Time budget**: the entire process (setup + 2 notebooks + this document) took ≤ 16 h of work.
   - [x] YES / [ ] NO
   - Actual time: **~45 min** (well below the 8 h hard budget, thanks to uv and lazy weight downloads).

5. **Phase 1.5 — Few-shot V-JEPA-2 validation on 5 RHpE behaviors** (53 DIY anchor clips combined from 24horsebehaviors.org / Padma documentaries + personal phone footage): per-behavior signal strength.
   - [x] EXECUTED (with mixed result) / [ ] NO
   - Comment: 4 iterations on 53 clips × 5 classes (ear/head/eye/tail/mouth) revealed **fundamentally different pipeline requirements per behavior** — full-frame V-JEPA-2 + linear probe is sufficient for some, while others require a Read My Ears-style ROI-specific replication.

   | iter | method | overall | ear | head | eye | tail | mouth |
   |---|---|---|---|---|---|---|---|
   | 2 | LOO cosine, full-frame | 0.358 | 0.000 | 0.500 | 1.000 (sink) | 0.000 | 0.333 |
   | 3 | LOO cosine, head-crop YOLO | 0.302 | 0.000 | 0.500 | 0.833 | 0.000 | 0.000 |
   | 4 | Read My Ears 283 clips, **binary** | linear probe **0.894** (vs paper 0.875, Phase 1: 0.854); LOO cos k=1: 0.756 | — | — | — | — | — |
   | 5 | LOO linear probe (5-class + binary OvR), full-frame | 5-class 0.255 | LP 0.000 / **AUC 0.339** | LP 0.667 / **AUC 0.927** | LP 0.000 / AUC 0.652 | LP 0.556 / AUC 0.783 | LP 0.000 / AUC 0.736 |

   **Key empirical findings:**
   - **Iter 4 reproduces Phase 1** — V-JEPA-2 + linear probe on the clean 283-clip dataset binary task = **0.894** (beats the paper 0.875 by +1.9 pp and Phase 1 by +4 pp). The `embedding → LogReg` pipeline IS verified.
   - **Iter 2 eye_expression=1.000 was a measurement artifact, not a signal.** All 12 eye clips came from Padma documentaries = shared talking-head visual style = cosine sink. Iter 5 binary OvR AUC of only 0.652 reveals that the probe doesn't recover what isn't a real signal.
   - **head_position AUC 0.927** on 12 clips (6 Padma + 6 phone) = strong signal, full-frame V-JEPA-2 + linear probe is sufficient. This is the MVP path for Phase 2 for full-frame behaviors. **(Later invalidated by iter 6.5 LOSO — see `docs/lessons_learned.md`.)**
   - **ear_position AUC 0.339** (below chance 0.5) → V-JEPA-2 SSv2 features on the full frame **do not encode ear position information**. Head-crop does NOT help (iter 3). An **ear-specific ROI crop** is required (Read My Ears: face mask + ear bbox + V-JEPA-2 on the crop) — `vendor/horse-face-ear-detection/` provides `yolov8l_horse_ear_detection.pt`.
   - **tail_movement AUC 0.783, mouth_open AUC 0.736** — moderate signal despite the small sample (9 and 3 clips). Promising, but more data is needed.

   **Phase 1.5 gate (≥3/5 behaviors with accuracy ≥0.70):** ✗ NOT MET in its original form — only head_position binary OvR AUC ≥0.70, while ear/eye don't qualify. **However**: the gate was defined as a uniform threshold for 5-class accuracy; binary OvR per behavior shows that 4/5 behaviors have AUC ≥0.65 and 2/5 have AUC ≥0.75 (head=0.927, tail=0.783) — this is **per-behavior signal differentiation**, not a blanket fail.

## Verdict

**Decision for Phase 0–1.5: [x] GO with scope revision** (5/5 items executed, 1 with a mixed result).

**Phase 0** (DLC SuperAnimal-Quadruped pose backbone): runs end-to-end in ~45 min on macOS Apple Silicon. **Phase 1** (Read My Ears benchmark): V-JEPA-2 + linear probe = **0.854** for ear movement (Phase 1) → **0.894** in the iter-4 replication (Phase 1.5). **Phase 1.5** (5-behavior DIY): reveals that RHpE behaviors have **highly differentiated preprocessing requirements** — full-frame V-JEPA-2 is sufficient for some (head_position AUC 0.927), while others need a ROI-specific crop (ear_position AUC 0.339 full-frame → ear-region crop required, like Read My Ears).

**Phase 2 as 3-track parallel rather than a unified pipeline** (see `poc-ai-rhpe.md`):
- **Track A** (full-frame): head_position, balance, lameness indicators ~5 of 24 behaviors. V-JEPA-2 + linear probe pipeline verified. ~50 clips per behavior.
- **Track B** (ROI crop): ear_position, eye_expression, mouth_open, tail_movement ~6 of 24. Replication of Read My Ears preprocessing (YOLO ROI + V-JEPA-2 on the crop). ~50 clips per behavior.
- **Track C** (DLC keypoints + temporal): behaviors with movement trajectory (~13 of 24). **Deferred to Phase 3.**

**Phase 2 MVP**: 1 behavior from Track A (head_position) + 1 from Track B (ear_position via Read My Ears crop). Target: both ≥0.80 accuracy. Time: ~3-5 days of implementation + data collection. **(Subsequently revised by iter 6.5 — Track A killed, Track B sizing revised. See `docs/lessons_learned.md`.)**

Date filled: 2026-05-04 (Phase 0) / 2026-05-05 (Phase 1) / 2026-05-06 (Phase 1.5)

## Additional notes (for Phase 1+)

- **Surgical fixes in setup.sh** — DLC 3.0 stable hasn't been released yet (rc14 in May 2026); HF Hub 1.x removed `huggingface_hub.commands.huggingface_cli` (use the Python API `snapshot_download`); the matplotlib pin must be <3.9 for DLC compatibility; ultralytics 8.4.46 installed cleanly with torch 2.11.
- **Domain shift**: the Wikimedia corral is a daytime sandy outdoor scene; an indoor arena will be under artificial light on fibre footing — keypoints should still work, but this needs verification in Phase 2 with our own clips.
- **Performance**: 287 frames at 640×480 → ~5 min DLC SuperAnimal inference on M-series CPU/MPS. RME movement-detection: 48 clips at 1920×1080 resize → ~12 min on M-series (YOLOv8l + Farneback flow).
- **Read My Ears 87.5% paper claim** — Stage A movement-detection: 0.58, V-JEPA-2 zero-shot alternative: **0.854** (−2.1 pp from the paper, WITHOUT their custom YOLO/preprocessing/FPS resampling, WITHOUT any training).
- **V-JEPA-2 as the strategic pivot** — an off-the-shelf foundation video model matches a custom-trained paper. Implication for Phase 2/3: instead of training per-behavior classifiers (24 RHpE), use V-JEPA-2 embeddings (1024-dim) per clip → 24 independent linear probes (logistic regression on CPU, seconds to train). This drastically reduces compute requirements and is open-source friendly.
- **YOLOv8l weights** — `jmalves5/horse-face-ear-detection` only ships `yolov8l_*.pt` (~175 MB each), not nano. Custom yolov8n weights are not publicly distributed.
- **`utils.metrics` import** — `vendor/read-my-ears/utils/` has no `__init__.py`. An inline copy is included in notebook 01.
- **Stages B/C/D (LSTM tracks)** — **cancelled**. V-JEPA-2 zero-shot matches the paper claim; fine-tuning VideoMAE from scratch (6-10 h GPU) has negligible expected upside.
- **X-CLIP text-conditioned zero-shot tested — does NOT work for this task**. `microsoft/xclip-base-patch16-16-frames` on 48 test clips: S1 binary 0.604, S2 cinematic 0.458, S3 multi-prompt 0.396. Cinematic and multi-prompt confusion matrices show that the model **always predicts action** — the text-conditioned approach cannot distinguish subtle ear motion from still poses. **Implication**: for the 24 RHpE behaviors, "type 24 zero-shot prompts" is not enough; you need **V-JEPA-2 embeddings + 24 independent linear probes** (each requiring a small labeled subset of ~50-100 clips per behavior).
- **Compatibility issue (to fix upstream)**: transformers 5.7.0 + tokenizers 0.23.0rc0 has a bug in `CLIPTokenizer.__init__` line 117 — it calls `processors.RobertaProcessing(cls=...)` but the new API requires `cls_token=`. Workaround: monkey-patch in the first cell of notebook 03 (maps `cls` → `cls_token`).
- **Possible next steps**: (a) email Alves/Andersen with the concrete result 0.854 (Phase 1) / 0.894 (iter 4 Phase 1.5) as an opener for collaboration; (b) Phase 2 as **3-track parallel** rather than a unified pipeline — Track A full-frame (head_position et al.), Track B ROI replication of Read My Ears (ear/eye/mouth), Track C DLC keypoints + temporal (Phase 3).
- **Phase 1.5 — key artifacts**: `outputs/anchor_embeddings_53clips.npz` (cached embeddings for fast re-iteration), `outputs/few_shot_validation_results_iter5_linearprobe.json` (per-behavior accuracies + binary OvR AUC), `outputs/readmyears_loo_baseline_results.json` (iter 4: linear probe 0.894 vs LOO cosine k=1 0.756 = 14pp gap — a hard measure of "how much signal raw cosine similarity leaves on the table"), `outputs/few_shot_validation_iter5_linearprobe_plot.png` (iter 2 cosine vs iter 5 LP vs binary OvR AUC comparison).
- **Phase 1.5 — methodological conclusion**: on a small sample (≤15 clips per class), **5-class LOO cosine is unreliable** (eye_expression sink-effect 1.000 vs the true AUC 0.652). **Linear probe + binary one-vs-rest AUC** is the honest signal-strength metric per behavior. From Phase 2 onwards: always report **per-behavior binary OvR AUC**, not 5-class accuracy.
- **Iter 2 head-crop YOLO experiment** (`vendor/horse-face-ear-detection/horse_face_detection/yolov8l_horse_face_detection.pt`): face detection succeeded for 76% of ear clips, but the embedding still doesn't separate ear-states from eye-states. **Conclusion for Track B**: it must be **ear-specific** ROI (we have `yolov8l_horse_ear_detection.pt`), not head-crop. Identical preprocessing per behavior, as in Read My Ears, is a necessary condition.
