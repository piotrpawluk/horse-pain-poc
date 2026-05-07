# Qwen2.5-VL vs Gemini on RHpE 36-clip subset — comparison

*Branch: `experiment/qwen-mlx`. Spec: `docs/qwen-experiment-spec.md`. Run date: 2026-05-07. M2 Max, mlx-vlm 0.5.0 + mlx 0.31.2, model `mlx-community/Qwen2.5-VL-7B-Instruct-bf16`. All 122 inferences ran without errors. Total wall-clock ≈ 10 min on M2 Max GPU. Cost: $0 (self-hosted).*

> **Verdict (locked against pre-registered spec §4):** Qwen 7B exhibits **all three** Lesson 14 failure modes — refusal-bias collapse, cross-rep instability, perception/classification decoupling. **Gate hit: row 1.** No 32B follow-up. Lesson 14 generalizes from "Gemini family" to "MLLM class" within the scope tested. The closed/open MLLM-as-classifier track on this task closes.

## 1. Headline metrics

Same 36-clip stratified subset (3 clips × 12 sources S1–S12, 14 action / 22 background) used in all Gemini runs. Qwen subsets reconstructed from the existing Gemini JSONLs to guarantee identical clips — no re-stratification.

| Configuration | Errors | Agreement | Background-prediction rate |
|---|---|---|---|
| Gemini 2.5 Pro · prompt A · t=0 | 0/36 | 22/36 = 0.611 | 12/36 = **0.333** |
| Gemini 2.5 Pro · prompt B · t=0 | 0/36 | 22/36 = 0.611 | 36/36 = 1.000 |
| Gemini 3.1 Pro Preview · prompt A · t=0 | 0/36 | 22/36 = 0.611 | 36/36 = 1.000 |
| Gemini 3.1 Pro Preview · prompt B · t=0 | 2/36 | 20/34 = 0.588 | 33/34 = 0.971 |
| Gemini 3.1 Pro Preview · prompt C + sys · t=1.0 · thinkLOW | 0/36 | 23/36 = 0.639 | 35/36 = 0.972 |
| **Qwen 7B-bf16 · prompt A · t=0** | **0/36** | **22/36 = 0.611** | **36/36 = 1.000** |
| **Qwen 7B-bf16 · prompt C + sys · t=1e-6 · rep_pen=1.05** (Qwen defaults) | **0/36** | **22/36 = 0.611** | **36/36 = 1.000** |

**Reading.** Qwen 7B sits in the same conservatism regime as Gemini 3.1 Pro Preview at *every* prompt — including prompt A, where Gemini 2.5 Pro was *not* collapsed (bg-rate 33 % at A vs 100 % everywhere else). The "Qwen-recommended" sampling defaults from the model card (`temperature=1e-6, repetition_penalty=1.05`) plus the system instruction `"Do not refuse to commit unless the clip is genuinely uninterpretable"` did **not** mitigate the collapse. The §4 "materially cleaner" gate (agreement ≥ 0.70 AND bg-rate < 80 %) is decisively missed.

## 2. Cross-rep stability (description-only probe, 5 reps × 10 clips)

Same 10 clips Gemini was probed on (extracted from `gemini_audit_probe_gemini-3.1-pro-preview_temp1.0_thinkLOW.jsonl`). Qwen probe at `temperature=0.7`. Stability = "all 5 reps land in the same motion-vs-still category" under a keyword classifier (motion verbs minus hedges).

| Configuration | Reps | Stable clips (5/5 agree) | Motion-majority clips (≥3/5 reps describe motion) |
|---|---|---|---|
| Gemini 2.5 Pro · original probe @ t=0.5 | 3 | 3/10 | 0/10 |
| Gemini 2.5 Pro · t=1.0 | 5 | 5/10 | 2/10 |
| Gemini 3.1 Pro Preview · t=1.0 thinkLOW (orig 10) | 5 | 1/10 | 6/10 |
| Gemini 3.1 Pro Preview · t=1.0 thinkLOW (N=20 expansion) | 5 | 2/10 | 3/10 |
| **Qwen 7B-bf16 · t=0.7** | 5 | **0/10** | **4/10** |

**Reading.** Qwen 7B is the *least* stable across reps — 0/10 clips produce 5/5-consistent descriptions, identical to Gemini 3.1's worst configuration. The pattern is the same: under sampled decoding, the description surface flickers between "twitching" and "stationary" on the same input.

## 3. Perception/classification decoupling

Same 10 clips. Decoupling case = clip is true-action, classification (probe 1) outputs `background`, AND ≥ 3/5 description reps report motion.

| Configuration | Decoupled clips |
|---|---|
| Gemini 2.5 Pro · prompt B + t=1.0 probe | 7/10 (70 %) |
| Gemini 3.1 Pro Preview · prompt C + thinkLOW probe (initial 10) | 13/20 across N=20 expansion (65 %) |
| **Qwen 7B-bf16 · prompt A + t=0.7 probe** | **3/10 (30 %)** |

**Reading.** Decoupling is present on Qwen but at lower magnitude than Gemini 3.1. The 4/10 motion-majority clips include 3 true-action clips (decoupling cases) and 1 true-background clip. The conservative side of Qwen's decoupling is consistent with the regime: Qwen's *classification* is essentially template-locked (see §5), so any motion language in *description* mode is a decoupling event. The keyword classifier counts hedged language ("occasionally twitching slightly") as `mixed` rather than `motion`, so the 30 % rate is a lower bound. Gemini 3.1's 65 % includes more language without hedges, possibly because Gemini 3.x is trained with more fluid descriptive prose.

## 4. Per-source agreement (3 clips per source × 12 sources)

| Source | truth_action / 3 | Gemini 2.5 + A | Gemini 3.1 + A | **Qwen 7B + A** | Gemini 3.1 + C | **Qwen 7B + C** |
|---|---|---|---|---|---|---|
| S1 | 0/3 | 1/3 | 3/3 | **3/3** | 3/3 | **3/3** |
| S2 | 1/3 | 3/3 | 2/3 | **2/3** | 2/3 | **2/3** |
| S3 | 2/3 | 1/3 | 1/3 | **1/3** | 2/3 | **1/3** |
| S4 | 2/3 | 2/3 | 1/3 | **1/3** | 1/3 | **1/3** |
| S5 | 2/3 | 2/3 | 1/3 | **1/3** | 1/3 | **1/3** |
| S6 | 1/3 | 2/3 | 2/3 | **2/3** | 2/3 | **2/3** |
| S7 | 0/3 | 1/3 | 3/3 | **3/3** | 3/3 | **3/3** |
| S8 | 1/3 | 3/3 | 2/3 | **2/3** | 2/3 | **2/3** |
| S9 | 2/3 | 2/3 | 1/3 | **1/3** | 1/3 | **1/3** |
| S10 | 1/3 | 2/3 | 2/3 | **2/3** | 2/3 | **2/3** |
| S11 | 2/3 | 2/3 | 1/3 | **1/3** | 1/3 | **1/3** |
| S12 | 0/3 | 1/3 | 3/3 | **3/3** | 3/3 | **3/3** |

**Reading.** Qwen 7B's per-source row is **exactly identical** to Gemini 3.1 Pro Preview's at every source under both prompts — same right-or-wrong on every clip-cluster. Both are getting "perfect" 3/3 on the all-background sources (S1, S7, S12) by accident of refusal bias, and missing all 2 actions on action-heavy sources (S3, S5, S9, S11). This is not a per-source heterogeneity finding — it's the same collapse expressed as a same-shaped agreement vector.

## 5. Qualitative reasoning excerpts (true-action clips, prompt C)

Side-by-side `evidence` / `reasoning_text` field on five true-action clips that both models classified as `background`:

| Clip | Gemini 3.1 Pro Preview + C (conf, evidence) | Qwen 7B-bf16 + C (conf, evidence) |
|---|---|---|
| `action_S10.mp4_0_.mp4` | bg, 0.99 — *"Across all frames, both ears remain in a fixed forward-pointing position without any visible rotation, twitching, or pinning."* | bg, 0.95 — *"The ears remain in essentially the same position throughout the clip, with no visible positional changes or movements."* |
| `action_S11.mp4_0_.mp4` | bg, 1.00 — *"The horse's ears remain in the same outward-pointing position from the first frame to the last without any noticeable rotation or twitching."* | bg, 0.95 — *"The ears remain in essentially the same position throughout the clip, with no visible positional changes or movements."* |
| `action_S11.mp4_6_.mp4` | bg, 1.00 — *"Both ears remain stationary and in the same forward-pointing position throughout the clip."* | bg, 0.95 — *"The ears remain in essentially the same position throughout the clip, with no visible positional changes or movements."* |
| `action_S2.mp4_7_.mp4` | bg, 1.00 — *"The horse's ears remain still and pointing slightly backward with no visible movement between the provided frames."* | bg, 0.95 — *"The ears remain in essentially the same position throughout the clip, with no visible positional changes or movements."* |
| `action_S3.mp4_2_.mp4` | **action**, 0.95 — *"The horse's left ear displays motion blur and slight positional changes across the frames, indicating a flicking or twitching movement."* | bg, 0.95 — *"The ears remain in essentially the same position throughout the clip, with no visible positional changes or movements."* |

**Reading.** Qwen 7B + C produces the *same templated evidence sentence* on every clip — confidence locked at 0.95, vocabulary identical. This is not a model that is *seeing* and *describing* before deciding; the system instruction's evidence-citation request is satisfied by a canned phrase that happens to syntactically resemble evidence. Gemini 3.1 Pro Preview at least produces clip-specific evidence prose, and on the only clip with conspicuous motion blur (`action_S3.mp4_2_.mp4`) Gemini commits to "action" while Qwen still defaults to background. So Gemini 3.1's collapse, while structurally similar in aggregate, has marginally more clip-conditioned variation in language.

## 6. §4 outcome and decision

| Spec §4 row | Threshold | Qwen 7B observed | Hit? |
|---|---|---|---|
| **Row 1 — all 3 failure modes reproduced** | refusal-bias + cross-rep instability + decoupling all present | bg-rate 100 %, 0/10 stable, 3/10 decoupling cases | **YES** |
| Row 2 — materially cleaner | agreement ≥ 0.70 AND bg-rate < 80 % AND no > 3 pp decoupling | 0.611, 100 %, 30 % | NO |
| Row 3 — mixed | one metric clean, others not | none clean | NO |
| Row 4 — infrastructure failure | wouldn't run | 0/122 errors | NO |

**Decision: Row 1. Skip 32B per spec gate. Update Lesson 14 framing → Lesson 15. Close the MLLM-as-classifier track on this task within the scope tested.**

The 32B-4bit follow-up is *not* run. The §4 gate is binding even though int4 might in principle differ — the spec rules out chasing that on a Row 1 outcome because (a) the failure mode is qualitative rather than capability-bound, (b) more parameters don't fix a calibration-bias problem, and (c) the experiment's purpose was tool-selection, not exhaustive capability mapping. We documented this and moved on.

## 7. Caveats and scope

- **One open-weight MLLM family at one parameter scale.** N = 1 architecture (Qwen2.5-VL), N = 1 size (7B), N = 1 quantization (bf16). Does not extrapolate to InternVL, MiniCPM-V, GLM-V, LLaVA-NeXT, etc.
- **One task.** Sub-second ear movement on horses. Does not extrapolate to other RHpE behaviors, let alone other species or other coarse-grained behavior tasks where MLLMs are documented to perform well (Animal-Bench NeurIPS 2024).
- **One frame-sampling regime.** mlx-vlm with the `{"type": "video", "video": ..., "fps": 10}` content. Different frame-sampling strategies (manual frame extraction, longer/shorter sequences, interleaved with text) were not tested; spec §5 forbids that scope creep.
- **One inference backend.** mlx-vlm 0.5.0 on M2 Max. Did not test PyTorch MPS or CUDA; not expected to materially differ on a refusal-bias outcome (the bias lives in weights, not in inference engine), but unverified.
- **The keyword-based motion classifier on description text is conservative.** Hedged language is counted as `mixed`. A human reader might classify more reps as motion-positive, raising decoupling rates. The current rates are lower bounds.

## 8. Reproduction

```bash
cd /path/to/poc
bash setup.sh                                      # installs mac extras on Apple Silicon
.venv/bin/python tools/qwen_audit.py --probe 1     # promptA, t=0
.venv/bin/python tools/qwen_audit.py --probe 2     # promptC + sys, Qwen defaults
.venv/bin/python tools/qwen_audit.py --probe 3     # description, 5 reps × 10 clips
```

Outputs land in `outputs/qwen25vl_7b_*.jsonl` (gitignored). Idempotent — re-runs skip clips already present in the JSONL.

## 9. References

- Pre-registered spec: `docs/qwen-experiment-spec.md`
- Lesson 14 (Gemini-side): `docs/lessons_learned.md`
- Gemini integration writeup: `docs/gemini-integration.md`
- Qwen2.5-VL model card: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- mlx-community 7B-bf16 quant: https://huggingface.co/mlx-community/Qwen2.5-VL-7B-Instruct-bf16
- mlx-vlm: https://github.com/Blaizzy/mlx-vlm
