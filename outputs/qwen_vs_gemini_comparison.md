# Qwen2.5-VL vs Gemini on RHpE 36-clip subset — comparison (v1 + v2)

> **⚠ Critical update 2026-05-07.** This document was originally written against the v1 Qwen JSONLs in PR #2. A code review of `tools/qwen_audit.py` against mlx-vlm 0.5.0 revealed that v1 had a load-bearing bug: the `apply_chat_template(prompt=messages_list, ...)` call hit a code path that strips multimodal content via `extract_text_from_content`, so **the model never received video tokens** — `prompt_tokens ≈ 111` (text-only baseline) instead of the expected ~5500 (text + video tokens). The v1 outputs are text-only inference disguised as video inference. Branch `fix/qwen-mlx-video-input` corrects the call shape (string-prompt + `video=` kwarg + `num_videos=1`), and the v2 outputs are produced with video genuinely reaching the model. Both v1 and v2 numbers are kept here as the dated record. **Pre-registered §4 row 1 still hits on v2** — but for legitimate reasons now. See [Lesson 16](../docs/lessons_learned.md) for the bug and [Lesson 15](../docs/lessons_learned.md) for the rewritten generalization claim.

*v1 branch: `experiment/qwen-mlx`. v2 branch: `fix/qwen-mlx-video-input`. Spec: `docs/qwen-experiment-spec.md` and `docs/qwen-fix-and-revalidate-spec.md`. Run dates: 2026-05-07 (both). M2 Max, mlx-vlm 0.5.0 + mlx 0.31.2, model `mlx-community/Qwen2.5-VL-7B-Instruct-bf16`. v1: 122 inferences in 10 min, $0. v2: 122 inferences in ~57 min, $0 (~5× slower because video is now actually being processed).*

> **Verdict (locked against pre-registered spec §4, evaluated on v2):** Qwen 7B v2 exhibits **all three** Lesson 14 failure modes — refusal-bias collapse, cross-rep instability, perception/classification decoupling. **Gate hit: row 1.** No 32B follow-up. Lesson 14 generalizes from "Gemini family" to "MLLM class" within the scope tested — and now the supporting evidence is real model behavior, not a tokenization artifact.

## 1. Headline metrics

Same 36-clip stratified subset (3 clips × 12 sources S1–S12, 14 action / 22 background) across all configurations. Qwen subsets reconstructed from the existing Gemini JSONLs to guarantee identical clips. Agreement and bg-rate are computed after a vocabulary normalization step on Qwen v2 prompt A only — `still` (which Qwen sometimes outputs, echoing the prompt's `STILL` wording) is mapped to `background` since they are semantically equivalent in context. Raw vocabulary is preserved in the JSONL `qwen_label` field; normalization is documented in `tools/qwen_audit.py` notes and applies only at agreement-computation time.

| Configuration | Errors | Agreement (normalized) | Background-prediction rate (normalized) | Raw vocabulary |
|---|---|---|---|---|
| Gemini 2.5 Pro · prompt A · t=0 | 0/36 | 22/36 = 0.611 | 12/36 = **0.333** | action/background only |
| Gemini 2.5 Pro · prompt B · t=0 | 0/36 | 22/36 = 0.611 | 36/36 = 1.000 | action/background only |
| Gemini 3.1 Pro Preview · prompt A · t=0 | 0/36 | 22/36 = 0.611 | 36/36 = 1.000 | action/background only |
| Gemini 3.1 Pro Preview · prompt B · t=0 | 2/36 | 20/34 = 0.588 | 33/34 = 0.971 | action/background only |
| Gemini 3.1 Pro Preview · prompt C + sys · t=1.0 · thinkLOW | 0/36 | 23/36 = 0.639 | 35/36 = 0.972 | action/background only |
| **Qwen 7B v1 · prompt A** *(buggy)* | 0/36 | 22/36 = 0.611 | 36/36 = 1.000 | `background` × 36 (templated) |
| **Qwen 7B v1 · prompt C** *(buggy)* | 0/36 | 22/36 = 0.611 | 36/36 = 1.000 | `background` × 36 (templated) |
| **Qwen 7B v2 · prompt A · t=0** | **0/36** | **22/36 = 0.611** | **34/36 = 0.944** | `still` × 18, `background` × 16, `action` × 2 |
| **Qwen 7B v2 · prompt C + sys · t=1e-6 · rep_pen=1.05** | **0/36** | **21/36 = 0.583** | **35/36 = 0.972** | `background` × 35, `action` × 1 |

**Reading the v1 → v2 transition.** v1 produces `background` × 36 with the same templated reasoning sentence on every clip (confidence locked at 0.95) — no perception, just the model's prior on a text-only prompt that mentions horses and ear regions. v2 produces clip-specific reasoning with vocabulary leakage on prompt A (the prompt's prose enumerates "MOVING ... or holding them STILL", and Qwen sometimes echoes "still" rather than the schema's "background") and includes 1 correctly-classified true-action clip (`action_S3.mp4_2_.mp4` on prompt A) plus 1 false-positive on a true-background clip (`background_S4.mp4_7_.mp4` on both prompts). v1 was structurally incapable of producing a false positive because everything collapsed to bg.

**Reading vs Gemini.** Qwen 7B v2 sits in the same conservatism regime as Gemini 3.1 Pro Preview at every prompt. Prompt A on v2 (94 % bg-rate after normalization) is slightly less collapsed than Gemini 3.1+A (100 %) but still fails the §4 < 80 % gate. Prompt C on v2 (97 %) matches Gemini 3.1+C (97 %) very closely. The "Qwen-recommended" defaults from the model card (`temperature=1e-6`, `repetition_penalty=1.05`) plus the inlined system instruction `"Do not refuse to commit unless the clip is genuinely uninterpretable"` (inlined into the user-text string per the mlx-vlm 0.5.0 system+video routing bug — see [Lesson 16](../docs/lessons_learned.md)) did **not** mitigate the collapse. The §4 "materially cleaner" gate (agreement ≥ 0.70 AND bg-rate < 80 %) is decisively missed on v2 just as on v1.

## 2. Cross-rep stability (description-only probe, 5 reps × 10 clips)

Same 10 clips Gemini was probed on. Qwen probe at `temperature=0.7`. Stability = "all 5 reps land in the same motion-vs-still category" under a keyword classifier (motion verbs minus hedges; see `tools/qwen_audit.py`).

| Configuration | Stable clips (5/5 agree) | Motion-majority clips (≥3/5 reps describe motion) |
|---|---|---|
| Gemini 2.5 Pro · t=1.0 | 5/10 | 2/10 |
| Gemini 3.1 Pro Preview · t=1.0 thinkLOW (orig 10) | 1/10 | 6/10 |
| Gemini 3.1 Pro Preview · t=1.0 thinkLOW (N=20 expansion) | 2/10 | 3/10 |
| **Qwen 7B v1 · t=0.7** *(buggy)* | 0/10 | 4/10 |
| **Qwen 7B v2 · t=0.7** | **0/10** | **4/10** |

**Reading.** v1 and v2 produce identical aggregate stability/motion-majority counts despite v1 being text-only and v2 being genuinely multimodal — a reminder that the keyword classifier sees only the description text, and Qwen's stylistic default ("upright, alert, with minimal/slight rotation") triggers the `mixed` category by mixing motion-verbs with hedges in roughly the same proportions whether the model is hallucinating from text alone or describing from frames. The substantively different observation is qualitative: v2 descriptions reference clip-specific details (e.g. "left ear slightly rotated", "perked forward", "minor twitching") while v1 descriptions are generic. Both produce 0/10 cross-rep-stable clips. Qwen 7B remains the least stable across reps among all tested configurations.

## 3. Perception/classification decoupling

Same 10 clips. Decoupling case = clip is true-action, classification (prompt A) outputs `background` (after normalization), AND ≥ 3/5 description reps report motion.

| Configuration | Decoupled clips |
|---|---|
| Gemini 2.5 Pro · prompt B + t=1.0 probe | 7/10 (70 %) |
| Gemini 3.1 Pro Preview · prompt C + thinkLOW probe (N=20) | 13/20 (65 %) |
| **Qwen 7B v1 · prompt A + t=0.7 probe** *(buggy)* | 3/10 (30 %) |
| **Qwen 7B v2 · prompt A + t=0.7 probe** | **2/10 (20 %)** |

**Reading.** Decoupling is present on Qwen v2 at slightly lower magnitude than v1 (one fewer case, due to v2 picking up `action_S3.mp4_2` correctly under prompt A — that clip was a v1 decoupling case but is now a v2 agreement case). Decoupling magnitude on Qwen is well below Gemini 3.1's 65 %, but the §4 spec only required *presence* of decoupling, which is unambiguous at 2/10. The keyword classifier remains conservative on hedged language; the 20 % rate is a lower bound.

## 4. Per-source agreement (3 clips per source × 12 sources, normalized)

Same patterns hold under v2 as v1 — the vector differs by ±1 on only 2 of 12 sources.

| Source | truth_action / 3 | Gemini 3.1 + A | **Qwen v1 + A** | **Qwen v2 + A** | Δ v1→v2 | Gemini 3.1 + C | **Qwen v1 + C** | **Qwen v2 + C** | Δ v1→v2 |
|---|---|---|---|---|---|---|---|---|---|
| S1 | 0/3 | 3/3 | 3/3 | 3/3 | — | 3/3 | 3/3 | 3/3 | — |
| S2 | 1/3 | 2/3 | 2/3 | 2/3 | — | 2/3 | 2/3 | 2/3 | — |
| S3 | 2/3 | 1/3 | 1/3 | **2/3** | **+1** ✓ | 2/3 | 1/3 | 1/3 | — |
| S4 | 2/3 | 1/3 | 1/3 | **0/3** | **−1** ✗ | 1/3 | 1/3 | **0/3** | **−1** ✗ |
| S5 | 2/3 | 1/3 | 1/3 | 1/3 | — | 1/3 | 1/3 | 1/3 | — |
| S6 | 1/3 | 2/3 | 2/3 | 2/3 | — | 2/3 | 2/3 | 2/3 | — |
| S7 | 0/3 | 3/3 | 3/3 | 3/3 | — | 3/3 | 3/3 | 3/3 | — |
| S8 | 1/3 | 2/3 | 2/3 | 2/3 | — | 2/3 | 2/3 | 2/3 | — |
| S9 | 2/3 | 1/3 | 1/3 | 1/3 | — | 1/3 | 1/3 | 1/3 | — |
| S10 | 1/3 | 2/3 | 2/3 | 2/3 | — | 2/3 | 2/3 | 2/3 | — |
| S11 | 2/3 | 1/3 | 1/3 | 1/3 | — | 1/3 | 1/3 | 1/3 | — |
| S12 | 0/3 | 3/3 | 3/3 | 3/3 | — | 3/3 | 3/3 | 3/3 | — |

**Reading.** v1 → v2 differs on exactly two clips: `action_S3.mp4_2_.mp4` (v1: bg, v2: action ✓) and `background_S4.mp4_7_.mp4` (v1: bg ✓, v2: action ✗ false positive). All ten remaining sources have identical v1 and v2 agreement counts despite v1 being a structurally degenerate output. The "structural per-source convergence" argument from the original PR #2 writeup is *strengthened* here, not weakened: when both a buggy text-only model and a genuinely-perceiving conservative model land in the same 22/36 ≈ 0.611 agreement, per-source vectors converge because the dominant mode in both is "lean to background majority". Two genuinely different mechanisms (v1 collapse from no input, v2 conservatism from real input) produce nearly indistinguishable per-source numbers.

## 5. Qualitative reasoning excerpts (true-action clips, prompt C)

| Clip | Gemini 3.1 + C | **Qwen v1 + C** *(buggy)* | **Qwen v2 + C** |
|---|---|---|---|
| `action_S10.mp4_0_.mp4` | bg, 0.99 — *"Across all frames, both ears remain in a fixed forward-pointing position without any visible rotation, twitching, or pinning."* | bg, 0.95 — *"The ears remain in essentially the same position throughout the clip, with no visible positional changes or movements."* | bg, 0.9 — *"The horse's ears remain in a consistent position throughout the frames, with no visible movement or change in their orientation."* |
| `action_S11.mp4_0_.mp4` | bg, 1.00 — *"…remain in the same outward-pointing position from the first frame to the last…"* | bg, 0.95 — *"…essentially the same position throughout the clip, with no visible positional changes…"* | bg, 0.9 — *"…remain in a consistent position throughout the frames, with no visible positional changes…"* |
| `action_S11.mp4_6_.mp4` | bg, 1.00 — *"Both ears remain stationary and in the same forward-pointing position…"* | bg, 0.95 — same templated sentence | bg, 0.9 — *"…consistent position throughout the frames, with no visible positional changes or twitching."* |
| `action_S2.mp4_7_.mp4` | bg, 1.00 — *"…remain still and pointing slightly backward with no visible movement…"* | bg, 0.95 — same templated sentence | bg, 0.9 — *"…consistent position throughout the frames, with no visible movement or change in their orientation."* |
| `action_S3.mp4_2_.mp4` | **action**, 0.95 — *"The horse's left ear displays motion blur and slight positional changes across the frames, indicating a flicking or twitching movement."* | bg, 0.95 — same templated sentence | bg, 0.9 — *"…consistent position throughout the clip, with no visible positional changes or twitching."* (still missed) |

**Reading.** Qwen v2 + C produces **near-templated** evidence sentences — small word swaps ("frames" ↔ "clip", "movement" ↔ "positional changes") on top of the same skeletal phrase, with confidence locked at 0.9 (down from v1's 0.95). The bug fix did not unlock genuinely clip-conditioned evidence under prompt C. Even on `action_S3.mp4_2_.mp4` — which v2 catches under the looser prompt A — prompt C's strict 2-class enum + system instruction routes the model back into the conservative templated response. Gemini 3.1 + C produces clip-specific evidence and catches the same clip; Qwen v2 + C produces near-templated evidence and misses it. The system instruction's evidence-citation request is being satisfied syntactically rather than semantically on Qwen, regardless of whether the model has the video.

## 6. §4 outcome and decision (re-evaluated on v2)

| Spec §4 row | Threshold | **Qwen v2 7B observed** | Hit? |
|---|---|---|---|
| **Row 1 — all 3 failure modes reproduced** | refusal-bias + cross-rep instability + decoupling all present | bg-rate 94–97 %, 0/10 stable, 2/10 decoupling | **YES (legitimately, post-fix)** |
| Row 2 — materially cleaner | agreement ≥ 0.70 AND bg-rate < 80 % AND no > 3 pp decoupling | 0.583–0.611, 94–97 %, 20 % | NO |
| Row 3 — mixed | one metric clean, others not | none clean | NO |
| Row 4 — infrastructure failure | wouldn't run | 0/122 errors on v2 | NO |

**Decision: Row 1 — legitimately, on v2.** No 32B follow-up. Lesson 15 is **replaced** (not amended) to reflect the post-fix framing. The MLLM-as-classifier track on this task remains closed within the scope tested. V-JEPA-2 + linear probe (LOSO 0.875) remains the spine.

## 7. What changed in the conclusion vs PR #2

| | PR #2 v1 (buggy) | This document (v2 fixed) |
|---|---|---|
| §4 outcome | Row 1 | Row 1 (still) |
| Mechanism for "background everywhere" | unknown — assumed model conservatism | now disambiguated: the v1 36/36 collapse was text-only inference; the v2 conservatism is genuine model behavior on properly-routed video |
| Generalization claim | "Lesson 14 generalizes from Gemini family to MLLM class" | same claim, now with valid evidence |
| Per-source vector convergence | "structural — uniform bg lean produces convergence regardless of perception" | confirmed: the convergence holds even when one side is text-only and the other side is perceiving |
| Decoupling magnitude | 30 % | 20 % (1 case fewer because v2 catches `action_S3.mp4_2` under prompt A) |
| Qwen catches any true-action? | no (0/14, by construction) | yes — 1/14 on prompt A (`action_S3.mp4_2_.mp4`), 0/14 on prompt C |

## 8. Caveats and scope (unchanged from v1 except where noted)

- **One open-weight MLLM family at one parameter scale.** Qwen2.5-VL, 7B, bf16 only. Does not extrapolate.
- **One task.** Sub-second ear movement on RHpE clips. Coarser tasks may behave differently.
- **One frame-sampling regime.** mlx-vlm 0.5.0 with `fps=10`. Not tested on alternative sampling.
- **One inference backend.** mlx-vlm. PyTorch MPS not tested in v2 (the spec §6 fallback was not needed since Phase 2 passed at the prompt-token level).
- **Vocabulary normalization is conservative.** `still` → `background` mapping is documented and applied only at agreement computation; it does not change the JSONL or alter raw model output.
- **The bug invalidated v1 results.** v1 numbers are preserved in this document and in the JSONLs as diagnostic evidence, NOT as model-behavior evidence. The PR #2 conclusion was structurally correct by accident; v2 makes it correct in fact.
- **The `_print_debug_tensors` helper in `qwen_audit.py` is broken.** It crashes with `Expected video as (T, C, H, W), got shape ()` because mlx-vlm's processor expects pre-loaded video tensors when called directly, not paths. Substantive Phase 2 verification was done at the formatted-prompt + prompt-token level instead. Documenting as a follow-up to fix in `tools/qwen_audit.py` if this code path becomes load-bearing.

## 9. Reproduction (v2)

```bash
git checkout fix/qwen-mlx-video-input
cd /path/to/poc
bash setup.sh
.venv/bin/python tools/qwen_audit.py --probe 1 --out-suffix _v2
.venv/bin/python tools/qwen_audit.py --probe 2 --out-suffix _v2
.venv/bin/python tools/qwen_audit.py --probe 3 --out-suffix _v2
```

Phase 2 (tensor verification) and Phase 3 (5-clip smoke) recipes are in `outputs/fix-verification-summary.md`.

## 10. References

- Original spec: `docs/qwen-experiment-spec.md`
- Fix spec: `docs/qwen-fix-and-revalidate-spec.md`
- Phase 2/3 verification: `outputs/fix-verification-summary.md`
- Lessons 14, 15, 16: `docs/lessons_learned.md`
- mlx-vlm: https://github.com/Blaizzy/mlx-vlm (0.5.0 used here)
- Qwen2.5-VL model card: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
