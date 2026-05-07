# Qwen Pipeline Fix and Re-validation — Spec for Claude Code

*Saved as `docs/qwen-fix-and-revalidate-spec.md` on branch `fix/qwen-mlx-video-input` (branched from `experiment/qwen-mlx`). **PR #2 is held — do not merge — until this spec completes.***

---

## 1. Context — what we found and why this is fix-first

PR #2 reports Qwen2.5-VL-7B-bf16 producing **36/36 background classifications** on the 36-clip RME audit subset, with per-source agreement element-wise identical to Gemini 3.1's. The result was pre-registered to merge under §4 row 1.

**This result is unsafe.** Code review of `tools/qwen_audit.py` against `tools/gemini_audit.py` and against mlx-vlm conventions identified a high-confidence bug in the Qwen video input path:

In `QwenRunner.run` (around line 109):

```python
formatted = self._apply_chat_template(
    self.processor, self.config, prompt=messages,
    num_images=0, num_audios=0,
)
```

`num_videos` is omitted and defaults to 0. mlx-vlm's `apply_chat_template` uses these `num_*` integers to insert vision-token placeholders into the rendered prompt. With all three at zero, **no `<|vision_start|><|video_pad|><|vision_end|>` placeholders are inserted**, so the formatted prompt is text-only regardless of what `messages` contains. The subsequent `generate(..., video=clip_path, ...)` call passes the video, but the prompt has no slot for video tokens to fill — so the model effectively sees a text-only prompt about "a horse clip" with no video signal. This produces exactly the observed 36/36 bg-collapse with templated reasoning sentences and confidence locked at 0.95.

**Independent confirmations:**
- mlx-vlm README and issue #192 show `num_*` parameters are load-bearing for modality token insertion (not optional metadata).
- mlx-vlm has a separate `mlx_vlm.video_generate` CLI specifically for video, distinct from the image path.
- `gemini_audit.py` line 286 uses the canonical Gemini Files API + `VideoMetadata(fps=fps)` path correctly, so Gemini's 35/36 bg is real model output. The "element-wise identical per-source vector" is structurally explained: when one model outputs uniform bg on everything, its per-source agreement is determined entirely by per-source bg/action distribution in the dataset — convergence is structural, not behavioral.

Because the diagnosis is high-confidence and the fix is one line, **this spec applies the fix first and only escalates to deeper diagnostics if behavior doesn't change**. The original diagnostic spec is preserved in §6 below as a fall-back path.

---

## 1a. Pre-flight verification (added at branch creation)

Before applying the fix, the agent traced `mlx_vlm.prompt_utils.apply_chat_template` on mlx-vlm 0.5.0 and ran an empirical render test. Result: the spec's diagnosis is correct in direction (the buggy call produces no video tokens) but the proposed mechanism is incomplete. The list-prompt branch in `apply_chat_template` calls `extract_text_from_content(content)` on the user message **before** `num_videos` is consulted — `extract_text_from_content` keeps only `type=text` items and silently drops every `type=video` item. With the video info gone from the message, `_format_video_message` (gated on `kwargs.get("video")` being truthy) is never reached, regardless of `num_videos`.

Empirical render comparison on `mlx-community/Qwen2.5-VL-7B-Instruct-bf16` with a real RME clip:

| Call shape | Rendered prompt length | Contains `<\|video_pad\|>`? |
|---|---|---|
| `prompt=messages_list_with_video_content`, no `num_videos` | 159 chars | **no** — bug |
| `prompt=messages_list_with_video_content`, `num_videos=1` (literal §2 fix) | 159 chars | **no** — fix doesn't propagate |
| `prompt=user_text_str`, `video=path`, `fps=10`, `num_videos=1` | 202 chars | **yes** ✓ |

The full fix is therefore not one line — it restructures `QwenRunner.run` to use the string-prompt + kwargs path. `num_videos=1` stays in the call (per spec §2) and the messages-list path is removed. This is the change applied in §2 below; the spec's original §2 wording is preserved unchanged for the dated record.

---

## 2. The fix (as updated by §1a)

**File:** `tools/qwen_audit.py`, in `QwenRunner.run`.

**Original spec change (insufficient on its own):**

```python
# BEFORE:
formatted = self._apply_chat_template(
    self.processor, self.config, prompt=messages,
    num_images=0, num_audios=0,
)

# AFTER (literal §2; verified to NOT propagate video tokens — see §1a):
formatted = self._apply_chat_template(
    self.processor, self.config, prompt=messages,
    num_images=0, num_audios=0, num_videos=1,
)
```

**Actual applied fix (kwarg restructure):**

```python
# AFTER (kwarg restructure; verified to insert video tokens — see §1a):
formatted = self._apply_chat_template(
    self.processor, self.config, prompt=user_text,
    video=clip_path, fps=10.0,
    num_images=0, num_audios=0, num_videos=1,
)
```

The `messages` list construction is removed; `system_instruction` (when set) is passed through `apply_chat_template`'s system-message support directly via `prompt=[{"role": "system", ...}, user_text]` form. No other code changes in Phase 1.

---

## 3. Procedure

### Phase 1 — Apply the fix (~5 min)

1. Branch: `fix/qwen-mlx-video-input` from `experiment/qwen-mlx`.
2. Apply the kwarg-restructure edit per §2 (revised). Add `--out-suffix` flag to `qwen_audit.py` so v2 outputs can land alongside v1 evidence.
3. Commit with message `fix(qwen): pass video as kwarg + num_videos=1 (apply_chat_template now emits vision tokens)`. Reference this spec.

### Phase 2 — Verify the fix at the tensor level (~15 min)

1. Add temporary debug prints inspecting the processor output for one action clip.
2. Run on a known-action clip (`action_S3.mp4_2_.mp4` or similar — S3 has strong ear motion per the iter-6.5 case study).
3. Capture shapes to `outputs/fix-verification-tensor-shapes.txt`.

**Pass condition:** `pixel_values_videos` (or equivalent) populated with shape `(1, T, C, H, W)` where T>1; `video_grid_thw` non-degenerate.
**Fail condition:** only image tensor or T=1.

### Phase 3 — Smoke test on 5 clips (~15 min)

3 action (S3, S3, S10) + 2 bg (S1, S12). Probe 1, t=0. Save to `outputs/fix-verification-smoke-test.jsonl`. Compare to `outputs/qwen25vl_7b_promptA.jsonl` (v1, pre-fix).

| Outcome | Meaning | Next |
|---|---|---|
| A. ≥1 action clip flips to "action" | Bug confirmed; v1 invalid | Phase 4 |
| B. Tensors valid (Phase 2 pass) but smoke still 100% bg | Bug existed but model genuinely conservative | Phase 4 |
| C. Tensors look unchanged from pre-fix, smoke still 100% bg | Bug class extends beyond `num_videos` | Phase 6 |

### Phase 4 — Full 36-clip re-run as v2 (~30–60 min)

Remove debug prints (separate commit). Re-run with `--out-suffix _v2`:
- `outputs/qwen25vl_7b_promptA_v2.jsonl`
- `outputs/qwen25vl_7b_promptC_v2.jsonl`
- `outputs/qwen25vl_7b_description_probe_v2.jsonl`

V1 outputs preserved untouched.

### Phase 5 — Update writeups (~30 min)

1. **Update `outputs/qwen_vs_gemini_comparison.md`** with both v1 and v2 numbers; mark v1 as "pre-fix, video signal not reaching model".
2. **Replace Lesson 15** in `docs/lessons_learned.md` (don't amend) per actual v2 outcome.
3. **Add Lesson 16**: mlx-vlm video-input bug — `apply_chat_template` requires the string-prompt + `video` kwarg path on Qwen2.5-VL; the structured-messages path silently drops video content via `extract_text_from_content`. Verify temporal tensors before drawing conclusions.
4. **Update README** cross-vendor section.
5. **Comment on PR #2** with outcome + merge recommendation based on v2.

### Phase 6 — Deeper diagnosis (only if Phase 2 fails or Phase 3 hits Outcome C)

Time-boxed 2 hours. Steps Q6.1–Q6.4 (mlx-vlm version, `mlx_vlm.video_generate` CLI, PyTorch MPS fallback via `qwen_vl_utils.process_vision_info`, resolution sanity).

---

## 4. Scope discipline

- **Do NOT** delete or overwrite v1 outputs.
- **Do NOT** modify `tools/gemini_audit.py`.
- **Do NOT** change prompts, temperature, sampling, or the 36-clip / 10-clip subsets.
- **Do NOT** start a 32B run, swap models, or expand scope.
- **Do NOT** merge to main, even after Phase 4. PR #2 review with v2 numbers first.
- **Do NOT** spend more than 3 h on Phases 1–5, or 2 h on Phase 6.
- **Do NOT** edit lessons 1–14. **Replace** the existing Lesson 15 (don't amend); add Lesson 16 separately.

---

## 5. Deliverables

1. The fix in `tools/qwen_audit.py` (Phase 1)
2. `outputs/fix-verification-tensor-shapes.txt` (Phase 2)
3. `outputs/fix-verification-smoke-test.jsonl` (Phase 3)
4. `outputs/fix-verification-summary.md` (which Phase 3 outcome was hit)
5. `outputs/qwen25vl_7b_*_v2.jsonl` (Phase 4, if reached)
6. Updated `outputs/qwen_vs_gemini_comparison.md` with both v1 and v2 (Phase 5)
7. Replaced Lesson 15 + new Lesson 16 (Phase 5)
8. Updated `README.md` (Phase 5)
9. (If Phase 6 ran) `outputs/fix-verification-pytorch-fallback.jsonl` + diagnosis notes
10. PR comment on #2 with outcome and merge recommendation

---

## 6. Quick start for the agent

1. Read this spec end-to-end including §1a pre-flight finding.
2. Branch `fix/qwen-mlx-video-input` from `experiment/qwen-mlx`.
3. Phase 1: apply the kwarg-restructure fix per §2 (revised).
4. Phase 2: tensor inspection on one action clip.
5. Phase 3: 5-clip smoke. Determine A/B/C.
6. Phase 4: full re-run as v2 if A or B.
7. Phase 5: update writeups, replace L15, add L16, comment on PR #2.
8. Phase 6: time-boxed deeper diagnosis if needed.
