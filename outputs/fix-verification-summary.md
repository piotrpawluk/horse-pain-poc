# Fix verification summary — `fix/qwen-mlx-video-input`

*Phase 2 + Phase 3 of `docs/qwen-fix-and-revalidate-spec.md`. Run 2026-05-07. Model: `mlx-community/Qwen2.5-VL-7B-Instruct-bf16` via mlx-vlm 0.5.0 on M2 Max.*

## Outcome: **A — bug confirmed, v1 results invalid**

At least one true-action clip (`action_S3.mp4_2_.mp4`) flips from v1's `background` (templated reasoning) to v2's `action` (clip-specific reasoning). Per spec §3 outcome table, this triggers Phase 4 (full 36-clip v2 re-run).

## Phase 2 — tensor / token-level verification

The temporary debug inspection (`--debug-tensors` flag, `_print_debug_tensors`) failed with `ValueError: Expected video as (T, C, H, W), got shape ().` because mlx-vlm's processor expects a pre-loaded video tensor object, not a path string, when invoked directly. **This failure is in the inspection helper only**, not in the actual generation pipeline — `mlx_vlm.generate(video=clip_path, ...)` loads the video internally and works correctly. The substantive verification evidence is therefore at the prompt-token level rather than processor-tensor-shape level:

| Signal | v1 (pre-fix) | v2 (post-fix) | Δ |
|---|---|---|---|
| Formatted prompt length | 159 chars (no `<\|video_pad\|>`) | 510 chars (`<\|video_pad\|>` present) | +351 chars |
| `prompt_tokens` reported by generate | ~111 | **5495** | **49× increase** |
| Wall-clock per inference | ~3–5 s | ~20 s | ~5× slower |
| Reasoning text content | identical templated sentence on every clip | clip-specific, varied vocabulary | qualitative change |

The 49× increase in `prompt_tokens` is the smoking-gun evidence: with v1 the model was tokenizing only a text prompt (~110 tokens of system header + user prompt + assistant scaffold), with v2 the model is tokenizing video frames into ~5380 vision tokens plus the ~110 text tokens. This is consistent with mlx-vlm's documented Qwen2.5-VL video tokenization (`min_pixels=224*224`, fps=10 on a ~1.5 s clip → roughly 5300 video tokens).

**Phase 2 status: PASS.** Tensor-level inspection helper is broken but doesn't matter — the prompt-token count, formatted-prompt content, and behavioral change collectively establish that video is reaching the model.

## Phase 3 — smoke test on 5 clips (probe 1, prompt A, t=0)

### v2 raw output

| Clip | Truth | v2 classification | v2 confidence | v2 reasoning |
|---|---|---|---|---|
| `action_S3.mp4_2_.mp4` | action | **action** ✓ | 0.8 | "The horse's ears are slightly moving, indicating active movement." |
| `action_S3.mp4_8_.mp4` | action | still ✗ | 0.9 | "The horse's ears are not moving, indicating they are still." |
| `action_S10.mp4_0_.mp4` | action | still ✗ | 0.9 | "The horse's ears are not moving, indicating they are still." |
| `background_S1.mp4_11_.mp4` | background | still ≈ ✓ | 0.9 | "The horse's ears are not moving; they remain in a fixed position throughout the video." |
| `background_S12.mp4_2_.mp4` | background | **background** ✓ | 0.9 | "The horse's ears are not actively moving; they appear still and relaxed." |

### v1 (pre-fix) baseline for the same 5 clips

All five clips returned `qwen_label = "background"` with confidence 0.95 and the templated reasoning *"The ears remain in essentially the same position throughout the clip, with no visible positional changes or movements."*

### Vocabulary observation: `still` vs `background`

Prompt A asks for `"action" | "background"` per the JSON schema, but the prompt body uses *"holding them STILL"* in capital letters. Qwen v2 sometimes outputs `"still"` (echoing the prompt's prose) rather than `"background"` (the schema label). The two are semantically equivalent in context — both denote "no positional change". Normalization rule for downstream comparison (Phase 4 + Phase 5):

```python
def normalize_qwen_label(label: str) -> str | None:
    if label is None: return None
    s = label.strip().lower()
    if s in ("action", "moving", "movement"): return "action"
    if s in ("background", "still", "stationary", "no action", "no_action"): return "background"
    return None  # genuine OOV
```

This will be applied **only at agreement-computation time**, not in the JSONL writes. The raw `qwen_label` field preserves whatever the model emits, so the vocabulary deviation stays auditable. With normalization applied, the smoke test reads as: 3/5 = 0.60 agreement, vs v1's 2/5 = 0.40 (where v1's "agreement" was entirely the bg-collapse-on-true-bg artifact).

## Decision

- **Phase 4: GO.** Re-run all three probes with the fixed pipeline; save with `--out-suffix _v2`.
- v1 outputs preserved untouched in `outputs/qwen25vl_7b_*.jsonl` (no `_v2`) as diagnostic evidence.
- Lesson 15 will be replaced (not amended) in Phase 5 per spec — the v1-driven framing was based on text-only inference disguised as video inference.
- Lesson 16 will document the mlx-vlm 0.5.0 video-input bug (structured-messages path drops video) and the system+video routing bug discovered during fix verification.
- PR #2 stays held until Phase 5 completes.
