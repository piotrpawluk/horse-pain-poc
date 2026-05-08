#!/usr/bin/env python3
"""Qwen2.5-VL replication of the Gemini Lesson 14 protocol on Read My Ears.

Branch: experiment/qwen-mlx
Spec: docs/qwen-experiment-spec.md

Three probes against the SAME 36-clip stratified subset Gemini was tested on
(2.5 Pro and 3.1 Pro Preview, May 2026):

  --probe 1   prompt A (generic)              temperature=0
  --probe 2   prompt C (best-practice + sys)  Qwen-recommended defaults
  --probe 3   description-only, 5 reps × 10   temperature=0.7

The 36-clip and 10-clip subsets are NOT re-stratified. They are reconstructed
from clip paths in the existing Gemini JSONL outputs:
  outputs/gemini_audit_results_gemini-2.5-pro_promptA.jsonl   → 36-clip set
  outputs/gemini_audit_probe_gemini-3.1-pro-preview_temp1.0_thinkLOW.jsonl
                                                              → 10-clip probe set

Prompts (PROMPT_A, PROMPT_C, SYSTEM_INSTRUCTION_C, PROBE_PROMPT) are imported
verbatim from gemini_audit.py — no paraphrase.

JSONL schema is diff-able with the Gemini outputs (extra fields padded null
on the Gemini side, extra Qwen-only fields added at the tail). Idempotent:
re-runs skip clips already in the output JSONL.

Usage:
    .venv/bin/python tools/qwen_audit.py --probe 1
    .venv/bin/python tools/qwen_audit.py --probe 2
    .venv/bin/python tools/qwen_audit.py --probe 3
    .venv/bin/python tools/qwen_audit.py --probe 1 --model-size 32B   # conditional
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

POC_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = POC_DIR / "outputs"
sys.path.insert(0, str(POC_DIR / "tools"))
import gemini_audit as ga  # PROMPT_A, PROMPT_C, SYSTEM_INSTRUCTION_C, PROBE_PROMPT, source_id


MODEL_PATHS = {
    "7B":                 "mlx-community/Qwen2.5-VL-7B-Instruct-bf16",
    "32B":                "mlx-community/Qwen2.5-VL-32B-Instruct-4bit",
    "qwen3-30b-a3b-4bit": "mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit",
    "qwen3-8b-bf16":      "mlx-community/Qwen3-VL-8B-Instruct-bf16",
    "qwen3-8b":           "mlx-community/Qwen3-VL-8B-Instruct",
}

# v2 priority chain — first to load AND generate cleanly wins.
# 2026-05-08: Qwen3-VL on mlx-vlm 0.5.0 has multiple known issues (GitHub
# #652 broadcast errors, Metal GPU page faults during inference, prose-not-
# JSON output adherence — all confirmed via WebSearch + empirical probes).
# Qwen2.5-VL-7B-bf16 with inlined-system mode (Lesson 16) is the working
# path; v2 applies all prompt-body refinements (Markdown structure, frame
# indices, sclera, negative anchor, temp=0) on top. Qwen3 variants kept in
# the chain as evidence — they will be probed and rejected for the record.
QWEN3_V2_PRIORITY = ["7B", "qwen3-30b-a3b-4bit", "qwen3-8b-bf16", "qwen3-8b"]

# Gemini outputs we mine for clip paths.
GEMINI_36_SOURCE = OUTPUTS / "gemini_audit_results_gemini-2.5-pro_promptA.jsonl"
GEMINI_10_PROBE_SOURCE = OUTPUTS / "gemini_audit_probe_gemini-3.1-pro-preview_temp1.0_thinkLOW.jsonl"


def load_36_clip_set() -> list[dict]:
    """Reconstruct the 36-clip set from the canonical Gemini A run.

    Returns rows with the fields needed downstream: clip, video_rel, split,
    source, true_label.
    """
    if not GEMINI_36_SOURCE.exists():
        sys.exit(f"ERROR: {GEMINI_36_SOURCE} not found — cannot reconstruct 36-clip set.")
    rows = []
    seen = set()
    with open(GEMINI_36_SOURCE) as f:
        for line in f:
            r = json.loads(line)
            key = r["clip"]
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "clip": r["clip"],
                "video_rel": r["video_rel"],
                "split": r["split"],
                "source": r["source"],
                "true_label": r["true_label"],
            })
    if len(rows) != 36:
        print(f"WARNING: expected 36 unique clips, got {len(rows)}", file=sys.stderr)
    return rows


def load_10_probe_clips() -> list[dict]:
    """Reconstruct the 10-clip probe set from the canonical Gemini 3.1 probe JSONL."""
    if not GEMINI_10_PROBE_SOURCE.exists():
        sys.exit(f"ERROR: {GEMINI_10_PROBE_SOURCE} not found — cannot reconstruct 10-clip set.")
    rows = []
    seen = set()
    with open(GEMINI_10_PROBE_SOURCE) as f:
        for line in f:
            r = json.loads(line)
            key = r.get("clip") or r.get("clip_path")
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "clip": r.get("clip") or r.get("clip_path"),
                "video_rel": r.get("video_rel"),
                "split": r.get("split"),
                "source": r.get("source"),
                "true_label": r.get("true_label"),
            })
    if len(rows) != 10:
        print(f"WARNING: expected 10 unique probe clips, got {len(rows)}", file=sys.stderr)
    return rows


# ----- Output handling --------------------------------------------------------

def output_path(model_size: str, prompt_id: str, suffix: str = "") -> Path:
    return OUTPUTS / f"qwen25vl_{model_size.lower()}_{prompt_id}{suffix}.jsonl"


def already_done(out: Path, key_fields: tuple) -> set:
    """Return set of keys (per key_fields tuple) already in the JSONL."""
    if not out.exists():
        return set()
    done = set()
    with open(out) as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add(tuple(r.get(k) for k in key_fields))
            except json.JSONDecodeError:
                continue
    return done


def append_row(out: Path, row: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ----- JSON parsing -----------------------------------------------------------

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def parse_qwen_json(text: str) -> dict:
    """Extract JSON object from Qwen output. Strips markdown fences if present."""
    s = text.strip()
    m = _FENCE_RE.match(s)
    if m:
        s = m.group(1).strip()
    # Best-effort: find the first { ... } balanced block.
    depth = 0
    start = -1
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = s[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
    # Last resort: try to parse whole string.
    return json.loads(s)


# ----- Inference --------------------------------------------------------------

class QwenRunner:
    """Lazy-loaded mlx-vlm wrapper. Loads model once per process."""

    def __init__(self, model_size: str):
        self.model_size = model_size
        self.model_path = MODEL_PATHS[model_size]
        print(f"[qwen] Loading {self.model_path} ...", flush=True)
        t0 = time.time()
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        self.model, self.processor = load(self.model_path)
        self.config = load_config(self.model_path)
        print(f"[qwen] Loaded in {time.time()-t0:.1f}s", flush=True)
        # Imports cached for inference.
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        self._generate = generate
        self._apply_chat_template = apply_chat_template

    # v2 (2026-05-08): system-role-mode resolution. Lesson 16 documented mlx-vlm
    # 0.5.0 stripping video tokens on messages_list+structured_content. v2
    # empirically re-verifies on the current mlx-vlm version + the Qwen3-VL
    # tokenizer (which may use a different video pad token than Qwen2.5).
    _VISION_MARKER_CANDIDATES = ("<|video_pad|>", "<|vision_pad|>", "<|video_start|>")

    def _resolved_video_marker(self, formatted: str) -> Optional[str]:
        """Return whichever vision marker (if any) appears in `formatted`."""
        for cand in self._VISION_MARKER_CANDIDATES:
            if cand in formatted:
                return cand
        return None

    def _format_messages_list(
        self, clip_path: str, user_text: str, system_instruction: Optional[str]
    ) -> str:
        """v2 mode: true `system` role via mlx-vlm structured-content messages."""
        messages = []
        if system_instruction:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_instruction}],
            })
        messages.append({
            "role": "user",
            "content": [
                {"type": "video", "video": clip_path},
                {"type": "text", "text": user_text},
            ],
        })
        return self._apply_chat_template(
            self.processor, self.config, prompt=messages,
            video=clip_path, fps=10.0,
            num_images=0, num_audios=0, num_videos=1,
        )

    def _format_manual_chatml(
        self, clip_path: str, user_text: str, system_instruction: Optional[str],
        *, json_prime: bool = False,
    ) -> str:
        """v2 fallback: bypass apply_chat_template, emit Qwen ChatML directly.

        `json_prime=True` appends a `{` after the assistant turn marker so the
        model is forced to continue from a JSON-object opening. mlx-vlm 0.5.0
        observed (Qwen3-VL-8B-Instruct-bf16) producing free-form prose
        descriptions instead of JSON at max_tokens=256 in messages_list mode;
        priming the assistant turn redirects generation into the JSON path.
        Caller must re-prepend the `{` to result.text before JSON parsing.
        """
        sys_block = (
            f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
            if system_instruction else ""
        )
        assistant_prime = "{" if json_prime else ""
        return (
            sys_block
            + "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
            + user_text
            + "<|im_end|>\n<|im_start|>assistant\n"
            + assistant_prime
        )

    def _format_inlined(
        self, clip_path: str, user_text: str, system_instruction: Optional[str]
    ) -> str:
        """v1 mode (Lesson 16): inline system into user-text, string prompt path."""
        prompt_str = (
            f"{system_instruction}\n\n{user_text}" if system_instruction else user_text
        )
        return self._apply_chat_template(
            self.processor, self.config, prompt=prompt_str,
            video=clip_path, fps=10.0,
            num_images=0, num_audios=0, num_videos=1,
        )

    def _format_for_mode(
        self, mode: str, clip_path: str, user_text: str,
        system_instruction: Optional[str], *, json_prime: bool = False,
    ) -> str:
        if mode == "messages_list":
            return self._format_messages_list(clip_path, user_text, system_instruction)
        if mode == "manual_chatml":
            return self._format_manual_chatml(
                clip_path, user_text, system_instruction, json_prime=json_prime,
            )
        if mode == "manual_chatml_jsonprime":
            return self._format_manual_chatml(
                clip_path, user_text, system_instruction, json_prime=True,
            )
        if mode == "inlined":
            return self._format_inlined(clip_path, user_text, system_instruction)
        raise ValueError(f"unknown system_role_mode: {mode}")

    def resolve_system_role_mode(
        self, clip_path: str, user_text: str, system_instruction: Optional[str],
        *, end_to_end: bool = True, temperature: float = 0.0,
        repetition_penalty: Optional[float] = 1.05, max_tokens: int = 32,
    ) -> tuple[str, str]:
        """Probe each mode on a single clip; return (mode, vision_marker_found).

        Tries (in priority order):
          1. manual_chatml_jsonprime — manual ChatML with `{` priming on the
             assistant turn (forces JSON output for Qwen3-VL which otherwise
             produces prose at max_tokens=256, observed 2026-05-08).
          2. messages_list — true `system` role via mlx-vlm structured content.
          3. manual_chatml — manual ChatML without priming.
          4. inlined — v1 fallback (system in user-text).

        Stops at the first mode that BOTH formats successfully (string
        contains a vision marker) AND generates without raising. Qwen3-VL on
        mlx-vlm has been observed to format messages_list cleanly but crash
        at generate-time with broadcast-shape errors (30B-A3B-4bit) or
        produce non-JSON prose (8B-bf16), so a string-only probe is
        insufficient.

        `end_to_end=False` reverts to format-only probe (cheaper but unsafe).
        """
        # Order matters: `inlined` is the v1 Lesson-16 known-good path for
        # Qwen2.5-VL, putting it first ensures Qwen2.5 lands on the proven
        # mode rather than `manual_chatml_jsonprime` which (observed
        # 2026-05-08) collapses to empty `{}` when the model treats the
        # primed `{` as a complete-and-stop signal. messages_list goes second
        # to test the system-role-split for models that support it cleanly.
        for mode in (
            "inlined", "messages_list", "manual_chatml", "manual_chatml_jsonprime",
        ):
            try:
                formatted = self._format_for_mode(
                    mode, clip_path, user_text, system_instruction,
                )
                marker = self._resolved_video_marker(formatted)
                if not marker:
                    print(f"[qwen v2] smoke mode={mode}: no vision marker in formatted "
                          f"prompt; fallthrough.", flush=True)
                    continue
                if end_to_end:
                    kwargs = dict(temperature=temperature, max_tokens=max_tokens,
                                  verbose=False)
                    if repetition_penalty is not None:
                        kwargs["repetition_penalty"] = repetition_penalty
                    self._generate(
                        self.model, self.processor, prompt=formatted,
                        video=clip_path, **kwargs,
                    )
                print(f"[qwen v2] smoke mode={mode}: OK marker={marker}", flush=True)
                return mode, marker
            except Exception as exc:  # noqa: BLE001 - empirical probe
                print(f"[qwen v2] smoke mode={mode} raised: "
                      f"{type(exc).__name__}: {exc}; fallthrough.",
                      flush=True)
        return "inlined", "<|video_pad|>"  # last resort

    def run(
        self,
        clip_path: str,
        user_text: str,
        *,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        repetition_penalty: Optional[float] = None,
        max_tokens: int = 256,
        debug_tensors: bool = False,
        system_role_mode: str = "inlined",
    ) -> dict:
        """Single inference. Returns a dict with text + token counts + latency.

        v1 (Lesson 16, mlx-vlm 0.5.0 era): inlined-system / string-prompt path
        was the only one that emitted `<|vision_start|><|video_pad|><|vision_end|>`
        — list-prompt + video-kwarg attached video tokens to every message.
        Default `system_role_mode="inlined"` preserves that v1 behavior.

        v2 (2026-05-08): per-vendor research argued true `system` role is the
        correct Qwen pattern. Callers should call `resolve_system_role_mode()`
        on a smoke clip first and pass the winning mode here. Modes:
          - "messages_list"  : structured-content messages with system role
          - "manual_chatml"  : ChatML string emitted directly (bypasses
                               apply_chat_template)
          - "inlined"        : v1 fallback, system in user-text
        """
        formatted = self._format_for_mode(
            system_role_mode, clip_path, user_text, system_instruction,
        )
        if debug_tensors:
            self._print_debug_tensors(formatted, clip_path)
        kwargs = dict(temperature=temperature, max_tokens=max_tokens, verbose=False)
        if repetition_penalty is not None:
            kwargs["repetition_penalty"] = repetition_penalty
        t0 = time.time()
        result = self._generate(
            self.model, self.processor, prompt=formatted, video=clip_path, **kwargs
        )
        dt = time.time() - t0
        marker = self._resolved_video_marker(formatted)
        text = getattr(result, "text", str(result))
        # JSON-prime mode prepends `{` to the assistant turn so the model
        # continues from inside a JSON object. mlx-vlm's `result.text` does
        # not include the primed prefix, so we re-attach it to produce a
        # parseable JSON string for downstream consumers.
        if system_role_mode == "manual_chatml_jsonprime":
            text = "{" + text
        return {
            "text": text,
            "prompt_tokens": getattr(result, "prompt_tokens", None),
            "candidate_tokens": getattr(result, "generation_tokens", None),
            "total_tokens": getattr(result, "total_tokens", None),
            "finish_reason": getattr(result, "finish_reason", None),
            "wall_clock_seconds": dt,
            "formatted_prompt_chars": len(formatted),
            "has_video_token": marker is not None,
            "video_marker": marker,
            "system_role_mode_used": system_role_mode,
        }

    def _print_debug_tensors(self, formatted: str, clip_path: str) -> None:
        """Phase 2 verification: inspect what the processor produced.

        Removed before merge — kept here as a diagnostic that runs only when
        --debug-tensors is passed. See docs/qwen-fix-and-revalidate-spec.md §3 Phase 2.
        """
        print(f"[debug] formatted prompt: {len(formatted)} chars, "
              f"<|video_pad|> present: {'<|video_pad|>' in formatted}", flush=True)
        try:
            processed = self.processor(
                text=[formatted], videos=[clip_path], return_tensors="pt", padding=True
            )
            for k, v in processed.items():
                if hasattr(v, "shape"):
                    print(f"[debug] {k}: shape={tuple(v.shape)}, dtype={v.dtype}",
                          flush=True)
                elif isinstance(v, list) and v and hasattr(v[0], "shape"):
                    shapes = [tuple(t.shape) for t in v]
                    print(f"[debug] {k}: list of {len(v)} tensors, shapes={shapes}",
                          flush=True)
                else:
                    s = repr(v)
                    print(f"[debug] {k}: {type(v).__name__} = "
                          f"{s if len(s) < 200 else s[:200] + '…'}", flush=True)
        except Exception as e:
            print(f"[debug] processor inspection failed: {type(e).__name__}: {e}",
                  flush=True)


# ----- Probe runners ----------------------------------------------------------

def probe_classify(
    runner: QwenRunner,
    prompt_id: str,
    *,
    out_suffix: str = "",
    rows: Optional[list] = None,
    out_override: Optional[Path] = None,
    debug_tensors: bool = False,
) -> None:
    """Probe 1 (A, t=0) or Probe 2 (C, Qwen defaults). One pass per clip.

    `rows` defaults to load_36_clip_set(); pass a smaller subset for smoke runs.
    `out_override` bypasses the standard output_path() naming when set.
    """
    if prompt_id == "promptA":
        prompt_text = ga.PROMPT_A
        system_instruction = None
        temperature = 0.0
        rep_penalty = None
    elif prompt_id == "promptC":
        prompt_text = ga.PROMPT_C
        system_instruction = ga.SYSTEM_INSTRUCTION_C
        # Qwen2.5-VL official generation_config defaults:
        # temperature=1e-06, repetition_penalty=1.05, do_sample=True.
        temperature = 1e-6
        rep_penalty = 1.05
    else:
        raise ValueError(f"unknown prompt_id: {prompt_id}")

    out = out_override or output_path(runner.model_size, prompt_id, suffix=out_suffix)
    if rows is None:
        rows = load_36_clip_set()
    done = already_done(out, key_fields=("clip",))
    todo = [r for r in rows if (r["clip"],) not in done]
    print(f"[probe {prompt_id}] {len(rows)} total, {len(done)} done, {len(todo)} todo → {out.name}",
          flush=True)

    for i, r in enumerate(todo, 1):
        try:
            res = runner.run(
                r["clip"], prompt_text,
                system_instruction=system_instruction,
                temperature=temperature, repetition_penalty=rep_penalty,
                debug_tensors=debug_tensors and i == 1,  # only on first clip
            )
            parsed = parse_qwen_json(res["text"])
            classification = parsed.get("classification")
            confidence = parsed.get("confidence")
            reasoning = (
                parsed.get("reasoning") or parsed.get("evidence")
                or parsed.get("observation") or ""
            )
            error = None
        except Exception as e:
            res = {"text": "", "prompt_tokens": None, "candidate_tokens": None,
                   "total_tokens": None, "finish_reason": None, "wall_clock_seconds": None}
            classification, confidence, reasoning = None, None, ""
            error = f"{type(e).__name__}: {e}"

        row = {
            "clip": r["clip"],
            "video_rel": r["video_rel"],
            "split": r["split"],
            "source": r["source"],
            "true_label": r["true_label"],
            "model": runner.model_path,
            "prompt_id": prompt_id,
            "temperature": temperature,
            "repetition_penalty": rep_penalty,
            "qwen_label": classification,
            "confidence": confidence,
            "reasoning_text": reasoning,
            "raw_text": res["text"],
            "prompt_tokens": res["prompt_tokens"],
            "candidate_tokens": res["candidate_tokens"],
            "total_tokens": res["total_tokens"],
            "finish_reason": res["finish_reason"],
            "wall_clock_seconds": res["wall_clock_seconds"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "error": error,
        }
        append_row(out, row)
        agree = "✓" if classification == r["true_label"] else "✗"
        print(f"  [{i:2d}/{len(todo)}] {r['source']} {Path(r['clip']).name[:40]:40} "
              f"truth={r['true_label']:<10} qwen={classification or 'ERR':<10} {agree} "
              f"({res['wall_clock_seconds'] or 0:.1f}s)",
              flush=True)


def probe_describe(
    runner: QwenRunner,
    reps: int = 5,
    temperature: float = 0.7,
    *,
    out_suffix: str = "",
) -> None:
    """Probe 3: description-only on the 10-clip Gemini probe subset, N reps each."""
    out = output_path(runner.model_size, "description_probe", suffix=out_suffix)
    rows = load_10_probe_clips()
    done = already_done(out, key_fields=("clip", "rep_index"))
    plan = [(r, k) for r in rows for k in range(reps) if (r["clip"], k) not in done]
    print(f"[probe description] {len(rows)} clips × {reps} reps = {len(rows)*reps} total, "
          f"{len(done)} done, {len(plan)} todo → {out.name}", flush=True)

    for i, (r, k) in enumerate(plan, 1):
        try:
            res = runner.run(
                r["clip"], ga.PROBE_PROMPT,
                system_instruction=None,
                temperature=temperature, repetition_penalty=None,
            )
            parsed = parse_qwen_json(res["text"])
            observation = parsed.get("observation", "")
            error = None
        except Exception as e:
            res = {"text": "", "prompt_tokens": None, "candidate_tokens": None,
                   "total_tokens": None, "finish_reason": None, "wall_clock_seconds": None}
            observation = ""
            error = f"{type(e).__name__}: {e}"

        row = {
            "clip": r["clip"],
            "video_rel": r["video_rel"],
            "split": r["split"],
            "source": r["source"],
            "true_label": r["true_label"],
            "model": runner.model_path,
            "prompt_id": "description_probe",
            "temperature": temperature,
            "repetition_penalty": None,
            "rep_index": k,
            "observation": observation,
            "raw_text": res["text"],
            "prompt_tokens": res["prompt_tokens"],
            "candidate_tokens": res["candidate_tokens"],
            "total_tokens": res["total_tokens"],
            "finish_reason": res["finish_reason"],
            "wall_clock_seconds": res["wall_clock_seconds"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "error": error,
        }
        append_row(out, row)
        snippet = (observation or res["text"]).replace("\n", " ")[:80]
        print(f"  [{i:2d}/{len(plan)}] {r['source']} rep{k} {Path(r['clip']).name[:30]:30} "
              f"({res['wall_clock_seconds'] or 0:.1f}s) {snippet}", flush=True)


# ----- Smoke-test subset (5 clips for fix verification) -----------------------

# Per docs/qwen-fix-and-revalidate-spec.md §3 Phase 3: 3 action clips on sources
# known for strong ear motion (S3 ×2, S10) + 2 obvious-background clips on
# stable sources (S1, S12). Selected from the 36-clip Gemini stratified subset.
SMOKE_TEST_CLIPS_SELECTOR = [
    ("S3",  "action",     "action_S3.mp4_2_.mp4"),
    ("S3",  "action",     "action_S3.mp4_8_.mp4"),
    ("S10", "action",     "action_S10.mp4_0_.mp4"),
    ("S1",  "background", "background_S1.mp4_11_.mp4"),
    ("S12", "background", "background_S12.mp4_2_.mp4"),
]

def load_smoke_clip_set() -> list[dict]:
    """Pick the 5 spec-defined smoke-test clips out of the 36-clip set."""
    full = load_36_clip_set()
    by_basename = {Path(r["clip"]).name: r for r in full}
    selected = []
    for src, lbl, name in SMOKE_TEST_CLIPS_SELECTOR:
        if name not in by_basename:
            sys.exit(f"ERROR: smoke-test clip {name!r} not in 36-clip set")
        r = by_basename[name]
        if r["source"] != src or r["true_label"] != lbl:
            sys.exit(f"ERROR: smoke-test clip {name!r} mismatch "
                     f"(expected {src}/{lbl}, got {r['source']}/{r['true_label']})")
        selected.append(r)
    return selected


# ----- CLI --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--probe", type=int, required=True, choices=[1, 2, 3],
                    help="1=promptA t=0, 2=promptC Qwen-defaults, 3=description 5×10")
    ap.add_argument("--model-size", default="7B", choices=list(MODEL_PATHS.keys()))
    ap.add_argument("--reps", type=int, default=5, help="probe 3 only")
    ap.add_argument("--describe-temp", type=float, default=0.7, help="probe 3 only")
    ap.add_argument("--out-suffix", default="",
                    help="append to output filename (e.g. '_v2' to preserve v1 evidence)")
    ap.add_argument("--smoke-test", action="store_true",
                    help="probe 1/2 only: run on the 5-clip smoke set "
                         "(3 action S3,S3,S10 + 2 bg S1,S12) instead of the full 36; "
                         "writes to outputs/fix-verification-smoke-test.jsonl regardless of --out-suffix")
    ap.add_argument("--debug-tensors", action="store_true",
                    help="print processor tensor shapes on the first clip only — "
                         "used for Phase 2 fix verification (see fix spec §3)")
    args = ap.parse_args()

    runner = QwenRunner(args.model_size)
    if args.probe == 1 or args.probe == 2:
        prompt_id = "promptA" if args.probe == 1 else "promptC"
        rows = load_smoke_clip_set() if args.smoke_test else None
        out_override = (
            OUTPUTS / "fix-verification-smoke-test.jsonl" if args.smoke_test else None
        )
        probe_classify(
            runner, prompt_id,
            out_suffix=args.out_suffix,
            rows=rows,
            out_override=out_override,
            debug_tensors=args.debug_tensors,
        )
    elif args.probe == 3:
        if args.smoke_test:
            sys.exit("ERROR: --smoke-test only applies to --probe 1 or 2")
        probe_describe(
            runner, reps=args.reps, temperature=args.describe_temp,
            out_suffix=args.out_suffix,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
