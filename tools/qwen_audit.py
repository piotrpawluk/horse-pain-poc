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
    "7B":  "mlx-community/Qwen2.5-VL-7B-Instruct-bf16",
    "32B": "mlx-community/Qwen2.5-VL-32B-Instruct-4bit",
}

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

    def run(
        self,
        clip_path: str,
        user_text: str,
        *,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        repetition_penalty: Optional[float] = None,
        max_tokens: int = 256,
    ) -> dict:
        """Single inference. Returns a dict with text + token counts + latency."""
        content = []
        content.append({"type": "video", "video": clip_path, "fps": 10})
        content.append({"type": "text", "text": user_text})
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": content})
        formatted = self._apply_chat_template(
            self.processor, self.config, prompt=messages, num_images=0, num_audios=0
        )
        kwargs = dict(temperature=temperature, max_tokens=max_tokens, verbose=False)
        if repetition_penalty is not None:
            kwargs["repetition_penalty"] = repetition_penalty
        t0 = time.time()
        result = self._generate(
            self.model, self.processor, prompt=formatted, video=clip_path, **kwargs
        )
        dt = time.time() - t0
        return {
            "text": getattr(result, "text", str(result)),
            "prompt_tokens": getattr(result, "prompt_tokens", None),
            "candidate_tokens": getattr(result, "generation_tokens", None),
            "total_tokens": getattr(result, "total_tokens", None),
            "finish_reason": getattr(result, "finish_reason", None),
            "wall_clock_seconds": dt,
        }


# ----- Probe runners ----------------------------------------------------------

def probe_classify(runner: QwenRunner, prompt_id: str) -> None:
    """Probe 1 (A, t=0) or Probe 2 (C, Qwen defaults). One pass per clip."""
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

    out = output_path(runner.model_size, prompt_id)
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


def probe_describe(runner: QwenRunner, reps: int = 5, temperature: float = 0.7) -> None:
    """Probe 3: description-only on the 10-clip Gemini probe subset, N reps each."""
    out = output_path(runner.model_size, "description_probe")
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


# ----- CLI --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--probe", type=int, required=True, choices=[1, 2, 3],
                    help="1=promptA t=0, 2=promptC Qwen-defaults, 3=description 5×10")
    ap.add_argument("--model-size", default="7B", choices=list(MODEL_PATHS.keys()))
    ap.add_argument("--reps", type=int, default=5, help="probe 3 only")
    ap.add_argument("--describe-temp", type=float, default=0.7, help="probe 3 only")
    args = ap.parse_args()

    runner = QwenRunner(args.model_size)
    if args.probe == 1:
        probe_classify(runner, "promptA")
    elif args.probe == 2:
        probe_classify(runner, "promptC")
    elif args.probe == 3:
        probe_describe(runner, reps=args.reps, temperature=args.describe_temp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
