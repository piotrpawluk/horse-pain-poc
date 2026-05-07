#!/usr/bin/env python3
"""Eye-region probe — cross-vendor MLLM test on RHpE eye behavior.

Branch: experiment/eye-probe
Spec: Plans/przeanalizuj-oba-dokumenty-w-serialized-crab.md (May 2026)

Three MLLMs run on the same 36-clip RME stratified subset Piotr's audit
covered. Per-model dedicated prompts + parameter configs informed by each
vendor's current best-practice docs (research notes inline below):

  - Gemini 2.5 Pro: broad direct-classify (PROMPT_A pattern that worked
    in audit-followup ear track), temperature=0, no system instruction.
    The over-detection corner of the bifurcation we identified on ear.

  - Gemini 3.1 Pro Preview: evidence-cite-then-classify with system
    instruction (Gemini 3 best-practice per Vertex prompting guide +
    philschmid 2026 notes). temperature=1.0, thinking_level=low.
    NEW: media_resolution="high" — Gemini 3 supports per-call media
    resolution; "high" is appropriate for small ROIs like eye region
    (Gemini 3 prompting guide 2026).

  - Qwen2.5-VL-7B-Instruct-bf16: system-prompt-style adapted from QwenLM
    video-grounding example (Qwen2.5-VL HF model card). temperature=1e-6,
    repetition_penalty=1.05 (Qwen generation_config defaults). System
    instruction inlined into user-text per mlx-vlm 0.5.0 video-routing
    bug workaround (Lesson 16).

Output: outputs/eye_probe_<model>_<config>.jsonl, one JSONL per (model,
config) pair. Same schema as ear-track JSONLs for downstream analysis.

User does the verification pass in outputs/eye_verification_clips.txt
(separate file, no MLLM outputs visible — blind-first methodology).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

POC_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = POC_DIR / "outputs"
sys.path.insert(0, str(POC_DIR / "tools"))

# ---- Per-model dedicated eye prompts ----------------------------------------

# Gemini 2.5 Pro — broad direct-classify (matches the over-detection mode that
# worked on ear). No system instruction; minimal scaffolding; let the model do
# its broad-perception thing. JSON schema strict.
EYE_PROMPT_GEMINI_25 = """\
Watch this short video clip of a horse. The clip shows the horse's
eye region. Determine whether the horse shows visible eye-region
change during the clip — eye motion, blinking, asymmetric expression,
tension, white-of-eye visible, eyelid drooping, or any other visible
change in or around the eyes.

Respond with strict JSON only:
{"classification": "action" | "background",
 "confidence": <float 0.0-1.0>,
 "reasoning": "<one sentence, max 25 words>"}
"""

EYE_SCHEMA_GEMINI_25 = {
    "type": "object",
    "properties": {
        "classification": {"type": "string", "enum": ["action", "background"]},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["classification", "confidence", "reasoning"],
}

# Gemini 3.1 Pro Preview — Vertex Gemini 3 prompting guide best-practice:
# system_instruction grounding role, evidence-cite-then-classify in the user
# prompt, no negative constraints. plus media_resolution="high" for small ROI.
EYE_PROMPT_GEMINI_31 = """\
The clip you are about to watch is approximately 0.2 to 2 seconds long
and shows the eye region of a horse.

Cite specific visual evidence from the clip — what you see in the eyes
or surrounding eye region across frames — and on that basis decide one
of two labels:
- "action": at least one eye shows a visible change during the clip
            (motion, blink, tension shift, asymmetry, eyelid movement,
            white-of-eye becoming more or less visible).
- "background": the eyes appear consistent throughout the clip with no
            visible change in their state or surrounding region.

Respond with strict JSON only:
{"evidence": "<one sentence citing specific frame-to-frame change or its absence>",
 "classification": "action" | "background",
 "confidence": <float 0.0-1.0>}
"""

EYE_SCHEMA_GEMINI_31 = {
    "type": "object",
    "properties": {
        "evidence": {"type": "string"},
        "classification": {"type": "string", "enum": ["action", "background"]},
        "confidence": {"type": "number"},
    },
    "required": ["evidence", "classification", "confidence"],
}

EYE_SYSTEM_INSTRUCTION_GEMINI_31 = (
    "You are a careful equine-behavior video annotator. Cite specific visible "
    "evidence from the frames you observe before reaching any conclusion. Do "
    "not refuse to commit unless the clip is genuinely uninterpretable (e.g., "
    "eyes are not visible or the eye region is occluded). When uncertain, "
    "report the most likely classification with appropriate confidence."
)

# Qwen 2.5-VL-7B-Instruct — adapted from the QwenLM video-grounding system-
# prompt template (Qwen2.5-VL HF model card). System instruction is INLINED
# into the user-text string per Lesson 16 (mlx-vlm 0.5.0 list+video routing
# bug). Same ear-track pattern.
EYE_PROMPT_QWEN = """\
You are a highly capable AI assistant trained to analyze short video
clips of horses and identify visible changes in the eye region. Cite
specific visible evidence from the frames you observe before reaching
any conclusion. When uncertain, report the most likely classification
with appropriate confidence.

The clip you are about to watch is approximately 0.2 to 2 seconds long
and shows the eye region of a horse.

Cite specific visual evidence from the clip — what you observe in the
eyes or surrounding eye region across frames — and on that basis
decide one of two labels:
- "action": at least one eye shows a visible change during the clip
            (motion, blink, tension shift, asymmetry, eyelid movement,
            white-of-eye becoming more or less visible).
- "background": the eyes appear consistent throughout the clip with
            no visible change.

Respond with strict JSON only:
{"evidence": "<one sentence citing specific frame-to-frame change or its absence>",
 "classification": "action" | "background",
 "confidence": <float 0.0-1.0>}
"""

# ---- Per-model configs ------------------------------------------------------

CONFIGS = {
    "gemini-2.5-pro": {
        "model_id": "gemini-2.5-pro",
        "prompt": EYE_PROMPT_GEMINI_25,
        "schema": EYE_SCHEMA_GEMINI_25,
        "temperature": 0.0,
        "thinking_level": None,
        "system_instruction": None,
        "media_resolution": None,
    },
    "gemini-3.1-pro-preview": {
        "model_id": "gemini-3.1-pro-preview",
        "prompt": EYE_PROMPT_GEMINI_31,
        "schema": EYE_SCHEMA_GEMINI_31,
        "temperature": 1.0,
        "thinking_level": "low",
        "system_instruction": EYE_SYSTEM_INSTRUCTION_GEMINI_31,
        "media_resolution": "high",
    },
    "qwen-7b-bf16": {
        "model_path": "mlx-community/Qwen2.5-VL-7B-Instruct-bf16",
        "prompt": EYE_PROMPT_QWEN,
        "system_instruction": None,  # already inlined in prompt above
        "temperature": 1e-6,
        "repetition_penalty": 1.05,
    },
}


# ---- Manifest reuse ---------------------------------------------------------

GEMINI_36_SOURCE = OUTPUTS / "gemini_audit_results_gemini-2.5-pro_promptA.jsonl"


def load_36_clip_set() -> list[dict]:
    """Reuse the 36-clip set from the audit-followup ear-track."""
    rows, seen = [], set()
    with open(GEMINI_36_SOURCE) as f:
        for line in f:
            r = json.loads(line)
            if r["clip"] in seen:
                continue
            seen.add(r["clip"])
            rows.append({
                "clip": r["clip"],
                "video_rel": r["video_rel"],
                "split": r["split"],
                "source": r["source"],
                "rme_label": r["true_label"],  # ear label, not eye — RME doesn't have eye labels
            })
    if len(rows) != 36:
        sys.exit(f"ERROR: expected 36 unique clips, got {len(rows)}")
    return rows


# ---- Output handling --------------------------------------------------------

def output_path(model_id: str) -> Path:
    safe = model_id.replace("/", "_")
    return OUTPUTS / f"eye_probe_{safe}.jsonl"


def already_done(out: Path) -> set:
    if not out.exists():
        return set()
    done = set()
    with open(out) as f:
        for line in f:
            try:
                done.add(json.loads(line)["clip"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def append_row(out: Path, row: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---- Gemini runner (with media_resolution support added for 3.x) ------------

def call_gemini(client, clip_path: str, cfg: dict) -> dict:
    """Variant of call_gemini_real with media_resolution support for Gemini 3.x."""
    from google.genai import types

    uploaded = client.files.upload(file=clip_path)
    while uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state.name != "ACTIVE":
        raise RuntimeError(f"upload state={uploaded.state.name}")

    video_part_kwargs = {
        "file_data": types.FileData(
            file_uri=uploaded.uri,
            mime_type=uploaded.mime_type,
        ),
        "video_metadata": types.VideoMetadata(fps=10.0),
    }
    video_part = types.Part(**video_part_kwargs)

    config_kwargs = {
        "response_mime_type": "application/json",
        "response_schema": cfg["schema"],
        "temperature": cfg["temperature"],
    }
    if cfg.get("system_instruction"):
        config_kwargs["system_instruction"] = cfg["system_instruction"]
    if cfg.get("thinking_level"):
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=getattr(types.ThinkingLevel, cfg["thinking_level"].upper())
        )
    if cfg.get("media_resolution"):
        # Gemini 3.x feature — improves fine-grained visual detail (small ROI like eye).
        try:
            config_kwargs["media_resolution"] = getattr(
                types.MediaResolution, f"MEDIA_RESOLUTION_{cfg['media_resolution'].upper()}"
            )
        except AttributeError:
            # Older SDK — fall back to string
            config_kwargs["media_resolution"] = cfg["media_resolution"].upper()

    t0 = time.time()
    response = client.models.generate_content(
        model=cfg["model_id"],
        contents=[video_part, cfg["prompt"]],
        config=types.GenerateContentConfig(**config_kwargs),
    )
    dt = time.time() - t0
    payload = json.loads(response.text)
    payload["_model_version"] = getattr(response, "model_version", None) or cfg["model_id"]
    payload["_finish_reason"] = (
        str(response.candidates[0].finish_reason)
        if response.candidates and response.candidates[0].finish_reason
        else None
    )
    um = getattr(response, "usage_metadata", None)
    payload["_prompt_tokens"] = getattr(um, "prompt_token_count", None) if um else None
    payload["_candidate_tokens"] = getattr(um, "candidates_token_count", None) if um else None
    payload["_total_tokens"] = getattr(um, "total_token_count", None) if um else None
    payload["_latency_s"] = dt
    return payload


def run_gemini(model_key: str) -> None:
    cfg = CONFIGS[model_key]
    out = output_path(model_key)

    # Lazy client init (loads .env)
    from dotenv import load_dotenv
    load_dotenv(POC_DIR / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY not set; check .env")
    from google import genai
    client = genai.Client(api_key=api_key)

    rows = load_36_clip_set()
    done = already_done(out)
    todo = [r for r in rows if r["clip"] not in done]
    print(f"[{model_key}] {len(rows)} total, {len(done)} done, {len(todo)} todo → {out.name}",
          flush=True)

    for i, r in enumerate(todo, 1):
        try:
            payload = call_gemini(client, r["clip"], cfg)
            cls = payload.get("classification")
            err = None
        except Exception as e:
            payload = {"_error": f"{type(e).__name__}: {e}"}
            cls = None
            err = payload["_error"]

        row = {
            "clip": r["clip"],
            "video_rel": r["video_rel"],
            "split": r["split"],
            "source": r["source"],
            "rme_label": r["rme_label"],
            "model": model_key,
            "model_version": payload.get("_model_version"),
            "prompt_id": "eye",
            "temperature": cfg["temperature"],
            "thinking_level": cfg.get("thinking_level"),
            "media_resolution": cfg.get("media_resolution"),
            "system_instruction_used": cfg.get("system_instruction") is not None,
            "classification": cls,
            "confidence": payload.get("confidence"),
            "evidence_or_reasoning": payload.get("evidence") or payload.get("reasoning"),
            "raw_payload": {k: v for k, v in payload.items() if not k.startswith("_")},
            "prompt_tokens": payload.get("_prompt_tokens"),
            "candidate_tokens": payload.get("_candidate_tokens"),
            "total_tokens": payload.get("_total_tokens"),
            "finish_reason": payload.get("_finish_reason"),
            "latency_s": payload.get("_latency_s"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "error": err,
        }
        append_row(out, row)
        print(f"  [{i:2d}/{len(todo)}] {r['source']} {Path(r['clip']).name[:40]:40} "
              f"→ {cls or 'ERR':<10} ({payload.get('_latency_s', 0) or 0:.1f}s)",
              flush=True)


def run_qwen() -> None:
    cfg = CONFIGS["qwen-7b-bf16"]
    out = output_path("qwen-7b-bf16")

    sys.path.insert(0, str(POC_DIR / "tools"))
    import qwen_audit as qa

    runner = qa.QwenRunner(model_size="7B")
    rows = load_36_clip_set()
    done = already_done(out)
    todo = [r for r in rows if r["clip"] not in done]
    print(f"[qwen-7b-bf16] {len(rows)} total, {len(done)} done, {len(todo)} todo → {out.name}",
          flush=True)

    for i, r in enumerate(todo, 1):
        try:
            res = runner.run(
                r["clip"], cfg["prompt"],
                system_instruction=None,  # already inlined in prompt
                temperature=cfg["temperature"],
                repetition_penalty=cfg["repetition_penalty"],
            )
            parsed = qa.parse_qwen_json(res["text"])
            cls = parsed.get("classification")
            confidence = parsed.get("confidence")
            evidence = parsed.get("evidence")
            err = None
        except Exception as e:
            res = {"text": "", "prompt_tokens": None, "candidate_tokens": None,
                   "total_tokens": None, "finish_reason": None, "wall_clock_seconds": None}
            cls, confidence, evidence = None, None, ""
            err = f"{type(e).__name__}: {e}"

        row = {
            "clip": r["clip"],
            "video_rel": r["video_rel"],
            "split": r["split"],
            "source": r["source"],
            "rme_label": r["rme_label"],
            "model": "qwen-7b-bf16",
            "model_version": cfg["model_path"],
            "prompt_id": "eye",
            "temperature": cfg["temperature"],
            "repetition_penalty": cfg["repetition_penalty"],
            "system_instruction_used": True,  # inlined into prompt
            "classification": cls,
            "confidence": confidence,
            "evidence_or_reasoning": evidence,
            "raw_text": res["text"],
            "prompt_tokens": res["prompt_tokens"],
            "candidate_tokens": res["candidate_tokens"],
            "total_tokens": res["total_tokens"],
            "finish_reason": res["finish_reason"],
            "latency_s": res["wall_clock_seconds"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "error": err,
        }
        append_row(out, row)
        print(f"  [{i:2d}/{len(todo)}] {r['source']} {Path(r['clip']).name[:40]:40} "
              f"→ {cls or 'ERR':<10} ({res['wall_clock_seconds'] or 0:.1f}s)",
              flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True,
                    choices=["gemini-2.5-pro", "gemini-3.1-pro-preview", "qwen-7b-bf16"],
                    help="Which MLLM to run")
    args = ap.parse_args()

    if args.model.startswith("gemini-"):
        run_gemini(args.model)
    else:
        run_qwen()
    return 0


if __name__ == "__main__":
    sys.exit(main())
