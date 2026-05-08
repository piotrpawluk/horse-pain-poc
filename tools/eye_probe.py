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

# Gemini 3.1 Pro Preview — v2 (2026-05-08): research-grounded refactor per
# Vertex Gemini 3 prompting guide. Drops conversational scaffolding (Gemini 3
# "favors directness over verbosity, may over-analyze verbose prompts"); uses
# XML tags consistently (no XML+markdown mix); persona/grounding moved to
# system_instruction; data first / instruction last; negative anchor as final
# line; inline JSON-schema text dropped (response_schema config handles it).
EYE_PROMPT_GEMINI_31 = """\
<task>
Classify the attached short clip (0.2–2 s) of a horse's eye region as
either "action" or "background".
</task>

<definitions>
- action: at least one eye exhibits a visible frame-to-frame change —
  eyelid motion, blink, tension shift, asymmetry, gaze shift, or a change
  in how much sclera (white of eye) is visible.
- background: the eye region appears stable across all frames; no
  detectable change in lid position, gaze, sclera exposure, or surrounding
  tissue.
</definitions>

<output>
Return JSON matching the provided schema:
- evidence: one sentence citing the specific visual change (or its
  absence) observed between frames.
- classification: "action" or "background".
- confidence: 0.0–1.0, calibrated to how clearly the evidence supports
  the label.
</output>

Based on the clip above, decide the label using only what is visible
across frames. If the clip yields fewer than two interpretable frames of
the eye, classify as "background".
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

# v2: concise binary-classifier role; "ground in pixel-level evidence" + the
# ambiguity-fallback constraint moved here per Gemini 3 guidance ("Place
# behavioral constraints and role definitions in the System Instruction").
EYE_SYSTEM_INSTRUCTION_GEMINI_31 = (
    "You are a binary visual classifier for equine ophthalmic motion analysis. "
    "Ground every decision strictly in pixel-level evidence visible across the "
    "provided video frames. Do not draw on outside knowledge of horse behavior. "
    "If frame-to-frame change is ambiguous or only one frame is interpretable, "
    "prefer \"background\" and reflect that in confidence."
)

# Qwen-VL — v2 (2026-05-08): research-grounded refactor per QwenLM
# video-grounding template (model-card issue #837) + Qwen3-VL upgrade.
# Markdown headers `### Task / ### Definitions / ### Guidelines / ### Output`
# is the team's own pattern. Frame-index evidence (`frames_compared`) anchors
# attention to specific frames rather than templated language. Negative
# anchor (eye not visible → bg 0.5) gives a defined fallback. Sclera rename
# (was "white-of-eye") matches the anatomical term VLMs respond to better.
# System role is now extracted to EYE_SYSTEM_INSTRUCTION_QWEN below — passed
# via QwenRunner's messages_list / manual_chatml path (resolved at smoke).
# v3 (2026-05-08): kill all escape hatches. v2 collapsed to 100% background
# via the negative anchor ("if eye not visible, output background 0.5") which
# was a fully-formed escape template the model picked verbatim. v3 design:
#   - Removes `confidence` (self-reported floats are noise on hard tasks).
#   - Removes the negative anchor (no "if eye not visible..." escape).
#   - Renames `frames_compared` → `most_changed_frame_pair` with i ≠ j
#     constraint. Generic [0,1]/[0,0] template is now syntactically invalid.
#   - Requires `observed_change` to name a specific eye feature (eyelid /
#     sclera / gaze / asymmetry / muscle tension); generic "no change" without
#     a named feature fails validation.
#   - Asymmetric rule: model must explicitly check ≥3 features across the pair
#     before claiming "background", raising the cost of background relative to
#     action. Forces the model to *look* rather than skip.
#   - Adds `frames_examined` and `eye_visible_in_frames` so we can detect the
#     "no eye reached the encoder" failure mode separately from prompt issues.
EYE_PROMPT_QWEN = """\
Examine the eye region across all frames of this clip.

Output schema (strict JSON):
{
  "frames_examined": <int: total number of frames you saw>,
  "eye_visible_in_frames": [<frame indices where at least one eye is clearly visible>],
  "most_changed_frame_pair": [<i>, <j>],
  "observed_change": "<one sentence comparing frame i and frame j; must name a specific eye feature (eyelid / sclera / gaze / asymmetry / muscle tension) and how it differs, or what you checked if nothing differs>",
  "classification": "action" | "background"
}

Rules:
1. i and j in most_changed_frame_pair MUST be different indices.
2. observed_change MUST reference frame i, frame j, and at least one named eye feature.
3. Choose "action" if any named eye feature differs between frame i and frame j.
4. Choose "background" ONLY if you have explicitly checked at least three named eye features across the pair and none differs.
"""

# v3: vet-vision-analyst role (was binary classifier in v2). The system
# prompt explicitly catalogues features to look for and forbids generic
# "no change" without naming what was checked. Output discipline ("raw
# JSON only, no markdown, no code fences") is in the system role per
# Qwen2.5-VL training pattern.
EYE_SYSTEM_INSTRUCTION_QWEN = """\
You are a veterinary vision analyst. You examine short video clips of a horse's eye region and report whether the eye changes during the clip.

Your job is to find evidence of motion if it is present. Examine specifically:
- eyelid position (upper and lower lid)
- blinking, squinting, or partial closure
- sclera (the white of the eye) becoming more or less visible
- pupil or gaze direction shift
- visible muscle tension or wrinkling around the orbit
- left-right asymmetry between the two eyes that changes across frames

You MUST pick two specific frames and compare them. Generic statements like "no change" or "the eyes look the same" are not acceptable. If the eyes truly look identical, name what you checked (e.g., "eyelid position identical in frames 0 and 7; sclera not visible in either; no asymmetry change").

Output raw JSON only. No markdown, no code fences, no preamble."""

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
    # v2 (2026-05-08): swapped to Qwen3-VL — model_size resolved at smoke-test
    # via QWEN3_V2_PRIORITY (30B-A3B-4bit → 8B-bf16 → 8B → 7B). System role
    # extracted, temp greedy (was 1e-6 sampling).
    "qwen3-vl": {
        "model_size": None,  # resolved at smoke; recorded in JSONL
        "prompt": EYE_PROMPT_QWEN,
        "system_instruction": EYE_SYSTEM_INSTRUCTION_QWEN,
        "temperature": 0.0,
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

def output_path(model_id: str, suffix: str = "_v2") -> Path:
    """v2 default suffix preserves v1 evidence; pass suffix='' to overwrite v1."""
    safe = model_id.replace("/", "_")
    return OUTPUTS / f"eye_probe_{safe}{suffix}.jsonl"


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

def call_gemini(client, clip_path: str, cfg: dict, *, max_retries: int = 3) -> dict:
    """Variant of call_gemini_real with media_resolution support for Gemini 3.x.

    v2 (2026-05-08): added retry logic — Gemini 3.1 Pro Preview was observed
    returning RemoteProtocolError ("Server disconnected without sending a
    response") on ~30% of calls. Retries with exponential backoff (5s, 15s, 30s)
    on transient errors before giving up.
    """
    from google.genai import types
    import httpx

    last_exc = None
    for attempt in range(max_retries):
        try:
            return _call_gemini_once(client, clip_path, cfg, types)
        except (httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                backoff = 5 * (3 ** attempt)  # 5s, 15s, 45s
                print(f"    [retry {attempt+1}/{max_retries}] {type(exc).__name__}: "
                      f"sleeping {backoff}s", flush=True)
                time.sleep(backoff)
            else:
                raise
        except Exception:
            # Non-transient — re-raise immediately, don't waste retries.
            raise
    raise last_exc  # unreachable, defensive


def _call_gemini_once(client, clip_path: str, cfg: dict, types) -> dict:
    """Single (un-retried) Gemini call. Extracted for retry wrapping."""
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
            "prompt_id": "eye_v3",
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


def validate_qwen_v3_response(parsed: dict) -> str:
    """v3: returns 'ok' or a failure tag. Use to track collapse rate.

    The user's v3 spec defines collapse modes explicitly:
      - collapse:same_frame      — most_changed_frame_pair has i == j
      - collapse:no_change_template — generic 'no change' phrasing
      - collapse:no_feature_named — observed_change cites no named eye feature
      - inconsistent:action_with_no_change — claims action while saying no change
    """
    pair = parsed.get("most_changed_frame_pair") or []
    if len(pair) != 2 or pair[0] == pair[1]:
        return "collapse:same_frame"
    obs = (parsed.get("observed_change") or "").lower()
    if any(t in obs for t in ["no change", "no visible change",
                              "look the same", "appear the same",
                              "looks the same", "appears the same"]):
        if parsed.get("classification") == "action":
            return "inconsistent:action_with_no_change"
        return "collapse:no_change_template"
    features = ["eyelid", "sclera", "gaze", "pupil", "asymmetry",
                "tension", "wrinkl", "lid", "blink", "squint", "orbit"]
    if not any(f in obs for f in features):
        return "collapse:no_feature_named"
    return "ok"


def _resolve_qwen3_runner(qa, smoke_clip: str, user_text: str,
                          system_instruction: str):
    """v2: walk QWEN3_V2_PRIORITY chain, return first runner that loads AND generates.

    Each candidate is `mlx_vlm.load`-ed and then probed end-to-end on
    `smoke_clip`. Failures at either step (404, OOM, broadcast-shape errors
    inside MLX, etc.) fall through to the next model in the chain. Returns
    (runner, size_key, repo, role_mode, vision_marker). If all fail, raises.
    """
    last_exc = None
    for size_key in qa.QWEN3_V2_PRIORITY:
        repo = qa.MODEL_PATHS[size_key]
        try:
            print(f"[qwen v2] trying {size_key} = {repo} ...", flush=True)
            runner = qa.QwenRunner(model_size=size_key)
            print(f"[qwen v2] loaded: {repo}", flush=True)
        except Exception as exc:  # noqa: BLE001 - empirical fallback
            print(f"[qwen v2] LOAD FAIL {size_key}: {type(exc).__name__}: {exc}",
                  flush=True)
            last_exc = exc
            continue

        # End-to-end probe: try each system_role_mode until one generates OK.
        try:
            role_mode, vision_marker = runner.resolve_system_role_mode(
                smoke_clip, user_text, system_instruction, end_to_end=True,
            )
            # resolve_system_role_mode may return ("inlined","<|video_pad|>") even
            # if every mode crashed — verify the resolved mode actually works.
            try:
                runner.run(
                    smoke_clip, user_text, system_instruction=system_instruction,
                    temperature=0.0, repetition_penalty=1.05, max_tokens=16,
                    system_role_mode=role_mode,
                )
            except Exception as gen_exc:  # noqa: BLE001
                print(f"[qwen v2] GEN FAIL {size_key} mode={role_mode}: "
                      f"{type(gen_exc).__name__}: {gen_exc}", flush=True)
                last_exc = gen_exc
                # Free the model before fallthrough — large MoE may be holding
                # ~16-20 GB of unified memory. Best-effort.
                try:
                    del runner.model, runner.processor
                except Exception:  # noqa: BLE001
                    pass
                continue
            print(f"[qwen v2] PROBE OK {size_key} mode={role_mode} "
                  f"marker={vision_marker}", flush=True)
            return runner, size_key, repo, role_mode, vision_marker
        except Exception as exc:  # noqa: BLE001
            print(f"[qwen v2] PROBE FAIL {size_key}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            last_exc = exc

    raise RuntimeError(
        f"all Qwen3 v2 candidates failed; last={last_exc}"
    ) from last_exc


def run_qwen() -> None:
    cfg = CONFIGS["qwen3-vl"]
    out = output_path("qwen3-vl")

    sys.path.insert(0, str(POC_DIR / "tools"))
    import qwen_audit as qa

    rows = load_36_clip_set()
    smoke_clip = rows[0]["clip"]
    runner, size_key, repo, role_mode, vision_marker = _resolve_qwen3_runner(
        qa, smoke_clip, cfg["prompt"], cfg["system_instruction"],
    )
    print(f"[qwen v2] resolved: model={repo} role_mode={role_mode} "
          f"vision_marker={vision_marker}", flush=True)

    done = already_done(out)
    todo = [r for r in rows if r["clip"] not in done]
    print(f"[qwen3-vl] {len(rows)} total, {len(done)} done, {len(todo)} todo → {out.name}",
          flush=True)

    for i, r in enumerate(todo, 1):
        try:
            res = runner.run(
                r["clip"], cfg["prompt"],
                system_instruction=cfg["system_instruction"],
                temperature=cfg["temperature"],
                repetition_penalty=cfg["repetition_penalty"],
                system_role_mode=role_mode,
                max_tokens=220,  # v3 user spec — tight budget, classification + short evidence
            )
            parsed = qa.parse_qwen_json(res["text"])
            cls = parsed.get("classification")
            # v3: confidence dropped (was a collapse vector); kept None for schema parity.
            confidence = parsed.get("confidence")
            # v3 evidence: observed_change is the structured-output field;
            # fall back to v2/v1 keys for compatibility with older runs.
            evidence = (
                parsed.get("observed_change")
                or parsed.get("evidence")
                or parsed.get("reasoning")
                or ""
            )
            # v3: most_changed_frame_pair (preferred) ↔ frames_compared (v2 key)
            frames_compared = (
                parsed.get("most_changed_frame_pair")
                or parsed.get("frames_compared")
            )
            validation_tag = validate_qwen_v3_response(parsed)
            err = None
        except Exception as e:
            res = {"text": "", "prompt_tokens": None, "candidate_tokens": None,
                   "total_tokens": None, "finish_reason": None,
                   "wall_clock_seconds": None, "video_marker": None,
                   "system_role_mode_used": role_mode}
            cls, confidence, evidence, frames_compared = None, None, "", None
            validation_tag = "error"
            err = f"{type(e).__name__}: {e}"

        row = {
            "clip": r["clip"],
            "video_rel": r["video_rel"],
            "split": r["split"],
            "source": r["source"],
            "rme_label": r["rme_label"],
            "model": "qwen3-vl",
            "model_version": repo,
            "model_size_key": size_key,
            "prompt_id": "eye_v3",
            "temperature": cfg["temperature"],
            "repetition_penalty": cfg["repetition_penalty"],
            "system_instruction_used": cfg["system_instruction"] is not None,
            "system_role_mode": res.get("system_role_mode_used"),
            "video_marker": res.get("video_marker"),
            "classification": cls,
            "confidence": confidence,
            "evidence_or_reasoning": evidence,
            "frames_compared": frames_compared,
            "validation_tag": validation_tag,  # v3: collapse-mode tracker
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
                    choices=["gemini-2.5-pro", "gemini-3.1-pro-preview", "qwen3-vl"],
                    help="Which MLLM to run (v2: qwen3-vl replaces qwen-7b-bf16)")
    args = ap.parse_args()

    if args.model.startswith("gemini-"):
        run_gemini(args.model)
    else:
        run_qwen()
    return 0


if __name__ == "__main__":
    sys.exit(main())
