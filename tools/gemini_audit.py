#!/usr/bin/env python3
"""Gemini label-noise audit for Read My Ears (RME).

For each clip in vendor/ReadMyEars_Dataset/data/{train,val,test}.csv,
this script asks Gemini 2.5 Pro to classify the clip as 'action' (ear movement)
or 'background' (no ear movement) and compares against the human label.
Disagreements flag clips for re-review.

Justification (from Standard research synthesis, 2026-05-06):
frontier multimodal LLMs do NOT replace V-JEPA-2 on fine-grained motion
(frame sampling at 1 fps misses sub-second deltas) but ARE useful as a
second-opinion layer for surfacing labeling errors.

Run modes:
    python tools/gemini_audit.py --dry-run          # no API calls; stub responses
    python tools/gemini_audit.py --limit 5          # smoke test with 5 clips
    python tools/gemini_audit.py                    # full 283-clip audit (~$6-12)

Output:
    outputs/gemini_audit_results.jsonl   # one row per clip (idempotent, resumable)
    outputs/gemini_audit_summary.json    # per-source agreement table

GDPR caveat: this uses AI Studio (no EU residency) — fine for RME (CC-BY-4.0,
public, anonymized), NOT for field-collected clips with identifiable persons.
For field data, switch to Vertex AI europe-west4. See docs/gemini-integration.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

POC_DIR = Path(__file__).resolve().parent.parent
RME_DATA = POC_DIR / "vendor" / "ReadMyEars_Dataset" / "data"
OUTPUTS = POC_DIR / "outputs"


def results_jsonl(model: str, version: str) -> Path:
    return OUTPUTS / f"gemini_audit_results_{model}_prompt{version.upper()}.jsonl"


def summary_json(model: str, version: str) -> Path:
    return OUTPUTS / f"gemini_audit_summary_{model}_prompt{version.upper()}.json"


def probe_jsonl(model: str) -> Path:
    return OUTPUTS / f"gemini_audit_probe_{model}.jsonl"


# ---- Probe prompt: pure description, no classification --------------------
# Used by --probe mode. We deliberately do NOT ask for a label; the probe
# tests whether 3 reps at temperature ≥0.5 produce stable perceptual
# descriptions of the same clip, independent of any classification
# pressure. If reps are stable → underlying perception is consistent.
# If reps vary → "perception" is sampling artifact, not stable observation.
PROBE_PROMPT = """\
Watch this short video clip of a horse. Describe in 1–2 sentences what
you observe in the ear region across the clip duration: ear position,
any rotation, twitching, or sustained displacement.

Do NOT classify. Just describe what you see.

Respond with strict JSON only:
{"observation": "<1-2 sentence description>"}
"""

PROBE_SCHEMA = {
    "type": "object",
    "properties": {"observation": {"type": "string"}},
    "required": ["observation"],
}


# ---- Prompt A: original generic phrasing -----------------------------------
# Used for the initial fps=1 → fps=10 fix iteration. Asks Gemini to judge
# "actively MOVING" vs "STILL" without anchoring on a labeling protocol.
PROMPT_A = """\
Watch this short video clip of a horse. The clip shows the horse's
ear region. Determine whether the horse is actively MOVING its ears
during the clip (rotation, twitch, or pinning) or holding them STILL.

Respond with strict JSON only:
{"classification": "action" | "background",
 "confidence": <float 0.0-1.0>,
 "reasoning": "<one sentence, max 25 words>"}
"""

SCHEMA_A = {
    "type": "object",
    "properties": {
        "classification": {"type": "string", "enum": ["action", "background"]},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["classification", "confidence", "reasoning"],
}

# ---- Prompt B: anchored on the actual RME labeling protocol ----------------
# RME labels descend from EquiFACS-coded source data (Alves et al. CVPR W'25,
# arXiv 2505.03554). EquiFACS coders apply intensity / duration thresholds —
# fleeting micro-adjustments do not meet the bar for a coded ear AU.
# This prompt anchors Gemini on that population's threshold rather than on
# editorial prose, and forces a perceptual description before classification
# so we can separate "different perception" from "different threshold".
PROMPT_B = """\
This is a clip from a research dataset where trained EquiFACS coders
classified each clip based on whether a specific ear-related Action
Unit (e.g., AD101 "ears forward", AD104 "ear flattener", AD103 "ear
rotator") was occurring during the clip. EquiFACS coders apply
intensity and duration thresholds — fleeting micro-adjustments and
postural settling typically do not meet the threshold for a coded AU.

Step 1: In one sentence, describe what you observe in the ear region
        across the clip duration (be specific about ear position
        changes, rotation, and any sustained displacement).

Step 2: Classify whether a trained EquiFACS coder would have marked
        this clip as containing a coded ear AU.

Respond with strict JSON only:
{"observation": "<one-sentence description>",
 "classification": "action" | "background",
 "confidence": <float 0.0-1.0>}
"""

SCHEMA_B = {
    "type": "object",
    "properties": {
        "observation": {"type": "string"},
        "classification": {"type": "string", "enum": ["action", "background"]},
        "confidence": {"type": "number"},
    },
    "required": ["observation", "classification", "confidence"],
}

# ---- Prompt C: Gemini 3.x best-practice --------------------------------
# Reflects research synthesis (2026-05-07):
#   - Temperature MUST be 1.0 on Gemini 3.x (Google: "strongly recommend
#     keeping at default 1.0; below-1.0 may cause looping or degraded output").
#   - thinking_level=LOW is recommended for classification tasks (HIGH adds
#     cost without quality and is the silent default on 3 Pro / 3.1 Pro).
#   - Vertex prompting guide: avoid broad negative constraints; place task
#     imperative at the end after the data context; use evidence-grounded
#     positive framing. Explicit reasoning request is required because
#     "Gemini 3 is less verbose by default" (Schmid 2025).
#   - System instructions separate role/task from per-clip data.
# This prompt is for the USER content; the SYSTEM instruction is set
# separately via SystemInstruction in the GenerateContentConfig.
PROMPT_C = """\
The clip you are about to watch is approximately 0.2 to 2 seconds long
and shows the ear region of a horse.

Cite specific visual evidence from the clip — what you see in the ears
across frames — and on that basis decide one of two labels:
- "action": at least one ear shows a positional change during the clip
            (rotation, twitching, or pinning).
- "background": the ears remain in essentially the same position
            throughout the clip.

Respond with strict JSON only:
{"evidence": "<one sentence citing specific frame-to-frame change or its absence>",
 "classification": "action" | "background",
 "confidence": <float 0.0-1.0>}
"""

SCHEMA_C = {
    "type": "object",
    "properties": {
        "evidence": {"type": "string"},
        "classification": {"type": "string", "enum": ["action", "background"]},
        "confidence": {"type": "number"},
    },
    "required": ["evidence", "classification", "confidence"],
}

# System instruction that pairs with PROMPT_C.
SYSTEM_INSTRUCTION_C = (
    "You are a careful equine-behavior video annotator. Cite specific visible "
    "evidence from the frames you observe before reaching any conclusion. "
    "Do not refuse to commit unless the clip is genuinely uninterpretable "
    "(e.g., ears are not visible). When uncertain, report the most likely "
    "classification with appropriate confidence."
)

PROMPTS = {
    "a": (PROMPT_A, SCHEMA_A),
    "b": (PROMPT_B, SCHEMA_B),
    "c": (PROMPT_C, SCHEMA_C),
}


# ---- Inventory --------------------------------------------------------------

def source_id(video_path: str) -> str:
    """Extract S1-S12 from filenames like 'videos/action_S8.mp4_3_.mp4'."""
    m = re.search(r"(action|background)_(S\d+)", video_path)
    return m.group(2) if m else "unknown"


def load_inventory() -> pd.DataFrame:
    """Concat train/val/test CSVs; resolve to absolute paths; keep only existing files."""
    frames = []
    for split in ("train", "val", "test"):
        csv = RME_DATA / f"{split}.csv"
        if not csv.exists():
            print(f"  WARNING: {csv} not found, skipping", file=sys.stderr)
            continue
        df = pd.read_csv(csv)
        df["split"] = split
        frames.append(df)
    if not frames:
        sys.exit(f"ERROR: no RME CSVs found in {RME_DATA}")
    inv = pd.concat(frames, ignore_index=True)
    inv["abs_path"] = inv["video"].apply(lambda v: str(RME_DATA / v))
    inv = inv[inv["abs_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    inv["source"] = inv["video"].apply(source_id)
    return inv


# ---- Output handling --------------------------------------------------------

def already_processed(model: str, version: str) -> set[str]:
    """Set of clip paths already in the JSONL output for this model + prompt version."""
    p = results_jsonl(model, version)
    if not p.exists():
        return set()
    done = set()
    with p.open() as f:
        for line in f:
            try:
                row = json.loads(line)
                if "clip" in row and not row.get("error"):
                    done.add(row["clip"])
            except json.JSONDecodeError:
                continue
    return done


def append_result(row: dict, model: str, version: str) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with results_jsonl(model, version).open("a") as f:
        f.write(json.dumps(row) + "\n")


# ---- Gemini call ------------------------------------------------------------

def call_gemini_real(
    client,
    clip_path: str,
    fps: float,
    prompt: str,
    schema: dict,
    model: str,
    temperature: float = 1.0,
    thinking_level: str | None = None,
    system_instruction: str | None = None,
) -> dict:
    """Upload clip, poll until ACTIVE, generate at the requested FPS, parse JSON.

    Default Gemini sampling is 1 fps, which collapses on sub-second RME clips
    (0.24-1.76s = 1-2 frames) into still-image classification. Setting fps≈10
    gives ~5-17 frames per clip, comparable to V-JEPA-2's 16-frame window.

    Returns dict with parsed payload + metadata fields:
        {**parsed_response, "_model_version": str, "_finish_reason": str,
         "_prompt_tokens": int, "_candidate_tokens": int, "_total_tokens": int}
    """
    from google.genai import types

    uploaded = client.files.upload(file=clip_path)
    while uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state.name != "ACTIVE":
        raise RuntimeError(f"upload state={uploaded.state.name}")

    video_part = types.Part(
        file_data=types.FileData(
            file_uri=uploaded.uri,
            mime_type=uploaded.mime_type,
        ),
        video_metadata=types.VideoMetadata(fps=fps),
    )

    config_kwargs = {
        "response_mime_type": "application/json",
        "response_schema": schema,
        "temperature": temperature,
    }
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if thinking_level:
        # ThinkingLevel enum: MINIMAL, LOW, MEDIUM, HIGH (default HIGH on 3.x)
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=getattr(types.ThinkingLevel, thinking_level.upper())
        )

    response = client.models.generate_content(
        model=model,
        contents=[video_part, prompt],
        config=types.GenerateContentConfig(**config_kwargs),
    )
    payload = json.loads(response.text)
    payload["_model_version"] = getattr(response, "model_version", None) or model
    payload["_finish_reason"] = (
        str(response.candidates[0].finish_reason)
        if response.candidates and response.candidates[0].finish_reason
        else None
    )
    um = getattr(response, "usage_metadata", None)
    payload["_prompt_tokens"] = getattr(um, "prompt_token_count", None) if um else None
    payload["_candidate_tokens"] = getattr(um, "candidates_token_count", None) if um else None
    payload["_total_tokens"] = getattr(um, "total_token_count", None) if um else None
    return payload


def call_gemini_stub(clip_path: str, true_label: str, version: str) -> dict:
    """Dry-run stub: deterministic fake response based on filename hash."""
    h = sum(ord(c) for c in clip_path) % 10
    cls = true_label if h < 7 else ("background" if true_label == "action" else "action")
    base = {
        "classification": cls,
        "confidence": 0.5 + (h / 20),
        "_model_version": "stub-dry-run",
        "_finish_reason": "STOP",
        "_prompt_tokens": 0,
        "_candidate_tokens": 0,
        "_total_tokens": 0,
    }
    if version == "a":
        base["reasoning"] = f"[DRY RUN STUB] hash-based pseudo-classification (h={h})."
    elif version == "b":
        base["observation"] = f"[DRY RUN STUB] hash-based fake observation (h={h})."
    elif version == "c":
        base["evidence"] = f"[DRY RUN STUB] hash-based fake evidence (h={h})."
    else:  # probe
        base.pop("classification", None)
        base.pop("confidence", None)
        base["observation"] = f"[DRY RUN STUB PROBE] hash-based observation (h={h})."
    return base


# ---- Main loop --------------------------------------------------------------

def run(args) -> None:
    inv = load_inventory()
    if args.per_source:
        # Stratified sample N per source; stable seed for reproducibility
        inv = (
            inv.groupby("source", group_keys=False)
            .apply(
                lambda g: g.sample(n=min(args.per_source, len(g)), random_state=42),
                include_groups=False,
            )
            .reset_index(drop=True)
        )
        inv["source"] = inv["video"].apply(source_id)
    if args.limit:
        inv = inv.head(args.limit)
    print(f"Inventory: {len(inv)} clips, {inv['source'].nunique()} sources")

    version = args.prompt_version
    model = args.model
    prompt, schema = PROMPTS[version]
    # System instruction defaults: pair PROMPT_C with SYSTEM_INSTRUCTION_C
    # unless an explicit --system-instruction overrides.
    system_instruction = args.system_instruction
    if system_instruction is None and version == "c":
        system_instruction = SYSTEM_INSTRUCTION_C
    print(f"Model: {model}")
    print(f"Prompt version: {version.upper()} (output → {results_jsonl(model, version).name})")
    print(f"Temperature: {args.temperature}, thinking_level: {args.thinking_level}, "
          f"system_instruction: {'<set>' if system_instruction else '<none>'}")

    done = already_processed(model, version) if not args.dry_run else set()
    if done:
        print(f"Resuming: {len(done)} clips already processed, skipping")

    client = None
    if not args.dry_run:
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    interval = 60.0 / args.rpm if args.rpm > 0 else 0.0
    last_call = 0.0

    n_processed = 0
    for _, row in inv.iterrows():
        if row["abs_path"] in done:
            continue

        # Rate limiting
        if not args.dry_run:
            wait = max(0, interval - (time.time() - last_call))
            if wait > 0:
                time.sleep(wait)

        t0 = time.time()
        try:
            if args.dry_run:
                pred = call_gemini_stub(row["abs_path"], row["label"], version)
            else:
                pred = call_gemini_real(
                    client, row["abs_path"], fps=args.fps, prompt=prompt,
                    schema=schema, model=model,
                    temperature=args.temperature,
                    thinking_level=args.thinking_level,
                    system_instruction=system_instruction,
                )
                last_call = time.time()
            err = None
        except Exception as exc:
            pred = {"classification": None, "confidence": None, "reasoning": None,
                    "observation": None, "evidence": None,
                    "_model_version": None, "_finish_reason": None,
                    "_prompt_tokens": None, "_candidate_tokens": None, "_total_tokens": None}
            err = f"{type(exc).__name__}: {exc}"

        latency = time.time() - t0
        result = {
            "clip": row["abs_path"],
            "video_rel": row["video"],
            "split": row["split"],
            "source": row["source"],
            "true_label": row["label"],
            "model": model,
            "model_version": pred.get("_model_version"),
            "prompt_version": version,
            "temperature": args.temperature,
            "thinking_level": args.thinking_level,
            "system_instruction_set": bool(system_instruction),
            "gemini_label": pred.get("classification"),
            "confidence": pred.get("confidence"),
            "reasoning": pred.get("reasoning"),       # populated by prompt A
            "observation": pred.get("observation"),   # populated by prompt B
            "evidence": pred.get("evidence"),         # populated by prompt C
            "agreement": pred.get("classification") == row["label"] if pred.get("classification") else None,
            "finish_reason": pred.get("_finish_reason"),
            "prompt_tokens": pred.get("_prompt_tokens"),
            "candidate_tokens": pred.get("_candidate_tokens"),
            "total_tokens": pred.get("_total_tokens"),
            "latency_s": round(latency, 2),
            "error": err,
        }
        append_result(result, model, version)
        n_processed += 1
        marker = "✓" if result["agreement"] else ("✗" if result["agreement"] is False else "·")
        print(f"  {marker}  {row['source']:5s} {Path(row['abs_path']).name:50s} "
              f"true={row['label']:10s} gemini={(pred.get('classification') or 'ERR'):10s} "
              f"({latency:.1f}s)")

    print(f"\nProcessed {n_processed} clips. Writing summary...")
    write_summary(model, version)


def write_summary(model: str, version: str) -> None:
    rj = results_jsonl(model, version)
    sj = summary_json(model, version)
    if not rj.exists():
        return
    rows = []
    with rj.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    by_source: dict[str, list] = defaultdict(list)
    for r in rows:
        if r.get("agreement") is not None:
            by_source[r["source"]].append(r)

    summary = {
        "total_clips": len(rows),
        "scored_clips": sum(len(v) for v in by_source.values()),
        "errors": sum(1 for r in rows if r.get("error")),
        "overall_agreement": (
            sum(r["agreement"] for r in rows if r.get("agreement") is not None)
            / max(1, sum(1 for r in rows if r.get("agreement") is not None))
        ),
        "per_source": {},
    }
    for src, src_rows in sorted(by_source.items()):
        n = len(src_rows)
        agree = sum(r["agreement"] for r in src_rows)
        disagreements = [r for r in src_rows if not r["agreement"]]
        summary["per_source"][src] = {
            "n": n,
            "agreement_rate": agree / n if n else 0.0,
            "disagreement_count": len(disagreements),
            "mean_confidence_on_disagreements": (
                sum(r["confidence"] for r in disagreements if r["confidence"] is not None)
                / max(1, len(disagreements))
            ),
            "disagreement_clips": [r["video_rel"] for r in disagreements][:10],
        }

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    sj.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary → {sj.relative_to(POC_DIR)}")
    print(f"  Overall agreement: {summary['overall_agreement']:.3f}  "
          f"({summary['scored_clips']} scored, {summary['errors']} errors)")
    print("\nPer-source agreement:")
    for src, info in summary["per_source"].items():
        print(f"  {src:5s} {info['n']:>3d} clips  "
              f"agreement={info['agreement_rate']:.3f}  "
              f"disagreements={info['disagreement_count']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="No API calls; use deterministic stub responses (validates code path)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N clips (smoke test)")
    parser.add_argument("--rpm", type=float, default=6.0,
                        help="Rate limit in requests per minute (default 6 = gemini-2.5-pro free tier)")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Gemini video sampling rate in fps (default 10; the API default of 1 fps "
                             "collapses on sub-second RME clips into still-image classification)")
    parser.add_argument("--per-source", type=int, default=None, metavar="N",
                        help="Stratified sampling: take exactly N clips per source (S1-S12). "
                             "Useful for diverse smoke tests before full runs.")
    parser.add_argument("--prompt-version", choices=["a", "b", "c"], default="a",
                        help="A = original generic 'is the horse moving its ears'. "
                             "B = EquiFACS-anchored with observation-then-classify. "
                             "C = Gemini 3.x best-practice (positive evidence-grounded, "
                             "pairs with SystemInstruction).")
    parser.add_argument("--model", default="gemini-2.5-pro",
                        help="Gemini model name (default: gemini-2.5-pro). "
                             "Examples: gemini-2.5-pro, gemini-3.1-pro-preview")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature. Default 1.0 (Google strongly recommends "
                             "1.0 for Gemini 3.x; below 1.0 may cause looping or degraded output).")
    parser.add_argument("--thinking-level", choices=["minimal", "low", "medium", "high"], default=None,
                        help="Gemini 3.x thinking level. Default: SDK default (HIGH on 3 Pro / 3.1 Pro). "
                             "Recommended for classification: 'low'.")
    parser.add_argument("--system-instruction", type=str, default=None,
                        help="System instruction string. If --prompt-version=c and this is unset, "
                             "the default SYSTEM_INSTRUCTION_C is used automatically.")
    parser.add_argument("--probe", action="store_true",
                        help="Run no-classification probe: load A and B JSONL for current --model, "
                             "find clips where they disagreed, sample N clips, run M reps each at "
                             "--probe-temp with description-only prompt. Tests perceptual stability.")
    parser.add_argument("--probe-clips", type=int, default=10,
                        help="Number of disagreement clips to probe (default 10)")
    parser.add_argument("--probe-reps", type=int, default=3,
                        help="Reps per clip in probe (default 3)")
    parser.add_argument("--probe-temp", type=float, default=0.5,
                        help="Temperature for probe (default 0.5; ≥0.5 reveals sampling variance)")
    parser.add_argument("--probe-suffix", type=str, default="",
                        help="Optional suffix appended to probe output filename "
                             "(e.g. '_temp1.0' to differentiate runs at different parameters).")
    parser.add_argument("--probe-source-model", type=str, default=None,
                        help="Model to use for selecting A-vs-B disagreement clips. "
                             "Defaults to --model. Set this to e.g. 'gemini-2.5-pro' to "
                             "probe a different model on clips selected from the 2.5 Pro "
                             "disagreement set (preserves comparability across probe runs).")
    parser.add_argument("--probe-skip-clips", type=str, default=None,
                        help="Path to a probe JSONL whose clips should be EXCLUDED from "
                             "selection. Useful for N-expansion: feed the original probe's "
                             "JSONL here so the next run picks fresh clips that don't overlap.")
    parser.add_argument("--probe-seed", type=int, default=42,
                        help="Random seed for sampling probe clips (default 42)")
    args = parser.parse_args()

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN — no API calls, stub responses only")
        print("=" * 60)
    else:
        load_dotenv(POC_DIR / ".env")
        if not os.environ.get("GEMINI_API_KEY"):
            print("ERROR: GEMINI_API_KEY is not set.", file=sys.stderr)
            print("       Copy .env.example to .env and fill in your key,", file=sys.stderr)
            print("       or export GEMINI_API_KEY=... in your shell.", file=sys.stderr)
            print("       Get a key at https://aistudio.google.com/app/apikey", file=sys.stderr)
            sys.exit(1)
        try:
            from google import genai  # noqa: F401
        except ImportError:
            print("ERROR: google-genai package not installed.", file=sys.stderr)
            print("       Run: uv pip install google-genai>=0.8 python-dotenv>=1.0", file=sys.stderr)
            sys.exit(1)

    if args.probe:
        run_probe(args)
    else:
        run(args)


# ---- Probe runner ----------------------------------------------------------

def run_probe(args) -> None:
    """No-classification probe: 3 reps per clip at temp ≥0.5 on the description-only prompt.

    Selects clips where A and B disagreed (on this --model). If A or B JSONL
    is missing or has no disagreements, exits cleanly.
    """
    model = args.model
    selection_model = args.probe_source_model or model
    a_path = results_jsonl(selection_model, "a")
    b_path = results_jsonl(selection_model, "b")
    if not a_path.exists() or not b_path.exists():
        sys.exit(
            f"ERROR: probe selection requires A and B JSONL for model={selection_model}.\n"
            f"  Missing: {a_path.name if not a_path.exists() else b_path.name}"
        )
    if selection_model != model:
        print(f"Selecting disagreement clips from {selection_model}, probing with {model}")

    def load_rows(p: Path) -> dict:
        out = {}
        with p.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    out[r["clip"]] = r
                except json.JSONDecodeError:
                    continue
        return out

    a_rows = load_rows(a_path)
    b_rows = load_rows(b_path)
    common_clips = set(a_rows) & set(b_rows)
    disagreements = [
        c for c in common_clips
        if a_rows[c]["gemini_label"] is not None
        and b_rows[c]["gemini_label"] is not None
        and a_rows[c]["gemini_label"] != b_rows[c]["gemini_label"]
    ]
    print(f"Found {len(disagreements)} A-vs-B disagreement clips on model={model}")
    if not disagreements:
        sys.exit("No disagreements to probe. Exiting.")

    # Optional exclusion: skip clips already probed in a prior JSONL
    if args.probe_skip_clips:
        skip_path = Path(args.probe_skip_clips)
        if not skip_path.exists():
            sys.exit(f"ERROR: --probe-skip-clips file not found: {skip_path}")
        skip_set = set()
        with skip_path.open() as f:
            for line in f:
                try:
                    skip_set.add(json.loads(line)["clip"])
                except (json.JSONDecodeError, KeyError):
                    continue
        before = len(disagreements)
        disagreements = [c for c in disagreements if c not in skip_set]
        print(f"  Excluding {before - len(disagreements)} clips already in {skip_path.name}; "
              f"{len(disagreements)} candidates remain")

    import random
    random.seed(args.probe_seed)
    selected = random.sample(disagreements, min(args.probe_clips, len(disagreements)))
    suffix = args.probe_suffix or ""
    out_path = OUTPUTS / f"gemini_audit_probe_{model}{suffix}.jsonl"
    print(f"Probing {len(selected)} clips × {args.probe_reps} reps at temp={args.probe_temp}, "
          f"thinking_level={args.thinking_level}")
    print(f"Output → {out_path.name}")

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN — no API calls, stub responses only")
        print("=" * 60)
        client = None
    else:
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    interval = 60.0 / args.rpm if args.rpm > 0 else 0.0
    last_call = 0.0

    for clip in selected:
        a_row = a_rows[clip]
        b_row = b_rows[clip]
        for rep_idx in range(args.probe_reps):
            if not args.dry_run:
                wait = max(0, interval - (time.time() - last_call))
                if wait > 0:
                    time.sleep(wait)
            t0 = time.time()
            try:
                if args.dry_run:
                    pred = call_gemini_stub(clip, a_row["true_label"], "probe")
                else:
                    pred = call_gemini_real(
                        client, clip, fps=args.fps, prompt=PROBE_PROMPT,
                        schema=PROBE_SCHEMA, model=model,
                        temperature=args.probe_temp,
                        thinking_level=args.thinking_level,
                        system_instruction=args.system_instruction,
                    )
                    last_call = time.time()
                err = None
            except Exception as exc:
                pred = {"observation": None, "_model_version": None, "_finish_reason": None,
                        "_prompt_tokens": None, "_candidate_tokens": None, "_total_tokens": None}
                err = f"{type(exc).__name__}: {exc}"
            latency = time.time() - t0

            row = {
                "clip": clip,
                "video_rel": a_row["video_rel"],
                "source": a_row["source"],
                "true_label": a_row["true_label"],
                "model": model,
                "model_version": pred.get("_model_version"),
                "rep_idx": rep_idx,
                "temperature": args.probe_temp,
                "thinking_level": args.thinking_level,
                "observation": pred.get("observation"),
                "a_label": a_row["gemini_label"],
                "b_label": b_row["gemini_label"],
                "finish_reason": pred.get("_finish_reason"),
                "prompt_tokens": pred.get("_prompt_tokens"),
                "candidate_tokens": pred.get("_candidate_tokens"),
                "total_tokens": pred.get("_total_tokens"),
                "latency_s": round(latency, 2),
                "error": err,
            }
            with out_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  {a_row['source']:5s} {Path(clip).name[:45]:45s} rep{rep_idx} "
                  f"({latency:.1f}s)  obs: {(pred.get('observation') or 'ERR')[:80]}...")

    print(f"\nProbe complete → {out_path.relative_to(POC_DIR)}")


if __name__ == "__main__":
    main()
