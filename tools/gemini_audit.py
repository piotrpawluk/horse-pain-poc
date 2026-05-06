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
RESULTS_JSONL = OUTPUTS / "gemini_audit_results.jsonl"
SUMMARY_JSON = OUTPUTS / "gemini_audit_summary.json"

MODEL = "gemini-2.5-pro"

PROMPT = """\
Watch this short video clip of a horse. The clip shows the horse's
ear region. Determine whether the horse is actively MOVING its ears
during the clip (rotation, twitch, or pinning) or holding them STILL.

Respond with strict JSON only:
{"classification": "action" | "background",
 "confidence": <float 0.0-1.0>,
 "reasoning": "<one sentence, max 25 words>"}
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "classification": {"type": "string", "enum": ["action", "background"]},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["classification", "confidence", "reasoning"],
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

def already_processed() -> set[str]:
    """Set of clip paths already in the JSONL output."""
    if not RESULTS_JSONL.exists():
        return set()
    done = set()
    with RESULTS_JSONL.open() as f:
        for line in f:
            try:
                row = json.loads(line)
                if "clip" in row and "error" not in row:
                    done.add(row["clip"])
            except json.JSONDecodeError:
                continue
    return done


def append_result(row: dict) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with RESULTS_JSONL.open("a") as f:
        f.write(json.dumps(row) + "\n")


# ---- Gemini call ------------------------------------------------------------

def call_gemini_real(client, clip_path: str) -> dict:
    """Upload clip, poll until ACTIVE, generate, parse JSON. Returns the parsed dict."""
    from google.genai import types

    uploaded = client.files.upload(file=clip_path)
    while uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state.name != "ACTIVE":
        raise RuntimeError(f"upload state={uploaded.state.name}")

    response = client.models.generate_content(
        model=MODEL,
        contents=[uploaded, PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0.0,
        ),
    )
    return json.loads(response.text)


def call_gemini_stub(clip_path: str, true_label: str) -> dict:
    """Dry-run stub: deterministic fake response based on filename hash."""
    h = sum(ord(c) for c in clip_path) % 10
    cls = true_label if h < 7 else ("background" if true_label == "action" else "action")
    return {
        "classification": cls,
        "confidence": 0.5 + (h / 20),
        "reasoning": f"[DRY RUN STUB] hash-based pseudo-classification (h={h}).",
    }


# ---- Main loop --------------------------------------------------------------

def run(args) -> None:
    inv = load_inventory()
    if args.limit:
        inv = inv.head(args.limit)
    print(f"Inventory: {len(inv)} clips, {inv['source'].nunique()} sources")

    done = already_processed() if not args.dry_run else set()
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
                pred = call_gemini_stub(row["abs_path"], row["label"])
            else:
                pred = call_gemini_real(client, row["abs_path"])
                last_call = time.time()
            err = None
        except Exception as exc:
            pred = {"classification": None, "confidence": None, "reasoning": None}
            err = f"{type(exc).__name__}: {exc}"

        latency = time.time() - t0
        result = {
            "clip": row["abs_path"],
            "video_rel": row["video"],
            "split": row["split"],
            "source": row["source"],
            "true_label": row["label"],
            "gemini_label": pred["classification"],
            "confidence": pred["confidence"],
            "reasoning": pred["reasoning"],
            "agreement": pred["classification"] == row["label"] if pred["classification"] else None,
            "latency_s": round(latency, 2),
            "error": err,
        }
        append_result(result)
        n_processed += 1
        marker = "✓" if result["agreement"] else ("✗" if result["agreement"] is False else "·")
        print(f"  {marker}  {row['source']:5s} {Path(row['abs_path']).name:50s} "
              f"true={row['label']:10s} gemini={pred['classification'] or 'ERR':10s} "
              f"({latency:.1f}s)")

    print(f"\nProcessed {n_processed} clips. Writing summary...")
    write_summary()


def write_summary() -> None:
    if not RESULTS_JSONL.exists():
        return
    rows = []
    with RESULTS_JSONL.open() as f:
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
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary → {SUMMARY_JSON.relative_to(POC_DIR)}")
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

    run(args)


if __name__ == "__main__":
    main()
