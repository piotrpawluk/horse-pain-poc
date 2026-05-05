#!/usr/bin/env python3
"""Subtitle search — parsuje .vtt z vendor/rhpe_materials/ i generuje
propozycje klipów per RHpE behavior.

Output: vendor/rhpe_anchored/suggestions.csv z kolumnami:
  behavior, source_video, timestamp_start, timestamp_end, keyword_match,
  context, status (=pending), clip_path (=puste).

User następnie weryfikuje przez Gradio UI w notebooks/04 i approve'uje
fragmenty do `vendor/rhpe_anchored/<behavior>/`.

Uruchom: python tools/subtitle_search.py [--max-per-behavior N]
"""

import argparse
import csv
import re
from datetime import timedelta
from pathlib import Path

import webvtt

# --- Konfiguracja: 5 behaviors × keywords ----------------------------------
BEHAVIOR_KEYWORDS = {
    "ear_position": [
        r"\bear[s]?\b",
        r"\bear[s]?\s+back\b",
        r"\brotated\s+ear",
        r"\bear[s]?\s+pinn",
        r"\bear[s]?\s+forward\b",
        r"\bear\s+position\b",
    ],
    "head_position": [
        r"\bhead\s+position\b",
        r"\bhead\s+carriage\b",
        r"\bhead\s+tilt\b",
        r"\bhead\s+nod\b",
        r"\bhead\s+toss\b",
        r"\bneck\s+position\b",
        r"\bbehind\s+the\s+vertical\b",
        r"\babove\s+the\s+bit\b",
    ],
    "mouth_open": [
        r"\bmouth\s+open\b",
        r"\bopen\s+mouth\b",
        r"\btongue\s+out\b",
        r"\btongue\s+protruding\b",
        r"\blips\s+apart\b",
        r"\bgaping\b",
        r"\bjaw\s+tens",
    ],
    "tail_movement": [
        r"\btail\s+swish\b",
        r"\btail\s+swishing\b",
        r"\btail\s+clamp",
        r"\btail\s+held\b",
        r"\btail\s+wring",
        r"\btail\s+lash",
    ],
    "eye_expression": [
        r"\bintense\s+stare\b",
        r"\bsclera\b",
        r"\bwhite\s+of\s+the\s+eye\b",
        r"\bclosed?\s+eye[s]?\b",
        r"\beye\s+expression\b",
        r"\beye[s]?\s+half[- ]closed\b",
        r"\bworried\s+eye\b",
    ],
}

VTT_GLOB = "vendor/rhpe_materials/videos/**/*.en.vtt"
OUTPUT_CSV = Path("vendor/rhpe_anchored/suggestions.csv")
CONTEXT_PADDING_SECONDS = 5  # ile sekund przed/po keyword match na clip


def vtt_time_to_seconds(t: str) -> float:
    """'00:01:23.456' lub '00:01:23.456 --> ...' → 83.456 (sekundy)."""
    parts = t.split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = "0"
        m, s = parts
    else:
        return 0.0
    return int(h) * 3600 + int(m) * 60 + float(s)


def seconds_to_hms(s: float) -> str:
    s = max(0, s)
    return str(timedelta(seconds=int(s)))


def find_matches_in_vtt(vtt_path: Path) -> list[dict]:
    """Zwraca listę match'ów per behavior dla pojedynczego pliku VTT."""
    rows = []
    try:
        captions = list(webvtt.read(str(vtt_path)))
    except Exception as e:
        print(f"  ⚠ {vtt_path.name}: {e}")
        return rows

    # Source video path (zamiana .en.vtt → .mp4 obok)
    source_video = vtt_path.with_suffix("").with_suffix(".mp4")
    if not source_video.exists():
        # YouTube auto-cap są .en.vtt, video jest .mp4 — czasem stem is identical sans suffix
        candidate = vtt_path.parent / (vtt_path.name.replace(".en.vtt", ".mp4"))
        source_video = candidate if candidate.exists() else vtt_path

    seen_timestamps_per_behavior: dict[str, set] = {b: set() for b in BEHAVIOR_KEYWORDS}

    for caption in captions:
        text = caption.text.lower()
        start_s = vtt_time_to_seconds(caption.start)
        for behavior, patterns in BEHAVIOR_KEYWORDS.items():
            for pat in patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if not m:
                    continue
                # de-dupe: skip jeśli mamy już match dla tego behavior w okolicy ±5s
                bucket = round(start_s / 5)
                if bucket in seen_timestamps_per_behavior[behavior]:
                    continue
                seen_timestamps_per_behavior[behavior].add(bucket)

                clip_start = max(0, start_s - CONTEXT_PADDING_SECONDS)
                clip_end = start_s + CONTEXT_PADDING_SECONDS + 5  # ~10-15s clip
                rows.append({
                    "behavior": behavior,
                    "source_video": str(source_video),
                    "timestamp_start": seconds_to_hms(clip_start),
                    "timestamp_end": seconds_to_hms(clip_end),
                    "keyword_match": m.group(0),
                    "context": caption.text.replace("\n", " ").strip()[:200],
                    "status": "pending",
                    "clip_path": "",
                })
                break  # jeden match per caption per behavior
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-per-behavior",
        type=int,
        default=200,
        help="Maksymalnie ile propozycji per behavior (domyślnie 200, dla user efficiency).",
    )
    args = parser.parse_args()

    # Zapewnij output dir
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Znajdź wszystkie .vtt
    vtt_files = sorted(Path(".").glob(VTT_GLOB))
    print(f"Znaleziono {len(vtt_files)} plików .en.vtt w {VTT_GLOB}")

    all_rows = []
    for vtt in vtt_files:
        rows = find_matches_in_vtt(vtt)
        all_rows.extend(rows)
        if rows:
            print(f"  ✓ {vtt.name[:80]:80s} → {len(rows)} match")

    # Sort: per behavior + per source
    all_rows.sort(key=lambda r: (r["behavior"], r["source_video"], r["timestamp_start"]))

    # Limit per behavior
    limited = []
    counts: dict[str, int] = {}
    for r in all_rows:
        b = r["behavior"]
        if counts.get(b, 0) < args.max_per_behavior:
            limited.append(r)
            counts[b] = counts.get(b, 0) + 1

    # Write CSV
    fieldnames = [
        "behavior", "source_video", "timestamp_start", "timestamp_end",
        "keyword_match", "context", "status", "clip_path",
    ]
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(limited)

    print(f"\n✓ Zapisano {len(limited)} propozycji do {OUTPUT_CSV}")
    print("\nPer behavior:")
    for b in BEHAVIOR_KEYWORDS:
        n = sum(1 for r in limited if r["behavior"] == b)
        print(f"  {b:20s} {n}")
    print("\nNastępny krok: jupyter lab notebooks/04_few_shot_rhpe_validation.ipynb")


if __name__ == "__main__":
    main()
