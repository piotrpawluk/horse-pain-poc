#!/usr/bin/env python3
"""Phase 10 — One-shot script: split needs_resplit=1 clips into per-window files.

Reads `poc/data/prudnik/labels_pending.csv`, identifies clips with
`needs_resplit == "1"`, computes per-clip window timestamps, ffmpeg-splits
the source .mov into derived clip files (IMG_NNNN_w1.mov, IMG_NNNN_w2.mov,
…), and extends `labels_pending.csv` + `inventory.csv` with derived rows
for the labeling tool to walk through in a second pass.

Archive policy: original per-clip labels stay in `labels_pending.csv` with
a new `archived_full_clip_label` column populated for the 80 flagged rows.
Derived rows have blank labels for the user's per-window pass.

Splitting algorithm (per `splitting_recommendations.md` method):
- duration >= 15s : uniform ~10s windows in [5, 15]s (n = max(2, round(D/10)))
- duration in [5, 15)s : 2 equal halves (forces split since flagged)
- duration < 5s : skip with log note (cannot meaningfully split)

ffmpeg uses `-c copy` for fast remux (no re-encode). Caveat: cuts may
land on the nearest keyframe rather than the exact requested timestamp;
acceptable for ~10s windows where ±0.5s drift doesn't change labeling
verdict.

Usage:
    python tools/phase10_split_flagged.py --dry-run   # plan only, no writes
    python tools/phase10_split_flagged.py             # bulk execution
    python tools/phase10_split_flagged.py --limit 1   # test on first flagged clip
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

POC = Path(__file__).resolve().parent.parent
LABELS_CSV = POC / "data" / "prudnik" / "labels_pending.csv"
INVENTORY_CSV = POC / "data" / "prudnik" / "inventory.csv"
VIDEO_DIR = POC / "data" / "prudnik"
LOG_MD = POC / "data" / "prudnik" / "splitting_log.md"

RME_TARGET_S = 10.0
RME_MIN_S = 5.0
RME_MAX_S = 15.0
VJEPA2_MIN_FRAMES = 16


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write df to path atomically (tmp + os.replace)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def propose_splits(
    duration_s: float,
    target: float = RME_TARGET_S,
    min_w: float = RME_MIN_S,
    max_w: float = RME_MAX_S,
) -> list[tuple[float, float]] | None:
    """Per-clip uniform window splits, or None if too short to split."""
    if duration_s < min_w:
        return None  # can't split below RME minimum
    if duration_s <= max_w:
        # 5-15s flagged clips: force 2 equal halves
        return [(0.0, duration_s / 2.0), (duration_s / 2.0, duration_s)]
    # >15s: uniform splits targeting ~target seconds
    n = max(2, round(duration_s / target))
    window_dur = duration_s / n
    while window_dur > max_w:
        n += 1
        window_dur = duration_s / n
    while window_dur < min_w and n > 1:
        n -= 1
        window_dur = duration_s / n
    splits = []
    for i in range(n):
        start = i * window_dur
        end = (i + 1) * window_dur if i < n - 1 else duration_s
        splits.append((start, end))
    return splits


def check_ffmpeg() -> bool:
    """Verify ffmpeg is installed and runnable."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5
        )
        return r.returncode == 0
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def run_ffmpeg_split(
    src: Path, start_s: float, end_s: float, dst: Path, dry_run: bool = False
) -> tuple[bool, str]:
    """Run ffmpeg to extract [start_s, end_s] from src into dst.

    Returns (success, stderr_excerpt).
    """
    duration = end_s - start_s
    cmd = [
        "ffmpeg",
        "-y",  # overwrite without prompt
        "-ss", f"{start_s:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        "-loglevel", "error",
        str(dst),
    ]
    if dry_run:
        return True, " ".join(cmd)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return False, r.stderr.strip()[:300]
        if not dst.exists() or dst.stat().st_size == 0:
            return False, "ffmpeg produced empty/missing file"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timed out"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split flagged Prudnik clips into per-window files via ffmpeg.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned splits without writing files or modifying CSVs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N flagged clips (for testing).",
    )
    args = parser.parse_args()

    if not check_ffmpeg():
        print(
            "[ERROR] ffmpeg not found in PATH. Install via `brew install ffmpeg`.",
            file=sys.stderr,
        )
        return 1

    if not LABELS_CSV.exists():
        print(f"[ERROR] {LABELS_CSV} not found", file=sys.stderr)
        return 1
    if not INVENTORY_CSV.exists():
        print(f"[ERROR] {INVENTORY_CSV} not found", file=sys.stderr)
        return 1

    df = pd.read_csv(LABELS_CSV, keep_default_na=False, dtype=str)
    inv = pd.read_csv(INVENTORY_CSV, keep_default_na=False, dtype=str)

    # Schema migration: ensure archived_full_clip_label column exists
    if "archived_full_clip_label" not in df.columns:
        df["archived_full_clip_label"] = ""

    flagged = df[df["needs_resplit"] == "1"].copy()
    print(f"[phase10-split] {len(flagged)} clips flagged needs_resplit=1")

    if args.limit is not None:
        flagged = flagged.head(args.limit)
        print(f"[phase10-split] LIMITED to first {len(flagged)} (test mode)")

    # Compute plans
    plans = []
    for _, row in flagged.iterrows():
        duration = float(row["duration_s"])
        splits = propose_splits(duration)
        plans.append({
            "clip_id": row["clip_id"],
            "filename": row["filename"],
            "duration": duration,
            "original_label": row["label"],
            "splits": splits,
        })

    n_to_split = sum(1 for p in plans if p["splits"])
    n_skip = sum(1 for p in plans if not p["splits"])
    total_windows = sum(len(p["splits"]) for p in plans if p["splits"])
    print()
    print("[phase10-split] Plan:")
    print(f"  clips to split:    {n_to_split}")
    print(f"  clips to skip:     {n_skip} (sub-{RME_MIN_S}s, cannot split)")
    print(f"  total windows:     {total_windows}")
    print()

    # Show first 10 plans
    print("  First 10 plans (sorted as in CSV):")
    for p in plans[:10]:
        if p["splits"]:
            split_str = "; ".join(f"{a:.2f}-{b:.2f}" for a, b in p["splits"])
            print(
                f"    {p['clip_id']:<14} {p['duration']:>6.2f}s  "
                f"→ {len(p['splits'])} windows: {split_str}"
            )
        else:
            print(
                f"    {p['clip_id']:<14} {p['duration']:>6.2f}s  "
                f"→ SKIP (too short)"
            )
    if len(plans) > 10:
        print(f"    … +{len(plans) - 10} more")
    print()

    if args.dry_run:
        print("[phase10-split] --dry-run: no files written, no CSV modified.")
        return 0

    # Bulk execution
    derived_rows: list[dict] = []
    inv_rows_to_add: list[dict] = []
    log_entries: list[str] = []
    failures: list[str] = []

    for plan in plans:
        clip_id = plan["clip_id"]
        if not plan["splits"]:
            log_entries.append(
                f"- `{clip_id}` SKIP — duration {plan['duration']:.2f}s "
                f"< {RME_MIN_S}s minimum"
            )
            continue

        src_path = VIDEO_DIR / plan["filename"]
        if not src_path.exists():
            failures.append(f"{clip_id}: source file missing ({plan['filename']})")
            continue

        inv_match_rows = inv[inv["clip_id"] == clip_id]
        if inv_match_rows.empty:
            failures.append(f"{clip_id}: no inventory entry")
            continue
        inv_match = inv_match_rows.iloc[0]

        plan_log_lines = [f"- `{clip_id}` ({plan['duration']:.2f}s, original label `{plan['original_label']}`):"]
        for w_idx, (start, end) in enumerate(plan["splits"], start=1):
            derived_id = f"{clip_id}_w{w_idx}"
            derived_filename = f"{derived_id}.mov"
            dst_path = VIDEO_DIR / derived_filename

            print(
                f"  splitting {clip_id} w{w_idx}: {start:.2f}-{end:.2f}s "
                f"→ {derived_filename}",
                flush=True,
            )
            ok, err = run_ffmpeg_split(src_path, start, end, dst_path)
            if not ok:
                failures.append(f"{derived_id}: {err}")
                plan_log_lines.append(
                    f"  - w{w_idx}: {start:.2f}-{end:.2f}s → ⚠ FAILED: {err[:80]}"
                )
                continue

            window_dur = end - start
            actual_size_mb = dst_path.stat().st_size / 1e6
            plan_log_lines.append(
                f"  - w{w_idx}: {start:.2f}-{end:.2f}s ({window_dur:.2f}s) "
                f"→ `{derived_filename}` ({actual_size_mb:.1f} MB)"
            )

            derived_rows.append({
                "clip_id": derived_id,
                "filename": derived_filename,
                "duration_s": f"{window_dur:.3f}",
                "window": f"{start:.2f}-{end:.2f}",
                "frame_type": "",
                "label": "",
                "multi_horse": "",
                "confidence": "",
                "notes": "",
                "labeler": "",
                "label_date": "",
                "needs_resplit": "0",
                "archived_full_clip_label": "",
            })

            # fps from inventory used to check V-JEPA-2 viability
            try:
                fps_val = float(inv_match["fps"])
            except (ValueError, KeyError):
                fps_val = 0.0
            vjepa2_min = VJEPA2_MIN_FRAMES / fps_val if fps_val > 0 else float("inf")

            inv_rows_to_add.append({
                "clip_id": derived_id,
                "filename": derived_filename,
                "duration_s": f"{window_dur:.3f}",
                "resolution": inv_match["resolution"],
                "fps": inv_match["fps"],
                "codec": inv_match["codec"],
                "pix_fmt": inv_match["pix_fmt"],
                "file_size_mb": f"{actual_size_mb:.1f}",
                "short_lt_5s": "TRUE" if window_dur < RME_MIN_S else "FALSE",
                "long_gt_15s": "TRUE" if window_dur > RME_MAX_S else "FALSE",
                "below_vjepa2_min": "TRUE" if window_dur < vjepa2_min else "FALSE",
                "portrait": inv_match["portrait"],
            })

        log_entries.extend(plan_log_lines)
        # Archive the original per-clip label for this clip
        df.loc[df["clip_id"] == clip_id, "archived_full_clip_label"] = plan["original_label"]

    # Compose new DataFrames
    derived_df = pd.DataFrame(
        derived_rows,
        columns=[
            "clip_id", "filename", "duration_s", "window",
            "frame_type", "label", "multi_horse", "confidence",
            "notes", "labeler", "label_date", "needs_resplit",
            "archived_full_clip_label",
        ],
    )
    df_new = pd.concat([df, derived_df], ignore_index=True)

    inv_derived_df = pd.DataFrame(
        inv_rows_to_add,
        columns=[
            "clip_id", "filename", "duration_s", "resolution", "fps",
            "codec", "pix_fmt", "file_size_mb",
            "short_lt_5s", "long_gt_15s", "below_vjepa2_min", "portrait",
        ],
    )
    inv_new = pd.concat([inv, inv_derived_df], ignore_index=True)

    # Atomic writes
    atomic_write_csv(df_new, LABELS_CSV)
    atomic_write_csv(inv_new, INVENTORY_CSV)

    # Splitting log
    log_md = "\n".join([
        "# Phase 10 — Splitting log (flagged clips)",
        "",
        f"*Generated {datetime.now(timezone.utc).isoformat()} by `tools/phase10_split_flagged.py`. "
        f"See `Plans/refactored-orbiting-fog.md` and `docs/labeling-protocol-2026-05.md` for context.*",
        "",
        "## Summary",
        "",
        f"- Processed flagged clips: **{len(flagged)}**",
        f"- Clips split into windows: **{n_to_split}**",
        f"- Clips skipped (sub-{RME_MIN_S}s, can't split): **{n_skip}**",
        f"- Total derived windows generated: **{len(derived_rows)}**",
        f"- ffmpeg failures: **{len(failures)}**",
        "",
        "## Splitting parameters",
        "",
        f"- Target window: **{RME_TARGET_S}s** (midpoint of RME 5–15s range)",
        f"- Window bounds: **[{RME_MIN_S}s, {RME_MAX_S}s]**",
        "- Method: uniform splits via ffmpeg `-c copy` (no re-encode)",
        "- 15+s clips: `n = max(2, round(D/10))`, window_dur = D/n bounded to [5, 15]",
        "- 5–15s clips: forced 2 equal halves (since flagged needs_resplit=1)",
        f"- <{RME_MIN_S}s clips: skipped (cannot split below RME minimum)",
        "",
        "## Archive policy",
        "",
        f"Original per-clip labels for the {n_to_split} split clips moved into the new "
        "`archived_full_clip_label` column. Original rows in `labels_pending.csv` are "
        "kept with their `label` field intact (preserving the per-clip judgment as "
        "historical context). Derived window rows have blank `label` for the user's "
        "per-window labeling pass.",
        "",
        "## Per-clip splits",
        "",
        *log_entries,
    ])

    if failures:
        log_md += "\n\n## Failures\n\n" + "\n".join(f"- {f}" for f in failures)

    LOG_MD.write_text(log_md)

    print()
    print("=" * 70)
    print("[phase10-split] Done.")
    print(f"  labels_pending.csv: {len(df)} → {len(df_new)} rows "
          f"(+{len(derived_rows)} derived)")
    print(f"  inventory.csv:      {len(inv)} → {len(inv_new)} rows")
    print(f"  derived .mov files: {len(derived_rows)} new in {VIDEO_DIR.name}/")
    print(f"  splitting log:      {LOG_MD.relative_to(POC.parent)}")
    if failures:
        print(f"  ⚠ failures:        {len(failures)} (see log)")
    print("=" * 70)
    print()
    print("[phase10-split] Next: re-launch labeling tool to walk derived rows:")
    print("    python tools/prudnik_label_app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
