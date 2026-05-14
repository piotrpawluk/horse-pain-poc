#!/usr/bin/env python3
"""Phase 10 — Gradio labeling tool for Prudnik unridden championship clips.

Single-user labeling app: walks through 122 clips one-by-one in browser,
captures binary ACTION/BACKGROUND ear-motion verdicts (RME-compatible per
`docs/labeling-protocol-2026-05.md` §2-§3), writes each label atomically to
`poc/data/prudnik/labels_pending.csv` with resume-on-restart support.

**STRICT MANUAL**: no classifier predictions / motion-magnitude / V-JEPA-2
output is shown anywhere in the UI — Phase 10a is a transfer test that
requires labels independent of the classifier under test (Lesson 21).

**Single user, single tab** assumption. Do NOT open two browser tabs on
the same CSV — last write wins, no file locking.

Usage:
    python tools/prudnik_label_app.py

Launches at http://localhost:7860 (auto-opens browser).

Plan: poc/Plans/refactored-orbiting-fog.md
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import pandas as pd

POC = Path(__file__).resolve().parent.parent
LABELS_CSV = POC / "data" / "prudnik" / "labels_pending.csv"
INVENTORY_CSV = POC / "data" / "prudnik" / "inventory.csv"
VIDEO_DIR = POC / "data" / "prudnik"

LABEL_CHOICES = ["action", "background", "action?", "background?"]
FRAME_CHOICES = ["head-zoom", "full-body"]
CONFIDENCE_CHOICES = ["high", "medium", "low"]

PROTOCOL_CARD_MD = """### Labeling rule — `docs/labeling-protocol-2026-05.md` §2-§3

**ACTION** = at least one ear shows a positional change during the clip
(rotation / twitch / flick / pinning) visible to a careful viewer at
normal-speed playback. Motion can be brief (<200 ms) or sustained.

**BACKGROUND** = ears remain essentially stationary throughout, OR head /
body / tail moves while ears are stationary.

**`?` suffix** — borderline; "I would defer to a second opinion".
Used for threshold-of-visibility motion, partial occlusion, or edit-cut
artifacts. ~10% of clips typically.

**Multi-horse**: foreground horse only governs verdict (per §3). Note
background-horse motion in `notes` with `fh:` / `bh:` notation if relevant,
but the verdict tracks FH only.

**Anatomical scope**: ears only. Head / body / tail movement without ear
motion = BACKGROUND.
"""

# Module-level state — single-user, single-tab assumption per design.
# DF is the source of truth; CSV is its atomic-write target.
DF: pd.DataFrame = pd.DataFrame()
INV: pd.DataFrame = pd.DataFrame()


# -----------------------------------------------------------------------
# State management
# -----------------------------------------------------------------------


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write df to path atomically via .tmp + os.replace.

    POSIX-atomic on same-directory rename. Ctrl-C mid-`to_csv` only corrupts
    the .tmp file; the real CSV is untouched until the rename completes.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_state() -> int:
    """Read labels_pending.csv + inventory.csv into module globals.

    Side effects (one-shot, idempotent):
      - Snapshot pristine CSV → labels_pending.csv.bak (first launch only)
      - Add `needs_resplit` column = "0" if missing (schema migration)

    Returns the initial cursor (first row with empty `label`).
    """
    global DF, INV
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"labels_pending.csv not found at {LABELS_CSV}")
    if not INVENTORY_CSV.exists():
        raise FileNotFoundError(f"inventory.csv not found at {INVENTORY_CSV}")

    # keep_default_na=False, dtype=str is load-bearing: pandas otherwise
    # converts empty cells to NaN (which becomes literal "nan" string on
    # round-trip) and coerces "0"/"1" flags to int (breaks empty-cell check).
    DF = pd.read_csv(LABELS_CSV, keep_default_na=False, dtype=str)

    if "needs_resplit" not in DF.columns:
        DF["needs_resplit"] = "0"
        atomic_write_csv(DF, LABELS_CSV)

    bak = LABELS_CSV.with_suffix(".csv.bak")
    if not bak.exists():
        shutil.copy2(LABELS_CSV, bak)

    INV = pd.read_csv(INVENTORY_CSV, keep_default_na=False, dtype=str)

    return next_unlabeled(0)


def next_unlabeled(start: int) -> int:
    """First index >= start where DF.loc[i, 'label'] == ''."""
    start = max(0, start)
    for i in range(start, len(DF)):
        if DF.loc[i, "label"] == "":
            return i
    return len(DF)


# -----------------------------------------------------------------------
# Render / save
# -----------------------------------------------------------------------


def progress_md(cursor: int) -> str:
    n_labeled = int((DF["label"] != "").sum())
    n_resplit = int((DF["needs_resplit"] == "1").sum())
    pos = min(cursor, len(DF) - 1) if len(DF) else 0
    return (
        f"**Clip {pos + 1} / {len(DF)}** &nbsp;·&nbsp; "
        f"labeled: {n_labeled} &nbsp;·&nbsp; needs_resplit: {n_resplit}"
    )


def metadata_md(cursor: int) -> str:
    """Recording metadata only — no classifier-derived fields."""
    if cursor >= len(DF):
        return "_All clips labeled._"
    row = DF.loc[cursor]
    clip_id = row["clip_id"]
    matches = INV[INV["clip_id"] == clip_id]
    if matches.empty:
        return f"**{clip_id}** · _(no inventory metadata available)_"
    m = matches.iloc[0]
    return (
        f"**{clip_id}** &nbsp;·&nbsp; {m['resolution']} &nbsp;·&nbsp; "
        f"{m['fps']} fps &nbsp;·&nbsp; {m['duration_s']}s &nbsp;·&nbsp; "
        f"{m['file_size_mb']} MB"
    )


def render_clip(cursor: int):
    """Return form values for the clip at row=cursor.

    Tuple order matches `all_outputs` in build_ui():
      (video_path, metadata_md, frame_type, label, confidence,
       multi_horse, notes, needs_resplit, status_md, progress_md)
    """
    if cursor >= len(DF):
        return (
            None,
            "**All clips labeled.** Close the app; CSV is committed.",
            None, None, "high", False, "", False,
            "All clips labeled — ready for Phase 10a Stage 1 pre-reg.",
            progress_md(cursor),
        )

    row = DF.loc[cursor]
    video_path = VIDEO_DIR / row["filename"]
    if not video_path.exists():
        return (
            None,
            f"**{row['clip_id']}** &nbsp;·&nbsp; ⚠ Video file missing: `{row['filename']}`",
            row["frame_type"] or None,
            row["label"] or None,
            row["confidence"] or "high",
            row["multi_horse"] == "1",
            row["notes"] or "",
            row["needs_resplit"] == "1",
            "⚠ Video file missing on disk — note absence in `notes`, label as best-effort or Skip.",
            progress_md(cursor),
        )

    return (
        str(video_path),
        metadata_md(cursor),
        row["frame_type"] or None,
        row["label"] or None,
        row["confidence"] or "high",
        row["multi_horse"] == "1",
        row["notes"] or "",
        row["needs_resplit"] == "1",
        "",
        progress_md(cursor),
    )


def save_label(
    cursor: int,
    frame_type: str,
    label: str,
    multi_horse: bool,
    confidence: str,
    notes: str,
    needs_resplit: bool,
    labeler: str,
) -> str | None:
    """Validate required fields, mutate DF, atomic-write CSV.

    Returns None on success, error string on validation failure.
    """
    if cursor >= len(DF):
        return "No current clip to save (cursor past end of data)."
    if not frame_type:
        return "`frame_type` is required (head-zoom or full-body)."
    if not label:
        return "`label` is required (action / background / action? / background?)."

    DF.loc[cursor, "frame_type"] = frame_type
    DF.loc[cursor, "label"] = label
    DF.loc[cursor, "multi_horse"] = "1" if multi_horse else "0"
    DF.loc[cursor, "confidence"] = confidence or ""
    DF.loc[cursor, "notes"] = notes or ""
    DF.loc[cursor, "needs_resplit"] = "1" if needs_resplit else "0"
    DF.loc[cursor, "labeler"] = labeler or ""
    DF.loc[cursor, "label_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    atomic_write_csv(DF, LABELS_CSV)
    return None


# -----------------------------------------------------------------------
# Gradio UI
# -----------------------------------------------------------------------


def build_ui(initial_cursor: int) -> gr.Blocks:
    """Construct the Gradio Blocks app.

    State: per-session cursor (gr.State(int)). DF + INV are module-level
    globals (single-user assumption).
    """
    with gr.Blocks(title="Prudnik labeling tool") as app:
        gr.Markdown(
            f"# Phase 10 — Prudnik labeling tool\n"
            f"*Reading* `{LABELS_CSV.name}` *·* {len(DF)} clips *·* "
            f"strict-manual labeling for Phase 10a transfer test "
            f"(see [Lesson 21](../../docs/lessons_learned.md) for framing)."
        )

        state_cursor = gr.State(initial_cursor)

        # Top bar: labeler + progress
        with gr.Row():
            labeler_in = gr.Textbox(
                label="Labeler initials (e.g., PP) — sticky across session",
                value="",
                scale=2,
            )
            progress_out = gr.Markdown(progress_md(initial_cursor))

        # Main layout: video left, form right
        with gr.Row():
            with gr.Column(scale=3):
                video_out = gr.Video(
                    autoplay=True,
                    loop=True,
                    height=540,
                    interactive=False,
                    show_label=False,
                )
                metadata_out = gr.Markdown(metadata_md(initial_cursor))
                status_out = gr.Markdown("")

            with gr.Column(scale=2):
                gr.Markdown(PROTOCOL_CARD_MD)

                label_in = gr.Radio(
                    choices=LABEL_CHOICES,
                    label="Verdict (required)",
                    value=None,
                )
                frame_type_in = gr.Radio(
                    choices=FRAME_CHOICES,
                    label="Frame type (required)",
                    value=None,
                )
                confidence_in = gr.Radio(
                    choices=CONFIDENCE_CHOICES,
                    label="Confidence",
                    value="high",
                )
                with gr.Row():
                    multi_horse_in = gr.Checkbox(
                        label="Multi-horse in frame",
                        value=False,
                    )
                    needs_resplit_in = gr.Checkbox(
                        label="Flag for re-split (within-clip variance)",
                        value=False,
                    )
                notes_in = gr.Textbox(
                    label="Notes (optional — `fh:` / `bh:` notation per §7)",
                    lines=2,
                    max_lines=4,
                    value="",
                )

        # Navigation row
        with gr.Row():
            prev_btn = gr.Button("← Prev", scale=1)
            submit_btn = gr.Button("Submit & Next", variant="primary", scale=2)
            skip_btn = gr.Button("Skip → Next Unlabeled", scale=1)

        with gr.Row():
            jump_idx = gr.Number(
                label="Jump to clip # (1-based)",
                value=1,
                precision=0,
                scale=1,
            )
            jump_btn = gr.Button("Go", scale=1)
            reload_btn = gr.Button("Reload CSV from disk", scale=1)

        # All form outputs in display order (matches render_clip tuple)
        all_outputs = [
            video_out,
            metadata_out,
            frame_type_in,
            label_in,
            confidence_in,
            multi_horse_in,
            notes_in,
            needs_resplit_in,
            status_out,
            progress_out,
        ]

        # Initial render after layout exists
        def initial_render():
            return render_clip(initial_cursor)

        app.load(fn=initial_render, inputs=[], outputs=all_outputs)

        # -- Handlers ---------------------------------------------------

        def on_submit(
            cursor, labeler, frame_type, label,
            multi_horse, confidence, notes, needs_resplit,
        ):
            err = save_label(
                cursor, frame_type, label, multi_horse, confidence,
                notes, needs_resplit, labeler,
            )
            if err:
                gr.Warning(err)
                return cursor, *render_clip(cursor)
            new_cursor = min(cursor + 1, len(DF))
            return new_cursor, *render_clip(new_cursor)

        submit_btn.click(
            fn=on_submit,
            inputs=[
                state_cursor, labeler_in,
                frame_type_in, label_in,
                multi_horse_in, confidence_in,
                notes_in, needs_resplit_in,
            ],
            outputs=[state_cursor, *all_outputs],
        )

        def on_prev(cursor):
            new_cursor = max(0, cursor - 1)
            return new_cursor, *render_clip(new_cursor)

        prev_btn.click(
            fn=on_prev,
            inputs=[state_cursor],
            outputs=[state_cursor, *all_outputs],
        )

        def on_skip(cursor):
            new_cursor = next_unlabeled(cursor + 1)
            return new_cursor, *render_clip(new_cursor)

        skip_btn.click(
            fn=on_skip,
            inputs=[state_cursor],
            outputs=[state_cursor, *all_outputs],
        )

        def on_jump(idx):
            try:
                target = int(idx) - 1  # 1-based input → 0-based index
            except (TypeError, ValueError):
                gr.Warning("Invalid index — enter a clip number.")
                return 0, *render_clip(0)
            new_cursor = max(0, min(target, len(DF) - 1))
            return new_cursor, *render_clip(new_cursor)

        jump_btn.click(
            fn=on_jump,
            inputs=[jump_idx],
            outputs=[state_cursor, *all_outputs],
        )

        def on_reload():
            new_cursor = load_state()
            gr.Info(f"Reloaded from disk. Cursor at clip {new_cursor + 1}.")
            return new_cursor, *render_clip(new_cursor)

        reload_btn.click(
            fn=on_reload,
            inputs=[],
            outputs=[state_cursor, *all_outputs],
        )

    return app


def main() -> int:
    initial_cursor = load_state()
    print()
    print(f"[prudnik-label] Loaded {len(DF)} clips. Cursor at row {initial_cursor + 1}.")
    print(f"  Labels CSV:  {LABELS_CSV}")
    print(f"  Inventory:   {INVENTORY_CSV}")
    print(f"  Video dir:   {VIDEO_DIR}")
    print(f"  Backup:      {LABELS_CSV.with_suffix('.csv.bak')}")
    print()
    print("[prudnik-label] Launching at http://127.0.0.1:7860")
    print("[prudnik-label] DO NOT open two browser tabs on the same CSV.")
    print("[prudnik-label] STRICT-MANUAL labeling — no classifier hints shown.")
    print()

    app = build_ui(initial_cursor)
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        allowed_paths=[str(VIDEO_DIR)],
        quiet=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
