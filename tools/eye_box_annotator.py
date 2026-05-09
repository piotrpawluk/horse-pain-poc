#!/usr/bin/env python3
"""Phase 5 eye-box annotation tool — minimal cv2 GUI.

Reads 102 PNG frames from outputs/eye_box_frames/ (extracted at deterministic
indices matching V-JEPA-2's frame sampling) and lets you draw a tight axis-
aligned bounding box around the visible eye on each. Saves to
outputs/eye_boxes_phase5a.json incrementally; resumes where you left off.

Locked rubric (per Phase 5 pre-reg, hash 2d253ae0...):

  Draw the SMALLEST axis-aligned rectangle containing the entire visible
  eye (sclera + lid + lash margin). Do NOT add extra margin yourself —
  the cropping pipeline adds margin programmatically (15 % primary; 10 /
  40 / 80 % sensitivity-2 curve).

  Partial occlusion: capture the visible portion only.
  Eye not visible at all in this frame: press 'x' (mark no_eye_visible).
  Memory of which clips were inverted in Phase 3: do NOT bias box
  tightness or margin asymmetry. Annotate purely on geometric visibility.

Keys:
  drag   draw / resize box (left mouse button)
  n      save current + go to NEXT frame
  p      go to PREVIOUS frame (saves current first)
  x      mark this frame as 'no_eye_visible'
  c      clear current box (start over)
  s      save progress to JSON now (incremental — 'n' also saves)
  q      save and quit

The window auto-resizes large frames (1920×1080 → ~1400 wide for screen
fit). Box coordinates are stored at full original resolution regardless
of display scale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

POC = Path(__file__).resolve().parent.parent
FRAMES_DIR = POC / "outputs/eye_box_frames"
DISPLAY_MAX_W = 1400

# Default = Phase 5a primary annotation; --phase5b switches to intra-rater paths
KEYMAP_5A = POC / "outputs/eye_box_keymap_phase5.json"
OUTPUT_5A = POC / "outputs/eye_boxes_phase5a.json"
KEYMAP_5B = POC / "outputs/eye_box_keymap_phase5b.json"
OUTPUT_5B = POC / "outputs/eye_boxes_phase5b.json"


class State:
    drawing = False
    ix = iy = -1
    box = None  # (x, y, w, h) in display coords, or "no_eye_visible", or None


state = State()


def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        state.drawing = True
        state.ix, state.iy = x, y
        state.box = (x, y, 0, 0)
    elif event == cv2.EVENT_MOUSEMOVE and state.drawing:
        x1 = min(state.ix, x)
        y1 = min(state.iy, y)
        x2 = max(state.ix, x)
        y2 = max(state.iy, y)
        state.box = (x1, y1, x2 - x1, y2 - y1)
    elif event == cv2.EVENT_LBUTTONUP:
        state.drawing = False
        x1 = min(state.ix, x)
        y1 = min(state.iy, y)
        x2 = max(state.ix, x)
        y2 = max(state.iy, y)
        if x2 - x1 < 5 or y2 - y1 < 5:
            state.box = None  # treat tiny boxes as accidental
        else:
            state.box = (x1, y1, x2 - x1, y2 - y1)


def task_done(annotations, uuid, label):
    return uuid in annotations and label in annotations[uuid]


def save_box(annotations, uuid, label, box, scale):
    if box is None:
        return
    annotations.setdefault(uuid, {})
    if box == "no_eye_visible":
        annotations[uuid][label] = "no_eye_visible"
    else:
        x, y, bw, bh = box
        annotations[uuid][label] = [
            int(round(x / scale)),
            int(round(y / scale)),
            int(round(bw / scale)),
            int(round(bh / scale)),
        ]


def write_json(annotations, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=2, sort_keys=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--phase5b", action="store_true",
        help="Re-annotation pass for intra-rater check (5 clips × 3 frames). "
             "Uses outputs/eye_box_keymap_phase5b.json + saves to "
             "outputs/eye_boxes_phase5b.json. Default is Phase 5a primary.",
    )
    args = ap.parse_args()

    if args.phase5b:
        keymap_path = KEYMAP_5B
        output_path = OUTPUT_5B
        phase_label = "5b"
    else:
        keymap_path = KEYMAP_5A
        output_path = OUTPUT_5A
        phase_label = "5a"
    print(f"[annotator] phase {phase_label}: keymap={keymap_path.name}",
          flush=True)

    if not keymap_path.exists():
        sys.exit(f"missing keymap: {keymap_path}")
    keymap = json.load(open(keymap_path))["keymap"]

    tasks = []
    for u in keymap:
        for label in ["first", "middle", "last"]:
            png = FRAMES_DIR / f"{u}_{label}.png"
            if png.exists():
                tasks.append((u, label, png))
    print(f"[annotator] loaded {len(tasks)} frame tasks across "
          f"{len(keymap)} clips", flush=True)

    annotations: dict = {}
    if output_path.exists():
        try:
            annotations = json.load(open(output_path))
            n_done = sum(1 for u, lab, _ in tasks if task_done(annotations, u, lab))
            print(f"[annotator] resuming with {n_done}/{len(tasks)} already done",
                  flush=True)
        except json.JSONDecodeError:
            print(f"[annotator] WARNING: existing {output_path.name} is "
                  f"unparseable; starting fresh", flush=True)
            annotations = {}

    # Start at first incomplete
    idx = 0
    while idx < len(tasks) and task_done(annotations, tasks[idx][0], tasks[idx][1]):
        idx += 1

    cv2.namedWindow("annotate", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("annotate", mouse_cb)

    while 0 <= idx < len(tasks):
        u, label, png_path = tasks[idx]
        img = cv2.imread(str(png_path))
        if img is None:
            print(f"  [skip] cannot load {png_path}", flush=True)
            idx += 1
            continue

        h, w = img.shape[:2]
        scale = min(1.0, DISPLAY_MAX_W / w)
        dw, dh = int(w * scale), int(h * scale)

        # Pre-load existing annotation if present
        existing = annotations.get(u, {}).get(label)
        if existing == "no_eye_visible":
            state.box = "no_eye_visible"
        elif isinstance(existing, list):
            ex_x, ex_y, ex_w, ex_h = existing
            state.box = (
                int(ex_x * scale), int(ex_y * scale),
                int(ex_w * scale), int(ex_h * scale),
            )
        else:
            state.box = None

        action = None
        while action is None:
            display = cv2.resize(img, (dw, dh)) if scale < 1.0 else img.copy()

            if state.box and state.box != "no_eye_visible":
                x, y, bw, bh = state.box
                cv2.rectangle(display, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                # Crosshair at center for sanity
                cx, cy = x + bw // 2, y + bh // 2
                cv2.line(display, (cx - 5, cy), (cx + 5, cy), (0, 255, 0), 1)
                cv2.line(display, (cx, cy - 5), (cx, cy + 5), (0, 255, 0), 1)

            n_done = sum(1 for tu, tlab, _ in tasks
                         if task_done(annotations, tu, tlab))
            status = (f"[{idx+1}/{len(tasks)}] done={n_done}  "
                      f"{u[:12]}...  {label}")
            if state.box == "no_eye_visible":
                status += "  [NO EYE VISIBLE]"
                color = (0, 0, 255)
            elif state.box:
                bw, bh = state.box[2], state.box[3]
                status += f"  box {bw}x{bh}"
                color = (0, 255, 0)
            else:
                status += "  [drag to draw]"
                color = (0, 200, 200)

            cv2.putText(display, status, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(
                display,
                "n=next  p=prev  x=no_eye  c=clear  s=save  q=quit",
                (10, dh - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1,
            )

            cv2.imshow("annotate", display)
            key = cv2.waitKey(15) & 0xFF

            if key == ord('q') or key == 27:  # 27 = ESC
                action = "quit"
            elif key == ord('n'):
                action = "next"
            elif key == ord('p'):
                action = "prev"
            elif key == ord('x'):
                state.box = "no_eye_visible"
            elif key == ord('c'):
                state.box = None
            elif key == ord('s'):
                save_box(annotations, u, label, state.box, scale)
                write_json(annotations, output_path)
                n_done = sum(1 for tu, tlab, _ in tasks
                             if task_done(annotations, tu, tlab))
                print(f"  [save] {n_done}/{len(tasks)} done", flush=True)

        # Action handling — save on next/quit, just navigate on prev
        if action in ("next", "quit"):
            save_box(annotations, u, label, state.box, scale)
            write_json(annotations, output_path)
        elif action == "prev":
            # Save what's been drawn so back-navigation doesn't lose it
            save_box(annotations, u, label, state.box, scale)
            write_json(annotations, output_path)

        if action == "next":
            idx += 1
        elif action == "prev":
            idx = max(0, idx - 1)
        elif action == "quit":
            break

    cv2.destroyAllWindows()

    n_done = sum(1 for tu, tlab, _ in tasks if task_done(annotations, tu, tlab))
    print(flush=True)
    print(f"=== {n_done}/{len(tasks)} annotations in {output_path.name} ===",
          flush=True)
    if n_done == len(tasks):
        print("All frames annotated.", flush=True)
    else:
        print(f"{len(tasks) - n_done} remaining — re-run to continue.",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
