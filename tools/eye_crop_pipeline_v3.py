#!/usr/bin/env python3
"""Track B Phase 5 — v3 gold-standard manual eye-crop pipeline.

Per locked Phase 5 pre-reg (track_b_phase5_preregistration.md, hash
ced5cae6...):

  - Inputs: outputs/eye_boxes_phase5a.json (tight eye-region boxes on
    first/middle/last frame per clip), outputs/eye_box_keymap_phase5.json
    (UUID → real-filename mapping).
  - Per clip: compute pairwise IoU between (first, middle) and (middle,
    last) boxes. If BOTH > 0.7 → STATIC mode (use middle box for all
    native frames). Else → INTERPOLATED mode (linearly interpolate box
    x/y/w/h across frame indices).
  - Apply margin (CLI flag --margin-pct, default 15) by expanding tight
    box on all sides, then square-pad to maintain aspect ratio. Clip
    to image bounds.
  - Crop each native frame, resize to 224×224, write mp4 at native fps.

Outputs (per margin):
  outputs/eye_crops_v3_m<pct>/<clip>.mp4
  outputs/eye_crops_v3_m<pct>_manifest.jsonl  (per-clip metadata)

Run for each margin in {10, 15, 40, 80} for the locked Phase 5 design:
  primary + sensitivity 1 share m=15; sensitivity 2 uses all four.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

POC_DIR = Path(__file__).resolve().parent.parent
CLIPS_DIR = POC_DIR / "vendor" / "ReadMyEars_Dataset" / "data" / "videos"
KEYMAP_PATH = POC_DIR / "outputs" / "eye_box_keymap_phase5.json"
BOXES_PATH = POC_DIR / "outputs" / "eye_boxes_phase5a.json"
CROP_RES = 224
IOU_INTERP_THRESHOLD = 0.7  # below → switch to interpolated mode


def iou(b1, b2):
    if b1 == "no_eye_visible" or b2 == "no_eye_visible":
        return None
    x1a, y1a, wa, ha = b1
    x2a, y2a = x1a + wa, y1a + ha
    x1b, y1b, wb, hb = b2
    x2b, y2b = x1b + wb, y1b + hb
    ix1 = max(x1a, x1b); iy1 = max(y1a, y1b)
    ix2 = min(x2a, x2b); iy2 = min(y2a, y2b)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    union = wa * ha + wb * hb - inter
    return inter / union if union > 0 else 0.0


def lerp_box(b_start, b_end, t):
    """Linear interpolation of (x, y, w, h)."""
    return [
        b_start[0] * (1 - t) + b_end[0] * t,
        b_start[1] * (1 - t) + b_end[1] * t,
        b_start[2] * (1 - t) + b_end[2] * t,
        b_start[3] * (1 - t) + b_end[3] * t,
    ]


def box_for_frame(b_first, b_mid, b_last, frame_idx, n_frames, mode):
    """Return tight box for the given frame index."""
    if mode == "static":
        return b_mid
    # interpolated: 3-keyframe lerp
    mid_idx = n_frames // 2
    if frame_idx <= mid_idx:
        if mid_idx == 0:
            return b_first
        t = frame_idx / mid_idx
        return lerp_box(b_first, b_mid, t)
    else:
        denom = (n_frames - 1) - mid_idx
        if denom == 0:
            return b_last
        t = (frame_idx - mid_idx) / denom
        return lerp_box(b_mid, b_last, t)


def apply_margin_and_square(tight_box, margin_pct, frame_shape):
    """Tight box + margin on all sides, then square-pad around center,
    clipped to frame."""
    x, y, w, h = tight_box
    mw = w * (margin_pct / 100.0)
    mh = h * (margin_pct / 100.0)
    # Apply margin
    x_m = x - mw
    y_m = y - mh
    w_m = w + 2 * mw
    h_m = h + 2 * mh
    # Square-pad
    side = max(w_m, h_m)
    cx = x_m + w_m / 2
    cy = y_m + h_m / 2
    sq_x1 = cx - side / 2
    sq_y1 = cy - side / 2
    sq_x2 = cx + side / 2
    sq_y2 = cy + side / 2
    # Clip to frame bounds
    h_frame, w_frame = frame_shape[:2]
    sq_x1 = max(0, int(round(sq_x1)))
    sq_y1 = max(0, int(round(sq_y1)))
    sq_x2 = min(w_frame, int(round(sq_x2)))
    sq_y2 = min(h_frame, int(round(sq_y2)))
    return sq_x1, sq_y1, sq_x2, sq_y2


def crop_clip(clip_path, boxes_3frame, margin_pct, output_path):
    """Crop a video clip using the 3-keyframe boxes + margin."""
    b_first = boxes_3frame["first"]
    b_mid = boxes_3frame["middle"]
    b_last = boxes_3frame["last"]
    if any(b == "no_eye_visible" for b in (b_first, b_mid, b_last)):
        return {"status": "fail", "error": "no_eye_visible_in_some_frame"}

    iou_fm = iou(b_first, b_mid)
    iou_ml = iou(b_mid, b_last)
    if iou_fm is None or iou_ml is None:
        return {"status": "fail", "error": "iou_compute_failed"}
    mode = ("static" if (iou_fm > IOU_INTERP_THRESHOLD
                          and iou_ml > IOU_INTERP_THRESHOLD)
            else "interpolated")

    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames < 1:
        cap.release()
        return {"status": "fail", "error": "no_frames"}

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps,
                              (CROP_RES, CROP_RES))
    frames_written = 0
    sample_eye_box_first = None
    sample_eye_box_last = None
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        tight = box_for_frame(b_first, b_mid, b_last, i, n_frames, mode)
        x1, y1, x2, y2 = apply_margin_and_square(
            tight, margin_pct, frame.shape,
        )
        if x2 <= x1 or y2 <= y1:
            continue
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_resized = cv2.resize(crop, (CROP_RES, CROP_RES),
                                   interpolation=cv2.INTER_AREA)
        writer.write(crop_resized)
        frames_written += 1
        if i == 0:
            sample_eye_box_first = [x1, y1, x2, y2]
        if i == n_frames - 1:
            sample_eye_box_last = [x1, y1, x2, y2]
    cap.release()
    writer.release()

    if frames_written == 0:
        Path(output_path).unlink(missing_ok=True)
        return {"status": "fail", "error": "no_frames_written"}

    return {
        "status": "ok",
        "frames_extracted": frames_written,
        "frames_native": n_frames,
        "fps": fps,
        "mode": mode,
        "iou_first_middle": iou_fm,
        "iou_middle_last": iou_ml,
        "sample_first_crop_box": sample_eye_box_first,
        "sample_last_crop_box": sample_eye_box_last,
        "output_path": str(Path(output_path).relative_to(POC_DIR)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin-pct", type=int, required=True,
                    choices=[10, 15, 40, 80],
                    help="Margin to apply around tight eye box.")
    args = ap.parse_args()

    margin = args.margin_pct
    crops_dir = POC_DIR / "outputs" / f"eye_crops_v3_m{margin}"
    crops_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (POC_DIR / "outputs"
                     / f"eye_crops_v3_m{margin}_manifest.jsonl")
    manifest_path.unlink(missing_ok=True)

    keymap = json.load(open(KEYMAP_PATH))["keymap"]
    boxes_5a = json.load(open(BOXES_PATH))
    print(f"[v3 m={margin}] {len(keymap)} clips, margin={margin}%",
          flush=True)

    n_ok = 0
    n_fail = 0
    n_static = 0
    n_interp = 0
    for u, real_clip in keymap.items():
        if u not in boxes_5a:
            print(f"  [skip] no annotations for {u} ({real_clip})", flush=True)
            n_fail += 1
            continue
        clip_path = CLIPS_DIR / real_clip
        if not clip_path.exists():
            n_fail += 1
            continue
        out_path = crops_dir / real_clip
        result = crop_clip(str(clip_path), boxes_5a[u], margin, str(out_path))
        result["clip"] = real_clip
        result["uuid"] = u
        result["margin_pct"] = margin
        with open(manifest_path, "a") as f:
            f.write(json.dumps(result) + "\n")
        if result["status"] == "ok":
            n_ok += 1
            if result["mode"] == "static":
                n_static += 1
            else:
                n_interp += 1
            print(f"  [ok]  {real_clip:35} mode={result['mode']:11} "
                  f"frames={result['frames_extracted']:3d}/{result['frames_native']:3d} "
                  f"IoU(f,m)={result['iou_first_middle']:.2f} "
                  f"IoU(m,l)={result['iou_middle_last']:.2f}",
                  flush=True)
        else:
            n_fail += 1
            print(f"  [fail] {real_clip}: {result.get('error')}", flush=True)

    print(flush=True)
    print(f"=== v3 m={margin}: {n_ok}/{len(keymap)} ok, "
          f"{n_static} static, {n_interp} interpolated, {n_fail} fail ===",
          flush=True)
    print(f"Manifest: {manifest_path.relative_to(POC_DIR)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
