#!/usr/bin/env python3
"""Phase 6 (b) — derive anatomical eye-position parameter from data.

Locked methodology (per user direction 2026-05-09):

  > Derive from data: compute the median position of Phase 5 manual eye
  > boxes within the corresponding face bboxes across all 34 clips. Use
  > that as the canonical anatomical position. Hash-lock the parameter
  > file before the (b) run. This is principled.

Procedure:

1. For each of the 34 Phase 5 clips listed in
   `outputs/eye_box_keymap_phase5.json`:
   - Open the video, seek to the middle native frame
     (matches `eye_boxes_phase5a.json` middle keyframe).
   - Run YOLOv8 face detection (conf=0.5, imgsz=640) — same parameters
     as the vendor `infer_face_then_ear.py`.
   - Take the highest-confidence face detection.
   - Compute relative position of the manual middle-keyframe eye box
     (x, y, w, h pixels) within the face bbox:
       rel_x = (eye_x - face_x) / face_w
       rel_y = (eye_y - face_y) / face_h
       rel_w =  eye_w / face_w
       rel_h =  eye_h / face_h
2. Aggregate across clips that pass:
   - median(rel_x), median(rel_y), median(rel_w), median(rel_h)
   - IQR distribution
3. Output: outputs/phase6b_position_param.json with median + IQR + per-clip
   raw values + face-detection failure log.

Outputs are hash-locked into `docs/preregistration_hashes.md` before the
(b) crop pipeline is run.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

POC_DIR = Path(__file__).resolve().parent.parent
CLIPS_DIR = POC_DIR / "vendor" / "ReadMyEars_Dataset" / "data" / "videos"
KEYMAP_PATH = POC_DIR / "outputs" / "eye_box_keymap_phase5.json"
BOXES_PATH = POC_DIR / "outputs" / "eye_boxes_phase5a.json"
FACE_MODEL = (POC_DIR / "vendor" / "horse-face-ear-detection"
              / "horse_face_detection" / "yolov8l_horse_face_detection.pt")
OUT_PATH = POC_DIR / "outputs" / "phase6b_position_param.json"

CONF_THRESHOLD = 0.5
IMG_SIZE = 640


def load_middle_frame(clip_path: Path):
    cap = cv2.VideoCapture(str(clip_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames < 1:
        cap.release()
        return None, 0
    mid_idx = n_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None, n_frames
    return frame, n_frames


def detect_face(model: YOLO, frame):
    """Return (x, y, w, h, conf) for highest-confidence face, or None."""
    results = model(frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE,
                    verbose=False)
    if not results:
        return None
    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    confs = res.boxes.conf.cpu().numpy()
    xyxy = res.boxes.xyxy.cpu().numpy()
    best = int(confs.argmax())
    x1, y1, x2, y2 = xyxy[best]
    return (
        float(x1), float(y1),
        float(x2 - x1), float(y2 - y1),
        float(confs[best]),
    )


def main() -> int:
    keymap = json.loads(KEYMAP_PATH.read_text())["keymap"]
    boxes = json.loads(BOXES_PATH.read_text())

    if not FACE_MODEL.exists():
        print(f"[ERROR] face model not found: {FACE_MODEL}",
              file=sys.stderr)
        return 1
    print(f"Loading {FACE_MODEL.name}…", flush=True)
    model = YOLO(str(FACE_MODEL))

    per_clip = []
    fails = []

    for u, real_clip in sorted(keymap.items()):
        if u not in boxes:
            fails.append({"uuid": u, "clip": real_clip,
                          "stage": "no_annotation"})
            continue
        clip_box = boxes[u]
        eye_mid = clip_box.get("middle")
        if eye_mid == "no_eye_visible" or eye_mid is None:
            fails.append({"uuid": u, "clip": real_clip,
                          "stage": "no_eye_visible_in_middle"})
            continue
        clip_path = CLIPS_DIR / real_clip
        if not clip_path.exists():
            fails.append({"uuid": u, "clip": real_clip,
                          "stage": "clip_missing"})
            continue
        frame, n_frames = load_middle_frame(clip_path)
        if frame is None:
            fails.append({"uuid": u, "clip": real_clip,
                          "stage": "frame_read_failed"})
            continue

        face = detect_face(model, frame)
        if face is None:
            fails.append({"uuid": u, "clip": real_clip,
                          "stage": "no_face_detected"})
            continue
        face_x, face_y, face_w, face_h, face_conf = face
        if face_w <= 0 or face_h <= 0:
            fails.append({"uuid": u, "clip": real_clip,
                          "stage": "invalid_face_bbox"})
            continue

        eye_x, eye_y, eye_w, eye_h = eye_mid
        rel_x = (eye_x - face_x) / face_w
        rel_y = (eye_y - face_y) / face_h
        rel_w = eye_w / face_w
        rel_h = eye_h / face_h

        per_clip.append({
            "uuid": u,
            "clip": real_clip,
            "frame_index": n_frames // 2,
            "n_frames": n_frames,
            "face_bbox": [face_x, face_y, face_w, face_h],
            "face_conf": face_conf,
            "eye_bbox_middle_keyframe": [eye_x, eye_y, eye_w, eye_h],
            "rel_x": rel_x,
            "rel_y": rel_y,
            "rel_w": rel_w,
            "rel_h": rel_h,
        })
        print(f"  [ok]  {real_clip:35} face_conf={face_conf:.2f} "
              f"rel=(x={rel_x:.3f}, y={rel_y:.3f}, "
              f"w={rel_w:.3f}, h={rel_h:.3f})", flush=True)

    n_ok = len(per_clip)
    n_fail = len(fails)
    n_total = len(keymap)

    if n_ok < 2:
        print(f"[ERROR] insufficient successful detections: {n_ok}",
              file=sys.stderr)
        return 1

    def stats(key):
        vals = sorted(r[key] for r in per_clip)
        return {
            "median": statistics.median(vals),
            "mean": statistics.fmean(vals),
            "stdev": statistics.pstdev(vals),
            "p25": vals[max(0, len(vals) // 4 - 1)],
            "p75": vals[min(len(vals) - 1, 3 * len(vals) // 4)],
            "min": vals[0],
            "max": vals[-1],
        }

    summary = {
        "tool": "tools/phase6b_derive_position.py",
        "face_model": str(FACE_MODEL.relative_to(POC_DIR)),
        "face_conf_threshold": CONF_THRESHOLD,
        "face_imgsz": IMG_SIZE,
        "frame_choice": "middle native frame, matches eye_boxes_phase5a.json"
                        " middle keyframe",
        "eye_boxes_source": str(BOXES_PATH.relative_to(POC_DIR)),
        "keymap_source": str(KEYMAP_PATH.relative_to(POC_DIR)),
        "n_clips_total": n_total,
        "n_clips_ok": n_ok,
        "n_clips_failed": n_fail,
        "failure_log": fails,
        "stats": {
            "rel_x": stats("rel_x"),
            "rel_y": stats("rel_y"),
            "rel_w": stats("rel_w"),
            "rel_h": stats("rel_h"),
        },
        "locked_anatomical_position": {
            "rel_x": stats("rel_x")["median"],
            "rel_y": stats("rel_y")["median"],
            "rel_w": stats("rel_w")["median"],
            "rel_h": stats("rel_h")["median"],
            "interpretation": "tight eye-region bbox at "
                              "(face_x + rel_x*face_w, face_y + rel_y*face_h, "
                              "rel_w*face_w, rel_h*face_h); apply margin "
                              "and square-pad in (b) crop pipeline as in "
                              "v3.",
        },
        "per_clip": per_clip,
    }

    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print()
    print(f"=== n_ok={n_ok}/{n_total}, n_fail={n_fail} ===")
    print(f"Median rel_x: {summary['stats']['rel_x']['median']:.4f} "
          f"(IQR [{summary['stats']['rel_x']['p25']:.4f}, "
          f"{summary['stats']['rel_x']['p75']:.4f}])")
    print(f"Median rel_y: {summary['stats']['rel_y']['median']:.4f} "
          f"(IQR [{summary['stats']['rel_y']['p25']:.4f}, "
          f"{summary['stats']['rel_y']['p75']:.4f}])")
    print(f"Median rel_w: {summary['stats']['rel_w']['median']:.4f} "
          f"(IQR [{summary['stats']['rel_w']['p25']:.4f}, "
          f"{summary['stats']['rel_w']['p75']:.4f}])")
    print(f"Median rel_h: {summary['stats']['rel_h']['median']:.4f} "
          f"(IQR [{summary['stats']['rel_h']['p25']:.4f}, "
          f"{summary['stats']['rel_h']['p75']:.4f}])")
    print()
    print(f"Wrote: {OUT_PATH.relative_to(POC_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
