#!/usr/bin/env python3
"""Track B Phase 6 (b) — face-bbox-positioned eye-crop pipeline.

Per locked Phase 6 (b) pre-registration
(`outputs/track_b_phase6b_preregistration.md`, hash
`c8c00fb81e98...`):

  - Inputs: video clips listed in `outputs/eye_box_keymap_phase5.json`
    (same 34 clips as Phase 5).
  - Per native frame: run YOLOv8l horse-face detector
    (`vendor/horse-face-ear-detection/horse_face_detection/
    yolov8l_horse_face_detection.pt`, conf=0.5, imgsz=640).
  - From face bbox, derive tight eye box at locked anatomical position
    (rel_x=0.5060, rel_y=0.3926, rel_w=0.1419, rel_h=0.0596) — values
    from `outputs/phase6b_position_param.json`.
  - Apply margin m=15% (matches Phase 5 primary), square-pad, clip to
    frame, resize to 224×224, write mp4 at native fps.

Locked failure handling rule:

  1. Per-frame attempt at conf=0.5.
  2. Single-frame failure → interpolate face bbox linearly between
     nearest temporally-preceding and following detected frames; edge
     cases use the nearest available detected frame.
  3. Clip-level threshold: > 25% frames fail → fall back to
     single-middle-frame face bbox applied across entire clip
     (manifest logs `fallback_mode: "single_middle_frame"`).
  4. Total clip failure (no face detected on any frame at conf ≥ 0.5)
     → drop clip from LOSO; manifest records status="fail".

Outputs:
  outputs/eye_crops_phase6b_m15/<clip>.mp4
  outputs/eye_crops_phase6b_m15_manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

POC_DIR = Path(__file__).resolve().parent.parent
CLIPS_DIR = POC_DIR / "vendor" / "ReadMyEars_Dataset" / "data" / "videos"
KEYMAP_PATH = POC_DIR / "outputs" / "eye_box_keymap_phase5.json"
POSITION_PARAM_PATH = POC_DIR / "outputs" / "phase6b_position_param.json"
FACE_MODEL = (POC_DIR / "vendor" / "horse-face-ear-detection"
              / "horse_face_detection" / "yolov8l_horse_face_detection.pt")
CROP_RES = 224

CONF_THRESHOLD = 0.5
IMG_SIZE = 640
FALLBACK_FAILURE_FRACTION = 0.25  # > 25% triggers single-middle fallback


def detect_face(model: YOLO, frame):
    """Highest-confidence face bbox on this frame, or None."""
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
    if x2 <= x1 or y2 <= y1:
        return None
    return (
        float(x1), float(y1),
        float(x2 - x1), float(y2 - y1),
        float(confs[best]),
    )


def lerp_bbox(b_a, b_b, t):
    """Linear interpolation between two (x, y, w, h, conf) tuples,
    excluding conf (interpolated separately, just for logging)."""
    return (
        b_a[0] * (1 - t) + b_b[0] * t,
        b_a[1] * (1 - t) + b_b[1] * t,
        b_a[2] * (1 - t) + b_b[2] * t,
        b_a[3] * (1 - t) + b_b[3] * t,
        b_a[4] * (1 - t) + b_b[4] * t,
    )


def interpolate_missing(per_frame_face, n_frames):
    """Fill in missing detections by interpolating between nearest
    detected neighbors. Returns (filled_list, n_interpolated_frames).
    Edge cases (failure on first/last frames) use nearest detected.
    Caller has already guaranteed at least one detection exists.
    """
    filled = list(per_frame_face)
    detected_indices = [i for i, b in enumerate(filled) if b is not None]
    if not detected_indices:
        return filled, 0
    n_interp = 0

    # Left edge: backfill from first detection.
    first_det = detected_indices[0]
    for i in range(first_det):
        filled[i] = filled[first_det]
        n_interp += 1
    # Right edge: forward-fill from last detection.
    last_det = detected_indices[-1]
    for i in range(last_det + 1, n_frames):
        filled[i] = filled[last_det]
        n_interp += 1
    # Interior gaps: linear interpolate between bracketing detections.
    for di in range(len(detected_indices) - 1):
        a, b = detected_indices[di], detected_indices[di + 1]
        if b - a > 1:
            for i in range(a + 1, b):
                t = (i - a) / (b - a)
                filled[i] = lerp_bbox(filled[a], filled[b], t)
                n_interp += 1
    return filled, n_interp


def derive_eye_box(face_bbox, pos):
    """Return (x, y, w, h) tight eye box from face bbox + locked
    anatomical position."""
    fx, fy, fw, fh, _conf = face_bbox
    rx, ry, rw, rh = pos["rel_x"], pos["rel_y"], pos["rel_w"], pos["rel_h"]
    return (
        fx + rx * fw,
        fy + ry * fh,
        rw * fw,
        rh * fh,
    )


def apply_margin_and_square(tight, margin_pct, frame_shape):
    """Same as v3: tight + margin + square-pad + frame-clip."""
    x, y, w, h = tight
    mw = w * (margin_pct / 100.0)
    mh = h * (margin_pct / 100.0)
    x_m = x - mw
    y_m = y - mh
    w_m = w + 2 * mw
    h_m = h + 2 * mh
    side = max(w_m, h_m)
    cx = x_m + w_m / 2
    cy = y_m + h_m / 2
    sq_x1 = cx - side / 2
    sq_y1 = cy - side / 2
    sq_x2 = cx + side / 2
    sq_y2 = cy + side / 2
    h_frame, w_frame = frame_shape[:2]
    sq_x1 = max(0, int(round(sq_x1)))
    sq_y1 = max(0, int(round(sq_y1)))
    sq_x2 = min(w_frame, int(round(sq_x2)))
    sq_y2 = min(h_frame, int(round(sq_y2)))
    return sq_x1, sq_y1, sq_x2, sq_y2


def crop_clip(model, clip_path, pos, margin_pct, output_path):
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames < 1:
        cap.release()
        return {"status": "fail", "error": "no_frames"}

    # Pass 1: detect face per frame.
    per_frame = []
    frames_cache = []
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frames_cache.append(frame)
        per_frame.append(detect_face(model, frame))
    cap.release()

    n_native = len(frames_cache)
    n_detected = sum(1 for b in per_frame if b is not None)
    n_failed_raw = n_native - n_detected

    if n_native == 0:
        return {"status": "fail", "error": "no_frames_read"}
    if n_detected == 0:
        return {"status": "fail",
                "error": "no_face_detected_any_frame",
                "frames_native": n_native, "frames_detected": 0}

    failure_pct = n_failed_raw / n_native
    if failure_pct > FALLBACK_FAILURE_FRACTION:
        # Locked single-middle-frame fallback.
        mid_idx = n_native // 2
        mid_box = per_frame[mid_idx]
        if mid_box is None:
            # Mid frame failed; use nearest detected.
            detected_idx = sorted(
                (abs(i - mid_idx), i)
                for i, b in enumerate(per_frame) if b is not None
            )
            _, picked = detected_idx[0]
            mid_box = per_frame[picked]
        fallback_mode = "single_middle_frame"
        per_frame = [mid_box] * n_native
        n_interp = 0
    elif n_failed_raw == 0:
        fallback_mode = "per_frame"
        n_interp = 0
    else:
        per_frame, n_interp = interpolate_missing(per_frame, n_native)
        fallback_mode = "interpolated"

    # Pass 2: write crops.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps,
                              (CROP_RES, CROP_RES))
    frames_written = 0
    sample_first = None
    sample_mid = None
    sample_last = None
    sample_eye_first = None
    sample_eye_mid = None
    sample_eye_last = None
    confs = []
    for i, frame in enumerate(frames_cache):
        face = per_frame[i]
        if face is None:
            continue
        confs.append(face[4])
        eye = derive_eye_box(face, pos)
        x1, y1, x2, y2 = apply_margin_and_square(eye, margin_pct,
                                                  frame.shape)
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
            sample_first = [face[0], face[1], face[2], face[3]]
            sample_eye_first = [x1, y1, x2, y2]
        if i == n_native // 2:
            sample_mid = [face[0], face[1], face[2], face[3]]
            sample_eye_mid = [x1, y1, x2, y2]
        if i == n_native - 1:
            sample_last = [face[0], face[1], face[2], face[3]]
            sample_eye_last = [x1, y1, x2, y2]
    writer.release()

    if frames_written == 0:
        Path(output_path).unlink(missing_ok=True)
        return {"status": "fail", "error": "no_frames_written",
                "frames_native": n_native,
                "frames_detected": n_detected}

    mean_conf = sum(confs) / len(confs) if confs else None
    return {
        "status": "ok",
        "frames_native": n_native,
        "frames_detected": n_detected,
        "frames_interpolated": n_interp,
        "frames_failed_raw": n_failed_raw,
        "frame_failure_pct": failure_pct,
        "fallback_mode": fallback_mode,
        "frames_extracted": frames_written,
        "fps": fps,
        "mean_face_conf": mean_conf,
        "sample_face_bbox_first": sample_first,
        "sample_face_bbox_middle": sample_mid,
        "sample_face_bbox_last": sample_last,
        "sample_eye_crop_first": sample_eye_first,
        "sample_eye_crop_middle": sample_eye_mid,
        "sample_eye_crop_last": sample_eye_last,
        "output_path": str(Path(output_path).relative_to(POC_DIR)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin-pct", type=int, default=15,
                    choices=[10, 15, 40, 80],
                    help="Margin to apply around tight eye box "
                         "(default 15 — matches Phase 5 primary).")
    args = ap.parse_args()

    margin = args.margin_pct
    crops_dir = POC_DIR / "outputs" / f"eye_crops_phase6b_m{margin}"
    crops_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (POC_DIR / "outputs"
                     / f"eye_crops_phase6b_m{margin}_manifest.jsonl")
    manifest_path.unlink(missing_ok=True)

    keymap = json.loads(KEYMAP_PATH.read_text())["keymap"]
    pos_data = json.loads(POSITION_PARAM_PATH.read_text())
    pos = pos_data["locked_anatomical_position"]
    print(f"[phase6b m={margin}] {len(keymap)} clips; "
          f"position rel=(x={pos['rel_x']:.4f}, y={pos['rel_y']:.4f}, "
          f"w={pos['rel_w']:.4f}, h={pos['rel_h']:.4f})", flush=True)
    print(f"[phase6b] loading {FACE_MODEL.name}…", flush=True)
    model = YOLO(str(FACE_MODEL))

    n_ok = 0
    n_fail = 0
    n_per_frame = 0
    n_interp = 0
    n_fallback = 0
    for u, real_clip in keymap.items():
        clip_path = CLIPS_DIR / real_clip
        if not clip_path.exists():
            n_fail += 1
            with open(manifest_path, "a") as f:
                f.write(json.dumps({
                    "clip": real_clip, "uuid": u, "margin_pct": margin,
                    "status": "fail", "error": "clip_missing",
                }) + "\n")
            continue
        out_path = crops_dir / real_clip
        result = crop_clip(model, str(clip_path), pos, margin,
                           str(out_path))
        result["clip"] = real_clip
        result["uuid"] = u
        result["margin_pct"] = margin
        with open(manifest_path, "a") as f:
            f.write(json.dumps(result) + "\n")
        if result["status"] == "ok":
            n_ok += 1
            mode = result["fallback_mode"]
            if mode == "per_frame":
                n_per_frame += 1
            elif mode == "interpolated":
                n_interp += 1
            else:
                n_fallback += 1
            print(f"  [ok]  {real_clip:35} mode={mode:18} "
                  f"detected={result['frames_detected']:3d}/"
                  f"{result['frames_native']:3d} "
                  f"interp={result['frames_interpolated']:2d} "
                  f"fail%={result['frame_failure_pct']:.1%} "
                  f"conf̄={result['mean_face_conf']:.2f}",
                  flush=True)
        else:
            n_fail += 1
            print(f"  [FAIL] {real_clip}: {result.get('error')}",
                  flush=True)

    print(flush=True)
    print(f"=== phase6b m={margin}: {n_ok}/{len(keymap)} ok, "
          f"{n_per_frame} per-frame, {n_interp} interpolated, "
          f"{n_fallback} single-middle, {n_fail} fail ===", flush=True)
    print(f"Manifest: {manifest_path.relative_to(POC_DIR)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
