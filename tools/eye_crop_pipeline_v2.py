#!/usr/bin/env python3
"""Track B Phase 4 — v2 profile-aware eye-crop pipeline.

Locked spec (track_b_phase4_preregistration.md, hash ced5cae6...):

  - Upper 40 % strip of YOLO face bbox is split into LEFT and RIGHT halves
    horizontally.
  - For each half: mean spatial-frequency content via Sobel edge magnitude,
    smoothed with a 32×32 box filter; the maximum smoothed value across the
    half is the half's score (the highest-frequency 32×32 region within the
    half — proxy for "most likely to contain the eye").
  - Half with higher score wins.
  - Tie-break (within 5 % of higher half): BOTH halves output as separate
    samples; both inherit the parent clip's source for LOSO grouping.
  - Output: square-padded around the chosen half's center, resized to 224×224,
    applied to all native frames of the clip.

Drift / per-frame detection logic mirrors v1 (`tools/eye_crop_pipeline.py`):
single YOLO face detection on the middle frame for static clips, per-frame
detection for clips with center drift > 10 % bbox width on the
first/middle/last sample. Half selection happens once per clip on the
middle frame's upper-40 % strip; the chosen half is then applied to all
frames using each frame's own face bbox.

Outputs:
  outputs/eye_crops_v2/<clip_basename>.mp4              — non-tied clips
  outputs/eye_crops_v2/<clip_basename>__left.mp4        — tied clips, left
  outputs/eye_crops_v2/<clip_basename>__right.mp4       — tied clips, right
  outputs/eye_crops_v2_manifest.jsonl                   — per-clip metadata
  outputs/eye_crops_v2_failed.txt                       — clips with no detection
  outputs/eye_crops_v2_contact_sheet.png                — middle-frame grid

Tie-break suffix convention: parent clip "action_S5.mp4_5_.mp4" with a tie
produces "action_S5.mp4_5___left.mp4" and "action_S5.mp4_5___right.mp4"
(triple-underscore delimiter, separator from any existing underscore noise).
LOSO source extraction strips "__left" / "__right" before parsing source.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

POC_DIR = Path(__file__).resolve().parent.parent
YOLO_WEIGHTS = (
    POC_DIR / "vendor" / "horse-face-ear-detection"
    / "horse_face_detection" / "yolov8l_horse_face_detection.pt"
)
CLIPS_DIR = POC_DIR / "vendor" / "ReadMyEars_Dataset" / "data" / "videos"
OUTPUTS = POC_DIR / "outputs"
CROPS_DIR = OUTPUTS / "eye_crops_v2"
MANIFEST = OUTPUTS / "eye_crops_v2_manifest.jsonl"
FAILED_LOG = OUTPUTS / "eye_crops_v2_failed.txt"
CONTACT = OUTPUTS / "eye_crops_v2_contact_sheet.png"

# Reuse the canonical 34-clip viable set from v1 manifest
V1_MANIFEST = OUTPUTS / "eye_crops_manifest.jsonl"
ANNOTATIONS = OUTPUTS / "eye_crops_annotations.md"

CROP_RES = 224
DRIFT_THRESHOLD = 0.10
EYE_VERTICAL_FRACTION = 0.40
CONF_THRESHOLD = 0.5
SOBEL_BOX = 32
TIE_BREAK_FRACTION = 0.05  # within 5 % → output both halves


def detect_face(model, frame, conf: float = CONF_THRESHOLD):
    """Highest-confidence detection or None."""
    results = model(frame, conf=conf, imgsz=640, verbose=False)
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        confs = r.boxes.conf.cpu().numpy()
        idx = int(np.argmax(confs))
        bbox = r.boxes.xyxy.cpu().numpy()[idx]
        return (*bbox.tolist(), float(confs[idx]))
    return None


def measure_drift(model, video_path: str):
    """Detect on first/middle/last; return (max_drift, bboxes_dict)."""
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bboxes = {}
    for label, idx in [("first", 0), ("mid", n // 2), ("last", max(0, n - 1))]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        det = detect_face(model, frame)
        if det:
            bboxes[label] = det[:4]
    cap.release()
    if "mid" not in bboxes:
        return None, bboxes
    mx1, my1, mx2, my2 = bboxes["mid"]
    bw, bh = mx2 - mx1, my2 - my1
    if bw <= 0 or bh <= 0:
        return None, bboxes
    mid_cx, mid_cy = (mx1 + mx2) / 2, (my1 + my2) / 2
    max_drift = 0.0
    for k, b in bboxes.items():
        if k == "mid":
            continue
        cx = (b[0] + b[2]) / 2
        cy = (b[1] + b[3]) / 2
        d = max(abs(cx - mid_cx) / bw, abs(cy - mid_cy) / bh)
        max_drift = max(max_drift, d)
    return max_drift, bboxes


def upper_strip_bounds(face_bbox, frame_shape):
    """Upper 40 % of face bbox, full bbox width, clipped to frame."""
    x1, y1, x2, y2 = face_bbox[:4]
    h_frame, w_frame = frame_shape[:2]
    bh = y2 - y1
    sy1 = max(0, int(round(y1)))
    sy2 = min(h_frame, int(round(y1 + bh * EYE_VERTICAL_FRACTION)))
    sx1 = max(0, int(round(x1)))
    sx2 = min(w_frame, int(round(x2)))
    return sx1, sy1, sx2, sy2


def half_freq_score(half_gray: np.ndarray, kernel: int = SOBEL_BOX) -> float:
    """Sobel edge magnitude → 32×32 box-filtered mean → max value over half.
    Proxy for 'is there a high-frequency-content sub-region in this half?'.
    """
    if half_gray.size == 0:
        return 0.0
    sx = cv2.Sobel(half_gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(half_gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sx * sx + sy * sy).astype(np.float32)
    k = min(kernel, magnitude.shape[0], magnitude.shape[1])
    if k < 3:
        return float(magnitude.mean())
    smoothed = cv2.boxFilter(magnitude, ddepth=-1, ksize=(k, k))
    return float(smoothed.max())


def select_halves(face_bbox, frame_bgr, frame_shape):
    """Returns (left_score, right_score, decision) where decision is one of
    'left', 'right', 'tie'. Strip + half computed from face_bbox on this frame.
    """
    sx1, sy1, sx2, sy2 = upper_strip_bounds(face_bbox, frame_shape)
    if sx2 <= sx1 or sy2 <= sy1:
        return 0.0, 0.0, "left"
    strip = frame_bgr[sy1:sy2, sx1:sx2]
    if strip.size == 0:
        return 0.0, 0.0, "left"
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    mid_x = strip.shape[1] // 2
    left_half = gray[:, :mid_x]
    right_half = gray[:, mid_x:]
    sl = half_freq_score(left_half)
    sr = half_freq_score(right_half)
    higher = max(sl, sr)
    if higher == 0:
        return sl, sr, "left"
    diff_frac = abs(sl - sr) / higher
    if diff_frac < TIE_BREAK_FRACTION:
        return sl, sr, "tie"
    return sl, sr, ("left" if sl > sr else "right")


def half_crop_box(face_bbox, frame_shape, side: str):
    """Square crop box around the chosen half's center."""
    sx1, sy1, sx2, sy2 = upper_strip_bounds(face_bbox, frame_shape)
    h_frame, w_frame = frame_shape[:2]
    strip_w = sx2 - sx1
    strip_h = sy2 - sy1
    half_w = strip_w // 2
    if side == "left":
        hx1, hx2 = sx1, sx1 + half_w
    else:
        hx1, hx2 = sx2 - half_w, sx2
    cx = (hx1 + hx2) / 2
    cy = (sy1 + sy2) / 2
    side_size = max(half_w, strip_h)
    sq_x1 = max(0, int(round(cx - side_size / 2)))
    sq_y1 = max(0, int(round(cy - side_size / 2)))
    sq_x2 = min(w_frame, int(round(cx + side_size / 2)))
    sq_y2 = min(h_frame, int(round(cy + side_size / 2)))
    return sq_x1, sq_y1, sq_x2, sq_y2


def crop_clip(video_path: str, model, *, per_frame: bool = False) -> dict:
    """Apply v2 to a single clip. Returns metadata dict; outputs MP4(s) to CROPS_DIR."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    static_face_bbox = None
    if not per_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, mid_frame = cap.read()
        if not ok:
            cap.release()
            return {"status": "fail", "error": "cannot_read_middle_frame"}
        det = detect_face(model, mid_frame)
        if det is None:
            cap.release()
            return {"status": "fail", "error": "no_detection_static"}
        static_face_bbox = det[:4]
    # For per-frame mode, we still need a half decision for the whole clip;
    # we use the middle frame for half-selection regardless of detection cadence.
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, mid_frame = cap.read()
    if not ok:
        cap.release()
        return {"status": "fail", "error": "cannot_read_middle_frame_for_half_selection"}
    if per_frame:
        det_mid = detect_face(model, mid_frame)
        if det_mid is None:
            cap.release()
            return {"status": "fail", "error": "no_detection_middle_frame_per_frame_mode"}
        face_for_selection = det_mid[:4]
    else:
        face_for_selection = static_face_bbox

    sl, sr, decision = select_halves(face_for_selection, mid_frame, mid_frame.shape)
    sides = ["left", "right"] if decision == "tie" else [decision]

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    base_name = Path(video_path).name  # e.g., "action_S5.mp4_5_.mp4"
    if base_name.endswith(".mp4"):
        stem = base_name[:-4]
    else:
        stem = base_name

    writers = {}
    out_paths = {}
    for side in sides:
        if decision == "tie":
            out_path = CROPS_DIR / f"{stem}__{side}.mp4"
        else:
            out_path = CROPS_DIR / f"{stem}.mp4"
        writers[side] = None  # lazy init at first frame
        out_paths[side] = out_path

    frames_written = {side: 0 for side in sides}
    skipped = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if per_frame:
            det = detect_face(model, frame)
            if det is None:
                skipped += 1
                continue
            face_bbox = det[:4]
        else:
            face_bbox = static_face_bbox

        for side in sides:
            x1, y1, x2, y2 = half_crop_box(face_bbox, frame.shape, side)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_resized = cv2.resize(crop, (CROP_RES, CROP_RES),
                                      interpolation=cv2.INTER_AREA)
            if writers[side] is None:
                writers[side] = cv2.VideoWriter(
                    str(out_paths[side]), fourcc, fps, (CROP_RES, CROP_RES)
                )
            writers[side].write(crop_resized)
            frames_written[side] += 1
    cap.release()
    for w in writers.values():
        if w is not None:
            w.release()

    if all(c == 0 for c in frames_written.values()):
        for op in out_paths.values():
            op.unlink(missing_ok=True)
        return {"status": "fail",
                "error": f"no_frames_written ({skipped} per-frame skipped)"}

    return {
        "status": "ok",
        "frames_native": n,
        "fps": fps,
        "frames_skipped_no_detection": skipped,
        "left_score": sl,
        "right_score": sr,
        "decision": decision,
        "tie_break_diff_fraction": (
            abs(sl - sr) / max(sl, sr) if max(sl, sr) > 0 else 0.0
        ),
        "outputs": [
            {
                "side": side,
                "path": str(out_paths[side].relative_to(POC_DIR)),
                "frames_extracted": frames_written[side],
            }
            for side in sides
        ],
        "per_frame_detection": per_frame,
        "face_bbox_used_for_half_selection": list(face_for_selection),
    }


def load_v1_viable_clips() -> list[str]:
    """Reuse the 34 viable clips from v1 (excludes YOLO-failed +
    manually-excluded). Per the locked Phase 4 protocol."""
    viable = []
    excluded = {"background_S1.mp4_11_.mp4"}  # manual eye<20% exclusion
    with open(V1_MANIFEST) as f:
        for line in f:
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            base = Path(row["clip"]).name
            if base in excluded:
                continue
            viable.append(base)
    return viable


def make_contact_sheet(manifest_rows: list[dict]) -> None:
    """Variable-length grid; one cell per output clip (some parent clips have 2)."""
    cells: list[tuple[str, Path]] = []
    for row in manifest_rows:
        if row.get("status") != "ok":
            continue
        for out in row["outputs"]:
            label = Path(row["clip"]).stem
            if row["decision"] == "tie":
                label = f"{label}__{out['side']}"
            cells.append((label[:24], POC_DIR / out["path"]))
    n_cells = len(cells)
    GRID = max(6, int(np.ceil(np.sqrt(n_cells))))
    PAD = 4
    LABEL_H = 18
    cell_size = CROP_RES + LABEL_H + PAD
    sheet = np.full((GRID * cell_size, GRID * cell_size, 3), 240,
                    dtype=np.uint8)
    for i, (label, path) in enumerate(cells[:GRID * GRID]):
        r, c = divmod(i, GRID)
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        y0 = r * cell_size + PAD
        x0 = c * cell_size + PAD
        sheet[y0:y0 + CROP_RES, x0:x0 + CROP_RES] = frame
        cv2.putText(sheet, label, (x0, y0 + CROP_RES + LABEL_H - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.imwrite(str(CONTACT), sheet)


def main() -> int:
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.unlink(missing_ok=True)
    FAILED_LOG.unlink(missing_ok=True)

    print(f"[v2] loading YOLO weights: {YOLO_WEIGHTS.name}", flush=True)
    model = YOLO(str(YOLO_WEIGHTS))

    viable = load_v1_viable_clips()
    print(f"[v2] {len(viable)} viable clips from v1 manifest", flush=True)

    manifest: list[dict] = []
    for i, base in enumerate(viable, 1):
        clip_path = CLIPS_DIR / base
        if not clip_path.exists():
            with open(FAILED_LOG, "a") as fh:
                fh.write(f"{base}\tmissing_file\n")
            manifest.append({"clip": base, "status": "fail",
                             "error": "missing_file"})
            continue

        drift, _ = measure_drift(model, str(clip_path))
        per_frame = drift is not None and drift > DRIFT_THRESHOLD
        result = crop_clip(str(clip_path), model, per_frame=per_frame)
        result["clip"] = base
        result["drift"] = drift
        manifest.append(result)

        if result["status"] == "fail":
            with open(FAILED_LOG, "a") as fh:
                fh.write(f"{base}\t{result.get('error', 'unknown')}\n")
            print(f"  [{i:2d}/{len(viable)}] FAIL {base:30}  "
                  f"{result.get('error')}", flush=True)
        else:
            tag = "PER-FRAME" if per_frame else "static   "
            d = drift if drift is not None else 0
            tie = " (TIE — both halves)" if result["decision"] == "tie" else ""
            print(f"  [{i:2d}/{len(viable)}] OK   {base:30}  "
                  f"L={result['left_score']:.2f} R={result['right_score']:.2f}  "
                  f"{result['decision']:<5}{tie}  drift={d:.2f} ({tag})",
                  flush=True)

    with open(MANIFEST, "w") as fh:
        for row in manifest:
            fh.write(json.dumps(row) + "\n")

    make_contact_sheet(manifest)

    n_ok = sum(1 for r in manifest if r.get("status") == "ok")
    n_fail = sum(1 for r in manifest if r.get("status") == "fail")
    n_tie = sum(1 for r in manifest
                if r.get("status") == "ok" and r.get("decision") == "tie")
    n_left = sum(1 for r in manifest
                 if r.get("status") == "ok" and r.get("decision") == "left")
    n_right = sum(1 for r in manifest
                  if r.get("status") == "ok" and r.get("decision") == "right")
    n_outputs = sum(len(r["outputs"]) for r in manifest if r.get("status") == "ok")
    print(flush=True)
    print(f"=== v2 done: {n_ok} clips → {n_outputs} output crops "
          f"({n_left} left, {n_right} right, {n_tie} ties × 2 halves), "
          f"{n_fail} failed ===", flush=True)
    print(f"Manifest:      {MANIFEST.relative_to(POC_DIR)}", flush=True)
    print(f"Contact sheet: {CONTACT.relative_to(POC_DIR)}", flush=True)
    if n_fail:
        print(f"Failed list:   {FAILED_LOG.relative_to(POC_DIR)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
