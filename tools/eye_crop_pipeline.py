#!/usr/bin/env python3
"""Track B Phase 1 — eye-region crop pipeline.

Pre-registration: outputs/track_b_phase1_preregistration.md

Per-clip flow:
  1. Drift check: detect face on first/middle/last frame; record max center
     drift as a fraction of bbox width.
  2. Crop strategy:
       - drift <= 10% bbox width: single detection on middle frame, applied
         to all native frames.
       - drift  > 10% bbox width: per-frame detection.
  3. Eye-region heuristic v1: upper 40% of face bbox, full bbox width,
     square-padded around center, resized to 224x224.
  4. YOLO returns 0 detections (any frame for static, or on a frame
     during per-frame): SKIP + LOG. Never fall back to whole-frame.

Outputs:
  outputs/eye_crops/<clip>.mp4         - 224x224 eye-crop video at native fps
  outputs/eye_crops_manifest.jsonl     - per-clip: status, drift, frame counts
  outputs/eye_crops_drift_log.jsonl    - per-clip: first/mid/last bboxes + drift
  outputs/eye_crops_failed.txt         - newline-separated failures with reason
  outputs/eye_crops_contact_sheet.png  - 6x6 grid of middle-frame crops

The 36-clip set is reconstructed from the canonical Gemini A run JSONL,
matching the audit-followup ear-track for direct comparability.
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
CROPS_DIR = OUTPUTS / "eye_crops"
DRIFT_LOG = OUTPUTS / "eye_crops_drift_log.jsonl"
FAILED_LOG = OUTPUTS / "eye_crops_failed.txt"
MANIFEST = OUTPUTS / "eye_crops_manifest.jsonl"
CONTACT = OUTPUTS / "eye_crops_contact_sheet.png"
GEMINI_36_SOURCE = OUTPUTS / "gemini_audit_results_gemini-2.5-pro_promptA.jsonl"

CROP_RES = 224
DRIFT_THRESHOLD = 0.10
EYE_VERTICAL_FRACTION = 0.40
CONF_THRESHOLD = 0.5


def load_36_clips() -> list[dict]:
    rows, seen = [], set()
    with open(GEMINI_36_SOURCE) as f:
        for line in f:
            r = json.loads(line)
            if r["clip"] in seen:
                continue
            seen.add(r["clip"])
            rows.append({"clip": r["clip"], "video_rel": r.get("video_rel")})
    if len(rows) != 36:
        sys.exit(f"expected 36 clips, got {len(rows)}")
    return rows


def detect_face(model, frame, conf: float = CONF_THRESHOLD):
    """Return (x1, y1, x2, y2, confidence) for the highest-confidence detection,
    or None if nothing detected."""
    results = model(frame, conf=conf, imgsz=640, verbose=False)
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        confs = r.boxes.conf.cpu().numpy()
        idx = int(np.argmax(confs))
        bbox = r.boxes.xyxy.cpu().numpy()[idx]
        return (*bbox.tolist(), float(confs[idx]))
    return None


def eye_region_from_face(bbox, frame_shape):
    """Heuristic v1: upper 40% / full width / square-padded around bbox center."""
    x1, y1, x2, y2 = bbox[:4]
    h_frame, w_frame = frame_shape[:2]
    bh = y2 - y1
    eye_y1 = y1
    eye_y2 = y1 + bh * EYE_VERTICAL_FRACTION
    eye_x1, eye_x2 = x1, x2
    cw = eye_x2 - eye_x1
    ch = eye_y2 - eye_y1
    side = max(cw, ch)
    cx = (eye_x1 + eye_x2) / 2
    cy = (eye_y1 + eye_y2) / 2
    sq_x1 = int(round(cx - side / 2))
    sq_y1 = int(round(cy - side / 2))
    sq_x2 = int(round(cx + side / 2))
    sq_y2 = int(round(cy + side / 2))
    sq_x1 = max(0, sq_x1)
    sq_y1 = max(0, sq_y1)
    sq_x2 = min(w_frame, sq_x2)
    sq_y2 = min(h_frame, sq_y2)
    return sq_x1, sq_y1, sq_x2, sq_y2


def measure_drift(model, video_path: str):
    """Detect on first/middle/last; return (max_drift, bboxes_dict).
    Drift = max(|Δcx|/bw, |Δcy|/bh) across the three samples vs middle."""
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


def crop_clip(video_path: str, model, *, per_frame: bool = False) -> dict:
    """Crop all native frames to eye region. Returns per-clip status dict."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    static_eye_bbox = None
    static_face_bbox = None
    if not per_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return {"status": "fail", "error": "cannot_read_middle_frame"}
        det = detect_face(model, frame)
        if det is None:
            cap.release()
            return {"status": "fail", "error": "no_detection_static"}
        static_face_bbox = det[:4]
        static_eye_bbox = eye_region_from_face(static_face_bbox, frame.shape)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = CROPS_DIR / Path(video_path).name
    out = None

    frames_written = 0
    skipped_no_detection = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if per_frame:
            det = detect_face(model, frame)
            if det is None:
                skipped_no_detection += 1
                continue
            eye_bbox = eye_region_from_face(det[:4], frame.shape)
        else:
            eye_bbox = static_eye_bbox

        x1, y1, x2, y2 = eye_bbox
        if x2 <= x1 or y2 <= y1:
            skipped_no_detection += 1
            continue
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            skipped_no_detection += 1
            continue
        crop_resized = cv2.resize(
            crop, (CROP_RES, CROP_RES), interpolation=cv2.INTER_AREA
        )
        if out is None:
            out = cv2.VideoWriter(
                str(out_path), fourcc, fps, (CROP_RES, CROP_RES)
            )
        out.write(crop_resized)
        frames_written += 1
    cap.release()
    if out:
        out.release()

    if frames_written == 0:
        out_path.unlink(missing_ok=True)
        return {"status": "fail",
                "error": f"no_frames_written ({skipped_no_detection} skipped)"}

    return {
        "status": "ok",
        "frames_extracted": frames_written,
        "frames_skipped": skipped_no_detection,
        "frames_native": n,
        "face_bbox": list(static_face_bbox) if not per_frame else "per_frame",
        "eye_bbox": list(static_eye_bbox) if not per_frame else "per_frame",
        "output_path": str(out_path.relative_to(POC_DIR)),
        "per_frame_detection": per_frame,
        "fps": fps,
    }


def make_contact_sheet(manifest_rows: list[dict]) -> None:
    """6x6 grid of middle-frame crops. 36 cells max."""
    GRID = 6
    PAD = 4
    LABEL_H = 18
    cell = CROP_RES + LABEL_H + PAD
    sheet = np.full((GRID * cell, GRID * cell, 3), 240, dtype=np.uint8)
    rows_with_crops = [r for r in manifest_rows if r.get("status") == "ok"]
    for i, row in enumerate(rows_with_crops[:GRID * GRID]):
        r, c = divmod(i, GRID)
        crop_path = POC_DIR / row["output_path"]
        cap = cv2.VideoCapture(str(crop_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        y0 = r * cell + PAD
        x0 = c * cell + PAD
        sheet[y0:y0 + CROP_RES, x0:x0 + CROP_RES] = frame
        label = Path(row["clip"]).stem[:24]
        cv2.putText(
            sheet, label, (x0, y0 + CROP_RES + LABEL_H - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
        )
    cv2.imwrite(str(CONTACT), sheet)


def main() -> int:
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    DRIFT_LOG.unlink(missing_ok=True)
    FAILED_LOG.unlink(missing_ok=True)
    MANIFEST.unlink(missing_ok=True)

    print(f"[track_b] loading YOLO weights: {YOLO_WEIGHTS.name}", flush=True)
    model = YOLO(str(YOLO_WEIGHTS))
    print(f"[track_b] loaded; classes={list(model.names.values())}", flush=True)

    rows = load_36_clips()
    manifest: list[dict] = []
    for i, r in enumerate(rows, 1):
        clip_path = CLIPS_DIR / Path(r["clip"]).name
        if not clip_path.exists():
            print(f"  [{i:2d}/36] MISSING {clip_path}", flush=True)
            with open(FAILED_LOG, "a") as fh:
                fh.write(f"{r['clip']}\tmissing_file\n")
            manifest.append({
                "clip": r["clip"],
                "status": "fail",
                "error": "missing_file",
            })
            continue

        drift, drift_bboxes = measure_drift(model, str(clip_path))
        with open(DRIFT_LOG, "a") as fh:
            fh.write(json.dumps({
                "clip": r["clip"],
                "drift": drift,
                "first_bbox": list(drift_bboxes.get("first", [])),
                "mid_bbox": list(drift_bboxes.get("mid", [])),
                "last_bbox": list(drift_bboxes.get("last", [])),
            }) + "\n")

        per_frame = drift is not None and drift > DRIFT_THRESHOLD
        result = crop_clip(str(clip_path), model, per_frame=per_frame)
        result["clip"] = r["clip"]
        result["drift"] = drift
        manifest.append(result)

        if result["status"] == "fail":
            with open(FAILED_LOG, "a") as fh:
                fh.write(f"{r['clip']}\t{result.get('error', 'unknown')}\n")
            print(f"  [{i:2d}/36] FAIL {r['clip']:30} {result.get('error')}",
                  flush=True)
        else:
            tag = "PER-FRAME" if per_frame else "static   "
            d = drift if drift is not None else 0
            print(
                f"  [{i:2d}/36] OK   {r['clip']:30} drift={d:.2f} "
                f"frames={result['frames_extracted']:3d}/{result['frames_native']:3d} "
                f"({tag})",
                flush=True,
            )

    with open(MANIFEST, "w") as fh:
        for row in manifest:
            fh.write(json.dumps(row) + "\n")

    make_contact_sheet(manifest)

    n_ok = sum(1 for r in manifest if r.get("status") == "ok")
    n_fail = sum(1 for r in manifest if r.get("status") == "fail")
    n_per_frame = sum(
        1 for r in manifest
        if r.get("status") == "ok" and r.get("per_frame_detection")
    )
    print(flush=True)
    print(f"=== Track B Phase 1: {n_ok}/36 cropped, {n_fail} failed, "
          f"{n_per_frame} per-frame ===", flush=True)
    print(f"Contact sheet: {CONTACT.relative_to(POC_DIR)}", flush=True)
    print(f"Manifest:      {MANIFEST.relative_to(POC_DIR)}", flush=True)
    print(f"Drift log:     {DRIFT_LOG.relative_to(POC_DIR)}", flush=True)
    if n_fail:
        print(f"Failed list:   {FAILED_LOG.relative_to(POC_DIR)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
