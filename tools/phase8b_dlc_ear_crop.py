#!/usr/bin/env python3
"""Phase 8b Step 5 — DLC ear-keypoint-anchored crop pipeline.

Per locked Phase 8b Stage 1 Decision 1: BOTH-EARS bbox encompassing 4
ear keypoints (right_earbase, right_earend, left_earbase, left_earend)
that meet confidence threshold 0.5, with margin 15% + square-pad to
224×224.

Frame-failure handling per Decision 1:
  - if ≥3 of 4 ear keypoints confident in a frame: bbox = enclosing
    rect of confident keypoints
  - if <3 of 4 confident: per-clip-locked fallback (use the frame with
    most confident ear keypoints across the clip; if none has ≥3,
    fallback to bounding all 4 keypoints regardless of confidence)

Inputs:
  outputs/phase8b_rme_dlc_keypoints.json  (Step 4 output)
  RME clips at vendor/ReadMyEars_Dataset/data/videos/

Outputs:
  outputs/eye_crops_v4_ear_dlc/<clip>.mp4
  outputs/eye_crops_v4_ear_dlc_manifest.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

POC_DIR = Path(__file__).resolve().parent.parent
KEYPOINTS_PATH = POC_DIR / "outputs" / "phase8b_rme_dlc_keypoints.json"
RME_VIDEOS = POC_DIR / "vendor" / "ReadMyEars_Dataset" / "data" / "videos"
CROPS_DIR = POC_DIR / "outputs" / "eye_crops_v4_ear_dlc"
MANIFEST_PATH = (POC_DIR / "outputs"
                 / "eye_crops_v4_ear_dlc_manifest.jsonl")

# Locked Phase 8b parameters
EAR_KP_INDICES = [6, 7, 11, 12]  # right_earbase, right_earend, left_earbase, left_earend
EAR_KP_NAMES = ["right_earbase", "right_earend",
                "left_earbase", "left_earend"]
CONF_THRESHOLD = 0.5
MIN_KP_CONFIDENT = 3
MARGIN_PCT = 15
CROP_RES = 224


def confident_indices(kps):
    """Return list of EAR_KP_INDICES that are confident in this frame."""
    if kps is None:
        return []
    return [idx for idx in EAR_KP_INDICES
            if kps[idx][2] >= CONF_THRESHOLD]


def enclosing_rect(kps, indices):
    """Min axis-aligned rect enclosing keypoints at given indices.
    Returns (x, y, w, h)."""
    xs = [kps[i][0] for i in indices]
    ys = [kps[i][1] for i in indices]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def select_clip_fallback_frame(per_frame_kps):
    """Decision 1 fallback: pick frame with most confident ear
    keypoints. Returns (frame_idx, confident_indices) or (None, []) if
    no frame has ≥1 confident keypoint."""
    best_idx = None
    best_count = 0
    best_indices = []
    for i, kps in enumerate(per_frame_kps):
        confs = confident_indices(kps)
        if len(confs) > best_count:
            best_count = len(confs)
            best_idx = i
            best_indices = confs
    return best_idx, best_indices


def apply_margin_and_square(tight, margin_pct, frame_shape):
    x, y, w, h = tight[:4]
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


def crop_clip(clip_path, info, output_path):
    keypoints = info["keypoints"]
    n_frames = info["n_frames"]

    # Identify per-clip fallback frame (used when frame has <3 confident)
    fallback_idx, fallback_kp_indices = select_clip_fallback_frame(
        keypoints
    )
    if fallback_idx is None:
        return {"status": "fail",
                "error": "no_confident_keypoints_any_frame",
                "n_frames": n_frames}

    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_native = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_native != n_frames:
        cap.release()
        return {"status": "fail",
                "error": f"frame_count_mismatch: video={n_native} keypoints={n_frames}"}

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps,
                              (CROP_RES, CROP_RES))
    n_per_frame_full = 0    # frames with ≥3 confident
    n_per_frame_fallback = 0   # frames with <3 confident, used clip-level fallback
    n_dropped = 0
    sample_first_box = None
    sample_mid_box = None
    sample_last_box = None
    fallback_clip_box = None  # fallback bbox computed from fallback frame

    # Pre-compute fallback clip box if needed (using frame with most
    # confident ear keypoints; if that frame has ≥3 confident, use
    # those; else use all 4 regardless of confidence)
    fallback_kps = keypoints[fallback_idx]
    if len(fallback_kp_indices) >= 1:
        if len(fallback_kp_indices) >= MIN_KP_CONFIDENT:
            fallback_kp_indices_for_box = fallback_kp_indices
        else:
            # Less than 3 confident on best frame: use all 4 regardless
            fallback_kp_indices_for_box = EAR_KP_INDICES
        fallback_clip_box = enclosing_rect(
            fallback_kps, fallback_kp_indices_for_box,
        )

    for i in range(n_native):
        ok, frame = cap.read()
        if not ok:
            break
        kps = keypoints[i]
        confs = confident_indices(kps) if kps is not None else []

        if len(confs) >= MIN_KP_CONFIDENT:
            tight = enclosing_rect(kps, confs)
            mode = "per_frame_confident"
            n_per_frame_full += 1
        elif fallback_clip_box is not None:
            tight = fallback_clip_box
            mode = "single_middle_frame_fallback"
            n_per_frame_fallback += 1
        else:
            n_dropped += 1
            continue

        x1, y1, x2, y2 = apply_margin_and_square(
            tight, MARGIN_PCT, frame.shape,
        )
        if x2 <= x1 or y2 <= y1:
            n_dropped += 1
            continue
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            n_dropped += 1
            continue
        crop_resized = cv2.resize(crop, (CROP_RES, CROP_RES),
                                   interpolation=cv2.INTER_AREA)
        writer.write(crop_resized)

        if i == 0:
            sample_first_box = [x1, y1, x2, y2]
        if i == n_native // 2:
            sample_mid_box = [x1, y1, x2, y2]
        if i == n_native - 1:
            sample_last_box = [x1, y1, x2, y2]
    cap.release()
    writer.release()

    frames_written = n_per_frame_full + n_per_frame_fallback
    if frames_written == 0:
        Path(output_path).unlink(missing_ok=True)
        return {"status": "fail", "error": "no_frames_written",
                "n_frames": n_native, "n_dropped": n_dropped}

    fallback_pct = (n_per_frame_fallback / frames_written
                     if frames_written > 0 else 0)

    return {
        "status": "ok",
        "frames_native": n_native,
        "frames_written": frames_written,
        "n_per_frame_confident": n_per_frame_full,
        "n_single_middle_fallback": n_per_frame_fallback,
        "fallback_pct": fallback_pct,
        "n_dropped": n_dropped,
        "fps": fps,
        "fallback_clip_frame_idx": fallback_idx,
        "fallback_clip_kp_indices_used": fallback_kp_indices_for_box
            if fallback_clip_box is not None else None,
        "sample_first_box": sample_first_box,
        "sample_mid_box": sample_mid_box,
        "sample_last_box": sample_last_box,
        "output_path": str(Path(output_path).relative_to(POC_DIR)),
    }


def main() -> int:
    keypoints_data = json.loads(KEYPOINTS_PATH.read_text())
    print(f"[phase8b crop] {len(keypoints_data['per_clip'])} clips "
          f"from Step 4 keypoints", flush=True)

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.unlink(missing_ok=True)

    n_ok = 0
    n_fail = 0
    n_per_frame_only = 0
    n_with_fallback = 0
    for clip, info in sorted(keypoints_data["per_clip"].items()):
        if info.get("status") != "ok":
            n_fail += 1
            with open(MANIFEST_PATH, "a") as f:
                f.write(json.dumps({
                    "clip": clip, "status": "fail",
                    "error": "no_keypoints_from_step4",
                }) + "\n")
            continue
        clip_path = RME_VIDEOS / clip
        if not clip_path.exists():
            n_fail += 1
            with open(MANIFEST_PATH, "a") as f:
                f.write(json.dumps({
                    "clip": clip, "status": "fail",
                    "error": "clip_missing",
                }) + "\n")
            continue
        out_path = CROPS_DIR / clip
        result = crop_clip(str(clip_path), info, str(out_path))
        result["clip"] = clip
        result["label"] = info.get("label")
        result["split"] = info.get("split")
        with open(MANIFEST_PATH, "a") as f:
            f.write(json.dumps(result) + "\n")
        if result["status"] == "ok":
            n_ok += 1
            if result["n_single_middle_fallback"] == 0:
                n_per_frame_only += 1
            else:
                n_with_fallback += 1
        else:
            n_fail += 1

    print(flush=True)
    print(f"=== phase8b crop pipeline: {n_ok}/{len(keypoints_data['per_clip'])} ok, "
          f"{n_per_frame_only} per-frame-only, "
          f"{n_with_fallback} with fallback, {n_fail} fail ===",
          flush=True)
    print(f"Manifest: {MANIFEST_PATH.relative_to(POC_DIR)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
