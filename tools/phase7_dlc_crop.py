#!/usr/bin/env python3
"""Phase 7 Step 3 — DLC keypoint-anchored eye-crop pipeline.

Reads:
- outputs/phase7_rme_dlc_keypoints.json (Step 1.5 output, hash 82ae2c8d11aa...)
- outputs/phase7_eye_side_assignment.json (Stage 1 lock, hash 95b879d2eb30...)
- outputs/phase6b_position_param.json (Phase 6b geometry hash d1521a35c2de...)

Locked parameters per Stage 1 + Stage 2 amendment:
- Confidence threshold (§7): 0.5  [Stage 2 hard-lock; meta-rule reframed]
- Frame-failure thresholds (§6): X_clip = 25%, Y_clip = 50%
- Absolute fallback dimensions (§5 option C): 62 × 47 px
- Head keypoint set (§2): {nose, upper_jaw, lower_jaw, mouth_end_right,
  mouth_end_left, right_eye, right_earbase, left_eye, left_earbase}
- Head-bbox proxy minimum (§5): ≥6 of 9 confident keypoints
- Margin (§3): 15% (matches Phase 5 primary)
- Crop size: 224×224 with cv2.INTER_AREA

Per-frame algorithm:
  1. Determine target eye keypoint per Stage 1 §4 rule:
     - Side-assigned clips: assigned eye unless its conf < 0.5 AND
       opposite eye conf > 0.7 → swap (per-frame)
     - Ambiguous clips: per-clip lock (computed once before per-frame
       loop) — pick the eye that was higher-confidence in the majority
       of confident frames; lock that eye for the entire clip
  2. If target eye conf < 0.5 → mark frame for interpolation
  3. Per-clip: count interpolation-needed frames
     - >50% → drop clip
     - 25-50% → use single-middle-keypoint (median target eye position)
       across all frames
     - <25% → interpolate per-frame from nearest confident neighbors
  4. Compute head-bbox proxy from 9 head keypoints with conf >= 0.5
     - ≥6 confident → use proxy; eye box dims = rel_w*proxy_w × rel_h*proxy_h
     - <6 confident → fallback dims = 62 × 47 px (absolute)
  5. Center eye box at target keypoint, apply margin 15%, square-pad,
     clip to frame, resize to 224×224

Outputs:
  outputs/eye_crops_v4_dlc/<clip>.mp4
  outputs/eye_crops_v4_dlc_manifest.jsonl  (per-clip metadata)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

POC_DIR = Path(__file__).resolve().parent.parent
CLIPS_DIR = POC_DIR / "vendor" / "ReadMyEars_Dataset" / "data" / "videos"
KEYPOINTS_PATH = POC_DIR / "outputs" / "phase7_rme_dlc_keypoints.json"
SIDE_PATH = POC_DIR / "outputs" / "phase7_eye_side_assignment_corrected.json"
POS_PARAM_PATH = POC_DIR / "outputs" / "phase6b_position_param.json"
CROPS_DIR = POC_DIR / "outputs" / "eye_crops_v4_dlc"
MANIFEST_PATH = POC_DIR / "outputs" / "eye_crops_v4_dlc_manifest.jsonl"

# Locked Stage 1 + Stage 2 parameters
CONF_THRESHOLD = 0.5  # Stage 2 §1 hard-lock
SWAP_OPPOSITE_THRESHOLD = 0.7  # Stage 1 §4 step 2
X_CLIP = 0.25  # Stage 2 §2 single-middle fallback trigger
Y_CLIP = 0.50  # Stage 2 §2 drop trigger
HEAD_KP_MIN_CONFIDENT = 6  # Stage 1 §5 minimum
HEAD_KP_INDICES = [0, 1, 2, 3, 4, 5, 6, 10, 11]  # locked Stage 1 §2
ABS_EYE_W_PX = 62  # Stage 2 §3
ABS_EYE_H_PX = 47  # Stage 2 §3
EYE_KP_IDX = {"right_eye": 5, "left_eye": 10}
MARGIN_PCT = 15  # Stage 1 + Phase 5 primary
CROP_RES = 224


def determine_per_clip_locked_eye(per_frame_keypoints):
    """For ambiguous-side clips: pick the eye that's higher-confidence
    in the majority of confident frames. Lock for the entire clip."""
    re_higher_count = 0
    le_higher_count = 0
    for kps in per_frame_keypoints:
        if kps is None:
            continue
        re_c = kps[EYE_KP_IDX["right_eye"]][2]
        le_c = kps[EYE_KP_IDX["left_eye"]][2]
        if max(re_c, le_c) < CONF_THRESHOLD:
            continue  # neither confident
        if re_c >= le_c:
            re_higher_count += 1
        else:
            le_higher_count += 1
    if re_higher_count >= le_higher_count:
        return "right_eye"
    return "left_eye"


def get_target_eye_keypoint(kps, target_eye_name, side):
    """Stage 1 §4 step 2 logic. Returns (kp_x, kp_y, kp_conf) or None
    if no usable eye keypoint at this frame."""
    if kps is None:
        return None
    if side == "ambiguous":
        # Per-clip lock already applied; just use target_eye_name
        kp = kps[EYE_KP_IDX[target_eye_name]]
        return kp if kp[2] >= CONF_THRESHOLD else None
    # Side-assigned clip: per-frame swap if assigned < 0.5 AND opposite > 0.7
    assigned = kps[EYE_KP_IDX[target_eye_name]]
    opposite_name = "left_eye" if target_eye_name == "right_eye" else "right_eye"
    opposite = kps[EYE_KP_IDX[opposite_name]]
    if assigned[2] >= CONF_THRESHOLD:
        return assigned
    # Assigned below threshold; check opposite
    if opposite[2] > SWAP_OPPOSITE_THRESHOLD:
        return opposite  # swap for this frame only
    return None  # neither usable


def interpolate_eye_keypoints(per_frame_target):
    """Fill None target eye keypoints by linear interpolation between
    nearest non-None neighbors. Returns (filled_list, n_interpolated,
    interpolated_indices_list)."""
    n = len(per_frame_target)
    valid_indices = [i for i, t in enumerate(per_frame_target)
                     if t is not None]
    if not valid_indices:
        return per_frame_target, 0, []
    filled = list(per_frame_target)
    interp_idx = []
    # Backfill before first valid
    first_valid = valid_indices[0]
    for i in range(first_valid):
        filled[i] = per_frame_target[first_valid]
        interp_idx.append(i)
    # Forward-fill after last valid
    last_valid = valid_indices[-1]
    for i in range(last_valid + 1, n):
        filled[i] = per_frame_target[last_valid]
        interp_idx.append(i)
    # Interior gaps: linear interpolation
    for di in range(len(valid_indices) - 1):
        a, b = valid_indices[di], valid_indices[di + 1]
        if b - a > 1:
            ka = per_frame_target[a]
            kb = per_frame_target[b]
            for i in range(a + 1, b):
                t = (i - a) / (b - a)
                filled[i] = (
                    ka[0] * (1 - t) + kb[0] * t,
                    ka[1] * (1 - t) + kb[1] * t,
                    ka[2] * (1 - t) + kb[2] * t,
                )
                interp_idx.append(i)
    return filled, len(interp_idx), interp_idx


def get_head_bbox_proxy(kps):
    """Stage 1 §5: minimum axis-aligned rectangle enclosing confident
    head keypoints. Returns (x, y, w, h) or None if <6 confident."""
    if kps is None:
        return None
    confident = [(kps[idx][0], kps[idx][1])
                 for idx in HEAD_KP_INDICES
                 if kps[idx][2] >= CONF_THRESHOLD]
    if len(confident) < HEAD_KP_MIN_CONFIDENT:
        return None
    xs = [p[0] for p in confident]
    ys = [p[1] for p in confident]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return (x1, y1, x2 - x1, y2 - y1)


def compute_eye_box(eye_kp, head_proxy, rel_w, rel_h):
    """Eye box centered at eye_kp with dimensions from head proxy or
    absolute fallback. Returns tight box (x, y, w, h)."""
    kp_x, kp_y = eye_kp[0], eye_kp[1]
    if head_proxy is not None:
        proxy_w, proxy_h = head_proxy[2], head_proxy[3]
        eye_w = rel_w * proxy_w
        eye_h = rel_h * proxy_h
        proxy_used = "head_keypoint_proxy"
    else:
        eye_w = ABS_EYE_W_PX
        eye_h = ABS_EYE_H_PX
        proxy_used = "abs_fallback"
    return (kp_x - eye_w / 2, kp_y - eye_h / 2,
            eye_w, eye_h, proxy_used)


def apply_margin_and_square(tight, margin_pct, frame_shape):
    """Same as Phase 5 v3 / Phase 6(b)."""
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


def crop_clip(clip_path, info, side_info, rel_w, rel_h, output_path):
    side = side_info.get("side", "ambiguous")
    target_eye_name = side_info.get(
        "target_eye_keypoint", "higher_conf_per_clip_lock_rule_a"
    )

    keypoints = info["keypoints"]  # n_frames × (39 × [x,y,c]) or None
    n_frames = info["n_frames"]

    # Per-clip lock for ambiguous side
    if side == "ambiguous":
        clip_locked_eye = determine_per_clip_locked_eye(keypoints)
        target_eye_name_resolved = clip_locked_eye
    else:
        clip_locked_eye = None
        target_eye_name_resolved = target_eye_name

    # Per-frame target eye keypoint
    per_frame_target = []
    per_frame_swap_log = []  # log frame indices where side-swap fired
    for i, kps in enumerate(keypoints):
        if side == "ambiguous":
            tgt = (kps[EYE_KP_IDX[target_eye_name_resolved]]
                   if kps is not None and
                   kps[EYE_KP_IDX[target_eye_name_resolved]][2] >= CONF_THRESHOLD
                   else None)
        else:
            tgt = get_target_eye_keypoint(
                kps, target_eye_name, side
            )
            # Log frame-level swap if it happened
            if (kps is not None and tgt is not None and
                    kps[EYE_KP_IDX[target_eye_name]][2] < CONF_THRESHOLD):
                per_frame_swap_log.append(i)
        per_frame_target.append(tgt)

    # Count frames needing interpolation (target is None)
    n_below_threshold = sum(1 for t in per_frame_target if t is None)
    fail_rate = n_below_threshold / n_frames if n_frames > 0 else 1.0

    if fail_rate > Y_CLIP:
        return {"status": "fail", "error": "drop_above_Y_clip",
                "fail_rate": fail_rate}

    if fail_rate > X_CLIP:
        # Single-middle-keypoint fallback
        valid = [t for t in per_frame_target if t is not None]
        if not valid:
            return {"status": "fail",
                    "error": "no_valid_target_keypoint_for_fallback"}
        med_x = sorted(t[0] for t in valid)[len(valid) // 2]
        med_y = sorted(t[1] for t in valid)[len(valid) // 2]
        med_c = sorted(t[2] for t in valid)[len(valid) // 2]
        per_frame_target = [(med_x, med_y, med_c)] * n_frames
        fallback_mode = "single_middle_keypoint"
        n_interpolated = 0
    else:
        # Interpolate from nearest neighbors
        per_frame_target, n_interpolated, _ = interpolate_eye_keypoints(
            per_frame_target
        )
        fallback_mode = ("per_frame" if n_interpolated == 0
                         else "interpolated")

    # Crop
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
    frames_written = 0
    n_proxy_failed = 0
    sample_eye_first = None
    sample_eye_mid = None
    sample_eye_last = None
    sample_proxy_used = []
    for i in range(n_native):
        ok, frame = cap.read()
        if not ok:
            break
        eye_kp = per_frame_target[i]
        kps = keypoints[i]
        head_proxy = get_head_bbox_proxy(kps)
        if head_proxy is None:
            n_proxy_failed += 1
        tight = compute_eye_box(eye_kp, head_proxy, rel_w, rel_h)
        x1, y1, x2, y2 = apply_margin_and_square(
            tight, MARGIN_PCT, frame.shape,
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
            sample_eye_first = [x1, y1, x2, y2]
            sample_proxy_used.append(tight[4])
        if i == n_native // 2:
            sample_eye_mid = [x1, y1, x2, y2]
            sample_proxy_used.append(tight[4])
        if i == n_native - 1:
            sample_eye_last = [x1, y1, x2, y2]
            sample_proxy_used.append(tight[4])
    cap.release()
    writer.release()

    if frames_written == 0:
        Path(output_path).unlink(missing_ok=True)
        return {"status": "fail", "error": "no_frames_written"}

    return {
        "status": "ok",
        "frames_native": n_native,
        "frames_written": frames_written,
        "fps": fps,
        "side": side,
        "target_eye_keypoint": target_eye_name,
        "clip_locked_eye": clip_locked_eye,  # set if ambiguous, else None
        "n_below_threshold_pre_fallback": n_below_threshold,
        "fail_rate_pre_fallback": fail_rate,
        "fallback_mode": fallback_mode,
        "n_frames_interpolated": n_interpolated,
        "n_frames_swap_to_opposite_eye": len(per_frame_swap_log),
        "frames_swap_indices": per_frame_swap_log,
        "n_frames_proxy_fallback_to_abs": n_proxy_failed,
        "sample_eye_crop_first": sample_eye_first,
        "sample_eye_crop_middle": sample_eye_mid,
        "sample_eye_crop_last": sample_eye_last,
        "sample_proxy_used": sample_proxy_used,
        "output_path": str(Path(output_path).relative_to(POC_DIR)),
    }


def main() -> int:
    keypoints_data = json.loads(KEYPOINTS_PATH.read_text())
    side_data = json.loads(SIDE_PATH.read_text())["per_clip"]
    pos_data = json.loads(POS_PARAM_PATH.read_text())
    rel_w = pos_data["locked_anatomical_position"]["rel_w"]
    rel_h = pos_data["locked_anatomical_position"]["rel_h"]
    print(f"[phase7_crop] rel_w={rel_w:.4f}, rel_h={rel_h:.4f} "
          f"(from Phase 6b position param)", flush=True)
    print(f"[phase7_crop] CONF_THRESHOLD={CONF_THRESHOLD} "
          f"(Stage 2 §1 hard-lock)", flush=True)
    print(f"[phase7_crop] X_CLIP={X_CLIP}, Y_CLIP={Y_CLIP} "
          f"(Stage 2 §2 defaults)", flush=True)
    print(f"[phase7_crop] head_kp_min_confident={HEAD_KP_MIN_CONFIDENT}/9, "
          f"abs_fallback={ABS_EYE_W_PX}×{ABS_EYE_H_PX}px", flush=True)

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.unlink(missing_ok=True)

    n_ok = 0
    n_fail = 0
    n_per_frame = 0
    n_interp = 0
    n_single_middle = 0
    n_drop = 0
    for clip, info in sorted(keypoints_data["per_clip"].items()):
        if info.get("status") != "ok":
            n_fail += 1
            with open(MANIFEST_PATH, "a") as f:
                f.write(json.dumps({
                    "clip": clip, "status": "fail",
                    "error": "no_keypoints_from_step1.5",
                }) + "\n")
            continue
        side_info = side_data.get(clip, {})
        clip_path = CLIPS_DIR / clip
        if not clip_path.exists():
            n_fail += 1
            with open(MANIFEST_PATH, "a") as f:
                f.write(json.dumps({
                    "clip": clip, "status": "fail",
                    "error": "clip_missing",
                }) + "\n")
            continue
        out_path = CROPS_DIR / clip
        result = crop_clip(str(clip_path), info, side_info,
                           rel_w, rel_h, str(out_path))
        result["clip"] = clip
        with open(MANIFEST_PATH, "a") as f:
            f.write(json.dumps(result) + "\n")
        if result["status"] == "ok":
            n_ok += 1
            mode = result["fallback_mode"]
            if mode == "per_frame":
                n_per_frame += 1
            elif mode == "interpolated":
                n_interp += 1
            else:
                n_single_middle += 1
            print(f"  [ok]  {clip:35} mode={mode:23} "
                  f"side={result['side']:23} "
                  f"swap={result['n_frames_swap_to_opposite_eye']:2d} "
                  f"interp={result['n_frames_interpolated']:2d} "
                  f"proxy_fallback={result['n_frames_proxy_fallback_to_abs']:2d}",
                  flush=True)
        else:
            err = result.get("error", "unknown")
            if "drop_above_Y" in err:
                n_drop += 1
            n_fail += 1
            print(f"  [FAIL] {clip}: {err}", flush=True)

    print(flush=True)
    print(f"=== phase7 crop pipeline: {n_ok}/34 ok, "
          f"{n_per_frame} per-frame, {n_interp} interpolated, "
          f"{n_single_middle} single-middle, {n_drop} drop, {n_fail} fail ===",
          flush=True)
    print(f"Manifest: {MANIFEST_PATH.relative_to(POC_DIR)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
