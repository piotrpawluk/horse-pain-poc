#!/usr/bin/env python3
"""Phase 7 Step 1.5 — DLC inference on 34 RME clips.

Per locked Stage 1 pre-reg §12 sequencing, runs after Wikimedia parity
PASSes and BEFORE Stage 2 amendment. Produces per-clip per-frame
keypoints; no crops, no LOSO, no gates fire yet.

The output drives Stage 2 amendment: confidence threshold, frame-failure
thresholds, fallback eye dimensions are derived from observations here.

Approach: call `deeplabcut.video_inference_superanimal()` once with all
34 clips as the videos list (model loaded once, batched internally).
Then aggregate per-clip JSONs into a single tracked artifact.

Output: outputs/phase7_rme_dlc_keypoints.json with structure:
  {
    "tool_hash": ...,
    "dlc_version": ...,
    "params": {...locked Stage 1 §1...},
    "per_clip": {
      "<filename>": {
        "n_frames": int,
        "primary_individual_indices": [...],
        "keypoints": [[[x,y,c], ...39x...] x n_frames],
        "bboxes": [[x,y,w,h], ...]
      }
    }
  }

Plus pre-Stage-2 instrumentation summary (descriptive only, no decisions):
  - Pooled p25 of target eye keypoint confidence (the value Stage 2 will
    feed into max(0.5, p25))
  - Per-clip target-eye failure rate (the rate Stage 2 will compare
    against Phase 0's ~9% baseline)
  - Median absolute eye_w, eye_h from Phase 5 manual boxes (the value
    Stage 2 will lock as fallback dimensions)
"""

from __future__ import annotations

import json
import shutil
import statistics
import sys
import time
from pathlib import Path

POC_DIR = Path(__file__).resolve().parent.parent
CLIPS_DIR = POC_DIR / "vendor" / "ReadMyEars_Dataset" / "data" / "videos"
KEYMAP_PATH = POC_DIR / "outputs" / "eye_box_keymap_phase5.json"
SIDE_ASSIGN_PATH = POC_DIR / "outputs" / "phase7_eye_side_assignment.json"
PHASE5_BOXES_PATH = POC_DIR / "outputs" / "eye_boxes_phase5a.json"
TMP_DIR = POC_DIR / "outputs" / ".tmp_phase7_dlc"
OUT_PATH = POC_DIR / "outputs" / "phase7_rme_dlc_keypoints.json"

# Locked DLC params per Stage 1 §1
SUPERANIMAL_NAME = "superanimal_quadruped"
MODEL_NAME = "hrnet_w32"
DETECTOR_NAME = "fasterrcnn_resnet50_fpn_v2"
VIDEO_ADAPT = False
PSEUDO_THRESHOLD = 0.1

# Locked keypoint indices per Stage 1 §2
EYE_KP_IDX = {"right_eye": 5, "left_eye": 10}
ALL_KEYPOINT_NAMES = [
    "nose", "upper_jaw", "lower_jaw", "mouth_end_right", "mouth_end_left",
    "right_eye", "right_earbase", "right_earend", "right_antler_base",
    "right_antler_end", "left_eye", "left_earbase", "left_earend",
    "left_antler_base", "left_antler_end", "neck_base", "neck_end",
    "throat_base", "throat_end", "back_base", "back_end", "back_middle",
    "tail_base", "tail_end", "front_left_thai", "front_left_knee",
    "front_left_paw", "front_right_thai", "front_right_knee",
    "front_right_paw", "back_left_paw", "back_left_thai",
    "back_right_thai", "back_left_knee", "back_right_knee",
    "back_right_paw", "belly_bottom", "body_middle_right",
    "body_middle_left",
]


def primary_individual_index(frame_entry):
    bbox_scores = frame_entry.get("bbox_scores", [])
    if not bbox_scores:
        return None
    valid = [(i, s) for i, s in enumerate(bbox_scores) if s >= 0]
    if not valid:
        return None
    best_i, _ = max(valid, key=lambda x: x[1])
    return best_i


def main() -> int:
    keymap = json.loads(KEYMAP_PATH.read_text())["keymap"]
    side_assignment = json.loads(SIDE_ASSIGN_PATH.read_text())["per_clip"]
    phase5_boxes = json.loads(PHASE5_BOXES_PATH.read_text())

    # Build video list (sorted for deterministic order)
    real_clips = sorted(keymap.values())
    video_paths = [str(CLIPS_DIR / clip) for clip in real_clips]
    missing = [v for v in video_paths if not Path(v).exists()]
    if missing:
        print(f"[ERROR] missing clips: {missing}", file=sys.stderr)
        return 1
    print(f"[step1.5] {len(video_paths)} clips queued for DLC inference",
          flush=True)

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True)

    print("[step1.5] loading deeplabcut + running batch inference...",
          flush=True)
    t0 = time.time()
    import deeplabcut
    dlc_version = deeplabcut.__version__
    print(f"[step1.5] DLC version: {dlc_version}", flush=True)

    deeplabcut.video_inference_superanimal(
        videos=video_paths,
        superanimal_name=SUPERANIMAL_NAME,
        model_name=MODEL_NAME,
        detector_name=DETECTOR_NAME,
        video_adapt=VIDEO_ADAPT,
        dest_folder=str(TMP_DIR),
        pseudo_threshold=PSEUDO_THRESHOLD,
        create_labeled_video=False,
    )
    elapsed = time.time() - t0
    print(f"[step1.5] inference complete in {elapsed:.1f}s "
          f"({elapsed/len(video_paths):.1f}s/clip avg)", flush=True)

    # Aggregate per-clip
    per_clip = {}
    target_eye_conf_pool = []  # for Stage 2 §7 pooled p25
    per_clip_target_eye_fail_rate = []  # for Stage 2 §6
    # Stage 1 pre-reg §2 head-keypoint set (9 of 39):
    #   nose=0, upper_jaw=1, lower_jaw=2,
    #   mouth_end_right=3, mouth_end_left=4,
    #   right_eye=5, right_earbase=6,
    #   left_eye=10, left_earbase=11
    head_kp_indices_locked = [0, 1, 2, 3, 4, 5, 6, 10, 11]
    assert len(head_kp_indices_locked) == 9

    for clip in real_clips:
        clip_stem = Path(clip).stem
        json_pattern = (
            f"{clip_stem}_{SUPERANIMAL_NAME}_{MODEL_NAME}_"
            f"{DETECTOR_NAME}_before_adapt.json"
        )
        json_path = TMP_DIR / json_pattern
        if not json_path.exists():
            print(f"[WARN] no JSON for {clip}", file=sys.stderr)
            per_clip[clip] = {"status": "fail", "error": "no_json"}
            continue

        frames = json.loads(json_path.read_text())
        n_frames = len(frames)
        side_info = side_assignment.get(clip, {})
        target_kp_name = side_info.get(
            "target_eye_keypoint", "higher_conf_per_clip_lock_rule_a"
        )

        per_frame_kps = []
        per_frame_bbox = []
        per_frame_primary_idx = []
        per_frame_target_eye_conf = []
        per_frame_head_n_confident_at_0_5 = []  # for context only
        for f in frames:
            pi = primary_individual_index(f)
            per_frame_primary_idx.append(pi)
            if pi is None:
                per_frame_kps.append(None)
                per_frame_bbox.append(None)
                per_frame_target_eye_conf.append(None)
                per_frame_head_n_confident_at_0_5.append(None)
                continue
            kps = f["bodyparts"][pi]  # 39 × [x, y, c]
            per_frame_kps.append(kps)
            per_frame_bbox.append(f["bboxes"][pi])

            # Target eye confidence (per side-assignment rule)
            if target_kp_name == "higher_conf_per_clip_lock_rule_a":
                # Ambiguous clip: report both right_eye and left_eye confs;
                # the per-clip lock decision happens in §4 step 1
                # (executed in Step 3 crop pipeline, not here).
                # For Stage 2 instrumentation: pool the higher of the two.
                re_c = kps[EYE_KP_IDX["right_eye"]][2]
                le_c = kps[EYE_KP_IDX["left_eye"]][2]
                target_c = max(re_c, le_c)
            elif target_kp_name in EYE_KP_IDX:
                target_c = kps[EYE_KP_IDX[target_kp_name]][2]
            else:
                target_c = None
            per_frame_target_eye_conf.append(target_c)
            if target_c is not None and target_c > 0:
                target_eye_conf_pool.append(target_c)

            # Head-keypoint confident-count at conservative 0.5
            # (informational — actual threshold is Stage 2 §7)
            n_conf_at_05 = sum(
                1 for idx in head_kp_indices_locked
                if kps[idx][2] >= 0.5
            )
            per_frame_head_n_confident_at_0_5.append(n_conf_at_05)

        # Per-clip target-eye failure rate at provisional 0.5 cutoff
        valid_confs = [c for c in per_frame_target_eye_conf if c is not None]
        if valid_confs:
            n_fail_at_0_5 = sum(1 for c in valid_confs if c < 0.5)
            fail_rate = n_fail_at_0_5 / len(valid_confs)
        else:
            fail_rate = None
        per_clip_target_eye_fail_rate.append((clip, fail_rate))

        per_clip[clip] = {
            "status": "ok",
            "n_frames": n_frames,
            "side_assignment": side_info,
            "target_eye_keypoint": target_kp_name,
            "primary_individual_index": per_frame_primary_idx,
            "keypoints": per_frame_kps,  # list of n_frames × 39 × [x,y,c]
            "bbox": per_frame_bbox,
            "target_eye_confidence_per_frame": per_frame_target_eye_conf,
            "head_keypoint_confident_count_at_0_5": (
                per_frame_head_n_confident_at_0_5
            ),
            "target_eye_fail_rate_at_0_5": fail_rate,
        }

    # Stage 2 instrumentation (DESCRIPTIVE ONLY — no thresholds locked here)
    pooled_p25 = (statistics.quantiles(target_eye_conf_pool, n=4)[0]
                  if target_eye_conf_pool else None)
    pooled_median = (statistics.median(target_eye_conf_pool)
                     if target_eye_conf_pool else None)
    pooled_p75 = (statistics.quantiles(target_eye_conf_pool, n=4)[2]
                  if target_eye_conf_pool else None)
    pooled_min = min(target_eye_conf_pool) if target_eye_conf_pool else None
    pooled_max = max(target_eye_conf_pool) if target_eye_conf_pool else None

    fail_rates = [r for _, r in per_clip_target_eye_fail_rate
                  if r is not None]
    pooled_fail_rate_provisional = (
        statistics.mean(fail_rates) if fail_rates else None
    )

    # Median absolute eye_w, eye_h from Phase 5 manual boxes
    abs_widths = []
    abs_heights = []
    for u, real_clip in keymap.items():
        if u not in phase5_boxes:
            continue
        mid = phase5_boxes[u].get("middle")
        if mid == "no_eye_visible" or mid is None:
            continue
        abs_widths.append(mid[2])
        abs_heights.append(mid[3])
    abs_w_median = statistics.median(abs_widths) if abs_widths else None
    abs_h_median = statistics.median(abs_heights) if abs_heights else None

    # Threshold-rule preview (NOT a lock — Stage 2 amendment will apply)
    threshold_preview = (max(0.5, pooled_p25)
                         if pooled_p25 is not None else None)

    summary = {
        "tool": "tools/phase7_run_dlc_on_rme.py",
        "stage1_pre_reg_section": "§12 Step 1.5",
        "dlc_version": dlc_version,
        "params": {
            "superanimal_name": SUPERANIMAL_NAME,
            "model_name": MODEL_NAME,
            "detector_name": DETECTOR_NAME,
            "video_adapt": VIDEO_ADAPT,
            "pseudo_threshold": PSEUDO_THRESHOLD,
        },
        "n_clips": len(real_clips),
        "per_clip": per_clip,
        "stage2_instrumentation_descriptive_only": {
            "purpose": (
                "Descriptive observations to feed Stage 2 amendment. "
                "No thresholds are locked by this file. Stage 2 amendment "
                "applies locked meta-rules from Stage 1 §6 and §7 to "
                "these observations."
            ),
            "pooled_target_eye_confidence_distribution": {
                "n": len(target_eye_conf_pool),
                "min": pooled_min,
                "p25": pooled_p25,
                "median": pooled_median,
                "p75": pooled_p75,
                "max": pooled_max,
            },
            "stage2_section_7_threshold_preview": {
                "rule": "max(0.5, pooled_p25_observed)",
                "pooled_p25_observed": pooled_p25,
                "resulting_threshold_preview": threshold_preview,
                "note": ("Stage 2 amendment will lock this concretely; "
                         "this file only previews it."),
            },
            "stage2_section_6_failure_rate_preview": {
                "rule": ("Default X=25%/Y=50% unless RME failure rate "
                         "≥ 2× Phase 0's ~9% baseline"),
                "phase0_baseline_failure_rate_at_0_5": (
                    "~9% (right_eye 24/287, left_eye 26/287; ~8.4% / 9.1%)"
                ),
                "phase7_rme_pooled_fail_rate_at_0_5_provisional": (
                    pooled_fail_rate_provisional
                ),
                "trigger_threshold_2x_phase0": 0.18,
                "note": ("Provisional fail rate at 0.5 cutoff. Stage 2 "
                         "amendment will recompute at the locked Stage 2 "
                         "threshold, not 0.5, and apply the meta-rule."),
            },
            "stage2_section_5_absolute_fallback_dimensions": {
                "source": "Phase 5 manual boxes middle keyframe",
                "abs_eye_w_median_px": abs_w_median,
                "abs_eye_h_median_px": abs_h_median,
                "n_clips_used": len(abs_widths),
                "note": ("Used as option C fallback in §5 step 2 when "
                         "head-bbox proxy has <6 of 9 confident "
                         "keypoints."),
            },
        },
    }
    summary["per_clip_target_eye_fail_rate_at_0_5"] = (
        per_clip_target_eye_fail_rate
    )

    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print()
    print(f"=== Step 1.5 complete: {len(per_clip)} clips processed ===")
    print()
    print("Stage 2 instrumentation preview (DESCRIPTIVE only, no locks "
          "fire here):")
    print()
    print(f"  Pooled target-eye confidence (n={len(target_eye_conf_pool)} "
          f"frames × clips):")
    print(f"    min:    {pooled_min}")
    print(f"    p25:    {pooled_p25}")
    print(f"    median: {pooled_median}")
    print(f"    p75:    {pooled_p75}")
    print(f"    max:    {pooled_max}")
    print()
    print(f"  §7 threshold preview = max(0.5, p25):  {threshold_preview}")
    print()
    print(f"  RME fail rate at 0.5: "
          f"{pooled_fail_rate_provisional:.4f} (Phase 0 ~0.085, "
          f"trigger ≥0.18)" if pooled_fail_rate_provisional is not None
          else "  RME fail rate at 0.5: N/A")
    print()
    print(f"  Phase 5 manual abs eye_w median: {abs_w_median:.2f} px")
    print(f"  Phase 5 manual abs eye_h median: {abs_h_median:.2f} px")
    print()
    print(f"Wrote: {OUT_PATH.relative_to(POC_DIR)}")
    print(f"(Tmp DLC outputs in {TMP_DIR.relative_to(POC_DIR)} — "
          f"keep for hash verification, delete after Stage 2 amendment "
          f"is committed.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
