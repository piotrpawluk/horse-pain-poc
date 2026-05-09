#!/usr/bin/env python3
"""Phase 7 Stage 2 amendment v2 — corrected side-assignment derivation.

Empirical correction to Stage 1 §4 mapping. Original lock was:
  manual eye on RIGHT of face center  → target = right_eye (DLC keypoint)
  manual eye on LEFT of face center   → target = left_eye

Empirically falsified: 24/34 RME clips show flipped rule produces
2.5× higher mean IoU vs Phase 5 manual boxes (0.42 vs 0.17).
Phase 0 Wikimedia confirms DLC labeling convention is
horse-anatomical (231/287 frames have right_eye_x < left_eye_x in
both-eyes-visible regime, consistent with anatomical right being
image-LEFT of anatomical left when horse faces camera).

The §4 anatomical reasoning was inverted: profile-LEFT horse (head
image-left) means camera sees horse's LEFT side (not right), so
visible eye is anatomical LEFT (DLC `left_eye`), not right.

Corrected mapping (horse-anatomical, empirically verified):
  manual eye on RIGHT of face center  → DLC `left_eye`
  manual eye on LEFT of face center   → DLC `right_eye`
  ambiguous (|offset| < 0.05)         → per-clip lock (rule (a),
                                         unchanged from Stage 1)

Output: outputs/phase7_eye_side_assignment_corrected.json
(hash-locked BEFORE the corrected re-run).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

POC_DIR = Path(__file__).resolve().parent.parent
EYE_BOXES_PATH = POC_DIR / "outputs" / "eye_boxes_phase5a.json"
KEYMAP_PATH = POC_DIR / "outputs" / "eye_box_keymap_phase5.json"
POSITION_PARAM_PATH = POC_DIR / "outputs" / "phase6b_position_param.json"
ORIGINAL_SIDE_ASSIGN_PATH = (
    POC_DIR / "outputs" / "phase7_eye_side_assignment.json"
)
OUT_PATH = (
    POC_DIR / "outputs" / "phase7_eye_side_assignment_corrected.json"
)

MARGIN_THRESHOLD = 0.05  # unchanged from Stage 1 §4


def main() -> int:
    boxes = json.loads(EYE_BOXES_PATH.read_text())
    keymap = json.loads(KEYMAP_PATH.read_text())["keymap"]
    pos = json.loads(POSITION_PARAM_PATH.read_text())
    per_clip_face = {r["clip"]: r["face_bbox"] for r in pos["per_clip"]}

    per_clip = {}
    side_counts = {
        "right_of_face_center": 0,
        "left_of_face_center": 0,
        "ambiguous": 0,
    }
    for u, real_clip in keymap.items():
        if u not in boxes:
            continue
        mid = boxes[u].get("middle")
        if mid == "no_eye_visible" or mid is None:
            continue
        if real_clip not in per_clip_face:
            continue
        face = per_clip_face[real_clip]
        fx, fy, fw, fh = face
        eye_cx = mid[0] + mid[2] / 2
        face_cx = fx + fw / 2
        offset = (eye_cx - face_cx) / fw

        if offset > MARGIN_THRESHOLD:
            side = "right_of_face_center"
            target = "left_eye"  # CORRECTED (was right_eye in Stage 1)
        elif offset < -MARGIN_THRESHOLD:
            side = "left_of_face_center"
            target = "right_eye"  # CORRECTED (was left_eye in Stage 1)
        else:
            side = "ambiguous"
            target = "higher_conf_per_clip_lock_rule_a"  # unchanged

        side_counts[side] += 1
        per_clip[real_clip] = {
            "uuid": u,
            "offset": offset,
            "side": side,
            "target_eye_keypoint": target,
        }

    # Verify: side counts should match Stage 1 (only target keypoint flips)
    original = json.loads(ORIGINAL_SIDE_ASSIGN_PATH.read_text())
    original_counts = original["side_counts"]
    side_match = side_counts == original_counts
    if not side_match:
        print(f"[ERROR] side counts diverged from Stage 1 — should be "
              f"identical (only target_eye_keypoint flips). "
              f"new={side_counts} stage1={original_counts}",
              file=sys.stderr)
        return 1

    def sha256(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    summary = {
        "tool": "tools/phase7_derive_side_assignment_corrected.py",
        "stage": "Stage 2 amendment v2 (post-empirical-falsification)",
        "correction_basis": (
            "RME 24/34 clips show flipped rule has higher IoU vs Phase 5 "
            "manual boxes (mean IoU 0.42 vs 0.17 locked). Phase 0 "
            "Wikimedia confirms horse-anatomical convention (231/287 "
            "frames have right_eye_x < left_eye_x in both-eyes-visible "
            "regime)."
        ),
        "margin_threshold": MARGIN_THRESHOLD,
        "mapping_corrected": {
            "right_of_face_center": "left_eye",
            "left_of_face_center": "right_eye",
            "ambiguous": "higher_conf_per_clip_lock_rule_a",
        },
        "mapping_original_stage1_falsified": {
            "right_of_face_center": "right_eye",
            "left_of_face_center": "left_eye",
            "ambiguous": "higher_conf_per_clip_lock_rule_a",
        },
        "inputs": {
            "eye_boxes_phase5a": {
                "path": str(EYE_BOXES_PATH.relative_to(POC_DIR)),
                "sha256": sha256(EYE_BOXES_PATH),
            },
            "eye_box_keymap_phase5": {
                "path": str(KEYMAP_PATH.relative_to(POC_DIR)),
                "sha256": sha256(KEYMAP_PATH),
            },
            "phase6b_position_param": {
                "path": str(POSITION_PARAM_PATH.relative_to(POC_DIR)),
                "sha256": sha256(POSITION_PARAM_PATH),
            },
            "stage1_side_assignment_falsified": {
                "path": str(ORIGINAL_SIDE_ASSIGN_PATH.relative_to(POC_DIR)),
                "sha256": sha256(ORIGINAL_SIDE_ASSIGN_PATH),
            },
        },
        "n_clips_total": len(per_clip),
        "side_counts": side_counts,
        "side_counts_match_stage1_originals": side_match,
        "ambiguous_clips_diagnostic_flag": (
            "Two of the 6 ambiguous-zone clips are V3_NEWLY_LOST in "
            "Phase 6(a) Set B: action_S5.mp4_5_ and "
            "background_S3.mp4_3_. Plus action_S5.mp4_5_ is part of "
            "the Sensitivity-1 swap pair with bg_S10_3_. The "
            "ambiguous-side rule (higher-conf per-clip lock) is "
            "unchanged from Stage 1; only the side-assigned mapping "
            "is corrected."
        ),
        "per_clip": per_clip,
    }

    OUT_PATH.write_text(json.dumps(summary, indent=2))

    print(f"Corrected side-assignment derivation complete: "
          f"{len(per_clip)}/{len(keymap)} clips")
    print(f"Side counts match Stage 1: {side_match}")
    print()
    for k, v in side_counts.items():
        print(f"  {k:25s} {v:3d}")
    print()
    print("Mapping corrected:")
    print("  right_of_face_center → left_eye  (was right_eye in Stage 1)")
    print("  left_of_face_center  → right_eye (was left_eye in Stage 1)")
    print("  ambiguous            → higher_conf_per_clip_lock (unchanged)")
    print()
    print(f"Wrote: {OUT_PATH.relative_to(POC_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
