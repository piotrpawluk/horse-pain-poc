#!/usr/bin/env python3
"""Phase 7 Stage 1 — derive per-clip eye-side assignment from Phase 5
manual annotations + Phase 6(b) cached face bboxes.

Per locked Stage 1 pre-registration §4
(`outputs/track_b_phase7_preregistration_stage1.md`):

For each of the 34 RME clips:
  offset = (eye_cx − face_cx) / face_width
where:
  eye_cx = manual middle-keyframe eye box centroid x from
           outputs/eye_boxes_phase5a.json
  face_cx, face_width = Phase 6(b) cached middle-frame face bbox from
           outputs/phase6b_position_param.json (per_clip array)

Margin = 0.05 (locked):
- offset > +0.05 → side = right_of_face_center;
  target DLC keypoint = right_eye
  (horse profile-left, anatomical RIGHT eye visible)
- offset < −0.05 → side = left_of_face_center;
  target DLC keypoint = left_eye
- |offset| ≤ 0.05 → side = ambiguous;
  target = higher_conf_per_frame_rule_a (per-clip lock at inference time)

Output: outputs/phase7_eye_side_assignment.json (hash-locked BEFORE any
DLC inference on the 34 RME clips).
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
OUT_PATH = POC_DIR / "outputs" / "phase7_eye_side_assignment.json"

MARGIN_THRESHOLD = 0.05  # locked per §4


def main() -> int:
    boxes = json.loads(EYE_BOXES_PATH.read_text())
    keymap = json.loads(KEYMAP_PATH.read_text())["keymap"]
    pos = json.loads(POSITION_PARAM_PATH.read_text())
    per_clip_face = {r["clip"]: r["face_bbox"]
                     for r in pos["per_clip"]}

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
            target = "right_eye"
        elif offset < -MARGIN_THRESHOLD:
            side = "left_of_face_center"
            target = "left_eye"
        else:
            side = "ambiguous"
            target = "higher_conf_per_clip_lock_rule_a"

        side_counts[side] += 1
        per_clip[real_clip] = {
            "uuid": u,
            "offset": offset,
            "side": side,
            "target_eye_keypoint": target,
        }

    # Reproducibility hashes of inputs.
    def sha256(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    summary = {
        "tool": "tools/phase7_derive_side_assignment.py",
        "stage1_pre_reg_section": "§4",
        "margin_threshold": MARGIN_THRESHOLD,
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
        },
        "n_clips_total": len(per_clip),
        "side_counts": side_counts,
        "ambiguous_clips_diagnostic_flag": (
            "Two of the ambiguous-zone clips are V3_NEWLY_LOST in "
            "Phase 6(a) Set B: action_S5.mp4_5_ and "
            "background_S3.mp4_3_. Plus action_S5.mp4_5_ is part of "
            "the Sensitivity-1 swap pair with bg_S10_3_. These clips' "
            "Phase 7 outcomes must be reported with explicit "
            "fallback_rule_applied notation. See §10 item 3."
        ),
        "per_clip": per_clip,
    }

    OUT_PATH.write_text(json.dumps(summary, indent=2))

    print(f"Side-assignment derivation complete: "
          f"{len(per_clip)}/{len(keymap)} clips")
    print()
    for k, v in side_counts.items():
        print(f"  {k:25s} {v:3d}")
    print()
    print("Ambiguous clips:")
    for c, d in sorted(per_clip.items(), key=lambda x: x[1]["offset"]):
        if d["side"] == "ambiguous":
            print(f"  {d['offset']:+.4f}  {c}")
    print()
    print(f"Wrote: {OUT_PATH.relative_to(POC_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
