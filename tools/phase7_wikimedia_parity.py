#!/usr/bin/env python3
"""Phase 7 Step 2 — Wikimedia parity check.

Per locked Stage 1 pre-reg §3:
- Compare new `tools/dlc_inference.py` output on the Wikimedia
  `data/sample_horse.mp4` against the Phase 0 reference JSON
  (`outputs/sample_horse_superanimal_quadruped_..._before_adapt.json`).
- Per-keypoint cosine similarity computed across frames.
- Gate: median per-keypoint cosine similarity ≥ 0.999 ideal,
  ≥ 0.99 acceptable WITH documented mechanism.
- Below 0.99 → HALT entire Phase 7 pending user investigation
  (per Stage 1 §3 clarification).

Per-frame matching: for each frame, take the individual with the
highest bbox_score (the primary horse detection). Keypoints with
confidence < 0.0 (i.e., not detected by DLC) are excluded from the
cosine-similarity computation per keypoint.

Outputs:
  outputs/phase7_dlc_wikimedia_parity.json — per-keypoint similarity,
                                              median, halt verdict.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

POC_DIR = Path(__file__).resolve().parent.parent
PHASE0_REF = POC_DIR / "outputs" / (
    "sample_horse_superanimal_quadruped_hrnet_w32_"
    "fasterrcnn_resnet50_fpn_v2_before_adapt.json"
)
PHASE7_NEW = POC_DIR / "outputs" / "phase7_wikimedia_parity_keypoints.json"
OUT_PATH = POC_DIR / "outputs" / "phase7_dlc_wikimedia_parity.json"

# Locked from Stage 1 pre-reg §2
HEAD_KEYPOINTS = {
    "nose": 0,
    "upper_jaw": 1,
    "lower_jaw": 2,
    "mouth_end_right": 3,
    "mouth_end_left": 4,
    "right_eye": 5,
    "right_earbase": 6,
    "right_earend": 7,
    "left_eye": 10,
    "left_earbase": 11,
    "left_earend": 12,
}
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
N_KEYPOINTS = 39
GATE_IDEAL = 0.999
GATE_ACCEPT = 0.99


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def primary_individual_keypoints(frame_entry):
    """Return keypoints of the highest-bbox-score individual in a
    single-frame entry. Returns None if no individual is valid
    (all bboxes are -1)."""
    bboxes = frame_entry.get("bboxes")
    bbox_scores = frame_entry.get("bbox_scores")
    bodyparts = frame_entry.get("bodyparts")
    if not bbox_scores or not bodyparts:
        return None
    valid = [(i, s) for i, s in enumerate(bbox_scores) if s >= 0]
    if not valid:
        return None
    best_i, _ = max(valid, key=lambda x: x[1])
    return bodyparts[best_i]


def cosine_similarity(v1, v2):
    """Cosine similarity between two equal-length vectors of floats."""
    if not v1 or not v2 or len(v1) != len(v2):
        return None
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0 or n2 == 0:
        return None
    return dot / (n1 * n2)


def main() -> int:
    if not PHASE0_REF.exists():
        print(f"[ERROR] Phase 0 reference not found: {PHASE0_REF}",
              file=sys.stderr)
        return 1
    if not PHASE7_NEW.exists():
        print(f"[ERROR] Phase 7 output not found: {PHASE7_NEW}",
              file=sys.stderr)
        return 1

    ref = json.loads(PHASE0_REF.read_text())
    new = json.loads(PHASE7_NEW.read_text())

    if len(ref) != len(new):
        print(f"[ERROR] frame count mismatch: ref={len(ref)} "
              f"new={len(new)}", file=sys.stderr)
        return 1

    # For each frame, get primary-individual keypoints from each.
    paired_frames = []
    n_skipped_no_indiv = 0
    for i, (rf, nf) in enumerate(zip(ref, new)):
        ref_kps = primary_individual_keypoints(rf)
        new_kps = primary_individual_keypoints(nf)
        if ref_kps is None or new_kps is None:
            n_skipped_no_indiv += 1
            continue
        paired_frames.append((i, ref_kps, new_kps))

    print(f"[parity] frames in ref/new: {len(ref)}/{len(new)}")
    print(f"[parity] paired frames (both have primary individual): "
          f"{len(paired_frames)}")
    print(f"[parity] skipped frames (missing individual): "
          f"{n_skipped_no_indiv}")
    print()

    # Per-keypoint cosine similarity across frames.
    per_kp = {}
    for kp_idx in range(N_KEYPOINTS):
        kp_name = ALL_KEYPOINT_NAMES[kp_idx]
        ref_xy_flat = []
        new_xy_flat = []
        n_used = 0
        for _, ref_kps, new_kps in paired_frames:
            r = ref_kps[kp_idx]
            n = new_kps[kp_idx]
            # Skip frames where either ref or new has zero-confidence
            # (DLC sometimes emits placeholder zeros)
            if r[2] <= 0 or n[2] <= 0:
                continue
            ref_xy_flat.extend([r[0], r[1]])
            new_xy_flat.extend([n[0], n[1]])
            n_used += 1
        cos = cosine_similarity(ref_xy_flat, new_xy_flat)
        per_kp[kp_name] = {
            "kp_idx": kp_idx,
            "n_frames_used": n_used,
            "cosine_sim": cos,
        }

    cos_values = [v["cosine_sim"] for v in per_kp.values()
                  if v["cosine_sim"] is not None]
    median_cos = statistics.median(cos_values) if cos_values else None
    min_cos = min(cos_values) if cos_values else None
    max_cos = max(cos_values) if cos_values else None
    n_below_ideal = sum(1 for v in cos_values if v < GATE_IDEAL)
    n_below_accept = sum(1 for v in cos_values if v < GATE_ACCEPT)

    if median_cos is None:
        verdict = "FAIL_NO_VALID_KEYPOINTS"
        next_action = ("No valid keypoints to compare. Investigate "
                       "Phase 7 output JSON structure.")
    elif median_cos >= GATE_IDEAL:
        verdict = "PASS_IDEAL"
        next_action = "Median ≥ 0.999. Proceed to Step 1.5."
    elif median_cos >= GATE_ACCEPT:
        verdict = "PASS_ACCEPT_WITH_DOC"
        next_action = (f"Median {median_cos:.6f} in [0.99, 0.999) "
                       "range. Acceptable IF deviation mechanism "
                       "is documented (e.g., MPS vs CPU floating-point "
                       "drift). User must confirm before Step 1.5.")
    else:
        verdict = "HALT_PHASE_7"
        next_action = (f"Median {median_cos:.6f} < 0.99. Per Stage 1 "
                       "pre-reg §3 (clarified): HALTS ENTIRE PHASE 7 "
                       "pending user investigation. Do NOT proceed "
                       "to Step 1.5 (DLC inference on RME clips).")

    summary = {
        "tool": "tools/phase7_wikimedia_parity.py",
        "stage1_pre_reg_section": "§3",
        "phase0_reference_path": str(PHASE0_REF.relative_to(POC_DIR)),
        "phase0_reference_sha256": sha256(PHASE0_REF),
        "phase7_new_path": str(PHASE7_NEW.relative_to(POC_DIR)),
        "phase7_new_sha256": sha256(PHASE7_NEW),
        "video_path": "data/sample_horse.mp4",
        "n_frames_ref": len(ref),
        "n_frames_new": len(new),
        "n_paired_frames": len(paired_frames),
        "n_skipped_no_individual": n_skipped_no_indiv,
        "gates": {
            "ideal": GATE_IDEAL,
            "acceptable": GATE_ACCEPT,
            "halt_below": GATE_ACCEPT,
        },
        "median_cosine_similarity_per_keypoint": median_cos,
        "min_cosine_similarity_per_keypoint": min_cos,
        "max_cosine_similarity_per_keypoint": max_cos,
        "n_keypoints_below_ideal_0_999": n_below_ideal,
        "n_keypoints_below_accept_0_99": n_below_accept,
        "verdict": verdict,
        "next_action": next_action,
        "per_keypoint": per_kp,
    }

    OUT_PATH.write_text(json.dumps(summary, indent=2))

    print(f"Per-keypoint cosine similarity (n_keypoints={len(cos_values)}):")
    print(f"  median: {median_cos}")
    print(f"  min:    {min_cos}")
    print(f"  max:    {max_cos}")
    print(f"  below 0.999: {n_below_ideal}")
    print(f"  below 0.99:  {n_below_accept}")
    print()
    print(f"VERDICT: {verdict}")
    print(f"NEXT: {next_action}")
    print()
    print(f"Wrote: {OUT_PATH.relative_to(POC_DIR)}")

    if verdict == "HALT_PHASE_7":
        return 2  # nonzero exit so any pipeline orchestrator halts
    return 0


if __name__ == "__main__":
    sys.exit(main())
