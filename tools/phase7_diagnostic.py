#!/usr/bin/env python3
"""Phase 7 diagnostic — DLC-specific dual-diagnostic routing.

Per locked Stage 1 pre-reg §9 + Stage 2 amendment:

1. Per-clip categories vs Phase 5 primary (DLC_NEWLY_RECOVERED /
   DLC_NEWLY_LOST / BOTH_RIGHT / BOTH_WRONG against v3 manual).
2. Per-clip categories vs Phase 6(b) face-bbox (DLC_BEATS_FBB /
   FBB_BEATS_DLC / BOTH_RIGHT / BOTH_WRONG).
3. Per-clip IoU between Phase 7 tight eye box (middle frame, before
   margin/square-pad) and Phase 5 manual middle keyframe box.
4. Routing diagnostics (mean keypoint conf × IoU on DLC_NEWLY_LOST
   clips, per Stage 1 §9 dual-axis).
5. Named-clip outcomes (4 OE clips + bg_S8_3 + action_S9_4 +
   action_S5_5 ↔ bg_S10_3 swap pair).
6. Locked routing decision per matrix.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

POC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_DIR / "tools"))
P5_PATH = POC_DIR / "outputs" / "eye_loso_results_phase5_primary.json"
P6B_PATH = POC_DIR / "outputs" / "eye_loso_results_phase6b.json"
P7_PATH = POC_DIR / "outputs" / "eye_loso_results_phase7.json"
KEYPOINTS_PATH = POC_DIR / "outputs" / "phase7_rme_dlc_keypoints.json"
SIDE_PATH = POC_DIR / "outputs" / "phase7_eye_side_assignment_corrected.json"
PHASE5_BOXES_PATH = POC_DIR / "outputs" / "eye_boxes_phase5a.json"
KEYMAP_PATH = POC_DIR / "outputs" / "eye_box_keymap_phase5.json"
POS_PARAM_PATH = POC_DIR / "outputs" / "phase6b_position_param.json"
MANIFEST_PATH = POC_DIR / "outputs" / "eye_crops_v4_dlc_manifest.jsonl"
OUT_JSON = POC_DIR / "outputs" / "phase7_diagnostic.json"
OUT_MD = POC_DIR / "outputs" / "phase7_diagnostic.md"

# Stage 1 §9 routing thresholds
CONF_BOTTLENECK = 0.5
CONF_HIGH = 0.7
IOU_OFF_EYE = 0.30
IOU_ON_EYE = 0.50
IOU_LOCKED_GATE = 0.6  # G3 reportable

# Stage 1 §10 named-clip set
ORIENTATION_EXTREME = {
    "background_S5.mp4_10_.mp4",
    "action_S9.mp4_7_.mp4",
    "action_S9.mp4_4_.mp4",
    "background_S1.mp4_12_.mp4",
}
SWAP_PAIR = {"action_S5.mp4_5_.mp4", "background_S10.mp4_3_.mp4"}

# Stage 1 §2 locked head set + eye keypoints
HEAD_KP_INDICES = [0, 1, 2, 3, 4, 5, 6, 10, 11]
EYE_KP_IDX = {"right_eye": 5, "left_eye": 10}
HEAD_KP_MIN = 6
CONF_THRESHOLD = 0.5  # Stage 2 §1
ABS_EYE_W_PX = 62
ABS_EYE_H_PX = 47


def is_correct(score, label):
    return (1 if score > 0 else 0) == label


def categorize(p_correct, q_correct, q_label_recov, q_label_lost):
    if p_correct and q_correct:
        return "BOTH_RIGHT"
    if (not p_correct) and (not q_correct):
        return "BOTH_WRONG"
    if (not p_correct) and q_correct:
        return q_label_recov
    return q_label_lost


def iou_xywh(b1, b2):
    if b1 is None or b2 is None:
        return None
    x1a, y1a, wa, ha = b1
    x1b, y1b, wb, hb = b2
    x2a, y2a = x1a + wa, y1a + ha
    x2b, y2b = x1b + wb, y1b + hb
    ix1 = max(x1a, x1b); iy1 = max(y1a, y1b)
    ix2 = min(x2a, x2b); iy2 = min(y2a, y2b)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = wa * ha + wb * hb - inter
    return inter / union if union > 0 else 0.0


def head_bbox_proxy(kps):
    """Same as crop tool. Returns (x, y, w, h) or None if <6 confident."""
    if kps is None:
        return None
    confident = [(kps[idx][0], kps[idx][1])
                 for idx in HEAD_KP_INDICES
                 if kps[idx][2] >= CONF_THRESHOLD]
    if len(confident) < HEAD_KP_MIN:
        return None
    xs = [p[0] for p in confident]
    ys = [p[1] for p in confident]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def main() -> int:
    p5 = json.loads(P5_PATH.read_text())
    p6b = json.loads(P6B_PATH.read_text())
    p7 = json.loads(P7_PATH.read_text())
    keypoints_data = json.loads(KEYPOINTS_PATH.read_text())
    side_data = json.loads(SIDE_PATH.read_text())["per_clip"]
    boxes_5a = json.loads(PHASE5_BOXES_PATH.read_text())
    keymap = json.loads(KEYMAP_PATH.read_text())["keymap"]
    pos_data = json.loads(POS_PARAM_PATH.read_text())
    rel_w = pos_data["locked_anatomical_position"]["rel_w"]
    rel_h = pos_data["locked_anatomical_position"]["rel_h"]

    real_to_uuid = {v: k for k, v in keymap.items()}

    p5_per = {r["clip"]: r for r in p5["per_clip"]}
    p6b_per = {r["clip"]: r for r in p6b["per_clip"]}
    p7_per = {r["clip"]: r for r in p7["per_clip"]}

    if set(p5_per.keys()) != set(p7_per.keys()):
        print("[ERROR] clip set mismatch P5 vs P7", file=sys.stderr)
        return 1
    label_mismatches = [c for c in p5_per
                        if p5_per[c]["label"] != p7_per[c]["label"]]
    if label_mismatches:
        print(f"[ERROR] label mismatches: {label_mismatches}", file=sys.stderr)
        return 1

    # Per-clip rows
    rows = []
    for clip in sorted(p5_per):
        r5 = p5_per[clip]
        r7 = p7_per[clip]
        r6 = p6b_per.get(clip)  # may be None if Phase 6(b) had different
        label = r5["label"]
        s5, s7 = r5["score"], r7["score"]
        c5 = is_correct(s5, label)
        c7 = is_correct(s7, label)
        cat_vs_p5 = categorize(c5, c7, "DLC_NEWLY_RECOVERED",
                                "DLC_NEWLY_LOST")
        cat_vs_p6b = None
        if r6 is not None:
            s6 = r6["score"]
            c6 = is_correct(s6, label)
            cat_vs_p6b = categorize(c6, c7, "DLC_BEATS_FBB",
                                     "FBB_BEATS_DLC")

        # Phase 5 manual middle keyframe box (reference for IoU)
        u = real_to_uuid.get(clip)
        ref_box = None
        if u and u in boxes_5a:
            ref_box = boxes_5a[u].get("middle")
            if ref_box == "no_eye_visible":
                ref_box = None

        # Phase 7 tight eye box at middle frame (reconstructed)
        kp_data = keypoints_data["per_clip"].get(clip)
        side_info = side_data.get(clip, {})
        side = side_info.get("side", "ambiguous")
        target_kp_name = side_info.get(
            "target_eye_keypoint", "higher_conf_per_clip_lock_rule_a"
        )
        p7_eye_box_mid = None
        kp_mid_conf = None
        head_proxy_mid = None
        if kp_data and kp_data.get("status") == "ok":
            kps_mid = kp_data["keypoints"][kp_data["n_frames"] // 2]
            if kps_mid is not None:
                # Resolve target eye name (per-clip lock for ambiguous)
                if side == "ambiguous":
                    # Compute majority-confident eye across confident frames
                    re_count = 0; le_count = 0
                    for kps in kp_data["keypoints"]:
                        if kps is None: continue
                        re_c = kps[EYE_KP_IDX["right_eye"]][2]
                        le_c = kps[EYE_KP_IDX["left_eye"]][2]
                        if max(re_c, le_c) < CONF_THRESHOLD: continue
                        if re_c >= le_c: re_count += 1
                        else: le_count += 1
                    target_kp_name_resolved = ("right_eye" if re_count >= le_count
                                                else "left_eye")
                else:
                    target_kp_name_resolved = target_kp_name
                kp_mid = kps_mid[EYE_KP_IDX[target_kp_name_resolved]]
                kp_mid_conf = kp_mid[2]
                head_proxy_mid = head_bbox_proxy(kps_mid)
                if head_proxy_mid is not None:
                    eye_w = rel_w * head_proxy_mid[2]
                    eye_h = rel_h * head_proxy_mid[3]
                else:
                    eye_w = ABS_EYE_W_PX
                    eye_h = ABS_EYE_H_PX
                p7_eye_box_mid = [kp_mid[0] - eye_w / 2,
                                   kp_mid[1] - eye_h / 2,
                                   eye_w, eye_h]

        iou_p7_vs_p5 = iou_xywh(p7_eye_box_mid, ref_box)

        # Mean target-eye-keypoint confidence across all frames in this clip
        if kp_data and kp_data.get("status") == "ok":
            confs = [c for c in kp_data["target_eye_confidence_per_frame"]
                     if c is not None]
            mean_kp_conf = (statistics.fmean(confs) if confs else None)
        else:
            mean_kp_conf = None

        rows.append({
            "clip": clip,
            "source": r5["source"],
            "label": label,
            "phase5_score": s5,
            "phase5_correct": c5,
            "phase6b_score": (r6["score"] if r6 else None),
            "phase6b_correct": (is_correct(r6["score"], label) if r6 else None),
            "phase7_score": s7,
            "phase7_correct": c7,
            "category_vs_phase5": cat_vs_p5,
            "category_vs_phase6b": cat_vs_p6b,
            "score_delta_vs_phase5": s7 - s5,
            "side": side,
            "target_eye_keypoint_name": target_kp_name,
            "phase5_manual_box_mid_xywh": ref_box,
            "phase7_tight_eye_box_mid_xywh": p7_eye_box_mid,
            "iou_phase7_vs_phase5_manual_mid": iou_p7_vs_p5,
            "mid_frame_target_eye_keypoint_conf": kp_mid_conf,
            "mean_target_eye_keypoint_conf_clip": mean_kp_conf,
            "is_orientation_extreme": clip in ORIENTATION_EXTREME,
            "is_swap_pair": clip in SWAP_PAIR,
        })

    # Aggregates
    cat_counts_p5 = Counter(r["category_vs_phase5"] for r in rows)
    cat_counts_p6b = Counter(r["category_vs_phase6b"] for r in rows
                              if r["category_vs_phase6b"] is not None)
    n_total = len(rows)

    # Locked routing diagnostics
    dlc_lost_rows = [r for r in rows
                     if r["category_vs_phase5"] == "DLC_NEWLY_LOST"]
    dlc_recov_rows = [r for r in rows
                      if r["category_vs_phase5"] == "DLC_NEWLY_RECOVERED"]
    n_lost = len(dlc_lost_rows)
    n_recov = len(dlc_recov_rows)

    confs_on_lost = [r["mean_target_eye_keypoint_conf_clip"]
                     for r in dlc_lost_rows
                     if r["mean_target_eye_keypoint_conf_clip"] is not None]
    ious_on_lost = [r["iou_phase7_vs_phase5_manual_mid"]
                    for r in dlc_lost_rows
                    if r["iou_phase7_vs_phase5_manual_mid"] is not None]
    mean_kp_conf_on_lost = (statistics.fmean(confs_on_lost)
                             if confs_on_lost else None)
    median_iou_on_lost = (statistics.median(ious_on_lost)
                           if ious_on_lost else None)

    # All-clip IoU stats (G3 reportable)
    all_ious = [r["iou_phase7_vs_phase5_manual_mid"] for r in rows
                if r["iou_phase7_vs_phase5_manual_mid"] is not None]
    median_iou_all = (statistics.median(all_ious) if all_ious else None)
    mean_iou_all = (statistics.fmean(all_ious) if all_ious else None)
    n_below_30 = sum(1 for v in all_ious if v <= IOU_OFF_EYE)
    n_above_50 = sum(1 for v in all_ious if v >= IOU_ON_EYE)

    # OE-concentration secondary diagnostic
    n_oe_in_lost = sum(1 for r in dlc_lost_rows
                       if r["is_orientation_extreme"])
    if n_lost > 0:
        oe_concentration_pct = n_oe_in_lost / n_lost
    else:
        oe_concentration_pct = None

    # Locked Stage 1 §9 routing
    auc_p5 = p5["pooled_auc"]
    auc_p7 = p7["pooled_auc"]
    g1 = auc_p7 >= 0.70
    g2_auc_floor = auc_p5 - 0.05  # 0.7485
    g2_auc = auc_p7 >= g2_auc_floor

    # Paired DeLong vs Phase 5 (used for OUTPERFORM determination per
    # Stage 1 §9 + G2 supportive)
    sys.path.insert(0, str(POC_DIR / "tools"))
    from eye_loso_lr_phase5 import delong_paired  # noqa: E402
    p5_per_aligned = {r["clip"]: r for r in p5["per_clip"]}
    p7_per_aligned = {r["clip"]: r for r in p7["per_clip"]}
    common_clips = sorted(set(p5_per_aligned) & set(p7_per_aligned))
    paired_y = [p5_per_aligned[c]["label"] for c in common_clips]
    scores_p5 = [p5_per_aligned[c]["score"] for c in common_clips]
    scores_p7 = [p7_per_aligned[c]["score"] for c in common_clips]
    paired_vs_p5 = delong_paired(paired_y, scores_p7, scores_p5)
    paired_p_vs_p5 = paired_vs_p5.get("p_two_sided")
    paired_z_vs_p5 = paired_vs_p5.get("z")
    paired_delta_vs_p5 = paired_vs_p5.get("delta")

    if g1 and g2_auc:
        if (auc_p7 > auc_p5 and paired_p_vs_p5 is not None and
                paired_p_vs_p5 < 0.05):
            verdict = "OUTPERFORM_PHASE_5"
            next_action = ("Strong finding. Investigate which clips DLC "
                           "handles better than human annotation.")
        elif auc_p7 > auc_p5:
            verdict = "OUTPERFORM_PHASE_5_AUC_ONLY"
            next_action = (f"AUC > Phase 5 (Δ +{auc_p7 - auc_p5:.4f}) "
                           f"but paired-DeLong p={paired_p_vs_p5:.4f} "
                           f"≥ 0.05; per G2 asymmetry (AUC load-bearing, "
                           f"paired p supportive), the AUC evidence "
                           f"governs. Phase 8 confirmation at higher N "
                           f"recommended.")
        else:
            verdict = "PASS"
            next_action = ("Viable Phase 8 scaling baseline; defer "
                           "custom YOLO unless N≥200 commits.")
    elif g1 and not g2_auc:
        # NON_INFERIORITY_FAIL — sub-route by mean conf and IoU on losses
        if (mean_kp_conf_on_lost is not None and
                median_iou_on_lost is not None):
            if (median_iou_on_lost < IOU_OFF_EYE or
                    mean_kp_conf_on_lost < CONF_BOTTLENECK):
                verdict = "NON_INFERIORITY_FAIL_LOCALIZATION"
                next_action = ("Geometry/side-assignment review; "
                               "possibly orientation-aware Phase 8a.")
            elif (median_iou_on_lost >= IOU_ON_EYE and
                    mean_kp_conf_on_lost >= CONF_HIGH):
                verdict = "NON_INFERIORITY_FAIL_SCOPE"
                next_action = ("Architecture/scope issue, not "
                               "localization. Larger margin (m=80%) "
                               "test or DLC + different feature extractor.")
            else:
                verdict = "NON_INFERIORITY_FAIL_AMBIGUOUS"
                next_action = ("Per-clip diagnostic needed; consider "
                               "running both YOLO and orientation-aware.")
        else:
            verdict = "NON_INFERIORITY_FAIL_NO_LOSSES"
            next_action = ("AUC drop without DLC_NEWLY_LOST clips — "
                           "unusual. Investigate prediction shift.")
    else:
        # G1 fails — apply locked dual-diagnostic routing
        if mean_kp_conf_on_lost is None:
            verdict = "G1_FAIL_NO_DIAGNOSTIC_DATA"
            next_action = "Investigate; routing data unavailable."
        elif mean_kp_conf_on_lost < CONF_BOTTLENECK:
            verdict = "CONFIDENCE_BOTTLENECK_FAIL"
            next_action = ("Custom YOLO horse-eye detector (Phase 8b, "
                           "2-3 day investment). DLC sees 'no eye' "
                           "reliably enough to need an eye-specific "
                           "detector.")
        elif (mean_kp_conf_on_lost >= CONF_HIGH and
                median_iou_on_lost is not None and
                median_iou_on_lost < IOU_OFF_EYE):
            verdict = "CONFIDENT_MISPLACEMENT_FAIL"
            next_action = ("Geometry/side-assignment review first; "
                           "if that doesn't recover, orientation-aware "
                           "Phase 8a.")
        elif (mean_kp_conf_on_lost >= CONF_HIGH and
                median_iou_on_lost is not None and
                median_iou_on_lost >= IOU_ON_EYE):
            verdict = "ON_EYE_WRONG_PREDICTION_FAIL"
            next_action = ("Architecture/scope issue: localization "
                           "works but V-JEPA-2 + LR signal isn't "
                           "extractable from this crop scale. Larger "
                           "margin test, or different feature extractor. "
                           "NOT custom YOLO (which would also crop tightly).")
        else:
            verdict = "AMBIGUOUS_FAIL"
            next_action = ("Run BOTH custom YOLO AND geometry review; "
                           "let empirical comparison disambiguate.")

    # Named-clip outcomes
    oe_breakdown = []
    for clip in sorted(ORIENTATION_EXTREME):
        r = next((row for row in rows if row["clip"] == clip), None)
        if r is None: continue
        oe_breakdown.append({
            "clip": clip,
            "category_vs_phase5": r["category_vs_phase5"],
            "category_vs_phase6b": r["category_vs_phase6b"],
            "phase5_score": r["phase5_score"],
            "phase6b_score": r["phase6b_score"],
            "phase7_score": r["phase7_score"],
            "iou_p7_vs_p5_manual": r["iou_phase7_vs_phase5_manual_mid"],
            "mid_kp_conf": r["mid_frame_target_eye_keypoint_conf"],
        })

    # action_S9_4 specific routing (per Stage 1 §10 hypothesis test)
    a94 = next((r for r in rows if r["clip"] == "action_S9.mp4_4_.mp4"),
               None)
    a94_routing = None
    if a94 is not None:
        iou = a94["iou_phase7_vs_phase5_manual_mid"]
        c7 = a94["phase7_correct"]
        if iou is None:
            label_a94 = "missing_iou"
        elif c7 and iou >= IOU_ON_EYE:
            label_a94 = ("CORRECT_ON_EYE → DLC localized correctly AND "
                         "preserved Phase 5 prediction. Best outcome.")
        elif c7 and iou < IOU_OFF_EYE:
            label_a94 = ("CORRECT_OFF_EYE → got it right despite DLC "
                         "missing the eye; whole-body cues again.")
        elif (not c7) and iou < IOU_OFF_EYE:
            label_a94 = ("WRONG_OFF_EYE → DLC missed eye on this clip; "
                         "orientation displacement persists. Custom "
                         "YOLO or orientation-aware needed.")
        elif (not c7) and iou >= IOU_ON_EYE:
            label_a94 = ("WRONG_ON_EYE → DLC localized correctly but "
                         "signal still missing. Off-axis motion or "
                         "catchlight is the mechanism; larger margin "
                         "(m=80%) or different feature-extractor next.")
        else:
            label_a94 = (f"AMBIGUOUS (correct={c7}, IoU={iou:.3f})")
        a94_routing = {
            "iou": iou,
            "phase7_correct": c7,
            "phase7_score": a94["phase7_score"],
            "phase5_score": a94["phase5_score"],
            "phase6b_score": a94["phase6b_score"],
            "mid_kp_conf": a94["mid_frame_target_eye_keypoint_conf"],
            "verdict": label_a94,
        }

    # Swap-pair outcomes
    swap_breakdown = []
    for clip in sorted(SWAP_PAIR):
        r = next((row for row in rows if row["clip"] == clip), None)
        if r is None: continue
        swap_breakdown.append({
            "clip": clip,
            "category_vs_phase5": r["category_vs_phase5"],
            "phase5_score": r["phase5_score"],
            "phase7_score": r["phase7_score"],
            "iou_p7_vs_p5": r["iou_phase7_vs_phase5_manual_mid"],
        })

    summary = {
        "tool": "tools/phase7_diagnostic.py",
        "phase5_pooled_auc": auc_p5,
        "phase6b_pooled_auc": p6b["pooled_auc"],
        "phase7_pooled_auc": auc_p7,
        "phase7_subject_bootstrap_ci":
            p7.get("auc_95_ci_subject_bootstrap"),
        "delta_vs_phase5": auc_p7 - auc_p5,
        "delta_vs_phase6b": auc_p7 - p6b["pooled_auc"],
        "n_clips": n_total,
        "category_counts_vs_phase5": dict(cat_counts_p5),
        "category_counts_vs_phase6b": dict(cat_counts_p6b),
        "n_dlc_newly_recovered": n_recov,
        "n_dlc_newly_lost": n_lost,
        "net_shift_vs_phase5": n_recov - n_lost,
        "n_orientation_extreme_in_lost": n_oe_in_lost,
        "oe_concentration_pct_secondary": oe_concentration_pct,
        "mean_target_eye_keypoint_conf_on_lost_clips": mean_kp_conf_on_lost,
        "median_iou_on_lost_clips": median_iou_on_lost,
        "iou_phase7_vs_phase5_manual_stats_all_clips": {
            "n": len(all_ious),
            "median": median_iou_all,
            "mean": mean_iou_all,
            "n_below_30": n_below_30,
            "n_above_50": n_above_50,
            "g3_locked_gate_0_6_pass":
                (median_iou_all is not None and median_iou_all >= IOU_LOCKED_GATE),
        },
        "gates": {
            "G1_auc_geq_0_70": g1,
            "G2_auc_geq_0_7485_load_bearing": g2_auc,
            "G2_paired_delong_vs_phase5_supportive": {
                "delta": paired_delta_vs_p5,
                "z": paired_z_vs_p5,
                "p_two_sided": paired_p_vs_p5,
                "supportive_pass": (paired_p_vs_p5 is not None and
                                     paired_p_vs_p5 < 0.05),
            },
            "G3_median_iou_geq_0_6_reportable":
                (median_iou_all is not None and median_iou_all >= IOU_LOCKED_GATE),
        },
        "verdict": verdict,
        "next_action": next_action,
        "action_S9_4_routing": a94_routing,
        "orientation_extreme_breakdown": oe_breakdown,
        "swap_pair_breakdown": swap_breakdown,
        "rows": rows,
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2))

    # Markdown writeup
    L = []
    a = L.append
    a("# Phase 7 diagnostic — DLC keypoint-anchored crop")
    a("")
    a(f"- Phase 5 primary AUC: **{auc_p5:.4f}**")
    a(f"- Phase 6(b) face-bbox AUC: **{p6b['pooled_auc']:.4f}**")
    a(f"- Phase 7 DLC AUC: **{auc_p7:.4f}**")
    a(f"- Δ vs Phase 5: **{auc_p7 - auc_p5:+.4f}**")
    a(f"- Δ vs Phase 6(b): **{auc_p7 - p6b['pooled_auc']:+.4f}**")
    if p7.get("auc_95_ci_subject_bootstrap"):
        boot = p7["auc_95_ci_subject_bootstrap"]
        a(f"- Phase 7 bootstrap CI: [{boot[0]:.4f}, {boot[1]:.4f}]")
    a("")
    a("## Locked gates")
    a("")
    a(f"- **G1** (AUC ≥ 0.70): {'PASS' if g1 else 'FAIL'} ({auc_p7:.4f})")
    a(f"- **G2 load-bearing** (AUC ≥ 0.7485): "
      f"{'PASS' if g2_auc else 'FAIL'}")
    g3_pass = (median_iou_all is not None and
                median_iou_all >= IOU_LOCKED_GATE)
    g3_str = (f"{median_iou_all:.4f}"
               if median_iou_all is not None else "—")
    a(f"- **G3 reportable** (median IoU ≥ 0.6): "
      f"{'PASS' if g3_pass else 'FAIL'} ({g3_str})")
    a("")
    a(f"**Verdict: {verdict}**")
    a("")
    a(f"**Next action**: {next_action}")
    a("")
    a("## Per-clip categories vs Phase 5 primary")
    a("")
    a("| Category | n | Share |")
    a("|---|---:|---:|")
    for cat in ["BOTH_RIGHT", "BOTH_WRONG",
                "DLC_NEWLY_RECOVERED", "DLC_NEWLY_LOST"]:
        n = cat_counts_p5.get(cat, 0)
        a(f"| {cat} | {n} | {n/n_total:.1%} |")
    a("")
    a(f"Net shift DLC_NEWLY_RECOVERED − DLC_NEWLY_LOST = "
      f"**{n_recov - n_lost:+d}** clips")
    a("")
    a("## Per-clip categories vs Phase 6(b) face-bbox")
    a("")
    a("| Category | n | Share |")
    a("|---|---:|---:|")
    for cat in ["BOTH_RIGHT", "BOTH_WRONG",
                "DLC_BEATS_FBB", "FBB_BEATS_DLC"]:
        n = cat_counts_p6b.get(cat, 0)
        a(f"| {cat} | {n} | {n/n_total:.1%} |")
    a("")
    a("## Routing diagnostic (Stage 1 §9 dual-axis)")
    a("")
    if mean_kp_conf_on_lost is not None:
        a(f"- DLC_NEWLY_LOST count: {n_lost}")
        a(f"- Mean target-eye-keypoint confidence on lost clips: "
          f"**{mean_kp_conf_on_lost:.4f}** "
          f"(thresholds: <{CONF_BOTTLENECK}=bottleneck, "
          f"≥{CONF_HIGH}=high)")
        if median_iou_on_lost is not None:
            a(f"- Median IoU vs Phase 5 manual on lost clips: "
              f"**{median_iou_on_lost:.4f}** "
              f"(thresholds: ≤{IOU_OFF_EYE}=off-eye, "
              f"≥{IOU_ON_EYE}=on-eye)")
        else:
            a("- Median IoU on lost clips: N/A")
    else:
        a("- No DLC_NEWLY_LOST clips → routing diagnostic N/A")
    a("")
    if oe_concentration_pct is not None:
        a("### Secondary diagnostic (informational, NOT routing)")
        a("")
        a(f"- OE-concentration (n_OE_in_lost / n_lost): "
          f"{n_oe_in_lost}/{n_lost} = {oe_concentration_pct:.1%}")
        a("  (Phase 6(b) was 6.2%; this is informational for whether "
          "DLC handles OE clips face-bbox failed on)")
        a("")
    a("## IoU vs Phase 5 manual eye boxes (all 34 clips)")
    a("")
    if median_iou_all is not None:
        a(f"- Median IoU: **{median_iou_all:.4f}** (G3 gate ≥ 0.6: "
          f"{'PASS' if median_iou_all >= IOU_LOCKED_GATE else 'FAIL'})")
        a(f"- Mean IoU: {mean_iou_all:.4f}")
        a(f"- Clips with IoU ≤ 0.30 (off-eye): {n_below_30}/{len(all_ious)}")
        a(f"- Clips with IoU ≥ 0.50 (on-eye): {n_above_50}/{len(all_ious)}")
    a("")
    a("## Orientation-extreme clips — per-clip breakdown")
    a("")
    a("| Clip | vs P5 | vs P6b | P5 score | P6b score | P7 score | "
      "IoU vs manual | mid kp conf |")
    a("|---|---|---|---:|---:|---:|---:|---:|")
    for r in oe_breakdown:
        iou_str = (f"{r['iou_p7_vs_p5_manual']:.3f}"
                    if r['iou_p7_vs_p5_manual'] is not None else "—")
        kc_str = (f"{r['mid_kp_conf']:.3f}"
                   if r['mid_kp_conf'] is not None else "—")
        p6b_str = (f"{r['phase6b_score']:+.3f}"
                    if r['phase6b_score'] is not None else "—")
        a(f"| `{r['clip']}` | {r['category_vs_phase5']} | "
          f"{r['category_vs_phase6b'] or '—'} | "
          f"{r['phase5_score']:+.3f} | {p6b_str} | "
          f"{r['phase7_score']:+.3f} | {iou_str} | {kc_str} |")
    a("")
    a("## Swap pair (action_S5_5 ↔ bg_S10_3, ambiguous-side fallback)")
    a("")
    a("| Clip | vs P5 | P5 score | P7 score | IoU vs manual |")
    a("|---|---|---:|---:|---:|")
    for r in swap_breakdown:
        iou_str = (f"{r['iou_p7_vs_p5']:.3f}"
                    if r['iou_p7_vs_p5'] is not None else "—")
        a(f"| `{r['clip']}` | {r['category_vs_phase5']} | "
          f"{r['phase5_score']:+.3f} | {r['phase7_score']:+.3f} | "
          f"{iou_str} |")
    a("")
    a("## action_S9.mp4_4_ (most informative single clip per Stage 1 §10)")
    a("")
    if a94_routing:
        a(f"- IoU vs Phase 5 manual: {a94_routing['iou']:.3f}"
          if a94_routing['iou'] is not None else "- IoU: N/A")
        a(f"- Mid-frame keypoint conf: "
          f"{a94_routing['mid_kp_conf']:.3f}"
          if a94_routing['mid_kp_conf'] is not None else "—")
        a(f"- Phase 5 score: {a94_routing['phase5_score']:+.3f}")
        p6b_score = a94_routing.get('phase6b_score')
        if p6b_score is not None:
            a(f"- Phase 6(b) score: {p6b_score:+.3f}")
        a(f"- Phase 7 score: {a94_routing['phase7_score']:+.3f}")
        a(f"- Phase 7 correct: "
          f"{'✓' if a94_routing['phase7_correct'] else '✗'}")
        a("")
        a(f"**Verdict (per locked routing in Stage 1 §10)**: "
          f"{a94_routing['verdict']}")
    a("")
    a("## All clips — per-clip table")
    a("")
    a("| Clip | Source | Truth | P5 | P6b | P7 | Δ vs P5 | "
      "vs P5 | vs P6b | IoU | mid kp conf |")
    a("|---|---|---|---:|---:|---:|---:|---|---|---:|---:|")
    for r in sorted(rows, key=lambda x: x["score_delta_vs_phase5"]):
        truth = "ACTION" if r["label"] == 1 else "BACKGROUND"
        iou_str = (f"{r['iou_phase7_vs_phase5_manual_mid']:.3f}"
                    if r['iou_phase7_vs_phase5_manual_mid'] is not None
                    else "—")
        kc_str = (f"{r['mid_frame_target_eye_keypoint_conf']:.3f}"
                   if r['mid_frame_target_eye_keypoint_conf'] is not None
                   else "—")
        p6b_str = (f"{r['phase6b_score']:+.3f}"
                    if r['phase6b_score'] is not None else "—")
        a(f"| `{r['clip']}` | {r['source']} | {truth} | "
          f"{r['phase5_score']:+.3f} | {p6b_str} | "
          f"{r['phase7_score']:+.3f} | "
          f"{r['score_delta_vs_phase5']:+.3f} | "
          f"{r['category_vs_phase5']} | "
          f"{r['category_vs_phase6b'] or '—'} | "
          f"{iou_str} | {kc_str} |")
    a("")

    OUT_MD.write_text("\n".join(L))

    # Console
    print(f"Phase 5 AUC: {auc_p5:.4f}")
    print(f"Phase 6(b) AUC: {p6b['pooled_auc']:.4f}")
    print(f"Phase 7 AUC: {auc_p7:.4f}  (Δ vs P5 {auc_p7 - auc_p5:+.4f}, "
          f"Δ vs P6b {auc_p7 - p6b['pooled_auc']:+.4f})")
    print()
    print("Categories vs Phase 5:")
    for cat in ["BOTH_RIGHT", "BOTH_WRONG",
                "DLC_NEWLY_RECOVERED", "DLC_NEWLY_LOST"]:
        print(f"  {cat:24s} {cat_counts_p5.get(cat, 0):3d}")
    print()
    print("Categories vs Phase 6(b):")
    for cat in ["BOTH_RIGHT", "BOTH_WRONG",
                "DLC_BEATS_FBB", "FBB_BEATS_DLC"]:
        print(f"  {cat:24s} {cat_counts_p6b.get(cat, 0):3d}")
    print()
    if mean_kp_conf_on_lost is not None:
        print(f"Mean KP conf on DLC_NEWLY_LOST clips: "
              f"{mean_kp_conf_on_lost:.4f}")
    if median_iou_on_lost is not None:
        print(f"Median IoU on DLC_NEWLY_LOST clips: "
              f"{median_iou_on_lost:.4f}")
    print(f"OE concentration (secondary): "
          f"{oe_concentration_pct:.1%}" if oe_concentration_pct is not None
          else "OE concentration: N/A")
    print()
    print(f"All-clip median IoU vs P5 manual: {median_iou_all:.4f}")
    print(f"Clips off-eye (IoU≤0.30): {n_below_30}/{len(all_ious)}")
    print(f"Clips on-eye (IoU≥0.50): {n_above_50}/{len(all_ious)}")
    print()
    print(f"VERDICT: {verdict}")
    print(f"NEXT: {next_action}")
    print()
    if a94_routing:
        verdict_str = a94_routing.get('verdict', '')
        print(f"action_S9_4: {verdict_str[:90]}")
    print()
    print(f"Wrote: {OUT_JSON.relative_to(POC_DIR)}")
    print(f"Wrote: {OUT_MD.relative_to(POC_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
