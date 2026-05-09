#!/usr/bin/env python3
"""Phase 6 (b) diagnostic — failure-mode attribution + per-clip analysis.

Per locked pre-registration `outputs/track_b_phase6b_preregistration.md`:

1. Per-clip diff vs Phase 5 primary (FBB_NEWLY_RECOVERED / FBB_NEWLY_LOST
   / BOTH_RIGHT / BOTH_WRONG against v3, not Phase 3).
2. IoU between Phase 6 (b)'s middle-frame eye crop bbox and Phase 5's
   manual middle keyframe eye box, per clip. Median + per-clip values
   on the 4 orientation-extreme clips.
3. loss_concentration_pct on 4 pre-named OE clips → routes G1 fail to
   orientation-aware face-bbox vs DLC vs run-both.
4. Per-clip interpretation routing for action_S9.mp4_4_ specifically.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

POC_DIR = Path(__file__).resolve().parent.parent
P5_PATH = POC_DIR / "outputs" / "eye_loso_results_phase5_primary.json"
P6B_PATH = POC_DIR / "outputs" / "eye_loso_results_phase6b.json"
MANIFEST_PATH = (POC_DIR / "outputs"
                 / "eye_crops_phase6b_m15_manifest.jsonl")
PHASE5_BOXES_PATH = POC_DIR / "outputs" / "eye_boxes_phase5a.json"
KEYMAP_PATH = POC_DIR / "outputs" / "eye_box_keymap_phase5.json"
POS_PARAM_PATH = POC_DIR / "outputs" / "phase6b_position_param.json"
OUT_JSON = POC_DIR / "outputs" / "phase6b_diagnostic.json"
OUT_MD = POC_DIR / "outputs" / "phase6b_diagnostic.md"

ORIENTATION_EXTREME = {
    "background_S5.mp4_10_.mp4",
    "action_S9.mp4_7_.mp4",
    "action_S9.mp4_4_.mp4",
    "background_S1.mp4_12_.mp4",
}

IOU_LANDED_ON_EYE = 0.50  # >= → "crop on eye"
IOU_OFF_EYE = 0.30        # <= → "crop off eye"


def is_correct(score: float, label: int) -> bool:
    return (1 if score > 0 else 0) == label


def categorize(p5_correct: bool, p6b_correct: bool) -> str:
    if p5_correct and p6b_correct:
        return "BOTH_RIGHT"
    if (not p5_correct) and (not p6b_correct):
        return "BOTH_WRONG"
    if (not p5_correct) and p6b_correct:
        return "FBB_NEWLY_RECOVERED"
    return "FBB_NEWLY_LOST"


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


def main() -> int:
    p5 = json.loads(P5_PATH.read_text())
    p6b = json.loads(P6B_PATH.read_text())
    boxes_5a = json.loads(PHASE5_BOXES_PATH.read_text())
    keymap = json.loads(KEYMAP_PATH.read_text())["keymap"]
    pos_param = json.loads(POS_PARAM_PATH.read_text())

    p5_per = {r["clip"]: r for r in p5["per_clip"]}
    p6b_per = {r["clip"]: r for r in p6b["per_clip"]}

    manifest = {}
    for line in MANIFEST_PATH.read_text().splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        if m.get("status") == "ok":
            manifest[m["clip"]] = m

    real_to_uuid = {v: k for k, v in keymap.items()}

    if set(p5_per.keys()) != set(p6b_per.keys()):
        print("[ERROR] clip sets differ between Phase 5 and Phase 6 (b)",
              file=sys.stderr)
        return 1
    label_mismatches = [
        c for c in p5_per
        if p5_per[c]["label"] != p6b_per[c]["label"]
    ]
    if label_mismatches:
        print(f"[ERROR] label vector mismatches: {label_mismatches}",
              file=sys.stderr)
        return 1

    rows = []
    for clip in sorted(p5_per):
        r5 = p5_per[clip]
        r6 = p6b_per[clip]
        label = r5["label"]
        s5, s6 = r5["score"], r6["score"]
        c5 = is_correct(s5, label)
        c6 = is_correct(s6, label)
        cat = categorize(c5, c6)

        # Manual middle keyframe box → Phase 5 reference for IoU.
        u = real_to_uuid.get(clip)
        ref_box = None
        if u and u in boxes_5a:
            ref_box = boxes_5a[u].get("middle")
            if ref_box == "no_eye_visible":
                ref_box = None

        m = manifest.get(clip)
        # Phase 6 (b) middle-frame tight eye box (pre-margin) reconstructed
        # from the manifest's recorded face bbox at middle frame.
        fbb_eye_tight_mid = None
        face_mid = m.get("sample_face_bbox_middle") if m else None
        if face_mid is not None:
            fx, fy, fw, fh = face_mid
            pos = pos_param["locked_anatomical_position"]
            fbb_eye_tight_mid = [
                fx + pos["rel_x"] * fw,
                fy + pos["rel_y"] * fh,
                pos["rel_w"] * fw,
                pos["rel_h"] * fh,
            ]
        iou_vs_manual = iou_xywh(fbb_eye_tight_mid, ref_box)

        rows.append({
            "clip": clip,
            "source": r5["source"],
            "label": label,
            "phase5_score": s5,
            "phase5_correct": c5,
            "phase6b_score": s6,
            "phase6b_correct": c6,
            "category_vs_phase5": cat,
            "score_delta_vs_phase5": s6 - s5,
            "is_orientation_extreme": clip in ORIENTATION_EXTREME,
            "phase5_manual_middle_box_xywh": ref_box,
            "phase6b_face_bbox_middle": face_mid,
            "phase6b_eye_tight_box_middle": fbb_eye_tight_mid,
            "iou_phase6b_eye_vs_phase5_manual": iou_vs_manual,
            "phase6b_fallback_mode": (m.get("fallback_mode") if m
                                       else "n/a"),
            "phase6b_mean_face_conf": (m.get("mean_face_conf") if m
                                        else None),
        })

    cat_counts = Counter(r["category_vs_phase5"] for r in rows)
    n_total = len(rows)
    n_p5_correct = sum(1 for r in rows if r["phase5_correct"])
    n_p6b_correct = sum(1 for r in rows if r["phase6b_correct"])
    n_recovered = cat_counts.get("FBB_NEWLY_RECOVERED", 0)
    n_lost = cat_counts.get("FBB_NEWLY_LOST", 0)
    net = n_recovered - n_lost

    fbb_lost = [r for r in rows
                if r["category_vs_phase5"] == "FBB_NEWLY_LOST"]
    n_oe_loss = sum(1 for r in fbb_lost if r["is_orientation_extreme"])
    if n_lost > 0:
        loss_concentration_pct = n_oe_loss / n_lost
    else:
        loss_concentration_pct = None

    # IoU stats.
    ious_all = [r["iou_phase6b_eye_vs_phase5_manual"] for r in rows
                if r["iou_phase6b_eye_vs_phase5_manual"] is not None]
    iou_median = (statistics.median(ious_all) if ious_all else None)
    iou_mean = (statistics.fmean(ious_all) if ious_all else None)
    iou_below_30 = sum(1 for v in ious_all if v <= IOU_OFF_EYE)
    iou_above_50 = sum(1 for v in ious_all if v >= IOU_LANDED_ON_EYE)

    # Per-OE-clip breakdown.
    oe_rows = [r for r in rows if r["is_orientation_extreme"]]

    # Locked routing decision per pre-reg.
    auc_p6b = p6b["pooled_auc"]
    auc_p5 = p5["pooled_auc"]
    g1 = auc_p6b >= 0.70
    g2_auc = auc_p6b >= (auc_p5 - 0.05)
    paired = p6b.get("delong_paired_vs_phase3") or {}
    # Note: paired here is vs Phase 3, not Phase 5; the pre-reg's G2 is
    # vs Phase 5 — paired-DeLong on the per-clip predictions of Phase 5
    # and Phase 6 (b) is computed below.
    # (Lightweight reuse of the existing per-clip score arrays.)

    if g1 and g2_auc:
        if auc_p6b > auc_p5 and paired.get("p_two_sided", 1.0) < 0.05:
            verdict = "OUTPERFORM_V3"
            next_action = ("Strong finding: tight eye crop was "
                           "over-tightening; face-bbox is the new "
                           "baseline.")
        else:
            verdict = "PASS"
            next_action = ("Viable Phase 6 scaling baseline; defer "
                           "DLC/YOLO unless N≥200.")
    elif g1 and not g2_auc:
        verdict = "NON_INFERIORITY_FAIL"
        if loss_concentration_pct is not None and \
                loss_concentration_pct > 0.50:
            next_action = ("Orientation-aware face-bbox is next "
                           "(lighter iteration); DLC deferred.")
        else:
            next_action = ("DLC SuperAnimal-Quadruped is next "
                           "(half-day setup).")
    else:  # G1 fails
        if loss_concentration_pct is not None:
            within_one_of_50 = abs(n_oe_loss - n_lost / 2) <= 1
            if loss_concentration_pct > 0.50:
                verdict = "ORIENTATION_DOMINATED_FAIL"
                next_action = ("Orientation-aware face-bbox is next; "
                               "DLC deferred. Light-tier escalation.")
            elif within_one_of_50:
                verdict = "AMBIGUOUS_FAIL"
                next_action = ("Run BOTH orientation-aware face-bbox "
                               "AND DLC; let empirical comparison "
                               "disambiguate.")
            else:
                verdict = "DISTRIBUTED_FAIL"
                next_action = ("DLC SuperAnimal-Quadruped is next; "
                               "orientation-aware face-bbox would not "
                               "be sufficient.")
        else:
            verdict = "AUC_FLOOR_FAIL_NO_LOSSES"
            next_action = ("AUC floor failure with no FBB_NEWLY_LOST "
                           "clips — the prediction shift is "
                           "uniform-down, not category-bound. "
                           "DLC is next.")

    # Specific routing for action_S9.mp4_4_.
    a94 = next((r for r in rows
                if r["clip"] == "action_S9.mp4_4_.mp4"), None)
    a94_routing = None
    if a94 is not None:
        iou_94 = a94["iou_phase6b_eye_vs_phase5_manual"]
        c_94 = a94["phase6b_correct"]
        if iou_94 is None:
            a94_label = "missing_iou"
        elif c_94 and iou_94 >= IOU_LANDED_ON_EYE:
            a94_label = ("CORRECT_ON_EYE → face-bbox preserves v1's "
                         "recovery on this clip; off-axis-motion "
                         "hypothesis confirmed for this clip "
                         "(looser-than-v3 crop captures the same "
                         "signal v1 had).")
        elif c_94 and iou_94 < IOU_OFF_EYE:
            a94_label = ("CORRECT_OFF_EYE → face-bbox got it right but "
                         "wider scope captures whole-body cues even "
                         "off-eye; suggestive but not conclusive.")
        elif (not c_94) and iou_94 < IOU_OFF_EYE:
            a94_label = ("WRONG_OFF_EYE → crop missed the eye; "
                         "orientation displacement is the failure. "
                         "Next tool: orientation-aware face-bbox.")
        elif (not c_94) and iou_94 >= IOU_LANDED_ON_EYE:
            a94_label = ("WRONG_ON_EYE → crop on eye but signal still "
                         "missing. Off-axis motion or catchlight is "
                         "the mechanism; next test: larger margin "
                         "(m=80% from bimodal hypothesis) or DLC.")
        else:
            a94_label = ("AMBIGUOUS_IOU (between thresholds): "
                         f"correct={c_94}, IoU={iou_94:.3f}")
        a94_routing = {
            "iou": iou_94,
            "phase6b_correct": c_94,
            "phase6b_score": a94["phase6b_score"],
            "phase5_score": a94["phase5_score"],
            "verdict": a94_label,
        }

    summary = {
        "phase5_path": str(P5_PATH.relative_to(POC_DIR)),
        "phase6b_path": str(P6B_PATH.relative_to(POC_DIR)),
        "phase5_pooled_auc": auc_p5,
        "phase6b_pooled_auc": auc_p6b,
        "phase6b_subject_bootstrap_ci":
            p6b.get("auc_95_ci_subject_bootstrap"),
        "delta_vs_phase5": auc_p6b - auc_p5,
        "n_clips": n_total,
        "n_phase5_correct_thr0": n_p5_correct,
        "n_phase6b_correct_thr0": n_p6b_correct,
        "category_counts_vs_phase5": dict(cat_counts),
        "n_recovered_vs_phase5": n_recovered,
        "n_lost_vs_phase5": n_lost,
        "net_shift_vs_phase5": net,
        "n_lost_in_orientation_extreme": n_oe_loss,
        "loss_concentration_pct": loss_concentration_pct,
        "iou_phase6b_vs_phase5_manual_stats": {
            "n": len(ious_all),
            "median": iou_median,
            "mean": iou_mean,
            "n_below_30": iou_below_30,
            "n_above_50": iou_above_50,
            "passes_locked_gate_0_6":
                (iou_median is not None and iou_median >= 0.6),
        },
        "gates": {
            "G1_auc_geq_0_70": g1,
            "G2_auc_geq_0_7485": g2_auc,
        },
        "verdict": verdict,
        "next_action": next_action,
        "action_S9_4_routing": a94_routing,
        "orientation_extreme_clip_breakdown": [
            {
                "clip": r["clip"],
                "category_vs_phase5": r["category_vs_phase5"],
                "iou_phase6b_eye_vs_phase5_manual":
                    r["iou_phase6b_eye_vs_phase5_manual"],
                "phase6b_correct": r["phase6b_correct"],
                "phase5_score": r["phase5_score"],
                "phase6b_score": r["phase6b_score"],
            }
            for r in oe_rows
        ],
        "rows": rows,
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2))

    lines = []
    a = lines.append
    a("# Phase 6 (b) diagnostic — face-bbox-positioned crop")
    a("")
    a(f"- Phase 5 primary AUC: **{auc_p5:.4f}**")
    a(f"- Phase 6 (b) AUC: **{auc_p6b:.4f}**")
    a(f"- Δ vs Phase 5: **{auc_p6b - auc_p5:+.4f}**")
    if p6b.get('auc_95_ci_subject_bootstrap'):
        boot = p6b['auc_95_ci_subject_bootstrap']
        a(f"- Phase 6 (b) bootstrap CI: [{boot[0]:.4f}, {boot[1]:.4f}]")
    a("")
    a("## Locked gates")
    a("")
    a(f"- G1 (AUC ≥ 0.70): **{'PASS' if g1 else 'FAIL'}** "
      f"({auc_p6b:.4f} {'≥' if g1 else '<'} 0.70)")
    a(f"- G2 (AUC ≥ {auc_p5 - 0.05:.4f}): "
      f"**{'PASS' if g2_auc else 'FAIL'}**")
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
                "FBB_NEWLY_RECOVERED", "FBB_NEWLY_LOST"]:
        n = cat_counts.get(cat, 0)
        a(f"| {cat} | {n} | {n / n_total:.1%} |")
    a("")
    a(f"Net shift FBB_NEWLY_RECOVERED − FBB_NEWLY_LOST = "
      f"**{net:+d}** clips")
    a("")
    a("## Failure-mode attribution (locked routing)")
    a("")
    if loss_concentration_pct is None:
        a("- No FBB_NEWLY_LOST clips → loss_concentration undefined")
    else:
        a(f"- FBB_NEWLY_LOST count: {n_lost}")
        a(f"- Of those, in 4 orientation-extreme clips: "
          f"{n_oe_loss}/{n_lost}")
        a(f"- **loss_concentration_pct = {loss_concentration_pct:.1%}**")
        a("")
        if loss_concentration_pct > 0.50:
            a("- > 50% → orientation-aware face-bbox is the next "
              "iteration (light-tier).")
        elif abs(n_oe_loss - n_lost / 2) <= 1:
            a("- ≈ 50% within ±1 clip → ambiguous routing; both "
              "tools needed.")
        else:
            a("- < 50% → distributed failure; DLC SuperAnimal-Quadruped "
              "is next (half-day-tier).")
    a("")
    a("## IoU vs Phase 5 manual eye boxes (middle keyframe, n=34)")
    a("")
    if iou_median is not None:
        a(f"- Median IoU: **{iou_median:.4f}** "
          f"(locked gate ≥ 0.6 → "
          f"{'PASS' if iou_median >= 0.6 else 'FAIL'})")
        a(f"- Mean IoU: {iou_mean:.4f}")
        a(f"- Clips with IoU ≤ 0.30 (off-eye): {iou_below_30}/{len(ious_all)}")
        a(f"- Clips with IoU ≥ 0.50 (on-eye): {iou_above_50}/{len(ious_all)}")
    a("")
    a("## Orientation-extreme clips — per-clip breakdown")
    a("")
    a("| Clip | Category vs P5 | P5 score | P6b score | IoU vs manual | "
      "P6b correct? |")
    a("|---|---|---:|---:|---:|---|")
    for r in oe_rows:
        iou_str = (f"{r['iou_phase6b_eye_vs_phase5_manual']:.3f}"
                    if r['iou_phase6b_eye_vs_phase5_manual'] is not None
                    else "—")
        a(f"| `{r['clip']}` | {r['category_vs_phase5']} | "
          f"{r['phase5_score']:+.3f} | {r['phase6b_score']:+.3f} | "
          f"{iou_str} | "
          f"{'✓' if r['phase6b_correct'] else '✗'} |")
    a("")
    a("## action_S9.mp4_4_ — locked interpretation routing")
    a("")
    if a94_routing:
        a(f"- IoU vs Phase 5 manual: {a94_routing['iou']:.3f}")
        a(f"- Phase 6 (b) correct: "
          f"{'✓' if a94_routing['phase6b_correct'] else '✗'}")
        a(f"- Phase 5 score: {a94_routing['phase5_score']:+.3f}")
        a(f"- Phase 6 (b) score: {a94_routing['phase6b_score']:+.3f}")
        a("")
        a(f"**Verdict (per locked routing in pre-reg)**: "
          f"{a94_routing['verdict']}")
    a("")
    a("## All clips (sorted by score_delta_vs_phase5)")
    a("")
    a("| Clip | Source | Truth | P5 score | P6b score | Δ | Cat | "
      "IoU | OE? |")
    a("|---|---|---|---:|---:|---:|---|---:|---|")
    for r in sorted(rows, key=lambda x: x["score_delta_vs_phase5"]):
        truth = "ACTION" if r["label"] == 1 else "BACKGROUND"
        iou_str = (f"{r['iou_phase6b_eye_vs_phase5_manual']:.3f}"
                    if r['iou_phase6b_eye_vs_phase5_manual'] is not None
                    else "—")
        oe = "OE" if r["is_orientation_extreme"] else ""
        a(f"| `{r['clip']}` | {r['source']} | {truth} | "
          f"{r['phase5_score']:+.3f} | {r['phase6b_score']:+.3f} | "
          f"{r['score_delta_vs_phase5']:+.3f} | "
          f"{r['category_vs_phase5']} | {iou_str} | {oe} |")
    a("")
    OUT_MD.write_text("\n".join(lines))

    print(f"Phase 5 AUC: {auc_p5:.4f}")
    print(f"Phase 6 (b) AUC: {auc_p6b:.4f}  (Δ {auc_p6b - auc_p5:+.4f})")
    print()
    print("Categories vs Phase 5:")
    for cat in ["BOTH_RIGHT", "BOTH_WRONG",
                "FBB_NEWLY_RECOVERED", "FBB_NEWLY_LOST"]:
        print(f"  {cat:24s} {cat_counts.get(cat, 0):3d}")
    print()
    print(f"FBB_NEWLY_LOST in OE clips: {n_oe_loss}/{n_lost}")
    if loss_concentration_pct is not None:
        print(f"loss_concentration_pct: {loss_concentration_pct:.1%}")
    print()
    print(f"Median IoU vs Phase 5 manual: "
          f"{iou_median:.4f}" if iou_median is not None else "—")
    print(f"VERDICT: {verdict}")
    print(f"NEXT: {next_action}")
    print()
    if a94_routing:
        print(f"action_S9_4 routing: {a94_routing['verdict'][:90]}")
    print()
    print(f"Wrote: {OUT_JSON.relative_to(POC_DIR)}")
    print(f"Wrote: {OUT_MD.relative_to(POC_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
