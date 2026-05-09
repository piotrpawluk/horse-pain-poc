#!/usr/bin/env python3
"""Phase 6 instrumentation (a) — per-clip Phase 3 vs Phase 5 prediction diff.

Pre-registered in `docs/phase5_audit.md`:

  > Pre-registered Phase 6 instrumentation: per-clip diff between Phase 3
  > and Phase 5 primary predictions. Which clips did Phase 3 get right
  > that v3 lost? Which clips did v3 newly recover? The mechanism behind
  > the prediction-shift is the next-tier diagnostic; Phase 6 carries it.

Methodology (locked here):

- Inputs: outputs/eye_loso_results.json (Phase 3, post per-clip alignment
  fix, pooled AUC 0.6813) and outputs/eye_loso_results_phase5_primary.json
  (Phase 5 primary, v3 m=15, pooled AUC 0.7985).
- Score source: per_clip[*].score from each JSON. The score reflects the
  RidgeClassifier decision_function output from the LOSO run.
- Reference label sets (3 variants computed):
    * "set_b_verification" (PRIMARY): outputs/eye_verification_clips.txt —
      Piotr's morning verification, the labels Phase 3 + Phase 5 primary
      were trained AND evaluated against. AUC 0.6813 / 0.7985 are computed
      against this set. This is the directly-decomposable variant.
    * "set_a_rme" (supplementary): RME filename taxonomy
      (action_*/background_*). The dataset publisher's labels for a
      different (ear-motion) behavior; not what Phase 5 primary trained
      on. Useful as a reference-point ablation.
    * "set_c_tightened": outputs/eye_relabel_unmasked.txt — Piotr's
      tightened-rubric relabel. Phase 5 sens_rubric's training labels.
      Useful as a reference-point ablation; allows label-set sensitivity
      check on the prediction-shift pattern.
- Correctness rule: RidgeClassifier decision_function threshold = 0.
  predicted_positive = (score > 0). Tie at score == 0 (will not occur
  numerically, but defined): treat as predicted_negative
  (matches sklearn's RidgeClassifier.predict convention which rounds
  ties down).
- Categories: BOTH_RIGHT, BOTH_WRONG, V3_NEWLY_RECOVERED
  (Phase 3 wrong, Phase 5 right), V3_NEWLY_LOST (Phase 3 right,
  Phase 5 wrong).

Outputs:
  outputs/phase6a_per_clip_prediction_diff.json  (structured, all 3
                                                   variants)
  outputs/phase6a_per_clip_prediction_diff.md    (human-readable, primary
                                                   + per-variant summary)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

POC_DIR = Path(__file__).resolve().parent.parent
PHASE3_PATH = POC_DIR / "outputs" / "eye_loso_results.json"
PHASE5_PATH = POC_DIR / "outputs" / "eye_loso_results_phase5_primary.json"
VERIF_PATH = POC_DIR / "outputs" / "eye_verification_clips.txt"
TIGHT_PATH = POC_DIR / "outputs" / "eye_relabel_unmasked.txt"
OUT_JSON = POC_DIR / "outputs" / "phase6a_per_clip_prediction_diff.json"
OUT_MD = POC_DIR / "outputs" / "phase6a_per_clip_prediction_diff.md"


def load_per_clip(path: Path) -> dict:
    """Load LOSO result JSON, return {clip: {label, score, source}}."""
    d = json.loads(path.read_text())
    return {row["clip"]: row for row in d["per_clip"]}, d


def parse_label_file(path: Path) -> dict[str, int]:
    """Parse `<path>/<basename> - <observation> - <ACTION|BACKGROUND>` lines.
    Returns {basename: 1_if_action_else_0}."""
    out = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(" - ")]
        if len(parts) < 2:
            continue
        verdict = parts[-1].upper()
        if verdict not in ("ACTION", "BACKGROUND"):
            continue
        path_or_name = parts[0]
        basename = path_or_name.rsplit("/", 1)[-1]
        out[basename] = 1 if verdict == "ACTION" else 0
    return out


def rme_filename_label(clip_basename: str) -> int:
    """RME taxonomy from filename: action_* → 1, background_* → 0."""
    return 1 if clip_basename.startswith("action_") else 0


def is_correct(score: float, label: int) -> bool:
    """RidgeClassifier threshold=0 correctness."""
    pred = 1 if score > 0 else 0
    return pred == label


def categorize(p3_correct: bool, p5_correct: bool) -> str:
    if p3_correct and p5_correct:
        return "BOTH_RIGHT"
    if (not p3_correct) and (not p5_correct):
        return "BOTH_WRONG"
    if (not p3_correct) and p5_correct:
        return "V3_NEWLY_RECOVERED"
    return "V3_NEWLY_LOST"  # p3 right, p5 wrong


def compute_variant(p3_clips, p5_clips, label_fn, variant_name):
    """Compute per-clip diff under a specific label-source.

    Args:
      p3_clips: {clip: {score, source, label}} from Phase 3 LOSO output.
      p5_clips: same structure for Phase 5 primary.
      label_fn: callable(clip_basename) -> int (1=action, 0=background) or None
                if the label is unknown for that clip under this set.
      variant_name: str identifier for the label set.
    """
    rows = []
    for clip in sorted(p3_clips):
        r3 = p3_clips[clip]
        r5 = p5_clips[clip]
        ref_label = label_fn(clip)
        if ref_label is None:
            continue
        s3, s5 = r3["score"], r5["score"]
        c3 = is_correct(s3, ref_label)
        c5 = is_correct(s5, ref_label)
        rows.append({
            "clip": clip,
            "source": r3["source"],
            "ref_label": ref_label,
            "phase3_label_in_loso": r3["label"],
            "phase3_score": s3,
            "phase3_pred": 1 if s3 > 0 else 0,
            "phase3_correct": c3,
            "phase5_label_in_loso": r5["label"],
            "phase5_score": s5,
            "phase5_pred": 1 if s5 > 0 else 0,
            "phase5_correct": c5,
            "category": categorize(c3, c5),
            "score_delta": s5 - s3,
        })
    cat_counts = Counter(r["category"] for r in rows)
    n_total = len(rows)
    n_p3_correct = sum(1 for r in rows if r["phase3_correct"])
    n_p5_correct = sum(1 for r in rows if r["phase5_correct"])
    n_recovered = cat_counts["V3_NEWLY_RECOVERED"]
    n_lost = cat_counts["V3_NEWLY_LOST"]
    if n_lost == 0:
        recovery_lost_ratio = "inf" if n_recovered > 0 else "0/0"
    else:
        recovery_lost_ratio = round(n_recovered / n_lost, 3)
    return {
        "variant": variant_name,
        "n_clips": n_total,
        "n_phase3_correct_thr0": n_p3_correct,
        "n_phase5_correct_thr0": n_p5_correct,
        "category_counts": dict(cat_counts),
        "n_recovered": n_recovered,
        "n_lost": n_lost,
        "net_shift_recovered_minus_lost": n_recovered - n_lost,
        "recovery_lost_ratio": recovery_lost_ratio,
        "rows": rows,
    }


def main() -> int:
    p3_clips, p3_meta = load_per_clip(PHASE3_PATH)
    p5_clips, p5_meta = load_per_clip(PHASE5_PATH)

    p3_set = set(p3_clips.keys())
    p5_set = set(p5_clips.keys())
    if p3_set != p5_set:
        only_p3 = p3_set - p5_set
        only_p5 = p5_set - p3_set
        print(f"[ERROR] clip sets differ. P3 only: {only_p3} | "
              f"P5 only: {only_p5}", file=sys.stderr)
        return 1

    label_mismatches = []
    for c in p3_set:
        if p3_clips[c]["label"] != p5_clips[c]["label"]:
            label_mismatches.append(
                (c, p3_clips[c]["label"], p5_clips[c]["label"])
            )
    if label_mismatches:
        print("[ERROR] in-LOSO label vectors differ between Phase 3 and "
              "Phase 5:", file=sys.stderr)
        for row in label_mismatches:
            print(f"  {row}", file=sys.stderr)
        return 1

    set_b = parse_label_file(VERIF_PATH)
    set_c = parse_label_file(TIGHT_PATH)

    set_a_fn = rme_filename_label
    set_b_fn = lambda clip: set_b.get(clip)
    set_c_fn = lambda clip: set_c.get(clip)

    primary = compute_variant(p3_clips, p5_clips, set_b_fn,
                              "set_b_verification (Phase 5 primary's "
                              "training labels — directly decomposable)")
    suppl_a = compute_variant(p3_clips, p5_clips, set_a_fn,
                              "set_a_rme_filename (RME filename taxonomy "
                              "— dataset publisher's labels)")
    suppl_c = compute_variant(p3_clips, p5_clips, set_c_fn,
                              "set_c_tightened (Phase 5 sens_rubric's "
                              "training labels)")

    primary_loso_consistency = all(
        r["ref_label"] == r["phase3_label_in_loso"] == r["phase5_label_in_loso"]
        for r in primary["rows"]
    )

    summary = {
        "phase3_path": str(PHASE3_PATH.relative_to(POC_DIR)),
        "phase5_path": str(PHASE5_PATH.relative_to(POC_DIR)),
        "phase3_pooled_auc": p3_meta["pooled_auc"],
        "phase5_pooled_auc": p5_meta["pooled_auc"],
        "phase3_labels_path_in_run": p3_meta.get("labels_path"),
        "phase5_labels_path_in_run": p5_meta.get("labels_path"),
        "primary_label_set_matches_in_loso_labels": primary_loso_consistency,
        "correctness_rule": "ridge_threshold_eq_0",
        "primary_variant": primary,
        "supplementary_set_a_rme_filename": suppl_a,
        "supplementary_set_c_tightened": suppl_c,
    }

    rows = primary["rows"]
    cat_counts = Counter(r["category"] for r in rows)
    n_total = primary["n_clips"]
    n_p3_correct = primary["n_phase3_correct_thr0"]
    n_p5_correct = primary["n_phase5_correct_thr0"]
    n_recovered = primary["n_recovered"]
    n_lost = primary["n_lost"]
    net_shift = primary["net_shift_recovered_minus_lost"]
    recovery_lost_ratio = primary["recovery_lost_ratio"]

    by_category = {
        "V3_NEWLY_RECOVERED": [r for r in rows
                                if r["category"] == "V3_NEWLY_RECOVERED"],
        "V3_NEWLY_LOST":      [r for r in rows
                                if r["category"] == "V3_NEWLY_LOST"],
        "BOTH_RIGHT":         [r for r in rows
                                if r["category"] == "BOTH_RIGHT"],
        "BOTH_WRONG":         [r for r in rows
                                if r["category"] == "BOTH_WRONG"],
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2))

    label_name = lambda x: "ACTION" if x == 1 else "BACKGROUND"
    lines = []
    a = lines.append
    a("# Phase 6 (a) — per-clip Phase 3 vs Phase 5 prediction diff")
    a("")
    a("Pre-registered diagnostic from `docs/phase5_audit.md`. Threshold=0 "
      "on RidgeClassifier decision_function, both pipelines share the same "
      "34 clips and the same in-LOSO label vector.")
    a("")
    a("## Label-source clarification (locked)")
    a("")
    a("Three reference label sets exist in the repo. The same 34 clips are "
      "labeled differently across them:")
    a("")
    a("- **Set A — RME filename taxonomy**: `action_*` / `background_*` "
      "from filenames. The dataset publisher's labels for the original "
      "ear-motion behavior. Not what Phase 5 primary trained on.")
    a("- **Set B — Piotr morning verification** (`outputs/eye_verification_clips.txt`): "
      "the labels Phase 3 + Phase 5 primary were both trained AND "
      "evaluated against. Phase 5 primary's reported AUC of 0.7985 is "
      "the AUC against this set. **This is the directly-decomposable "
      "label set and the primary diagnostic uses it.**")
    a("- **Set C — Piotr tightened-rubric relabel** (`outputs/eye_relabel_unmasked.txt`): "
      "Phase 5 sens_rubric's training labels. Differs from Set B on "
      "several clips (e.g., `bg_S1_12` is ACTION in Set B, BACKGROUND "
      "in Set C).")
    a("")
    a("**Primary diagnostic** uses Set B (matches the AUCs it decomposes); "
      "**supplementary** Sets A and C are computed for sensitivity. "
      "Categories under each set are *not* the same — the prediction-shift "
      "structure depends on the reference label.")
    a("")
    a("## Primary (Set B — verification labels)")
    a("")
    a(f"- Phase 3 pooled AUC: **{p3_meta['pooled_auc']:.4f}** "
      f"(per-clip correct@thr=0: {n_p3_correct}/{n_total})")
    a(f"- Phase 5 primary pooled AUC: **{p5_meta['pooled_auc']:.4f}** "
      f"(per-clip correct@thr=0: {n_p5_correct}/{n_total})")
    a(f"- Δ AUC: +{p5_meta['pooled_auc'] - p3_meta['pooled_auc']:.4f}")
    a("")
    a("| Category | Count | Share |")
    a("|---|---:|---:|")
    for cat in ["BOTH_RIGHT", "BOTH_WRONG",
                "V3_NEWLY_RECOVERED", "V3_NEWLY_LOST"]:
        n = cat_counts.get(cat, 0)
        a(f"| {cat} | {n} | {n / n_total:.1%} |")
    a("")
    a(f"**Net shift (recovered − lost): {net_shift:+d} clips** "
      f"(ratio recovered/lost = {recovery_lost_ratio})")
    a("")
    a("## Mechanical interpretation rule (user-locked decision)")
    a("")
    a("- If `recovered ≫ lost` → cropping is a **uniform lever**. (b) gate:"
      " geometric face-bbox crop captures the same rescue mechanism.")
    a("- If `recovered ≈ lost` → cropping **shifts** which clips classify "
      "well; doesn't uniformly improve. (b) gate: preserve recoveries "
      "without losing new losses (harder target, different criteria).")
    a("")
    a("## Supplementary — sensitivity to label choice")
    a("")
    a("| Variant | n | P3 right@thr=0 | P5 right@thr=0 | Both right | "
      "Both wrong | Recovered | Lost | Net | Ratio |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for v in [primary, suppl_a, suppl_c]:
        cc = v["category_counts"]
        a(f"| {v['variant'].split(' (')[0]} | {v['n_clips']} | "
          f"{v['n_phase3_correct_thr0']} | {v['n_phase5_correct_thr0']} | "
          f"{cc.get('BOTH_RIGHT', 0)} | {cc.get('BOTH_WRONG', 0)} | "
          f"{v['n_recovered']} | {v['n_lost']} | "
          f"{v['net_shift_recovered_minus_lost']:+d} | "
          f"{v['recovery_lost_ratio']} |")
    a("")
    a("Reading: if the recovered/lost ratio holds direction across all "
      "three reference sets, the prediction-shift finding is robust to "
      "label-reference choice. If it inverts under any set, the finding "
      "is label-set-specific.")
    a("")

    for cat in ["V3_NEWLY_RECOVERED", "V3_NEWLY_LOST",
                "BOTH_RIGHT", "BOTH_WRONG"]:
        items = by_category[cat]
        a(f"## {cat} (n={len(items)})")
        a("")
        if not items:
            a("_(none)_")
            a("")
            continue
        a("| Clip | Source | Truth (Set B) | P3 score | P5 score | Δ score |")
        a("|---|---|---|---:|---:|---:|")
        for r in sorted(items, key=lambda x: x["score_delta"], reverse=True):
            a(f"| `{r['clip']}` | {r['source']} | "
              f"{label_name(r['ref_label'])} | "
              f"{r['phase3_score']:+.3f} | "
              f"{r['phase5_score']:+.3f} | "
              f"{r['score_delta']:+.3f} |")
        a("")

    OUT_MD.write_text("\n".join(lines))

    print(f"Phase 3 AUC: {p3_meta['pooled_auc']:.4f} "
          f"({n_p3_correct}/{n_total} correct@thr=0 under Set B)")
    print(f"Phase 5 AUC: {p5_meta['pooled_auc']:.4f} "
          f"({n_p5_correct}/{n_total} correct@thr=0 under Set B)")
    print()
    print("Set B (PRIMARY, verification labels = Phase 5 primary's training):")
    for cat in ["BOTH_RIGHT", "BOTH_WRONG",
                "V3_NEWLY_RECOVERED", "V3_NEWLY_LOST"]:
        print(f"  {cat:24s} {cat_counts.get(cat, 0):3d}")
    print(f"  Net shift recovered-lost: {net_shift:+d} "
          f"(ratio {recovery_lost_ratio})")
    print()
    print("Set A (RME filename taxonomy — supplementary):")
    for cat in ["BOTH_RIGHT", "BOTH_WRONG",
                "V3_NEWLY_RECOVERED", "V3_NEWLY_LOST"]:
        print(f"  {cat:24s} {suppl_a['category_counts'].get(cat, 0):3d}")
    print(f"  Net shift recovered-lost: "
          f"{suppl_a['net_shift_recovered_minus_lost']:+d} "
          f"(ratio {suppl_a['recovery_lost_ratio']})")
    print()
    print("Set C (tightened-rubric — supplementary):")
    for cat in ["BOTH_RIGHT", "BOTH_WRONG",
                "V3_NEWLY_RECOVERED", "V3_NEWLY_LOST"]:
        print(f"  {cat:24s} {suppl_c['category_counts'].get(cat, 0):3d}")
    print(f"  Net shift recovered-lost: "
          f"{suppl_c['net_shift_recovered_minus_lost']:+d} "
          f"(ratio {suppl_c['recovery_lost_ratio']})")
    print()
    print(f"Wrote: {OUT_JSON.relative_to(POC_DIR)}")
    print(f"Wrote: {OUT_MD.relative_to(POC_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
