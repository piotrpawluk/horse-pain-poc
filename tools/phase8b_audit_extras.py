#!/usr/bin/env python3
"""Phase 8b audit extras — per-clip 4-way categorization, bootstrap shape,
and named fallback-clip table.

Computes the three additional artifacts the audit doc requires beyond
phase8b_diagnostic.json:

  1. Per-clip 4-way joint categorization at threshold 0 (RidgeClassifier
     decision_function default boundary):
       BOTH_RIGHT          — DLC correct, whole-frame correct
       BOTH_WRONG          — DLC wrong, whole-frame wrong
       DLC_BEATS_WHOLE_FRAME — DLC correct, whole-frame wrong
       WHOLE_FRAME_BEATS_DLC — DLC wrong, whole-frame correct

  2. Bootstrap distribution shape (skewness via Fisher-Pearson coefficient
     plus quartiles) on Δ AUC subject-bootstrap distribution.

  3. Named fallback-clip table — the 26 clips with single-middle-frame
     fallback applied to ≥1 frame, with fallback fraction per clip.

Output: outputs/phase8b_audit_extras.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

POC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_DIR / "tools"))
from phase8b_diagnostic import (  # noqa: E402
    compute_whole_frame_loso,
    subject_bootstrap_ci_on_delta,
)

LOSO_PATH = POC_DIR / "outputs" / "eye_loso_results_phase8b.json"
MANIFEST_PATH = POC_DIR / "outputs" / "eye_crops_v4_ear_dlc_manifest.jsonl"
OUT_PATH = POC_DIR / "outputs" / "phase8b_audit_extras.json"


def fisher_pearson_skew(x: np.ndarray) -> float:
    """Sample skewness (Fisher-Pearson g1)."""
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    return float(((x - m) ** 3).mean() / (s ** 3))


def main() -> int:
    # 1. Load Phase 8b per-clip predictions
    loso = json.loads(LOSO_PATH.read_text())
    p8b_per_clip = {row["clip"]: row for row in loso["per_clip"]}
    p8b_clips = list(p8b_per_clip.keys())
    p8b_scores = np.array([p8b_per_clip[c]["score"] for c in p8b_clips])
    labels = np.array([p8b_per_clip[c]["label"] for c in p8b_clips])
    sources = np.array([p8b_per_clip[c]["source"] for c in p8b_clips])

    # 2. Reproduce whole-frame predictions
    wf_preds, wf_labels, wf_fns, wf_groups = compute_whole_frame_loso()
    wf_per_clip = {fn: pred for fn, pred in zip(wf_fns, wf_preds)}

    # Align by clip name
    aligned_p8b = []
    aligned_wf = []
    aligned_label = []
    aligned_clip = []
    aligned_source = []
    n_dropped = 0
    for c in p8b_clips:
        if c not in wf_per_clip:
            n_dropped += 1
            continue
        aligned_p8b.append(p8b_per_clip[c]["score"])
        aligned_wf.append(wf_per_clip[c])
        aligned_label.append(p8b_per_clip[c]["label"])
        aligned_clip.append(c)
        aligned_source.append(p8b_per_clip[c]["source"])
    aligned_p8b = np.array(aligned_p8b)
    aligned_wf = np.array(aligned_wf)
    aligned_label = np.array(aligned_label)

    # 3. Per-clip categorization at threshold 0 (RidgeClassifier default)
    p8b_pred = (aligned_p8b > 0).astype(int)
    wf_pred = (aligned_wf > 0).astype(int)
    p8b_correct = p8b_pred == aligned_label
    wf_correct = wf_pred == aligned_label

    cats = {
        "BOTH_RIGHT": [],
        "BOTH_WRONG": [],
        "DLC_BEATS_WHOLE_FRAME": [],
        "WHOLE_FRAME_BEATS_DLC": [],
    }
    for i, c in enumerate(aligned_clip):
        if p8b_correct[i] and wf_correct[i]:
            tag = "BOTH_RIGHT"
        elif (not p8b_correct[i]) and (not wf_correct[i]):
            tag = "BOTH_WRONG"
        elif p8b_correct[i] and (not wf_correct[i]):
            tag = "DLC_BEATS_WHOLE_FRAME"
        else:
            tag = "WHOLE_FRAME_BEATS_DLC"
        cats[tag].append({
            "clip": c, "source": aligned_source[i],
            "label": int(aligned_label[i]),
            "p8b_score": float(aligned_p8b[i]),
            "wf_score": float(aligned_wf[i]),
        })
    cat_counts = {k: len(v) for k, v in cats.items()}

    # 4. Bootstrap distribution shape on Δ
    src_arr = np.array(aligned_source)
    boot = subject_bootstrap_ci_on_delta(
        aligned_p8b, aligned_wf, aligned_label, src_arr,
    )
    # Re-run bootstrap to get the full distribution (the function only
    # returns summary stats); shape descriptors below.
    rng = np.random.default_rng(42)
    sources_uniq = sorted(set(aligned_source), key=lambda s: int(s[1:]))
    from collections import defaultdict
    from sklearn.metrics import roc_auc_score
    src_to_idx = defaultdict(list)
    for i, g in enumerate(aligned_source):
        src_to_idx[g].append(i)
    deltas = []
    for _ in range(10000):
        boot_sources = rng.choice(sources_uniq, size=len(sources_uniq),
                                   replace=True)
        boot_idx = []
        for s in boot_sources:
            boot_idx.extend(src_to_idx[s])
        boot_idx = np.array(boot_idx)
        y_b = aligned_label[boot_idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            a_p = roc_auc_score(y_b, aligned_p8b[boot_idx])
            a_w = roc_auc_score(y_b, aligned_wf[boot_idx])
        except ValueError:
            continue
        deltas.append(a_p - a_w)
    deltas_arr = np.array(deltas)
    shape = {
        "n_kept": len(deltas_arr),
        "mean": float(deltas_arr.mean()),
        "median": float(np.median(deltas_arr)),
        "std": float(deltas_arr.std()),
        "min": float(deltas_arr.min()),
        "max": float(deltas_arr.max()),
        "q25": float(np.percentile(deltas_arr, 25)),
        "q75": float(np.percentile(deltas_arr, 75)),
        "ci_95_low": float(np.percentile(deltas_arr, 2.5)),
        "ci_95_high": float(np.percentile(deltas_arr, 97.5)),
        "p_delta_le_zero": float((deltas_arr <= 0).mean()),
        "skewness_fisher_pearson": fisher_pearson_skew(deltas_arr),
    }

    # 5. Fallback clip table from manifest
    fallback_rows = []
    with open(MANIFEST_PATH) as f:
        for line in f:
            row = json.loads(line)
            if (row.get("status") == "ok"
                    and row.get("n_single_middle_fallback", 0) > 0):
                fallback_rows.append({
                    "clip": row["clip"],
                    "source": row["clip"].split(".mp4")[0].split("_")[-1]
                    if "S" in row["clip"] else "?",
                    "label": row.get("label"),
                    "frames_written": row["frames_written"],
                    "n_per_frame_confident": row["n_per_frame_confident"],
                    "n_single_middle_fallback":
                    row["n_single_middle_fallback"],
                    "fallback_pct": row["fallback_pct"],
                })
    fallback_rows.sort(key=lambda r: -r["fallback_pct"])

    out = {
        "tool": "tools/phase8b_audit_extras.py",
        "n_aligned": len(aligned_clip),
        "n_dropped_from_alignment": n_dropped,
        "per_clip_4way_counts": cat_counts,
        "per_clip_4way_examples": {
            k: v[:5] for k, v in cats.items()
        },
        "per_clip_4way_full": cats,
        "delta_bootstrap_shape": shape,
        "fallback_clip_table": fallback_rows,
        "n_fallback_clips": len(fallback_rows),
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote: {OUT_PATH.relative_to(POC_DIR)}")
    print()
    print(f"Per-clip 4-way (n={len(aligned_clip)}):")
    for k, v in cat_counts.items():
        print(f"  {k:30s} {v:4d}")
    print()
    print("Bootstrap Δ shape:")
    print(f"  mean={shape['mean']:+.4f}  median={shape['median']:+.4f}  "
          f"std={shape['std']:.4f}")
    print(f"  q25={shape['q25']:+.4f}  q75={shape['q75']:+.4f}")
    print(f"  ci95=[{shape['ci_95_low']:+.4f}, {shape['ci_95_high']:+.4f}]"
          f"  skew={shape['skewness_fisher_pearson']:+.3f}")
    print(f"  P(Δ ≤ 0) = {shape['p_delta_le_zero']:.4f}")
    print()
    print(f"Fallback clips: {len(fallback_rows)}")
    print("  top 5 by fallback %:")
    for r in fallback_rows[:5]:
        print(f"    {r['clip']:50s}  fb={r['fallback_pct']*100:5.1f}%  "
              f"({r['n_single_middle_fallback']}/{r['frames_written']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
