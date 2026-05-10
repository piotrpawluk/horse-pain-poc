#!/usr/bin/env python3
"""Phase 8b Step 8 — Diagnostic + paired DeLong vs whole-frame baseline.

Per locked Phase 8b Stage 1 §"Test hierarchy":
  - Load-bearing: AUC-vs-gate (G1 ≥0.80 / G2 0.65-0.80 / G3 <0.65)
    [computed by eye_loso_lr_phase8b.py]
  - Supportive: paired DeLong vs whole-frame baseline
    [computed here, n=283 adequately powered]
  - Precision: subject-bootstrap CI on Δ AUC vs whole-frame
    [computed here, supplementary]

Per locked Phase 8b §"Verdict-reporting protocol": surface BOTH G1-G3
(absolute) AND G4 (paired DeLong vs whole-frame) jointly per the 7-row
matrix.

Inputs:
  outputs/eye_loso_results_phase8b.json (Step 7 output, ear-cropped)
  outputs/vjepa2_embeddings.npz (RME whole-frame baseline features)
  outputs/iter65_sanity5_loso_rme_results.json (whole-frame AUC reference)

Outputs:
  outputs/phase8b_diagnostic.{json,md}
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

POC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_DIR / "tools"))
from eye_loso_lr_phase5 import delong_paired  # noqa: E402

P8B_LOSO_PATH = POC_DIR / "outputs" / "eye_loso_results_phase8b.json"
WHOLE_FRAME_EMB = POC_DIR / "outputs" / "vjepa2_embeddings.npz"
SANITY5_PATH = POC_DIR / "outputs" / "iter65_sanity5_loso_rme_results.json"
RME_DATA = POC_DIR / "vendor" / "ReadMyEars_Dataset" / "data"
MANIFEST_PATH = (POC_DIR / "outputs"
                 / "eye_crops_v4_ear_dlc_manifest.jsonl")
OUT_JSON = POC_DIR / "outputs" / "phase8b_diagnostic.json"
OUT_MD = POC_DIR / "outputs" / "phase8b_diagnostic.md"

BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 42

# Locked gates (Phase 8b §8)
G1_STRONG = 0.80
G2_MODEST_LOW = 0.65
G3_FAILS_HIGH = 0.65
G4_SIG = 0.05


def extract_source(filename):
    m = re.search(r"S(\d+)\.mp4", filename)
    return f"S{m.group(1)}" if m else "S?"


def compute_whole_frame_loso():
    """Reproduce whole-frame V-JEPA-2 LOSO predictions (matches Sanity 5
    ssv2_motion). Returns (preds_in_clip_order, labels, clips, sources)."""
    d = np.load(WHOLE_FRAME_EMB, allow_pickle=True)
    X = d["embs"]
    y = d["labels"]
    fns = [str(f) for f in d["filenames"]]
    groups = np.array([extract_source(f) for f in fns])
    sources = sorted(set(groups), key=lambda s: int(s[1:]))
    preds = np.zeros(len(y), dtype=float)
    for s in sources:
        train = groups != s
        test = groups == s
        if not test.any():
            continue
        if len(np.unique(y[train])) < 2:
            continue
        scaler = StandardScaler().fit(X[train])
        clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
        clf.fit(scaler.transform(X[train]), y[train])
        preds[test] = clf.decision_function(scaler.transform(X[test]))
    return preds, y, fns, groups


def subject_bootstrap_ci_on_delta(preds_p8b, preds_wf, labels, groups,
                                   n_bootstrap=BOOTSTRAP_B,
                                   seed=BOOTSTRAP_SEED):
    """Subject-bootstrap CI on Δ AUC = AUC_p8b − AUC_whole_frame."""
    rng = np.random.default_rng(seed)
    sources = sorted(set(groups), key=lambda s: int(s[1:]))
    src_to_idx = defaultdict(list)
    for i, g in enumerate(groups):
        src_to_idx[g].append(i)

    delta_dist = []
    for _ in range(n_bootstrap):
        boot_sources = rng.choice(sources, size=len(sources), replace=True)
        boot_idx = []
        for s in boot_sources:
            boot_idx.extend(src_to_idx[s])
        boot_idx = np.array(boot_idx)
        y_b = labels[boot_idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            auc_p8b = roc_auc_score(y_b, preds_p8b[boot_idx])
            auc_wf = roc_auc_score(y_b, preds_wf[boot_idx])
        except ValueError:
            continue
        delta_dist.append(auc_p8b - auc_wf)
    arr = np.array(delta_dist)
    return {
        "n_kept": len(arr),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "ci_95_low": float(np.percentile(arr, 2.5)),
        "ci_95_high": float(np.percentile(arr, 97.5)),
        "p_delta_le_zero": float((arr <= 0).mean()),
    }


def per_source_aucs(preds, labels, groups):
    """Per-source AUC distribution. Returns dict {source: auc}."""
    sources = sorted(set(groups), key=lambda s: int(s[1:]))
    out = {}
    for s in sources:
        mask = groups == s
        if not mask.any():
            continue
        y_s = labels[mask]
        if len(np.unique(y_s)) < 2:
            out[s] = None
            continue
        out[s] = float(roc_auc_score(y_s, preds[mask]))
    return out


def joint_verdict(g1_strong, g2_modest, g3_fails, g4_sig, delta):
    """Apply locked 7-row verdict-reporting matrix."""
    if g1_strong:
        if g4_sig and delta > 0:
            return ("OUTPERFORM", "Strong absolute AUC AND DLC > "
                                   "whole-frame significantly. Best case.")
        if g4_sig and delta < 0:
            return ("STRONG_BUT_DEGRADES",
                    "Strong absolute AUC BUT significantly < whole-frame. "
                    "Locked-gate-passes does NOT mean methodology-improves-"
                    "baseline. Worth flagging.")
        return ("COMPETITIVE",
                "Strong absolute AUC; paired test inconclusive. DLC "
                "competitive with whole-frame but indistinguishable.")
    if g2_modest:
        if g4_sig and delta > 0:
            return ("UNLIKELY_COMBINATION",
                    "Modest absolute AUC + significant positive Δ — "
                    "unusual; flag for investigation.")
        if g4_sig and delta < 0:
            return ("MODEST_DEGRADES",
                    "Modest absolute AUC AND significantly < whole-frame. "
                    "DLC ear cropping clearly degrades whole-frame baseline.")
        return ("MODEST_INCONCLUSIVE",
                "Modest absolute AUC; paired test inconclusive. DLC "
                "loses some information vs whole-frame, possibly noise.")
    if g3_fails:
        return ("FAILS",
                "Cross-behavior generalization fails regardless of "
                "comparison. Eye-specific narrow narrative.")
    return ("UNCLASSIFIED", "Unexpected gate combination; investigate.")


def main() -> int:
    print("[phase8b diag] loading inputs...", flush=True)
    p8b = json.loads(P8B_LOSO_PATH.read_text())
    sanity5 = json.loads(SANITY5_PATH.read_text())
    whole_frame_auc_expected = sanity5["best_config"]["loso_auc"]

    p8b_per_clip = {r["clip"]: r for r in p8b["per_clip"]}
    auc_p8b = p8b["pooled_auc"]
    boot_p8b = p8b.get("auc_95_ci_subject_bootstrap")

    # Reproduce whole-frame LOSO predictions (clip-aligned)
    print("[phase8b diag] reproducing whole-frame LOSO predictions...",
          flush=True)
    wf_preds, wf_labels, wf_fns, wf_groups = compute_whole_frame_loso()
    wf_auc = float(roc_auc_score(wf_labels, wf_preds))
    print(f"[phase8b diag] whole-frame AUC reproduced: {wf_auc:.10f} "
          f"(expected {whole_frame_auc_expected:.10f})", flush=True)

    # Align p8b predictions to whole-frame clip order for paired tests
    p8b_preds_aligned = []
    p8b_labels_aligned = []
    p8b_groups_aligned = []
    aligned_clips = []
    for fn in wf_fns:
        if fn in p8b_per_clip:
            r = p8b_per_clip[fn]
            p8b_preds_aligned.append(r["score"])
            p8b_labels_aligned.append(r["label"])
            p8b_groups_aligned.append(extract_source(fn))
            aligned_clips.append(fn)
        else:
            # Phase 8b dropped this clip; can't be paired
            pass
    p8b_preds_arr = np.array(p8b_preds_aligned)
    p8b_labels_arr = np.array(p8b_labels_aligned)
    p8b_groups_arr = np.array(p8b_groups_aligned)

    # Subset whole-frame to same clips for fair paired comparison
    wf_idx_for_aligned = {fn: i for i, fn in enumerate(wf_fns)}
    wf_preds_aligned = np.array(
        [wf_preds[wf_idx_for_aligned[c]] for c in aligned_clips]
    )
    wf_labels_aligned = np.array(
        [wf_labels[wf_idx_for_aligned[c]] for c in aligned_clips]
    )

    if len(aligned_clips) < len(wf_fns):
        print(f"[phase8b diag] WARN: {len(wf_fns) - len(aligned_clips)} "
              f"clips dropped from Phase 8b vs whole-frame baseline; "
              f"paired comparison runs on intersection ({len(aligned_clips)} "
              f"clips)", flush=True)

    # Verify label vectors match on aligned clips
    if not np.array_equal(p8b_labels_arr, wf_labels_aligned):
        print("[ERROR] label mismatch between Phase 8b and whole-frame on "
              "aligned clips", file=sys.stderr)
        return 1

    # Recompute pooled AUCs on aligned clips for paired tests
    auc_p8b_aligned = float(roc_auc_score(p8b_labels_arr, p8b_preds_arr))
    auc_wf_aligned = float(roc_auc_score(wf_labels_aligned,
                                          wf_preds_aligned))
    delta_observed = auc_p8b_aligned - auc_wf_aligned

    # Paired DeLong (load-bearing-supportive)
    print("[phase8b diag] paired DeLong vs whole-frame...", flush=True)
    paired = delong_paired(p8b_labels_arr.tolist(),
                            p8b_preds_arr.tolist(),
                            wf_preds_aligned.tolist())

    # Subject-bootstrap CI on Δ (precision)
    print("[phase8b diag] subject-bootstrap CI on Δ (B=10000)...",
          flush=True)
    bootstrap = subject_bootstrap_ci_on_delta(
        p8b_preds_arr, wf_preds_aligned,
        p8b_labels_arr, p8b_groups_arr,
    )

    # Per-source breakdown
    p8b_per_src = per_source_aucs(p8b_preds_arr, p8b_labels_arr,
                                    p8b_groups_arr)
    wf_per_src = per_source_aucs(wf_preds_aligned, wf_labels_aligned,
                                   p8b_groups_arr)
    per_src_delta = {}
    for s in p8b_per_src:
        if p8b_per_src[s] is not None and wf_per_src.get(s) is not None:
            per_src_delta[s] = p8b_per_src[s] - wf_per_src[s]
        else:
            per_src_delta[s] = None
    n_pos_delta = sum(1 for d in per_src_delta.values()
                      if d is not None and d > 0)
    n_total_delta = sum(1 for d in per_src_delta.values() if d is not None)

    # Manifest stats (Decision 1 fallback handling outcomes)
    manifest_stats = {"n_per_frame_only": 0, "n_with_fallback": 0,
                       "n_failed": 0, "fallback_pct_distribution": []}
    if MANIFEST_PATH.exists():
        for line in MANIFEST_PATH.read_text().splitlines():
            if not line.strip():
                continue
            m = json.loads(line)
            if m.get("status") == "ok":
                if m.get("n_single_middle_fallback", 0) == 0:
                    manifest_stats["n_per_frame_only"] += 1
                else:
                    manifest_stats["n_with_fallback"] += 1
                manifest_stats["fallback_pct_distribution"].append(
                    m.get("fallback_pct", 0)
                )
            else:
                manifest_stats["n_failed"] += 1

    # Locked gate evaluation
    g1 = auc_p8b >= G1_STRONG
    g2 = G2_MODEST_LOW <= auc_p8b < G1_STRONG
    g3 = auc_p8b < G3_FAILS_HIGH
    paired_p = paired.get("p_two_sided")
    g4_sig = paired_p is not None and paired_p < G4_SIG

    verdict_label, verdict_text = joint_verdict(
        g1, g2, g3, g4_sig, delta_observed
    )

    summary = {
        "tool": "tools/phase8b_diagnostic.py",
        "phase8b_pooled_auc": auc_p8b,
        "whole_frame_baseline_auc_full": wf_auc,
        "whole_frame_baseline_auc_aligned": auc_wf_aligned,
        "phase8b_aligned_auc_check": auc_p8b_aligned,
        "n_clips_phase8b": len(p8b_per_clip),
        "n_clips_aligned": len(aligned_clips),
        "n_clips_dropped_from_p8b_vs_baseline":
            len(wf_fns) - len(aligned_clips),
        "delta_observed_aligned": delta_observed,
        "phase8b_subject_bootstrap_ci": boot_p8b,
        "paired_delong_vs_whole_frame": paired,
        "subject_bootstrap_ci_on_delta": bootstrap,
        "per_source_phase8b": p8b_per_src,
        "per_source_whole_frame": wf_per_src,
        "per_source_delta": per_src_delta,
        "n_sources_positive_delta": n_pos_delta,
        "n_sources_total_delta": n_total_delta,
        "manifest_stats": manifest_stats,
        "gates": {
            "G1_strong_auc_geq_0_80": g1,
            "G2_modest_0_65_to_0_80": g2,
            "G3_fails_below_0_65": g3,
            "G4_paired_sig_below_0_05": g4_sig,
        },
        "verdict_label": verdict_label,
        "verdict_text": verdict_text,
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))

    # Markdown
    lines = []
    a = lines.append
    a("# Phase 8b diagnostic — DLC ear-keypoint cropping vs whole-frame")
    a("")
    a(f"- Phase 8b pooled AUC: **{auc_p8b:.4f}**")
    a(f"- Whole-frame baseline AUC (full RME 283 clips): **{wf_auc:.4f}**")
    a(f"- Whole-frame baseline AUC (aligned to Phase 8b clips, {len(aligned_clips)} clips): {auc_wf_aligned:.4f}")
    a(f"- Δ observed (aligned): **{delta_observed:+.4f}**")
    a("")
    if boot_p8b:
        a(f"- Phase 8b bootstrap CI: [{boot_p8b[0]:.4f}, {boot_p8b[1]:.4f}]")
    a(f"- Subject-bootstrap CI on Δ (B=10000): "
      f"[{bootstrap['ci_95_low']:+.4f}, {bootstrap['ci_95_high']:+.4f}], "
      f"median {bootstrap['median']:+.4f}")
    a(f"- Paired DeLong vs whole-frame: Δ={paired.get('delta'):+.4f}, "
      f"z={paired.get('z'):.3f}, p={paired.get('p_two_sided'):.4f}")
    a("")
    a("## Locked gates (Phase 8b §8)")
    a("")
    a(f"- **G1** (AUC ≥ 0.80, strong): "
      f"{'PASS' if g1 else 'FAIL'} ({auc_p8b:.4f})")
    a(f"- **G2** (0.65 ≤ AUC < 0.80, modest): "
      f"{'PASS' if g2 else 'FAIL'}")
    a(f"- **G3** (AUC < 0.65, fails): "
      f"{'PASS' if g3 else 'FAIL'}")
    a(f"- **G4 supportive** (paired DeLong p < 0.05): "
      f"{'PASS' if g4_sig else 'FAIL'} (p={paired_p:.4f})")
    a("")
    a("## Joint verdict (per locked verdict-reporting protocol)")
    a("")
    a(f"**{verdict_label}** — {verdict_text}")
    a("")
    a("## Per-source AUC")
    a("")
    a("| Source | Phase 8b AUC | Whole-frame AUC | Δ | Sign |")
    a("|---|---:|---:|---:|---|")
    for s in sorted(p8b_per_src.keys(), key=lambda x: int(x[1:])):
        a_p8b = p8b_per_src.get(s)
        a_wf = wf_per_src.get(s)
        d = per_src_delta.get(s)
        a_p8b_str = f"{a_p8b:.4f}" if a_p8b is not None else "—"
        a_wf_str = f"{a_wf:.4f}" if a_wf is not None else "—"
        d_str = f"{d:+.4f}" if d is not None else "—"
        sign = "+" if (d is not None and d > 0) else ("−" if (d is not None and d < 0) else "")
        a(f"| {s} | {a_p8b_str} | {a_wf_str} | {d_str} | {sign} |")
    a("")
    a(f"Sources with Δ > 0: {n_pos_delta}/{n_total_delta}")
    a("")
    a("## Crop pipeline statistics (Decision 1 fallback handling)")
    a("")
    a(f"- Per-frame-only clips (no fallback): "
      f"{manifest_stats['n_per_frame_only']}/{len(p8b_per_clip)}")
    a(f"- Clips with single-middle-frame fallback applied to some frames: "
      f"{manifest_stats['n_with_fallback']}/{len(p8b_per_clip)}")
    a(f"- Failed clips (status=fail): {manifest_stats['n_failed']}")
    a("")
    if manifest_stats["fallback_pct_distribution"]:
        fp = manifest_stats["fallback_pct_distribution"]
        a(f"- Fallback frame % distribution: median={np.median(fp):.1%}, "
          f"max={max(fp):.1%}, n_clips_with_any_fallback={sum(1 for x in fp if x > 0)}")
        a("")

    OUT_MD.write_text("\n".join(lines))

    print()
    print(f"Phase 8b AUC: {auc_p8b:.4f}")
    print(f"Whole-frame baseline AUC: {wf_auc:.4f}")
    print(f"Δ aligned: {delta_observed:+.4f}")
    print(f"Bootstrap CI on Δ: [{bootstrap['ci_95_low']:+.4f}, "
          f"{bootstrap['ci_95_high']:+.4f}]")
    print(f"Paired DeLong p: {paired_p:.4f}")
    print(f"Per-source +Δ: {n_pos_delta}/{n_total_delta}")
    print()
    print(f"VERDICT: {verdict_label}")
    print(f"NEXT: {verdict_text}")
    print()
    print(f"Wrote: {OUT_JSON.relative_to(POC_DIR)}")
    print(f"Wrote: {OUT_MD.relative_to(POC_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
