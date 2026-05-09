#!/usr/bin/env python3
"""Track B Phase 5 LOSO LR runner — primary + sensitivity 1 + margin curve.

Runs all 5 locked Phase 5 configs in sequence:

  1. primary               — v3 (m=15) + original Phase 3 labels
  2. sensitivity_rubric    — v3 (m=15) + tightened Phase 4 labels
  3. sensitivity_margin_10 — v3 (m=10) + original labels
  4. sensitivity_margin_40 — v3 (m=40) + original labels
  5. sensitivity_margin_80 — v3 (m=80) + original labels

For PRIMARY only:
  - Subject-bootstrap CI (B=10000) — the locked decision metric
  - DeLong-paired test vs Phase 3 (same clips, same labels) — top-band gate
  - Factor-(d) suppression verdict per locked criterion
  - Permutation test n=1000 for parity with Phase 3/4 protocol

For sensitivity runs:
  - Pooled AUC + DeLong CI + subject-bootstrap CI + permutation p
  - Δ vs primary noted

Reuses helpers from tools/eye_loso_lr.py (parse_labels, delong_ci,
permutation_test, post-fix loso_pooled with proper per_clip alignment).
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

POC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_DIR / "tools"))

from eye_loso_lr import (  # noqa: E402
    delong_ci, permutation_test, parse_labels,
    decision_branch as _decision_branch,
    ALPHA, CLASS_WEIGHT, N_PERMUTATIONS, PERMUTATION_SEED,
)

LABELS_ORIGINAL = POC_DIR / "outputs" / "eye_verification_clips.txt"
LABELS_TIGHTENED = POC_DIR / "outputs" / "eye_relabel_unmasked.txt"
PHASE3_RESULTS = POC_DIR / "outputs" / "eye_loso_results.json"

PHASE3_AUC = 0.6813186813186813
REGRESSION_THRESHOLD = PHASE3_AUC - 0.05  # 0.6313

# Persistent BG-target clips for factor-(d) (locked in Phase 4 + Phase 5 pre-regs)
PERSISTENT_BG_TARGETS = [
    "action_S5.mp4_2_.mp4",
    "background_S6.mp4_2_.mp4",
    "background_S6.mp4_3_.mp4",
]

SOURCE_RE = re.compile(r"_(S\d+)\.mp4_")


def extract_source(filename: str) -> str:
    m = SOURCE_RE.search(filename)
    if not m:
        raise ValueError(f"cannot extract source from {filename}")
    return m.group(1)


def loso_pooled(X, y, groups, sources, aligned_filenames):
    """Standard LOSO pooled with post-fix per_clip alignment."""
    aligned_arr = np.array(aligned_filenames)
    all_preds: list[float] = []
    all_truth: list[int] = []
    all_clips: list[str] = []
    fold_aucs: list[float] = []
    fold_log: list[dict] = []
    for source in sources:
        train_idx = (groups != source)
        test_idx = (groups == source)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = RidgeClassifier(alpha=ALPHA, class_weight=CLASS_WEIGHT)
        clf.fit(X_tr, y[train_idx])
        p = clf.decision_function(X_te)
        all_preds.extend(p.tolist())
        all_truth.extend(y[test_idx].tolist())
        all_clips.extend(aligned_arr[test_idx].tolist())
        n_test = int(test_idx.sum())
        n_pos = int(y[test_idx].sum())
        n_neg = n_test - n_pos
        defined = (len(np.unique(y[test_idx])) >= 2)
        fold_auc = (
            float(roc_auc_score(y[test_idx], p)) if defined else None
        )
        if defined:
            fold_aucs.append(fold_auc)
        fold_log.append({
            "source": source, "n_test": n_test, "n_pos": n_pos,
            "n_neg": n_neg, "defined": defined, "fold_auc": fold_auc,
        })
    return (np.array(all_preds), np.array(all_truth),
            np.array(all_clips), fold_aucs, fold_log)


def subject_bootstrap_ci(per_clip, n_boot=10000, seed=42, alpha=0.05):
    """Source-resampled bootstrap CI on pooled AUC."""
    rng = np.random.default_rng(seed)
    sources = sorted(set(p["source"] for p in per_clip))
    src_to_pred = {
        s: [(p["label"], p["score"]) for p in per_clip if p["source"] == s]
        for s in sources
    }
    boot_aucs = []
    for _ in range(n_boot):
        sampled = rng.choice(sources, size=len(sources), replace=True)
        truth, scores = [], []
        for s in sampled:
            for lab, sc in src_to_pred[s]:
                truth.append(lab); scores.append(sc)
        if len(set(truth)) < 2:
            continue
        boot_aucs.append(roc_auc_score(truth, scores))
    boot_aucs = np.array(boot_aucs)
    ci_low, ci_high = np.percentile(
        boot_aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)]
    )
    return float(ci_low), float(ci_high), float(boot_aucs.mean()), len(boot_aucs)


def delong_paired(y_true, scores_a, scores_b):
    """Paired DeLong test for AUC_a vs AUC_b on same (truth, scores) pairs."""
    y = np.asarray(y_true)
    sa = np.asarray(scores_a)
    sb = np.asarray(scores_b)
    pos_mask = y == 1
    neg_mask = y == 0
    pos_a, neg_a = sa[pos_mask], sa[neg_mask]
    pos_b, neg_b = sb[pos_mask], sb[neg_mask]
    m, n = len(pos_a), len(neg_a)
    if m == 0 or n == 0:
        return None
    auc_a = float(roc_auc_score(y, sa))
    auc_b = float(roc_auc_score(y, sb))
    V10_a = np.array([
        np.mean((p > neg_a).astype(float) + 0.5 * (p == neg_a).astype(float))
        for p in pos_a
    ])
    V10_b = np.array([
        np.mean((p > neg_b).astype(float) + 0.5 * (p == neg_b).astype(float))
        for p in pos_b
    ])
    V01_a = np.array([
        np.mean((pos_a > q).astype(float) + 0.5 * (pos_a == q).astype(float))
        for q in neg_a
    ])
    V01_b = np.array([
        np.mean((pos_b > q).astype(float) + 0.5 * (pos_b == q).astype(float))
        for q in neg_b
    ])
    var_a = np.var(V10_a, ddof=1) / m + np.var(V01_a, ddof=1) / n
    var_b = np.var(V10_b, ddof=1) / m + np.var(V01_b, ddof=1) / n
    cov_V10 = np.cov(V10_a, V10_b, ddof=1)[0, 1]
    cov_V01 = np.cov(V01_a, V01_b, ddof=1)[0, 1]
    cov_ab = cov_V10 / m + cov_V01 / n
    delta = auc_a - auc_b
    var_delta = var_a + var_b - 2 * cov_ab
    if var_delta <= 0:
        return {
            "auc_a": auc_a, "auc_b": auc_b, "delta": delta,
            "z": 0.0, "p_two_sided": 1.0, "se_delta": 0.0,
        }
    se_delta = float(math.sqrt(var_delta))
    z = float(delta / se_delta)
    p_two_sided = float(math.erfc(abs(z) / math.sqrt(2)))
    return {
        "auc_a": auc_a, "auc_b": auc_b, "delta": float(delta),
        "z": z, "p_two_sided": p_two_sided, "se_delta": se_delta,
    }


def compute_factor_d_verdict(per_clip):
    """≥2 of 3 persistent BG-targets below median Phase 5 BG-clip score → SUPPRESSED."""
    scores = {p["clip"]: p["score"] for p in per_clip}
    labels = {p["clip"]: p["label"] for p in per_clip}
    bg_scores = sorted([scores[c] for c, lab in labels.items() if lab == 0])
    median_bg = float(np.median(bg_scores)) if bg_scores else None
    targets = []
    n_below = 0
    for tgt in PERSISTENT_BG_TARGETS:
        if tgt not in scores:
            targets.append({"clip": tgt, "score": None,
                            "below_median": None, "note": "not_in_aligned_set"})
            continue
        s = scores[tgt]
        below = (median_bg is not None) and (s < median_bg)
        if below:
            n_below += 1
        targets.append({
            "clip": tgt, "score": float(s),
            "label_in_phase5": int(labels[tgt]),
            "below_median": bool(below),
        })
    verdict = "suppressed" if n_below >= 2 else "persistent"
    return {
        "persistent_bg_clips": targets,
        "median_bg_score": median_bg,
        "n_bg_clips": len(bg_scores),
        "n_targets_below_median": n_below,
        "n_targets_total": len(PERSISTENT_BG_TARGETS),
        "verdict": verdict,
    }


def run_loso(emb_path, labels_path, out_path, mode_label, *,
             do_factor_d=False, do_paired_vs_phase3=False,
             do_permutation=True):
    print(f"\n[{mode_label}] embeddings: {emb_path.name}", flush=True)
    print(f"[{mode_label}] labels: {labels_path.name}", flush=True)
    emb = np.load(emb_path, allow_pickle=True)
    X = emb["embs"]
    filenames = [str(x) for x in emb["filenames"]]
    label_map = parse_labels(labels_path)
    aligned_filenames = [fn for fn in filenames if fn in label_map]
    in_emb_not_lbl = [fn for fn in filenames if fn not in label_map]
    if in_emb_not_lbl:
        print(f"[{mode_label}] HARD ERROR: embeddings without labels: {in_emb_not_lbl}",
              flush=True)
        sys.exit(2)
    y = np.array([label_map[fn]["label"] for fn in aligned_filenames])
    groups = np.array([extract_source(fn) for fn in aligned_filenames])
    sources = sorted(set(groups), key=lambda s: int(s[1:]))
    assert len(X) == len(y) == len(groups), "length mismatch"
    assert set(np.unique(y)) == {0, 1}, "non-binary labels"
    assert not np.isnan(X).any(), "NaN in features"
    print(f"[{mode_label}] aligned: {len(aligned_filenames)} samples, "
          f"{len(sources)} sources, "
          f"bg={int((y==0).sum())} action={int((y==1).sum())}", flush=True)

    t0 = time.time()
    preds, truth, clips, fold_aucs, fold_log = loso_pooled(
        X, y, groups, sources, aligned_filenames,
    )
    pooled_auc, var, ci_low, ci_high = delong_ci(truth, preds)
    print(f"[{mode_label}] pooled AUC = {pooled_auc:.4f} "
          f"(DeLong CI [{ci_low:.4f}, {ci_high:.4f}]) "
          f"in {time.time()-t0:.1f}s", flush=True)
    fold_dist = {
        "min": (min(fold_aucs) if fold_aucs else None),
        "median": (float(np.median(fold_aucs)) if fold_aucs else None),
        "max": (max(fold_aucs) if fold_aucs else None),
        "n_defined": len(fold_aucs),
        "n_skipped": len(sources) - len(fold_aucs),
    }
    print(f"[{mode_label}] fold_dist: {fold_dist}", flush=True)

    per_clip = [
        {"clip": str(clips[i]), "source": extract_source(str(clips[i])),
         "label": int(truth[i]), "score": float(preds[i])}
        for i in range(len(clips))
    ]

    boot_low, boot_high, boot_mean, boot_n = subject_bootstrap_ci(
        per_clip, n_boot=10000, seed=42,
    )
    print(f"[{mode_label}] subject-bootstrap CI [{boot_low:.4f}, {boot_high:.4f}]"
          f" (B={boot_n})", flush=True)

    p_value = None
    null_mean = None
    if do_permutation:
        null_aucs, p_value = permutation_test(
            X, y, groups, sources, pooled_auc,
            n_perm=N_PERMUTATIONS, seed=PERMUTATION_SEED,
        )
        null_mean = float(null_aucs.mean())
        print(f"[{mode_label}] permutation p={p_value:.4f} (null mean={null_mean:.3f})",
              flush=True)

    paired_result = None
    if do_paired_vs_phase3:
        # Pair Phase 3's predictions to Phase 5 primary's predictions on same clips
        ph3 = json.load(open(PHASE3_RESULTS))
        ph3_per_clip = {p["clip"]: (p["label"], p["score"])
                        for p in ph3["per_clip"]}
        # Build paired arrays
        paired_y, scores_p3, scores_p5 = [], [], []
        for entry in per_clip:
            clip = entry["clip"]
            if clip in ph3_per_clip:
                lab_p3, sc_p3 = ph3_per_clip[clip]
                if lab_p3 == entry["label"]:  # same clip, same label
                    paired_y.append(entry["label"])
                    scores_p3.append(sc_p3)
                    scores_p5.append(entry["score"])
        if len(paired_y) >= 4 and len(set(paired_y)) >= 2:
            paired_result = delong_paired(paired_y, scores_p5, scores_p3)
            print(f"[{mode_label}] DeLong-paired vs Phase 3: "
                  f"Δ={paired_result['delta']:+.4f}  z={paired_result['z']:.3f}  "
                  f"p={paired_result['p_two_sided']:.4f}  "
                  f"(n_paired={len(paired_y)})", flush=True)

    factor_d = None
    if do_factor_d:
        factor_d = compute_factor_d_verdict(per_clip)
        print(f"[{mode_label}] factor (d): {factor_d['verdict']} "
              f"({factor_d['n_targets_below_median']}/3 below median)",
              flush=True)

    decision = _decision_branch(pooled_auc)
    delta = pooled_auc - PHASE3_AUC
    regression = pooled_auc < REGRESSION_THRESHOLD

    result = {
        "mode": mode_label,
        "pooled_auc": pooled_auc,
        "delta_vs_phase3": delta,
        "regression_vs_phase3": bool(regression),
        "auc_95_ci_delong": [ci_low, ci_high],
        "auc_95_ci_subject_bootstrap": [boot_low, boot_high],
        "bootstrap_mean": boot_mean,
        "bootstrap_n_kept": boot_n,
        "p_value_permutation": p_value,
        "permutation_null_mean": null_mean,
        "n_permutations": N_PERMUTATIONS if do_permutation else None,
        "fold_dist": fold_dist,
        "fold_log": fold_log,
        "n_aligned": len(aligned_filenames),
        "decision_per_pre_reg_aux": decision,
        "factor_d_suppression": factor_d,
        "delong_paired_vs_phase3": paired_result,
        "config": {
            "classifier": "RidgeClassifier",
            "alpha": ALPHA,
            "class_weight": CLASS_WEIGHT,
            "scaler": "StandardScaler refit per fold",
            "primary_metric": "pooled-prediction AUC",
            "ci_level_test": "subject-bootstrap with replacement, B=10000",
            "ci_method_companion": "DeLong (1988) analytical",
            "delta_test": ("DeLong-paired (1988) for v3 vs Phase 3"
                            if do_paired_vs_phase3 else None),
        },
        "embeddings_path": str(emb_path.relative_to(POC_DIR)),
        "labels_path": str(labels_path.relative_to(POC_DIR)),
        "phase3_pooled_auc_reference": PHASE3_AUC,
        "regression_threshold": REGRESSION_THRESHOLD,
        "per_clip": per_clip,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{mode_label}] saved: {out_path.relative_to(POC_DIR)}", flush=True)
    return result


def main():
    runs = [
        ("primary",
         POC_DIR / "outputs/vjepa2_embeddings_eye_v3_m15.npz",
         LABELS_ORIGINAL,
         POC_DIR / "outputs/eye_loso_results_phase5_primary.json",
         {"do_factor_d": True, "do_paired_vs_phase3": True,
          "do_permutation": True}),
        ("sensitivity_rubric",
         POC_DIR / "outputs/vjepa2_embeddings_eye_v3_m15.npz",
         LABELS_TIGHTENED,
         POC_DIR / "outputs/eye_loso_results_phase5_sens_rubric.json",
         {"do_factor_d": False, "do_paired_vs_phase3": False,
          "do_permutation": True}),
        ("sensitivity_margin_10",
         POC_DIR / "outputs/vjepa2_embeddings_eye_v3_m10.npz",
         LABELS_ORIGINAL,
         POC_DIR / "outputs/eye_loso_results_phase5_sens_margin_10.json",
         {"do_factor_d": False, "do_paired_vs_phase3": False,
          "do_permutation": False}),
        ("sensitivity_margin_40",
         POC_DIR / "outputs/vjepa2_embeddings_eye_v3_m40.npz",
         LABELS_ORIGINAL,
         POC_DIR / "outputs/eye_loso_results_phase5_sens_margin_40.json",
         {"do_factor_d": False, "do_paired_vs_phase3": False,
          "do_permutation": False}),
        ("sensitivity_margin_80",
         POC_DIR / "outputs/vjepa2_embeddings_eye_v3_m80.npz",
         LABELS_ORIGINAL,
         POC_DIR / "outputs/eye_loso_results_phase5_sens_margin_80.json",
         {"do_factor_d": False, "do_paired_vs_phase3": False,
          "do_permutation": False}),
    ]
    results = {}
    for label, emb, lbl, out, kwargs in runs:
        r = run_loso(emb, lbl, out, label, **kwargs)
        results[label] = r

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 5 SUMMARY")
    print("=" * 70)
    print(f"{'config':<30}{'AUC':<8}{'Δ P3':<10}{'Boot CI':<22}{'p':<8}")
    for label, r in results.items():
        delta = r["delta_vs_phase3"]
        boot = r["auc_95_ci_subject_bootstrap"]
        boot_str = f"[{boot[0]:.3f}, {boot[1]:.3f}]"
        p = r.get("p_value_permutation")
        p_str = f"{p:.3f}" if p is not None else "—"
        print(f"{label:<30}{r['pooled_auc']:.4f}  {delta:+.4f}   "
              f"{boot_str:<22}{p_str:<8}")

    # Margin curve quick view
    print("\nMARGIN CURVE (v3 + original labels):")
    for m in [10, 15, 40, 80]:
        if m == 15:
            r = results["primary"]
        else:
            r = results[f"sensitivity_margin_{m}"]
        boot = r["auc_95_ci_subject_bootstrap"]
        print(f"  m={m:>3}%  AUC={r['pooled_auc']:.4f}  "
              f"boot CI [{boot[0]:.3f}, {boot[1]:.3f}]")

    # Phase 5 primary highlights
    primary = results["primary"]
    print("\nPRIMARY HIGHLIGHTS (v3 m=15 + original labels):")
    print(f"  pooled AUC: {primary['pooled_auc']:.4f}")
    print(f"  Δ vs Phase 3 (0.6813): {primary['delta_vs_phase3']:+.4f}")
    print(f"  bootstrap CI: {primary['auc_95_ci_subject_bootstrap']}")
    if primary["delong_paired_vs_phase3"]:
        dp = primary["delong_paired_vs_phase3"]
        print(f"  DeLong-paired Δ vs Phase 3: Δ={dp['delta']:+.4f}, "
              f"z={dp['z']:.3f}, p={dp['p_two_sided']:.4f}")
    print(f"  permutation p: {primary['p_value_permutation']:.4f}")
    if primary["factor_d_suppression"]:
        fd = primary["factor_d_suppression"]
        print(f"  factor (d): {fd['verdict']} "
              f"({fd['n_targets_below_median']}/3 below median {fd['median_bg_score']:+.3f})")


if __name__ == "__main__":
    main()
