#!/usr/bin/env python3
"""Track B Phase 4 LOSO LR runner with factor-(d) verdict + ablation modes.

Locked spec: outputs/track_b_phase4_preregistration.md (hash ced5cae6...).

Reuses helpers from tools/eye_loso_lr.py for the core LOSO + DeLong CI +
permutation test machinery (post-fix per_clip alignment). This script
adds:

  - __left / __right suffix handling on filenames produced by the v2
    tie-break-both-halves rule. Source extraction strips the suffix
    before regex; label lookup also strips it (both halves of a tied
    clip share the parent clip's label).
  - Mode flag: primary | ablation_a | ablation_b
      primary     = v2 crops + tightened-rubric labels   (Phase 4 main run)
      ablation_a  = v1 crops + tightened-rubric labels   (isolates relabel effect)
      ablation_b  = v2 crops + original Phase 3 labels   (isolates v2 crop effect)
  - Factor-(d) verdict computation per locked criterion: ≥2 of 3
    persistent BG-target clips score strictly below median of all 20
    Phase 4 BG-labeled clips → factor (d) SUPPRESSED; ≤1 → PERSISTENT.
    Tie-break duplicates use max-score-per-clip convention.
  - Subject-bootstrap CI alongside DeLong (locked Phase 4 reporting).

Output JSON schema matches Phase 3 plus:
  - factor_d_suppression: {persistent_bg_clips, scores, median_bg_score,
                           n_below_median, verdict}
  - bootstrap CI alongside DeLong CI
  - mode field
"""

from __future__ import annotations

import argparse
import json
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

EMB_V2 = POC_DIR / "outputs" / "vjepa2_embeddings_eye_v2.npz"
EMB_V1 = POC_DIR / "outputs" / "vjepa2_embeddings_eye.npz"
LABELS_TIGHTENED = POC_DIR / "outputs" / "eye_relabel_unmasked.txt"
LABELS_ORIGINAL = POC_DIR / "outputs" / "eye_verification_clips.txt"

PHASE3_AUC = 0.6813186813186813
REGRESSION_THRESHOLD = PHASE3_AUC - 0.05  # 0.6313

PERSISTENT_BG_TARGETS = [
    "action_S5.mp4_2_.mp4",
    "background_S6.mp4_2_.mp4",
    "background_S6.mp4_3_.mp4",
]

SOURCE_RE = re.compile(r"_(S\d+)\.mp4_")
HALF_SUFFIX_RE = re.compile(r"__(?:left|right)\.mp4$")


def strip_half_suffix(filename: str) -> str:
    """v2 tie produces clip__left.mp4 / clip__right.mp4. Strip back to clip.mp4."""
    m = HALF_SUFFIX_RE.search(filename)
    if m:
        return filename[: m.start()] + ".mp4"
    return filename


def extract_source(filename: str) -> str:
    parent = strip_half_suffix(filename)
    m = SOURCE_RE.search(parent)
    if not m:
        raise ValueError(f"cannot extract source from {filename}")
    return m.group(1)


def loso_pooled(X, y, groups, sources, aligned_filenames):
    """Same as eye_loso_lr.loso_pooled but with extract_source localized here."""
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
            float(roc_auc_score(y[test_idx], p))
            if defined else None
        )
        if defined:
            fold_aucs.append(fold_auc)
        fold_log.append({
            "source": source,
            "n_test": n_test,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "defined": defined,
            "fold_auc": fold_auc,
        })
    return (np.array(all_preds), np.array(all_truth),
            np.array(all_clips), fold_aucs, fold_log)


def subject_bootstrap_ci(per_clip, n_boot=2000, seed=42, alpha=0.05):
    """Source-resampled bootstrap (matches Phase 3 phase3_subject_bootstrap_ci.json)."""
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
                truth.append(lab)
                scores.append(sc)
        if len(set(truth)) < 2:
            continue
        boot_aucs.append(roc_auc_score(truth, scores))
    boot_aucs = np.array(boot_aucs)
    ci_low, ci_high = np.percentile(boot_aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(ci_low), float(ci_high), float(boot_aucs.mean()), len(boot_aucs)


def compute_factor_d_verdict(per_clip):
    """Locked criterion (track_b_phase4_preregistration.md, hash ced5cae6...):
    >=2 of 3 persistent BG-target clips score strictly below median of all 20
    Phase 4 BG-labeled clips → SUPPRESSED; <=1 → PERSISTENT.
    Tie-break duplicates use max-score-per-clip convention.
    """
    # Aggregate to parent-clip level, max score per parent clip
    parent_scores = {}
    parent_labels = {}
    for entry in per_clip:
        parent = strip_half_suffix(entry["clip"])
        s = entry["score"]
        if parent not in parent_scores or s > parent_scores[parent]:
            parent_scores[parent] = s
        parent_labels[parent] = entry["label"]

    bg_scores = sorted(
        [parent_scores[c] for c, lab in parent_labels.items() if lab == 0]
    )
    median_bg = float(np.median(bg_scores)) if bg_scores else None

    targets = []
    n_below = 0
    for tgt in PERSISTENT_BG_TARGETS:
        if tgt not in parent_scores:
            targets.append({"clip": tgt, "score": None, "below_median": None,
                            "note": "not_in_aligned_set"})
            continue
        s = parent_scores[tgt]
        below = (median_bg is not None) and (s < median_bg)
        if below:
            n_below += 1
        targets.append({
            "clip": tgt, "score": float(s),
            "label_in_phase4": parent_labels[tgt],
            "below_median": bool(below),
        })

    verdict = "suppressed" if n_below >= 2 else "persistent"
    return {
        "persistent_bg_clips": targets,
        "median_bg_score": median_bg,
        "n_bg_clips_aggregated_to_parent": len(bg_scores),
        "n_targets_below_median": n_below,
        "n_targets_total": len(PERSISTENT_BG_TARGETS),
        "verdict": verdict,
        "criterion": (
            "≥2 of 3 persistent BG-target clips score strictly below "
            "median of all Phase 4 BG-labeled clips' decision_function scores "
            "(parent-clip max convention) → suppressed; ≤1 → persistent"
        ),
    }


def run_loso(emb_path, labels_path, out_path, mode):
    print(f"\n[phase4 {mode}] embeddings: {emb_path}", flush=True)
    print(f"[phase4 {mode}] labels: {labels_path}", flush=True)
    emb = np.load(emb_path, allow_pickle=True)
    X = emb["embs"]
    filenames = [str(x) for x in emb["filenames"]]
    print(f"[phase4 {mode}] {X.shape} embeddings, {len(filenames)} filenames",
          flush=True)

    # Parse labels — for v2 with __half suffix, label of the half = label of parent
    label_map = parse_labels(labels_path)
    print(f"[phase4 {mode}] {len(label_map)} label entries parsed", flush=True)

    # Alignment: map each embedding filename to a label (stripping __half if needed)
    aligned_filenames = []
    aligned_y = []
    aligned_groups = []
    in_emb_not_lbl = []
    for fn in filenames:
        parent = strip_half_suffix(fn)
        if parent in label_map:
            aligned_filenames.append(fn)
            aligned_y.append(label_map[parent]["label"])
            aligned_groups.append(extract_source(fn))
        else:
            in_emb_not_lbl.append((fn, parent))

    if in_emb_not_lbl:
        print(f"[phase4 {mode}] HARD ERROR: {len(in_emb_not_lbl)} embeddings "
              f"have no label:", flush=True)
        for fn, parent in in_emb_not_lbl:
            print(f"   {fn} (parent={parent})", flush=True)
        sys.exit(2)

    # Embeddings without parent in labels would have failed above
    # Labels not used (no embedding for them) get logged
    parent_set = {strip_half_suffix(fn) for fn in filenames}
    in_lbl_not_emb = sorted(set(label_map.keys()) - parent_set)
    alignment_log = []
    for parent in in_lbl_not_emb:
        info = label_map[parent]
        # Identify excluded clips
        if parent == "background_S4.mp4_7_.mp4":
            reason = "YOLO_no_detection_static"
        elif parent == "background_S1.mp4_11_.mp4":
            reason = "manual_excluded_eye_under_20pct"
        else:
            reason = "no_embedding_in_v2"
        alignment_log.append({"clip": parent, "label": info["verdict"],
                              "reason": reason, "action": "dropped"})
        print(f"[phase4 {mode}] dropped (no embedding): {parent:32}  "
              f"label={info['verdict']:<10} reason={reason}", flush=True)

    y = np.array(aligned_y)
    groups = np.array(aligned_groups)
    sources = sorted(set(groups), key=lambda s: int(s[1:]))

    # Defensive asserts
    assert len(X) == len(y) == len(groups), \
        f"length mismatch: X={len(X)}, y={len(y)}, groups={len(groups)}"
    assert set(np.unique(y)) == {0, 1}, f"non-binary labels: {np.unique(y)}"
    assert not np.isnan(X).any(), "NaN in V-JEPA-2 features"
    print(f"[phase4 {mode}] aligned: {len(aligned_filenames)} samples, "
          f"{len(sources)} sources, "
          f"class balance: bg={int((y == 0).sum())}, "
          f"action={int((y == 1).sum())}", flush=True)

    # LOSO
    print(f"[phase4 {mode}] running LOSO with α={ALPHA}, "
          f"class_weight='{CLASS_WEIGHT}'...", flush=True)
    t0 = time.time()
    preds, truth, clips_in_loso, fold_aucs, fold_log = loso_pooled(
        X, y, groups, sources, aligned_filenames,
    )
    pooled_auc, var, ci_low, ci_high = delong_ci(truth, preds)
    print(f"[phase4 {mode}] pooled AUC = {pooled_auc:.4f} "
          f"(DeLong 95% CI [{ci_low:.4f}, {ci_high:.4f}]) "
          f"in {time.time()-t0:.1f}s", flush=True)

    fold_dist = {
        "min": (min(fold_aucs) if fold_aucs else None),
        "median": (float(np.median(fold_aucs)) if fold_aucs else None),
        "max": (max(fold_aucs) if fold_aucs else None),
        "n_defined": len(fold_aucs),
        "n_skipped": len(sources) - len(fold_aucs),
    }
    print(f"[phase4 {mode}] fold_dist: {fold_dist}", flush=True)

    # Permutation test
    print(f"[phase4 {mode}] permutation test n={N_PERMUTATIONS}, "
          f"seed={PERMUTATION_SEED}...", flush=True)
    null_aucs, p_value = permutation_test(
        X, y, groups, sources, pooled_auc,
        n_perm=N_PERMUTATIONS, seed=PERMUTATION_SEED,
    )
    print(f"[phase4 {mode}] p_value = {p_value:.4f} "
          f"(null mean={null_aucs.mean():.3f})", flush=True)

    # Per-clip
    per_clip = [
        {
            "clip": str(clips_in_loso[i]),
            "parent_clip": strip_half_suffix(str(clips_in_loso[i])),
            "source": extract_source(str(clips_in_loso[i])),
            "label": int(truth[i]),
            "score": float(preds[i]),
        }
        for i in range(len(clips_in_loso))
    ]

    # Subject-bootstrap CI
    print(f"[phase4 {mode}] subject-bootstrap CI...", flush=True)
    boot_low, boot_high, boot_mean, boot_n = subject_bootstrap_ci(
        per_clip, n_boot=2000, seed=42,
    )
    print(f"[phase4 {mode}] bootstrap 95% CI [{boot_low:.4f}, {boot_high:.4f}]",
          flush=True)

    # Factor-(d) verdict (only meaningful for primary)
    factor_d = compute_factor_d_verdict(per_clip)
    if mode == "primary":
        print(f"[phase4 {mode}] factor (d) verdict: {factor_d['verdict']} "
              f"({factor_d['n_targets_below_median']}/3 below median)",
              flush=True)
        for t in factor_d["persistent_bg_clips"]:
            if t.get("score") is not None:
                marker = "BELOW" if t["below_median"] else "above"
                print(f"  {t['clip']:30}  score={t['score']:+.3f}  "
                      f"vs median {factor_d['median_bg_score']:+.3f}  ({marker})",
                      flush=True)

    decision = _decision_branch(pooled_auc)
    delta_vs_phase3 = pooled_auc - PHASE3_AUC
    regression = pooled_auc < REGRESSION_THRESHOLD
    print(f"[phase4 {mode}] decision_per_pre_reg: {decision}  "
          f"(Δ vs Phase 3 = {delta_vs_phase3:+.4f}; "
          f"regression={regression})", flush=True)

    result = {
        "mode": mode,
        "pooled_auc": pooled_auc,
        "delta_vs_phase3": delta_vs_phase3,
        "regression_vs_phase3": bool(regression),
        "auc_95_ci_delong": [ci_low, ci_high],
        "auc_95_ci_subject_bootstrap": [boot_low, boot_high],
        "bootstrap_mean": boot_mean,
        "bootstrap_n_kept": boot_n,
        "p_value": p_value,
        "n_permutations": N_PERMUTATIONS,
        "permutation_null_mean": float(null_aucs.mean()),
        "permutation_seed": PERMUTATION_SEED,
        "fold_dist": fold_dist,
        "fold_log": fold_log,
        "n_aligned": len(aligned_filenames),
        "n_dropped_no_embedding": len(in_lbl_not_emb),
        "alignment_log": alignment_log,
        "decision_per_pre_reg": decision,
        "factor_d_suppression": factor_d if mode == "primary" else None,
        "config": {
            "classifier": "RidgeClassifier",
            "alpha": ALPHA,
            "class_weight": CLASS_WEIGHT,
            "scaler": "StandardScaler refit per fold",
            "primary_metric": "pooled-prediction AUC",
            "permutation_design": "global label shuffle, full LOSO per permutation",
            "p_value_form": "(sum(null >= observed) + 1) / (n + 1)",
            "ci_method_primary": "DeLong (1988) analytical",
            "ci_method_companion": "subject-bootstrap with replacement, n=2000",
        },
        "embeddings_path": str(Path(emb_path).relative_to(POC_DIR)),
        "labels_path": str(Path(labels_path).relative_to(POC_DIR)),
        "phase3_pooled_auc_reference": PHASE3_AUC,
        "regression_threshold": REGRESSION_THRESHOLD,
        "per_clip": per_clip,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    out_resolved = Path(out_path).resolve()
    try:
        out_pretty = out_resolved.relative_to(POC_DIR)
    except ValueError:
        out_pretty = out_resolved
    print(f"[phase4 {mode}] saved: {out_pretty}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
                    choices=["primary", "ablation_a", "ablation_b"],
                    required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.mode == "primary":
        emb, lbl = EMB_V2, LABELS_TIGHTENED
    elif args.mode == "ablation_a":
        emb, lbl = EMB_V1, LABELS_TIGHTENED
    elif args.mode == "ablation_b":
        emb, lbl = EMB_V2, LABELS_ORIGINAL
    else:
        sys.exit(f"unknown mode: {args.mode}")

    run_loso(emb, lbl, args.out, args.mode)


if __name__ == "__main__":
    main()
