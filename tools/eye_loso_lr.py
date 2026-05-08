#!/usr/bin/env python3
"""Track B Phase 3 — V-JEPA-2 + Ridge LOSO on user's eye labels.

Locked design (final review pass on 2026-05-08):

  - Pooled-prediction AUC over all aligned (truth, score) pairs concatenated
    across folds is the PRIMARY metric (more interpretable at n=34 than
    mean-of-fold; no hidden bias from class-degenerate skip).
  - Per-fold AUC distribution (min/median/max + n_defined + n_skipped) is
    SECONDARY — the spread diagnostic.
  - DeLong (1988) analytical 95 % CI on the pooled AUC. CI before p-value
    in the writeup, effect size before significance.
  - Permutation test (n=1000) shuffles labels GLOBALLY once per permutation,
    re-runs the FULL LOSO loop, recomputes pooled AUC. Group structure not
    preserved under the null — matches the 0.8746 ear baseline protocol.
  - +1-form p-value (floor 1/(n+1)).
  - Scaler is REFIT per fold (train statistics only); fresh RidgeClassifier
    per fold (no state carryover, audit-friendly even though Ridge overwrites
    on fit).
  - Decision per pre-registration (track_b_phase1_preregistration.md):
      pooled AUC ≥ 0.65   → adopt v1 crop, write up Track A
      0.55 ≤ AUC < 0.65   → run pre-committed v2 profile-aware crop ONCE
      AUC < 0.55          → conclude V-JEPA-2 + LR is eye-track-failed
  - Alignment contract: embeddings npz is canonical (34 rows). Labels CSV
    has 36 rows; the 2 excluded clips are dropped with explicit log lines.
    Any embedding without a label → hard error (fail fast).

Output: outputs/eye_loso_results.json with pooled_auc, auc_95_ci, p_value,
fold_dist, per-clip predictions, alignment log, and decision_per_pre_reg.
"""

from __future__ import annotations

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
EMB_PATH = POC_DIR / "outputs" / "vjepa2_embeddings_eye.npz"
LABELS_PATH = POC_DIR / "outputs" / "eye_verification_clips.txt"
OUT_PATH = POC_DIR / "outputs" / "eye_loso_results.json"

ALPHA = 1.0
CLASS_WEIGHT = "balanced"
N_PERMUTATIONS = 1000
PERMUTATION_SEED = 42
SOURCE_RE = re.compile(r"_(S\d+)\.mp4_")


def parse_labels(path: Path) -> dict[str, dict]:
    """Parse the verification file; return {basename: {label, observation}}."""
    out: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(" - ")]
            if len(parts) < 3:
                continue
            clip_path, observation = parts[0], " - ".join(parts[1:-1])
            verdict = parts[-1].upper().replace("?", "").strip()
            if verdict not in {"ACTION", "BACKGROUND"}:
                print(f"[labels] WARNING: unparseable verdict on '{line[:80]}'",
                      flush=True)
                continue
            basename = Path(clip_path).name
            out[basename] = {
                "label": 1 if verdict == "ACTION" else 0,
                "verdict": verdict,
                "observation": observation,
            }
    return out


def extract_source(filename: str) -> str:
    m = SOURCE_RE.search(filename)
    if not m:
        raise ValueError(f"cannot extract source from {filename}")
    return m.group(1)


def delong_ci(y_true, y_score, alpha: float = 0.05) -> tuple[float, float, float, float]:
    """DeLong (1988) analytical variance + Wald CI for a single AUC.
    Returns (auc, variance, ci_low, ci_high). CI clipped to [0, 1]."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    auc = float(roc_auc_score(y_true, y_score))
    # Per-positive structural components
    V10 = np.array([
        np.mean((p > neg).astype(float) + 0.5 * (p == neg).astype(float))
        for p in pos
    ])
    V01 = np.array([
        np.mean((pos > q).astype(float) + 0.5 * (pos == q).astype(float))
        for q in neg
    ])
    s10 = float(np.var(V10, ddof=1)) if m > 1 else 0.0
    s01 = float(np.var(V01, ddof=1)) if n > 1 else 0.0
    var = s10 / m + s01 / n
    se = float(np.sqrt(var))
    z = 1.959963984540054  # qnorm(0.975)
    ci_low = max(0.0, auc - z * se)
    ci_high = min(1.0, auc + z * se)
    return auc, var, ci_low, ci_high


def loso_pooled(X, y, groups, sources):
    """Single LOSO pass returning pooled (truth, score) and per-fold AUCs."""
    all_preds: list[float] = []
    all_truth: list[int] = []
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
    return np.array(all_preds), np.array(all_truth), fold_aucs, fold_log


def permutation_test(X, y, groups, sources, observed_auc,
                     n_perm: int = N_PERMUTATIONS,
                     seed: int = PERMUTATION_SEED):
    """Globally-shuffled labels, full LOSO loop per permutation, pooled AUC under null."""
    rng = np.random.default_rng(seed)
    null_aucs: list[float] = []
    t0 = time.time()
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        preds: list[float] = []
        truth: list[int] = []
        for source in sources:
            tr = (groups != source)
            te = (groups == source)
            scaler = StandardScaler()
            clf = RidgeClassifier(alpha=ALPHA, class_weight=CLASS_WEIGHT)
            clf.fit(scaler.fit_transform(X[tr]), y_perm[tr])
            preds.extend(clf.decision_function(scaler.transform(X[te])).tolist())
            truth.extend(y_perm[te].tolist())
        null_aucs.append(float(roc_auc_score(truth, preds)))
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [perm {i+1}/{n_perm}] elapsed {elapsed:.0f}s, "
                  f"rate {(i+1)/elapsed:.1f}/s", flush=True)
    null_aucs = np.array(null_aucs)
    p_value = (np.sum(null_aucs >= observed_auc) + 1) / (n_perm + 1)
    return null_aucs, float(p_value)


def decision_branch(auc: float) -> str:
    if auc >= 0.65:
        return ">=0.65"
    if auc >= 0.55:
        return "0.55-0.65"
    return "<0.55"


def main() -> int:
    # --- Load embeddings (canonical row set) ----------------------------------
    if not EMB_PATH.exists():
        sys.exit(f"missing embeddings: {EMB_PATH}")
    emb = np.load(EMB_PATH, allow_pickle=True)
    X = emb["embs"]
    filenames = [str(x) for x in emb["filenames"]]
    print(f"[loso] embeddings: {X.shape} from {EMB_PATH.name}", flush=True)
    print(f"[loso] parity_passed (Phase 2 audit): "
          f"{bool(emb['parity_passed'])}", flush=True)

    # --- Load labels ----------------------------------------------------------
    if not LABELS_PATH.exists():
        sys.exit(f"missing labels: {LABELS_PATH}")
    label_map = parse_labels(LABELS_PATH)
    print(f"[loso] labels parsed: {len(label_map)} entries from "
          f"{LABELS_PATH.name}", flush=True)

    # --- Alignment contract ---------------------------------------------------
    emb_set = set(filenames)
    lbl_set = set(label_map.keys())
    in_lbl_not_emb = sorted(lbl_set - emb_set)
    in_emb_not_lbl = sorted(emb_set - lbl_set)

    alignment_log: list[dict] = []
    for fn in in_lbl_not_emb:
        info = label_map[fn]
        # Determine why this clip has no embedding
        if fn == "background_S4.mp4_7_.mp4":
            reason = "YOLO_no_detection_static"
        elif fn == "background_S1.mp4_11_.mp4":
            reason = "manual_excluded_eye_under_20pct"
        else:
            reason = "unknown_no_embedding"
        alignment_log.append({
            "clip": fn, "label": info["verdict"], "reason": reason,
            "action": "dropped"
        })
        print(f"[loso] dropped (no embedding): {fn:32} "
              f"label={info['verdict']:<10} reason={reason}", flush=True)

    if in_emb_not_lbl:
        print(f"[loso] HARD ERROR: {len(in_emb_not_lbl)} embeddings have no label:",
              flush=True)
        for fn in in_emb_not_lbl:
            print(f"   missing label: {fn}", flush=True)
        sys.exit(2)

    # --- Build aligned arrays -------------------------------------------------
    aligned_filenames = [fn for fn in filenames if fn in label_map]
    if len(aligned_filenames) != len(filenames):
        sys.exit("alignment failure (this should be unreachable)")
    y = np.array([label_map[fn]["label"] for fn in aligned_filenames])
    groups = np.array([extract_source(fn) for fn in aligned_filenames])
    sources = sorted(set(groups), key=lambda s: int(s[1:]))

    # --- Defensive asserts ----------------------------------------------------
    assert len(X) == len(y) == len(groups), \
        f"length mismatch: X={len(X)}, y={len(y)}, groups={len(groups)}"
    assert set(np.unique(y)) == {0, 1}, \
        f"non-binary labels: {np.unique(y)}"
    assert not np.isnan(X).any(), "NaN in V-JEPA-2 features"
    print(f"[loso] aligned: {len(aligned_filenames)} rows, "
          f"{len(sources)} sources, "
          f"class balance: bg={int((y == 0).sum())}, "
          f"action={int((y == 1).sum())}", flush=True)
    print(f"[loso] sources: {sources}", flush=True)

    # --- LOSO pooled ----------------------------------------------------------
    print(f"[loso] running LOSO with α={ALPHA}, "
          f"class_weight='{CLASS_WEIGHT}'...", flush=True)
    t0 = time.time()
    preds, truth, fold_aucs, fold_log = loso_pooled(X, y, groups, sources)
    pooled_auc, var, ci_low, ci_high = delong_ci(truth, preds)
    print(f"[loso] pooled AUC = {pooled_auc:.4f} "
          f"(95% CI [{ci_low:.4f}, {ci_high:.4f}]) "
          f"in {time.time() - t0:.1f}s", flush=True)

    fold_dist = {
        "min": (min(fold_aucs) if fold_aucs else None),
        "median": (float(np.median(fold_aucs)) if fold_aucs else None),
        "max": (max(fold_aucs) if fold_aucs else None),
        "n_defined": len(fold_aucs),
        "n_skipped": len(sources) - len(fold_aucs),
    }
    print(f"[loso] fold_dist: {fold_dist}", flush=True)

    # --- Permutation test -----------------------------------------------------
    print(f"[loso] permutation test n={N_PERMUTATIONS}, "
          f"seed={PERMUTATION_SEED}...", flush=True)
    null_aucs, p_value = permutation_test(
        X, y, groups, sources, pooled_auc,
        n_perm=N_PERMUTATIONS, seed=PERMUTATION_SEED,
    )
    print(f"[loso] p_value = {p_value:.4f} "
          f"(null mean={null_aucs.mean():.3f}, std={null_aucs.std():.3f})",
          flush=True)

    # --- Decision -------------------------------------------------------------
    decision = decision_branch(pooled_auc)
    print(f"[loso] decision_per_pre_reg: {decision}", flush=True)

    # --- Output JSON ----------------------------------------------------------
    per_clip = [
        {
            "clip": aligned_filenames[i],
            "source": groups[i],
            "label": int(truth[i]),
            "score": float(preds[i]),
        }
        for i in range(len(aligned_filenames))
    ]

    result = {
        "pooled_auc": pooled_auc,
        "auc_95_ci": [ci_low, ci_high],
        "ci_method": "delong",
        "auc_variance_delong": float(var),
        "p_value": p_value,
        "n_permutations": N_PERMUTATIONS,
        "permutation_null_mean": float(null_aucs.mean()),
        "permutation_null_std": float(null_aucs.std()),
        "permutation_seed": PERMUTATION_SEED,
        "fold_dist": fold_dist,
        "fold_log": fold_log,
        "n_aligned": len(aligned_filenames),
        "n_dropped_no_embedding": len(in_lbl_not_emb),
        "alignment_log": alignment_log,
        "decision_per_pre_reg": decision,
        "config": {
            "classifier": "RidgeClassifier",
            "alpha": ALPHA,
            "class_weight": CLASS_WEIGHT,
            "scaler": "StandardScaler refit per fold",
            "primary_metric": "pooled-prediction AUC",
            "permutation_design": "global label shuffle, full LOSO per permutation",
            "p_value_form": "(sum(null >= observed) + 1) / (n + 1)",
        },
        "embeddings_path": str(EMB_PATH.relative_to(POC_DIR)),
        "labels_path": str(LABELS_PATH.relative_to(POC_DIR)),
        "per_clip": per_clip,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[loso] saved: {OUT_PATH.relative_to(POC_DIR)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
