#!/usr/bin/env python3
"""Phase 10a-prelim Step 1.5 — train + save the deployed classifier.

Trains RidgeClassifier(alpha=1.0, class_weight='balanced') + StandardScaler
on ALL 283 RME DLC-cropped V-JEPA-2 features (Phase 8b's
`outputs/vjepa2_embeddings_ear_v4.npz`). Saves parameters to JSON for
reproducible inference on Prudnik subset.

Per D3 of `outputs/track_b_phase10a_prelim_preregistration.md`.

Verifies bit-exact reproducibility: re-loads the JSON, manually computes
decision_function on a known RME clip, compares to sklearn's
clf.decision_function output. Must be ≤ 1e-10 difference.

Outputs:
    outputs/phase10a_prelim_deployed_classifier.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

POC = Path(__file__).resolve().parent.parent
EMB_PATH = POC / "outputs" / "vjepa2_embeddings_ear_v4.npz"
OUT_JSON = POC / "outputs" / "phase10a_prelim_deployed_classifier.json"

# Locked hyperparameters (same as Phase 8b)
ALPHA = 1.0
CLASS_WEIGHT = "balanced"


def manual_decision_function(
    x: np.ndarray, mean: np.ndarray, scale: np.ndarray,
    coef: np.ndarray, intercept: float,
) -> float:
    """Reproduce sklearn's StandardScaler + RidgeClassifier.decision_function from saved params."""
    scaled = (x - mean) / scale
    return float(np.dot(scaled, coef) + intercept)


def main() -> int:
    if not EMB_PATH.exists():
        print(f"[ERROR] Embeddings not found: {EMB_PATH}", file=sys.stderr)
        return 1

    print(f"[step1.5] Loading {EMB_PATH.name}…")
    npz = np.load(EMB_PATH, allow_pickle=True)
    X = npz["embs"].astype(np.float64)  # (283, 1024)
    y = npz["labels"].astype(np.int64)  # (283,) — 1=action, 0=background
    filenames = npz["filenames"]
    print(f"           shape: {X.shape}, labels: {(y == 1).sum()} action / {(y == 0).sum()} background")

    # Fit StandardScaler + RidgeClassifier on ALL 283 clips (D3)
    print(f"[step1.5] Fitting StandardScaler + RidgeClassifier(alpha={ALPHA}, "
          f"class_weight={CLASS_WEIGHT!r}) on full 283…")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = RidgeClassifier(alpha=ALPHA, class_weight=CLASS_WEIGHT)
    clf.fit(X_scaled, y)

    # In-sample AUC sanity (expect very high — fit on full data; bounded by Ridge regularization)
    in_sample_scores = clf.decision_function(X_scaled)
    in_sample_auc = float(roc_auc_score(y, in_sample_scores))
    print(f"[step1.5] In-sample AUC on RME (fit-on-all): {in_sample_auc:.4f} "
          f"(expected ≈ 0.95+; the all-data classifier is closer to overfitting than Phase 8b's "
          f"LOSO per-fold classifiers)")
    print("[step1.5] Phase 8b reference (LOSO pooled AUC): 0.9008 — this is the OOF reference")

    # Pack params for JSON
    params = {
        "model_type": "sklearn.linear_model.RidgeClassifier",
        "alpha": ALPHA,
        "class_weight": CLASS_WEIGHT,
        "n_features": int(X.shape[1]),
        "n_train_clips": int(X.shape[0]),
        "n_train_action": int((y == 1).sum()),
        "n_train_background": int((y == 0).sum()),
        "scaler_type": "sklearn.preprocessing.StandardScaler",
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "ridge_coef": clf.coef_.flatten().tolist(),
        "ridge_intercept": float(clf.intercept_.flatten()[0]),
        "in_sample_auc_on_RME": in_sample_auc,
        "phase8b_LOSO_reference_AUC": 0.9008,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source_embeddings": str(EMB_PATH.relative_to(POC)),
        "preregistration": "outputs/track_b_phase10a_prelim_preregistration.md",
        "notes": (
            "Deployed classifier for Phase 10a-prelim transfer test. Trained on "
            "ALL 283 RME clips (no LOSO). Decision_function output distribution "
            "differs from Phase 8b's per-fold classifiers — see D4 classifier-shift "
            "caveat in pre-reg. Applied to Prudnik via DLC ear-keypoint crop + "
            "V-JEPA-2 → this classifier → sigmoid(score/T_median=0.494)."
        ),
    }

    # Reproducibility verification — bit-exact reload + manual predict
    print("[step1.5] Saving + verifying bit-exact reproducibility…")
    OUT_JSON.write_text(json.dumps(params, indent=2))

    reload = json.loads(OUT_JSON.read_text())
    mean_arr = np.array(reload["scaler_mean"])
    scale_arr = np.array(reload["scaler_scale"])
    coef_arr = np.array(reload["ridge_coef"])
    intercept = reload["ridge_intercept"]

    # Test on first 5 clips
    print("[step1.5] Bit-exact check (first 5 clips):")
    max_delta = 0.0
    for idx in range(min(5, X.shape[0])):
        sklearn_score = float(clf.decision_function(scaler.transform(X[idx:idx + 1]))[0])
        manual_score = manual_decision_function(
            X[idx], mean_arr, scale_arr, coef_arr, intercept
        )
        delta = abs(sklearn_score - manual_score)
        max_delta = max(max_delta, delta)
        marker = "✓" if delta < 1e-10 else "✗"
        print(f"  {marker} {filenames[idx]:<30}  sklearn={sklearn_score:.6f}  "
              f"manual={manual_score:.6f}  |Δ|={delta:.2e}")

    bit_exact = max_delta < 1e-10
    print(f"\n[step1.5] Max |Δ| across 5 sample clips: {max_delta:.2e}  "
          f"{'✓ bit-exact (≤1e-10)' if bit_exact else '✗ NOT bit-exact'}")

    if not bit_exact:
        print("[ERROR] Reproducibility check FAILED — JSON dump cannot bit-exact "
              "reproduce sklearn output. Investigate before proceeding to Step 1.",
              file=sys.stderr)
        return 2

    print()
    print("=" * 70)
    print("[step1.5] Deployed classifier saved to:")
    print(f"  {OUT_JSON.relative_to(POC.parent)}")
    print(f"  ({OUT_JSON.stat().st_size / 1024:.1f} KB; "
          f"{params['n_features']} feature weights + scaler params)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
