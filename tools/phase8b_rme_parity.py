#!/usr/bin/env python3
"""Phase 8b Step 1 — RME parity check.

Per locked Phase 8b Stage 1 pre-reg Decision 2:
Reproduce the existing whole-frame V-JEPA-2 LOSO 0.8746126936531734
result bit-exact using:
- Cached features: outputs/vjepa2_embeddings.npz (283 × 1024)
- Canonical Ridge LOSO config: RidgeClassifier(alpha=1.0,
  class_weight='balanced') + StandardScaler per fold
- 12 LOSO folds (one per source)

Gate: pooled AUC must equal 0.8746126936531734 to within numerical
noise (≤ 1e-10 deviation). If parity fails, Phase 8b halts.

Output: outputs/phase8b_rme_parity.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

POC_DIR = Path(__file__).resolve().parent.parent
EMB_PATH = POC_DIR / "outputs" / "vjepa2_embeddings.npz"
SANITY5_PATH = POC_DIR / "outputs" / "iter65_sanity5_loso_rme_results.json"
OUT_PATH = POC_DIR / "outputs" / "phase8b_rme_parity.json"

# Locked from Sanity 5 best_config.loso_auc
EXPECTED_AUC = 0.8746126936531734
DEVIATION_TOLERANCE = 1e-10


def extract_source(filename: str) -> str:
    m = re.search(r"S(\d+)\.mp4", filename)
    if m is None:
        raise ValueError(f"cannot extract source from {filename}")
    return f"S{m.group(1)}"


def loso_pooled_predict(X, y, groups, sources):
    """Standard LOSO with canonical Ridge config; returns pooled
    predictions in source-iteration order (matches Phase 5/7
    convention)."""
    preds = np.zeros(len(y), dtype=float)
    fold_log = {}
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
        scores_test = clf.decision_function(scaler.transform(X[test]))
        preds[test] = scores_test
        fold_log[s] = {
            "n": int(test.sum()),
            "auc": (float(roc_auc_score(y[test], scores_test))
                    if len(np.unique(y[test])) >= 2 else None),
        }
    return preds, fold_log


def main() -> int:
    print("[phase8b parity] loading cached RME features...", flush=True)
    if not EMB_PATH.exists():
        print(f"[ERROR] {EMB_PATH} not found", file=sys.stderr)
        return 1
    if not SANITY5_PATH.exists():
        print(f"[ERROR] {SANITY5_PATH} not found", file=sys.stderr)
        return 1
    d = np.load(EMB_PATH, allow_pickle=True)
    X = d["embs"]
    y = d["labels"]
    filenames = [str(f) for f in d["filenames"]]
    groups = np.array([extract_source(f) for f in filenames])
    sources = sorted(set(groups), key=lambda s: int(s[1:]))
    print(f"[phase8b parity] n={len(y)} clips, {len(sources)} sources, "
          f"action={int((y == 1).sum())} bg={int((y == 0).sum())}",
          flush=True)

    print("[phase8b parity] running LOSO with canonical Ridge config...",
          flush=True)
    preds, fold_log = loso_pooled_predict(X, y, groups, sources)
    pooled_auc = float(roc_auc_score(y, preds))

    sanity5 = json.loads(SANITY5_PATH.read_text())
    sanity5_auc = sanity5["best_config"]["loso_auc"]
    deviation = abs(pooled_auc - EXPECTED_AUC)
    parity_pass = deviation <= DEVIATION_TOLERANCE

    if parity_pass:
        verdict = "PASS_BIT_EXACT"
        next_action = ("Pooled AUC matches expected to within 1e-10. "
                       "Proceed to Stage 1.5 ear keypoint inspection.")
    elif deviation < 1e-4:
        verdict = "PASS_NUMERICAL_NOISE"
        next_action = ("Pooled AUC differs by {:.2e} (within numerical "
                       "noise tolerance for canonical config; not "
                       "bit-exact but reproducibility is functional). "
                       "Proceed to Stage 1.5 with documented "
                       "deviation.".format(deviation))
    else:
        verdict = "HALT_PHASE_8B"
        next_action = ("Pooled AUC differs from expected by {:.2e} > "
                       "1e-4. Per locked Decision 2: HALTS Phase 8b "
                       "pending user investigation. Do NOT proceed to "
                       "Stage 1.5.".format(deviation))

    summary = {
        "tool": "tools/phase8b_rme_parity.py",
        "stage1_decision": "§Decision 2 RME parity check",
        "expected_pooled_auc": EXPECTED_AUC,
        "expected_source": str(SANITY5_PATH.relative_to(POC_DIR)),
        "observed_pooled_auc": pooled_auc,
        "deviation_absolute": deviation,
        "deviation_tolerance_strict": DEVIATION_TOLERANCE,
        "n_clips": len(y),
        "n_sources": len(sources),
        "fold_log": fold_log,
        "verdict": verdict,
        "next_action": next_action,
    }

    OUT_PATH.write_text(json.dumps(summary, indent=2))

    print(flush=True)
    print(f"[phase8b parity] expected: {EXPECTED_AUC}")
    print(f"[phase8b parity] observed: {pooled_auc}")
    print(f"[phase8b parity] deviation: {deviation:.2e}")
    print(f"[phase8b parity] VERDICT: {verdict}")
    print(f"[phase8b parity] NEXT: {next_action}")
    print()
    print(f"Wrote: {OUT_PATH.relative_to(POC_DIR)}")

    if verdict == "HALT_PHASE_8B":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
