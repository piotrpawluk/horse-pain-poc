#!/usr/bin/env python3
"""Phase 7 LOSO runner — DLC keypoint-anchored crop variant.

Calls canonical `run_loso(...)` from `tools/eye_loso_lr_phase5.py` with
locked Phase 7 parameters:

- Embeddings: outputs/vjepa2_embeddings_eye_v4.npz (Phase 7 v4 crops)
- Labels: outputs/eye_verification_clips.txt (Set B = Phase 5 primary's
  training labels = locked reference per Stage 1 §4 and §8)
- Output: outputs/eye_loso_results_phase7.json
- Instrumentation matches Phase 5 primary + Phase 6(b): factor-d,
  paired-DeLong vs Phase 3, permutation. Paired DeLong vs Phase 5
  primary is computed by the diagnostic step.
"""

from __future__ import annotations

import sys
from pathlib import Path

POC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_DIR / "tools"))

from eye_loso_lr_phase5 import (  # noqa: E402
    run_loso, LABELS_ORIGINAL,
)

EMB_PATH = POC_DIR / "outputs" / "vjepa2_embeddings_eye_v4.npz"
OUT_PATH = POC_DIR / "outputs" / "eye_loso_results_phase7.json"


def main() -> int:
    if not EMB_PATH.exists():
        print(f"[ERROR] embeddings not found: {EMB_PATH}",
              file=sys.stderr)
        return 1
    r = run_loso(
        EMB_PATH, LABELS_ORIGINAL, OUT_PATH, "phase7_dlc_keypoint",
        do_factor_d=True,
        do_paired_vs_phase3=True,
        do_permutation=True,
    )

    print()
    print("=" * 70)
    print("PHASE 7 — DLC keypoint-anchored crop result vs locked gates")
    print("=" * 70)
    auc = r["pooled_auc"]
    boot = r["auc_95_ci_subject_bootstrap"]
    delta_p3 = r["delta_vs_phase3"]
    perm_p = r.get("p_value_permutation")
    print(f"  Pooled AUC:                 {auc:.4f}")
    print(f"  Subject-bootstrap 95% CI:   [{boot[0]:.4f}, {boot[1]:.4f}]")
    print(f"  Δ vs Phase 3 (0.6813):      {delta_p3:+.4f}")
    perm_str = f"{perm_p:.4f}" if perm_p is not None else "—"
    print(f"  Permutation p (vs chance):  {perm_str}")
    print()
    print("Locked gates (per Stage 1 §8):")
    g1 = auc >= 0.70
    g2_auc_floor = 0.7485  # Phase 5 primary 0.7985 - 0.05 protection floor
    g2_auc = auc >= g2_auc_floor
    print(f"  G1 (AUC >= 0.70):                       "
          f"{'PASS' if g1 else 'FAIL'} ({auc:.4f} {'≥' if g1 else '<'} 0.70)")
    print(f"  G2 load-bearing (AUC >= {g2_auc_floor:.4f}): "
          f"{'PASS' if g2_auc else 'FAIL'}")
    print("  G2 supportive (paired DeLong vs Phase 5): computed in diagnostic")
    print("  G3 reportable (median IoU vs Phase 5):    computed in diagnostic")
    print()
    print(f"Result file: {OUT_PATH.relative_to(POC_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
