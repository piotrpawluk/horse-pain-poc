#!/usr/bin/env python3
"""Phase 6 (b) LOSO runner — face-bbox-positioned crop variant.

Calls the canonical `run_loso(...)` from `tools/eye_loso_lr_phase5.py`
with Phase 6 (b) parameters locked in
`outputs/track_b_phase6b_preregistration.md`:

- Embeddings: outputs/vjepa2_embeddings_phase6b_m15.npz
- Labels: outputs/eye_verification_clips.txt (Set B = Phase 5 primary's
  training labels)
- Output: outputs/eye_loso_results_phase6b.json
- Instrumentation: factor-d, paired-DeLong vs Phase 3, permutation
  (matches Phase 5 primary's instrumentation for apples-to-apples
  comparison; paired-DeLong vs Phase 5 primary is computed by the
  diagnostic step, not here).
"""

from __future__ import annotations

import sys
from pathlib import Path

POC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_DIR / "tools"))

from eye_loso_lr_phase5 import (  # noqa: E402
    run_loso, LABELS_ORIGINAL,
)

EMB_PATH = POC_DIR / "outputs" / "vjepa2_embeddings_phase6b_m15.npz"
OUT_PATH = POC_DIR / "outputs" / "eye_loso_results_phase6b.json"


def main() -> int:
    if not EMB_PATH.exists():
        print(f"[ERROR] embeddings not found: {EMB_PATH}",
              file=sys.stderr)
        return 1
    r = run_loso(
        EMB_PATH, LABELS_ORIGINAL, OUT_PATH, "phase6b_face_bbox_m15",
        do_factor_d=True,
        do_paired_vs_phase3=True,
        do_permutation=True,
    )

    print()
    print("=" * 70)
    print("PHASE 6 (b) — face-bbox m=15 result vs locked gates")
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
    print("Locked gates:")
    g1 = auc >= 0.70
    print(f"  G1 (AUC >= 0.70):           "
          f"{'PASS' if g1 else 'FAIL'} ({auc:.4f} {'≥' if g1 else '<'} 0.70)")
    print("  G2 (paired vs Phase 5 + AUC ≥ 0.7485): computed in diagnostic")
    print()
    print(f"Result file: {OUT_PATH.relative_to(POC_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
