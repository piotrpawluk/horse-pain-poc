#!/usr/bin/env python3
"""Phase 8b Step 7 — LOSO runner for ear-cropped V-JEPA-2 features.

Wraps canonical run_loso(...) from Phase 5 module with Phase 8b-specific
inputs:
- Embeddings: outputs/vjepa2_embeddings_ear_v4.npz (Step 6 output)
- Labels: parsed from RME train/val/test CSVs (action=1, background=0)
- Output: outputs/eye_loso_results_phase8b.json
- Instrumentation: factor-d, paired-DeLong vs Phase 3 (eye), permutation
  (Phase 5 module conventions). Paired DeLong vs whole-frame RME baseline
  is computed by phase8b_diagnostic.py per locked test hierarchy.

Note: this module's labels file (outputs/phase8b_rme_labels.txt) is
generated here from RME CSVs in the format the Phase 5 module expects:
  <abs_path> - <observation> - <ACTION|BACKGROUND>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

POC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_DIR / "tools"))
from eye_loso_lr_phase5 import run_loso  # noqa: E402

EMB_PATH = POC_DIR / "outputs" / "vjepa2_embeddings_ear_v4.npz"
RME_DATA = POC_DIR / "vendor" / "ReadMyEars_Dataset" / "data"
LABELS_PATH = POC_DIR / "outputs" / "phase8b_rme_labels.txt"
OUT_PATH = POC_DIR / "outputs" / "eye_loso_results_phase8b.json"


def build_labels_file():
    """Generate Phase-5-module-compatible labels file from RME CSVs."""
    dfs = []
    for split in ["train", "val", "test"]:
        path = RME_DATA / f"{split}.csv"
        if path.exists():
            df = pd.read_csv(path)
            dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    full["basename"] = full["video"].apply(lambda v: Path(v).name)
    full = full.drop_duplicates(subset=["basename"])
    lines = ["# Phase 8b RME labels — generated from train/val/test CSVs",
              "# Format: <abs_path> - <split-source> - <ACTION|BACKGROUND>"]
    for _, row in full.iterrows():
        abs_path = str(RME_DATA / row["video"])
        verdict = "ACTION" if row["label"] == "action" else "BACKGROUND"
        lines.append(f"{abs_path} - rme_{row['video']} - {verdict}")
    LABELS_PATH.write_text("\n".join(lines) + "\n")


def main() -> int:
    if not EMB_PATH.exists():
        print(f"[ERROR] embeddings not found: {EMB_PATH}",
              file=sys.stderr)
        return 1
    build_labels_file()
    print(f"[phase8b loso] labels file built at "
          f"{LABELS_PATH.relative_to(POC_DIR)}", flush=True)

    r = run_loso(
        EMB_PATH, LABELS_PATH, OUT_PATH, "phase8b_ear_dlc",
        do_factor_d=False,  # factor-d is Phase 5 specific; not relevant here
        do_paired_vs_phase3=False,  # Phase 3 is eye-region, not ear; skip
        do_permutation=True,
    )

    print()
    print("=" * 70)
    print("PHASE 8b — DLC ear-keypoint-cropped V-JEPA-2 LOSO")
    print("=" * 70)
    auc = r["pooled_auc"]
    boot = r["auc_95_ci_subject_bootstrap"]
    perm_p = r.get("p_value_permutation")
    print(f"  Pooled AUC:                 {auc:.4f}")
    print(f"  Subject-bootstrap 95% CI:   [{boot[0]:.4f}, {boot[1]:.4f}]")
    perm_str = f"{perm_p:.4f}" if perm_p is not None else "—"
    print(f"  Permutation p (vs chance):  {perm_str}")
    print()
    print("Locked gates (per Phase 8b §8):")
    g1_strong = auc >= 0.80
    g2_modest = 0.65 <= auc < 0.80
    g3_fails = auc < 0.65
    print(f"  G1 (AUC ≥ 0.80, strong):    "
          f"{'PASS' if g1_strong else 'FAIL'}")
    print(f"  G2 (0.65 ≤ AUC < 0.80, modest): "
          f"{'PASS' if g2_modest else 'FAIL'}")
    print(f"  G3 (AUC < 0.65, fails):     "
          f"{'PASS' if g3_fails else 'FAIL'}")
    print("  G4 (paired DeLong vs whole-frame): computed in diagnostic")
    print()
    print(f"Result file: {OUT_PATH.relative_to(POC_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
