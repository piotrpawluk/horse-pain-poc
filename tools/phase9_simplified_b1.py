#!/usr/bin/env python3
"""Phase 9 — simplified-B1 long-form aggregation + G1 sanity-check redesign.

Operates strictly downstream of Phase 8c output JSONs. No new training,
no new compute on the classifier or calibration.

All design decisions locked in:
    outputs/track_b_phase9_preregistration.md

Decisions referenced inline:
  D1: max-window-prob + threshold aggregation; session = Σ_k present_k;
      flag if ≥ 8/24 per Dyson 2018 + Dyson & Pollard 2023
      (presence/absence rule, see v2/research/dyson_scoring_check.md).
  D2: each RME clip = one window for the K = 1 demonstration.
  D3: strict downstream consumption of Phase 8c outputs (no
      classifier or calibration recomputation).
  D4: parameterised compute_session_score() for arbitrary K with
      synthetic K = 24 unit tests covering 4 boundary cases.
  D5: G1 redesign — G1a per-source AUC invariance (bit-exact) +
      G1b bounded pooled drift ratio (k = 0.04 constant).
  D6: per-source FPR reportable diagnostic; no per-source τ_S.

Outputs:
  outputs/phase9_simplified_b1_results.json   pooled + per-source + gates
  outputs/phase9_audit_extras.json            per-clip presence + session
  outputs/phase9_per_source_confusion.png     12-panel confusion grid
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


POC_DIR = Path(__file__).resolve().parent.parent
PHASE8C_RESULTS = POC_DIR / "outputs" / "phase8c_calibration_results.json"
PHASE8C_EXTRAS = POC_DIR / "outputs" / "phase8c_audit_extras.json"
OUT_RESULTS = POC_DIR / "outputs" / "phase9_simplified_b1_results.json"
OUT_EXTRAS = POC_DIR / "outputs" / "phase9_audit_extras.json"
OUT_FIG = POC_DIR / "outputs" / "phase9_per_source_confusion.png"

# Locked from pre-reg
FPR_TARGET = 0.05
FPR_SANITY_TOL = 0.005  # G3 threshold
N_BEHAVIORS_RHPE = 24
SESSION_THRESHOLD = 8
G1B_K_CONSTANT = 0.04
G1A_INVARIANCE_TOL = 1e-10
DEFAULT_BEHAVIORS = ("ear",)


# --------------------------------------------------------------------- #
# D3 — Phase 8c output loaders (strictly downstream, no recomputation)
# --------------------------------------------------------------------- #


def load_phase8c_outputs() -> dict:
    """Load Phase 8c calibration results + audit extras. No transformation."""
    if not PHASE8C_RESULTS.exists():
        print(
            f"[ERROR] Phase 8c results not found: {PHASE8C_RESULTS}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not PHASE8C_EXTRAS.exists():
        print(
            f"[ERROR] Phase 8c extras not found: {PHASE8C_EXTRAS}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    results = json.loads(PHASE8C_RESULTS.read_text())
    extras = json.loads(PHASE8C_EXTRAS.read_text())
    return {
        "per_clip": extras["per_clip"],
        "tau_ear": results["operating_point"]["tau_ear"],
        "T_per_src": results["per_fold_T"],
        "phase8b_pooled_auc": results["phase8b_pooled_auc_reference"],
    }


# --------------------------------------------------------------------- #
# D1 + D4 — aggregation rule and parameterised session-score API
# --------------------------------------------------------------------- #


def compute_session_score(
    presence_vector: list[int] | np.ndarray,
    threshold: int = SESSION_THRESHOLD,
) -> tuple[int, bool]:
    """Locked aggregation: session_score = sum(present_k); flag if ≥ threshold.

    Per Dyson 2018 J Vet Behav 23:47-57 + Dyson & Pollard 2023 Animals
    13(12):1940 — presence/absence per behavior, session score is count
    of distinct behaviors observed, threshold ≥ 8/24 flags pain.

    For K = 1 demonstration session_score ∈ {0, 1}; flag never fires.
    Parameterised to handle arbitrary K ≥ 1.
    """
    score = int(np.sum(presence_vector))
    flag = score >= threshold
    return score, flag


def compute_present_k_for_ear(
    per_clip: list[dict], tau_ear: float
) -> list[dict]:
    """D1 + D2 — per-clip present_ear via max-window-prob + threshold.

    For K = 1 RME demonstration, each clip = one window (D2). The
    max-window-prob is simply prob_post_cal. present_ear = 1 iff
    prob_post_cal ≥ τ_ear.

    Returns per_clip extended with present_ear, session_score (= present_ear
    at K = 1), and would_session_flag (always False at K = 1 since
    session_score ∈ {0, 1} < 8).
    """
    out = []
    for rec in per_clip:
        present_ear = int(rec["prob_post_cal"] >= tau_ear)
        score, flag = compute_session_score(
            [present_ear], threshold=SESSION_THRESHOLD
        )
        out.append(
            {
                **rec,
                "present_ear": present_ear,
                "session_score_at_K_eq_1": score,
                "would_session_flag_at_K_eq_1": flag,
            }
        )
    return out


# --------------------------------------------------------------------- #
# D6 — per-source confusion + pooled FPR/TPR diagnostic
# --------------------------------------------------------------------- #


def per_source_confusion(per_clip_with_presence: list[dict]) -> dict:
    """Per-source 2×2 confusion at τ_ear: predicted present vs true label.

    Returns dict mapping source → {n, n_pos, n_neg, TP, FP, TN, FN,
    FPR, TPR, accuracy}.
    """
    out: dict[str, dict] = {}
    by_src: dict[str, list[dict]] = {}
    for rec in per_clip_with_presence:
        by_src.setdefault(rec["source"], []).append(rec)
    for src in sorted(by_src, key=lambda s: int(s[1:])):
        clips = by_src[src]
        n = len(clips)
        n_pos = sum(1 for r in clips if r["label"] == 1)
        n_neg = n - n_pos
        TP = sum(1 for r in clips if r["label"] == 1 and r["present_ear"] == 1)
        FN = sum(1 for r in clips if r["label"] == 1 and r["present_ear"] == 0)
        FP = sum(1 for r in clips if r["label"] == 0 and r["present_ear"] == 1)
        TN = sum(1 for r in clips if r["label"] == 0 and r["present_ear"] == 0)
        out[src] = {
            "n": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
            "FPR": (FP / n_neg) if n_neg > 0 else float("nan"),
            "TPR": (TP / n_pos) if n_pos > 0 else float("nan"),
            "accuracy": (TP + TN) / n if n > 0 else float("nan"),
        }
    return out


def pooled_fpr_tpr(per_clip_with_presence: list[dict]) -> dict:
    """Pooled FPR/TPR across all 283 clips at τ_ear (G3 sanity)."""
    labels = np.array([r["label"] for r in per_clip_with_presence])
    presents = np.array([r["present_ear"] for r in per_clip_with_presence])
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    TP = int(((labels == 1) & (presents == 1)).sum())
    FP = int(((labels == 0) & (presents == 1)).sum())
    TN = int(((labels == 0) & (presents == 0)).sum())
    FN = int(((labels == 1) & (presents == 0)).sum())
    return {
        "n": int(len(labels)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "FPR": FP / n_neg if n_neg > 0 else float("nan"),
        "TPR": TP / n_pos if n_pos > 0 else float("nan"),
        "accuracy": (TP + TN) / len(labels) if len(labels) > 0 else float("nan"),
    }


# --------------------------------------------------------------------- #
# D5 — G1 redesign: G1a per-source AUC invariance + G1b bounded drift
# --------------------------------------------------------------------- #


def g1a_per_source_auc_invariance(per_clip: list[dict]) -> dict:
    """G1a — for each source S, AUC(pre_cal[S]) = AUC(post_cal[S]) bit-exact.

    Per-source T preserves within-source rank ordering by construction
    (T > 0 monotonic per fold). G1a confirms this on Phase 8c data and
    establishes the gate machinery for future per-source calibration
    phases.
    """
    by_src: dict[str, list[dict]] = {}
    for rec in per_clip:
        by_src.setdefault(rec["source"], []).append(rec)

    per_source = {}
    all_invariant = True
    for src in sorted(by_src, key=lambda s: int(s[1:])):
        clips = by_src[src]
        labels = np.array([r["label"] for r in clips])
        pre = np.array([r["prob_pre_cal_T1"] for r in clips])
        post = np.array([r["prob_post_cal"] for r in clips])
        n_pos = int(labels.sum())
        n_neg = int(len(labels) - n_pos)
        if n_pos == 0 or n_neg == 0:
            per_source[src] = {
                "n": len(clips),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "auc_pre_cal": None,
                "auc_post_cal": None,
                "delta": None,
                "invariant": None,
                "note": "AUC undefined (degenerate labels)",
            }
            continue
        auc_pre = float(roc_auc_score(labels, pre))
        auc_post = float(roc_auc_score(labels, post))
        delta = abs(auc_post - auc_pre)
        invariant = delta <= G1A_INVARIANCE_TOL
        if not invariant:
            all_invariant = False
        per_source[src] = {
            "n": len(clips),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "auc_pre_cal": auc_pre,
            "auc_post_cal": auc_post,
            "delta": delta,
            "invariant": invariant,
        }
    return {
        "per_source": per_source,
        "all_12_invariant": all_invariant,
        "tolerance": G1A_INVARIANCE_TOL,
        "verdict": "PASS" if all_invariant else "FAIL",
    }


def g1b_pooled_drift(per_clip: list[dict], T_per_src: dict) -> dict:
    """G1b — bounded pooled AUC drift conditional on T variance.

    Per pre-reg D5 (amended):
        ΔAUC_pooled        = |AUC(pre_cal_pooled) − AUC(post_cal_pooled)|
        range(T_per_source) = max(T_S) − min(T_S) over 12 LOSO folds
        Locked invariant:   ΔAUC_pooled ≤ k × range(T_per_source) with k = 0.04
        Reportable:         ratio = ΔAUC_pooled / range(T_per_source)
        FLAG if ratio > 0.04 (cross-source rank-shuffle exceeds Phase 8c baseline)
    """
    labels = np.array([r["label"] for r in per_clip])
    pre = np.array([r["prob_pre_cal_T1"] for r in per_clip])
    post = np.array([r["prob_post_cal"] for r in per_clip])
    auc_pre_pooled = float(roc_auc_score(labels, pre))
    auc_post_pooled = float(roc_auc_score(labels, post))
    delta_auc_pooled = abs(auc_post_pooled - auc_pre_pooled)

    T_values = list(T_per_src.values())
    T_max = float(max(T_values))
    T_min = float(min(T_values))
    T_range = T_max - T_min

    if T_range == 0:
        ratio = float("inf") if delta_auc_pooled > 0 else 0.0
    else:
        ratio = delta_auc_pooled / T_range

    bound = G1B_K_CONSTANT * T_range
    invariant_holds = delta_auc_pooled <= bound
    flag = ratio > G1B_K_CONSTANT
    return {
        "auc_pre_cal_pooled": auc_pre_pooled,
        "auc_post_cal_pooled": auc_post_pooled,
        "delta_auc_pooled": delta_auc_pooled,
        "T_max": T_max,
        "T_min": T_min,
        "T_range": T_range,
        "k_constant": G1B_K_CONSTANT,
        "bound": bound,
        "ratio": ratio,
        "invariant_holds": invariant_holds,
        "flag": flag,
        "verdict": "FLAG" if flag else "WITHIN_BOUND",
    }


# --------------------------------------------------------------------- #
# G5 — synthetic K = 24 unit tests (D4)
# --------------------------------------------------------------------- #


def synthetic_k24_unit_tests() -> dict:
    """4 boundary cases for compute_session_score at K = 24.

    Per pre-reg D4 + Test 6 + G5:
      all-zero     → (0, False)   sub-threshold
      seven        → (7, False)   just-below-threshold; OFF-BY-ONE GUARD
      exactly-eight → (8, True)   at threshold (≥ direction)
      all-one      → (24, True)   super-threshold
    """
    cases = []
    # all-zero
    presence = [0] * N_BEHAVIORS_RHPE
    score, flag = compute_session_score(presence)
    cases.append(
        {
            "name": "all-zero",
            "presence_sum": 0,
            "expected_score": 0,
            "expected_flag": False,
            "observed_score": score,
            "observed_flag": flag,
            "pass": (score == 0) and (flag is False),
        }
    )
    # seven (off-by-one guard)
    presence = [1] * 7 + [0] * (N_BEHAVIORS_RHPE - 7)
    score, flag = compute_session_score(presence)
    cases.append(
        {
            "name": "seven (off-by-one guard)",
            "presence_sum": 7,
            "expected_score": 7,
            "expected_flag": False,
            "observed_score": score,
            "observed_flag": flag,
            "pass": (score == 7) and (flag is False),
        }
    )
    # exactly-eight (at threshold)
    presence = [1] * 8 + [0] * (N_BEHAVIORS_RHPE - 8)
    score, flag = compute_session_score(presence)
    cases.append(
        {
            "name": "exactly-eight (at threshold)",
            "presence_sum": 8,
            "expected_score": 8,
            "expected_flag": True,
            "observed_score": score,
            "observed_flag": flag,
            "pass": (score == 8) and (flag is True),
        }
    )
    # all-one
    presence = [1] * N_BEHAVIORS_RHPE
    score, flag = compute_session_score(presence)
    cases.append(
        {
            "name": "all-one (super-threshold)",
            "presence_sum": 24,
            "expected_score": 24,
            "expected_flag": True,
            "observed_score": score,
            "observed_flag": flag,
            "pass": (score == 24) and (flag is True),
        }
    )

    all_pass = all(c["pass"] for c in cases)
    return {
        "cases": cases,
        "all_pass": all_pass,
        "verdict": "PASS" if all_pass else "FAIL",
        "n_behaviors": N_BEHAVIORS_RHPE,
        "threshold": SESSION_THRESHOLD,
    }


# --------------------------------------------------------------------- #
# Confusion-matrix plot (12-panel grid)
# --------------------------------------------------------------------- #


def per_source_confusion_plot(confusions: dict, out_path: Path) -> None:
    """3×4 grid of per-source 2×2 confusion matrices at τ_ear."""
    sources = sorted(confusions, key=lambda s: int(s[1:]))
    fig, axes = plt.subplots(3, 4, figsize=(13, 9))
    for idx, src in enumerate(sources):
        ax = axes[idx // 4][idx % 4]
        c = confusions[src]
        matrix = np.array(
            [
                [c["TN"], c["FP"]],
                [c["FN"], c["TP"]],
            ]
        )
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max() or 1)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(matrix[i, j]),
                    ha="center",
                    va="center",
                    color="black" if matrix[i, j] < matrix.max() / 2 else "white",
                    fontsize=14,
                    fontweight="bold",
                )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["not\npresent", "present"], fontsize=8)
        ax.set_yticklabels(["bg", "action"], fontsize=8)
        ax.set_xlabel("predicted", fontsize=9)
        ax.set_ylabel("true", fontsize=9)
        fpr = c["FPR"]
        tpr = c["TPR"]
        ax.set_title(
            f"{src} (n={c['n']}, +/−={c['n_pos']}/{c['n_neg']})\n"
            f"FPR={fpr:.3f}  TPR={tpr:.3f}",
            fontsize=9,
        )

    # hide any unused subplots
    for idx in range(len(sources), 12):
        axes[idx // 4][idx % 4].axis("off")

    fig.suptitle(
        "Phase 9 — per-source confusion at τ_ear  "
        "(RME ear movement, n=283, 12 sources, τ_ear from Phase 8c D4)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- #
# G2 sanity — present_ear must match phase8c above_tau_ear bit-exact
# --------------------------------------------------------------------- #


def g2_phase8c_consistency_check(per_clip_with_presence: list[dict]) -> dict:
    """G2 — verify per-clip present_ear matches Phase 8c above_tau_ear bit-exact.

    The phase8c_audit_extras.json computes above_tau_ear using the same
    rule (prob_post_cal ≥ τ_ear). Phase 9's present_ear must agree on
    all 283 clips — any mismatch indicates a tool bug.
    """
    mismatches = []
    for rec in per_clip_with_presence:
        if int(rec["above_tau_ear"]) != int(rec["present_ear"]):
            mismatches.append(
                {
                    "clip": rec["clip"],
                    "source": rec["source"],
                    "prob_post_cal": rec["prob_post_cal"],
                    "above_tau_ear": rec["above_tau_ear"],
                    "present_ear": rec["present_ear"],
                }
            )
    return {
        "n_clips_checked": len(per_clip_with_presence),
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:10],  # cap reported sample
        "verdict": "PASS" if not mismatches else "FAIL",
    }


# --------------------------------------------------------------------- #
# Main — orchestration
# --------------------------------------------------------------------- #


def main() -> int:
    print(
        "[phase9] loading Phase 8c outputs "
        f"({PHASE8C_RESULTS.relative_to(POC_DIR)}, "
        f"{PHASE8C_EXTRAS.relative_to(POC_DIR)})",
        flush=True,
    )
    p8c = load_phase8c_outputs()
    tau_ear = p8c["tau_ear"]
    T_per_src = p8c["T_per_src"]
    per_clip_raw = p8c["per_clip"]
    print(
        f"[phase9] n_clips={len(per_clip_raw)}, τ_ear={tau_ear:.4f}, "
        f"12 LOSO T values loaded",
        flush=True,
    )

    # G5 — synthetic K = 24 unit tests (run early so threshold logic
    # is verified before applying compute_session_score to real data)
    print("[phase9] G5 — synthetic K = 24 unit tests (D4)...", flush=True)
    g5 = synthetic_k24_unit_tests()
    for c in g5["cases"]:
        marker = "✓" if c["pass"] else "✗"
        print(
            f"  {marker} {c['name']}: score={c['observed_score']} "
            f"flag={c['observed_flag']}",
            flush=True,
        )
    if not g5["all_pass"]:
        print(
            "[ERROR] G5 unit tests FAILED — threshold logic bug. Halting.",
            file=sys.stderr,
        )
        return 2

    # D1 + D2 — compute present_ear per clip via max-window-prob threshold
    print(
        "[phase9] computing present_ear via max-window-prob ≥ τ_ear (D1+D2)...",
        flush=True,
    )
    per_clip = compute_present_k_for_ear(per_clip_raw, tau_ear)

    # G2 — sanity check: present_ear must match Phase 8c above_tau_ear bit-exact
    print(
        "[phase9] G2 — Phase 8c consistency check (present_ear vs above_tau_ear)...",
        flush=True,
    )
    g2 = g2_phase8c_consistency_check(per_clip)
    print(
        f"  G2: {g2['verdict']}  ({g2['n_clips_checked']} clips, "
        f"{g2['n_mismatches']} mismatches)",
        flush=True,
    )
    if g2["verdict"] == "FAIL":
        print(
            "[ERROR] G2 consistency check FAILED — present_ear ≠ above_tau_ear. "
            "Tool bug or τ_ear mismatch. Halting.",
            file=sys.stderr,
        )
        return 2

    # D6 + G3 — per-source + pooled FPR/TPR
    print("[phase9] D6 — per-source confusion at τ_ear...", flush=True)
    per_src = per_source_confusion(per_clip)
    pooled = pooled_fpr_tpr(per_clip)
    fpr_pooled_drift = abs(pooled["FPR"] - FPR_TARGET)
    g3_pass = fpr_pooled_drift <= FPR_SANITY_TOL
    print(
        f"  Pooled FPR={pooled['FPR']:.4f} (target {FPR_TARGET}, "
        f"|drift|={fpr_pooled_drift:.4f}, tol {FPR_SANITY_TOL}); "
        f"G3 {'PASS' if g3_pass else 'FAIL'}",
        flush=True,
    )

    # G4 / D5 G1a — per-source AUC invariance bit-exact (HALT on FAIL per pre-reg amendment)
    print(
        "[phase9] G4 (D5 G1a) — per-source AUC invariance check...", flush=True
    )
    g1a = g1a_per_source_auc_invariance(per_clip_raw)
    print(
        f"  G1a: {g1a['verdict']}  "
        f"({sum(1 for s in g1a['per_source'].values() if s.get('invariant')) }/"
        f"{len(g1a['per_source'])} sources invariant)",
        flush=True,
    )
    if g1a["verdict"] == "FAIL":
        print(
            "[ERROR] G4 (G1a per-source AUC invariance) FAILED — "
            "per-source rank ordering not preserved bit-exact. "
            "Downstream outputs NOT written. "
            "Investigate tool bug or per-source non-monotonicity.",
            file=sys.stderr,
        )
        return 2

    # D5 G1b — bounded pooled drift
    print("[phase9] D5 G1b — bounded pooled drift ratio...", flush=True)
    g1b = g1b_pooled_drift(per_clip_raw, T_per_src)
    print(
        f"  ΔAUC_pooled={g1b['delta_auc_pooled']:.6f}  "
        f"range(T)={g1b['T_range']:.4f}  "
        f"ratio={g1b['ratio']:.4f}  bound (k×range)={g1b['bound']:.4f}  "
        f"verdict {g1b['verdict']}",
        flush=True,
    )

    # Confusion matrix plot
    print(
        f"[phase9] writing per-source confusion plot to "
        f"{OUT_FIG.relative_to(POC_DIR)}",
        flush=True,
    )
    per_source_confusion_plot(per_src, OUT_FIG)

    # Assemble results JSON
    g1_verdict = "PASS"  # tool execution reached this point
    g2_verdict = g2["verdict"]
    g3_verdict = "PASS" if g3_pass else "FAIL"
    g4_verdict = g1a["verdict"]
    g5_verdict = g5["verdict"]

    results = {
        "mode": "phase9_simplified_b1",
        "preregistration": "outputs/track_b_phase9_preregistration.md",
        "inputs": {
            "phase8c_calibration_results": str(
                PHASE8C_RESULTS.relative_to(POC_DIR)
            ),
            "phase8c_audit_extras": str(PHASE8C_EXTRAS.relative_to(POC_DIR)),
        },
        "tau_ear_consumed": tau_ear,
        "n_clips": len(per_clip),
        "n_sources": len(per_src),
        "scoring_rule_source": (
            "Dyson 2018 J Vet Behav 23:47-57 + Dyson & Pollard 2023 "
            "Animals 13(12):1940 — presence/absence per behavior, "
            "session score = sum of presences, threshold ≥ 8/24. "
            "See v2/research/dyson_scoring_check.md"
        ),
        "gates": {
            "G1_tool_execution": g1_verdict,
            "G2_phase8c_consistency": g2_verdict,
            "G3_pooled_FPR_sanity": g3_verdict,
            "G4_g1a_per_source_auc_invariance": g4_verdict,
            "G5_synthetic_unit_tests": g5_verdict,
        },
        "pooled_diagnostic": {
            **pooled,
            "FPR_target": FPR_TARGET,
            "FPR_drift": fpr_pooled_drift,
            "FPR_sanity_tol": FPR_SANITY_TOL,
        },
        "per_source_confusion": per_src,
        "g1a_per_source_auc_invariance": g1a,
        "g1b_pooled_drift": g1b,
        "g5_synthetic_unit_tests": g5,
        "scaffold_for_K_eq_24": {
            "api_signature": (
                "compute_session_score(presence_vector: list[int] | np.ndarray, "
                "threshold: int = 8) -> tuple[int, bool]"
            ),
            "exercised_at_K_eq_1": True,
            "exercised_with_real_data_at_K_gt_1": False,
            "validated_at_K_eq_24_via_synthetic_unit_tests_only": True,
            "note": (
                "Mechanism-only validation at K = 24; clinical claim is "
                "blocked on multi-behavior probe development (Phase 10+)."
            ),
        },
        "limitations": [
            "L1: single-behavior scope (K = 1) — ≥ 8/24 cannot fire.",
            "L2: single-window-per-session — RME clip structure.",
            "L3: independence assumption carries forward from Phase 8c.",
            "L4: single-observer label noise carries forward.",
            "L5: k = 0.04 is N = 1 phase empirical anchor (G1b reportable).",
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUT_RESULTS.write_text(json.dumps(results, indent=2))
    print(f"[phase9] wrote {OUT_RESULTS.relative_to(POC_DIR)}", flush=True)

    # Per-clip extras
    extras = {
        "mode": "phase9_audit_extras",
        "tau_ear": tau_ear,
        "scoring_threshold": SESSION_THRESHOLD,
        "per_clip": [
            {
                "clip": r["clip"],
                "source": r["source"],
                "label": int(r["label"]),
                "prob_post_cal": float(r["prob_post_cal"]),
                "above_tau_ear": int(r["above_tau_ear"]),
                "present_ear": int(r["present_ear"]),
                "session_score_at_K_eq_1": int(r["session_score_at_K_eq_1"]),
                "would_session_flag_at_K_eq_1": bool(
                    r["would_session_flag_at_K_eq_1"]
                ),
            }
            for r in per_clip
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUT_EXTRAS.write_text(json.dumps(extras, indent=2))
    print(f"[phase9] wrote {OUT_EXTRAS.relative_to(POC_DIR)}", flush=True)

    # Summary
    print()
    print("=" * 72)
    print("PHASE 9 — simplified-B1 + G1 sanity check redesign")
    print("=" * 72)
    print(f"  τ_ear (from Phase 8c D4):        {tau_ear:.4f}")
    print(f"  n_clips / n_sources:             {len(per_clip)} / {len(per_src)}")
    print()
    print("  Pooled diagnostic at τ_ear:")
    print(
        f"    FPR={pooled['FPR']:.4f} (target {FPR_TARGET})  "
        f"TPR={pooled['TPR']:.4f}  "
        f"accuracy={pooled['accuracy']:.4f}"
    )
    print(
        f"    TP={pooled['TP']} FP={pooled['FP']} "
        f"TN={pooled['TN']} FN={pooled['FN']}"
    )
    print()
    print("  Gate verdicts:")
    print(f"    G1 tool execution:              {g1_verdict}")
    print(f"    G2 Phase 8c consistency:        {g2_verdict}")
    print(f"    G3 pooled FPR sanity:           {g3_verdict}  "
          f"(|drift|={fpr_pooled_drift:.4f}, tol {FPR_SANITY_TOL})")
    print(f"    G4 G1a per-source AUC inv.:     {g4_verdict}  "
          f"(12/12 expected; tol {G1A_INVARIANCE_TOL})")
    print(f"    G5 synthetic K=24 unit tests:   {g5_verdict}  "
          f"({sum(1 for c in g5['cases'] if c['pass'])}/{len(g5['cases'])})")
    print()
    print("  G1b reportable (D5 amended):")
    print(
        f"    ΔAUC_pooled={g1b['delta_auc_pooled']:.6f}  "
        f"range(T_per_source)={g1b['T_range']:.4f}"
    )
    print(
        f"    ratio={g1b['ratio']:.4f}  "
        f"k_bound={g1b['k_constant']}  "
        f"verdict {g1b['verdict']}"
    )
    print()
    print("  ⚠ Limitations carry forward — see results JSON for full list.")
    print("    Single-behavior K=1 demonstration; ≥8/24 cannot fire.")
    print()
    print("Result files:")
    print(f"  {OUT_RESULTS.relative_to(POC_DIR)}")
    print(f"  {OUT_EXTRAS.relative_to(POC_DIR)}")
    print(f"  {OUT_FIG.relative_to(POC_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
