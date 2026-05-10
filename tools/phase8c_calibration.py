#!/usr/bin/env python3
"""Phase 8c (B3 calibration package) — temperature scaling on Phase 8b RME ear output.

Operates strictly downstream on outputs/eye_loso_results_phase8b.json.
No new training, no new compute on the spine.

All design decisions locked in:
    outputs/track_b_phase8c_preregistration.md

Decisions referenced inline:
  D1: Temperature scaling — p = sigmoid(score / T), single parameter, NLL fit.
  D2: Source-aware calibration LOSO — for each held-out source S, fit T_S on
      the other 11 sources' OOF scores; apply T_S to S's OOF scores.
  D3: ECE (10 equal-frequency bins) + Brier + NLL + reliability diagram +
      per-fold T values.
  D4: τ_ear at FPR=0.05 on negative-source clips (pooled), B=1000 source-
      bootstrap CI on TPR.
  D5: Session-level OP via exact Poisson-binomial CDF (24 Bernoulli convolution)
      + Poisson(λ=1.2) approximation; both reported. Independence assumption
      surfaced inline as load-bearing limitation.

Outputs:
  outputs/phase8c_calibration_results.json   main numerical results + gates
  outputs/phase8c_audit_extras.json          per-clip extras for downstream use
  outputs/phase8c_reliability_diagram.png    2-panel pre/post figure
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive PNG generation
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.stats import poisson  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


POC_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = POC_DIR / "outputs" / "eye_loso_results_phase8b.json"
OUT_RESULTS = POC_DIR / "outputs" / "phase8c_calibration_results.json"
OUT_EXTRAS = POC_DIR / "outputs" / "phase8c_audit_extras.json"
OUT_FIG = POC_DIR / "outputs" / "phase8c_reliability_diagram.png"

# Locked from pre-reg
N_BINS = 10
FPR_TARGET = 0.05
BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 42
N_BEHAVIORS_RHPE = 24
SESSION_THRESHOLD = 8
T_PLAUSIBILITY_BAND = (0.3, 5.0)
ECE_QUALITY_BANDS = {"well_calibrated": 0.05, "acceptable": 0.10}


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def fit_temperature(scores: np.ndarray, labels: np.ndarray) -> float:
    """Fit T > 0 by minimizing NLL of sigmoid(score / T) vs labels (D1)."""

    def nll(T: float) -> float:
        if T <= 0:
            return float("inf")
        p = sigmoid(scores / T)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -float(np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

    result = minimize_scalar(
        nll, bounds=(0.01, 100.0), method="bounded", options={"xatol": 1e-6}
    )
    return float(result.x)


def calibration_loso(per_clip: list[dict]) -> dict:
    """Source-aware calibration LOSO (D2).

    For each source S, fit T_S on the OTHER 11 sources' OOF scores,
    then apply T_S to S's OOF scores. Returns per-source T plus
    per-clip pre-cal (T=1 baseline) and post-cal probabilities.
    """
    sources = sorted({rec["source"] for rec in per_clip})
    scores = np.array([rec["score"] for rec in per_clip], dtype=float)
    labels = np.array([rec["label"] for rec in per_clip], dtype=int)
    src_arr = np.array([rec["source"] for rec in per_clip])

    pre_cal = sigmoid(scores)  # T=1 baseline
    post_cal = np.zeros_like(scores)
    T_per_src: dict[str, float] = {}

    for S in sources:
        train_mask = src_arr != S
        test_mask = src_arr == S
        T_S = fit_temperature(scores[train_mask], labels[train_mask])
        T_per_src[S] = T_S
        post_cal[test_mask] = sigmoid(scores[test_mask] / T_S)

    return {
        "sources": sources,
        "T_per_src": T_per_src,
        "scores": scores,
        "labels": labels,
        "src_arr": src_arr,
        "pre_cal": pre_cal,
        "post_cal": post_cal,
    }


def equal_freq_bins(probs: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin indices [0..n_bins-1] for equal-frequency (quantile) binning (D3)."""
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(probs, quantiles)
    edges[-1] += 1e-9  # ensure max prob falls in last bin
    return np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)


def compute_ece(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS
) -> tuple[float, list[dict]]:
    """ECE = sum_b (n_b / N) * |acc(b) - conf(b)|, equal-freq bins (D3)."""
    bins = equal_freq_bins(probs, n_bins)
    N = len(probs)
    ece = 0.0
    bin_records: list[dict] = []
    for b in range(n_bins):
        mask = bins == b
        n_b = int(mask.sum())
        if n_b == 0:
            bin_records.append({"bin": b, "n": 0})
            continue
        conf_b = float(probs[mask].mean())
        acc_b = float(labels[mask].mean())
        ece += (n_b / N) * abs(acc_b - conf_b)
        bin_records.append(
            {
                "bin": b,
                "n": n_b,
                "conf_mean": conf_b,
                "acc": acc_b,
                "gap": acc_b - conf_b,
            }
        )
    return float(ece), bin_records


def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def compute_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def select_tau(
    probs: np.ndarray, labels: np.ndarray, fpr: float = FPR_TARGET
) -> tuple[float, float, int, int]:
    """τ such that FPR on label=0 clips equals target; return (tau, tpr, n_neg, n_pos) (D4)."""
    neg = probs[labels == 0]
    pos = probs[labels == 1]
    tau = float(np.quantile(neg, 1 - fpr))
    tpr = float(np.mean(pos >= tau))
    return tau, tpr, int(len(neg)), int(len(pos))


def bootstrap_tpr(
    probs: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    tau: float,
    B: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Source-resampled bootstrap CI on TPR at fixed tau (D4).

    Subject-bootstrap convention: resample sources WITH replacement;
    within each sampled source, keep all clips. Matches Phase 8b's
    subject-bootstrap protocol.
    """
    rng = np.random.default_rng(seed)
    unique_src = np.array(sorted(set(sources)))
    tprs: list[float] = []
    for _ in range(B):
        sampled_src = rng.choice(unique_src, size=len(unique_src), replace=True)
        boot_idx_parts = [np.where(sources == s)[0] for s in sampled_src]
        boot_idx = np.concatenate(boot_idx_parts)
        boot_probs = probs[boot_idx]
        boot_labels = labels[boot_idx]
        pos = boot_probs[boot_labels == 1]
        if len(pos) == 0:
            continue  # defensive; near-zero probability at this n
        tprs.append(float(np.mean(pos >= tau)))
    tprs_arr = np.array(tprs)
    return float(np.quantile(tprs_arr, 0.025)), float(np.quantile(tprs_arr, 0.975))


def session_op_under_independence(
    per_behavior_fpr: float = FPR_TARGET,
    n_behaviors: int = N_BEHAVIORS_RHPE,
    threshold: int = SESSION_THRESHOLD,
) -> dict:
    """Session-level OP under independence (D5).

    Reports BOTH the exact Poisson-binomial CDF (24-Bernoulli convolution,
    generalizes to non-uniform per-behavior FPRs) AND the Poisson(λ) tail
    approximation. The exact computation is the load-bearing one;
    the approximation is reported to demonstrate it isn't load-bearing.
    """
    # Exact Poisson-binomial via convolution.
    # Generalizes to non-uniform p; for uniform p this reduces to Binomial(n, p).
    p_per_behavior = np.full(n_behaviors, per_behavior_fpr)
    pmf = np.array([1.0 - p_per_behavior[0], p_per_behavior[0]])
    for k in range(1, n_behaviors):
        bern = np.array([1.0 - p_per_behavior[k], p_per_behavior[k]])
        pmf = np.convolve(pmf, bern)
    p_session_exact = float(pmf[threshold:].sum())

    # Poisson(λ) approximation
    lam = float(n_behaviors * per_behavior_fpr)
    p_session_poisson = float(1.0 - poisson.cdf(threshold - 1, lam))

    abs_err = abs(p_session_exact - p_session_poisson)
    rel_err = abs_err / p_session_exact if p_session_exact > 0 else float("nan")

    return {
        "n_behaviors": n_behaviors,
        "per_behavior_fpr": per_behavior_fpr,
        "session_threshold": threshold,
        "exact_poisson_binomial": {
            "P_session_flagged_under_H0": p_session_exact,
            "method": (
                f"convolution of {n_behaviors} Bernoulli(p={per_behavior_fpr}); "
                f"reduces to Binomial(n={n_behaviors}, p={per_behavior_fpr}) under uniform p"
            ),
        },
        "poisson_approximation": {
            "P_session_flagged_under_H0": p_session_poisson,
            "lambda": lam,
            "method": f"Poisson(λ={lam}) tail beyond {threshold - 1}",
        },
        "approximation_absolute_error": abs_err,
        "approximation_relative_error": rel_err,
        "independence_assumption_status": (
            "UNVERIFIED — see pre-reg Limitation 1: RHpE behavior co-occurrence "
            "matrices not located in q3_architecture survey; tail swish + ear "
            "pinning, eye-cluster behaviors, and head/body co-occurrence are "
            "plausible violations. Direction of violation unknown."
        ),
    }


def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI on a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def reliability_diagram(
    pre_cal: np.ndarray,
    post_cal: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    n_bins: int = N_BINS,
) -> None:
    """2-panel reliability diagram: pre-cal | post-cal, equal-freq bins (D3)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)

    panels = [
        (axes[0], pre_cal, "Pre-calibration (T=1, raw sigmoid of decision_function)"),
        (axes[1], post_cal, "Post-calibration (LOSO T_S per source)"),
    ]

    for ax, probs, title in panels:
        bins = equal_freq_bins(probs, n_bins)
        bin_data = []
        for b in range(n_bins):
            mask = bins == b
            n_b = int(mask.sum())
            if n_b == 0:
                continue
            conf_b = float(probs[mask].mean())
            acc_b = float(labels[mask].mean())
            ci_low, ci_high = wilson_ci(acc_b, n_b)
            bin_data.append((conf_b, acc_b, n_b, ci_low, ci_high))

        if bin_data:
            confs, accs, ns, lows, highs = (np.array(x) for x in zip(*bin_data))

            # bin-count underlay
            ax2 = ax.twinx()
            ax2.bar(confs, ns, width=0.06, alpha=0.18, color="gray", edgecolor="none")
            ax2.set_ylabel("clips per bin", color="gray", fontsize=9)
            ax2.tick_params(axis="y", labelcolor="gray", labelsize=8)
            ax2.set_ylim(0, max(ns) * 3 if max(ns) > 0 else 1)

            err_lower = accs - lows
            err_upper = highs - accs
            ax.errorbar(
                confs,
                accs,
                yerr=[err_lower, err_upper],
                fmt="o-",
                color="tab:blue",
                capsize=3,
                label="observed accuracy (Wilson 95% CI)",
            )

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("predicted probability (bin mean)")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("observed accuracy")
    fig.suptitle(
        f"Phase 8c — RME ear-movement reliability diagram "
        f"(n=283, 12 sources, {n_bins} equal-freq bins)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def quality_band(ece: float) -> str:
    """G3 verdict bands (locked in pre-reg Decision 3)."""
    if ece < ECE_QUALITY_BANDS["well_calibrated"]:
        return "well_calibrated"
    if ece < ECE_QUALITY_BANDS["acceptable"]:
        return "acceptable"
    return "poorly_calibrated"


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"[ERROR] Phase 8b output not found: {INPUT_PATH}", file=sys.stderr)
        return 1

    print(
        f"[phase8c] loading Phase 8b output from "
        f"{INPUT_PATH.relative_to(POC_DIR)}",
        flush=True,
    )
    phase8b = json.loads(INPUT_PATH.read_text())
    per_clip = phase8b["per_clip"]
    n_clips = len(per_clip)
    phase8b_pooled_auc = phase8b["pooled_auc"]
    print(
        f"[phase8c] n_clips={n_clips}, Phase 8b pooled AUC={phase8b_pooled_auc:.4f}",
        flush=True,
    )

    print(
        "[phase8c] running source-aware calibration LOSO (D2) — 12 folds...",
        flush=True,
    )
    cal = calibration_loso(per_clip)

    pre_cal = cal["pre_cal"]
    post_cal = cal["post_cal"]
    labels = cal["labels"]
    scores = cal["scores"]
    src_arr = cal["src_arr"]

    # Per-fold T diagnostic
    T_values = list(cal["T_per_src"].values())
    T_median = float(np.median(T_values))
    T_q25 = float(np.quantile(T_values, 0.25))
    T_q75 = float(np.quantile(T_values, 0.75))
    T_in_band = all(
        T_PLAUSIBILITY_BAND[0] <= T <= T_PLAUSIBILITY_BAND[1] for T in T_values
    )

    # G1: AUC invariance sanity check
    auc_pre = float(roc_auc_score(labels, pre_cal))
    auc_post = float(roc_auc_score(labels, post_cal))
    auc_invariance_pass = abs(auc_pre - auc_post) <= 1e-10
    print(
        f"[phase8c] AUC pre={auc_pre:.6f}, post={auc_post:.6f}, "
        f"invariance={'PASS' if auc_invariance_pass else 'FAIL'}",
        flush=True,
    )

    # D3 metrics (pooled)
    ece_pre, bin_pre = compute_ece(pre_cal, labels)
    ece_post, bin_post = compute_ece(post_cal, labels)
    brier_pre = compute_brier(pre_cal, labels)
    brier_post = compute_brier(post_cal, labels)
    nll_post = compute_nll(post_cal, labels)

    # Per-source ECE/Brier diagnostic
    per_source_metrics = {}
    for S in cal["sources"]:
        mask = src_arr == S
        n_S = int(mask.sum())
        if n_S == 0:
            continue
        # per-source bins reduced when n_S small to avoid degeneracy
        n_bins_S = min(N_BINS, max(2, n_S // 5))
        ece_S_pre, _ = compute_ece(pre_cal[mask], labels[mask], n_bins=n_bins_S)
        ece_S_post, _ = compute_ece(post_cal[mask], labels[mask], n_bins=n_bins_S)
        brier_S_pre = compute_brier(pre_cal[mask], labels[mask])
        brier_S_post = compute_brier(post_cal[mask], labels[mask])
        per_source_metrics[S] = {
            "n_clips": n_S,
            "n_bins_used": n_bins_S,
            "ece_pre_cal": ece_S_pre,
            "ece_post_cal": ece_S_post,
            "brier_pre_cal": brier_S_pre,
            "brier_post_cal": brier_S_post,
            "T_S": cal["T_per_src"][S],
        }

    # D4: τ_ear at FPR=0.05 + bootstrap
    print(f"[phase8c] selecting τ_ear at FPR={FPR_TARGET} (D4)...", flush=True)
    tau_ear, tpr_at_tau, n_neg, n_pos = select_tau(post_cal, labels)
    print(
        f"[phase8c] bootstrapping TPR (B={BOOTSTRAP_B}, source-resampled, "
        f"seed={BOOTSTRAP_SEED})...",
        flush=True,
    )
    tpr_ci_low, tpr_ci_high = bootstrap_tpr(post_cal, labels, src_arr, tau_ear)

    # D5: session-level operating point
    session_op = session_op_under_independence()

    # Reliability diagram
    print(
        f"[phase8c] writing reliability diagram to "
        f"{OUT_FIG.relative_to(POC_DIR)}",
        flush=True,
    )
    reliability_diagram(pre_cal, post_cal, labels, OUT_FIG)

    # Gates
    g1 = "PASS" if auc_invariance_pass else "FAIL"
    g2 = "PASS" if ece_post < ece_pre else "NEUTRAL_OR_DEGRADES"
    g3 = quality_band(ece_post)
    g4 = "PASS" if T_in_band else "FAIL"

    results = {
        "mode": "phase8c_calibration",
        "input_file": str(INPUT_PATH.relative_to(POC_DIR)),
        "preregistration": "outputs/track_b_phase8c_preregistration.md",
        "n_clips": n_clips,
        "n_sources": len(cal["sources"]),
        "phase8b_pooled_auc_reference": phase8b_pooled_auc,
        "pre_cal_pooled_auc": auc_pre,
        "post_cal_pooled_auc": auc_post,
        "auc_invariance_check": g1,
        "per_fold_T": cal["T_per_src"],
        "T_median": T_median,
        "T_iqr": [T_q25, T_q75],
        "T_in_plausibility_band": T_in_band,
        "T_plausibility_band": list(T_PLAUSIBILITY_BAND),
        "ece": {
            "pre_cal": ece_pre,
            "post_cal": ece_post,
            "delta": ece_post - ece_pre,
            "n_bins": N_BINS,
            "binning": "equal-frequency",
        },
        "brier": {
            "pre_cal": brier_pre,
            "post_cal": brier_post,
            "delta": brier_post - brier_pre,
        },
        "nll": {"post_cal": nll_post},
        "per_source_metrics": per_source_metrics,
        "operating_point": {
            "fpr_target": FPR_TARGET,
            "tau_ear": tau_ear,
            "tpr_at_tau": tpr_at_tau,
            "tpr_bootstrap_95_ci": [tpr_ci_low, tpr_ci_high],
            "n_negatives_used": n_neg,
            "n_positives_used": n_pos,
            "bootstrap_B": BOOTSTRAP_B,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "session_level_under_independence": session_op,
        "gates": {
            "G1_sanity_auc_invariance": g1,
            "G2_calibration_improvement": g2,
            "G3_calibration_quality_band": g3,
            "G4_T_plausibility": g4,
        },
        "config": {
            "calibration_method": "temperature_scaling_one_parameter",
            "fitting_objective": "NLL on training-fold scores",
            "calibration_loso": (
                "source-aware (per source S, fit T on other 11 sources' OOF scores)"
            ),
            "ece_n_bins": N_BINS,
            "ece_binning": "equal-frequency",
            "fpr_target": FPR_TARGET,
            "bootstrap_B": BOOTSTRAP_B,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "n_behaviors_rhpe": N_BEHAVIORS_RHPE,
            "session_threshold": SESSION_THRESHOLD,
            "scoring_rule_source": (
                "Dyson 2018 J Vet Behav 23:47-57 + Dyson & Pollard 2023 "
                "Animals 13(12):1940 — presence/absence per behavior, "
                "session score = sum of presences, threshold ≥8/24. "
                "See v2/research/dyson_scoring_check.md"
            ),
        },
        "limitations": [
            "L1: independence assumption for session-level operating point is "
            "unverified — see pre-reg Limitation 1.",
            "L2: single-behavior scope (ear movement only) — see pre-reg "
            "Limitation 2.",
            "L3: single-observer label noise from RME paper carries forward — "
            "see pre-reg Limitation 3.",
            "L4: RidgeClassifier vs LogisticRegression deferred to Phase 9 — "
            "see pre-reg Limitation 4.",
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    OUT_RESULTS.write_text(json.dumps(results, indent=2))
    print(f"[phase8c] wrote {OUT_RESULTS.relative_to(POC_DIR)}", flush=True)

    # Per-clip extras
    bins_pre = equal_freq_bins(pre_cal, N_BINS)
    bins_post = equal_freq_bins(post_cal, N_BINS)
    above_tau = post_cal >= tau_ear
    extras = {
        "mode": "phase8c_audit_extras",
        "tau_ear": tau_ear,
        "fpr_target": FPR_TARGET,
        "per_clip": [
            {
                "clip": rec["clip"],
                "source": rec["source"],
                "label": int(rec["label"]),
                "score_pre_cal": float(scores[i]),
                "prob_pre_cal_T1": float(pre_cal[i]),
                "T_applied": float(cal["T_per_src"][rec["source"]]),
                "prob_post_cal": float(post_cal[i]),
                "ece_bin_pre_cal": int(bins_pre[i]),
                "ece_bin_post_cal": int(bins_post[i]),
                "above_tau_ear": bool(above_tau[i]),
            }
            for i, rec in enumerate(per_clip)
        ],
        "ece_bin_records_pre_cal": bin_pre,
        "ece_bin_records_post_cal": bin_post,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUT_EXTRAS.write_text(json.dumps(extras, indent=2))
    print(f"[phase8c] wrote {OUT_EXTRAS.relative_to(POC_DIR)}", flush=True)

    # Summary
    print()
    print("=" * 70)
    print("PHASE 8c — calibration package on Phase 8b ear output")
    print("=" * 70)
    print(
        f"  AUC invariance (G1):             {g1}  "
        f"(pre={auc_pre:.6f}, post={auc_post:.6f})"
    )
    print(f"  ECE pre-cal:                     {ece_pre:.4f}")
    print(
        f"  ECE post-cal:                    {ece_post:.4f}  "
        f"(Δ={ece_post - ece_pre:+.4f})"
    )
    print(f"  Calibration improvement (G2):    {g2}")
    print(f"  Calibration quality band (G3):   {g3}")
    print(f"  Brier pre-cal:                   {brier_pre:.4f}")
    print(
        f"  Brier post-cal:                  {brier_post:.4f}  "
        f"(Δ={brier_post - brier_pre:+.4f})"
    )
    print(f"  NLL post-cal:                    {nll_post:.4f}")
    print(f"  T median:                        {T_median:.3f}")
    print(f"  T IQR:                           [{T_q25:.3f}, {T_q75:.3f}]")
    print(f"  T plausibility (G4):             {g4}  ({len(T_values)} folds)")
    print(f"  τ_ear at FPR={FPR_TARGET}:               {tau_ear:.4f}")
    print(
        f"  TPR at τ_ear:                    {tpr_at_tau:.4f}  "
        f"95% CI=[{tpr_ci_low:.4f}, {tpr_ci_high:.4f}]"
    )
    print(
        f"  Session OP P(≥8|H0) exact:       "
        f"{session_op['exact_poisson_binomial']['P_session_flagged_under_H0']:.3e}"
    )
    print(
        f"  Session OP P(≥8|H0) Poisson:     "
        f"{session_op['poisson_approximation']['P_session_flagged_under_H0']:.3e}"
    )
    print(
        "  ⚠ independence assumption:       UNVERIFIED (load-bearing limitation)"
    )
    print()
    print("Result files:")
    print(f"  {OUT_RESULTS.relative_to(POC_DIR)}")
    print(f"  {OUT_EXTRAS.relative_to(POC_DIR)}")
    print(f"  {OUT_FIG.relative_to(POC_DIR)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
