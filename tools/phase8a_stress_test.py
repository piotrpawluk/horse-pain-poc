#!/usr/bin/env python3
"""Phase 8a stress-test of Phase 7's +0.048 AUC advantage.

Runs 5 locked tests on existing artifacts per
`outputs/track_b_phase8a_preregistration_stage1.md` (hash
c0bf2bdca8d5...):

1. Subject-bootstrap CI on Δ AUC (B=10000, seed=42, parent-clip
   resampling). Three-band gate.
2. Per-source LOSO ablation (drop each of 12 sources, recompute pooled
   AUC for both pipelines). Three-band gate.
3. Per-clip score divergence (v3 vs v4-corrected). Reportable.
4. V-JEPA-2 feature similarity (cosine sim per clip). Two-band gate.
5. G3-IoU-conditional AUC (3 IoU buckets). Reportable.

Plus 4 diagnostic cross-checks per pre-reg.

Inputs (existing artifacts, hash-pinned):
  outputs/eye_loso_results_phase5_primary.json
  outputs/eye_loso_results_phase7.json
  outputs/vjepa2_embeddings_eye_v3_m15.npz
  outputs/vjepa2_embeddings_eye_v4.npz
  outputs/phase7_diagnostic.json (for IoU per clip)

Output:
  outputs/phase8a_results.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

POC_DIR = Path(__file__).resolve().parent.parent
P5_PATH = POC_DIR / "outputs" / "eye_loso_results_phase5_primary.json"
P7_PATH = POC_DIR / "outputs" / "eye_loso_results_phase7.json"
P7_DIAG_PATH = POC_DIR / "outputs" / "phase7_diagnostic.json"
EMB_V3_PATH = POC_DIR / "outputs" / "vjepa2_embeddings_eye_v3_m15.npz"
EMB_V4_PATH = POC_DIR / "outputs" / "vjepa2_embeddings_eye_v4.npz"
LABELS_PATH = POC_DIR / "outputs" / "eye_verification_clips.txt"
OUT_PATH = POC_DIR / "outputs" / "phase8a_results.json"

# Locked Phase 8a parameters per pre-reg
BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 42
TEST1_GATES = {
    "robust_lb": 0.02,
    "retraction_lb": -0.02,
    "retraction_median_split": -0.01,
}
TEST2_GATES = {"robust_min": 9, "modest_min": 6}
TEST4_THRESHOLD = 0.7  # cosine sim median
TEST5_BUCKETS = [
    ("off_eye", 0.0, 0.30),
    ("mid", 0.30, 0.50),
    ("on_eye", 0.50, 1.01),
]


def parse_labels(path):
    """Re-parse labels file from raw text (matches eye_loso_lr.py)."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(" - ")]
        if len(parts) < 2:
            continue
        verdict = parts[-1].upper()
        if verdict not in ("ACTION", "BACKGROUND"):
            continue
        path_or_name = parts[0]
        basename = path_or_name.rsplit("/", 1)[-1]
        out[basename] = 1 if verdict == "ACTION" else 0
    return out


def extract_source(filename):
    """Source ID from filename like background_S5.mp4_10_.mp4 → S5."""
    import re
    m = re.search(r"S(\d+)\.mp4", filename)
    if m is None:
        raise ValueError(f"cannot extract source from {filename}")
    return f"S{m.group(1)}"


def loso_pooled_predict(X, y, groups, sources, filenames):
    """Run LOSO with the canonical Ridge config; return (preds, truth,
    clips, fold_log) ordered by source-iteration (matching Phase 5/7
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


def auc_pooled(y, scores):
    return float(roc_auc_score(y, scores))


# ---------- Test 1: Subject-bootstrap CI on Δ AUC ----------

def test1_bootstrap_ci(p5_predictions, p7_predictions, labels, groups,
                       n_bootstrap=BOOTSTRAP_B, seed=BOOTSTRAP_SEED):
    """Subject-bootstrap CI on Δ AUC = AUC(p7) − AUC(p5).

    Parent-clip resampling at the source level, matching Phase 5/6/7
    bootstrap conventions.
    """
    rng = np.random.default_rng(seed)
    sources = sorted(set(groups))
    source_to_indices = defaultdict(list)
    for i, g in enumerate(groups):
        source_to_indices[g].append(i)

    delta_distribution = []
    n_kept = 0
    for _ in range(n_bootstrap):
        # Resample sources with replacement
        boot_sources = rng.choice(sources, size=len(sources), replace=True)
        boot_indices = []
        for s in boot_sources:
            boot_indices.extend(source_to_indices[s])
        boot_indices = np.array(boot_indices)
        y_boot = labels[boot_indices]
        if len(np.unique(y_boot)) < 2:
            continue  # degenerate fold; skip
        p5_boot = p5_predictions[boot_indices]
        p7_boot = p7_predictions[boot_indices]
        try:
            auc_p5 = roc_auc_score(y_boot, p5_boot)
            auc_p7 = roc_auc_score(y_boot, p7_boot)
        except ValueError:
            continue
        delta_distribution.append(auc_p7 - auc_p5)
        n_kept += 1

    delta_arr = np.array(delta_distribution)
    return {
        "n_bootstrap_kept": n_kept,
        "n_bootstrap_total": n_bootstrap,
        "delta_mean": float(delta_arr.mean()),
        "delta_median": float(np.median(delta_arr)),
        "delta_std": float(delta_arr.std()),
        "ci_95_low": float(np.percentile(delta_arr, 2.5)),
        "ci_95_high": float(np.percentile(delta_arr, 97.5)),
        "ci_95_low_one_sided": float(np.percentile(delta_arr, 5.0)),
        "p_delta_le_zero": float((delta_arr <= 0).mean()),
        "p_delta_le_neg_002": float((delta_arr <= -0.02).mean()),
        "p_delta_ge_002": float((delta_arr >= 0.02).mean()),
    }


def test1_route(bootstrap_result):
    """Route per locked three-band gate."""
    lb = bootstrap_result["ci_95_low"]
    if lb >= TEST1_GATES["robust_lb"]:
        verdict = "ROBUST"
        next_action = ("Phase 7 verdict robust at n=34. Phase 9 N "
                       "expansion's marginal value drops; bootstrap "
                       "answered the +0.048 question. Phase 8b proceeds "
                       "with confidence.")
    elif lb > TEST1_GATES["retraction_lb"]:
        verdict = "UNDERPOWERED"
        next_action = ("Bootstrap CI confirms paired-DeLong's "
                       "inconclusive verdict, just quantified. Phase 9 "
                       "N expansion is the resolution. Phase 8b "
                       "proceeds; verdict-band recalibration may apply.")
    else:
        # Retraction band: choose replacement verdict mechanically
        median = bootstrap_result["delta_median"]
        if median > TEST1_GATES["retraction_median_split"]:
            replacement = "UNDERPOWERED_INDISTINGUISHABLE"
        else:
            replacement = "PHASE_7_FAVORABLE_BY_CHANCE"
        verdict = "RETRACTION"
        next_action = (f"Bootstrap LB={lb:.4f} ≤ -0.02 fires retraction "
                       f"trigger. Replacement verdict: {replacement} "
                       f"(median Δ_bootstrap = {median:+.4f}). Phase 7 "
                       f"audit doc receives appended retraction section "
                       f"per locked procedure. Phase 8b BLOCKED until "
                       f"retraction commit lands.")
    return verdict, next_action


# ---------- Test 2: Per-source LOSO ablation ----------

def test2_per_source_ablation(emb_v3, emb_v4, labels, groups, filenames):
    """Drop each source individually, recompute pooled AUC for both
    pipelines on remaining 11 sources."""
    sources = sorted(set(groups))
    results = {}
    for drop_s in sources:
        keep = groups != drop_s
        X_v3 = emb_v3[keep]
        X_v4 = emb_v4[keep]
        y = labels[keep]
        g = groups[keep]
        f = [filenames[i] for i in range(len(filenames)) if keep[i]]
        kept_sources = sorted(set(g))

        preds_v3, _ = loso_pooled_predict(X_v3, y, g, kept_sources, f)
        preds_v4, _ = loso_pooled_predict(X_v4, y, g, kept_sources, f)
        if len(np.unique(y)) < 2:
            results[drop_s] = {"auc_v3": None, "auc_v4": None,
                               "delta": None}
            continue
        auc_v3 = auc_pooled(y, preds_v3)
        auc_v4 = auc_pooled(y, preds_v4)
        results[drop_s] = {
            "auc_v3": float(auc_v3),
            "auc_v4": float(auc_v4),
            "delta": float(auc_v4 - auc_v3),
            "n_remaining": int(keep.sum()),
        }
    n_positive_delta = sum(
        1 for r in results.values()
        if r["delta"] is not None and r["delta"] > 0
    )
    n_total = sum(1 for r in results.values() if r["delta"] is not None)
    return results, n_positive_delta, n_total


def test2_route(n_positive, n_total):
    if n_positive >= TEST2_GATES["robust_min"]:
        return "ROBUST"
    elif n_positive >= TEST2_GATES["modest_min"]:
        return "MODEST"
    else:
        return "FRAGILE"


# ---------- Test 3: Per-clip score divergence ----------

def test3_per_clip_score_divergence(p5_per_clip, p7_per_clip):
    rows = []
    for clip in sorted(p5_per_clip):
        r5 = p5_per_clip[clip]
        r7 = p7_per_clip[clip]
        if r5["label"] != r7["label"]:
            continue
        delta = r7["score"] - r5["score"]
        rows.append({
            "clip": clip,
            "label": r5["label"],
            "phase5_score": r5["score"],
            "phase7_score": r7["score"],
            "delta": delta,
        })
    deltas = np.array([r["delta"] for r in rows])
    return rows, {
        "n": len(rows),
        "mean": float(deltas.mean()),
        "median": float(np.median(deltas)),
        "std": float(deltas.std()),
        "p25": float(np.percentile(deltas, 25)),
        "p75": float(np.percentile(deltas, 75)),
        "min": float(deltas.min()),
        "max": float(deltas.max()),
    }


# ---------- Test 4: V-JEPA-2 feature similarity ----------

def cos_sim(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def test4_feature_similarity(emb_v3, emb_v4, filenames_v3, filenames_v4):
    if filenames_v3 != filenames_v4:
        common = sorted(set(filenames_v3) & set(filenames_v4))
        idx_v3 = {f: i for i, f in enumerate(filenames_v3)}
        idx_v4 = {f: i for i, f in enumerate(filenames_v4)}
    else:
        common = filenames_v3
        idx_v3 = {f: i for i, f in enumerate(filenames_v3)}
        idx_v4 = idx_v3
    per_clip = {}
    for f in common:
        c = cos_sim(emb_v3[idx_v3[f]], emb_v4[idx_v4[f]])
        per_clip[f] = c

    cos_values = list(per_clip.values())
    # Internal baseline 1: v3-vs-v3 cross-clip (different clips on same model)
    n_pairs = 0
    cross_clip_v3 = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = list(range(len(common)))
    n_target = min(500, len(common) * (len(common) - 1) // 2)
    seen = set()
    while len(cross_clip_v3) < n_target and n_pairs < n_target * 5:
        i, j = rng.choice(indices, size=2, replace=False)
        key = (min(i, j), max(i, j))
        n_pairs += 1
        if key in seen:
            continue
        seen.add(key)
        f_i = common[i]
        f_j = common[j]
        cross_clip_v3.append(
            cos_sim(emb_v3[idx_v3[f_i]], emb_v3[idx_v3[f_j]])
        )

    return {
        "per_clip": per_clip,
        "v3_v4_same_clip": {
            "n": len(cos_values),
            "median": float(np.median(cos_values)),
            "mean": float(np.mean(cos_values)),
            "std": float(np.std(cos_values)),
            "p25": float(np.percentile(cos_values, 25)),
            "p75": float(np.percentile(cos_values, 75)),
            "min": float(min(cos_values)),
            "max": float(max(cos_values)),
        },
        "v3_v3_different_clips_baseline": {
            "n_pairs": len(cross_clip_v3),
            "median": float(np.median(cross_clip_v3)),
            "mean": float(np.mean(cross_clip_v3)),
            "std": float(np.std(cross_clip_v3)),
            "p25": float(np.percentile(cross_clip_v3, 25)),
            "p75": float(np.percentile(cross_clip_v3, 75)),
            "note": ("Internal baseline: cosine sim between v3 features "
                     "of DIFFERENT clips. Calibrates 'what's a similar "
                     "feature?' for this V-JEPA-2 + clip distribution. "
                     "v3-v4-same-clip > this baseline → features encode "
                     "clip-specific information beyond background "
                     "structure."),
        },
    }


def test4_route(test4_result):
    median = test4_result["v3_v4_same_clip"]["median"]
    if median >= TEST4_THRESHOLD:
        verdict = "SAME_FEATURES_DIFFERENT_GEOMETRY"
    else:
        verdict = "DIFFERENT_FEATURES_SIMILAR_PREDICTION"
    # Calibration: how does median compare to cross-clip baseline?
    baseline_median = (
        test4_result["v3_v3_different_clips_baseline"]["median"]
    )
    margin = median - baseline_median
    return verdict, margin


# ---------- Test 5: G3-IoU-conditional AUC ----------

def test5_iou_conditional_auc(p7_per_clip, p7_diagnostic, labels):
    """Bucket clips by IoU (Phase 7 v4 vs Phase 5 manual middle box);
    compute per-bucket AUC."""
    iou_per_clip = {}
    for r in p7_diagnostic["rows"]:
        clip = r["clip"]
        iou = r.get("iou_phase7_vs_phase5_manual_mid")
        if iou is not None:
            iou_per_clip[clip] = iou

    bucket_data = {name: {"clips": [], "scores": [], "labels": []}
                    for name, _, _ in TEST5_BUCKETS}
    for clip, iou in iou_per_clip.items():
        if clip not in p7_per_clip:
            continue
        r7 = p7_per_clip[clip]
        for name, lo, hi in TEST5_BUCKETS:
            if lo <= iou < hi:
                bucket_data[name]["clips"].append(clip)
                bucket_data[name]["scores"].append(r7["score"])
                bucket_data[name]["labels"].append(r7["label"])
                break

    bucket_results = {}
    for name, data in bucket_data.items():
        n = len(data["clips"])
        if n < 2 or len(set(data["labels"])) < 2:
            bucket_results[name] = {
                "n": n,
                "auc": None,
                "ci_95_low": None,
                "ci_95_high": None,
                "note": ("undefined: <2 clips or single-class subset"),
            }
            continue
        scores = np.array(data["scores"])
        y = np.array(data["labels"])
        auc = float(roc_auc_score(y, scores))
        # Bootstrap CI per bucket (B=10000, seed=42)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        boot_aucs = []
        for _ in range(BOOTSTRAP_B):
            idx = rng.integers(0, n, size=n)
            y_boot = y[idx]
            s_boot = scores[idx]
            if len(np.unique(y_boot)) < 2:
                continue
            try:
                boot_aucs.append(roc_auc_score(y_boot, s_boot))
            except ValueError:
                continue
        if boot_aucs:
            ci_low = float(np.percentile(boot_aucs, 2.5))
            ci_high = float(np.percentile(boot_aucs, 97.5))
        else:
            ci_low = ci_high = None
        bucket_results[name] = {
            "n": n,
            "auc": auc,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "clips": data["clips"],
        }
    return bucket_results


# ---------- Diagnostic cross-checks ----------

def cross_checks(test1_result, test2_n_positive, test2_n_total,
                  test3_rows, test3_summary, test4_per_clip,
                  test5_buckets, p7_diag_rows):
    cross = {}
    # 1. Test 1 + 2 cross-check
    test1_lb = test1_result["ci_95_low"]
    test1_robust = test1_lb >= TEST1_GATES["robust_lb"]
    test2_robust = test2_n_positive >= TEST2_GATES["robust_min"]
    cross["test1_test2_consistency"] = {
        "test1_routes_robust": test1_robust,
        "test2_routes_robust": test2_robust,
        "consistent": test1_robust == test2_robust,
        "note": ("If inconsistent: bootstrap may be driven by 1-2 "
                  "high-Δ sources. Check per-source ablation for the "
                  "most-positive-Δ source(s) to see if their removal "
                  "would shift the bootstrap LB."),
    }

    # 2. Test 3 outliers
    rows_sorted = sorted(test3_rows, key=lambda r: r["delta"])
    cross["test3_outliers"] = {
        "top3_negative_delta": [
            {"clip": r["clip"], "label": r["label"],
             "p5_score": r["phase5_score"], "p7_score": r["phase7_score"],
             "delta": r["delta"]}
            for r in rows_sorted[:3]
        ],
        "top3_positive_delta": [
            {"clip": r["clip"], "label": r["label"],
             "p5_score": r["phase5_score"], "p7_score": r["phase7_score"],
             "delta": r["delta"]}
            for r in rows_sorted[-3:][::-1]
        ],
    }

    # 3. Test 4 + 5 cross-tab (cos_sim × IoU × score_delta per clip)
    iou_per_clip = {}
    for r in p7_diag_rows:
        clip = r["clip"]
        iou = r.get("iou_phase7_vs_phase5_manual_mid")
        if iou is not None:
            iou_per_clip[clip] = iou

    score_delta_per_clip = {r["clip"]: r["delta"] for r in test3_rows}

    cross_table = []
    for clip, cos in test4_per_clip.items():
        cross_table.append({
            "clip": clip,
            "cos_sim": cos,
            "iou": iou_per_clip.get(clip),
            "score_delta": score_delta_per_clip.get(clip),
        })
    cross["test4_test5_per_clip_table"] = sorted(
        cross_table,
        key=lambda x: x["cos_sim"] if x["cos_sim"] is not None else 0
    )

    # 4. Per-source per-bucket cross-tab (12 × 3)
    bucket_by_source = defaultdict(lambda: Counter())
    for r in p7_diag_rows:
        clip = r["clip"]
        src = r["source"]
        iou = r.get("iou_phase7_vs_phase5_manual_mid")
        if iou is None:
            continue
        for name, lo, hi in TEST5_BUCKETS:
            if lo <= iou < hi:
                bucket_by_source[src][name] += 1
                break
    cross["per_source_per_bucket"] = {
        s: dict(counts) for s, counts in sorted(bucket_by_source.items())
    }
    return cross


# ---------- Main ----------

def main() -> int:
    print("[phase8a] loading inputs...", flush=True)
    p5 = json.loads(P5_PATH.read_text())
    p7 = json.loads(P7_PATH.read_text())
    p7_diag = json.loads(P7_DIAG_PATH.read_text())
    emb_v3_data = np.load(EMB_V3_PATH, allow_pickle=True)
    emb_v4_data = np.load(EMB_V4_PATH, allow_pickle=True)
    label_map = parse_labels(LABELS_PATH)

    p5_per_clip = {r["clip"]: r for r in p5["per_clip"]}
    p7_per_clip = {r["clip"]: r for r in p7["per_clip"]}

    common_clips = sorted(set(p5_per_clip) & set(p7_per_clip))
    p5_predictions = np.array(
        [p5_per_clip[c]["score"] for c in common_clips]
    )
    p7_predictions = np.array(
        [p7_per_clip[c]["score"] for c in common_clips]
    )
    labels = np.array([p5_per_clip[c]["label"] for c in common_clips])
    groups = np.array([extract_source(c) for c in common_clips])

    print(f"[phase8a] {len(common_clips)} common clips, "
          f"{len(set(groups))} sources", flush=True)
    print(f"[phase8a] Phase 5 AUC: {p5['pooled_auc']:.4f}")
    print(f"[phase8a] Phase 7 AUC: {p7['pooled_auc']:.4f}")
    print(f"[phase8a] observed Δ: "
          f"{p7['pooled_auc'] - p5['pooled_auc']:+.4f}")
    print()

    # === Test 1 ===
    print("[test1] subject-bootstrap CI on Δ AUC...", flush=True)
    t1_result = test1_bootstrap_ci(
        p5_predictions, p7_predictions, labels, groups
    )
    t1_verdict, t1_next = test1_route(t1_result)
    print(f"[test1] LB={t1_result['ci_95_low']:.4f} → {t1_verdict}")

    # === Test 2 ===
    print("[test2] per-source LOSO ablation...", flush=True)
    emb_v3 = emb_v3_data["embs"]
    emb_v4 = emb_v4_data["embs"]
    fns_v3 = [str(x) for x in emb_v3_data["filenames"]]
    fns_v4 = [str(x) for x in emb_v4_data["filenames"]]
    common_emb = sorted(set(fns_v3) & set(fns_v4) & set(common_clips))
    idx_v3_emb = {f: i for i, f in enumerate(fns_v3)}
    idx_v4_emb = {f: i for i, f in enumerate(fns_v4)}
    aligned_v3 = np.stack([emb_v3[idx_v3_emb[f]] for f in common_emb])
    aligned_v4 = np.stack([emb_v4[idx_v4_emb[f]] for f in common_emb])
    aligned_y = np.array([label_map[f] for f in common_emb])
    aligned_groups = np.array([extract_source(f) for f in common_emb])
    t2_results, t2_n_positive, t2_n_total = test2_per_source_ablation(
        aligned_v3, aligned_v4, aligned_y, aligned_groups, common_emb
    )
    t2_verdict = test2_route(t2_n_positive, t2_n_total)
    print(f"[test2] {t2_n_positive}/{t2_n_total} sources +Δ → {t2_verdict}")

    # === Test 3 ===
    print("[test3] per-clip score divergence...", flush=True)
    t3_rows, t3_summary = test3_per_clip_score_divergence(
        p5_per_clip, p7_per_clip
    )
    print(f"[test3] mean Δ={t3_summary['mean']:+.4f}, "
          f"median={t3_summary['median']:+.4f}, "
          f"min={t3_summary['min']:+.4f}, max={t3_summary['max']:+.4f}")

    # === Test 4 ===
    print("[test4] V-JEPA-2 feature similarity...", flush=True)
    t4_result = test4_feature_similarity(
        emb_v3, emb_v4, fns_v3, fns_v4
    )
    t4_verdict, t4_margin = test4_route(t4_result)
    print(f"[test4] median cos_sim={t4_result['v3_v4_same_clip']['median']:.4f} "
          f"(baseline {t4_result['v3_v3_different_clips_baseline']['median']:.4f}) "
          f"→ {t4_verdict}")

    # === Test 5 ===
    print("[test5] G3-IoU-conditional AUC...", flush=True)
    t5_buckets = test5_iou_conditional_auc(
        p7_per_clip, p7_diag, labels
    )
    for name, b in t5_buckets.items():
        if b["auc"] is None:
            print(f"[test5] {name:8s}: n={b['n']} (no AUC)")
        else:
            print(f"[test5] {name:8s}: n={b['n']}, AUC={b['auc']:.4f} "
                  f"CI [{b['ci_95_low']:.4f}, {b['ci_95_high']:.4f}]")

    # === Cross-checks ===
    print()
    print("[cross-checks] computing diagnostic cross-checks...",
          flush=True)
    cross = cross_checks(
        t1_result, t2_n_positive, t2_n_total,
        t3_rows, t3_summary, t4_result["per_clip"],
        t5_buckets, p7_diag["rows"],
    )

    # Assemble result
    result = {
        "tool": "tools/phase8a_stress_test.py",
        "pre_reg": "outputs/track_b_phase8a_preregistration_stage1.md",
        "pre_reg_hash_at_run":
            "c0bf2bdca8d534ec50683525078f4ced7f5f3fab86c7bd5696a6ed96f95d392c",
        "phase5_pooled_auc": p5["pooled_auc"],
        "phase7_pooled_auc": p7["pooled_auc"],
        "observed_delta": p7["pooled_auc"] - p5["pooled_auc"],
        "n_clips": len(common_clips),
        "test1_subject_bootstrap_ci_on_delta_auc": {
            "result": t1_result,
            "verdict": t1_verdict,
            "next_action": t1_next,
        },
        "test2_per_source_loso_ablation": {
            "results_by_dropped_source": t2_results,
            "n_positive_delta": t2_n_positive,
            "n_total": t2_n_total,
            "verdict": t2_verdict,
        },
        "test3_per_clip_score_divergence": {
            "summary": t3_summary,
            "rows": t3_rows,
        },
        "test4_feature_similarity": {
            "result": t4_result,
            "verdict": t4_verdict,
            "median_minus_cross_clip_baseline_margin": t4_margin,
            "threshold_heuristic_note": (
                "The 0.7 threshold for SAME_FEATURES_DIFFERENT_GEOMETRY "
                "vs DIFFERENT_FEATURES_SIMILAR_PREDICTION is a heuristic "
                "anchored on the project's intra-rater IoU threshold "
                "(0.765 from Phase 5b). It is NOT empirically validated "
                "for V-JEPA-2 feature similarity. Audit-doc framing "
                "should cite this caveat explicitly. The cross-clip "
                "baseline (median cos sim between v3 features of "
                "different clips) provides post-hoc calibration: "
                "v3_v4_same_clip > cross_clip_baseline indicates v4 "
                "features encode clip-specific information beyond "
                "background structure."
            ),
        },
        "test5_g3_iou_conditional_auc": {
            "buckets": t5_buckets,
        },
        "cross_checks": cross,
    }

    OUT_PATH.write_text(json.dumps(result, indent=2, default=str))
    print()
    print(f"Wrote: {OUT_PATH.relative_to(POC_DIR)}")
    print()
    print("=" * 70)
    print("PHASE 8a SUMMARY (per locked pre-reg gates)")
    print("=" * 70)
    print(f"Test 1 (bootstrap CI on Δ): {t1_verdict}")
    print(f"  LB={t1_result['ci_95_low']:+.4f}, "
          f"median={t1_result['delta_median']:+.4f}, "
          f"95% CI [{t1_result['ci_95_low']:+.4f}, "
          f"{t1_result['ci_95_high']:+.4f}]")
    print(f"Test 2 (per-source ablation): {t2_verdict}")
    print(f"  {t2_n_positive}/{t2_n_total} sources show +Δ")
    print("Test 3 (score divergence): reportable")
    print(f"  mean={t3_summary['mean']:+.4f}, "
          f"std={t3_summary['std']:.4f}")
    print(f"Test 4 (feature similarity): {t4_verdict}")
    print(f"  median cos={t4_result['v3_v4_same_clip']['median']:.4f}, "
          f"baseline={t4_result['v3_v3_different_clips_baseline']['median']:.4f}")
    print("Test 5 (IoU-conditional AUC): reportable")
    for name, b in t5_buckets.items():
        if b["auc"] is None:
            print(f"  {name}: n={b['n']} (undefined)")
        else:
            print(f"  {name}: n={b['n']}, AUC={b['auc']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
