"""Full-pipeline smoke test — DLC + V-JEPA-2 + classifier on Modal CUDA vs cached Phase 8b CPU baseline.

Reference clip: `action_S1.mp4_0_.mp4` (first row of Phase 8b's
`outputs/vjepa2_embeddings_ear_v4.npz` — index 0, label=action).

The clip is the ALREADY-CROPPED ear region used as V-JEPA-2 input during
Phase 8b training. We DO NOT re-run DLC + ear-bbox on this clip (the crop
geometry was baked in during Phase 8b). The DLC stage is validated
separately by `smoke_test.py`. This file's job is to validate the
**downstream** stages on Modal CUDA:

  1. V-JEPA-2 features on Modal CUDA vs cached Phase 8b MPS baseline
  2. Classifier score on Modal-computed features vs cached score
  3. Binary classification flip vs cached binary class

Pass criteria (locked, per critic discipline):
  - V-JEPA-2 cosine similarity ≥ 0.999 (matches PARITY_THRESHOLD in extract_vjepa2.py)
  - Classifier score |Δ| ≤ 0.01 (T_median=0.494, σ slope ≈ 0.25 → ~0.0025 effect on σ output)
  - Binary classification flip count = 0

This test gates Phase 8b-CUDA retrain. Pass → retrain; fail-loud → investigate.

Audit-doc transition-commit framing: this test produces the "tolerance
equivalence measurement" the audit doc cites when documenting the
compute-substrate transition (Lesson 23 candidate).

Usage:
    modal run tools/cloud_dlc/smoke_test_full_pipeline.py
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

POC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POC))

# Reference clip + cached state
EAR_CROPS_DIR = POC / "outputs" / "eye_crops_v4_ear_dlc"
REFERENCE_FILENAME = "action_S1.mp4_0_.mp4"
REFERENCE_CLIP_PATH = EAR_CROPS_DIR / REFERENCE_FILENAME

VJEPA2_NPZ = POC / "outputs" / "vjepa2_embeddings_ear_v4.npz"
CLASSIFIER_JSON = POC / "outputs" / "phase10a_prelim_deployed_classifier.json"
RESULT_JSON = POC / "outputs" / "cloud_dlc_smoke_test_full_pipeline_result.json"

# Locked pass/fail thresholds
COSINE_PASS = 0.999
SCORE_DELTA_PASS = 0.01
T_MEDIAN = 0.494  # from Phase 8c calibration
TAU_EAR = 0.8138  # from Phase 8c calibration

from tools.cloud_dlc.app import app, run_vjepa2_remote  # noqa: E402,F401


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def manual_decision_function(
    x: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    coef: np.ndarray,
    intercept: float,
) -> float:
    """Same logic as tools/phase10a_train_deployed_classifier.py:40."""
    scaled = (x - mean) / scale
    return float(np.dot(scaled, coef) + intercept)


def _fail_loud(reason: str, result: dict) -> None:
    print("=" * 70, file=sys.stderr)
    print("[full_pipeline] FAIL — DO NOT PROCEED TO PHASE 8b-CUDA RETRAIN", file=sys.stderr)
    print(f"[full_pipeline] reason: {reason}", file=sys.stderr)
    print(f"[full_pipeline] result: {RESULT_JSON.relative_to(POC.parent)}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(result, indent=2, default=str))
    sys.exit(2)


@app.local_entrypoint()
def main() -> None:
    # ── Pre-flight: verify cached artifacts exist ─────────────────────────
    if not REFERENCE_CLIP_PATH.exists():
        print(f"[FAIL] reference clip missing: {REFERENCE_CLIP_PATH}", file=sys.stderr)
        sys.exit(1)
    if not VJEPA2_NPZ.exists():
        print(f"[FAIL] Phase 8b V-JEPA-2 cache missing: {VJEPA2_NPZ}", file=sys.stderr)
        sys.exit(1)
    if not CLASSIFIER_JSON.exists():
        print(f"[FAIL] deployed classifier missing: {CLASSIFIER_JSON}", file=sys.stderr)
        sys.exit(1)

    # ── Local: load cached state for reference clip ───────────────────────
    npz = np.load(VJEPA2_NPZ, allow_pickle=True)
    filenames = npz["filenames"]
    cached_embs = npz["embs"]

    try:
        ref_idx = int(np.where(filenames == REFERENCE_FILENAME)[0][0])
    except IndexError:
        print(f"[FAIL] {REFERENCE_FILENAME} not in {VJEPA2_NPZ.name}", file=sys.stderr)
        sys.exit(1)

    cached_emb = cached_embs[ref_idx].astype(np.float64)
    cached_label = int(npz["labels"][ref_idx])

    clf = json.loads(CLASSIFIER_JSON.read_text())
    scaler_mean = np.array(clf["scaler_mean"], dtype=np.float64)
    scaler_scale = np.array(clf["scaler_scale"], dtype=np.float64)
    ridge_coef = np.array(clf["ridge_coef"], dtype=np.float64)
    ridge_intercept = float(clf["ridge_intercept"])

    # Local CPU classifier score using cached MPS-computed features
    local_score = manual_decision_function(
        cached_emb, scaler_mean, scaler_scale, ridge_coef, ridge_intercept
    )
    local_prob = sigmoid(local_score / T_MEDIAN)
    local_binary = 1 if local_prob >= TAU_EAR else 0

    print(f"[full_pipeline] reference clip      : {REFERENCE_FILENAME}")
    print(f"[full_pipeline] cached emb shape    : {cached_emb.shape}")
    print(f"[full_pipeline] cached emb norm     : {np.linalg.norm(cached_emb):.4f}")
    print(f"[full_pipeline] cached label        : {cached_label} "
          f"({'action' if cached_label == 1 else 'background'})")
    print(f"[full_pipeline] local score (cached emb → cached classifier): {local_score:.6f}")
    print(f"[full_pipeline] local prob          : {local_prob:.6f}")
    print(f"[full_pipeline] local binary @ τ_ear: {local_binary}")
    print()

    # ── Modal: run V-JEPA-2 on CUDA for the same clip ─────────────────────
    print("[full_pipeline] invoking Modal T4 V-JEPA-2…")
    clip_bytes = REFERENCE_CLIP_PATH.read_bytes()
    print(f"[full_pipeline]   uploading {REFERENCE_CLIP_PATH.name} "
          f"({len(clip_bytes) / 1e6:.2f} MB)")
    result = run_vjepa2_remote.remote(clip_bytes, REFERENCE_FILENAME)

    cloud_emb = np.frombuffer(result["embedding_bytes"], dtype=np.float32).astype(np.float64)
    cloud_emb = cloud_emb.reshape(result["embedding_shape"])

    print(f"[full_pipeline] cloud emb shape     : {cloud_emb.shape}")
    print(f"[full_pipeline] cloud emb norm      : {np.linalg.norm(cloud_emb):.4f}")
    print(f"[full_pipeline] cloud wall-clock    : {result['wall_clock_s']:.2f} s "
          f"(model load: {result['model_load_s']:.1f}s, inference: {result['inference_s']:.1f}s)")
    print(f"[full_pipeline] cloud GPU           : {result['runtime_env'].get('torch_device_name')}")
    print(f"[full_pipeline] cloud transformers  : "
          f"{result['runtime_env'].get('transformers_pip_version')}")
    print(f"[full_pipeline] cloud cuDNN         : "
          f"{result['runtime_env'].get('torch_cudnn_version')}")
    n_det_warns = len(result.get("determinism_warnings", []))
    n_total_warns = result.get("all_warnings_count", 0)
    print(f"[full_pipeline] determinism warns   : {n_det_warns} of {n_total_warns} total")
    if n_det_warns > 0:
        print("[full_pipeline]   ⚠ non-deterministic ops detected (see result JSON):")
        for w in result["determinism_warnings"][:5]:
            print(f"     - {w['category']}: {w['message'][:120]}")
    else:
        print("[full_pipeline]   ✓ no determinism warnings — bit-exact reproducible run")
    print()

    # ── Stage 1: V-JEPA-2 feature equivalence ─────────────────────────────
    if cloud_emb.shape != cached_emb.shape:
        result_obj = {
            "verdict": "FAIL",
            "failure_reason": (
                f"shape mismatch: cloud={cloud_emb.shape} vs cached={cached_emb.shape}"
            ),
            "stage": "vjepa2_features",
        }
        _fail_loud("shape mismatch", result_obj)

    dot = float(np.dot(cached_emb, cloud_emb))
    cached_norm = float(np.linalg.norm(cached_emb))
    cloud_norm = float(np.linalg.norm(cloud_emb))
    cosine = dot / (cached_norm * cloud_norm)
    l2_delta = float(np.linalg.norm(cached_emb - cloud_emb))
    max_abs_delta = float(np.max(np.abs(cached_emb - cloud_emb)))

    print("[full_pipeline] Stage 1 — V-JEPA-2 feature equivalence:")
    print(f"  cosine similarity   : {cosine:.6f} (pass ≥ {COSINE_PASS})")
    print(f"  L2 distance         : {l2_delta:.4f}")
    print(f"  max abs element Δ   : {max_abs_delta:.4e}")
    stage1_pass = cosine >= COSINE_PASS

    # ── Stage 2: Classifier score equivalence ─────────────────────────────
    cloud_score = manual_decision_function(
        cloud_emb, scaler_mean, scaler_scale, ridge_coef, ridge_intercept
    )
    cloud_prob = sigmoid(cloud_score / T_MEDIAN)
    cloud_binary = 1 if cloud_prob >= TAU_EAR else 0

    score_delta = abs(local_score - cloud_score)
    prob_delta = abs(local_prob - cloud_prob)
    print()
    print("[full_pipeline] Stage 2 — Classifier score equivalence:")
    print(f"  local score         : {local_score:.6f}")
    print(f"  cloud score         : {cloud_score:.6f}")
    print(f"  |Δ| score           : {score_delta:.6f} (pass ≤ {SCORE_DELTA_PASS})")
    print(f"  |Δ| prob (T_median) : {prob_delta:.6f}")
    stage2_pass = score_delta <= SCORE_DELTA_PASS

    # ── Stage 3: Binary classification flip ───────────────────────────────
    print()
    print("[full_pipeline] Stage 3 — Binary classification:")
    print(f"  local prob          : {local_prob:.6f}  → binary = {local_binary}")
    print(f"  cloud prob          : {cloud_prob:.6f}  → binary = {cloud_binary}")
    print(f"  boundary distance   : "
          f"local |prob - τ_ear|={abs(local_prob - TAU_EAR):.4f}  "
          f"cloud |prob - τ_ear|={abs(cloud_prob - TAU_EAR):.4f}")
    binary_flip = local_binary != cloud_binary
    stage3_pass = not binary_flip

    overall_pass = stage1_pass and stage2_pass and stage3_pass

    # ── Result JSON ───────────────────────────────────────────────────────
    smoke_result = {
        "smoke_test_ran_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_clip": REFERENCE_FILENAME,
        "reference_clip_path": str(REFERENCE_CLIP_PATH.relative_to(POC)),
        "verdict": "PASS" if overall_pass else "FAIL",
        "thresholds": {
            "cosine_pass": COSINE_PASS,
            "score_delta_pass": SCORE_DELTA_PASS,
            "binary_flip_allowed": 0,
            "T_median": T_MEDIAN,
            "tau_ear": TAU_EAR,
        },
        "stage_1_vjepa2_features": {
            "pass": stage1_pass,
            "cosine_similarity": round(cosine, 6),
            "l2_distance": round(l2_delta, 4),
            "max_abs_element_delta": float(f"{max_abs_delta:.4e}"),
            "cached_norm": round(cached_norm, 4),
            "cloud_norm": round(cloud_norm, 4),
            "n_features": int(cached_emb.shape[0]),
        },
        "stage_2_classifier_score": {
            "pass": stage2_pass,
            "local_score": round(local_score, 6),
            "cloud_score": round(cloud_score, 6),
            "score_delta": round(score_delta, 6),
            "local_prob": round(local_prob, 6),
            "cloud_prob": round(cloud_prob, 6),
            "prob_delta": round(prob_delta, 6),
        },
        "stage_3_binary_classification": {
            "pass": stage3_pass,
            "local_binary": local_binary,
            "cloud_binary": cloud_binary,
            "binary_flip": binary_flip,
            "local_boundary_distance": round(abs(local_prob - TAU_EAR), 4),
            "cloud_boundary_distance": round(abs(cloud_prob - TAU_EAR), 4),
            "cached_label": cached_label,
        },
        "cloud_t4_wall_clock_s": round(result["wall_clock_s"], 2),
        "cloud_model_load_s": round(result["model_load_s"], 2),
        "cloud_inference_s": round(result["inference_s"], 2),
        "cloud_runtime_env": result["runtime_env"],
        "determinism_capture": {
            "determinism_warnings": result.get("determinism_warnings", []),
            "all_warnings_count": result.get("all_warnings_count", 0),
            "all_warnings_sample": result.get("all_warnings_sample", []),
            "interpretation": (
                "Empty determinism_warnings list with torch_deterministic_warn_only=True "
                "indicates the V-JEPA-2 forward pass hit no non-deterministic ops on CUDA; "
                "the run is effectively bit-exact reproducible across Modal workers. "
                "Non-empty list names the specific ops that fell back to non-deterministic "
                "implementations — these become audit footnotes (specific op + magnitude "
                "of nondeterminism if measurable)."
            ),
        },
        "classifier_json_path": str(CLASSIFIER_JSON.relative_to(POC)),
        "classifier_n_features": clf["n_features"],
        "classifier_n_train_clips": clf["n_train_clips"],
        "notes": (
            "Tolerance-equivalence measurement for compute-substrate transition "
            "(local MPS V-JEPA-2 → Modal CUDA V-JEPA-2). DLC keypoint substitution "
            "is validated separately by smoke_test.py. This test gates Phase 8b-CUDA "
            "retraining. PASS → retrain on Modal CUDA produces a tolerance-equivalent "
            "deployed classifier; FAIL → investigate before any production GPU run. "
            "Lesson 23 candidate (compute-substrate transitions): this JSON is the "
            "tolerance-equivalence measurement cited in the transition commit."
        ),
    }

    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(smoke_result, indent=2, default=str))

    print()
    print("=" * 70)
    print(f"[full_pipeline] verdict             : {smoke_result['verdict']}")
    print(f"[full_pipeline] Stage 1 (V-JEPA-2)  : "
          f"{'PASS' if stage1_pass else 'FAIL'} (cosine {cosine:.6f} vs ≥{COSINE_PASS})")
    print(f"[full_pipeline] Stage 2 (score)     : "
          f"{'PASS' if stage2_pass else 'FAIL'} (|Δ|={score_delta:.6f} vs ≤{SCORE_DELTA_PASS})")
    print(f"[full_pipeline] Stage 3 (binary)    : "
          f"{'PASS' if stage3_pass else 'FAIL'} (flip={binary_flip})")
    print(f"[full_pipeline] result JSON         : {RESULT_JSON.relative_to(POC.parent)}")
    print("=" * 70)

    if not overall_pass:
        if not stage1_pass:
            reason = f"V-JEPA-2 cosine {cosine:.6f} < {COSINE_PASS}"
        elif not stage2_pass:
            reason = f"score |Δ| {score_delta:.6f} > {SCORE_DELTA_PASS}"
        else:
            reason = f"binary flip: local={local_binary} cloud={cloud_binary}"
        _fail_loud(reason, smoke_result)

    print("[full_pipeline] PASS — Phase 8b-CUDA retrain authorized.")
    print("[full_pipeline] Methodology preservation verified across full feature pipeline.")
