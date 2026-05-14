#!/usr/bin/env python3
"""Phase 10a-prelim Step 1 — Prudnik transfer test on the labeled subset.

Pipeline (per `outputs/track_b_phase10a_prelim_preregistration.md`):

    Subset (D1) → DLC ear-keypoint inference (D2) → ear-bbox crop (Phase 8b
    locked geometry) → V-JEPA-2 ViT-L features → deployed classifier (D3
    JSON) → temperature scaling T=0.494 (D4) → pooled AUC + bootstrap CI
    + permutation p + ECE + per-clip residuals + reliability diagram → 3-
    band verdict (D6) routing Phase 10a-full continuation.

Sanity-check posture (user-locked):
  1. Bit-exact classifier reproducibility from JSON (re-verified here)
  2. DLC inference time warning if > 4× expected (~3 s/clip)
  3. G6 trivial PASS check (T > 0 monotonic → pre-cal AUC = post-cal AUC)
  4. First-clip full pipeline trace before chewing through all 79

Resumable: per-clip DLC keypoints cached in JSON, V-JEPA-2 features in NPZ.
Re-run skips already-processed clips automatically.

Usage:
    python tools/phase10a_prelim_run.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

POC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC / "tools"))

# Reuse Phase 8b cropping + V-JEPA-2 helpers
from phase8b_dlc_ear_crop import (  # noqa: E402
    apply_margin_and_square,
    confident_indices,
    enclosing_rect,
    select_clip_fallback_frame,
)
from extract_vjepa2 import (  # noqa: E402
    extract_embedding,
    pick_device,
)

# --- Paths ---------------------------------------------------------------
LABELS_CSV = POC / "data" / "prudnik" / "labels_pending.csv"
INVENTORY_CSV = POC / "data" / "prudnik" / "inventory.csv"
VIDEO_DIR = POC / "data" / "prudnik"
DEPLOYED_JSON = POC / "outputs" / "phase10a_prelim_deployed_classifier.json"
DLC_RESULTS_DIR = POC / "outputs" / "phase10a_prelim_dlc_outputs"
DLC_CACHE = POC / "outputs" / "phase10a_prelim_dlc_keypoints.json"
VJEPA2_CACHE = POC / "outputs" / "phase10a_prelim_vjepa2_features.npz"
CROPPED_DIR = POC / "data" / "prudnik" / "ear_crops_phase10a"

OUT_RESULTS = POC / "outputs" / "phase10a_prelim_results.json"
OUT_EXTRAS = POC / "outputs" / "phase10a_prelim_audit_extras.json"
OUT_FIG = POC / "outputs" / "phase10a_prelim_reliability_diagram.png"

# --- Locked constants from pre-reg --------------------------------------
T_MEDIAN = 0.494  # D4 — Phase 8c per-fold T median
VJEPA2_MIN_FRAMES = 16  # D1 filter
BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 42
PERMUTATION_B = 1000
PERMUTATION_SEED = 42
N_ECE_BINS = 10
ECE_BIN_FLOOR = 2  # min(n_bins, max(2, n//5)) per Phase 8c convention
DECISION_BOUNDARY = 0.0  # RidgeClassifier.decision_function default

# DLC params (carry from Phase 8b)
SUPERANIMAL_NAME = "superanimal_quadruped"
MODEL_NAME = "hrnet_w32"
DETECTOR_NAME = "fasterrcnn_resnet50_fpn_v2"
PSEUDO_THRESHOLD = 0.1
VIDEO_ADAPT = False

# Ear-bbox config (carry from Phase 8b)
EAR_KEYPOINT_NAMES = ["right_earbase", "right_earend", "left_earbase", "left_earend"]
EAR_CONF_THRESHOLD = 0.5
EAR_MIN_CONFIDENT_KPS = 3  # ≥ 3 of 4 ear keypoints confident
MARGIN_PCT = 0.15
TARGET_SIZE = 224

# 3-band verdict thresholds (D6)
VERDICT_STRONG_AUC = 0.75
VERDICT_AMBIGUOUS_LOW_AUC = 0.60

# Sanity-check timing — corrected from pre-reg estimate after verifying
# Phase 8b's actual was 31.1 s/clip wall (8813.3s / 283 clips on RME, audit
# doc Step 4). Pre-reg's "~3 s/clip" was off by 10×. Threshold = 2× actual.
DLC_EXPECTED_S_PER_CLIP = 31.0
DLC_TIMEOUT_FACTOR = 2.0


# -----------------------------------------------------------------------
# Module globals (populated in main)
# -----------------------------------------------------------------------
DEVICE: str = ""
DEPLOYED: dict = {}


def atomic_write_json(obj, path: Path) -> None:
    """Atomic JSON write via .tmp + os.replace."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


# -----------------------------------------------------------------------
# 1. Deployed classifier — load + bit-exact reproducibility (sanity #1)
# -----------------------------------------------------------------------


def load_deployed_classifier() -> dict:
    """Load the JSON-dumped deployed classifier from Step 1.5."""
    if not DEPLOYED_JSON.exists():
        print(
            f"[ERROR] Deployed classifier not found at {DEPLOYED_JSON}.\n"
            f"        Run Step 1.5 first:\n"
            f"        python tools/phase10a_train_deployed_classifier.py",
            file=sys.stderr,
        )
        raise SystemExit(1)
    params = json.loads(DEPLOYED_JSON.read_text())
    # Convert to numpy for inference
    params["_scaler_mean_np"] = np.asarray(params["scaler_mean"])
    params["_scaler_scale_np"] = np.asarray(params["scaler_scale"])
    params["_coef_np"] = np.asarray(params["ridge_coef"])
    params["_intercept_f"] = float(params["ridge_intercept"])
    return params


def predict_decision_function(feature: np.ndarray, params: dict) -> float:
    """Manual reproduction of sklearn Ridge.decision_function + StandardScaler."""
    scaled = (feature - params["_scaler_mean_np"]) / params["_scaler_scale_np"]
    return float(np.dot(scaled, params["_coef_np"]) + params["_intercept_f"])


def verify_classifier_reproducibility(params: dict) -> None:
    """Sanity check #1: bit-exact reload+predict on the original training data."""
    print("[phase10a-prelim] Sanity #1: bit-exact classifier reproducibility…")
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler

    emb_path = POC / "outputs" / "vjepa2_embeddings_ear_v4.npz"
    npz = np.load(emb_path, allow_pickle=True)
    X = npz["embs"].astype(np.float64)
    y = npz["labels"].astype(np.int64)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = RidgeClassifier(alpha=params["alpha"], class_weight=params["class_weight"])
    clf.fit(Xs, y)

    max_delta = 0.0
    for idx in range(min(3, X.shape[0])):
        sklearn_score = float(clf.decision_function(scaler.transform(X[idx : idx + 1]))[0])
        manual_score = predict_decision_function(X[idx], params)
        max_delta = max(max_delta, abs(sklearn_score - manual_score))

    print(f"  max |Δ| across 3 sample clips: {max_delta:.2e}")
    if max_delta > 1e-10:
        print(
            f"[ERROR] Bit-exact reload check FAILED (max |Δ|={max_delta:.2e} > 1e-10). "
            f"Classifier JSON may be stale or corrupted. Halting.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    print("  ✓ bit-exact (≤ 1e-10)")


# -----------------------------------------------------------------------
# 2. Subset selection (D1)
# -----------------------------------------------------------------------


def select_subset() -> pd.DataFrame:
    """Filter labels_pending.csv per D1 (locked).

    Filters:
      - label != ""                            (actually labeled)
      - label not in ("action?", "background?") (exclude borderlines)
      - confidence != "low"
      - archived_full_clip_label == ""         (exclude split-roots)
      - duration_s * fps ≥ 16                  (V-JEPA-2 minimum)
    """
    df = pd.read_csv(LABELS_CSV, keep_default_na=False, dtype=str)
    inv = pd.read_csv(INVENTORY_CSV, keep_default_na=False, dtype=str)

    df = df.merge(
        inv[["clip_id", "fps", "resolution"]],
        on="clip_id",
        how="left",
        suffixes=("", "_inv"),
    )

    df["dur_num"] = df["duration_s"].astype(float)
    df["fps_num"] = df["fps"].astype(float)
    df["min_frames_needed"] = (df["dur_num"] * df["fps_num"]).astype(float)

    mask = (
        (df["label"] != "")
        & (~df["label"].isin(["action?", "background?"]))
        & (df["confidence"] != "low")
        & (df.get("archived_full_clip_label", "") == "")
        & (df["min_frames_needed"] >= float(VJEPA2_MIN_FRAMES))
    )
    subset = df[mask].copy()
    subset["label_binary"] = (subset["label"] == "action").astype(int)
    subset = subset.reset_index(drop=True)
    return subset


# -----------------------------------------------------------------------
# 3. DLC inference (with cache + timing — sanity #2)
# -----------------------------------------------------------------------


def run_dlc_on_clip(clip_path: Path, out_dir: Path) -> Path:
    """Run DLC SuperAnimal-Quadruped inference on one clip.

    Returns the path to the resulting H5 keypoints file.
    """
    import deeplabcut

    out_dir.mkdir(parents=True, exist_ok=True)
    deeplabcut.video_inference_superanimal(
        videos=[str(clip_path)],
        superanimal_name=SUPERANIMAL_NAME,
        model_name=MODEL_NAME,
        detector_name=DETECTOR_NAME,
        video_adapt=VIDEO_ADAPT,
        pseudo_threshold=PSEUDO_THRESHOLD,
        dest_folder=str(out_dir),
    )
    # DLC writes results to dest_folder with a derived name
    stem = clip_path.stem
    candidates = list(out_dir.glob(f"{stem}*.h5"))
    if not candidates:
        raise RuntimeError(f"DLC produced no .h5 for {clip_path.name}")
    return candidates[0]


def parse_dlc_h5_for_ears(h5_path: Path) -> list[dict]:
    """Read DLC h5 → per-frame ear keypoints in Phase 8b's bbox format.

    Returns list of per-frame dicts: {kp_name: {x, y, confidence}}
    Filters to only ear keypoints (the 4 we care about).
    """
    df = pd.read_hdf(h5_path)
    # DLC 3.0 multi-individual format: columns are MultiIndex
    # (scorer, individual, bodypart, coord). For SuperAnimal we have
    # individual=animal_X. Use the first individual ("animal_1" or similar).
    if df.empty:
        return []

    # Find the column structure
    if isinstance(df.columns, pd.MultiIndex):
        # Get all individuals
        levels = df.columns.names
        if "individuals" in levels:
            individuals = df.columns.get_level_values("individuals").unique().tolist()
            # Pick first individual (the primary subject)
            primary = individuals[0]
            df_p = df.xs(primary, level="individuals", axis=1)
        else:
            df_p = df
        # Drop scorer level too so columns are just (bodyparts, coords) and
        # row[(bp, "x")] indexing works. xs above only removed individuals.
        if "scorer" in df_p.columns.names:
            df_p = df_p.droplevel("scorer", axis=1)
        bodyparts = df_p.columns.get_level_values("bodyparts").unique().tolist()
    else:
        bodyparts = []
        df_p = df

    per_frame = []
    for frame_idx in range(len(df_p)):
        row = df_p.iloc[frame_idx]
        frame_kps: dict[str, dict] = {}
        for bp in EAR_KEYPOINT_NAMES:
            if bp in bodyparts:
                try:
                    x = float(row[(bp, "x")])
                    y = float(row[(bp, "y")])
                    c = float(row[(bp, "likelihood")])
                    frame_kps[bp] = {"x": x, "y": y, "confidence": c}
                except (KeyError, ValueError):
                    pass
        per_frame.append(frame_kps)
    return per_frame


def load_or_compute_dlc(subset: pd.DataFrame) -> dict:
    """Run DLC on the subset using BATCH mode (single DLC call with all
    uncached videos). Matches Phase 8b's pattern; avoids per-call model-load
    overhead. Returns {clip_id: per_frame_keypoints}.
    """
    cache = {}
    if DLC_CACHE.exists():
        cache = json.loads(DLC_CACHE.read_text())
        print(f"  loaded {len(cache)} cached DLC results from {DLC_CACHE.name}")

    DLC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build list of clips that still need DLC inference
    todo = []
    clip_paths_by_id: dict[str, Path] = {}
    for _, row in subset.iterrows():
        clip_id = row["clip_id"]
        if clip_id in cache:
            continue
        clip_path = VIDEO_DIR / row["filename"]
        if not clip_path.exists():
            print(f"    ⚠ skip {clip_id}: file missing {row['filename']}")
            cache[clip_id] = []
            continue
        # Check if h5 already exists in DLC_RESULTS_DIR (partial recovery)
        existing_h5 = list(DLC_RESULTS_DIR.glob(f"{clip_path.stem}*.h5"))
        if existing_h5:
            try:
                per_frame = parse_dlc_h5_for_ears(existing_h5[0])
                cache[clip_id] = per_frame
                print(f"    recovered {clip_id} from existing h5 ({len(per_frame)} frames)")
                continue
            except Exception as e:
                print(f"    ⚠ failed to parse existing h5 for {clip_id}: {e}")
        todo.append((clip_id, clip_path))
        clip_paths_by_id[clip_id] = clip_path

    if not todo:
        print(f"  all {len(subset)} clips already have DLC results; skipping inference")
        atomic_write_json(cache, DLC_CACHE)
        return cache

    expected_total_s = len(todo) * DLC_EXPECTED_S_PER_CLIP
    print(
        f"  running DLC on {len(todo)} clips in batch mode "
        f"(expected ~{expected_total_s / 60:.0f} min @ {DLC_EXPECTED_S_PER_CLIP:.0f}s/clip per Phase 8b reference)…",
        flush=True,
    )
    print(
        f"  ⚠ This is the slow step. Sanity #2 threshold: warn if > "
        f"{DLC_TIMEOUT_FACTOR}× expected = {expected_total_s * DLC_TIMEOUT_FACTOR / 60:.0f} min.",
        flush=True,
    )

    t0 = time.time()
    import deeplabcut

    video_paths = [str(p) for _, p in todo]
    try:
        deeplabcut.video_inference_superanimal(
            videos=video_paths,
            superanimal_name=SUPERANIMAL_NAME,
            model_name=MODEL_NAME,
            detector_name=DETECTOR_NAME,
            video_adapt=VIDEO_ADAPT,
            pseudo_threshold=PSEUDO_THRESHOLD,
            dest_folder=str(DLC_RESULTS_DIR),
            create_labeled_video=False,
            plot_bboxes=False,
        )
    except Exception as e:
        print(f"  ⚠ batch DLC call raised: {e}")
        print("     — attempting to parse any h5 files that DID write before the error")

    elapsed = time.time() - t0
    print(
        f"  DLC batch call elapsed: {elapsed:.1f}s "
        f"(expected ~{expected_total_s:.0f}s; {elapsed/max(1, expected_total_s):.1f}× ratio)"
    )
    if elapsed > expected_total_s * DLC_TIMEOUT_FACTOR:
        print(
            f"  ⚠ Sanity #2 FIRED: DLC took > {DLC_TIMEOUT_FACTOR}× expected. "
            f"Possible causes: orientation handling, GPU not engaged, CPU fallback. "
            f"Note in audit doc."
        )

    # Parse each clip's h5 result
    print("  parsing per-clip h5 outputs…", flush=True)
    for clip_id, clip_path in todo:
        stem = clip_path.stem
        h5_candidates = list(DLC_RESULTS_DIR.glob(f"{stem}*.h5"))
        if not h5_candidates:
            print(f"    ⚠ {clip_id}: no h5 produced")
            cache[clip_id] = []
            continue
        try:
            per_frame = parse_dlc_h5_for_ears(h5_candidates[0])
            cache[clip_id] = per_frame
        except Exception as e:
            print(f"    ⚠ {clip_id}: h5 parse failed: {e}")
            cache[clip_id] = []

    atomic_write_json(cache, DLC_CACHE)
    n_ok = sum(1 for v in cache.values() if v)
    print(f"  cached {n_ok}/{len(cache)} clips with valid DLC outputs")
    return cache


# -----------------------------------------------------------------------
# 4. Ear-crop pipeline (reuse Phase 8b helpers)
# -----------------------------------------------------------------------


def keypoints_to_phase8b_format(per_frame_kps: list[dict]) -> list[list[tuple]]:
    """Convert our DLC parse output to Phase 8b's confident_indices format.

    Phase 8b's `confident_indices` indexes EAR_KP_INDICES = [6, 7, 11, 12] into
    a list of (x, y, confidence) tuples in SuperAnimal-Quadruped canonical
    order (where 6=right_earbase, 7=right_earend, 11=left_earbase, 12=left_earend).
    We build a 13-element list per frame with placeholder (0,0,0) tuples for
    non-ear positions and the actual ear keypoints at indices 6, 7, 11, 12.
    """
    NAME_TO_IDX = {
        "right_earbase": 6,
        "right_earend": 7,
        "left_earbase": 11,
        "left_earend": 12,
    }
    out = []
    for frame in per_frame_kps:
        frame_list = [(0.0, 0.0, 0.0)] * 13
        for name, idx in NAME_TO_IDX.items():
            kp = frame.get(name)
            if kp is not None:
                frame_list[idx] = (kp["x"], kp["y"], kp["confidence"])
        out.append(frame_list)
    return out


def compute_ear_bboxes(
    per_frame_kps_list: list[list[dict]],
    frame_shape: tuple[int, int],
) -> tuple[list[tuple | None], int]:
    """Per-frame ear bbox following Phase 8b's locked geometry.

    Returns (per_frame_bbox, n_fallback_frames). Each bbox is (x1, y1, x2, y2)
    in pixel coords, post-margin + square-pad to TARGET_SIZE-friendly aspect.

    Frames with <3 of 4 confident → fall back to clip-level single-middle-frame.
    """
    n_frames = len(per_frame_kps_list)
    per_frame_confident = []
    for kps in per_frame_kps_list:
        idx_conf = confident_indices(kps)
        per_frame_confident.append(idx_conf)

    # Phase 8b fallback rule: select_clip_fallback_frame returns
    # (frame_idx, confident_indices) — see phase8b_dlc_ear_crop.py:66.
    fallback_idx, fb_conf = select_clip_fallback_frame(per_frame_kps_list)
    fallback_rect = None
    if fallback_idx is not None and len(fb_conf) >= 1:
        fallback_kps = per_frame_kps_list[fallback_idx]
        fb_tight = enclosing_rect(fallback_kps, fb_conf)
        if fb_tight is not None:
            fallback_rect = apply_margin_and_square(fb_tight, MARGIN_PCT, frame_shape)

    per_frame_bbox = []
    n_fallback = 0
    for i, idx_conf in enumerate(per_frame_confident):
        if len(idx_conf) >= EAR_MIN_CONFIDENT_KPS:
            tight = enclosing_rect(per_frame_kps_list[i], idx_conf)
            if tight is not None:
                bbox = apply_margin_and_square(tight, MARGIN_PCT, frame_shape)
                per_frame_bbox.append(bbox)
                continue
        # fallback
        n_fallback += 1
        per_frame_bbox.append(fallback_rect)

    return per_frame_bbox, n_fallback


def crop_clip_with_bboxes(
    clip_path: Path, per_frame_bbox: list, out_path: Path
) -> bool:
    """Read clip, crop each frame per per_frame_bbox, write to out_path."""
    import cv2

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (TARGET_SIZE, TARGET_SIZE))

    frame_idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx < len(per_frame_bbox) and per_frame_bbox[frame_idx] is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in per_frame_bbox[frame_idx]]
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            cropped = frame[y1:y2, x1:x2]
            if cropped.size > 0:
                resized = cv2.resize(cropped, (TARGET_SIZE, TARGET_SIZE))
                writer.write(resized)
                written += 1
        frame_idx += 1

    cap.release()
    writer.release()
    return written > 0


# -----------------------------------------------------------------------
# 5. V-JEPA-2 feature extraction (reuse extract_vjepa2 helpers)
# -----------------------------------------------------------------------


def load_or_compute_vjepa2(
    subset: pd.DataFrame, cropped_paths: dict[str, Path]
) -> dict[str, np.ndarray]:
    """Extract V-JEPA-2 features per cropped clip. Cache to NPZ."""
    cache: dict[str, np.ndarray] = {}
    if VJEPA2_CACHE.exists():
        npz = np.load(VJEPA2_CACHE, allow_pickle=True)
        for k in npz.files:
            cache[k] = npz[k]
        print(f"  loaded {len(cache)} cached V-JEPA-2 features from {VJEPA2_CACHE.name}")

    todo = [r["clip_id"] for _, r in subset.iterrows() if r["clip_id"] not in cache]
    if not todo:
        return cache

    print("  loading V-JEPA-2 model (this may take ~30s first time)…", flush=True)
    from transformers import AutoVideoProcessor, VJEPA2Model

    global DEVICE
    DEVICE = pick_device()
    print(f"  device: {DEVICE}")
    MODEL_ID = "facebook/vjepa2-vitl-fpc16-256-ssv2"
    processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
    model = VJEPA2Model.from_pretrained(MODEL_ID).to(DEVICE).eval()

    print(f"  extracting features for {len(todo)} clips…", flush=True)
    t0 = time.time()
    for i, clip_id in enumerate(todo):
        cp = cropped_paths.get(clip_id)
        if cp is None or not cp.exists():
            print(f"    ⚠ skip {clip_id}: cropped clip missing")
            continue
        try:
            emb = extract_embedding(cp, model, processor, DEVICE)
            cache[clip_id] = emb
            if (i + 1) % 10 == 0:
                print(f"    [{i+1}/{len(todo)}] done", flush=True)
        except Exception as e:
            print(f"    ⚠ {clip_id} FAILED: {e}")

    elapsed = time.time() - t0
    print(f"  V-JEPA-2 elapsed: {elapsed:.1f}s ({elapsed/max(1,len(todo)):.1f}s/clip avg)")

    # Save cache as NPZ
    np.savez(VJEPA2_CACHE, **{k: v for k, v in cache.items() if isinstance(v, np.ndarray)})
    return cache


# -----------------------------------------------------------------------
# 6. Metrics, bootstrap, ECE
# -----------------------------------------------------------------------


def clip_bootstrap_auc_ci(
    labels: np.ndarray, scores: np.ndarray, B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float, float, list[float]]:
    """Clip-level bootstrap (with replacement) CI on AUC.

    Returns (auc_point, auc_lo, auc_hi, sample_aucs).
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), []
    auc_point = float(roc_auc_score(labels, scores))
    samples = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        ly = labels[idx]
        ls = scores[idx]
        if (ly == 0).all() or (ly == 1).all():
            continue
        samples.append(float(roc_auc_score(ly, ls)))
    if not samples:
        return auc_point, float("nan"), float("nan"), []
    lo = float(np.quantile(samples, 0.025))
    hi = float(np.quantile(samples, 0.975))
    return auc_point, lo, hi, samples


def permutation_p_value(
    labels: np.ndarray, scores: np.ndarray, B: int = PERMUTATION_B, seed: int = PERMUTATION_SEED
) -> float:
    """Permutation test: P(AUC_shuffled ≥ AUC_observed)."""
    rng = np.random.default_rng(seed)
    auc_obs = float(roc_auc_score(labels, scores))
    n_geq = 0
    for _ in range(B):
        shuffled = rng.permutation(labels)
        auc_s = float(roc_auc_score(shuffled, scores))
        if auc_s >= auc_obs:
            n_geq += 1
    return (n_geq + 1) / (B + 1)


def equal_freq_bins(probs: np.ndarray, n_bins: int) -> np.ndarray:
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(probs, quantiles)
    edges[-1] += 1e-9
    return np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int) -> tuple[float, list]:
    bins = equal_freq_bins(probs, n_bins)
    N = len(probs)
    ece = 0.0
    records = []
    for b in range(n_bins):
        mask = bins == b
        n_b = int(mask.sum())
        if n_b == 0:
            records.append({"bin": b, "n": 0})
            continue
        conf_b = float(probs[mask].mean())
        acc_b = float(labels[mask].mean())
        ece += (n_b / N) * abs(acc_b - conf_b)
        records.append({"bin": b, "n": n_b, "conf": conf_b, "acc": acc_b, "gap": acc_b - conf_b})
    return float(ece), records


def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def reliability_diagram(
    pre_cal: np.ndarray, post_cal: np.ndarray, labels: np.ndarray, out: Path, n_bins: int
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, probs, title in [
        (axes[0], pre_cal, "Pre-cal (T=1)"),
        (axes[1], post_cal, f"Post-cal (T={T_MEDIAN})"),
    ]:
        bins = equal_freq_bins(probs, n_bins)
        bd = []
        for b in range(n_bins):
            mask = bins == b
            n_b = int(mask.sum())
            if n_b == 0:
                continue
            conf_b = float(probs[mask].mean())
            acc_b = float(labels[mask].mean())
            lo, hi = wilson_ci(acc_b, n_b)
            bd.append((conf_b, acc_b, n_b, lo, hi))
        if bd:
            confs, accs, ns, los, his = (np.array(x) for x in zip(*bd))
            ax2 = ax.twinx()
            ax2.bar(confs, ns, width=0.06, alpha=0.18, color="gray")
            ax2.set_ylim(0, max(ns) * 3 if max(ns) > 0 else 1)
            ax2.set_ylabel("clips per bin", color="gray", fontsize=9)
            ax.errorbar(
                confs, accs, yerr=[accs - los, his - accs], fmt="o-", color="tab:blue",
                capsize=3, label="observed accuracy (Wilson 95%)",
            )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("predicted probability")
        ax.set_title(title)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("observed accuracy")
    fig.suptitle(
        f"Phase 10a-prelim — Prudnik transfer reliability (n={len(labels)} after D1 filters)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def verdict_band(auc: float) -> str:
    """3-band verdict (D6 locked)."""
    if auc >= VERDICT_STRONG_AUC:
        return "STRONG_TRANSFER"
    if auc >= VERDICT_AMBIGUOUS_LOW_AUC:
        return "AMBIGUOUS_TRANSFER"
    return "WEAK_OR_NO_TRANSFER"


def routing_decision(band: str) -> str:
    return {
        "STRONG_TRANSFER": "CONTINUE full Phase 10a labeling (3 days). Expect Phase 10a-full to confirm with tighter CI.",
        "AMBIGUOUS_TRANSFER": "CONTINUE labeling AND investigate per-clip residuals (L1 selection bias, domain shift, or DLC reliability).",
        "WEAK_OR_NO_TRANSFER": "PAUSE labeling. Diagnose first — portrait orientation (data_quality_notes §1) is the leading hypothesis.",
    }[band]


# -----------------------------------------------------------------------
# main orchestration
# -----------------------------------------------------------------------


def main() -> int:
    print("=" * 70)
    print("PHASE 10a-prelim — Prudnik transfer test")
    print("Pre-reg: outputs/track_b_phase10a_prelim_preregistration.md")
    print("=" * 70)
    print()

    # Step 0: load classifier + bit-exact reproducibility check (sanity #1)
    global DEPLOYED
    DEPLOYED = load_deployed_classifier()
    print(
        f"[phase10a-prelim] Loaded deployed classifier (Step 1.5 output): "
        f"{DEPLOYED['n_features']} features, in-sample RME AUC = "
        f"{DEPLOYED['in_sample_auc_on_RME']:.4f}"
    )
    verify_classifier_reproducibility(DEPLOYED)
    print()

    # Step 1: subset selection (D1)
    print("[phase10a-prelim] D1 — selecting subset…")
    subset = select_subset()
    print(
        f"  n = {len(subset)}  "
        f"({(subset['label'] == 'action').sum()} action, "
        f"{(subset['label'] == 'background').sum()} background)"
    )
    if len(subset) < 10:
        print("[ERROR] subset too small to run preliminary; halting.", file=sys.stderr)
        return 1
    print()

    # Step 2: DLC inference (with cache, sanity #2 timing)
    print("[phase10a-prelim] D2 step 1 — DLC SuperAnimal-Quadruped inference (with cache)…")
    keypoints = load_or_compute_dlc(subset)
    print()

    # Step 3: ear crop per Phase 8b geometry
    print("[phase10a-prelim] D2 step 2 — ear-bbox crop (Phase 8b locked geometry)…")
    CROPPED_DIR.mkdir(parents=True, exist_ok=True)
    cropped_paths = {}
    crop_diag = {}
    import cv2

    for i, row in subset.iterrows():
        clip_id = row["clip_id"]
        clip_path = VIDEO_DIR / row["filename"]
        per_frame = keypoints.get(clip_id, [])
        if not per_frame or not clip_path.exists():
            print(f"  ⚠ skip {clip_id}: no keypoints or missing video")
            continue

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            continue
        ok, frame0 = cap.read()
        cap.release()
        if not ok or frame0 is None:
            continue
        frame_shape = (frame0.shape[0], frame0.shape[1])

        kps_phase8b = keypoints_to_phase8b_format(per_frame)
        per_frame_bbox, n_fallback = compute_ear_bboxes(kps_phase8b, frame_shape)
        out_clip = CROPPED_DIR / f"{clip_id}_eared.mp4"
        if out_clip.exists():
            cropped_paths[clip_id] = out_clip
            crop_diag[clip_id] = {"n_frames": len(per_frame), "n_fallback": n_fallback, "cached": True}
            continue
        ok = crop_clip_with_bboxes(clip_path, per_frame_bbox, out_clip)
        if ok:
            cropped_paths[clip_id] = out_clip
            crop_diag[clip_id] = {"n_frames": len(per_frame), "n_fallback": n_fallback, "cached": False}
        else:
            print(f"  ⚠ crop failed for {clip_id}")

    print(f"  cropped clips ready: {len(cropped_paths)} / {len(subset)}")
    print()

    # Step 4: V-JEPA-2 feature extraction
    print("[phase10a-prelim] D2 step 3 — V-JEPA-2 ViT-L feature extraction (with cache)…")
    features = load_or_compute_vjepa2(subset, cropped_paths)
    print(f"  features ready: {len(features)} / {len(subset)}")
    print()

    # Step 5: predict per clip
    print("[phase10a-prelim] D2 step 4 + D4 — apply deployed classifier + temperature scaling…")
    per_clip = []
    for _, row in subset.iterrows():
        clip_id = row["clip_id"]
        feat = features.get(clip_id)
        if feat is None:
            continue
        feat = np.asarray(feat, dtype=np.float64).flatten()
        if feat.shape[0] != DEPLOYED["n_features"]:
            print(f"  ⚠ {clip_id}: feature dim {feat.shape[0]} != {DEPLOYED['n_features']}")
            continue
        score = predict_decision_function(feat, DEPLOYED)
        prob_pre = float(sigmoid(np.array([score]))[0])
        prob_post = float(sigmoid(np.array([score / T_MEDIAN]))[0])
        per_clip.append(
            {
                "clip_id": clip_id,
                "filename": row["filename"],
                "frame_type": row["frame_type"],
                "label": row["label"],
                "label_binary": int(row["label_binary"]),
                "multi_horse": row["multi_horse"],
                "confidence": row["confidence"],
                "duration_s": float(row["dur_num"]),
                "fps": float(row["fps_num"]),
                "decision_score": score,
                "prob_pre_cal": prob_pre,
                "prob_post_cal": prob_post,
                "correct_at_boundary": int((score >= 0) == (row["label"] == "action")),
                "dlc_n_frames": crop_diag.get(clip_id, {}).get("n_frames", 0),
                "dlc_n_fallback": crop_diag.get(clip_id, {}).get("n_fallback", 0),
            }
        )

    if len(per_clip) < 10:
        print(f"[ERROR] only {len(per_clip)} per-clip predictions; halting.", file=sys.stderr)
        return 1
    print(f"  per-clip predictions: {len(per_clip)} (subset was {len(subset)})")
    print()

    # Sanity #4: first-clip trace
    print("[phase10a-prelim] Sanity #4 — first-clip full pipeline trace:")
    first = per_clip[0]
    print(f"  clip:         {first['clip_id']}  ({first['label']}, conf={first['confidence']})")
    print(f"  duration:     {first['duration_s']:.2f}s @ {first['fps']:.1f} fps")
    print(f"  DLC frames:   {first['dlc_n_frames']}, fallback fired: {first['dlc_n_fallback']}")
    feat0 = features[first["clip_id"]].flatten()
    print(f"  V-JEPA-2:     dim={len(feat0)}, mean={feat0.mean():.4f}, "
          f"std={feat0.std():.4f}, min={feat0.min():.4f}, max={feat0.max():.4f}")
    print(f"  decision:     score={first['decision_score']:+.4f}  "
          f"pre-cal prob={first['prob_pre_cal']:.4f}  post-cal prob={first['prob_post_cal']:.4f}")
    print(f"  correct?      {first['correct_at_boundary']} (decision > 0 ↔ label = action)")
    print()

    # Step 6: metrics
    print("[phase10a-prelim] computing metrics…")
    labels_arr = np.array([r["label_binary"] for r in per_clip])
    scores_arr = np.array([r["decision_score"] for r in per_clip])
    pre_cal = np.array([r["prob_pre_cal"] for r in per_clip])
    post_cal = np.array([r["prob_post_cal"] for r in per_clip])

    auc_point, auc_lo, auc_hi, _ = clip_bootstrap_auc_ci(labels_arr, scores_arr)
    perm_p = permutation_p_value(labels_arr, scores_arr)
    n_bins_eff = min(N_ECE_BINS, max(ECE_BIN_FLOOR, len(per_clip) // 5))
    ece_pre, ece_bins_pre = compute_ece(pre_cal, labels_arr, n_bins_eff)
    ece_post, ece_bins_post = compute_ece(post_cal, labels_arr, n_bins_eff)

    # G6: pre-cal AUC = post-cal AUC bit-exact (T > 0 monotonic)
    auc_pre = float(roc_auc_score(labels_arr, pre_cal))
    auc_post = float(roc_auc_score(labels_arr, post_cal))
    g6_pass = abs(auc_pre - auc_post) <= 1e-10

    band = verdict_band(auc_point)
    routing = routing_decision(band)

    # Frame_type subgroup (Test 7) — n ≥ 20 guard
    by_ft = {}
    for ft in ("head-zoom", "full-body"):
        ft_mask = np.array([r["frame_type"] == ft for r in per_clip])
        n_ft = int(ft_mask.sum())
        if n_ft >= 20:
            try:
                auc_ft = float(roc_auc_score(labels_arr[ft_mask], scores_arr[ft_mask]))
                by_ft[ft] = {"n": n_ft, "auc": auc_ft, "auc_reported": True}
            except ValueError:
                by_ft[ft] = {"n": n_ft, "auc": None, "auc_reported": False, "reason": "degenerate labels"}
        else:
            by_ft[ft] = {"n": n_ft, "auc": None, "auc_reported": False, "reason": "n<20 (guard per Test 7)"}

    print(f"  pooled AUC:    {auc_point:.4f}  95% CI [{auc_lo:.4f}, {auc_hi:.4f}]")
    print(f"  permutation p: {perm_p:.4f}")
    print(f"  ECE pre-cal:   {ece_pre:.4f}")
    print(f"  ECE post-cal:  {ece_post:.4f}")
    print(f"  G6 pre==post AUC: {'PASS' if g6_pass else 'FAIL'}  (|Δ|={abs(auc_pre-auc_post):.2e})")
    print(f"  verdict band:  {band}")
    print(f"  routing:       {routing}")
    print()

    # Step 7: reliability diagram
    print("[phase10a-prelim] writing reliability diagram…")
    reliability_diagram(pre_cal, post_cal, labels_arr, OUT_FIG, n_bins_eff)
    print(f"  {OUT_FIG.relative_to(POC)}")
    print()

    # Step 8: write results JSON
    results = {
        "mode": "phase10a_prelim",
        "preregistration": "outputs/track_b_phase10a_prelim_preregistration.md",
        "n_subset": len(per_clip),
        "n_action": int((labels_arr == 1).sum()),
        "n_background": int((labels_arr == 0).sum()),
        "pooled_auc": auc_point,
        "pooled_auc_bootstrap_95ci": [auc_lo, auc_hi],
        "pre_cal_auc": auc_pre,
        "post_cal_auc": auc_post,
        "g6_auc_invariance_pass": g6_pass,
        "permutation_p_vs_chance": perm_p,
        "ece_pre_cal": ece_pre,
        "ece_post_cal": ece_post,
        "ece_n_bins_used": n_bins_eff,
        "verdict_band": band,
        "routing_recommendation": routing,
        "frame_type_subgroup_auc": by_ft,
        "operating_point_evaluation": "DEFERRED to Phase 10a-full per D5 (n_neg too small)",
        "limitations_load_bearing": {
            "L1_selection_bias": (
                "n=91 was filtered to high/medium-confidence, clearly-classifiable; "
                "remaining 180 derived rows are likely harder. Preliminary AUC is "
                "likely an UPPER BOUND on full Phase 10a-full."
            ),
            "L2_single_source": "Prudnik = 1 source; clip-bootstrap not source-bootstrap.",
            "L3_n_neg_16": "n_neg too small for operating-point analysis.",
            "L6_portrait_orientation": "All clips portrait; leading hypothesis if WEAK_OR_NO_TRANSFER.",
        },
        "config": {
            "deployed_classifier": str(DEPLOYED_JSON.relative_to(POC)),
            "t_median_used": T_MEDIAN,
            "bootstrap_B": BOOTSTRAP_B,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "permutation_B": PERMUTATION_B,
            "decision_boundary": DECISION_BOUNDARY,
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(results, OUT_RESULTS)
    print(f"[phase10a-prelim] wrote {OUT_RESULTS.relative_to(POC)}")

    extras = {
        "per_clip": per_clip,
        "ece_bins_pre_cal": ece_bins_pre,
        "ece_bins_post_cal": ece_bins_post,
        "dlc_per_clip_summary": crop_diag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(extras, OUT_EXTRAS)
    print(f"[phase10a-prelim] wrote {OUT_EXTRAS.relative_to(POC)}")
    print()

    # Final summary
    print("=" * 70)
    print("PHASE 10a-prelim SUMMARY")
    print("=" * 70)
    print(f"  n={len(per_clip)}  ({(labels_arr==1).sum()} action / {(labels_arr==0).sum()} bg)")
    print(f"  pooled AUC = {auc_point:.4f}  95% CI [{auc_lo:.4f}, {auc_hi:.4f}]")
    print(f"  permutation p (vs chance) = {perm_p:.4f}")
    print(f"  ECE pre/post: {ece_pre:.4f} → {ece_post:.4f}")
    print()
    print(f"  VERDICT BAND: {band}  (AUC threshold: {VERDICT_STRONG_AUC} / {VERDICT_AMBIGUOUS_LOW_AUC})")
    print(f"  ROUTING:      {routing}")
    print()
    print(
        f"  ⚠ L1 (selection bias): n={len(per_clip)} is NOT a random sample of Prudnik. "
        f"Pooled AUC is likely an UPPER BOUND on full Phase 10a-full."
    )
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
