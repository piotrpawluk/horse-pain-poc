"""Compare two DLC h5 outputs frame-by-frame, keypoint-by-keypoint.

Pass/fail criterion (locked, per smoke-test discipline ask #2):
    max |Δ|_xy ≤ 0.5 px across all (frame, bodypart, coord) entries

Likelihood differences are reported but DO NOT gate pass/fail (softmax outputs
are more sensitive to FP precision; <0.05 likelihood drift is typical between
CPU and CUDA and methodologically benign).

NaN handling: if one h5 has NaN at (frame, bodypart) and the other has a
real number, that's a STRUCTURAL mismatch (detector disagreed on whether the
bodypart was visible at that frame). Counted and reported separately; any
nonzero structural mismatch is FAIL regardless of |Δ|_xy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLD_PX = 0.5
LIKELIHOOD_REPORT_ONLY = True


def _load_dlc_h5(path: Path) -> pd.DataFrame:
    """Load DLC h5; return DataFrame with columns (scorer, bodypart, coord)."""
    df = pd.read_hdf(path)
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError(f"{path}: expected MultiIndex columns, got {type(df.columns)}")
    return df


def _xy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return only x and y coords (drop likelihood)."""
    mask = df.columns.get_level_values(-1).isin(["x", "y"])
    return df.loc[:, mask]


def _likelihood_columns(df: pd.DataFrame) -> pd.DataFrame:
    mask = df.columns.get_level_values(-1) == "likelihood"
    return df.loc[:, mask]


def compare_h5(local_path: str | Path, cloud_path: str | Path) -> dict:
    """Compare two DLC h5 outputs; return structured metrics + pass/fail."""
    local_path = Path(local_path)
    cloud_path = Path(cloud_path)

    df_local = _load_dlc_h5(local_path)
    df_cloud = _load_dlc_h5(cloud_path)

    # Structural alignment
    if df_local.shape[0] != df_cloud.shape[0]:
        return {
            "passes_threshold": False,
            "failure_reason": "FRAME_COUNT_MISMATCH",
            "local_n_frames": int(df_local.shape[0]),
            "cloud_n_frames": int(df_cloud.shape[0]),
            "threshold_px": THRESHOLD_PX,
        }

    # Column alignment — DLC SuperAnimal h5 has 4-level MultiIndex
    # (scorer, individuals, bodyparts, coords). Match by FULL tuple identity;
    # using only [-2:] (bodyparts, coords) collapses the 10 animals into 1 slot
    # and produces garbage deltas. Scorer level is identical between local CPU
    # and cloud CUDA (same model name).
    local_cols = list(df_local.columns)
    cloud_cols = list(df_cloud.columns)
    if set(local_cols) != set(cloud_cols):
        return {
            "passes_threshold": False,
            "failure_reason": "COLUMN_SET_MISMATCH",
            "local_only": sorted(str(c) for c in set(local_cols) - set(cloud_cols)),
            "cloud_only": sorted(str(c) for c in set(cloud_cols) - set(local_cols)),
            "threshold_px": THRESHOLD_PX,
        }

    # Reorder cloud columns to match local (full-tuple identity)
    df_cloud_aligned = df_cloud[local_cols]

    # XY diff
    xy_local = _xy_columns(df_local).to_numpy(dtype=np.float64)
    xy_cloud = _xy_columns(df_cloud_aligned).to_numpy(dtype=np.float64)

    nan_mismatch_mask = np.isnan(xy_local) != np.isnan(xy_cloud)
    n_nan_mismatches = int(nan_mismatch_mask.sum())

    # Only compare cells where BOTH are finite
    both_finite = np.isfinite(xy_local) & np.isfinite(xy_cloud)
    deltas = np.abs(xy_local - xy_cloud)
    deltas_finite = np.where(both_finite, deltas, np.nan)

    max_delta_xy = float(np.nanmax(deltas_finite)) if np.any(both_finite) else float("nan")
    mean_delta_xy = float(np.nanmean(deltas_finite)) if np.any(both_finite) else float("nan")
    p99_delta_xy = float(np.nanpercentile(deltas_finite, 99)) if np.any(both_finite) else float("nan")
    p999_delta_xy = float(np.nanpercentile(deltas_finite, 99.9)) if np.any(both_finite) else float("nan")

    # Per-bodypart max delta
    bodyparts = sorted(set(c[-2] for c in df_local.columns if c[-1] in ("x", "y")))
    per_bodypart_max = {}
    for bp in bodyparts:
        bp_cols_local = [c for c in df_local.columns if c[-2] == bp and c[-1] in ("x", "y")]
        bp_cols_cloud = [c for c in df_cloud_aligned.columns if c[-2] == bp and c[-1] in ("x", "y")]
        bp_local = df_local[bp_cols_local].to_numpy(dtype=np.float64)
        bp_cloud = df_cloud_aligned[bp_cols_cloud].to_numpy(dtype=np.float64)
        bp_finite = np.isfinite(bp_local) & np.isfinite(bp_cloud)
        if not np.any(bp_finite):
            per_bodypart_max[bp] = None
            continue
        bp_delta = np.where(bp_finite, np.abs(bp_local - bp_cloud), np.nan)
        per_bodypart_max[bp] = round(float(np.nanmax(bp_delta)), 4)

    # Likelihood diff (report-only)
    lik_local = _likelihood_columns(df_local).to_numpy(dtype=np.float64)
    lik_cloud = _likelihood_columns(df_cloud_aligned).to_numpy(dtype=np.float64)
    lik_both_finite = np.isfinite(lik_local) & np.isfinite(lik_cloud)
    if np.any(lik_both_finite):
        lik_delta = np.where(lik_both_finite, np.abs(lik_local - lik_cloud), np.nan)
        max_delta_likelihood = float(np.nanmax(lik_delta))
        mean_delta_likelihood = float(np.nanmean(lik_delta))
    else:
        max_delta_likelihood = float("nan")
        mean_delta_likelihood = float("nan")

    # Pass/fail
    passes = (max_delta_xy <= THRESHOLD_PX) and (n_nan_mismatches == 0)
    failure_reason = None
    if not passes:
        if n_nan_mismatches > 0:
            failure_reason = f"STRUCTURAL_NAN_MISMATCH ({n_nan_mismatches} cells)"
        elif max_delta_xy > THRESHOLD_PX:
            failure_reason = f"XY_DELTA_EXCEEDS_THRESHOLD ({max_delta_xy:.4f} > {THRESHOLD_PX})"
        else:
            failure_reason = "UNKNOWN"

    return {
        "passes_threshold": bool(passes),
        "failure_reason": failure_reason,
        "threshold_px": THRESHOLD_PX,
        "max_delta_xy_px": round(max_delta_xy, 4),
        "mean_delta_xy_px": round(mean_delta_xy, 4),
        "p99_delta_xy_px": round(p99_delta_xy, 4),
        "p999_delta_xy_px": round(p999_delta_xy, 4),
        "max_delta_likelihood": round(max_delta_likelihood, 6),
        "mean_delta_likelihood": round(mean_delta_likelihood, 6),
        "n_frames": int(df_local.shape[0]),
        "n_bodyparts": len(bodyparts),
        "n_nan_mismatches": n_nan_mismatches,
        "per_bodypart_max_delta_px": per_bodypart_max,
        "local_h5": str(local_path),
        "cloud_h5": str(cloud_path),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare two DLC h5 outputs.")
    parser.add_argument("local_h5", type=Path)
    parser.add_argument("cloud_h5", type=Path)
    parser.add_argument("--json", action="store_true", help="emit raw JSON only")
    args = parser.parse_args()

    result = compare_h5(args.local_h5, args.cloud_h5)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        verdict = "PASS" if result["passes_threshold"] else "FAIL"
        print(f"[validate] verdict           : {verdict}")
        print(f"[validate] max |Δ|_xy        : {result['max_delta_xy_px']} px "
              f"(threshold {result['threshold_px']} px)")
        print(f"[validate] mean |Δ|_xy       : {result['mean_delta_xy_px']} px")
        print(f"[validate] p99 |Δ|_xy        : {result['p99_delta_xy_px']} px")
        print(f"[validate] max |Δ|_likelihood: {result['max_delta_likelihood']}")
        print(f"[validate] n frames          : {result['n_frames']}")
        print(f"[validate] n bodyparts       : {result['n_bodyparts']}")
        print(f"[validate] n NaN mismatches  : {result['n_nan_mismatches']}")
        if result["failure_reason"]:
            print(f"[validate] failure reason    : {result['failure_reason']}")
    raise SystemExit(0 if result["passes_threshold"] else 1)
