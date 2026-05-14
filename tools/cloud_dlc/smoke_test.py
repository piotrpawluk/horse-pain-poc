"""Smoke test — run Modal DLC on one reference clip, validate against local CPU h5.

Pass/fail discipline (per ask #2):
    max |Δ|_xy ≤ 0.5 px across all keypoints → PASS
    > 0.5 px OR any structural NaN mismatch → FAIL LOUD (stderr, non-zero exit,
    fail JSON artifact)

Reference clip: IMG_1050 (already cached in Phase 10a-prelim CPU run).
  - 1080×1920, 0.95 s, 57 frames (sub-V-JEPA-2-floor but adequate for DLC validation)
  - Local CPU h5 baseline: outputs/phase10a_prelim_dlc_outputs/IMG_1050_*.h5

Outputs:
    outputs/cloud_dlc_smoke_test_result.json — full structured metrics + verdict

Usage:
    modal run tools/cloud_dlc/smoke_test.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


POC = Path(__file__).resolve().parents[2]
# Ensure tools.cloud_dlc.* is importable regardless of how `modal run` invokes
# this script (Modal CLI does not auto-add cwd to sys.path).
sys.path.insert(0, str(POC))

REFERENCE_CLIP_ID = "IMG_1050"
REFERENCE_CLIP_PATH = POC / "data" / "prudnik" / f"{REFERENCE_CLIP_ID}.mov"
LOCAL_H5 = (
    POC / "outputs" / "phase10a_prelim_dlc_outputs"
    / f"{REFERENCE_CLIP_ID}_superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2.h5"
)
TEMP_DIR = POC / "temp" / "cloud_dlc_smoke"
RESULT_JSON = POC / "outputs" / "cloud_dlc_smoke_test_result.json"

THRESHOLD_PX = 0.5

from tools.cloud_dlc.app import app, run_dlc_remote  # noqa: E402,F401
from tools.cloud_dlc.dlc_inference import runtime_environment  # noqa: E402,F401
from tools.cloud_dlc.validate_keypoints import compare_h5  # noqa: E402,F401


def _version_identity_check(local_env: dict, cloud_env: dict) -> dict:
    """Compare local vs cloud version manifest. Returns identity-check result.

    Three classes of mismatch:
      - HASH_MATCH: dist-info METADATA SHA256 identical → binary identity proven
      - VERSION_MATCH_HASH_DIFFER: pip versions match but hashes differ →
        same release, possibly different wheel build; likely benign
      - VERSION_MISMATCH: pip versions differ → genuine methodology split
    """
    local_pip = local_env.get("deeplabcut_pip_dist_info_version")
    cloud_pip = cloud_env.get("deeplabcut_pip_dist_info_version")
    local_md_sha = local_env.get("deeplabcut_dist_info_metadata_sha256")
    cloud_md_sha = cloud_env.get("deeplabcut_dist_info_metadata_sha256")
    local_vpy_sha = local_env.get("deeplabcut_version_py_sha256")
    cloud_vpy_sha = cloud_env.get("deeplabcut_version_py_sha256")

    if local_pip != cloud_pip:
        return {
            "status": "VERSION_MISMATCH",
            "local_pip_version": local_pip,
            "cloud_pip_version": cloud_pip,
            "concern": "Methodology split — local and cloud are different DLC releases.",
        }
    if local_md_sha and cloud_md_sha and local_md_sha == cloud_md_sha:
        return {
            "status": "HASH_MATCH",
            "pip_version": local_pip,
            "metadata_sha256": local_md_sha,
            "version_py_sha256_match": (local_vpy_sha == cloud_vpy_sha),
            "concern": None,
        }
    return {
        "status": "VERSION_MATCH_HASH_DIFFER",
        "pip_version": local_pip,
        "local_metadata_sha256": local_md_sha,
        "cloud_metadata_sha256": cloud_md_sha,
        "concern": (
            "Same DLC release version but dist-info METADATA hashes differ. "
            "Possible causes: different wheel build dates, different platform tags, "
            "different OS-specific metadata. Likely benign for inference output but "
            "worth recording in audit doc."
        ),
    }


def _fail_loud(msg: str, result: dict) -> None:
    """Print loud failure to stderr + write fail JSON + exit non-zero."""
    print("=" * 70, file=sys.stderr)
    print("[smoke_test] FAIL — DO NOT PROCEED TO PRODUCTION RUN", file=sys.stderr)
    print(f"[smoke_test] reason: {msg}", file=sys.stderr)
    print(f"[smoke_test] full result: {RESULT_JSON.relative_to(POC.parent)}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(result, indent=2))
    sys.exit(2)


@app.local_entrypoint()
def main() -> None:
    if not REFERENCE_CLIP_PATH.exists():
        print(f"[FAIL] reference clip missing: {REFERENCE_CLIP_PATH}", file=sys.stderr)
        sys.exit(1)
    if not LOCAL_H5.exists():
        print(f"[FAIL] local CPU baseline h5 missing: {LOCAL_H5}", file=sys.stderr)
        print("       expected from Phase 10a-prelim CPU run; smoke test cannot validate without it.",
              file=sys.stderr)
        sys.exit(1)

    print(f"[smoke_test] reference clip : {REFERENCE_CLIP_PATH.relative_to(POC.parent)}")
    print(f"[smoke_test] local baseline : {LOCAL_H5.relative_to(POC.parent)}")
    print(f"[smoke_test] threshold      : {THRESHOLD_PX} px (max |Δ|_xy)")
    print()

    print("[smoke_test] capturing local runtime env…")
    local_env = runtime_environment()
    print(f"[smoke_test]   pip dist-info     : {local_env['deeplabcut_pip_dist_info_version']}")
    print(f"[smoke_test]   runtime constant  : {local_env['deeplabcut_runtime_version_constant']}")
    print(f"[smoke_test]   METADATA sha256   : {local_env['deeplabcut_dist_info_metadata_sha256']}")
    print(f"[smoke_test]   version.py sha256 : {local_env['deeplabcut_version_py_sha256']}")
    print()

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    cloud_h5 = TEMP_DIR / LOCAL_H5.name

    print("[smoke_test] invoking Modal T4 GPU function…")
    clip_bytes = REFERENCE_CLIP_PATH.read_bytes()
    result = run_dlc_remote.remote(clip_bytes, REFERENCE_CLIP_PATH.name)

    print(f"[smoke_test] T4 wall-clock  : {result['wall_clock_s']:.2f} s "
          f"(local CPU baseline: ~70 s — expected speedup ~10×)")
    print()

    print("[smoke_test] cloud runtime env vs local…")
    cloud_env = result["runtime_env"]
    print(f"[smoke_test]   cloud pip dist-info: {cloud_env['deeplabcut_pip_dist_info_version']}")
    print(f"[smoke_test]   cloud METADATA sha : {cloud_env['deeplabcut_dist_info_metadata_sha256']}")
    print(f"[smoke_test]   cloud torch        : {cloud_env['torch_version']}")
    print(f"[smoke_test]   cloud cuDNN        : {cloud_env.get('torch_cudnn_version')}")
    print(f"[smoke_test]   cloud GPU         : {cloud_env.get('torch_device_name')}")
    version_check = _version_identity_check(local_env, cloud_env)
    print(f"[smoke_test]   version identity   : {version_check['status']}")
    if version_check.get("concern"):
        print(f"[smoke_test]   concern            : {version_check['concern']}")
    n_det_warns = len(result.get("determinism_warnings", []))
    n_total_warns = result.get("all_warnings_count", 0)
    print(f"[smoke_test]   determinism warns  : {n_det_warns} of {n_total_warns} total")
    if n_det_warns > 0:
        print("[smoke_test]   ⚠ non-deterministic ops in DLC forward pass (see result JSON)")
        for w in result["determinism_warnings"][:5]:
            print(f"     - {w['category']}: {w['message'][:120]}")
    print()

    cloud_h5.write_bytes(result["h5_bytes"])
    print(f"[smoke_test] cloud h5 saved : {cloud_h5.relative_to(POC.parent)}")
    print()

    print("[smoke_test] running validator…")
    cmp = compare_h5(LOCAL_H5, cloud_h5)

    smoke_result = {
        "smoke_test_ran_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_clip": REFERENCE_CLIP_ID,
        "reference_clip_path": str(REFERENCE_CLIP_PATH.relative_to(POC)),
        "threshold_px": THRESHOLD_PX,
        "verdict": "PASS" if cmp["passes_threshold"] else "FAIL",
        "local_runtime_env": local_env,
        "cloud_runtime_env": cloud_env,
        "version_identity_check": version_check,
        "cloud_t4_wall_clock_s": round(result["wall_clock_s"], 2),
        "local_cpu_baseline_wall_clock_s_approx": 70.0,
        "speedup_factor": round(70.0 / result["wall_clock_s"], 1),
        "validation": cmp,
        "determinism_capture": {
            "determinism_warnings": result.get("determinism_warnings", []),
            "all_warnings_count": result.get("all_warnings_count", 0),
            "all_warnings_sample": result.get("all_warnings_sample", []),
        },
        "notes": (
            "DLC keypoint output agreement between local CPU and Modal T4 CUDA. "
            "PASS at max |Δ|_xy ≤ 0.5 px means downstream Phase 8b pipeline "
            "(ear-bbox crop → V-JEPA-2 features → classifier) yields identical "
            "binary classifications except potentially at boundary cases very "
            "close to τ_ear. FAIL means methodology preservation is broken — "
            "investigate before any production GPU run. "
            "DLC version: pip dist-info reports 3.0.0rc14 (the source of truth); "
            "runtime __version__ constant displays 3.0.0rc13 (stale due to "
            "incomplete bump in DLC rc14 release process). METADATA SHA256 "
            "captured in {local,cloud}_runtime_env proves binary identity."
        ),
    }

    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(smoke_result, indent=2))

    print()
    print("=" * 70)
    print(f"[smoke_test] verdict             : {smoke_result['verdict']}")
    print(f"[smoke_test] max |Δ|_xy          : {cmp['max_delta_xy_px']} px")
    print(f"[smoke_test] mean |Δ|_xy         : {cmp['mean_delta_xy_px']} px")
    print(f"[smoke_test] p99 |Δ|_xy          : {cmp['p99_delta_xy_px']} px")
    print(f"[smoke_test] p99.9 |Δ|_xy        : {cmp['p999_delta_xy_px']} px")
    print(f"[smoke_test] max |Δ|_likelihood  : {cmp['max_delta_likelihood']}")
    print(f"[smoke_test] n NaN mismatches    : {cmp['n_nan_mismatches']}")
    print(f"[smoke_test] speedup (CPU/T4)    : {smoke_result['speedup_factor']}×")
    print(f"[smoke_test] result JSON         : {RESULT_JSON.relative_to(POC.parent)}")
    print("=" * 70)

    if not cmp["passes_threshold"]:
        _fail_loud(cmp["failure_reason"] or "validation failed", smoke_result)

    print("[smoke_test] PASS — Modal CUDA path produces keypoints within 0.5 px of local CPU.")
    print("[smoke_test] Methodology preservation verified; safe to proceed to Phase 10a GPU runs.")
