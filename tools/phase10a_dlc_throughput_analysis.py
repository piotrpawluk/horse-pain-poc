#!/usr/bin/env python3
"""Phase 10a DLC throughput analysis — post-hoc reconstruction from h5 mtimes.

Reads `outputs/phase10a_prelim_dlc_outputs/*.h5` file mtimes (writes happen at
inference completion, so consecutive mtime gaps = per-clip wall-clock).
Joins with `data/prudnik/inventory.csv` for frame counts to derive per-frame
rates. Outputs JSON for inclusion in audit_extras.

Also integrates the MPS probe finding (probe wall-clock + stage decomposition)
to distinguish fixable engineering finding from methodological finding.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev

POC = Path(__file__).resolve().parent.parent
DLC_DIR = POC / "outputs" / "phase10a_prelim_dlc_outputs"
INVENTORY_CSV = POC / "data" / "prudnik" / "inventory.csv"
OUT_JSON = POC / "outputs" / "phase10a_dlc_throughput.json"

H5_SUFFIX = "_superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2.h5"


def load_inventory() -> dict[str, dict]:
    inv: dict[str, dict] = {}
    with INVENTORY_CSV.open() as f:
        header = f.readline().strip().split(",")
        for line in f:
            row = dict(zip(header, line.strip().split(",")))
            inv[row["clip_id"]] = {
                "duration_s": float(row["duration_s"]),
                "fps": float(row["fps"]),
                "resolution": row["resolution"],
                "portrait": row["portrait"] == "TRUE",
                "n_frames": int(round(float(row["duration_s"]) * float(row["fps"]))),
            }
    return inv


def main() -> int:
    if not DLC_DIR.exists():
        print(f"[FAIL] DLC dir not found: {DLC_DIR}", file=sys.stderr)
        return 1

    h5_files = sorted(DLC_DIR.glob(f"*{H5_SUFFIX}"), key=lambda p: p.stat().st_mtime)
    if not h5_files:
        print(f"[FAIL] no h5 files in {DLC_DIR}", file=sys.stderr)
        return 1

    inv = load_inventory()
    print(f"[throughput] {len(h5_files)} h5 files; {len(inv)} inventory rows")

    per_clip: list[dict] = []
    prev_mtime: float | None = None
    for h5 in h5_files:
        clip_id = h5.name.removesuffix(H5_SUFFIX)
        mtime = h5.stat().st_mtime
        inv_row = inv.get(clip_id, {})
        n_frames = inv_row.get("n_frames")
        if prev_mtime is None:
            wall_s = None
        else:
            wall_s = mtime - prev_mtime
        s_per_frame = (wall_s / n_frames) if (wall_s is not None and n_frames) else None
        per_clip.append({
            "clip_id": clip_id,
            "h5_completed_at": datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
            "duration_s": inv_row.get("duration_s"),
            "n_frames": n_frames,
            "resolution": inv_row.get("resolution"),
            "portrait": inv_row.get("portrait"),
            "wall_clock_s": round(wall_s, 2) if wall_s is not None else None,
            "s_per_frame": round(s_per_frame, 3) if s_per_frame is not None else None,
        })
        prev_mtime = mtime

    # First clip has no wall-clock (no prior anchor). Skip from distribution.
    timed = [c for c in per_clip if c["s_per_frame"] is not None]
    rates = [c["s_per_frame"] for c in timed]

    # Slowest clip
    slowest = max(timed, key=lambda c: c["s_per_frame"]) if timed else None

    summary = {
        "device_routing": {
            "dlc_version": "3.0.0rc13",
            "torch_version": "2.11.0",
            "device_arg_used": "auto (default)",
            "actual_routing": "CPU (Faster R-CNN detector forces CPU fallback regardless of device kwarg)",
            "mps_available_on_hardware": True,
            "mps_smoke_test_pass": True,  # from probe
            "hardware": "MacBook Pro 14\" 2023, M2 Max, 96 GB unified memory",
        },
        "mps_probe": {
            "ran_at": "2026-05-13",
            "clip": "IMG_1050",
            "clip_resolution": "1080x1920",
            "clip_frames": 57,
            "wall_clock_s_with_mps_kwarg": 118.36,
            "wall_clock_s_cpu_baseline": 70.0,
            "cpu_pct_during_probe": 321,
            "stage_decomposition": {
                "detector_faster_rcnn_s_per_frame": 1.68,
                "detector_interpretation": "Silent fallback to CPU — Faster R-CNN ops not MPS-compatible",
                "pose_hrnet_w32_s_per_frame": 0.17,
                "pose_interpretation": "MPS engaged — pose model runs ~10× faster than detector",
            },
            "verdict": "METHODOLOGICAL_FINDING",
            "rationale": (
                "Detector dominates wall-clock (~80%) and silently falls back to CPU. "
                "MPS-engaged pose stage shows only ~6× speedup over its CPU equivalent, "
                "insufficient to overcome detector CPU bottleneck + MPS↔CPU shuttle "
                "overhead + first-call MPS init. device='auto' (pure CPU) is already "
                "the optimal routing for this DLC version on this hardware for these "
                "clips. NO ENGINEERING FIX AVAILABLE without swapping detector."
            ),
            "phase11_optimization_candidate": (
                "Replace Faster R-CNN with MPS-compatible bbox detector "
                "(e.g., Yolov8 from Ultralytics) — would unlock pose-stage speedup "
                "(~6×) and likely give 4-8× total throughput on portrait clips."
            ),
        },
        "cpu_run_throughput": {
            "n_clips_timed": len(timed),
            "n_clips_total_completed": len(per_clip),
            "s_per_frame_mean": round(mean(rates), 3) if rates else None,
            "s_per_frame_median": round(median(rates), 3) if rates else None,
            "s_per_frame_stdev": round(stdev(rates), 3) if len(rates) >= 2 else None,
            "s_per_frame_min": round(min(rates), 3) if rates else None,
            "s_per_frame_max": round(max(rates), 3) if rates else None,
            "slowest_clip": {
                "clip_id": slowest["clip_id"],
                "s_per_frame": slowest["s_per_frame"],
                "wall_clock_s": slowest["wall_clock_s"],
                "n_frames": slowest["n_frames"],
                "resolution": slowest["resolution"],
            } if slowest else None,
        },
        "phase8b_reference": {
            "s_per_clip_landscape": 31.1,
            "note": (
                "Phase 8b RME (RMA) was landscape 1920×1080 at ~31.1 s/clip. "
                "Prudnik portrait at ~1.18 s/frame × ~300 frames ≈ 350 s/clip ≈ 10× more expensive. "
                "ATTRIBUTION (load-bearing for audit interpretation): the 10× gap is "
                "DETECTOR-ARCHITECTURE-SPECIFIC, NOT ORIENTATION-FUNDAMENTAL. "
                "Faster R-CNN silently CPU-falls-back on these clips regardless of "
                "device kwarg, and CPU execution scales near-linearly with pixel count. "
                "The pose stage (hrnet_w32) runs cleanly on MPS (~0.17 s/frame in probe) "
                "and that performance is preserved across orientations. A reader who "
                "infers 'portrait orientation is intrinsically 10× more expensive' has "
                "misread this finding — the correct reading is 'Faster R-CNN's CPU-bound "
                "execution on portrait dimensions costs 10× more than landscape; "
                "pose-stage MPS performance is preserved across orientations'. "
                "This distinction matters for the orientation hypothesis (Limitation 6) "
                "in the audit doc: compute cost does NOT validate or invalidate the "
                "orientation-induced distribution-shift hypothesis for the classifier. "
                "Those are orthogonal concerns."
            ),
        },
        "implication_for_phase10a_full": {
            "compute_budget_per_clip_s": round(mean(rates) * 60, 0) if rates else None,
            "n_remaining_clips_in_prelim": "TBD when run completes",
            "fix_available": False,
            "honest_compute_plan": (
                "Phase 10a-full (~283 clips equivalent at portrait scale) would take "
                "roughly: 283 × mean_clip_wall_s ≈ many hours. Plan as overnight job. "
                "No software fix available within current DLC + detector pipeline."
            ),
        },
        "per_clip_detail": per_clip,
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"[throughput] wrote {OUT_JSON.relative_to(POC.parent)}")

    print()
    print("=" * 70)
    print("DLC THROUGHPUT — Prudnik portrait clips (CPU run)")
    print("=" * 70)
    print(f"  N clips completed     : {len(per_clip)}")
    print(f"  N clips timed         : {len(timed)} (first clip has no prior anchor)")
    if rates:
        print(f"  s/frame mean          : {mean(rates):.2f}")
        print(f"  s/frame median        : {median(rates):.2f}")
        print(f"  s/frame range         : [{min(rates):.2f}, {max(rates):.2f}]")
        print(f"  Slowest clip          : {slowest['clip_id']} "
              f"({slowest['s_per_frame']:.2f} s/frame, "
              f"{slowest['n_frames']} frames, {slowest['resolution']})")
    print()
    print("MPS PROBE VERDICT: METHODOLOGICAL_FINDING")
    print("  Faster R-CNN detector silently falls back to CPU even with device='mps'.")
    print("  Pose stage IS faster on MPS (0.17 vs 1.0+ s/frame on detector) but cannot")
    print("  overcome detector bottleneck. device='auto' = optimal on this hardware.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
