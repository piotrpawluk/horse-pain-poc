#!/usr/bin/env python3
"""Phase 10a MPS probe — does DLC SuperAnimal engage MPS on M2 Max?

Runs DLC SuperAnimal-Quadruped with device='mps' on IMG_1050 (0.95s, 57 frames).
CPU baseline for same clip ≈ 70s (1.23 s/frame, derived from h5 mtime gap in
running Phase 10a-prelim CPU process).

Three possible outcomes:
  - MPS engages → wall-clock 5-15s, CPU% stays well below 600%
  - MPS silently falls back to CPU → wall-clock ~70s, CPU% spikes to 400-600%
  - MPS errors (unsupported op) → exception thrown
"""

from __future__ import annotations

import resource
import shutil
import sys
import time
from pathlib import Path

POC = Path(__file__).resolve().parent.parent
SRC_VIDEO = POC / "data" / "prudnik" / "IMG_1050.mov"
PROBE_DIR = POC / "temp" / "phase10a_mps_probe"


def get_cpu_times() -> tuple[float, float]:
    """Return (user, sys) CPU time consumed so far by this process."""
    u = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return u.ru_utime + c.ru_utime, u.ru_stime + c.ru_stime


def main() -> int:
    if not SRC_VIDEO.exists():
        print(f"[FAIL] source not found: {SRC_VIDEO}", file=sys.stderr)
        return 1

    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    probe_video = PROBE_DIR / SRC_VIDEO.name
    if probe_video.exists():
        probe_video.unlink()
    shutil.copy2(SRC_VIDEO, probe_video)

    print(f"[probe] clip          : {SRC_VIDEO.name}")
    print("[probe] duration      : 0.950 s @ 60 fps → 57 frames")
    print("[probe] CPU baseline  : ~70 s (1.23 s/frame from running Phase 10a-prelim)")
    print("[probe] target device : mps")
    print(f"[probe] temp dir      : {PROBE_DIR.relative_to(POC.parent)}")
    print()

    print("[probe] importing torch + deeplabcut…")
    import torch
    import deeplabcut

    print(f"[probe] torch         : {torch.__version__}")
    print(f"[probe] mps_available : {torch.backends.mps.is_available()}")
    print(f"[probe] mps_built     : {torch.backends.mps.is_built()}")
    print(f"[probe] dlc           : {deeplabcut.__version__}")
    print()

    # Quick MPS smoke test BEFORE invoking DLC — proves MPS itself responsive
    print("[probe] MPS smoke test: torch tensor on mps…")
    smoke_start = time.perf_counter()
    try:
        x = torch.randn(2048, 2048, device="mps")
        y = (x @ x.T).sum().item()
        smoke_elapsed = time.perf_counter() - smoke_start
        print(f"[probe]   PASS — 2048×2048 matmul on MPS: {smoke_elapsed*1000:.0f} ms, result={y:.2f}")
    except Exception as e:
        print(f"[probe]   FAIL — MPS smoke test errored: {type(e).__name__}: {e}")
        return 2
    print()

    # Probe DLC with device='mps'
    print("[probe] invoking DLC.video_inference_superanimal(device='mps')…")
    cpu_user_0, cpu_sys_0 = get_cpu_times()
    wall_0 = time.perf_counter()
    error = None
    try:
        deeplabcut.video_inference_superanimal(
            videos=[str(probe_video)],
            superanimal_name="superanimal_quadruped",
            model_name="hrnet_w32",
            detector_name="fasterrcnn_resnet50_fpn_v2",
            video_adapt=False,
            pseudo_threshold=0.1,
            dest_folder=str(PROBE_DIR),
            create_labeled_video=False,
            plot_bboxes=False,
            device="mps",
        )
    except Exception as e:
        error = e

    wall_elapsed = time.perf_counter() - wall_0
    cpu_user_1, cpu_sys_1 = get_cpu_times()
    cpu_user_delta = cpu_user_1 - cpu_user_0
    cpu_sys_delta = cpu_sys_1 - cpu_sys_0
    cpu_total_delta = cpu_user_delta + cpu_sys_delta
    cpu_pct = (cpu_total_delta / wall_elapsed) * 100 if wall_elapsed > 0 else 0.0

    print()
    print("=" * 70)
    print(f"[probe] wall-clock      : {wall_elapsed:.2f} s")
    print(f"[probe] CPU user time   : {cpu_user_delta:.2f} s")
    print(f"[probe] CPU sys time    : {cpu_sys_delta:.2f} s")
    print(f"[probe] CPU total time  : {cpu_total_delta:.2f} s")
    print(f"[probe] CPU% (vs wall)  : {cpu_pct:.0f}%")
    print(f"[probe] per-frame rate  : {wall_elapsed/57:.3f} s/frame")
    print("=" * 70)

    if error is not None:
        print(f"[VERDICT] MPS_ERROR — DLC raised {type(error).__name__}: {error}")
        print("[VERDICT] → CPU run continues; MPS path unusable for this DLC version.")
        return 3

    # Interpret
    cpu_baseline_per_frame = 1.23
    mps_per_frame = wall_elapsed / 57
    speedup = cpu_baseline_per_frame / mps_per_frame

    if cpu_pct > 400 and mps_per_frame > 0.8 * cpu_baseline_per_frame:
        verdict = "SILENT_FALLBACK_TO_CPU"
        recommendation = (
            "MPS device flag did NOT engage — CPU% > 400% and per-frame rate matches CPU baseline. "
            "Likely unsupported torch ops in SuperAnimal-Quadruped + hrnet_w32 + Faster R-CNN. "
            "DO NOT kill-restart with device='mps' — same throughput. Let CPU run finish."
        )
    elif mps_per_frame < 0.3 * cpu_baseline_per_frame:
        verdict = "MPS_ENGAGED_FAST"
        recommendation = (
            f"MPS engaged cleanly — {speedup:.1f}× speedup. "
            "Recommend (B): kill CPU run, restart with device='mps', cache preserves done clips. "
            "Apply 5-min CPU%-monitoring gate at restart to confirm no silent fallback."
        )
    elif mps_per_frame < 0.7 * cpu_baseline_per_frame:
        verdict = "MPS_ENGAGED_PARTIAL"
        recommendation = (
            f"MPS partially engaged — {speedup:.1f}× speedup. "
            "Marginal benefit; cost-of-switch (~20 min monitoring overhead) ≈ saved compute. "
            "Recommend let CPU run finish unless overnight budget matters."
        )
    else:
        verdict = "AMBIGUOUS"
        recommendation = (
            f"Inconclusive — per-frame {mps_per_frame:.2f} s vs CPU {cpu_baseline_per_frame:.2f} s, "
            f"CPU% {cpu_pct:.0f}%. Manually inspect log + decide."
        )

    print(f"[VERDICT] {verdict}")
    print(f"[VERDICT] speedup_factor: {speedup:.2f}×")
    print(f"[VERDICT] {recommendation}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
