#!/usr/bin/env python3
"""Phase 7 — DLC SuperAnimal-Quadruped per-frame inference tool.

Productionizes Phase 0's notebook-only `deeplabcut.video_inference_superanimal()`
call as a reproducible, CLI-driven artifact in `tools/`.

Per locked Phase 7 Stage 1 pre-registration §1, §2:
- DLC version: 3.0.0rc13 (matches installed; re-running entire chain
  is required if rc14+ ever ships)
- superanimal_name = 'superanimal_quadruped'
- model_name = 'hrnet_w32'
- detector_name = 'fasterrcnn_resnet50_fpn_v2'
- video_adapt = False (matches Phase 0 'before_adapt' suffix)
- pseudo_threshold = 0.1 (matches Phase 0 notebook)

The 39-keypoint output schema (per individual): list of [x, y, conf]
arrays. Eye keypoints at indices 5 (right_eye) and 10 (left_eye).

Usage:
  python tools/dlc_inference.py --video <path> --out-dir <dir> [--device auto]

Outputs to `--out-dir`:
  <video_basename>_superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2_before_adapt.json
  <video_basename>_superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2.h5
  <video_basename>_superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2_labeled_before_adapt.mp4
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

# DLC import is heavy; defer to main()
POC_DIR = Path(__file__).resolve().parent.parent

# Locked DLC params (per Stage 1 pre-reg §1)
SUPERANIMAL_NAME = "superanimal_quadruped"
MODEL_NAME = "hrnet_w32"
DETECTOR_NAME = "fasterrcnn_resnet50_fpn_v2"
VIDEO_ADAPT = False
PSEUDO_THRESHOLD = 0.1


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True,
                    help="Input video path (mp4).")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output directory for DLC artifacts.")
    ap.add_argument("--device", default="auto",
                    help="Inference device (auto|cpu|cuda|mps); "
                         "default auto. Locked DLC default.")
    args = ap.parse_args()

    if not args.video.exists():
        print(f"[ERROR] video not found: {args.video}", file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[dlc] video: {args.video}")
    print(f"[dlc] sha256: {sha256(args.video)}")
    print(f"[dlc] out_dir: {args.out_dir}")
    print(f"[dlc] superanimal: {SUPERANIMAL_NAME}")
    print(f"[dlc] model: {MODEL_NAME}")
    print(f"[dlc] detector: {DETECTOR_NAME}")
    print(f"[dlc] video_adapt: {VIDEO_ADAPT}")
    print(f"[dlc] pseudo_threshold: {PSEUDO_THRESHOLD}")
    print(f"[dlc] device: {args.device}")
    print(flush=True)

    t0 = time.time()
    print("[dlc] loading deeplabcut...", flush=True)
    import deeplabcut
    print(f"[dlc] DLC version: {deeplabcut.__version__}", flush=True)
    if deeplabcut.__version__ != "3.0.0rc13":
        print(f"[WARN] DLC version mismatch: got "
              f"{deeplabcut.__version__}, expected 3.0.0rc13. "
              f"Per Stage 1 pre-reg §1, version pin reproduction-event "
              f"clause may apply.", file=sys.stderr, flush=True)

    print("[dlc] starting video_inference_superanimal...", flush=True)
    deeplabcut.video_inference_superanimal(
        videos=[str(args.video)],
        superanimal_name=SUPERANIMAL_NAME,
        model_name=MODEL_NAME,
        detector_name=DETECTOR_NAME,
        video_adapt=VIDEO_ADAPT,
        dest_folder=str(args.out_dir),
        pseudo_threshold=PSEUDO_THRESHOLD,
    )
    elapsed = time.time() - t0
    print(f"[dlc] inference complete in {elapsed:.1f}s", flush=True)

    # Sanity check outputs
    expected_json_pattern = (
        f"{args.video.stem}_{SUPERANIMAL_NAME}_{MODEL_NAME}_"
        f"{DETECTOR_NAME}_before_adapt.json"
    )
    json_out = args.out_dir / expected_json_pattern
    if not json_out.exists():
        # DLC may write json with slightly different naming; search
        candidates = list(args.out_dir.glob(
            f"{args.video.stem}_*_before_adapt.json"
        ))
        if not candidates:
            print(f"[ERROR] no JSON output found in {args.out_dir}",
                  file=sys.stderr)
            return 1
        json_out = candidates[0]
        print(f"[dlc] resolved JSON output: {json_out.name}", flush=True)

    print(f"[dlc] JSON output sha256: {sha256(json_out)}")
    print("[dlc] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
