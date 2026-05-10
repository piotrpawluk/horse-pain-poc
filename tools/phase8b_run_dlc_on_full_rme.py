#!/usr/bin/env python3
"""Phase 8b Step 4 — DLC SuperAnimal-Quadruped on full 283-clip RME.

Runs DLC inference on every clip in train.csv + val.csv + test.csv
(285 listed; expected 283 after dedup of any duplicates and missing-
file filter). Produces per-clip per-frame keypoints with the same
schema as `outputs/phase7_rme_dlc_keypoints.json`.

Locked DLC params (carry forward from Phase 7 §1):
  superanimal_name = 'superanimal_quadruped'
  model_name = 'hrnet_w32'
  detector_name = 'fasterrcnn_resnet50_fpn_v2'
  video_adapt = False
  pseudo_threshold = 0.1
  create_labeled_video = False  (Phase 7 cosmetic deviation lock)

Output: outputs/phase8b_rme_dlc_keypoints.json
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

POC_DIR = Path(__file__).resolve().parent.parent
RME_DATA = POC_DIR / "vendor" / "ReadMyEars_Dataset" / "data"
RME_VIDEOS = RME_DATA / "videos"
TMP_DIR = POC_DIR / "outputs" / ".tmp_phase8b_dlc"
OUT_PATH = POC_DIR / "outputs" / "phase8b_rme_dlc_keypoints.json"

SUPERANIMAL_NAME = "superanimal_quadruped"
MODEL_NAME = "hrnet_w32"
DETECTOR_NAME = "fasterrcnn_resnet50_fpn_v2"
VIDEO_ADAPT = False
PSEUDO_THRESHOLD = 0.1
CREATE_LABELED_VIDEO = False


def load_all_rme_clips():
    """Union of train/val/test split CSVs; clip basename → label dict.
    Filters to clips that exist on disk."""
    dfs = []
    for split in ["train", "val", "test"]:
        path = RME_DATA / f"{split}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["split"] = split
            dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    full["basename"] = full["video"].apply(lambda v: Path(v).name)
    full = full.drop_duplicates(subset=["basename"])
    full["abs_path"] = full["video"].apply(lambda v: str(RME_DATA / v))
    full = full[full["abs_path"].apply(Path).apply(lambda p: p.exists())]
    return full


def primary_individual_index(frame_entry):
    bbox_scores = frame_entry.get("bbox_scores", [])
    if not bbox_scores:
        return None
    valid = [(i, s) for i, s in enumerate(bbox_scores) if s >= 0]
    if not valid:
        return None
    best_i, _ = max(valid, key=lambda x: x[1])
    return best_i


def main() -> int:
    df = load_all_rme_clips()
    print(f"[phase8b dlc] {len(df)} RME clips loaded; "
          f"{(df['label'] == 'action').sum()} action, "
          f"{(df['label'] == 'background').sum()} background", flush=True)

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True)

    print("[phase8b dlc] loading deeplabcut...", flush=True)
    t0 = time.time()
    import deeplabcut
    dlc_version = deeplabcut.__version__
    print(f"[phase8b dlc] DLC version: {dlc_version}", flush=True)

    video_paths = df["abs_path"].tolist()
    print(f"[phase8b dlc] starting batch inference on {len(video_paths)} "
          f"clips... ETA ~30 sec/clip = "
          f"{len(video_paths) * 30 / 60:.0f} min", flush=True)

    deeplabcut.video_inference_superanimal(
        videos=video_paths,
        superanimal_name=SUPERANIMAL_NAME,
        model_name=MODEL_NAME,
        detector_name=DETECTOR_NAME,
        video_adapt=VIDEO_ADAPT,
        dest_folder=str(TMP_DIR),
        pseudo_threshold=PSEUDO_THRESHOLD,
        create_labeled_video=CREATE_LABELED_VIDEO,
    )
    elapsed = time.time() - t0
    print(f"[phase8b dlc] inference complete in {elapsed:.1f}s "
          f"({elapsed/len(video_paths):.1f}s/clip avg)", flush=True)

    # Aggregate per-clip
    per_clip = {}
    n_ok = 0
    n_fail = 0
    for _, row in df.iterrows():
        clip_basename = row["basename"]
        clip_stem = Path(clip_basename).stem
        json_pattern = (
            f"{clip_stem}_{SUPERANIMAL_NAME}_{MODEL_NAME}_"
            f"{DETECTOR_NAME}_before_adapt.json"
        )
        json_path = TMP_DIR / json_pattern
        if not json_path.exists():
            per_clip[clip_basename] = {"status": "fail",
                                        "error": "no_json"}
            n_fail += 1
            continue

        frames = json.loads(json_path.read_text())
        n_frames = len(frames)
        per_frame_kps = []
        per_frame_bbox = []
        per_frame_pi = []
        for f in frames:
            pi = primary_individual_index(f)
            per_frame_pi.append(pi)
            if pi is None:
                per_frame_kps.append(None)
                per_frame_bbox.append(None)
                continue
            kps = f["bodyparts"][pi]
            per_frame_kps.append(kps)
            per_frame_bbox.append(f["bboxes"][pi])

        per_clip[clip_basename] = {
            "status": "ok",
            "n_frames": n_frames,
            "label": row["label"],
            "split": row["split"],
            "primary_individual_index": per_frame_pi,
            "keypoints": per_frame_kps,
            "bbox": per_frame_bbox,
        }
        n_ok += 1

    summary = {
        "tool": "tools/phase8b_run_dlc_on_full_rme.py",
        "stage1_decision": "§Sequencing Step 4",
        "dlc_version": dlc_version,
        "params": {
            "superanimal_name": SUPERANIMAL_NAME,
            "model_name": MODEL_NAME,
            "detector_name": DETECTOR_NAME,
            "video_adapt": VIDEO_ADAPT,
            "pseudo_threshold": PSEUDO_THRESHOLD,
            "create_labeled_video": CREATE_LABELED_VIDEO,
        },
        "n_clips_total": len(df),
        "n_clips_ok": n_ok,
        "n_clips_failed": n_fail,
        "per_clip": per_clip,
    }

    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print()
    print(f"=== Step 4 complete: {n_ok}/{len(df)} clips, "
          f"{n_fail} failed ===")
    print(f"Wrote: {OUT_PATH.relative_to(POC_DIR)}")
    print(f"Tmp DLC outputs in {TMP_DIR.relative_to(POC_DIR)} — "
          f"keep until Step 5 crops are written, then delete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
