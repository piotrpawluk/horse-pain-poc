#!/usr/bin/env python3
"""Phase 8b Step 2 — Stage 1.5 ear keypoint quality inspection.

Per locked Phase 8b Stage 1 pre-reg Decision 3:

  Run DLC SuperAnimal-Quadruped on a stratified sample of ~10 RME
  clips (5 action + 5 background; sources spread across S1, S5, S8,
  S10, S12). For each clip, inspect mean confidence of the 4 ear
  keypoints (right_earbase, right_earend, left_earbase, left_earend)
  across frames. Compute fraction of frames per clip with ≥3 of 4 ear
  keypoints confident (≥0.5).

Sample: deterministic — alphabetical-first action + alphabetical-first
background per source from {S1, S5, S8, S10, S12} = 10 clips. Locked
here (not adjustable post-observation).

Locked gate: ≥80% of frames in ≥80% of sample clips meet the
≥3-of-4 threshold. Pre-reg also has reference-floor verification
(33/34 = 97.1% on Phase 5 clips).

Locked ear keypoint confidence threshold: 0.5 (matches Phase 7
Stage 2 §1 hard-lock).

Output: outputs/phase8b_stage15_inspection.json
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

POC_DIR = Path(__file__).resolve().parent.parent
RME_VIDEOS = POC_DIR / "vendor" / "ReadMyEars_Dataset" / "data" / "videos"
TMP_DIR = POC_DIR / "outputs" / ".tmp_phase8b_stage15"
OUT_PATH = POC_DIR / "outputs" / "phase8b_stage15_inspection.json"

# Locked DLC params per Phase 7 §1 (carry forward)
SUPERANIMAL_NAME = "superanimal_quadruped"
MODEL_NAME = "hrnet_w32"
DETECTOR_NAME = "fasterrcnn_resnet50_fpn_v2"
VIDEO_ADAPT = False
PSEUDO_THRESHOLD = 0.1
CREATE_LABELED_VIDEO = False  # Phase 7 §1 cosmetic deviation lock

# Locked Phase 8b params per pre-reg Decision 3
EAR_KP_INDICES = [6, 7, 11, 12]  # right_earbase, right_earend, left_earbase, left_earend
EAR_KP_NAMES = ["right_earbase", "right_earend",
                "left_earbase", "left_earend"]
CONF_THRESHOLD = 0.5
MIN_KP_CONFIDENT = 3  # of 4
PER_CLIP_FRAME_FRACTION_GATE = 0.80
SAMPLE_CLIPS_FRACTION_GATE = 0.80

# Locked stratified sample (deterministic alphabetical-first
# action + bg per source from S1, S5, S8, S10, S12)
LOCKED_SAMPLE = [
    "action_S1.mp4_0_.mp4",
    "background_S1.mp4_0_.mp4",
    "action_S5.mp4_0_.mp4",
    "background_S5.mp4_0_.mp4",
    "action_S8.mp4_0_.mp4",
    "background_S8.mp4_0_.mp4",
    "action_S10.mp4_0_.mp4",
    "background_S10.mp4_0_.mp4",
    "action_S12.mp4_0_.mp4",
    "background_S12.mp4_0_.mp4",
]


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
    # Verify locked sample clips exist
    sample_paths = []
    missing = []
    for clip in LOCKED_SAMPLE:
        p = RME_VIDEOS / clip
        if p.exists():
            sample_paths.append(p)
        else:
            missing.append(clip)
    if missing:
        print(f"[ERROR] missing clips: {missing}", file=sys.stderr)
        return 1
    print(f"[stage1.5] {len(sample_paths)} locked sample clips found",
          flush=True)

    if TMP_DIR.exists():
        import shutil
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True)

    print("[stage1.5] loading deeplabcut + running batch inference...",
          flush=True)
    t0 = time.time()
    import deeplabcut
    dlc_version = deeplabcut.__version__
    print(f"[stage1.5] DLC version: {dlc_version}", flush=True)

    deeplabcut.video_inference_superanimal(
        videos=[str(p) for p in sample_paths],
        superanimal_name=SUPERANIMAL_NAME,
        model_name=MODEL_NAME,
        detector_name=DETECTOR_NAME,
        video_adapt=VIDEO_ADAPT,
        dest_folder=str(TMP_DIR),
        pseudo_threshold=PSEUDO_THRESHOLD,
        create_labeled_video=CREATE_LABELED_VIDEO,
    )
    elapsed = time.time() - t0
    print(f"[stage1.5] inference complete in {elapsed:.1f}s "
          f"({elapsed/len(sample_paths):.1f}s/clip avg)", flush=True)

    # Aggregate per-clip
    per_clip_results = {}
    n_clips_meet = 0
    pooled_kp_confs = {idx: [] for idx in EAR_KP_INDICES}
    for clip in LOCKED_SAMPLE:
        clip_stem = Path(clip).stem
        json_pattern = (
            f"{clip_stem}_{SUPERANIMAL_NAME}_{MODEL_NAME}_"
            f"{DETECTOR_NAME}_before_adapt.json"
        )
        json_path = TMP_DIR / json_pattern
        if not json_path.exists():
            print(f"[WARN] no JSON for {clip}", file=sys.stderr)
            per_clip_results[clip] = {"status": "fail",
                                       "error": "no_json"}
            continue

        frames = json.loads(json_path.read_text())
        n_frames = len(frames)
        n_meeting = 0
        per_kp_confs = {idx: [] for idx in EAR_KP_INDICES}
        for f in frames:
            pi = primary_individual_index(f)
            if pi is None:
                continue
            kps = f["bodyparts"][pi]
            n_confident = sum(1 for idx in EAR_KP_INDICES
                              if kps[idx][2] >= CONF_THRESHOLD)
            if n_confident >= MIN_KP_CONFIDENT:
                n_meeting += 1
            for idx in EAR_KP_INDICES:
                per_kp_confs[idx].append(kps[idx][2])
                pooled_kp_confs[idx].append(kps[idx][2])

        pct_meeting = n_meeting / n_frames if n_frames > 0 else 0
        meets_gate = pct_meeting >= PER_CLIP_FRAME_FRACTION_GATE
        if meets_gate:
            n_clips_meet += 1

        kp_summary = {}
        for idx, name in zip(EAR_KP_INDICES, EAR_KP_NAMES):
            cs = per_kp_confs[idx]
            if cs:
                kp_summary[name] = {
                    "median": statistics.median(cs),
                    "mean": statistics.fmean(cs),
                    "min": min(cs),
                    "n_below_0_5": sum(1 for c in cs if c < 0.5),
                    "n_total": len(cs),
                }
        per_clip_results[clip] = {
            "status": "ok",
            "n_frames": n_frames,
            "n_meeting_3_of_4_at_0_5": n_meeting,
            "pct_meeting": pct_meeting,
            "meets_gate_80pct": meets_gate,
            "per_kp": kp_summary,
        }

    # Locked gate: ≥80% of clips meet the ≥80% per-clip threshold
    n_eligible = sum(1 for r in per_clip_results.values()
                     if r.get("status") == "ok")
    pct_clips_meeting = (n_clips_meet / n_eligible
                         if n_eligible > 0 else 0)
    gate_pass = pct_clips_meeting >= SAMPLE_CLIPS_FRACTION_GATE

    pooled_summary = {}
    for idx, name in zip(EAR_KP_INDICES, EAR_KP_NAMES):
        cs = pooled_kp_confs[idx]
        if cs:
            pooled_summary[name] = {
                "median": statistics.median(cs),
                "mean": statistics.fmean(cs),
                "p25": sorted(cs)[len(cs) // 4],
                "p75": sorted(cs)[3 * len(cs) // 4],
                "min": min(cs),
                "n_total": len(cs),
                "n_below_0_5": sum(1 for c in cs if c < 0.5),
                "frac_below_0_5": sum(1 for c in cs if c < 0.5) / len(cs),
            }

    if gate_pass:
        verdict = "PASS"
        next_action = ("Phase 8b proceeds to Step 3 (build crop "
                       "pipeline). Decision 1 geometry (both-ears bbox "
                       "with ≥3-of-4 confidence gate) confirmed "
                       "appropriate for RME ear keypoint quality.")
    else:
        verdict = "FAIL_GATE"
        next_action = (f"Sample shows ear keypoint reliability below "
                       f"locked gate ({pct_clips_meeting:.1%} < 80%). "
                       f"Per anti-pattern locks, do NOT lower the "
                       f"≥3-of-4 threshold mid-phase. User "
                       f"investigation required: revisit Decision 1 "
                       f"geometry or Decision 3 threshold before "
                       f"proceeding to Step 3.")

    summary = {
        "tool": "tools/phase8b_stage15_ear_inspection.py",
        "stage1_decision": "§Decision 3 Stage 1.5 ear keypoint quality",
        "dlc_version": dlc_version,
        "params": {
            "superanimal_name": SUPERANIMAL_NAME,
            "model_name": MODEL_NAME,
            "detector_name": DETECTOR_NAME,
            "video_adapt": VIDEO_ADAPT,
            "pseudo_threshold": PSEUDO_THRESHOLD,
            "create_labeled_video": CREATE_LABELED_VIDEO,
        },
        "ear_keypoint_indices": dict(zip(EAR_KP_NAMES, EAR_KP_INDICES)),
        "conf_threshold": CONF_THRESHOLD,
        "min_kp_confident": MIN_KP_CONFIDENT,
        "per_clip_frame_fraction_gate": PER_CLIP_FRAME_FRACTION_GATE,
        "sample_clips_fraction_gate": SAMPLE_CLIPS_FRACTION_GATE,
        "locked_sample": LOCKED_SAMPLE,
        "n_clips_in_sample": len(LOCKED_SAMPLE),
        "n_clips_eligible_for_analysis": n_eligible,
        "n_clips_meeting_per_clip_gate": n_clips_meet,
        "fraction_clips_meeting_per_clip_gate": pct_clips_meeting,
        "verdict": verdict,
        "next_action": next_action,
        "pooled_kp_confidence_summary": pooled_summary,
        "per_clip_results": per_clip_results,
    }

    OUT_PATH.write_text(json.dumps(summary, indent=2))

    print()
    print("=== Stage 1.5 inspection complete ===")
    print()
    print("Locked sample (10 clips, alphabetical-first per source × label):")
    for c, r in per_clip_results.items():
        if r.get("status") == "ok":
            print(f"  {c:35s}  {r['pct_meeting']:.1%} meeting "
                  f"(n_meeting/n_frames = {r['n_meeting_3_of_4_at_0_5']}/{r['n_frames']})  "
                  f"{'✓' if r['meets_gate_80pct'] else '✗'}")
        else:
            print(f"  {c:35s}  FAILED: {r.get('error')}")
    print()
    print(f"Pooled ear keypoint confidence (n={len(pooled_kp_confs[EAR_KP_INDICES[0]])} frames):")
    for name in EAR_KP_NAMES:
        s = pooled_summary[name]
        print(f"  {name:18s}: median={s['median']:.3f} mean={s['mean']:.3f} "
              f"p25={s['p25']:.3f} below-0.5={s['frac_below_0_5']:.1%}")
    print()
    print("Locked gate: ≥80% of clips meet the ≥80% per-clip frame threshold")
    print(f"Result: {n_clips_meet}/{n_eligible} = {pct_clips_meeting:.1%}")
    print(f"VERDICT: {verdict}")
    print(f"NEXT: {next_action}")
    print()
    print(f"Wrote: {OUT_PATH.relative_to(POC_DIR)}")

    if verdict == "FAIL_GATE":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
