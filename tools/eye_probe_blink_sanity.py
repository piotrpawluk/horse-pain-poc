#!/usr/bin/env python3
"""6-clip blink sanity test for MiniCPM-V 4.5.

User's hand-labeled set:
  3 BLINK:    action_S2.mp4_2_, action_S2.mp4_8_, action_S4.mp4_4_
  3 NO-BLINK: action_S4.mp4_2_, action_S2.mp4_7_, action_S4.mp4_10_

Pre-registered gate (outputs/eye_probe_preregistration_minicpm.md):
  5/6 correct -> proceed to 36-clip; <5/6 -> stop.

Frame policy: native FPS (all native frames). Removes frame-count as a
confound for the architecturally-right model on the pre-flight test.

Each clip is graded against expected_diagnostic_minicpm_blink.json — the
predictions were committed BEFORE the run.

Output: outputs/eye_probe_minicpm_blink_sanity.jsonl + printed summary.
Exit code 0 = pass gate; 1 = fail gate / structural failure.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

POC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_DIR / "tools"))

from minicpm_audit import MiniCPMRunner, parse_minicpm_json

CLIPS_DIR = POC_DIR / "vendor" / "ReadMyEars_Dataset" / "data" / "videos"
EXPECTED = POC_DIR / "outputs" / "expected_diagnostic_minicpm_blink.json"
OUT = POC_DIR / "outputs" / "eye_probe_minicpm_blink_sanity.jsonl"

CLIPS = [
    "action_S2.mp4_2_.mp4",
    "action_S2.mp4_8_.mp4",
    "action_S4.mp4_4_.mp4",
    "action_S4.mp4_2_.mp4",
    "action_S2.mp4_7_.mp4",
    "action_S4.mp4_10_.mp4",
]

BLINK_LANGUAGE = ["eyelid", "blink", "lid", "closure", "closing", "shut", "close"]
NON_BLINK_FEATURES = ["gaze", "asymmetry", "tension", "wrinkl", "pupil",
                       "sclera", "muscle", "orbit"]


def validate_v3_response(parsed: dict) -> str:
    """Inlined copy of validate_qwen_v3_response from eye_probe.py.

    Avoids the heavy import chain (qwen_audit -> gemini_audit -> google-genai).
    """
    pair = parsed.get("most_changed_frame_pair") or []
    if len(pair) != 2 or pair[0] == pair[1]:
        return "collapse:same_frame"
    obs = (parsed.get("observed_change") or "").lower()
    if any(t in obs for t in [
        "no change", "no visible change", "look the same",
        "appear the same", "looks the same", "appears the same",
    ]):
        if parsed.get("classification") == "action":
            return "inconsistent:action_with_no_change"
        return "collapse:no_change_template"
    features = ["eyelid", "sclera", "gaze", "pupil", "asymmetry",
                "tension", "wrinkl", "lid", "blink", "squint", "orbit",
                "muscle"]
    if not any(f in obs for f in features):
        return "collapse:no_feature_named"
    return "ok"


def grade_clip(clip: str, parsed: dict, expected_clip: dict) -> tuple[bool, str]:
    label = expected_clip["user_label"]
    obs = (parsed.get("observed_change") or "").lower()
    cls = parsed.get("classification")
    has_blink_lang = any(t in obs for t in BLINK_LANGUAGE)
    has_non_blink_feat = any(t in obs for t in NON_BLINK_FEATURES)

    if label == "blink":
        if cls == "action" and has_blink_lang:
            return True, "correct: action + blink language"
        if cls == "action":
            return False, f"action but no blink language ({obs[:60]!r})"
        return False, f"missed blink: classified {cls}"

    # no_blink
    if cls == "background":
        return True, "correct: background"
    if cls == "action" and not has_blink_lang and has_non_blink_feat:
        return True, f"correct: action w/ non-blink feature ({obs[:60]!r})"
    if cls == "action" and has_blink_lang:
        return False, f"FALSE POSITIVE blink ({obs[:60]!r})"
    return False, f"unclear: cls={cls} obs={obs[:60]!r}"


def main() -> int:
    expected = json.loads(EXPECTED.read_text())["clips"]
    runner = MiniCPMRunner()
    print(f"[sanity] model={runner.model_tag}", flush=True)
    print(f"[sanity] prompt={runner.prompt_name}", flush=True)
    print("[sanity] frame policy: native FPS (all frames)", flush=True)
    print(flush=True)

    rows: list[dict] = []
    OUT.unlink(missing_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    for i, clip_name in enumerate(CLIPS, 1):
        clip_path = str(CLIPS_DIR / clip_name)
        if not Path(clip_path).exists():
            print(f"  [{i}/6] MISSING: {clip_path}", flush=True)
            continue

        try:
            res = runner.run(clip_path, fps="native", temperature=0.0)
            parsed = parse_minicpm_json(res["text"])
            cls = parsed.get("classification")
            tag = validate_v3_response(parsed)
            err = None
        except Exception as e:
            res = {"text": "", "frame_count": None, "fps_used": "native",
                   "prompt_eval_count": None, "eval_count": None,
                   "wall_clock_seconds": None, "model_version": None,
                   "total_duration_ns": None, "load_duration_ns": None,
                   "prompt_eval_duration_ns": None, "eval_duration_ns": None}
            parsed = {}
            cls = None
            tag = "error"
            err = f"{type(e).__name__}: {e}"

        if err is None:
            ok, why = grade_clip(clip_name, parsed, expected[clip_name])
        else:
            ok, why = False, err

        row = {
            "clip": clip_name,
            "user_label": expected[clip_name]["user_label"],
            "model": runner.model_tag,
            "model_version": res.get("model_version"),
            "prompt_id": runner.prompt_name,
            "frame_count": res["frame_count"],
            "fps_policy": "native",
            "classification": cls,
            "observed_change": parsed.get("observed_change"),
            "most_changed_frame_pair": parsed.get("most_changed_frame_pair"),
            "frames_examined": parsed.get("frames_examined"),
            "eye_visible_in_frames": parsed.get("eye_visible_in_frames"),
            "validation_tag": tag,
            "graded_correct": ok,
            "grader_note": why,
            "raw_text": res["text"],
            "prompt_eval_count": res["prompt_eval_count"],
            "eval_count": res["eval_count"],
            "wall_clock_seconds": res["wall_clock_seconds"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "error": err,
        }
        rows.append(row)
        with open(OUT, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        mark = "[OK]  " if ok else "[FAIL]"
        latency = res.get("wall_clock_seconds") or 0
        print(f"  [{i}/6] {clip_name:24} cls={cls or 'ERR':<11} "
              f"frames={res['frame_count']} ({latency:.1f}s) {mark} {why}",
              flush=True)

    n_ok = sum(r["graded_correct"] for r in rows)
    print(flush=True)
    print(f"=== BLINK SANITY: {n_ok}/{len(rows)} correct ===", flush=True)

    # Template-collapse check (same observed_change across clips)
    obs_strings = [r.get("observed_change") or "" for r in rows]
    nonempty = [s for s in obs_strings if s]
    if nonempty:
        max_repeat = max(nonempty.count(s) for s in nonempty)
        if max_repeat >= 3:
            print(f"!!! STRUCTURAL FAIL: {max_repeat} clips share identical "
                  f"observed_change (template hallucination)",
                  flush=True)
            print("    Sanity result invalidated regardless of label match.",
                  flush=True)
            return 1

    if n_ok >= 5:
        print(f"PASS gate: {n_ok}/6 >= 5/6 -> proceed to 36-clip run",
              flush=True)
        return 0
    print(f"FAIL gate: {n_ok}/6 < 5/6 -> STOP, do not run 36-clip",
          flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
