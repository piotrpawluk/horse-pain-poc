#!/usr/bin/env python3
"""MiniCPM-V 4.5 audit wrapper via Ollama HTTP API.

Mirrors QwenRunner from qwen_audit.py: load once, run() per clip, return
text + tokens + frame_count + latency. The Ollama daemon is presumed running
on localhost:11434.

Frame extraction: ffmpeg -> PNG sequence -> base64 -> Ollama /api/chat images[].

The prompt is loaded from tools/prompts/<prompt_name>.txt and parsed on
====SYSTEM==== / ====USER==== markers. Bumping the prompt requires a new
versioned file (eye_v4_minicpm.txt etc.) — never edit a frozen file.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import requests

POC_DIR = Path(__file__).resolve().parent.parent
PROMPT_DIR = POC_DIR / "tools" / "prompts"
OLLAMA_URL = "http://localhost:11434"


def load_prompt(prompt_name: str) -> tuple[str, str]:
    """Load a versioned prompt. Returns (system, user)."""
    path = PROMPT_DIR / f"{prompt_name}.txt"
    text = path.read_text()
    sys_match = re.search(
        r"====SYSTEM====\s*\n(.*?)(?=\n====USER====)", text, re.DOTALL)
    user_match = re.search(
        r"====USER====\s*\n(.*?)\Z", text, re.DOTALL)
    if not sys_match or not user_match:
        raise ValueError(
            f"{path}: missing ====SYSTEM==== or ====USER==== marker")
    return sys_match.group(1).strip(), user_match.group(1).strip()


def extract_frames(clip_path: str, *, fps: float | str = "native",
                   tmp_dir: Path | None = None) -> list[Path]:
    """Extract frames from clip via ffmpeg.

    fps='native' -> no -vf filter (all native frames).
    fps=<float>  -> ffmpeg -vf fps=<value>.
    """
    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="minicpm_frames_"))
    pattern = tmp_dir / "f%03d.png"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", clip_path]
    if fps != "native":
        cmd += ["-vf", f"fps={fps}"]
    cmd += [str(pattern)]
    subprocess.run(cmd, check=True)
    return sorted(tmp_dir.glob("*.png"))


def call_ollama_chat(model: str, system: str, user: str,
                     image_paths: list[Path], *,
                     temperature: float = 0.0,
                     num_predict: int = 320,
                     timeout: int = 300) -> dict:
    """POST to /api/chat with system + user(images, text). Non-streaming."""
    images_b64 = [
        base64.b64encode(p.read_bytes()).decode("ascii")
        for p in image_paths
    ]
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user, "images": images_b64},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    t0 = time.time()
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    payload["_latency_s"] = time.time() - t0
    return payload


def parse_minicpm_json(text: str) -> dict:
    """Strip ```json fences, pick first {...} block, parse."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in: {text[:200]!r}")
    return json.loads(m.group(0))


class MiniCPMRunner:
    def __init__(self, model_tag: str = "openbmb/minicpm-v4.5",
                 prompt_name: str = "eye_v3_minicpm"):
        self.model_tag = model_tag
        self.prompt_name = prompt_name
        self.system, self.user = load_prompt(prompt_name)
        # Verify model is registered with Ollama.
        r = requests.post(
            f"{OLLAMA_URL}/api/show",
            json={"name": model_tag}, timeout=10,
        )
        r.raise_for_status()

    def run(self, clip_path: str, *, fps: float | str = "native",
            temperature: float = 0.0, num_predict: int = 320,
            timeout: int = 300) -> dict:
        with tempfile.TemporaryDirectory(prefix="minicpm_frames_") as tmp:
            frames = extract_frames(clip_path, fps=fps, tmp_dir=Path(tmp))
            if not frames:
                raise RuntimeError(f"no frames extracted from {clip_path}")
            payload = call_ollama_chat(
                self.model_tag, self.system, self.user, frames,
                temperature=temperature, num_predict=num_predict,
                timeout=timeout,
            )
        text = (payload.get("message") or {}).get("content", "")
        return {
            "text": text,
            "frame_count": len(frames),
            "fps_used": fps,
            "prompt_eval_count": payload.get("prompt_eval_count"),
            "eval_count": payload.get("eval_count"),
            "total_duration_ns": payload.get("total_duration"),
            "load_duration_ns": payload.get("load_duration"),
            "prompt_eval_duration_ns": payload.get("prompt_eval_duration"),
            "eval_duration_ns": payload.get("eval_duration"),
            "wall_clock_seconds": payload.get("_latency_s"),
            "model_version": payload.get("model"),
        }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--fps", default="native",
                    help="'native' or a float like 10")
    ap.add_argument("--prompt", default="eye_v3_minicpm")
    ap.add_argument("--model", default="openbmb/minicpm-v4.5")
    args = ap.parse_args()
    fps = args.fps if args.fps == "native" else float(args.fps)
    runner = MiniCPMRunner(model_tag=args.model, prompt_name=args.prompt)
    result = runner.run(args.clip, fps=fps)
    print(json.dumps(result, indent=2, ensure_ascii=False))
