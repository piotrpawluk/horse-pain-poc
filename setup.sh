#!/usr/bin/env bash
# Phase 0 — idempotent installer.
# Run from poc/ (cd poc && bash setup.sh).
# Safe to re-run — every step has an existence guard.

set -euo pipefail

POC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$POC_DIR"

echo "==> Phase 0 setup in $POC_DIR"

# 1. uv (if missing)
if ! command -v uv >/dev/null 2>&1; then
    echo "==> [1/6] Installing uv (Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv lands in ~/.local/bin or ~/.cargo/bin — add to PATH for this session.
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
else
    echo "==> [1/6] uv already installed: $(uv --version)"
fi

# 2. virtualenv .venv (Python 3.11 — DLC 3.x is stable on 3.10–3.11)
if [ ! -d ".venv" ]; then
    echo "==> [2/6] Creating virtualenv (Python 3.11) in .venv/..."
    uv venv --python 3.11
else
    echo "==> [2/6] .venv already exists, skipping"
fi

# 3. dependencies from pyproject.toml
echo "==> [3/6] Installing dependencies (DLC, torch, HF, jupyter)..."
uv pip install --python .venv/bin/python --prerelease=allow \
    "deeplabcut[modelzoo]>=3.0.0rc14" \
    "torch>=2.3" \
    "torchvision" \
    "huggingface_hub>=0.24" \
    "transformers>=4.44" \
    "opencv-python-headless>=4.10" \
    "matplotlib>=3.7,<3.9" \
    "numpy>=1.26,<2.1" \
    "pandas>=2.2" \
    "jupyter>=1.0" \
    "ipykernel>=6.29" \
    "yt-dlp>=2024.8" \
    "ultralytics>=8.3"

# 4. SuperAnimal-Quadruped weights — DLC 3.x lazy-fetches them on the first
#    `video_inference_superanimal` call. We try proactively so notebook 00 doesn't
#    stall on the download. Failure here is OK.
WEIGHTS_DIR="checkpoints/superanimal-quadruped"
if [ ! -d "$WEIGHTS_DIR" ] || [ -z "$(ls -A "$WEIGHTS_DIR" 2>/dev/null)" ]; then
    echo "==> [4/6] Trying to fetch SuperAnimal-Quadruped weights from HuggingFace..."
    .venv/bin/python - <<PY || echo "    (NOTE: download failed — DLC will fetch lazily on first use in notebook 00)"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped",
    local_dir="$WEIGHTS_DIR",
    allow_patterns=["*.pth", "*.json", "*.yaml"],
)
print("    weights downloaded to $WEIGHTS_DIR")
PY
else
    echo "==> [4/6] Weights already present in $WEIGHTS_DIR, skipping"
fi

# 5. clone read-my-ears
if [ ! -d "vendor/read-my-ears/.git" ]; then
    echo "==> [5/6] Cloning jmalves5/read-my-ears..."
    git clone --depth 1 https://github.com/jmalves5/read-my-ears vendor/read-my-ears
else
    echo "==> [5/6] vendor/read-my-ears already present, skipping"
fi

# 6. sample horse video (if missing)
#    Strategies in order: (a) DLC examples, (b) Pexels CC0 known URL, (c) instructions for the user.
SAMPLE_VIDEO="data/sample_horse.mp4"
if [ ! -f "$SAMPLE_VIDEO" ]; then
    echo "==> [6/6] Fetching sample horse video..."
    # (a) DLC ships examples — check
    .venv/bin/python - <<'PY' || true
import os, shutil, glob
import deeplabcut
dlc_dir = os.path.dirname(deeplabcut.__file__)
candidates = (
    glob.glob(os.path.join(dlc_dir, "**", "*horse*.mp4"), recursive=True)
    + glob.glob(os.path.join(dlc_dir, "**", "*Horse*.mp4"), recursive=True)
    + glob.glob(os.path.join(dlc_dir, "examples", "**", "*.mp4"), recursive=True)
)
target = "data/sample_horse.mp4"
if candidates:
    src = candidates[0]
    print(f"    (a) Copying {src} -> {target}")
    shutil.copy(src, target)
else:
    print("    (a) No DLC sample horse video in the package.")
PY
    # (b) Wikimedia Commons — stable CC URL, horse walking in a corral (9.6 s)
    if [ ! -f "$SAMPLE_VIDEO" ]; then
        echo "    (b) Fetching from Wikimedia Commons (Horse walking in corral, CC)..."
        SAMPLE_OGV="data/sample_horse.ogv"
        curl -fsSL --max-time 60 \
            -A "horse-training-poc/0.1 (peter@example.org)" \
            -o "$SAMPLE_OGV" \
            "https://upload.wikimedia.org/wikipedia/commons/c/c7/Horse_walking_in_corral_MVI_7490.MOV.ogv" || \
            rm -f "$SAMPLE_OGV"
        if [ -f "$SAMPLE_OGV" ]; then
            echo "    (b) Downloaded: $SAMPLE_OGV ($(du -h $SAMPLE_OGV | cut -f1))"
            # Convert ogv → mp4 (DLC/OpenCV prefer mp4)
            .venv/bin/python - <<'PY' || rm -f "$SAMPLE_VIDEO"
import cv2
src = "data/sample_horse.ogv"
dst = "data/sample_horse.mp4"
cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"    converting: {w}x{h} @ {fps:.1f}fps")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(dst, fourcc, fps, (w, h))
n = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    out.write(fr)
    n += 1
cap.release(); out.release()
print(f"    wrote {n} frames to {dst}")
PY
            rm -f "$SAMPLE_OGV"
        fi
    fi
    # (c) instructions for the user
    if [ ! -f "$SAMPLE_VIDEO" ]; then
        cat <<'MSG'
    ⚠ Could not auto-fetch a sample video.
    Drop in any short (10–60 s) clip of a horse in an arena at:
        data/sample_horse.mp4
    Suggested CC0 sources:
      - https://www.pexels.com/search/videos/horse/
      - https://www.pixabay.com/videos/search/horse/
      - https://commons.wikimedia.org/wiki/Category:Videos_of_horses
    Without this file, notebook 00 will stop at the sanity check.
MSG
    fi
else
    echo "==> [6/6] $SAMPLE_VIDEO already present, skipping"
fi

# 7. clone horse-face-ear-detection — YOLOv8n custom weights for movement-detection
HFED="vendor/horse-face-ear-detection"
if [ ! -d "$HFED/.git" ]; then
    echo "==> [7/8] Cloning jmalves5/horse-face-ear-detection (YOLOv8n custom weights)..."
    git clone --depth 1 https://github.com/jmalves5/horse-face-ear-detection "$HFED" || \
        echo "    (NOTE: clone failed — Stage A will not work without these weights)"
else
    echo "==> [7/8] $HFED already present, skipping"
fi

# 8. download a subset of HF dataset joaomalves/read-my-ears (test split + ~20 clips)
RME_DATA="vendor/ReadMyEars_Dataset/data"
if [ ! -f "$RME_DATA/test.csv" ]; then
    echo "==> [8/8] Fetching subset of joaomalves/read-my-ears (CSVs + S1 clips)..."
    .venv/bin/python - <<'PY' || echo "    (NOTE: HF download failed — Stage A may have no data)"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="joaomalves/read-my-ears",
    repo_type="dataset",
    local_dir="vendor/ReadMyEars_Dataset/data",
    allow_patterns=["test.csv", "train.csv", "val.csv", "videos/action_S1.*", "videos/background_S1.*"],
)
print("    downloaded to vendor/ReadMyEars_Dataset/data/")
PY
else
    echo "==> [8/8] HF dataset subset already present, skipping"
fi

echo ""
echo "==> ✓ Setup ready."
echo "==> Notebook 00 (DLC smoke):     jupyter lab notebooks/00_smoke_dlc_sample.ipynb"
echo "==> Notebook 01 (RME movement):  jupyter lab notebooks/01_read_my_ears_replicate.ipynb"
echo "==> Colab fallback:              notebooks/99_colab_fallback.ipynb"
