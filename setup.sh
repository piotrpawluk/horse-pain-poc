#!/usr/bin/env bash
# Faza 0 — idempotentny installer.
# Uruchamiać z poc/ (cd poc && bash setup.sh).
# Można odpalić wielokrotnie — każdy krok ma guard na istnienie.

set -euo pipefail

POC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$POC_DIR"

echo "==> Faza 0 setup w $POC_DIR"

# 1. uv (jeśli brak)
if ! command -v uv >/dev/null 2>&1; then
    echo "==> [1/6] Instaluję uv (Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Po instalacji uv ląduje w ~/.local/bin lub ~/.cargo/bin — dopisać do PATH na tę sesję.
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
else
    echo "==> [1/6] uv już zainstalowane: $(uv --version)"
fi

# 2. virtualenv .venv (Python 3.11 — DLC 3.x stabilne na 3.10–3.11)
if [ ! -d ".venv" ]; then
    echo "==> [2/6] Tworzę virtualenv (Python 3.11) w .venv/..."
    uv venv --python 3.11
else
    echo "==> [2/6] .venv już istnieje, pomijam"
fi

# 3. zależności z pyproject.toml
echo "==> [3/6] Instaluję zależności (DLC, torch, HF, jupyter)..."
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

# 4. weights SuperAnimal-Quadruped — DLC 3.x pobiera lazy przy pierwszym `video_inference_superanimal`.
#    Próbujemy proaktywnie, żeby Notebook 00 nie zatrzymał się na downloadzie. Niewykonanie = OK.
WEIGHTS_DIR="checkpoints/superanimal-quadruped"
if [ ! -d "$WEIGHTS_DIR" ] || [ -z "$(ls -A "$WEIGHTS_DIR" 2>/dev/null)" ]; then
    echo "==> [4/6] Próbuję pobrać SuperAnimal-Quadruped weights z HuggingFace..."
    .venv/bin/python - <<PY || echo "    (UWAGA: pobranie padło — DLC pobierze lazy przy pierwszym użyciu w notebooku 00)"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped",
    local_dir="$WEIGHTS_DIR",
    allow_patterns=["*.pth", "*.json", "*.yaml"],
)
print("    weights pobrane do $WEIGHTS_DIR")
PY
else
    echo "==> [4/6] Weights już istnieją w $WEIGHTS_DIR, pomijam"
fi

# 5. clone read-my-ears
if [ ! -d "vendor/read-my-ears/.git" ]; then
    echo "==> [5/6] Klonuję jmalves5/read-my-ears..."
    git clone --depth 1 https://github.com/jmalves5/read-my-ears vendor/read-my-ears
else
    echo "==> [5/6] vendor/read-my-ears już istnieje, pomijam"
fi

# 6. sample horse video (jeśli brak)
#    Strategie po kolei: (a) DLC examples, (b) Pexels CC0 known URL, (c) instrukcja dla user'a.
SAMPLE_VIDEO="data/sample_horse.mp4"
if [ ! -f "$SAMPLE_VIDEO" ]; then
    echo "==> [6/6] Pobieram sample horse video..."
    # (a) DLC ma examples — sprawdźmy
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
    print(f"    (a) Kopiuję {src} -> {target}")
    shutil.copy(src, target)
else:
    print("    (a) Brak DLC sample horse video w pakiecie.")
PY
    # (b) Wikimedia Commons — stabilne CC URL, koń chodzący w corralu (9.6s)
    if [ ! -f "$SAMPLE_VIDEO" ]; then
        echo "    (b) Pobieram z Wikimedia Commons (Horse walking in corral, CC)..."
        SAMPLE_OGV="data/sample_horse.ogv"
        curl -fsSL --max-time 60 \
            -A "horse-training-poc/0.1 (peter@example.org)" \
            -o "$SAMPLE_OGV" \
            "https://upload.wikimedia.org/wikipedia/commons/c/c7/Horse_walking_in_corral_MVI_7490.MOV.ogv" || \
            rm -f "$SAMPLE_OGV"
        if [ -f "$SAMPLE_OGV" ]; then
            echo "    (b) Pobrane: $SAMPLE_OGV ($(du -h $SAMPLE_OGV | cut -f1))"
            # Konwersja ogv → mp4 (DLC/OpenCV preferuje mp4)
            .venv/bin/python - <<'PY' || rm -f "$SAMPLE_VIDEO"
import cv2
src = "data/sample_horse.ogv"
dst = "data/sample_horse.mp4"
cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"    konwersja: {w}x{h} @ {fps:.1f}fps")
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
print(f"    zapisano {n} klatek do {dst}")
PY
            rm -f "$SAMPLE_OGV"
        fi
    fi
    # (c) instrukcja dla usera
    if [ ! -f "$SAMPLE_VIDEO" ]; then
        cat <<'MSG'
    ⚠ Nie udało się automatycznie pobrać sample video.
    Wgraj ręcznie dowolny krótki (10–60s) klip konia w hali do:
        data/sample_horse.mp4
    Sugerowane źródła CC0:
      - https://www.pexels.com/search/videos/horse/
      - https://www.pixabay.com/videos/search/horse/
      - https://commons.wikimedia.org/wiki/Category:Videos_of_horses
    Bez tego pliku notebook 00 zatrzyma się na sanity check.
MSG
    fi
else
    echo "==> [6/6] $SAMPLE_VIDEO już istnieje, pomijam"
fi

# 7. clone horse-face-ear-detection — YOLOv8n custom weights dla movement-detection
HFED="vendor/horse-face-ear-detection"
if [ ! -d "$HFED/.git" ]; then
    echo "==> [7/8] Klonuję jmalves5/horse-face-ear-detection (YOLOv8n custom weights)..."
    git clone --depth 1 https://github.com/jmalves5/horse-face-ear-detection "$HFED" || \
        echo "    (UWAGA: clone padło — Etap A nie zadziała bez tych weights)"
else
    echo "==> [7/8] $HFED już istnieje, pomijam"
fi

# 8. download subsetu HF dataset joaomalves/read-my-ears (test split + ~20 klipów)
RME_DATA="vendor/ReadMyEars_Dataset/data"
if [ ! -f "$RME_DATA/test.csv" ]; then
    echo "==> [8/8] Pobieram subset joaomalves/read-my-ears (CSVs + S1 klipy)..."
    .venv/bin/python - <<'PY' || echo "    (UWAGA: HF download padł — Etap A może nie mieć danych)"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="joaomalves/read-my-ears",
    repo_type="dataset",
    local_dir="vendor/ReadMyEars_Dataset/data",
    allow_patterns=["test.csv", "train.csv", "val.csv", "videos/action_S1.*", "videos/background_S1.*"],
)
print("    pobrane do vendor/ReadMyEars_Dataset/data/")
PY
else
    echo "==> [8/8] HF dataset subset już istnieje, pomijam"
fi

echo ""
echo "==> ✓ Setup gotowy."
echo "==> Notebook 00 (DLC smoke):     jupyter lab notebooks/00_smoke_dlc_sample.ipynb"
echo "==> Notebook 01 (RME movement):  jupyter lab notebooks/01_read_my_ears_replicate.ipynb"
echo "==> Colab fallback:              notebooks/99_colab_fallback.ipynb"
