"""Modal app — thin wrapper around `dlc_inference.run_dlc_inference`.

The vanilla function lives in `dlc_inference.py` (portable). This file holds
ONLY the Modal-specific decoration, image definition, and IO marshalling.

If Modal becomes friction (price changes, API churn, sunset), swap this file
for a RunPod/Vast.ai wrapper — `dlc_inference.py` is untouched.

Local versions captured 2026-05-13 via `uv pip freeze --python .venv/bin/python`
on /Users/peterpawluk/horse-training/poc/.venv (Python 3.11.13). Modal image
pins the major direct deps; pip resolves transitives.

DLC VERSION TRUTH (audit-load-bearing — do not collapse):
  - pip dist-info Version: 3.0.0rc14 (the source of truth — bytes installed)
  - runtime `deeplabcut.__version__` constant: 3.0.0rc13 (display string lags
    because the rc14 release process had an incomplete version bump; see
    DLC GitHub PRs #3096 + #3109)
  - rc14 contains substantive inference-affecting changes vs rc13:
    PR #3105 disables `torch.autocast` by default in inference,
    PR #3154 filters low-confidence bbox detections,
    PR #3078 fixes RTMPose likelihood, plus several PyTorch speed fixes
  - Phase 8b ran on this same install, so Phase 8b is also rc14-bytes
  - Modal image pin `deeplabcut==3.0.0rc14` matches the pip artifact bytewise
  - Smoke-test result JSON captures BOTH version values as separate fields
    plus dist-info METADATA SHA256 to prove binary identity across local
    and Modal
"""

from __future__ import annotations

import modal

# Image definition — versions pinned to match local 2026-05-13 capture.
# Linear-algebra ops at the bit level are torch-version-stable; CUDA vs CPU
# kernel differences are addressed downstream by the smoke-test validator
# (0.5 px max delta threshold against local CPU baseline).
DLC_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    # CUBLAS_WORKSPACE_CONFIG MUST be set before torch is imported for
    # torch.use_deterministic_algorithms(True) to work with CUDA matmul. Setting
    # it on the image .env() ensures it's present in every worker process.
    .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    # apt layer: ffmpeg for DLC video I/O + opencv-headless deps.
    # libxext6/libsm6 are belt-and-suspenders for matplotlib/cv2 transitives
    # that occasionally try to import GUI libs even in headless mode.
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "libxext6", "libsm6")
    # Install uv, then use `uv pip install --override` to install the exact
    # local-matching numpy 2.4.4 past DLC rc14's declared `numpy<2` constraint.
    # Local install (via uv) has been running on numpy 2.4.4 for the entire
    # Phase 8b + Phase 10a-prelim chain, producing valid keypoint h5 files —
    # so numpy 2.x is empirically compatible with DLC rc14 despite the
    # declared constraint. Modal-vs-local methodology preservation requires
    # matching the empirically-validated runtime, not the declared constraint.
    # See `outputs/cloud_dlc_audit_footnotes.md` Footnote 1 + Lesson 22.
    .run_commands(
        "python -m pip install --no-cache-dir uv==0.9.0",
        # Override file: pin local-empirically-validated versions past DLC's
        # over-conservative declared constraints. DLC rc14 declares numpy<2
        # and pandas<3, but the local install (running Phase 8b + the
        # in-flight Phase 10a-prelim CPU process) uses numpy 2.4.4 + pandas
        # 3.0.2 and produces valid h5 keypoint outputs. Audit-doc Footnote 7
        # documents the constraint-vs-runtime divergence.
        "printf '%s\\n' "
        "'numpy==2.4.4' "
        "'pandas==3.0.2' "
        "'scipy==1.17.1' "
        "'scikit-learn==1.8.0' "
        "'scikit-image==0.26.0' "
        "'matplotlib==3.8.4' "
        "'h5py==3.16.0' "
        "'tables==3.11.1' "
        "'networkx==3.6.1' "
        "'imageio==2.37.3' "
        "'pyyaml==6.0.3' "
        "'tqdm==4.67.3' "
        "'huggingface-hub==1.14.0' "
        "'timm==1.0.26' "
        "'einops==0.8.2' "
        "'albumentations==1.4.3' "
        "'filterpy==1.4.5' "
        "'opencv-python-headless==4.11.0.86' "
        "'torch==2.11.0' "
        "'torchvision==0.26.0' "
        "'transformers==5.8.0' "
        "'safetensors==0.8.0rc0' "
        "'tokenizers==0.23.0rc0' "
        "'protobuf==6.33.6' "
        "> /tmp/overrides.txt",
        "uv pip install --system --override /tmp/overrides.txt "
        "torch==2.11.0 torchvision==0.26.0 "
        "deeplabcut==3.0.0rc14 numpy==2.4.4 "
        "h5py==3.16.0 tables==3.11.1 pandas==3.0.2 scipy==1.17.1 "
        "scikit-image==0.26.0 scikit-learn==1.8.0 matplotlib==3.8.4 "
        "opencv-python-headless==4.11.0.86 imageio==2.37.3 "
        "imageio-ffmpeg==0.6.0 pyyaml==6.0.3 'ruamel.yaml==0.19.1' "
        "tqdm==4.67.3 albumentations==1.4.3 filterpy==1.4.5 "
        "networkx==3.6.1 huggingface-hub==1.14.0 timm==1.0.26 "
        "einops==0.8.2 "
        # HuggingFace stack for V-JEPA-2 ViT-L inference (pinned to local
        # versions captured 2026-05-13 via uv pip freeze). transformers 5.8.0
        # is recent; if it pulls a different torch transitively, the override
        # file forces torch==2.11.0 to win. safetensors==0.8.0rc0 and
        # tokenizers==0.23.0rc0 are release-candidate versions; pinned exact
        # for methodology reproducibility.
        "transformers==5.8.0 'safetensors==0.8.0rc0' "
        "'tokenizers==0.23.0rc0' sentencepiece==0.2.1 "
        "protobuf==6.33.6",
    )
)

app = modal.App("phase10a-dlc")


@app.function(
    image=DLC_IMAGE,
    gpu="T4",
    timeout=3600,
)
def run_dlc_remote(video_bytes: bytes, clip_filename: str) -> dict:
    """Run DLC inference on cloud T4; return h5 bytes + runtime metadata.

    Returns a dict (Modal-serializable) with:
      - h5_bytes: bytes — raw h5 file contents
      - runtime_env: dict — version capture for audit
      - wall_clock_s: float — DLC inference wall-clock on T4
    """
    import tempfile
    import time
    from pathlib import Path

    # vendored import — Modal serializes function source; relative imports
    # don't work, so we copy the vanilla function inline here.
    # If you need to update dlc_inference.py, also update the inline copy below.
    # (Modal does support `add_local_python_source` for cleaner imports; using
    # inline-copy here for explicit version visibility in the cloud function.)

    SUPERANIMAL_NAME = "superanimal_quadruped"
    MODEL_NAME = "hrnet_w32"
    DETECTOR_NAME = "fasterrcnn_resnet50_fpn_v2"
    H5_SUFFIX = f"_{SUPERANIMAL_NAME}_{MODEL_NAME}_{DETECTOR_NAME}.h5"

    workdir = Path(tempfile.mkdtemp(prefix="dlc_modal_"))
    video_path = workdir / clip_filename
    video_path.write_bytes(video_bytes)

    import random
    import warnings
    import deeplabcut
    import numpy as np
    import torch

    # CUDA determinism (CUBLAS_WORKSPACE_CONFIG is set on the image .env()).
    # warn_only=True is a DELIBERATE relaxation: some torch ops have no
    # deterministic CUDA implementation. Strict mode (warn_only=False) would
    # raise immediately. The pragmatic path is to permit the warning AND
    # capture it for audit — see warnings.catch_warnings block below. If the
    # captured_warnings list is empty in the return dict, the run was
    # effectively bit-exact-deterministic. If non-empty, the specific ops are
    # named and audit doc can footnote them.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)

    wall_0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        deeplabcut.video_inference_superanimal(
            videos=[str(video_path)],
            superanimal_name=SUPERANIMAL_NAME,
            model_name=MODEL_NAME,
            detector_name=DETECTOR_NAME,
            video_adapt=False,
            pseudo_threshold=0.1,
            dest_folder=str(workdir),
            create_labeled_video=False,
            plot_bboxes=False,
            device="cuda",
        )
    wall_elapsed = time.perf_counter() - wall_0

    # Serialize warnings — keep only torch determinism-related ones to avoid
    # noise (DLC + torch emit many UserWarnings unrelated to determinism).
    serialized_warnings = []
    determinism_warnings = []
    for w in captured_warnings:
        entry = {
            "category": (w.category.__name__ if w.category else None),
            "message": str(w.message),
            "filename": w.filename,
            "lineno": w.lineno,
        }
        serialized_warnings.append(entry)
        msg_lower = str(w.message).lower()
        if (
            "deterministic" in msg_lower
            or "nondeterministic" in msg_lower
            or "non-deterministic" in msg_lower
        ):
            determinism_warnings.append(entry)

    h5_out = workdir / f"{video_path.stem}{H5_SUFFIX}"
    if not h5_out.exists():
        raise RuntimeError(f"DLC did not write expected h5: {h5_out}")

    import hashlib
    import sys

    try:
        from importlib.metadata import distribution, version as _v
        pip_dlc = _v("deeplabcut")
        dist = distribution("deeplabcut")
        metadata_text = dist.read_text("METADATA") or ""
        metadata_sha256 = hashlib.sha256(metadata_text.encode("utf-8")).hexdigest()
    except Exception:
        pip_dlc = None
        metadata_sha256 = None

    try:
        version_py_sha256 = hashlib.sha256(
            (Path(deeplabcut.__file__).parent / "version.py").read_bytes()
        ).hexdigest()
    except Exception:
        version_py_sha256 = None

    return {
        "h5_bytes": h5_out.read_bytes(),
        "wall_clock_s": wall_elapsed,
        "determinism_warnings": determinism_warnings,
        "all_warnings_count": len(serialized_warnings),
        "all_warnings_sample": serialized_warnings[:20],
        "runtime_env": {
            "python_version": sys.version.split()[0],
            "deeplabcut_pip_dist_info_version": pip_dlc,
            "deeplabcut_runtime_version_constant": deeplabcut.__version__,
            "deeplabcut_dist_info_metadata_sha256": metadata_sha256,
            "deeplabcut_version_py_sha256": version_py_sha256,
            "torch_version": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
            "torch_cuda_arch_list": (
                torch.cuda.get_arch_list() if torch.cuda.is_available() else None
            ),
            "torch_device_name": torch.cuda.get_device_name(0),
            "torch_cudnn_version": (
                torch.backends.cudnn.version() if torch.cuda.is_available() else None
            ),
            "torch_deterministic_algorithms": True,
            "torch_deterministic_warn_only": True,
            "torch_cudnn_deterministic": True,
            "torch_cudnn_benchmark": False,
            "seed": 42,
        },
    }


@app.function(
    image=DLC_IMAGE,
    gpu="T4",
    timeout=1800,
)
def run_vjepa2_remote(video_bytes: bytes, clip_filename: str) -> dict:
    """Run V-JEPA-2 ViT-L inference on cloud T4 for one clip.

    Mirrors `tools/extract_vjepa2.py` exactly: 16 evenly-spaced frames as
    RGB uint8 → AutoVideoProcessor → VJEPA2Model → mean-pool last_hidden_state
    over patch tokens → (1024,) float32 numpy.

    Returns dict with:
      - embedding_bytes: bytes — (1024,) float32 numpy ndarray serialized
      - runtime_env: dict — version + GPU metadata for audit
      - wall_clock_s: float — V-JEPA-2 inference wall-clock on T4
    """
    import hashlib
    import random
    import sys
    import tempfile
    import time
    from pathlib import Path

    import cv2
    import numpy as np
    import torch
    from transformers import AutoVideoProcessor, VJEPA2Model

    # CUDA determinism mirrors run_dlc_remote. warn_only=True permits ops
    # without deterministic implementations to proceed, but warnings are
    # captured below and surfaced in the result JSON for audit.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)

    # Constants pinned to match tools/extract_vjepa2.py exactly
    MODEL_ID = "facebook/vjepa2-vitl-fpc16-256-ssv2"
    NUM_FRAMES = 16

    workdir = Path(tempfile.mkdtemp(prefix="vjepa2_modal_"))
    video_path = workdir / clip_filename
    video_path.write_bytes(video_bytes)

    # Load model + processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_load = time.perf_counter()
    processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
    model = VJEPA2Model.from_pretrained(MODEL_ID).to(device).eval()
    load_elapsed = time.perf_counter() - t_load

    # read_clip_frames + extract_embedding — verbatim from extract_vjepa2.py
    cap = cv2.VideoCapture(str(video_path))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_total < 1:
        cap.release()
        raise RuntimeError(f"video has 0 frames: {video_path}")
    indices = np.linspace(0, n_total - 1, NUM_FRAMES).astype(int)
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, f = cap.read()
        if not ok:
            f = frames[-1] if frames else np.zeros((256, 256, 3), dtype=np.uint8)
            frames.append(f)
            continue
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    frames_arr = np.stack(frames)

    t_infer = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        with torch.no_grad():
            inputs = processor(videos=list(frames_arr), return_tensors="pt").to(device)
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().float().numpy()
    infer_elapsed = time.perf_counter() - t_infer

    # Serialize captured warnings — split determinism-related vs others
    serialized_warnings = []
    determinism_warnings = []
    for w in captured_warnings:
        entry = {
            "category": (w.category.__name__ if w.category else None),
            "message": str(w.message),
            "filename": w.filename,
            "lineno": w.lineno,
        }
        serialized_warnings.append(entry)
        msg_lower = str(w.message).lower()
        if (
            "deterministic" in msg_lower
            or "nondeterministic" in msg_lower
            or "non-deterministic" in msg_lower
        ):
            determinism_warnings.append(entry)

    # Capture version manifest for audit
    try:
        from importlib.metadata import distribution, version as _v
        tf_v = _v("transformers")
        tf_dist = distribution("transformers")
        tf_meta = tf_dist.read_text("METADATA") or ""
        tf_meta_sha = hashlib.sha256(tf_meta.encode("utf-8")).hexdigest()
    except Exception:
        tf_v = None
        tf_meta_sha = None

    return {
        "embedding_bytes": emb.tobytes(),
        "embedding_shape": list(emb.shape),
        "embedding_dtype": str(emb.dtype),
        "wall_clock_s": load_elapsed + infer_elapsed,
        "model_load_s": load_elapsed,
        "inference_s": infer_elapsed,
        "n_frames_read": int(len(frames_arr)),
        "determinism_warnings": determinism_warnings,
        "all_warnings_count": len(serialized_warnings),
        "all_warnings_sample": serialized_warnings[:20],
        "runtime_env": {
            "python_version": sys.version.split()[0],
            "transformers_pip_version": tf_v,
            "transformers_metadata_sha256": tf_meta_sha,
            "model_id": MODEL_ID,
            "num_frames": NUM_FRAMES,
            "torch_version": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
            "torch_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "torch_cudnn_version": (
                torch.backends.cudnn.version() if torch.cuda.is_available() else None
            ),
            "torch_deterministic_algorithms": True,
            "torch_deterministic_warn_only": True,
            "torch_cudnn_deterministic": True,
            "torch_cudnn_benchmark": False,
            "seed": 42,
        },
    }


@app.local_entrypoint()
def smoke_test_one_clip(clip_path: str = "data/prudnik/IMG_1050.mov") -> None:
    """Convenience entrypoint: `modal run tools/cloud_dlc/app.py::smoke_test_one_clip`."""
    from pathlib import Path

    p = Path(clip_path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / clip_path
    if not p.exists():
        raise FileNotFoundError(f"clip not found: {p}")

    print(f"[modal] uploading {p.name} ({p.stat().st_size / 1e6:.1f} MB) to T4…")
    result = run_dlc_remote.remote(p.read_bytes(), p.name)
    print(f"[modal] T4 wall-clock : {result['wall_clock_s']:.2f} s")
    print(f"[modal] runtime env  : {result['runtime_env']}")
    print(f"[modal] h5 bytes recv: {len(result['h5_bytes']) / 1024:.1f} KB")
    out = Path(__file__).resolve().parents[2] / "temp" / "cloud_dlc_smoke" / f"{p.stem}.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(result["h5_bytes"])
    print(f"[modal] saved        : {out}")
