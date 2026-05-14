"""Vanilla DLC SuperAnimal-Quadruped inference — portable across providers.

Pure Python function: no Modal decorators, no cloud-specific imports. Same
signature works on local CPU, Modal serverless, RunPod Pods, Vast.ai, or any
CUDA Linux box. Modal-specific wrapping lives in `app.py`.

This is the portability insurance — if Modal becomes friction, swap the
wrapper, keep this function.

Parameter contract MUST match Phase 10a-prelim local pipeline EXACTLY
(see `tools/phase10a_prelim_run.py:load_or_compute_dlc`) for methodology
preservation.
"""

from __future__ import annotations

from pathlib import Path

# Locked parameters — DO NOT CHANGE without updating Phase 10a-prelim pre-reg D-locks
SUPERANIMAL_NAME = "superanimal_quadruped"
MODEL_NAME = "hrnet_w32"
DETECTOR_NAME = "fasterrcnn_resnet50_fpn_v2"
PSEUDO_THRESHOLD = 0.1
VIDEO_ADAPT = False
CREATE_LABELED_VIDEO = False
PLOT_BBOXES = False

# Expected DLC output filename suffix (deterministic from name/model/detector)
H5_SUFFIX = (
    f"_{SUPERANIMAL_NAME}_{MODEL_NAME}_{DETECTOR_NAME}.h5"
)


def expected_h5_path(video_path: Path | str, dest_folder: Path | str) -> Path:
    """Return the h5 path DLC will write for `video_path` into `dest_folder`."""
    video_path = Path(video_path)
    dest_folder = Path(dest_folder)
    return dest_folder / f"{video_path.stem}{H5_SUFFIX}"


def run_dlc_inference(
    video_path: str | Path,
    dest_folder: str | Path,
    device: str = "auto",
) -> Path:
    """Run DLC SuperAnimal-Quadruped on one video; return path to h5 output.

    Parameters match Phase 10a-prelim pre-reg D-locks. Caller is responsible
    for ensuring `dest_folder` exists and is writable.

    `device` accepts: "auto" (CPU on Mac / CUDA on Linux GPU), "cuda", "cpu",
    "mps" (silent-fallback to CPU for Faster R-CNN ops — see Phase 10a MPS probe).
    """
    import deeplabcut

    video_path = Path(video_path)
    dest_folder = Path(dest_folder)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    dest_folder.mkdir(parents=True, exist_ok=True)

    deeplabcut.video_inference_superanimal(
        videos=[str(video_path)],
        superanimal_name=SUPERANIMAL_NAME,
        model_name=MODEL_NAME,
        detector_name=DETECTOR_NAME,
        video_adapt=VIDEO_ADAPT,
        pseudo_threshold=PSEUDO_THRESHOLD,
        dest_folder=str(dest_folder),
        create_labeled_video=CREATE_LABELED_VIDEO,
        plot_bboxes=PLOT_BBOXES,
        device=device,
    )

    h5_out = expected_h5_path(video_path, dest_folder)
    if not h5_out.exists():
        raise RuntimeError(
            f"DLC inference returned without writing expected h5: {h5_out}. "
            f"Check DLC logs and dest_folder contents."
        )
    return h5_out


def runtime_environment() -> dict:
    """Capture runtime versions + dist-info hash for audit reproducibility.

    Returns keys (load-bearing for audit — do NOT collapse pip_dist_info_version
    and runtime_version_constant into one field; they may legitimately differ
    due to in-code __version__ bumps lagging actual releases — see DLC rc14
    case where pip artifact is rc14 but runtime constant displays rc13).
    """
    import hashlib
    import sys
    import deeplabcut
    import torch

    try:
        from importlib.metadata import distribution, version as _v
        pip_dlc = _v("deeplabcut")
        dist = distribution("deeplabcut")
        # METADATA file SHA256 — proves dist-info bytes match across installs
        metadata_bytes = dist.read_text("METADATA").encode("utf-8") if dist.read_text("METADATA") else b""
        metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest() if metadata_bytes else None
    except Exception:
        pip_dlc = None
        metadata_sha256 = None

    # version.py SHA256 — independent proof of the runtime-constant-lag claim
    try:
        version_py = Path(deeplabcut.__file__).parent / "version.py"
        version_py_sha256 = hashlib.sha256(version_py.read_bytes()).hexdigest()
    except Exception:
        version_py_sha256 = None

    return {
        "python_version": sys.version.split()[0],
        "deeplabcut_pip_dist_info_version": pip_dlc,
        "deeplabcut_runtime_version_constant": deeplabcut.__version__,
        "deeplabcut_dist_info_metadata_sha256": metadata_sha256,
        "deeplabcut_version_py_sha256": version_py_sha256,
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": (torch.version.cuda if torch.cuda.is_available() else None),
        "torch_cuda_arch_list": (
            torch.cuda.get_arch_list() if torch.cuda.is_available() else None
        ),
        "torch_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "torch_cudnn_version": (
            torch.backends.cudnn.version() if torch.cuda.is_available() else None
        ),
        "torch_cudnn_enabled": (
            torch.backends.cudnn.enabled if torch.cuda.is_available() else None
        ),
    }
