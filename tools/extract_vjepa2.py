#!/usr/bin/env python3
"""V-JEPA-2 ViT-L embedding extractor (lifted from notebooks/02_vjepa2_zeroshot.ipynb).

Forward-pass parity with the cached outputs/vjepa2_embeddings.npz is verified
by --parity test before extraction: re-extract one cached clip, assert cosine
similarity ≥ 0.999 against the cached embedding. Phase 2 of Track B cannot
proceed without parity — silent regression vs ear baseline 0.8746 would
poison every downstream LOSO comparison.

Output schema matches the cached npz:
  embs       (N, 1024)  V-JEPA-2 ViT-L mean-pooled patch tokens
  labels     (N,)       1 if filename starts with 'action_', 0 if 'background_', -1 otherwise
  splits     (N,)       informational tag (used for downstream LOSO partitioning)
  filenames  (N,)       basename per row

Usage:
  .venv/bin/python tools/extract_vjepa2.py \\
      --clips-dir outputs/eye_crops \\
      --out outputs/vjepa2_embeddings_eye.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoVideoProcessor, VJEPA2Model

POC_DIR = Path(__file__).resolve().parent.parent
MODEL_ID = "facebook/vjepa2-vitl-fpc16-256-ssv2"
NUM_FRAMES = 16
DEFAULT_CACHED = POC_DIR / "outputs" / "vjepa2_embeddings.npz"
DEFAULT_PARITY_CLIPS = (
    POC_DIR / "vendor" / "ReadMyEars_Dataset" / "data" / "videos"
)
PARITY_THRESHOLD = 0.999


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def read_clip_frames(clip_path: Path, num_frames: int = NUM_FRAMES):
    """N evenly-spaced frames from clip as (T, H, W, C) RGB uint8.
    Mirrors notebooks/02_vjepa2_zeroshot.ipynb cell 6 verbatim for parity."""
    cap = cv2.VideoCapture(str(clip_path))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_total < 1:
        cap.release()
        return None
    indices = np.linspace(0, n_total - 1, num_frames).astype(int)
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, f = cap.read()
        if not ok:
            f = (frames[-1] if frames
                 else np.zeros((256, 256, 3), dtype=np.uint8))
            frames.append(f)
            continue
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


@torch.no_grad()
def extract_embedding(clip_path: Path, model, processor, device: str,
                      num_frames: int = NUM_FRAMES):
    frames = read_clip_frames(clip_path, num_frames=num_frames)
    if frames is None:
        return None
    inputs = processor(videos=list(frames), return_tensors="pt").to(device)
    outputs = model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
    return emb.cpu().float().numpy()


def parity_test(model, processor, device: str, cached_path: Path,
                clips_dir: Path) -> bool:
    """Genuine re-extraction parity: load cached embedding from npz, then
    invoke the lifted CLI forward pass (extract_embedding) on the SAME source
    clip, then compare. NOT a cache-to-cache tautology — the only thing taken
    from the cache is the embedding to compare AGAINST; the comparison
    embedding is freshly computed by reading frames, running them through the
    processor, and forwarding through the model in eval+no_grad mode.
    Bit-exact match (cos = 1.0, ‖Δ‖ = 0) means the lifted forward pass is
    deterministic and reproduces the notebook output exactly — the strongest
    parity signal."""
    cached = np.load(cached_path, allow_pickle=True)
    filenames = cached["filenames"]
    embs = cached["embs"]
    print(f"[parity] cached {cached_path.name}: {embs.shape}", flush=True)

    for fn, cached_emb in zip(filenames, embs):
        clip_path = clips_dir / str(fn)
        if not clip_path.exists():
            continue
        print(f"[parity] re-extracting {fn} via lifted CLI forward pass...",
              flush=True)
        t0 = time.time()
        fresh = extract_embedding(clip_path, model, processor, device)
        if fresh is None:
            continue
        dt = time.time() - t0
        cached_norm = np.linalg.norm(cached_emb)
        fresh_norm = np.linalg.norm(fresh)
        if cached_norm == 0 or fresh_norm == 0:
            print(f"[parity] zero-norm — skipping {fn}", flush=True)
            continue
        cos = float(np.dot(cached_emb, fresh) / (cached_norm * fresh_norm))
        delta = float(np.linalg.norm(cached_emb - fresh))
        print(f"[parity] {fn}: cos={cos:.6f}, ‖Δ‖={delta:.4f}, "
              f"latency={dt:.1f}s", flush=True)
        if cos >= PARITY_THRESHOLD:
            print(f"[parity] PASS: cos={cos:.6f} ≥ {PARITY_THRESHOLD}",
                  flush=True)
            return True
        print(f"[parity] FAIL: cos={cos:.6f} < {PARITY_THRESHOLD}", flush=True)
        return False

    print(f"[parity] no cached clips found in {clips_dir}", flush=True)
    return False


def derive_label(filename: str) -> int:
    if filename.startswith("action"):
        return 1
    if filename.startswith("background"):
        return 0
    return -1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parity-against", type=Path, default=DEFAULT_CACHED)
    ap.add_argument("--parity-clips-dir", type=Path,
                    default=DEFAULT_PARITY_CLIPS)
    ap.add_argument("--skip-parity", action="store_true",
                    help="Skip parity check (NOT recommended; logged in npz)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="Filenames to exclude from extraction (repeatable). "
                         "Used to honor the manual eye-visible exclusion list "
                         "from outputs/eye_crops_annotations.md.")
    ap.add_argument("--split-tag", default="eye_crops")
    args = ap.parse_args()

    device = pick_device()
    print(f"[v-jepa2] device={device}", flush=True)
    print(f"[v-jepa2] model={MODEL_ID}", flush=True)
    t0 = time.time()
    processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
    model = VJEPA2Model.from_pretrained(MODEL_ID).to(device).eval()
    print(f"[v-jepa2] loaded in {time.time()-t0:.1f}s; "
          f"hidden={model.config.hidden_size}", flush=True)

    if not args.skip_parity:
        if not args.parity_against.exists():
            print(f"[parity] cached file missing: {args.parity_against}",
                  flush=True)
            sys.exit(2)
        ok = parity_test(
            model, processor, device,
            args.parity_against, args.parity_clips_dir,
        )
        if not ok:
            print("[v-jepa2] HALT: parity failed; "
                  "extraction would be incomparable with cached ear baseline",
                  flush=True)
            sys.exit(2)

    excluded = set(args.exclude or [])
    clips_all = sorted(p for p in args.clips_dir.glob("*.mp4"))
    clips = [p for p in clips_all if p.name not in excluded]
    n_excluded = len(clips_all) - len(clips)
    print(f"[v-jepa2] {len(clips_all)} clips in {args.clips_dir} "
          f"({n_excluded} excluded by --exclude); extracting on {len(clips)}",
          flush=True)
    if excluded:
        for fn in sorted(excluded):
            present = "skipped (found)" if (args.clips_dir / fn).exists() else "skipped (not present)"
            print(f"  --exclude {fn}: {present}", flush=True)

    embs: list[np.ndarray] = []
    labels: list[int] = []
    splits: list[str] = []
    filenames: list[str] = []
    t_run = time.time()
    for i, clip in enumerate(clips, 1):
        try:
            emb = extract_embedding(clip, model, processor, device)
            if emb is None:
                print(f"  [{i:3d}/{len(clips)}] {clip.name} SKIP (no frames)",
                      flush=True)
                continue
            embs.append(emb)
            labels.append(derive_label(clip.name))
            splits.append(args.split_tag)
            filenames.append(clip.name)
            print(f"  [{i:3d}/{len(clips)}] {clip.name:38} → {emb.shape}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i:3d}/{len(clips)}] {clip.name} ERR: "
                  f"{type(e).__name__}: {e}", flush=True)

    if not embs:
        print("[v-jepa2] no embeddings extracted", flush=True)
        sys.exit(1)

    embs_arr = np.stack(embs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        embs=embs_arr,
        labels=np.array(labels),
        splits=np.array(splits),
        filenames=np.array(filenames),
        parity_passed=np.array(not args.skip_parity),
        parity_threshold=np.array(PARITY_THRESHOLD),
        model_id=np.array(MODEL_ID),
    )
    out_resolved = args.out.resolve()
    try:
        out_pretty = out_resolved.relative_to(POC_DIR)
    except ValueError:
        out_pretty = out_resolved
    print(f"\n[v-jepa2] saved: {out_pretty} "
          f"({embs_arr.shape}, {embs_arr.nbytes / 1e6:.1f} MB) "
          f"in {time.time()-t_run:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
