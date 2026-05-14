# Cloud DLC audit footnotes — pre-drafted for Phase 10a audit doc

Drafted 2026-05-13 during Phase 10a-prelim cloud DLC scaffolding, ahead of
the actual production GPU runs, to ensure the audit-doc author has the
correct framing ready when the time comes (instead of reconstructing it
under time pressure).

These are **drop-in snippets** for the eventual Phase 10a audit doc's
"Methodology & limitations" or "Compute & reproducibility" section.

---

## Footnote 1 — DLC version identity (load-bearing)

> DLC inference performed on the pip artifact `deeplabcut==3.0.0rc14`,
> verified by dist-info `METADATA` SHA256 = `273ec59ddd47b004c16385252a638a034579ea2cd180fdbe553915bbb1a4c02e`
> (captured 2026-05-13 from local install at `.venv/lib/python3.11/site-packages/deeplabcut-3.0.0rc14.dist-info/METADATA`).
> The runtime `deeplabcut.__version__` constant displays `"3.0.0rc13"` due to
> an incomplete version bump in the DLC rc14 release process (see DLC GitHub
> PRs #3096 + #3109 — two separate version-update commits, one of which
> did not update `version.py`). The pip dist-info `Version:` field is
> authoritative; the in-code constant is a stale display. Phase 8b ran
> on this same install, so the methodology preservation chain is
> consistent across Phases 8b → 10a-prelim → 10a-full → Phase 11+.
> See `docs/lessons_learned.md` Lesson 22 for the generalizable rule.

## Footnote 2 — CPU vs CUDA tolerance-band agreement (load-bearing)

> Modal T4 CUDA DLC inference and local CPU inference agree at the
> sub-pixel level: smoke-test on reference clip IMG_1050 produced
> `max |Δ|_xy = [PASTE FROM smoke_test_result.json]` px, with the
> threshold for "methodology preserved" pre-locked at 0.5 px (per
> `tools/cloud_dlc/smoke_test.py:THRESHOLD_PX`). Downstream
> classifications in the Phase 8b pipeline (ear-bbox crop at integer-pixel
> resolution → V-JEPA-2 features → RidgeClassifier(α=1.0) → T-scaled
> sigmoid) are sensitive to keypoint location at the integer-pixel level;
> sub-pixel agreement implies bbox crops are identical (Python int
> truncation), V-JEPA-2 inputs are identical, decision_function scores
> are identical, and binary classifications at τ_ear = 0.8138 are identical.
> Boundary-case agreement (clips where `|score - τ_ear| < 0.05`) is
> tracked separately as `boundary_case_flip_count` in audit_extras (see
> Footnote 3); the count is expected to be 0 if methodology preservation
> holds.

## Footnote 3 — Boundary-case flip instrumentation (planned, not yet run)

> `tools/cloud_dlc/run_phase10a.py` (built post-smoke-test-pass) logs
> for every clip the absolute difference between local CPU and Modal CUDA
> decision_function scores plus the per-clip binary classification on
> both paths. Any clip where the binary classification flips is recorded
> with full provenance (`clip_id`, `cpu_score`, `cuda_score`,
> `cpu_label`, `cuda_label`, `delta_score`). The total flip count appears
> in `audit_extras.cpu_vs_cuda_boundary_flips` and is reported alongside
> the headline AUC. Expectation from the CPU-vs-MPS-probe 0.0 px result
> on IMG_1050: zero flips, providing empirical evidence that the
> tolerance-band methodology preserves the binary verdict structure.

## Footnote 4 — Modal worker fleet variance (limitation)

> Methodology preservation is verified by tolerance-band agreement on
> one representative Modal T4 worker via the smoke test, NOT by
> exhaustive bounding of variance across the entire Modal worker pool.
> Modal allocates workers from a fleet; individual T4 SKU revisions,
> host CUDA driver versions, and (in principle) cuDNN libraries may vary
> across invocations. PyTorch ships bundled cuDNN with its wheel
> (`torch==2.11.0` → cuDNN [PASTE FROM smoke_test_result.json
> cloud_runtime_env.torch_cudnn_version] across all workers), so the
> primary CUDA library is constant; only host driver may vary, which
> historically does not affect float32 inference output. Per-invocation
> cuDNN version + GPU device name + CUDA arch list are captured in
> `audit_extras.modal_per_invocation_env_capture` (see also Footnote 5).
> This is an honest known unknown; risk is bounded by the smoke-test
> threshold (≤0.5 px keypoint divergence under any worker reachable
> via free-tier T4 allocation) but not eliminated.

## Footnote 5 — Per-invocation environment capture

> Every `run_dlc_remote` invocation captures its worker's runtime
> environment (Python version, DLC version + METADATA SHA256, torch
> version, cuDNN version, CUDA arch list, GPU device name) and returns
> it alongside the h5 bytes. The full per-clip capture is preserved in
> `outputs/phase10a_run_modal_env.jsonl` for audit. If any field
> drifts across workers (e.g., one clip lands on a worker with a
> different cuDNN library), the drift is captured and reportable.

## Footnote 6 — apt layer composition

> Modal Docker image includes apt packages `ffmpeg`, `libgl1`,
> `libglib2.0-0`, `libxext6`, `libsm6`. ffmpeg is required for DLC's
> video I/O; libgl1 + libglib2.0-0 + libxext6 + libsm6 are
> belt-and-suspenders for matplotlib / cv2 transitive imports that
> occasionally try to load GUI libraries even in headless mode.
> opencv-python-headless is pinned over opencv-python to avoid GUI
> deps entirely; the apt layer is redundant defense.

## Footnote 7 — DLC rc14 inference-affecting changes vs rc13

> Phase 10a uses DLC pip artifact rc14, which contains the following
> inference-affecting changes vs rc13 (per GitHub release notes for
> v3.0.0rc14, published 2026-03-23):
>
> - PR #3105: disable `torch.autocast` by default in inference settings
> - PR #3154: filter low-confidence bbox detections from Faster R-CNN
> - PR #3078: fix RTMPose likelihood computation
> - PR #3110, #3117, #3121: PyTorch inference speed-ups (functional
>   equivalence preserved per upstream test suite)
> - PRs #3074, #3079, #3092: cleanup + override handling in `analyze_videos()`
>
> All Phase 8b RME work also ran on rc14 bytes (verified by install
> chronology), so the rc14-vs-rc13 distinction does not affect
> Phase-to-Phase methodology preservation.

## Footnote 8 — Pre-release dependency versions

> The local + Modal environment pins `safetensors==0.8.0rc0` and
> `tokenizers==0.23.0rc0` — both are HuggingFace release candidates, not
> stable releases. Pinning RC versions is methodologically correct (matches
> local for binary identity) but introduces a known dependency-stability
> caveat: upstream may issue stable releases (e.g., `0.8.0`, `0.23.0`)
> with subtle behavioral changes vs the RC. If a future re-installation
> picks up a stable version that differs in tokenizer normalization or
> safetensor loading order, V-JEPA-2 feature outputs could shift below
> the cosine threshold. The transformers METADATA SHA256 captured in
> `cloud_runtime_env.transformers_metadata_sha256` catches transformers
> version drift; analogous SHA256 capture for safetensors / tokenizers
> would close the gap fully. Not done in current scaffolding — flagged
> as Phase 11+ hardening (Lesson 22 extension).

## Footnote 9 — Determinism flag relaxation + warning capture

> `torch.use_deterministic_algorithms(True, warn_only=True)` is the
> DELIBERATE relaxation from strict mode (`warn_only=False`). Strict mode
> would raise immediately if any torch op without a deterministic CUDA
> implementation is encountered; pragmatic mode permits these ops but
> we capture the resulting warnings via `warnings.catch_warnings(record=True)`
> and serialize them into the smoke-test result JSON under
> `determinism_capture.determinism_warnings`. Audit interpretation:
>
> - **Empty `determinism_warnings` list** → V-JEPA-2 (or DLC) forward pass
>   hit no non-deterministic ops on CUDA. The relaxation is moot for this
>   workload; effectively equivalent to strict mode. Bit-exact reproducible
>   across Modal workers.
>
> - **Non-empty `determinism_warnings` list** → specific ops named in
>   the warnings are non-deterministic on CUDA. The audit doc must
>   footnote: (a) the specific op(s), (b) whether the inference output
>   downstream of those ops is bit-stable empirically (rerun twice; if
>   identical, the non-determinism is unobserved at the output level),
>   (c) the magnitude of any observed drift between runs. Sub-pixel
>   keypoint drift driven by non-deterministic ops is benign and falls
>   under the existing tolerance-band methodology argument.
>
> The flag choice (pragmatic with capture, not strict) preserves
> ability to ship the inference run while making any non-determinism
> visible in the audit chain rather than buried as a silent default.

---

## How to use this file

When Phase 10a audit doc draft begins (after smoke-test PASS + production
run completion):

1. Open `outputs/cloud_dlc_smoke_test_result.json` to fill in the
   `[PASTE FROM ...]` placeholders in footnotes 2 and 4.
2. Open `outputs/phase10a_run_modal_env.jsonl` (created by
   `run_phase10a.py` when built) to fill in any worker-fleet drift
   observations.
3. Footnote ordering in the audit doc is by load-bearingness:
   1 + 2 are mandatory inline; 3–7 go in the "Methodology limitations"
   subsection or appendix.

This file becomes stale once the audit doc lands; delete or archive
after that point.
