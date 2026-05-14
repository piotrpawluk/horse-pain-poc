# `tools/cloud_dlc/` — Modal cloud DLC + V-JEPA-2 inference

Run the Phase 8b/10a inference stack (DLC SuperAnimal-Quadruped + V-JEPA-2 ViT-L) on Modal serverless GPUs (T4) instead of the local M2 Max. Built 2026-05-13 to unblock Phase 10a-full (89-clip portrait DLC is ~10× slower on local CPU than on CUDA, blocking timely thesis work). Free tier ($30/mo) covers all expected project usage.

## Files

| File | Purpose |
|---|---|
| `dlc_inference.py` | Vanilla portable DLC inference function + `runtime_environment()` version capture. No Modal deps — provider-portable. |
| `app.py` | Modal app (`phase10a-dlc`). Image pinned to local versions (DLC rc14, torch 2.11, transformers 5.8); `run_dlc_remote` + `run_vjepa2_remote` functions with CUDA determinism flags + warning capture. |
| `validate_keypoints.py` | DLC h5 diff validator. 4-level MultiIndex aware. Locked 0.5 px max-Δ threshold for keypoint identity. Self-test: identical h5 vs itself → 0 px. |
| `smoke_test.py` | DLC-only smoke test (IMG_1050 Prudnik clip). Compares Modal CUDA h5 against local CPU baseline. Includes `_version_identity_check()` (HASH_MATCH / VERSION_MATCH_HASH_DIFFER / VERSION_MISMATCH). |
| `smoke_test_full_pipeline.py` | End-to-end smoke test (action_S1.mp4_0_.mp4 Phase 8b RME clip). Three stages: V-JEPA-2 features (cosine ≥ 0.999), classifier score (\|Δ\| ≤ 0.01), binary classification (flip = 0). Gates Phase 8b-CUDA retrain. |
| `FAILURE_PLAYBOOK.md` | Decision trees for the three realistic smoke-test failure modes (small drift, large drift, structural NaN mismatch). |

## Quick start

```bash
# One-time auth (opens browser, ~30s)
.venv/bin/python -m modal setup

# DLC-only smoke test (~5 min wallclock first run with image build; ~30s warm)
.venv/bin/python -m modal run tools/cloud_dlc/smoke_test.py

# Full-pipeline smoke test (~10 min first run; ~1 min warm) — gates Phase 8b-CUDA retrain
.venv/bin/python -m modal run tools/cloud_dlc/smoke_test_full_pipeline.py
```

Smoke test result JSONs land at `outputs/cloud_dlc_smoke_test_result.json` and `outputs/cloud_dlc_smoke_test_full_pipeline_result.json`. Each captures full version manifest (pip dist-info versions, METADATA SHA256s, cuDNN version, GPU device name) for the transition-commit audit doc.

## Cost expectation

Inside Modal $30/mo free tier for all expected project usage. Empirical:

- Smoke test (one clip): ~$0.50 first run with image rebuild, ~$0.05 warm
- Phase 8b-CUDA retrain (283 clips): ~$1.50–2.50, ~2 hours wallclock
- Phase 10a-prelim CUDA re-run (89 clips): ~$0.30, ~25 min wallclock
- Phase 10a-full + Phase 11+ K=2 expansion: well inside free tier

Set a billing alert at $10/mo in Modal dashboard as bug-detection insurance (not budget management — actual usage stays at $0).

## When smoke test fails

`outputs/cloud_dlc_smoke_test_*.json` carries the structured failure reason. Read `FAILURE_PLAYBOOK.md` for decision trees — three failure modes (`(0.5, 2.0]` px drift, `> 2.0` px drift, `STRUCTURAL_NAN_MISMATCH`) each with auto-decide steps and escalation triggers.

## Methodology constraints (load-bearing — do not change without audit-doc update)

- **DLC version**: pip artifact `deeplabcut==3.0.0rc14`. Runtime `__version__` constant displays `"3.0.0rc13"` — this is a known stale-display bug in DLC's rc14 release process; pip dist-info is the source of truth. See `docs/lessons_learned.md` Lesson 22.
- **numpy / pandas constraint override**: Modal image uses `uv pip install --override` to install numpy 2.4.4 and pandas 3.0.2 past DLC rc14's declared `numpy<2, pandas<3` constraints. Local install (running phase10a-prelim CPU process) demonstrates empirical compatibility at these versions; the declared constraints are conservative. See `outputs/cloud_dlc_audit_footnotes.md` Footnote 7.
- **CUDA determinism**: `torch.use_deterministic_algorithms(True, warn_only=True)` with warning capture. Empty `determinism_warnings` list in result JSON → bit-exact reproducibility; non-empty list names the specific non-deterministic ops for audit footnoting. See `outputs/cloud_dlc_audit_footnotes.md` Footnote 9.
- **Pre-release HF deps**: `safetensors==0.8.0rc0`, `tokenizers==0.23.0rc0` are pinned to match local install (matches binary identity, accepts RC-version risk). See `outputs/cloud_dlc_audit_footnotes.md` Footnote 8.

## Transition-commit audit doc

`outputs/cloud_dlc_audit_footnotes.md` carries 9 pre-drafted footnotes ready to fold into the Phase 10a-prelim or Phase 10a-full audit doc when the compute-substrate transition is committed. Footnotes 1 + 2 are mandatory inline; 3–9 go in methodology limitations / appendix. The audit doc author opens the smoke-test result JSON, copies values into the `[PASTE FROM …]` placeholders, and the methodology preservation chain is complete.

Lesson 23 (compute-substrate transitions extend the audit chain rather than replace it) gets written at the transition commit when empirical content exists, not before.

## Provider-portability

`dlc_inference.py` is Modal-free by design. To swap providers (RunPod, Vast.ai, Lambda Labs, etc.) without touching the inference logic:

1. Write a new wrapper in place of `app.py` calling `run_dlc_inference()` directly
2. Reuse `validate_keypoints.py` + `smoke_test*.py` unchanged
3. Re-run smoke tests against the new substrate; expect the same identity-check / tolerance-band agreement

The architecture choice (vanilla function in one file, Modal decoration in a separate thin wrapper) is the portability insurance. ~30 min of work to swap, no methodology change required.
