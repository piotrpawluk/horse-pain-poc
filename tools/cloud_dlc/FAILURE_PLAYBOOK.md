# Cloud DLC Smoke-Test Failure Playbook

`tools/cloud_dlc/smoke_test.py` exits non-zero with structured JSON if the
Modal CUDA path diverges from the local CPU baseline beyond the locked
0.5 px threshold. This document covers the realistic failure modes and
the decision tree for each.

The smoke-test result lands at `outputs/cloud_dlc_smoke_test_result.json`.
Inspect `validation.failure_reason` + `validation.max_delta_xy_px` first.

---

## Failure mode A — `max |Δ|_xy ∈ (0.5, 2.0] px` (small numerical drift)

**Most likely cause:** non-deterministic cuDNN kernels. PyTorch's
default cuDNN backend can pick different conv kernels across runs
even on identical input, producing sub-pixel keypoint drift that
sometimes peaks above 0.5 px.

**Decision tree:**

1. **Rerun the smoke test once.** If `max |Δ|_xy` lands ≤ 0.5 px on
   a second run, the threshold is fine — first run hit a non-determinism
   spike. Document the first-run delta in audit_extras and proceed.
2. **If second run also exceeds 0.5 px**: enable cuDNN determinism in
   the Modal image (`torch.backends.cudnn.deterministic = True`,
   `torch.backends.cudnn.benchmark = False`). Rerun. Expected to bring
   delta into 0.0–0.2 px range at a small throughput cost (~10–20% slower).
3. **If still > 0.5 px after determinism flag**: switch GPU type T4 → A10G.
   Larger GPU, fewer kernel variants, less non-determinism. Per-hour cost
   doubles (~$1.10) but still inside the free tier for this project's volume.
4. **If A10G also > 0.5 px**: relax threshold to 1.0 px with explicit
   audit-doc justification: *"DLC SuperAnimal-Quadruped + Faster R-CNN
   detector exhibits ≤1.0 px CUDA-vs-CPU drift on this Modal worker pool.
   Downstream classification at boundary cases tracked separately in
   audit_extras (boundary_case_flip_count)."* This is a methodological
   compromise, not an engineering failure — flag it explicitly.

**Do NOT do:** silently raise the threshold without audit justification.
The 0.5 px number was chosen because it preserves the methodology argument
that DLC outputs are effectively device-stable. Raising it without
documentation breaks that chain.

---

## Failure mode B — `max |Δ|_xy > 2.0 px` (genuine methodology divergence)

**Most likely cause:** something substantive differs between local and
Modal — different DLC version, different model weights, different
preprocessing pipeline, or a wrong device routing somewhere.

**Decision tree:**

1. **Check `version_identity_check.status` in the smoke-test JSON.**
   - `VERSION_MISMATCH`: local and Modal are on different DLC releases.
     Update `tools/cloud_dlc/app.py:DLC_IMAGE` pip pin to match local.
     Rebuild Modal image (next `modal run` triggers rebuild on image change).
   - `VERSION_MATCH_HASH_DIFFER`: same pip version but different METADATA
     SHA256. Possible — wheel built on different platform tag. Inspect
     `local_runtime_env` and `cloud_runtime_env` field-by-field for any
     mismatch (Python version, torch version, torchvision version).
   - `HASH_MATCH`: dist-info bytes identical. Then the divergence is NOT
     in the package itself — proceed to step 2.
2. **Check that SuperAnimal-Quadruped weights match.** DLC downloads
   weights on first use; if Modal pulls a different snapshot than local
   has cached, outputs diverge. Inspect Modal logs for the weights download
   URL + hash. Compare against local: `find ~/.cache -name "*hrnet_w32*"`.
3. **Check input preprocessing.** Both code paths (local and Modal) must
   call `deeplabcut.video_inference_superanimal()` with identical kwargs.
   The `device` arg differs (`"auto"` local vs `"cuda"` Modal) but everything
   else is locked in `dlc_inference.py:run_dlc_inference()` and the inlined
   Modal copy in `app.py:run_dlc_remote()`. Diff the two by hand to verify.
4. **If preprocessing matches and weights match and versions match**: the
   divergence is in the model's GPU kernel implementation vs CPU
   implementation. This is genuinely rare — surface to user before
   any production run; methodology preservation argument needs a more
   robust foundation than tolerance-band agreement.

**Do NOT do:** ship a production run with > 2 px divergence. The
downstream pipeline (ear-bbox crop → V-JEPA-2 → classifier) is sensitive
to ear-keypoint location at the integer-pixel level; 2 px crop shift
changes which pixels go into the encoder.

---

## Failure mode C — `STRUCTURAL_NAN_MISMATCH` (detector behavior divergence)

**Most likely cause:** Faster R-CNN detector assigned individuals
differently between CPU and CUDA runs. SuperAnimal-Quadruped outputs
up to 10 animals per frame; some are NaN when no detection is made.
If local CPU detects 2 animals in frame F and Modal CUDA detects 3
(or assigns them to different `animal0..animal9` slots), the NaN
patterns differ even when the actually-detected horse's coordinates
match.

**Decision tree:**

1. **Inspect the NaN pattern per `animal_*` index.** Read both h5 files,
   group by individual, count NaN frames per animal slot. If local has
   animal0=populated, animal1=NaN, animal2=NaN... and cloud has
   animal0=populated, animal1=populated, animal2=NaN, the detector saw
   a phantom second horse on CUDA that CPU missed (or vice versa).
2. **If only `animal0` differs in NaN pattern**: that's the foreground
   horse — methodology IS broken. Same as Failure mode B step 4.
3. **If only `animal1..animal9` differ**: the foreground horse
   keypoints agree, the detector just disagreed on phantom detections.
   For Prudnik foreground-only labeling, only `animal0` is downstream-
   relevant. Filter the comparison to animal0 columns and rerun the
   validator on the filtered subset; if `max |Δ|_xy ≤ 0.5 px` on the
   animal0 subset, the methodology is preserved for the actual use case.
   Document the multi-individual divergence as audit footnote.
4. **If both animal0 AND phantom slots differ**: treat as Failure mode B.

---

## When to escalate to user instead of auto-deciding

Auto-decide (CC proceeds without asking) is appropriate for:

- Failure mode A step 1 (rerun once)
- Failure mode A step 2 (enable determinism flag)
- Failure mode C step 3 (animal0-only filtered comparison)

Escalate to user for:

- Any step that involves relaxing the 0.5 px threshold
- Any step that involves switching GPU type (cost change)
- Any Failure mode B outcome
- Any Failure mode C escalation beyond animal0 filtering
- Any audit-doc-load-bearing decision (everything above qualifies)

---

## What "PASS" looks like

`max |Δ|_xy ≤ 0.5 px`, `n_nan_mismatches = 0`, `version_identity_check.status =
HASH_MATCH`. Result JSON written to `outputs/cloud_dlc_smoke_test_result.json`,
exit code 0. Safe to proceed to building `tools/cloud_dlc/run_phase10a.py`
for production GPU runs.

Expected from the MPS-probe-vs-CPU 0.0 px result: actual CUDA-vs-CPU
delta should land at 0.0–0.1 px. The threshold is generously conservative.
