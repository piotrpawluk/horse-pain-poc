# Qwen2.5-VL on RHpE — Experimental Spec for Claude Code

*Saved as `docs/qwen-experiment-spec.md` on branch `experiment/qwen-mlx`. Self-contained spec — agent should not need to ask clarifying questions before starting.*

---

## 1. Context

### 1.1 What this repo is
`horse-pain-poc` — methodology-first PoC for automated RHpE (Ridden Horse Pain Ethogram) scoring. Phase 0–1.5 is complete. Read `README.md`, `GATE.md`, and `docs/lessons_learned.md` before doing anything else; they contain hard-won decisions that should not be re-relitigated by this experiment.

Key established facts the agent must respect:

- **V-JEPA-2 + linear probe is the spine.** LOSO AUC 0.875 on Read My Ears ear-movement, replicating Alves et al. CVPR W'25 under source-aware split. Nothing in this experiment displaces V-JEPA-2 + LR.
- **The 53-clip DIY anchor dataset is disqualified for training** due to session leakage (Lesson 1, iter 6.5). Do not use it as labels.
- **Field data collection follows `docs/recording-protocol.md`.** Out of scope for this spec.

### 1.2 Prior MLLM testing on this task
We tested Gemini 2.5 Pro and Gemini 3.1 Pro Preview on a 36-clip stratified subset of the Read My Ears test split, at fps=10, with three prompts and at Google's officially recommended parameters. Three robust failure modes were identified (Lesson 14):

1. **Refusal-bias collapse.** Gemini 3.1 Pro classifies 35/36 clips as "background" regardless of true label, even with system instruction + `thinking_level=low` + `temperature=1.0`.
2. **Cross-rep instability.** 0/10 probed clips produce consistent motion/still descriptions across 5 reps at `temperature=1.0`.
3. **Perception/classification decoupling.** Description-mode reports motion in 3–5/5 reps on action clips while classification-mode on the same clips outputs "background."

Existing artifacts (locate them in the repo before starting):

- The 36-clip stratified manifest used by all Gemini probes
- The verbatim text of Prompt A (generic) and Prompt C (best-practice with system instruction)
- Output JSONLs for Gemini 2.5 and 3.1 runs under `outputs/`
- `gemini_audit.py` or equivalent script with the call structure

### 1.3 The hypothesis under test
Qwen2.5-VL differs from Gemini in ways that could plausibly avoid the failure modes:

- **Open-weight, self-hostable** → no API conservatism / RLHF drift outside our control
- **Different post-training regime** (Alibaba, not Google) → likely different bias profile on classification commitment
- **Different vision encoder architecture** → different perceptual ceiling
- **Different training corpus** → different exposure to relevant visual domains

If Qwen reproduces all three Gemini failure modes, Lesson 14 generalizes from "Gemini family" to "MLLM class" and the closed-source-MLLM track of the project closes for good. If Qwen behaves materially differently, open-weight MLLMs may have a role and a follow-up at 32B is warranted.

### 1.4 What we are NOT doing
- **Not writing a paper.** Tool selection only. The bar is "does Qwen belong in the pipeline?" not "is this publishable?"
- **Not trying to beat V-JEPA-2 + LR's LOSO 0.875.** V-JEPA-2 + LR is the spine regardless.
- **Not scope-creeping** into InternVL, MiniCPM-V, GLM-V, or other models in this run.
- **Not iterating prompts beyond the 3 already tested on Gemini.** This is replication, not Qwen-specific optimization.
- **Not testing more than 2 model sizes** (7B baseline; 32B only if 7B shows signal).

---

## 2. Setup

### 2.1 Hardware
M2 Max MacBook Pro, 96 GB unified memory, macOS Tahoe 26.3.1. Apple Silicon native is the target deployment surface.

### 2.2 Branch and dependency strategy
- Create branch `experiment/qwen-mlx` from current `main`
- Add `[project.optional-dependencies]` group `mac` to `pyproject.toml` containing `mlx-vlm` and any required transitive deps (pin to currently-released versions at branch creation; document the exact pins in the PR)
- **Do not add MLX deps to the main `[project.dependencies]` group** — `mlx` is Apple-Silicon-only and would break the Linux/Colab path in `setup.sh` and `notebooks/99_colab_fallback.ipynb`
- Update `setup.sh` to detect Apple Silicon (`uname -sm | grep -q "Darwin arm64"`) and prompt to install the `mac` extra; skip on other platforms with a printed note

### 2.3 Models
- **Primary:** `mlx-community/Qwen2.5-VL-7B-Instruct` (fp16, ~14 GB resident)
- **Conditional follow-up:** `mlx-community/Qwen2.5-VL-32B-Instruct-4bit` (~16–20 GB resident at int4)
- **Skip 72B** — marginal even at int4 on 96 GB and not worth the latency for a pilot

### 2.4 MLX → MPS fallback
If `mlx-vlm` cannot load Qwen2.5-VL-7B cleanly within 30 minutes of debugging (model loading, processor errors, video frame handling, etc.), fall back to PyTorch MPS via the official `Qwen/Qwen2.5-VL-7B-Instruct` HuggingFace weights using the standard `transformers` loader. Pin `transformers` and `tokenizers` per the official Qwen2.5-VL `requirements_web_demo.txt` rather than letting pip resolve. Document which path was used in the notebook header.

### 2.5 Dataset
- Same 36-clip stratified subset used for the Gemini runs. Locate the manifest file in the repo and reuse it verbatim — do not re-stratify.
- Frame extraction at fps=10, identical preprocessing to the Gemini runs. Reuse the existing extraction code if present; do not re-implement.
- Qwen2.5-VL takes a list of images as video input natively — extract frames as PNG/JPEG and pass as image sequence. This is the documented Qwen pattern, not a workaround.

### 2.6 Notebook
- Create `notebooks/05_qwen_mlx_video.ipynb`
- Follow the structural pattern of `notebooks/02_vjepa2_zeroshot.ipynb` (setup → load → smoke test → run → analysis → decision note)
- Header cell: state which inference backend was used (MLX or MPS fallback), exact model identifier, branch name, date

---

## 3. Procedure

### Phase 1 — Environment setup (~30 min)
1. Create branch `experiment/qwen-mlx` from `main`
2. Add `mac` optional dep group to `pyproject.toml`
3. Update `setup.sh` with Apple Silicon detection and mac-extra install path
4. Verify `mlx-vlm` imports and Qwen2.5-VL-7B downloads successfully (~14 GB, may take 10+ min on first run)
5. Smoke test: classify ONE clip from the test set with prompt A. Confirm structured output (classification + reasoning + token count). If smoke test fails, do not proceed — debug or fall back to MPS.

### Phase 2 — Replicate the 3-probe Gemini protocol on Qwen 7B (~1.5h)

For each probe, log the same fields the Gemini runs logged so the JSONL formats are directly diff-able: `clip_id, true_label, model, prompt_id, temperature, classification, reasoning_text, token_counts, finish_reason, timestamp, wall_clock_seconds`.

**Probe 1 — Classification, prompt A (generic), `temperature=0`:**
- Use the EXACT prompt A text from the Gemini runs (locate and copy verbatim — do not paraphrase)
- Single pass per clip
- Output: `outputs/qwen25vl_7b_promptA.jsonl`

**Probe 2 — Classification, prompt C (best-practice), Qwen-recommended params:**
- Use the EXACT prompt C text from the Gemini 3.1 runs
- Apply parameters from Qwen2.5-VL's official model card / generation config (this gives Qwen its fair shot — do not carry over Gemini's `thinking_level` or other Gemini-specific knobs)
- Output: `outputs/qwen25vl_7b_promptC.jsonl`

**Probe 3 — Description-mode probe, 5 reps:**
- Use the SAME 10 clips Gemini was probed on (locate them — they are a subset of the 36)
- Prompt: "Describe what you observe about the horse's ears in this clip in 2–3 sentences." No classification framing. No mention of "action" or "background."
- 5 reps per clip at `temperature=0.5` or higher (use Qwen's recommended sampling temperature for description tasks)
- Output: `outputs/qwen25vl_7b_description_probe.jsonl`

### Phase 3 — Comparative analysis (~1h)

Create `outputs/qwen_vs_gemini_comparison.md` containing:

- **Headline metrics table** with Gemini 2.5, Gemini 3.1, Qwen 7B side-by-side on:
  - Agreement with ground truth on 36 clips (prompt A and prompt C separately)
  - Background-prediction rate (the refusal-bias signature — Gemini 3.1 was 35/36)
  - Description-mode cross-rep stability (fraction of clips with consistent motion/still description across 5 reps)
  - Perception/classification decoupling cases (description says motion, classification says background, on the same clip)
- **Per-source breakdown** (S1, S3, S4, ..., S12) of agreement — does Qwen show the inverse-LOSO pattern from the Gemini 2.5 fps=10 run?
- **Qualitative comparison of reasoning text** on 3–5 representative disagreement clips — paste excerpts side by side
- **Verdict** in 3–5 sentences against the pre-registered outcomes (§4 below)

### Phase 4 (conditional) — 32B follow-up

**Only run if Qwen 7B shows materially different behavior than Gemini on at least one of the three failure-mode metrics** (see §4). If 7B reproduces all three Gemini failure modes, do NOT run 32B. Document the gating decision in the analysis doc.

If 32B runs: same three probes, same output format, save to `outputs/qwen25vl_32b_*.jsonl`. Add a row to the headline table.

### Phase 5 — Document (~30 min)

1. Add **Lesson 15** to `docs/lessons_learned.md` with the headline finding (3–5 sentences, following the Lesson 1–14 format)
2. Update the **Status table in `README.md`** with a new row for Qwen2.5-VL
3. Update or rename `docs/gemini-integration.md` to `docs/mllm-integration.md` if the conclusion generalizes (i.e. Qwen reproduced the Gemini failure modes)
4. Commit everything to branch with descriptive commit messages
5. Open PR with the analysis doc summary as the PR description

---

## 4. Pre-registration: outcome interpretation

Before running, the agent commits to the following interpretation table. **Do not retrofit interpretation after seeing results.**

| Outcome | Interpretation | Next step |
|---|---|---|
| Qwen 7B exhibits all 3 failure modes (refusal-bias collapse, cross-rep instability, perception/classification decoupling) | **Lesson 14 generalizes to "MLLM class" rather than "Gemini family."** Strong negative result for the entire MLLM-as-classifier track. | Do NOT run 32B. Update Lesson 14 framing. Close the closed/open MLLM track. Merge branch. |
| Qwen 7B shows materially cleaner classification: agreement ≥0.70 AND background-prediction rate <80% AND no >3pp decoupling on the 10-clip probe | **Open-weight MLLM may have a role in the pipeline.** Worth deeper investigation. | Run 32B. Compare 7B vs 32B. Investigate where Qwen's behavior diverges from Gemini's. |
| Mixed: e.g., no refusal collapse but high instability, or one metric clean and others not | **Partial finding.** Worth documenting but not pivoting on. | Run 32B only if it would specifically resolve the partial result (e.g., 7B is on the edge of a threshold). Otherwise stop. |
| Qwen 7B fails to run, produces garbled output, or hits an infrastructure wall | **Infrastructure issue, not a scientific finding.** | Debug for up to 1 additional hour, then either re-run or document the failure and stop. Do NOT draw conclusions about Qwen capability from infrastructure failure. |

---

## 5. Scope discipline

- **Do NOT** add prompt engineering iterations beyond the 3 probes from the Gemini protocol. The goal is controlled replication.
- **Do NOT** test other models, frameworks, or architectures in this run — even if they look promising mid-experiment.
- **Do NOT** rewrite shared infrastructure (data loaders, frame extractors, JSONL schemas). Reuse what the Gemini run built.
- **Do NOT** merge to main unless results are coherent and useful. If 7B fails outright at infrastructure level, leave the branch alone for now.
- **Do NOT** spend more than **4 hours of wall-clock time** total on this. If something blocks past 4 hours, stop and surface the blocker.
- **Do NOT** modify `lessons_learned.md` lessons 1–14 — only append Lesson 15.

---

## 6. Deliverables (PR contents)

At PR time, branch `experiment/qwen-mlx` should contain:

1. Updated `pyproject.toml` with `mac` extra
2. Updated `setup.sh` with Apple Silicon detection
3. New `notebooks/05_qwen_mlx_video.ipynb`, fully executable end-to-end
4. Output JSONLs for each probe under `outputs/`
5. `outputs/qwen_vs_gemini_comparison.md` with the full analysis
6. New Lesson 15 appended to `docs/lessons_learned.md`
7. Updated Status table row in `README.md`
8. (Conditional) `docs/mllm-integration.md` if rename triggered
9. PR description summarizing the headline finding in 3–5 sentences and stating which row of the §4 outcome table was hit

---

## 7. Quick start for the agent

```
1. Read README.md, GATE.md, docs/lessons_learned.md (especially Lesson 14), docs/recording-protocol.md
2. Locate Gemini artifacts: 36-clip manifest, prompt A and prompt C verbatim text, output JSONLs, gemini_audit.py
3. Create branch experiment/qwen-mlx
4. Wire pyproject.toml mac extra + setup.sh detection
5. Smoke test mlx-vlm + Qwen2.5-VL-7B-Instruct on one clip
6. Run probes 1, 2, 3 on 7B
7. Generate comparison doc against pre-registered outcomes
8. Conditionally run 32B
9. Write Lesson 15, update README, open PR
```

If anything in this spec conflicts with what's actually in the repo, surface the conflict before proceeding rather than guessing.

---

## 8. Implementation note (added at save time)

The 36-clip stratified subset and 10-clip description-probe subset are NOT saved as standalone manifest files in the repo. They are produced dynamically by `tools/gemini_audit.py` (per-source random selection for the 36; disagreement-based filtering for the 10). To guarantee identical clips across the Gemini → Qwen comparison, the Qwen runner reads clip paths directly from the existing Gemini output JSONLs in `outputs/`:

- 36-clip subset → union of clip paths across `outputs/gemini_audit_results_*_promptA.jsonl` (sanity-check |union| == 36 and identical across the 5 promptA/B/C output files).
- 10-clip probe subset → clip paths in `outputs/gemini_audit_probe_gemini-3.1-pro-preview_temp1.0_thinkLOW.jsonl` (deduped from 5 reps × 10 clips).

This is binding — no re-stratification.
