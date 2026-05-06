# Gemini integration — label-noise audit

Status: scaffolded on the `gemini-augmentation` branch. Run-ready once a `GEMINI_API_KEY` is provided.

## What it is

`tools/gemini_audit.py` runs **Gemini 2.5 Pro** over the 283 Read My Ears clips, asks it to classify each clip as `action` (ear movement) or `background` (no ear movement), and compares against the human label. Disagreements flag clips for re-review.

## Why this and not "replace V-JEPA-2 with Gemini"

Standard research synthesis (2026-05-06) established that frontier multimodal LLMs **do not replace** V-JEPA-2 + linear probe for fine-grained subtle behavior classification. The mechanism: Gemini samples video at 1 fps by default; ear movements happen on sub-second timescales. The same gap that broke X-CLIP zero-shot on this task. Direct precedent: Dussert et al. 2025 (*Methods in Ecology and Evolution*, [DOI 10.1111/2041-210X.70059](https://doi.org/10.1111/2041-210X.70059)) tested CogVLM/MobileVLM/CLIP/SigLIP zero-shot on three-class behavior VQA from camera traps — multimodal LLMs underperformed trained baselines.

What MLLMs **are** good at: second-opinion classification on ambiguous cases. The audit uses that capability.

## Setup

1. Get an AI Studio API key at <https://aistudio.google.com/app/apikey>.
2. Copy the env template and fill in your key:
   ```bash
   cd /path/to/poc
   cp .env.example .env
   # edit .env, set GEMINI_API_KEY=...
   ```
3. Install the new dependencies (already pinned in `pyproject.toml`):
   ```bash
   uv pip install --python .venv/bin/python "google-genai>=0.8" "python-dotenv>=1.0"
   ```

## Run

**Dry run first** (no API calls; validates the code path with stubbed responses):

```bash
python tools/gemini_audit.py --dry-run --limit 3
```

**Smoke test** with 5 real clips (~$0.10–0.20 cost):

```bash
python tools/gemini_audit.py --limit 5
```

**Full run** over all 283 clips (~$6–12, ~50–60 min at 6 RPM rate limit):

```bash
python tools/gemini_audit.py
```

The script is **idempotent** — re-running after a partial failure resumes from the last completed clip.

## Outputs

- `outputs/gemini_audit_results.jsonl` — one row per clip (clip path, source, true label, Gemini label, confidence, reasoning, agreement flag, latency, error)
- `outputs/gemini_audit_summary.json` — per-source agreement table, disagreement counts, mean confidence on disagreements

Both are gitignored (`outputs/` is in `.gitignore`).

## Cost estimate

Gemini 3 Pro pricing (May 2026): ~$2 per 1M input tokens, video sampled at 1 fps = ~258 tokens/sec. A 30-second RME clip ≈ 7,740 input tokens ≈ $0.015 input + output. Per clip: **~$0.02–0.04**. Full 283-clip run: **~$6–12**.

For 1,000 audited clips: ~$20–40. For 10,000: ~$200–400. Beyond ~10k, V-JEPA-2 + linear probe wins on cost (electricity only).

## GDPR caveat — read before using on field data

This setup uses **Google AI Studio** (`generativelanguage.googleapis.com`). AI Studio has **no EU data residency** as of May 2026.

That is **acceptable for Read My Ears**: the dataset is CC-BY-4.0, public, and contains anonymized ear-region clips with no identifiable persons. Sending it to AI Studio creates no new GDPR exposure beyond what's already public.

That is **not acceptable for our future field-collected clips**. Field data from the HKiJ peer network will contain:
- Identifiable horses (markings, environment matching the owner's property)
- Identifiable riders/handlers in frame
- Recordings made under consent for "research only", not for "third-party AI processing"

When field clips arrive, **switch to Vertex AI `europe-west4`** before using the audit on them:
- Vertex provides regional pinning (data stays in the EU)
- Different SDK code path: `genai.Client(vertexai=True, project=..., location="europe-west4")`
- Note: as of May 2026, Vertex `europe-west4` only supports Gemini 2.5 (not 3.x) — fine for this audit; you'd need to re-test prompts.

This migration is documented as a TODO, not implemented yet, because it requires GCP project setup that isn't in scope for the current scaffold.

## What to do with the disagreement clips

After a full run, open `outputs/gemini_audit_summary.json` and look for:

1. **Source-clustered disagreements.** If disagreements concentrate on the weak-fold sources we already identified in Sanity 5 (S8 = two-horses confound, S9 = instrumented medical context), that's a **methodological finding** worth noting in the Andersen/Zamansky email. It would be independent confirmation, from a completely different family of models, that those two folds are genuinely harder.
2. **Confident disagreements** (Gemini classifies oppositely with confidence ≥ 0.8). These are candidates for re-review by a certified RHpE assessor — possible label noise in the original dataset.
3. **Low-confidence disagreements** (confidence < 0.5 on either side). Probably ambiguous edge cases; document but don't chase.

## What this is NOT

- Not a clinical decision-support tool
- Not a replacement for the V-JEPA-2 + linear probe production classifier
- Not validated against a certified RHpE assessor's adjudication of the disagreement set (yet)
- Not deployed on field data with personal-data exposure (use Vertex `europe-west4` for that)

## References

- Standard research synthesis (2026-05-06) establishing replacement vs augmentation tradeoffs — see project memory and the Andersen email briefing in `Plans/`.
- V-JEPA-2 paper: <https://arxiv.org/abs/2506.09985>
- Gemini video understanding docs: <https://ai.google.dev/gemini-api/docs/video-understanding>
- Vertex AI data residency: <https://cloud.google.com/vertex-ai/generative-ai/docs/learn/data-residency>
- Animal-Bench (NeurIPS 2024): <https://proceedings.neurips.cc/paper_files/paper/2024/file/8fa604a81e5a236e2f38e917109571a3-Paper-Conference.pdf>
