# Gemini integration — label-noise audit experiment (status: hypothesis falsified)

This file documents an experiment that tested whether off-the-shelf frontier multimodal LLMs (Gemini 2.5 Pro and Gemini 3.1 Pro Preview, May 2026, AI Studio API) can serve as label-noise auditors on the 283-clip Read My Ears dataset (Alves et al. CVPR W'25, CC-BY-4.0). **The hypothesis was falsified at N=36 stratified clips; this document records the experiment, its findings, and how to reproduce them.**

The original framing — "Gemini disagreements with the human label flag candidates for re-review" — assumed the model would be a reasonably calibrated witness. It is not, on this task, at the parameters and prompts we tested.

## What we tested

Three prompt variants on two model generations × per-clip + cross-rep probe:

- **Prompt A** (generic) — *"is the horse actively moving its ears?"*
- **Prompt B** (EquiFACS-anchored) — *"would a trained EquiFACS coder mark this as a coded ear AU?"*, observation-then-classify
- **Prompt C** (Gemini-3.x best practice) — system-instruction-grounded, evidence-cite-then-classify, no negative constraints, temp=1.0, thinking_level=low

All prompt definitions live verbatim in `tools/gemini_audit.py`. Schemas are strict JSON. Video uploaded via File API at `videoMetadata.fps=10` (clips are 0.24–1.76 s long; default fps=1 collapses them to 1–2 frames each).

## Headline findings

See [`lessons_learned.md` Lesson 14](lessons_learned.md) for the full writeup. Three results:

1. **Cross-rep instability is structural.** On Gemini 3.1 Pro Preview at Google's officially recommended Gemini 3.x parameters (temp=1.0, thinking_level=low), 0/20 clips produce consistent motion/still descriptions across 5 reps. On Gemini 2.5 Pro at temp=1.0, 2/10 clips consistent. The 80% flip rate is not a low-temperature artifact.
2. **Perception/classification decoupling is generation-general.** On both 2.5 Pro (with Prompt B) and 3.1 Pro Preview (with Prompt C), the same model on the same clip reports motion in the majority of description-only reps but commits to "background" in classification mode. 13/20 clips on 3.1 + 7/10 clips on 2.5 show this pattern. The mechanism is plausibly that post-training selects harder for conservative classification commitments than for conservative descriptive language.
3. **Conservatism on 3.1 Pro Preview is structural, not parameter-driven.** With the best-practice prompt + Google's recommended parameters, 3.1 still classifies 35/36 clips as background regardless of true label.

**Operational implication.** Off-the-shelf Gemini 2.5 Pro and 3.1 Pro Preview, at the parameters tested in May 2026, are not reliable label-noise auditors for fine-grained equine ear-movement classification on Read My Ears. We do not extrapolate to Claude, GPT-class, or other multimodal LLMs without testing.

## Reproduce

```bash
cd /path/to/poc
echo "GEMINI_API_KEY=<your key>" > .env

# Classifier (Prompt C on 3.1 Pro Preview, best-practice config):
python tools/gemini_audit.py \
    --model gemini-3.1-pro-preview \
    --prompt-version c \
    --per-source 3 \
    --fps 10 \
    --temperature 1.0 \
    --thinking-level low

# Description-only probe (5 reps × 10 disagreement clips on 3.1 Pro Preview):
python tools/gemini_audit.py --probe \
    --model gemini-3.1-pro-preview \
    --probe-source-model gemini-2.5-pro \
    --probe-clips 10 \
    --probe-reps 5 \
    --probe-temp 1.0 \
    --thinking-level low \
    --probe-suffix _temp1.0_thinkLOW \
    --fps 10
```

For 2.5 Pro, **omit `--thinking-level`** — Gemini 2.5 Pro returns `400 INVALID_ARGUMENT: Thinking level is not supported for this model`.

JSONL outputs land in `outputs/`:
- `gemini_audit_results_{model}_prompt{A,B,C}.jsonl` — per-clip classifications
- `gemini_audit_probe_{model}{suffix}.jsonl` — per-rep description-only observations
- `gemini_audit_summary_{model}_prompt{A,B,C}.json` — per-source agreement summaries

All gitignored (under `outputs/`).

## Cost estimate

May 2026 Gemini API pricing (preview models priced lower than stable releases):

- Per-clip classifier (~1100 tokens prompt + ~50 tokens output): ≈ $0.005 on 3.1 Pro Preview, ≈ $0.04 on 2.5 Pro
- Full 36-clip stratified run: $0.20 (3.1) – $1.50 (2.5)
- N=10 × 5-rep probe: $0.25 (3.1) – $2.00 (2.5)
- Replication of all experiments end-to-end (this writeup): under $5

## GDPR caveat (still applies; unchanged from original setup)

This setup uses **Google AI Studio** (`generativelanguage.googleapis.com`). AI Studio has **no EU data residency** as of May 2026.

That is **acceptable for Read My Ears**: dataset is CC-BY-4.0, public, anonymized, no identifiable persons. That is **not acceptable for our future field-collected clips** — when field clips arrive, switch to Vertex AI `europe-west4` before running anything from this repo on them.

## What this experiment is and is not

It is a **supporting methodological observation** for the V-JEPA-2 + linear probe pipeline. The thesis of the project remains V-JEPA-2 LOSO 0.875 source-aware replication on Read My Ears (Sanity 5). This experiment exists to document why we did not pivot to "MLLM-as-judge" — we tested it, it doesn't work in this regime, V-JEPA-2 stays the backbone.

It is not a publication-grade evaluation of frontier multimodal LLMs. N=36 stratified is a convenience sample, not a benchmark. The findings are dated (May 2026), scoped to two specific Gemini models accessed via a specific API endpoint, and should not be extrapolated to other models or other behaviors without testing.

## References

- [Lesson 14 in `lessons_learned.md`](lessons_learned.md) — full writeup with numbers
- [Gemini 3 Developer Guide](https://ai.google.dev/gemini-api/docs/gemini-3) — official parameter recommendations
- [Gemini 3.1 Pro model card (DeepMind)](https://deepmind.google/models/model-cards/gemini-3-1-pro/) — model specifications
- [Vertex AI Gemini 3 prompting guide](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/gemini-3-prompting-guide) — prompt-engineering best practices
- [V-JEPA-2 paper (Meta, Jun 2025)](https://arxiv.org/abs/2506.09985) — the actual production classifier
- [Read My Ears paper (Alves et al. CVPR W'25)](https://arxiv.org/abs/2505.03554) — source dataset + EquiFACS labels
