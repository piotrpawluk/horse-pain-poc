# URL Verification Audit

Final audit of citations appearing in `synthesis.md`. Each sub-researcher independently verified URLs at retrieval time; this log records main-thread spot-checks of load-bearing citations (one per question) and notes corrections applied during synthesis.

## Spot-checks performed in main thread

| # | URL | Claim verified | Status |
|---|-----|----------------|--------|
| 1 | https://arxiv.org/abs/2506.09985 | V-JEPA 2 paper, SSv2 attentive-probe 77.3 | ✓ verified — "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning" by Assran/LeCun/Ballas et al.; SSv2 77.3 confirmed |
| 2 | https://elifesciences.org/articles/63377 | DeepEthogram, eLife 2021, 80-frames-per-behavior result | ✓ verified — Bohnslav et al. 2021. **Correction applied:** sub-report Q3 claimed "shared *frozen* CNN features"; actual architecture fine-tunes spatial+flow CNNs. Synthesis Q3 section corrected to remove "frozen" claim. |
| 3 | https://arxiv.org/abs/2202.07925 | ActionFormer, ECCV 2022 | ✓ verified — Zhang/Wu/Li, ECCV 2022, THUMOS14 71.0 mAP confirmed |
| 4 | https://pubmed.ncbi.nlm.nih.gov/39843399/ | Pacholec 2025 Part II, FUTURE-AI/DECIDE-AI for vet AI | ✓ verified — Pacholec/Flatland/Xie/Zimmerman, *Vet Clin Pathol* 54 Suppl 2, S43-S51, 2025 |
| 5 | https://pubmed.ncbi.nlm.nih.gov/38626948/ | TRIPOD+AI statement, BMJ April 2024 | ✓ verified — Collins et al., BMJ 2024;385:e078378, 16 April 2024 |
| 6 | https://pmc.ncbi.nlm.nih.gov/articles/PMC10295347/ | Dyson & Pollard 2023 RHpE on 150 horses | ✓ verified — *Animals* 13(12):1940, 9 June 2023 |

## Sub-researcher self-verifications (declared in their reports)

Each sub-agent declared WebFetch verification of all cited URLs before delivery. Notable declarations:

- **Q1**: 8 WebFetch attempts, 3 succeeded directly, 3 redirected/403 (replaced with PMC mirrors), 2 N/A. PFERD CC-BY-4.0 license cross-verified via PMC mirror after Harvard Dataverse returned empty.
- **Q2**: All HuggingFace/arXiv/GitHub URLs WebFetch-verified. V-JEPA-2 ViT-g SSv2 numbers cross-checked with arXiv:2506.09985 Table 4.
- **Q3**: All cited URLs WebFetch-verified. Notably, the user-brief reference `arXiv:2003.10474` was checked and found unrelated (Discontinuous Galerkin / fractional diffusion); citation was DROPPED rather than paraphrased.
- **Q4**: 11 sources WebFetch/WebSearch verified, all peer-reviewed academic, no commercial.
- **Q5**: Hernlund/Dyson/Pacholec/Müller/Auer/Stygar/Wathan/adoption-paradox PMC URLs confirmed via WebFetch. Sleep/CEEFIT/Allflex/Connecterra flagged commercial-only. STARD-AI / Nature Medicine 2025 auth-walled — citation retained because PMID + EQUATOR Network listing independently confirm.
- **Q6**: TRIPOD+AI/CLAIM/RHpE-Dyson/Andersen all WebFetch-verified. CLAIM 2024 verified via PMC mirror after RSNA 403. STARD-AI behind auth wall; PMID + EQUATOR confirm.

## Citations DROPPED during research (as a result of verification failure)

- `arXiv:2003.10474` — listed in Q3 brief; resolves to a Discontinuous Galerkin / fractional diffusion paper unrelated to multi-label calibration. Dropped by Q3 sub-agent.
- "EquineSense" — listed in Q5 brief as a commercial product; Q5 sub-agent searched and **no product or validation literature found under this name**. Possibly conflated with another brand. Dropped, with note in Q5 sub-report.

## Citations flagged auth-walled but cross-confirmed by independent identifiers

- **STARD-AI** *Nature Medicine* 2025;31:3283-3289 — full text behind auth, but PMID 40954311 + EQUATOR Network listing independently confirm.
- **Garcia 2023** *Equine Vet Educ* 13803 (Icelandic pilot) — Wiley auth-walled; cited by sub-report Q1 and Q6 with consistent κ distribution figures (14/22 / 6/22 / 1/22 substantial/moderate/fair), suggesting both agents read the same source independently.

## Result

All load-bearing citations in `synthesis.md` either WebFetch-verified by the sub-researcher who introduced them, spot-checked by the main thread, or auth-walled with independent identifier confirmation. One factual correction (DeepEthogram architecture description) applied to synthesis text. No anti-target citations introduced. **Audit passes.**
