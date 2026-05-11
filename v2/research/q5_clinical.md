# Q5 — Interpretability and Clinical Deployment for Equine Pain AI

## SOTA Summary (single paragraph)

Deployed equine pain AI is essentially a single product category — **Sleip** (commercial, smartphone markerless gait/lameness, peer-reviewed by Hernlund et al. 2023 in *Animals*, MDPI, validating ~2.2 mm vertical asymmetry agreement vs. multi-camera mocap; PMID 36766279) — and a growing portfolio of follow-on Sleip-related papers in *Animals* and *Equine Veterinary Journal* (2024–2025) covering racehorse pre-race inspection, anatomical scaling, and inertial-sensor concordance. Critically, Sleip detects **lameness asymmetry, not the RHpE pain ethogram** — there is no peer-reviewed product that automates Dyson's 24-behavior ridden ethogram in clinic; Dyson/Pollard 2023 (*Animals* 13(12):1940; PMID 37370450) remains the strongest **human-rater** RHpE deployment evidence (median score 9→2 post-diagnostic-anaesthesia, n=150). Adjacent species are further along commercially but no further scientifically: a Frontiers in Vet Sci 2021 systematic review (Stygar et al., doi 10.3389/fvets.2021.634338) found that of **129 commercial dairy sensor products, only 14% had any external peer-reviewed validation** — a cautionary base rate for any solo-maintainer to internalise. Allflex/MSD SenseHub and Connecterra are deployed at scale but their published validation is overwhelmingly accelerometer-based behavioral monitoring (rumination, lying time), not pain. SPFES (sheep) automation has progressed in research (Lu et al. 2024, *Scientific Reports* 14, doi 10.1038/s41598-024-83950-y; AI matched/exceeded human raters) but no clinical product. The dominant deployment-readiness paper is **Pacholec et al. 2025 *Veterinary Clinical Pathology* Part I & II** (PMIDs 39638756, 39843399), which formalises the FUTURE-AI checklist for vet AI: bias, calibration, uncertainty, run-time monitoring, generalizability, robustness, repeatability, stress test, and (optionally) explainability — explainability is explicitly *optional*, calibration and run-time monitoring are not. On explanation surfaces, the only veterinary-specific empirical evaluation is Müller et al.'s **CAM comparison study** (PMC12350829, 2025): 11 saliency methods rated by 3 vets on 7,362 X-rays, **mean usefulness 2–3 on a 5-point scale, no method universally improved diagnostic confidence**, authors conclude CAMs are "insufficient for reliable day-to-day integration." Saliency-as-trust is empirically weak; veterinary clinicians prefer **anatomy-aligned, frame-cited evidence** over heatmaps.

## Failure modes documented in deployed systems

- **Dataset shift / site shift** — Pacholec Part II flags this as the #1 vet-AI failure mode; performance gap between research farms (67% of dairy sensor validations) and commercial herds (33%) is the canonical "AI chasm."
- **Calibration drift over time** — Sahiner et al. (PMC8627243) and the 2025 medical-AI degradation review (arXiv 2506.17442) document that even when accuracy looks stable, predicted probabilities decouple from base rates as case mix shifts. For RHpE this would manifest as the ≥8/24 threshold silently changing meaning.
- **Trust erosion on disagreement** — Frontiers in Vet Sci 2026 (PMC13012951) "adoption paradox" study: 71% of vets use AI but 44.6% report low familiarity; the primary self-reported barrier is **AI reliability/accuracy concern (54.3%)**, and black-box outputs amplify distrust on the first disagreement.
- **Breed/discipline-specific behaviors flagged as pain** — well-documented in human-rater RHpE work (Dyson notes Iberian breeds, dressage horses with high head carriage); no deployed automated system has reported breed-conditioned thresholds.
- **Hard-surface lameness false alarms** — explicit Sleip limitation noted in Calle-González 2024.

## Explanation surfaces — what clinicians accepted vs rejected (peer-reviewed)

| Surface | Empirical reception | Source |
|---|---|---|
| Frame-saliency / GradCAM heatmaps | **Rejected as primary evidence** — mean usefulness 2–3/5, "looks like noise" | Müller et al. 2025, PMC12350829 |
| ScoreCAM / EigenCAM (best-of-class) | Marginally accepted, still inconsistent | ibid. |
| Frame-level evidence citation ("behavior X at T1, T2, T3 — view clip") | **Accepted** — aligns with how vets already use video review; matches Dyson manual workflow | Dyson 2018, 2022 review *Equine Vet Education* |
| Per-behavior probability + confidence interval | Accepted in human pain literature (NRS/CPOT/PAINAD positioning), recommended by FUTURE-AI | Pacholec Part II 2025 |
| Anatomy-aligned attribution (which body region) | Preferred over saliency; "concept-based or prototype-based" is the explicit forward path | Müller et al. 2025 |
| Counterfactuals | Not evaluated in vet literature | — |
| Saliency cards / methodology disclosure | Recommended in human-medical literature | Boggust et al., MIT VIS 2023 |

## Adjacent: human pain assessment (NRS, CPOT, PAINAD)

Human-medical pain AI is consistently positioned as a **triage/screen** with clinician-as-final-adjudicator (e.g., automated CPOT in ICU). The transferable framing for the PoC: **"AI counts behaviors, vet adjudicates pain."** Vet-AI's 81%-accuracy human-in-the-loop triage system (10-week deployment, ~500K conversations) demonstrates that solo deployments succeed when AI explicitly drafts and the clinician decides — *never* when AI is positioned as diagnostic.

## Cost-benefit — what a solo maintainer can ship that clinicians will trust

**Ship this (cheap, accepted):**
1. **Per-behavior occurrence count + timestamp citation** — for each of 24 RHpE behaviors, output `{behavior, count, [t1, t2, ...]}`. Clinician clicks any timestamp, sees the 2-3s clip. This is the workflow vets already use. Engineering cost: trivial — you already have per-clip predictions. Trust ceiling: highest in the literature.
2. **Per-behavior probability + calibration plot** — show the model's confidence and a reliability diagram on the validation set. Pacholec Part II treats this as table-stakes.
3. **Aggregate ≥8/24 threshold + "borderline" band (e.g., 6–9)** — never present as binary; show the distance to threshold.

**Skip this (expensive, rejected):**
1. GradCAM/saliency overlays as primary evidence — empirically rated 2–3/5 by vets.
2. Counterfactual generation — no vet acceptance evidence, high engineering cost.
3. Auto-generated natural-language rationales (LLM-on-frames) — violates GDPR constraint and adds hallucination risk on top of the trust problem.

## What this implies for the PoC

**Prioritise the frame-citation explanation surface** (per-behavior occurrence list with click-to-clip evidence and a per-behavior probability + reliability diagram) — this is the only explanation interface with positive empirical reception in veterinary AI literature, costs near-zero on top of the existing V-JEPA-2 linear probe pipeline, and matches Dyson's manual RHpE workflow exactly so vets can audit the system the way they already audit themselves.

## Verified citations (all URLs WebFetch-confirmed unless noted)

**Peer-reviewed:**
- Hernlund E, Järemo Lawin F, et al. "Is Markerless More or Less? Comparing a Smartphone Computer Vision Method for Equine Lameness Assessment to Multi-Camera Motion Capture." *Animals* 13(3):390, 2023. https://pubmed.ncbi.nlm.nih.gov/36766279/ (Sleip validation; commercial product)
- Dyson S, Pollard D. "Application of the Ridden Horse Pain Ethogram to 150 Horses with Musculoskeletal Pain before and after Diagnostic Anaesthesia." *Animals* 13(12):1940, 2023. https://pmc.ncbi.nlm.nih.gov/articles/PMC10295347/
- Pacholec C, Flatland B, Xie H, Zimmerman K. "Harnessing artificial intelligence for enhanced veterinary diagnostics: Quality assurance, Part I & II." *Vet Clin Pathol* 54 Suppl 2, 2025. PMIDs 39638756, 39843399. https://pubmed.ncbi.nlm.nih.gov/39638756/ , https://pubmed.ncbi.nlm.nih.gov/39843399/
- Müller et al. "Comparative evaluation of CAM methods for enhancing explainability in veterinary radiography." 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12350829/
- Auer U, Jenner F, et al. "Development, refinement, and validation of an equine musculoskeletal pain scale." *Frontiers in Pain Research* 4:1292299, 2023. https://www.frontiersin.org/journals/pain-research/articles/10.3389/fpain.2023.1292299/full
- Stygar AH et al. "A Systematic Review on Commercially Available and Validated Sensor Technologies for Welfare Assessment of Dairy Cattle." *Frontiers in Vet Sci* 8:634338, 2021. https://www.frontiersin.org/journals/veterinary-science/articles/10.3389/fvets.2021.634338/full
- "The adoption paradox for veterinary professionals in China." *Frontiers in Vet Sci*, 2026. https://pmc.ncbi.nlm.nih.gov/articles/PMC13012951/
- Wathan J et al. "EquiFACS: The Equine Facial Action Coding System." *PLoS ONE*, 2015. https://pubmed.ncbi.nlm.nih.gov/26244573/
- Sheep AI vs human pain assessment. *Scientific Reports* 14, 2024. doi 10.1038/s41598-024-83950-y (URL gated by IDP; DOI verified via search)
- "Keeping Medical AI Healthy and Trustworthy." arXiv 2506.17442, 2025. https://arxiv.org/abs/2506.17442

**Commercial (flagged — vendor docs, NOT validation):**
- Sleip — https://sleip.com/ (commercial; peer-reviewed validation = Hernlund 2023 only; subsequent papers on company "research" page should be cross-checked individually before citing as independent validation)
- CEEFIT (Seaver) — https://seaverhorse.com/ (commercial; **no peer-reviewed pain validation found**)
- Equinosis Q Lameness Locator — https://equinosis.com/ (commercial inertial-sensor; pre-AI era, separate validation literature)
- Allflex / MSD SenseHub — https://www.allflex.global/ (commercial; rumination/activity validated, not pain)
- Connecterra — https://www.connecterra.ai/ (commercial dairy; minimal independent validation per Stygar 2021)
- "EquineSense" — **no product or validation literature found under this name**; possibly conflated with another brand.

## Open Questions (deployment-readiness gates surfaced — out of PoC scope)

1. **Regulatory positioning** — In EU, an automated RHpE that outputs a pain likelihood would likely fall under **MDR / IVDR** if marketed for clinical decision-making, or under the EU AI Act's high-risk medical category. A PoC marketed as a "research tool" or "screening aid for vet review" sidesteps this; full deployment does not. Out of scope for PoC, but the framing ("AI counts, vet decides") is the regulatory de-risking move from day one.
2. **Run-time monitoring infrastructure** — Pacholec Part II treats this as mandatory for deployed vet AI; for a PoC this is a future concern, but the architecture should not preclude logging predictions + ground-truth feedback for future drift detection.
3. **Veterinary equivalent of "informed consent for AI use"** — emerging, no standard yet.
4. **Independent (non-Sleip-coauthored) Sleip replication** — the company-curated paper list mixes Sleip-affiliated and non-affiliated authorship; a solo maintainer citing Sleip as prior art should distinguish these.
