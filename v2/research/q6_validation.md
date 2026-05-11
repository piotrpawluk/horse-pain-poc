# Q6 — Validation methodology beyond LOSO

## State-of-the-art (single paragraph)

For a 24-behavior RHpE PoC targeting clinical utility, the 2024–2026 medical-AI reporting consensus (TRIPOD+AI, BMJ 2024, Collins et al., PMID 38626948, [verified](https://pubmed.ncbi.nlm.nih.gov/38626948/); CLAIM 2024 Update, Radiology:AI, Tejani et al., PMID 38809149, [verified](https://pmc.ncbi.nlm.nih.gov/articles/PMC11304031); STARD-AI, Nat Med 2025, Sounderajah et al., PMID 40954311; MI-CLAIM, Nat Med 2020, Norgeot et al., PMID 32908275) treats six items as load-bearing: (i) calibration via reliability diagram + ECE + Brier (CLAIM Item 38, TRIPOD+AI); (ii) per-class metrics with bootstrap CIs when N is small (CLAIM Item 38); (iii) external/multi-site testing or explicit justification (CLAIM Item 33; STARD-AI external-validation expansion); (iv) subgroup/fairness reporting (TRIPOD+AI 2024); (v) inter-rater reliability with κ for the reference standard; and (vi) sample-size justification (CLAIM Item 21). The peer-reviewed equine-pain-ML literature partially meets these: Andersen et al. (PLOS ONE 2021, PMID 34665834, n=7 horses, single rater, no AUC/calibration, [verified](https://pmc.ncbi.nlm.nih.gov/articles/PMC8525760/)) and Broomé et al. ("Dynamics are Important...", CVPR 2019, plus PLOS ONE 2022 pain-domain-transfer) report no calibration and no κ. RHpE methodology papers (Dyson et al., Animals 2023, n=150, PMID 37370450, [verified](https://pmc.ncbi.nlm.nih.gov/articles/PMC10295347/); Dyson British Eventing, Animals 2022, n=1,010 starts, [verified](https://pmc.ncbi.nlm.nih.gov/articles/PMC8909886/); Garcia 2023 Icelandic pilot, EVE 13803) report κ per behavior with the Landis–Koch bands (substantial 14/22, moderate 6/22, fair 1/22 in Icelandic horses); Bonferroni was used for multiple-comparison control. Equine composite-pain validation (van Loon & van Dierendonck EQUUS-COMPASS, Vet J 2015, PMID 26526526) anchors ICC ≥0.93 as the inter-observer benchmark. For small-N AUC comparison, DeLong is documented underpowered (≈0.24 power vs. 0.89 bootstrap at n≈53 events; Lee 2013 PMID 23656853); permutation tests (Venkatraman–Begg) and hierarchical bootstrap dominate. Calibration on N≈300 favors **Platt/temperature scaling** (parametric, low overfit) over isotonic (needs N≳1000; sklearn docs; Ojeda 2023 *Stat Med* 10.1002/sim.9921). For 24 simultaneous binary tests, Benjamini–Hochberg FDR is the reported standard (more powerful than Bonferroni; precedent in Dyson 2022 BE study used Bonferroni-conservative).

## Cost-benefit for solo maintainer

| Validation step | Solo-shippable? | Cost |
|---|---|---|
| Per-behavior AUC + bootstrap 95% CI | **Yes** | 1 day; standard scikit-learn |
| Reliability diagram + ECE + Brier | **Yes** | 0.5 day; per-behavior + at ≥8/24 operating point |
| Temperature scaling on held-out fold | **Yes** | 0.5 day; one parameter, fits N=300 |
| Source-aware LOSO (already locked) | **Yes** | already in pipeline |
| BH-FDR across 24 behaviors | **Yes** | trivial; `statsmodels.multipletests` |
| Permutation test (vs. DeLong) for paired AUC | **Yes** | 1 day; replaces underpowered DeLong |
| Multi-rater κ on 100% of clips | **No** — needs ≥2 vets | Collaboration required |
| Multi-rater κ on 20% audit subset | **Partially** | 1 vet collaborator, ~10 hr labeling |
| Held-out clinic / cross-country test | **No** | Requires partner clinics (UK + EU breeds) |
| Prospective clinical deployment | **No** | Multi-year, IRB-equivalent, multi-site |
| TRIPOD+AI / CLAIM 2024 checklist completion | **Yes** | 0.5 day; doc artifact |

## What this implies for the PoC

The single most load-bearing validation step missing today is **multi-rater κ on at least an audit subset (≥20% of clips, ≥2 vets)** — without it, every per-behavior AUC reported is upper-bounded by single-rater label noise and cannot be defended under TRIPOD+AI or STARD-AI. Calibration at the ≥8/24 operating point (reliability diagram + ECE + Brier) is the second-most load-bearing and is solo-shippable in a day.

## Open Questions

- For "fair-to-moderate" RHpE behaviors (κ 0.4–0.7, ~7/24 in the Icelandic pilot), the literature does not support single-expert ground truth. A solo PoC may need to **report these behaviors with a κ-weighted reliability flag** rather than treat them as gold-standard targets — or drop them from the aggregate ≥8/24 score, which changes the clinical claim.
- Held-out cross-country (e.g., UK warmblood vs. Icelandic vs. Iberian) external testing is treated as load-bearing by STARD-AI 2025 but is structurally unreachable solo. The PoC framing may need to be downgraded from "clinical screening tool" to "research prototype with documented external-validation gap" until a clinic partnership lands.
- TRIPOD+AI / CLAIM 2024 / STARD-AI have **not** been formally adopted by the equine veterinary community (no equine-specific adaptation found in 2024–2026 search). The PoC can lead by adopting them voluntarily, but reviewers in *Equine Veterinary Journal* / *Animals* may not enforce them.

## Citations (all peer-reviewed unless flagged)

- Collins GS et al. **TRIPOD+AI statement.** *BMJ* 2024;385:e078378. PMID 38626948. https://pubmed.ncbi.nlm.nih.gov/38626948/ [verified]
- Tejani AS et al. **CLAIM 2024 Update.** *Radiology: AI* 2024. PMID 38809149. https://pmc.ncbi.nlm.nih.gov/articles/PMC11304031 [verified]
- Sounderajah V et al. **STARD-AI.** *Nature Medicine* 2025;31:3283-3289. PMID 40954311. [verified via search; full-text behind auth]
- Norgeot B et al. **MI-CLAIM checklist.** *Nature Medicine* 2020;26:1320-1324. PMID 32908275.
- Dyson S et al. **RHpE applied to 150 horses pre/post diagnostic anaesthesia.** *Animals* 2023;13(12):1940. PMID 37370450. https://pmc.ncbi.nlm.nih.gov/articles/PMC10295347/ [verified]
- Dyson S, Pollard D. **RHpE in British Eventing.** *Animals* 2022. https://pmc.ncbi.nlm.nih.gov/articles/PMC8909886/ [verified] — uses Bonferroni; n=1,010 starts; per-behavior frequencies reported.
- Garcia MR et al. **RHpE in Icelandic horses pilot.** *Equine Vet Educ* 2023. doi:10.1111/eve.13803 — per-behavior κ: 14/22 substantial-to-perfect, 6/22 moderate, 1/22 fair.
- Andersen PH et al. **Pain assessment in horses via facial expression deep learning.** *PLOS ONE* 2021;16(10):e0258672. PMID 34665834. https://pmc.ncbi.nlm.nih.gov/articles/PMC8525760/ [verified] — n=7 horses, single rater, no AUC/calibration reported.
- Broomé S et al. **Dynamics are Important for the Recognition of Equine Pain in Video.** CVPR 2019. https://openaccess.thecvf.com/content_CVPR_2019/papers/Broome_Dynamics_Are_Important_for_the_Recognition_of_Equine_Pain_in_CVPR_2019_paper.pdf
- Broomé S et al. **Sharing pain: pain domain transfer for video recognition of low-grade orthopedic pain in horses.** *PLOS ONE* 2022.
- van Loon JPAM, Van Dierendonck MC. **EQUUS-COMPASS / EQUUS-FAP construction.** *Vet J* 2015. PMID 26526526 — ICC 0.98 / 0.93 inter-observer benchmark for equine pain scales.
- Ojeda FM et al. **Calibrating ML approaches for probability estimation: comparison.** *Statistics in Medicine* 2023. doi:10.1002/sim.9921 — Platt/temperature dominate isotonic at small N.
- Lee H et al. **Misuse of DeLong test for nested models.** *PMC3684152* 2013 — documents DeLong power collapse at small N; F-test/bootstrap recommended.
- Venkatraman ES, Begg CB. **A distribution-free procedure for comparing ROC curves.** *Biometrika* 1996 — permutation alternative to DeLong.
- scikit-learn calibration docs https://scikit-learn.org/stable/modules/calibration.html [non-peer-reviewed; tooling reference] — isotonic needs N≳1000.

## Flags

- **Commercial / non-peer-reviewed:** scikit-learn docs (tooling), Mad Barn / 24horsebehaviors.org / The Horse / EquiManagement (consumer/educational, **not cited as evidence**).
- **All other citations peer-reviewed.**
- **Could not verify full-text** behind auth: Nature Medicine STARD-AI 2025 (auth-walled). Citation retained because the PMID + EQUATOR Network listing independently confirm it.
