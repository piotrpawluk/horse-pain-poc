# RHpE Recognition PoC — Technology & Methodology Survey

*Synthesis of 6 parallel sub-investigations, 2026-05-10. URLs verified via WebFetch (full audit log: `url-verification.md`).*

---

## Q1 — Dataset landscape

**No public 24-behavior RHpE labeled video corpus exists.** The canonical labeled archives are held privately by the Dyson group (RVC), the KTH/SLU Sweden line (Broomé, Andersen, Rhodin, Ask), and the Phelipon/Lansade/Razzaq 2025 group. Dyson's validation studies (Dyson 2020 *Animals* 10(6):1044; Dyson & Pollard 2023 *Animals* 13(12):1940, [PMC10295347](https://pmc.ncbi.nlm.nih.gov/articles/PMC10295347/), peer-reviewed, verified) use single-rater real-time scoring, not a sharable archive. The SLU/KTH line publishes **code only** ([github.com/sofiabroome/painface-recognition](https://github.com/sofiabroome/painface-recognition)) — datasets PF/EOP(j) are confidential and on-request via Rhodin/Ask at SLU. **Phelipon 2025** *Scientific Reports* ([PMC12018932](https://pmc.ncbi.nlm.nih.gov/articles/PMC12018932/), peer-reviewed) is the closest RHpE-aligned resource (1,036 ridden-horse images, comfortable/uncomfortable binary), image-only and not openly redistributable.

**Adjacent open data:** **PFERD** (*Scientific Data* 2024, peer-reviewed, [doi:10.7910/DVN/2EXONE](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/2EXONE), CC-BY 4.0) is pose-only on 5 horses with markered ground truth — useful for keypoint pretraining, not pain. **EquiFACS** (Wathan et al. *PLOS One* 2015) provides FAU-coding manuals only; underlying video not redistributed.

**Build-from-scratch cost.** Dyson 2020 anchors trained-vet RHpE Cohen's κ at **0.72 (SD 0.22)**, total-score ICC 0.97. A 10-min ridden clip takes a vet 25–40 min to RHpE-score; at £100–150/hr equine-vet rates that's ~£50–100 per single-rater clip. For κ≥0.7 on ≥20% of corpus with ≥2 raters, a 300-clip κ-validated set ≈ 360 vet-hours ≈ **£40–60k** plus filming logistics with GDPR-consented subjects. **Not feasible solo — deferred until resources change.** Collaboration shortlist (descending plausibility): SLU/Uppsala (Rhodin, Ask, Andersen) > KTH RPL (Kjellström/Broomé) > Dyson group.

**Implication for PoC:** Pilot 50–100 in-house clips with 1 vet + 1 self-trained rater for a κ spot-check, while opening collaboration with SLU/Uppsala before locking architecture. Treat dataset access as the project's binding constraint, not the architecture.

---

## Q2 — Video-encoder backbones

**V-JEPA-2 ViT-g is the single configuration with published frozen-probe evidence of beating the current ViT-L floor on a fine-grained motion benchmark.** The V-JEPA-2 paper ([arXiv:2506.09985](https://arxiv.org/abs/2506.09985), verified) Table 4 reports SSv2 attentive-probe scores: V-JEPA-2 ViT-g 75.3 / ViT-g/384 77.3 vs **InternVideo2-1B 69.7** (lateral) and **VideoMAE-V2 (1B) 56.1** (catastrophic 19–21 pt loss). The pixel-reconstruction (VideoMAE-V2) and feature-prediction (V-JEPA-2) gap on fine-grained motion is decisive at frozen. **DINOv2** is image-only and lost prior project's RME LOSO at 0.514 — anti-target.

**Hiera, MViT-v2, Video-Swin** ([arXiv:2306.00989](https://arxiv.org/abs/2306.00989), [arXiv:2112.01526](https://arxiv.org/abs/2112.01526), [arXiv:2106.13230](https://arxiv.org/abs/2106.13230)) have **zero published frozen low-N evidence on SSv2-class fine-grained tasks** — their reported wins are end-to-end fine-tuned on coarse benchmarks (Kinetics-400-class), which does not extrapolate to clip-level frozen probing at N≤500. Drop from shortlist.

**No animal-specific video foundation model exists** as of 2026-05. **PFERD** is pose-only; the cattle TimeSformer (*npj Vet Sci* 2026) is supervised fine-tuning, not a frozen backbone. Re-survey in 6 months.

**M2 Max viability.** All 1B-class video encoders fit at fp16; 96 GB is non-binding. PyTorch MPS works for all candidates via `transformers`/`timm`/`torchvision`; **no MLX-native ports exist** as of 2026-05. Frozen-extraction MPS throughput at bs=1: ~6–15 clips/s at ViT-L, ~1–3 clips/s at 1B scale. A 15-min RHpE session ≈ 450 clips → 5 min wall on ViT-L, 15–25 min at 1B (offline-acceptable). Feature dimensions (1024–1536) are comparable across candidates — no dimensionality advantage at N≤500.

**Implication for PoC:** Run **one** A/B of V-JEPA-2 ViT-g vs the current ViT-L floor *after* Q3 architecture is settled. Backbone choice is unlikely to be the dominant bottleneck — the gap from ViT-L→ViT-g is plausibly +5–7 pts on SSv2-class, smaller than what calibrated downstream architecture decisions buy.

---

## Q3 — Multi-behavior architecture (24 binary detectors)

**The peer-reviewed pattern for multi-label fine-grained behavior at small N is per-class binary heads with sigmoid cross-entropy.** AVA ([arXiv:1705.08421](https://arxiv.org/abs/1705.08421), CVPR 2018, peer-reviewed) explicitly switched to per-class sigmoid for 80 atomic actions with frequent multi-label co-occurrence — the canonical reference. **DeepEthogram** (Bohnslav et al., *eLife* 2021;10:e63377, verified) is the closest multi-behavior animal-behavior analogue: independent per-behavior binary heads, >90% accuracy with ~80 example frames per behavior. (Correction to one sub-report: DeepEthogram fine-tunes its spatial+flow CNNs rather than running them frozen — but the "independent binary heads + small-N viability" lesson is the load-bearing finding regardless.)

**MTL caution.** Standley et al. *Which Tasks Should Be Learned Together?* ([arXiv:1905.07553](https://arxiv.org/abs/1905.07553), ICML 2020) and Crawshaw's MTL survey ([arXiv:2009.09796](https://arxiv.org/abs/2009.09796)) both document shared-trunk MTL as fragile at small N: tasks compete, negative transfer is common, optimal task-grouping must be empirically discovered — costs the PoC cannot absorb at 24 tasks × ~300 clips. Hard parameter sharing (Ruder [arXiv:1706.05098](https://arxiv.org/abs/1706.05098)) regularizes mainly when features are not yet learned; with V-JEPA-2 frozen and strong, that argument evaporates.

**Calibration is decisive.** RHpE thresholds at ≥8/24 on per-behavior occurrence counts (Dyson 2022, *Equine Vet Educ* 13468). Independent probes give cleanly calibrated per-behavior probabilities via per-probe Platt/temperature scaling; shared heads can break per-class calibration without per-head calibration steps.

**Cost-benefit ranking for solo maintainer:** **(a) 24 independent linear probes** — minutes to train (24 sklearn fits), best calibration, lowest complexity, start here. (b) Multi-head MLP — modest expressivity gain; only if (a) demonstrably underfits. (c) Shared trainable trunk + 24 heads — expected negative transfer at this N; gradient-balancing complexity unjustified. (d) Behavior-as-token sequence transformer — needs dense temporal labels; underpowered at N≈300/behavior. **Reject.**

**Implication for PoC:** Start with **24 independent linear probes** on frozen V-JEPA-2 ViT-L features, sigmoid per behavior, per-probe isotonic or Platt calibration, source-aware LOSO. Architectural answer is invariant to whether the PoC ships 24 or a high-κ subset of 6–8.

---

## Q4 — Long-form video: clip-prob streams → per-behavior counts

**The bridge problem has a hard-to-beat simple-baseline answer.** The weakly-supervised TAL literature ([Baraka & Mohd Noor 2022](https://doi.org/10.1007/s00521-022-07102-x) survey; [Lee & Byun ICCV 2021](https://arxiv.org/abs/2108.05029)) reports that calibrated clip classifiers + threshold + temporal NMS trail learned localization heads by only ~3–6 mAP points at IoU=0.5 — small enough that on **count accuracy** (which RHpE actually needs), the simple baseline often wins. Every modern TAL method — **ActionFormer** ([arXiv:2202.07925](https://arxiv.org/abs/2202.07925), ECCV 2022, verified), **TriDet** ([arXiv:2303.07347](https://arxiv.org/abs/2303.07347), CVPR 2023), **TemporalMaxer** ([arXiv:2303.09055](https://arxiv.org/abs/2303.09055)), **TadTR** ([arXiv:2106.10271](https://arxiv.org/abs/2106.10271)), **BMN** ([arXiv:1907.09702](https://arxiv.org/abs/1907.09702)) — requires segment-level temporal annotation the PoC does not have. Action segmentation models (**MS-TCN**, **ASFormer**, **DiffAct**) similarly require per-frame multi-label supervision; **DiffAct** is also MPS-unfriendly.

**Counting-aware losses (RepNet-style)** are tuned for periodic actions and are a wrong-domain fit for the heterogeneous RHpE ethogram.

**The real fork** is between (i) calibrating the clip classifier and post-processing with isotonic/Platt → hysteresis threshold → temporal NMS → event count (1–2 days, MPS-friendly, zero new annotation, calibrated counts directly), or (ii) collecting **point-level supervision** (one frame per behavior instance, à la Lee & Byun 2021) and training a lightweight per-frame head on frozen V-JEPA-2 features. (ii) gives more lift per annotation hour than full segment annotation but still requires new labels.

**Implication for PoC:** **Simple-baseline-first.** Calibrate the existing V-JEPA-2 clip probe per-behavior, apply hysteresis threshold + temporal NMS over the sliding-window stream, count events. Escalate to a learned per-frame head only if Stage-1.5 LOSO verification shows count-vs-vet error breaches κ=0.7. **Crucial open question** (cross-cuts Q5): if RHpE is clinically scored on per-behavior **presence/absence** rather than counts, the bridge collapses entirely — most-window-above-threshold = "behavior present" is sufficient.

---

## Q5 — Interpretability and clinical deployment

**Deployed equine pain AI is a single product category — Sleip — and it does not automate the RHpE.** Sleip (commercial smartphone markerless gait/lameness; [Hernlund et al. 2023 *Animals*](https://pubmed.ncbi.nlm.nih.gov/36766279/), peer-reviewed, ~2.2 mm vertical asymmetry agreement vs multi-camera mocap) detects lameness asymmetry, not the 24-behavior RHpE. **No peer-reviewed product automates Dyson's RHpE in clinic.** Adjacent species offer a sobering base rate: [Stygar et al. 2021](https://www.frontiersin.org/journals/veterinary-science/articles/10.3389/fvets.2021.634338/full) systematic review found that of **129 commercial dairy sensor products, only 14% had any external peer-reviewed validation**.

**The deployment-readiness reference is Pacholec et al. 2025 *Vet Clin Pathol* Parts I & II** ([PMID 39638756](https://pubmed.ncbi.nlm.nih.gov/39638756/), [PMID 39843399](https://pubmed.ncbi.nlm.nih.gov/39843399/), verified), which formalises FUTURE-AI/DECIDE-AI for veterinary AI: **calibration and run-time monitoring are mandatory; explainability is explicitly optional**. On explainability, the only veterinary-specific empirical evaluation is [Müller et al. 2025 (PMC12350829)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12350829/): 11 saliency methods rated by 3 vets on 7,362 X-rays — mean usefulness 2–3/5, **CAMs deemed "insufficient for reliable day-to-day integration."** Saliency-as-trust is empirically weak.

**Failure modes documented.** (i) Site shift / dataset shift (Pacholec Part II flags this as the #1 vet-AI failure mode); (ii) calibration drift decoupling predicted probabilities from base rates ([arXiv:2506.17442](https://arxiv.org/abs/2506.17442)) — RHpE's ≥8/24 threshold would silently change meaning; (iii) trust erosion on disagreement ([adoption paradox PMC13012951](https://pmc.ncbi.nlm.nih.gov/articles/PMC13012951/), 71% AI usage, 54.3% cite reliability concerns); (iv) breed/discipline-specific behaviors flagged as pain (Iberian, dressage); (v) hard-surface lameness false alarms.

**Clinician-accepted explanation surfaces (peer-reviewed evidence):** frame-level evidence citation (per-behavior occurrence list with click-to-clip) **accepted** — matches Dyson's manual workflow; per-behavior probability + reliability diagram **recommended** by Pacholec Part II as table-stakes; anatomy-aligned (concept/prototype-based) attribution **preferred** over saliency; **frame saliency / GradCAM rejected** as primary evidence (Müller 2025); auto-generated LLM rationales out — GDPR-violating + hallucination risk.

**Implication for PoC:** Prioritise **frame-citation explanation** (per-behavior occurrence list with timestamp click-to-clip + per-behavior probability + reliability diagram). This is the only explanation interface with positive empirical reception in the veterinary AI literature, costs near-zero on top of the existing pipeline, and mirrors how vets already audit RHpE manually.

---

## Q6 — Validation methodology beyond LOSO

**The 2024–2026 medical-AI reporting consensus treats six items as load-bearing for clinical-utility claims.** [TRIPOD+AI](https://pubmed.ncbi.nlm.nih.gov/38626948/) (Collins et al., *BMJ* 2024;385:e078378, verified), CLAIM 2024 Update ([PMC11304031](https://pmc.ncbi.nlm.nih.gov/articles/PMC11304031), Tejani et al.), STARD-AI (*Nature Medicine* 2025;31:3283-3289, Sounderajah et al.), and MI-CLAIM 2020 (*Nature Medicine* 26:1320-1324, Norgeot et al.) collectively require: (i) calibration via reliability diagram + ECE + Brier; (ii) per-class metrics with bootstrap CIs; (iii) external/multi-site testing or explicit justification; (iv) subgroup/fairness reporting; (v) inter-rater κ on the reference standard; (vi) sample-size justification.

**The peer-reviewed equine-pain-ML literature meets few of these.** Andersen et al. (*PLOS ONE* 2021, [PMC8525760](https://pmc.ncbi.nlm.nih.gov/articles/PMC8525760/), n=7 horses, single rater, no AUC/calibration). Broomé et al. CVPR 2019 / *PLOS ONE* 2022 — no calibration, no κ. **No direct precedent to inherit.** RHpE methodology papers report κ per behavior — Garcia 2023 Icelandic pilot (*Equine Vet Educ* 13803) is the empirical floor: **14/22 substantial-to-perfect, 6/22 moderate, 1/22 fair**. Behaviors with κ in the 0.4–0.7 band cannot be defended as gold-standard targets under TRIPOD+AI.

**Statistical practice.** **DeLong is underpowered at small N** (Lee 2013 PMC3684152: ~0.24 power vs ~0.89 hierarchical bootstrap at n≈53 events). Use **permutation (Venkatraman–Begg) or hierarchical bootstrap** for paired AUC. For 24 simultaneous binary tests, **Benjamini–Hochberg FDR** is more powerful than Bonferroni and is the modern standard. For calibration at N≈300, **Platt or temperature scaling beats isotonic** (Ojeda 2023 *Stat Med* 10.1002/sim.9921 — isotonic needs N≳1000).

**Solo-shippable:** per-behavior AUC + bootstrap CI; reliability diagram + ECE + Brier; temperature scaling; source-aware LOSO (already in pipeline); BH-FDR; permutation tests; TRIPOD+AI/CLAIM checklist. **Collaboration-required:** multi-rater κ on full corpus; held-out cross-country/clinic test; prospective deployment.

**Implication for PoC:** The single most load-bearing missing step is **multi-rater κ on at least a 20% audit subset (≥2 vets)**. Without it, every per-behavior AUC reported is upper-bounded by single-rater label noise and cannot be defended under TRIPOD+AI or STARD-AI. Calibration at the ≥8/24 operating point is the second-most load-bearing and is **solo-shippable in one day**.

---

## Cross-cutting synthesis & ranking for next 6 months

**Investment-worthiness ranking (highest → lowest):**

**1. Q6 — Validation methodology.** Highest leverage. Without multi-rater κ on an audit subset, no validation claim survives TRIPOD+AI/STARD-AI review and the entire downstream stack (Q2 backbone choice, Q3 architecture, Q4 long-form pipeline) reports inflated metrics against noisy labels. Solo-shippable items (calibration, BH-FDR, permutation tests, TRIPOD+AI checklist) cost ~1 week and immediately upgrade the project's publication ceiling.

**2. Q1 — Datasets / collaboration.** Binding constraint. Public 24-behavior corpus does not exist; solo build is £40–60k. The only realistic path is **opening SLU/Uppsala collaboration now** (Rhodin/Ask published contacts) before locking architecture. This is a "send 3 emails this week" investment with a multiplier on every other axis.

**3. Q5 — Explanation surface.** Frame-citation interface is the single highest-leverage *technology* bet — clinician-accepted per peer-reviewed evidence (Müller 2025, Pacholec Part II), near-zero engineering cost on top of the existing V-JEPA-2 probe, and is the only explanation surface that distinguishes a research prototype from a "screening aid for vet review." Skip GradCAM/saliency entirely.

**4. Q4 — Long-form pipeline.** Simple-baseline-first (clip classifier + isotonic + threshold + NMS + count) operationalises the validated AUC into a session-level deliverable in 1–2 days. Defer learned TAL until point-level supervision is justified.

**5. Q3 — Multi-behavior architecture.** Mostly answered: 24 independent linear probes, sigmoid + per-probe calibration, source-aware LOSO. Decision is invariant to whether the PoC ships 24 or 6–8 behaviors.

**6. Q2 — Backbone A/B.** Lowest leverage in the next phase. Run V-JEPA-2 ViT-g vs ViT-L *once*, after Q3/Q4 are operational. The ≤+5–7 pt SSv2-frozen gap is plausibly smaller than gains from calibration + threshold tuning + correct multi-rater labels.

**Single highest-leverage technology bet: the simple long-form pipeline + frame-citation explanation surface (Q4 + Q5 combined).** It turns the existing AUC 0.87–0.90 clip result into a working session-level deliverable that vets can audit clip-by-clip, with zero new annotation and ~3 days of work.

**Single highest-leverage methodology bet: multi-rater κ on a 20% audit subset (Q6).** Without it, no metric in the project is publishable under 2024–2026 medical-AI reporting standards. This is also the work that *requires* solving Q1's collaboration problem first — making the methodology bet and the dataset bet structurally linked.

**Cross-cutting dependencies:**
- **Q1 (collaboration access) gates Q6 (multi-rater κ).** Solo maintainer cannot meet the TRIPOD+AI bar without a vet co-rater. This is the project's central bottleneck, not architecture.
- **Q5 may collapse part of Q4.** If RHpE is clinically scored on presence/absence rather than counts, the long-form bridge reduces to thresholding (sub-day work) and learned TAL becomes irrelevant. Confirm with collaborating vet before building counting infrastructure.
- **Q2 and Q3 are weakly coupled at this N.** Architecture (independent probes) is invariant to backbone; backbone (ViT-L floor + one ViT-g A/B) is invariant to architecture. Run independently.
- **Q6 over-rules Q2.** If multi-rater κ shows ~7/24 behaviors at κ<0.7, those must be flagged or dropped — changing the aggregate ≥8/24 claim and feeding back into how independent-probe outputs pool into the session score.

---

## Open Questions (survey-irreducible — require experimental investigation)

- **Is RHpE clinically scored as count-sensitive or presence/absence?** Dyson's threshold uses ≥8/24 *distinct* behaviors. If clinical signal is per-behavior presence over a session, the Q4 bridge problem collapses. Resolvable only via a vet collaborator's review of clinic workflow.
- **Should the PoC scope to 6–8 high-κ behaviors instead of 24?** Garcia 2023 Icelandic κ distribution (14/22 substantial, 6/22 moderate, 1/22 fair) suggests rare/low-κ behaviors may not be defensible targets. Architectural answer is invariant; clinical claim is not.
- **Will Phelipon 2025's corresponding author release the 1,036-image set under research licence?** Fastest legitimate bootstrap. Resolvable by direct contact.
- **Can per-source isotonic calibration fit with 12 LOSO sources, or does one global calibrator suffice?** Empirical.
- **Are RHpE behaviors strongly correlated** (tail swish + ear pinning co-occur)? Not in published literature; computable on PoC training set early. Decides whether multi-head MLP plausibly beats independent probes.
- **Is Sleip's published validation as independent as it appears?** Company-curated list mixes Sleip-affiliated and non-affiliated authorship; distinguish before citing as prior art.

---

*Appendix sub-reports: `q1_datasets.md` … `q6_validation.md`. URL audit: `url-verification.md`.*
