# Q3 — Multi-behavior architecture for 24 RHpE detectors over frozen V-JEPA-2

## SOTA summary
For multi-label fine-grained behavior classification at small N over frozen self-supervised features, the dominant peer-reviewed pattern is **per-class binary heads with sigmoid cross-entropy on shared frozen features** — i.e., binary relevance, option (a)/(b). The AVA baseline (Gu et al., CVPR 2018, arXiv:1705.08421) explicitly switched to **per-class sigmoid loss** for 80 atomic actions with frequent multi-label co-occurrence — the canonical reference. DeepEthogram (Bohnslav et al., *eLife* 2021;10:e63377), the closest analogue to RHpE, uses **independent per-behavior binary heads** and reports >90% accuracy with as few as ~80 example frames per behavior, validating small-N viability on shared frozen CNN features. Standley et al., *Which Tasks Should Be Learned Together?* (ICML 2020, arXiv:1905.07553) and Crawshaw's MTL survey (arXiv:2009.09796) both document shared-trunk MTL as fragile: tasks compete, negative transfer is common, and groupings must be empirically discovered — costs the PoC cannot absorb at ~300 clips/task. Ruder (arXiv:1706.05098) notes hard parameter sharing regularizes mainly when features are not yet learned; with V-JEPA-2 frozen and strong, that argument evaporates.

## Cost-benefit
- **(a) 24 independent linear probes** — M2 training: minutes (24 sklearn fits). Calibration: best; per-probe isotonic/Platt directly supports ≥8/24 thresholding. Complexity: lowest. Risk: ignores label correlation, but RHpE is scored as independent occurrence counts (Dyson 2022, *Equine Vet Educ*).
- **(b) Multi-head MLP on shared frozen features** — Modest expressivity gain; can break per-head calibration without per-head temperature scaling. Justifiable only if (a) underfits.
- **(c) Shared trainable trunk + 24 heads** — Expected negative transfer at 300×24 with V-JEPA-2 already strong (Standley 2020); adds gradient-balancing complexity. Solo-maintainer cost: high.
- **(d) Sequence labeling / behavior-as-token transformer** — Needs dense temporal labels (PoC has clip labels); underpowered at N≈300/behavior. Reject.

## What this implies for the PoC
**Start with (a): 24 independent linear probes** on frozen V-JEPA-2 ViT-L features, sigmoid per behavior, per-probe isotonic calibration, source-aware LOSO. This mirrors AVA and DeepEthogram precedent, preserves the calibrated ≥8/24 workflow, and is the only option a solo maintainer can ship and debug behavior-by-behavior. Escalate to (b) only if specific behaviors demonstrably underfit. Best-in-principle could plausibly be (b) with shared task-similarity discovery, but for solo-maintainer start, (a) dominates.

## Sources (URLs verified via WebFetch)
- Gu et al., AVA, CVPR 2018 — arXiv:1705.08421 (peer-reviewed)
- Bohnslav et al., DeepEthogram, *eLife* 2021;10:e63377 (peer-reviewed)
- Standley et al., ICML 2020 — arXiv:1905.07553 (peer-reviewed)
- Crawshaw, MTL Survey, 2020 — arXiv:2009.09796 (preprint)
- Ruder, MTL Overview, 2017 — arXiv:1706.05098 (preprint)
- Dyson, RHpE, *Equine Vet Educ* 2022, doi:10.1111/eve.13468 (peer-reviewed)
- Dyson & Pollard, *Animals* 2022 — PMC8909886 (peer-reviewed)
- arXiv:2003.10474 from brief is unrelated (Discontinuous Galerkin / fractional diffusion). Dropped.

### Open Questions
- Published RHpE behavior-correlation matrices not located; compute co-occurrence on the PoC's own training set early. If 6–8 ear/mouth/eye-driven behaviors carry most signal, Q5/Q6 may consider phased 6–8 → 24 rollout. Architectural answer (independent probes) is invariant to that choice.
- Per-behavior calibration under source-aware LOSO with rare behaviors (<30 positives/source) may need pooled isotonic or empirical-Bayes calibration — flag for Q6.
