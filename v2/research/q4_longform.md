# Q4 — Long-form Video Handling: Clip-prob Streams → Per-behavior Counts

## SOTA Summary (one paragraph)

Modern temporal action localization (TAL) is dominated by anchor-free, multiscale-feature-pyramid transformers — **ActionFormer** (arXiv:2202.07925, ECCV 2022), **TriDet** (arXiv:2303.07347, CVPR 2023), and the surprisingly strong baseline **TemporalMaxer** (arXiv:2303.09055, 2023) that replaces self-attention with parameter-free max-pooling and still beats ActionFormer on THUMOS14 with ~3× fewer FLOPs. End-to-end query-based detectors (**TadTR**, arXiv:2106.10271, IEEE TIP 2022) and the older proposal-generation **BMN** (arXiv:1907.09702, ICCV 2019) remain reference points. For dense per-frame multi-label segmentation, **MS-TCN** (arXiv:1903.01945, CVPR 2019) and **ASFormer** (arXiv:2110.08568, BMVC 2021) are standard, with **DiffAct** (arXiv:2303.17959, ICCV 2023) the current diffusion-based SOTA on 50Salads/Breakfast/GTEA. Critically, all of these need frame- or segment-level temporal annotation. With clip-level labels only, the relevant literature is **weakly-supervised TAL (WSTAL)**: the **STPN** lineage (Nguyen et al., arXiv:1712.05080, CVPR 2018) uses video-level labels via MIL+attention, and the survey by Baraka & Mohd Noor ("Weakly-supervised temporal action localization: a survey", Neural Comput. Appl. 2022, doi:10.1007/s00521-022-07102-x) confirms that point-level supervision (Lee & Byun, "Learning Action Completeness from Points", arXiv:2108.05029, ICCV 2021 oral) closes most of the gap to fully-supervised at ~6× cheaper annotation. Counting-aware/repetition heads (e.g. RepNet-style) exist but are tuned for periodic actions, not heterogeneous ethogram behaviors.

## Cost-Benefit (M2 Max, clip-level labels, solo maintainer)

| Approach | Trainable on clip-level labels? | Sample-efficient at N=100–500? | M2 Max cost | Calibrated counts? |
|---|---|---|---|---|
| **Threshold + temporal NMS/hysteresis on V-JEPA-2 probs** | Yes (already done) | Yes | Hours | Yes, via Platt/isotonic on held-out source |
| **MS-TCN / ASFormer per-frame head on V-JEPA-2 features** | Only with pseudo-labels from clip probs | Marginal | 1–2 days impl | Needs post-hoc calibration |
| **ActionFormer / TriDet / TemporalMaxer** | No — needs segment annotation | No (built for THUMOS-scale) | 2–4 days impl, MPS-friendly (1D convs / pooling) | Counts via NMS, not native |
| **TadTR / BMN** | No | No | 3–5 days, CUDA-leaning | No |
| **WSTAL (STPN-style MIL+attention)** | Yes | Plausible | 2–3 days impl | Weak; thresholding still needed |
| **Point-supervised TAL (Lee 2021)** | Needs ~1 frame/instance label — feasible to add | Strong (designed for sparse labels) | 2–3 days impl | Better than pure MIL |
| **DiffAct** | No | No | Heavy, MPS-unfriendly | No |
| **Counting-aware losses (e.g. RepNet)** | Limited to periodic actions | N/A | N/A | Native counts, wrong domain |

**Published evals**: every TAL/segmentation method above is benchmarked on THUMOS14 / ActivityNet / EPIC-Kitchens / 50Salads / Breakfast — **none on animal-behavior or pain ethograms**. Generalization to RHpE is untested. The action-segmentation lineage (MS-TCN/ASFormer) is the closest in spirit because per-behavior occurrence counting maps cleanly onto per-frame multi-label segmentation followed by run-length counting.

The WSTAL survey (Baraka & Mohd Noor 2022) and the Lee 2021 point-supervision paper both reach the same empirical conclusion the action-detection community has internalized: **a well-calibrated clip classifier + threshold + temporal NMS/hysteresis is a hard-to-beat baseline**, and learned localization heads only meaningfully outperform it once segment- or point-level annotation is added. Several WSTAL papers explicitly report that simple thresholded score streams trail SOTA learned methods by only ~3–6 mAP points at IoU=0.5 — small enough that, with calibration, the simple baseline often wins on count accuracy (which is what RHpE needs).

## What this implies for the PoC

**Recommendation: simple-baseline-first.** Train the clip classifier (V-JEPA-2 + linear probe, already validated at AUC 0.87–0.90), produce sliding-window probability streams, and post-process with per-behavior isotonic calibration → hysteresis threshold → temporal NMS → event count. This is 1–2 days of work, fully MPS-compatible, doesn't require new annotations, and gives calibrated counts directly. Only escalate to a learned TAL/segmentation head (MS-TCN or TemporalMaxer-style 1D pooling on frozen V-JEPA-2 features) if Stage-1.5 verification shows count error >κ=0.7 against expert RHpE scoring on held-out sources — and even then, the WSTAL literature suggests adding **point-level labels** (one frame per behavior instance, à la Lee 2021) gives more lift per annotation hour than full segment annotation.

## Open Questions

- **Is RHpE actually count-sensitive, or threshold-sensitive?** Dyson 2018 thresholds at ≥8/24 *distinct* behaviors. If clinical signal is presence/absence per behavior over the 10–15 min session, then per-behavior **binary occurrence** (did it happen at all?) is sufficient and counting is over-engineering. Confirm with Q1/Q5 (clinical deployment, validation): does inter-rater agreement on RHpE reward count accuracy or only presence-detection? If the latter, the bridge collapses to "any window above threshold → behavior present" and TAL/segmentation become irrelevant.
- **Source-aware LOSO calibration**: per-source isotonic calibration assumes enough held-out source-specific clips; with 12 LOSO sources this may be thin. Worth empirically checking whether one global calibrator suffices.

## Sources (verified)

- ActionFormer — arXiv:2202.07925 (ECCV 2022) — peer-reviewed
- BMN — arXiv:1907.09702 (ICCV 2019) — peer-reviewed
- TriDet — arXiv:2303.07347 (CVPR 2023) — peer-reviewed
- TemporalMaxer — arXiv:2303.09055 (2023) — preprint
- TadTR — arXiv:2106.10271 (IEEE TIP 2022) — peer-reviewed
- MS-TCN — arXiv:1903.01945 (CVPR 2019) — peer-reviewed
- ASFormer — arXiv:2110.08568 (BMVC 2021) — peer-reviewed
- DiffAct — arXiv:2303.17959 (ICCV 2023) — peer-reviewed
- STPN (Nguyen et al.) — arXiv:1712.05080 (CVPR 2018) — peer-reviewed
- Lee & Byun, point-supervised TAL — arXiv:2108.05029 (ICCV 2021 oral) — peer-reviewed
- Baraka & Mohd Noor, WSTAL survey — Neural Computing and Applications 2022, doi:10.1007/s00521-022-07102-x — peer-reviewed

All commercial-tool flags: none — all peer-reviewed academic.
