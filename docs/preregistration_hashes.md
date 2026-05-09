# Pre-registration audit hashes

The pre-registration documents and per-clip annotations are kept in
`poc/outputs/` and so are not versioned in git (per project `.gitignore`
policy on inference outputs and ad-hoc methodology docs). To keep the
audit trail real — i.e., to prove the documents existed at a specific
content state before any unblinded run that depends on them — this file
records SHA-256 hashes and freeze timestamps. The hashes are committed to
git in this versioned file; mismatching the recorded hash later means
the document was edited after freeze.

The protection this gives:

- **Existence + content lock at freeze time.** The hash + commit timestamp
  proves the document was in this exact state at this exact moment, with
  the commit serving as the witness.
- **Tamper-evidence.** If a future re-run or downstream audit recomputes
  the hash and gets a different value, the change is detectable.

Verify any time with:

```bash
shasum -a 256 poc/outputs/<filename>
shasum -a 256 poc/docs/<filename>
```

## Frozen documents

| Frozen at | SHA-256 | Bytes | Path | Purpose |
|---|---|---|---|---|
| 2026-05-08 | `e1ced120e6d5116347809a85d38be51fab4dec2965488e0f176d73a9630b7d6f` | 5594 | `outputs/eye_probe_preregistration_minicpm.md` | Pre-registration for MiniCPM-V 4.5 sanity (Lesson 18 evidence). Frozen BEFORE the 6-clip MiniCPM run. |
| 2026-05-08 | `fc647963f2b9a8e9d7256811e13a0375c6081ce95beb5bc9f21e8a51aa2003c0` | 5569 | `outputs/expected_diagnostic_minicpm_blink.json` | Pre-committed per-clip predictions for the 6-clip blink sanity test. Frozen BEFORE the MiniCPM run. |
| 2026-05-08 | `6b3791a49eb84b77d27e450ffed08bee4a807feac97663eb09d18d6c5c355766` | 4813 | `outputs/track_b_phase1_preregistration.md` | Pre-registration for Track B Phase 1 + locked v2 profile-aware fallback rule. Frozen BEFORE the eye-crop pipeline run. |
| 2026-05-08 (post-review) | `18a5754f9e247b992b9c578e49726cc3cab1c77db28e6eff5bc78ccac41994ea` | 7201 | `outputs/eye_crops_annotations.md` | Per-clip eye-visible Y/N + post-review corrections (drift accounting, threshold disambiguation, parity clarification, embedding-count fix). Frozen BEFORE Phase 3 LOSO LR. |
| 2026-05-08 | `9b391640ded3aeaf9efaed236756ad1c4a4f53fa7c87a57ef3de480ab08f192a` | 6255 | `outputs/eye_verification_clips.txt` | User-provided eye labels for the 36-clip RME stratified subset (blind verification pass). Frozen post-collection, BEFORE Phase 3 LOSO LR run. |
| 2026-05-08 | `722b916cc29f46bef33fcb9f8b7bd37ef1acf08ac804affc7b2d11c06c162346` | 7518 | `docs/phase3_auc_method.md` | Phase 3 method + result audit doc — locks pooled-AUC primary, DeLong CI, permutation design, and decision-per-pre-reg. Frozen at run time alongside `outputs/eye_loso_results.json`. |
| 2026-05-09 | `b92bc109975ed8e73224505d59fb410d8fc17e079713f7a9787a90a39ee4b6e0` | 15758 | `outputs/eye_probe_results.md` | Track A writeup — **corrected** inverted-fold diagnostic for S5/S6 from post-fix per_clip data, plus newly-identified fourth contributing factor (BG clips ranked above ACT clips on source-correlated training features). Phase 3 numbers and decision unchanged. Earlier hashes `915f48d3...`, `5c681bfb...`, `9a4c1673...` superseded. |
| 2026-05-09 | `bc5c18441c6be8107fe7741339dc908ac1e1a80469886c065d2059aaee277727` | 7434 | `docs/phase3_per_clip_correction.md` | Phase 3 per_clip alignment correction — bug history, scope of impact (pooled metric and decision unaffected), bit-exact reproduction proof, regression test pin. Versioned in `docs/`. |
| 2026-05-09 | `6fc5451f819b27e0fc3d259ae0f388196a96b33e3fcb224fa18d7070b6ffbe62` | 7437 | `outputs/eye_loso_results.PRE_FIX.json` | Pre-fix Phase 3 LOSO output preserved as bug evidence (corrupted per_clip; pooled metrics correct). |
| 2026-05-09 | `70aedd173b6d367ed4944a5a4d07fc5562768cad32798ed9c00e753d1a6a9489` | 7437 | `outputs/eye_loso_results.json` | Post-fix Phase 3 LOSO output — same pooled AUC 0.6813, CI, p-value as pre-fix, with corrected per_clip alignment. |
| 2026-05-09 | `7a311dbdf8e442bfbdfa23efe133a052129a76a12441d10b2858e2555501bbf3` | 12969 | `docs/phase4_audit.md` | Phase 4 audit doc — primary AUC 0.5854 (regression), Ablation A 0.5857 (relabel hurts), Ablation B 0.5536 (v2 crop hurts), factor-(d) SUPPRESSED 2/3 below median, contact-sheet inspection flagged the v2 heuristic failure pre-run, Phase 5 design space narrowed. |
| 2026-05-09 | `05c01ec40366f97e71ae7d70442f7a32d72c8452043a6de1c69ea4724a8e35f3` | 17799 | `outputs/eye_probe_results.md` | Track A writeup — Phase 4 result section added at end. Phase 3 numbers and decision unchanged. Earlier hashes `915f48d3...`, `5c681bfb...`, `9a4c1673...`, `b92bc109...` superseded. |
| 2026-05-09 | `6f99cd9236dd88dc738140c208070330cb9dbccd1a95f7f5916abee86d10a029` | 10809 | `outputs/eye_loso_results_phase4.json` | Phase 4 primary LOSO result. Pooled AUC 0.5854, regression triggered, factor-(d) suppressed (2/3 below median). |
| 2026-05-09 | `ad017ff62205eaa1611971d15382cf6575eb3b62a8adebf0d6373cbb5aef1b4d` | 9157 | `outputs/eye_loso_results_phase4_ablation_A.json` | Ablation A (v1 crops + tightened labels) — isolates relabel effect. AUC 0.5857. |
| 2026-05-09 | `2e0706b3811dccfd02c39e926775ea443401f048141764349b9195d9dabd217c` | 10239 | `outputs/eye_loso_results_phase4_ablation_B.json` | Ablation B (v2 crops + original labels) — isolates v2 crop effect. AUC 0.5536, deepest regression. |
| 2026-05-09 | `5c7ca4d1f8fa7339b1695331f24ffdc8407c60c30366c4a3f763d67cc1cac315` | 18711 | `outputs/eye_crops_v2_manifest.jsonl` | v2 crop pipeline per-clip metadata: 34 clips → 38 outputs (4 ties × 2 halves), L/R scores, decisions. |
| 2026-05-09 | `33f22d0595eefc791ace8ef51de28699cc26dcb72a573f7a1ba406c2b5c19878` | 2396275 | `outputs/eye_crops_v2_contact_sheet.png` | v2 contact sheet — visible heuristic-failure evidence (ears captured rather than eyes). |
| 2026-05-09 | `2d253ae034e10d0b39d3b7bc1ceb15610d83ed1204a55f7020764d1f6bcf724f` | 15628 | `outputs/track_b_phase5_preregistration.md` | **Phase 5 pre-registration**, frozen BEFORE annotation. Locks: gold-standard manual eye-region bbox annotation (3 frames per clip with IoU-based interpolation), 2×2 cropping × labels factorial, 4-anchor margin curve {10,15,40,80}%, MDE-aware 3-disjoint-band primary thresholds (Top: Δ≥0.10 ∧ paired p<0.05; Middle: AUC≥0.6313 ∧ ¬Top; Regression: AUC<0.6313), sensitivity-1 3-row including positive-tail Δ>+0.085 (Phase 4 vindication band), sensitivity-2 4-shape noise-tolerant categorical (monotone-up/-down/inverted-U/flat), factor-(d) criterion same as Phase 4, intra-rater consistency on 5 clips with ≥4-6h gap (≥48h ideal deferred to Phase 6), Step 0a P3 reproduction sanity passed bit-exact, Step 0b MDE-80% ≈ 0.085. |
| 2026-05-09 | `8d658e608bb9feb06ed5cfc0de89f3cf2459a94b1ce2bd5596c1663d912b1730` | 2449 | `outputs/eye_box_keymap_phase5.json` | Phase 5 UUID → real-filename mapping (mask_seed=45). Hash committed BEFORE user opens annotation PNGs, binding the masking; user does not consult keymap until after annotation submission. |
| 2026-05-09 | `02563083d8e990734e37ce3623e585ac257990380a3bba9c01329ec41c4ebe29` | 87 | `outputs/eye_loso_results.PHASE5_REPRO_CHECK.json` | Step 0a — Phase 3 reproduction sanity, bit-exact pooled AUC 0.6813186813186813 confirmed before any Phase 5 work. Catches code drift in numpy/scikit-learn/V-JEPA-2 pipeline. |
| 2026-05-09 | `e193a359397b442d63078ae264f99c6f7cccfbf6b9013d5fd83742f3521b5df0` | 438 | `outputs/eye_box_keymap_phase5b.json` | Phase 5b intra-rater re-annotation keymap (mask_seed=46, distinct from Phase 5a's seed=45). 5 randomly-selected clips with NEW UUIDs so memory of (Phase 5a UUID → eye position) doesn't carry across the ≥4-6h gap. Hash committed BEFORE user opens the re-annotation; audit can verify mapping wasn't edited after the unblinded re-annotation. The 5 selected real clips include `action_S6.mp4_2_` — one of the 3 persistent factor-(d) BG-target clips, by random chance — so we'll have IoU data on a critical clip. |
| 2026-05-09 | `dd36072b19bdde482d00bca2deb5b149db191cefab1483c418e71d915a9efae5` | 17057 | `docs/phase5_audit.md` | Phase 5 audit doc — primary AUC 0.7985 (Δ vs P3 = +0.117, paired-DeLong p = 0.286, perm p = 0.010, bootstrap CI [0.584, 0.964]) → MIDDLE band per locked rule. Sensitivity 1 Δ = +0.019 (clean) → Phase 4 rubric direction vindicated. Sensitivity 2 (margin curve {10,15,40,80}%) = FLAT under noise tolerance. Factor (d) suppressed (2/3 below median). Intra-rater median IoU = 0.765 (gate cleared at 0.6). **Includes locked pre-result interpretation for `background_S8.mp4_3_` IoU outlier (perceptual-floor + catchlight-confounder)** and locked alternative interpretation for the margin curve (catchlight dilution as candidate mechanism for any future monotone-improving shape at higher N). Earlier hash `355ceff9...` superseded — interpretation lock added before any per-fold S8 inspection. |
| 2026-05-09 | `4d601ff6eca4899eac09b210cf7b467400f9670ba96c372389eb1b765732a39e` | 20768 | `outputs/eye_probe_results.md` | Track A writeup — Phase 5 result section with Middle-band locked sentence + locked S8/catchlight pre-registration. Earlier hashes `915f48d3...`, `5c681bfb...`, `9a4c1673...`, `b92bc109...`, `05c01ec4...`, `81c8b72e...` superseded. |
| 2026-05-09 | `275ab92e5bd2141d02fd5205881ba57cd979880e070e2cb457c39872843f226a` | 8091 | `outputs/eye_boxes_phase5a.json` | Phase 5a primary annotations: 34 clips × 3 frames = 102 tight eye-region bounding boxes. 0 no_eye_visible. |
| 2026-05-09 | `488f2c66daeb1e7b53e94f7587417ed7c3f93fd466340e67d85209e939afef60` | 1193 | `outputs/eye_boxes_phase5b.json` | Phase 5b intra-rater re-annotations: 5 clips × 3 frames = 15 boxes after ≥4-6h gap. |
| 2026-05-09 | `3697a4a448922733aebeb4bd6fbdd562e63a00b22870cfdadce8e2449657156a` | 1987 | `outputs/phase5_intra_rater_iou.json` | Phase 5 intra-rater IoU result: median 0.765 across 15 frames; min 0.533, max 0.877. Locked gate at 0.6 cleared → "gold-standard" framing supported. |
| 2026-05-09 | `ed478c6e87e53ba2bebd4fab9dbaaa0e08f520cf9098fdd8e375511fb1967daa` | 8280 | `outputs/eye_loso_results_phase5_primary.json` | Phase 5 primary LOSO: v3 (m=15) + original labels. AUC 0.7985, Δ vs P3 +0.117, decision Middle. |
| 2026-05-09 | `9a08d23a55e7bdb8ba601c44bead0f547022aa88448f61f363e0cb03125e8de6` | 7400 | `outputs/eye_loso_results_phase5_sens_rubric.json` | Phase 5 sensitivity 1: v3 (m=15) + tightened labels. AUC 0.8179, Δ vs primary +0.019 (within ±0.085 MDE → clean). |
| 2026-05-09 | `5cdb1dcf1de499626387b893e17111892ec9b450a1e536444db192216e961cc1` | 7363 | `outputs/eye_loso_results_phase5_sens_margin_10.json` | Phase 5 sensitivity 2 m=10%: AUC 0.7546. |
| 2026-05-09 | `0012761c59faf909c824efc5a27655837354c1660cee05137baf94dbe281bd38` | 7362 | `outputs/eye_loso_results_phase5_sens_margin_40.json` | Phase 5 sensitivity 2 m=40%: AUC 0.7473. |
| 2026-05-09 | `a7afd4eda5264d23830bf110f7741d08569f0e16616759059152dee8e0e4cc85` | 7365 | `outputs/eye_loso_results_phase5_sens_margin_80.json` | Phase 5 sensitivity 2 m=80%: AUC 0.7949. Margin curve verdict: FLAT under noise tolerance. |
| 2026-05-09 | `bc9d585fbee36bc39a0b5323a1897988099c8d5855eae465785fff22bed5ab67` | 14774 | `outputs/eye_crops_v3_m10_manifest.jsonl` | v3 cropping manifest m=10% — 34/34, 13 static, 21 interpolated. |
| 2026-05-09 | `6ccd4927c77408045eabf6157739e18e6d3ead1641b46a2d3de554e88e3a806e` | 14774 | `outputs/eye_crops_v3_m15_manifest.jsonl` | v3 cropping manifest m=15% (primary). |
| 2026-05-09 | `f0c5bce23a8410b76216267e47a4c6dfb54c40f5f8dc08f2f385004171a8c40c` | 14773 | `outputs/eye_crops_v3_m40_manifest.jsonl` | v3 cropping manifest m=40%. |
| 2026-05-09 | `2c3394cd2efc1011b5bfb1c216d9889081fd248a266b761e1544fa709bc69577` | 14774 | `outputs/eye_crops_v3_m80_manifest.jsonl` | v3 cropping manifest m=80%. |
| 2026-05-09 | `d274bc9ebdb7c0f8ddfaaabe46929c711e481e6b5cc2994d5e57e3612ddf7bab` | 146215 | `outputs/vjepa2_embeddings_eye_v3_m10.npz` | V-JEPA-2 embeddings v3 m=10%, parity cos=1.0. |
| 2026-05-09 | `1fc3f8e9d83d7e3a1a01513dc79489cccb979781084904f156fd64694ff60a5f` | 146215 | `outputs/vjepa2_embeddings_eye_v3_m15.npz` | V-JEPA-2 embeddings v3 m=15% (primary). |
| 2026-05-09 | `fe0690513e479f5b53d7b7561c21cce368c6cec7aa1af5690d792feeb69f178a` | 146215 | `outputs/vjepa2_embeddings_eye_v3_m40.npz` | V-JEPA-2 embeddings v3 m=40%. |
| 2026-05-09 | `df39930a37d9b70a191c64fbb66e9addeca7901a6654a41df9726379852ec45a` | 146215 | `outputs/vjepa2_embeddings_eye_v3_m80.npz` | V-JEPA-2 embeddings v3 m=80%. |
| 2026-05-08 | `eb2f1a15cd5347df1a328153ba6524192bbdd2caa7d89158c439d6ab9c6c7396` | 16073 | `docs/methodology_discipline_pattern.md` | **Six**-element discipline pattern (added element 6: empirical-anchor data-structure-dependent rules before pre-reg lock-in) plus sub-second-filter case study under anti-pattern #6 ("confidently-wrong dataset assumptions"). Earlier hashes `99fbe4f3...` and `8bd143a0...` superseded. |
| 2026-05-09 | `ced5cae66e54e187e957ef5b21aea94c16c03eebc88e5436dea76b028bcf06c9` | 17884 | `outputs/track_b_phase4_preregistration.md` | **Phase 4 pre-registration** — adds factor-(d) suppression criterion locked BEFORE Phase 4 LOSO run. Mechanical decision rule: ≥2 of 3 persistent BG-target clips (`action_S5.mp4_2_`, `background_S6.mp4_2_`, `background_S6.mp4_3_`) score strictly below median of all 20 Phase 4 BG clips → factor (d) SUPPRESSED; ≤1 → PERSISTENT. Tie-break-both-halves duplicates use max-score-per-clip convention. Earlier hashes `0b970228...`, `35607470...` superseded by this revision. Numerical thresholds and 4-band decision rule unchanged from `0b970228...` (post-unmask interpretation lock + post-unmask factor-(d) operationalization stack on top). |
| 2026-05-08 | `8bb065d763f4b01995c06ce9687b215066d7f9c0625d9843cae81296f95b504d` | 1342 | `outputs/eye_relabel_blind.txt` | **Phase 4 masked re-label file** with locked tightened rubric ("ACTION = visible at native resolution at normal-speed playback; no zoom, no frame-stepping"). 34 entries `clip_M001`...`clip_M034`, randomized `mask_seed=44`. Hash committed BEFORE user opens the file, binding the rubric content. |
| 2026-05-08 | `2ee51a27bf33312d26722b4c7170385a8e5e1bc195eed39ca362da6be9633c48` | 1531 | `outputs/eye_relabel_keymap.json` | **Phase 4 clip-mask → real-filename mapping**. Hash committed BEFORE user opens the masked file, so the audit can prove the mapping was not edited after the unblinded re-label. User does not consult this file until after re-label submission. |
| 2026-05-08 | `fbf392eb36efe04161eccffc59fbb1320bb6f632d1fb2846e5844d35b5d15206` | 458 | `outputs/phase3_subject_bootstrap_ci.json` | Subject-bootstrap CI on Phase 3 predictions (n=2000 source-resamples, seed=42). 95 % CI [0.4138, 0.8980], 9 pp wider lower bound than DeLong. Computed retrospectively as methodological adjunct; Phase 3 numbers and decision unchanged. |

## Notes

- The MiniCPM pre-registration and predictions JSON were frozen earlier on the same day as the run; the audit trail for those is "this commit verifies the file was in this state before the LOSO/sanity it gates."
- The Track B Phase 1 pre-registration was frozen before the Phase 1 pipeline ran. The v2 fallback spec inside it is the binding rule for what happens if Phase 3 LOSO AUC lands in the ambiguous 0.55–0.65 zone.
- The annotations file was edited once after the initial Phase 2 run to add a corrections section addressing four reviewer findings (count discrepancy, drift accounting, threshold ambiguity, parity-test clarification). The hash here is the post-correction state, frozen before Phase 3.
- `eye_verification_clips.txt` has been at this content state since the user provided the completed labels before Phase 3 launched. The hash makes the labels-as-given immutable for any future re-run.
- `docs/phase3_auc_method.md` is unique among these files in that it is versioned (in `docs/`, not `outputs/`). Including its hash here is belt-and-braces — git has the version history but the hash provides the same tamper-evidence as for the gitignored docs.
