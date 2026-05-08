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
| 2026-05-08 | `eb2f1a15cd5347df1a328153ba6524192bbdd2caa7d89158c439d6ab9c6c7396` | 16073 | `docs/methodology_discipline_pattern.md` | **Six**-element discipline pattern (added element 6: empirical-anchor data-structure-dependent rules before pre-reg lock-in) plus sub-second-filter case study under anti-pattern #6 ("confidently-wrong dataset assumptions"). Earlier hashes `99fbe4f3...` and `8bd143a0...` superseded. |
| 2026-05-09 | `35607470600b63796c9bda02f92e88c85c1dac5d87cc147d80bbe50e1be47c02` | 14885 | `outputs/track_b_phase4_preregistration.md` | **Phase 4 pre-registration** with interpretation lock for the fourth factor (source-correlated training-feature carryover) added BEFORE unmask. Numerical thresholds unchanged from earlier hash `0b970228...`; what's locked is interpretation of each band given the corrected Phase 3 diagnostic identified factor (d) as a residual structural confound not addressable by Phase 4's (a)+(b)+(c) interventions. The 0.65–0.72 band is explicitly framed as "Phase 4 succeeded on (a)(b)(c), factor (d) is residual" — NOT partial failure. Earlier hash `0b970228...` superseded by this revision. |
| 2026-05-08 | `8bb065d763f4b01995c06ce9687b215066d7f9c0625d9843cae81296f95b504d` | 1342 | `outputs/eye_relabel_blind.txt` | **Phase 4 masked re-label file** with locked tightened rubric ("ACTION = visible at native resolution at normal-speed playback; no zoom, no frame-stepping"). 34 entries `clip_M001`...`clip_M034`, randomized `mask_seed=44`. Hash committed BEFORE user opens the file, binding the rubric content. |
| 2026-05-08 | `2ee51a27bf33312d26722b4c7170385a8e5e1bc195eed39ca362da6be9633c48` | 1531 | `outputs/eye_relabel_keymap.json` | **Phase 4 clip-mask → real-filename mapping**. Hash committed BEFORE user opens the masked file, so the audit can prove the mapping was not edited after the unblinded re-label. User does not consult this file until after re-label submission. |
| 2026-05-08 | `fbf392eb36efe04161eccffc59fbb1320bb6f632d1fb2846e5844d35b5d15206` | 458 | `outputs/phase3_subject_bootstrap_ci.json` | Subject-bootstrap CI on Phase 3 predictions (n=2000 source-resamples, seed=42). 95 % CI [0.4138, 0.8980], 9 pp wider lower bound than DeLong. Computed retrospectively as methodological adjunct; Phase 3 numbers and decision unchanged. |

## Notes

- The MiniCPM pre-registration and predictions JSON were frozen earlier on the same day as the run; the audit trail for those is "this commit verifies the file was in this state before the LOSO/sanity it gates."
- The Track B Phase 1 pre-registration was frozen before the Phase 1 pipeline ran. The v2 fallback spec inside it is the binding rule for what happens if Phase 3 LOSO AUC lands in the ambiguous 0.55–0.65 zone.
- The annotations file was edited once after the initial Phase 2 run to add a corrections section addressing four reviewer findings (count discrepancy, drift accounting, threshold ambiguity, parity-test clarification). The hash here is the post-correction state, frozen before Phase 3.
- `eye_verification_clips.txt` has been at this content state since the user provided the completed labels before Phase 3 launched. The hash makes the labels-as-given immutable for any future re-run.
- `docs/phase3_auc_method.md` is unique among these files in that it is versioned (in `docs/`, not `outputs/`). Including its hash here is belt-and-braces — git has the version history but the hash provides the same tamper-evidence as for the gitignored docs.
