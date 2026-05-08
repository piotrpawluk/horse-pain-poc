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
| 2026-05-08 | `5c681bfbda3f111fa5528786e93da4ceaaf041e50eb870eca3281dfe69f42a2a` | 12539 | `outputs/eye_probe_results.md` | Track A writeup — Phase 3 pilot result with locked headline order (AUC → CI → p), per-fold spread as load-bearing diagnostic, **inverted-fold diagnostic for S5/S6 from post-Phase-3 manual inspection** (3 named contributing factors, no Phase 3 modifications), claim-scope sentence, reproducibility paragraph. Earlier hash `915f48d3...` superseded by this revision. |
| 2026-05-08 | `8bd143a0c729daca50af5698970f966a61bf12d6b1922f2f0c66266544815fed` | 12630 | `docs/methodology_discipline_pattern.md` | Five-element discipline pattern + crop-inspection case study under "result-conditioned heuristic revision" anti-pattern. Versioned in `docs/`. Earlier hash `99fbe4f3...` superseded by this revision. |

## Notes

- The MiniCPM pre-registration and predictions JSON were frozen earlier on the same day as the run; the audit trail for those is "this commit verifies the file was in this state before the LOSO/sanity it gates."
- The Track B Phase 1 pre-registration was frozen before the Phase 1 pipeline ran. The v2 fallback spec inside it is the binding rule for what happens if Phase 3 LOSO AUC lands in the ambiguous 0.55–0.65 zone.
- The annotations file was edited once after the initial Phase 2 run to add a corrections section addressing four reviewer findings (count discrepancy, drift accounting, threshold ambiguity, parity-test clarification). The hash here is the post-correction state, frozen before Phase 3.
- `eye_verification_clips.txt` has been at this content state since the user provided the completed labels before Phase 3 launched. The hash makes the labels-as-given immutable for any future re-run.
- `docs/phase3_auc_method.md` is unique among these files in that it is versioned (in `docs/`, not `outputs/`). Including its hash here is belt-and-braces — git has the version history but the hash provides the same tamper-evidence as for the gitignored docs.
