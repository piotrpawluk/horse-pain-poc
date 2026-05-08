# Phase 3 per_clip alignment correction

**Filed:** 2026-05-09, after Phase 3 numbers were already locked and committed (`6f4352e`) but before Phase 4 launched.

**Scope:** A class of bug — silently-misaligned arrays where each component is internally correct but the post-loop composition pairs them wrong — was identified in `tools/eye_loso_lr.py` after the user's blind Phase 4 re-label submission produced a label diff vs Phase 3 that didn't match what the verification file said. Investigation traced the discrepancy to corrupted per-clip data in `outputs/eye_loso_results.json`, not a labels-file change.

## What the bug was

`loso_pooled()` accumulated `all_preds` and `all_truth` in **LOSO traversal order** (S1 clips, S2 clips, ..., S12 clips, in the order each fold was processed). The post-loop construction of `per_clip` then iterated over `aligned_filenames` in **alphabetical sort order** and paired `aligned_filenames[i]` with `truth[i]` and `preds[i]`:

```python
# pre-fix code:
per_clip = [
    {"clip": aligned_filenames[i],   # ← alphabetical order
     "label": int(truth[i]),         # ← LOSO traversal order
     "score": float(preds[i])}       # ← LOSO traversal order
    for i in range(len(aligned_filenames))
]
```

The two orderings are different (e.g., the alphabetically-first clip was `action_S10.mp4_0_.mp4` with source S10, while the LOSO traversal started with source S1's clips first). All 34 of 34 per-clip entries had the wrong label and score paired with the wrong clip name.

## What was *not* affected

The pooled metric and decision were entirely unaffected because they operate on parallel arrays:

- `all_preds` and `all_truth` were appended in the *same* LOSO loop iteration. So the (truth, score) pairs are correctly aligned with each other, just internally — both are in LOSO traversal order.
- `roc_auc_score(all_truth, all_preds)` paired correctly → **pooled AUC = 0.6813** correct.
- `delong_ci(all_truth, all_preds)` paired correctly → **DeLong CI [0.4866, 0.8760]** correct.
- Subject-bootstrap CI uses the source-grouped predictions, derived from the LOSO loop output → **bootstrap CI [0.4138, 0.8980]** correct.
- Per-fold AUCs use `roc_auc_score(y[test_idx], p)` *inside* the loop, never touching `aligned_filenames` → **per-fold AUC distribution {min 0.000, median 1.000, max 1.000, n_defined 8, n_skipped 4}** correct.
- Permutation null distribution → unaffected, same parallel-array property → **p = 0.0579** correct.
- `decision_per_pre_reg = ">=0.65"` correct.

The corruption was exclusively in the post-loop *join* between names and values, which is a derived artifact (the per-clip JSON), not the primary metric pipeline.

## Verification: pooled metrics reproduce bit-exactly

After the fix, re-running `eye_loso_lr.py` on the same labels and same embeddings produced:

| Quantity | Pre-fix value | Post-fix value | Match |
|---|---|---|---|
| pooled_auc | 0.6813186813186813 | 0.6813186813186813 | ✓ exact |
| auc_95_ci_low (DeLong) | 0.4866050016311529 | 0.4866050016311529 | ✓ exact |
| auc_95_ci_high (DeLong) | 0.8760323610062097 | 0.8760323610062097 | ✓ exact |
| p_value (permutation) | 0.057942 | 0.057942 | ✓ exact |
| permutation_null_mean | 0.486513 | 0.486513 | ✓ exact |
| fold_dist | {min 0.0, median 1.0, max 1.0, n_def 8, n_skip 4} | identical | ✓ exact |

All 34 of 34 per_clip entries had updated label/score pairings vs the pre-fix file. The pre-fix artifact is preserved at `outputs/eye_loso_results.PRE_FIX.json` for audit comparison; the fixed artifact is at `outputs/eye_loso_results.json`.

## What was affected (and corrected)

- **`outputs/eye_loso_results.json` per_clip field** — regenerated with correct alignment. The label and score for every clip now correspond to that clip's actual labeled value and the prediction the model made on its embedding.
- **The "Inverted-fold diagnostic (S5/S6)" subsection in `outputs/eye_probe_results.md`** — the original version cited specific clips by name with attributed labels (e.g., it cited `action_S5.mp4_2_` as a "sub-pixel gaze change ACTION" but the verification file shows that clip was labeled BACKGROUND with observation "eyes still"). The narrative *substance* (S5/S6 inversion driven by sub-pixel labels + crop misposition + short clips) was sourced from the verification file and the user's manual crop inspection — both independent of the corrupted per-clip data — and survives. The specific named clips and the per-clip score rankings were corrected from the post-fix per_clip data. Notably, the corrected ranking revealed a fourth contributing pattern (BG clips ranked higher than ACT clips on source-correlated training features) that the original three-factor diagnostic did not name. Track A writeup updated accordingly.

## Fix applied

`tools/eye_loso_lr.py` now propagates clip identity through the LOSO loop alongside truth/score:

```python
def loso_pooled(X, y, groups, sources, aligned_filenames):
    aligned_arr = np.array(aligned_filenames)
    all_preds, all_truth, all_clips = [], [], []
    ...
    for source in sources:
        ...
        all_preds.extend(p.tolist())
        all_truth.extend(y[test_idx].tolist())
        all_clips.extend(aligned_arr[test_idx].tolist())   # ← new
        ...
    return (np.array(all_preds), np.array(all_truth),
            np.array(all_clips), fold_aucs, fold_log)

# per_clip now uses LOSO-order names matching truth/preds:
per_clip = [
    {"clip": str(clips_in_loso_order[i]),
     "source": extract_source(str(clips_in_loso_order[i])),
     "label": int(truth[i]),
     "score": float(preds[i])}
    for i in range(len(clips_in_loso_order))
]
```

## Regression test pinned

`tools/test_eye_loso_lr_alignment.py` runs the LOSO pipeline on synthetic data designed so that alphabetical filename order ≠ LOSO source order (the precondition that exposes the bug class). The test asserts that for every per_clip entry, its label matches the gold label of the clip it claims to be. The test fails loudly if any future refactor reintroduces the misalignment.

```
$ python tools/test_eye_loso_lr_alignment.py
test_clip_label_pairing_matches_loso_traversal ... ok
test_pooled_auc_independent_of_per_clip_order ... ok
Ran 2 tests in 0.030s
OK
```

The second test documents the property that the pooled AUC is invariant to per_clip alignment — present so the failure mode is recognized as "per_clip-only corruption, headline metric unaffected" if it ever recurs.

## Discipline-pattern element triggered

This is exactly the failure class element 3 of `docs/methodology_discipline_pattern.md` is about: **"catch implementation bugs in writing."** The bug looked correct line-by-line — `aligned_filenames[i]` is a valid clip name, `truth[i]` is a valid label — but the *composition* paired them wrong. The same shape as the precomputed-scaler leakage that was caught in pre-run code review. The lesson: composition-class bugs need both pre-run code review (element 3) AND regression tests (this addition) to be reliably caught.

The pre-registration discipline (element 5: never amend a closed phase with a new question) holds: Phase 3 numbers and decision are unchanged; this is a data-integrity correction on a derived artifact, not a Phase 3 modification. The audit chain is extended (this document + the regenerated per_clip JSON + the regression test) rather than rewritten.
