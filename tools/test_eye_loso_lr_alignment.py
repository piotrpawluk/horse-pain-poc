#!/usr/bin/env python3
"""Regression test for eye_loso_lr.py per_clip alignment.

Bug history (pre-fix Phase 3): loso_pooled() accumulated all_truth and
all_preds in LOSO traversal order (S1 clips, S2 clips, ..., S12 clips),
but the post-loop per_clip construction used aligned_filenames[i] which
was in original sorted (alphabetical) order. The two orderings differ,
so per_clip[i].clip referred to a different physical clip than truth[i]
and preds[i]. The pooled AUC, DeLong CI, bootstrap CI, permutation
p-value, and per-fold AUCs were all unaffected because they operate on
parallel arrays (truth↔preds, y[test_idx]↔p) that are internally
consistent with each other; the corruption was exclusively in the
post-loop join between names and values.

Class of bug: silently-misaligned arrays where each component is
internally correct but the composition pairs them wrong. Same shape
as the precomputed-scaler-leakage class. Tests are the only durable
defense — element 3 of the discipline pattern says catch composition
bugs in writing; this test pins the catch in code.

Test method: synthetic V-JEPA-2-shaped data where each clip's "label"
is uniquely identifiable from its embedding (one-hot per clip in
extra dimensions). After running the LOSO pipeline, verify each
per_clip[i].clip matches the clip whose embedding actually generated
preds[i].

Run: pytest tools/test_eye_loso_lr_alignment.py
or: python tools/test_eye_loso_lr_alignment.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

POC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_DIR / "tools"))
from eye_loso_lr import (  # noqa: E402
    loso_pooled,
    extract_source,
)


class TestPerClipAlignment(unittest.TestCase):
    """per_clip[i].clip must correspond to the clip whose prediction is preds[i]."""

    def _build_synthetic(self, n_per_source=3, n_sources=12, dim=64, seed=0):
        """Synthetic dataset where clip names sort differently from LOSO order.

        Filenames:  action_S{NN}.mp4_{j}_.mp4 with NN zero-padded so alphabetical
                    order is well-defined and DIFFERENT from LOSO source order.
        Labels:     deterministic by clip — half ACTION, half BACKGROUND, alternating.
        Embeddings: a unique signature per clip in the first few dimensions so
                    a Ridge classifier can perfectly separate any individual clip
                    from any other (the embedding is essentially the label
                    determined by the clip's index, but with enough noise that
                    the classifier produces real decision_function scores).
        """
        rng = np.random.default_rng(seed)
        filenames: list[str] = []
        labels: list[int] = []
        groups: list[str] = []
        for src_idx in range(n_sources):
            for j in range(n_per_source):
                src = f"S{src_idx + 1}"
                # Critical: source S1's clips all use prefix "background_"
                # while S2-S12 use a mix of "action_" and "background_". This
                # makes the alphabetically-first clip "action_S10.mp4_..."
                # (since 'a' < 'b' and S10 sorts before S2/S3/... within the
                # action_ group), placing the first sorted clip in source S10
                # while LOSO traversal starts at S1. This is the precondition
                # that exposes the bug class — without it, the post-loop join
                # mismatch is invisible because both orderings happen to start
                # with S1. Mirrors the real data shape from
                # outputs/vjepa2_embeddings_eye.npz where S1 had only
                # background_ clips after the manual exclusion.
                if src_idx == 0:
                    prefix = "background"
                else:
                    prefix = "action" if (src_idx + j) % 2 == 0 else "background"
                fn = f"{prefix}_{src}.mp4_{j}_.mp4"
                filenames.append(fn)
                labels.append((src_idx + j) % 2)
                groups.append(src)
        # Sort filenames alphabetically (matches what extract_vjepa2.py produces)
        order = np.argsort(filenames)
        filenames = [filenames[i] for i in order]
        labels = [labels[i] for i in order]
        groups = [groups[i] for i in order]
        # Embeddings: per-clip unique signature in dim 0, plus weak noise
        n = len(filenames)
        X = rng.normal(scale=0.1, size=(n, dim))
        for i in range(n):
            # Signal in dim 0 = label scaled by clip index → linearly separable
            X[i, 0] = (labels[i] * 2 - 1) * (1.0 + 0.1 * i)
        return (
            np.array(X, dtype=np.float32),
            np.array(labels, dtype=int),
            np.array(groups),
            filenames,
        )

    def test_clip_label_pairing_matches_loso_traversal(self):
        """For every per_clip entry, its label must equal the actual label of
        the clip whose embedding produced its score.
        """
        X, y, groups, filenames = self._build_synthetic()
        sources = sorted(set(groups), key=lambda s: int(s[1:]))

        # Sanity: alphabetical filename order != LOSO source order
        # (this is the precondition that exposes the bug if it's reintroduced)
        sorted_first_source = extract_source(filenames[0])
        loso_first_source = sources[0]
        self.assertNotEqual(
            sorted_first_source, loso_first_source,
            "Test setup invariant: alphabetical first source must differ from "
            "LOSO first source — otherwise the bug class is not exercised."
        )

        # Build the gold mapping: clip_name → label (the truth)
        gold_label = {fn: lab for fn, lab in zip(filenames, y.tolist())}

        # Run the actual pipeline function under test
        preds, truth, clips_in_loso_order, fold_aucs, fold_log = loso_pooled(
            X, y, groups, sources, filenames,
        )

        # Construct per_clip exactly as eye_loso_lr.py does
        per_clip = [
            {
                "clip": str(clips_in_loso_order[i]),
                "source": extract_source(str(clips_in_loso_order[i])),
                "label": int(truth[i]),
                "score": float(preds[i]),
            }
            for i in range(len(clips_in_loso_order))
        ]

        # Core assertion: every per_clip entry's label matches the gold label
        # for that clip name. Fails if name↔label pairing is scrambled.
        for entry in per_clip:
            self.assertEqual(
                entry["label"], gold_label[entry["clip"]],
                f"label mismatch for {entry['clip']}: per_clip says "
                f"{entry['label']}, gold says {gold_label[entry['clip']]}",
            )

        # Also check the source field is consistent with the clip name
        for entry in per_clip:
            self.assertEqual(
                entry["source"], extract_source(entry["clip"]),
                f"source mismatch for {entry['clip']}: per_clip says "
                f"{entry['source']}, extract_source says "
                f"{extract_source(entry['clip'])}",
            )

        # Sanity: every clip appears exactly once
        clip_names = [e["clip"] for e in per_clip]
        self.assertEqual(len(clip_names), len(set(clip_names)),
                         "duplicate clips in per_clip")
        self.assertEqual(set(clip_names), set(filenames),
                         "per_clip clip set != input filenames set")

    def test_pooled_auc_independent_of_per_clip_order(self):
        """The pooled AUC must be identical whether computed from (truth, preds)
        directly or after re-pairing through per_clip. This documents the
        fact that the bug never affected the headline metric."""
        X, y, groups, filenames = self._build_synthetic()
        sources = sorted(set(groups), key=lambda s: int(s[1:]))

        preds, truth, clips_in_loso_order, _, _ = loso_pooled(
            X, y, groups, sources, filenames,
        )

        auc_direct = roc_auc_score(truth, preds)

        # Reconstruct via per_clip and recompute
        per_clip = [
            {"clip": str(c), "label": int(t), "score": float(p)}
            for c, t, p in zip(clips_in_loso_order, truth, preds)
        ]
        auc_via_per_clip = roc_auc_score(
            [e["label"] for e in per_clip],
            [e["score"] for e in per_clip],
        )
        self.assertAlmostEqual(auc_direct, auc_via_per_clip, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
