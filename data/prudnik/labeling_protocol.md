# Phase 10 — Prudnik labeling protocol (thin wrapper)

*Created 2026-05-12 for the 122-clip Prudnik unridden championship transfer test.
This file is a **thin wrapper**: the canonical labeling protocol is
`docs/labeling-protocol-2026-05.md` (verbatim, no semantic changes). This
wrapper adds **Prudnik-specific operational notes** layered on top.*

## Canonical reference (verbatim, no changes)

**Use `docs/labeling-protocol-2026-05.md` as the authoritative rule set for all label decisions.** All of §1–§7 of that doc apply unchanged:

- §2 **Primary decision rule** — ACTION = any visible positional change of at least one ear during the clip; BACKGROUND = ears essentially stationary
- §3 **FH-only rule** — verdict applies to the foreground horse; background-horse motion excluded
- §4 **Anatomical scope** — verdict determined by ear motion only; head / body / tail motion without ear motion → BACKGROUND
- §5 **Borderline `?` annotation** — `ACTION?` or `BACKGROUND?` for cases at the threshold of visibility
- §6 **Edge cases** — edit-cut artifacts, occluded ears, off-camera ears, head-tilt-induced passive ear rotation
- §7 **Worked examples** — strong action, slight action, subthreshold, head-only motion, multi-subject

## Framing context (Lesson 21)

These labels detect **ear MOTION** (any direction, any duration ≥ visible at normal speed), **not** RHpE Behavior #7 (sustained ears-back ≥5s). Phase 10a is a **transfer test of the existing RME-trained classifier** to a new context (Polish championship, unridden, portrait orientation). It does NOT validate RHpE Behavior #7 detection — that's Phase 11+ scope. See `docs/lessons_learned.md` Lesson 21 for the framing decision rationale (interpretation A adopted).

## Prudnik-specific operational notes

### 1. Frame type categorization (`frame_type` column)

You said the dataset is ~70% head-zoom / ~30% full-body. Per-clip categorization:

- **`head-zoom`** — ear/face/nose dominates the frame, body minimally visible or absent. The classifier should perform near RME-trained distribution.
- **`full-body`** — body visible (tail, posture, gait may be present), head + ears occupy smaller fraction of frame. **Out of RME training distribution** — pipeline may need a different cropping strategy (DLC keypoint-anchored ear crop becomes more useful here than for head-zoom clips, where the ear region is already centered).

Categorize during labeling — adds one column to `labels_pending.csv`, takes ~1s per clip.

### 2. Splitting workflow

For clips >15s (68 of 122), `splitting_recommendations.md` proposes uniform timestamp boundaries targeting ~10s windows. Choose one per long clip:

- **Accept proposed splits** — replicate the clip's row in `labels_pending.csv`, one row per window. Set `clip_id` like `IMG_1056_w1`, `IMG_1056_w2`, etc. Set `window` to the time range (e.g., `0.00-10.45`).
- **Override** — adjust split timestamps to capture natural action/background transitions, then proceed as above.
- **Single window** (for clips just over 15s) — keep one row; `window` = `full`; add a note that duration overflowed slightly.

For clips ≤15s (54 of 122), single row, `window` = `full`, no split decision needed.

### 3. Column conventions for `labels_pending.csv`

| Column | Values | Notes |
|---|---|---|
| `clip_id` | `IMG_NNNN` or `IMG_NNNN_wK` | Stem of filename; append `_w1`, `_w2`, … for split windows |
| `filename` | `IMG_NNNN.mov` | Original filename (unchanged even for split windows) |
| `duration_s` | float | Clip duration; for split rows, set to window duration |
| `window` | `full` or `START-END` (seconds) | `full` for unsplit; `0.00-10.45` for split windows |
| `frame_type` | `head-zoom` / `full-body` | Per §1 above |
| `label` | `action` / `background` / `action?` / `background?` | Per canonical protocol §2 + §5 |
| `multi_horse` | `0` / `1` | `1` if multiple horses visible (FH-only rule still governs verdict per §3) |
| `confidence` | `high` / `medium` / `low` | Your subjective certainty on the call |
| `notes` | free text | Observation notes (e.g., `"fh: slight left ear rotation, bh: ears still"` — matches the audit notation in canonical protocol §7) |
| `labeler` | free text | Your name / initial |
| `label_date` | `YYYY-MM-DD` | Calendar date of labeling |

### 4. Quality concerns to keep in mind (see `data_quality_notes.md`)

- **All 122 clips are PORTRAIT orientation** (W < H). This is the single biggest distribution shift from RME training data. Phase 10a transfer test will tell us how much it matters; during labeling, note any clips where the portrait framing visibly affects horse visibility (handler obscuring ear region from below, etc.) — useful diagnostic data if the transfer test underperforms.
- **fps mix (30 + 60)** — 4 clips at 30 fps, 118 at 60 fps. Single-clip operating-point work is fps-invariant under V-JEPA-2's 16-frame sampling; no labeling-time concern.
- **Sub-5s clips** (12 of 122) — flag any that are unusable for labeling (cut too short to observe ear state, etc.) with `confidence = low` + `notes = "too short to assess"`. Decision on dropping these from the transfer test is deferred to Phase 10a Stage 1 pre-reg.

### 5. Multi-horse handling (canonical protocol §3)

For multi-subject scenes, **identify the foreground horse first**, then judge ear motion on FH only. Background-horse motion is recorded in `notes` (using the `fh:` / `bh:` notation from the canonical protocol §7) but **does not affect the verdict**. Set `multi_horse = 1` for filtering / diagnostic purposes downstream.

### 6. Borderline `?` annotation (canonical protocol §5)

Use `?` suffix (`action?` / `background?`) freely — borderline cases are **explicit uncertainty signals**, not labeling errors. The canonical protocol's within-observer Cohen's κ on borderline cases was 0.586 (moderate); inter-rater κ in the multi-rater track (when SLU / Palichleb / Bek-Kaczkowska responses arrive) will be measured separately. Keeping the `?` signal in `labels_pending.csv` preserves the option to either drop borderlines from training (Strict variant) or include with reduced weight downstream.

## Send checklist (when labeling is done)

- [ ] All 122 clip-rows in `labels_pending.csv` have non-empty `label`, `frame_type`, `confidence`, `labeler`, `label_date`
- [ ] For split-derived rows: `clip_id` follows `IMG_NNNN_wK` convention, `window` is `START-END` not `full`, `duration_s` reflects window not clip
- [ ] `multi_horse` set for any clip with visible 2+ horses (per §5 above)
- [ ] Sub-5s clips noted with `confidence = low` + `notes = "too short to assess"` if applicable (per §4 above)
- [ ] Spot-check 5–10 worked examples from canonical protocol §7 mentally — your labeling on similar Prudnik clips should match the protocol's intuition

After labeling lands, Phase 10a Stage 1 pre-reg drafts: subset definitions (head-zoom vs full-body, sub-5s exclude, multi-horse handling), expected AUC range (RME baseline 0.875), gates (G1–G5 mirroring Phase 9), expected per-source breakdown. Standard discipline-pattern cycle from there.

## What this protocol does NOT do

- **No new label semantics.** All decisions defer to `docs/labeling-protocol-2026-05.md` verbatim.
- **No content-aware splitting.** `splitting_recommendations.md` proposes mechanical uniform splits; natural-breakpoint identification is the labeler's call during the labeling pass.
- **No exclusion decisions.** Sub-5s clips, multi-horse clips, and quality-concern clips are kept in `labels_pending.csv`; Phase 10a Stage 1 pre-reg locks the subset definitions for the transfer test.
- **No multi-rater κ resolution.** Single-observer (user) labels for Phase 10a; multi-rater κ track is gated on SLU / Palichleb response (`v2/outreach/`).
