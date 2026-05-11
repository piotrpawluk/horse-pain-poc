# Dyson RHpE Scoring Rule — Literature Check

*Resolves the count-vs-presence open question from `synthesis.md` (Q4 cross-cut + Q5 implication).*
*Date: 2026-05-10. Sources verified via WebFetch + WebSearch.*

---

## Question

The synthesis (`synthesis.md`, Q4 cross-cut and Q5 implication) flagged this as the single open question that could collapse a substantial chunk of the Q4 long-form pipeline scope:

> Is RHpE clinically scored as count-sensitive (frequency of each behavior matters) or presence/absence (each of 24 is either observed at least once during the session or not, sum ≥8 → pain inferred)?

If presence/absence, learned TAL is irrelevant and the long-form bridge collapses to per-behavior thresholding.

## Answer

**Presence/absence.** Each of the 24 RHpE behaviors is scored as binary (occurred at least once during the ridden session = 1, did not occur = 0). The diagnostic threshold ≥8/24 counts **distinct behaviors observed**, not the frequency of any single behavior.

## Evidence

### Dyson & Pollard 2023 (*Animals* 13(12):1940, [PMC10295347](https://pmc.ncbi.nlm.nih.gov/articles/PMC10295347/))

Peer-reviewed. Direct quote (verified via WebFetch 2026-05-10):

> "An RHpE score of ≥8/24 reflects the likely presence of musculoskeletal pain."

The scoring counts whether each behavior was present during ridden exercise — a yes/no per behavior, totalled.

### Dyson 2018 (J Vet Behav 23:47–57) — original RHpE paper, as quoted in application papers

> "the maximum individual occurrence score for lame horses was 14 (out of 24 possible markers), with a median and mean score of 9 (±2 SD) compared with a maximum score of 6 for nonlame horses, with a median and mean score of 2 (±1.4)."

The phrasing **"markers"** (and "out of 24") is structurally a binary set, not a frequency sum. Max possible = 24 = the count of behaviors in the ethogram itself.

### Dyson 2020 (*Animals* 10(6):1044) and PMC7341225 (Application paper)

Application studies report scores in the range 0–24 with the same ≥8/24 threshold convention — consistent with binary-per-behavior.

## What this implies for the PoC

**The Q4 long-form pipeline collapses from "calibrated stream → hysteresis → temporal NMS → event count" to the much simpler:**

```
Per behavior k ∈ {1..24}:
   max_window_prob_k = max over sliding windows of calibrated P(behavior_k | window)
   present_k = (max_window_prob_k > τ_k)

Session score = Σ_k present_k
Flag for vet review if session_score ≥ 8
```

Practical implications:

1. **No temporal NMS, no event counting, no learned TAL ever needed.** Action segmentation literature (MS-TCN, ASFormer, ActionFormer, TriDet, point-supervised TAL) becomes structurally irrelevant for the RHpE clinical claim. Defer permanently unless a *different* clinical use-case (e.g. behavioural-frequency research) is in scope.

2. **B1 (long-form pipeline) drops from ~2 days to sub-day.** The whole bridge is: aggregate sliding-window probabilities, take max per behavior, threshold once.

3. **B3 (calibration) becomes even more load-bearing.** The per-behavior threshold τ_k determines `present_k`, and τ_k depends entirely on calibrated probabilities. **Temperature scaling per behavior** (Ojeda 2023; Platt-equivalent at ~1 parameter) is the right choice at n=283 / 12 sources — isotonic overfits below ~N=1000.

4. **Per-behavior τ_k selection criterion** can be set on the calibration set: choose τ_k that achieves a fixed per-behavior false-positive rate (e.g. 0.05) on negative-source clips, consistent with the Dyson 2018 baseline-rate finding (median 2/24 in non-lame horses ⇒ ~0.08 per-behavior FPR baseline).

5. **The 24-binary aggregation is itself calibratable.** Once each `present_k` has a known per-behavior FPR/TPR, the session-level "≥8/24" rule has a derivable session-level operating point (sum of independent Bernoullis; Poisson approximation reasonable). This makes the operating-point claim defensible under TRIPOD+AI/STARD-AI without additional methodology.

## Caveat — what this does NOT settle

- **Frequency-aware research questions** (e.g. "is tail-swish frequency a stronger pain signal than presence?") are outside the operational RHpE scoring rule but may be of clinical interest. If a future phase explores those, the long-form pipeline would need extension. **Not in current scope.**
- **Severity grading beyond present/absent** for individual behaviors — the original ethogram includes graded entries for some behaviors (e.g. ear position has multiple sub-codes). Confirm against the per-behavior coding manual before locking τ_k thresholds. *Action:* read the Dyson 2018 supplementary appendix or the 24horsebehaviors.org per-behavior definitions.
- **Clinician-practice drift from the published rule.** Most Dyson-trained vets follow the published convention, but informal adaptations exist in field practice. Worth a one-question check with a clinician contact (low priority — published rule governs).

## References

- Dyson, S. (2018). Development of an ethogram for a pain scoring system in ridden horses and its application to determine the presence of musculoskeletal pain. *J. Vet. Behav. Clin. Appl. Res.* 23:47–57. [Source via ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1558787817301727).
- Dyson, S. & Pollard, D. (2023). [Application of the Ridden Horse Pain Ethogram and Its Relationship with Gait](https://pmc.ncbi.nlm.nih.gov/articles/PMC10295347/), *Animals* 13(12):1940.
- Dyson, S. (2020). [Application of the Ridden Horse Pain Ethogram to elite Polish dressage horses](https://pmc.ncbi.nlm.nih.gov/articles/PMC7341225/), *Animals* 10(6):1044.
- Dyson, S. & Pollard, D. (2022). [British Eventing application of RHpE](https://pmc.ncbi.nlm.nih.gov/articles/PMC8909886/), *Animals* 12(5):590.
- 24 Behaviors of the Ridden Horse in Pain — [24horsebehaviors.org](https://www.24horsebehaviors.org/) (per-behavior coding reference).

## Decision

**Lock B1 scope to the simplified presence/absence pipeline.** Pre-register with explicit reference to Dyson 2018 J Vet Behav 23:47–57 and Dyson & Pollard 2023 Animals 13(12):1940 as the primary scoring-rule sources.
