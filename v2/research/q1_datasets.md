# Q1 — Dataset Landscape for RHpE Multi-Behavior Detection

## State of the art
No public, RHpE-labeled video corpus exists. Full-24 RHpE labeled video sets are held privately by the Dyson group ("available on reasonable request"), and the validation studies use single-rater real-time scoring rather than a sharable archive (*Animals* 2020 10(6):1044; *Animals* 2023 13(12):1940, both peer-reviewed). The KTH/SLU Sweden line (Broomé, Kjellström, Andersen, Rhodin, Ask) produced two pain-video datasets, PF and EOP(j) — both **explicitly non-public** and labeled with the orthopedic CPS scale, not RHpE (*PLOS One* 2022; arXiv:2105.10313). The closest RHpE-aligned resource is Phelipon, Lansade & Razzaq 2025 (*Scientific Reports*, peer-reviewed): 1,036 ridden-horse images, binary comfortable/uncomfortable, mixed public/private provenance — image-only and not openly redistributable.

## Named datasets (URLs verified)

| Dataset | URL | License | RHpE relevance |
|---|---|---|---|
| **PFERD** (*Sci Data* 2024, peer-reviewed) | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/2EXONE | **CC-BY 4.0** | Pose only (5 horses, 10-cam + markers); keypoint pretraining. |
| **EquiFACS** (*PLOS One* 2015, peer-reviewed) | https://github.com/AnimalFACS/EquiFACS | Manual free; video corpus not redistributed | FAU coding, facial only. |
| **Horsing Around** (*Data* 2019, peer-reviewed) | https://www.mdpi.com/2306-5729/4/4/131 | Open via 4TU | IMU-centric; video as ground truth only. |
| **Phelipon 2025** (*Sci Reports*, peer-reviewed) | https://pmc.ncbi.nlm.nih.gov/articles/PMC12018932/ | Not redistributable; on-request | Closest RHpE/HGS image labels publicly described. |
| **Broomé painface-recognition** (*PLOS One* 2022, peer-reviewed) | https://github.com/sofiabroome/painface-recognition | Code open; datasets **confidential** (marie.rhodin@slu.se / katrina.ask@slu.se) | Pain video, not ridden, not RHpE. |

## Cost-benefit to construct from scratch
Empirical anchors (*Animals* 2020 10(6):1044): trained-vet RHpE Cohen's κ averages **0.72 (SD 0.22)**, total-score ICC 0.97, but eye/muzzle behaviors drag κ down. Protocol = 5–10 min ridden bout. A vet scoring one 10-min clip with all 24 behaviors takes 25–40 min; at £100–150/hr equine-vet rates that's **~£50–100/clip single-rater**. κ≥0.7 demands ≥2 raters on ≥20% of corpus, so a 300-clip κ-validated set ≈ 360 vet-hours ≈ **£40–60k** plus rater training and 8–12 weekends of GDPR-consented filming. **Not feasible solo without a grant or vet co-author — deferred until resources change.**

Collaboration shortlist (descending plausibility): (1) **SLU/Uppsala** (Rhodin, Ask, Andersen) — published contacts, data-sharing posture; (2) **KTH RPL** (Kjellström/Broomé network); (3) **Dyson group / 24horsebehaviors.org** (commercial outreach) — owns the canonical corpus, historically restrictive; (4) **RVC, Wageningen** — low yield via cold outreach.

## What this implies for the PoC
No off-the-shelf 24-behavior video corpus exists and building one is out of scope solo; pilot 50–100 in-house clips with 1 vet + 1 self-trained rater for a κ spot-check, while opening collaboration with SLU/Uppsala before locking architecture.

### Open Questions
- Empirical κ data (eye/muzzle behaviors drag agreement down) suggests v1 could target the **6–8 highest-κ, most-frequently-scored RHpE behaviors** (ear position, tail swish, head toss, mouth opening) instead of all 24. Surfacing — not unilaterally re-scoping.
- Worth confirming whether Phelipon (2025) corresponding author will release the 1,036-image set under a research licence; fastest legitimate bootstrap.
