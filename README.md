# horse-pain-poc

[![status](https://img.shields.io/badge/Faza_1.5-sanity_checks_done-blue)](GATE.md)
[![best](https://img.shields.io/badge/V--JEPA--2_ear_movement-0.91_(bg--masked)-success)](GATE.md)
[![data](https://img.shields.io/badge/szukam-polskich_klipów-orange)](#współpraca--szukam-danych-z-polskich-stajni)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10--3.11-blue)](pyproject.toml)

PoC stosu open-source dla automatycznej **klasyfikacji pojedynczych behaviors** z **Ridden Horse Pain Ethogram (RHpE, Sue Dyson, 24 zachowania)** — building blocks pod późniejsze multi-behavior pain assessment, nie wykrywanie bólu samo w sobie.

> **Co to znaczy konkretnie:** RHpE wymaga ≥8 z 24 behaviors razem żeby wnioskować o bólu mięśniowo-szkieletowym (Dyson 2018). Tu budujemy klasyfikatory **pojedynczych behaviors** — aktualnie ear_movement, w roadmap pozostałe 23. Multi-behavior pain assessment to horyzont ≥2 lata, wymaga walidacji klinicznej i kolaboracji z certyfikowanymi RHpE assessorami. **Aktualne MVP = jedna z 24 cegiełek.**

![DLC SuperAnimal-Quadruped keypoints na sample horse video — 5 klatek z overlay'em szkieletu](docs/example_output.png)
*5 klatek z notebooka `00_smoke_dlc_sample.ipynb` — DLC SuperAnimal-Quadruped zero-shot na [Horse_walking_in_corral_MVI_7490](https://commons.wikimedia.org/wiki/File:Horse_walking_in_corral_MVI_7490.MOV.ogv) (Wikimedia Commons CC).*

To **nie jest narzędzie diagnostyczne**. To research prototype:
- **Faza 0** (~45 min, [GATE.md](GATE.md)): sanity-check zerolot pose-estimation z DLC
- **Faza 1** (~3h): replikacja Read My Ears (Alves CVPR W'25) na ich [HF dataset](https://huggingface.co/datasets/joaomalves/read-my-ears) → V-JEPA-2 + linear probe = 0.854
- **Faza 1.5** (~10h): few-shot walidacja na 5 RHpE behaviors z DIY anchor klipami (53)
- **iter 6.5** (~3h): 4 sanity checks które ujawniły session leakage w 4/5 behaviors → patrz [`docs/lessons_learned.md`](docs/lessons_learned.md)

## Współpraca — szukam danych z polskich stajni

**Faza 2 wymaga zróżnicowanego datasetu** z polskich realiów (hala jeździecka, oświetlenie, rasy, sprzęt) — to czego nie ma w żadnym publicznym datasetcie.

**Czego potrzebuję:** klipy wideo z różnych ośrodków/koni/jeźdźców (LOSO wymaga ≥10 unique sessions — patrz [Lesson 8](docs/lessons_learned.md)). Dobra próbka wymaga konkretnego protokołu nagrywania żeby nie powtórzyć session leakage z iter 1.5.

**👉 Warto przeczytać przed nagrywaniem:** [`docs/recording-protocol.md`](docs/recording-protocol.md) (1 strona, ~5 min). Główna zasada: **5 różnych momentów z normalnej sesji treningowej** (ten sam koń, ta sama kamera, ten sam kąt — naturalna codzienna praca, bez prowokowania konkretnych reakcji). Twoja rola = nagrać; ocena behavior na klipach to robota certyfikowanego RHpE assessora po fakcie. Protocol zawiera również szablon RODO-compliant zgody.

**Co dostaję ja:** materiał do treningu modelu (open-source, MIT) + walidacja czy V-JEPA-2 + linear probe pipeline działa na realnych polskich klipach.

**Co dostajesz Ty:**
- Zdjęcia / timeline ruchu uszu / pose-estimation Twoich koni jako bonus
- Współautorstwo w writeup'ie jeśli zostanie publikowany
- Spokój ducha — RODO + pisemne zgody jeźdźców/właścicieli ogarniam, surowe klipy nigdzie poza moim laptopem nie wychodzą

**Ramy etyczne**: welfare > PoC. Jeśli koń wykazuje sygnały bólu w trakcie nagrywania, sesja się przerywa, koń idzie do weterynarza. Bez indukowanego bólu/strachu. Naturalna codzienna praca pod jeźdźcem.

Kontakt: piotr.pawluk@gmail.com lub Issues w tym repo.

## Replication results — Read My Ears ear-movement classification

Pełen pipeline na 283 klipach (HF dataset `joaomalves/read-my-ears`):

| Approach | Accuracy / AUC | Notes |
|----------|---------------|-------|
| Paper claim (Alves CVPR W'25) | 0.875 | Z ich custom YOLOv8n + face_masked_clips |
| **V-JEPA-2 + linear probe** (iter 4 reprodukcja) | **0.894** AUC | Foundation video model z półki, BEZ custom preprocessing, BEZ treningu modelu |
| V-JEPA-2 + linear probe (Faza 1, 48 test split) | 0.854 | Initial reprodukcja na test split |
| V-JEPA-2 + linear probe + bg-masked (iter 6.5 Sanity 2) | 0.911 AUC | Mask outside ear bbox → signal w ear region (drop tylko -6pp) |
| DINOv2 + linear probe + bg-masked (iter 6.5 Sanity 4) | 0.809 AUC | Image-only baseline, słabszy od V-JEPA-2 o 10pp |
| V-JEPA-2 + LOO cosine k=1 | 0.756 | Same embeddingi bez probe'a — gap 14pp pokazuje wartość LR |
| X-CLIP zerolot S1 (binary prompts) | 0.604 | Text-conditioned NIE działa dla subtle motion |
| Etap A movement-detection (1:1) | 0.583 | Pełna replikacja ich pipeline'u (YOLOv8l + optical flow) |

**Kluczowe wnioski (post Faza 1):**

1. **V-JEPA-2 zerolot przebija paper claim** — `facebook/vjepa2-vitl-fpc16-256-ssv2` (Meta, czerwiec 2025) jako ekstraktor cech 1024-D + sklearn `LogisticRegression`. Trening klasyfikatora < 1s na CPU. Embedding extraction 283 klipów ~6 min na M-series MPS.
2. **Sygnał jest faktycznie w region ucha** (Sanity 2: bg-masked drop tylko -6pp). Read My Ears protocol robust.
3. **V-JEPA-2 ≫ DINOv2 dla ear movement** o 10pp. Temporal info ma istotny wkład.

## Faza 1.5 + iter 6.5 — what we learned by being wrong

Faza 1.5: zebrałem 53 anchor klipów (5 RHpE behaviors × ~10 klipów) z legalnie dostępnych źródeł (24horsebehaviors.org, YouTube wykłady Dyson/Andersen). Plan był: walidacja czy V-JEPA-2 + LOO cosine wykrywa behaviors poza ear movement.

Po 5 iteracjach wynik był optymistyczny: head_position binary OvR AUC 0.927 z linear probe. To wyglądało jak gotowy MVP dla Fazy 2.

**Wtedy iter 6.5 sanity checks ujawniły session leakage.** LOSO (leave-one-session-out) zamiast LOO (leave-one-out):

| behavior | LOO | LOSO | Δ | verdict |
|---|---|---|---|---|
| ear_position | 0.237 | 0.494 | +0.257 | leakage |
| head_position | **0.898** | **0.561** | **−0.337** | **leakage** |
| mouth_open | 0.740 | 0.513 | −0.227 | leakage |
| tail_movement | 0.747 | **0.253** | **−0.495** | **catastrophic anti-correlation** |
| eye_expression | 0.394 | 0.773 | +0.379 | partial (wszystkie sesje Padma) |

Klasyfikator nie nauczył się behavior'u — uczył się **rozpoznawać sesję nagraniową** (telefon w Lesznowola vs YouTube documentary studio). LOSO drop −34pp dla head_position oznacza że 0.898 było artefaktem, nie sygnałem. Tail_movement LOSO 0.253 (gorsze niż chance) to spektakularny przykład anty-korelacji cross-session.

**Co iter 6.5 oznacza praktycznie:**
- Track A "head_position MVP" — **kill**. Brak fundamentu.
- Track B "ear_position via Read My Ears ROI replikacja" — **proceed**, ale wymaga zróżnicowanego datasetu (≥10 unique sessions, balanced labels).
- Read My Ears sam ma random clip-level split (te same source videos w train/val/test), więc ich 0.875 jest prawdopodobnie inflowany podobnym mechanizmem. Nasz bg-masked Sanity 2 = 0.911 sugeruje że *jakiś* signal jest w uchu, ale ich oryginalny baseline jest upper bound.

**Pełna analiza metodologiczna:** [`docs/lessons_learned.md`](docs/lessons_learned.md) — 8 lekcji od iter 1 do iter 6.5, w tym dlaczego LOO nie jest bezpieczny baseline w tej dziedzinie i dlaczego sample size trzeba liczyć w sesjach, nie klipach.

## Co znajdziesz w tym repo

```
.
├── setup.sh                  idempotentny installer oparty o uv (macOS / Linux)
├── pyproject.toml            pinowane deps (DLC 3.0.0rc14, torch 2.11, transformers 5.7, gradio, webvtt-py)
├── GATE.md                   binarne kryteria GO/NO-GO + lekcje
├── docs/
│   └── lessons_learned.md    8 lekcji metodologicznych z iter 1-6.5 (must-read)
├── notebooks/
│   ├── 00_smoke_dlc_sample.ipynb            DLC SuperAnimal-Quadruped zero-shot
│   ├── 01_read_my_ears_replicate.ipynb      Read My Ears (CVPR W'25) replikacja
│   ├── 04_few_shot_rhpe_validation.ipynb    Faza 1.5 — Gradio UI clipping + V-JEPA-2 eval
│   └── 99_colab_fallback.ipynb              backup do Google Colab T4
├── tools/
│   └── subtitle_search.py                   VTT keyword parser (Faza 1.5 anchor clipping)
└── .gitignore
```

`data/`, `checkpoints/`, `outputs/`, `vendor/` są gitignored — pobiera je
`setup.sh` (sample horse video z Wikimedia Commons CC, weights z HuggingFace).

## Quickstart (macOS Apple Silicon, lokalnie)

```bash
git clone https://github.com/piotrpawluk/horse-pain-poc
cd horse-pain-poc
bash setup.sh
source .venv/bin/activate
jupyter lab notebooks/00_smoke_dlc_sample.ipynb
```

Notebooki w kolejności: `00` (DLC sanity) → `01` (Read My Ears replikacja) → `04` (Faza 1.5 Gradio UI clipping + V-JEPA-2 eval). Pełne wyniki + lekcje w [`GATE.md`](GATE.md), pełna metodologia w [`docs/lessons_learned.md`](docs/lessons_learned.md).

## Quickstart (Google Colab, fallback)

Otwórz `notebooks/99_colab_fallback.ipynb` w [Google Colab](https://colab.research.google.com/) (File → Upload notebook). Free T4 wystarczy. Nie wymaga lokalnego setupu.

## Setup gotchas (dla replikacji)

1. **DLC 3.0** stable nie wydane (maj 2026); pinować `>=3.0.0rc14` z `--prerelease=allow`
2. **matplotlib pin `<3.9`** (DLC requirement)
3. **HF Hub 1.x** usunął `huggingface_hub.commands.huggingface_cli` — używać Python API `snapshot_download`
4. **HEVC w klipach z iPhone** (.MOV) — OpenCV powinien czytać native; macOS TCC quarantine xattrs blokują niektóre files (`xattr -d com.apple.quarantine <file>` po imporcie z Photos library)
5. **Inferencja**: V-JEPA-2 ViT-L 16 frames @ 256×256 ≈ 1.3s/clip MPS; 283 klipy ~6 min

## Stack rationale (skrót)

- **V-JEPA-2 ViT-L** ([Meta, czerwiec 2025](https://arxiv.org/abs/2506.09985)) — foundation video model, encoder features 1024-D, używany jako pretrain-only backbone (SSv2 fine-tune nie modyfikuje encodera, tylko dodaje głowę — patrz [Lesson 4](docs/lessons_learned.md))
- **DINOv2 large** (image-only, 1024-D) — alternatywny image-only baseline; w naszym pipeline słabszy o 10pp od V-JEPA-2
- **DeepLabCut SuperAnimal-Quadruped** ([Nature Comm 2024](https://www.nature.com/articles/s41467-024-48792-2)) — zero-shot pose dla 45+ gatunków, zaplanowany dla Track C (temporal behaviors w Fazie 3)
- **Read My Ears** (Alves et al., [CVPR W'25](https://arxiv.org/abs/2505.03554)) — bazowy pipeline ROI: face mask + ear bbox + classifier; pokazana skalowalność na 24 RHpE behaviors via per-behavior ROI replikacja
- **scikit-learn RidgeClassifier / LogisticRegression** — linear probe na cached embeddings, sekundy treningu na CPU
- **uv** zamiast conda — 10× szybszy installer
- **AutoML cloud** odrzucony jako strategic dead-end: Vertex AutoML legacy, Azure Custom Vision retires 09.2028, AWS Rekognition nie ma natywnego video

## Etyka / disclaimer

To jest **research prototype, nie narzędzie diagnostyczne**. Każde
zastosowanie kliniczne wymaga walidacji przez certyfikowanego RHpE
assessor'a i konsultacji weterynaryjnej. Welfare zwierzęcia jest
nadrzędne wobec PoC — jeśli model wykryje sygnały bólu w trakcie
zbierania danych, należy przerwać sesję i skierować konia do weterynarza.

## Licencja

MIT — zobacz [LICENSE](LICENSE).

## Krótkie podziękowania

- Mathis Lab — DeepLabCut + SuperAnimal-Quadruped
- Alves, Andersen, Zamansky et al. — Read My Ears
- Sue Dyson — RHpE jako framework
- Wikimedia Commons — sample horse video pod licencją CC
