# horse-pain-poc

[![status](https://img.shields.io/badge/Faza_0-GO-success)](GATE.md)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10--3.11-blue)](pyproject.toml)

Faza 0 sanity-check stosu open-source dla automatycznej detekcji bólu u koni
zgodnie z **Ridden Horse Pain Ethogram (RHpE, Sue Dyson, 24 zachowania)**.

![DLC SuperAnimal-Quadruped keypoints na sample horse video — 5 klatek z overlay'em szkieletu](docs/example_output.png)
*5 klatek z notebooka `00_smoke_dlc_sample.ipynb` — DLC SuperAnimal-Quadruped zero-shot na [Horse_walking_in_corral_MVI_7490](https://commons.wikimedia.org/wiki/File:Horse_walking_in_corral_MVI_7490.MOV.ogv) (Wikimedia Commons CC).*

To **nie jest narzędzie diagnostyczne**. To 45-minutowy weekend exploration
(`bash setup.sh && jupyter lab notebooks/00_smoke_dlc_sample.ipynb`)
sprawdzający, czy zerolot pose-estimation + behavior-classification działa
na sprzęcie pojedynczego użytkownika i daje sygnał wystarczający do
zaplanowania większego projektu.

## Co znajdziesz w tym repo

```
.
├── setup.sh                  idempotentny installer oparty o uv (macOS / Linux)
├── pyproject.toml            pinowane deps (DLC 3.0.0rc14, torch 2.11, transformers 5.7)
├── GATE.md                   4 binarne kryteria GO/NO-GO + lekcje
├── notebooks/
│   ├── 00_smoke_dlc_sample.ipynb       DLC SuperAnimal-Quadruped zero-shot
│   ├── 01_read_my_ears_replicate.ipynb Read My Ears (CVPR W'25) replikacja + DLC ear proxy
│   └── 99_colab_fallback.ipynb         backup do Google Colab T4
└── .gitignore
```

`data/`, `checkpoints/`, `outputs/`, `vendor/` są gitignored — pobiera je
`setup.sh` (sample horse video z Wikimedia Commons CC, weights z HuggingFace).

## Quickstart (macOS Apple Silicon, lokalnie)

```bash
git clone https://github.com/<your-fork>/horse-pain-poc
cd horse-pain-poc
bash setup.sh
source .venv/bin/activate
jupyter lab notebooks/00_smoke_dlc_sample.ipynb
```

Po uruchomieniu wszystkich komórek notebooków `00` i `01` wypełnij
[`GATE.md`](GATE.md) — 4 binarne kryteria. **3/4 = GO** dla rozszerzenia
do większego projektu, **<3/4 = NO-GO**.

## Quickstart (Google Colab, fallback)

Otwórz `notebooks/99_colab_fallback.ipynb` w
[Google Colab](https://colab.research.google.com/) (File → Upload notebook).
Free T4 wystarczy. Nie wymaga lokalnego setupu.

## Wynik referencyjny

Dla sample [Horse_walking_in_corral_MVI_7490](https://commons.wikimedia.org/wiki/File:Horse_walking_in_corral_MVI_7490.MOV.ogv)
(Wikimedia Commons, 9.6s, 287 klatek, 2 konie):

- **DLC SuperAnimal-Quadruped** wykrył oba konie i nałożył pełen szkielet
  na ≥90% klatek (głowa, kark, łopatki, biodra, kończyny, uszy)
- **DLC ear keypoints proxy** (rolling std pozycji `*_earbase`/`*_earend`)
  wygenerował sensowny timeline ruchu uszu
- **Read My Ears 1:1** padł na `ModuleNotFoundError: ultralytics` —
  wymaga rozszerzenia deps + custom weights (TODO Faza 1)

Czas: ~45 min. Koszt: 0 PLN.

## Lekcje (z `GATE.md`, dla replikacji)

1. **DLC 3.0** stable nie wydane (maj 2026); pinować `>=3.0.0rc14` z
   `--prerelease=allow`
2. **matplotlib pin `<3.9`** (DLC requirement)
3. **HF Hub 1.x** usunął `huggingface_hub.commands.huggingface_cli` —
   używać Python API `snapshot_download`
4. **Sample video URL** — weryfikować przez Wikimedia API:
   `curl 'https://commons.wikimedia.org/w/api.php?action=query&titles=File:NAME&prop=imageinfo&iiprop=url&format=json'`
5. **Inferencja**: 287 klatek 640×480 → ~5 min na M-series CPU/MPS;
   30s 1080p (Faza 2 / 30-50 klipów) liczyć ~10-15 min/klip → wówczas
   warto rozważyć RunPod RTX4090 (~$0.34/h)

## Stack rationale (skrót)

- **DeepLabCut SuperAnimal-Quadruped** — zero-shot pose dla 45+ gatunków
  ([Nature Comm 2024](https://www.nature.com/articles/s41467-024-48792-2)),
  out-of-the-box dla koni, autorzy publikują demo na Colab
- **Read My Ears** (Alves et al., [CVPR W'25](https://arxiv.org/abs/2505.03554))
  — ear movement detection 87,5% accuracy
- **uv** zamiast conda — 10× szybszy, prostszy pojedynczy tool
- **planowany backbone** (Faza 1+): VideoMAE-v2 lub V-JEPA-2 fine-tune
  (foundation models post-2024 obsoletują pre-foundation Broomé 2019)
- **AutoML cloud (Vertex/Azure/AWS)** odrzucony jako strategic dead-end:
  Vertex AutoML to legacy (Google przesuwa na Gemini fine-tuning),
  Azure Custom Vision retires 09.2028, AWS Rekognition Custom Labels
  nie ma natywnego video

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
