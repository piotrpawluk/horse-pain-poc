# PoC AI — Faza 0 (Weekend exploration)

Sanity check stosu open-source dla detekcji bólu u koni (RHpE).
Pełen plan: [`../poc-ai-rhpe.md`](../poc-ai-rhpe.md). Plan Fazy 0: [`../Plans/przeanalizuj-oba-dokumenty-w-serialized-crab.md`](../Plans/przeanalizuj-oba-dokumenty-w-serialized-crab.md).

## Quickstart

```bash
cd /Users/peterpawluk/horse-training/poc
bash setup.sh                            # idempotentny — można odpalić ponownie
source .venv/bin/activate
jupyter lab notebooks/00_smoke_dlc_sample.ipynb
```

Po `00` uruchom `01_read_my_ears_replicate.ipynb`. Wypełnij [`GATE.md`](GATE.md) i zdecyduj GO / NO-GO.

## Gdy macOS pada na DLC / torch install

Otwórz [`notebooks/99_colab_fallback.ipynb`](notebooks/99_colab_fallback.ipynb) bezpośrednio w [Google Colab](https://colab.research.google.com/) (File → Open notebook → GitHub / Upload). T4 free tier wystarczy.

## Co jest w repo, co nie

Wersjonowane: `setup.sh`, `pyproject.toml`, 3 notebooki (z wyczyszczonymi outputs), `README.md`, `GATE.md`.
Gitignored: `.venv/`, `data/`, `checkpoints/`, `outputs/`, `vendor/`. Wszystkie binaria pobiera `setup.sh`.

## Twardy budżet czasu

8 h hard / 16 h soft. Przy braku rezultatu po 16 h → `GATE = NO-GO` i redirect do wariantu A z `poc-ai-rhpe.md` (sam wideo + raport assessor'a, bez ML).
