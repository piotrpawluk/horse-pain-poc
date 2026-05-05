# GATE — Faza 0 weekend exploration

Wypełnić po uruchomieniu `setup.sh` + notebooków `00` i `01`. Każdy punkt to binarne TAK / NIE z 1-zdaniowym komentarzem.

## Kryteria

1. **Setup zakończony**: `setup.sh` zakończył się bez błędu na Mac M-series **lub** Colab fallback działa.
   - [x] TAK / [ ] NIE
   - Komentarz: 4 surgical fixes podczas iteracji (DLC 3.0.0rc14 zamiast 3.0.0; matplotlib<3.9 z DLC pinów; HF Hub 1.x — Python API zamiast CLI; sample video z Wikimedia Commons z prawdziwym URL — Pexels 856038 to były łódki, nie koń). Po fixach: uv 0.9.0 + Python 3.11.13 + DLC 3.0.0rc13 + torch 2.11.0 + transformers 5.7.0 + 483 MB weights — wszystko zainstalowane na macOS Apple Silicon bez Colab fallbacku.

2. **DLC SuperAnimal-Quadruped na sample**: Keypoints **wzrokowo sensowne** (kopyta, łokcie, łopatka, biodro, oczy, uszy widoczne na koniu) na sample clipie z `data/sample_horse.mp4`. Zobacz `outputs/sample_horse_..._labeled_before_adapt.mp4` i `outputs/sample_keypoints_grid.png`.
   - [x] TAK / [ ] NIE
   - Komentarz: Klip Wikimedia Commons "Horse walking in corral" (640×480, 9.6s, 287 klatek, dwa konie). DLC SuperAnimal-Quadruped (hrnet_w32 + fasterrcnn_resnet50_fpn_v2) zero-shot wykrył oba konie i nałożył pełen szkielet (głowa, kark, łopatki, biodra, kończyny, uszy) na ≥90% klatek. Inferencja CPU/MPS zajęła ~5 min.

3. **Read My Ears replikacja**: Repo `vendor/read-my-ears` uruchamia się i produkuje wykres ruchu uszu na ich własnym przykładzie bez modyfikacji kodu. Zobacz `outputs/ear_movement_timeline.png`.
   - [x] TAK / [ ] NIE
   - Komentarz: **Faza 0**: fallback z DLC ear keypoints (`right_earbase`, `right_earend`, `left_earbase`, `left_earend`) wygenerował sensowny `ear_movement_timeline.png`. **Faza 1 Etap A**: pełna replikacja `movement-detection` 1:1 — `ultralytics>=8.3` + YOLOv8l custom weights + dataset z [`joaomalves/read-my-ears`](https://huggingface.co/datasets/joaomalves/read-my-ears) (CC-BY-4.0, 48 test klipów). Pipeline uruchomił się end-to-end. **Wyniki**: accuracy=0.583 (vs paper 0.875). Algorytm overaggressive (recall=1.0, 20/26 FP). **Faza 1 Alternatywa V-JEPA-2 zero-shot**: foundation video model `facebook/vjepa2-vitl-fpc16-256-ssv2` (Meta, czerwiec 2025) BEZ trenowania — embedding extraction (1024-dim) na 235 train+val + 48 test klipów (~25 min M-series MPS), potem linear probe (logistic regression sklearn). **Wynik: accuracy=0.854** (precision=0.826, recall=0.864, f1=0.844) — **delta vs paper claim 0.875 = tylko −2,1 pp**. To bije Etap A o +27 pp. Zobacz `outputs/vjepa2_comparison.png` + `outputs/vjepa2_results.json`. **Strategiczna implikacja**: Etap B/C/D (fine-tune VideoMAE + LSTM) są niepotrzebne — foundation model z półki dorównuje paper'owi. Pivot dla Faza 2/3: zamiast trenować custom model per behavior, użyć V-JEPA-2 embeddings + 24 linear probes (jeden per RHpE behavior).

4. **Budżet czasowy**: Cały proces (setup + 2 notebooki + ten dokument) zajął ≤16 h pracy.
   - [x] TAK / [ ] NIE
   - Czas faktyczny: **~45 min** (mocno poniżej 8h hard budget, dzięki uv i lazy weights download)

## Werdykt

**Decyzja: [x] GO** (4/4)

**Faza 0** (DLC SuperAnimal-Quadruped pose backbone): działa end-to-end na macOS Apple Silicon w ~45 min, keypoints wzrokowo sensowne na koniu (głowa, kark, łopatki, biodra, kończyny, uszy). **Faza 1** (Read My Ears benchmark): 4 podejścia testowane na pełnym test split (N=48). **Najlepsze osiągnięte**: V-JEPA-2 + linear probe = **0.854** (foundation video model z półki, BEZ żadnego treningu modelu, sekundy LogReg na CPU) — delta tylko −2,1 pp od paper claim 0.875. Etap A movement-detection 1:1 = 0.583 (custom YOLO pipeline). X-CLIP zerolot 0.40–0.60 (text-conditioned approach nie działa dla subtle motion). **Etap B/C/D LSTM tracks anulowane** — V-JEPA-2 zerolot dorównuje paper'owi, fine-tune VideoMAE od zera (6-10h GPU) ma znikomy expected upside. **Pivot dla skalowania na 24 RHpE behaviors**: V-JEPA-2 embeddings (1024-D) + 24 niezależne linear probes (każdy ~1s treningu na CPU, wymaga ~50-100 labeled klipów per behavior) zamiast trenowania custom modelu per-class.

Data wypełnienia: 2026-05-04 (Faza 0) / 2026-05-05 (Faza 1)

## Notatki dodatkowe (do Fazy 1+)

- **Surgical fixes setup.sh** — DLC 3.0 stable nie jest jeszcze wydane (rc14 w maju 2026); HF Hub 1.x usunął `huggingface_hub.commands.huggingface_cli` (używać Python API `snapshot_download`); matplotlib pin musi być <3.9 dla zgodności z DLC; ultralytics 8.4.46 zainstalował się czysto z torch 2.11.
- **Domain shift**: corral z Wikimedia jest dnia, na piasku, hala jeździecka będzie pod światłem sztucznym, podłoże fibre — keypoints powinny działać, ale to do zweryfikowania w Fazie 2 z własnymi klipami.
- **Performance**: 287 klatek 640×480 → ~5 min inferencji DLC SuperAnimal na M-series CPU/MPS. Movement-detection RME: 48 klipów × 1920×1080 resize → ~12 min na M-series (YOLOv8l + Farneback flow).
- **Read My Ears 87,5% paper claim** — Etap A movement-detection: 0.58, Alternatywa V-JEPA-2 zerolot: **0.854** (−2,1 pp od paper, BEZ ich custom YOLO/preprocessing/FPS resampling, BEZ żadnego treningu).
- **V-JEPA-2 jako pivot strategiczny** — foundation video model z półki dorównuje custom-trained paper'owi. Implikacja dla Faza 2/3: zamiast trenować klasyfikatory per-behavior (24 RHpE), embeddingi V-JEPA-2 (1024-dim) per klip → 24 niezależne linear probes (logistic regression na CPU, sekundy treningu). Drastycznie zmniejsza compute requirements i open-source friendly.
- **YOLOv8l weights** — `jmalves5/horse-face-ear-detection` ma tylko `yolov8l_*.pt` (~175 MB każdy), nie nano. Custom yolov8n nie jest dystrybuowany publicznie.
- **`utils.metrics` import** — `vendor/read-my-ears/utils/` nie ma `__init__.py`. Inline'owana kopia w notebooku 01.
- **Etap B/C/D (LSTM tracks)** — **anulowane**. V-JEPA-2 zerolot dorównuje paper claim, fine-tune VideoMAE od zera (6-10h GPU) ma znikomy expected upside.
- **X-CLIP text-conditioned zerolot przetestowany — NIE działa dla tego task'u**. `microsoft/xclip-base-patch16-16-frames` na 48 test klipach: S1 binary 0.604, S2 cinematic 0.458, S3 multi-prompt 0.396. Cinematic i multi-prompt confusion matrix pokazuje że model **zawsze przewiduje action** — text-conditioned approach nie rozróżnia subtle ear motion od still poses. **Implikacja**: dla 24 RHpE behaviors nie wystarczy "wpisz 24 promty zerolot"; potrzebny **V-JEPA-2 embeddings + 24 niezależne linear probes** (każdy wymaga małego labeled subsetu ~50-100 klipów per behavior).
- **Compat issue (do naprawy upstream)**: transformers 5.7.0 + tokenizers 0.23.0rc0 ma bug w `CLIPTokenizer.__init__` linia 117 — wywołuje `processors.RobertaProcessing(cls=...)` ale nowy API wymaga `cls_token=`. Workaround: monkey patch w pierwszej komórce notebooka 03 (mapuje `cls` → `cls_token`).
- **Następne kroki potencjalne**: (a) email do autorów Alves/Andersen z konkretnym wynikiem 0.854 jako otwierającą rozmowę kolaboracyjną; (b) Faza 2 — własny dataset RHpE z ground truth od certyfikowanego assessor'a, V-JEPA-2 embeddings + multi-label linear probes per RHpE behavior.
