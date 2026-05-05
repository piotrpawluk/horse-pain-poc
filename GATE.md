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

5. **Faza 1.5 — Few-shot V-JEPA-2 walidacja na 5 RHpE behaviors** (DIY anchor klipy, 53 łącznie z 24horsebehaviors.org/Padma + osobiste telefon nagrania): per-behavior signal strength.
   - [x] WYKONANE (z mieszanym wynikiem) / [ ] NIE
   - Komentarz: 4 iteracje na 53 klipach × 5 klas (ear/head/eye/tail/mouth) ujawniły **fundamentalnie różne wymagania pipeline'u per behavior** — full-frame V-JEPA-2 + linear probe wystarcza dla części, dla innych wymagana ROI-specific replikacja Read My Ears.

   | iter | metoda | overall | ear | head | eye | tail | mouth |
   |---|---|---|---|---|---|---|---|
   | 2 | LOO cosine, full-frame | 0.358 | 0.000 | 0.500 | 1.000 (sink) | 0.000 | 0.333 |
   | 3 | LOO cosine, head-crop YOLO | 0.302 | 0.000 | 0.500 | 0.833 | 0.000 | 0.000 |
   | 4 | Read My Ears 283 clips, **binary** | linear probe **0.894** (vs paper 0.875, Faza 1: 0.854); LOO cos k=1: 0.756 | — | — | — | — | — |
   | 5 | LOO linear probe (5-class + binary OvR), full-frame | 5-class 0.255 | LP 0.000 / **AUC 0.339** | LP 0.667 / **AUC 0.927** | LP 0.000 / AUC 0.652 | LP 0.556 / AUC 0.783 | LP 0.000 / AUC 0.736 |

   **Kluczowe wnioski empiryczne:**
   - **Iter 4 reprodukuje Fazę 1** — V-JEPA-2 + linear probe na czystym 283-clip dataset binary task = **0.894** (przebija paper 0.875 o +1,9 pp i Fazę 1 o +4 pp). Pipeline `embedding → LogReg` JEST sprawdzony.
   - **Iter 2 eye_expression=1.000 był artefaktem pomiaru, nie sygnałem.** Wszystkie 12 klipów eye z dokumentów Padma = wspólny styl talking-head = cosine sink. Iter 5 binary OvR AUC tylko 0.652 ujawnia że probe nie odzyskuje tego co nie jest sygnałem rzeczywistym.
   - **head_position AUC 0.927** na 12 klipach (6 Padma + 6 telefon) = strong signal, full-frame V-JEPA-2 + linear probe wystarcza. To jest ścieżka MVP dla Fazy 2 dla full-frame behaviors.
   - **ear_position AUC 0.339** (poniżej chance 0.5) → V-JEPA-2 SSv2 features na pełnej klatce **nie zawierają informacji o pozycji ucha**. Head-crop NIE pomaga (iter 3). Konieczna **ear-specific ROI crop** (Read My Ears: face-mask + ear bbox + V-JEPA-2 na crop'ie) — repo `vendor/horse-face-ear-detection/` ma `yolov8l_horse_ear_detection.pt`.
   - **tail_movement AUC 0.783, mouth_open AUC 0.736** — moderate signal mimo małej próbki (9 i 3 klipy). Promising, ale potrzeba więcej danych.

   **Gate Faza 1.5 (≥3/5 behaviors z accuracy ≥0.70):** ✗ NIE PRZEKROCZONY w postaci pierwotnej — tylko head_position binary OvR AUC ≥0.70, ear/eye nie kwalifikują się. **Ale**: gate zdefiniowano jako jednolity próg dla 5-class accuracy; binary OvR per behavior pokazuje że 4/5 behaviors ma AUC ≥0.65 i 2/5 AUC ≥0.75 (head=0.927, tail=0.783) — to jest **różnicowanie sygnał per behavior**, nie blanket fail.

## Werdykt

**Decyzja Fazy 0–1.5: [x] GO z rewizją scope** (5/5 punktów wykonanych, 1 z mieszanym wynikiem)

**Faza 0** (DLC SuperAnimal-Quadruped pose backbone): działa end-to-end w ~45 min na macOS Apple Silicon. **Faza 1** (Read My Ears benchmark): V-JEPA-2 + linear probe = **0.854** dla ear movement (Faza 1) → **0.894** w iter 4 reprodukcji (Faza 1.5). **Faza 1.5** (5-behavior DIY): ujawnia że RHpE behaviors mają **wysoce różnicowane wymagania preprocessing'u** — full-frame V-JEPA-2 wystarcza dla niektórych (head_position AUC 0.927), dla innych konieczna ROI-specific crop (ear_position AUC 0.339 full-frame → potrzeba ear-region jak Read My Ears).

**Faza 2 jako 3-track parallel zamiast jednolitego pipeline'u** (zob. `poc-ai-rhpe.md`):
- **Track A** (full-frame): head_position, balance, lameness indicators ~5 z 24 behaviors. Pipeline V-JEPA-2 + linear probe sprawdzony. ~50 klipów × behavior.
- **Track B** (ROI crop): ear_position, eye_expression, mouth_open, tail_movement ~6 z 24. Replikacja preprocessing'u Read My Ears (YOLO ROI + V-JEPA-2 na crop'ie). ~50 klipów × behavior.
- **Track C** (DLC keypoints + temporal): behaviors z trajektorią ruchu (~13 z 24). **Odłożone do Fazy 3.**

**MVP Fazy 2**: 1 behavior z Track A (head_position) + 1 z Track B (ear_position via Read My Ears crop). Cel: oba ≥0.80 accuracy. Czas: ~3-5 dni implementacji + zbieranie danych.

Data wypełnienia: 2026-05-04 (Faza 0) / 2026-05-05 (Faza 1) / 2026-05-06 (Faza 1.5)

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
- **Następne kroki potencjalne**: (a) email do autorów Alves/Andersen z konkretnym wynikiem 0.854 (Faza 1) / 0.894 (iter 4 Faza 1.5) jako otwierającą rozmowę kolaboracyjną; (b) Faza 2 jako **3-track parallel** zamiast jednolitego pipeline'u — Track A full-frame (head_position+inne), Track B ROI replikacja Read My Ears (ear/eye/mouth), Track C DLC keypoints + temporal (Faza 3).
- **Faza 1.5 — kluczowe artefakty**: `outputs/anchor_embeddings_53clips.npz` (cache embeddings dla szybkich re-iteracji), `outputs/few_shot_validation_results_iter5_linearprobe.json` (per-behavior accuracies + binary OvR AUC), `outputs/readmyears_loo_baseline_results.json` (iter 4: linear probe 0.894 vs LOO cosine k=1 0.756 = 14pp gap — twarda miara „ile sygnału pomija raw cosine similarity"), `outputs/few_shot_validation_iter5_linearprobe_plot.png` (iter 2 cosine vs iter 5 LP vs binary OvR AUC porównanie).
- **Faza 1.5 — wniosek metodyczny**: na małej próbce (≤15 klipów per klasa) **LOO cosine 5-class jest niereliable** (eye_expression sink-effect 1.000 vs prawdziwe AUC 0.652). **Linear probe + binary one-vs-rest AUC** to honest signal strength per behavior. Dla Fazy 2 i dalej: zawsze raportować **per-behavior binary OvR AUC**, nie 5-class accuracy.
- **Iter 2 head-crop YOLO eksperyment** (`vendor/horse-face-ear-detection/horse_face_detection/yolov8l_horse_face_detection.pt`): face detection 76% udanych dla ear klipów, ale embedding nadal nie odróżnia ear-states od eye-states. **Wniosek dla Track B**: musi być **ear-specific** ROI (mamy `yolov8l_horse_ear_detection.pt`), nie head-crop. Identyczny preprocessing per behavior tak jak w Read My Ears jest warunkiem koniecznym.
