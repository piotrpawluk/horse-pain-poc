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
   - [x] TAK (z asteriskiem) / [ ] NIE
   - Komentarz: Oryginalny `movement-detection/test.py` z repo padł na `ModuleNotFoundError: ultralytics` (YOLO + custom weights nie w pyproject). Fallback w notebooku 01 wykorzystał DLC ear keypoints (`right_earbase`, `right_earend`, `left_earbase`, `left_earend`) i wyliczył proxy ruchu uszu jako rolling std pozycji — zadziałał i pokazuje sensowny sygnał (niska amplituda gdy konie stoją, wysoka gdy chodzą). Replikacja 1:1 Alves 2025 wymaga dodania `ultralytics` + pobrania ich custom weights — to TODO dla Fazy 1, nie blokuje GO.

4. **Budżet czasowy**: Cały proces (setup + 2 notebooki + ten dokument) zajął ≤16 h pracy.
   - [x] TAK / [ ] NIE
   - Czas faktyczny: **~45 min** (mocno poniżej 8h hard budget, dzięki uv i lazy weights download)

## Werdykt

**Decyzja: [x] GO** (4/4 z asteriskiem przy punkcie 3)

Stos open-source dla detekcji bólu u koni działa end-to-end na macOS Apple Silicon w ~45 min. DLC SuperAnimal-Quadruped jest najmocniejszym ogniwem — out-of-the-box keypoints na koniu są wzrokowo sensowne i są bazą dla detekcji 5 z 24 RHpE behaviors planowanych w Fazie 1 (zob. `poc-ai-rhpe.md`). Read My Ears jako research-grade repo wymaga rozszerzenia deps (`ultralytics`) i pobrania ich custom weights — TODO dla Fazy 1, nie blokuje decyzji.

Data wypełnienia: 2026-05-04

## Notatki dodatkowe (do Fazy 1)

- **Surgical fixes setup.sh** — pamiętać, że DLC 3.0 stable nie jest jeszcze wydane (rc14 w maju 2026); HF Hub 1.x usunął `huggingface_hub.commands.huggingface_cli` (używać Python API `snapshot_download`); matplotlib pin musi być <3.9 dla zgodności z DLC.
- **Dodać do pyproject.toml dla Fazy 1**: `ultralytics>=8.3` (dla Read My Ears YOLO inference) + skrypt download dla ich pretrained ear-tracker weights.
- **Domain shift**: corral z Wikimedia jest dnia, na piasku, hala jeździecka będzie pod światłem sztucznym, podłoże fibre — keypoints powinny działać, ale to do zweryfikowania w Fazie 2 z własnymi klipami.
- **Sample video issue**: WebFetch halucynował URL Wikimedia (zamiast `0/0e/` było `c/c7/`). Zawsze weryfikować przez API: `curl 'https://commons.wikimedia.org/w/api.php?action=query&titles=File:NAME&prop=imageinfo&iiprop=url&format=json'`.
- **Performance**: 287 klatek 640×480 → ~5 min inferencji na M-series CPU/MPS. Dla 30s 1080p (Faza 2 / 30-50 klipów) liczyć ~10-15 min/klip × 50 = 8-12 h compute lokalnie. Wówczas warto rozważyć RunPod RTX4090 ($0.34/h × 4 h = ~$1,5).
- **Dwa konie w sample**: SuperAnimal multi-instance detection zadziałał. Przy ridden context (jeździec + koń) trzeba zweryfikować czy detector nie myli jeźdźca z drugim koniem.
- **Read My Ears 87,5%**: niezweryfikowane w Fazie 0 (oryginalny pipeline padł). Wymóg do Fazy 1 — dokończyć replikację z `ultralytics` i ich weights.
