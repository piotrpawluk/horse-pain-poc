# Protocol nagrywania klipów — projekt RHpE PoC

*Wersja 1.0 — 2026-05-06. Jedna strona. Przeczytaj przed nagrywaniem.*

## Po co protocol

Pierwsza iteracja PoC zebrała 53 klipy "jak Bóg da" (różne ośrodki, telefon, YouTube, miks formatów). Sanity check ujawnił że model uczył się **rozpoznawać sesję nagraniową** (telefon Lesznowola vs YouTube documentary), nie pozycji uszu/ogona/głowy. LOSO drop −34pp dla head_position, −49pp dla tail_movement. Pełna analiza: [`lessons_learned.md`](lessons_learned.md).

**Wniosek**: 30 klipów zebranych według intuicji ≠ 30 klipów które uczą model. Ten protocol zapobiega tej pomyłce.

## Twardy wymóg metodologiczny

**W obrębie jednej sesji nagraniowej musisz uchwycić więcej niż jeden stan tego samego behavior.** Tzn. ten sam koń, ta sama kamera, ten sam kąt, to samo światło — ale w trakcie 30 minut nagrywania uchwycić np. uszy do przodu **i** uszy do tyłu/na boki. Bez tego model uczy się "ośrodek X = uszy w pozycji Y", nie "behavior".

## Wymagania techniczne

| pozycja | wymóg |
|---|---|
| Kamera | smartfon na **statywie** (nie z ręki — drgania cały czas zmieniają tło) |
| Kąt | jeden wybrany kąt na całą sesję; preferowane: bok (90°) lub ¾ z przodu |
| Odległość | cała sylwetka konia widoczna + ~30% margines (nie close-up) |
| Rozdzielczość | 1080p wystarczy (4K niepotrzebne, więcej miejsca) |
| Klatkaż | 25-30 fps (default smartfona) |
| Długość | 30 sekund per klip × 5 klipów per sesja = 2.5 min nagrań netto |
| Format | .mp4 lub .mov bez problemu |
| Oświetlenie | konsystentne w obrębie sesji (nie nagrywać częściowo w słońcu, częściowo w cieniu) |

## Procedura: 1 sesja = 5 klipów × różne sytuacje

Cel jest **uchwycić różne stany behavior'u w stałym kontekście wizualnym**. Bez indukowania bólu, bez stresowania konia. Wykorzystujemy naturalne, etyczne bodźce:

| # | sytuacja | typowo wywołuje |
|---|---|---|
| 1 | **Spokojny relaks** — koń stoi luźno, brak interakcji, otoczenie ciche | uszy luźne/na boki, oczy spokojne, postawa zrelaksowana |
| 2 | **Uwaga skierowana** — wprowadzenie znajomego bodźca (właściciel woła z zewnątrz, ktoś przechodzi) | uszy do przodu, oczy szeroko otwarte |
| 3 | **Praca pod jeźdźcem (lekka)** — chód lub stęp, normalna lekcja, bez forsowania | różne stany w zależności od konia + sytuacji |
| 4 | **Coś co wymaga kompensacji** — nierówne podłoże, zmiana kierunku, krótki kłus po stępie | model behavior pod lekkim wysiłkiem |
| 5 | **Po wysiłku, relaks** — koniec lekcji, rozluźnienie | często powrót do spokojnego stanu, ale czasem sygnały zmęczenia |

Można dostosować — kluczowe jest **różnorodność stanów w obrębie tej samej sesji wizualnej**.

## Co NIE robić

- ❌ Nie nagrywać 30 minut ciągiem — split na 5 osobnych klipów
- ❌ Nie zmieniać pozycji kamery między klipami w obrębie sesji
- ❌ Nie miksować świateł (część w hali, część na padoku) w jednej sesji
- ❌ Nie nagrywać close-upów — całą sylwetka
- ❌ Nie indukować strachu/bólu — bodźce muszą być etyczne

## Ramy etyczne (must-read)

**Welfare > PoC.** Jeśli koń wykazuje sygnały bólu w trakcie nagrywania, sesja się przerywa, koń idzie do weterynarza. Bez wyjątku.

**Brak indukowanego bólu/strachu.** Wszystkie bodźce wymienione powyżej są naturalnymi elementami codziennej pracy z koniem.

**RODO**: jeźdźcy/właściciele/osoby w kadrze muszą podpisać **pisemną zgodę** (szablon poniżej + drugi szablon dla osób trzecich w kadrze). Możliwość wycofania zgody w dowolnym momencie. Kontakt: piotr.pawluk@gmail.com.

**Co dzieje się z surowymi klipami**: przechowywane lokalnie na moim laptopie (zaszyfrowany dysk), używane wyłącznie do treningu/walidacji modelu, NIE publikowane jako wideo w żadnej formie. Jeśli writeup będzie publikowany, opisuje tylko *embeddingi* i *metryki accuracy*, nie identyfikowalne wideo.

## Co przesłać + format nazewnictwa

Po nagraniu sesji prześlij mi wszystkie 5 klipów + krótki opis (ramka tekstowa wystarczy):

```
{horse_name}_{session_id}_{situation}_{date}.mp4

np. miszka_s01_relaks_2026-05-12.mp4
    miszka_s01_uwaga_2026-05-12.mp4
    miszka_s01_praca_2026-05-12.mp4
    miszka_s01_kompensacja_2026-05-12.mp4
    miszka_s01_porelaks_2026-05-12.mp4
```

W mailu lub formularzu — krótki opis konia (rasa, wiek, doświadczenie, ewentualne historyczne problemy), opis sesji (gdzie, kiedy, kto jeździł, jakie bodźce użyto).

**Sposób przesłania**: WeTransfer/Google Drive/Dropbox z linkiem do mnie (piotr.pawluk@gmail.com), albo dysk na spotkanie. Nie wysyłaj przez Messengera — kompresuje wideo i degraduje jakość.

## Szablon zgody (do wydruku, podpisania przez jeźdźca/właściciela)

```
ZGODA NA WYKORZYSTANIE NAGRAŃ DO PROJEKTU BADAWCZEGO

Ja, [imię i nazwisko], wyrażam zgodę na nagrywanie i wykorzystanie
nagrań wideo z udziałem mojej osoby i/lub mojego konia [imię konia]
w ramach projektu badawczego "horse-pain-poc" prowadzonego przez
Piotra Pawluk (piotr.pawluk@gmail.com).

Przyjmuję do wiadomości, że:
- Nagrania będą używane wyłącznie do treningu i walidacji modelu AI
- Surowe nagrania nie będą publikowane w żadnej formie
- Mogę wycofać zgodę w dowolnym momencie (kontakt mailowy)
- W publikacjach naukowych mogą pojawić się anonimowe metryki / embeddingi,
  bez możliwości identyfikacji konia ani osoby

Data: ______________

Podpis: ______________
```

## Po stronie współpracownika — minimum

1. Przeczytaj ten dokument w całości (5 minut)
2. Daj znać czy masz pytania (mail / Issues w repo)
3. Umów ze mną sesję nagraniową lub nagraj samemu wg powyższych zasad
4. Prześlij pliki + krótki opis

Każdy zwrot z konkretnym pytaniem o protocol = realny współpracownik. Każde "spadnij, mam 50 klipów z YouTube'a" = polite decline (tamte klipy mają session leakage z definicji).

---

**Aktualizacje protocolu**: ten dokument jest wersjonowany w repo. Jeśli go zmienię, dam znać współpracownikom przed kolejną sesją nagraniową.
