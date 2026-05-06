# Protocol nagrywania klipów — projekt RHpE PoC

*Wersja 2.0 — 2026-05-06. Warto przeczytać przed nagrywaniem — oszczędzi nam obu czasu.*

## Welfare zwierzęcia jest nadrzędne

Welfare > PoC. Bez wyjątków. To pierwsza zasada projektu.

- Jeśli koń w trakcie nagrywania pokazuje sygnały bólu lub silnego dyskomfortu — sesja się przerywa, koń idzie do weterynarza.
- Bodźce użyte w trakcie nagrywania są naturalnymi elementami codziennej pracy z koniem. Nie indukujemy strachu, nie prowokujemy bólu, nie wymuszamy stanów dyskomfortu.
- Jeśli cokolwiek w tym protocolu wydaje Ci się sprzeczne z welfare Twojego konia — nie rób tego, daj znać, dostosujemy.

## Co konkretnie próbujemy zrobić

**Klasyfikacja pojedynczych behaviors z RHpE** (ridden horse pain ethogram, Sue Dyson, 24 zachowania) — aktualnie pozycja uszu (ucho do przodu / w bok / do tyłu).

To **nie jest** detektor bólu. RHpE wymaga ≥8 z 24 behaviors razem żeby wnioskować o bólu mięśniowo-szkieletowym. Tu budujemy fundament — pojedyncze klasyfikatory per-behavior, które w odległej przyszłości złożą się w narzędzie wspomagające ocenę. Aktualne MVP = jedna z 24 cegiełek.

## Kto ocenia co widzi model na klipie

Ty nie. Twoja rola = **nagrać**. Ground truth (czyli "ten klip pokazuje uszy do przodu", "ten klip pokazuje uszy do tyłu") robi **certyfikowany RHpE assessor po fakcie**. Nie próbuj przewidywać co model "powinien" zobaczyć w klipie ani prowokować konkretnych zachowań — to zafałszowuje dane i przenosi etyczne dylematy na Ciebie.

Cała obietnica jest prostsza niż się wydaje: **nagraj normalną zróżnicowaną sesję treningową, jaką i tak robisz**.

## Po co osobny protocol

Pierwsza iteracja PoC zebrała 53 klipy bez konkretnych zasad nagrywania (różne ośrodki, telefon, YouTube, miks formatów). Sanity check ujawnił że model uczył się rozpoznawać sesję nagraniową (telefon Lesznowola vs YouTube documentary), a nie behavior konia. LOSO drop −34pp dla head_position, −49pp dla tail_movement. Pełna analiza: [`lessons_learned.md`](lessons_learned.md).

Wniosek: 30 klipów zebranych intuicyjnie ≠ 30 klipów które uczą model. Ten dokument opisuje co działa i dlaczego, żeby nasz wspólny czas miał sens.

## Co działa technicznie (i dlaczego)

| pozycja | rekomendacja | dlaczego |
| --- | --- | --- |
| Kamera | smartfon na statywie | drgania ręki cały czas zmieniają tło — model uczy się szumu zamiast konia |
| Kąt | jeden wybrany kąt na całą sesję; preferowany bok 90° lub ¾ z przodu | spójny widok pozwala modelowi porównywać klipy zamiast uczyć się różnic kąta |
| Odległość | cała sylwetka konia + ~30% margines | close-upy pomijają kontekst (postawa, ogon); zbyt szeroko gubi mimikę |
| Rozdzielczość | 1080p | więcej miejsca w przesyłaniu, model i tak downsampluje |
| Klatkaż | 25–30 fps (default smartfona) | bezbolesny default, nie kombinujmy |
| Długość | ~30 s per klip | krótsze gubi temporal context, dłuższe mieszają wiele behaviors |
| Format | .mp4 lub .mov | oba bez problemu |
| Oświetlenie | spójne w obrębie sesji | mieszanie hala / słońce / cień w jednej sesji = model znów uczy się oświetlenia |

## Procedura: 1 sesja = 5 klipów z normalnej sesji treningowej

Cel: uchwycić różne momenty zwykłej, codziennej sesji treningowej. Nie chodzi o specjalne aranżowanie — wystarczy nagrać 5 różnych momentów z tego co i tak robisz.

| # | typowy moment | typowy czas |
| --- | --- | --- |
| 1 | Moment przed lub po sesji — koń stoi luźno | ~30 s |
| 2 | Rozgrzewka — pierwsze minuty stępu | ~30 s |
| 3 | Główna część — stęp na luźnych wodzach albo lekki kłus | ~30 s |
| 4 | Zmiany kierunku, przejścia chód-stęp-chód | ~30 s |
| 5 | Rozluźnienie — koniec sesji, długie wodze | ~30 s |

To tylko sugestie momentów. Możesz dostosować pod swoją sesję — kluczowa jest **różnorodność momentów**, nie konkretna lista. Jeśli Twoja sesja wygląda inaczej (praca z ziemi, lonża, hipoterapia), wybierz 5 sensownie różnych momentów z tego co robisz.

**W trakcie nagrywania:**

- nie prowokuj konkretnych reakcji konia (cofania, spinania się, stanów stresu)
- nie wymuszaj sytuacji których normalnie nie ma w treningu
- nie modyfikuj jak pracujesz z koniem dla potrzeb nagrania

Im bardziej "naturalnie jak zawsze" — tym lepiej dla projektu.

## Co nie zadziała (i dlaczego warto wiedzieć)

- 30-minutowy ciągły film — pomijamy podział na momenty, trudniej analizować temporal context. Lepiej 5 osobnych klipów po 30 s.
- Kamera w ręku — drgania zmieniają tło, model uczy się drgań a nie konia.
- Mieszanie kątów / świateł / lokalizacji w jednej sesji — łamie spójność, robi z tego de facto różne sesje.
- Klipy z YouTube albo z innych źródeł — ich session leakage jest niepoznawalny, nie da się ich włączyć do balanced LOSO.
- Close-upy na samą głowę — gubimy postawę i ogon, model traci kontekst.

## RODO i własność intelektualna

Krótko: jeźdźcy / właściciele / osoby w kadrze podpisują pisemną zgodę przed sesją. To twardy wymóg prawny, niezależny ode mnie. Szablon zgody niżej.

Co dzieje się z surowymi klipami:

- przechowywane lokalnie na zaszyfrowanym dysku, nie w chmurze publicznej
- używane wyłącznie do treningu i walidacji modelu
- nie publikowane jako wideo w żadnej formie
- jeśli writeup będzie publikowany, zawiera tylko embeddingi i metryki accuracy — bez identyfikowalnych ujęć

Możliwość wycofania zgody w dowolnym momencie. Kontakt: <piotr.pawluk@gmail.com>.

## Format przesłania

Po nagraniu sesji proszę o przesłanie 5 klipów + krótki opis. Pomocne nazewnictwo:

```
{horse_name}_{session_id}_{moment}_{date}.mp4

np. miszka_s01_spokoj_2026-05-12.mp4
    miszka_s01_rozgrzewka_2026-05-12.mp4
    miszka_s01_glowna_2026-05-12.mp4
    miszka_s01_zmiany_2026-05-12.mp4
    miszka_s01_rozluzni_2026-05-12.mp4
```

W mailu / formularzu pomocny jest krótki opis: koń (rasa, wiek, doświadczenie, ewentualne historyczne problemy weterynaryjne), sesja (gdzie, kiedy, kto jeździł, jakie były naturalne bodźce w trakcie).

Sposób przesłania: WeTransfer / Google Drive / Dropbox z linkiem do mnie (<piotr.pawluk@gmail.com>), albo pendrive na spotkanie. Messenger kompresuje wideo i degraduje jakość — lepiej unikać.

## Szablon zgody (do wydruku / podpisania)

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

## Jak to wygląda od Twojej strony

1. Przeczytanie tego dokumentu (~5 minut). Pytania na priv / mailem / przez Issues w repo.
2. Umówienie sesji (sam, ze mną, jak wygodnie).
3. Nagranie 5 klipów wg powyższych wskazówek z normalnej sesji treningowej.
4. Przesłanie plików + krótkiego opisu.

Jeśli coś z tego nie pasuje albo masz pomysł jak zrobić to lepiej — daj znać. Protocol jest wersjonowany w repo, mogę go zaktualizować jeśli wynikną sensowne uwagi.

---

*Aktualizacje protocolu wersjonowane w repo. Jeśli zmienię, dam znać współpracownikom przed kolejną sesją.*
