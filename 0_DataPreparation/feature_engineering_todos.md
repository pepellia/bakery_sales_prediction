# Feature Engineering und Modellierung TODOs

## Modellierungsansätze
- [ ] Separate Modelle pro Warengruppe
    - [ ] Ein Modell für jede einzelne Warengruppe trainieren
    - [ ] Features für jedes Modell:
        - Zeitliche Features
        - Wetter-Features
        - Event-Features
    - [ ] Vergleich der Performance zwischen den Warengruppen
    - [ ] Analyse, welche Features für welche Warengruppe wichtig sind

- [ ] Ein gemeinsames Modell für alle Warengruppen
    - [ ] Warengruppe als kategorisches Feature
    - [ ] Modell lernt Abhängigkeiten zwischen Warengruppen implizit
    - [ ] Analyse der gelernten Zusammenhänge zwischen Warengruppen
    - [ ] Vergleich der Performance mit separaten Modellen

## Temperatur-Features
- [ ] Kontinuierliche Temperatur als Feature beibehalten
- [ ] Zusätzliche Temperatur-Kategorisierung einführen:
    - [ ] Basis-Kategorien definieren (kalt, mild, warm, etc.)
    - [ ] Jahreszeitabhängige Kategorien entwickeln
        - z.B. 15°C im Winter = "warm", im Sommer = "kühl"
        - Jahreszeiten definieren:
            - Winter: Dez, Jan, Feb
            - Frühling: März, Apr, Mai
            - Sommer: Jun, Jul, Aug
            - Herbst: Sep, Okt, Nov
    - [ ] Schwellenwerte für jede Jahreszeit definieren

## Datum-basierte Features
- [ ] Bestehende Features beibehalten:
    - Jahr
    - Monat
    - Wochentag
- [ ] Neue Features hinzufügen:
    - [ ] Tag im Monat (1-31)
    - [ ] Woche im Jahr (1-52)
    - [ ] Position im Monat:
        - Anfang (Tag 1-10)
        - Mitte (Tag 11-20)
        - Ende (Tag 21-31)
    - [ ] Quartal (1-4)
    - [ ] Jahreszeit (als kategorisches Feature)
    - [ ] Ist Wochenende (Boolean)

## Event-Features erweitern
- [ ] Feiertage
    - [ ] Ist Feiertag (Boolean)
    - [ ] Tage bis zum nächsten Feiertag
    - [ ] Tage seit letztem Feiertag
- [ ] Schulferien
    - [ ] Ist Schulferien (Boolean)
    - [ ] Ferientyp (Sommer, Herbst, etc.)
- [ ] Lokale Events
    - [ ] Kieler Woche (bereits vorhanden)
    - [ ] Windjammer (bereits vorhanden)
    - [ ] Weitere lokale Events?

## Wetter-Features erweitern
- [ ] Bestehend:
    - Temperatur
    - Bewölkung
    - Windgeschwindigkeit
- [ ] Neu hinzufügen:
    - [ ] Niederschlag
    - [ ] Luftfeuchtigkeit
    - [ ] Wetteränderung zum Vortag
    - [ ] Wettervorhersage (falls verfügbar)

## Vergleich und Evaluation
- [ ] Metriken für Modellvergleich definieren
- [ ] Vergleichsanalyse:
    - [ ] Performance-Vergleich (separate vs. gemeinsames Modell)
    - [ ] Feature-Importance für verschiedene Ansätze
    - [ ] Vor- und Nachteile der Ansätze dokumentieren
- [ ] Fehleranalyse:
    - [ ] Wo macht welches Modell die größten Fehler?
    - [ ] Gibt es systematische Unterschiede?
    - [ ] Welcher Ansatz eignet sich besser für welche Szenarien?

## Nächste Schritte
1. Implementierung beider Modellierungsansätze
2. Feature Engineering für beide Ansätze
3. Training und Evaluation
4. Vergleichsanalyse
5. Entscheidung für finalen Ansatz (oder Kombination?)

## Fragen zu klären
- Welche weiteren lokalen Events sind relevant?
- Welche Wetter-Daten sind verfügbar?
- Wie weit in die Vergangenheit reichen die Daten?
- Gibt es saisonale Produkte/Warengruppen?
- Welche Warengruppen haben starke Abhängigkeiten untereinander?
- Wie stark unterscheiden sich die Einflussfaktoren zwischen den Warengruppen?