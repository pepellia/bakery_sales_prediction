# Übersicht der Python-Skripte für Bakery Sales Prediction

## Aktuelle Skripte (Stand: Dezember 2024)

1. `analyse_umsatz.py` (18. November 2024)
   - Hauptskript für die explorative Datenanalyse der Umsatzdaten
   - Visualisierung von Umsatzverteilungen und Korrelationen
   - Analyse von Temperatur- und Wettereinflüssen

2. `create_windjammer_csv.py` (21. November 2024)
   - Erstellung der Windjammer-Ereignisdaten
   - Identifikation der Windjammerparade-Tage
   - Generierung der CSV-Datei mit Parade-Terminen

3. `analyse_windjammer.py` (21. November 2024)
   - Spezialisierte Analyse des Windjammerparade-Einflusses
   - Vergleich von normalen und Parade-Tagen
   - Berechnung von Umsatzunterschieden

4. `data_preparation.py` (3. Dezember 2024)
   - Zentrale Datenvorbereitungsklasse
   - Feature Engineering und Datenaufbereitung
   - Bereitstellung der Trainings- und Testdaten

5. `linear_regression_umsatz.py` (3. Dezember 2024)
   - Implementierung des linearen Regressionsmodells
   - Feature Engineering und Modelltraining
   - Modellauswertung und Visualisierungen

## Funktionen und Features

### Datenanalyse und Visualisierung
- Umsatzverteilungen und Trends
- Temperatur- und Wettereinflüsse
- Kieler Woche und Windjammerparade-Effekte
- Korrelationsanalysen

### Modellierung
- Feature Engineering mit zeitbasierten Features
- One-hot Encoding für kategorische Variablen
- Modelltraining und Evaluation
- Vorhersagegenauigkeitsanalysen

### Visualisierungen
- Umsatzvorhersagen (umsatz_prediction.png)
- Wochentagsanalysen (umsatz_by_weekday.png)
- Warengruppenanalysen (umsatz_by_group.png)
- Zeitliche Trends (umsatz_trend.png)
- Feature-Wichtigkeit (feature_importance.png)

## Verwendete Datendateien
- umsatzdaten_gekuerzt.csv (Hauptdatensatz)
- windjammer_dates.csv (Windjammerparade-Termine)
- wetter_data.csv (Wetterdaten)