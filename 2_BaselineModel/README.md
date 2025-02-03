# Baseline-Modell für Bäckerei-Verkaufsprognosen

Dieses Verzeichnis enthält die Implementierung des Baseline-Modells für die Vorhersage von Bäckerei-Verkäufen. Das Modell verwendet grundlegende statistische Methoden und Wochentagsmuster, um Verkaufsprognosen zu erstellen.

## Skript-Übersicht

### analyze_submission.py
Analysiert die Vorhersageergebnisse und erstellt detaillierte Auswertungen der Modellperformance. Das Skript generiert verschiedene Visualisierungen und Metriken zur Bewertung der Vorhersagegenauigkeit.

### simple_weekday_model.py
Implementiert ein einfaches wochentagsbasiertes Vorhersagemodell. Dieses Modell berücksichtigt die durchschnittlichen Verkaufszahlen pro Wochentag als Grundlage für die Prognosen.

### simple_weekday_model_group1.py
Eine Variation des Wochentagsmodells, speziell angepasst für die erste Produktgruppe. Enthält spezifische Anpassungen und Optimierungen für diese Produktkategorie.

### model_training.py
Hauptskript für das Training des Baseline-Modells. Beinhaltet die Datenaufbereitung, Modelltraining und Speicherung der trainierten Modelle.

### weekday_model_by_product.py
Erweiterte Version des Wochentagsmodells, die separate Vorhersagen für jedes einzelne Produkt erstellt. Berücksichtigt produktspezifische Verkaufsmuster.

### evaluate_simple_model.py
Umfassendes Evaluierungsskript für das Baseline-Modell. Berechnet verschiedene Leistungsmetriken und erstellt Visualisierungen zur Modellbewertung.

## Verzeichnisstruktur

- `output/`: Enthält generierte Modellergebnisse und Vorhersagen
- `visualizations/`: Speicherort für erzeugte Grafiken und Visualisierungen

## Verwendung

1. Führen Sie zuerst `model_training.py` aus, um das Baseline-Modell zu trainieren
2. Nutzen Sie `weekday_model_by_product.py` oder `simple_weekday_model.py` für Vorhersagen
3. Analysieren Sie die Ergebnisse mit `analyze_submission.py`
4. Verwenden Sie `evaluate_simple_model.py` für eine detaillierte Modellbewertung

## Hinweise

- Die Modelle basieren hauptsächlich auf Wochentagsmustern
- Verschiedene Varianten des Modells sind für unterschiedliche Anwendungsfälle verfügbar
- Die Evaluierungsskripte bieten umfangreiche Möglichkeiten zur Performanceanalyse
