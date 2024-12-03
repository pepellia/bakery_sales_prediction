# Baseline Model: Lineare Regression

## Zwei Ansätze

### 1. Komplettes Modell (`model_training.py`)
Nutzt alle verfügbaren Features für die Vorhersage.

**Features:**
- Numerische Features (10):
  - Jahr, Monat, Wochentag
  - Tag_im_Monat, Woche_im_Jahr, Quartal
  - ist_Wochenende
  - Temperatur, Bewoelkung, Windgeschwindigkeit

- Kategorische Features nach One-Hot-Encoding (19):
  - Position_im_Monat (3: Anfang, Mitte, Ende)
  - Jahreszeit (4: winter, spring, summer, fall)
  - Temp_Kategorie_Basis (3: kalt, mild, warm)
  - Temp_Kategorie_Saison (3: kalt, mild, warm)
  - Warengruppe (6: 1-6)

**Performance auf Testdaten (2017-08-01 bis 2018-07-31):**
- RMSE: 81.67
- MAE: 50.98
- R²: 0.6885

### 2. Vereinfachtes Modell (`evaluate_simple_model.py`)
Nutzt nur die 5 wichtigsten Features, die durch Feature-Selektion identifiziert wurden.

**Lineare Modellgleichung:**
```
Umsatz = 85.08 + 
         23.87 × ist_Wochenende + 
         69.39 × Jahreszeit_summer + 
         306.89 × Warengruppe_2 + 
         60.44 × Warengruppe_3 + 
         177.19 × Warengruppe_5
```

**Interpretation:**
- Grundumsatz: 85.08€ pro Tag
- +23.87€ am Wochenende
- +69.39€ im Sommer
- Warengruppen-Effekte:
  - Warengruppe 2: +306.89€
  - Warengruppe 3: +60.44€
  - Warengruppe 5: +177.19€

## Methodik

### Datenaufteilung
- Training: 01.07.2013 bis 31.07.2017
- Test: 01.08.2017 bis 31.07.2018

### Feature-Verarbeitung
1. Numerische Features:
   - Standardisierung (Mittelwert 0, Standardabweichung 1)
   - Behandlung fehlender Werte durch Mittelwert

2. Kategorische Features:
   - One-Hot-Encoding
   - Behandlung unbekannter Kategorien

### Modelltraining
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

### Evaluationsmetriken
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- Adjustiertes R² (für Feature-Selektion)

## Visualisierungen
- Feature Importance (`feature_importance.png`)
- Vorhersagen vs. tatsächliche Werte (`predictions_vs_actual.png`)
- RMSE pro Warengruppe (`rmse_by_group.png`)
- R² Score pro Warengruppe (`r2_by_group.png`)
- Vorhersagefehler pro Warengruppe (`prediction_errors.png`)
- Zeitreihenvergleich pro Warengruppe (`time_series_comparison_group_*.png`)
