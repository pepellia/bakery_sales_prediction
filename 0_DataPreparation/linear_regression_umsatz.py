# Benötigte Bibliotheken importieren
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Daten einlesen
print("Lese Daten ein...")
df = pd.read_csv('umsatzdaten_gekuerzt.csv')
print(f"Datensatz geladen: {len(df)} Zeilen")

# Datum in datetime umwandeln
print("\nKonvertiere Datum...")
df['Datum'] = pd.to_datetime(df['Datum'])

# Feature Engineering
print("\nErstelle Features...")
df['Jahr'] = df['Datum'].dt.year
df['Monat'] = df['Datum'].dt.month
df['Wochentag'] = df['Datum'].dt.dayofweek  # 0 = Montag, 6 = Sonntag

# One-hot encoding für Warengruppe
print("Erstelle One-hot encoding für Warengruppen...")
warengruppen_dummies = pd.get_dummies(df['Warengruppe'], prefix='Warengruppe')
df = pd.concat([df, warengruppen_dummies], axis=1)

# Features für das Modell auswählen
feature_columns = ['Jahr', 'Monat', 'Wochentag'] + list(warengruppen_dummies.columns)
X = df[feature_columns]
y = df['Umsatz']

print("\nTeile Daten in Training und Test...")
# Daten in Trainings- und Testset aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTrainiere Modell...")
# Lineares Regressionsmodell erstellen und trainieren
model = LinearRegression()
model.fit(X_train, y_train)

print("\nMache Vorhersagen...")
# Vorhersagen machen
y_pred = model.predict(X_test)

# Modellperformance evaluieren
print('\nModell Performance:')
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
print('R² Score: %.2f' % r2_score(y_test, y_pred))

# Koeffizienten des Modells anzeigen
print('\nModell Koeffizienten:')
for idx, feature in enumerate(feature_columns):
    print(f'{feature}: {model.coef_[idx]:.4f}')

print("\nFertig! Modell wurde erfolgreich trainiert.")

print("\nErstelle Visualisierungen...")
import matplotlib.pyplot as plt
import seaborn as sns

# Stil für die Plots setzen
sns.set_theme(style="whitegrid")

# 1. Tatsächliche vs. Vorhergesagte Umsätze
plt.figure(figsize=(12, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Tatsächliche Umsätze')
plt.ylabel('Vorhergesagte Umsätze')
plt.title('Tatsächliche vs. Vorhergesagte Umsätze')
plt.tight_layout()
plt.savefig('umsatz_prediction.png')
plt.close()

# 2. Durchschnittlicher Umsatz nach Wochentag
plt.figure(figsize=(12, 6))
avg_by_day = df.groupby('Wochentag')['Umsatz'].mean()
avg_by_day.plot(kind='bar')
plt.title('Durchschnittlicher Umsatz nach Wochentag')
plt.xlabel('Wochentag (0 = Montag, 6 = Sonntag)')
plt.ylabel('Durchschnittlicher Umsatz')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('umsatz_by_weekday.png')
plt.close()

# 3. Durchschnittlicher Umsatz nach Warengruppe
plt.figure(figsize=(12, 6))
avg_by_group = df.groupby('Warengruppe')['Umsatz'].agg(['mean', 'std']).round(2)
avg_by_group['mean'].plot(kind='bar', yerr=avg_by_group['std'], capsize=5)
plt.title('Durchschnittlicher Umsatz nach Warengruppe (mit Standardabweichung)')
plt.xlabel('Warengruppe')
plt.ylabel('Durchschnittlicher Umsatz')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('umsatz_by_group.png')
plt.close()

# 4. Umsatztrend über die Zeit
plt.figure(figsize=(15, 6))
monthly_sales = df.groupby([df['Datum'].dt.to_period('M')])['Umsatz'].mean()
monthly_sales.plot(kind='line', marker='o')
plt.title('Durchschnittlicher Umsatz pro Monat')
plt.xlabel('Datum')
plt.ylabel('Durchschnittlicher Umsatz')
plt.grid(True)
plt.tight_layout()
plt.savefig('umsatz_trend.png')
plt.close()

# 5. Modellkoeffizienten Visualisierung
plt.figure(figsize=(12, 6))
coef_df = pd.DataFrame({'Feature': feature_columns, 'Coefficient': model.coef_})
coef_df = coef_df.sort_values('Coefficient', ascending=True)
sns.barplot(data=coef_df, x='Coefficient', y='Feature')
plt.title('Einfluss der Features auf den Umsatz')
plt.xlabel('Koeffizient')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("\nVisualisierungen wurden erstellt und gespeichert:")
print("1. umsatz_prediction.png - Tatsächliche vs. Vorhergesagte Umsätze")
print("2. umsatz_by_weekday.png - Umsatz nach Wochentag")
print("3. umsatz_by_group.png - Umsatz nach Warengruppe")
print("4. umsatz_trend.png - Umsatztrend über die Zeit")
print("5. feature_importance.png - Einfluss der Features")

# Zusätzliche statistische Zusammenfassung
print("\nStatistische Zusammenfassung nach Warengruppe:")
print(df.groupby('Warengruppe')['Umsatz'].describe().round(2))
