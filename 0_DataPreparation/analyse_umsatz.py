import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.switch_backend("Agg")

# Read CSV files
umsatz_df = pd.read_csv("umsatzdaten_gekuerzt.csv")
kiwo_df = pd.read_csv("kiwo.csv")
wetter_df = pd.read_csv("wetter.csv")

# Convert Datum to datetime type for all dataframes
umsatz_df["Datum"] = pd.to_datetime(umsatz_df["Datum"])
kiwo_df["Datum"] = pd.to_datetime(kiwo_df["Datum"])
wetter_df["Datum"] = pd.to_datetime(wetter_df["Datum"])

# Merge all dataframes
# First merge umsatz with kiwo
merged_df = umsatz_df.merge(kiwo_df, on="Datum", how="left")

# Then merge with weather data
final_df = merged_df.merge(wetter_df, on="Datum", how="left")

# Stelle sicher, dass KielerWoche als 0 und 1 codiert ist
final_df['KielerWoche'] = final_df['KielerWoche'].fillna(0)  # Falls NaN-Werte existieren
final_df['KielerWoche'] = final_df['KielerWoche'].astype(int)  # Konvertiere zu int

# Deskriptive Statistiken
print("Deskriptive Statistiken:")
print(final_df.describe())

# Separate Plots erstellen und speichern
# 1. Umsatzverteilung
plt.figure(figsize=(10, 6))
sns.histplot(data=final_df, x="Umsatz", bins=30)
plt.title("Verteilung der Umsätze")
plt.savefig("umsatz_verteilung.png")
plt.close()

# 2. Temperaturverteilung
plt.figure(figsize=(10, 6))
sns.histplot(data=final_df, x="Temperatur", bins=30)
plt.title("Verteilung der Temperaturen")
plt.savefig("temperatur_verteilung.png")
plt.close()

# 3. Umsatz vs. Temperatur
plt.figure(figsize=(10, 6))
sns.scatterplot(data=final_df, x="Temperatur", y="Umsatz")
plt.title("Umsatz vs. Temperatur")
plt.savefig("umsatz_vs_temperatur.png")
plt.close()

# Verbesserter Boxplot für Kieler Woche
plt.figure(figsize=(12, 6))
sns.boxplot(data=final_df, x='KielerWoche', y='Umsatz')
plt.title('Umsatz Vergleich: Während vs. Außerhalb der Kieler Woche')
plt.xlabel('Kieler Woche (0 = Außerhalb, 1 = Während)')
plt.ylabel('Umsatz')
plt.savefig('umsatz_kiwo_vergleich.png')
plt.close()

# Violin Plot für detailliertere Verteilungsvisualisierung
plt.figure(figsize=(12, 6))
sns.violinplot(data=final_df, x='KielerWoche', y='Umsatz')
plt.title('Umsatzverteilung: Während vs. Außerhalb der Kieler Woche')
plt.xlabel('Kieler Woche (0 = Außerhalb, 1 = Während)')
plt.ylabel('Umsatz')
plt.savefig('umsatz_kiwo_violin.png')
plt.close()

# Statistische Kennzahlen für Kieler Woche
print("\nStatistische Kennzahlen für Umsätze während/außerhalb der Kieler Woche:")
print(final_df.groupby('KielerWoche')['Umsatz'].describe())

# Korrelationen mit Umsatz
correlations = final_df[["Umsatz", "KielerWoche", "Temperatur", "Bewoelkung", "Windgeschwindigkeit"]].corr()
umsatz_corr = correlations.loc["Umsatz"]
# Sortiere nach absolutem Korrelationswert (außer Umsatz selbst)
umsatz_corr_sorted = pd.concat([
    pd.Series({'Umsatz': 1.0}),
    umsatz_corr[1:].abs().sort_values(ascending=False).map(lambda x: umsatz_corr[umsatz_corr.abs() == x].iloc[0])
])

print("\nKorrelationen mit Umsatz (sortiert nach Stärke):")
print(umsatz_corr_sorted)

# Korrelationsmatrix als Heatmap
plt.figure(figsize=(12, 2))
umsatz_corr_df = pd.DataFrame(umsatz_corr_sorted)
sns.heatmap(umsatz_corr_df, 
            annot=True,               # Zeigt die Werte an
            cmap="coolwarm",          # Farbschema
            center=0,                 # Zentriert die Farbskala bei 0
            fmt=".3f",                # Zeigt 3 Dezimalstellen
            annot_kws={"size": 10})   # Schriftgröße der Zahlen

plt.title("Korrelationen mit Umsatz (sortiert nach Stärke)")
plt.savefig("korrelationsmatrix.png", 
            bbox_inches='tight',      # Verhindert abgeschnittene Labels
            dpi=300)                  # Höhere Auflösung
plt.close()
