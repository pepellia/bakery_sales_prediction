import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Daten einlesen
umsatz_df = pd.read_csv('umsatzdaten_gekuerzt.csv')
windjammer_df = pd.read_csv('windjammer.csv')

# Datum zu datetime konvertieren
umsatz_df['Datum'] = pd.to_datetime(umsatz_df['Datum'])
windjammer_df['Datum'] = pd.to_datetime(windjammer_df['Datum'])

# Wochentag hinzufügen (0 = Montag, 6 = Sonntag)
umsatz_df['Wochentag'] = umsatz_df['Datum'].dt.dayofweek

# Windjammerparade-Flag hinzufügen
umsatz_df['Windjammerparade'] = umsatz_df['Datum'].isin(windjammer_df['Datum']).astype(int)

# 1. Vergleich: Allgemeiner Umsatz vs. Umsatz bei Windjammerparade
print("\n1. Umsatzvergleich: Allgemein vs. Windjammerparade")
print("\nAllgemeiner Umsatz:")
print(umsatz_df[umsatz_df['Windjammerparade'] == 0]['Umsatz'].describe())
print("\nUmsatz an Windjammerparade-Tagen:")
print(umsatz_df[umsatz_df['Windjammerparade'] == 1]['Umsatz'].describe())

# 2. Vergleich: Samstage ohne Parade vs. Samstage mit Parade
samstage_df = umsatz_df[umsatz_df['Wochentag'] == 5]  # 5 = Samstag
print("\n2. Umsatzvergleich: Normale Samstage vs. Windjammerparade-Samstage")
print("\nUmsatz an normalen Samstagen:")
print(samstage_df[samstage_df['Windjammerparade'] == 0]['Umsatz'].describe())
print("\nUmsatz an Windjammerparade-Samstagen:")
print(samstage_df[samstage_df['Windjammerparade'] == 1]['Umsatz'].describe())

# Visualisierung
plt.figure(figsize=(12, 5))

# Box-Plot: Allgemeiner Vergleich
plt.subplot(1, 2, 1)
sns.boxplot(data=umsatz_df, x='Windjammerparade', y='Umsatz')
plt.title('Umsatzvergleich: Allgemein vs. Windjammerparade')
plt.xlabel('Windjammerparade')
plt.xticks([0, 1], ['Andere Tage', 'Parade'])

# Box-Plot: Samstags-Vergleich
plt.subplot(1, 2, 2)
sns.boxplot(data=samstage_df, x='Windjammerparade', y='Umsatz')
plt.title('Umsatzvergleich: Normale Samstage vs. Parade')
plt.xlabel('Windjammerparade')
plt.xticks([0, 1], ['Normale Samstage', 'Parade'])

plt.tight_layout()
plt.savefig('windjammer_vergleich.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nDie Visualisierung wurde in 'windjammer_vergleich.png' gespeichert.")

# Prozentuale Unterschiede berechnen
avg_normal = umsatz_df[umsatz_df['Windjammerparade'] == 0]['Umsatz'].mean()
avg_parade = umsatz_df[umsatz_df['Windjammerparade'] == 1]['Umsatz'].mean()
avg_normal_samstag = samstage_df[samstage_df['Windjammerparade'] == 0]['Umsatz'].mean()
avg_parade_samstag = samstage_df[samstage_df['Windjammerparade'] == 1]['Umsatz'].mean()

print(f"\nProzentuale Unterschiede:")
print(f"Parade vs. normale Tage: {((avg_parade / avg_normal) - 1) * 100:.1f}% mehr Umsatz")
print(f"Parade vs. normale Samstage: {((avg_parade_samstag / avg_normal_samstag) - 1) * 100:.1f}% mehr Umsatz")
