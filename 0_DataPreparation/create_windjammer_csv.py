import pandas as pd

# Kieler Woche Daten einlesen
kiwo_df = pd.read_csv('kiwo.csv')
kiwo_df['Datum'] = pd.to_datetime(kiwo_df['Datum'])

# Wochentag hinzufügen (0 = Montag, 6 = Sonntag)
kiwo_df['Wochentag'] = kiwo_df['Datum'].dt.dayofweek

# Gruppiere nach Jahr, um die Kieler Wochen zu identifizieren
kiwo_df['Jahr'] = kiwo_df['Datum'].dt.year
kiwo_gruppen = kiwo_df.groupby('Jahr')

# Liste für Windjammerparade-Tage
windjammer_tage = []

# Für jedes Jahr
for jahr, gruppe in kiwo_gruppen:
    # Finde alle Samstage (Wochentag 5) während der Kieler Woche
    samstage = gruppe[gruppe['Wochentag'] == 5].sort_values('Datum')
    
    # Wenn es mindestens zwei Samstage gibt, nimm den zweiten
    if len(samstage) >= 2:
        windjammer_tag = samstage.iloc[1]['Datum']
        windjammer_tage.append({
            'Datum': windjammer_tag,
            'Windjammerparade': 1
        })

# Erstelle DataFrame und speichere als CSV
windjammer_df = pd.DataFrame(windjammer_tage)
windjammer_df.to_csv('windjammer.csv', index=False)

print("Windjammerparade-Tage:")
print(windjammer_df['Datum'].dt.strftime('%Y-%m-%d').to_string(index=False))
