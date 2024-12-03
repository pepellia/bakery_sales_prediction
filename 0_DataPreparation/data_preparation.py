import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os

class DataPreparation:
    def __init__(self):
        """
        Initialisiert die DataPreparation-Klasse.
        Speichert Transformationen, die auf Trainingsdaten berechnet wurden.
        """
        self.train_temp_mean = None
        self.train_temp_std = None
        self.numerical_scaler = StandardScaler()
        self.categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Temperatur-Kategorien für verschiedene Jahreszeiten
        self.temp_categories = {
            'winter': {'kalt': (-float('inf'), 5), 'mild': (5, 10), 'warm': (10, float('inf'))},
            'spring': {'kalt': (-float('inf'), 10), 'mild': (10, 15), 'warm': (15, float('inf'))},
            'summer': {'kalt': (-float('inf'), 15), 'mild': (15, 20), 'warm': (20, float('inf'))},
            'fall': {'kalt': (-float('inf'), 8), 'mild': (8, 13), 'warm': (13, float('inf'))}
        }
    
    def load_data(self):
        """
        Lädt alle benötigten Datensätze.
        """
        # Get the current directory (where data_preparation.py is located)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Hauptdaten laden
        umsatz_df = pd.read_csv(os.path.join(current_dir, "umsatzdaten_gekuerzt.csv"))
        wetter_df = pd.read_csv(os.path.join(current_dir, "wetter.csv"))
        kiwo_df = pd.read_csv(os.path.join(current_dir, "kiwo.csv"))
        
        # Datum in datetime umwandeln
        for df in [umsatz_df, wetter_df, kiwo_df]:
            df['Datum'] = pd.to_datetime(df['Datum'])
            
        # Daten zusammenführen
        merged_df = umsatz_df.merge(wetter_df, on="Datum", how="left")
        final_df = merged_df.merge(kiwo_df, on="Datum", how="left")
        
        # KielerWoche mit 0 auffüllen, wo NaN
        final_df['KielerWoche'] = final_df['KielerWoche'].fillna(0)
        
        return final_df
    
    def split_data(self, df):
        """
        Teilt die Daten zeitbasiert:
        - Training: 01.07.2013 bis 31.07.2017
        - Test: 01.08.2017 bis 31.07.2018
        """
        train_mask = (df['Datum'] >= '2013-07-01') & (df['Datum'] <= '2017-07-31')
        test_mask = (df['Datum'] >= '2017-08-01') & (df['Datum'] <= '2018-07-31')
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        print(f"Trainingsdaten: von {train_df['Datum'].min()} bis {train_df['Datum'].max()}")
        print(f"Testdaten: von {test_df['Datum'].min()} bis {test_df['Datum'].max()}")
        print(f"Anzahl Trainingssamples: {len(train_df)}")
        print(f"Anzahl Testsamples: {len(test_df)}")
        
        return train_df, test_df
    
    def get_season(self, month):
        """
        Bestimmt die Jahreszeit basierend auf dem Monat.
        """
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:  # 9, 10, 11
            return 'fall'
    
    def get_temp_category(self, temp, season):
        """
        Bestimmt die Temperaturkategorie basierend auf Temperatur und Jahreszeit.
        """
        categories = self.temp_categories[season]
        for category, (min_temp, max_temp) in categories.items():
            if min_temp <= temp < max_temp:
                return category
        return 'mild'  # Fallback
    
    def handle_missing_values(self, df, is_training=True):
        """
        Behandelt fehlende Werte.
        Berechnet Statistiken nur aus den Trainingsdaten.
        """
        df = df.copy()
        
        if is_training:
            # Berechne Statistiken aus Trainingsdaten
            self.train_temp_mean = df['Temperatur'].mean()
            self.train_temp_std = df['Temperatur'].std()
            
        # Wende die Behandlung an
        df['Temperatur'] = df['Temperatur'].fillna(self.train_temp_mean)
        df['Bewoelkung'] = df['Bewoelkung'].fillna(df['Bewoelkung'].mean())
        df['Windgeschwindigkeit'] = df['Windgeschwindigkeit'].fillna(df['Windgeschwindigkeit'].mean())
        
        return df
    
    def create_date_features(self, df):
        """
        Erstellt Features aus dem Datum.
        """
        df = df.copy()
        
        # Basis-Zeitfeatures
        df['Jahr'] = df['Datum'].dt.year
        df['Monat'] = df['Datum'].dt.month
        df['Wochentag'] = df['Datum'].dt.dayofweek
        
        # Erweiterte Zeitfeatures
        df['Tag_im_Monat'] = df['Datum'].dt.day
        df['Woche_im_Jahr'] = df['Datum'].dt.isocalendar().week
        df['Quartal'] = df['Datum'].dt.quarter
        df['ist_Wochenende'] = df['Wochentag'].isin([5, 6]).astype(int)
        
        # Position im Monat (Anfang/Mitte/Ende)
        df['Position_im_Monat'] = pd.cut(df['Tag_im_Monat'], 
                                       bins=[0, 10, 20, 31], 
                                       labels=['Anfang', 'Mitte', 'Ende'])
        
        # Jahreszeit
        df['Jahreszeit'] = df['Monat'].map(lambda x: self.get_season(x))
        
        return df
    
    def create_temperature_features(self, df):
        """
        Erstellt Features aus der Temperatur.
        """
        df = df.copy()
        
        # Kontinuierliche Temperatur beibehalten
        
        # Basis-Temperaturkategorien (jahreszeit-unabhängig)
        df['Temp_Kategorie_Basis'] = pd.cut(df['Temperatur'],
                                          bins=[-float('inf'), 10, 20, float('inf')],
                                          labels=['kalt', 'mild', 'warm'])
        
        # Jahreszeitabhängige Temperaturkategorien
        df['Temp_Kategorie_Saison'] = df.apply(
            lambda row: self.get_temp_category(row['Temperatur'], 
                                             self.get_season(row['Monat'])), 
            axis=1
        )
        
        return df
    
    def prepare_data(self, df, is_training=True):
        """
        Führt die gesamte Datenaufbereitung durch.
        """
        # Fehlende Werte behandeln
        df = self.handle_missing_values(df, is_training)
        
        # Features erstellen
        df = self.create_date_features(df)
        df = self.create_temperature_features(df)
        
        # Features in numerisch und kategorisch aufteilen
        numerical_features = [
            'Jahr', 'Monat', 'Wochentag', 'Tag_im_Monat',
            'Woche_im_Jahr', 'Quartal', 'ist_Wochenende',
            'Temperatur', 'Bewoelkung', 'Windgeschwindigkeit'
        ]
        
        categorical_features = [
            'Position_im_Monat', 'Jahreszeit', 
            'Temp_Kategorie_Basis', 'Temp_Kategorie_Saison',
            'Warengruppe'  # Warengruppe als kategorisches Feature
        ]
        
        # Feature Matrices erstellen
        X_numerical = df[numerical_features]
        X_categorical = df[categorical_features]
        
        # Standardisierung und Encoding
        if is_training:
            X_numerical_scaled = pd.DataFrame(
                self.numerical_scaler.fit_transform(X_numerical),
                columns=X_numerical.columns
            )
            X_categorical_encoded = pd.DataFrame(
                self.categorical_encoder.fit_transform(X_categorical),
                columns=self.categorical_encoder.get_feature_names_out(categorical_features)
            )
        else:
            X_numerical_scaled = pd.DataFrame(
                self.numerical_scaler.transform(X_numerical),
                columns=X_numerical.columns
            )
            X_categorical_encoded = pd.DataFrame(
                self.categorical_encoder.transform(X_categorical),
                columns=self.categorical_encoder.get_feature_names_out(categorical_features)
            )
        
        # Features zusammenführen
        X = pd.concat([X_numerical_scaled, X_categorical_encoded], axis=1)
        y = df['Umsatz']
        
        return X, y

# Beispiel für die Verwendung:
if __name__ == "__main__":
    # Instanz erstellen
    prep = DataPreparation()
    
    # Daten laden
    print("Lade Daten...")
    data = prep.load_data()
    
    # In Train und Test aufteilen
    print("\nTeile Daten in Training und Test...")
    train_data, test_data = prep.split_data(data)
    
    # Trainingsdaten aufbereiten
    print("\nBereite Trainingsdaten auf...")
    X_train, y_train = prep.prepare_data(train_data, is_training=True)
    
    # Testdaten aufbereiten
    print("\nBereite Testdaten auf...")
    X_test, y_test = prep.prepare_data(test_data, is_training=False)
    
    print("\nFeature-Übersicht:")
    print(f"Anzahl Features: {X_train.shape[1]}")
    print("\nNumerische Features:", X_train.filter(like='Temperatur').columns.tolist())
    print("\nKategorische Features (encoded):", X_train.filter(like='Warengruppe').columns.tolist())
    
    print("\nErste Zeilen der aufbereiteten Daten:")
    print(X_train.head())