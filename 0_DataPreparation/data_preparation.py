import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
from config import (TRAIN_PATH, TEST_PATH, WEATHER_PATH, KIWO_PATH,
                   HOLIDAYS_PATH, SCHOOL_HOLIDAYS_PATH, FEATURES_DIR)

class DataPreparation:
    def __init__(self):
        """
        Initialize the DataPreparation class.
        Stores transformations computed on training data.
        """
        self.train_temp_mean = None
        self.train_temp_std = None
        self.numerical_scaler = StandardScaler()
        self.categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
        
        # Temperature categories for different seasons
        self.temp_categories = {
            'winter': {'cold': (-float('inf'), 5), 'mild': (5, 10), 'warm': (10, float('inf'))},
            'spring': {'cold': (-float('inf'), 10), 'mild': (10, 15), 'warm': (15, float('inf'))},
            'summer': {'cold': (-float('inf'), 15), 'mild': (15, 20), 'warm': (20, float('inf'))},
            'fall': {'cold': (-float('inf'), 8), 'mild': (8, 13), 'warm': (13, float('inf'))}
        }
    
    def load_data(self):
        """
        Load all required datasets.
        """
        print("Loading and preparing data...")
        
        # Load main data (YYYY-MM-DD format)
        train_df = pd.read_csv(TRAIN_PATH)
        wetter_df = pd.read_csv(WEATHER_PATH)
        kiwo_df = pd.read_csv(KIWO_PATH)
        
        # Convert dates for main data (YYYY-MM-DD format)
        for df in [train_df, wetter_df, kiwo_df]:
            df['Datum'] = pd.to_datetime(df['Datum'])
        
        # Load new datasets with semicolon separator (DD.MM.YYYY format)
        feiertage_df = pd.read_csv(HOLIDAYS_PATH, sep=';')
        schulferien_df = pd.read_csv(SCHOOL_HOLIDAYS_PATH, sep=';')
        
        # Convert dates for new datasets (DD.MM.YYYY format)
        for df in [feiertage_df, schulferien_df]:
            df['Datum'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y')
        
        # Rename 'Ferientag' to 'Schulferien'
        schulferien_df = schulferien_df.rename(columns={'Ferientag': 'Schulferien'})
        
        print(f"Number of holidays: {feiertage_df['Feiertag'].sum()}")
        print(f"Number of school holidays: {schulferien_df['Schulferien'].sum()}")
        
        # Merge data
        merged_df = train_df.merge(wetter_df, on="Datum", how="left")
        merged_df = merged_df.merge(kiwo_df, on="Datum", how="left")
        merged_df = merged_df.merge(feiertage_df, on="Datum", how="left")
        merged_df = merged_df.merge(schulferien_df, on="Datum", how="left")
        
        # Fill missing values
        merged_df['KielerWoche'] = merged_df['KielerWoche'].fillna(0)
        merged_df['Feiertag'] = merged_df['Feiertag'].fillna(0)
        merged_df['Schulferien'] = merged_df['Schulferien'].fillna(0)
        
        print(f"Dataset contains {len(merged_df)} rows")
        
        return merged_df
    
    def save_features(self, X, y, filename):
        """
        Save generated features to a file
        """
        # Combine features and target
        data = pd.concat([X, y], axis=1)
        
        # Create output path
        output_path = os.path.join(FEATURES_DIR, filename)
        
        # Save to CSV
        data.to_csv(output_path, index=False)
        print(f"Features saved to: {output_path}")
    
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
            'Warengruppe', 'Feiertag', 'Schulferien'  # Warengruppe als kategorisches Feature
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

# Example usage:
if __name__ == "__main__":
    # Create instance
    prep = DataPreparation()
    
    # Load data
    print("Loading data...")
    data = prep.load_data()
    
    # Split into train and test
    print("\nSplitting data into training and test...")
    train_data, test_data = prep.split_data(data)
    
    # Prepare training data
    print("\nPreparing training data...")
    X_train, y_train = prep.prepare_data(train_data, is_training=True)
    
    # Save features
    prep.save_features(X_train, y_train, 'train_features.csv')
    
    # Prepare test data
    print("\nPreparing test data...")
    X_test, y_test = prep.prepare_data(test_data, is_training=False)
    
    # Save features
    prep.save_features(X_test, y_test, 'test_features.csv')