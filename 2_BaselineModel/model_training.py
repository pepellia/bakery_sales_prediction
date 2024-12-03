import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sys
import os

# Add the DataPreparation directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), '0_DataPreparation'))
from data_preparation import DataPreparation
import matplotlib.pyplot as plt
import seaborn as sns

class BakeryModel:
    def __init__(self):
        self.data_prep = DataPreparation()
        self.model = LinearRegression()
        sns.set_style("whitegrid")
    
    def train_model(self, train_data, verbose=True):
        """
        Trainiert ein gemeinsames Modell für alle Warengruppen.
        """
        # Features und Zielgröße vorbereiten
        X_train, y_train = self.data_prep.prepare_data(train_data, is_training=True)
        
        # Modell trainieren
        self.model.fit(X_train, y_train)
        
        # Trainingsmetriken berechnen
        y_pred = self.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        train_mae = mean_absolute_error(y_train, y_pred)
        train_r2 = r2_score(y_train, y_pred)
        
        if verbose:
            print("\nTrainingsmetriken (2013-07-01 bis 2017-07-31):")
            print(f"RMSE: {train_rmse:.2f}")
            print(f"MAE: {train_mae:.2f}")
            print(f"R²: {train_r2:.4f}")
            
            # Feature Importance analysieren und visualisieren
            self._analyze_feature_importance(X_train)
    
    def evaluate_model(self, test_data, verbose=True):
        """
        Evaluiert das Modell auf den Testdaten.
        """
        # Features und Zielgröße vorbereiten
        X_test, y_test = self.data_prep.prepare_data(test_data, is_training=False)
        
        # Vorhersagen machen
        y_pred = self.model.predict(X_test)
        
        # Metriken berechnen
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        
        if verbose:
            print("\nTestmetriken (2017-08-01 bis 2018-07-31):")
            print(f"RMSE: {test_rmse:.2f}")
            print(f"MAE: {test_mae:.2f}")
            print(f"R²: {test_r2:.4f}")
        
        # Vorhersagen vs. tatsächliche Werte visualisieren
        self._plot_predictions(test_data, y_test, y_pred)
        
        # Analyse pro Warengruppe
        self._analyze_by_group(test_data, y_pred)
        
        return test_rmse, test_mae, test_r2
    
    def _analyze_feature_importance(self, X):
        """
        Analysiert und visualisiert die Feature-Wichtigkeit.
        """
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': np.abs(self.model.coef_)
        })
        importance = importance.sort_values('Importance', ascending=False)
        
        print("\nTop 10 wichtigste Features:")
        print(importance.head(10))
        
        # Feature Importance Visualisierung
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance.head(10), x='Importance', y='Feature')
        plt.title('Top 10 wichtigste Features')
        plt.xlabel('Absolute Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    
    def _plot_predictions(self, test_data, y_true, y_pred):
        """
        Visualisiert Vorhersagen vs. tatsächliche Werte.
        """
        # Scatter Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Tatsächliche Werte')
        plt.ylabel('Vorhersagen')
        plt.title('Vorhersagen vs. Tatsächliche Werte\n(Testdaten: 2017-08-01 bis 2018-07-31)')
        plt.tight_layout()
        plt.savefig('predictions_vs_actual.png')
        plt.close()
        
        # Zeitreihenplot
        test_data = test_data.copy()
        test_data['Predictions'] = y_pred
        
        plt.figure(figsize=(15, 6))
        for group in sorted(test_data['Warengruppe'].unique()):
            group_data = test_data[test_data['Warengruppe'] == group]
            plt.figure(figsize=(15, 6))
            plt.plot(group_data['Datum'], group_data['Umsatz'], label='Tatsächlich', alpha=0.7)
            plt.plot(group_data['Datum'], group_data['Predictions'], label='Vorhersage', alpha=0.7)
            plt.title(f'Zeitreihenvergleich - Warengruppe {group}\n(Testdaten: 2017-08-01 bis 2018-07-31)')
            plt.xlabel('Datum')
            plt.ylabel('Umsatz')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'time_series_comparison_group_{group}.png')
            plt.close()
    
    def _analyze_by_group(self, test_data, predictions):
        """
        Analysiert die Modellperformance pro Warengruppe.
        """
        # Original Testdaten mit Vorhersagen zusammenführen
        test_data = test_data.copy()
        test_data['predictions'] = predictions
        
        # Performance-Metriken pro Warengruppe
        metrics = []
        print("\nAnalyse pro Warengruppe:")
        for group in sorted(test_data['Warengruppe'].unique()):
            group_data = test_data[test_data['Warengruppe'] == group]
            group_rmse = np.sqrt(mean_squared_error(group_data['Umsatz'], 
                                                  group_data['predictions']))
            group_mae = mean_absolute_error(group_data['Umsatz'], 
                                          group_data['predictions'])
            group_r2 = r2_score(group_data['Umsatz'], 
                              group_data['predictions'])
            
            metrics.append({
                'Warengruppe': group,
                'RMSE': group_rmse,
                'MAE': group_mae,
                'R2': group_r2
            })
            
            print(f"\nWarengruppe {group}:")
            print(f"RMSE: {group_rmse:.2f}")
            print(f"MAE: {group_mae:.2f}")
            print(f"R²: {group_r2:.4f}")
        
        # Performance-Visualisierung pro Warengruppe
        metrics_df = pd.DataFrame(metrics)
        
        # RMSE pro Warengruppe
        plt.figure(figsize=(10, 6))
        sns.barplot(data=metrics_df, x='Warengruppe', y='RMSE')
        plt.title('RMSE pro Warengruppe\n(Testdaten: 2017-08-01 bis 2018-07-31)')
        plt.tight_layout()
        plt.savefig('rmse_by_group.png')
        plt.close()
        
        # R² pro Warengruppe
        plt.figure(figsize=(10, 6))
        sns.barplot(data=metrics_df, x='Warengruppe', y='R2')
        plt.title('R² Score pro Warengruppe\n(Testdaten: 2017-08-01 bis 2018-07-31)')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.tight_layout()
        plt.savefig('r2_by_group.png')
        plt.close()
        
        # Boxplot der Vorhersagefehler pro Warengruppe
        test_data['error'] = test_data['Umsatz'] - test_data['predictions']
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=test_data, x='Warengruppe', y='error')
        plt.title('Vorhersagefehler pro Warengruppe\n(Testdaten: 2017-08-01 bis 2018-07-31)')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.tight_layout()
        plt.savefig('prediction_errors.png')
        plt.close()

if __name__ == "__main__":
    # Modell instanziieren
    print("Initialisiere Modell...")
    model = BakeryModel()
    
    # Daten laden
    print("\nLade Daten...")
    data = model.data_prep.load_data()
    
    # Daten aufteilen
    print("\nTeile Daten in Training und Test...")
    train_data, test_data = model.data_prep.split_data(data)
    
    # Modell trainieren
    print("\nTrainiere Modell...")
    model.train_model(train_data)
    
    # Modell evaluieren
    print("\nEvaluiere Modell...")
    model.evaluate_model(test_data)