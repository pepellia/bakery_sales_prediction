import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sys
import os

# Add the DataPreparation directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), '0_DataPreparation'))
from data_preparation import DataPreparation
import matplotlib.pyplot as plt
import seaborn as sns

class SimpleModel:
    def __init__(self):
        # Koeffizienten aus der gefundenen Modellgleichung
        self.intercept = 85.08
        self.coefficients = {
            'ist_Wochenende': 23.87,
            'Jahreszeit_summer': 69.39,
            'Warengruppe_2': 306.89,
            'Warengruppe_3': 60.44,
            'Warengruppe_5': 177.19
        }
    
    def predict(self, X):
        """
        Macht Vorhersagen basierend auf der einfachen Modellgleichung
        """
        predictions = np.full(len(X), self.intercept)
        
        for feature, coef in self.coefficients.items():
            if feature in X.columns:
                predictions += coef * X[feature]
        
        return predictions

def evaluate_simple_model():
    # Daten laden
    print("Lade Daten...")
    data_prep = DataPreparation()
    data = data_prep.load_data()
    train_data, test_data = data_prep.split_data(data)
    
    # Features vorbereiten
    X_train, y_train = data_prep.prepare_data(train_data, is_training=True)
    X_test, y_test = data_prep.prepare_data(test_data, is_training=False)
    
    # Modell instanziieren und Vorhersagen machen
    model = SimpleModel()
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metriken berechnen
    train_metrics = {
        'R²': r2_score(y_train, y_pred_train),
        'Adj. R²': 1 - (1 - r2_score(y_train, y_pred_train)) * (len(y_train) - 1) / (len(y_train) - len(model.coefficients) - 1),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'MAE': mean_absolute_error(y_train, y_pred_train)
    }
    
    test_metrics = {
        'R²': r2_score(y_test, y_pred_test),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'MAE': mean_absolute_error(y_test, y_pred_test)
    }
    
    # Ergebnisse ausgeben
    print("\nTrainingsdaten (2013-07-01 bis 2017-07-31):")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nTestdaten (2017-08-01 bis 2018-07-31):")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Analyse pro Warengruppe
    print("\nAnalyse pro Warengruppe (Testdaten):")
    
    # Vorhersagen pro Warengruppe machen
    test_predictions = {}
    for group in range(1, 7):  # Alle Warengruppen 1-6
        group_data = test_data[test_data['Warengruppe'] == group].copy()
        if group in [2, 3, 5]:  # Nur für Warengruppen im Modell
            X_group, _ = data_prep.prepare_data(group_data, is_training=False)
            group_predictions = model.predict(X_group)
            test_predictions[group] = group_predictions
            
            group_rmse = np.sqrt(mean_squared_error(group_data['Umsatz'], group_predictions))
            group_mae = mean_absolute_error(group_data['Umsatz'], group_predictions)
            group_r2 = r2_score(group_data['Umsatz'], group_predictions)
            
            print(f"\nWarengruppe {group}:")
            print(f"RMSE: {group_rmse:.2f}")
            print(f"MAE: {group_mae:.2f}")
            print(f"R²: {group_r2:.4f}")
        else:
            print(f"\nWarengruppe {group}:")
            print("Nicht im Modell enthalten")
    
    # Visualisierungen
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Tatsächliche Werte')
    plt.ylabel('Vorhersagen')
    plt.title('Vorhersagen vs. Tatsächliche Werte\n(Testdaten: 2017-08-01 bis 2018-07-31)')
    plt.tight_layout()
    plt.savefig('simple_model_predictions.png')
    plt.close()
    
    # Zeitreihenplot für jede Warengruppe im Modell
    for group in [2, 3, 5]:
        plt.figure(figsize=(12, 6))
        group_data = test_data[test_data['Warengruppe'] == group].copy()
        group_data = group_data.sort_values('Datum')
        
        plt.plot(group_data['Datum'], group_data['Umsatz'], label='Tatsächlich', alpha=0.7)
        if group in test_predictions:
            plt.plot(group_data['Datum'], test_predictions[group], label='Vorhersage', alpha=0.7)
        
        plt.title(f'Zeitreihenvergleich - Warengruppe {group}')
        plt.xlabel('Datum')
        plt.ylabel('Umsatz')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'simple_model_timeseries_group_{group}.png')
        plt.close()

if __name__ == "__main__":
    evaluate_simple_model()