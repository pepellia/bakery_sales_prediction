import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import sys
import os
from itertools import combinations

# Add the DataPreparation directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), '0_DataPreparation'))
from data_preparation import DataPreparation

class FeatureSelector:
    def __init__(self):
        self.data_prep = DataPreparation()
        self.best_features = None
        self.best_model = None
        self.best_adj_r2 = -float('inf')
        
    def adjusted_r2(self, r2, n, p):
        """
        Berechnet das adjustierte R².
        n: Anzahl der Beobachtungen
        p: Anzahl der Features
        """
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    def evaluate_feature_combination(self, X, y, feature_cols):
        """
        Evaluiert eine Kombination von Features.
        """
        X_subset = X[list(feature_cols)]
        model = LinearRegression()
        model.fit(X_subset, y)
        y_pred = model.predict(X_subset)
        r2 = r2_score(y, y_pred)
        adj_r2 = self.adjusted_r2(r2, len(y), len(feature_cols))
        return adj_r2, model
    
    def find_best_features(self, max_features=10, verbose=True):
        """
        Findet die beste Feature-Kombination.
        """
        # Daten laden und vorbereiten
        data = self.data_prep.load_data()
        train_data, _ = self.data_prep.split_data(data)
        X_train, y_train = self.data_prep.prepare_data(train_data, is_training=True)
        
        all_features = X_train.columns.tolist()
        
        print(f"Suche beste Kombination aus maximal {max_features} Features...")
        print(f"Gesamtanzahl Features: {len(all_features)}")
        
        # Teste verschiedene Feature-Kombinationen
        for n_features in range(1, max_features + 1):
            if verbose:
                print(f"\nTeste Kombinationen mit {n_features} Features...")
            
            for feature_combo in combinations(all_features, n_features):
                adj_r2, model = self.evaluate_feature_combination(X_train, y_train, feature_combo)
                
                if adj_r2 > self.best_adj_r2:
                    self.best_adj_r2 = adj_r2
                    self.best_features = feature_combo
                    self.best_model = model
                    
                    if verbose:
                        print(f"Neue beste Kombination gefunden!")
                        print(f"Features: {feature_combo}")
                        print(f"Adj. R²: {adj_r2:.4f}")
        
        # Finale Ausgabe
        print("\nBeste Feature-Kombination gefunden:")
        print("Features:", self.best_features)
        print(f"Adjustiertes R²: {self.best_adj_r2:.4f}")
        
        # Modellgleichung erstellen
        coef_dict = dict(zip(self.best_features, self.best_model.coef_))
        intercept = self.best_model.intercept_
        
        print("\nLineare Modellgleichung:")
        equation = f"Umsatz = {intercept:.2f}"
        for feature, coef in coef_dict.items():
            equation += f" + ({coef:.2f} × {feature})"
        print(equation)
        
        return self.best_features, self.best_model, self.best_adj_r2

if __name__ == "__main__":
    selector = FeatureSelector()
    best_features, best_model, best_adj_r2 = selector.find_best_features(max_features=5)
