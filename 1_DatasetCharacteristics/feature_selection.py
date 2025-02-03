# Required libraries
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Union, Tuple, Any
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from itertools import combinations
from datetime import datetime

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Import configuration
from bakery_sales_prediction.config import (TRAIN_PATH, WARENGRUPPEN, 
                                          get_warengruppe_name, WEATHER_PATH, 
                                          KIWO_PATH, WINDJAMMER_PATH)

class FeatureSelector:
    def __init__(self):
        self.best_features: Optional[List[str]] = None
        self.best_model: Optional[LinearRegression] = None
        self.best_adj_r2: float = -float('inf')
        self.output_dir: str = os.path.join(SCRIPT_DIR, 'output')
        self.viz_dir: str = os.path.join(SCRIPT_DIR, 'visualizations')
        
        # Create output directories
        for directory in [self.output_dir, self.viz_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare data with all available features"""
        print("Loading and preparing data...")
        
        # Load main sales data
        df = pd.read_csv(TRAIN_PATH)
        df['Datum'] = pd.to_datetime(df['Datum'])
        
        # Load and merge additional data
        weather_df = pd.read_csv(WEATHER_PATH)
        weather_df['Datum'] = pd.to_datetime(weather_df['Datum'])
        
        kiwo_df = pd.read_csv(KIWO_PATH)
        kiwo_df['Datum'] = pd.to_datetime(kiwo_df['Datum'])
        kiwo_df['is_kiwo'] = 1
        
        windjammer_df = pd.read_csv(WINDJAMMER_PATH)
        windjammer_df['Datum'] = pd.to_datetime(windjammer_df['Datum'])
        windjammer_df['is_windjammer'] = 1
        
        # Merge datasets
        df = pd.merge(df, weather_df, on='Datum', how='left')
        df = pd.merge(df, kiwo_df[['Datum', 'is_kiwo']], on='Datum', how='left')
        df = pd.merge(df, windjammer_df[['Datum', 'is_windjammer']], on='Datum', how='left')
        
        # Fill missing values
        df['is_kiwo'] = df['is_kiwo'].fillna(0)
        df['is_windjammer'] = df['is_windjammer'].fillna(0)
        
        # Fill missing weather data with median values
        numeric_columns = df.select_dtypes(include='number').columns.tolist()
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Create features
        df['Jahr'] = df['Datum'].dt.year
        df['Monat'] = df['Datum'].dt.month
        df['Wochentag'] = df['Datum'].dt.dayofweek
        df['Tag_im_Monat'] = df['Datum'].dt.day
        df['Woche_im_Jahr'] = df['Datum'].dt.isocalendar().week
        df['is_weekend'] = df['Wochentag'].isin([5, 6]).astype(int)
        
        # Create categorical features
        df['month_position'] = pd.cut(df['Tag_im_Monat'], 
                                    bins=[0, 10, 20, 31], 
                                    labels=['start', 'middle', 'end'])
        month_pos_dummies = pd.get_dummies(df['month_position'], prefix='month_pos')
        
        df['season'] = pd.cut(df['Monat'], 
                             bins=[0, 3, 6, 9, 12], 
                             labels=['winter', 'spring', 'summer', 'fall'])
        season_dummies = pd.get_dummies(df['season'], prefix='season')
        
        # Product group dummies
        product_dummies = pd.get_dummies(df['Warengruppe'], prefix='Warengruppe')
        
        # Temperature categories
        if 'Temperatur' in df.columns:
            df['temp_category'] = pd.cut(df['Temperatur'], 
                                       bins=[-np.inf, 5, 15, 25, np.inf],
                                       labels=['cold', 'mild', 'warm', 'hot'])
            temp_dummies = pd.get_dummies(df['temp_category'], prefix='temp')
            
            # Add all dummy variables
            df = pd.concat([df, month_pos_dummies, season_dummies, 
                          product_dummies, temp_dummies], axis=1)
        
        return df
    
    def analyze_missing_values(self, df):
        """Analyze missing values in the dataset"""
        print("\nAnalyzing missing values...")
        
        # Calculate missing value statistics
        missing_stats = pd.DataFrame({
            'Missing Values': df.isnull().sum(),
            'Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        missing_stats = missing_stats[missing_stats['Missing Values'] > 0]
        
        if not missing_stats.empty:
            # Save missing value statistics
            missing_stats.to_csv(os.path.join(self.output_dir, 'missing_values.csv'))
            
            # Visualize missing values
            plt.figure(figsize=(12, 6))
            sns.barplot(data=missing_stats.reset_index(), 
                       x='index', y='Percentage')
            plt.title('Missing Values by Feature')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'missing_values.png'))
            plt.close()
    
    def analyze_feature_distributions(self, df: pd.DataFrame) -> None:
        """Analyze and visualize feature distributions"""
        print("\nAnalyzing feature distributions...")
        
        numeric_features = df.select_dtypes(include='number').columns.tolist()
        
        # Create distribution plots
        for i, feature in enumerate(numeric_features):
            if feature not in ['Warengruppe', 'Umsatz']:
                plt.figure(figsize=(10, 6))
                sns.histplot(data=df, x=feature, stat='count', binwidth=None)
                plt.title(f'Distribution of {feature}')
                plt.tight_layout()
                plt.savefig(os.path.join(self.viz_dir, f'dist_{feature}.png'))
                plt.close()
        
        # Save distribution statistics
        dist_stats = df[numeric_features].describe()
        dist_stats.to_csv(os.path.join(self.output_dir, 'feature_statistics.csv'))
    
    def analyze_correlations(self, df: pd.DataFrame) -> None:
        """Analyze feature correlations"""
        print("Analyzing feature correlations...")
        
        numeric_features = df.select_dtypes(include='number').columns.tolist()
        corr_matrix = df[numeric_features].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlations')
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'correlation_matrix.png'))
        plt.close()
        
        # Save correlation matrix
        corr_matrix.to_csv(os.path.join(self.output_dir, 'correlation_matrix.csv'))
    
    def select_features_univariate(self, X: pd.DataFrame, y: pd.Series, k: int = 10) -> pd.DataFrame:
        """Select top k features using univariate selection"""
        print(f"\nSelecting top {k} features using univariate selection...")
        
        # Handle any remaining missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X_imputed, y)
        
        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': selector.scores_
        })
        feature_scores = feature_scores.sort_values('Score', ascending=False)
        
        # Visualize feature scores
        plt.figure(figsize=(12, 6))
        sns.barplot(data=feature_scores.head(k), x='Score', y='Feature')
        plt.title(f'Top {k} Features by F-score')
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'univariate_feature_scores.png'))
        plt.close()
        
        return feature_scores
    
    def adjusted_r2(self, r2: float, n: int, p: int) -> float:
        """Calculate adjusted R²"""
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    def evaluate_feature_combination(self, X: pd.DataFrame, y: pd.Series, feature_cols: List[str]) -> Tuple[float, float, LinearRegression]:
        """Evaluate a combination of features"""
        X_subset = X[feature_cols]
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X_subset), columns=X_subset.columns)
        
        model = LinearRegression()
        model.fit(X_imputed, y)
        y_pred = model.predict(X_imputed)
        
        r2 = float(r2_score(y, y_pred))
        adj_r2 = float(self.adjusted_r2(r2, len(y), len(feature_cols)))
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        return adj_r2, rmse, model
    
    def find_best_features(self, max_features: int = 10, verbose: bool = True) -> tuple[Optional[List[str]], Optional[LinearRegression], float]:
        """Find the best feature combination"""
        # Load and prepare data
        df = self.load_and_prepare_data()
        
        # Analyze missing values
        self.analyze_missing_values(df)
        
        # Analyze distributions and correlations
        self.analyze_feature_distributions(df)
        self.analyze_correlations(df)
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in 
                       ['Datum', 'Umsatz', 'Warengruppe', 'month_position', 
                        'season', 'temp_category', 'Temperatur']]
        X = df[feature_cols]
        y = df['Umsatz']
        
        # Perform univariate feature selection
        feature_scores = self.select_features_univariate(X, y, k=max_features)
        
        print(f"\nSearching for best combination of up to {max_features} features...")
        
        # Test different feature combinations
        for n_features in range(1, max_features + 1):
            if verbose:
                print(f"\nTesting combinations with {n_features} features...")
            
            top_features = feature_scores.head(n_features)['Feature'].tolist()
            adj_r2, rmse, model = self.evaluate_feature_combination(X, y, top_features)
            
            if adj_r2 > self.best_adj_r2:
                self.best_adj_r2 = adj_r2
                self.best_features = top_features
                self.best_model = model
                
                if verbose:
                    print(f"New best combination found!")
                    print(f"Features: {top_features}")
                    print(f"Adj. R²: {adj_r2:.4f}")
                    print(f"RMSE: {rmse:.2f}")
        
        # Save results
        self.save_results()
        
        return self.best_features, self.best_model, self.best_adj_r2
    
    def save_results(self) -> None:
        """Save analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_dir, f'feature_selection_results_{timestamp}.txt')
        
        with open(results_file, 'w') as f:
            f.write("Feature Selection Results\n")
            f.write("=======================\n\n")
            
            f.write("Best Feature Combination:\n")
            if isinstance(self.best_features, list) and self.best_features:
                for feature in self.best_features:
                    if isinstance(feature, str) and feature.startswith('Warengruppe_'):
                        try:
                            group_nr = int(feature.split('_')[1])
                            group_name = get_warengruppe_name(group_nr)
                            f.write(f"- {feature} ({group_name})\n")
                        except (IndexError, ValueError):
                            f.write(f"- {feature}\n")
                    else:
                        f.write(f"- {feature}\n")
            else:
                f.write("No features selected - model not fitted\n")
            
            f.write(f"\nAdjusted R²: {self.best_adj_r2:.4f}\n")
            
            f.write("\nModel Coefficients:\n")
            if isinstance(self.best_features, list) and self.best_features and self.best_model is not None:
                features_list = list(self.best_features)
                coefficients = self.best_model.coef_
                if len(features_list) == len(coefficients):
                    coef_dict = dict(zip(features_list, coefficients))
                    for feature, coef in coef_dict.items():
                        f.write(f"{feature}: {coef:.4f}\n")
                    
                    f.write(f"\nIntercept: {self.best_model.intercept_:.4f}\n")
                else:
                    f.write("Error: Mismatch between features and coefficients\n")
            else:
                f.write("No model coefficients available - model not fitted\n")
        
        print(f"\nResults saved to: {results_file}")

def main():
    selector = FeatureSelector()
    best_features, best_model, best_adj_r2 = selector.find_best_features(max_features=15)

if __name__ == "__main__":
    main()
