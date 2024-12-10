import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.join(PROJECT_ROOT, '0_DataPreparation'))
from config import (TRAIN_PATH, VIZ_DIR, WARENGRUPPEN, get_warengruppe_name)

def prepare_features(df):
    """Prepare basic features for the model"""
    print("Preparing features...")
    
    # Convert date to datetime
    df['Datum'] = pd.to_datetime(df['Datum'])
    
    # Create basic temporal features
    df['Jahr'] = df['Datum'].dt.year
    df['Monat'] = df['Datum'].dt.month
    df['Wochentag'] = df['Datum'].dt.dayofweek
    
    # Create one-hot encoding for product groups
    product_dummies = pd.get_dummies(df['Warengruppe'], prefix='Warengruppe')
    df = pd.concat([df, product_dummies], axis=1)
    
    return df

def train_model(X_train, y_train, X_test, y_test):
    """Train and evaluate the linear regression model"""
    print("\nTraining model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return model, y_pred, rmse, r2

def analyze_feature_importance(model, feature_names, viz_dir):
    """Analyze and visualize feature importance"""
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(model.coef_)
    })
    
    # Rename product group features
    for col in feature_importance['Feature']:
        if col.startswith('Warengruppe_'):
            try:
                group_nr = int(col.split('_')[1])
                new_name = f"Warengruppe_{get_warengruppe_name(group_nr)}"
                feature_importance.loc[feature_importance['Feature'] == col, 'Feature'] = new_name
            except ValueError:
                continue
    
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance.head(10), x='Importance', y='Feature')
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return feature_importance

def visualize_predictions(y_test, y_pred, viz_dir):
    """Create prediction vs actual visualization"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    plt.title('Prediction vs Actual Values')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'prediction_vs_actual.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run the linear regression analysis"""
    # Create output directories
    viz_dir = os.path.join(VIZ_DIR, 'linear_regression')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(TRAIN_PATH)
    
    # Prepare features
    df = prepare_features(df)
    
    # Select features for the model
    feature_columns = ['Jahr', 'Monat', 'Wochentag'] + [col for col in df.columns if col.startswith('Warengruppe_')]
    X = df[feature_columns]
    y = df['Umsatz']
    
    # Split data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model, y_pred, rmse, r2 = train_model(X_train, y_train, X_test, y_test)
    
    # Print model performance
    print("\nModel Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(model, feature_columns, viz_dir)
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Create prediction visualization
    visualize_predictions(y_test, y_pred, viz_dir)
    
    print("\nVisualizations saved to:")
    print(f"1. {os.path.join(viz_dir, 'feature_importance.png')} - Feature Importance")
    print(f"2. {os.path.join(viz_dir, 'prediction_vs_actual.png')} - Prediction vs Actual Values")

if __name__ == "__main__":
    main()
