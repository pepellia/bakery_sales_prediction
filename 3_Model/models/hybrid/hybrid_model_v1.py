import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
from datetime import datetime
from typing import Optional, Tuple, Union, List, Dict, Any
from numpy.typing import NDArray, ArrayLike

# Add the parent directory to Python path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))  # Add parent of project root

from bakery_sales_prediction.config import (
    TRAIN_PATH, TEST_PATH, WEATHER_PATH, KIWO_PATH
)

# Define output directories relative to project root
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "3_Model", "output")
VISUALIZATION_DIR = os.path.join(PROJECT_ROOT, "3_Model", "visualization")

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

def plot_feature_importance(feature_importance: pd.DataFrame, output_dir: str) -> None:
    """Plot feature importance."""
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=feature_importance.head(15),
        x='importance',
        y='feature'
    )
    plt.title('Top 15 Most Important Features')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()

def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, set_name: str, output_dir: str) -> None:
    """Plot predicted vs actual values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title(f'Predicted vs Actual Sales ({set_name} Set)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'predictions_vs_actual_{set_name}.png'))
    plt.close()

def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray, set_name: str, output_dir: str) -> None:
    """Plot error distribution."""
    errors = y_pred - y_true
    plt.figure(figsize=(10, 6))
    sns.histplot(data=errors, binwidth=5, kde=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title(f'Error Distribution ({set_name} Set)')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'error_distribution_{set_name}.png'))
    plt.close()

def plot_sales_by_group(train_df: pd.DataFrame, output_dir: str) -> None:
    """Plot average sales by product group."""
    avg_sales = train_df.groupby('Warengruppe')['Umsatz'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    avg_sales.plot(kind='bar')
    plt.title('Average Sales by Product Group')
    plt.xlabel('Product Group')
    plt.ylabel('Average Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sales_by_group.png'))
    plt.close()

def plot_temporal_patterns(train_df: pd.DataFrame, output_dir: str) -> None:
    """Plot temporal patterns in sales."""
    # Daily pattern
    daily_sales = train_df.groupby('weekday')['Umsatz'].mean()
    plt.figure(figsize=(10, 6))
    daily_sales.plot(kind='line', marker='o')
    plt.title('Average Sales by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Sales')
    plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sales_by_weekday.png'))
    plt.close()
    
    # Monthly pattern
    monthly_sales = train_df.groupby('month')['Umsatz'].mean()
    plt.figure(figsize=(10, 6))
    monthly_sales.plot(kind='line', marker='o')
    plt.title('Average Sales by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Sales')
    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sales_by_month.png'))
    plt.close()

def apply_fourier_transform(data: np.ndarray, n_harmonics: int = 5) -> np.ndarray:
    """Apply Fourier transform and extract dominant frequencies."""
    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, d=1)[:N//2]
    
    # Get dominant frequencies
    yf_half = np.array(yf[:N//2])  # Convert to numpy array explicitly
    amplitudes = np.abs(yf_half)  # Calculate absolute values
    indices = np.argsort(amplitudes)[::-1][:n_harmonics]
    
    # Extract real and imaginary components
    fourier_features = np.zeros((N, n_harmonics * 2))
    for i, idx in enumerate(indices):
        fourier_features[:, i*2] = np.real(yf[idx]) * np.cos(2 * np.pi * xf[idx] * np.arange(N))
        fourier_features[:, i*2+1] = np.imag(yf[idx]) * np.sin(2 * np.pi * xf[idx] * np.arange(N))
    
    return fourier_features

def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare all data sources."""
    # Load all data sources
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    weather_df = pd.read_csv(WEATHER_PATH)
    kiwo_df = pd.read_csv(KIWO_PATH)
    
    # Convert dates
    for df in [train_df, test_df, weather_df, kiwo_df]:
        df['Datum'] = pd.to_datetime(df['Datum'])
    
    # Merge all data
    train_df = pd.merge(train_df, weather_df, on='Datum', how='left')
    train_df = pd.merge(train_df, kiwo_df, on='Datum', how='left')
    
    test_df = pd.merge(test_df, weather_df, on='Datum', how='left')
    test_df = pd.merge(test_df, kiwo_df, on='Datum', how='left')
    
    # Fill missing values
    for df in [train_df, test_df]:
        df['Bewoelkung'] = df['Bewoelkung'].fillna(df['Bewoelkung'].mean())
        df['Temperatur'] = df['Temperatur'].fillna(df['Temperatur'].mean())
        df['Windgeschwindigkeit'] = df['Windgeschwindigkeit'].fillna(df['Windgeschwindigkeit'].mean())
        df['Wettercode'] = df['Wettercode'].fillna(-1)
        df['KielerWoche'] = df['KielerWoche'].fillna(0)
    
    # Create temporal features
    for df in [train_df, test_df]:
        df['weekday'] = df['Datum'].dt.dayofweek
        df['month'] = df['Datum'].dt.month
        df['day_of_month'] = df['Datum'].dt.day
        df['week_of_year'] = df['Datum'].dt.isocalendar().week.astype(int)
        df['quarter'] = df['Datum'].dt.quarter
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
        
        # Add special event features
        df['is_silvester'] = ((df['month'] == 12) & (df['day_of_month'] == 31)).astype(int)
        
        # Create seasonal temperature categories
        def get_season(month: int) -> str:
            if month in [12, 1, 2]: return 'winter'
            elif month in [3, 4, 5]: return 'spring'
            elif month in [6, 7, 8]: return 'summer'
            else: return 'fall'
        
        df['season'] = df['month'].map(get_season)
        
        # Create temperature categories based on season
        temp_categories = {
            'winter': {'cold': (-np.inf, 5), 'mild': (5, 12), 'warm': (12, np.inf)},
            'spring': {'cold': (-np.inf, 10), 'mild': (10, 18), 'warm': (18, np.inf)},
            'summer': {'cold': (-np.inf, 15), 'mild': (15, 23), 'warm': (23, np.inf)},
            'fall': {'cold': (-np.inf, 8), 'mild': (8, 15), 'warm': (15, np.inf)}
        }
        
        def get_temp_category(row: pd.Series) -> str:
            categories = temp_categories[row['season']]
            temp = row['Temperatur']
            for category, (min_temp, max_temp) in categories.items():
                if min_temp <= temp < max_temp:
                    return category
            return 'mild'  # fallback
        
        df['temp_category'] = df.apply(get_temp_category, axis=1)
    
    return train_df, test_df

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and validation sets."""
    val_mask = (df['Datum'] >= '2017-02-01') & (df['Datum'] <= '2017-07-31')
    
    train_df = df[~val_mask].copy()
    val_df = df[val_mask].copy()
    
    return train_df, val_df

def prepare_features(df: pd.DataFrame, reference_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Prepare features including Fourier components."""
    # Group by product category and calculate Fourier features
    fourier_features = []
    for group in sorted(df['Warengruppe'].unique()):
        mask = df['Warengruppe'] == group
        if 'Umsatz' in df.columns:
            group_data = df[mask]['Umsatz'].values
        else:
            group_data = np.zeros(len(df[mask]))
        group_fourier = apply_fourier_transform(group_data, n_harmonics=5)
        fourier_features.append(group_fourier)
    
    fourier_features = np.concatenate(fourier_features)
    
    # Create feature columns for Fourier components
    fourier_cols = [f'fourier_{i}' for i in range(fourier_features.shape[1])]
    fourier_df = pd.DataFrame(fourier_features, columns=fourier_cols, index=df.index)
    
    # Create dummy variables for categorical features
    categorical_features = ['weekday', 'temp_category', 'Warengruppe']
    df_encoded = pd.get_dummies(df, columns=categorical_features)
    
    # If reference_df is provided, ensure consistent columns
    if reference_df is not None:
        reference_encoded = pd.get_dummies(reference_df, columns=categorical_features)
        missing_cols = set(reference_encoded.columns) - set(df_encoded.columns)
        for col in missing_cols:
            df_encoded[col] = 0
        df_encoded = df_encoded[reference_encoded.columns]
    
    # Combine with other features
    feature_cols = [
        'month', 'day_of_month', 'week_of_year', 'quarter', 'is_weekend',
        'Bewoelkung', 'Temperatur', 'Windgeschwindigkeit', 'Wettercode',
        'KielerWoche', 'is_silvester'
    ]
    
    # Add dummy variable columns
    dummy_cols = [col for col in df_encoded.columns if col.startswith(('weekday_', 'temp_category_', 'Warengruppe_'))]
    feature_cols.extend(dummy_cols)
    
    # Combine all features
    X = pd.concat([df_encoded[feature_cols], fourier_df], axis=1)
    y = df['Umsatz'].values if 'Umsatz' in df.columns else None
    
    if y is not None:
        y = y.astype(np.float64)
    
    return X, y

def train_and_evaluate() -> None:
    """Train the model and evaluate its performance."""
    # Load and prepare data
    train_df, test_df = load_and_prepare_data()
    train_df, val_df = split_data(train_df)
    
    # Create visualization directory
    viz_dir = os.path.join(VISUALIZATION_DIR, "hybrid")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Plot data patterns
    plot_sales_by_group(train_df, viz_dir)
    plot_temporal_patterns(train_df, viz_dir)
    
    # Prepare features for each set
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df, reference_df=train_df)
    X_test, _ = prepare_features(test_df, reference_df=train_df)
    
    if y_train is None or y_val is None:
        raise ValueError("Training or validation targets are missing")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_split=15,
        min_samples_leaf=5,
        max_features='sqrt',
        bootstrap=True,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    test_pred = model.predict(X_test_scaled)
    
    # Calculate MAE for training and validation
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Create visualization plots
    plot_feature_importance(feature_importance, viz_dir)
    plot_predictions_vs_actual(y_train, train_pred, 'train', viz_dir)
    plot_predictions_vs_actual(y_val, val_pred, 'validation', viz_dir)
    plot_error_distribution(y_train, train_pred, 'train', viz_dir)
    plot_error_distribution(y_val, val_pred, 'validation', viz_dir)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "model_name": "hybrid_model_v1",
        "timestamp": timestamp,
        "metrics": {
            "train_mae": float(train_mae),
            "val_mae": float(val_mae)
        },
        "parameters": {
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_split": 15,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "bootstrap": True
        },
        "feature_importance": feature_importance.head(20).to_dict('records')
    }
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(MODEL_OUTPUT_DIR, "hybrid")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    results_file = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save predictions
    test_df['Umsatz'] = test_pred
    submission = test_df[['id', 'Umsatz']]
    submission_file = os.path.join(output_dir, f"submission_{timestamp}.csv")
    submission.to_csv(submission_file, index=False)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Submission saved to: {submission_file}")
    print(f"Visualizations saved to: {viz_dir}")
    print("\nModel Performance:")
    print(f"Training MAE: {train_mae:.2f}")
    print(f"Validation MAE: {val_mae:.2f}")
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string())

if __name__ == "__main__":
    train_and_evaluate() 