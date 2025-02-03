import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple, Union, Any
from numpy.typing import NDArray
from pandas import Series, DataFrame

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Import configuration
from bakery_sales_prediction.config import (TRAIN_PATH, TEST_PATH, 
                                          SAMPLE_SUBMISSION_PATH,
                                          WARENGRUPPEN, get_warengruppe_name)

# Create output directories
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output', SCRIPT_NAME)
VIZ_DIR = os.path.join(SCRIPT_DIR, 'visualizations', SCRIPT_NAME)
for directory in [OUTPUT_DIR, VIZ_DIR]:
    os.makedirs(directory, exist_ok=True)

# Set seaborn style
sns.set_theme()

def calculate_metrics(actual: Union[Series, NDArray[np.float64]], 
                     predicted: Union[Series, NDArray[np.float64]]) -> Tuple[float, float, float]:
    """Calculate RMSE, MAE, and R² metrics.
    
    Args:
        actual: Series or array of actual values
        predicted: Series or array of predicted values
        
    Returns:
        Tuple of (RMSE, MAE, R²)
    """
    # Convert to numpy arrays if needed
    actual_arr = actual.to_numpy() if isinstance(actual, Series) else np.asarray(actual, dtype=np.float64)
    predicted_arr = predicted.to_numpy() if isinstance(predicted, Series) else np.asarray(predicted, dtype=np.float64)
    
    mse = float(np.mean((actual_arr - predicted_arr) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(actual_arr - predicted_arr)))
    r2 = float(1 - mse / np.var(actual_arr))
    return rmse, mae, r2

def create_time_series_plot(data: DataFrame, title: str, output_filename: str) -> None:
    """Create and save a time series plot comparing actual vs predicted values."""
    plt.figure(figsize=(15, 6))
    plt.plot(data['Datum'], data['Umsatz'], label='Actual', alpha=0.5)
    plt.plot(data['Datum'], data['Vorhersage'], label='Predicted', alpha=0.5)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Sales (€)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, output_filename))
    plt.close()

def main():
    """Main function to run the simple weekday model for product group 1"""
    print("Starting simple weekday model for product group 1...")
    
    # Read and prepare data
    print("\nLoading data...")
    df = pd.read_csv(TRAIN_PATH)
    df = DataFrame(df)  # Ensure it's a DataFrame
    df['Datum'] = pd.to_datetime(df['Datum'])
    
    # Filter for Warengruppe 1 only
    df_filtered = DataFrame(df[df['Warengruppe'] == 1].copy())
    
    # Extract weekday (0 = Monday, 6 = Sunday)
    weekdays = Series([d.weekday() for d in df_filtered['Datum']])
    df_filtered = DataFrame(df_filtered.assign(Wochentag=weekdays))
    
    # Add product group names
    warengruppe_names = Series([get_warengruppe_name(w) for w in df_filtered['Warengruppe']])
    df_filtered = DataFrame(df_filtered.assign(Warengruppe_Name=warengruppe_names))
    
    # Split data based on date
    train_mask = (df_filtered['Datum'] >= '2013-07-01') & (df_filtered['Datum'] <= '2017-07-31')
    val_mask = (df_filtered['Datum'] >= '2017-08-01') & (df_filtered['Datum'] <= '2018-07-31')
    
    train_data = DataFrame(df_filtered[train_mask].copy())
    val_data = DataFrame(df_filtered[val_mask].copy())
    
    # Prepare features using one-hot encoding for weekday
    encoder = OneHotEncoder(sparse_output=False)
    
    # Fit encoder on training data (weekday only)
    X_train_encoded = encoder.fit_transform(train_data[['Wochentag']])
    X_val_encoded = encoder.transform(val_data[['Wochentag']])
    
    # Train the model
    print("\nTraining model...")
    model = LinearRegression()
    model.fit(X_train_encoded, train_data['Umsatz'])
    
    # Make predictions for both train and validation sets
    train_predictions = Series(model.predict(X_train_encoded), index=train_data.index)
    val_predictions = Series(model.predict(X_val_encoded), index=val_data.index)
    
    train_data = DataFrame(train_data.assign(Vorhersage=train_predictions))
    val_data = DataFrame(val_data.assign(Vorhersage=val_predictions))
    
    # Calculate metrics
    train_rmse, train_mae, train_r2 = calculate_metrics(
        Series(train_data['Umsatz']).astype(np.float64),
        Series(train_data['Vorhersage']).astype(np.float64)
    )
    val_rmse, val_mae, val_r2 = calculate_metrics(
        Series(val_data['Umsatz']).astype(np.float64),
        Series(val_data['Vorhersage']).astype(np.float64)
    )
    
    # Print metrics
    print(f"\nModel Performance for Product Group 1 - Training Set (2013-07-01 to 2017-07-31):")
    print(f"RMSE: {train_rmse:.2f}€")
    print(f"MAE: {train_mae:.2f}€")
    print(f"R²: {train_r2:.4f}")
    
    print(f"\nModel Performance for Product Group 1 - Validation Set (2017-08-01 to 2018-07-31):")
    print(f"RMSE: {val_rmse:.2f}€")
    print(f"MAE: {val_mae:.2f}€")
    print(f"R²: {val_r2:.4f}")
    
    # Save metrics
    metrics = {
        'training': {
            'RMSE': train_rmse,
            'MAE': train_mae,
            'R²': train_r2
        },
        'validation': {
            'RMSE': val_rmse,
            'MAE': val_mae,
            'R²': val_r2
        }
    }
    pd.DataFrame(metrics).to_json(os.path.join(OUTPUT_DIR, 'metrics.json'))
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Time series plots
    for period, data in [('train', train_data), ('val', val_data)]:
        title = f'Product Group 1 - {"Training" if period == "train" else "Validation"} Period Sales'
        output_filename = f'group1_{period}_sales.png'
        create_time_series_plot(data, title, output_filename)
    
    # Weekly pattern visualization
    plt.figure(figsize=(10, 6))
    weekday_means = pd.DataFrame(train_data).groupby('Wochentag')['Umsatz'].mean()
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    plt.bar(weekday_names, weekday_means)
    plt.title('Average Daily Sales for Product Group 1')
    plt.xlabel('Weekday')
    plt.ylabel('Average Sales (€)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'group1_weekday_pattern.png'))
    plt.close()
    
    # Generate predictions for test set
    print("\nGenerating predictions for test set...")
    test_df = pd.read_csv(TEST_PATH)
    test_df = DataFrame(test_df)
    test_df['Datum'] = pd.to_datetime(test_df['id'].astype(str).str[:6], format='%y%m%d')
    test_df_filtered = DataFrame(test_df[test_df['Warengruppe'] == 1].copy())
    
    weekdays = Series([d.weekday() for d in test_df_filtered['Datum']])
    test_df_filtered = DataFrame(test_df_filtered.assign(Wochentag=weekdays))
    
    X_test_encoded = encoder.transform(test_df_filtered[['Wochentag']])
    test_predictions = model.predict(X_test_encoded)
    
    # Create submission file
    submission = pd.DataFrame({
        'id': test_df_filtered['id'],
        'Umsatz': test_predictions
    })
    
    submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission file saved to: {submission_path}")

if __name__ == "__main__":
    main()
