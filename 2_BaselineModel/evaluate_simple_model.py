import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

class SimpleModel:
    def __init__(self):
        # Coefficients from the simple model equation
        self.intercept = 85.08
        self.coefficients = {
            'is_weekend': 23.87,
            'season_summer': 69.39,
            'warengruppe_2': 306.89,  # Brötchen
            'warengruppe_3': 60.44,   # Croissant
            'warengruppe_5': 177.19   # Kuchen
        }
    
    def predict(self, X):
        """Make predictions based on the simple model equation"""
        predictions = np.full(len(X), self.intercept)
        
        # Map old feature names to new ones
        feature_mapping = {
            'is_weekend': X['is_weekend'],
            'season_summer': (X['season'] == 'summer').astype(int),
            'warengruppe_2': (X['Warengruppe'] == 2).astype(int),
            'warengruppe_3': (X['Warengruppe'] == 3).astype(int),
            'warengruppe_5': (X['Warengruppe'] == 5).astype(int)
        }
        
        for feature, coef in self.coefficients.items():
            predictions += coef * feature_mapping[feature]
        
        return predictions

def prepare_data(df):
    """Prepare features for the simple model"""
    df = df.copy()
    
    # Create weekend feature
    df['is_weekend'] = df['Datum'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Create season feature
    df['month'] = df['Datum'].dt.month
    df['season'] = pd.cut(df['month'], 
                         bins=[0, 3, 6, 9, 12], 
                         labels=['winter', 'spring', 'summer', 'fall'])
    
    return df

def evaluate_simple_model():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    train_df['Datum'] = pd.to_datetime(train_df['Datum'])
    
    # Create train/validation split based on dates
    train_mask = (train_df['Datum'] >= '2013-07-01') & (train_df['Datum'] <= '2017-07-31')
    val_mask = (train_df['Datum'] >= '2017-08-01') & (train_df['Datum'] <= '2018-07-31')
    
    train_data = train_df[train_mask].copy()
    val_data = train_df[val_mask].copy()
    
    # Prepare features
    train_data = prepare_data(train_data)
    val_data = prepare_data(val_data)
    
    # Initialize model and make predictions
    model = SimpleModel()
    train_pred = model.predict(train_data)
    val_pred = model.predict(val_data)
    
    # Calculate metrics
    train_metrics = {
        'R²': r2_score(train_data['Umsatz'], train_pred),
        'Adj. R²': 1 - (1 - r2_score(train_data['Umsatz'], train_pred)) * (len(train_data) - 1) / (len(train_data) - len(model.coefficients) - 1),
        'RMSE': np.sqrt(mean_squared_error(train_data['Umsatz'], train_pred)),
        'MAE': mean_absolute_error(train_data['Umsatz'], train_pred)
    }
    
    val_metrics = {
        'R²': r2_score(val_data['Umsatz'], val_pred),
        'RMSE': np.sqrt(mean_squared_error(val_data['Umsatz'], val_pred)),
        'MAE': mean_absolute_error(val_data['Umsatz'], val_pred)
    }
    
    # Print results
    print("\nTraining Data (2013-07-01 to 2017-07-31):")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nValidation Data (2017-08-01 to 2018-07-31):")
    for metric, value in val_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Analysis by product group
    print("\nAnalysis by Product Group (Validation Data):")
    
    # Make predictions by product group
    val_predictions = {}
    product_metrics = {}
    
    for group, name in WARENGRUPPEN.items():
        group_data = val_data[val_data['Warengruppe'] == group].copy()
        if group in [2, 3, 5]:  # Only for groups in the model
            group_pred = model.predict(group_data)
            val_predictions[group] = group_pred
            
            metrics = {
                'RMSE': np.sqrt(mean_squared_error(group_data['Umsatz'], group_pred)),
                'MAE': mean_absolute_error(group_data['Umsatz'], group_pred),
                'R²': r2_score(group_data['Umsatz'], group_pred)
            }
            product_metrics[group] = metrics
            
            print(f"\n{name} (Group {group}):")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        else:
            print(f"\n{name} (Group {group}):")
            print("Not included in model")
    
    # Save metrics
    metrics = {
        'training': train_metrics,
        'validation': val_metrics,
        'products': product_metrics
    }
    pd.DataFrame(metrics).to_json(os.path.join(OUTPUT_DIR, 'metrics.json'))
    
    # Visualizations
    # 1. Actual vs Predicted Plot
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(train_data['Umsatz'], train_pred, alpha=0.5, label='Training')
    plt.scatter(val_data['Umsatz'], val_pred, alpha=0.5, label='Validation')
    max_val = max(train_data['Umsatz'].max(), val_data['Umsatz'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Sales (€)')
    plt.ylabel('Predicted Sales (€)')
    plt.title('Actual vs Predicted Sales')
    plt.legend()
    
    # 2. Prediction Errors by Product Group
    plt.subplot(1, 2, 2)
    error_data = []
    
    # Training errors
    train_errors = train_pred - train_data['Umsatz']
    train_error_df = pd.DataFrame({
        'Product': train_data['Warengruppe'].map(get_warengruppe_name),
        'Error': train_errors,
        'Set': 'Training'
    })
    error_data.append(train_error_df)
    
    # Validation errors
    val_errors = val_pred - val_data['Umsatz']
    val_error_df = pd.DataFrame({
        'Product': val_data['Warengruppe'].map(get_warengruppe_name),
        'Error': val_errors,
        'Set': 'Validation'
    })
    error_data.append(val_error_df)
    
    # Combine error data
    error_df = pd.concat(error_data, ignore_index=True)
    
    # Create box plot
    sns.boxplot(data=error_df, x='Product', y='Error', hue='Set')
    plt.xticks(rotation=45)
    plt.xlabel('Product Group')
    plt.ylabel('Prediction Error (€)')
    plt.title('Prediction Errors by Product Group')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'prediction_analysis.png'))
    plt.close()
    
    # Time series plots for each product group in the model
    for group in [2, 3, 5]:
        name = get_warengruppe_name(group)
        plt.figure(figsize=(15, 6))
        
        # Training data
        train_group = train_data[train_data['Warengruppe'] == group].sort_values('Datum')
        train_group_pred = model.predict(train_group)
        plt.plot(train_group['Datum'], train_group['Umsatz'], 
                label='Actual (Train)', alpha=0.7)
        plt.plot(train_group['Datum'], train_group_pred, 
                label='Predicted (Train)', alpha=0.7, linestyle='--')
        
        # Validation data
        val_group = val_data[val_data['Warengruppe'] == group].sort_values('Datum')
        val_group_pred = model.predict(val_group)
        plt.plot(val_group['Datum'], val_group['Umsatz'], 
                label='Actual (Val)', alpha=0.7)
        plt.plot(val_group['Datum'], val_group_pred, 
                label='Predicted (Val)', alpha=0.7, linestyle='--')
        
        plt.title(f'Time Series Comparison - {name} (Group {group})')
        plt.xlabel('Date')
        plt.ylabel('Sales (€)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, f'timeseries_group_{group}.png'))
        plt.close()

if __name__ == "__main__":
    evaluate_simple_model() 