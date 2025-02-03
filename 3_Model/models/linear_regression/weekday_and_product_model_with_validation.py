"""
Linear Regression Model with Weekday and Product Features (with validation)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # 3_Model directory

# Data paths
DATA_DIR = os.path.join(MODEL_ROOT, '..', '0_DataPreparation', 'input', 'competition_data')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'sample_submission.csv')
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_DIR, 'sample_submission.csv')

# Product group configurations
WARENGRUPPEN = {
    1: 'Brot',
    2: 'Brötchen',
    3: 'Croissant',
    4: 'Konditorei',
    5: 'Kuchen',
    6: 'Saisonbrot'
}

def get_warengruppe_name(code):
    """Get the name of a product group by its code"""
    return WARENGRUPPEN.get(code, f'Unknown ({code})')

# Create output directories
MODEL_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_DIR = os.path.join(MODEL_ROOT, 'output', 'linear_regression', MODEL_NAME)
VIZ_DIR = os.path.join(MODEL_ROOT, 'visualizations', 'linear_regression', MODEL_NAME)

for directory in [OUTPUT_DIR, VIZ_DIR]:
    os.makedirs(directory, exist_ok=True)

def load_and_prepare_data():
    """Load and prepare the data with enhanced date features"""
    # Load training data
    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_PATH)
    train_df['Datum'] = pd.to_datetime(train_df['Datum'])
    
    # Create train/validation split based on dates
    train_mask = (train_df['Datum'] >= '2013-07-01') & (train_df['Datum'] <= '2017-07-31')
    val_mask = (train_df['Datum'] >= '2017-08-01') & (train_df['Datum'] <= '2018-07-31')
    
    train_data = train_df[train_mask].copy()
    val_data = train_df[val_mask].copy()
    
    # Load test data for final predictions
    print("Loading test data...")
    test_df = pd.read_csv(TEST_PATH)
    test_df['Datum'] = pd.to_datetime('20' + test_df['id'].astype(str).str[:6], format='%Y%m%d')
    test_df['Warengruppe'] = test_df['id'].astype(str).str[-1].astype(int)
    
    # Add features to all datasets
    for df in [train_data, val_data, test_df]:
        # Basic date components
        df['Wochentag'] = df['Datum'].dt.dayofweek
        df['is_weekend'] = df['Datum'].dt.dayofweek.isin([5, 6]).astype(int)
        df['week_of_year'] = df['Datum'].dt.isocalendar().week
        df['month'] = df['Datum'].dt.month
        df['day_of_month'] = df['Datum'].dt.day
        
        # Month position (start, middle, end)
        df['month_position'] = pd.cut(df['day_of_month'], 
                                    bins=[0, 10, 20, 31], 
                                    labels=['start', 'middle', 'end'])
        
        # Seasonal features
        df['season'] = pd.cut(df['month'], 
                            bins=[0, 3, 6, 9, 12], 
                            labels=['winter', 'spring', 'summer', 'fall'])
        
        # Cyclical encoding of week and month
        df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Add product group names
        df['Warengruppe_Name'] = df['Warengruppe'].map(get_warengruppe_name)
    
    return train_data, val_data, test_df

def create_feature_pipeline():
    """Create a pipeline for feature preprocessing"""
    # Define the preprocessing steps for different types of features
    numeric_features = ['Wochentag', 'week_sin', 'week_cos', 
                       'month_sin', 'month_cos', 'day_of_month']
    categorical_features = ['Warengruppe', 'month_position', 'season']
    binary_features = ['is_weekend']
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('bin', 'passthrough', binary_features)
        ])
    
    # Create pipeline with preprocessor and model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    return pipeline

def evaluate_model(y_true, y_pred, dataset_name):
    """Calculate and print evaluation metrics"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - mse / np.var(y_true)
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }
    
    print(f"\nMetrics for {dataset_name}:")
    print(f"RMSE: {rmse:.2f}€")
    print(f"MAE: {mae:.2f}€")
    print(f"R²: {r2:.3f}")
    
    return metrics

def plot_predictions(train_df, train_pred, val_df=None, val_pred=None):
    """Create visualizations of the model's predictions"""
    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    
    # Training data
    plt.scatter(train_df['Umsatz'], train_pred, alpha=0.5, label='Training')
    
    # Validation data if available
    if val_df is not None and val_pred is not None:
        plt.scatter(val_df['Umsatz'], val_pred, alpha=0.5, label='Validation')
    
    max_val = max(train_df['Umsatz'].max(), train_pred.max())
    if val_df is not None and val_pred is not None:
        max_val = max(max_val, val_df['Umsatz'].max(), val_pred.max())
    
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Sales (€)')
    plt.ylabel('Predicted Sales (€)')
    plt.title('Actual vs Predicted Sales')
    plt.legend()
    
    # Plot 2: Prediction Errors by Product Group
    plt.subplot(1, 2, 2)
    error_data = []
    
    # Training errors
    train_errors = train_pred - train_df['Umsatz']
    train_error_df = pd.DataFrame({
        'Product': train_df['Warengruppe_Name'],
        'Error': train_errors,
        'Set': 'Training'
    })
    error_data.append(train_error_df)
    
    # Validation errors if available
    if val_df is not None and val_pred is not None:
        val_errors = val_pred - val_df['Umsatz']
        val_error_df = pd.DataFrame({
            'Product': val_df['Warengruppe_Name'],
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

def plot_weekday_patterns(train_df, train_pred, test_df, test_pred, val_df=None, val_pred=None):
    """Create plots showing weekly patterns for each product group"""
    # Add predictions to data
    train_data = train_df.copy()
    train_data['Predicted_Sales'] = train_pred
    test_data = test_df.copy()
    test_data['Predicted_Sales'] = test_pred
    
    if val_df is not None and val_pred is not None:
        val_data = val_df.copy()
        val_data['Predicted_Sales'] = val_pred
    
    # Create a figure with subplots for each product group
    n_products = len(WARENGRUPPEN)
    fig, axes = plt.subplots(n_products, 1, figsize=(12, 4*n_products))
    
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                     'Friday', 'Saturday', 'Sunday']
    
    for (product_group, product_name), ax in zip(WARENGRUPPEN.items(), axes):
        # Get data for this product
        train_product = train_data[train_data['Warengruppe'] == product_group]
        test_product = test_data[test_data['Warengruppe'] == product_group]
        
        # Calculate average sales by weekday
        train_weekday_avg = train_product.groupby('Wochentag').agg({
            'Umsatz': 'mean',
            'Predicted_Sales': 'mean'
        })
        test_weekday_avg = test_product.groupby('Wochentag')['Predicted_Sales'].mean()
        
        # Plot weekday patterns
        ax.plot(weekday_names, train_weekday_avg['Umsatz'], 
                marker='o', label='Actual (Train)', color='blue')
        ax.plot(weekday_names, train_weekday_avg['Predicted_Sales'], 
                marker='o', label='Predicted (Train)', color='red', linestyle='--')
        
        # Add validation data if available
        if val_df is not None and val_pred is not None:
            val_product = val_data[val_data['Warengruppe'] == product_group]
            val_weekday_avg = val_product.groupby('Wochentag').agg({
                'Umsatz': 'mean',
                'Predicted_Sales': 'mean'
            })
            ax.plot(weekday_names, val_weekday_avg['Umsatz'],
                   marker='o', label='Actual (Val)', color='green')
            ax.plot(weekday_names, val_weekday_avg['Predicted_Sales'],
                   marker='o', label='Predicted (Val)', color='orange', linestyle='--')
        
        ax.plot(weekday_names, test_weekday_avg, 
                marker='o', label='Predicted (Test)', color='purple', linestyle='--')
        
        ax.set_title(f'{product_name} - Weekly Sales Pattern')
        ax.set_xlabel('Weekday')
        ax.set_ylabel('Average Sales (€)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'weekly_patterns.png'))
    plt.close()

def plot_seasonal_patterns(train_df, train_pred, val_df=None, val_pred=None):
    """Create visualizations of seasonal patterns"""
    # Add predictions to data
    train_data = train_df.copy()
    train_data['Predicted_Sales'] = train_pred
    
    if val_df is not None and val_pred is not None:
        val_data = val_df.copy()
        val_data['Predicted_Sales'] = val_pred
    
    # Plot 1: Sales by Season
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    
    # Training data
    seasonal_data = train_data.groupby(['season', 'Warengruppe_Name']).agg({
        'Umsatz': 'mean',
        'Predicted_Sales': 'mean'
    }).reset_index()
    
    # Validation data if available
    if val_df is not None and val_pred is not None:
        val_seasonal = val_data.groupby(['season', 'Warengruppe_Name']).agg({
            'Umsatz': 'mean',
            'Predicted_Sales': 'mean'
        }).reset_index()
    
    for product in train_data['Warengruppe_Name'].unique():
        # Training data
        product_data = seasonal_data[seasonal_data['Warengruppe_Name'] == product]
        plt.plot(product_data['season'], product_data['Umsatz'], 
                marker='o', label=f'{product} (Train Actual)')
        plt.plot(product_data['season'], product_data['Predicted_Sales'], 
                marker='o', linestyle='--', label=f'{product} (Train Pred)')
        
        # Validation data if available
        if val_df is not None and val_pred is not None:
            val_product = val_seasonal[val_seasonal['Warengruppe_Name'] == product]
            plt.plot(val_product['season'], val_product['Umsatz'],
                    marker='s', label=f'{product} (Val Actual)')
            plt.plot(val_product['season'], val_product['Predicted_Sales'],
                    marker='s', linestyle='--', label=f'{product} (Val Pred)')
    
    plt.title('Seasonal Sales Patterns by Product')
    plt.xlabel('Season')
    plt.ylabel('Average Sales (€)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Month Position Effect
    plt.subplot(1, 2, 2)
    
    # Training data
    position_data = train_data.groupby(['month_position', 'Warengruppe_Name']).agg({
        'Umsatz': 'mean',
        'Predicted_Sales': 'mean'
    }).reset_index()
    
    # Validation data if available
    if val_df is not None and val_pred is not None:
        val_position = val_data.groupby(['month_position', 'Warengruppe_Name']).agg({
            'Umsatz': 'mean',
            'Predicted_Sales': 'mean'
        }).reset_index()
    
    for product in train_data['Warengruppe_Name'].unique():
        # Training data
        product_data = position_data[position_data['Warengruppe_Name'] == product]
        plt.plot(product_data['month_position'], product_data['Umsatz'], 
                marker='o', label=f'{product} (Train Actual)')
        plt.plot(product_data['month_position'], product_data['Predicted_Sales'], 
                marker='o', linestyle='--', label=f'{product} (Train Pred)')
        
        # Validation data if available
        if val_df is not None and val_pred is not None:
            val_product = val_position[val_position['Warengruppe_Name'] == product]
            plt.plot(val_product['month_position'], val_product['Umsatz'],
                    marker='s', label=f'{product} (Val Actual)')
            plt.plot(val_product['month_position'], val_product['Predicted_Sales'],
                    marker='s', linestyle='--', label=f'{product} (Val Pred)')
    
    plt.title('Sales Patterns by Month Position')
    plt.xlabel('Month Position')
    plt.ylabel('Average Sales (€)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'seasonal_patterns.png'))
    plt.close()

def generate_submission_predictions(model, feature_columns, train_df):
    """Generate predictions for submission"""
    # Load submission template
    submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    
    # Extract dates and product groups from IDs
    dates = pd.to_datetime('20' + submission_df['id'].astype(str).str[:6])
    product_groups = submission_df['id'].astype(str).str[-1].astype(int)
    
    # Create features for prediction
    pred_data = pd.DataFrame({
        'Datum': dates,
        'Warengruppe': product_groups
    })
    
    # Add date features
    pred_data['Wochentag'] = pred_data['Datum'].dt.dayofweek
    pred_data['is_weekend'] = pred_data['Datum'].dt.dayofweek.isin([5, 6]).astype(int)
    pred_data['week_of_year'] = pred_data['Datum'].dt.isocalendar().week
    pred_data['month'] = pred_data['Datum'].dt.month
    pred_data['day_of_month'] = pred_data['Datum'].dt.day
    
    # Month position
    pred_data['month_position'] = pd.cut(pred_data['day_of_month'], 
                                       bins=[0, 10, 20, 31], 
                                       labels=['start', 'middle', 'end'])
    
    # Seasonal features
    pred_data['season'] = pd.cut(pred_data['month'], 
                                bins=[0, 3, 6, 9, 12], 
                                labels=['winter', 'spring', 'summer', 'fall'])
    
    # Cyclical features
    pred_data['week_sin'] = np.sin(2 * np.pi * pred_data['week_of_year'] / 52)
    pred_data['week_cos'] = np.cos(2 * np.pi * pred_data['week_of_year'] / 52)
    pred_data['month_sin'] = np.sin(2 * np.pi * pred_data['month'] / 12)
    pred_data['month_cos'] = np.cos(2 * np.pi * pred_data['month'] / 12)
    
    # Generate predictions
    predictions = model.predict(pred_data[feature_columns])
    
    # Add predictions to submission dataframe
    submission_df['Umsatz'] = predictions
    
    # Save submission file
    submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission file has been saved to: {submission_path}")
    print(f"Number of predictions: {len(submission_df)}")
    
    return submission_df

def main():
    """Main function to run the combined model with validation"""
    print("Starting combined model training with validation...")
    
    # Load and prepare data
    train_df, val_df, test_df = load_and_prepare_data()
    
    # Create and train model
    print("\nTraining model...")
    feature_columns = ['Wochentag', 'Warengruppe', 'is_weekend', 
                      'week_sin', 'week_cos', 'month_sin', 'month_cos',
                      'day_of_month', 'month_position', 'season']
    target_column = 'Umsatz'
    
    model = create_feature_pipeline()
    model.fit(train_df[feature_columns], train_df[target_column])
    
    # Make predictions
    train_pred = model.predict(train_df[feature_columns])
    val_pred = model.predict(val_df[feature_columns])
    test_pred = model.predict(test_df[feature_columns])
    
    # Evaluate model
    print("\nModel Evaluation:")
    train_metrics = evaluate_model(train_df[target_column], train_pred, "Training Data")
    val_metrics = evaluate_model(val_df[target_column], val_pred, "Validation Data")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_predictions(train_df, train_pred, val_df, val_pred)
    plot_weekday_patterns(train_df, train_pred, test_df, test_pred, val_df, val_pred)
    plot_seasonal_patterns(train_df, train_pred, val_df, val_pred)
    
    # Generate submission file
    print("\nGenerating submission file...")
    submission_df = generate_submission_predictions(model, feature_columns, train_df)
    
    # Save metrics
    metrics = {
        'training': train_metrics,
        'validation': val_metrics
    }
    pd.DataFrame(metrics).to_json(os.path.join(OUTPUT_DIR, 'metrics.json'))
    
    print(f"\nResults have been saved to: {OUTPUT_DIR}")
    print(f"Visualizations have been saved to: {VIZ_DIR}")

if __name__ == "__main__":
    main() 