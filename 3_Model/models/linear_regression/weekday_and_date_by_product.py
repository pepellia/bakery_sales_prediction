"""
Linear regression model with weekday and date features, trained separately for each product group.
Includes train/validation split and enhanced evaluation metrics.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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

# Define timeframes
TRAIN_START = '2013-07-01'
TRAIN_END = '2017-07-31'
VAL_START = '2017-08-01'
VAL_END = '2018-07-31'

def load_and_prepare_data():
    """Load and prepare the data with enhanced date features"""
    # Load training data
    print("Loading training data...")
    train_data = pd.read_csv(TRAIN_PATH)
    train_data['Datum'] = pd.to_datetime(train_data['Datum'])
    
    # Create train/validation split
    train_df = train_data[(train_data['Datum'] >= TRAIN_START) & 
                         (train_data['Datum'] <= TRAIN_END)].copy()
    val_df = train_data[(train_data['Datum'] >= VAL_START) & 
                       (train_data['Datum'] <= VAL_END)].copy()
    
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(TEST_PATH)
    test_df['Datum'] = pd.to_datetime('20' + test_df['id'].astype(str).str[:6], format='%Y%m%d')
    test_df['Warengruppe'] = test_df['id'].astype(str).str[-1].astype(int)
    
    print(f"Data split sizes:")
    print(f"Training set: {len(train_df)} samples ({TRAIN_START} to {TRAIN_END})")
    print(f"Validation set: {len(val_df)} samples ({VAL_START} to {VAL_END})")
    print(f"Test set: {len(test_df)} samples")
    
    # Add date features to all datasets
    for df in [train_df, val_df, test_df]:
        add_date_features(df)
    
    return train_df, val_df, test_df

def add_date_features(df):
    """Add date-based features to the dataframe"""
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
    df['Warengruppe_Name'] = df['Warengruppe'].map(WARENGRUPPEN)

def create_feature_pipeline():
    """Create a pipeline for feature transformation"""
    categorical_features = ['Wochentag', 'month_position', 'season']
    binary_features = ['is_weekend']
    numeric_features = ['week_sin', 'week_cos', 'month_sin', 'month_cos', 
                       'day_of_month']
    
    # Create preprocessing steps
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
    numeric_transformer = StandardScaler()
    
    # Combine all transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('bin', 'passthrough', binary_features),
            ('num', numeric_transformer, numeric_features)
        ])
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    return pipeline

def train_model_for_product(train_df, val_df, test_df, product_group):
    """Train and evaluate model for a specific product group"""
    # Filter data for this product group
    product_train = train_df[train_df['Warengruppe'] == product_group].copy()
    product_val = val_df[val_df['Warengruppe'] == product_group].copy()
    product_test = test_df[test_df['Warengruppe'] == product_group].copy()
    
    # Check if we have enough data
    if len(product_train) == 0 or len(product_val) == 0:
        print(f"\nWarning: Insufficient data for {WARENGRUPPEN[product_group]}")
        return None, None, {
            'train_metrics': None,
            'val_metrics': None
        }
    
    # Prepare features and target
    feature_columns = ['Wochentag', 'is_weekend', 'week_sin', 'week_cos', 
                      'month_sin', 'month_cos', 'day_of_month', 
                      'month_position', 'season']
    target_column = 'Umsatz'
    
    # Create and train model
    model = create_feature_pipeline()
    model.fit(product_train[feature_columns], product_train[target_column])
    
    # Make predictions
    train_pred = model.predict(product_train[feature_columns])
    val_pred = model.predict(product_val[feature_columns])
    test_pred = model.predict(product_test[feature_columns])
    
    # Calculate metrics
    train_metrics = calculate_metrics(product_train[target_column], train_pred)
    val_metrics = calculate_metrics(product_val[target_column], val_pred)
    
    print(f"\nMetrics for {WARENGRUPPEN[product_group]}:")
    print("\nTraining Metrics:")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.3f}")
    print("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        print(f"{metric}: {value:.3f}")
    
    return model, (train_pred, val_pred, test_pred), {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    return {
        'R²': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred)
    }

def plot_predictions(train_df, val_df, test_df, predictions_dict, product_group):
    """Plot actual vs predicted values for a product group"""
    product_name = WARENGRUPPEN[product_group]
    train_pred, val_pred, test_pred = predictions_dict[product_group]
    
    # Prepare data
    train_data = train_df[train_df['Warengruppe'] == product_group].copy()
    val_data = val_df[val_df['Warengruppe'] == product_group].copy()
    test_data = test_df[test_df['Warengruppe'] == product_group].copy()
    
    train_data['Predicted'] = train_pred
    val_data['Predicted'] = val_pred
    test_data['Predicted'] = test_pred
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Time series plot
    ax1.plot(train_data['Datum'], train_data['Umsatz'], 
             label='Actual (Train)', alpha=0.6)
    ax1.plot(train_data['Datum'], train_data['Predicted'], 
             label='Predicted (Train)', alpha=0.6)
    ax1.plot(val_data['Datum'], val_data['Umsatz'],
             label='Actual (Validation)', alpha=0.6)
    ax1.plot(val_data['Datum'], val_data['Predicted'],
             label='Predicted (Validation)', alpha=0.6)
    ax1.plot(test_data['Datum'], test_data['Predicted'], 
             label='Predicted (Test)', alpha=0.6)
    
    ax1.set_title(f'{product_name} - Time Series Comparison')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sales (€)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot (training and validation data)
    ax2.scatter(train_data['Umsatz'], train_data['Predicted'], 
                alpha=0.5, label='Train')
    ax2.scatter(val_data['Umsatz'], val_data['Predicted'],
                alpha=0.5, label='Validation')
    
    # Add perfect prediction line
    all_actual = pd.concat([train_data['Umsatz'], val_data['Umsatz']])
    all_pred = pd.concat([train_data['Predicted'], val_data['Predicted']])
    
    # Convert Series to numpy arrays for type-safe comparison
    actual_values = all_actual.to_numpy()
    pred_values = all_pred.to_numpy()
    
    max_val = float(max(np.max(actual_values), np.max(pred_values)))
    min_val = float(min(np.min(actual_values), np.min(pred_values)))
    
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', 
             label='Perfect Prediction')
    
    ax2.set_title(f'{product_name} - Actual vs Predicted')
    ax2.set_xlabel('Actual Sales (€)')
    ax2.set_ylabel('Predicted Sales (€)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f'predictions_{product_group}.png'))
    plt.close()

def plot_predictions_overview(train_df, train_pred):
    """Create overview visualizations of the model's predictions"""
    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(train_df['Umsatz'], train_pred, alpha=0.5, label='Training')
    max_val = max(train_df['Umsatz'].max(), train_pred.max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Sales (€)')
    plt.ylabel('Predicted Sales (€)')
    plt.title('Actual vs Predicted Sales')
    plt.legend()
    
    # Plot 2: Prediction Errors by Product Group
    plt.subplot(1, 2, 2)
    train_errors = train_pred - train_df['Umsatz']
    error_df = pd.DataFrame({
        'Product': train_df['Warengruppe_Name'],
        'Error': train_errors
    })
    sns.boxplot(data=error_df, x='Product', y='Error')
    plt.xticks(rotation=45)
    plt.xlabel('Product Group')
    plt.ylabel('Prediction Error (€)')
    plt.title('Prediction Errors by Product Group')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'prediction_analysis.png'))
    plt.close()

def plot_weekday_patterns_overview(train_df, val_df, val_pred, test_df, test_pred):
    """Create plots showing weekly patterns for each product group"""
    # Add predictions to data
    train_data = train_df.copy()
    val_data = val_df.copy()
    test_data = test_df.copy()
    
    val_data['Predicted_Sales'] = val_pred
    test_data['Predicted_Sales'] = test_pred
    
    # Create a figure with subplots for each product group
    n_products = len(WARENGRUPPEN)
    fig, axes = plt.subplots(n_products, 1, figsize=(12, 4*n_products))
    
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                     'Friday', 'Saturday', 'Sunday']
    
    for (product_group, product_name), ax in zip(WARENGRUPPEN.items(), axes):
        # Get data for this product
        train_product = train_data[train_data['Warengruppe'] == product_group]
        val_product = val_data[val_data['Warengruppe'] == product_group]
        test_product = test_data[test_data['Warengruppe'] == product_group]
        
        # Calculate average sales by weekday
        train_weekday_avg = train_product.groupby('Wochentag')['Umsatz'].mean()
        val_weekday_avg = val_product.groupby('Wochentag').agg({
            'Umsatz': 'mean',
            'Predicted_Sales': 'mean'
        })
        test_weekday_avg = test_product.groupby('Wochentag')['Predicted_Sales'].mean()
        
        # Plot weekday patterns with different styles and slight offsets
        ax.plot(weekday_names, train_weekday_avg, 
                marker='o', label='Actual (Train)', color='blue', linewidth=2)
        ax.plot(weekday_names, val_weekday_avg['Umsatz'], 
                marker='s', label='Actual (Validation)', color='green', linewidth=2)
        ax.plot(weekday_names, val_weekday_avg['Predicted_Sales'], 
                marker='s', label='Predicted (Validation)', color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.plot(weekday_names, test_weekday_avg, 
                marker='^', label='Predicted (Test)', color='purple', linestyle=':', linewidth=2)
        
        ax.set_title(f'{product_name} - Weekly Sales Pattern')
        ax.set_xlabel('Weekday')
        ax.set_ylabel('Average Sales (€)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper left')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'weekly_patterns.png'))
    plt.close()

def plot_product_comparison(train_df, train_pred):
    """Create visualizations comparing predictions across product groups"""
    # Add predictions to data
    train_data = train_df.copy()
    train_data['Predicted_Sales'] = train_pred
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Box plot of actual vs predicted by product
    plot_data = pd.melt(train_data, 
                        id_vars=['Warengruppe_Name'],
                        value_vars=['Umsatz', 'Predicted_Sales'],
                        var_name='Type',
                        value_name='Sales')
    
    sns.boxplot(data=plot_data, x='Warengruppe_Name', y='Sales', hue='Type', ax=ax1)
    ax1.set_title('Distribution of Actual vs Predicted Sales by Product')
    ax1.set_xlabel('Product')
    ax1.set_ylabel('Sales (€)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Heatmap of prediction differences
    pivot_actual = train_data.pivot_table(
        values='Umsatz',
        index='Warengruppe_Name',
        columns='Wochentag',
        aggfunc='mean'
    )
    
    pivot_pred = train_data.pivot_table(
        values='Predicted_Sales',
        index='Warengruppe_Name',
        columns='Wochentag',
        aggfunc='mean'
    )
    
    # Calculate differences
    diff_matrix = ((pivot_pred - pivot_actual) / pivot_actual * 100)
    
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                     'Friday', 'Saturday', 'Sunday']
    diff_matrix.columns = weekday_names
    
    sns.heatmap(diff_matrix, cmap='RdYlBu_r', center=0, annot=True, fmt='.1f', ax=ax2,
                cbar_kws={'label': 'Prediction Difference (%)'})
    ax2.set_title('Prediction Difference by Product and Weekday (%)')
    ax2.set_xlabel('Weekday')
    ax2.set_ylabel('Product')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'product_comparison.png'))
    plt.close()

def main():
    """Main execution function"""
    print("Starting model training and evaluation...")
    
    # Load and prepare data
    train_df, val_df, test_df = load_and_prepare_data()
    
    # Train models for each product group
    models = {}
    predictions = {}
    metrics = {}
    
    feature_columns = ['Wochentag', 'is_weekend', 'week_sin', 'week_cos', 
                      'month_sin', 'month_cos', 'day_of_month', 
                      'month_position', 'season']
    
    # Store all predictions for overview plots
    all_train_pred = []
    all_val_pred = []
    all_test_pred = []
    train_indices = []
    val_indices = []
    test_indices = []
    
    for product_group in WARENGRUPPEN.keys():
        print(f"\nProcessing {WARENGRUPPEN[product_group]}...")
        model, preds, model_metrics = train_model_for_product(
            train_df, val_df, test_df, product_group
        )
        models[product_group] = model
        predictions[product_group] = preds
        metrics[product_group] = model_metrics
        
        # Store predictions for this product group
        if model is not None and preds is not None:
            train_mask = train_df['Warengruppe'] == product_group
            val_mask = val_df['Warengruppe'] == product_group
            test_mask = test_df['Warengruppe'] == product_group
            
            # Store predictions and corresponding indices
            all_train_pred.extend(preds[0])
            all_val_pred.extend(preds[1])
            all_test_pred.extend(preds[2])
            train_indices.extend(train_df[train_mask].index)
            val_indices.extend(val_df[val_mask].index)
            test_indices.extend(test_df[test_mask].index)
            
            # Generate individual product plots
            plot_predictions(train_df, val_df, test_df, predictions, product_group)
    
    # Generate overview plots only if we have predictions
    if all_train_pred and all_val_pred and all_test_pred:
        print("\nGenerating overview visualizations...")
        # Create DataFrames with predictions in the correct order
        train_pred_series = pd.Series(index=train_indices, data=all_train_pred)
        val_pred_series = pd.Series(index=val_indices, data=all_val_pred)
        test_pred_series = pd.Series(index=test_indices, data=all_test_pred)
        
        plot_predictions_overview(train_df, train_pred_series)
        plot_weekday_patterns_overview(train_df, val_df, val_pred_series,
                                     test_df, test_pred_series)
        plot_product_comparison(train_df, train_pred_series)
    
    # Generate submission file
    print("\nGenerating submission file...")
    submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    
    # Combine all test predictions
    all_predictions = []
    for product_group in WARENGRUPPEN.keys():
        # Get predictions for this product group
        if models[product_group] is None or predictions[product_group] is None:
            # Use mean from training if no model or predictions
            product_preds = [train_df[train_df['Warengruppe'] == product_group]['Umsatz'].mean()] * len(test_df)
        else:
            product_preds = predictions[product_group][2]
        all_predictions.extend(product_preds)
    
    submission_df['Umsatz'] = all_predictions
    submission_df.to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)
    
    # Save metrics
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nTraining complete! Results saved to:")
    print(f"- Metrics: {os.path.join(OUTPUT_DIR, 'metrics.json')}")
    print(f"- Submission: {os.path.join(OUTPUT_DIR, 'submission.csv')}")
    print(f"- Visualizations: {VIZ_DIR}")

if __name__ == "__main__":
    main() 