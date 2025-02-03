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
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    
    # Load test data
    print("\nLoading test data...")
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

def plot_predictions(train_df, train_pred):
    """Create visualizations of the model's predictions"""
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

def plot_weekday_patterns(train_df, train_pred, val_df, val_pred, test_df, test_pred):
    """Create plots showing weekly patterns for each product group"""
    # Add predictions to data
    train_data = train_df.copy()
    val_data = val_df.copy()
    test_data = test_df.copy()
    
    train_data['Predicted_Sales'] = train_pred
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
        train_weekday_avg = train_product.groupby('Wochentag').agg({
            'Umsatz': 'mean',
            'Predicted_Sales': 'mean'
        })
        val_weekday_avg = val_product.groupby('Wochentag').agg({
            'Umsatz': 'mean',
            'Predicted_Sales': 'mean'
        })
        test_weekday_avg = test_product.groupby('Wochentag')['Predicted_Sales'].mean()
        
        # Plot weekday patterns
        ax.plot(weekday_names, train_weekday_avg['Umsatz'], 
                marker='o', label='Actual (Train)', color='blue', linewidth=2)
        ax.plot(weekday_names, train_weekday_avg['Predicted_Sales'], 
                marker='o', label='Predicted (Train)', color='red', linestyle='--', linewidth=2)
        ax.plot(weekday_names, val_weekday_avg['Umsatz'], 
                marker='s', label='Actual (Validation)', color='green', linewidth=2)
        ax.plot(weekday_names, val_weekday_avg['Predicted_Sales'], 
                marker='s', label='Predicted (Validation)', color='orange', linestyle='--', linewidth=2)
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

def plot_seasonal_patterns(train_df, train_pred, val_df, val_pred):
    """Create visualizations of seasonal patterns"""
    # Add predictions to data
    train_data = train_df.copy()
    val_data = val_df.copy()
    train_data['Predicted_Sales'] = train_pred
    val_data['Predicted_Sales'] = val_pred
    
    # Plot 1: Sales by Season
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    
    # Training data
    train_seasonal = train_data.groupby(['season', 'Warengruppe_Name']).agg({
        'Umsatz': 'mean',
        'Predicted_Sales': 'mean'
    }).reset_index()
    
    # Validation data
    val_seasonal = val_data.groupby(['season', 'Warengruppe_Name']).agg({
        'Umsatz': 'mean',
        'Predicted_Sales': 'mean'
    }).reset_index()
    
    for product in train_data['Warengruppe_Name'].unique():
        # Training data
        train_product = train_seasonal[train_seasonal['Warengruppe_Name'] == product]
        plt.plot(train_product['season'], train_product['Umsatz'], 
                marker='o', label=f'{product} (Train Actual)')
        plt.plot(train_product['season'], train_product['Predicted_Sales'], 
                marker='o', linestyle='--', label=f'{product} (Train Pred)')
        
        # Validation data
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
    train_position = train_data.groupby(['month_position', 'Warengruppe_Name']).agg({
        'Umsatz': 'mean',
        'Predicted_Sales': 'mean'
    }).reset_index()
    
    # Validation data
    val_position = val_data.groupby(['month_position', 'Warengruppe_Name']).agg({
        'Umsatz': 'mean',
        'Predicted_Sales': 'mean'
    }).reset_index()
    
    for product in train_data['Warengruppe_Name'].unique():
        # Training data
        train_product = train_position[train_position['Warengruppe_Name'] == product]
        plt.plot(train_product['month_position'], train_product['Umsatz'], 
                marker='o', label=f'{product} (Train Actual)')
        plt.plot(train_product['month_position'], train_product['Predicted_Sales'], 
                marker='o', linestyle='--', label=f'{product} (Train Pred)')
        
        # Validation data
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

def main():
    """Main function to run the model"""
    print("Starting model training with date features...")
    
    # Load and prepare data
    train_data, val_data, test_df = load_and_prepare_data()
    
    # Define feature columns
    feature_columns = [
        'Wochentag', 'is_weekend', 'week_sin', 'week_cos',
        'month_sin', 'month_cos', 'day_of_month',
        'Warengruppe', 'month_position', 'season'
    ]
    
    # Create and train model
    print("\nTraining model...")
    model = create_feature_pipeline()
    model.fit(train_data[feature_columns], train_data['Umsatz'])
    
    # Generate predictions
    train_pred = model.predict(train_data[feature_columns])
    val_pred = model.predict(val_data[feature_columns])
    test_pred = model.predict(test_df[feature_columns])
    
    # Evaluate model
    print("\nModel Performance:")
    train_metrics = evaluate_model(train_data['Umsatz'], train_pred, "Training Data")
    val_metrics = evaluate_model(val_data['Umsatz'], val_pred, "Validation Data")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_predictions(train_data, train_pred)
    plot_weekday_patterns(train_data, train_pred, val_data, val_pred, test_df, test_pred)
    plot_seasonal_patterns(train_data, train_pred, val_data, val_pred)
    
    # Generate submission file
    print("\nGenerating submission file...")
    submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    submission_df['Umsatz'] = test_pred
    
    # Save submission file
    submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission file has been saved to: {submission_path}")
    print(f"Number of predictions: {len(submission_df)}")
    
    # Save metrics
    metrics = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }
    metrics_path = os.path.join(OUTPUT_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nResults have been saved to: {OUTPUT_DIR}")
    print(f"Visualizations have been saved to: {VIZ_DIR}")

if __name__ == "__main__":
    main() 