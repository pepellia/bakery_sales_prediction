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
    """Load and prepare the data with basic features"""
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
        
        # Add product group names
        df['Warengruppe_Name'] = df['Warengruppe'].map(get_warengruppe_name)
    
    return train_data, val_data, test_df

def create_feature_pipeline():
    """Create a pipeline for feature preprocessing"""
    # Define the preprocessing steps for numerical and categorical features
    numeric_features = ['Wochentag']
    categorical_features = ['Warengruppe', 'is_weekend']
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
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
        
        # Plot weekday patterns with different styles
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
    """Main function to run the model"""
    print("Starting weekday and product model training...")
    
    # Load and prepare data
    train_data, val_data, test_df = load_and_prepare_data()
    
    # Define feature columns
    feature_columns = ['Wochentag', 'is_weekend', 'Warengruppe']
    
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
    plot_product_comparison(train_data, train_pred)
    
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