import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'competition_data')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
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
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output', SCRIPT_NAME)
VIZ_DIR = os.path.join(SCRIPT_DIR, 'visualizations', SCRIPT_NAME)

# Define output files
TRAIN_PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, 'train_predictions.csv')
VAL_PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, 'val_predictions.csv')

# Create directories
for directory in [OUTPUT_DIR, VIZ_DIR]:
    os.makedirs(directory, exist_ok=True)

def load_and_prepare_data():
    """
    Load data and prepare it by:
    1. Converting date to datetime
    2. Adding weekday feature
    3. Creating train/validation split
    """
    # Load training data
    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_PATH)
    train_df['Datum'] = pd.to_datetime(train_df['Datum'])
    train_df['Wochentag'] = train_df['Datum'].dt.dayofweek
    train_df['Warengruppe_Name'] = train_df['Warengruppe'].map(get_warengruppe_name)
    
    # Create train/validation split
    train_mask = (train_df['Datum'] >= '2013-07-01') & (train_df['Datum'] <= '2017-07-31')
    val_mask = (train_df['Datum'] >= '2017-08-01') & (train_df['Datum'] <= '2018-07-31')
    
    train_data = train_df[train_mask].copy()
    val_data = train_df[val_mask].copy()
    
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(TEST_PATH)
    test_df['Datum'] = pd.to_datetime(test_df['id'].astype(str).str[:6], format='%y%m%d')
    test_df['Wochentag'] = test_df['Datum'].dt.dayofweek
    test_df['Warengruppe_Name'] = test_df['Warengruppe'].map(get_warengruppe_name)
    
    return train_data, val_data, test_df

def calculate_weekday_averages(df):
    """Calculate average sales by weekday and product group"""
    weekday_avg = df.groupby(['Wochentag', 'Warengruppe'])['Umsatz'].mean().reset_index()
    return weekday_avg

def generate_predictions(weekday_avg, data_df):
    """Generate predictions using weekday averages"""
    predictions = pd.merge(
        data_df[['id' if 'id' in data_df.columns else 'Datum', 'Datum', 'Wochentag', 'Warengruppe', 'Warengruppe_Name']], 
        weekday_avg, 
        on=['Wochentag', 'Warengruppe']
    )
    return predictions

def plot_weekday_patterns(train_df, val_df=None):
    """Plot weekday sales patterns"""
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                     'Friday', 'Saturday', 'Sunday']
    
    # Overall weekday pattern
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=train_df, x='Wochentag', y='Umsatz', color='blue')
    if val_df is not None:
        sns.boxplot(data=val_df, x='Wochentag', y='Umsatz', color='green')
    plt.title('Sales Distribution by Weekday')
    plt.xlabel('Weekday')
    plt.ylabel('Sales (€)')
    plt.xticks(range(7), weekday_names)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'weekday_sales_distribution.png'))
    plt.close()
    
    # Weekday pattern by product group
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    sns.boxplot(data=train_df, x='Wochentag', y='Umsatz', hue='Warengruppe_Name')
    plt.title('Training: Sales Distribution by Weekday and Product Group')
    plt.xlabel('Weekday')
    plt.ylabel('Sales (€)')
    plt.xticks(range(7), weekday_names)
    plt.legend(title='Product Group', bbox_to_anchor=(1.05, 1))
    
    if val_df is not None:
        plt.subplot(2, 1, 2)
        sns.boxplot(data=val_df, x='Wochentag', y='Umsatz', hue='Warengruppe_Name')
        plt.title('Validation: Sales Distribution by Weekday and Product Group')
        plt.xlabel('Weekday')
        plt.ylabel('Sales (€)')
        plt.xticks(range(7), weekday_names)
        plt.legend(title='Product Group', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'weekday_product_sales_distribution.png'))
    plt.close()

def evaluate_predictions(actual_data, predictions, dataset_name=""):
    """Evaluate model predictions"""
    # Calculate overall metrics
    rmse = np.sqrt(mean_squared_error(actual_data['Umsatz'], predictions['Umsatz']))
    mae = mean_absolute_error(actual_data['Umsatz'], predictions['Umsatz'])
    r2 = r2_score(actual_data['Umsatz'], predictions['Umsatz'])
    
    # Calculate metrics by product group
    metrics_by_group = []
    for group, name in WARENGRUPPEN.items():
        group_actual = actual_data[actual_data['Warengruppe'] == group]['Umsatz']
        group_pred = predictions[predictions['Warengruppe'] == group]['Umsatz']
        
        if len(group_actual) > 0:  # Only calculate if we have data for this group
            metrics_by_group.append({
                'Product_Group': group,
                'Name': name,
                'RMSE': np.sqrt(mean_squared_error(group_actual, group_pred)),
                'MAE': mean_absolute_error(group_actual, group_pred),
                'R2': r2_score(group_actual, group_pred)
            })
    
    metrics_df = pd.DataFrame(metrics_by_group)
    
    # Save metrics
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, f'metrics_by_group_{dataset_name.lower()}.csv'), 
                     index=False)
    
    # Plot metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # RMSE plot
    sns.barplot(data=metrics_df, x='Name', y='RMSE', ax=ax1)
    ax1.set_title(f'RMSE by Product Group - {dataset_name}')
    ax1.set_xlabel('Product Group')
    ax1.tick_params(axis='x', rotation=45)
    
    # R² plot
    sns.barplot(data=metrics_df, x='Name', y='R2', ax=ax2)
    ax2.set_title(f'R² Score by Product Group - {dataset_name}')
    ax2.set_xlabel('Product Group')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f'model_metrics_{dataset_name.lower()}.png'))
    plt.close()
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }, metrics_df

def main():
    """Main function to run the baseline model"""
    print("Starting baseline weekday model analysis...")
    
    # Load and prepare data
    train_data, val_data, test_df = load_and_prepare_data()
    
    # Calculate weekday averages from training data
    print("\nCalculating weekday averages...")
    weekday_avg = calculate_weekday_averages(train_data)
    
    # Plot weekday patterns
    print("Creating visualizations...")
    plot_weekday_patterns(train_data, val_data)
    
    # Generate predictions and evaluate
    print("\nEvaluating model performance...")
    
    # Training predictions and evaluation
    train_predictions = generate_predictions(weekday_avg, train_data)
    train_metrics, train_group_metrics = evaluate_predictions(train_data, train_predictions, "Training")
    
    # Save training predictions
    pd.DataFrame({
        'Datum': train_predictions['Datum'],
        'predicted_sales': train_predictions['Umsatz']
    }).to_csv(TRAIN_PREDICTIONS_FILE, index=False)
    
    print("\nTraining Performance (2013-07-01 to 2017-07-31):")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Validation predictions and evaluation
    val_predictions = generate_predictions(weekday_avg, val_data)
    val_metrics, val_group_metrics = evaluate_predictions(val_data, val_predictions, "Validation")
    
    # Save validation predictions
    pd.DataFrame({
        'Datum': val_predictions['Datum'],
        'predicted_sales': val_predictions['Umsatz']
    }).to_csv(VAL_PREDICTIONS_FILE, index=False)
    
    print("\nValidation Performance (2017-08-01 to 2018-07-31):")
    for metric, value in val_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save all metrics
    metrics = {
        'training': train_metrics,
        'validation': val_metrics
    }
    pd.DataFrame(metrics).to_json(os.path.join(OUTPUT_DIR, 'metrics.json'))
    
    # Generate predictions for test set
    print("\nGenerating predictions for test set...")
    test_predictions = generate_predictions(weekday_avg, test_df)
    
    # Create submission file
    submission = pd.DataFrame({
        'id': test_predictions['id'],
        'Umsatz': test_predictions['Umsatz']
    })
    
    submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission file saved to: {submission_path}")

if __name__ == "__main__":
    main()
