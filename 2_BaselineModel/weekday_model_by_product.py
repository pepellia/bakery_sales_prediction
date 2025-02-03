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

def load_and_prepare_data():
    """
    Load data and prepare it by:
    1. Converting date to datetime
    2. Adding weekday feature
    """
    # Load training data
    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_PATH)
    train_df['Datum'] = pd.to_datetime(train_df['Datum'])
    train_df['Wochentag'] = train_df['Datum'].dt.dayofweek
    train_df['Warengruppe_Name'] = train_df['Warengruppe'].map(get_warengruppe_name)
    
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(TEST_PATH)
    test_df['Datum'] = pd.to_datetime(test_df['id'].astype(str).str[:6], format='%y%m%d')
    test_df['Wochentag'] = test_df['Datum'].dt.dayofweek
    test_df['Warengruppe_Name'] = test_df['Warengruppe'].map(get_warengruppe_name)
    
    return train_df, test_df

def calculate_weekday_averages(train_df):
    """Calculate average sales by weekday and product group"""
    weekday_avg = train_df.groupby(['Warengruppe', 'Wochentag'])['Umsatz'].mean().unstack()
    
    # Create a new index with product names
    new_index = [get_warengruppe_name(idx) for idx in weekday_avg.index]
    weekday_avg.index = new_index
    
    return weekday_avg

def generate_predictions(weekday_avg, test_df):
    """Generate predictions using weekday averages"""
    # Convert weekday averages to long format for merging
    weekday_pred = weekday_avg.stack().reset_index()
    weekday_pred.columns = ['Warengruppe_Name', 'Wochentag', 'Predicted_Sales']
    
    # Merge with test data
    predictions = pd.merge(
        test_df[['id', 'Datum', 'Wochentag', 'Warengruppe', 'Warengruppe_Name']], 
        weekday_pred,
        on=['Warengruppe_Name', 'Wochentag']
    )
    
    return predictions[['id', 'Predicted_Sales']]

def plot_weekday_patterns(weekday_avg):
    """Plot weekday sales patterns by product group"""
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                     'Friday', 'Saturday', 'Sunday']
    
    # Heatmap of weekday patterns
    plt.figure(figsize=(15, 8))
    sns.heatmap(weekday_avg, 
                cmap='YlOrRd', 
                annot=True, 
                fmt='.0f',
                cbar_kws={'label': 'Average Daily Sales (€)'})
    
    plt.title('Average Daily Sales by Product Group and Weekday')
    plt.xlabel('Weekday')
    plt.ylabel('Product Group')
    plt.xticks(np.arange(7) + 0.5, weekday_names, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'weekday_patterns_heatmap.png'))
    plt.close()
    
    # Line plots for each product group
    plt.figure(figsize=(15, 10))
    for product_name in weekday_avg.index:
        plt.plot(weekday_names, weekday_avg.loc[product_name], 
                marker='o', label=product_name)
    
    plt.title('Weekday Sales Patterns by Product Group')
    plt.xlabel('Weekday')
    plt.ylabel('Average Sales (€)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Product Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'weekday_patterns_lines.png'))
    plt.close()

def evaluate_model(train_df, weekday_avg):
    """Evaluate model performance on training data"""
    # Convert weekday averages to long format for merging
    weekday_pred = weekday_avg.stack().reset_index()
    weekday_pred.columns = ['Warengruppe_Name', 'Wochentag', 'Predicted_Sales']
    
    # Merge with training data
    train_pred = pd.merge(
        train_df[['Datum', 'Wochentag', 'Warengruppe_Name', 'Umsatz']], 
        weekday_pred,
        on=['Warengruppe_Name', 'Wochentag']
    )
    
    # Calculate overall metrics
    rmse = np.sqrt(mean_squared_error(train_pred['Umsatz'], train_pred['Predicted_Sales']))
    mae = mean_absolute_error(train_pred['Umsatz'], train_pred['Predicted_Sales'])
    r2 = r2_score(train_pred['Umsatz'], train_pred['Predicted_Sales'])
    
    # Calculate metrics by product group
    metrics_by_group = []
    for group_name in weekday_avg.index:
        group_data = train_pred[train_pred['Warengruppe_Name'] == group_name]
        metrics_by_group.append({
            'Product_Group': group_name,
            'RMSE': np.sqrt(mean_squared_error(group_data['Umsatz'], group_data['Predicted_Sales'])),
            'MAE': mean_absolute_error(group_data['Umsatz'], group_data['Predicted_Sales']),
            'R2': r2_score(group_data['Umsatz'], group_data['Predicted_Sales'])
        })
    
    metrics_df = pd.DataFrame(metrics_by_group)
    
    # Plot metrics by product group
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # RMSE plot
    sns.barplot(data=metrics_df, x='Product_Group', y='RMSE', ax=ax1)
    ax1.set_title('RMSE by Product Group')
    ax1.set_xlabel('Product Group')
    ax1.set_ylabel('RMSE')
    ax1.tick_params(axis='x', rotation=45)
    
    # R² plot
    sns.barplot(data=metrics_df, x='Product_Group', y='R2', ax=ax2)
    ax2.set_title('R² Score by Product Group')
    ax2.set_xlabel('Product Group')
    ax2.set_ylabel('R²')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'model_metrics_by_product.png'))
    plt.close()
    
    # Save metrics to file
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'model_metrics_by_product.csv'), index=False)
    
    return rmse, mae, r2, metrics_df

def main():
    """Main function to run the baseline model by product group"""
    print("Starting baseline weekday model by product group analysis...")
    
    # Load and prepare data
    train_df, test_df = load_and_prepare_data()
    
    # Calculate weekday averages by product group
    print("\nCalculating weekday averages by product group...")
    weekday_avg = calculate_weekday_averages(train_df)
    
    # Plot weekday patterns
    print("Creating visualizations...")
    plot_weekday_patterns(weekday_avg)
    
    # Evaluate model
    print("\nEvaluating model performance...")
    rmse, mae, r2, metrics_df = evaluate_model(train_df, weekday_avg)
    
    print("\nOverall Model Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    
    print("\nPerformance by Product Group:")
    print(metrics_df.to_string(index=False))
    
    # Generate predictions for test set
    print("\nGenerating predictions for test set...")
    predictions = generate_predictions(weekday_avg, test_df)
    
    # Save predictions
    submission_path = os.path.join(OUTPUT_DIR, 'weekday_model_by_product_submission.csv')
    predictions.to_csv(submission_path, index=False)
    print(f"\nPredictions saved to: {submission_path}")

if __name__ == "__main__":
    main() 