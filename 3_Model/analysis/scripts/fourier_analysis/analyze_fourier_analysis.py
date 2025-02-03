"""
Analyzes Fourier analysis model predictions for bakery sales forecasting.
Generates visualizations and statistics including:
- Weekly sales patterns per product
- Weekday sales trends across periods
- Timeline comparisons of daily sales
- Product-specific and overall statistics
- Weekday-product sales breakdowns

This script analyzes the predictions from the Fourier analysis model, which uses
frequency decomposition to capture periodic patterns in the sales data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse

# Set style for better visualizations
plt.style.use('seaborn-v0_8')

# German product names mapping
PRODUCT_NAMES = {
    1: 'Brot',
    2: 'Brötchen',
    3: 'Croissant',
    4: 'Konditorei',
    5: 'Kuchen',
    6: 'Saisonbrot'
}

# Set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, '..', '..', '..')
MODEL_DIR = os.path.join(BASE_DIR, 'output', 'fourier_analysis')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis', 'results', 'fourier_analysis')
VISUALIZATION_DIR = os.path.join(BASE_DIR, 'analysis', 'visualizations', 'fourier_analysis')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def load_data():
    """Load all necessary data for analysis"""
    # Load predictions
    pred_path = os.path.join(MODEL_DIR, 'submission.csv')
    if not os.path.exists(pred_path):
        print(f"Prediction file not found at: {pred_path}")
        return None, None, None
    
    # Load training and validation data
    train_path = os.path.join(BASE_DIR, '..', '0_DataPreparation', 'input', 'competition_data', 'train.csv')
    train_df = pd.read_csv(train_path)
    train_df['Datum'] = pd.to_datetime(train_df['Datum'])
    train_df['weekday'] = train_df['Datum'].dt.dayofweek
    
    # Split into training and validation using fixed date ranges
    train_mask = (train_df['Datum'] >= '2013-07-01') & (train_df['Datum'] <= '2017-07-31')
    val_mask = (train_df['Datum'] >= '2017-08-01') & (train_df['Datum'] <= '2018-07-31')
    val_df = train_df[val_mask].copy()
    train_df = train_df[train_mask].copy()
    
    # Load and prepare predictions
    pred_df = pd.read_csv(pred_path)
    pred_df['date'] = pd.to_datetime('20' + pred_df['id'].astype(str).str[:6], format='%Y%m%d')
    pred_df['product'] = pred_df['id'].astype(str).str[-1].astype(int)
    pred_df['weekday'] = pred_df['date'].dt.dayofweek
    
    return train_df, val_df, pred_df

def plot_weekly_patterns(train_df, val_df, pred_df):
    """Plot weekly sales patterns for each product"""
    print("\nCreating weekly pattern plots...")
    
    for product in range(1, 7):
        plt.figure(figsize=(10, 6))
        
        # Training data
        train_product = train_df[train_df['Warengruppe'] == product]
        train_avg = train_product.groupby('weekday')['Umsatz'].mean()
        plt.plot(train_avg.index, train_avg.values, 'b-', label='Actual (Train)', marker='o')
        
        # Validation data
        val_product = val_df[val_df['Warengruppe'] == product]
        val_avg = val_product.groupby('weekday')['Umsatz'].mean()
        plt.plot(val_avg.index, val_avg.values, 'g-', label='Actual (Validation)', marker='o')
        
        # Predictions
        pred_product = pred_df[pred_df['product'] == product]
        pred_avg = pred_product.groupby('weekday')['Umsatz'].mean()
        plt.plot(pred_avg.index, pred_avg.values, 'r--', label='Predicted (Test)', marker='^')
        
        plt.title(f'{PRODUCT_NAMES[product]} - Weekly Sales Pattern')
        plt.xlabel('Weekday')
        plt.ylabel('Average Sales (€)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(range(7), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_DIR, f'weekly_pattern_product_{product}.png'))
        plt.close()

def plot_weekday_patterns(train_df, val_df, pred_df):
    """Plot average sales by weekday across all periods"""
    print("\nCreating weekday pattern plot...")
    
    plt.figure(figsize=(12, 6))
    
    # Calculate averages
    train_avg = train_df.groupby('weekday')['Umsatz'].mean()
    val_avg = val_df.groupby('weekday')['Umsatz'].mean()
    pred_avg = pred_df.groupby('weekday')['Umsatz'].mean()
    
    # Plot
    plt.plot(train_avg.index, train_avg.values, 'o-', label='Training', color='#1f77b4', alpha=0.8)
    plt.plot(val_avg.index, val_avg.values, 's-', label='Validation', color='#ff7f0e', alpha=0.8)
    plt.plot(pred_avg.index, pred_avg.values, '^-', label='Test', color='#2ca02c', alpha=0.8)
    
    plt.title('Average Sales by Weekday - All Periods')
    plt.xlabel('Weekday')
    plt.ylabel('Average Sales (€)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(range(7), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'weekday_patterns.png'))
    plt.close()

def plot_timeline_sales(train_df, val_df, pred_df):
    """Plot timeline of total daily sales for all periods"""
    print("\nCreating timeline sales plots...")
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=False)
    
    # Training period
    daily_train = train_df.groupby('Datum')['Umsatz'].sum()
    ax1.plot(daily_train.index, daily_train.values, 'b-', linewidth=1)
    ax1.set_title('Training Period: Total Daily Sales (2013-07-01 to 2017-07-31)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('Total Sales (€)')
    
    # Validation period
    daily_val = val_df.groupby('Datum')['Umsatz'].sum()
    ax2.plot(daily_val.index, daily_val.values, 'g-', linewidth=1)
    ax2.set_title('Validation Period: Total Daily Sales (2017-08-01 to 2018-07-31)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel('Total Sales (€)')
    
    # Test period (predictions)
    daily_pred = pred_df.groupby('date')['Umsatz'].sum()
    ax3.plot(daily_pred.index, daily_pred.values, 'r-', linewidth=1)
    ax3.set_title('Test Period: Predicted Total Daily Sales')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylabel('Total Sales (€)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'timeline_sales.png'))
    plt.close()

def analyze_predictions():
    """Analyze predictions from the Fourier analysis model"""
    print("\nAnalyzing Fourier analysis model predictions...")
    
    # Load all data
    train_df, val_df, pred_df = load_data()
    if pred_df is None:
        print("No prediction file found!")
        return
    
    print(f"Loaded predictions from Fourier analysis model")
    
    # Create the visualization plots
    plot_weekly_patterns(train_df, val_df, pred_df)
    plot_weekday_patterns(train_df, val_df, pred_df)
    plot_timeline_sales(train_df, val_df, pred_df)
    
    # Basic statistics
    stats = {
        'mean': pred_df['Umsatz'].mean(),
        'std': pred_df['Umsatz'].std(),
        'min': pred_df['Umsatz'].min(),
        'max': pred_df['Umsatz'].max(),
        'zeros': (pred_df['Umsatz'] == 0).sum(),
        'negative': (pred_df['Umsatz'] < 0).sum()
    }
    
    # Save statistics to file
    stats_df = pd.DataFrame([stats])
    stats_path = os.path.join(OUTPUT_DIR, 'prediction_statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    print("\nPrediction Statistics:")
    print(stats_df.round(2))
    
    # Analyze by product
    product_stats = pred_df.groupby('product')['Umsatz'].agg(['count', 'mean', 'std', 'min', 'max'])
    product_stats_path = os.path.join(OUTPUT_DIR, 'product_statistics.csv')
    product_stats.to_csv(product_stats_path)
    print("\nProduct Statistics:")
    print(product_stats.round(2))
    
    # Analyze predictions by product and weekday
    pivot_table = pred_df.pivot_table(
        values='Umsatz',
        index='product',
        columns='weekday',
        aggfunc='mean'
    )
    pivot_table.columns = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_path = os.path.join(OUTPUT_DIR, 'product_weekday_predictions.csv')
    pivot_table.to_csv(pivot_path)
    print("\nProduct by Weekday Average Predictions:")
    print(pivot_table.round(2))

    print(f"\nAnalysis complete.")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Visualizations saved to: {VISUALIZATION_DIR}")

if __name__ == "__main__":
    analyze_predictions() 